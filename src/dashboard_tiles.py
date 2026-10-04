"""Local imagery pyramids and Terrarium display tiles, derived from saved captures."""

import math
import struct
import threading
import zlib
from itertools import accumulate

from PIL import Image

from . import terrain
from .common import atomic_path, contained


def unpredict(data, width):
    """Undo the TIFF floating point predictor, giving big-endian Float32 rows."""
    rows = bytearray()
    for y in range(0, len(data), width * 4):
        # Each row holds byte differences across four planes, most significant first.
        planes = bytes(v & 255 for v in accumulate(data[y:y + width * 4]))
        row = bytearray(width * 4)
        for plane in range(4):
            row[plane::4] = planes[plane * width:(plane + 1) * width]
        rows += row
    return bytes(rows)


def terrain_tile(path):
    """Decode one saved 512-pixel terrain tile without libtiff."""
    with path.open("rb") as stream:
        source = terrain.TIFF(stream)
        size = source.value(322)
        per_row = 512 // size
        image = Image.new("F", (512, 512))
        for i, (offset, length) in enumerate(zip(source.values(324), source.values(325))):
            data = source.read(offset, length)
            if source.value(259) != 1:
                data = zlib.decompress(data)
            if source.value(317, 1) == 3:
                data, raw_mode = unpredict(data, size), "F;32BF"
            else:
                raw_mode = "F;32F" if source.order == "<" else "F;32BF"
            with Image.frombytes("F", (size, size), data, "raw", raw_mode) as block:
                image.paste(block, (i % per_row * size, i // per_row * size))
        nodata = float(b"".join(source.values(42113)).rstrip(b"\0"))
    # RGB elevation has no NoData representation. Unknown heights use sea level.
    image.putdata([v if math.isfinite(v) and v != nodata else 0
                   for (v,) in struct.iter_unpack("=f", image.tobytes())])
    return image


def terrarium(image):
    rgb = bytearray(image.width * image.height * 3)
    for i, (height,) in enumerate(struct.iter_unpack("=f", image.tobytes())):
        value = max(0, min(16777215, round((height + 32768) * 256)))
        rgb[i * 3:i * 3 + 3] = value.to_bytes(3, "big")
    return Image.frombytes("RGB", image.size, bytes(rgb))


def tile_state(grid, count, z, x, y):
    """Last saved patch in a tile, and whether it is complete (downloads are row-major)."""
    factor = 2 ** (grid["zoom"] - z)
    left = max(x * factor - grid["x0"], 0)
    right = min((x + 1) * factor - grid["x0"], grid["columns"])
    top = max(y * factor - grid["y0"], 0)
    bottom = min((y + 1) * factor - grid["y0"], grid["rows"])
    if left >= right or top >= bottom:
        return 0, True
    columns = grid["columns"]
    row = min(bottom - 1, (count - 1) // columns)
    end = min(right, count - row * columns)
    if end <= left:
        row, end = row - 1, right
    return (row * columns + end if row >= top else 0), count >= (bottom - 1) * columns + right


class Tiles:
    def __init__(self, cache):
        self.cache = cache
        self.locks = {kind: threading.Lock() for kind in ("satellite", "terrain")}
        self.versions = {}

    def get(self, folder, run, kind, z, x, y):
        folder = folder.resolve()
        stage = next((stage for stage in run["stages"] if stage["mode"] == kind), None)
        if stage is None or z < 0 or z > stage["grid"]["zoom"] or not (0 <= x < 2**z and 0 <= y < 2**z):
            raise FileNotFoundError("Tile is outside this capture.")
        grid, results = stage["grid"], stage["results"]
        version, complete = tile_state(grid, len(results), z, x, y)
        # Native imagery is already browser-readable; avoid decoding or re-encoding it.
        if kind == "satellite" and version and z == grid["zoom"]:
            return contained(folder, results[version - 1]["filename"]), complete
        filename = self.cache / folder.name / kind / str(z) / str(x) / f"{y}.png"
        # Published cache files are atomic. Ready tiles never wait for an overview build.
        if self.versions.get(filename, -1) >= version:
            return filename, complete
        # Bound decoding to one tree per layer, without terrain blocking satellite imagery.
        with self.locks[kind]:
            if self.versions.get(filename, -1) < version:
                with self._image(folder, kind, grid, results, z, x, y) as image:
                    if kind == "terrain":
                        with terrarium(image) as rgb, atomic_path(filename) as temporary:
                            rgb.save(temporary, format="PNG")
                        self.versions[filename] = version
            return filename, complete

    def _image(self, folder, kind, grid, results, z, x, y):
        """Satellite tiles are cached as display PNGs, terrain as Float32 TIFFs."""
        elevation = kind == "terrain"
        mode, size = ("F", 512) if elevation else ("RGBA", 256)
        filename = self.cache / folder.name / kind / str(z) / str(x) / f"{y}.{'tif' if elevation else 'png'}"
        version, _ = tile_state(grid, len(results), z, x, y)
        if self.versions.get(filename, -1) >= version:
            with Image.open(filename) as image:
                return image.copy()
        native = version and z == grid["zoom"]
        if native and not elevation:
            with Image.open(contained(folder, results[version - 1]["filename"])) as patch:
                return patch.convert("RGBA")
        if native:
            image = terrain_tile(contained(folder, results[version - 1]["filename"]))
        else:
            image = Image.new(mode, (size, size))
            children = [(dx, dy) for dy in range(2) for dx in range(2)] if version else []
            for dx, dy in children:
                with (
                    self._image(folder, kind, grid, results, z + 1, x * 2 + dx, y * 2 + dy) as child,
                    child.resize((size // 2, size // 2), Image.Resampling.BOX) as smaller,
                ):
                    image.paste(smaller, (dx * size // 2, dy * size // 2))
        with atomic_path(filename) as temporary:
            image.save(temporary, format="TIFF" if elevation else "PNG")
        self.versions[filename] = version
        return image
