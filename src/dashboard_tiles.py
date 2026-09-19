"""Local imagery pyramids and Terrarium display tiles, derived from saved captures."""

import math
import struct
import sys
import threading
from io import BytesIO

from PIL import Image

from . import terrain
from .common import atomic_path, contained


def terrain_tile(path, grid, x, y):
    """Decode just four compressed TIFF blocks, not the entire terrain mosaic."""
    with path.open("rb") as stream:
        source = terrain.TIFF(stream)
        columns = grid["columns"] * 2
        bx, by = (x - grid["x0"]) * 2, (y - grid["y0"]) * 2
        offsets, sizes = source.values(324), source.values(325)
        indices = [(by + dy) * columns + bx + dx for dy in range(2) for dx in range(2)]
        if any(i < 0 or i >= len(offsets) or not offsets[i] or not sizes[i] for i in indices):
            return Image.new("F", (512, 512))
        blocks = [source.read(offsets[i], sizes[i]) for i in indices]
        tags = {key: value for key, value in source.tags.items()
                if key in (258, 259, 262, 274, 277, 284, 317, 322, 323, 339)}

        def tag(code, values):
            tags[code] = 4, len(values), struct.pack(source.order + "I" * len(values), *values)

        tag(256, [512])
        tag(257, [512])
        tag(324, [0] * 4)
        tag(325, [len(block) for block in blocks])
        cursor = len(terrain.header(tags, source.order))
        locations = []
        for block in blocks:
            locations.append(cursor)
            cursor += len(block)
        tag(324, locations)
        data = terrain.header(tags, source.order) + b"".join(blocks)
        nodata = float(b"".join(source.values(42113)).rstrip(b"\0"))
    with Image.open(BytesIO(data)) as decoded:
        # libtiff returns native floats, even when the source TIFF is big-endian.
        raw_mode = "F;32F" if sys.byteorder == "little" else "F;32BF"
        decoded.tile = [(codec, extent, offset, (raw_mode, *args[1:]))
                        if codec == "libtiff" else (codec, extent, offset, args)
                        for codec, extent, offset, args in decoded.tile]
        image = decoded.convert("F")
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
        self.lock = threading.Lock()
        self.versions = {}

    def get(self, folder, run, kind, z, x, y):
        folder = folder.resolve()
        mode = "satellite" if kind == "satellite" else "osm"
        stage = next((stage for stage in run["stages"] if stage["mode"] == mode), None)
        if stage is None or z < 0 or z > stage["grid"]["zoom"] or not (0 <= x < 2**z and 0 <= y < 2**z):
            raise FileNotFoundError("Tile is outside this capture.")
        directory = self.cache / folder.name / kind
        if kind == "terrain":
            stat = contained(folder, "terrain.tif").stat()
            directory /= f"{stat.st_mtime_ns}-{stat.st_size}"
        grid = stage["grid"]
        count = len(stage["results"]) if kind == "satellite" else grid["columns"] * grid["rows"]
        version, complete = tile_state(grid, count, z, x, y)
        filename = directory / str(z) / str(x) / f"{y}.png"
        # Limit concurrent decoding and avoid two requests writing the same cache tile.
        with self.lock:
            if self.versions.get(filename, -1) < version:
                with self._image(folder, directory, grid, stage["results"], count, kind, z, x, y) as image:
                    if kind == "terrain":
                        with terrarium(image) as rgb, atomic_path(filename) as temporary:
                            rgb.save(temporary, format="PNG")
                        self.versions[filename] = version
            return filename, complete

    def _image(self, folder, directory, grid, results, count, kind, z, x, y):
        elevation = kind == "terrain"
        mode, size = ("F", 512) if elevation else ("RGBA", 256)
        suffix = "tif" if elevation else "png"
        filename = directory / str(z) / str(x) / f"{y}.{suffix}"
        version, _ = tile_state(grid, count, z, x, y)
        if self.versions.get(filename, -1) >= version:
            with Image.open(filename) as image:
                return image.copy()
        image = Image.new(mode, (size, size))
        if version and z == grid["zoom"]:
            if elevation:
                image.close()
                image = terrain_tile(contained(folder, "terrain.tif"), grid, x, y)
            else:
                item = results[version - 1]
                path = contained(folder, item["filename"])
                if path.is_file():
                    with Image.open(path) as patch:
                        image.paste(patch)
        elif version:
            for dy in range(2):
                for dx in range(2):
                    with (
                        self._image(folder, directory, grid, results, count, kind, z + 1, x * 2 + dx, y * 2 + dy) as child,
                        child.resize((size // 2, size // 2), Image.Resampling.BOX) as smaller,
                    ):
                        image.paste(smaller, (dx * size // 2, dy * size // 2))
        with atomic_path(filename) as temporary:
            image.save(temporary, format="TIFF" if elevation else "PNG")
        self.versions[filename] = version
        return image
