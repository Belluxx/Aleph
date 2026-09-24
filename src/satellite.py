"""Lossless RGBA Cloud Optimized GeoTIFF and PNG exports using Pillow and the stdlib.

Directories precede all pixels; tiled overviews precede full-resolution data.
Compressed blocks are spooled to disk, bounding COG decoding to small windows.
The PNG reuses those blocks, decoding one tile row at a time.
No GDAL or libtiff encoder is required.
"""

import math
import struct
import tempfile
import zlib
from array import array

from PIL import Image

from .common import atomic_path, contained
from .geo import MERCATOR_RADIUS

BLOCK = 256
TIFF_LIMIT = 2**32


def region(box, read_tile):
    """Assemble a small pixel window from 256-pixel tiles, including alpha."""
    left, top, right, bottom = box
    image = Image.new("RGBA", (right - left, bottom - top))
    try:
        for y in range(top // BLOCK, (bottom - 1) // BLOCK + 1):
            for x in range(left // BLOCK, (right - 1) // BLOCK + 1):
                with read_tile(x, y) as tile:
                    image.paste(tile, (x * BLOCK - left, y * BLOCK - top))
        return image
    except BaseException:
        image.close()
        raise


class Level:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.columns = math.ceil(width / BLOCK)
        self.rows = math.ceil(height / BLOCK)
        self.offsets, self.sizes = array("Q"), array("I")

    def read(self, stream, x, y):
        i = y * self.columns + x
        stream.seek(self.offsets[i])
        raw = zlib.decompress(stream.read(self.sizes[i]))
        return Image.frombytes("RGBA", (BLOCK, BLOCK), raw)

    def write(self, stream, image):
        with Image.new("RGBA", (BLOCK, BLOCK)) as tile:
            tile.paste(image, (0, 0))
            data = zlib.compress(tile.tobytes())
        stream.seek(0, 2)
        self.offsets.append(stream.tell())
        self.sizes.append(len(data))
        stream.write(data)


def header(levels, grid, *, big=False):
    """Encode just the TIFF directories and GeoKeys, with final block offsets.

    Classic TIFF uses 32-bit offsets; BigTIFF uses 64-bit offsets. Pixel data
    follows this header in smallest-to-largest level order, row-major per level.
    """
    offset_format, count_format = ("Q", "Q") if big else ("I", "H")
    inline = 8 if big else 4
    directories = []
    scale = 2 * math.pi * MERCATOR_RADIUS / (BLOCK * 2**grid["zoom"])
    half = math.pi * MERCATOR_RADIUS
    for i, level in enumerate(levels):
        # (TIFF type, values). ExtraSamples=2 means unassociated alpha.
        tags = {
            254: (4, [int(i != 0)]), 256: (4, [level.width]), 257: (4, [level.height]),
            258: (3, [8, 8, 8, 8]), 259: (3, [8]), 262: (3, [2]),
            277: (3, [4]), 284: (3, [1]), 322: (4, [BLOCK]), 323: (4, [BLOCK]),
            324: (16 if big else 4, [0] * len(level.sizes)),
            325: (4, level.sizes), 338: (3, [2]),
        }
        if i == 0:
            tags.update({
                33550: (12, [scale, scale, 0]),
                33922: (12, [0, 0, 0, grid["left"] * scale - half,
                            half - grid["top"] * scale, 0]),
                34735: (3, [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857]),
            })
        directories.append(tags)

    data = bytearray(struct.pack("<2sHHHQ", b"II", 43, 8, 0, 16) if big
                     else struct.pack("<2sHI", b"II", 42, 8))
    locations = []
    for tags in directories:
        data.extend(b"\0" * (-len(data) % 8))
        locations.append(len(data))
        data.extend(b"\0" * ((8 if big else 2) + len(tags) * (20 if big else 12) + inline))
    deferred = []
    offset_fields = []
    for i, tags in enumerate(directories):
        cursor = locations[i]
        struct.pack_into("<" + count_format, data, cursor, len(tags))
        cursor += 8 if big else 2
        for code, (kind, values) in sorted(tags.items()):
            struct.pack_into("<HH" + offset_format, data, cursor, code, kind, len(values))
            field = cursor + (12 if big else 8)
            raw = struct.pack("<" + {3: "H", 4: "I", 12: "d", 16: "Q"}[kind] * len(values), *values)
            if len(raw) <= inline:
                data[field:field + len(raw)] = raw
                if code == 324:
                    offset_fields.append((i, field))
            else:
                deferred.append((code in (324, 325), i, code, field, raw))
            cursor += 20 if big else 12
        struct.pack_into("<" + offset_format, data, cursor,
                         locations[i + 1] if i + 1 < len(locations) else 0)
    # Keep metadata before potentially large tile offset/count arrays.
    for _, i, code, field, raw in sorted(deferred, key=lambda item: item[0]):
        data.extend(b"\0" * (-len(data) % 8))
        struct.pack_into("<" + offset_format, data, field, len(data))
        if code == 324:
            offset_fields.append((i, len(data)))
        data.extend(raw)
    data.extend(b"\0" * (-len(data) % 8))
    starts, cursor = {}, len(data)
    for i in reversed(range(len(levels))):
        starts[i] = cursor
        cursor += sum(levels[i].sizes)
    if not big and cursor >= TIFF_LIMIT:
        return header(levels, grid, big=True)
    for i, field in offset_fields:
        cursor = starts[i]
        for size in levels[i].sizes:
            struct.pack_into("<" + offset_format, data, field, cursor)
            cursor += size
            field += inline
    return data


def write_png(path, level, spool, progress):
    """Stream the full-resolution pixels as PNG, decoding one tile row at a time."""
    with path.open("wb") as output:
        def chunk(kind, data):
            output.write(struct.pack(">I", len(data)))
            output.write(kind)
            output.write(data)
            output.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind))))

        output.write(b"\x89PNG\r\n\x1a\n")
        chunk(b"IHDR", struct.pack(">2I5B", level.width, level.height, 8, 6, 0, 0, 0))
        compressor = zlib.compressobj()
        progress("Building satellite.png", 0, level.rows)
        for row, top in enumerate(range(0, level.height, BLOCK), 1):
            with region((0, top, level.width, min(top + BLOCK, level.height)),
                        lambda x, y: level.read(spool, x, y)) as strip:
                for y in range(strip.height):
                    with strip.crop((0, y, level.width, y + 1)) as scanline:
                        data = compressor.compress(b"\0" + scanline.tobytes())
                    if data:
                        chunk(b"IDAT", data)
            progress("Building satellite.png", row, level.rows)
        chunk(b"IDAT", compressor.flush())
        chunk(b"IEND", b"")


def merge(stage, folder, progress):
    """Write satellite.tif and satellite.png, retaining transparent gaps."""
    grid = stage["grid"]
    levels = [Level(grid["width"], grid["height"])]
    while max(levels[-1].width, levels[-1].height) > BLOCK:
        previous = levels[-1]
        levels.append(Level((previous.width + 1) // 2, (previous.height + 1) // 2))
    count = sum(level.columns * level.rows for level in levels)
    done = 0
    progress("Building satellite.tif", done, count)

    def source(x, y):
        index = (y - grid["y0"]) * grid["columns"] + x - grid["x0"]
        if index >= len(stage["results"]):
            return Image.new("RGBA", (BLOCK, BLOCK))
        tile = stage["results"][index]
        if (tile["x"], tile["y"], tile["zoom"]) != (x, y, grid["zoom"]):
            raise ValueError("Invalid saved satellite tile order.")
        with Image.open(contained(folder, tile["filename"])) as image:
            if image.format not in ("JPEG", "PNG") or image.size != (BLOCK, BLOCK):
                raise ValueError("Expected a 256 × 256 satellite JPEG or PNG.")
            return image.convert("RGBA")

    with (
        atomic_path(folder / "satellite.tif") as temporary,
        atomic_path(folder / "satellite.png") as png_temporary,
        tempfile.TemporaryFile(dir=folder) as spool,
    ):
        for i, level in enumerate(levels):
            for y in range(0, level.height, BLOCK):
                for x in range(0, level.width, BLOCK):
                    width, height = min(BLOCK, level.width - x), min(BLOCK, level.height - y)
                    if i == 0:
                        left, top = grid["left"] + x, grid["top"] + y
                        with region((left, top, left + width, top + height), source) as image:
                            level.write(spool, image)
                    else:
                        previous = levels[i - 1]
                        # Fractional windows preserve the extent for odd dimensions.
                        sx, sy = previous.width / level.width, previous.height / level.height
                        box = (x * sx, y * sy, (x + width) * sx, (y + height) * sy)
                        window = (math.floor(box[0]), math.floor(box[1]),
                                  math.ceil(box[2]), math.ceil(box[3]))
                        local = tuple(v - window[j % 2] for j, v in enumerate(box))
                        with (
                            region(window, lambda tx, ty: previous.read(spool, tx, ty)) as image,
                            image.resize((width, height), Image.Resampling.BOX, box=local) as smaller,
                        ):
                            level.write(spool, smaller)
                    done += 1
                    progress("Building satellite.tif", done, count)
        with temporary.open("wb") as output:
            output.write(header(levels, grid))
            total_bytes = sum(sum(level.sizes) for level in levels)
            copied = 0
            progress("Saving satellite.tif", copied, total_bytes, "bytes")
            for level in reversed(levels):
                spool.seek(level.offsets[0])
                remaining = sum(level.sizes)
                while remaining:
                    data = spool.read(min(1024 * 1024, remaining))
                    if not data:
                        raise OSError("Truncated satellite export spool.")
                    output.write(data)
                    remaining -= len(data)
                    copied += len(data)
                    progress("Saving satellite.tif", copied, total_bytes, "bytes")
        write_png(png_temporary, levels[0], spool, progress)
