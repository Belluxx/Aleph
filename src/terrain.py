"""Join Float32 GeoTIFFs, copying 256-pixel blocks and splitting larger ones."""

import math
import struct
import zlib

from .common import atomic_path
from .geo import MERCATOR_RADIUS

TYPES = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d"}
COMPATIBLE = (258, 259, 262, 277, 284, 317, 339, 34735, 34736, 34737, 42113)
TIFF_LIMIT = 2**32


class TIFF:
    def __init__(self, stream):
        self.stream = stream
        stream.seek(0, 2)
        self.size = stream.tell()
        magic = self.read(0, 8)
        if magic[:4] not in (b"II*\0", b"MM\0*"):
            raise ValueError("Unsupported terrain TIFF header.")
        self.order = "<" if magic[:2] == b"II" else ">"
        offset = struct.unpack(self.order + "I", magic[4:])[0]
        count = struct.unpack(self.order + "H", self.read(offset, 2))[0]
        if count > 128:
            raise ValueError("Unexpected terrain TIFF directory.")
        self.tags = {}
        for i in range(count):
            entry = self.read(offset + 2 + i * 12, 12)
            code, kind, length, location = struct.unpack(self.order + "HHII", entry)
            if kind not in TYPES or code in self.tags:
                raise ValueError("Unsupported terrain TIFF tag.")
            size = struct.calcsize(self.order + TYPES[kind]) * length
            raw = entry[8 : 8 + size] if size <= 4 else self.read(location, size)
            self.tags[code] = kind, length, raw
        required = {258: 32, 339: 3, 277: 1, 262: 1}
        if any(self.value(key) != value for key, value in required.items()) or self.value(
            259
        ) not in (1, 8, 32946):
            raise ValueError("Expected tiled Float32 terrain.")
        if (self.value(322), self.value(323)) not in ((256, 256), (512, 512)):
            raise ValueError("Expected 256- or 512-pixel terrain blocks.")
        if self.value(274, 1) != 1 or self.value(284, 1) != 1 or self.value(317, 1) not in (1, 3):
            raise ValueError("Unsupported terrain layout.")
        if any(key not in self.tags for key in (33550, 33922, 34735, 42113, 324, 325)):
            raise ValueError("Missing terrain georeferencing or blocks.")
        keys = self.values(34735)
        self.keys = {
            keys[i]: keys[i + 3]
            for i in range(4, len(keys) - 3, 4)
            if keys[i + 1 : i + 3] == (0, 1)
        }
        if self.keys.get(3072) != 3857 or self.keys.get(1025) not in (1, 2):
            raise ValueError("Expected EPSG:3857 terrain.")

    def read(self, offset, size):
        if offset < 0 or offset + size > self.size:
            raise ValueError("Truncated terrain TIFF.")
        self.stream.seek(offset)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated terrain TIFF.")
        return data

    def values(self, code):
        kind, count, raw = self.tags[code]
        return struct.unpack(self.order + TYPES[kind] * count, raw)

    def value(self, code, default=None):
        return self.values(code)[0] if code in self.tags else default

    def blocks(self):
        """Yield four 256-pixel blocks, preserving the source pixel encoding."""
        if self.value(322) == 256:
            for offset, size in zip(self.values(324), self.values(325)):
                yield self.read(offset, size)
            return
        data = self.read(self.value(324), self.value(325))
        compressed = self.value(259) != 1
        if compressed:
            data = zlib.decompress(data)
        if len(data) != 512 * 512 * 4:
            raise ValueError("Invalid terrain block size.")
        blocks = [bytearray() for _ in range(4)]
        for y in range(512):
            row = data[y * 2048:(y + 1) * 2048]
            left, right = blocks[y // 256 * 2:y // 256 * 2 + 2]
            if self.value(317, 1) == 3:
                # Float prediction differences span four byte planes per row.
                # Rebase only the first byte of each half-plane after splitting.
                carry = 0
                for plane in range(4):
                    a = bytearray(row[plane * 512:plane * 512 + 256])
                    b = bytearray(row[plane * 512 + 256:(plane + 1) * 512])
                    a_sum, b_sum = sum(a), sum(b)
                    a[0] = (a[0] + carry) & 255
                    b[0] = (b[0] + a_sum) & 255
                    carry = b_sum
                    left.extend(a)
                    right.extend(b)
            else:
                left.extend(row[:1024])
                right.extend(row[1024:])
        for block in blocks:
            yield zlib.compress(block) if compressed else block


def header(tags, order):
    directory = bytearray(
        (b"II" if order == "<" else b"MM") + struct.pack(order + "HIH", 42, 8, len(tags))
    )
    data = bytearray()
    start = 8 + 2 + len(tags) * 12 + 4
    for code, (kind, count, raw) in sorted(tags.items()):
        directory.extend(struct.pack(order + "HHI", code, kind, count))
        if len(raw) <= 4:
            directory.extend(raw.ljust(4, b"\0"))
        else:
            data.extend(b"\0" * (-(start + len(data)) % 8))
            directory.extend(struct.pack(order + "I", start + len(data)))
            data.extend(raw)
    return directory + b"\0" * 4 + data


def origin(source, tile):
    scale = 2 * math.pi * MERCATOR_RADIUS / 2 ** tile["zoom"] / 512
    half = math.pi * MERCATOR_RADIUS
    shift = scale / 2 if source.keys[1025] == 2 else 0
    return scale, -half + tile["x"] * scale * 512 + shift, half - tile["y"] * scale * 512 - shift


def validate(source, tile):
    scale, west, north = origin(source, tile)
    pixel, tie = source.values(33550), source.values(33922)
    if not (
        (source.value(256), source.value(257)) == (512, 512)
        and len(pixel) == 3 and len(tie) == 6
        and all(math.isfinite(v) for v in pixel + tie)
        and all(abs(pixel[i] - scale) <= scale * 1e-8 for i in (0, 1))
        and tie[:3] == (0, 0, 0)
        and abs(tie[3] - west) <= scale * 1e-5
        and abs(tie[4] - north) <= scale * 1e-5
    ):
        raise ValueError("Terrain georeferencing does not match the download grid.")
    offsets, sizes = source.values(324), source.values(325)
    count = (512 // source.value(322)) ** 2
    if len(offsets) != count or len(sizes) != count:
        raise ValueError("Invalid terrain block directory.")
    if any(not offset or not size or offset + size > source.size for offset, size in zip(offsets, sizes)):
        raise ValueError("Missing or truncated saved terrain block.")


def merge(path, folder, results, grid, progress):
    """One source file open at a time; usual 256-pixel blocks are copied as-is."""
    columns, rows = grid["columns"], grid["rows"]
    if len(results) != columns * rows:
        raise ValueError("Download all terrain tiles before building terrain.tif.")
    offsets, sizes = [0] * (columns * rows * 4), [0] * (columns * rows * 4)
    tags, compatible, order = {}, None, None

    def set_tag(code, kind, values):
        tags[code] = kind, len(values), struct.pack(order + TYPES[kind] * len(values), *values)

    progress("Building terrain.tif", 0, len(results))
    with atomic_path(path) as temporary, temporary.open("wb") as output:
        for i, tile in enumerate(results):
            row, column = divmod(i, columns)
            if (tile["x"], tile["y"], tile["zoom"]) != (grid["x0"] + column, grid["y0"] + row, grid["zoom"]):
                raise ValueError("Invalid saved terrain tile order.")
            with (folder / tile["filename"]).open("rb") as stream:
                source = TIFF(stream)
                validate(source, tile)
                signature = {key: source.tags[key] for key in COMPATIBLE if key in source.tags}
                if compatible is None:
                    compatible, order = signature, source.order
                    tags.update(signature)
                    scale, west, north = origin(source, tile)
                    set_tag(256, 4, [columns * 512])
                    set_tag(257, 4, [rows * 512])
                    set_tag(322, 4, [256])
                    set_tag(323, 4, [256])
                    set_tag(33550, 12, [scale, scale, 0])
                    set_tag(33922, 12, [0, 0, 0, west, north, 0])
                    set_tag(324, 4, offsets)
                    set_tag(325, 4, sizes)
                    output.write(header(tags, order))
                elif order != source.order or compatible != signature:
                    raise ValueError("Terrain tile formats differ.")
                for j, data in enumerate(source.blocks()):
                    block = (row * 2 + j // 2) * columns * 2 + column * 2 + j % 2
                    size = len(data)
                    if output.tell() + size >= TIFF_LIMIT:
                        raise ValueError("Terrain exceeds the 4 GiB TIFF limit. Use a smaller area or lower terrain zoom.")
                    offsets[block], sizes[block] = output.tell(), size
                    output.write(data)
            progress("Building terrain.tif", i + 1, len(results))
        set_tag(324, 4, offsets)
        set_tag(325, 4, sizes)
        output.seek(0)
        output.write(header(tags, order))
