"""Small offline responses and a TIFF fixture written independently of src.terrain."""

import json
import struct
import zlib
from io import BytesIO

from PIL import Image


def image_bytes(size, color):
    with Image.new("RGB", size, color) as image, BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def metadata(pano_id="panorama_0001", lat=0, lon=0):
    return json.dumps([None, [[
        [1], [2, pano_id], None, None, None,
        [[None, [[None, None, lat, lon]]]],
        [None] * 7 + [[2024, 6]],
    ]]]).encode()


def terrain_bytes(x, y, zoom, heights=(1.25, -10.5, 200.75, -9999), *, order="<", raster_type=1):
    # Four independently compressed 256 x 256 Float32 blocks per provider tile.
    blocks = [zlib.compress(struct.pack(order + "f", height) * (256 * 256))
              for height in heights]
    scale = 40075016.68557849 / (2**zoom * 512)
    shift = scale / 2 if raster_type == 2 else 0
    west = -20037508.342789244 + x * scale * 512 + shift
    north = 20037508.342789244 - y * scale * 512 - shift
    tags = {
        256: (4, [512]), 257: (4, [512]), 258: (3, [32]), 259: (3, [8]),
        262: (3, [1]), 277: (3, [1]), 284: (3, [1]),
        322: (4, [256]), 323: (4, [256]), 324: (4, [0] * 4),
        325: (4, list(map(len, blocks))), 339: (3, [3]),
        33550: (12, [scale, scale, 0]),
        33922: (12, [0, 0, 0, west, north, 0]),
        34735: (3, [1, 1, 0, 2, 1025, 0, 1, raster_type, 3072, 0, 1, 3857]),
        42113: (2, b"-9999\0"),
    }
    payload = bytearray()
    entries = bytearray()
    payload_start = 8 + 2 + 12 * len(tags) + 4
    offset_location = None
    for code, (kind, values) in sorted(tags.items()):
        raw = values if kind == 2 else struct.pack(order + {3: "H", 4: "I", 12: "d"}[kind] * len(values), *values)
        if len(raw) > 4:
            location = payload_start + len(payload)
            if code == 324:
                offset_location = location
            payload.extend(raw)
            raw = struct.pack(order + "I", location)
        entries.extend(struct.pack(order + "HHI", code, kind, len(values)) + raw.ljust(4, b"\0"))
    data = bytearray(b"II" if order == "<" else b"MM")
    data.extend(struct.pack(order + "HIH", 42, 8, len(tags)) + entries + b"\0" * 4 + payload)
    offsets = []
    for block in blocks:
        offsets.append(len(data))
        data.extend(block)
    struct.pack_into(order + "4I", data, offset_location, *offsets)
    return bytes(data)
