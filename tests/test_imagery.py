"""Check exported pixels, placement, and the custom TIFF binary layouts."""

import math
import struct
import unittest
import zlib
from io import BytesIO
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src import geo, satellite, terrain


def quiet(*args):
    pass


class SatelliteTests(unittest.TestCase):
    def test_tile_aligned_bounds_do_not_add_extra_pixels_or_tiles(self):
        for zoom, x, y in ((1, 0, 0), (1, 1, 1), (17, 23456, 43210), (21, 2097151, 2097151)):
            with self.subTest(zoom=zoom, x=x, y=y):
                north, west = geo.coordinate(x * 256, y * 256, zoom)
                south, east = geo.coordinate((x + 1) * 256, (y + 1) * 256, zoom)
                self.assertEqual(geo.grid((south, west, north, east), zoom), dict(
                    zoom=zoom, left=x * 256, top=y * 256, width=256, height=256,
                    x0=x, y0=y, columns=1, rows=1))

        north, west = geo.coordinate(1024 - 0.01, 1280 - 0.01, 21)
        south, east = geo.coordinate(1280 + 0.01, 1536 + 0.01, 21)
        self.assertEqual(geo.grid((south, west, north, east), 21), dict(
            zoom=21, left=1023, top=1279, width=258, height=258,
            x0=3, y0=4, columns=3, rows=3))

    def test_partial_mosaic_crops_pixels_and_writes_georeferenced_overview(self):
        grid = dict(zoom=4, left=1027, top=1287, width=301, height=259,
                    x0=4, y0=5, columns=2, rows=2)
        with TemporaryDirectory() as directory, Image.new("RGBA", (512, 512)) as canvas:
            folder = Path(directory)
            results = []
            for i in range(3):  # Fourth tile has not been downloaded yet.
                row, column = divmod(i, 2)
                with Image.new("RGBA", (256, 256)) as image:
                    image.putdata([(x, y, i * 60, 200) for y in range(256) for x in range(256)])
                    image.save(folder / f"{i}.png")
                    canvas.paste(image, (column * 256, row * 256))
                results.append(dict(x=4 + column, y=5 + row, zoom=4, filename=f"{i}.png"))
            stage = dict(grid=grid, results=results)
            with canvas.crop((3, 7, 304, 266)) as expected:
                # Force BigTIFF using a small file; never allocate gigabytes.
                for limit, magic in ((2**32, b"II*\0"), (1, b"II+\0")):
                    with self.subTest(format=magic), patch("src.satellite.TIFF_LIMIT", limit):
                        satellite.merge(stage, folder, quiet)
                        self.assertEqual((folder / "satellite.tif").read_bytes()[:4], magic)
                        with Image.open(folder / "satellite.png") as png:
                            self.assertEqual(png.size, expected.size)
                            self.assertEqual(png.tobytes(), expected.tobytes())
                        with Image.open(folder / "satellite.tif") as tif:
                            self.assertEqual(tif.size, expected.size)
                            self.assertEqual(tif.tobytes(), expected.tobytes())
                            self.assertEqual(tif.n_frames, 2)
                            scale = 2 * math.pi * 6_378_137 / 4096
                            self.assertEqual(tif.tag_v2[33550], (scale, scale, 0))
                            self.assertEqual(tif.tag_v2[33922], (
                                0, 0, 0, 1027 * scale - math.pi * 6_378_137,
                                math.pi * 6_378_137 - 1287 * scale, 0))
                            self.assertIn(3857, tif.tag_v2[34735])
                            full_resolution_start = min(tif.tag_v2[324])
                            tif.seek(1)
                            self.assertEqual(tif.size, (151, 130))
                            self.assertEqual(tif.tag_v2[254], 1)
                            self.assertLess(max(tif.tag_v2[324]), full_resolution_start)
                            with expected.resize((151, 130), Image.Resampling.BOX) as overview:
                                self.assertEqual(tif.tobytes(), overview.tobytes())


def float_tile(order, predictor, x=8):
    """Synthetic 512px source; predictor oracle is independent of block splitting."""
    raw = b"".join(struct.pack(order + "f", ((i * 257 + x * 131) % 10000 - 5000) / 37)
                   for i in range(512 * 512))
    if predictor == 3:
        encoded = bytearray()
        for y in range(512):
            row = raw[y * 2048:(y + 1) * 2048]
            planes = b"".join(row[i::4] for i in range(4))
            encoded.extend(bytes([planes[0]]) + bytes((b - a) & 255 for a, b in zip(planes, planes[1:])))
        data = zlib.compress(encoded)
    else:
        data = zlib.compress(raw)
    scale = 2 * math.pi * 6_378_137 / 16 / 512
    tags = {}

    def tag(code, kind, values):
        tags[code] = kind, len(values), struct.pack(order + terrain.TYPES[kind] * len(values), *values)

    for code, value in {256: 512, 257: 512, 258: 32, 259: 8, 262: 1, 277: 1,
                        317: predictor, 322: 512, 323: 512, 339: 3}.items():
        tag(code, 3, [value])
    tag(33550, 12, [scale, scale, 0])
    tag(33922, 12, [0, 0, 0, -math.pi * 6_378_137 + x * 512 * scale,
                    math.pi * 6_378_137 - 5 * 512 * scale, 0])
    tag(34735, 3, [1, 1, 0, 2, 1025, 0, 1, 1, 3072, 0, 1, 3857])
    tag(42113, 2, [bytes([v]) for v in b"-9999\0"])
    tag(324, 4, [0])
    tag(325, 4, [len(data)])
    tag(324, 4, [len(terrain.header(tags, order))])
    return terrain.header(tags, order) + data, raw


def decode_float_block(data, predictor):
    raw = zlib.decompress(data)
    if predictor != 3:
        return raw
    decoded = bytearray()
    for y in range(256):
        planes = bytes(v & 255 for v in accumulate(raw[y * 1024:(y + 1) * 1024]))
        decoded.extend(v for pixel in zip(*(planes[i * 256:(i + 1) * 256] for i in range(4))) for v in pixel)
    return bytes(decoded)


class TerrainTests(unittest.TestCase):
    def test_merge_preserves_float_pixels_when_splitting_predictor_blocks(self):
        for order, predictor in (("<", 1), ("<", 3), (">", 3)):
            with self.subTest(order=order, predictor=predictor), TemporaryDirectory() as directory:
                folder = Path(directory)
                results, originals = [], []
                for x in (8, 9):
                    data, raw = float_tile(order, predictor, x)
                    (folder / f"{x}.tif").write_bytes(data)
                    originals.append(raw)
                    results.append(dict(x=x, y=5, zoom=4, filename=f"{x}.tif"))
                terrain.merge(folder / "terrain.tif", folder, results,
                              dict(columns=2, rows=1, x0=8, y0=5, zoom=4), quiet)
                with (folder / "terrain.tif").open("rb") as stream:
                    merged = terrain.TIFF(stream)
                    self.assertEqual((merged.value(256), merged.value(257)), (1024, 512))
                    self.assertEqual(merged.order, order)
                    self.assertEqual(merged.value(317), predictor)
                    for i, block in enumerate(merged.blocks()):
                        row, column = divmod(i, 4)
                        original = originals[column // 2]
                        left = column % 2 * 1024
                        expected = b"".join(original[y * 2048 + left:y * 2048 + left + 1024]
                                            for y in range(row * 256, (row + 1) * 256))
                        self.assertEqual(decode_float_block(block, predictor), expected)
                    self.assertEqual(len(merged.values(324)), 8)

    def test_truncated_or_misplaced_terrain_is_rejected(self):
        data, _ = float_tile("<", 3)
        for payload, x in ((data[:-1], 8), (data, 9)):
            with self.subTest(truncated=len(payload) < len(data), x=x), self.assertRaises(ValueError):
                terrain.validate(terrain.TIFF(BytesIO(payload)), dict(x=x, y=5, zoom=4))
