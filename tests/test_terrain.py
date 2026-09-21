import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from src import terrain
from tests.fixtures import terrain_bytes


class TerrainTests(unittest.TestCase):
    def test_merge_preserves_float_blocks_in_spatial_order_and_georeferencing(self):
        for order in ("<", ">"):
            for raster_type in (1, 2):
                with self.subTest(order=order, raster_type=raster_type), tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    results = []
                    for i in range(4):
                        x, y = 1 + i % 2, 1 + i // 2
                        filename = f"{i}.tif"
                        heights = [i * 4 + j + 0.25 for j in range(4)]
                        (folder / filename).write_bytes(terrain_bytes(x, y, 2, heights, order=order, raster_type=raster_type))
                        results.append(dict(x=x, y=y, zoom=2, filename=filename))
                    path = folder / "terrain.tif"
                    terrain.merge(path, folder, results, dict(x0=1, y0=1, zoom=2, columns=2, rows=2), Mock())

                    # Pillow reads the output directory independently of our TIFF reader.
                    with Image.open(path) as mosaic:
                        self.assertEqual(mosaic.size, (1024, 1024))
                        tags = mosaic.tag_v2
                        scale = 40075016.68557849 / 2048
                        shift = scale / 2 if raster_type == 2 else 0
                        self.assertEqual(tags[33550], (scale, scale, 0))
                        self.assertEqual(tags[33922], (0, 0, 0, -10018754.171394622 + shift, 10018754.171394622 - shift, 0))
                        self.assertEqual(tags[34735][7], raster_type)
                        offsets, sizes = tags[324], tags[325]
                    data = path.read_bytes()
                    self.assertEqual(data[:2], b"II" if order == "<" else b"MM")
                    self.assertEqual(len(offsets), 16)
                    self.assertEqual(len(sizes), 16)
                    # Two block rows per source tile, interleaved across tile columns.
                    expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
                    for offset, size, height in zip(offsets, sizes, expected):
                        block = zlib.compress(struct.pack(order + "f", height + 0.25) * (256 * 256))
                        self.assertEqual(data[offset:offset + size], block)

    def test_mixed_block_sizes_preserve_every_float_pixel(self):
        for order in ("<", ">"):
            for predictor in (1, 3):
                with self.subTest(order=order, predictor=predictor), tempfile.TemporaryDirectory() as directory:
                    folder = Path(directory)
                    results = []
                    for i, block_size in enumerate((512, 256, 256, 512)):
                        tile = dict(x=1 + i % 2, y=1 + i // 2, zoom=2, filename=f"{i}.tif")
                        (folder / tile["filename"]).write_bytes(terrain_bytes(
                            tile["x"], tile["y"], 2, order=order, block_size=block_size, predictor=predictor))
                        results.append(tile)
                    path = folder / "terrain.tif"
                    terrain.merge(path, folder, results, dict(x0=1, y0=1, zoom=2, columns=2, rows=2), Mock())
                    with Image.open(path) as mosaic:
                        self.assertEqual((mosaic.tag_v2[322], mosaic.tag_v2[323]), (256, 256))
                        for i, tile in enumerate(results):
                            x, y = i % 2 * 512, i // 2 * 512
                            with Image.open(folder / tile["filename"]) as source:
                                self.assertEqual(mosaic.crop((x, y, x + 512, y + 512)).tobytes(), source.tobytes())

    def test_invalid_later_tile_leaves_previous_mosaic_intact(self):
        corruptions = {
            "wrong georeferencing": terrain_bytes(3, 1, 2),
            "truncated block": terrain_bytes(2, 1, 2)[:-1],
            "incompatible byte order": terrain_bytes(2, 1, 2, order=">"),
        }
        for name, data in corruptions.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "first.tif").write_bytes(terrain_bytes(1, 1, 2))
                (folder / "second.tif").write_bytes(data)
                path = folder / "terrain.tif"
                original = terrain_bytes(1, 1, 2)
                path.write_bytes(original)
                results = [dict(x=1, y=1, zoom=2, filename="first.tif"),
                           dict(x=2, y=1, zoom=2, filename="second.tif")]
                with self.assertRaises(ValueError):
                    terrain.merge(path, folder, results, dict(x0=1, y0=1, zoom=2, columns=2, rows=1), Mock())
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual({p.name for p in folder.iterdir()}, {"first.tif", "second.tif", "terrain.tif"})
