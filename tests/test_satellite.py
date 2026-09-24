"""Check COG layout and pixels with Pillow's independent TIFF reader."""

import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src import geo, satellite


class SatelliteTests(unittest.TestCase):
    def test_cog_layout_georeferencing_and_overviews_in_classic_and_bigtiff(self):
        # Odd dimensions and a crop across source tiles expose overview seams,
        # padded-edge darkening, and incorrect georeferencing to whole tile edges.
        north, west = geo.coordinate(383, 391, 4)
        south, east = geo.coordinate(1408, 1160, 4)
        grid = geo.grid((south, west, north, east), 4)
        for big in (False, True):
            with self.subTest(big=big), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                stage = dict(grid=grid, results=[])
                expected = self.enterContext(Image.new("RGBA", (1025, 769)))
                for i, tile in enumerate(geo.tiles(grid)):
                    name = f"{i}.png"
                    # Transparent red must not bleed into the blue overview.
                    color = (255, 0, 0, 0) if i % 3 == 0 else (0, 80 + i * 5, 200, 255)
                    with Image.new("RGBA", (256, 256), color) as image:
                        image.save(folder / name)
                        expected.paste(image, (tile["x"] * 256 - grid["left"],
                                               tile["y"] * 256 - grid["top"]))
                    stage["results"].append(dict(tile, filename=name))
                with patch.object(satellite, "TIFF_LIMIT", 0 if big else 2**32):
                    satellite.merge(stage, folder, Mock())
                path = folder / "satellite.tif"
                data = path.read_bytes()
                self.assertEqual(struct.unpack_from("<H", data, 2)[0], 43 if big else 42)
                ifds, offsets, ends, metadata_ends = [], [], [], []
                with Image.open(path) as image:
                    self.assertEqual(image.n_frames, 4)
                    for frame, size in enumerate(((1025, 769), (513, 385), (257, 193), (129, 97))):
                        image.seek(frame)
                        tags = image.tag_v2
                        self.assertEqual(image.size, size)
                        self.assertEqual(image.mode, "RGBA")
                        self.assertEqual(tags[254], int(frame != 0))
                        self.assertEqual((tags[322], tags[323]), (256, 256))
                        self.assertEqual(tags[259], 8)  # Deflate, lossless.
                        self.assertEqual(tags[338], (2,))  # Unassociated alpha.
                        if frame == 0:
                            scale = 2 * math.pi * geo.MERCATOR_RADIUS / 4096
                            self.assertEqual(tags[33550], (scale, scale, 0))
                            self.assertEqual(tags[33922], (0, 0, 0, 383 * scale - math.pi * geo.MERCATOR_RADIUS,
                                                         math.pi * geo.MERCATOR_RADIUS - 391 * scale, 0))
                            self.assertEqual(tags[34735], (1, 1, 0, 3, 1024, 0, 1, 1,
                                                          1025, 0, 1, 1, 3072, 0, 1, 3857))
                        else:
                            self.assertNotIn(33550, tags)  # Overviews inherit the full image extent.
                            smaller = expected.resize(size, Image.Resampling.BOX)
                            expected.close()
                            expected = self.enterContext(smaller)
                        self.assertEqual(image.tobytes(), expected.tobytes())
                        tile_offsets, tile_sizes = tags[324], tags[325]
                        self.assertEqual(len(tile_offsets), math.ceil(size[0] / 256) * math.ceil(size[1] / 256))
                        self.assertTrue(all(b == a + count for a, b, count in
                                            zip(tile_offsets, tile_offsets[1:], tile_sizes)))
                        offsets.append(tile_offsets[0])
                        ends.append(tile_offsets[-1] + tile_sizes[-1])
                    # Inspect byte positions independently of the writer. All IFDs
                    # and out-of-line tag values must precede the first tile.
                    position = struct.unpack_from("<Q" if big else "<I", data, 8 if big else 4)[0]
                    while position:
                        ifds.append(position)
                        self.assertEqual(position % (8 if big else 2), 0)
                        count = struct.unpack_from("<Q" if big else "<H", data, position)[0]
                        cursor = position + (8 if big else 2)
                        for _ in range(count):
                            _, kind, length, value = struct.unpack_from("<HHQQ" if big else "<HHII", data, cursor)
                            size = {3: 2, 4: 4, 12: 8, 16: 8}[kind] * length
                            if size > (8 if big else 4):
                                metadata_ends.append(value + size)
                            cursor += 20 if big else 12
                        metadata_ends.append(cursor + (8 if big else 4))
                        position = struct.unpack_from("<Q" if big else "<I", data, cursor)[0]
                self.assertEqual(len(ifds), 4)
                self.assertEqual(ifds, sorted(ifds))
                self.assertLessEqual(max(metadata_ends), min(offsets))
                self.assertEqual(offsets, sorted(offsets, reverse=True))
                self.assertEqual(ends[1:], offsets[:-1])
                self.assertEqual(ends[0], len(data))

    def test_bigtiff_offsets_cross_four_gib_without_allocating_pixel_data(self):
        full, overview = satellite.Level(768, 512), satellite.Level(256, 171)
        full.sizes.extend([2**30] * 6)
        overview.sizes.append(100)
        data = satellite.header([full, overview], dict(left=0, top=0, zoom=1))
        self.assertEqual(data[:4], b"II+\0")
        # IFD 0: read LONG8 tile offsets directly from its directory.
        count = struct.unpack_from("<Q", data, 16)[0]
        entries = [struct.unpack_from("<HHQQ", data, 24 + i * 20) for i in range(count)]
        _, kind, count, pointer = next(entry for entry in entries if entry[0] == 324)
        self.assertEqual((kind, count), (16, 6))
        offsets = struct.unpack_from("<6Q", data, pointer)
        self.assertEqual(offsets[0], len(data) + 100)
        self.assertGreater(offsets[-1], 2**32)
