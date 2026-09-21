import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.dashboard_tiles import Tiles
from tests.fixtures import image_bytes


class DashboardTileTests(unittest.TestCase):
    def test_native_and_cached_tiles_do_not_wait_for_generation(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor() as workers:
            root = Path(directory).resolve()
            folder = root / "capture"
            folder.mkdir()
            with Image.new("RGB", (256, 256), "red") as image:
                image.save(folder / "original.jpg")
            with Image.new("RGBA", (256, 256)) as image:
                image.save(folder / "gap.png")
            stage = dict(mode="satellite", grid=dict(zoom=2, x0=0, y0=0, columns=2, rows=2),
                         results=[dict(filename=name) for name in
                                  ("original.jpg", "gap.png", "original.jpg", "original.jpg")])
            run = dict(stages=[stage])
            tiles = Tiles(root / "cache")
            overview, _ = tiles.get(folder, run, "satellite", 1, 0, 0)
            # Only the overview needs encoding; originals remain the native tiles.
            self.assertEqual(list((root / "cache").rglob("*.png")), [overview])
            with tiles.locks["satellite"], tiles.locks["terrain"], patch.object(
                Image, "open", side_effect=AssertionError("Ready tiles must not be decoded")
            ):
                for z, x, expected in ((2, 0, folder / "original.jpg"), (2, 1, folder / "gap.png"),
                                       (1, 0, overview)):
                    with self.subTest(zoom=z, x=x):
                        result = workers.submit(tiles.get, folder, run, "satellite", z, x, 0)
                        self.assertEqual(result.result(timeout=2), (expected, True))

    def test_cached_overview_refreshes_as_capture_fills_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "capture"
            folder.mkdir()
            tiles = Tiles(root / "cache")
            stage = dict(mode="satellite", grid=dict(zoom=2, x0=0, y0=0, columns=2, rows=2), results=[])
            run = dict(stages=[stage])
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            centers = [(64, 64), (192, 64), (64, 192), (192, 192)]
            for count in range(5):
                with self.subTest(saved_tiles=count):
                    if count:
                        filename = f"{count}.png"
                        (folder / filename).write_bytes(image_bytes((256, 256), colors[count - 1]))
                        stage["results"].append(dict(filename=filename))
                    path, complete = tiles.get(folder, run, "satellite", 1, 0, 0)
                    self.assertEqual(complete, count == 4)
                    with Image.open(path) as image:
                        self.assertEqual([image.getpixel(point) for point in centers],
                                         [(*color, 255) if i < count else (0, 0, 0, 0)
                                          for i, color in enumerate(colors)])
                    # Repeating a request must return the same usable cached content.
                    original = path.read_bytes()
                    cached, cached_complete = tiles.get(folder, run, "satellite", 1, 0, 0)
                    self.assertEqual(cached.read_bytes(), original)
                    self.assertEqual(cached_complete, complete)
