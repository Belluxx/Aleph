import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from src import capture, geo
from src.common import Client, MissingImagery
from tests.fixtures import image_bytes, metadata, terrain_bytes


def street_run():
    return dict(
        format="aleph-python", version=capture.FORMAT_VERSION,
        bounds=[-0.001, -0.001, 0.001, 0.001],
        options=capture.settings(dict(include=["streetview"])),
        started_at="2024-06-01T00:00:00+00:00", state="planned",
        stages=[dict(
            mode="streetview", results=[],
            paths=[dict(id="way-1-1", osm_id=1, name="Main Road", highway="residential",
                        layer="0", points=[[0, -0.001], [0, 0.001]])],
            samples=[dict(pano_id="panorama_0001", lat=0, lon=0, heading=90,
                          path_index=0, path_meters=111, stop_in_path=1)],
        )],
    )


class CaptureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.progress = Mock()
        self.client = Mock(spec=Client)
        self.client.get.side_effect = AssertionError("Unexpected image download")
        self.client.maps.export.side_effect = AssertionError("Unexpected OSM download")

    def test_spacing_and_delay_require_valid_numbers_without_workload_caps(self):
        for step in (0.5, 2000):
            options = capture.settings(dict(step=step, delay=60))
            self.assertEqual((options["step"], options["delay"]), (step, 60))
        self.assertEqual(capture.settings(dict(delay=0))["delay"], 0)
        for key, values in (("step", (0, -1, float("inf"), float("nan"))),
                            ("delay", (-1, float("inf"), float("nan")))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    capture.settings({key: value})

    def test_interrupted_second_side_resumes_from_saved_checkpoint(self):
        photo = image_bytes((1024, 576), "red")
        for error, state in ((KeyboardInterrupt(), "stopped"), (OSError("offline"), "failed")):
            with self.subTest(state=state):
                folder = self.folder / state
                folder.mkdir()
                run = street_run()
                self.client.get.side_effect = [metadata(), photo, error]
                with self.assertRaises(type(error)) as raised:
                    capture.download(run, folder, self.client, self.progress)
                self.assertIs(raised.exception, error)
                saved = capture.load(folder)
                self.assertEqual(saved["state"], state)
                results = saved["stages"][0]["results"]
                self.assertEqual([(p["side"], p["heading"]) for p in results], [("left", 0)])
                first = folder / results[0]["filename"]
                original = first.read_bytes()
                self.assertTrue(saved["exports_saved"])

                self.client.get.reset_mock()
                self.client.get.side_effect = [metadata(), photo]
                capture.download(saved, folder, self.client, self.progress)
                completed = capture.load(folder)
                self.assertEqual(completed["state"], "complete")
                self.assertNotIn("error", completed)
                self.assertIn("finished_at", completed)
                self.assertEqual(first.read_bytes(), original)
                self.assertEqual([(p["side"], p["heading"]) for p in completed["stages"][0]["results"]],
                                 [("left", 0), ("right", 180)])
                self.assertEqual(self.client.get.call_count, 2)  # Metadata, then only the missing side.
                request = self.client.get.call_args_list[-1].args[0]
                self.assertEqual(parse_qs(urlsplit(request).query)["yaw"], ["180"])
                exported = json.loads((folder / "streetview/photos.geojson").read_bytes())
                self.assertEqual(len(exported["features"]), 2)

    def test_missing_imagery_skips_one_side_and_exports_only_saved_photos(self):
        run = street_run()
        self.client.get.side_effect = [metadata(), MissingImagery("removed"), image_bytes((1024, 576), "blue")]
        capture.download(run, self.folder, self.client, self.progress)
        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "complete")
        self.assertEqual([p["status"] for p in saved["stages"][0]["results"]], ["skipped", "saved"])
        exported = json.loads((self.folder / "streetview/photos.geojson").read_bytes())
        self.assertEqual([p["properties"]["side"] for p in exported["features"]], ["right"])

    def test_changed_panorama_or_invalid_image_does_not_advance_checkpoint(self):
        responses = {
            "changed identity": [metadata(pano_id="panorama_0002")],
            "moved panorama": [metadata(lat=0.001)],
            "wrong dimensions": [metadata(), image_bytes((256, 256), "red")],
            "undecodable image": [metadata(), b"not an image"],
        }
        for name, response in responses.items():
            with self.subTest(case=name):
                run = street_run()
                self.client.get.reset_mock()
                self.client.get.side_effect = response
                with self.assertRaises((ValueError, OSError)):
                    capture.download(run, self.folder, self.client, self.progress)
                saved = capture.load(self.folder)
                self.assertEqual(saved["state"], "failed")
                self.assertEqual(saved["stages"][0]["results"], [])
                self.assertEqual(self.client.get.call_count, len(response))
                self.assertFalse((self.folder / "streetview/photos").exists())

    def test_failed_satellite_export_resumes_offline_and_crops_tiles_correctly(self):
        north, west = geo.coordinate(384, 384, 2)
        south, east = geo.coordinate(640, 640, 2)
        run = capture.plan(self.client, [south, west, north, east],
                           dict(include=["satellite"], satellite_zoom=2, satellite_format="png"), self.progress)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.client.get.side_effect = [image_bytes((256, 256), color) for color in colors]
        previous = image_bytes((1, 1), "black")
        (self.folder / "satellite.tif").write_bytes(previous)
        with patch("src.satellite.zlib.compress", side_effect=OSError("export disk full")):
            with self.assertRaisesRegex(OSError, "export disk full"):
                capture.download(run, self.folder, self.client, self.progress)
        self.assertEqual((self.folder / "satellite.tif").read_bytes(), previous)
        self.assertFalse(list(self.folder.glob(".satellite.tif-*")))
        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "failed")
        self.assertFalse(saved["exports_saved"])
        self.assertEqual(len(saved["stages"][0]["results"]), 4)

        self.client.get.reset_mock()
        self.client.get.side_effect = AssertionError("Saved tiles must not be downloaded again")
        (self.folder / "satellite.png").write_bytes(previous)
        with patch("src.satellite.zlib.compressobj", side_effect=OSError("PNG disk full")):
            with self.assertRaisesRegex(OSError, "PNG disk full"):
                capture.download(saved, self.folder, self.client, self.progress)
        for extension in ("tif", "png"):
            self.assertEqual((self.folder / f"satellite.{extension}").read_bytes(), previous)
            self.assertFalse(list(self.folder.glob(f".satellite.{extension}-*")))
        self.assertFalse(capture.load(self.folder)["exports_saved"])
        capture.download(saved, self.folder, self.client, self.progress)
        self.client.get.assert_not_called()
        self.assertEqual(capture.load(self.folder)["state"], "complete")
        with Image.open(self.folder / "satellite.tif") as mosaic:
            self.assertEqual(mosaic.size, (256, 256))
            self.assertEqual([mosaic.getpixel(point) for point in ((0, 0), (255, 0), (0, 255), (255, 255))],
                             [(*color, 255) for color in colors])

    def test_satellite_cog_matches_mosaic_for_empty_partial_and_complete_captures(self):
        north, west = geo.coordinate(384, 384, 3)
        south, east = geo.coordinate(896, 984, 3)
        run = capture.plan(self.client, [south, west, north, east],
                           dict(include=["satellite"], satellite_zoom=3), self.progress)
        stage = run["stages"][0]
        patches = []
        for i, tile in enumerate(geo.tiles(stage["grid"])):
            filename = f"patch-{i}.png"
            # Vary both coordinates and alpha to expose cropping and paste errors.
            pixels = bytes(v for y in range(256) for x in range(256)
                           for v in (x, y, i * 20, (x + y) % 256))
            with Image.frombytes("RGBA", (256, 256), pixels) as image:
                image.save(self.folder / filename)
            patches.append(dict(tile, filename=filename))
        for count in (0, 4, 9):
            with self.subTest(saved_tiles=count), Image.new("RGBA", (512, 600)) as expected:
                stage["results"] = patches[:count]
                for photo in stage["results"]:
                    with Image.open(self.folder / photo["filename"]) as image:
                        expected.paste(image, (photo["x"] * 256 - 384, photo["y"] * 256 - 384))
                with patch.object(Image, "new", wraps=Image.new) as allocate:
                    capture.export(run, self.folder, self.progress)
                self.assertTrue(allocate.called)
                self.assertTrue(all(max(call.args[1]) <= 513 for call in allocate.call_args_list))
                with Image.open(self.folder / "satellite.tif") as actual:
                    self.assertEqual(actual.format, "TIFF")
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.mode, "RGBA")
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                with Image.open(self.folder / "satellite.png") as actual:
                    actual.verify()
                with Image.open(self.folder / "satellite.png") as actual:
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.mode, "RGBA")
                    self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_satellite_tile_formats_preserve_source_bytes_and_transparent_gaps(self):
        north, west = geo.coordinate(256, 256, 2)
        south, east = geo.coordinate(512, 512, 2)
        for source in ("JPEG", "PNG", "missing"):
            for extension in ("jpg", "png"):
                with self.subTest(source=source, requested=extension):
                    folder = self.folder / f"{source}-{extension}"
                    folder.mkdir()
                    run = capture.plan(self.client, [south, west, north, east],
                                       dict(include=["satellite"], satellite_zoom=2,
                                            satellite_format=extension), self.progress)
                    with BytesIO() as buffer, Image.new("RGB", (256, 256), (72, 105, 210)) as image:
                        image.save(buffer, format="PNG" if source == "missing" else source)
                        data = buffer.getvalue()
                    self.client.get.side_effect = [MissingImagery("missing") if source == "missing" else data]
                    capture.download(run, folder, self.client, self.progress)
                    filename = run["stages"][0]["results"][0]["filename"]
                    expected = "png" if source == "missing" else extension
                    self.assertEqual(Path(filename).suffix, "." + expected)
                    with Image.open(folder / filename) as tile, tile.convert("RGBA") as rgba:
                        self.assertEqual(tile.format, "JPEG" if expected == "jpg" else "PNG")
                        if source == "missing":
                            self.assertEqual(rgba.getchannel("A").getextrema(), (0, 0))
                        elif tile.format == source:
                            self.assertEqual((folder / filename).read_bytes(), data)
                        for name in ("satellite.png", "satellite.tif"):
                            with Image.open(folder / name) as merged:
                                self.assertEqual(merged.tobytes(), rgba.tobytes())

    def test_satellite_404_leaves_transparent_tile_without_retrying_on_resume(self):
        north, west = geo.coordinate(384, 384, 2)
        south, east = geo.coordinate(640, 640, 2)
        run = capture.plan(self.client, [south, west, north, east],
                           dict(include=["satellite"], satellite_zoom=2, satellite_format="png"), self.progress)
        missing = HTTPError("https://mt1.google.com/vt/", 404, "Not Found", {}, BytesIO())
        responses = [BytesIO(image_bytes((256, 256), "red")), missing,
                     BytesIO(image_bytes((256, 256), "blue")),
                     BytesIO(image_bytes((256, 256), "green"))]
        with patch("src.common.urlopen", side_effect=responses) as get:
            capture.download(run, self.folder, Client(), self.progress)
            self.assertEqual(get.call_count, 4)
        self.assertTrue(missing.fp.closed)
        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "complete")
        original = (self.folder / "satellite.tif").read_bytes()
        with Image.open(self.folder / "satellite.tif") as mosaic:
            self.assertEqual(mosaic.size, (256, 256))
            self.assertEqual(mosaic.getpixel((0, 0)), (255, 0, 0, 255))
            self.assertEqual(mosaic.crop((128, 0, 256, 128)).getchannel("A").getextrema(), (0, 0))
            self.assertEqual(mosaic.getpixel((0, 255)), (0, 0, 255, 255))
            self.assertEqual(mosaic.getpixel((255, 255)), (0, 128, 0, 255))
        self.client.get.reset_mock()
        capture.download(saved, self.folder, self.client, self.progress)
        self.client.get.assert_not_called()
        self.assertEqual((self.folder / "satellite.tif").read_bytes(), original)

    def test_osm_failure_still_saves_terrain_and_resume_fetches_only_map(self):
        run = capture.plan(self.client, [1, 1, 2, 2],
                           dict(include=["osm"], terrain_zoom=1), self.progress)
        failure = OSError("Geofabrik unavailable")
        self.client.maps.export.side_effect = failure

        def download_terrain(address):
            self.progress.assert_called_with("Downloading terrain tiles (map failed: Geofabrik unavailable)", 0, 1)
            return terrain_bytes(1, 0, 1)

        self.client.get.side_effect = download_terrain
        with self.assertRaises(OSError) as raised:
            capture.download(run, self.folder, self.client, self.progress)
        self.assertIs(raised.exception, failure)
        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "failed")
        self.assertTrue(saved["exports_saved"])
        self.assertEqual(len(saved["stages"][0]["results"]), 1)
        mosaic = (self.folder / "terrain.tif").read_bytes()
        self.assertFalse((self.folder / "map.osm").exists())

        osm = b'<osm version="0.6"><node id="1" lat="1" lon="1"/><way id="2"><nd ref="1"/></way></osm>'
        def export(area, path, progress):
            path.write_bytes(osm)
            return dict(source_url="https://download.geofabrik.de/test.osm.pbf", osm_data_at="2026-09-20T20:00:00Z")

        self.client.maps.export.side_effect = export
        self.client.get.reset_mock()
        self.client.get.side_effect = AssertionError("Completed terrain must not be downloaded again")
        capture.download(saved, self.folder, self.client, self.progress)
        self.client.get.assert_not_called()
        self.assertEqual((self.folder / "map.osm").read_bytes(), osm)
        self.assertEqual((self.folder / "terrain.tif").read_bytes(), mosaic)
        completed = capture.load(self.folder)
        self.assertEqual(completed["state"], "complete")
        self.assertEqual(len(completed["stages"][0]["results"]), 2)
