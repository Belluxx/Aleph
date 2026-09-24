"""Exercise real checkpoints and exports with only the provider replaced."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from src import capture, streetview
from src.common import MissingImagery
from src.geo import coordinate


def image_bytes(size=(256, 256)):
    with BytesIO() as buffer, Image.new("RGB", size, "red") as image:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def metadata(pano_id="test_panorama_123", lat=45.0, lon=9.0):
    # Minimal synthetic photometa response, including Google's XSSI prefix.
    return b")]}'\n" + json.dumps([None, [[
        [1], [2, pano_id], None, None, None,
        [[None, [[None, None, lat, lon]]]],
        [None, None, None, None, None, None, None, [2024, 2]],
    ]]]).encode()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.progress = Mock()

    def satellite_run(self, columns=1):
        north, west = coordinate(1024, 1024, 4)
        south, east = coordinate(1024 + columns * 256, 1280, 4)
        return capture.plan(None, (south, west, north, east),
                            {"include": ["satellite"], "satellite_zoom": 4}, self.progress)

    def test_resume_keeps_saved_tiles_and_transparent_missing_imagery(self):
        run = self.satellite_run(columns=3)
        client = Mock()
        client.get.side_effect = [image_bytes(), MissingImagery("No tile"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            capture.download(run, self.folder, client, self.progress)

        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "stopped")
        results = saved["stages"][0]["results"]
        self.assertEqual(len(results), 2)
        original = (self.folder / results[0]["filename"]).read_bytes()
        with Image.open(self.folder / "satellite.png") as image:
            self.assertEqual(image.getpixel((300, 100)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((600, 100)), (0, 0, 0, 0))

        client.get.reset_mock(side_effect=True)
        client.get.return_value = image_bytes()
        capture.download(saved, self.folder, client, self.progress)
        client.get.assert_called_once()
        self.assertIn("&x=6&y=4&z=4", client.get.call_args.args[0])
        completed = capture.load(self.folder)
        self.assertEqual(completed["state"], "complete")
        self.assertNotIn("error", completed)
        self.assertTrue(completed["exports_saved"])
        self.assertEqual(len(completed["stages"][0]["results"]), 3)
        self.assertEqual((self.folder / results[0]["filename"]).read_bytes(), original)
        with Image.open(self.folder / "satellite.png") as image:
            self.assertEqual(image.getpixel((300, 100)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((600, 100))[3], 255)

    def test_failed_export_can_resume_without_redownloading(self):
        run = self.satellite_run()
        client = Mock()
        client.get.return_value = image_bytes()
        with (
            patch("src.capture.satellite.merge", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            capture.download(run, self.folder, client, self.progress)

        saved = capture.load(self.folder)
        self.assertEqual(saved["state"], "failed")
        self.assertFalse(saved["exports_saved"])
        self.assertEqual(len(saved["stages"][0]["results"]), 1)
        client.get.reset_mock()
        capture.download(saved, self.folder, client, self.progress)
        client.get.assert_not_called()
        self.assertEqual(capture.load(self.folder)["state"], "complete")
        self.assertTrue((self.folder / "satellite.tif").is_file())

    def street_run(self):
        return dict(
            format="aleph-python", version=capture.FORMAT_VERSION,
            bounds=[44.99, 8.99, 45.01, 9.01],
            options=capture.settings({"include": ["streetview"]}),
            state="planned", stages=[dict(
                mode="streetview", results=[],
                paths=[dict(id="way-1-1", name="Test street", osm_id=1, highway="residential", layer="0")],
                samples=[dict(pano_id="test_panorama_123", lat=45.0, lon=9.0,
                              heading=350, path_index=0, stop_in_path=1)],
            )],
        )

    def test_resume_between_street_sides_after_missing_left_photo(self):
        run = self.street_run()
        client = Mock()
        client.get.side_effect = [metadata(), MissingImagery("No left view"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            capture.download(run, self.folder, client, self.progress)
        saved = capture.load(self.folder)
        self.assertEqual(saved["stages"][0]["results"][0]["status"], "skipped")

        client.get.reset_mock()
        client.get.side_effect = [metadata(), image_bytes((1024, 576))]
        capture.download(saved, self.folder, client, self.progress)
        photos = capture.load(self.folder)["stages"][0]["results"]
        self.assertEqual([(p["side"], p["status"]) for p in photos],
                         [("left", "skipped"), ("right", "saved")])
        self.assertEqual(photos[1]["sequence"], 2)
        self.assertEqual(photos[1]["heading"], 80)
        self.assertEqual(client.get.call_count, 2)
        query = parse_qs(urlsplit(client.get.call_args.args[0]).query)
        self.assertEqual(query["yaw"], ["80"])
        exported = json.loads((self.folder / "streetview/photos.geojson").read_text())
        self.assertEqual(len(exported["features"]), 1)
        self.assertEqual(exported["features"][0]["geometry"]["coordinates"], [9.0, 45.0])

    def test_changed_panorama_identity_or_position_stops_before_saving_photo(self):
        for changed in (metadata(pano_id="other_panorama_123"), metadata(lat=45.001)):
            with self.subTest(metadata=changed):
                run = self.street_run()
                client = Mock()
                client.get.return_value = changed
                with self.assertRaisesRegex(ValueError, "identity or position changed"):
                    capture.download(run, self.folder, client, self.progress)
                client.get.assert_called_once()
                self.assertEqual(capture.load(self.folder)["stages"][0]["results"], [])


class GoogleResponseTests(unittest.TestCase):
    def test_coverage_skips_sentinels_and_non_google_panoramas(self):
        payload = [None, [None, [
            [[1]],
            [[[3, "user_panorama_123"]]],
            [[[2, "test_panorama_123"], None, [[None, None, 45.0, 9.0]]]],
        ]]]
        self.assertEqual(streetview.parse_coverage(b")]}'\n" + json.dumps(payload).encode()),
                         [dict(pano_id="test_panorama_123", lat=45.0, lon=9.0)])
        self.assertEqual(streetview.parse_coverage(b"[null, null]"), [])
        with self.assertRaisesRegex(ValueError, "Unreadable Google panorama coverage"):
            streetview.parse_coverage(b"[null, [null]]")

    def test_metadata_reads_prefixed_response_and_optional_date(self):
        self.assertEqual(streetview.parse_metadata(metadata()), dict(
            pano_id="test_panorama_123", lat=45.0, lon=9.0, imagery_date="2024-02"))
        payload = json.loads(metadata()[5:])
        payload[1][0].pop()
        self.assertIsNone(streetview.parse_metadata(json.dumps(payload).encode())["imagery_date"])

    def test_missing_panorama_is_distinct_from_changed_response_schema(self):
        with self.assertRaises(MissingImagery):
            streetview.parse_metadata(b'[null, [[[2]]]]')
        for payload in (b'[null, [[[9]]]]', b'[null, []]', b'<html>rate limited</html>'):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Unreadable Google"):
                streetview.parse_metadata(payload)
