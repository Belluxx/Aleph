import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from src import common


class PersistenceTests(unittest.TestCase):
    def test_failed_checkpoint_write_keeps_previous_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            common.write_json(path, {"saved": 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                common.write_json(path, {"saved": 2, "invalid": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    common.write_json(path, {"saved": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])
            common.write_json(path, {"saved": 2})
            self.assertEqual(json.loads(path.read_bytes()), {"saved": 2})

    def test_output_paths_cannot_escape_through_traversal_or_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            (capture / "linked").symlink_to(root, target_is_directory=True)
            for name in ("../secret", str(root / "secret"), "linked/secret", "..\\secret", "", "."):
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    common.contained(capture, name)
            self.assertEqual(common.contained(capture, "photos/one.jpg"), capture.resolve() / "photos/one.jpg")


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.elapsed = 0
        self.sleeps = []

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.elapsed += seconds

        self.enterContext(patch.object(common.time, "monotonic", side_effect=lambda: self.elapsed))
        self.enterContext(patch.object(common.time, "sleep", side_effect=sleep))
        self.urlopen = self.enterContext(patch.object(common, "urlopen"))

    def test_response_read_failure_retries_and_respects_delay_and_backoff(self):
        class BrokenResponse(BytesIO):
            def read(self):
                raise OSError("connection dropped during read")

        broken = BrokenResponse()
        self.urlopen.side_effect = [
            broken, OSError("offline"), OSError("offline"), BytesIO(b"complete"),
        ]
        self.assertEqual(common.Client(delay=3).get("https://example.test/tile"), b"complete")
        self.assertTrue(broken.closed)
        self.assertEqual(self.urlopen.call_count, 4)
        self.assertEqual(self.sleeps, [3, 3, 4])

    def test_only_opted_in_404_becomes_missing_imagery_without_retries(self):
        for code, missing_ok, expected in ((404, True, common.MissingImagery), (404, False, OSError), (500, True, OSError)):
            with self.subTest(code=code, missing_ok=missing_ok):
                attempts = 1 if code == 404 and missing_ok else 4
                errors = [HTTPError("https://example.test/tile", code, "unavailable", {}, BytesIO()) for _ in range(attempts)]
                self.urlopen.reset_mock(side_effect=True)
                self.urlopen.side_effect = errors
                self.sleeps.clear()
                with self.assertRaises(expected) as raised:
                    common.Client().get("https://example.test/tile", missing_ok=missing_ok)
                self.assertIs(raised.exception.__cause__, errors[-1])
                self.assertEqual(self.urlopen.call_count, attempts)
                self.assertEqual(self.sleeps, [] if attempts == 1 else [1, 2, 4])
                self.assertTrue(all(error.fp.closed for error in errors))

    def test_cancellation_during_backoff_stops_before_another_request(self):
        def cancel():
            if self.elapsed:
                raise KeyboardInterrupt()

        self.urlopen.side_effect = OSError("offline")
        with self.assertRaises(KeyboardInterrupt):
            common.Client(cancel=cancel).get("https://example.test/tile")
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertLessEqual(self.elapsed, 0.2)

    def test_streamed_download_retries_truncation_and_preserves_old_file_on_cancellation(self):
        def response(data):
            value = BytesIO(data)
            value.headers = {"Content-Length": "8"}
            return value

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region.pbf"
            path.write_bytes(b"previous")
            self.urlopen.side_effect = [response(b"short"), response(b"complete")]
            common.Client().get("https://example.test/region.pbf", destination=path)
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(self.urlopen.call_count, 2)
            self.urlopen.side_effect = [response(b"replaced")]

            def cancel(*args):
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                common.Client().get("https://example.test/region.pbf", destination=path, progress=cancel)
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(list(path.parent.iterdir()), [path])
