"""Interrupted transfers must never replace a usable cached file."""

import unittest
from io import BytesIO
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.common import Client, contained


def response(data, length):
    stream = BytesIO(data)
    stream.headers = {"Content-Length": str(length)}
    return stream


class DownloadTests(unittest.TestCase):
    def test_truncated_download_preserves_previous_file_and_cleans_up(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "region.pbf"
            target.write_bytes(b"previous snapshot")
            with (
                patch("src.common.urlopen", side_effect=lambda *a, **k: response(b"short", 20)) as get,
                patch("src.common.time.sleep"),
                patch("src.common.time.monotonic", side_effect=count(step=10)),
                self.assertRaisesRegex(OSError, "Incomplete download"),
            ):
                Client().get("https://example.test/region", destination=target)
            self.assertEqual(get.call_count, 4)
            self.assertEqual(target.read_bytes(), b"previous snapshot")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_retry_replaces_file_only_after_complete_download(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "region.pbf"
            target.write_bytes(b"previous snapshot")

            def progress(downloaded, expected):
                self.assertEqual(target.read_bytes(), b"previous snapshot")

            with (
                patch("src.common.urlopen", side_effect=[response(b"short", 20), response(b"complete", 8)]),
                patch("src.common.time.sleep"),
                patch("src.common.time.monotonic", side_effect=count(step=10)),
            ):
                result = Client().get("https://example.test/region", destination=target, progress=progress)
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_interrupt_during_streaming_preserves_file_without_retry(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "region.pbf"
            target.write_bytes(b"previous snapshot")

            def interrupt(*args):
                raise KeyboardInterrupt

            with (
                patch("src.common.urlopen", return_value=response(b"new snapshot", 12)) as get,
                self.assertRaises(KeyboardInterrupt),
            ):
                Client().get("https://example.test/region", destination=target, progress=interrupt)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(target.read_bytes(), b"previous snapshot")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_containment_rejects_traversal_and_symlink_escape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            root.mkdir()
            (root / "escape").symlink_to(Path(directory), target_is_directory=True)
            for name in ("../secret", "/etc/passwd", "..\\secret", "escape/secret", ".", ""):
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    contained(root, name)
            self.assertEqual(contained(root, "photos/view.jpg"), root.resolve() / "photos/view.jpg")
