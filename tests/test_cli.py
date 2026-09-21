import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import aleph
from src.common import Client
from src.osm import Source
from tests.fixtures import image_bytes


class ConfirmationTests(unittest.TestCase):
    def test_declining_or_closed_stdin_creates_no_capture_and_downloads_nothing(self):
        for answer in ("", EOFError()):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "captures"
                with (
                    patch("builtins.input", side_effect=[answer]),
                    patch.object(Client, "get", side_effect=AssertionError("Unexpected download")) as get,
                    patch.object(Source, "region", side_effect=AssertionError("Unexpected OSM request")) as region,
                    redirect_stdout(StringIO()) as stdout,
                    redirect_stderr(StringIO()) as stderr,
                ):
                    status = aleph.main(["capture", "create", "--bbox", "1", "1", "2", "2", "--sources", "satellite",
                                             "--satellite-zoom", "1", "-o", str(output)])
                self.assertEqual(status, 0)
                self.assertIn("Cancelled", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())
                get.assert_not_called()
                region.assert_not_called()


class CaptureCommandTests(unittest.TestCase):
    def test_create_plan_resume_and_export(self):
        def invoke(*args):
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
                self.assertEqual(aleph.main(["capture", *args, "--json"]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["command"], "capture")
            self.assertEqual(result["action"], args[0])
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Client, "get", side_effect=AssertionError("Planning should not download imagery")):
                planned = invoke("create", "--bbox", "1", "1", "2", "2", "--sources", "satellite",
                                 "--satellite-zoom", "1", "--plan", "-o", directory)
            self.assertEqual(planned["status"], "planned")
            folder = planned["folder"]
            with patch.object(Client, "get", return_value=image_bytes((256, 256), "red")):
                resumed = invoke("resume", folder, "--no-plan")
            self.assertEqual(resumed["status"], "complete")
            with patch.object(Client, "get", side_effect=AssertionError("Export must stay offline")):
                exported = invoke("export", folder)
            self.assertEqual(exported["folder"], folder)
            self.assertTrue((Path(folder) / "satellite.png").is_file())
