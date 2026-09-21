import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET

from src.common import Client
from src.osm import Source, covers


def polygon(ring, *holes):
    return dict(type="Polygon", coordinates=[ring, *holes])


class RegionTests(unittest.TestCase):
    def test_region_must_cover_rectangle_interior_and_avoid_holes(self):
        outer = [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]
        hole = [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1], [-.1, -.1]]
        self.assertTrue(covers(polygon(outer), (-1, -1, 1, 1)))
        self.assertFalse(covers(polygon(outer, hole), (-1, -1, 1, 1)))
        notch = [[-2, -2], [2, -2], [2, 2], [.1, 2], [.1, 0], [-.1, 0], [-.1, 2], [-2, 2], [-2, -2]]
        self.assertFalse(covers(polygon(notch), (-1, -1, 1, 1)))
        self.assertFalse(covers(polygon(outer), (-3, -1, 1, 1)))


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.client = Client(cache_dir=self.folder)
        self.client.get = Mock(side_effect=AssertionError("Cached extracts must work offline"))
        self.source = self.client.maps
        self.source.directory.mkdir()
        url = "https://download.geofabrik.de/test-latest.osm.pbf"
        self.path = self.source.directory / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".osm.pbf")
        index = dict(features=[dict(type="Feature", properties=dict(name="Test region", urls=dict(pbf=url)),
                                   geometry=polygon([[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]))])
        (self.source.directory / "index.json").write_text(json.dumps(index))
        self.path.write_bytes((Path(__file__).parent / "data" / "dense.osm.pbf").read_bytes())
        self.area = (-.0001, -.0001, .0001, .0005)

    def test_export_completes_ways_and_preserves_partial_relation_members(self):
        output = self.folder / "map.osm"
        metadata = self.source.export(self.area, output, lambda *args: None)
        root = ET.parse(output).getroot()
        self.assertEqual({int(n.attrib["id"]) for n in root.findall("node")}, {1, 2, 6})
        self.assertEqual({int(w.attrib["id"]) for w in root.findall("way")}, {10, 13})
        self.assertEqual({int(r.attrib["id"]) for r in root.findall("relation")}, {19, 20})
        self.assertEqual(metadata["osm_data_at"], "2026-09-20T20:00:00Z")
        self.client.get.assert_not_called()

    def test_roads_pois_and_connected_street_use_cached_region(self):
        ways, metadata = self.source.data(self.area)
        self.assertEqual([way["id"] for way in ways], [10])
        self.assertEqual(ways[0]["nodes"], [1, 2])
        self.assertEqual(ways[0]["points"][-1], (0, .001))
        self.assertEqual(metadata["osm_data_at"], "2026-09-20T20:00:00Z")
        pois, _ = self.source.data(self.area, poi=True)
        relations = {item["id"]: item["center"] for item in pois if item["type"] == "relation"}
        self.assertEqual(relations, {19: (0, .001), 20: (0, .001)})
        name, street = self.source.street(dict(id="way/10", lat=0, lon=0))
        self.assertEqual(name, "Main Street")
        self.assertEqual({w["id"] for w in street}, {10, 11})
        self.assertEqual(self.source.data((1, 1, 1.1, 1.1), poi=True)[0], [])
        self.client.get.assert_not_called()

    def test_failed_extraction_keeps_previous_map(self):
        output = self.folder / "map.osm"
        output.write_bytes(b"previous")
        with patch.object(Source, "extract", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                self.source.export(self.area, output, lambda *args: None)
        self.assertEqual(output.read_bytes(), b"previous")
        self.assertEqual(list(self.folder.glob(".map.osm-*")), [])

    def test_refresh_rejects_invalid_download_without_replacing_cached_region(self):
        original = self.path.read_bytes()
        self.client.refresh = True
        self.source.index = json.loads((self.source.directory / "index.json").read_bytes())["features"]

        def download(url, **kwargs):
            Path(kwargs["destination"]).write_bytes(b"not a PBF")

        self.client.get.side_effect = download
        with self.assertRaises(ValueError):
            self.source.region(self.area, lambda *args: None)
        self.assertEqual(self.path.read_bytes(), original)

    def test_cancellation_stops_local_processing(self):
        output = self.folder / "map.osm"
        output.write_bytes(b"previous")
        self.source.region(self.area, lambda *args: None)
        self.client.cancel = Mock(side_effect=[None, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.source.export(self.area, output, lambda *args: None)
        self.assertEqual(output.read_bytes(), b"previous")
        self.assertEqual(list(self.folder.glob(".map.osm-*")), [])
