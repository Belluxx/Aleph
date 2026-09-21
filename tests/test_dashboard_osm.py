import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from src.dashboard import Catalog, Handler
from src.dashboard_osm import display_osm
from src.geo import signed_area


class DashboardOSMTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.folder / "map.osm"
        self.root = ET.Element("osm", version="0.6")

    def nodes(self, points):
        for identity, (lon, lat) in points.items():
            ET.SubElement(self.root, "node", id=str(identity), lon=str(lon), lat=str(lat))

    def way(self, identity, refs, **tags):
        element = ET.SubElement(self.root, "way", id=str(identity), user="unused metadata")
        for ref in refs:
            ET.SubElement(element, "nd", ref=str(ref))
        for key, value in tags.items():
            ET.SubElement(element, "tag", k=key, v=value)

    def relation(self, identity, members, **tags):
        element = ET.SubElement(self.root, "relation", id=str(identity))
        for ref, role in members:
            ET.SubElement(element, "member", type="way", ref=str(ref), role=role)
        for key, value in tags.items():
            ET.SubElement(element, "tag", k=key, v=value)

    def save(self):
        ET.ElementTree(self.root).write(self.path)

    def test_only_rendered_geometry_and_properties_with_building_popup_fields(self):
        self.nodes({1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)})
        ring = [1, 2, 3, 4, 1]
        self.way(1, ring, building="yes", name="Library", height="30 ft", min_height="50",
                 amenity="library", source="survey", **{"addr:street": "Unused"})
        self.way(2, ring, landuse="forest", name="Unused forest name")
        self.way(3, ring, natural="water", water="pond", name="Unused pond name")
        self.way(4, ring, highway="residential", junction="roundabout", name="Unused road name")
        self.way(5, [1, 2], waterway="stream", name="Unused stream name")
        self.way(6, ring, amenity="parking")
        self.way(7, ring, building="no")
        self.way(8, ring, highway="pedestrian", area="yes")
        self.way(9, ring, natural="wood", area="no")
        self.way(10, [1, 999], highway="service")  # Never bridge missing nodes.
        self.way(11, ring, building="house", **{"building:levels": "2"})
        self.way(12, ring, building="shed")
        self.save()
        features = display_osm(self.path)["features"]
        self.assertEqual(len(features), 7)
        building, forest, pond, road, stream, house, shed = features
        self.assertEqual(building["geometry"]["type"], "Polygon")
        self.assertEqual(building["properties"], dict(building="yes", name="Library", _height=9.144,
                                                     _base=9.144, _estimated=False))
        self.assertEqual(forest["properties"], {"land": True})
        self.assertEqual(pond["properties"], {"water": True})
        self.assertEqual(road["geometry"]["type"], "LineString")
        self.assertEqual(road["properties"], {"road": True})
        self.assertEqual(stream["properties"], {"waterway": True})
        self.assertEqual(house["properties"]["_height"], 6)
        self.assertFalse(house["properties"]["_estimated"])
        self.assertEqual(shed["properties"]["_height"], 9)
        self.assertTrue(shed["properties"]["_estimated"])

    def test_split_reversed_multipolygon_with_hole_and_incomplete_distant_member(self):
        self.nodes({1: (0, 0), 2: (8, 0), 3: (8, 8), 4: (0, 8),
                    5: (2, 2), 6: (4, 2), 7: (4, 4), 8: (2, 4),
                    9: (10, 0), 10: (12, 0), 11: (12, 2), 12: (10, 2)})
        self.way(1, [1, 2, 3])
        self.way(2, [1, 4, 3])
        self.way(3, [5, 6, 7, 8, 5], natural="water")
        self.way(4, [9, 10, 11, 12, 9], natural="water")
        self.relation(1, [(2, "outer"), (3, "inner"), (1, "outer"), (4, "outer"), (999, "outer")],
                      type="multipolygon", natural="water", name="Unused")
        self.save()
        features = display_osm(self.path)["features"]
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["properties"], {"water": True})
        geometry = features[0]["geometry"]
        self.assertEqual(geometry["type"], "MultiPolygon")
        polygons = geometry["coordinates"]
        self.assertEqual(sorted(len(p) for p in polygons), [1, 2])
        self.assertEqual(sum(signed_area(r) for p in polygons for r in p), 64)
        self.assertTrue(all(signed_area(p[0]) > 0 for p in polygons))
        self.assertTrue(all(signed_area(r) < 0 for p in polygons for r in p[1:]))

    def test_outer_way_tags_and_independently_tagged_inner_are_preserved(self):
        self.nodes({1: (0, 0), 2: (8, 0), 3: (8, 8), 4: (0, 8),
                    5: (2, 2), 6: (4, 2), 7: (4, 4), 8: (2, 4)})
        self.way(1, [1, 2, 3, 4, 1], landuse="forest")
        self.way(2, [5, 6, 7, 8, 5], natural="water")
        self.relation(1, [(1, "outer"), (2, "inner")], type="multipolygon")
        self.save()
        features = display_osm(self.path)["features"]
        self.assertEqual([f["properties"] for f in features], [{"land": True}, {"water": True}])
        self.assertEqual(len(features[0]["geometry"]["coordinates"][0]), 2)

    def test_endpoint_reconverts_without_cache_and_reports_invalid_xml(self):
        server = SimpleNamespace(catalog=Catalog(self.folder.parent), hosts={"localhost"})

        def request():
            handler = object.__new__(Handler)
            handler.server = server
            handler.headers = {"Host": "localhost"}
            handler.command, handler.request_version = "GET", "HTTP/1.1"
            handler.path = f"/api/captures/{self.folder.name}/osm.geojson"
            handler.requestline = f"GET {handler.path} HTTP/1.1"
            handler.wfile = BytesIO()
            handler.dispatch()
            headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
            self.assertIn(b"Cache-Control: no-store", headers)
            return headers, json.loads(body)

        self.save()
        headers, data = request()
        self.assertIn(b"200 OK", headers)
        self.assertEqual(data, dict(type="FeatureCollection", features=[]))
        self.nodes({1: (0, 0), 2: (1, 1)})
        self.way(1, [1, 2], highway="service")
        self.save()
        self.assertEqual(len(request()[1]["features"]), 1)
        self.assertEqual(list(self.folder.iterdir()), [self.path])
        self.assertEqual(server.catalog.cache, {})
        self.path.write_text("<osm><way>")
        headers, data = request()
        self.assertIn(b"400 Bad Request", headers)
        self.assertIn("invalid XML", data["error"])
