import json
import unittest

from src import geo, streetview
from src.common import MissingImagery
from tests.fixtures import metadata


class StreetViewTests(unittest.TestCase):
    def test_coverage_distinguishes_empty_tiles_from_unreadable_responses(self):
        row = [[[2, "panorama_0001"], None, [[None, None, 12, 34]]]]
        data = b")]}'\n" + json.dumps([None, [None, [row, [[1]], [[[3, "user_photo"]]]]]]).encode()
        self.assertEqual(streetview.parse_coverage(data), [dict(pano_id="panorama_0001", lat=12, lon=34)])
        for empty in (b"[null,null]", b"[null,[]]", b"[null,[null,[]]]"):
            with self.subTest(empty=empty):
                self.assertEqual(streetview.parse_coverage(empty), [])
        for invalid in (b"<html>unavailable</html>", b"[null,{}]", b"[null,[null,[[[]]]]]"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "coverage"):
                streetview.parse_coverage(invalid)

    def test_metadata_distinguishes_removed_panorama_from_protocol_errors(self):
        expected = dict(pano_id="panorama_0001", lat=0, lon=0, imagery_date="2024-06")
        self.assertEqual(streetview.parse_metadata(metadata()), expected)
        message = json.loads(metadata())
        message[1][0][0][0] = 3
        message[1][0][6] = None  # Dates are optional, even for usable imagery.
        self.assertEqual(streetview.parse_metadata(json.dumps(message).encode()), dict(expected, imagery_date=None))
        with self.assertRaises(MissingImagery):
            streetview.parse_metadata(b"[null,[[[2]]]]")
        for invalid in (b"[null,[[[99]]]]", b"[null,[]]", metadata(lat=91), metadata(pano_id="../bad")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "metadata"):
                streetview.parse_metadata(invalid)

    def test_roads_keep_crossings_but_do_not_bridge_outside_excursions(self):
        points = [(0.25, -1), (0.25, 2), (0.75, 2), (0.75, -1)]
        way = dict(type="way", id=1, tags=dict(highway="residential"),
                   nodes=[1, 2, 3, 4], points=points)
        roads = streetview.roads([way], (0, 0, 1, 1), "roads")
        self.assertEqual([road["points"] for road in roads],
                         [[(0.25, 0), (0.25, 1)], [(0.75, 1), (0.75, 0)]])

    def test_matching_rejects_ambiguous_crossings_but_keeps_connected_roads(self):
        horizontal = [(0, -0.001), (0, 0.001)]
        crossing = [(-0.001, 0), (0.001, 0)]
        connector = [(0, 0), (0.001, 0)]
        cases = (
            ("intersection", [horizontal, crossing], ["0", "0"], (0, 0), None),
            ("junction", [horizontal, connector], ["0", "0"], (0, 0), "through-road-at-connector"),
            ("bridge", [horizontal, connector], ["0", "1"], (0, 0), None),
            ("reversed joined sections", [[(0, -0.001), (0, 0)], [(0, 0.001), (0, 0)]],
             ["0", "0"], (0, 0), "joined-road-sections"),
            ("outside matching radius", [horizontal], ["0"], (0.001, 0), None),
        )
        for name, points, layers, point, method in cases:
            with self.subTest(case=name):
                parts = [dict(osm_id=i, name="Main Road", highway="residential", layer=layer)
                         for i, layer in enumerate(layers)]
                lines = [geo.Line(line) for line in points]
                result = streetview.match_road(parts, lines, geo.RoadIndex(lines), point, 30)
                if method is None:
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result["match_method"], method)
                    self.assertEqual(result["path_index"], 0)
                    self.assertAlmostEqual(result["road_distance"], 0, places=5)
