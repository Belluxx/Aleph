"""Named-street route sampling behavior."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from PIL import Image

from src import quick, routes
from src.geo import Line, destination


class RouteSamplingTests(unittest.TestCase):
    def test_total_stops_are_distributed_by_length_with_exact_rounding(self):
        sections = [dict(points=[(0, 0), (0, lon)]) for lon in (0.007, 0.002, 0.001)]
        for total, expected in ((10, [7, 2, 1]), (4, [3, 1, 0]), (1, [1, 0, 0])):
            with self.subTest(stops=total):
                self.assertEqual(routes.allocate_stops(sections, total), expected)
        self.assertEqual(routes.allocate_stops([sections[1]], 10), [10])
        self.assertEqual(routes.allocate_stops([sections[1]] * 3, 2), [1, 1, 0])

    def test_meter_sampling_covers_endpoints_and_handles_short_routes(self):
        route = dict(points=[(0, 0), destination((0, 0), 90, 101)])
        length = Line(route["points"]).length
        step, targets = routes.sampling(route, step=10)
        self.assertEqual(len(targets), 12)
        self.assertEqual(targets[0], 0)
        self.assertEqual(targets[-1], length)
        self.assertLessEqual(step, 10)
        self.assertEqual(routes.sampling(route, stops=12), (step, targets))
        self.assertEqual(routes.sampling(route, stops=1), (length, [length / 2]))
        self.assertEqual(routes.sampling(route, step=200), (length, [0, length]))

    def test_loop_sampling_does_not_duplicate_the_start(self):
        route = dict(points=[(0, 0), (0, 0.001), (0.001, 0.001), (0, 0)])
        length = Line(route["points"]).length
        step, targets = routes.sampling(route, stops=4)
        self.assertEqual(len(targets), 4)
        self.assertAlmostEqual(step, length / 4)
        self.assertEqual(targets[0], step / 2)
        self.assertEqual(targets[-1], length - step / 2)

    def test_matcher_uses_meter_targets_and_reports_missing_coverage(self):
        points = [(0, 0), destination((0, 0), 90, 101)]
        route = dict(points=points, way_ids=[1])
        context = [dict(points=points, osm_id=1)]
        line = Line(points)
        views = [dict(pano_id=str(i), lat=line.at(meters)[0], lon=line.at(meters)[1])
                 for i, meters in enumerate([0, 50, 101])]
        # A second panorama at the same position must not create another stop.
        views.append(dict(views[1], pano_id="duplicate"))
        step, targets = routes.sampling(route, step=10)
        selected, gaps = routes.match_views(views, route, context, step, targets)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(gaps), len(targets) - 3)
        self.assertEqual(sorted(item["stop"] for item in selected + gaps),
                         list(range(1, len(targets) + 1)))
        for item in selected:
            self.assertLessEqual(abs(item["path_meters"] - item["target_meters"]), step / 2 + 0.01)


class BranchedStreetTests(unittest.TestCase):
    def setUp(self):
        self.ways = [dict(id=i, nodes=[0, i], points=[(0, 0), end],
                          tags=dict(highway="residential", name="Test Street"))
                     for i, end in enumerate([(0, -0.001), (0, 0.001), (0.001, 0)], 1)]
        self.client = Mock()
        self.client.maps.street.return_value = ("Test Street", self.ways)
        self.client.maps.data.return_value = (self.ways, [])
        self.place = dict(id="way/1", lat=0, lon=0, label="Test Street")

    def test_all_sections_default_and_optional_selection_with_reverse(self):
        sections = routes.resolve(self.client, self.place)
        self.assertEqual([section["route"] for section in sections], [1, 2, 3])
        self.assertEqual([section["way_ids"] for section in sections], [[1], [2], [3]])
        for section in sections:
            with self.subTest(route=section["route"]):
                selected = routes.resolve(self.client, self.place, route=section["route"], reverse=True)
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]["route"], section["route"])
                self.assertEqual(selected[0]["points"], section["points"][::-1])
                self.assertEqual(selected[0]["start"], section["end"])
                self.assertEqual(selected[0]["end"], section["start"])
        for invalid in (0, -1, 4):
            with self.subTest(route=invalid), self.assertRaisesRegex(ValueError, "--route must be from 1 to 3"):
                routes.resolve(self.client, self.place, route=invalid)

    def test_capture_combines_sections_and_tracks_missing_coverage(self):
        # Panoramas halfway along two branches; the third has no coverage.
        views = {str(i): dict(pano_id=str(i), lat=0, lon=lon)
                 for i, lon in enumerate([-0.0005, 0.0005], 1)}
        with BytesIO() as buffer, Image.new("RGB", (1024, 576)) as image:
            image.save(buffer, format="PNG")
            pixels = buffer.getvalue()
        self.client.get.side_effect = lambda address, **kwargs: pixels if kwargs.get("missing_ok") else views[address]
        cases = [({"stops": 1}, [1, 2, 3], 1, [1]),
                 ({"stops": 2}, [1, 2, 3], 2, [1, 2]),
                 ({"stops": 3}, [1, 2, 3], 3, [1, 2]),
                 ({"step": 70}, [1, 2, 3], 9, [1, 2]),
                 ({"stops": 1, "route": 2}, [2], 1, [2])]
        for options, section_ids, requested, photographed in cases:
            with (self.subTest(options=options), TemporaryDirectory() as directory,
                  patch("src.quick.places.choose", return_value=self.place),
                  patch("src.quick.coverage", return_value=list(views.values())) as coverage,
                  patch("src.quick.streetview.metadata_url", side_effect=lambda pano_id: pano_id),
                  patch("src.quick.streetview.parse_metadata", side_effect=lambda data, **kwargs: data)):
                result = quick.street_photos(self.client, directory, lambda *args: None,
                                             street="Test Street", view="both", **options)
                self.assertEqual([section["route"] for section in result["routes"]], section_ids)
                self.assertEqual(result["requested_stops"], requested)
                self.assertEqual(result["saved_stops"], len(photographed))
                self.assertEqual([photo["route"] for photo in result["photos"]],
                                 [route for route in photographed for _ in range(2)])
                self.assertEqual([photo["view"] for photo in result["photos"]],
                                 [view for _ in photographed for view in ("left", "right")])
                self.assertEqual(len(result["gaps"]), requested - len(photographed))
                self.assertEqual(result["status"], "partial" if result["gaps"] else "complete")
                stops = {photo["stop"] for photo in result["photos"]}
                stops.update(gap["stop"] for gap in result["gaps"])
                self.assertEqual(stops, set(range(1, requested + 1)))
                if result["gaps"]:
                    self.assertIn(3, {gap["route"] for gap in result["gaps"]})
                coverage.assert_called_once()
                folder = Path(result["folder"])
                self.assertEqual(list(Path(directory).resolve().iterdir()), [folder])
                self.assertEqual(len(list(folder.glob("*.jpg"))), 2 * len(photographed))
                self.assertEqual(json.loads((folder / "result.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()
