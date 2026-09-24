"""Named-street route sampling behavior."""

import unittest

from src import routes
from src.geo import Line, destination


class RouteSamplingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
