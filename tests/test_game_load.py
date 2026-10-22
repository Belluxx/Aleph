import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.game import load
from src.game.types import Frame

FRAME = Frame(52.5, 13.4, 30.0, 6_371_010.0)
# Clockwise in latitude and longitude, seen from above: north-west, north-east, south-east, south-west.
OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
<node id="1" lat="52.5001" lon="13.3999"/>
<node id="2" lat="52.5001" lon="13.4001"/>
<node id="3" lat="52.4999" lon="13.4001"/>
<node id="4" lat="52.4999" lon="13.3999"/>
<node id="5" lat="52.50005" lon="13.39995"/>
<node id="6" lat="52.50005" lon="13.40005"/>
<node id="7" lat="52.49995" lon="13.40005"/>
<node id="8" lat="52.49995" lon="13.39995"><tag k="highway" v="street_lamp"/></node>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
  <tag k="building" v="yes"/><tag k="roof:shape" v="gabled"/></way>
<way id="11"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
<way id="12"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/></way>
<way id="13"><nd ref="4"/><nd ref="3"/><tag k="highway" v="footway"/></way>
<relation id="20"><member type="way" ref="11" role="outer"/><member type="way" ref="12" role="inner"/>
  <tag k="type" v="multipolygon"/><tag k="leisure" v="park"/></relation>
</osm>
"""


class OsmTest(unittest.TestCase):
    def features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.osm"
            path.write_text(OSM)
            return {f.id: f for f in load.read_osm(path, FRAME)}

    def test_orientation_follows_the_view_from_above(self):
        # z points south, so counter-clockwise from above has a negative shoelace area in (x, z).
        east_then_north = np.array([[0, 0], [1, 0], [1, -1], [0, -1]])
        self.assertLess(load.signed_area(east_then_north), 0)
        features = self.features()
        building, park = features["way/10"], features["relation/20"]
        self.assertEqual((building.kind, len(building.points), building.tags["roof:shape"]), ("building", 4, "gabled"))
        self.assertLess(load.signed_area(building.points), 0)
        self.assertEqual((park.kind, len(park.holes)), ("area", 1))
        self.assertLess(load.signed_area(park.points), 0)
        self.assertGreater(load.signed_area(park.holes[0]), 0)

    def test_kinds_and_ids(self):
        features = self.features()
        self.assertEqual(features["way/13"].kind, "line")
        self.assertEqual(features["node/8"].kind, "node")
        self.assertEqual(features["node/8"].points.shape, (1, 2))
        # Untagged multipolygon members are not features of their own.
        self.assertNotIn("way/11", features)
        self.assertNotIn("way/12", features)

    def test_local_round_trip(self):
        x, z = FRAME.local(52.501, 13.402)
        lat, lon = load.degrees(FRAME, x, -z)
        self.assertAlmostEqual(float(lat), 52.501, places=9)
        self.assertAlmostEqual(float(lon), 13.402, places=9)


class MeshTest(unittest.TestCase):
    def test_weld_merges_seams_and_drops_collapsed_triangles(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0.0004], [1, 0, 1], [0, 0, 1]], np.float32)
        positions, triangles = load.weld(positions, np.array([[0, 2, 1], [3, 5, 4], [0, 1, 3]]))
        self.assertEqual((len(positions), len(triangles)), (4, 2))

    def test_rasterize_fills_cells_under_a_triangle(self):
        # A sloped triangle over a 4 m square: y = x + 1, covering cells with centres below the diagonal.
        positions = np.array([[0, 1, 0], [4, 5, 0], [0, 1, 4]], np.float32)
        top, bottom, up = load.rasterize(positions, np.array([[0, 2, 1]]), 0.0, 0.0, 16, 16, 0.25)
        x = (np.arange(16) + 0.5) * 0.25
        centres = x[None, :] + x[:, None] < 4
        self.assertTrue(np.isfinite(top[centres]).all())
        self.assertTrue(np.allclose(top[centres], np.broadcast_to(x + 1, (16, 16))[centres], atol=1e-5))
        self.assertTrue(np.isnan(top[~centres & (x[None, :] + x[:, None] > 4.3)]).all())
        # 45° is not facing up; a flat copy is.
        self.assertTrue(np.all(up[centres] == 0))
        flat = positions * (1, 0, 1)
        _, _, up = load.rasterize(flat, np.array([[0, 2, 1]]), 0.0, 0.0, 16, 16, 0.25)
        self.assertTrue(np.all(up[centres] == 1))
        self.assertTrue(np.array_equal(np.isfinite(bottom), np.isfinite(top)))

    def test_rasterize_marks_walls(self):
        positions = np.array([[1.1, 0, 1.1], [1.6, 0, 1.1], [1.1, 6, 1.1]], np.float32)
        top, bottom, up = load.rasterize(positions, np.array([[0, 1, 2]]), 0.0, 0.0, 8, 8, 0.25)
        # The wall's area lands in its centroid's cell.
        self.assertEqual((top[4, 4], bottom[4, 4], up[4, 5]), (6, 0, 0))


if __name__ == "__main__":
    unittest.main()
