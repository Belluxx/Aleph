"""Tree crowns from a synthetic canopy, and props kept off buildings and roadways."""

import unittest
from unittest.mock import patch

import numpy as np

from src.game import ground, scatter
from src.game.types import Feature, Ground, Raster, Reference
from tests import test_game_ground as test_ground


class CrownTest(unittest.TestCase):
    def test_two_crowns_and_no_lamp_post(self):
        cell = 0.5
        z, x = (np.mgrid[0:80, 0:80] + 0.5) * cell
        canopy = 15 * np.exp(-((x - 12) ** 2 + (z - 20) ** 2) / 18) + 10 * np.exp(-((x - 24) ** 2 + (z - 20) ** 2) / 10)
        canopy[60, 60] = 8  # A lamp post: one tall cell.
        found = scatter.crowns(Raster(canopy, 0, 0, cell))
        self.assertEqual(len(found), 2)
        (x0, z0, h0, r0), (x1, z1, h1, r1) = sorted(found)
        self.assertAlmostEqual(x0, 12, delta=0.5)
        self.assertAlmostEqual(x1, 24, delta=0.5)
        self.assertAlmostEqual(h0, 15, delta=0.5)
        self.assertGreater(r0, r1)


class PropTest(unittest.TestCase):
    def test_props_stay_off_buildings_and_roads(self):
        square = lambda x0, z0, x1, z1: np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], float)[::-1]
        building = square(10, 10, 30, 30)
        osm = [Feature("way/1", "building", {"building": "yes"}, building),
               Feature("way/2", "line", {"highway": "residential"}, np.array([[-50, 0], [50, 0]], float)),
               Feature("node/1", "node", {"amenity": "bench"}, np.array([[20, 20]], float)),  # In the building.
               Feature("node/2", "node", {"amenity": "bench"}, np.array([[0, 20]], float)),
               Feature("node/3", "node", {"highway": "street_lamp"}, np.array([[5, 0.5]], float)),  # On the road.
               Feature("node/4", "node", {"natural": "tree"}, np.array([[25, 25]], float)),  # In the building.
               Feature("node/5", "node", {"highway": "traffic_signals"}, np.array([[-20, 0]], float))]
        flat = lambda cell: Raster(np.zeros((int(100 / cell), int(100 / cell)), np.float32), -50, -50, cell)
        reference = Reference(None, None, square(-50, -50, 50, 50), np.zeros((0, 3)), np.zeros((0, 3), int),
                              flat(0.25), flat(0.25), flat(0.25), osm)
        ground = Ground(flat(0.5), Raster(np.zeros((200, 200), np.uint8), -50, -50, 0.5))
        with patch.object(scatter, "_size", lambda name: (15.0, 5.0)), \
                patch.object(scatter, "_available", lambda name, notes: False):
            instances = scatter.build(reference, ground, [], {}, lambda *a, **k: None)
        assets = [i.asset for i in instances]
        self.assertEqual(assets.count("prop/bench"), 1)
        self.assertIn("prop/traffic-signal", assets)
        self.assertGreater(sum("lamp" in a for a in assets), 2)  # The mapped lamp, moved, and filled ones.
        self.assertFalse(any(a.startswith("tree/") for a in assets))
        for instance in instances:
            x, _, z = instance.position
            self.assertFalse(scatter._inside(building, x, z), instance)
            self.assertGreaterEqual(abs(z), 3.0 - 1e-6, instance)  # The road is 6 m wide along z = 0.



class ParkingTest(unittest.TestCase):
    def test_parked_cars_stay_on_the_carriageway(self):
        """Cars park along a street with parking lanes: inside the carriageway, along it, clear of the crossing,
        the junction, and each other; none stand on the sidewalks."""
        street = Feature("way/1", "line", {"highway": "residential", "lanes": "2", "parking:both": "lane"},
                         np.array([[-30.0, 0.0], [30.0, 0.0]]))
        side = Feature("way/2", "line", {"highway": "residential"}, np.array([[10.0, 0.0], [10.0, 30.0]]))
        crossing = Feature("way/3", "line", {"highway": "footway", "footway": "crossing", "crossing": "zebra"},
                           np.array([[-12.0, -8.0], [-12.0, 8.0]]))
        houses = [Feature(f"way/{4 + k}", "building", {"building": "yes"},
                          np.array([[x0, -20.0], [x0 + 15, -20.0], [x0 + 15, -12.0], [x0, -12.0]]))
                  for k, x0 in enumerate((-25.0, 0.0))]
        built = ground.build(test_ground.reference([street, side, crossing] + houses), {}, lambda *a, **k: None)
        with patch.object(scatter, "_size", lambda name: (15.0, 5.0)), \
                patch.object(scatter, "_available", lambda name, notes: True):
            instances = scatter.build(test_ground.reference([street, side, crossing] + houses), built, [], {},
                                      lambda *a, **k: None)
        cars = [i for i in instances if i.asset in scatter.CARS]
        self.assertGreater(len(cars), 3)
        lanes = Raster(built.data["carriageway"], built.height.x0, built.height.z0, built.height.cell)
        boxes = []
        for car in cars:
            x, _, z = car.position
            direction = np.array([np.sin(car.yaw), np.cos(car.yaw)])
            length, width = scatter.VEHICLES[car.asset]
            corners = scatter._footprint(np.array([x, z]), direction, length, width)
            self.assertTrue((lanes.sample(corners[:, 0], corners[:, 1]) > 0).all(), car)
            self.assertGreater(abs(direction[0]), 0.99, car)  # Parked along the street.
            self.assertGreater(np.abs(corners[:, 0] + 12).min(), 1.0, car)  # Off the zebra.
            self.assertGreater(np.abs(corners[:, 0] - 10).min(), 6.0, car)  # Clear of the junction.
            box = (np.array([x, z]), direction, length / 2, width / 2)
            self.assertFalse(any(scatter._overlap(box, other) for other in boxes), car)
            boxes.append(box)
        self.assertEqual(len(built.data["parked"]), len(cars))


if __name__ == "__main__":
    unittest.main()
