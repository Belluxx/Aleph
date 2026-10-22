import unittest

import numpy as np

from src.game import ground, library, roads
from src.game.types import CELL, Feature, Frame, Raster, Reference

LEVEL = 10.0


def reference(osm, hole=False, car=False, pit=False):
    """A flat 48 m square capture at y = LEVEL, optionally with a mesh hole, a car-sized bump, or a pit 8 m deep
    under and around HOUSE, as meshes show under some buildings and in deep courtyards."""
    play = np.array([[-24.0, 24.0], [24.0, 24.0], [24.0, -24.0], [-24.0, -24.0]])
    size = int(56 / CELL)
    bottom = np.full((size, size), LEVEL)
    if hole:
        bottom[60:100, 60:100] = np.nan
    if car:
        bottom[150:158, 150:168] += 1.5
    if pit:
        bottom[16:80, 48:176] -= 8.0
    raster = lambda values: Raster(values, -28.0, -28.0)
    return Reference(folder=None, frame=Frame(52.5, 13.4, 0.0, 6371010.0), play=play, positions=np.zeros((0, 3)),
                     triangles=np.zeros((0, 3), int), top=raster(bottom.copy()), bottom=raster(bottom),
                     up=raster(np.ones((size, size))), osm=osm)


ROAD = Feature("way/1", "line", {"highway": "primary", "lanes": "2"}, np.array([[-30.0, 0.0], [30.0, 0.0]]))
HOUSE = Feature("way/2", "building", {"building": "yes"}, np.array([[-10.0, -20.0], [10.0, -20.0], [10.0, -12.0],
                                                                      [-10.0, -12.0]]))
# A paved square beside the house, laid out in granite slabs framed by mosaic.
SQUARE = Feature("way/3", "area", {"place": "square", "surface": "paving_stones"},
                 np.array([[12.0, -22.0], [22.0, -22.0], [22.0, -6.0], [12.0, -6.0]]))


def progress(*args, **kwargs):
    pass


class RoadTests(unittest.TestCase):
    def test_width_from_tags(self):
        self.assertAlmostEqual(roads.width({"highway": "primary", "lanes": "2"}), 6.5)
        self.assertAlmostEqual(roads.width({"highway": "residential", "width": "7.5 m"}), 7.5)
        self.assertAlmostEqual(roads.width({"highway": "primary", "lanes": "2", "parking:both": "lane"}), 10.5)
        self.assertAlmostEqual(roads.width({"highway": "primary", "oneway": "yes"}), 6.75)

    def test_buffered_road_surface(self):
        built = ground.build(reference([ROAD, HOUSE]), {}, progress)
        x, z = np.meshgrid(*ground.coordinates(built.height))
        asphalt = built.surface.values == ground.SURFACES.index("asphalt")
        # 6.5 m wide: vertices within 3.25 m of the centreline, none beyond.
        self.assertTrue(asphalt[np.abs(z) < 3.0].all())
        self.assertFalse(asphalt[np.abs(z) > 3.5].any())


class HeightTests(unittest.TestCase):
    def test_fills_holes_and_drops_cars(self):
        built = ground.build(reference([], hole=True, car=True), {}, progress)
        hole = built.height.sample(np.array([-8.0]), np.array([-8.0]))[0]
        car = built.height.sample(np.array([11.75]), np.array([10.5]))[0]
        self.assertAlmostEqual(hole, LEVEL, delta=0.02)
        self.assertAlmostEqual(car, LEVEL, delta=0.05)

    def test_ignores_pits_under_buildings(self):
        built = ground.build(reference([HOUSE], pit=True), {}, progress)
        x, z = np.array([0.0, 11.0, -11.0]), np.array([-16.0, -16.0, -10.0])
        self.assertTrue(np.allclose(built.height.sample(x, z), LEVEL, atol=0.1))


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ground = ground.build(reference([ROAD, HOUSE, SQUARE]), {}, progress)

    def test_curb_faces_the_road(self):
        parts = ground.chunk(self.ground, (-16, -16, 16, 16), 0)
        curb = next(part for part in parts if part.material == "ground/curb")
        faces = curb.indices.reshape(-1, 3)
        p, n = curb.positions[faces], curb.normals[faces].mean(axis=1)
        side = np.abs(n[:, 1]) < 0.1
        self.assertTrue(side.any())
        # Side faces stand at the road edges (|z| = 3.25) and face the centreline.
        centre = p[side].mean(axis=1)
        self.assertTrue(np.allclose(np.abs(centre[:, 2]), 3.25, atol=0.05))
        self.assertTrue((n[side, 2] * centre[:, 2] < 0).all())
        # Counter-clockwise seen from where the normal points.
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        self.assertTrue((np.einsum("ij,ij->i", normal, n) > 0).all())
        self.assertTrue(len(ground.colliders(self.ground, (-24, -24, 24, 24))) > 0)

    def test_chunks_tile_without_gaps_or_overlaps(self):
        bare = ground.build(reference([ROAD]), {}, progress)
        # The house's core, more than RIM inside its outline, is left out, with its corners cut by the grid.
        # Gutters, sidewalk strips, and the square's pattern are cut from the ground, never laid over it.
        materials = set()
        for built, lods, house in ((bare, (0, 1, 2), False), (self.ground, (0, 1), True)):
            for lod in lods:
                for rect in [(-20, -20, 0, 0), (0, -20, 20, 0), (-20, 0, 0, 20), (0, 0, 20, 20)]:
                    area = 0.0
                    for part in ground.chunk(built, rect, lod):
                        self.assertIn(part.material, library.MATERIALS)
                        materials.add(part.material)
                        # Paint and decals go whole to the chunk holding their centre.
                        if part.material in ground.LIFT:
                            continue
                        p = part.positions
                        self.assertTrue((p[:, 0] >= rect[0] - 1e-4).all() and (p[:, 0] <= rect[2] + 1e-4).all())
                        self.assertTrue((p[:, 2] >= rect[1] - 1e-4).all() and (p[:, 2] <= rect[3] + 1e-4).all())
                        t = p[part.indices.reshape(-1, 3)].astype(float)
                        area += np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 1]).sum() / 2
                    core = house * max(0.0, min(rect[2], 10 - ground.RIM) - max(rect[0], -10 + ground.RIM)) * max(
                        0.0, min(rect[3], -12 - ground.RIM) - max(rect[1], -20 + ground.RIM))
                    self.assertAlmostEqual(area, 400 - core, delta=0.5 if house else 0.01, msg=(lod, rect))
        for laid in ("ground/gutter", "ground/mosaic", "ground/paving-granite"):
            self.assertIn(ground.resolve(laid), materials)


if __name__ == "__main__":
    unittest.main()
