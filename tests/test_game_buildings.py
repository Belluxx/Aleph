import random
import unittest

import numpy as np

from src.game import buildings, facades, roofs
from src.game.types import Feature, Frame, Ground, Raster, Reference

STYLE = dict(buildings.STYLE, name="test", ornament=0.7, balconies=0.5, parapet=[0.5, 0.5],
             walls={"wall/plaster-white": 1}, plinth=dict(height=0.5, material="wall/sandstone"))
LOOK = dict(reveal=0.2, bay=3.0, door_width=1.2, door_height=2.4, window_width=1.3, window_height=1.8, sill=0.9,
            balconies=0.5)


def scene(surface, rings, tags=None, size=60.0, extra=()):
    """A Reference whose top raster is surface(x, z) inside rings and 0 elsewhere, on flat ground at y = 0."""
    n = int(size / 0.25)
    top = Raster(np.zeros((n, n)), -size / 2, -size / 2)
    x, z = top.centres()
    inside = roofs.inside(rings, x, z)
    top.values[inside] = surface(x[inside], z[inside])
    house = Feature("way/1", "building", dict(tags or {"building": "yes"}), np.asarray(rings[0], float),
                    [np.asarray(h, float) for h in rings[1:]])
    road = Feature("way/2", "line", {"highway": "residential"}, np.array([[-size / 2, 12.0], [size / 2, 12.0]]))
    play = np.array([[-size / 2, -size / 2], [-size / 2, size / 2], [size / 2, size / 2], [size / 2, -size / 2]])
    reference = Reference(None, Frame(52.5, 13.4, 0.0, 6371010.0), play, np.zeros((0, 3), np.float32),
                          np.zeros((0, 3), np.int32), top, top, top, [house, road, *extra])
    ground = Ground(Raster(np.zeros((n // 2, n // 2)), -size / 2, -size / 2, 0.5),
                    Raster(np.zeros((n // 2, n // 2), np.uint8), -size / 2, -size / 2, 0.5))
    return reference, ground


def closed(parts):
    """Whether triangles form closed shells: every directed edge has its reverse exactly once."""
    edges = {}
    for part in parts:
        p = np.round(part.positions.astype(float), 4)
        keys = [tuple(v) for v in p]
        for a, b, c in part.indices.reshape(-1, 3):
            for e in ((keys[a], keys[b]), (keys[b], keys[c]), (keys[c], keys[a])):
                edges[e] = edges.get(e, 0) + 1
    return all(edges.get((b, a)) == n for (a, b), n in edges.items())


RECT = np.array([[-10.0, 6.0], [10.0, 6.0], [10.0, -6.0], [-10.0, -6.0]])  # Counter-clockwise from above.


class FacadeTest(unittest.TestCase):
    def test_bays_stay_inside_walls_without_overlap(self):
        for length in np.linspace(0.5, 40, 80):
            spans = facades.bays(length, 3.0, 0.6)
            for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                self.assertLessEqual(a1, b0 + 1e-9)
            for a0, a1 in spans:
                self.assertGreaterEqual(a0, 0.6 - 1e-9)
                self.assertLessEqual(a1, length - 0.6 + 1e-9)

    def test_openings_inside_walls_and_floors_without_overlap(self):
        floors = [(0.0, 4.2, "ground"), (4.2, 7.6, "upper"), (7.6, 11.0, "upper")]
        for length in np.linspace(1.0, 30, 59):
            for shop in (False, True):
                found = facades.layout(length, floors, LOOK, random.Random(1), door=length / 3, shop=shop)
                for o in found:
                    self.assertGreaterEqual(o.u0, 0.0)
                    self.assertLessEqual(o.u1, length)
                    self.assertTrue(any(y0 <= o.v0 and o.v1 <= y1 for y0, y1, _ in floors))
                for i, a in enumerate(found):
                    for b in found[i + 1:]:
                        self.assertTrue(a.u1 <= b.u0 or b.u1 <= a.u0 or a.v1 <= b.v0 or b.v1 <= a.v0)
                if length > 8:
                    self.assertEqual(sum(o.kind == "door" for o in found), 1)


class RoofTest(unittest.TestCase):
    def fitted(self, shape):
        rng = np.random.default_rng(0)
        x, z = rng.uniform(-9.5, 9.5, 2000), rng.uniform(-5.5, 5.5, 2000)
        y = roofs.roof(shape, RECT, 10.0, 4.0).y(x, z) + rng.normal(0, 0.15, len(x))
        return roofs.fit(x, z, y, RECT)

    def test_fit_finds_shapes(self):
        for shape in ("flat", "gabled", "hipped", "pyramidal"):
            roof, rms = self.fitted(shape)
            self.assertEqual(roof.shape, shape)
            self.assertLess(rms, 0.3)
            if shape != "flat":
                self.assertAlmostEqual(roof.ridge, 14.0, delta=0.3)

    def test_surface_covers_footprint_once(self):
        hole = [[-3.0, -2.0], [-3.0, 2.0], [3.0, 2.0], [3.0, -2.0]]  # Clockwise from above.
        for shape in ("flat", "gabled", "hipped", "mansard", "skillion"):
            tris = roofs.surface(roofs.roof(shape, RECT, 10.0, 3.0), RECT, [hole])
            area = sum(roofs.orient(t[0, [0, 2]], t[1, [0, 2]], t[2, [0, 2]]) for t in tris) / 2
            self.assertAlmostEqual(area, 240 - 24, places=6)


class BuildTest(unittest.TestCase):
    def test_gabled_house(self):
        gable = roofs.roof("gabled", RECT, 9.0, 4.0)
        reference, ground = scene(lambda x, z: gable.y(x, z), [RECT])
        model, = buildings.build(reference, ground, {"style": "auto"}, lambda *a, **k: None)
        self.assertEqual(model.extras["kind"], "building")
        self.assertEqual(model.extras["roof"], "gabled")
        self.assertAlmostEqual(model.extras["height"], 13.0, delta=0.5)
        self.assertLess(model.extras["fit_rms"], 0.3)
        counts = [sum(len(p.indices) for p in parts) // 3 for parts in model.lods]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        hull = np.concatenate([c.vertices for c in model.colliders])
        self.assertGreaterEqual(hull[:, 1].max(), 12.9)
        self.assertTrue(all(c.shape == "convex" for c in model.colliders))

    def test_draft_shells_are_closed(self):
        courtyard = [np.array([[-4.0, -3.0], [-4.0, 3.0], [4.0, 3.0], [4.0, -3.0]])]
        for shape, rings in (("gabled", [RECT]), ("hipped", [RECT]), ("skillion", [RECT]),
                             ("flat", [RECT, *courtyard])):
            reference, ground = scene(lambda x, z: np.full(len(x), 11.0), rings,
                                      {"building": "yes", "roof:shape": shape, "height": "12"})
            reference.top.values[:] = np.nan  # Unmeasured: tags decide.
            for quality in ("draft", "high"):
                model, = buildings.build(reference, ground, {"quality": quality}, lambda *a, **k: None)
                shell = [p for p in model.lods[2] if not p.material.startswith(("glass/", "door/", "shop/"))]
                self.assertTrue(closed(shell), (shape, quality))  # Window quads float just off the walls.

    def test_glass_sits_behind_frames_inside_the_walls(self):
        """Every pane is recessed into the wall, its frame between it and the facade, and nothing floats off the
        wall further than balconies, portals and steps reach."""
        reference, ground = scene(lambda x, z: np.full(len(x), 15.0), [RECT],
                                  {"building": "apartments", "building:levels": "4", "start_date": "1890"})
        model, = buildings.build(reference, ground, {"style": "berlin-gruenderzeit"}, lambda *a, **k: None)
        inside = lambda points: roofs.inside([RECT], points[:, 0], points[:, 2])
        depth = lambda points: roofs.distance([RECT], points[:, 0], points[:, 2])
        parts = {p.material: p.positions.astype(float) for p in model.lods[0]}
        glass = parts["glass/window"]
        self.assertTrue(inside(glass).all())
        self.assertGreaterEqual(depth(glass).min(), 0.12)
        frames = parts["trim/wood-painted"]
        frames = frames[frames[:, 1] < 14.0]  # Window frames, not the eaves.
        self.assertTrue(inside(frames).all())
        self.assertLessEqual(depth(frames).max(), depth(glass).max() + 1e-4)
        everything = np.concatenate([p.positions for p in model.lods[0]]).astype(float)
        outside = ~inside(everything)
        self.assertLessEqual(depth(everything[outside]).max(), 1.5)

    def test_overlapping_pieces_build_one_wall(self):
        """An outline kept beside a taller part sharing two of its walls: each stretch of facade is built once."""
        part = Feature("way/3", "building", {"building:part": "yes", "height": "20"},
                       np.array([[0.0, 0.0], [10.0, 0.0], [10.0, -6.0], [0.0, -6.0]]))
        reference, ground = scene(lambda x, z: np.full(len(x), 11.0), [RECT], {"building": "yes", "height": "12"},
                                  extra=[part])
        reference.top.values[:] = np.nan
        model, = buildings.build(reference, ground, {"quality": "draft"}, lambda *a, **k: None)
        self.assertEqual(len(model.extras["parts"]), 1)
        # Wall triangles in the plane x = 10 facing east: no point of it may be covered twice.
        tris = []
        for p in model.lods[0]:
            for t in p.indices.reshape(-1, 3):
                v = p.positions[t].astype(float)
                if np.allclose(v[:, 0], 10.0, atol=1e-3) and p.normals[t[0]][0] > 0.9:
                    tris.append(v[:, [2, 1]])
        self.assertTrue(tris)
        for z in np.linspace(-5.9, 5.9, 25):
            for y in np.linspace(0.1, 19.9, 25):
                hits = sum(roofs.inside([t], np.array([z]), np.array([y]))[0] for t in tris)
                self.assertLessEqual(hits, 1, (z, y))

    def test_footprint_shift_is_found(self):
        moved = RECT + (1.0, -0.5)
        reference, ground = scene(lambda x, z: np.full(len(x), 15.0), [moved])
        reference.osm[0].points = RECT.copy()
        model, = buildings.build(reference, ground, {"quality": "draft"}, lambda *a, **k: None)
        self.assertEqual(model.extras["shift"], [1.0, -0.5])

    def test_overrides(self):
        reference, ground = scene(lambda x, z: np.full(len(x), 15.0), [RECT])
        quiet = lambda *a, **k: None
        self.assertEqual(buildings.build(reference, ground, {"buildings": {"way/1": {"mode": "drop"}}}, quiet), [])
        model, = buildings.build(reference, ground, {"buildings": {"way/1": {"mode": "replace", "file": "t.glb"}}},
                                 quiet)
        self.assertEqual((model.name, model.extras["replace"]), ("landmark/way-1", "t.glb"))
        model, = buildings.build(reference, ground, {"buildings": {"way/1": {"height": 30}}}, quiet)
        self.assertAlmostEqual(model.extras["height"], 30.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
