import math
import unittest

import numpy as np

from src.game import buildings, classical, sculpt
from src.game.types import Feature
from tests.test_game_buildings import scene


def octagon(cx, cz, r):
    a = np.linspace(0, 2 * math.pi, 9)[:-1]
    return np.column_stack([cx + r * np.cos(a), cz - r * np.sin(a)])


def welded_edges(p, t):
    """Directed edges of triangles after welding equal positions."""
    _, key = np.unique(np.round(p, 5), axis=0, return_inverse=True)
    t = key.ravel()[t]
    return np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])


class ColumnTest(unittest.TestCase):
    def test_column_stays_in_its_footprint_and_faces_out(self):
        # The plinth is 1.32 diameters wide; only Ionic volutes reach a little further.
        for order in classical.ORDERS:
            for detail in range(3):
                geo = classical.Geo()
                classical.column(geo, "stone", "carved", (3.0, -2.0), 0.0, 10.0, 1.0, order, detail)
                for part in geo.parts():
                    p = part.positions.astype(float)
                    t = part.indices.reshape(-1, 3)
                    self.assertTrue((np.abs(p[:, [0, 2]] - (3.0, -2.0)) <= 0.8 + 1e-6).all(), (order, detail))
                    self.assertTrue((p[:, 1] >= -1e-6).all() and (p[:, 1] <= 10.0 + 1e-6).all(), (order, detail))
                    # Wound faces agree with their normals, and the shaft's faces look away from the axis.
                    face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
                    self.assertTrue(((face * part.normals[t].sum(axis=1)).sum(axis=1) >= -1e-9).all())
                    shaft = (p[t][:, :, 1].min(axis=1) > 2.0) & (p[t][:, :, 1].max(axis=1) < 7.0)
                    radial = p[t][shaft].mean(axis=1) - (3.0, 0.0, -2.0)
                    radial[:, 1] = 0
                    self.assertTrue(((face[shaft] * radial).sum(axis=1) > 0).all(), (order, detail))


class SculptTest(unittest.TestCase):
    def test_surface_nets_make_a_closed_sphere(self):
        n, h = 24, 0.1
        x, y, z = np.meshgrid(*[np.arange(n) * h - 1.15] * 3, indexing="ij")
        field = 1.0 - np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Positive inside a sphere of radius 1.
        p, t = sculpt.surface_nets(field, 0.0, np.array([-1.2, -1.2, -1.2]), h)
        edges = welded_edges(p, t)
        pairs = {tuple(e) for e in edges}
        self.assertEqual(len(pairs), len(edges))  # No edge is used twice the same way…
        self.assertTrue(all((b, a) in pairs for a, b in pairs))  # …and every edge has its twin.
        self.assertEqual(len(np.unique(p, axis=0)) - len(pairs) // 2 + len(t), 2)  # Euler characteristic.
        r = np.linalg.norm(p, axis=1)
        self.assertLess(abs(r.mean() - 1.0), 0.05)

    def test_remeshed_sculpture_is_closed_and_stands_on_its_support(self):
        # An open skin over a block (no bottom), like a statue in the capture's mesh.
        corners = np.array([[-1, 2, -1], [1, 2, -1], [1, 2, 1], [-1, 2, 1], [-1, 5, -1], [1, 5, -1], [1, 5, 1],
                            [-1, 5, 1]], float)
        quads = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        tris = np.array([[corners[a], corners[b], corners[c]] for q in quads for a, b, c in ((q[0], q[1], q[2]),
                                                                                             (q[0], q[2], q[3]))])
        ring = np.array([[-3.0, 3.0], [3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]])
        p, n, uv, t = sculpt.remesh(tris, ring, 2.0, 5.5, 0.15)
        pairs = {tuple(e) for e in welded_edges(p, t)}
        self.assertTrue(all((b, a) in pairs for a, b in pairs))
        self.assertGreaterEqual(p[:, 1].min(), 2.0 - 0.06)
        self.assertLess(p[:, 1].min(), 2.2)
        self.assertLess(abs(p[:, 1].max() - 5.0), 0.3)


class GateTest(unittest.TestCase):
    def test_columns_and_entablature_become_a_temple_front(self):
        outline = np.array([[-8.0, 3.0], [8.0, 3.0], [8.0, -3.0], [-8.0, -3.0]])
        parts = [Feature(f"way/{10 + k}", "building", {"building:part": "yes", "height": "8"},
                         octagon(-6.0 + 4.0 * k, z, 0.5)) for k in range(4) for z in (-2.0,)]
        parts += [Feature("way/20", "building", {"building:part": "yes", "height": "10", "min_height": "8"},
                          np.array([[-7.0, -1.0], [7.0, -1.0], [7.0, -3.0], [-7.0, -3.0]])[::-1].copy())]
        reference, ground = scene(lambda x, z: np.full(len(x), 10.0), [outline],
                                  {"building": "gate", "historic": "monument"}, extra=parts)
        models = buildings.build(reference, ground, {}, lambda *a, **k: None)
        gate = [m for m in models if m.extras.get("osm") == "way/1"][0]
        self.assertEqual(gate.extras["kind"], "landmark")
        self.assertEqual(gate.extras["roles"].get("column"), 4)
        self.assertEqual(gate.extras["roles"].get("entablature"), 1)
        lod0, lod1, lod2 = gate.extras["triangles"]
        self.assertGreater(lod0, lod1)
        self.assertGreater(lod1, lod2)
        self.assertGreaterEqual(len(gate.colliders), 5)


if __name__ == "__main__":
    unittest.main()
