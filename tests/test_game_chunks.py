import importlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.game import chunks
from src.game.types import Asset, Collider, Ground, Instance, Material, Model, Part, Raster, Reference, Frame

# Other stages may be missing or mid-edit; build only needs names to patch.
for name in ("ground", "buildings", "scatter", "library"):
    try:
        importlib.import_module(f"src.game.{name}")
    except Exception:
        sys.modules[f"src.game.{name}"] = types.ModuleType(f"src.game.{name}")
from src.game import build  # noqa: E402

PLAY = np.array([[-100.0, 60.0], [100.0, 60.0], [100.0, -60.0], [-100.0, -60.0]])


def quad(material, x0, z0, x1, z1, y=0.0):
    positions = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], np.float32)
    return Part(material, positions, np.tile([0, 1, 0], (4, 1)).astype(np.float32), positions[:, [0, 2]],
                np.array([0, 3, 2, 0, 2, 1], np.uint32))


def glb(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    binary = data[28 + length:]

    def positions(accessor):
        a = gltf["accessors"][accessor]
        view = gltf["bufferViews"][a["bufferView"]]
        return np.frombuffer(binary, "<f4", a["count"] * 3, view["byteOffset"]).reshape(-1, 3)

    return gltf, positions


class ChunkTest(unittest.TestCase):
    def test_grid_is_centred_and_skips_chunks_outside(self):
        triangle = np.array([[-100.0, 60.0], [100.0, 60.0], [-100.0, -60.0]])
        cells = chunks.grid(triangle, 64)
        # 200 × 120 m needs 4 × 2 chunks, centred: x from -128, z from -64.
        self.assertEqual(cells[0].rect, (-128.0, -64.0, -64.0, 0.0))
        self.assertEqual(len(chunks.grid(PLAY, 64)), 8)
        # The hypotenuse runs from (100, 60) through (0, 0), leaving 2_0 and 3_0 outside.
        self.assertEqual([cell.id for cell in cells], ["0_0", "1_0", "0_1", "1_1", "2_1", "3_1"])

    def test_assign_uses_the_holding_or_nearest_chunk(self):
        cells = chunks.grid(PLAY, 64)
        index = chunks.assign(cells, [(-70, -10), (70, 30), (500, 500)])
        self.assertEqual([cells[i].id for i in index], ["0_0", "3_1", "3_1"])

    def test_heights_layout(self):
        # Ground rising 1 m per metre east and 10 m per metre south, sampled at nodes.
        x, z = np.meshgrid(np.arange(200) * 0.5 + 0.25 - 50, np.arange(200) * 0.5 + 0.25 - 50)
        height = Raster(x + 10 * z, -50.0, -50.0, 0.5)
        values = chunks.heights(height, (0.0, 4.0, 8.0, 16.0))
        self.assertEqual(values.shape, (25, 17))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ground.height"
            chunks.write_heights(path, values)
            raw = np.frombuffer(path.read_bytes(), "<f4").reshape(25, 17)
        # Column i, row j lies at x = 0 + 0.5 i, z = 4 + 0.5 j.
        self.assertAlmostEqual(float(raw[2, 3]), 1.5 + 10 * 5.0, places=4)


class FakeStages:
    """Small stand-ins for the other stages, following types.py."""

    def __init__(self):
        self.installed = None

    def reference(self, folder, config, progress):
        return Reference(folder, Frame(52.5, 13.4, 30.0, 6_371_010.0), PLAY, np.zeros((0, 3), np.float32),
                         np.zeros((0, 3), np.int32), *(Raster(np.full((4, 4), np.nan), -100, -60) for _ in range(3)),
                         osm=[], dem=None)

    def ground(self, reference, config, progress):
        x, z = np.meshgrid(np.arange(400) * 0.5 - 100, np.arange(240) * 0.5 - 60)
        return Ground(Raster(np.full(x.shape, 2.0), -100, -60, 0.5), Raster(np.zeros(x.shape, int), -100, -60, 0.5))

    def buildings(self, reference, ground, config, progress):
        box = quad("plaster/white", 60, 20, 70, 30, y=12.0)
        return [Model("building/way-1", (65.0, 25.0), [[box], [box], [box]],
                      colliders=[Collider("convex", vertices=box.positions)], extras=dict(tint=np.float32(0.5)))]

    def scatter(self, reference, ground, models, config, progress):
        return [Instance("tree/linden", (-70.0, 2.0, -10.0), 1.0, 1.1)]

    def chunk(self, ground, rect, lod):
        return [quad("asphalt", *rect, y=2.0)]

    def material(self, name):
        return Material(name, 2.0, base_color=f"materials/{name}/color.png", extras=dict(attribution="ambientCG (CC0)"))

    def asset(self, name):
        return Asset(name, [f"assets/{name}/lod{i}.glb" for i in range(3)],
                     [Collider("capsule", params=dict(radius=0.25, half_height=1.5, center=[0, 1.5, 0]))], 12.0, 4.0)

    def install(self, names, folder):
        self.installed = names, folder


class RunTest(unittest.TestCase):
    def test_scene_and_chunk_files(self):
        fake = FakeStages()
        ground = types.SimpleNamespace(build=fake.ground, chunk=fake.chunk, colliders=lambda ground, rect: [])
        library = types.SimpleNamespace(material=fake.material, asset=fake.asset, install=fake.install)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(build.load, "build", fake.reference), \
                mock.patch.object(build, "ground", ground), mock.patch.object(build, "library", library), \
                mock.patch.object(build, "buildings", types.SimpleNamespace(build=fake.buildings, accounting=lambda *args: {})), \
                mock.patch.object(build, "scatter", types.SimpleNamespace(build=fake.scatter)):
            folder = Path(folder)
            (folder / "manifest.json").write_text(json.dumps(dict(started_at="2026-06-21T10:00:00+00:00")))
            build.run(folder, dict(chunk_size=64), lambda *args: None)
            game = folder / "game"
            scene = json.loads((game / "scene.json").read_text())
            self.assertEqual((scene["format"], scene["version"], len(scene["chunks"])), ("aleph-game", 1, 8))
            self.assertEqual(scene["attribution"], ["© OpenStreetMap contributors", "ambientCG (CC0)"])
            self.assertEqual(scene["assets"]["tree/linden"]["lods"][0], "library/assets/tree/linden/lod0.glb")
            self.assertGreater(scene["sun"]["elevation"], 50)
            entry = {c["id"]: c for c in scene["chunks"]}["3_1"]
            self.assertEqual(entry["center"], [96.0, 0.0, 32.0])
            self.assertEqual(entry["ground"], dict(file="chunks/3_1/ground.height", size=[129, 129], spacing=0.5,
                                                   offset=[64.0, 0.0]))
            self.assertEqual(entry["max"][1], 12.0)
            self.assertEqual(sorted(fake.installed[0]), ["asphalt", "plaster/white", "tree/linden"])
            tree = {c["id"]: c for c in scene["chunks"]}["0_0"]["instances"]
            self.assertEqual(tree, [dict(asset="tree/linden", position=[-70.0, 2.0, -10.0], yaw=1.0, scale=1.1,
                                         tint=[1.0, 1.0, 1.0])])

            gltf, positions = glb(game / entry["lods"][0])
            names = [node["name"] for node in gltf["nodes"]]
            self.assertIn("building/way-1", names)
            self.assertIn("collider/building/way-1/0", names)
            building = next(n for n in gltf["nodes"] if n["name"] == "building/way-1")
            self.assertEqual(building["extras"]["aleph"]["tint"], 0.5)
            box = positions(gltf["meshes"][building["mesh"]]["primitives"][0]["attributes"]["POSITION"])
            # Vertices are relative to the chunk centre (96, 0, 32).
            self.assertTrue(np.allclose(box.min(axis=0), (-36, 12, -12)))
            image = gltf["images"][0]["uri"]
            self.assertTrue((game / "chunks/3_1" / image).resolve().is_relative_to((game / "library").resolve()))
            lod2, _ = glb(game / entry["lods"][2])
            self.assertFalse(any(n["name"].startswith("collider/") for n in lod2["nodes"]))
            self.assertEqual((game / entry["ground"]["file"]).stat().st_size, 129 * 129 * 4)
            report = json.loads((game / "report.json").read_text())
            self.assertEqual(report["triangles"]["chunks"]["3_1"]["lod0"], 4)

    def test_sun_follows_the_day(self):
        azimuth, elevation = build.sun(52.5, 13.4, "2026-06-21T11:06:00+00:00")
        self.assertAlmostEqual(azimuth, 180, delta=3)
        self.assertAlmostEqual(elevation, 90 - 52.5 + 23.44, delta=0.5)
        self.assertLess(build.sun(52.5, 13.4, "2026-06-21T23:00:00Z")[1], 0)


if __name__ == "__main__":
    unittest.main()
