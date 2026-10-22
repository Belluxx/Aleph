"""Library materials, models, and packaging, offline: downloads fail and procedural maps stand in."""

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.game import library


def offline(*args):
    raise OSError("offline")


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.previous = library._root
        library.root(self.folder.name)
        self.fetch = patch.object(library, "_fetch", offline)
        self.fetch.start()

    def tearDown(self):
        self.fetch.stop()
        library._root = self.previous
        self.folder.cleanup()

    def test_orm_packs_occlusion_roughness_metallic(self):
        orm = library.pack_orm(np.full((2, 2), 0.25), np.full((2, 2), 0.5), np.ones((2, 2)))
        self.assertEqual(orm.shape, (2, 2, 3))
        self.assertEqual(orm[0, 0].tolist(), [64, 128, 255])

    def test_install_mirrors_layout_and_texture_uris_resolve(self):
        dest = Path(self.folder.name) / "game" / "library"
        files = library.install(["wall/plaster-beige", "prop/bollard"], dest)
        beige = library.material("wall/plaster-beige")
        self.assertEqual(beige.normal, "materials/wall/plaster-white/normal.png")  # Variants share their base's maps.
        self.assertIn("procedural", beige.extras["source"])
        for relative in files:
            self.assertTrue((dest / relative).is_file(), relative)
        for relative in library.asset("prop/bollard").lods:
            data = (dest / relative).read_bytes()
            length = struct.unpack_from("<I", data, 12)[0]
            for image in json.loads(data[20:20 + length]).get("images", []):
                self.assertTrue(((dest / relative).parent / image["uri"]).resolve().is_file(), image["uri"])

    def test_install_copies_height_maps_and_aliases_share_maps(self):
        dest = Path(self.folder.name) / "game" / "library"
        files = library.install(["wall/ashlar-sandstone", "stone/sandstone-triplanar", "decal/manhole"], dest)
        ashlar = library.material("wall/ashlar-sandstone")
        self.assertEqual(ashlar.extras["height"], "materials/wall/ashlar-sandstone/height.png")
        self.assertGreater(ashlar.extras["parallax"], 0)
        self.assertIn(ashlar.extras["height"], files)
        self.assertTrue((dest / ashlar.extras["height"]).is_file())
        alias = library.material("stone/sandstone-triplanar")
        self.assertTrue(alias.extras["triplanar"])
        self.assertEqual(alias.base_color, library.material("stone/sandstone-smooth").base_color)
        self.assertEqual(library.material("decal/manhole").alpha, "BLEND")

    def test_viewer_flags(self):
        flags = dict(interior=["glass/window", "shop/glass", "door/metal-glass", "wall/glass-curtain"],
                     wind=["veg/leaves-broad", "veg/leaves-conifer"],
                     triplanar=["metal/bronze-triplanar", "metal/gold-triplanar", "stone/sandstone-triplanar",
                                "metal/copper-patina-triplanar"],
                     decal=[name for name in library.MATERIALS if name.startswith("decal/")])
        for flag, names in flags.items():
            for name in names:
                spec = library.MATERIALS[name]
                self.assertTrue(spec.get("extras", {}).get(flag), f"{name} lacks {flag}")

    def test_vehicles_are_grounded_and_within_budget(self):
        from src.game import assets

        for name, limit in (("vehicle/sedan", 6000), ("vehicle/bus", 10000)):
            model = assets.MODELS[name]()
            counts = [sum(len(part.indices) // 3 for part in parts) for parts in model.lods]
            self.assertLessEqual(counts[0], limit, name)
            self.assertLess(counts[2], counts[1], name)
            self.assertLess(counts[1], counts[0], name)
            points = np.concatenate([part.positions for part in model.lods[0]])
            self.assertAlmostEqual(float(points[:, 1].min()), 0.0, delta=0.02)  # Tyres touch the ground.
            self.assertGreater(points[:, 2].max() - points[:, 2].min(), points[:, 0].max() - points[:, 0].min())  # Long along z.
            self.assertEqual(model.colliders[0].shape, "convex")

    def test_styles_share_keys_and_name_library_materials(self):
        keys = set(library.style("berlin-gruenderzeit"))
        for name in library.styles():
            style = library.style(name)
            self.assertEqual(set(style), keys, name)
            used = [*style["walls"], style["roofs"]["flat"], *style["roofs"]["pitched"], style["trim"], style["glass"],
                    style["shop"], style["window"]["frame"], style["plinth"]["material"], *style["doors"]]
            self.assertTrue(all(material in library.MATERIALS for material in used), name)


if __name__ == "__main__":
    unittest.main()
