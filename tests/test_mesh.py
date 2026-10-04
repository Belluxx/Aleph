"""Decode synthetic Google Earth nodes and download them through the capture engine."""

import json
import struct
import time
import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from src import capture, mesh
from src.common import MissingImagery, write_bytes

RADIUS = 6371010.0
AREA = (-1e-4, -1e-4, 1e-4, 1e-4)  # About 22 m square around latitude 0, longitude 0.


def varint(value):
    out = bytearray()
    while True:
        out.append((value & 127) | (128 if value > 127 else 0))
        value >>= 7
        if not value:
            return bytes(out)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    return varint(number << 3 | 2) + varint(len(value)) + value


def counted(values):
    return b"".join(map(varint, [len(values), *values]))


def deltas(values):
    return bytes((b - a) & 255 for a, b in zip([0, *values], values))


def node(vertices=(), strip=(), runs=(), skirts=b"", jpeg=b""):
    """Vertex x, y, z map to meters east, north, and up, offset by 100 m horizontally."""
    matrix = struct.pack("<16d", 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, RADIUS, -100, -100, 1)
    data = field(1, matrix)
    if not vertices:
        return data
    xs, ys, zs = zip(*[(e + 100, n + 100, u) for e, n, u in vertices])
    us, vs = [10 * i for i in range(len(vertices))], [5 * i for i in range(len(vertices))]
    zeros, indices = 0, []
    for index in strip:
        indices.append(0 if index == zeros else zeros - index)
        zeros += index == zeros
    texture = field(1, jpeg) + field(2, mesh.JPEG)
    return data + field(2, b"".join([
        field(1, deltas(xs) + deltas(ys) + deltas(zs)),
        field(3, counted(indices)),
        field(6, texture),
        field(7, b"\xff\0\xff\0" + deltas(us) + deltas(vs) + bytes(len(us)) + bytes(len(vs))),
        field(8, counted(runs)),
        field(13, skirts),
    ]))


def read_glb(path):
    data = Path(path).read_bytes()
    size = struct.unpack("<I", data[12:16])[0]
    gltf, body = json.loads(data[20:20 + size]), data[28 + size:]

    def view(index, kind):
        v = gltf["bufferViews"][index]
        return array(kind, body[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])

    return gltf, view


class MeshTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def test_export_drops_refined_octants_skirts_upper_layers_and_outside_triangles(self):
        vertices = [(0, 0, 1), (2, 0, 0), (0, 2, 0), (2, 2, 0), (4, 0, 0), (4, 2, 0), (0, 4, 0), (60, 60, 0), (-60, -60, 0)]
        # Triangles: 0 kept, 1 skirt, 2–3 degenerate, 4 octant refined by "01", 5 octant of the
        # unavailable "02" (kept, odd so flipped), 6 centered outside, 7 in the fourth layer.
        strip = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8]
        runs = [6, 1, 1, 0, 0, 0, 0, 0, 1] + [0] * 15 + [1]
        write_bytes(self.folder / "0.bin", node(vertices, strip, runs, skirts=bytes([0b10]), jpeg=b"jpeg-0"))
        write_bytes(self.folder / "01.bin", node())
        stage = dict(radius=RADIUS, nodes=[["0", 1, None, 0b110], ["01", 1, None, 0], ["02", 1, None, 0]],
                     results=[dict(filename="0.bin"), dict(filename="01.bin"), dict(status="skipped")])
        mesh.export(self.folder / "mesh.glb", self.folder, stage, AREA, Mock())

        gltf, view = read_glb(self.folder / "mesh.glb")
        self.assertEqual([m["name"] for m in gltf["meshes"]], ["0"])
        position, uv, indices = (gltf["accessors"][i]["bufferView"] for i in range(3))
        self.assertEqual(list(view(indices, "H")), [0, 1, 2, 3, 4, 5])
        # Vertices 0, 1, 2, 4, 6, 5 as x east, y up, z south.
        self.assertEqual(list(view(position, "f")), [0, 1, 0, 2, 0, 0, 0, 0, -2, 4, 0, 0, 0, 0, -4, 4, 0, -2])
        self.assertEqual(list(view(uv, "f"))[6:8], [40.5 / 256, 20.5 / 256])
        self.assertEqual(view(gltf["images"][0]["bufferView"], "B").tobytes(), b"jpeg-0")

    def test_parallel_download_keeps_node_order_across_interrupt_and_resume(self):
        nodes = [[f"0{i:02d}", 1, None, 0] for i in range(20)]
        run = dict(format="aleph-python", version=capture.FORMAT_VERSION, bounds=list(AREA),
                   options=capture.settings({"include": ["mesh"]}), started_at="", state="planned",
                   stages=[dict(mode="mesh", level=21, radius=RADIUS, nodes=nodes, results=[])])

        def payload(path):
            return node() + field(4, path.encode())

        def get(address, interrupt=None, **_):
            path = address.split("!1s")[1].split("!")[0]
            time.sleep(0.02 if int(path) % 2 else 0)  # Finish out of order.
            if path == interrupt:
                raise KeyboardInterrupt()
            if path == "005":
                raise MissingImagery("Gone.")
            return payload(path)

        client = Mock(delay=0)
        client.get.side_effect = lambda address, **kwargs: get(address, "012", **kwargs)
        with self.assertRaises(KeyboardInterrupt):
            capture.download(run, self.folder, client, Mock())

        saved = capture.load(self.folder)
        results = saved["stages"][0]["results"]
        self.assertEqual((saved["state"], len(results)), ("stopped", 12))
        self.assertEqual(results[5]["status"], "skipped")
        for node_, result in zip(nodes, results):
            if "filename" in result:
                self.assertEqual((self.folder / result["filename"]).read_bytes(), payload(node_[0]))

        client.get.reset_mock(side_effect=True)
        client.get.side_effect = get
        capture.download(saved, self.folder, client, Mock())
        requested = sorted(call.args[0].split("!1s")[1].split("!")[0] for call in client.get.call_args_list)
        self.assertEqual(requested, [node_[0] for node_ in nodes[12:]])
        self.assertEqual(capture.load(self.folder)["state"], "complete")


if __name__ == "__main__":
    unittest.main()
