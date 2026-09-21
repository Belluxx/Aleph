"""PBF wire-format boundaries and extraction checked without external packages."""

import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from src.pbf import PBF, unpack


def integer(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def field(number, value):
    if isinstance(value, bytes):
        return integer(number * 8 + 2) + integer(len(value)) + value
    return integer(number * 8) + integer(value)


def packed(values, signed=False):
    return b"".join(integer((v << 1) ^ (v >> 63) if signed else v) for v in values)


def blob(kind, payload, compressed=False):
    data = field(2, len(payload)) + field(3, zlib.compress(payload)) if compressed else field(1, payload)
    header = field(1, kind) + field(3, len(data))
    return struct.pack(">I", len(header)) + header + data


def snapshot(group, *, options=b"", features=(b"OsmSchema-V0.6", b"DenseNodes")):
    header = b"".join(field(4, feature) for feature in features)
    data = field(1, field(1, b"")) + field(2, group) + options
    return blob(b"OSMHeader", header) + blob(b"OSMData", data, True)


class PBFTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.folder / "region.osm.pbf"
        self.output = self.folder / "map.osm"
        self.area = (-.0001, -.0001, .0001, .0005)

    def test_dense_and_normal_nodes_preserve_tags_metadata_and_complete_ways(self):
        results = []
        for name in ("dense.osm.pbf", "nodes.osm.pbf"):
            source = PBF(Path(__file__).parent / "data" / name)
            source.export(self.area, self.output)
            root = ET.parse(self.output).getroot()
            self.assertEqual({n.attrib["id"] for n in root.findall("node")}, {"1", "2", "6"})
            node = root.find("node")
            self.assertEqual(node.attrib["version"], "3")
            self.assertEqual(node.attrib["timestamp"], "2026-09-19T12:34:56Z")
            self.assertEqual(node.find("tag[@k='name']").attrib["v"], 'Caffè & "Tea" <Terrace>')
            results.append(source.read(source.select(self.area), centers=True))
        self.assertEqual(*results)

    def test_nondefault_offsets_granularity_and_split_packed_fields(self):
        ids = field(1, packed([10], True)) + field(1, packed([1], True))
        dense = ids + field(8, packed([-10, 20], True)) + field(9, packed([5, -10], True))
        options = field(17, 10_000_000) + field(19, (1 << 64) - 1_000_000_000) + field(20, 2_000_000_000)
        self.path.write_bytes(snapshot(field(2, dense), options=options))
        source = PBF(self.path)
        selected = source.select((-2, 1, 0, 3))
        self.assertEqual(selected, ({10, 11}, set(), set()))
        self.assertEqual(list(source.nodes(selected[0])), [(10, -1.1, 2.05), (11, -.9, 1.95)])

    def test_integer_boundaries_and_malformed_varints(self):
        values = [0, 1, 127, 128, 16384, 1 << 32, (1 << 64) - 1]
        self.assertEqual(list(unpack(packed(values))), values)
        signed = [0, -1, 63, -64, 64, -65, 1 << 40, -(1 << 63)]
        self.assertEqual(list(unpack(packed(signed, True), True)), signed)
        for data in (b"\x80", b"\x80" * 10, b"\xff" * 9 + b"\x02"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                unpack(data)

    def test_bulk_integer_decoding_and_lane_boundaries(self):
        rng = random.Random(17)
        for signed in (False, True):
            limit = 1 << (27 if signed else 28)
            values = [rng.randrange(-limit if signed else 0, limit) for _ in range(2000)]
            values[0] = -(1 << 63) if signed else (1 << 64) - 1
            values[1:9] = [0, 1, 63, 64, 127, 128, 16383, 16384]
            if signed:
                values[9:14] = [-1, -64, -65, -8192, -8193]
            data = packed(values, signed)
            self.assertEqual(list(unpack(data, signed)), values)
            with self.assertRaises(ValueError):
                unpack(data + b"\x80", signed)
        values = [1 << 40] + [1] * 1000 + [1 << 30, 0, 3]
        self.assertEqual(list(unpack(packed(values, True), True)), values)

    def test_rejects_unknown_features_and_truncated_files(self):
        for data in (b"", b"\x00\x00", snapshot(b"", features=(b"HistoricalInformation",)), snapshot(b"")[:-1]):
            self.path.write_bytes(data)
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                PBF(self.path)

    def test_missing_way_node_preserves_previous_output(self):
        node = field(1, 2) + field(8, 0) + field(9, 0)
        way = field(1, 10) + field(8, packed([1, 998], True))
        self.path.write_bytes(snapshot(field(1, node) + field(3, way)))
        self.output.write_bytes(b"previous")
        with self.assertRaisesRegex(ValueError, "incomplete way"):
            PBF(self.path).export(self.area, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous")
        self.assertEqual(list(self.folder.glob(".map.osm-*")), [])

    def test_standalone_script_runs_without_site_packages(self):
        source = Path(__file__).parent / "data" / "dense.osm.pbf"
        script = Path(__file__).resolve().parents[1] / "src" / "pbf.py"
        subprocess.run([sys.executable, "-S", str(script), str(source), *map(str, self.area),
                        "-o", str(self.output)], check=True, capture_output=True)
        self.assertEqual(len(ET.parse(self.output).getroot().findall("way")), 2)
