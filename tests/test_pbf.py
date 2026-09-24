"""Independent protobuf fixtures for optimized decoding and extract topology."""

import struct
import unittest
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from src import pbf


def uint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def packed(values, signed=False):
    return b"".join(uint(2 * v if v >= 0 else -2 * v - 1) if signed else uint(v) for v in values)


def field(number, value):
    if isinstance(value, bytes):
        return uint(number * 8 + 2) + uint(len(value)) + value
    return uint(number * 8) + uint(value)


def block(kind, payload):
    blob = field(2, len(payload)) + field(3, zlib.compress(payload))
    header = field(1, kind) + field(3, len(blob))
    return struct.pack(">I", len(header)) + header + blob


class PBFTests(unittest.TestCase):
    def test_packed_integer_fast_paths_match_independent_encoder(self):
        # Exercise byte translation, mixed runs, bulk lanes, and the long-varint
        # fallback on both sides of the 512-byte optimization threshold.
        cases = {
            "single_byte": list(range(64)) * 10,
            "short": [0, 63, 64, 127, 128, 8191, 8192, 2**20, 2**27 - 1],
            "bulk": [2**40] + [0, 64, 128, 8192, 2**20, 2**27 - 1] * 100,
            "mixed_runs": ([0] * 40 + [128, 2**40]) * 20,
            "long": [2**40, 2**62, 127] * 100,
        }
        for name, values in cases.items():
            for signed in (False, True):
                expected = [(-v if i % 2 else v) for i, v in enumerate(values)] if signed else values
                with self.subTest(case=name, signed=signed):
                    self.assertEqual(list(pbf.unpack(packed(expected, signed), signed)), expected)
        self.assertEqual(list(pbf.unpack(uint(2**64 - 1))), [2**64 - 1])
        self.assertEqual(list(pbf.unpack(packed([-2**63, 2**63 - 1], True), True)), [-2**63, 2**63 - 1])

    def test_truncated_or_overflowing_integers_fail_in_every_fast_path(self):
        prefixes = [b"", b"\x00" * 600, packed([128, 8192] * 200), packed([2**40] * 100)]
        for prefix in prefixes:
            for suffix in (b"\x80", uint(2**64)):
                for signed in (False, True):
                    with self.subTest(prefix_length=len(prefix), suffix=suffix, signed=signed):
                        with self.assertRaises(ValueError):
                            pbf.unpack(prefix + suffix, signed)

    def test_dense_columns_can_be_split_and_have_negative_deltas(self):
        message = (field(1, packed([100, 2], True)) + field(1, packed([3], True))
                   + field(8, packed([450000000, -20, 5], True))
                   + field(9, packed([90000000, 30, -40], True)))
        _, ids, lats, lons = pbf.dense(message)
        self.assertEqual(ids, [100, 102, 105])
        self.assertEqual(lats, [450000000, 449999980, 449999985])
        self.assertEqual(lons, [90000000, 90000030, 89999990])
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            pbf.dense(message + field(9, packed([1], True)))

    def test_blob_rejects_truncation_trailing_data_and_wrong_size(self):
        payload = b"OSM payload"
        compressed = zlib.compress(payload)
        self.assertEqual(pbf.inflate(field(2, len(payload)) + field(3, compressed)), payload)
        for blob in (field(3, compressed[:-1]), field(3, compressed + b"trailing"),
                     field(2, len(payload) + 1) + field(3, compressed)):
            with self.subTest(blob=blob), self.assertRaises(ValueError):
                pbf.inflate(blob)

    def test_export_completes_multipolygon_members_outside_selection(self):
        # Only node 1 is in the rectangle. Its outer way pulls in the other
        # outer way and the entirely outside hole, but not unrelated way 40.
        strings = [b"", b"type", b"multipolygon", b"outer", b"inner", b"name", b'A & "B"']
        nodes = b""
        for identity, lat, lon in [(1, 0, 0), (2, 0, 4), (3, 4, 4), (4, 4, 0),
                                   (5, 2, 2), (6, 2, 3), (7, 3, 2), (8, 8, 8), (9, 9, 9)]:
            node = (field(1, identity * 2) + field(8, lat * 10_000_000 * 2)
                    + field(9, lon * 10_000_000 * 2))
            nodes += field(1, node)
        ways = b""
        for identity, refs in [(10, [4, 1, 2]), (20, [2, 3, 4]), (30, [5, 6, 7, 5]), (40, [8, 9])]:
            deltas = [refs[0]] + [b - a for a, b in zip(refs, refs[1:])]
            ways += field(3, field(1, identity) + field(8, packed(deltas, True)))
        relation = (field(1, 100) + field(2, packed([1, 5])) + field(3, packed([2, 6]))
                    + field(8, packed([3, 3, 4])) + field(9, packed([10, 10, 10], True))
                    + field(10, packed([1, 1, 1])))
        data = (field(1, b"".join(field(1, s) for s in strings))
                + field(2, nodes) + field(2, ways + field(4, relation)))
        snapshot = (block(b"OSMHeader", field(4, b"OsmSchema-V0.6"))
                    + block(b"OSMData", data))
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / "region.pbf", Path(directory) / "map.osm"
            source.write_bytes(snapshot)
            pbf.PBF(source).export((-0.1, -0.1, 0.1, 0.1), target)
            root = ET.parse(target).getroot()
        self.assertEqual([int(n.get("id")) for n in root.findall("node")], list(range(1, 8)))
        self.assertEqual([w.get("id") for w in root.findall("way")], ["10", "20", "30"])
        self.assertEqual([n.get("ref") for n in root.find("way[@id='30']").findall("nd")], ["5", "6", "7", "5"])
        self.assertEqual([(m.get("ref"), m.get("role")) for m in root.find("relation").findall("member")],
                         [("10", "outer"), ("20", "outer"), ("30", "inner")])
        self.assertEqual(root.find("relation/tag[@k='name']").get("v"), 'A & "B"')
        self.assertEqual(float(root.find("node[@id='6']").get("lon")), 3.0)
