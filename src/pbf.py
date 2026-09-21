"""Standalone, standard-library reader/extractor for sorted OSM snapshot PBFs.

Run: python src/pbf.py REGION.osm.pbf S W N E -o map.osm
Schema: https://github.com/openstreetmap/OSM-binary/tree/master/osmpbf
Supports raw/zlib blobs, ordinary/dense nodes, ways, relations and OSM metadata.
"""

import argparse
import bisect
import math
import os
import re
import struct
import sys
import tempfile
import zlib
from array import array
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from datetime import datetime, timezone
from functools import lru_cache
from itertools import accumulate, chain
from multiprocessing import get_context
from pathlib import Path
from xml.sax.saxutils import quoteattr

MAX_BLOB = 32 * 1024 * 1024
KINDS = {1: "node", 2: "node", 3: "way", 4: "relation"}
MEMBERS = ("node", "way", "relation")
POI_KEYS = ("amenity", "tourism", "shop", "leisure", "historic", "office")
SIGNED_SMALL = tuple((v >> 1) ^ -(v & 1) for v in range(128))
SIGNED_BYTES = bytes(((v >> 1) ^ -(v & 1)) & 255 for v in range(256))
MULTIBYTE = re.compile(rb"[\x80-\xff]")
VARINTS = re.compile(rb"[\x80-\xff]+[\x00-\x7f]")
SHORT_RUN = re.compile(rb"[\x00-\x7f]{32}")
LONG_VARINT = re.compile(rb"[\x80-\xff]{4}")
TAG_ROWS = re.compile(rb"(?<![\x80-\xff])\x00")
quoted = lru_cache(maxsize=8192)(quoteattr)
_SPATIAL = None


def varint(data, offset):
    value = shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 1 << 64:
                break
            return value, offset
        shift += 7
    raise ValueError("Invalid PBF varint.")


def fields(data):
    offset = 0
    while offset < len(data):
        key = data[offset]
        offset += 1
        if key >= 128:
            key, offset = varint(data, offset - 1)
        number, wire = key >> 3, key & 7
        if not number:
            raise ValueError("Invalid PBF field number.")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                if offset >= len(data):
                    raise ValueError("Truncated PBF length.")
                size = data[offset]
                offset += 1
                if size >= 128:
                    size, offset = varint(data, offset - 1)
            else:
                size = 8 if wire == 1 else 4
            end = offset + size
            if end > len(data):
                raise ValueError("Truncated PBF field.")
            value, offset = data[offset:end], end
        else:
            raise ValueError("Unsupported PBF wire type.")
        yield number, value


@lru_cache(maxsize=16)
def lanes(count):
    """One bit at the start of each 32-bit lane of a Python integer."""
    return int.from_bytes(b"\x01\0\0\0" * count, "little")


def unpack_bulk(data, signed):
    """Decode short varints together using C-backed bytes and big integers.

    Each input byte becomes a 32-bit lane. Two masked shifts combine up to
    four base-128 digits. Mark continuation lanes with 0x80, delete them in
    C, then compact the remaining digits into ordinary 32-bit integers.
    """
    count = len(data)
    one = lanes(count)
    low = one * 127
    raw = int.from_bytes(data.decode("latin1").encode("utf-32le"), "little")
    more = (raw >> 7) & one
    value = raw & low
    mask = more * 0xffffffff
    value |= ((value >> 32) << 7) & mask
    mask &= mask >> 32
    value |= ((value >> 64) << 14) & mask
    drop = (more << 32) & one
    value = ((value & low) | ((value & (low << 7)) << 1)
             | ((value & (low << 14)) << 2) | ((value & (low << 21)) << 3))
    value = (value & ((one ^ drop) * 0xffffffff)) | (drop * 0x80808080)
    packed = value.to_bytes(count * 4, "little").translate(None, b"\x80")
    one = lanes(len(packed) // 4)
    low = one * 127
    value = int.from_bytes(packed, "little")
    value = ((value & low) | ((value & (low << 8)) >> 1)
             | ((value & (low << 16)) >> 2) | ((value & (low << 24)) >> 3))
    if signed:
        value = ((value >> 1) & (one * 0x7fffffff)) ^ ((value & one) * 0xffffffff)
    result = array("i" if signed else "I", value.to_bytes(len(packed), "little"))
    if sys.byteorder != "little":
        result.byteswap()
    return result


def unpack(data, signed=False):
    """Decode packed integers, with a C-level fast path for single-byte values."""
    if not MULTIBYTE.search(data):
        return array("b", data.translate(SIGNED_BYTES)) if signed else data
    if len(data) > 512 and SHORT_RUN.search(data):
        values, offset = [], 0
        for match in VARINTS.finditer(data):
            tail = data[offset:match.start()]
            values.extend(array("b", tail.translate(SIGNED_BYTES)) if signed else tail)
            value, _ = varint(match[0], 0)
            values.append(zigzag(value) if signed else value)
            offset = match.end()
        tail = data[offset:]
        if MULTIBYTE.search(tail):
            raise ValueError("Truncated PBF packed integer.")
        values.extend(array("b", tail.translate(SIGNED_BYTES)) if signed else tail)
        return values
    if len(data) > 512:
        first, offset = varint(data, 0)
        tail = data[offset:]
        if not LONG_VARINT.search(tail):
            if tail and tail[-1] >= 128:
                raise ValueError("Truncated PBF packed integer.")
            values = [zigzag(first) if signed else first]
            values.extend(unpack_bulk(tail, signed))
            return values
    values = []
    packed = iter(data)
    small = SIGNED_SMALL if signed else range(128)
    try:
        for byte in packed:
            if byte < 128:
                values.append(small[byte])
                continue
            value = byte & 127
            byte = next(packed)
            value |= (byte & 127) << 7
            if byte >= 128:
                byte = next(packed)
                value |= (byte & 127) << 14
                shift = 21
                while byte >= 128:
                    byte = next(packed)
                    value |= (byte & 127) << shift
                    shift += 7
                    if shift > 70:
                        raise ValueError("Invalid PBF packed integer.")
                if value >= 1 << 64:
                    raise ValueError("PBF integer overflow.")
            values.append((value >> 1) ^ -(value & 1) if signed else value)
    except StopIteration:
        raise ValueError("Truncated PBF packed integer.") from None
    return values


def zigzag(value):
    return (value >> 1) ^ -(value & 1)


def signed64(value):
    return value - (1 << 64) if value >= 1 << 63 else value


def inflate(data):
    blob = dict(fields(data))
    if 1 in blob:
        raw = blob[1]
    elif 3 in blob:
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(blob[3], MAX_BLOB + 1)
        except zlib.error as error:
            raise ValueError("Invalid PBF zlib data.") from error
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Truncated or oversized PBF zlib blob.")
    else:
        raise ValueError("PBF compression must be raw or zlib.")
    if len(raw) > MAX_BLOB or (2 in blob and len(raw) != blob[2]):
        raise ValueError("Invalid PBF blob size.")
    return raw


def block(path, entry):
    with open(path, "rb") as stream:
        stream.seek(entry[0])
        data = stream.read(entry[1])
    if len(data) != entry[1]:
        raise ValueError("Truncated PBF blob.")
    values, groups = {}, []
    for key, value in fields(inflate(data)):
        if key == 2:
            groups.append(value)
        else:
            values[key] = value
    if 1 not in values:
        raise ValueError("PBF block has no string table.")
    if not 0 < values.get(17, 100) < 1 << 31 or not 0 < values.get(18, 1000) < 1 << 31:
        raise ValueError("Invalid PBF coordinate or date granularity.")
    return values, groups


def columns(message):
    values = {}
    for key, value in fields(message):
        # Packed repeated fields may occur in more than one segment.
        if key in values and isinstance(value, bytes):
            values[key] += value
        else:
            values[key] = value
    return values


def dense(message):
    values = columns(message)
    ids = list(accumulate(unpack(values.get(1, b""), True)))
    lats = list(accumulate(unpack(values.get(8, b""), True)))
    lons = list(accumulate(unpack(values.get(9, b""), True)))
    if not len(ids) == len(lats) == len(lons):
        raise ValueError("Mismatched dense PBF coordinate columns.")
    return values, ids, lats, lons


def scan_nodes(task):
    path, entry, area, retain = task
    values, groups = block(path, entry)
    gran = values.get(17, 100)
    lat_offset, lon_offset = signed64(values.get(19, 0)), signed64(values.get(20, 0))
    if gran <= 0:
        raise ValueError("Invalid PBF coordinate granularity.")
    if area is not None:
        south, west, north, east = area
        south, north = math.ceil((south * 1e9 - lat_offset) / gran), math.floor((north * 1e9 - lat_offset) / gran)
        west, east = math.ceil((west * 1e9 - lon_offset) / gran), math.floor((east * 1e9 - lon_offset) / gran)
    selected, first, last, kinds = array("q"), None, None, 0
    coordinates = (array("q"), array("q")) if retain else None
    for group in groups:
        for kind, message in fields(group):
            if kind not in KINDS:
                raise ValueError("Unsupported OSM primitive in PBF snapshot.")
            kinds |= 1 << kind
            if kind == 2:
                if area is None:
                    ids = list(accumulate(unpack(columns(message).get(1, b""), True)))
                else:
                    _, ids, lats, lons = dense(message)
            elif kind == 1:
                node = dict(fields(message))
                ids, lats, lons = [zigzag(node[1])], [zigzag(node[8])], [zigzag(node[9])]
            else:
                continue
            if ids:
                if any(a >= b for a, b in zip(ids, ids[1:])) or (last is not None and ids[0] <= last):
                    raise ValueError("PBF nodes must be sorted by unique ID.")
                first = ids[0] if first is None else first
                last = ids[-1]
                if area is not None:
                    if retain:
                        hits = [i for i, (lat, lon) in enumerate(zip(lats, lons))
                                if south <= lat <= north and west <= lon <= east]
                        selected.extend(ids[i] for i in hits)
                        coordinates[0].extend(lats[i] for i in hits)
                        coordinates[1].extend(lons[i] for i in hits)
                    else:
                        selected.extend(identity for identity, lat, lon in zip(ids, lats, lons)
                                        if south <= lat <= north and west <= lon <= east)
    return kinds, first, last, selected, coordinates


def spatial_nodes(ids):
    global _SPATIAL
    _SPATIAL = set(ids)


def way_references(message):
    """Skip tags and metadata without allocating a dictionary for every way."""
    offset, end = 0, len(message)
    refs = b""
    while offset < end:
        key = message[offset]
        offset += 1
        if key >= 128 or key & 7 not in (0, 2):
            return columns(message).get(8, b"")
        if key & 7 == 0:
            while offset < end and message[offset] >= 128:
                offset += 1
            offset += 1
        else:
            size = message[offset]
            offset += 1
            if size >= 128:
                size, offset = varint(message, offset - 1)
            stop = offset + size
            if stop > end:
                raise ValueError("Truncated PBF way.")
            if key == 66:
                refs += message[offset:stop]
            offset = stop
    return refs


def references(data):
    if not data:
        return iter(())
    first, offset = varint(data, 0)
    return accumulate(chain((zigzag(first),), unpack(data[offset:], True)))


def scan_ways(task):
    path, entry, xml = task
    values, groups = block(path, entry)
    selected, identities, nodes = [], array("q"), set()
    strings = None
    for group in groups:
        for kind, message in fields(group):
            if kind == 3:
                packed = way_references(message)
                if not _SPATIAL.isdisjoint(references(packed)):
                    row = columns(message)
                    refs = list(references(packed))
                    if strings is None:
                        strings = StringTable(values[1])
                    item = dict(type="way", id=row[1], nodes=refs, tags=tags(row, strings),
                                attrs=info(dict(fields(row.get(4, b""))), strings, values.get(18, 1000)))
                    if xml:
                        identities.append(row[1])
                        nodes.update(refs)
                        selected.append(xml_object(item))
                    else:
                        selected.append(item)
    return (identities, array("q", nodes), "".join(selected).encode()) if xml else selected


class StringTable:
    """Decode UTF-8 only for strings referenced by selected objects."""

    def __init__(self, data):
        self.values = [value for key, value in fields(data) if key == 1]

    def __getitem__(self, index):
        try:
            if index < 0:
                raise IndexError(index)
            value = self.values[index]
            if isinstance(value, bytes):
                value = self.values[index] = value.decode("utf-8")
            return value
        except IndexError as error:
            raise ValueError("PBF string index is out of range.") from error


def tags(values, strings):
    keys, vals = unpack(values.get(2, b"")), unpack(values.get(3, b""))
    if len(keys) != len(vals):
        raise ValueError("Mismatched PBF tag columns.")
    return {strings[k]: strings[v] for k, v in zip(keys, vals)}


@lru_cache(maxsize=65536)
def timestamp(value, granularity=1000):
    return datetime.fromtimestamp(value * granularity / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def info(values, strings, date_gran):
    attrs = {}
    for key, value in values.items():
        if key == 1:
            attrs["version"] = value
        elif key == 2:
            attrs["timestamp"] = timestamp(value, date_gran)
        elif key == 3 and value:
            attrs["changeset"] = value
        elif key == 4 and value:
            attrs["uid"] = value
        elif key == 5 and value:
            attrs["user"] = strings[value]
        elif key == 6:
            attrs["visible"] = "true" if value else "false"
    return attrs


def xml_object(item):
    kind = item["type"]
    # IDs, coordinates and metadata values are generated numbers/ISO dates.
    # Only user names, tags and member roles need XML escaping.
    attributes = f'id="{item["id"]}"' + "".join(
        f' {key}={quoted(str(value))}' if key == "user" else f' {key}="{value}"'
        for key, value in item.get("attrs", {}).items())
    children = []
    if kind == "node":
        lat, lon = (f"{item[key]:.9f}".rstrip("0").rstrip(".") for key in ("lat", "lon"))
        attributes += f' lat="{lat}" lon="{lon}"'
    elif kind == "way":
        children = [f'<nd ref="{ref}"/>' for ref in item["nodes"]]
    else:
        children = [f'<member type="{kind}" ref="{ref}" role={quoted(role)}/>'
                    for kind, ref, role in item["members"]]
    children.extend(f'<tag k={quoted(k)} v={quoted(v)}/>' for k, v in item["tags"].items())
    return (f"<{kind} {attributes}>" + "".join(children) + f"</{kind}>\n" if children
            else f"<{kind} {attributes}/>\n")


def read_nodes(task):
    path, entry, wanted, mode, coordinates = task
    known = dict(zip(wanted, zip(*coordinates))) if coordinates else None
    wanted = set(wanted)
    values, groups = block(path, entry)
    strings = StringTable(values[1]) if mode != "points" else []
    gran, date_gran = values.get(17, 100), values.get(18, 1000)
    lat_offset, lon_offset = signed64(values.get(19, 0)), signed64(values.get(20, 0))
    output = []
    for group in groups:
        for kind, message in fields(group):
            if kind not in (1, 2):
                continue
            if kind == 1:
                node = dict(fields(message))
                identity = zigzag(node[1])
                if identity not in wanted:
                    continue
                rows = [(identity, zigzag(node[8]), zigzag(node[9]),
                         tags(node, strings) if mode != "points" else {},
                         info(dict(fields(node.get(4, b""))), strings, date_gran) if mode == "xml" else {})]
            else:
                if known is None:
                    node, ids, lats, lons = dense(message)
                else:
                    node = columns(message)
                    ids = list(accumulate(unpack(node.get(1, b""), True)))
                    lats, lons = [0] * len(ids), [0] * len(ids)
                if len(wanted) * 8 < len(ids):
                    selected = sorted(i for identity in wanted
                                      if (i := bisect.bisect_left(ids, identity)) < len(ids) and ids[i] == identity)
                else:
                    selected = [i for i, identity in enumerate(ids) if identity in wanted]
                if not selected:
                    continue
                if known is not None:
                    for i in selected:
                        lats[i], lons[i] = known[ids[i]]
                tag_rows, metadata = {}, {}
                if mode != "points":
                    data = node.get(10, b"")
                    if data:
                        rows = TAG_ROWS.split(data)
                        if len(rows) != len(ids) + 1 or rows[-1]:
                            raise ValueError("Mismatched dense PBF tag rows.")
                        for i in selected:
                            if rows[i]:
                                packed = unpack(rows[i])
                                if len(packed) % 2:
                                    raise ValueError("Truncated dense PBF tags.")
                                tag_rows[i] = {strings[k]: strings[v] for k, v in zip(packed[::2], packed[1::2])}
                    if mode == "xml":
                        for key, data in columns(node.get(5, b"")).items():
                            if data and key in (2, 3, 4, 5):
                                first, offset = varint(data, 0)
                                if not data[offset:].strip(b"\0"):
                                    metadata[key] = [zigzag(first)] * (len(data) - offset + 1)
                                else:
                                    metadata[key] = list(accumulate(unpack(data, True)))
                            else:
                                metadata[key] = unpack(data)
                            if len(metadata[key]) != len(ids):
                                raise ValueError("Mismatched dense PBF metadata columns.")
                if mode == "xml":
                    extra = {k: v for k, v in metadata.items() if k not in (1, 2)}
                    simple = 1 in metadata and 2 in metadata and all(v.count(v[0]) == len(v) for v in extra.values())
                    suffix = "".join(f' {k}={quoted(str(v))}' for k, v in
                                     info({k: v[0] for k, v in extra.items()}, strings, date_gran).items()) if simple else ""
                    precision = 7 if gran % 100 == lat_offset % 100 == lon_offset % 100 == 0 else 9
                    for i in selected:
                        lat, lon = (lat_offset + gran * lats[i]) / 1e9, (lon_offset + gran * lons[i]) / 1e9
                        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                            raise ValueError("Invalid PBF node coordinates.")
                        if simple:
                            attrs = f' version="{metadata[1][i]}" timestamp="{timestamp(metadata[2][i], date_gran)}"{suffix}'
                        else:
                            attrs = "".join(f' {k}={quoted(str(v))}' for k, v in
                                            info({k: v[i] for k, v in metadata.items()}, strings, date_gran).items())
                        head = f'<node id="{ids[i]}"{attrs} lat="{lat:.{precision}f}" lon="{lon:.{precision}f}"'
                        node_tags = tag_rows.get(i)
                        output.append(head + ">" + "".join(f'<tag k={quoted(k)} v={quoted(v)}/>'
                                      for k, v in node_tags.items()) + "</node>\n" if node_tags else head + "/>\n")
                    continue
                rows = ((ids[i], lats[i], lons[i], tag_rows.get(i, {}), {}) for i in selected)
            for identity, lat, lon, node_tags, attrs in rows:
                lat, lon = (lat_offset + gran * lat) / 1e9, (lon_offset + gran * lon) / 1e9
                if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    raise ValueError("Invalid PBF node coordinates.")
                if mode == "points":
                    output.append((identity, lat, lon))
                else:
                    item = dict(type="node", id=identity, lat=lat, lon=lon, tags=node_tags, attrs=attrs)
                    output.append(xml_object(item) if mode == "xml" else item)
    return "".join(output).encode("utf-8") if mode == "xml" else output


class PBF:
    """Read a Geofabrik snapshot; all indexing is temporary and in memory."""

    def __init__(self, path, cancel=lambda: None):
        self.path, self.cancel = Path(path), cancel
        self.entries = []
        self.selected_ways = {}
        self.selected_relations = {}
        self.selected_coordinates = {}
        self.timestamp = None
        seen_header = False
        with self.path.open("rb") as stream:
            size = self.path.stat().st_size
            while stream.tell() < size:
                cancel()
                length = stream.read(4)
                if len(length) != 4 or not 0 < (length := struct.unpack(">I", length)[0]) < 65536:
                    raise ValueError("Invalid PBF block header.")
                header = dict(fields(stream.read(length)))
                count = header.get(3, 0)
                if not 0 < count <= MAX_BLOB or stream.tell() + count > size:
                    raise ValueError("Truncated or oversized PBF block.")
                if header.get(1) == b"OSMHeader" and not seen_header:
                    seen_header = True
                    for key, value in fields(inflate(stream.read(count))):
                        if key == 4 and value not in (b"OsmSchema-V0.6", b"DenseNodes"):
                            raise ValueError(f"Unsupported required PBF feature: {value!r}")
                        if key == 32:
                            self.timestamp = timestamp(value)
                elif header.get(1) == b"OSMData" and seen_header:
                    self.entries.append([stream.tell(), count, 0, None, None])
                    stream.seek(count, 1)
                else:
                    raise ValueError("Unsupported PBF block type or order.")
        if not seen_header:
            raise ValueError("PBF snapshot contains no header.")

    def parallel(self, function, tasks, initializer=None, initargs=()):
        workers = min(8, os.cpu_count() or 1) if self.path.stat().st_size > 16 * 1024 * 1024 else 1
        if workers == 1:
            if initializer is not None:
                initializer(*initargs)
            for task in tasks:
                self.cancel()
                yield function(task)
            return
        pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=initializer, initargs=initargs)
        pending, tasks = deque(), iter(tasks)
        try:
            for _ in range(workers * 8):
                if (task := next(tasks, None)) is not None:
                    pending.append(pool.submit(function, task))
            while pending:
                self.cancel()
                try:
                    result = pending[0].result(timeout=0.2)
                except TimeoutError:
                    continue
                pending.popleft()
                if (task := next(tasks, None)) is not None:
                    pending.append(pool.submit(function, task))
                yield result
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    def raw(self, kinds, wanted=None):
        if wanted is not None and not wanted:
            return
        for entry in self.entries:
            self.cancel()
            if entry[2] and not any(entry[2] & (1 << k) for k in kinds):
                continue
            values, groups = block(self.path, entry)
            strings = None
            for group in groups:
                for kind, message in fields(group):
                    entry[2] |= 1 << kind
                    if kind not in kinds:
                        continue
                    # Geofabrik writes ID first. Skip other fields for unselected objects.
                    if wanted is not None and message[:1] == b"\x08" and varint(message, 1)[0] not in wanted:
                        continue
                    row = columns(message)
                    if wanted is not None and row[1] not in wanted:
                        continue
                    if strings is None:
                        strings = StringTable(values[1])
                    yield kind, row, strings, values.get(18, 1000)

    def ways(self, wanted=None, name=None):
        if wanted is not None and wanted <= self.selected_ways.keys():
            for identity in sorted(wanted):
                yield self.selected_ways[identity]
            return
        for _, row, strings, date_gran in self.raw((3,), wanted):
            properties = tags(row, strings)
            if name is not None and (properties.get("name") != name or not properties.get("highway")):
                continue
            way = dict(type="way", id=row[1], tags=properties,
                       nodes=list(accumulate(unpack(row.get(8, b""), True))),
                       attrs=info(dict(fields(row.get(4, b""))), strings, date_gran))
            if wanted is not None or name is not None:
                self.selected_ways[way["id"]] = way
            yield way

    def relations(self, wanted=None):
        rows = ((4, *self.selected_relations[i]) for i in sorted(wanted)) if (
            wanted is not None and wanted <= self.selected_relations.keys()) else self.raw((4,), wanted)
        for _, row, strings, date_gran in rows:
            refs = list(accumulate(unpack(row.get(9, b""), True)))
            roles, kinds = unpack(row.get(8, b"")), unpack(row.get(10, b""))
            if not len(refs) == len(roles) == len(kinds):
                raise ValueError("Mismatched PBF relation member columns.")
            if any(kind > 2 for kind in kinds):
                raise ValueError("Invalid PBF relation member type.")
            yield dict(type="relation", id=row[1], tags=tags(row, strings),
                       members=[(MEMBERS[kind], ref, strings[role]) for kind, ref, role in zip(kinds, refs, roles)],
                       attrs=info(dict(fields(row.get(4, b""))), strings, date_gran))

    def select(self, area, xml=False):
        self.way_xml = []
        self.selected_coordinates.clear()
        spatial, nodes, ways, relations = set(), set(), set(), set()
        tasks = ((self.path, entry, area, xml) for entry in self.entries)
        last = None
        for entry, (kinds, first, end, selected, coordinates) in zip(self.entries, self.parallel(scan_nodes, tasks)):
            entry[2:] = kinds, first, end
            if first is not None:
                if last is not None and first <= last:
                    raise ValueError("PBF node blocks must be sorted by unique ID.")
                last = end
            spatial.update(selected)
            if coordinates is not None and selected:
                self.selected_coordinates[entry[0]] = selected, coordinates
        nodes.update(spatial)
        tasks = ((self.path, entry, xml) for entry in self.entries if entry[2] & 8)
        for selected in self.parallel(scan_ways, tasks, spatial_nodes, (array("q", spatial),)):
            if xml:
                identities, refs, chunk = selected
                ways.update(identities)
                nodes.update(refs)
                self.way_xml.append(chunk)
                continue
            for way in selected:
                ways.add(way["id"])
                nodes.update(way["nodes"])
                self.selected_ways[way["id"]] = way
        parents, pending = defaultdict(list), []
        rows = {}
        for _, row, strings, date_gran in self.raw((4,)):
            identity = row[1]
            rows[identity] = row, strings, date_gran
            refs = accumulate(unpack(row.get(9, b""), True))
            for kind, ref in zip(unpack(row.get(10, b"")), refs):
                if kind == 2:
                    parents[ref].append(identity)
                elif ref in (spatial if kind == 0 else ways):
                    pending.append(identity)
        while pending:
            identity = pending.pop()
            if identity not in relations:
                relations.add(identity)
                pending.extend(parents[identity])
        self.selected_relations = {i: rows[i] for i in relations}
        return nodes, ways, relations

    def nodes(self, wanted, mode="points"):
        ordered = sorted(wanted)
        if not ordered:
            return
        unknown = [entry for entry in self.entries if not entry[2] or (entry[2] & 6 and entry[3] is None)]
        for entry, (kinds, first, last, _, _) in zip(
                unknown, self.parallel(scan_nodes, ((self.path, e, None, False) for e in unknown))):
            entry[2:] = kinds, first, last
        tasks = []
        for entry in self.entries:
            if entry[3] is not None:
                ids = ordered[bisect.bisect_left(ordered, entry[3]):bisect.bisect_right(ordered, entry[4])]
                if ids:
                    saved = self.selected_coordinates.get(entry[0])
                    coordinates = saved[1] if saved is not None and saved[0] == array("q", ids) else None
                    tasks.append((self.path, entry, ids, mode, coordinates))
        for rows in self.parallel(read_nodes, tasks):
            self.cancel()
            if mode == "xml":
                yield rows
            else:
                yield from rows

    def complete(self, selected):
        nodes, ways, relations = selected
        pending = set(relations)
        while pending:
            unseen = set()
            for relation in self.relations(pending):
                for kind, ref, _ in relation["members"]:
                    if kind == "node":
                        nodes.add(ref)
                    elif kind == "way":
                        ways.add(ref)
                    elif ref not in relations:
                        unseen.add(ref)
            relations.update(unseen)
            pending = unseen
        for way in self.ways(ways):
            nodes.update(way["nodes"])

    def read(self, selected, centers=False):
        wanted_nodes, wanted_ways, wanted_relations = selected
        points, items, boxes = {}, [], {}
        for node in self.nodes(wanted_nodes, "objects" if centers else "points"):
            identity, lat, lon = (node["id"], node["lat"], node["lon"]) if centers else node
            points[identity] = lat, lon
            if centers:
                boxes[("node", identity)] = lat, lon, lat, lon
                if node["tags"]:
                    items.append(dict(type="node", id=identity, tags=node["tags"], center=(lat, lon)))
        for way in self.ways(wanted_ways):
            if any(ref not in points for ref in way["nodes"]):
                raise ValueError("PBF extract contains incomplete way geometry.")
            geometry = [points[ref] for ref in way["nodes"]]
            items.append(dict(type="way", id=way["id"], tags=way["tags"], nodes=way["nodes"], points=geometry))
            if centers and geometry:
                lats, lons = zip(*geometry)
                boxes[("way", way["id"])] = min(lats), min(lons), max(lats), max(lons)
        pending = [dict(type="relation", id=r["id"], tags=r["tags"], members=[m[:2] for m in r["members"]])
                   for r in self.relations(wanted_relations)]
        items.extend(pending)
        while centers and pending:
            unresolved = []
            for item in pending:
                members = [boxes[ref] for ref in item["members"] if ref in boxes]
                if not members or len(members) != len(item["members"]):
                    unresolved.append(item)
                    continue
                boxes[("relation", item["id"])] = (min(b[0] for b in members), min(b[1] for b in members),
                                                    max(b[2] for b in members), max(b[3] for b in members))
            if len(unresolved) == len(pending):
                break
            pending = unresolved
        for item in items:
            if (box := boxes.get((item["type"], item["id"]))) is not None:
                item["center"] = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return items

    def export(self, area, output):
        selected = self.select(area, xml=True)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                south, west, north, east = area
                stream.write((f'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="Aleph">\n'
                              f'<bounds minlat="{south}" minlon="{west}" maxlat="{north}" maxlon="{east}"/>\n').encode())
                written = 0
                for chunk in self.nodes(selected[0], "xml"):
                    stream.write(chunk)
                    written += chunk.count(b"<node ")
                if written != len(selected[0]):
                    raise ValueError("PBF extract contains incomplete way geometry.")
                for chunk in self.way_xml:
                    stream.write(chunk)
                for item in self.relations(selected[2]):
                    stream.write(xml_object(item).encode("utf-8"))
                stream.write(b"</osm>\n")
            self.cancel()
            os.replace(name, output)
        finally:
            Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bbox", nargs=4, type=float, metavar="COORD")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    south, west, north, east = args.bbox
    if not all(math.isfinite(v) for v in args.bbox) or not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        parser.error("Use a nonempty bounding box in south west north east order.")
    try:
        PBF(args.source).export(args.bbox, args.output)
    except (OSError, ValueError, zlib.error, KeyError, IndexError) as error:
        parser.exit(1, f"PBF extraction failed: {error}\n")


if __name__ == "__main__":
    main()
