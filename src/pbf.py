"""Reader/extractor for sorted OSM snapshot PBFs.

Schema: https://github.com/openstreetmap/OSM-binary/tree/master/osmpbf
Supports raw/zlib blobs, ordinary/dense nodes, ways, relations and OSM metadata.
"""

import bisect
import math
import os
import re
import struct
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

import numpy as np

from .common import atomic_path

MAX_BLOB = 32 * 1024 * 1024
MEMBERS = ("node", "way", "relation")
SIGNED_BYTES = bytes(((v >> 1) ^ -(v & 1)) & 255 for v in range(256))
NUMPY_BYTES = 256  # Shorter packed fields decode faster in Python than through numpy's per-call overhead.
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


def varints(data, signed=False):
    """Packed varints as a uint64 array, or int64 when zigzag-encoded."""
    raw = np.frombuffer(data, np.uint8)
    if not raw.size:
        return np.zeros(0, np.int64 if signed else np.uint64)
    if raw[-1] >= 128:
        raise ValueError("Truncated PBF packed integer.")
    ends = np.flatnonzero(raw < 128)
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    # Ten bytes hold 64 bits only when the last carries a single bit.
    if lengths.max() > 10 or (raw[ends[lengths == 10]] > 1).any():
        raise ValueError("PBF integer overflow.")
    shifts = (np.arange(raw.size) - np.repeat(starts, lengths)).astype(np.uint64) * np.uint64(7)
    values = np.bitwise_or.reduceat((raw & 127).astype(np.uint64) << shifts, starts)
    if signed:
        return (values >> np.uint64(1)).astype(np.int64) ^ -(values & np.uint64(1)).astype(np.int64)
    return values


def unpack(data, signed=False):
    """Packed integers as a sequence of Python ints."""
    if len(data) >= NUMPY_BYTES:
        return varints(data, signed).tolist()
    if data.isascii():
        return array("b", data.translate(SIGNED_BYTES)) if signed else data
    values, value, shift = [], 0, 0
    for byte in data:
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 1 << 64:
                raise ValueError("PBF integer overflow.")
            values.append((value >> 1) ^ -(value & 1) if signed else value)
            value = shift = 0
        elif (shift := shift + 7) > 63:
            raise ValueError("Invalid PBF packed integer.")
    if shift:
        raise ValueError("Truncated PBF packed integer.")
    return values


def zigzag(value):
    return (value >> 1) ^ -(value & 1)


def signed64(value):
    return value - (1 << 64) if value >= 1 << 63 else value


def column(data):
    """A delta-coded column as an int64 array."""
    return np.cumsum(varints(data, True))


def deltas(data):
    return column(data).tolist() if len(data) >= NUMPY_BYTES else list(accumulate(unpack(data, True)))


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
    ids = deltas(values.get(1, b""))
    lats = deltas(values.get(8, b""))
    lons = deltas(values.get(9, b""))
    if not len(ids) == len(lats) == len(lons):
        raise ValueError("Mismatched dense PBF coordinate columns.")
    return values, ids, lats, lons


def scan_nodes(task):
    path, entry, area, retain = task
    values, groups = block(path, entry)
    gran = values.get(17, 100)
    lat_offset, lon_offset = signed64(values.get(19, 0)), signed64(values.get(20, 0))
    if area is not None:
        south, west, north, east = area
        south, north = math.ceil((south * 1e9 - lat_offset) / gran), math.floor((north * 1e9 - lat_offset) / gran)
        west, east = math.ceil((west * 1e9 - lon_offset) / gran), math.floor((east * 1e9 - lon_offset) / gran)
    hits, first, last, kinds = [], None, None, 0
    for group in groups:
        for kind, message in fields(group):
            if kind not in (1, 2, 3, 4):
                raise ValueError("Unsupported OSM primitive in PBF snapshot.")
            kinds |= 1 << kind
            if kind == 2:
                row = columns(message)
                ids = column(row.get(1, b""))
                if area is not None:
                    lats, lons = column(row.get(8, b"")), column(row.get(9, b""))
                    if not ids.size == lats.size == lons.size:
                        raise ValueError("Mismatched dense PBF coordinate columns.")
            elif kind == 1:
                node = dict(fields(message))
                ids, lats, lons = (np.array([zigzag(node[key])]) for key in (1, 8, 9))
            else:
                continue
            if ids.size:
                if (np.diff(ids) <= 0).any() or (last is not None and ids[0] <= last):
                    raise ValueError("PBF nodes must be sorted by unique ID.")
                first = int(ids[0]) if first is None else first
                last = int(ids[-1])
                if area is not None:
                    inside = (lats >= south) & (lats <= north) & (lons >= west) & (lons <= east)
                    hits.append((ids[inside], lats[inside], lons[inside]))
    selected, lats, lons = (np.concatenate(parts) for parts in zip(*hits)) if hits else [np.zeros(0, np.int64)] * 3
    coordinates = (array("q", lats.tobytes()), array("q", lons.tobytes())) if retain else None
    return kinds, first, last, array("q", selected.tobytes()), coordinates


def spatial_nodes(ids):
    global _SPATIAL
    _SPATIAL = np.sort(np.array(ids, np.int64))


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


def touching(messages):
    """Which ways reference a spatial node, decoding all of a block's references at once."""
    refs = [way_references(message) for message in messages]
    data = b"".join(refs)
    if not data:
        return np.zeros(len(messages), bool)
    # The way of each reference, from where its last byte falls among the ways' byte ranges.
    owners = np.searchsorted(np.cumsum([len(r) for r in refs]), np.flatnonzero(np.frombuffer(data, np.uint8) < 128),
                             side="right")
    totals = np.cumsum(varints(data, True))
    # Deltas restart in each way: subtract the running total reached before its first reference.
    firsts = np.searchsorted(owners, owners, side="left")
    nodes = totals - np.where(firsts > 0, totals[firsts - 1], 0)
    found = np.minimum(np.searchsorted(_SPATIAL, nodes), _SPATIAL.size - 1)
    hit = np.zeros(len(messages), bool)
    if _SPATIAL.size:
        hit[owners[_SPATIAL[found] == nodes]] = True
    return hit


def identity(message):
    # Geofabrik writes ID first.
    return varint(message, 1)[0] if message[:1] == b"\x08" else columns(message)[1]


def way_item(row, strings, date_gran):
    return dict(type="way", id=row[1], tags=tags(row, strings), nodes=deltas(row.get(8, b"")),
                attrs=info(dict(fields(row.get(4, b""))), strings, date_gran))


def relation_item(row, strings, date_gran):
    refs = deltas(row.get(9, b""))
    roles, kinds = unpack(row.get(8, b"")), unpack(row.get(10, b""))
    if not len(refs) == len(roles) == len(kinds):
        raise ValueError("Mismatched PBF relation member columns.")
    if any(kind > 2 for kind in kinds):
        raise ValueError("Invalid PBF relation member type.")
    return dict(type="relation", id=row[1], tags=tags(row, strings),
                members=[(MEMBERS[kind], ref, strings[role]) for kind, ref, role in zip(kinds, refs, roles)],
                attrs=info(dict(fields(row.get(4, b""))), strings, date_gran))


def scan_objects(task):
    """A block's ways or relations: all of them, those with wanted IDs, or highways with a name.

    Also returns the block's object kinds and the ID range of the requested kind.
    """
    path, entry, kind, wanted, name = task
    values, groups = block(path, entry)
    kinds, items, first, last = 0, [], None, None
    strings = StringTable(values[1])
    # Objects can only carry a name present in their block's string table.
    absent = name is not None and name.encode() not in strings.values
    wanted = None if wanted is None else set(wanted)
    for group in groups:
        for number, message in fields(group):
            kinds |= 1 << number
            if number != kind:
                continue
            current = identity(message)
            first, last = current if first is None else first, current
            if absent or (wanted is not None and current not in wanted):
                continue
            item = (way_item if kind == 3 else relation_item)(columns(message), strings, values.get(18, 1000))
            if name is None or (item["tags"].get("name") == name and item["tags"].get("highway")):
                items.append(item)
    return kinds, (first, last), items


def scan_ways(task):
    """Select spatial ways or requested member IDs and record the block's ID range."""
    path, entry, xml, wanted = task
    values, groups = block(path, entry)
    messages = [message for group in groups for kind, message in fields(group) if kind == 3]
    if not messages:
        raise ValueError("PBF way block contains no ways.")

    if wanted is not None:
        chosen = [message for message in messages if identity(message) in wanted]
    else:
        chosen = [message for message, hit in zip(messages, touching(messages)) if hit]
    selected, identities, nodes = [], array("q"), set()
    strings = StringTable(values[1]) if chosen else None
    for message in chosen:
        item = way_item(columns(message), strings, values.get(18, 1000))
        if xml:
            identities.append(item["id"])
            nodes.update(item["nodes"])
            selected.append(xml_object(item))
        else:
            selected.append(item)
    bounds = identity(messages[0]), identity(messages[-1])
    result = (identities, array("q", nodes), "".join(selected).encode()) if xml else selected
    return bounds, result


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


def xml_attrs(attrs):
    # Metadata values other than user names are generated numbers/ISO dates.
    return "".join(
        f' {key}={quoted(str(value))}' if key == "user" else f' {key}="{value}"'
        for key, value in attrs.items())


def xml_object(item):
    kind = item["type"]
    attributes = f'id="{item["id"]}"' + xml_attrs(item.get("attrs", {}))
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
                    ids = deltas(node.get(1, b""))
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
                                    metadata[key] = deltas(data)
                            else:
                                metadata[key] = unpack(data)
                            if len(metadata[key]) != len(ids):
                                raise ValueError("Mismatched dense PBF metadata columns.")
                if mode == "xml":
                    extra = {k: v for k, v in metadata.items() if k not in (1, 2)}
                    simple = 1 in metadata and 2 in metadata and all(v.count(v[0]) == len(v) for v in extra.values())
                    suffix = xml_attrs(info({k: v[0] for k, v in extra.items()}, strings, date_gran)) if simple else ""
                    precision = 7 if gran % 100 == lat_offset % 100 == lon_offset % 100 == 0 else 9
                    for i in selected:
                        lat, lon = (lat_offset + gran * lats[i]) / 1e9, (lon_offset + gran * lons[i]) / 1e9
                        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                            raise ValueError("Invalid PBF node coordinates.")
                        if simple:
                            attrs = f' version="{metadata[1][i]}" timestamp="{timestamp(metadata[2][i], date_gran)}"{suffix}'
                        else:
                            attrs = xml_attrs(info({k: v[i] for k, v in metadata.items()}, strings, date_gran))
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
        self.spans = {}  # Block offset → first and last way or relation ID.
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

    def objects(self, kind, wanted=None, name=None):
        """Ways (3) or relations (4) in file order, decoded in parallel; see scan_objects."""
        if wanted is not None:
            if not wanted:
                return
            wanted = sorted(wanted)
        tasks = []
        for entry in self.entries:
            if entry[2] and not entry[2] & 1 << kind:
                continue
            ids = wanted
            if wanted is not None and entry[0] in self.spans:
                # Blocks are sorted by ID, so send each only the IDs within its range.
                first, last = self.spans[entry[0]]
                ids = wanted[bisect.bisect_left(wanted, first):bisect.bisect_right(wanted, last)]
                if not ids:
                    continue
            tasks.append((entry, (self.path, entry, kind, None if ids is None else array("q", ids), name)))
        results = self.parallel(scan_objects, (task for _, task in tasks))
        for (entry, _), (kinds, span, items) in zip(tasks, results):
            entry[2] |= kinds
            if span[0] is not None:
                self.spans[entry[0]] = span
            yield from items

    def ways(self, wanted=None, name=None):
        if wanted is not None and wanted <= self.selected_ways.keys():
            for identity in sorted(wanted):
                yield self.selected_ways[identity]
            return
        for way in self.objects(3, wanted, name):
            if wanted is not None or name is not None:
                self.selected_ways[way["id"]] = way
            yield way

    def relations(self, wanted=None):
        if wanted is not None and wanted <= self.selected_relations.keys():
            for identity in sorted(wanted):
                yield self.selected_relations[identity]
            return
        yield from self.objects(4, wanted)

    def select(self, area, xml=False):
        self.way_xml = {}
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
        entries = [entry for entry in self.entries if entry[2] & 8]
        tasks = ((self.path, entry, xml, None) for entry in entries)
        way_blocks = []
        results = self.parallel(scan_ways, tasks, spatial_nodes, (array("q", spatial),))
        for entry, (bounds, selected) in zip(entries, results, strict=True):
            self.spans[entry[0]] = bounds
            if xml:
                identities, refs, chunk = selected
                ways.update(identities)
                nodes.update(refs)
                self.way_xml[entry[0]] = identities, chunk
                way_blocks.append((entry, bounds))
                continue
            for way in selected:
                ways.add(way["id"])
                nodes.update(way["nodes"])
                self.selected_ways[way["id"]] = way
        parents, pending = defaultdict(list), []
        rows, extra_ways = {}, set()
        for relation in self.relations():
            identity = relation["id"]
            rows[identity] = relation
            matched = False
            for kind, ref, _ in relation["members"]:
                if kind == "relation":
                    parents[ref].append(identity)
                elif ref in (spatial if kind == "node" else ways):
                    matched = True
            if matched:
                pending.append(identity)
                if xml and relation["tags"].get("type") == "multipolygon":
                    for kind, ref, _ in relation["members"]:
                        if kind == "node":
                            nodes.add(ref)
                        elif kind == "way":
                            extra_ways.add(ref)
        while pending:
            identity = pending.pop()
            if identity not in relations:
                relations.add(identity)
                pending.extend(parents[identity])
        self.selected_relations = {i: rows[i] for i in relations}
        if xml:
            # Revisit only blocks containing missing polygon members. Keep the
            # existing XML and merge added ways in ID order without duplicates.
            # Members absent from the regional file keep their original references.
            missing = sorted(extra_ways - ways)
            tasks, entries = [], []
            for entry, (first, last) in way_blocks:
                wanted = missing[bisect.bisect_left(missing, first):bisect.bisect_right(missing, last)]
                if wanted:
                    entries.append(entry)
                    tasks.append((self.path, entry, True, set(wanted)))
            if tasks:
                for entry, (_, (identities, refs, chunk)) in zip(entries, self.parallel(scan_ways, tasks)):
                    nodes.update(refs)
                    ways.update(identities)
                    previous, original = self.way_xml[entry[0]]
                    merged = sorted(chain(zip(previous, original.splitlines(keepends=True)),
                                          zip(identities, chunk.splitlines(keepends=True))))
                    self.way_xml[entry[0]] = (), b"".join(part for _, part in merged)
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
        south, west, north, east = area
        with atomic_path(output) as temporary, temporary.open("wb") as stream:
            stream.write((f'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="Aleph">\n'
                          f'<bounds minlat="{south}" minlon="{west}" maxlat="{north}" maxlon="{east}"/>\n').encode())
            written = 0
            for chunk in self.nodes(selected[0], "xml"):
                stream.write(chunk)
                written += chunk.count(b"<node ")
            if written != len(selected[0]):
                raise ValueError("PBF extract contains incomplete way geometry.")
            for _, chunk in self.way_xml.values():
                stream.write(chunk)
            for item in self.relations(selected[2]):
                stream.write(xml_object(item).encode("utf-8"))
            stream.write(b"</osm>\n")
            self.cancel()
