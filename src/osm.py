"""Download Geofabrik regions and select local OSM data with Osmium."""

import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from itertools import pairwise
from xml.etree import ElementTree as ET

import osmium

from .common import APP_AGENT, RequestError, atomic_path, write_bytes
from .geo import clip, extent, ring_contains, signed_area

INDEX = "https://download.geofabrik.de/index-v1.json"
POI_KEYS = ("amenity", "tourism", "shop", "leisure", "historic", "office")


def polygons(geometry):
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]


def covers(geometry, area):
    """Require the whole rectangle, including its interior, to lie in the region."""
    south, west, north, east = area
    shapes = polygons(geometry)
    for point in ((west, south), (west, north), (east, south), (east, north)):
        if not any(ring_contains(polygon[0], point) and not any(ring_contains(hole, point) for hole in polygon[1:])
                   for polygon in shapes):
            return False
    for polygon in shapes:
        for ring in polygon:
            for a, b in pairwise(ring):
                segment = clip((a[1], a[0]), (b[1], b[0]), area)
                if segment:
                    lat = (segment[0][0] + segment[1][0]) / 2
                    lon = (segment[0][1] + segment[1][1]) / 2
                    if south < lat < north and west < lon < east:
                        return False
    return True


def region_size(feature):
    return sum(abs(signed_area(polygon[0])) for polygon in polygons(feature["geometry"]))


def objects(path, cancel=lambda: None):
    """Stream complete OSM elements without retaining the XML tree."""
    try:
        with path.open("rb") as stream:
            parser = ET.iterparse(stream, events=("start", "end"))
            _, root = next(parser)
            if root.tag != "osm":
                raise ValueError("OSM file contains no osm root.")
            for index, (event, obj) in enumerate(parser):
                if index % 65536 == 0:
                    cancel()
                if event == "end" and obj.tag in ("node", "way", "relation"):
                    yield obj
                    root.clear()
    except (ET.ParseError, StopIteration) as error:
        raise ValueError("OSM file contains invalid XML.") from error


def read_data(objects, *, centers=False):
    """Copy selected Osmium objects, retaining node IDs for road topology."""
    nodes, items, boxes, relations = {}, [], {}, []
    kinds = {"n": "node", "w": "way", "r": "relation"}
    for obj in objects:
        identity, kind = obj.id, kinds[obj.type_str()]
        if kind == "node":
            lat, lon = obj.lat, obj.lon
            nodes[identity] = (lat, lon)
            if not centers:
                continue
            boxes[("node", identity)] = (lat, lon, lat, lon)
        tags = dict(obj.tags)
        item = dict(type=kind, id=identity, tags=tags)
        if kind == "way":
            refs = [n.ref for n in obj.nodes]
            if any(ref not in nodes for ref in refs):
                raise OSError("The Geofabrik extract contains incomplete way geometry.")
            points = [nodes[ref] for ref in refs]
            item.update(nodes=refs, points=points)
            if centers and points:
                boxes[("way", identity)] = extent(points)
        elif kind == "relation":
            item["members"] = [(kinds[m.type], m.ref) for m in obj.members]
            relations.append(item)
        if tags or kind != "node":
            items.append(item)
    pending = relations
    while pending:
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
        box = boxes.get((item["type"], item["id"]))
        if box:
            item["center"] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return items


class Source:
    def __init__(self, client):
        self.client = client
        self.directory = client.cache_dir / "geofabrik"
        self.index = None
        self.regions = {}

    @contextmanager
    def processing(self):
        self.client.check_cancel()
        try:
            yield
        except RuntimeError as error:
            raise OSError(f"Osmium failed: {error}") from error
        self.client.check_cancel()

    def scan(self, source, *filters, entities=osmium.osm.OBJECT):
        with self.processing(), osmium.io.Reader(source, entities) as reader:
            for index, obj in enumerate(osmium.OsmFileIterator(reader, *filters)):
                if index % 8192 == 0:
                    self.client.check_cancel()
                yield obj

    def region(self, area, progress):
        if self.index is None:
            path = self.directory / "index.json"
            refresh = self.client.refresh or not path.is_file()
            if refresh:
                progress("Finding Geofabrik region")
                data = self.client.get(INDEX, user_agent=APP_AGENT)
            else:
                data = path.read_bytes()
            index = json.loads(data)
            if not isinstance(index.get("features"), list):
                raise OSError("Geofabrik returned an invalid region index.")
            if refresh:
                write_bytes(path, data)
            self.index = sorted(index["features"], key=region_size)
        feature = next((f for f in self.index if f["properties"].get("urls", {}).get("pbf")
                        and covers(f["geometry"], area)), None)
        if feature is None:
            raise OSError("No Geofabrik region covers this rectangle. Select a smaller area.")
        props = feature["properties"]
        url = props["urls"]["pbf"]
        if url not in self.regions:
            path = self.directory / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".osm.pbf")
            refresh = self.client.refresh or not path.is_file()
            with (atomic_path(path) if refresh else nullcontext(path)) as local:
                if refresh:
                    label = f"Downloading Geofabrik {props['name']} (regional file)"
                    progress(label)
                    self.client.get(url, destination=local, timeout=180, user_agent=APP_AGENT,
                                    progress=lambda done, total: progress(label, done, total))
                with self.processing(), osmium.io.Reader(osmium.io.File(local, "pbf"), osmium.osm.NOTHING) as reader:
                    stamp = reader.header().get("osmosis_replication_timestamp")
            self.regions[url] = path, dict(source_url=url, region=props["name"], osm_data_at=stamp or None)
        return self.regions[url]

    def select(self, source, area):
        """Select the rectangle, complete its ways, and retain parent relations."""
        south, west, north, east = area
        box = osmium.osm.Box(west, south, east, north)
        spatial, selected = osmium.IdTracker(), osmium.IdTracker()
        parents, pending = defaultdict(list), []
        for obj in self.scan(source):
            if obj.is_node():
                if box.contains(obj.location):
                    spatial.add_node(obj.id)
                    selected.add_node(obj.id)
            elif obj.is_way():
                if spatial.contains_any_references(obj):
                    spatial.add_way(obj.id)
                    selected.add_way(obj.id)
                    selected.add_references(obj)
            else:
                if spatial.contains_any_references(obj):
                    pending.append(obj.id)
                for member in obj.members:
                    if member.type == "r":
                        parents[member.ref].append(obj.id)
        while pending:
            identity = pending.pop()
            if identity not in selected.relation_ids():
                selected.add_relation(identity)
                pending.extend(parents[identity])
        return selected

    def extract(self, source, area, output):
        selected = self.select(source, area)
        south, west, north, east = area
        header = osmium.io.Header()
        header.add_box(osmium.osm.Box(west, south, east, north))
        with self.processing(), osmium.SimpleWriter(osmium.io.File(output, "osm"), header=header, overwrite=True) as writer:
            for obj in self.scan(source, selected.id_filter()):
                writer.add(obj)

    def export(self, area, output, progress):
        source, metadata = self.region(area, progress)
        progress("Extracting OSM map")
        with atomic_path(output) as temporary:
            self.extract(source, area, temporary)
        return metadata

    def data(self, area, *, poi=False, progress=lambda *args: None):
        source, metadata = self.region(area, progress)
        progress("Reading nearby places" if poi else "Reading roads")
        local = self.select(source, area)
        selected = osmium.IdTracker()
        entities = osmium.osm.OBJECT if poi else osmium.osm.WAY
        keys = POI_KEYS if poi else ("highway",)
        add = {"n": selected.add_node, "w": selected.add_way, "r": selected.add_relation}
        for obj in self.scan(source, local.id_filter(), osmium.filter.KeyFilter(*keys), entities=entities):
            add[obj.type_str()](obj.id)
            selected.add_references(obj)
        if poi and len(selected.relation_ids()):
            # Complete only selected POIs, including nested relations outside the rectangle.
            with self.processing():
                while True:
                    self.client.check_cancel()
                    count = len(selected.relation_ids())
                    selected.complete_backward_references(source, relation_depth=1)
                    if len(selected.relation_ids()) == count:
                        break
        return read_data(self.scan(source, selected.id_filter()), centers=poi), metadata

    def street(self, place, progress=lambda *args: None):
        identity = int(place["id"].split("/")[1])
        lat, lon = place["lat"], place["lon"]
        area = (lat - 1e-6, lon - 1e-6, lat + 1e-6, lon + 1e-6)
        while True:
            source, _ = self.region(area, progress)
            progress("Reading named street")
            tags = next((dict(w.tags) for w in self.scan(source, osmium.filter.IdFilter([identity]),
                                                       entities=osmium.osm.WAY)), {})
            if not tags.get("highway") or not tags.get("name"):
                raise RequestError("place_not_found", "The selected street is absent or unnamed in the Geofabrik snapshot.")
            name = tags["name"]
            selected = osmium.IdTracker()
            for way in self.scan(source, osmium.filter.TagFilter(("name", name)), osmium.filter.KeyFilter("highway"),
                                 entities=osmium.osm.WAY):
                selected.add_way(way.id)
                selected.add_references(way)
            ways = {w["id"]: w for w in read_data(self.scan(source, selected.id_filter()))}
            by_node = defaultdict(list)
            for way in ways.values():
                for node in way["nodes"]:
                    by_node[node].append(way["id"])
            connected, pending = set(), [identity]
            while pending:
                current = pending.pop()
                if current in connected:
                    continue
                connected.add(current)
                for node in ways[current]["nodes"]:
                    pending.extend(by_node.pop(node, ()))
            selected = [ways[i] for i in sorted(connected)]
            expanded = extent([area[:2], area[2:]] + [p for w in selected for p in w["points"]])
            next_source, _ = self.region(expanded, progress)
            if next_source == source:
                return name, selected
            area = expanded
