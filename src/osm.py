"""Download Geofabrik regions and read them with the bundled PBF extractor."""

import hashlib
import json
from collections import defaultdict
from contextlib import nullcontext
from itertools import pairwise

from .common import APP_AGENT, RequestError, atomic_path, write_bytes
from .geo import clip, extent, ring_contains, signed_area
from .pbf import PBF

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


class Source:
    def __init__(self, client):
        self.client = client
        self.directory = client.cache_dir / "geofabrik"
        self.index = None
        self.regions = {}

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
                stamp = PBF(local, self.client.check_cancel).timestamp
            self.regions[url] = path, dict(source_url=url, region=props["name"], osm_data_at=stamp or None)
        return self.regions[url]

    def export(self, area, output, progress):
        source, metadata = self.region(area, progress)
        progress("Extracting OSM map")
        PBF(source, self.client.check_cancel).export(area, output)
        return metadata

    def data(self, area, *, poi=False, progress=lambda *args: None):
        source, metadata = self.region(area, progress)
        progress("Reading nearby places" if poi else "Reading roads")
        pbf = PBF(source, self.client.check_cancel)
        local = pbf.select(area)
        selected = (set(), set(), set())
        keys = POI_KEYS if poi else ("highway",)
        if poi:
            for node in pbf.nodes(local[0], "objects"):
                if any(key in node["tags"] for key in keys):
                    selected[0].add(node["id"])
            for relation in pbf.relations(local[2]):
                if any(key in relation["tags"] for key in keys):
                    selected[2].add(relation["id"])
        for way in pbf.ways(local[1]):
            if any(key in way["tags"] for key in keys):
                selected[1].add(way["id"])
                selected[0].update(way["nodes"])
        if selected[2]:
            pbf.complete(selected)
        return pbf.read(selected, centers=poi), metadata

    def street(self, place, progress=lambda *args: None):
        identity = int(place["id"].split("/")[1])
        lat, lon = place["lat"], place["lon"]
        area = (lat - 1e-6, lon - 1e-6, lat + 1e-6, lon + 1e-6)
        while True:
            source, _ = self.region(area, progress)
            progress("Reading named street")
            pbf = PBF(source, self.client.check_cancel)
            seed = next(pbf.ways({identity}), None)
            tags = {} if seed is None else seed["tags"]
            if not tags.get("highway") or not tags.get("name"):
                raise RequestError("place_not_found", "The selected street is absent or unnamed in the Geofabrik snapshot.")
            name = tags["name"]
            ways = {way["id"]: way for way in pbf.ways(name=name)}
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
            nodes = {node for identity in connected for node in ways[identity]["nodes"]}
            selected = pbf.read((nodes, connected, set()))
            expanded = extent([area[:2], area[2:]] + [p for w in selected for p in w["points"]])
            next_source, _ = self.region(expanded, progress)
            if next_source == source:
                return name, selected
            area = expanded
