"""Convert saved OSM to just the geometry and properties used by the dashboard."""

import math
import re
from collections import defaultdict
from xml.etree import ElementTree as ET

from .geo import feature, ring_contains, signed_area

DISPLAY_TAGS = {"building", "landuse", "natural", "water", "highway", "waterway"}
TAGS = DISPLAY_TAGS | {"name", "height", "building:levels", "min_height", "type"}
AREA_TAGS = {
    "building", "landuse", "amenity", "leisure", "boundary", "place", "shop",
    "tourism", "historic", "public_transport", "office", "building:part", "military",
    "ruins", "area:highway", "craft", "golf", "indoor",
}
AREA_VALUES = {
    "highway": {"services", "rest_area", "escape", "elevator"},
    "waterway": {"riverbank", "dock", "boatyard", "dam"},
    "barrier": {"city_wall", "ditch", "hedge", "retaining_wall", "wall", "spikes"},
    "railway": {"station", "turntable", "roundhouse", "platform"},
    "power": {"plant", "substation", "generator", "transformer"},
}
LINE_VALUES = {
    "natural": {"coastline", "cliff", "ridge", "arete", "tree_row"},
    "man_made": {"cutline", "embankment", "pipeline"}, "aeroway": {"taxiway"},
}


def is_area(tags):
    if "area" in tags:
        return tags["area"] != "no"
    return (any(tags.get(key, "no") != "no" for key in AREA_TAGS)
            or any(tags.get(key) in values for key, values in AREA_VALUES.items())
            or any(key in tags and tags[key] != "no" and tags[key] not in values
                   for key, values in LINE_VALUES.items()))


def meters(value, fallback):
    match = re.fullmatch(r"([\d.]+)\s*(m|ft|')?", str(value).strip(), re.I)
    if match:
        try:
            number = float(match[1]) * (0.3048 if (match[2] or "").lower() in ("ft", "'") else 1)
            if math.isfinite(number):
                return max(0, min(1500, number))
        except ValueError:
            pass
    return fallback


def properties(tags, polygon):
    if not polygon:
        return {layer: True for key, layer in (("highway", "road"), ("waterway", "waterway")) if key in tags}
    building = tags.get("building", "no") != "no"
    land = "building" not in tags and ("landuse" in tags or tags.get("natural") in
                                       ("wood", "grassland", "scrub", "heath"))
    water = tags.get("natural") == "water" or "water" in tags or tags.get("landuse") == "reservoir"
    result = {layer: True for layer, visible in (("land", land), ("water", water)) if visible}
    if building:
        height = meters(tags.get("height"), meters(tags.get("building:levels"), 3) * 3)
        result.update(building=tags["building"], _height=height, _base=min(height, meters(tags.get("min_height"), 0)),
                      _estimated=not tags.get("height") and not tags.get("building:levels"))
        if "name" in tags:
            result["name"] = tags["name"]
    return result


def objects(path):
    """Stream complete OSM elements without retaining the XML tree."""
    try:
        with path.open("rb") as stream:
            parser = ET.iterparse(stream, events=("start", "end"))
            _, root = next(parser)
            if root.tag != "osm":
                raise ValueError("OSM file contains no osm root.")
            for event, obj in parser:
                if event == "end" and obj.tag in ("node", "way", "relation"):
                    yield obj
                    root.clear()
    except (ET.ParseError, StopIteration) as error:
        raise ValueError("OSM file contains invalid XML.") from error


def read_osm(path):
    nodes, ways, relations = {}, {}, []
    for element in objects(path):
        identity = int(element.attrib["id"])
        if element.tag == "node":
            nodes[identity] = [float(element.attrib["lon"]), float(element.attrib["lat"])]
        else:
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}
            area = is_area(tags)
            tags = {key: value for key, value in tags.items() if key in TAGS}
            if element.tag == "way":
                refs = [int(nd.attrib["ref"]) for nd in element.findall("nd")]
                ways[identity] = refs, tags, area
            elif tags.get("type") in ("multipolygon", "boundary", "route", "waterway"):
                members = [(int(m.attrib["ref"]), m.get("role", ""))
                           for m in element.findall("member") if m.get("type") == "way"]
                relations.append((tags, members))
    return nodes, ways, relations


def rings(identities, ways, nodes):
    pending = {identity: ways[identity][0] for identity in identities if identity in ways
               and len(ways[identity][0]) >= 2 and all(n in nodes for n in ways[identity][0])}
    ends = defaultdict(set)
    for identity, refs in pending.items():
        ends[refs[0]].add(identity)
        ends[refs[-1]].add(identity)
    while pending:
        identity, refs = pending.popitem()
        ring, used = list(refs), {identity}
        while ring[0] != ring[-1]:
            identity = next((i for i in ends[ring[-1]] if i in pending), None)
            if identity is None:
                break
            refs = pending.pop(identity)
            used.add(identity)
            ring.extend(refs[1:] if refs[0] == ring[-1] else refs[-2::-1])
        if len(ring) >= 4 and ring[0] == ring[-1]:
            yield [nodes[n] for n in ring], used


def display_osm(path):
    nodes, ways, relations = read_osm(path)
    features, consumed = [], set()
    for tags, members in relations:
        if tags["type"] in ("route", "waterway"):
            props = properties(tags, False)
            if not props:
                continue
            lines = [[nodes[n] for n in ways[ref][0]] for ref, _ in members if ref in ways
                     and len(ways[ref][0]) >= 2 and all(n in nodes for n in ways[ref][0])]
            if lines:
                features.append(feature("MultiLineString", lines, props))
            continue
        outers = [ref for ref, role in members if role in ("", "outer")]
        # Older multipolygons can carry their area tags on a single outer way.
        if not DISPLAY_TAGS.intersection(tags) and len(outers) == 1 and outers[0] in ways:
            tags = dict(ways[outers[0]][1], **tags)
        props = properties(tags, True)
        if not props:
            continue
        polygons, used = [], set()
        for ring, identities in rings(outers, ways, nodes):
            if signed_area(ring) < 0:
                ring.reverse()
            polygons.append([ring])
            used.update(identities)
        for ring, identities in rings([ref for ref, role in members if role == "inner"], ways, nodes):
            owners = [p for p in polygons if ring_contains(p[0], ring[0])]
            if owners:
                if signed_area(ring) > 0:
                    ring.reverse()
                min(owners, key=lambda p: abs(signed_area(p[0]))).append(ring)
                used.update(identities)
        if polygons:
            features.append(feature("MultiPolygon", polygons, props))
            # Retain independently tagged members, but do not fill holes twice.
            consumed.update(ref for ref in used if all(tags.get(k) == v
                            for k, v in ways[ref][1].items() if k != "type"))
    for identity, (refs, tags, area) in ways.items():
        if identity in consumed or len(refs) < 2 or any(n not in nodes for n in refs):
            continue
        polygon = area and len(refs) >= 4 and refs[0] == refs[-1]
        props = properties(tags, polygon)
        if props:
            coordinates = [nodes[n] for n in refs]
            if polygon and signed_area(coordinates) < 0:
                coordinates.reverse()
            features.append(feature("Polygon" if polygon else "LineString",
                                    [coordinates] if polygon else coordinates, props))
    return dict(type="FeatureCollection", features=features)
