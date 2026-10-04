"""Browser-ready views of saved captures: imagery and terrain tiles, and OSM as GeoJSON."""

import math
import struct
import sys
import threading
import zlib
from collections import OrderedDict, defaultdict
from io import BytesIO
from itertools import accumulate
from xml.etree import ElementTree as ET

from PIL import Image

from . import terrain
from .common import contained
from .geo import collection, feature, ring_contains, signed_area

CACHE_BYTES = 384 * 2**20
OVERVIEWS = 6  # Zoom levels built below a grid's own zoom.

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
TAGS = {"building", "landuse", "natural", "water", "leisure", "highway", "waterway", "name", "type"}
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
PATHS = {"footway", "path", "cycleway", "steps", "pedestrian", "track", "bridleway", "corridor"}


def saved(grid, count, z, x, y):
    """One past the last saved grid tile inside tile z/x/y, or 0; downloads are row-major."""
    factor = 2 ** (grid["zoom"] - z)
    left = max(x * factor - grid["x0"], 0)
    right = min((x + 1) * factor - grid["x0"], grid["columns"])
    top = max(y * factor - grid["y0"], 0)
    bottom = min((y + 1) * factor - grid["y0"], grid["rows"])
    if left >= right or top >= bottom:
        return 0
    columns = grid["columns"]
    row = min(bottom - 1, (count - 1) // columns)
    end = min(right, count - row * columns)
    if end <= left:
        row, end = row - 1, right
    return row * columns + end if row >= top else 0


def unpredict(data, width):
    """Undo the TIFF floating point predictor, giving big-endian Float32 rows."""
    rows = bytearray()
    for y in range(0, len(data), width * 4):
        # Each row holds byte differences across four planes, most significant first.
        planes = bytes(v & 255 for v in accumulate(data[y:y + width * 4]))
        row = bytearray(width * 4)
        for plane in range(4):
            row[plane::4] = planes[plane * width:(plane + 1) * width]
        rows += row
    return bytes(rows)


def undifference(data, width, order):
    """Undo the TIFF horizontal predictor for 16-bit samples."""
    fmt = f"{order}{width}H"
    return b"".join(struct.pack(fmt, *(v & 0xFFFF for v in accumulate(struct.unpack(fmt, data[y:y + width * 2]))))
                    for y in range(0, len(data), width * 2))


def heights(path):
    """Decode one saved 512-pixel Int16 or Float32 terrain tile without libtiff."""
    with path.open("rb") as stream:
        source = terrain.TIFF(stream)
        size = source.value(322)
        per_row = 512 // size
        image = Image.new("F", (512, 512))
        for i, (offset, length) in enumerate(zip(source.values(324), source.values(325))):
            data = source.read(offset, length)
            if source.value(259) != 1:
                data = zlib.decompress(data)
            big = source.order == ">"
            if source.value(317, 1) == 3:
                data, big = unpredict(data, size), True
            elif source.value(317, 1) == 2:
                data = undifference(data, size, source.order)
            mode, raw_mode = ("F", "F;32BF" if big else "F;32F") if source.float else ("I", "I;16BS" if big else "I;16S")
            with (
                Image.frombytes(mode, (size, size), data, "raw", raw_mode) as block,
                block.convert("F") as values,
            ):
                image.paste(values, (i % per_row * size, i // per_row * size))
        nodata = float(b"".join(source.values(42113)).rstrip(b"\0"))
    # Terrarium has no NoData value. Unknown heights use sea level.
    image.putdata([v if math.isfinite(v) and v != nodata else 0
                   for (v,) in struct.iter_unpack("=f", image.tobytes())])
    return image


def terrarium(image):
    """Encode heights as RGB, where meters = R * 256 + G + B / 256 - 32768."""
    with image.point(lambda h: h * 256 + 32768 * 256) as scaled, scaled.convert("I") as values:
        # Native 32-bit integers hold the three channels below an unused high byte.
        raw = "BGRX" if sys.byteorder == "little" else "XRGB"
        return Image.frombytes("RGB", image.size, values.tobytes(), "raw", raw)


def png(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()


class Tiles:
    """Satellite and terrain tiles; overviews merge four children and are cached by saved progress."""

    def __init__(self):
        self.cache = OrderedDict()
        self.used = 0
        self.lock = threading.Lock()

    def get(self, folder, stage, z, x, y):
        """Return (bytes, content type), or None where nothing is saved yet."""
        grid = stage["grid"]
        if not grid["zoom"] - OVERVIEWS <= z <= grid["zoom"] or not (0 <= x < 2**z and 0 <= y < 2**z):
            return None
        if stage["mode"] == "satellite" and z == grid["zoom"]:
            # Saved patches are already browser images.
            version = saved(grid, len(stage["results"]), z, x, y)
            if not version:
                return None
            path = contained(folder, stage["results"][version - 1]["filename"])
            return path.read_bytes(), "image/png" if path.suffix == ".png" else "image/jpeg"
        image = self.image(folder, stage, z, x, y)
        if image is None:
            return None
        if stage["mode"] == "terrain":
            with terrarium(image) as rgb:
                return png(rgb), "image/png"
        return png(image), "image/png"

    def image(self, folder, stage, z, x, y):
        grid, results = stage["grid"], stage["results"]
        version = saved(grid, len(results), z, x, y)
        if not version:
            return None
        key = folder, stage["mode"], z, x, y
        with self.lock:
            entry = self.cache.get(key)
            if entry and entry[0] == version:
                self.cache.move_to_end(key)
                return entry[1]
        elevation = stage["mode"] == "terrain"
        if z == grid["zoom"]:
            path = contained(folder, results[version - 1]["filename"])
            if elevation:
                image = heights(path)
            else:
                with Image.open(path) as patch:
                    image = patch.convert("RGBA")
        else:
            size = 512 if elevation else 256
            half = size // 2
            image = Image.new("F" if elevation else "RGBA", (size, size))
            missing, means = [], []
            for dy in range(2):
                for dx in range(2):
                    child = self.image(folder, stage, z + 1, 2 * x + dx, 2 * y + dy)
                    if child is None:
                        missing.append((dx * half, dy * half))
                        continue
                    with child.resize((half, half), Image.Resampling.BOX) as smaller:
                        image.paste(smaller, (dx * half, dy * half))
                        if elevation:
                            with smaller.resize((1, 1), Image.Resampling.BOX) as mean:
                                means.append(mean.getpixel((0, 0)))
            # Sea level outside the grid would draw cliffs at its edges; use the tile's mean height.
            for left, top in missing if means else ():
                image.paste(sum(means) / len(means), (left, top, left + half, top + half))
        with self.lock:
            previous = self.cache.pop(key, None)
            if previous:
                self.used -= previous[1].width * previous[1].height * 4
            self.cache[key] = version, image
            self.used += image.width * image.height * 4
            while self.used > CACHE_BYTES:
                _, (_, dropped) = self.cache.popitem(last=False)
                self.used -= dropped.width * dropped.height * 4
        return image


def is_area(tags):
    if "area" in tags:
        return tags["area"] != "no"
    return (any(tags.get(key, "no") != "no" for key in AREA_TAGS)
            or any(tags.get(key) in values for key, values in AREA_VALUES.items())
            or any(key in tags and tags[key] != "no" and tags[key] not in values
                   for key, values in LINE_VALUES.items()))


def properties(tags, polygon):
    """Map styling class of an OSM object, or None when it is not drawn."""
    if not polygon:
        highway = tags.get("highway")
        if highway:
            road = "major" if highway.removesuffix("_link") in MAJOR_ROADS else "path" if highway in PATHS else "minor"
            return dict(kind="road", road=road, name=tags.get("name"))
        return dict(kind="waterway") if "waterway" in tags else None
    if tags.get("building", "no") != "no":
        return dict(kind="building", name=tags.get("name"))
    if tags.get("natural") == "water" or "water" in tags or tags.get("landuse") == "reservoir":
        return dict(kind="water")
    if "landuse" in tags or "leisure" in tags or tags.get("natural") in ("wood", "grassland", "scrub", "heath"):
        return dict(kind="land")
    return None


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


def osm(path):
    """Buildings, roads, water, and land from a saved map.osm."""
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
        if len(outers) == 1 and outers[0] in ways and not properties(tags, True):
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
    return collection(features)
