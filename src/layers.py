"""Browser-ready views of saved captures: imagery and terrain tiles, and OSM as GeoJSON."""

import math
import threading
import zlib
from collections import OrderedDict, defaultdict
from io import BytesIO
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from . import terrain
from .common import contained
from .geo import MERCATOR_RADIUS, collection, coordinate, feature, ring_contains, signed_area

CACHE_BYTES = 384 * 2**20
OVERVIEWS = dict(satellite=6, terrain=8)  # Most zoom levels built below a grid's own zoom.
# Missing heights. Overviews average it in, which keeps those pixels far below any real height.
NODATA = -1e6
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # West, east, north, south.
# Opacity per unit of brightness change from flat ground, and its maximum. Shadows span a much wider range.
SHADOW = 160, 120
HIGHLIGHT = 400, 120
RELIEF = 4  # Overview levels per doubling of slope exaggeration.

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


def lowest(stage):
    """Lowest zoom with tiles: a few levels past where the grid fits in one tile, within the overview limit."""
    grid = stage["grid"]
    levels = math.ceil(math.log2(max(grid["columns"], grid["rows"]))) + 3
    return max(0, grid["zoom"] - min(levels, OVERVIEWS[stage["mode"]]))


def heights(path):
    """Decode one saved 512-pixel Int16 or Float32 terrain tile without libtiff."""
    with path.open("rb") as stream:
        source = terrain.TIFF(stream)
        size, order, predictor = source.value(322), source.order, source.value(317, 1)
        per_row = 512 // size
        values = np.empty((512, 512), np.float32)
        for i, (offset, length) in enumerate(zip(source.values(324), source.values(325))):
            data = source.read(offset, length)
            if source.value(259) != 1:
                data = zlib.decompress(data)
            if source.float and predictor == 3:
                # Rows hold four byte planes, most significant first, with differences running on across planes.
                rows = np.cumsum(np.frombuffer(data, np.uint8).reshape(size, 4 * size), axis=1, dtype=np.uint8)
                block = rows.reshape(size, 4, size).transpose(0, 2, 1).copy().view(">f4")
            elif source.float:
                block = np.frombuffer(data, order + "f4")
            else:
                block = np.frombuffer(data, order + "u2").reshape(size, size)
                if predictor == 2:
                    block = np.cumsum(block, axis=1, dtype=np.uint16)
                block = block.astype(np.int16)
            y, x = divmod(i, per_row)
            values[y * size:(y + 1) * size, x * size:(x + 1) * size] = block.reshape(size, size)
        nodata = float(b"".join(source.values(42113)).rstrip(b"\0"))
    values[~np.isfinite(values) | (values == nodata)] = NODATA
    return Image.fromarray(values, "F")


def hillshade(center, neighbors, meters_per_pixel, exaggeration):
    """Shadows and highlights lit from the northwest, transparent where heights are missing."""
    center = np.asarray(center, np.float32)
    # Without a neighbor, continue the slope at the edge so adjacent tiles still meet smoothly.
    padded = np.pad(center, 1, mode="reflect", reflect_type="odd")
    # Neighbors' facing edges, where known, keep slopes continuous across tiles.
    for (dx, dy), neighbor in zip(NEIGHBORS, neighbors):
        if neighbor is None:
            continue
        neighbor = np.asarray(neighbor, np.float32)
        target, edge = {(-1, 0): (np.s_[1:-1, 0], np.s_[:, -1]), (1, 0): (np.s_[1:-1, -1], np.s_[:, 0]),
                        (0, -1): (np.s_[0, 1:-1], np.s_[-1, :]), (0, 1): (np.s_[-1, 1:-1], np.s_[0, :])}[dx, dy]
        padded[target] = np.where(neighbor[edge] > NODATA / 2, neighbor[edge], padded[target])
    scale = exaggeration / (2 * meters_per_pixel)
    east = (padded[1:-1, 2:] - padded[1:-1, :-2]) * scale
    south = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * scale
    # Light comes from x west, y north, 45° up: (-0.5, -0.5, 0.7071) with x east and y south.
    shade = (east * 0.5 + south * 0.5 + 0.7071) / np.sqrt(east * east + south * south + 1) - 0.7071
    (dark, darkest), (light, lightest) = SHADOW, HIGHLIGHT
    alpha = np.minimum(np.maximum(-shade, 0) * dark, darkest) + np.minimum(np.maximum(shade, 0) * light, lightest)
    # Slopes next to a missing height are unreliable, so the known area shrinks by a pixel inside the tile.
    known = np.pad(center > NODATA / 2, 1, mode="edge")
    inner = np.logical_and.reduce([known[y:y + center.shape[0], x:x + center.shape[1]] for y in range(3) for x in range(3)])
    tone = np.where(shade > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack((tone, tone, tone, np.where(inner, alpha, 0).astype(np.uint8))), "RGBA")


def terrarium(image, base):
    """Heights above base as Terrarium RGB for MapLibre terrain, which has no transparency: missing heights sit at base."""
    values = np.asarray(image, np.float64)
    values = np.clip(np.where(values > NODATA / 2, values - base, 0) + 32768, 0, 65535)
    return Image.fromarray(np.dstack((values // 256, values % 256, values % 1 * 256)).astype(np.uint8), "RGB")


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
        # One build per layer at a time: parallel builds of overlapping overviews would repeat the same decoding.
        self.builds = defaultdict(threading.Lock)
        self.bases = {}  # Folder → (saved count, lowest height).

    def get(self, folder, stage, z, x, y, dem=False):
        """Return (bytes, content type), or None where nothing is saved yet; dem asks for terrain heights, not shading."""
        grid = stage["grid"]
        if not lowest(stage) <= z <= grid["zoom"] or not (0 <= x < 2**z and 0 <= y < 2**z):
            return None
        if stage["mode"] == "satellite" and z == grid["zoom"]:
            # Saved patches are already browser images.
            version = saved(grid, len(stage["results"]), z, x, y)
            if not version:
                return None
            path = contained(folder, stage["results"][version - 1]["filename"])
            return path.read_bytes(), "image/png" if path.suffix == ".png" else "image/jpeg"
        with self.lock:
            build = self.builds[folder, stage["mode"]]
        with build:
            image = self.image(folder, stage, z, x, y)
            base = self.base(folder, stage) if dem and image is not None else None
        if image is None:
            return None
        if dem:
            with terrarium(image, base) as encoded:
                return png(encoded), "image/png"
        if stage["mode"] == "terrain":
            # Building neighbors would cost as much as the tile itself; edges are extrapolated without them.
            neighbors = [self.cached(folder, stage, z, x + dx, y + dy) for dx, dy in NEIGHBORS]
            latitude = math.radians(coordinate((x + 0.5) * 256, (y + 0.5) * 256, z)[0])
            meters = 2 * math.pi * MERCATOR_RADIUS * math.cos(latitude) / (512 * 2**z)
            # Zoomed out, averaged heights flatten slopes; exaggerate them to keep relief visible.
            with hillshade(image, neighbors, meters, 2 ** ((grid["zoom"] - z) / RELIEF)) as shaded:
                return png(shaded), "image/png"
        return png(image), "image/png"

    def base(self, folder, stage):
        """Lowest saved height, so terrain rises from the flat map around it; callers hold the build lock."""
        version = len(stage["results"])
        if self.bases.get(folder, (None,))[0] != version:
            lowest = np.inf
            for result in stage["results"]:
                with heights(contained(folder, result["filename"])) as image:
                    values = np.asarray(image)
                    lowest = min(lowest, values.min(initial=np.inf, where=values > NODATA / 2))
            self.bases[folder] = version, 0.0 if lowest == np.inf else float(lowest)
        return self.bases[folder][1]

    def cached(self, folder, stage, z, x, y):
        version = saved(stage["grid"], len(stage["results"]), z, x, y)
        with self.lock:
            entry = self.cache.get((folder, stage["mode"], z, x, y))
            if entry and version and entry[0] == version:
                self.cache.move_to_end((folder, stage["mode"], z, x, y))
                return entry[1]
        return None

    def image(self, folder, stage, z, x, y):
        grid, results = stage["grid"], stage["results"]
        version = saved(grid, len(results), z, x, y)
        if not version:
            return None
        key = folder, stage["mode"], z, x, y
        if (image := self.cached(folder, stage, z, x, y)) is not None:
            return image
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
            image = Image.new("F", (size, size), NODATA) if elevation else Image.new("RGBA", (size, size))
            for dy in range(2):
                for dx in range(2):
                    child = self.image(folder, stage, z + 1, 2 * x + dx, 2 * y + dy)
                    if child is not None:
                        with child.resize((half, half), Image.Resampling.BOX) as smaller:
                            image.paste(smaller, (dx * half, dy * half))
        with self.lock:
            previous = self.cache.pop(key, None)
            if previous:
                self.used -= previous[1].width * previous[1].height * 4
            self.cache[key] = version, image
            self.used += image.width * image.height * 4
            while self.used > CACHE_BYTES:
                # Finer tiles are cheapest to rebuild, so the least recently used of them go first.
                victim = max(self.cache, key=lambda cached: cached[2])
                _, dropped = self.cache.pop(victim)
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
