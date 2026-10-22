"""S3 road reading: what OSM says about the ground's surfaces, road widths, crossings, paint, and wear.

Everything here works in local metres on Features; ground.py turns the results into fields and meshes. Paint and
wear come back as flat quads per material, corners (a, b, c, d) with a–b across and b–c along, for ground.py to drape.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np

LANE = 3.25  # Metres per lane when only lanes are tagged.
PARKING_LANE = 2.0  # Metres added per side with a tagged parking lane.
BIKE = 1.6  # Metres of a painted cycle lane without a width tag.
CURB = 0.13  # Metres a curbed roadway sits below its sidewalks.

# Carriageway width in metres per highway type, when neither width nor lanes are tagged.
ROADWAYS = dict(motorway=11.0, trunk=10.0, primary=9.0, secondary=8.0, tertiary=7.0, unclassified=6.0,
                residential=6.0, living_street=5.0, service=4.0, busway=3.5, road=6.0, track=3.0, raceway=8.0)
PATHS = dict(pedestrian=6.0, footway=2.0, cycleway=2.0, path=1.5, bridleway=2.0, steps=2.0, platform=2.5)
# Roadways that get curbs and paint in town; service roads, tracks, and living streets stay flush.
CURBED = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "busway", "road"}
PAINTED = CURBED - {"unclassified", "residential", "road"}
# area:highway values that are carriageway; everything else in area:highway is walked on.
AREA_ROADWAYS = set(ROADWAYS) | {"emergency", "crossing", "bus_stop", "parking_lane"}
# Parking areas beside the carriageway, which sit at road level rather than on the sidewalk.
STREET_PARKING = {"street_side", "lane", "on_street", "layby"}

# OSM surface values to library ground materials.
SURFACES = {
    "asphalt": "asphalt", "chipseal": "asphalt",
    "paving_stones": "paving", "paving_stones:lanes": "paving", "tactile_paving": "paving", "bricks": "paving",
    "granite": "paving-granite", "stone": "paving", "clay": "paving", "wood": "paving",
    "sett": "cobblestone", "cobblestone": "cobblestone", "unhewn_cobblestone": "cobblestone",
    "cobblestone:flattened": "cobblestone",
    "concrete": "concrete", "concrete:plates": "concrete", "concrete:lanes": "concrete", "metal": "concrete",
    # Water-bound sandy paths, as in the Tiergarten, read as packed earth rather than loose grey gravel.
    "gravel": "gravel", "fine_gravel": "dirt", "compacted": "dirt", "pebblestone": "gravel", "rock": "gravel",
    "shells": "gravel",
    "dirt": "dirt", "earth": "dirt", "ground": "dirt", "mud": "dirt", "sand": "dirt", "woodchips": "dirt",
    "unpaved": "dirt",
    "grass": "grass", "grass_paver": "grass", "artificial_turf": "grass",
}
# Land cover areas and their default materials; specific covers come before grass so they win where they overlap.
LANDUSE = dict(construction="dirt", brownfield="dirt", landfill="dirt", quarry="gravel", railway="gravel",
               grass="grass", meadow="grass", village_green="grass", recreation_ground="grass", cemetery="grass",
               allotments="grass", orchard="grass", vineyard="grass", farmland="grass", greenfield="grass",
               flowerbed="dirt", plant_nursery="dirt")
LEISURE = dict(park="grass", garden="grass", pitch="grass", playground="dirt", dog_park="dirt", golf_course="grass",
               common="grass", nature_reserve="grass", track="gravel")
NATURAL = dict(wood="grass", scrub="dirt", heath="grass", grassland="grass", sand="dirt", beach="dirt",
               bare_rock="gravel", scree="gravel", shingle="gravel", mud="dirt", wetland="grass")
# Kept out of the ground: things above or below it, and things not built.
GONE = {"construction", "proposed", "abandoned", "razed", "demolished", "disused"}
# Unnamed surface patches smaller than this (m²), such as plates around a monument's columns, read as blotches;
# the paving around them is drawn instead. Soil patches stay: they are tree pits and beds.
SPECK = 8.0
SOIL = {"ground/dirt", "ground/grass", "ground/gravel"}


@dataclass
class Source:
    """One OSM feature's contribution to a ground layer."""

    group: str  # "core", "water", "island", "road", or "walk"; see ground.GROUPS.
    rank: int  # Priority inside the group, lower wins.
    material: str
    rings: list = None  # Polygon rings, or
    line: np.ndarray = None  # a centreline with
    half: float = 0.0  # half its width.
    curbed: bool = False  # Roadway with raised sidewalks beside it in town.
    tags: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class Crossing:
    """A crossing: the walked or ridden line across the carriageway and its paint ("zebra", "dashes", or None)."""

    line: np.ndarray
    paint: str = None
    width: float = 4.0
    cycle: bool = False  # A cycle crossing, painted as a Radfahrerfurt: wide dashes along both edges.


def number(text):
    """First number in a tag value like "7", "7.5 m", or "2;3"; None when there is none."""
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", text or "")
    return float(match.group(1)) if match else None


def oneway(tags):
    return (tags.get("oneway") in ("yes", "1", "-1", "true") or tags.get("junction") in ("roundabout", "circular")
            or (tags.get("highway") in ("motorway", "motorway_link") and tags.get("oneway") != "no"))


def highway(tags):
    """The highway type with _link removed, or None."""
    value = tags.get("highway")
    return value.removesuffix("_link") if value else None


def lanes(tags):
    count = number(tags.get("lanes"))
    return int(count) if count and count >= 1 else None


PARKED = ("lane", "parallel", "diagonal", "perpendicular", "yes", "half_on_kerb", "marked")


def parking_side(tags, side):
    """How cars park on one side ("left" or "right" of the way's direction) inside the carriageway: their
    orientation ("parallel", "diagonal", or "perpendicular"), or None where they don't."""
    for scope in (side, "both"):
        for key in (f"parking:{scope}", f"parking:lane:{scope}"):
            value = tags.get(key)
            if value is None:
                continue
            restriction = tags.get(f"parking:{scope}:restriction") or tags.get(f"parking:condition:{scope}")
            if value not in PARKED or restriction in ("no_stopping", "no_parking", "no"):
                return None
            orientation = tags.get(f"parking:{scope}:orientation", value)
            return orientation if orientation in ("diagonal", "perpendicular") else "parallel"
    return None


def parking(tags):
    """Metres of parking lanes beside the carriageway."""
    return sum(PARKING_LANE for side in ("left", "right") if parking_side(tags, side))


def cycle_lanes(tags):
    """Painted cycle lanes inside the carriageway: [(side, width, kind, colour)], side 1 right of the way's direction
    and -1 left; kind "exclusive" (solid line) or "advisory" (dashed); colour the surface:colour tag or None."""
    result = []
    for side, sign in (("right", 1), ("left", -1)):
        prefix = next((f"cycleway:{scope}" for scope in (side, "both") if f"cycleway:{scope}" in tags), None)
        if prefix is None and "cycleway" in tags and not (oneway(tags) and sign == (1 if tags.get("oneway") == "-1" else -1)):
            prefix = "cycleway"
        if prefix is None or tags.get(prefix) != "lane":
            continue
        size = number(tags.get(f"{prefix}:width")) or BIKE
        kind = "advisory" if tags.get(f"{prefix}:lane") == "advisory" else "exclusive"
        colour = tags.get(f"{prefix}:surface:colour") or tags.get("cycleway:surface:colour")
        result.append((sign, float(np.clip(size, 1.0, 2.5)), kind, colour))
    return result


def width(tags):
    """Paved width in metres of a highway line: width tag, else lanes, else a default, plus parking lanes."""
    kind = highway(tags)
    tagged = number(tags.get("width") or tags.get("est_width"))
    if tagged and 1 <= tagged <= 60:
        return tagged
    if kind in ROADWAYS:
        count = lanes(tags)
        base = count * LANE if count else ROADWAYS[kind] * (0.75 if oneway(tags) and kind in CURBED else 1)
        return base + parking(tags) + sum(lane[1] for lane in cycle_lanes(tags))
    if kind == "footway" and tags.get("footway") == "sidewalk":
        return 2.5
    return PATHS.get(kind, 2.0)


def material(tags, default, plaza=False):
    """Library material for a feature's surface tag, else the default; road materials keep a worn variant, and
    paved plazas get large granite slabs, as on Berlin's squares."""
    name = SURFACES.get(tags.get("surface", "").split(";")[0], None)
    if tags.get("surface") == "paved" or name is None:
        name = default
    if plaza and name in ("paving", "concrete"):
        name = "paving-granite"
    if name == "asphalt" and (tags.get("smoothness") in ("intermediate", "bad", "very_bad", "horrible")
                              or highway(tags) in ("service", "track")):
        name = "asphalt-old"
    return "ground/" + name


def present(tags):
    """Whether a feature lies on the ground today: not a bridge, tunnel, roof, or plan."""
    return not (tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") not in ("no", "building_passage")
                or tags.get("indoor", "no") != "no" or tags.get("location") in ("underground", "roof", "rooftop")
                or (number(tags.get("layer")) or 0) < 0 and tags.get("covered") != "yes"
                or any(key in tags for key in GONE) or tags.get("highway") in GONE)


def rings(feature):
    return [feature.points] + list(feature.holes)


def solid(feature, parted):
    """Whether a building stands on its whole footprint, so no ground shows inside it."""
    tags = feature.tags
    if (tags.get("building") in ("roof", "canopy", "carport", "bridge", "gate", "construction")
            or tags.get("building:part") in ("roof", "balcony", "bridge")
            or (number(tags.get("min_height")) or 0) > 0 or (number(tags.get("building:min_level")) or 0) > 0
            or (number(tags.get("layer")) or 0) > 0 or tags.get("location") == "underground"):
        return False
    return feature.id not in parted


def sources(osm):
    """Ground layer sources from OSM features: solid buildings as cores, then roads, paths, water, and land cover."""
    buildings = [f for f in osm if f.kind == "building"]
    parted = envelopes(buildings)
    result = []
    for f in osm:
        tags = f.tags
        if f.kind == "building":
            if solid(f, parted):
                result.append(Source("core", 0, "", rings=rings(f), tags=tags, id=f.id))
            continue
        if f.kind not in ("area", "line") or not present(tags):
            continue
        source = area(f) if f.kind == "area" else line(f)
        if source:
            result.append(source)
    return squares(result)


def squares(sources):
    """Sources with plain paving inside a square or pedestrian area, such as the footways and the service loop
    mapped across Pariser Platz, paved in the square's granite slabs too."""
    rings_ = [s.rings[0] for s in sources if s.rings is not None and plaza(s.tags) and s.material == "ground/paving-granite"]
    if not rings_:
        return sources
    for s in sources:
        if s.material not in ("ground/paving", "ground/concrete") or s.group not in ("walk", "road"):
            continue
        points, _ = resample(s.rings[0] if s.rings is not None else s.line, 2.0)
        # Mostly inside one square: a sidewalk that only touches one keeps its own paving.
        if any(contains(ring, points).mean() > 0.8 for ring in rings_):
            s.material = "ground/paving-granite"
    return sources


def envelopes(buildings):
    """Ids of building outlines mostly covered by raised parts, such as a gate over its passages.

    Their parts that reach the ground stand in for them; outlines with only a few raised parts on top stay solid.
    """
    raised = [f for f in buildings if "building:part" in f.tags and (number(f.tags.get("min_height")) or 0) > 0]
    if not raised:
        return set()
    centres = np.array([part.points.mean(axis=0) for part in raised])
    sizes = np.array([abs(signed_area(part.points)) for part in raised])
    return {f.id for f in buildings if "building:part" not in f.tags
            and sizes[contains(f.points, centres)].sum() >= 0.5 * abs(signed_area(f.points))}


def signed_area(ring):
    x, z = ring[:, 0], ring[:, 1]
    return float(np.dot(x, np.roll(z, -1)) - np.dot(np.roll(x, -1), z)) / 2


def contains(ring, points):
    """Even-odd test of (n, 2) points against a ring."""
    x, z = points[:, 0], points[:, 1]
    result = np.zeros(len(points), bool)
    for (ax, az), (bx, bz) in zip(ring, np.roll(ring, -1, axis=0)):
        if az != bz:
            result ^= ((az > z) != (bz > z)) & (x < ax + (z - az) * (bx - ax) / (bz - az))
    return result


def plaza(tags):
    """Whether a feature is a pedestrian square or street, paved in large slabs rather than sidewalk paving."""
    return (tags.get("area:highway") == "pedestrian" or tags.get("highway") == "pedestrian"
            or tags.get("place") == "square")


def area(f):
    tags = f.tags
    shape = dict(rings=rings(f), tags=tags, id=f.id)
    value = tags.get("area:highway")
    if (tags.get("natural") == "water" or "water" in tags or tags.get("waterway") == "riverbank"
            or tags.get("landuse") in ("reservoir", "basin") or tags.get("amenity") == "fountain"):
        return Source("water", 0, "water/surface", **shape)
    if value == "traffic_island":
        return Source("island", 0, material(tags, "paving"), **shape)
    if value in AREA_ROADWAYS:
        kind = "primary" if value in ("crossing", "emergency", "bus_stop", "parking_lane") else value
        return Source("road", 0, material(tags, "asphalt"), curbed=kind in CURBED, **shape)
    if value:
        default = "asphalt" if value == "cycleway" else "dirt" if value == "path" else "paving"
        return Source("walk", 0, material(tags, default, plaza(tags)), **shape)
    kind = highway(tags)
    if kind in ROADWAYS:
        return Source("road", 0, material(tags, "gravel" if kind == "track" else "asphalt"), curbed=kind in CURBED,
                      **shape)
    if kind in PATHS or tags.get("place") == "square":
        return Source("walk", 0 if kind else 2, material(tags, "paving", plaza(tags)), **shape)
    if tags.get("amenity") == "parking" and tags.get("parking") in STREET_PARKING:
        # Parking bays along the kerb lie at road level, with the curb between them and the sidewalk.
        return Source("road", 0, material(tags, "asphalt"), curbed=True, **shape)
    if tags.get("amenity") in ("parking", "bicycle_parking", "motorcycle_parking") and tags.get("parking") not in (
            "underground", "multi-storey", "rooftop"):
        return Source("walk", 0, material(tags, "asphalt"), **shape)
    for key, table in (("landuse", LANDUSE), ("leisure", LEISURE), ("natural", NATURAL)):
        if tags.get(key) in table:
            cover = material(tags, table[tags[key]])
            # Bare covers outrank grass where land uses overlap.
            return Source("walk", 2 if cover != "ground/grass" else 3, cover, **shape)
    if tags.get("surface") in SURFACES:
        cover = material(tags, "paving")
        if cover not in SOIL and abs(signed_area(f.points)) < SPECK:
            return None
        return Source("walk", 2, cover, **shape)
    return None


def line(f):
    tags = f.tags
    kind = highway(tags)
    if tags.get("area") == "yes" or tags.get("highway") in ("corridor", "elevator", "proposed", "construction"):
        return None
    if kind in ROADWAYS:
        return Source("road", 1, material(tags, "gravel" if kind == "track" else "asphalt"), line=f.points,
                      half=width(tags) / 2, curbed=kind in CURBED, tags=tags, id=f.id)
    if kind in PATHS:
        default = {"path": "dirt", "bridleway": "dirt", "cycleway": "asphalt"}.get(kind, "paving")
        return Source("walk", 1, material(tags, default, plaza(tags)), line=f.points, half=width(tags) / 2, tags=tags,
                      id=f.id)
    if tags.get("waterway") in ("river", "canal", "stream", "ditch", "drain") and tags.get("tunnel", "no") == "no":
        half = (number(tags.get("width")) or {"river": 20, "canal": 12, "stream": 2}.get(tags["waterway"], 1)) / 2
        return Source("water", 1, "water/surface", line=f.points, half=half, tags=tags, id=f.id)
    return None


# Crossings and paint.

def paint(tags):
    """Crossing paint from crossing tags: "zebra", "dashes", or None."""
    markings = tags.get("crossing:markings")
    if markings == "no" or tags.get("crossing") in ("unmarked", "no", "informal"):
        return None
    if markings in ("zebra", "yes", "ladder", "ladder:skewed", "zebra:double", "zebra:paired", "zebra:bicolour"):
        return "zebra"
    if markings in ("dashes", "dots", "lines", "ladder:paired", "surface"):
        return "dashes"
    if tags.get("crossing") in ("zebra", "marked", "uncontrolled") or tags.get("crossing_ref") == "zebra":
        return "zebra"
    if tags.get("crossing") == "traffic_signals":
        return "dashes"
    return None


def crossings(osm, roads):
    """Crossings from crossing ways, and from crossing nodes on roads that no crossing way passes."""
    result = []
    for f in osm:
        if f.kind == "line" and (f.tags.get("footway") == "crossing" or f.tags.get("cycleway") == "crossing"
                                 or f.tags.get("path") == "crossing"):
            if f.tags.get("cycleway") == "crossing" or highway(f.tags) == "cycleway":
                # Cycle crossings are as wide as the cycle way, between two dashed edge lines.
                result.append(Crossing(f.points, paint(f.tags), max(1.6, min(width(f.tags), 3.0) + 0.3), cycle=True))
                continue
            result.append(Crossing(f.points, paint(f.tags), 4.0 if paint(f.tags) == "zebra" else 3.0))
    ways = [c.line for c in result]
    for f in osm:
        if f.kind != "node" or f.tags.get("highway") != "crossing" and f.tags.get("railway") != "crossing":
            continue
        point = f.points[0]
        if any(distance(point, way).min() < 3.0 for way in ways):
            continue
        road = nearest(point, roads)
        if road is None:
            continue
        direction, half = road
        across = np.array([-direction[1], direction[0]])
        result.append(Crossing(np.array([point - across * (half + 1.5), point + across * (half + 1.5)]),
                               paint(f.tags), 4.0 if paint(f.tags) == "zebra" else 3.0))
    return result


def distance(point, polyline):
    """Distances from a point to each segment of a polyline."""
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * d - point, axis=1)


def nearest(point, roads):
    """Unit direction and half width of the road line passing within a metre of a point, or None."""
    best = None
    for road in roads:
        if road.line is None or len(road.line) < 2:
            continue
        d = distance(point, road.line)
        i = int(np.argmin(d))
        if d[i] < 1.0 and (best is None or d[i] < best[0]):
            step = road.line[i + 1] - road.line[i]
            best = d[i], step / max(np.linalg.norm(step), 1e-9), road.half
    return best and best[1:]


def resample(points, step):
    """Points along a polyline every step metres or less, keeping its corners, and their distances along it."""
    out, along, total = [points[0]], [0.0], 0.0
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        count = max(1, math.ceil(length / step))
        t = np.arange(1, count + 1) / count
        out.extend(a + t[:, None] * (b - a))
        along.extend(total + t * length)
        total += length
    return np.array(out), np.array(along)


def offset(points, distance):
    """A polyline shifted sideways, to the right of its direction for positive distances (z points south)."""
    tangent = np.gradient(points, axis=0) if len(points) > 2 else np.repeat(points[1:] - points[:1], len(points), 0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    return points + distance * np.column_stack((-tangent[:, 1], tangent[:, 0]))


def dashes(along, dash, gap):
    """Mask of resampled points inside dashes that start at the line's start."""
    if gap <= 0:
        return np.ones(len(along), bool)
    return (along % (dash + gap)) < dash


def strip(points, along, keep, half):
    """Quads of a painted strip along points, one per consecutive pair that keep marks."""
    right = offset(points, 1.0) - points
    left_side, right_side = points - right * half, points + right * half
    pairs = np.flatnonzero(keep[:-1] & keep[1:])
    quads = np.stack([left_side[pairs], right_side[pairs], right_side[pairs + 1], left_side[pairs + 1]], axis=1)
    u = np.stack([along[pairs], along[pairs], along[pairs + 1], along[pairs + 1]], axis=1)
    v = np.broadcast_to(np.array([-half, half, half, -half]), u.shape)
    return quads, np.stack([u, v], axis=2)


def lane_lines(road):
    """Lateral offsets and (dash, gap) patterns of a painted road's lane lines."""
    tags = road.tags
    total = width(tags) - parking(tags) - sum(lane[1] for lane in cycle_lanes(tags))
    count = lanes(tags) or max(1, round(total / LANE))
    if count < 2 or total < 5:
        return []
    step = total / count
    # Cycle lanes and parking narrow the general lanes from their own side.
    shift = sum(lane[1] * lane[0] for lane in cycle_lanes(tags)) / -2
    shift += (bool(parking_side(tags, "left")) - bool(parking_side(tags, "right"))) * PARKING_LANE / 2
    lines = [(shift - total / 2 + i * step, (3.0, 6.0)) for i in range(1, count)]
    if oneway(tags):
        return lines
    backward = int(number(tags.get("lanes:backward")) or count // 2)
    centre = shift - total / 2 + backward * step
    solid = (0.0, 0.0) if count >= 4 else (3.0, 6.0)
    return [(o, solid if abs(o - centre) < 1e-6 else pattern) for o, pattern in lines]


ASPHALT = ("ground/asphalt", "ground/asphalt-old")


def markings(roads, crossings_, field, junction):
    """Painted quads per material: lane lines, cycle lanes, stop lines, and crossing paint.

    field(points) is the signed distance into the carriageway (metres, positive inside); junction(points) says which
    points lie where roads meet or cross walks, where lane lines stop. Returns {material: (quads (n, 4, 2),
    uvs (n, 4, 2))}, with uvs in metres along and across each mark.
    """
    inside = lambda points: field(points) > 0.3
    paint = {"ground/marking-white": [], "ground/bike-lane-red": []}
    for road in roads:
        tags = road.tags
        if road.line is None or road.material not in ASPHALT:
            continue
        points, along = resample(road.line, 1.0)
        if highway(tags) in PAINTED:
            for lateral, (dash, gap) in lane_lines(road):
                shifted = offset(points, lateral)
                keep = dashes(along, dash, gap) & inside(shifted) & ~junction(shifted)
                paint["ground/marking-white"].append(strip(shifted, along, keep, 0.06 if gap else 0.075))
        for side, size, kind, colour in cycle_lanes(tags):
            for material, quads in cycle_lane(road, points, along, side, size, kind, colour, field, junction).items():
                paint[material].append(quads)
    for crossing in crossings_:
        if crossing.paint:
            paint["ground/marking-white"].append(crossing_paint(crossing, inside))
    paint["ground/marking-white"] += stop_lines(roads, crossings_, inside)
    result = {}
    for material, parts in paint.items():
        parts = [part for part in parts if len(part[0])]
        if parts:
            result[material] = tuple(np.concatenate(values) for values in zip(*parts))
    return result


def snap(points, normals, target, field, steps=4):
    """Points moved along their unit normals until field (falling along the normals, about one per metre) equals
    target: (moved points, mask of those that got there without travelling far or leaving the field's range)."""
    moved = points.copy()
    travel = np.zeros(len(points))
    for _ in range(steps):
        error = np.clip(field(moved) - target, -1.5, 1.5)
        moved += normals * error[:, None]
        travel += error
    value = field(moved)
    return moved, (np.abs(value - target) < 0.1) & (np.abs(travel) < 2.5) & (value < 5.0)


def cycle_lane(road, points, along, side, size, kind, colour, field, junction):
    """A cycle lane along one side of a road, against the curb or the parking lane: its line and, where tagged red
    or untagged, a red surface; lanes tagged without colour turn red only on the last 12 m before junctions."""
    right = offset(points, 1.0) - points
    normals = right * side  # Towards this side's curb.
    parked = PARKING_LANE if parking_side(road.tags, "right" if side > 0 else "left") else 0.0
    start = offset(points, side * max(road.half - parked - size / 2, 0.5))
    centre, ok = snap(start, normals, parked + size / 2, field)
    ok &= ~junction(centre)
    result = {}
    red = ok.copy()
    if colour not in (None, "red"):
        # Distance along the road to the nearest junction point, either way.
        stops = along[~ok]
        near = np.min(np.abs(along[:, None] - stops[None, :]), axis=1) if len(stops) else np.full(len(along), np.inf)
        red &= near <= 12.0
    if red.sum() >= 2:
        result["ground/bike-lane-red"] = strip(centre, along, red, size / 2)
    inner = centre - normals * (size / 2)
    if kind == "advisory":
        result["ground/marking-white"] = strip(inner, along, ok & dashes(along, 1.0, 1.0), 0.06)
    else:
        result["ground/marking-white"] = strip(inner, along, ok, 0.125)
    return result


def crossing_paint(crossing, inside):
    """Zebra bars, or dashed edge lines following the crossing's line, over the parts that lie on the carriageway."""
    points, along = resample(crossing.line, 0.25)
    on = inside(points)
    if on.sum() < 4:
        return np.zeros((0, 4, 2)), np.zeros((0, 4, 2))
    half = crossing.width / 2
    if crossing.paint == "zebra":
        # Zebras are straight: bars 0.5 m wide every metre, their long side along the traffic.
        hits = np.flatnonzero(on)
        start, end = points[hits[0]], points[hits[-1]]
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            return np.zeros((0, 4, 2)), np.zeros((0, 4, 2))
        walk = (end - start) / length
        across = np.array([-walk[1], walk[0]])
        t = np.arange(0.25, length - 0.25, 1.0)
        centres, bar = start + t[:, None] * walk, 0.25
        quads = np.stack([centres - walk * bar - across * half, centres + walk * bar - across * half,
                          centres + walk * bar + across * half, centres - walk * bar + across * half], axis=1)
        u = np.stack([t - bar, t + bar, t + bar, t - bar], axis=1)
        v = np.broadcast_to(np.array([-half, -half, half, half]), u.shape)
        return quads, np.stack([u, v], axis=2)
    # Signalised crossings: dashed lines along both edges, following the line as it bends (cycle crossings run
    # diagonally through junctions). Walkers get 0.5 m dashes 0.12 m wide; cyclists 0.5 m dashes, 0.25 m wide,
    # with 0.2 m gaps, as German rules mark them.
    dash, gap, thick = (0.5, 0.2, 0.125) if crossing.cycle else (0.5, 0.5, 0.06)
    quads, uvs = [], []
    for side in (-half, half):
        edge = offset(points, side)
        keep = inside(edge) & on & dashes(along, dash, gap)
        q, w = strip(edge, along, keep, thick)
        quads.append(q)
        uvs.append(w)
    return np.concatenate(quads), np.concatenate(uvs)


def stop_lines(roads, crossings_, inside):
    """Stop lines 0.5 m wide across the approaching lanes, a metre before signalised crossings of painted roads."""
    result = []
    painted = [road for road in roads if road.line is not None and highway(road.tags) in PAINTED]
    for crossing in crossings_:
        if crossing.paint != "dashes" or crossing.cycle:
            continue
        points, _ = resample(crossing.line, 0.25)
        on = inside(points)
        if on.sum() < 4:
            continue
        centre = points[on].mean(axis=0)
        best = None
        for road in painted:
            d = distance(centre, road.line)
            i = int(np.argmin(d))
            if d[i] < road.half + 1.0 and (best is None or d[i] < best[0]):
                best = d[i], road, i
        if best is None:
            continue
        _, road, i = best
        a, b = road.line[i], road.line[i + 1]
        direction = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        q = a + direction * float(np.clip((centre - a) @ direction, 0, np.linalg.norm(b - a)))
        # How far the crossing's paint reaches along the road, both ways, from where it meets the centreline.
        reach = float(np.abs((points[on] - q) @ direction).max()) + crossing.width / 2 + 1.0
        if oneway(road.tags):
            approaches = [(-direction if road.tags.get("oneway") == "-1" else direction, -road.half)]
        else:
            approaches = [(direction, 0.0), (-direction, 0.0)]
        for travel, low in approaches:
            right = np.array([-travel[1], travel[0]])
            s = np.arange(low, road.half + 0.01, 0.25)
            axis = q - travel * (reach + 0.25) + s[:, None] * right
            keep = inside(axis)
            if keep.sum() >= 4:
                result.append(strip(axis, s, keep, 0.25))
    return result


# Wear on the carriageway: decals with unit uvs, u along (b–c) and v across (a–b).

DECAL_UVS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def rectangle(centre, along, length, breadth):
    """Corners (a, b, c, d) of a rectangle with a–b across and b–c along a unit direction."""
    across = np.array([-along[1], along[0]])
    l, w = along * length / 2, across * breadth / 2
    return np.array([centre - l - w, centre - l + w, centre + l + w, centre + l - w])


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def wear(roads, field, junction, curbed, seed=0):
    """Decal quads on curbed carriageways, per material: manholes every 30–60 m, drains at the curb every 20–30 m,
    patches, utility trenches, cracks, and oil drips along the lanes.

    field(points) is the signed distance into the carriageway, junction(points) marks junctions and crossings, and
    curbed(points) where the carriageway has a raised curb. Returns {material: (quads (n, 4, 2), uvs (n, 4, 2))}.
    """
    found = {}

    def put(material, corners, clear):
        """Keep a decal whose corners all lie clear metres inside the carriageway."""
        if (field(corners) >= clear).all():
            found.setdefault(material, []).append(corners)
            return True
        return False

    for road in roads:
        if road.line is None or road.material not in ASPHALT or not road.curbed or len(road.line) < 2:
            continue
        rng = np.random.default_rng([seed, int(re.sub(r"\D", "", road.id) or 0) % (1 << 31)])
        points, along = resample(road.line, 1.0)
        total = float(along[-1])
        right = offset(points, 1.0) - points
        tangent = np.column_stack((right[:, 1], -right[:, 0]))  # The right normal turned back to the direction.
        shift = lambda k, lateral: points[k] + right[k] * lateral
        at = lambda s: int(np.clip(np.searchsorted(along, s), 0, len(points) - 1))
        half = road.half
        # Manholes sit in the lanes, about in the wheel-free middle.
        s = rng.uniform(5, 35)
        while s < total:
            k = at(s)
            lateral = rng.choice([-0.5, 0.0, 0.5]) * half + rng.normal(0, 0.3)
            centre = shift(k, lateral)
            put("decal/manhole", rectangle(centre, rotate(tangent[k], rng.uniform(0, 2 * math.pi)), 0.85, 0.85), 0.5)
            s += rng.uniform(30, 60)
        # Drains in the gutter, on each curbed side.
        for side in (1, -1):
            s = rng.uniform(3, 20)
            while s < total:
                k = at(s)
                normal = np.array([-tangent[k][1], tangent[k][0]]) * side
                start = points[k:k + 1] + normal * max(half - 0.3, 0.5)
                spot, ok = snap(start, normal[None], 0.3, field)
                if ok[0] and curbed(spot)[0] and not junction(spot)[0]:
                    put("decal/drain", rectangle(spot[0], tangent[k], 0.55, 0.55), 0.0)
                s += rng.uniform(20, 30)
        # Patches: about one every 30 m, as utilities and repairs leave them; now and then a long trench by the curb.
        for _ in range(rng.poisson(total / 30)):
            k = at(rng.uniform(0, total))
            centre = shift(k, rng.uniform(-1, 1) * max(half - 1.2, 0.0))
            corners = rectangle(centre, rotate(tangent[k], rng.normal(0, 0.05)), rng.uniform(1.2, 4.0),
                                rng.uniform(0.8, 2.2))
            if not junction(corners).any():
                put("decal/patch", corners, 0.25)
        for _ in range(rng.poisson(total / 180)):
            k, length = at(rng.uniform(0, total)), rng.uniform(6, 14)
            centre = shift(k, rng.choice([-1, 1]) * max(half - 1.0, 0.0))
            corners = rectangle(centre, tangent[k], length, rng.uniform(0.6, 0.9))
            if not junction(corners).any():
                put("decal/patch", corners, 0.15)
        # Cracks: about one every 15 m, mostly along the road where the pavement settles.
        for _ in range(rng.poisson(total / 15)):
            k = at(rng.uniform(0, total))
            centre = shift(k, rng.uniform(-1, 1) * max(half - 0.6, 0.0))
            turn = rng.normal(0, 0.3) if rng.random() < 0.7 else rng.uniform(0, math.pi)
            corners = rectangle(centre, rotate(tangent[k], turn), rng.uniform(1.5, 3.5), rng.uniform(0.6, 1.0))
            if not junction(corners).any():
                put("decal/crack", corners, 0.2)
        # Oil drips in the middle of the lanes, where engines stand in traffic.
        for _ in range(rng.poisson(total / 40)):
            k = at(rng.uniform(0, total))
            centre = shift(k, (rng.choice([-0.5, 0.5]) if not oneway(road.tags) else 0) * half)
            corners = rectangle(centre, rotate(tangent[k], rng.uniform(0, math.pi)), rng.uniform(0.6, 1.3),
                                rng.uniform(0.5, 1.0))
            put("decal/oil-stain", corners, 0.3)
    return {material: (np.array(quads), np.broadcast_to(DECAL_UVS, (len(quads), 4, 2)).copy())
            for material, quads in found.items()}
