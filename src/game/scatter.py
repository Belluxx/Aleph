"""S6: trees, bushes, street props, vehicles, and grass as library instances, from the canopy height model, OSM, and
simple rules.

Trees are canopy peaks of reference.top − ground merged with OSM trees (OSM wins position, the canopy wins height).
Props come from OSM nodes and lines, plus street lamps along roads that have none mapped. Nothing stands inside a
building footprint or on a roadway, except traffic signals, which move to the road edge, and vehicles: parked along
tagged parking lanes and in parking bays, buses at stops, taxis at ranks, and a few cars waiting at signals.

Scatter hands ground what lies under its instances, through ground.data: street trees ("trees") for tree pits and
fallen leaves, and parked cars ("parked") for oil stains; ground.chunk runs after scatter and draws them.

Yaw turns an asset about +y, so its front (+z) points along (sin yaw, cos yaw) in (x, z); see assets.py.
"""

import math
import warnings
import zlib

import numpy as np

from . import roads as roads_
from .ground import line_field
from .types import SURFACES, Instance, Raster

TREE_MIN = 3.0  # Metres of canopy height that count as a tree.
BUSH = (0.8, 2.5)  # Metres of canopy height that count as low vegetation.
MATCH = 2.5  # Metres within which an OSM tree and a detected crown are the same tree.
LAMP_SPACING = 27.0  # Metres between filled-in street lamps.
LAMP_GAP = 20.0  # Metres from a mapped lamp within which no lamp is added.
EDGE = 0.6  # Metres beyond the roadway edge where lamps, signs, and signals stand.
# Berlin street tree mix, for broadleaved trees without genus or species.
MIX = dict(linden=0.38, maple=0.25, plane=0.12, chestnut=0.13, birch=0.07, pine=0.05)
GENUS = dict(tilia="linden", linde="linden", platanus="plane", platane="plane", acer="maple", ahorn="maple",
             aesculus="chestnut", castanea="chestnut", kastanie="chestnut", quercus="chestnut", eiche="chestnut",
             betula="birch", birke="birch", populus="birch", pappel="birch", pinus="pine", kiefer="pine", picea="pine",
             fichte="pine", abies="pine", tanne="pine", larix="pine", taxus="pine", ulmus="plane", ulme="plane",
             fraxinus="maple", esche="maple", carpinus="maple", robinia="linden", prunus="maple", sorbus="maple")
VEHICLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
           "service", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
LIT = VEHICLE - {"service"}  # Roads that get filled-in lamps.
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link",
         "secondary_link", "tertiary_link"}
WIDTH = dict(motorway=12, trunk=10, primary=10, secondary=9, tertiary=7, residential=6, unclassified=6, living_street=5,
             service=4)
# Vehicle models by type, as vehicles.py builds them (the default colour first), and their length and width in
# metres; parked cars mix the first four types, then a colour evenly. Berlin's streets are mostly grey, black,
# white, and silver.
TYPES = dict(sedan=((4.85, 1.84), ("", "-black", "-white", "-blue", "-red", "-grey")),
             hatchback=((4.08, 1.78), ("", "-white", "-grey", "-blue", "-silver", "-green")),
             suv=((4.7, 1.92), ("", "-white", "-grey", "-silver", "-blue")),
             van=((5.93, 2.0), ("", "-silver", "-grey")),
             taxi=((4.85, 1.84), ("",)), bus=((13.7, 2.55), ("",)))
VEHICLES = {f"vehicle/{kind}{colour}": size for kind, (size, colours) in TYPES.items() for colour in colours}
SHARES = dict(sedan=0.36, hatchback=0.32, suv=0.2, van=0.12)
CARS = {f"vehicle/{kind}{colour}": share / len(TYPES[kind][1])
        for kind, share in SHARES.items() for colour in TYPES[kind][1]}
OCCUPANCY = 0.8  # Share of parking spaces taken.
WAITING = 10  # Most cars waiting at signals in one capture.
BUSES = 3  # Most buses standing at stops.
STREETS = {"residential", "living_street", "unclassified"}  # Untagged streets where people park anyway.
GRASS = 4000  # Alpha-tested clumps dominate foliage cost in shadows and the prepass. Most grass clumps, all within GRASS_REACH metres of walked paths.
GRASS_REACH = 3.0
# Assets the library may not have yet; missing ones are left out with a warning.
OPTIONAL = ("vehicle/", "grass/", "flowers/", "prop/advertising-column", "prop/planter", "prop/bicycle")


def build(reference, ground, models, config, progress):
    """Instances of library assets: trees always; bushes, hedges, props, vehicles, and grass unless quality is
    "draft"."""
    site = Site(reference, ground)
    progress("Finding trees")
    chm = canopy(reference.top, ground.height)
    x, z = chm.centres()
    # Building cells, grown by 1 m so facades and eaves don't read as canopy.
    excluded = _max_filter(site.building.values.astype(float), 2)[site.building.index(x, z)] > 0
    trees = place_trees(reference, site, chm, excluded)
    instances = list(trees)
    notes = ground.data.setdefault("warnings", []) if ground.data is not None else []
    have = {}

    def available(name):
        if name not in have:
            have[name] = not name.startswith(OPTIONAL) or _available(name, notes)
        return have[name]

    if config.get("quality") != "draft":
        progress("Placing bushes and props")
        instances += place_bushes(reference, site, trees, chm, excluded)
        props = [i for i in place_props(reference, site) if site.allowed(i)]
        instances += props
        instances += place_furniture(reference, site, props + list(trees), available)
        progress("Parking cars")
        instances += place_vehicles(reference, site, available)
        progress("Growing grass")
        instances += place_grass(reference, site, available)
    instances = [i for i in instances if site.allowed(i) and available(i.asset)]
    if ground.data is not None:
        ground.data["trees"] = [(i.position[0], i.position[2], 0.3 * _size(i.asset)[0] * i.scale)
                                for i in instances if i.asset.startswith("tree/")]
        kept = {id(i) for i in instances}
        ground.data["parked"] = [(i.position[0], i.position[2], i.yaw, VEHICLES[i.asset][0]) for i in site.parked
                                 if id(i) in kept]
    return instances


def _available(name, notes):
    """Whether the library can make an asset; a missing one is noted once and its instances left out."""
    from . import library

    try:
        library.asset(name)
        return True
    except Exception as error:  # Not in this library version yet, or its generator failed.
        note = f"Left out {name}: the library can't make it ({type(error).__name__}: {error})."
        warnings.warn(note)
        notes.append(note)
        return False


# Where things may stand


class Site:
    """Building, roadway, and play-area tests on the ground grid, plus OSM road lines for directions."""

    def __init__(self, reference, ground):
        self.height = ground.height
        self.surface = ground.surface
        self.play = np.asarray(reference.play, float)
        self.footprints = [_ring(f) for f in reference.osm if f.kind == "building"]
        # Footprint bounding boxes grown by a metre, (x0, z0, x1, z1) each, for quick rejection.
        self.bounds = np.array([np.r_[ring.min(0) - 1, ring.max(0) + 1] for ring in self.footprints]).reshape(-1, 4)
        building = ground.surface.values == SURFACES.index("building") if ground.surface is not None else False
        self.building = Raster(np.asarray(building | _fill(ground.height, self.footprints), bool), ground.height.x0,
                               ground.height.z0, ground.height.cell)
        self.roads = [(f, f.tags["highway"]) for f in reference.osm if f.kind == "line" and f.tags.get("highway") in VEHICLE]
        self.paths = [f for f in reference.osm if f.kind == "line" and "highway" in f.tags]
        road = ground.data.get("road") if ground.data else None
        if road is None:  # Without ground's roadway field, buffer OSM road lines by their usual width.
            road = _road_field(ground.height, self.roads)
        self.road = Raster(np.asarray(road, np.float32), ground.height.x0, ground.height.z0, ground.height.cell)
        # The carriageway, without traffic islands: where vehicles may stand.
        lanes = ground.data.get("carriageway", road) if ground.data else road
        self.carriageway = Raster(np.asarray(lanes, np.float32), ground.height.x0, ground.height.z0, ground.height.cell)
        self.parked = []  # Parked vehicle instances, whose oil stains ground draws.
        gz, gx = np.gradient(self.road.values, self.road.cell)  # Points into the roadway.
        self.gradient = [Raster(g, self.road.x0, self.road.z0, self.road.cell) for g in (gx, gz)]

    def ground_y(self, x, z):
        return float(self.height.sample(x, z))

    def surface_at(self, x, z):
        if self.surface is None:
            return "grass"
        return SURFACES[int(self.surface.values[self.surface.index(x, z)])]

    def on_road(self, x, z, margin=0.0):
        return self.road.sample(x, z) > -margin

    def in_building(self, x, z):
        row, col = self.building.index(x, z)
        if self.building.values[row, col]:
            return True
        near = np.flatnonzero((self.bounds[:, 0] <= x) & (x <= self.bounds[:, 2])
                              & (self.bounds[:, 1] <= z) & (z <= self.bounds[:, 3]))
        return any(_inside(self.footprints[k], x, z) for k in near)

    def free(self, x, z, margin=0.3):
        return _inside(self.play, x, z) and not self.on_road(x, z, margin) and not self.in_building(x, z)

    def allowed(self, instance):
        x, _, z = instance.position
        road_ok = instance.asset == "prop/traffic-signal" or instance.asset in VEHICLES or not self.on_road(x, z)
        return _inside(self.play, x, z) and road_ok and not self.in_building(x, z)

    def to_edge(self, x, z, distance):
        """Move a point along the roadway field until it stands distance metres outside the roadway."""
        for _ in range(6):
            value = float(self.road.sample(x, z))
            gx, gz = (float(g.sample(x, z)) for g in self.gradient)
            norm = math.hypot(gx, gz)
            if norm < 0.2 or abs(value + distance) < 0.05:
                break
            step = (value + distance) / norm
            x, z = x - gx / norm * step, z - gz / norm * step
        return x, z

    def nearest(self, x, z, lines):
        """Nearest point, unit direction, and feature among lines (OSM features), or None."""
        best = None
        for feature in lines:
            found = _project(feature.points, x, z)
            if found and (best is None or found[0] < best[0]):
                best = found + (feature,)
        return best

    def face(self, x, z, lines=None):
        """Yaw that turns an asset's front towards the nearest road (or given lines); random-free and stable."""
        found = self.nearest(x, z, self.roads_only() if lines is None else lines)
        if found is None or found[0] < 1e-6:
            return 0.0
        q = found[1]
        return math.atan2(q[0] - x, q[1] - z)

    def roads_only(self):
        return [f for f, _ in self.roads]


def _ring(feature):
    return np.asarray(feature.points, float)


def _project(points, x, z):
    """Distance, nearest point, and unit direction of a polyline at the point nearest (x, z); None if too short."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return None
    a, b = p[:-1], p[1:]
    d = b - a
    length = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((x - a[:, 0]) * d[:, 0] + (z - a[:, 1]) * d[:, 1]) / length, 0, 1)
    q = a + d * t[:, None]
    dist = np.hypot(q[:, 0] - x, q[:, 1] - z)
    k = int(np.argmin(dist))
    return dist[k], q[k], d[k] / math.sqrt(length[k])


def _closed(feature):
    """Whether a feature's points form a ring: areas and relations omit the repeated point, closed ways keep it."""
    p = feature.points
    return feature.kind in ("area", "building") or feature.id.startswith("relation/") or (len(p) > 3 and np.allclose(p[0], p[-1]))


def _inside(ring, x, z):
    """Even-odd point-in-polygon test for one point or arrays."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    result = np.zeros(np.broadcast(x, z).shape, bool)
    for (x0, z0), (x1, z1) in zip(ring, np.roll(ring, -1, axis=0)):
        crosses = (z0 > z) != (z1 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
        result ^= crosses & (x < at)
    return result if result.shape else bool(result)


def _near(ring, x, z, margin=1.0):
    lo, hi = ring.min(0) - margin, ring.max(0) + margin
    return lo[0] <= x <= hi[0] and lo[1] <= z <= hi[1]


def _fill(grid, rings):
    """Cells of grid whose centres lie inside any ring."""
    mask = np.zeros(grid.shape, bool)
    rows, cols = grid.shape
    for ring in rings:
        (c0, r0), (c1, r1) = (np.floor((ring.min(0) - [grid.x0, grid.z0]) / grid.cell).astype(int),
                              np.ceil((ring.max(0) - [grid.x0, grid.z0]) / grid.cell).astype(int))
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, cols), min(r1, rows)
        if c0 >= c1 or r0 >= r1:
            continue
        z, x = np.mgrid[r0:r1, c0:c1]
        mask[r0:r1, c0:c1] |= _inside(ring, grid.x0 + (x + 0.5) * grid.cell, grid.z0 + (z + 0.5) * grid.cell)
    return mask


def _road_field(grid, roads):
    """Signed distance to the nearest roadway edge (positive inside), from OSM centrelines and widths."""
    x, z = grid.centres()
    field = np.full(grid.shape, -6.0)
    for feature, kind in roads:
        half = _width(feature.tags, kind) / 2
        for a, b in zip(feature.points[:-1], feature.points[1:]):
            d = b - a
            t = np.clip(((x - a[0]) * d[0] + (z - a[1]) * d[1]) / max(d.dot(d), 1e-12), 0, 1)
            field = np.maximum(field, half - np.hypot(x - a[0] - d[0] * t, z - a[1] - d[1] * t))
    return np.clip(field, -6, 6)


def _width(tags, kind):
    try:
        return float(tags["width"].split()[0])
    except (KeyError, ValueError):
        pass
    try:
        return float(tags["lanes"]) * 3.25
    except (KeyError, ValueError):
        return WIDTH.get(kind.removesuffix("_link"), 6)


def _rng(*key):
    """Deterministic randomness per place, so rebuilds put the same tree in the same spot."""
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# Trees


def canopy(top, ground):
    """Canopy height model on a 0.5 m grid aligned with the top raster: top − ground height, 0 where no mesh."""
    rows, cols = top.shape[0] // 2 * 2, top.shape[1] // 2 * 2
    x, z = top.centres()
    height = top.values[:rows, :cols] - ground.sample(x[:rows, :cols], z[:rows, :cols])
    height = np.nan_to_num(height, nan=0.0).reshape(rows // 2, 2, cols // 2, 2).max(axis=(1, 3))
    return Raster(np.clip(height, 0, None).astype(np.float32), top.x0, top.z0, top.cell * 2)


def _max_filter(values, radius):
    """Square maximum filter of a 2-D array, radius in cells."""
    out = values.copy()
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in (0, 1)], constant_values=-np.inf)
        out = np.max([np.take(padded, range(k, k + values.shape[axis]), axis=axis) for k in range(2 * radius + 1)], axis=0)
    return out


def _smooth(values):
    """3 × 3 box blur, twice: suppresses single-cell spikes such as lamp posts."""
    for _ in range(2):
        padded = np.pad(values, 1, mode="edge")
        values = sum(padded[r:r + values.shape[0], c:c + values.shape[1]] for r in range(3) for c in range(3)) / 9
    return values


def crowns(chm, excluded=None, minimum=TREE_MIN):
    """Tree crowns in a canopy height model: [(x, z, height, radius)].

    Peaks are local maxima whose neighbourhood (≈ 0.25 × height, 1.5–6 m) holds nothing higher, so tall trees
    suppress nearby sub-peaks; the crown radius is the median distance along 16 rays to where the canopy falls below
    half the peak or starts rising into a neighbour.
    """
    values = np.where(excluded, 0, chm.values) if excluded is not None else chm.values
    smooth = _smooth(values.astype(float))
    smooth = smooth + _rng("crowns").random(smooth.shape) * 1e-4  # Break ties on flat tops.
    radii = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    filters = {r: _max_filter(smooth, max(1, round(r / chm.cell))) for r in radii}
    want = np.clip(0.25 * smooth, radii[0], radii[-1])
    peak = np.zeros(smooth.shape, bool)
    for low, high in zip(radii, radii[1:] + (np.inf,)):
        band = (want >= low) & (want < high)
        peak |= band & (smooth >= filters[low])
    peak &= smooth >= minimum
    result = []
    rows, cols = chm.shape
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for row, col in zip(*np.nonzero(peak)):
        h = float(values[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].max())
        reach = []
        for a in angles:
            previous, distance = h, 0.0
            for step in np.arange(1, min(h, 12) / chm.cell + 1):
                r, c = int(round(row + math.sin(a) * step)), int(round(col + math.cos(a) * step))
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                v = smooth[r, c]
                if v < 0.5 * h or (step * chm.cell > 1.0 and v > previous + 0.3):
                    break
                previous, distance = min(previous, v), step * chm.cell
            reach.append(distance)
        radius = float(np.median(reach))
        if radius >= 1.0:
            x, z = chm.x0 + (col + 0.5) * chm.cell, chm.z0 + (row + 0.5) * chm.cell
            result.append((x, z, h, radius))
    return result


def _species(tags, aspect, rng, mixed=False):
    """Tree species from OSM genus/species/taxon tags, leaf type, or the street mix weighted by crown shape."""
    from .assets import TREES

    text = " ".join(str(tags.get(k, "")) for k in ("genus", "species", "taxon", "genus:de", "species:de", "taxon:de")).lower()
    for key, species in GENUS.items():
        if key in text:
            return species
    if tags.get("leaf_type") == "needleleaved":
        return "pine"
    names = [s for s in MIX if s != "pine" or mixed or tags.get("leaf_type") == "mixed"]
    weights = np.array([MIX[s] for s in names])
    if aspect is not None:  # Wide crowns suit planes and chestnuts, narrow ones birches.
        weights = weights * np.exp(-((aspect - np.array([TREES[s]["ratio"] for s in names])) / 0.12) ** 2) + 1e-6
    return names[rng.choice(len(names), p=weights / weights.sum())]


def _size(asset):
    """(height, radius) of a library asset as generated."""
    from . import library

    a = library.asset(asset)
    return a.height, a.radius


def place_trees(reference, site, chm, excluded):
    found = crowns(chm, excluded)
    osm = [f for f in reference.osm if f.kind == "node" and f.tags.get("natural") == "tree"]
    for f in reference.osm:  # Tree rows: a tree every 8 m along the line.
        if f.kind == "line" and f.tags.get("natural") == "tree_row":
            for p in _along(f.points, 8.0, 4.0):
                osm.append(type(f)(f.id, "node", dict(f.tags, natural="tree"), p[None]))
    woods = [(_ring(f), f.tags) for f in reference.osm if f.kind == "area" and f.tags.get("natural") == "wood"]
    trees = []  # (x, z, height or None, radius or None, tags)
    taken = set()
    pairs = sorted((math.hypot(t.points[0][0] - c[0], t.points[0][1] - c[1]), i, j)
                   for i, t in enumerate(osm) for j, c in enumerate(found)
                   if abs(t.points[0][0] - c[0]) < MATCH and abs(t.points[0][1] - c[1]) < MATCH)
    matched = {}
    for distance, i, j in pairs:
        if distance <= MATCH and i not in matched and j not in taken:
            matched[i] = j
            taken.add(j)
    for i, t in enumerate(osm):
        x0, z0 = t.points[0]
        if i in matched:
            _, _, h, r = found[matched[i]]
        else:
            h, r = _local_height(chm, x0, z0), None
            try:
                h = float(t.tags["height"].split()[0])
            except (KeyError, ValueError):
                pass
        trees.append((x0, z0, h, r, t.tags))
    osm_points = np.array([t.points[0] for t in osm]).reshape(-1, 2)
    for j, (x0, z0, h, r) in enumerate(found):
        if j in taken:
            continue
        if len(osm_points) and np.hypot(*(osm_points - [x0, z0]).T).min() < max(MATCH, 0.6 * r):
            continue  # Part of a mapped tree's crown.
        if site.on_road(x0, z0, 0.2):
            x0, z0 = site.to_edge(x0, z0, 0.5)  # Crowns overhang roads; their trunks stand at the edge.
            if site.on_road(x0, z0) or r < 2.0:
                continue
        tags = next((t for ring, t in woods if _inside(ring, x0, z0)), {"leaf_type": "broadleaved"})
        trees.append((x0, z0, h, r, dict(tags, natural="tree")))
    instances = []
    for x0, z0, h, r, tags in trees:
        if not site.free(x0, z0, 0.0):
            continue
        rng = _rng("tree", round(x0, 1), round(z0, 1))
        aspect = r / h if h and r else None
        species = _species(tags, aspect, rng, mixed=tags.get("leaf_type") == "mixed")
        name = f"tree/{species}"
        model_height, _ = _size(name)
        height = h if h and h >= TREE_MIN else model_height * rng.uniform(0.45, 0.65)  # Unmeasured: a young tree.
        scale = float(np.clip(height / model_height, 0.3, 1.6))
        tint = tuple(float(v) for v in np.clip(1 + rng.normal(0, [0.04, 0.035, 0.05]), 0.85, 1.15))
        instances.append(Instance(name, (x0, site.ground_y(x0, z0), z0), float(rng.uniform(0, 2 * math.pi)), scale, tint))
    return instances


def _local_height(chm, x, z, radius=MATCH):
    """Highest canopy within radius of a point, or None below tree height."""
    row, col = chm.index(x, z)
    k = int(radius / chm.cell)
    patch = chm.values[max(row - k, 0):row + k + 1, max(col - k, 0):col + k + 1]
    h = float(patch.max()) if patch.size else 0.0
    return h if h >= TREE_MIN else None


def _along(points, spacing, start):
    """Points every spacing metres along a polyline, the first at start."""
    points = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = np.concatenate(([0], np.cumsum(lengths)))
    if total[-1] <= 0:
        return np.empty((0, 2))
    at = np.arange(min(start, total[-1] / 2), total[-1], spacing)
    return np.column_stack([np.interp(at, total, points[:, k]) for k in (0, 1)])


def _direction(points, at):
    """Unit direction of a polyline at the points nearest to at."""
    points = np.asarray(points, float)
    d = np.diff(points, axis=0)
    mid = (points[:-1] + points[1:]) / 2
    k = np.argmin(np.hypot(*(mid[None] - np.asarray(at)[:, None]).transpose(2, 0, 1)), axis=1)
    return d[k] / np.maximum(np.linalg.norm(d[k], axis=1, keepdims=True), 1e-12)


# Bushes and hedges


def place_bushes(reference, site, trees, chm, excluded):
    instances = []
    for f in reference.osm:
        tags = f.tags
        if tags.get("barrier") == "hedge" and f.kind in ("line", "area"):
            points = np.vstack((f.points, f.points[:1])) if _closed(f) and not np.allclose(f.points[0], f.points[-1]) else f.points
            try:
                scale = float(np.clip(float(tags["height"].split()[0]) / 1.5, 0.5, 2.0))
            except (KeyError, ValueError):
                scale = 0.85
            spots = _along(points, 1.8 * scale, 0.9 * scale)
            for (x, z), (dx, dz) in zip(spots, _direction(points, spots) if len(spots) else []):
                if site.free(x, z, 0.0):
                    # The segment runs along its local x, which yaw turns to (cos yaw, −sin yaw).
                    instances.append(Instance("bush/hedge-segment", (x, site.ground_y(x, z), z), math.atan2(-dz, dx), scale))
        elif tags.get("natural") == "shrub" and f.kind == "node":
            x, z = f.points[0]
            rng = _rng("shrub", f.id)
            instances.append(Instance("bush/round", (x, site.ground_y(x, z), z), float(rng.uniform(0, 2 * math.pi)),
                                      float(rng.uniform(0.8, 1.2))))
        elif tags.get("natural") == "scrub" and f.kind == "area":
            ring = _ring(f)
            rng = _rng("scrub", f.id)
            lo, hi = ring.min(0), ring.max(0)
            count = min(int(np.prod(hi - lo) / 6), 400)  # About one bush per 6 m² of bounding box.
            for x, z in rng.uniform(lo, hi, (count, 2)):
                if _inside(ring, x, z) and site.free(x, z, 0.3):
                    instances.append(Instance("bush/round", (x, site.ground_y(x, z), z), float(rng.uniform(0, 2 * math.pi)),
                                              float(rng.uniform(0.7, 1.3))))
    instances += _low_vegetation(site, trees, instances, chm, excluded)
    return instances


def _low_vegetation(site, trees, placed, chm, excluded):
    """Bushes at low canopy peaks off roads and buildings, away from tree trunks and other bushes."""
    low = Raster(np.where(chm.values > BUSH[1], 0, chm.values), chm.x0, chm.z0, chm.cell)
    found = crowns(low, excluded, BUSH[0])
    others = np.array([(i.position[0], i.position[2]) for i in list(trees) + list(placed)]).reshape(-1, 2)
    result = []
    for x0, z0, h, r in found:
        if r < 0.6 or not site.free(x0, z0, 0.5) or site.surface_at(x0, z0) not in ("grass", "dirt"):
            continue  # On paving, low bumps are people, bins, or cars.
        if len(others) and np.hypot(*(others - [x0, z0]).T).min() < 2.5:
            continue
        others = np.vstack((others, [x0, z0]))
        rng = _rng("bush", round(x0, 1), round(z0, 1))
        result.append(Instance("bush/round", (x0, site.ground_y(x0, z0), z0), float(rng.uniform(0, 2 * math.pi)),
                               float(np.clip(h / 1.3, 0.6, 1.6))))
    return result


# Props


def _lamp(tags, kind):
    """Berlin's lanterns stand in parks and side streets, gas lamps among them; modern masts on main roads."""
    if tags.get("support") == "ground" or tags.get("lamp_mount") == "ground":
        return None  # Floodlights set in the ground, as around landmarks: nothing to stand there.
    if tags.get("lamp_type") in ("gaslight", "gas") or "lamp_flames" in tags:
        return "prop/street-lamp-berlin"
    if "bent_mast" in tags.get("lamp_mount", "") or tags.get("lamp_mount") == "cast_steel_mast":
        return "prop/street-lamp"
    return "prop/street-lamp" if kind in MAJOR else "prop/street-lamp-berlin"


def _sign(tags):
    codes = (tags.get("traffic_sign", "") + "," + tags.get("highway", "")).replace(";", ",")
    if "206" in codes or "stop" in codes.split(","):
        return "prop/sign-stop"
    if "205" in codes or "give_way" in codes:
        return "prop/sign-give-way"
    if "283" in codes or "286" in codes:
        return "prop/sign-no-stopping"
    return "prop/sign"


def _bearing(tags):
    """Yaw facing drivers who see a sign whose OSM direction (compass degrees of travel) is given, else None."""
    try:
        theta = math.radians(float(tags["direction"]))
    except (KeyError, ValueError):
        return None
    # Travel along (sin θ, −cos θ) in (x, z); the face looks back against it.
    return math.atan2(-math.sin(theta), math.cos(theta))


def place_props(reference, site):
    instances = []
    roads = site.roads_only()
    road_kind = {f.id: kind for f, kind in site.roads}

    def put(asset, x, z, yaw=None, scale=1.0, margin=0.2):
        if site.on_road(x, z, margin):
            x, z = site.to_edge(x, z, max(margin, 0.3))
        if site.free(x, z, 0.0):
            yaw = site.face(x, z) if yaw is None else yaw
            instances.append(Instance(asset, (x, site.ground_y(x, z), z), float(yaw), scale))
            return True
        return False

    lamps = []
    for f in reference.osm:
        tags = f.tags
        if f.kind == "node":
            x, z = f.points[0]
            highway = tags.get("highway")
            if highway == "street_lamp":
                near = site.nearest(x, z, roads)
                kind = road_kind.get(near[3].id) if near and near[0] < 25 else None
                asset = _lamp(tags, kind)
                if asset and put(asset, x, z):
                    lamps.append((x, z))
            elif highway in ("traffic_signals", "stop", "give_way"):
                _at_road_edge(site, instances, f, "prop/traffic-signal" if highway == "traffic_signals" else _sign(tags))
            elif "traffic_sign" in tags:
                put(_sign(tags), x, z, _bearing(tags))
            elif tags.get("amenity") == "bench":
                put("prop/bench", x, z, site.face(x, z, site.paths), margin=0.5)
            elif tags.get("amenity") == "waste_basket":
                put("prop/waste-basket", x, z, site.face(x, z, site.paths))
            elif tags.get("barrier") == "bollard":
                put("prop/bollard", x, z, margin=0.0)
            elif tags.get("amenity") == "bicycle_parking":
                _bicycle_stands(site, put, x, z, tags)
            elif tags.get("emergency") == "fire_hydrant" and tags.get("fire_hydrant:type", "pillar") == "pillar":
                put("prop/hydrant", x, z)
        elif f.kind == "line" and tags.get("barrier") == "bollard":
            for x, z in _along(f.points, 1.5, 0.0):
                put("prop/bollard", x, z, margin=0.0)
        elif f.kind == "area" and tags.get("amenity") == "bicycle_parking":
            x, z = _ring(f).mean(0)
            _bicycle_stands(site, put, x, z, tags)
    instances += _fill_lamps(site, lamps)
    return instances


def _at_road_edge(site, instances, feature, asset):
    """Signals and stop signs mapped on the road move to its right-hand edge, facing oncoming traffic."""
    x, z = feature.points[0]
    near = site.nearest(x, z, site.roads_only())
    if near is None or near[0] > 15:
        return
    _, q, (dx, dz), way = near
    if way.tags.get("oneway") == "-1" or feature.tags.get("traffic_signals:direction", feature.tags.get("direction")) == "backward":
        dx, dz = -dx, -dz
    # Right of travel along (dx, dz), seen from above with x east and z south, is (−dz, dx).
    x, z = site.to_edge(q[0] - dz * 2, q[1] + dx * 2, 0.4)
    if not site.on_road(x, z) and site.free(x, z, -1.0):
        instances.append(Instance(asset, (x, site.ground_y(x, z), z), math.atan2(-dx, -dz)))


def _bicycle_stands(site, put, x, z, tags):
    """A row of stands along the nearest road, each hoop at right angles to it."""
    try:
        count = int(np.clip(int(tags.get("capacity", 2)) // 2, 1, 8))
    except ValueError:
        count = 1
    near = site.nearest(x, z, site.roads_only())
    dx, dz = near[2] if near else (1.0, 0.0)
    yaw = math.atan2(-dx, -dz) + math.pi / 2  # Hoop (local x) across the road direction.
    for k in range(count):
        offset = (k - (count - 1) / 2) * 1.2
        put("prop/bicycle-stand", x + dx * offset, z + dz * offset, yaw)


def _fill_lamps(site, mapped):
    """Street lamps every LAMP_SPACING m along lit roads that have none within LAMP_GAP m, at the sidewalk edge."""
    result, placed = [], list(mapped)
    for feature, kind in site.roads:
        if kind not in LIT:
            continue
        spots = _along(feature.points, LAMP_SPACING, LAMP_SPACING / 2)
        if not len(spots):
            continue
        sides = (1,) if feature.tags.get("oneway") in ("yes", "-1", "1") else (1, -1)
        half = _width(feature.tags, kind) / 2
        for (x, z), (dx, dz) in zip(spots, _direction(feature.points, spots)):
            for side in sides:
                if feature.tags.get("oneway") == "-1":
                    side = -side
                px, pz = x - dz * side * (half + 1.0), z + dx * side * (half + 1.0)
                px, pz = site.to_edge(px, pz, EDGE)
                if placed and np.hypot(*(np.array(placed) - [px, pz]).T).min() < LAMP_GAP:
                    continue
                if not site.free(px, pz, EDGE * 0.5):
                    continue
                placed.append((px, pz))
                asset = "prop/street-lamp" if kind in MAJOR else "prop/street-lamp-berlin"
                result.append(Instance(asset, (px, site.ground_y(px, pz), pz), math.atan2(x - px, z - pz)))
    return result


# Vehicles


class Traffic:
    """Where vehicles may stand: inside the carriageway (or a parking area), clear of crossings, junctions, and
    driveways, and not overlapping each other."""

    def __init__(self, reference, site):
        self.site = site
        grid = site.height
        x, z = grid.centres()
        self.boxes = []  # (centre, unit direction, half length, half width)
        lanes = site.carriageway
        # Crossings, and paths or service roads where they enter the carriageway (driveways, kerb ramps).
        crossing = np.zeros(grid.shape, bool)
        for f in reference.osm:
            tags = f.tags
            if f.kind != "line" or "highway" not in tags or tags.get("highway") in VEHICLE - {"service"}:
                continue
            points, _ = roads_.resample(np.asarray(f.points, float), 0.5)
            reach = 2.5 if tags.get("highway") == "service" else 1.5
            touching = points[lanes.sample(points[:, 0], points[:, 1]) > -0.5]
            for px, pz in touching:
                r, c = grid.index(px, pz)
                k = int(reach / grid.cell) + 1
                rs, cs = slice(max(r - k, 0), r + k + 1), slice(max(c - k, 0), c + k + 1)
                crossing[rs, cs] |= np.hypot(x[rs, cs] - px, z[rs, cs] - pz) < reach
        for f in reference.osm:
            if f.kind == "node" and f.tags.get("highway") == "crossing":
                crossing |= np.hypot(x - f.points[0][0], z - f.points[0][1]) < 3.5
        # Junctions: where carriageways of two streets meet, and 6 m around, for parking.
        streets = {}
        for f, kind in site.roads:
            key = f.tags.get("name") or f.tags.get("ref") or f.id
            field = streets.setdefault(key, np.full(grid.shape, -6.0, np.float32))
            line_field(grid, np.asarray(f.points, float), _width(f.tags, kind) / 2, field)
        count = sum((field > -0.5).astype(int) for field in streets.values()) if streets else np.zeros(grid.shape)
        junction = count >= 2
        self.moving = crossing | junction  # Where no vehicle stands, even in traffic.
        self.parking = self.moving | (_max_filter(junction.astype(float), int(6 / grid.cell)) > 0)

    def keep_clear(self, x, z, radius):
        """No parking within radius metres of a point, such as a bus stop or signal."""
        gx, gz = self.site.height.centres()
        self.parking |= np.hypot(gx - x, gz - z) < radius

    def fits(self, centre, direction, length, width, parked=True, inside=None):
        """Whether a vehicle footprint is free: on the carriageway (or inside(points), for parking areas)."""
        grid = self.site.height
        points = _footprint(centre, direction, length, width)
        if inside is not None:
            if not inside(points).all():
                return False
        elif not (self.site.carriageway.sample(points[:, 0], points[:, 1]) > 0.1).all():
            return False
        r, c = grid.index(points[:, 0], points[:, 1])
        if (self.parking if parked else self.moving)[r, c].any():
            return False
        box = (np.asarray(centre, float), np.asarray(direction, float), length / 2 + 0.4, width / 2 + 0.15)
        return not any(_overlap(box, other) for other in self.boxes)

    def add(self, centre, direction, length, width):
        self.boxes.append((np.asarray(centre, float), np.asarray(direction, float), length / 2, width / 2))


def _footprint(centre, direction, length, width):
    """Corners, edge midpoints, and centre of a vehicle's rectangle."""
    d = np.asarray(direction, float)
    n = np.array([-d[1], d[0]])
    return np.array([centre + d * a * length / 2 + n * b * width / 2
                     for a in (-1, 0, 1) for b in (-1, 0, 1)])


def _overlap(a, b):
    """Separating-axis test of two oriented rectangles (centre, unit direction, half length, half width)."""
    axes = [a[1], np.array([-a[1][1], a[1][0]]), b[1], np.array([-b[1][1], b[1][0]])]
    delta = b[0] - a[0]
    for n in axes:
        reach = sum(h * abs(v @ n) for (_, d, hl, hw) in (a, b)
                    for v, h in ((d, hl), (np.array([-d[1], d[0]]), hw)))
        if abs(delta @ n) > reach:
            return False
    return True


def _yaw(direction):
    """Yaw that points an asset's front (+z) along a direction in (x, z)."""
    return math.atan2(direction[0], direction[1])


def _pick(rng, mix):
    names = list(mix)
    weights = np.array([mix[n] for n in names])
    return names[rng.choice(len(names), p=weights / weights.sum())]


def place_vehicles(reference, site, available):
    """Buses at stops, taxis at ranks, parked cars in parking areas and along parking lanes, and a few cars waiting
    at traffic signals."""
    traffic = Traffic(reference, site)
    result = []
    cars = {name: share for name, share in CARS.items() if available(name)}
    vehicle = lambda name, centre, direction: Instance(name, (float(centre[0]), site.ground_y(*centre),
                                                              float(centre[1])), _yaw(direction))

    def park(name, centre, direction, inside=None, parked=True):
        length, width = VEHICLES[name]
        if traffic.fits(centre, direction, length, width, parked, inside):
            traffic.add(centre, direction, length, width)
            result.append(vehicle(name, centre, direction))
            if parked:
                site.parked.append(result[-1])
            return True
        return False

    if available("vehicle/bus"):
        _buses(reference, site, traffic, park)
    if available("vehicle/taxi"):
        _taxi_ranks(reference, site, traffic, park)
    if cars or available("vehicle/taxi"):
        _parking_areas(reference, site, cars, park)
    if cars:
        _parking_lanes(site, cars, park)
        _waiting(reference, site, cars, park)
    return result


def _kerb(site, line, at, side, depth):
    """Centre and travel direction of a vehicle depth metres wide standing against one side's curb (side 1 right of
    the line's direction, -1 left) at the line point nearest at; None where there is no curb nearby."""
    found = _project(line, at[0], at[1])
    if found is None:
        return None
    _, q, (dx, dz) = found
    outward = np.array([-dz, dx]) * side
    lanes = lambda points: site.carriageway.sample(points[:, 0], points[:, 1])
    start = q + outward * max(float(lanes(q[None])[0]) - depth / 2 - 0.3, 0.0)
    spot, ok = roads_.snap(start[None], outward[None], depth / 2 + 0.3, lanes)
    return (spot[0], np.array([dx, dz])) if ok[0] else None


def _buses(reference, site, traffic, park):
    """A bus at some stops, against the curb beside the stop, its front a little past the sign."""
    count = 0
    for f in reference.osm:
        tags = f.tags
        if f.kind != "node" or not (tags.get("highway") == "bus_stop" or (tags.get("public_transport") == "platform"
                                                                          and tags.get("bus") == "yes")):
            continue
        x, z = f.points[0]
        traffic.keep_clear(x, z, 18.0)
        near = site.nearest(x, z, site.roads_only())
        if near is None or near[0] > 20 or count >= BUSES or _rng("bus", f.id).random() > 0.6:
            continue
        _, q, (dx, dz), way = near
        side = 1 if (x - q[0]) * -dz + (z - q[1]) * dx > 0 else -1
        travel = np.array([dx, dz]) * (side if not roads_.oneway(way.tags) else (-1 if way.tags.get("oneway") == "-1" else 1))
        placed = _kerb(site, way.points, q, side, VEHICLES["vehicle/bus"][1])
        if placed is None:
            continue
        spot, _ = placed
        for shift in (-4.0, -1.0, -7.0, 2.0):
            if park("vehicle/bus", spot + travel * shift, travel, parked=False):
                count += 1
                break


def _taxi_ranks(reference, site, traffic, park):
    """Three or four taxis queued at the curb by each taxi rank node."""
    for f in reference.osm:
        if f.kind != "node" or f.tags.get("amenity") != "taxi":
            continue
        x, z = f.points[0]
        near = site.nearest(x, z, site.roads_only())
        if near is None or near[0] > 25:
            continue
        _, q, (dx, dz), way = near
        side = 1 if (x - q[0]) * -dz + (z - q[1]) * dx > 0 else -1
        travel = np.array([dx, dz]) * (side if not roads_.oneway(way.tags) else 1)
        placed = _kerb(site, way.points, q, side, VEHICLES["vehicle/taxi"][1])
        if placed is None:
            continue
        spot, _ = placed
        length = VEHICLES["vehicle/taxi"][0] + 1.0
        for k in range(int(_rng("taxi", f.id).integers(3, 5))):
            park("vehicle/taxi", spot - travel * k * length, travel)
        traffic.keep_clear(x, z, 12.0)


def _parking_areas(reference, site, cars, park):
    """Cars in mapped parking areas: a row along street-side bays, stalls in rows on surface lots; taxis where the
    area is designated for them."""
    for f in reference.osm:
        tags = f.tags
        if f.kind != "area" or tags.get("amenity") != "parking" or tags.get("parking") in (
                "underground", "multi-storey", "rooftop") or tags.get("access") in ("private",) and _rng(
                "private", f.id).random() < 0.5:
            continue
        ring = _ring(f)
        rng = _rng("lot", f.id)
        mix = {"vehicle/taxi": 1.0} if tags.get("taxi") == "designated" else cars
        if not mix:
            continue
        edges = np.roll(ring, -1, axis=0) - ring
        axis = edges[np.argmax(np.hypot(*edges.T))]
        axis = axis / np.linalg.norm(axis)
        normal = np.array([-axis[1], axis[0]])
        u, v = ring @ axis, ring @ normal
        # Footprints must stay 0.2 m inside the outline.
        shrunk = lambda points: _inside(ring, points[:, 0], points[:, 1]) & np.array(
            [min(np.hypot(*(ring - p).T).min(), 9) > 0.2 for p in points])
        across = v.max() - v.min()
        orientation = tags.get("orientation") or tags.get("parking:orientation")
        # Cars in a bay face the way traffic flows in the lane beside it.
        centre = ring.mean(axis=0)
        near = site.nearest(*centre, site.roads_only())
        flow = rng.choice([-1, 1])
        if near is not None and near[0] < 15:
            _, q, (dx, dz), way = near
            side = 1 if (centre[0] - q[0]) * -dz + (centre[1] - q[1]) * dx > 0 else -1
            travel = np.array([dx, dz]) * ((-1 if way.tags.get("oneway") == "-1" else 1) if roads_.oneway(way.tags)
                                           else side)
            flow = 1 if travel @ axis > 0 else -1
        if across < 7.0:
            # A bay along the street: cars along its axis, or side by side for perpendicular parking.
            perpendicular = orientation in ("perpendicular", "diagonal") or across > 4.0
            middle = (v.max() + v.min()) / 2
            s = u.min() + rng.uniform(0.3, 1.5)
            while s < u.max():
                name = _pick(rng, mix)
                length, width = VEHICLES[name]
                step = (width + 0.7) if perpendicular else length + rng.uniform(0.7, 1.6)
                centre = axis * (s + step / 2) + normal * middle
                direction = normal * rng.choice([-1, 1]) if perpendicular else axis * flow
                if rng.random() < OCCUPANCY:
                    park(name, centre, direction, inside=shrunk)
                s += step
            continue
        # A surface lot: back-to-back rows of 2.5 m stalls with 6 m aisles between pairs of rows.
        for row in np.arange(v.min() + 2.5, v.max() - 2.0, 8.0):
            facing = 1 if int(round((row - v.min()) / 8.0)) % 2 == 0 else -1
            for s in np.arange(u.min() + 1.5, u.max() - 1.0, 2.5):
                name = _pick(rng, mix)
                if rng.random() < OCCUPANCY * 0.85:
                    park(name, axis * s + normal * row, normal * facing, inside=shrunk)


def _parking_lanes(site, cars, park):
    """Parked cars along roads whose parking tags put them on the carriageway, and along untagged residential
    streets wide enough to park on both sides, or on the right."""
    for f, kind in site.roads:
        tags = f.tags
        sides = {side: roads_.parking_side(tags, side) for side in ("right", "left")}
        untagged = not any(key.startswith("parking") for key in tags)
        if not any(sides.values()) and not (untagged and kind in STREETS):
            continue
        points, along = roads_.resample(np.asarray(f.points, float), 0.5)
        if along[-1] < 8:
            continue
        right = roads_.offset(points, 1.0) - points
        middle = site.carriageway.sample(points[:, 0], points[:, 1])
        rng = _rng("parking", f.id)
        for side, sign in (("right", 1), ("left", -1)):
            orientation = sides[side]
            if orientation is None and untagged:
                # Untagged: people park where the street leaves two lanes free.
                wide = np.median(middle)
                orientation = "parallel" if wide >= 4.5 or (wide >= 3.6 and side == "right") else None
            if orientation is None:
                continue
            travel = (-1 if tags.get("oneway") == "-1" else 1) if roads_.oneway(tags) else sign
            s = rng.uniform(0.5, 4.0)
            while s < along[-1] - 2:
                name = _pick(rng, cars)
                length, width = VEHICLES[name]
                if orientation == "parallel":
                    step, depth = length + rng.uniform(0.8, 2.0), width
                else:
                    step, depth = width + 0.8, length * (0.95 if orientation == "perpendicular" else 0.8)
                k = int(np.clip(np.searchsorted(along, s + step / 2), 0, len(points) - 1))
                if rng.random() < OCCUPANCY:
                    placed = _kerb(site, f.points, points[k], sign, depth)
                    if placed is not None:
                        spot, tangent = placed
                        if orientation == "parallel":
                            direction = tangent * travel
                        else:
                            outward = np.array([-tangent[1], tangent[0]]) * sign
                            turn = 0.0 if orientation == "perpendicular" else math.radians(30) * travel
                            direction = roads_.rotate(outward if rng.random() < 0.7 else -outward, turn)
                        park(name, spot, direction)
                s += step


def _waiting(reference, site, cars, park):
    """A handful of cars queued at traffic signals, in a lane of the approach, the first one 2 m before the
    signal's stop line."""
    count = 0
    for f in reference.osm:
        if f.kind != "node" or f.tags.get("highway") != "traffic_signals" or count >= WAITING:
            continue
        x, z = f.points[0]
        near = site.nearest(x, z, site.roads_only())
        if near is None or near[0] > 3.0:
            continue
        _, q, (dx, dz), way = near
        tags = way.tags
        rng = _rng("waiting", f.id)
        total = roads_.width(tags) - roads_.parking(tags) - sum(lane[1] for lane in roads_.cycle_lanes(tags))
        count_ = roads_.lanes(tags) or max(1, round(total / roads_.LANE))
        step = total / count_
        backward = 0 if roads_.oneway(tags) else int(roads_.number(tags.get("lanes:backward")) or count_ // 2)
        ways = [(1, range(backward, count_))] + ([] if roads_.oneway(tags) else [(-1, range(0, backward))])
        wanted = f.tags.get("traffic_signals:direction", f.tags.get("direction"))
        for sign, lanes in ways:
            if not len(lanes) or rng.random() > 0.55 or wanted in ("forward", "backward") and (
                    (wanted == "forward") != (sign > 0)):
                continue
            travel = np.array([dx, dz]) * sign * (-1 if tags.get("oneway") == "-1" else 1)
            lateral = -total / 2 + (rng.choice(list(lanes)) + 0.5) * step
            spot = q + np.array([-dz, dx]) * lateral
            gap = 2.0
            for _ in range(int(rng.integers(1, 4))):
                name = _pick(rng, cars)
                length = VEHICLES[name][0]
                centre = spot - travel * (gap + length / 2)
                if count >= WAITING or not park(name, centre, travel, parked=False):
                    break
                count += 1
                gap += length + rng.uniform(1.2, 2.5)


# Street furniture, flowers, and grass


def place_furniture(reference, site, placed, available):
    """Advertising columns on wide sidewalks near junctions, planters on squares and outside cafés, bicycles at some
    stands, and flower beds in gardens and mapped flower beds."""
    result = []
    taken = [(i.position[0], i.position[2]) for i in placed]

    def room(x, z, clearance):
        return not taken or np.hypot(*(np.array(taken) - [x, z]).T).min() >= clearance

    def put(asset, x, z, yaw, clearance=1.5, scale=1.0):
        if available(asset) and site.free(x, z, 0.3) and room(x, z, clearance):
            taken.append((x, z))
            result.append(Instance(asset, (float(x), site.ground_y(x, z), float(z)), float(yaw), scale))
            return True
        return False

    paved = lambda x, z: site.surface_at(x, z) == "paving"
    # Litfaßsäulen: near street corners, on sidewalks with 3 m of paving behind the spot, not on squares.
    squares = [_ring(f) for f in reference.osm if f.kind == "area" and roads_.plaza(f.tags)]
    on_square = lambda x, z: any(_near(ring, x, z, 0) and _inside(ring, x, z) for ring in squares)
    columns = 0
    for f, kind in site.roads:
        if kind not in LIT or columns >= 8:
            continue
        rng = _rng("column", f.id)
        points = np.asarray(f.points, float)
        for end, inward in ((points[0], 1), (points[-1], -1)):
            if rng.random() > 0.35:
                continue
            d = _direction(points, end[None])[0]
            for side in (1, -1):
                x, z = site.to_edge(*(end + d * 8 * inward + np.array([-d[1], d[0]]) * side * 6), 1.2)
                g = np.array([float(r.sample(x, z)) for r in site.gradient])
                away = -g / max(np.linalg.norm(g), 1e-9)  # The roadway field rises into the road.
                spot = np.array([x, z])
                # A real sidewalk: paved for 3 m behind the spot, no other roadway within 6 m behind it (medians and
                # the islands of squares are out), and buildings nearby.
                behind = [spot + away * k for k in (1.5, 3.0, 4.5, 6.0)]
                if (paved(*spot) and paved(*behind[0]) and paved(*behind[1])
                        and not any(site.on_road(*b, 0.0) or site.in_building(*b) for b in behind[:3])
                        and not site.on_road(*behind[3], 0.0) and _urban(site, x, z) and not on_square(x, z)
                        and put("prop/advertising-column", x, z, rng.uniform(0, 2 * math.pi), 4.0)):
                    columns += 1
                    break
    # Planters: along the edges of squares, and a pair outside cafés and restaurants.
    for f in reference.osm:
        tags = f.tags
        if f.kind == "area" and (tags.get("place") == "square" or tags.get("highway") == "pedestrian"):
            ring = _ring(f)
            closed = np.vstack([ring, ring[:1]])
            spots = _along(closed, 14.0, 7.0)
            rng = _rng("planter", f.id)
            count = 0
            for (x, z), (dx, dz) in zip(spots, _direction(closed, spots) if len(spots) else []):
                if count >= 4 or rng.random() > 0.25:
                    continue
                inward = np.array([-dz, dx])
                if not _inside(ring, *(np.array([x, z]) + inward * 2.0)):
                    inward = -inward
                px, pz = np.array([x, z]) + inward * 2.0
                if paved(px, pz) and put("prop/planter", px, pz, math.atan2(dx, dz), 2.5):
                    count += 1
        elif f.kind == "node" and tags.get("amenity") in ("cafe", "restaurant", "bar", "pub", "ice_cream"):
            x, z = f.points[0]
            near = site.nearest(x, z, site.roads_only())
            if near is None or near[0] > 30:
                continue
            q, (dx, dz) = near[1], near[2]
            toward = (q - [x, z]) / max(np.linalg.norm(q - [x, z]), 1e-9)
            spot = np.array([x, z], float)
            for _ in range(60):  # Out of the building, then 0.8 m onto the sidewalk.
                if not site.in_building(*spot):
                    break
                spot = spot + toward * 0.25
            spot = spot + toward * 0.8
            for side in (-1.3, 1.3):
                px, pz = spot + np.array([dx, dz]) * side
                if paved(px, pz) and not site.on_road(px, pz, 1.5):
                    put("prop/planter", px, pz, math.atan2(dx, dz), 1.0)
    # Bicycles leaning on some stands, along the hoop, on either side.
    for stand in placed:
        if stand.asset != "prop/bicycle-stand" or not available("prop/bicycle"):
            continue
        x, _, z = stand.position
        rng = _rng("bicycle", round(x, 1), round(z, 1))
        for side, chance in ((-0.32, 0.5), (0.32, 0.15)):
            if rng.random() < chance:
                bx, bz = x + math.sin(stand.yaw) * side, z + math.cos(stand.yaw) * side
                yaw = stand.yaw + math.pi / 2 + (math.pi if rng.random() < 0.5 else 0.0)
                if site.free(bx, bz, 0.3):
                    result.append(Instance("prop/bicycle", (bx, site.ground_y(bx, bz), bz), yaw))
    # Flower beds: filling mapped flower beds, and now and then beside paths in gardens.
    beds = 0
    for f in reference.osm:
        tags = f.tags
        if not available("flowers/bed") or beds >= 60:
            break
        if f.kind == "area" and tags.get("landuse") == "flowerbed":
            ring = _ring(f)
            lo, hi = ring.min(0), ring.max(0)
            for x in np.arange(lo[0] + 0.9, hi[0], 1.8):
                for z in np.arange(lo[1] + 0.9, hi[1], 1.8):
                    if beds < 60 and _inside(ring, x, z) and put("flowers/bed", x, z, _rng("bed", x, z).uniform(0, 6.3), 1.2):
                        beds += 1
        elif f.kind == "line" and tags.get("highway") in ("footway", "path") and site.surface_at(*f.points[0]) != "paving":
            rng = _rng("beds", f.id)
            for (x, z), (dx, dz) in zip(_along(f.points, 35.0, 12.0), _direction(f.points, _along(f.points, 35.0, 12.0))):
                if beds >= 60 or rng.random() > 0.25:
                    continue
                side = rng.choice([-1, 1]) * (_width(tags, "footway") / 2 + 1.4)
                px, pz = x - dz * side, z + dx * side
                if _in_garden(reference, px, pz) and site.surface_at(px, pz) == "grass" and put(
                        "flowers/bed", px, pz, math.atan2(dx, dz), 2.0):
                    beds += 1
    return result


def _urban(site, x, z, reach=25.0):
    """Whether a building stands within reach metres of a point."""
    near = ((site.bounds[:, 0] - reach <= x) & (x <= site.bounds[:, 2] + reach)
            & (site.bounds[:, 1] - reach <= z) & (z <= site.bounds[:, 3] + reach))
    return bool(near.any())


def _in_garden(reference, x, z):
    """Whether a point lies in a garden or a small park, where beds are planted; large parks stay natural."""
    for f in reference.osm:
        if f.kind == "area" and f.tags.get("leisure") in ("garden", "park"):
            ring = _ring(f)
            if f.tags["leisure"] == "park" and abs(roads_.signed_area(ring)) > 20000:
                continue
            if _near(ring, x, z, 0) and _inside(ring, x, z):
                return True
    return False


def place_grass(reference, site, available):
    """Grass clumps on lawns within GRASS_REACH metres of walked paths, thinning out away from them."""
    if not available("grass/clump") or site.surface is None:
        return []
    codes = site.surface.values
    grass = codes == SURFACES.index("grass")
    walked = np.isin(codes, [SURFACES.index(name) for name in ("paving", "gravel", "dirt")])
    # Distance in cells from walked ground, by repeated growth.
    distance = np.full(codes.shape, np.inf)
    front = walked
    steps = int(GRASS_REACH / site.surface.cell)
    for step in range(1, steps + 1):
        grown = _max_filter(front.astype(float), 1) > 0
        distance[grown & ~walked & np.isinf(distance)] = step
        front = grown
    cell = site.surface.cell
    metres = distance * cell
    density = 1.4 * np.exp(-metres / 1.1)  # Clumps per square metre: a dense edge, thinning out.
    rng = _rng("grass", codes.shape)
    pick = grass & np.isfinite(metres) & (rng.random(codes.shape) < density * cell * cell)
    pick &= ~site.building.values
    rows, cols = np.nonzero(pick)
    x = site.surface.x0 + (cols + rng.random(len(cols))) * cell
    z = site.surface.z0 + (rows + rng.random(len(rows))) * cell
    keep = _inside(site.play, x, z) & (site.road.sample(x, z) < -0.3)
    x, z = x[keep], z[keep]
    if len(x) > GRASS:
        chosen = np.sort(rng.choice(len(x), GRASS, replace=False))
        x, z = x[chosen], z[chosen]
    y = site.height.sample(x, z)
    yaw, scale = rng.uniform(0, 2 * math.pi, len(x)), rng.uniform(0.6, 1.3, len(x))
    return [Instance("grass/clump", (float(a), float(b), float(c)), float(d), float(e))
            for a, b, c, d, e in zip(x, y, z, yaw, scale)]
