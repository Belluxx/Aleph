"""S3: the ground. A calm heightfield measured from the mesh, surface layers from OSM, and per-chunk meshes.

Each layer is a signed distance field on the ground grid (metres, positive inside). Chunks cut every grid triangle
along the fields' zero lines, where the fields are linear, so road and lawn edges come out straight and smooth at
any grid size, neighbouring pieces share their cuts exactly, and no two surfaces overlap or z-fight. Roadways sit
roads.CURB below their sidewalks, with a granite curb face along the cut and a granite strip on top.

Pieces are then laid out the Berlin way by more cuts: a sett gutter along curbs, sidewalks of small-cobble mosaic
with a granite slab strip, and squares of large granite slabs framed by mosaic bands. Paint, wear, grime, tree pits,
and fallen leaves are thin decal quads draped a centimetre or two above the surface.
"""

import math
import warnings
import zlib

import numpy as np

from . import library, roads
from .types import GROUND_CELL, SURFACES, Collider, Ground, Part, Raster

BAND = 6.0  # Metres of signed distance kept on each side of an edge; fields are clamped beyond.
# Layer groups, cut in this order: building cores (no ground drawn), water, traffic islands, roadways, curb tops,
# and everything walked on, which ends with the default lawn.
GROUPS = ("core", "water", "island", "road", "curb", "walk")
STRIDES = (1, 2, 8)  # Grid cells per mesh cell at LOD 0, 1, and 2: 0.5 m, 1 m, and 4 m.
CURB_WIDTH = 0.15  # Metres of granite on top of a curb.
GAP = 0.75  # Metres paved areas grow into unmapped slivers beside them.
APRON = 2.5  # Metres of paving around buildings where nothing else is mapped.
RIM = 1.5  # Metres of ground kept inside building footprints, under walls fitted slightly off the outline.
DECAL = 0.015  # Metres paint floats above the road.
SHORE = 0.6  # Metres the water surface reaches past its outline, under the bank.
URBAN = 60.0  # Metres from a building within which roadways get curbs.
# Mapped areas (rank 0) whose outline replaces the buffered lines of each group inside them.
COVER = dict(road={"road"}, walk={"road", "island", "walk"}, water={"water"})
SURFACE = {"ground/asphalt": "asphalt", "ground/asphalt-old": "asphalt", "ground/paving": "paving",
           "ground/paving-granite": "paving", "ground/cobblestone": "paving", "ground/concrete": "paving",
           "ground/curb": "paving", "ground/grass": "grass", "ground/dirt": "dirt", "ground/gravel": "gravel",
           "water/surface": "water"}
BED = "ground/dirt"  # Material under water.
# Stand-ins for materials the library may not have yet; None leaves the detail out.
FALLBACK = {"ground/paving-granite": "ground/paving", "ground/mosaic": "ground/cobblestone",
            "ground/gutter": "ground/cobblestone", "ground/tree-grate": "ground/dirt"}
GUTTER = 0.4  # Metres of sett gutter on the road side of curbs.
SIDEWALK = 5.5  # Metres from a curbed roadway within which plain paving is laid out as a sidewalk.
PLAZA = 9.0, 0.8  # Metres between the mosaic bands framing a square's granite slabs, and their width.
# Metres decals float above the ground, per material, so overlapping ones don't fight.
LIFT = {"decal/patch": 0.010, "decal/grime": 0.011, "ground/tree-grate": 0.012, "ground/bike-lane-red": 0.012,
        "decal/crack": 0.013, "decal/oil-stain": 0.014, "ground/marking-white": DECAL, "decal/leaves": 0.017,
        "decal/manhole": 0.018, "decal/drain": 0.018}


def build(reference, config, progress):
    progress("Reading roads")
    grid = vertices(reference.play)
    sources = roads.sources(reference.osm)
    progress("Drawing surfaces")
    layers, fields, building, curbed = surfaces(grid, sources)
    road = union(fields, layers, "road")
    crossings = roads.crossings(reference.osm, [s for s in sources if s.group == "road" and s.line is not None])
    depth = depression(grid, reference.osm, crossings, curbed, building)
    label = labels(fields)
    is_road = np.array([layer[0] == "road" for layer in layers])[label]
    progress("Measuring ground")
    notes = []
    base = heights(reference, grid, building, road, depth * is_road, notes)
    level = water(reference, grid, sources, base)
    height = Raster((base - depth * is_road).astype(np.float32), grid.x0, grid.z0, grid.cell)
    codes = np.array([SURFACES.index(SURFACE.get(layer[2], "building")) for layer in layers], np.uint8)[label]
    codes[building > 0] = SURFACES.index("building")
    ground = Ground(height, Raster(codes, grid.x0, grid.z0, grid.cell), dict(
        layers=layers, fields=fields, road=road, base=base.astype(np.float32), depth=depth.astype(np.float32),
        normals=normals(base), level=level, crossings=crossings, warnings=notes))
    # The carriageway: roadway not taken by islands, water, or buildings.
    over = np.max([union(fields, layers, group) for group in ("core", "water", "island")], axis=0)
    ground.data["carriageway"] = np.minimum(road, -over).astype(np.float32)
    ground.data["sidewalk"], ground.data["zone"] = sidewalks(curbed, building)
    ground.data["plaza"], ground.data["wide"] = plaza_angles(grid, sources)
    progress("Painting roads")
    ground.data["markings"] = paint(ground, sources, crossings)
    ground.data["decals"] = wear(ground, sources, crossings)
    progress("Fitting curbs")
    ground.data["curbs"] = curb_boxes(ground)
    return ground


def chunk(ground, rect, lod):
    """Ground parts inside rect at LOD 0, 1, or 2: terrain pieces per material, curbs, paint, and water."""
    stride = STRIDES[lod]
    mesh = Cells(ground, rect, stride)
    if not mesh.count:
        return []
    polys = mesh.polygons()
    x0, z0, x1, z1 = rect
    for values in (mesh.x - x0, x1 - mesh.x, mesh.z - z0, z1 - mesh.z):
        polys, _, _ = split(polys, values)
    pieces, cuts = partition(mesh, polys, curbs=lod < 2)
    parts = {}
    for index, poly in pieces:
        group, _, material = ground.data["layers"][index]
        if group == "core":
            continue
        if group == "water":
            add(parts, BED, terrain(mesh, poly, False, lod < 2))
            continue
        for name, piece, angle in styled(mesh, group, material, poly, lod):
            add(parts, name, terrain(mesh, piece, group == "road", lod < 2, angle))
    if lod < 2:
        add(parts, "ground/curb", curb_faces(mesh, cuts))
        overlays = [ground.data["markings"], ground.data.get("decals", {}), late(ground)]
        for overlay in overlays:
            for material, (positions, uvs) in overlay.items():
                add(parts, material, decals(positions, uvs, rect))
    add(parts, "water/surface", water_plane(mesh, polys))
    # Library materials that don't exist yet fall back to near relatives, or leave the detail out.
    result = {}
    for name, chunks in parts.items():
        final = resolve(name)
        if final is not None:
            result.setdefault(final, []).extend(chunks if final == name or tiling(final) == tiling(name)
                                                else [retile(c, final) for c in chunks])
    return [part for name, chunks in result.items() if (part := merge(name, chunks)) is not None]


def colliders(ground, rect):
    """Thin convex boxes along raised curbs whose centre lies inside rect."""
    boxes, centres = ground.data["curbs"]
    x0, z0, x1, z1 = rect
    keep = (centres[:, 0] >= x0) & (centres[:, 0] < x1) & (centres[:, 1] >= z0) & (centres[:, 1] < z1)
    return [Collider("convex", vertices=box.astype(np.float32)) for box in boxes[keep]]


# Grid and fields.

def vertices(play):
    """Raster whose cell centres are the ground vertices: multiples of GROUND_CELL, aligned to 4 m."""
    lo = np.floor(np.min(play, axis=0) / 4) * 4 - 4
    hi = np.ceil(np.max(play, axis=0) / 4) * 4 + 4
    cols, rows = np.round((hi - lo) / GROUND_CELL).astype(int) + 1
    return Raster(np.zeros((rows, cols), np.float32), lo[0] - GROUND_CELL / 2, lo[1] - GROUND_CELL / 2, GROUND_CELL)


def coordinates(grid):
    """x of each column and z of each row of vertices."""
    rows, cols = grid.shape
    return grid.x0 + (np.arange(cols) + 0.5) * grid.cell, grid.z0 + (np.arange(rows) + 0.5) * grid.cell


def window(grid, lo, hi):
    """Row and column slices of the vertices inside the box lo–hi."""
    rows, cols = grid.shape
    c0, r0 = np.maximum(np.ceil((np.asarray(lo) - (grid.x0, grid.z0)) / grid.cell - 0.5).astype(int), 0)
    c1, r1 = np.floor((np.asarray(hi) - (grid.x0, grid.z0)) / grid.cell - 0.5).astype(int) + 1
    return slice(r0, min(r1, rows)), slice(c0, min(c1, cols))


def segment_distance(x, z, a, b):
    d = b - a
    t = np.clip(((x - a[0]) * d[0] + (z - a[1]) * d[1]) / max(float(d @ d), 1e-12), 0, 1)
    return np.hypot(x - a[0] - t * d[0], z - a[1] - t * d[1])


def polygon_field(grid, rings, field):
    """Raise field to the signed distance of a polygon with holes, where that is higher."""
    points = np.concatenate(rings)
    rs, cs = window(grid, points.min(axis=0) - BAND, points.max(axis=0) + BAND)
    if rs.start >= rs.stop or cs.start >= cs.stop:
        return
    xs, zs = coordinates(grid)
    xs, zs = xs[cs], zs[rs]
    a = np.concatenate(rings)
    b = np.concatenate([np.roll(ring, -1, axis=0) for ring in rings])
    # Even-odd fill: count edge crossings to the right of each vertex along its row.
    sloped = a[:, 1] != b[:, 1]
    a1, b1 = a[sloped], b[sloped]
    hit = (a1[:, 1][:, None] > zs) != (b1[:, 1][:, None] > zs)
    edge, row = np.nonzero(hit)
    at = a1[edge, 0] + (zs[row] - a1[edge, 1]) * (b1[edge, 0] - a1[edge, 0]) / (b1[edge, 1] - a1[edge, 1])
    first = np.clip(np.ceil((at - xs[0]) / grid.cell), 0, len(xs)).astype(int)  # Columns at or right of the hit.
    counts = np.zeros((len(zs), len(xs) + 1), np.int32)
    np.add.at(counts, (row, first), 1)
    inside = np.cumsum(counts, axis=1)[:, :-1] % 2 == 1
    distance = np.full(inside.shape, BAND)
    for p, q in zip(a, b):
        r, c = window(grid, np.minimum(p, q) - BAND, np.maximum(p, q) + BAND)
        r, c = slice(max(r.start, rs.start) - rs.start, min(r.stop, rs.stop) - rs.start), \
            slice(max(c.start, cs.start) - cs.start, min(c.stop, cs.stop) - cs.start)
        if r.start < r.stop and c.start < c.stop:
            d = segment_distance(xs[c][None, :], zs[r][:, None], p, q)
            np.minimum(distance[r, c], d, out=distance[r, c])
    np.maximum(field[rs, cs], np.where(inside, distance, -distance), out=field[rs, cs])


def line_field(grid, points, half, field):
    """Raise field to the signed distance of a polyline buffered by half its width, where that is higher."""
    xs, zs = coordinates(grid)
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    for p, q in zip(points[:-1], points[1:]):
        r, c = window(grid, np.minimum(p, q) - half - BAND, np.maximum(p, q) + half + BAND)
        if r.start < r.stop and c.start < c.stop:
            d = half - segment_distance(xs[c][None, :], zs[r][:, None], p, q)
            np.maximum(field[r, c], np.clip(d, -BAND, BAND), out=field[r, c])


def surfaces(grid, sources):
    """Layers (group, rank, material), their fields, and the building and curbed-roadway fields.

    Lines that run inside mapped areas of their kind are dropped there, since the areas trace the real edges.
    """
    shape = grid.shape
    blank = lambda: np.full(shape, -BAND, np.float32)
    building, curbed = blank(), blank()
    keys = sorted({(s.group, s.rank, s.material) for s in sources if s.group != "core"}
                  | {("walk", 8, s.material) for s in sources if s.rings is not None and s.rank == 0
                     and s.group in ("walk", "island")}
                  | {("core", 0, ""), ("curb", 0, "ground/curb"), ("walk", 7, "ground/paving"),
                     ("walk", 9, "ground/grass")},
                  key=lambda key: (GROUPS.index(key[0]), key[1], key[2]))
    fields = {key: blank() for key in keys}
    covers = {}
    for s in sorted(sources, key=lambda s: s.line is not None):
        if s.group == "core":
            polygon_field(grid, s.rings, building)
            continue
        target = fields[(s.group, s.rank, s.material)]
        if s.rings is not None:
            polygon_field(grid, s.rings, target)
            if s.curbed:
                polygon_field(grid, s.rings, curbed)
            if (s.group, s.rank) in (("walk", 0), ("island", 0)):
                # Mapped paved areas also fill slivers between them that the mapper left, rather than lawn.
                np.maximum(fields[("walk", 8, s.material)], target + GAP, out=fields[("walk", 8, s.material)])
            continue
        if s.group not in covers:
            covers[s.group] = highest([f for (group, rank, _), f in fields.items()
                                       if rank == 0 and group in COVER[s.group]], shape)
        for piece in uncovered(grid, s.line, covers[s.group]):
            line_field(grid, piece, s.half, target)
            if s.curbed:
                line_field(grid, piece, s.half, curbed)
    fields[("core", 0, "")] = building - RIM
    fields[("walk", 7, "ground/paving")] = building + APRON
    fields[("walk", 9, "ground/grass")][:] = BAND
    return keys, np.stack([fields[key] for key in keys]), building, curbed


def uncovered(grid, line, cover, step=1.0):
    """Pieces of a polyline outside a field's positive part, each overlapping the cover by one step."""
    points, _ = roads.resample(line, step)
    row, col = grid.index(points[:, 0], points[:, 1])
    free = cover[row, col] <= 0.5
    free = free | np.r_[free[1:], False] | np.r_[False, free[:-1]]
    edges = np.flatnonzero(np.diff(np.r_[0, free.astype(int), 0]))
    return [points[a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]


def union(fields, layers, group):
    return highest([f for f, layer in zip(fields, layers) if layer[0] == group], fields.shape[1:])


def highest(fields, shape):
    """Elementwise maximum of fields, or -BAND everywhere without any."""
    return np.max(fields, axis=0) if fields else np.full(shape, -BAND, np.float32)


def labels(fields):
    """Index of the first layer whose field is positive at each vertex."""
    return np.argmax(fields > 0, axis=0)


# Heights.

def box(values, radius, axis):
    """Mean over 2 radius + 1 cells along an axis, clamping at the edges."""
    pad = [(0, 0)] * values.ndim
    pad[axis] = (radius + 1, radius)
    total = np.cumsum(np.pad(values, pad, mode="edge"), axis=axis)
    n = values.shape[axis]
    return (np.take(total, np.arange(2 * radius + 1, n + 2 * radius + 1), axis=axis)
            - np.take(total, np.arange(n), axis=axis)) / (2 * radius + 1)


def blur(values, sigma, weights=None):
    """Near-gaussian blur with sigma in cells (three box passes); with weights, a normalized convolution."""
    radius = max(1, round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2))
    if weights is not None:
        return blur(np.where(weights > 0, values * weights, 0), sigma) / np.maximum(blur(weights, sigma), 1e-9)
    out = values.astype(np.float64)
    for _ in range(3):
        out = box(box(out, radius, 0), radius, 1)
    return out


def fill(values, iterations=30):
    """NaNs filled smoothly from their surroundings: push-pull through a pyramid, then relaxation."""
    known = np.isfinite(values)
    if known.all() or not known.any():
        return np.where(known, values, 0.0)
    levels = [(np.where(known, values, 0.0), known.astype(float))]
    while max(levels[-1][0].shape) > 1:
        v, w = levels[-1]
        pad = [(0, v.shape[0] % 2), (0, v.shape[1] % 2)]
        v, w = np.pad(v, pad), np.pad(w, pad)
        levels.append((v[::2, ::2] + v[1::2, ::2] + v[::2, 1::2] + v[1::2, 1::2],
                       w[::2, ::2] + w[1::2, ::2] + w[::2, 1::2] + w[1::2, 1::2]))
    estimate = levels[-1][0] / np.maximum(levels[-1][1], 1e-9)
    for v, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(estimate, 2, axis=0), 2, axis=1)
        up = blur(np.pad(up, 1, mode="edge"), 0.8)[1:-1, 1:-1][:v.shape[0], :v.shape[1]]
        estimate = np.where(w > 0, v / np.maximum(w, 1e-9), up)
    result = np.where(known, values, estimate)
    for _ in range(iterations):
        p = np.pad(result, 1, mode="edge")
        result = np.where(known, values, (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4)
    return result


def gather(raster, x, z, offsets=0.125):
    """Raster values at four points around each vertex, NaN outside the raster: (4, rows, cols)."""
    rows, cols = raster.shape
    out = []
    for dx in (-offsets, offsets):
        for dz in (-offsets, offsets):
            col = np.floor((x + dx - raster.x0) / raster.cell).astype(int)
            row = np.floor((z + dz - raster.z0) / raster.cell).astype(int)
            valid = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
            out.append(np.where(valid, raster.values[row.clip(0, rows - 1), col.clip(0, cols - 1)], np.nan))
    return np.stack(out)


def blocks(values, size, q, minimum):
    """Percentile q of the finite values in size × size blocks; NaN where fewer than minimum."""
    rows, cols = values.shape
    padded = np.pad(values, ((0, -rows % size), (0, -cols % size)), constant_values=np.nan)
    r, c = padded.shape[0] // size, padded.shape[1] // size
    v = padded.reshape(r, size, c, size).transpose(0, 2, 1, 3).reshape(r, c, size * size)
    v = np.sort(v, axis=2)  # NaNs sort last.
    count = np.isfinite(v).sum(axis=2)
    pick = np.clip(np.round(q / 100 * (count - 1)).astype(int), 0, size * size - 1)
    result = np.take_along_axis(v, pick[..., None], axis=2)[..., 0]
    return np.where(count >= minimum, result, np.nan)


def heights(reference, grid, building, road, lowered, notes):
    """Sidewalk-level ground y per vertex: the lowest robust mesh surface outside buildings, filled and smoothed.

    Cars, kiosks, and canopy stand above a local low percentile and are dropped; holes are filled from the DEM
    shifted onto the samples, plus the samples' smooth residual from it; roads are smoothed harder than the rest.
    Roadway samples are raised by how far roadways are lowered, so the curb step isn't smoothed away twice.
    """
    xs, zs = coordinates(grid)
    x, z = np.meshgrid(xs, zs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN cells outside the mesh.
        g = np.nanmedian(gather(reference.bottom, x, z), axis=0)
        up = np.nanmean(gather(reference.up, x, z), axis=0)
    g[(building > -1.0) | ~(up >= 0.5)] = np.nan
    g += lowered
    # Drop samples off a smooth low reference: first at 32 m blocks, against pits like courtyards the mesh sees
    # into; then at 4 m blocks, against cars and kiosks; then twice against the samples' own local mean.
    for size, q, above, below in ((64, 50, 2.5, 2.5), (8, 25, 0.4, 1.0)):
        low = reference_surface(g, grid, size, q)
        g[(g - low > above) | (g - low < -below)] = np.nan
    for above, below in ((0.25, 0.6), (0.15, 0.4)):
        known = np.isfinite(g).astype(float)
        local = blur(np.nan_to_num(g), 3.0 / grid.cell, known)
        g[(g - local > above) | (g - local < -below)] = np.nan
    base = fill_ground(g, dem_at(reference, x, z), notes)
    # Gentle everywhere, harder along roads, where cars and photogrammetry ripples would read as bumps.
    soft = blur(base, 1.0 / grid.cell)
    hard = blur(base, 3.0 / grid.cell)
    weight = np.clip(blur((road > -1.0).astype(float), 1.0 / grid.cell), 0, 1)
    return soft * (1 - weight) + hard * weight


def reference_surface(g, grid, size, q):
    """Percentile q of samples in size-cell blocks, median filtered over 3 × 3 blocks, filled, and interpolated."""
    coarse = blocks(g, size, q, size * size // 6)
    if not np.isfinite(coarse).any():
        return np.full(g.shape, np.nan)
    padded = np.pad(coarse, 1, constant_values=np.nan)
    rows, cols = coarse.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Blocks without samples nearby.
        coarse = np.nanmedian([padded[1 + i:rows + 1 + i, 1 + j:cols + 1 + j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
                              axis=0)
    x, z = np.meshgrid(*coordinates(grid))
    return Raster(fill(coarse), grid.x0, grid.z0, grid.cell * size).sample(x, z)


def dem_at(reference, x, z):
    """The DEM at vertices, or None without one."""
    dem = reference.dem
    if dem is None or not np.isfinite(dem.values).any():
        return None
    return Raster(fill(np.where(np.isfinite(dem.values), dem.values, np.nan)), dem.x0, dem.z0, dem.cell).sample(x, z)


def fill_ground(g, dem, notes):
    """Samples with their holes filled: along the DEM shifted onto them where the DEM agrees with them in shape.

    Terrain tiles in cities often include buildings, smoothed; then the DEM would raise the ground around them,
    so it is only trusted when its residual from the samples is small.
    """
    known = np.isfinite(g)
    if not known.any():
        return np.zeros(g.shape) if dem is None else dem
    if dem is not None:
        residual = g[known] - dem[known]
        offset = np.median(residual)
        spread = np.median(np.abs(residual - offset))
        if spread < 0.5:
            return dem + offset + fill(g - dem - offset)
        notes.append(f"terrain.tif departs from the measured ground by {spread:.1f} m (median spread), so ground "
                     "holes were filled from the mesh alone.")
    return fill(g)


def water(reference, grid, sources, base):
    """Water level per vertex near water (NaN elsewhere); lowers base into a bed and keeps banks above the water."""
    level = np.full(base.shape, np.nan, np.float32)
    x, z = np.meshgrid(*coordinates(grid))
    for s in sources:
        if s.group != "water":
            continue
        f = np.full(grid.shape, -BAND, np.float32)
        if s.rings is not None:
            polygon_field(grid, s.rings, f)
        else:
            line_field(grid, s.line, s.half, f)
        # Each body of water gets its own level, measured on the mesh and kept below its shore.
        near, inner, ring = f > -3.0, f > 1.0, (f > -3.0) & (f <= -1.0)
        if not near.any():
            continue
        bottom = reference.bottom.sample(x[inner], z[inner])
        bottom = bottom[np.isfinite(bottom)]
        shore = np.percentile(base[ring], 10) if ring.any() else np.median(base[near])
        surface = min(float(np.median(bottom)), shore - 0.1) if len(bottom) else shore - 0.3
        level[near] = surface
        # The bed falls away inside; the bank crosses the water 0.375 m outside the outline, under the plane's edge.
        profile = surface - np.clip(0.15 + 0.4 * f, -0.25, 1.2)
        under, bank = f >= -0.375, (f < -0.375) & (f >= -1.0)
        base[under] = np.minimum(base[under], profile[under])
        base[bank] = np.maximum(base[bank], profile[bank])
    return level


def depression(grid, osm, crossings, curbed, building):
    """How far roadways sit below their sidewalks per vertex: a curb in town, lowered at crossings."""
    near = blur((building > 0).astype(float), URBAN / grid.cell / 2) > 1e-3
    depth = roads.CURB * ((curbed > -1.0) & near)
    depth = blur(depth, 0.75 / grid.cell)
    lowered = np.full(grid.shape, -BAND, np.float32)
    for crossing in crossings:
        line_field(grid, crossing.line, crossing.width / 2 + 0.3, lowered)
    for f in osm:
        kerb = f.tags.get("kerb")
        if kerb in ("lowered", "flush", "no") and (f.kind == "node" or f.tags.get("barrier") == "kerb"):
            line_field(grid, f.points, 1.5 if f.kind == "node" else 0.6, lowered)
    return depth * (0.25 + 0.75 * np.clip(-lowered / 1.5, 0, 1))


def normals(base):
    """Unit vertex normals of the base heightfield."""
    dz, dx = np.gradient(base, GROUND_CELL)
    n = np.stack([-dx, np.ones_like(base), -dz], axis=-1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


# Cutting grid triangles along fields.

class Cells:
    """Grid triangles covering a rect at a stride, with their corner values."""

    def __init__(self, ground, rect, stride):
        grid = ground.height
        self.ground = ground
        rows, cols = grid.shape
        x0, z0, x1, z1 = rect or (-math.inf, -math.inf, math.inf, math.inf)
        step = GROUND_CELL * stride
        # Vertices at multiples of the stride from the grid origin, which is 4 m aligned.
        index = lambda v, origin, round_: (round_((v - origin - GROUND_CELL / 2) / step) * stride if math.isfinite(v)
                                           else 0 if v < 0 else 1 << 30)
        c0, r0 = max(0, index(x0, grid.x0, math.floor)), max(0, index(z0, grid.z0, math.floor))
        c1, r1 = min(cols - 1, index(x1, grid.x0, math.ceil)), min(rows - 1, index(z1, grid.z0, math.ceil))
        rs, cs = np.arange(r0, r1 + 1, stride), np.arange(c0, c1 + 1, stride)
        self.count = 0
        if len(rs) < 2 or len(cs) < 2:
            return
        r, c = np.meshgrid(rs[:-1], cs[:-1], indexing="ij")
        r, c = r.ravel(), c.ravel()
        s = stride
        # Two triangles per cell, split along the same diagonal everywhere.
        a, b, d, e = (r, c), (r + s, c), (r + s, c + s), (r, c + s)
        self.rows = np.concatenate([np.stack([a[0], b[0], d[0]], 1), np.stack([a[0], d[0], e[0]], 1)])
        self.cols = np.concatenate([np.stack([a[1], b[1], d[1]], 1), np.stack([a[1], d[1], e[1]], 1)])
        self.count = len(self.rows)
        self.x = grid.x0 + (self.cols + 0.5) * GROUND_CELL
        self.z = grid.z0 + (self.rows + 0.5) * GROUND_CELL

    def keep(self, mask):
        """Drop the triangles outside mask."""
        self.rows, self.cols, self.x, self.z = self.rows[mask], self.cols[mask], self.x[mask], self.z[mask]
        self.count = int(mask.sum())

    def corners(self, values):
        """(t, 3) corner values of a vertex array, or (t, 3, k) for arrays with trailing axes."""
        return values[self.rows, self.cols]

    def polygons(self):
        bary = np.broadcast_to(np.eye(3), (self.count, 3, 3)).copy()
        return np.arange(self.count), bary, np.full(self.count, 3)


def evaluate(polys, corner):
    """Values at polygon vertices of a field linear on each triangle, from its (t, 3) corner values."""
    tri, bary, _ = polys
    return np.einsum("nmk,nk->nm", bary, corner[tri])


def split(polys, corner):
    """Split convex polygons by a field linear on each triangle, given its (t, 3) corner values: (positive part,
    the rest, cut segments).

    Polygons are (triangle index, barycentric vertices (n, m, 3), vertex count); cut segments are
    (triangle index, two barycentric endpoints (n, 2, 3)).
    """
    tri, bary, count = polys
    if not len(tri):
        return polys, polys, (tri, np.zeros((0, 2, 3)))
    values = evaluate(polys, corner)
    m = bary.shape[1]
    valid = np.arange(m) < count[:, None]
    positive = (values > 0) & valid
    n_positive = positive.sum(axis=1)
    whole_in, whole_out = n_positive == count, n_positive == 0
    mixed = ~whole_in & ~whole_out
    take = lambda mask: (tri[mask], bary[mask], count[mask])
    if not mixed.any():
        return take(whole_in), take(whole_out), (tri[:0], np.zeros((0, 2, 3)))
    t, b, n, s = tri[mixed], bary[mixed], count[mixed], values[mixed]
    ok, pos = valid[mixed], positive[mixed]
    following = np.where(np.arange(m) + 1 < n[:, None], np.arange(m) + 1, 0)
    s1 = np.take_along_axis(s, following, 1)
    b1 = np.take_along_axis(b, following[..., None], 1)
    pos1 = np.take_along_axis(pos, following, 1)
    cross = ok & (pos != pos1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(cross, s / (s - s1), 0)
    cut = b + u[..., None] * (b1 - b)

    def side(keep):
        candidates = np.stack([b, cut], axis=2).reshape(len(t), 2 * m, 3)
        chosen = np.stack([keep & ok, cross], axis=2).reshape(len(t), 2 * m)
        order = np.argsort(~chosen, axis=1, kind="stable")
        counts = chosen.sum(axis=1)
        width = max(3, int(counts.max()))
        out = np.take_along_axis(candidates, order[:, :width, None], 1)
        good = counts >= 3
        return t[good], out[good], counts[good]

    inner, outer = side(pos), side(~pos)
    first = np.argmax(cross, axis=1)
    second = m - 1 - np.argmax(cross[:, ::-1], axis=1)
    rows = np.arange(len(t))
    segments = (t, np.stack([cut[rows, first], cut[rows, second]], axis=1))
    return concat(take(whole_in), inner), concat(take(whole_out), outer), segments


def concat(*polys):
    width = max(p[1].shape[1] for p in polys)
    pad = lambda bary: np.pad(bary, ((0, 0), (0, width - bary.shape[1]), (0, 0)))
    return tuple(np.concatenate(values) for values in zip(*[(p[0], pad(p[1]), p[2]) for p in polys]))


def partition(mesh, polys, curbs=True):
    """Pieces per layer index, and the roadway's edges as (segments, corner field, sign) for curb faces.

    The roadway is cut first, so every roadway edge is a known cut: its outline, and where buildings, water, and
    traffic islands take ground back from it. The field rises into the road where sign is 1, falls where -1.
    """
    layers, fields = mesh.ground.data["layers"], mesh.ground.data["fields"]
    corner = mesh.corners(np.moveaxis(fields, 0, -1))  # (t, 3, layers)
    members = {group: [i for i, layer in enumerate(layers) if layer[0] == group] for group in GROUPS}
    field = lambda group: corner[..., members[group]].max(axis=-1, initial=-BAND)
    road, rest, segments = split(polys, field("road"))
    cuts = [(segments, field("road"), 1)]
    regions = {group: [] for group in GROUPS}
    for group in ("core", "water", "island"):
        taken, road, segments = split(road, field(group))
        cuts.append((segments, field(group), -1))
        regions[group].append(taken)
    regions["road"].append(road)
    for group in GROUPS:
        if group == "road":
            continue
        if group == "curb":
            if not curbs:
                continue
            depth = mesh.corners(mesh.ground.data["depth"])
            inside, rest, _ = split(rest, field("road") + CURB_WIDTH * np.clip(depth / roads.CURB, 0, 1))
        elif group == GROUPS[-1]:
            inside = rest
        else:
            inside, rest, _ = split(rest, field(group))
        regions[group].append(inside)
    if curbs and regions["island"]:
        # Traffic islands get a granite rim along the lowered roadway, not where they meet other paving.
        depth = mesh.corners(mesh.ground.data["depth"])
        width = CURB_WIDTH * np.clip(depth / roads.CURB, 0, 1)
        rim, inner, _ = split(concat(*regions["island"]), np.minimum(width - field("island"), field("road") + width))
        regions["island"], regions["curb"] = [inner], regions["curb"] + [rim]
    pieces = []
    for group in GROUPS:
        inside = regions[group][0] if len(regions[group]) == 1 else concat(*regions[group]) if regions[group] else None
        if inside is None or not members[group]:
            continue
        for index in members[group][:-1]:
            part, inside, _ = split(inside, corner[..., index])
            pieces.append((index, part))
        pieces.append((members[group][-1], inside))
    return pieces, cuts


# Laying out pieces: gutters, sidewalks, and squares.

def resolve(name):
    """The library material to use for name: itself, else its fallback, else None."""
    while name is not None and name not in library.MATERIALS:
        name = FALLBACK.get(name)
    return name


def tiling(name):
    """Whether a material tiles by the metre (uvs in metres) rather than spanning unit uvs."""
    spec = library.MATERIALS.get(name, {})
    return library.MATERIALS.get(spec.get("like"), spec).get("size", 1.0) > 0


def retile(chunk, material):
    """Triangles with uvs in metres from their positions, for a tiling stand-in of a unit-uv material."""
    positions, normals_, uvs, colors = chunk
    if tiling(material):
        uvs = np.stack([positions[:, 0], -positions[:, 2]], axis=-1)
    return positions, normals_, uvs, colors


def sidewalks(curbed, building):
    """Per vertex: the granite strip field of sidewalks (positive on the strip) and the sidewalk zone field
    (positive within SIDEWALK metres of a curbed roadway).

    Berlin sidewalks run, from the curb: a band of small-cobble mosaic where trees and lamps stand, a strip of large
    granite slabs to walk on, and mosaic again up to the facades. Both bands scale with the sidewalk's width, the
    distance from the curb to the nearest building (or BAND metres when there is none).
    """
    d = np.clip(-curbed, 0, None)
    width = d + np.clip(-building, 0, None)
    near = np.clip(0.3 * width, 0.8, 1.8)
    granite = np.clip(width - near - 0.6, 0, 2.4)
    strip = np.minimum(np.minimum(d - near, near + granite - d), granite - 1.0)
    return strip.astype(np.float32), (SIDEWALK - d).astype(np.float32)


def plaza_angles(grid, sources):
    """Per vertex, the direction (radians) the slab pattern of the square or pedestrian street there follows, NaN
    elsewhere; and a field positive where squares are wide enough for the mosaic bands framing their slabs.

    Squares follow their longest side, from the largest square down, so paving mapped inside a square lines up with
    it; pedestrian streets follow their centreline. Pedestrian areas only a sidewalk wide get plain slabs.
    """
    angle = np.full(grid.shape, np.nan, np.float32)
    depth = np.full(grid.shape, -BAND, np.float32)
    granite = [s for s in sources if s.material == "ground/paving-granite"]
    areas = sorted([s for s in granite if s.rings is not None],
                   key=lambda s: (not roads.plaza(s.tags), -abs(roads.signed_area(s.rings[0]))))
    for s in areas:
        f = np.full(grid.shape, -BAND, np.float32)
        ring = s.rings[0]
        edges = np.roll(ring, -1, axis=0) - ring
        longest = edges[np.argmax(np.hypot(edges[:, 0], edges[:, 1]))]
        polygon_field(grid, s.rings, f)
        np.maximum(depth, f, out=depth)
        # Squares set the angle over their whole outline; paving inside them only where no square did.
        cover = (f > -3.0) if roads.plaza(s.tags) else (f > -1.0) & np.isnan(angle)
        angle[cover] = math.atan2(longest[1], longest[0]) % (math.pi / 2)
    for s in granite:
        if s.line is None:
            continue
        f = np.full(grid.shape, -BAND, np.float32)
        for p, q in zip(s.line[:-1], s.line[1:]):
            f[:] = -BAND
            line_field(grid, np.array([p, q]), s.half + 1.0, f)
            angle[(f > 0) & np.isnan(angle)] = math.atan2(q[1] - p[1], q[0] - p[0]) % (math.pi / 2)
    # Within 6 m of a point 4.5 m or more inside a square: the square is at least 9 m across there.
    return angle, (spread(depth, round(BAND / grid.cell)) - 4.5).astype(np.float32)


def spread(values, radius):
    """Square maximum filter of a 2-D array, radius in cells, by doubling steps."""
    out, reach = values.copy(), 0
    while reach < radius:
        step = min(reach + 1, radius - reach)
        for axis in (0, 1):
            shifted = [np.roll(out, k, axis=axis) for k in (-step, step)]
            # Rolled-in rows from the far side don't count.
            index = [slice(None)] * 2
            index[axis] = slice(-step, None)
            shifted[0][tuple(index)] = -np.inf
            index[axis] = slice(None, step)
            shifted[1][tuple(index)] = -np.inf
            out = np.maximum(out, np.maximum(*shifted))
        reach += step
    return out


def styled(mesh, group, material, polys, lod, squares=True):
    """A layer's pieces laid out as [(material, polys, uv angle per mesh triangle or None)]: a sett gutter along
    curbs on roadways, mosaic and a granite strip on sidewalks, and granite slabs framed by mosaic on squares."""
    data = mesh.ground.data
    if group == "road" and material in roads.ASPHALT and lod < 2:
        depth = mesh.corners(data["depth"]) / roads.CURB
        gutter, rest, _ = split(polys, np.minimum(GUTTER - mesh.corners(data["carriageway"]), depth - 0.3))
        return [(material, rest, None), ("ground/gutter", gutter, along(mesh, data["carriageway"]))]
    if material == "ground/paving" and squares:
        # Plain paving next to a square, such as the apron along its facades, belongs to the square.
        inside = np.isfinite(mesh.corners(data["plaza"])).any(axis=1)[polys[0]]
        if inside.any():
            return (styled(mesh, group, "ground/paving-granite", select(polys, inside), lod)
                    + styled(mesh, group, material, select(polys, ~inside), lod, squares=False))
    if group in ("walk", "island") and material == "ground/paving":
        zone, rest, _ = split(polys, mesh.corners(data["zone"]))
        if group == "island":
            return [("ground/mosaic", zone, None), (material, rest, None)]
        granite, mosaic, _ = split(zone, mesh.corners(data["sidewalk"]))
        return [("ground/paving-granite", granite, along(mesh, data["zone"])), ("ground/mosaic", mosaic, None),
                (material, rest, None)]
    if material == "ground/paving-granite":
        angle = np.nan_to_num(np.nanmax(mesh.corners(data["plaza"]), axis=1, initial=-np.inf), neginf=0.0)
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        x, z = mesh.x, mesh.z
        period, band = PLAZA
        # Mosaic bands frame the slabs where the square is wide; narrow pedestrian areas keep plain slabs.
        slabs, narrow, _ = split(polys, mesh.corners(data["wide"]))
        bands = []
        for u in (x * c + z * s, z * c - x * s):
            # The band nearest each triangle, cut by two straight lines, so the cuts are exact.
            k = np.round(u.mean(axis=1) / period)[:, None] * period
            inside, outside_low, _ = split(slabs, u - (k - band / 2))
            framed, outside_high, _ = split(inside, (k + band / 2) - u)
            bands.append(framed)
            slabs = concat(outside_low, outside_high)
        slabs = concat(slabs, narrow)
        return [(material, slabs, angle), ("ground/mosaic", concat(*bands), None)]
    return [(material, polys, None)]


def select(polys, mask):
    return tuple(v[mask] for v in polys)


def along(mesh, field):
    """Per mesh triangle, the direction of the field's level lines, in 5° steps: uvs turned this way lay slabs and
    setts along the curb, and triangles along one straight curb share it exactly, so the pattern runs on."""
    gx, gz = gradient(mesh, np.arange(mesh.count), mesh.corners(field))
    step = math.radians(5)
    return np.round(np.arctan2(gx, -gz) % math.pi / step) * step


# Meshes.

def fan(polys):
    """Triangles of convex polygons as (triangle index, barycentric corners (n, 3, 3))."""
    tri, bary, count = polys
    out_t, out_b = [], []
    for k in range(1, bary.shape[1] - 1):
        keep = count > k + 1
        out_t.append(tri[keep])
        out_b.append(np.stack([bary[keep, 0], bary[keep, k], bary[keep, k + 1]], axis=1))
    if not out_t:
        return tri[:0], np.zeros((0, 3, 3))
    return np.concatenate(out_t), np.concatenate(out_b)


def interpolate(tri, bary, corner):
    """Values at barycentric points: corner is (t, 3) or (t, 3, k)."""
    c = corner[tri]
    return np.einsum("npk,nk...->np...", bary, c)


def terrain(mesh, polys, lowered, colored, angle=None):
    """Positions, normals, uvs, and colours of a layer's pieces, as unindexed triangles; uvs are metres in x and −z,
    turned by angle (per mesh triangle) where given."""
    tri, bary = fan(polys)
    if not len(tri):
        return None
    data = mesh.ground.data
    x = interpolate(tri, bary, mesh.x)
    z = interpolate(tri, bary, mesh.z)
    y = interpolate(tri, bary, mesh.corners(data["base"]))
    if lowered:
        y = y - interpolate(tri, bary, mesh.corners(data["depth"]))
    normal = interpolate(tri, bary, mesh.corners(data["normals"]))
    positions = np.stack([x, y, z], axis=-1)
    colors = None
    if colored:
        # Subtle: the textures already carry their own variation, and dark shades read black in the shade.
        shade = 0.94 + 0.06 * noise(x, z, 7.0) * (0.7 + 0.3 * noise(x, z, 1.3, 7))
        if lowered:
            # Grime in the gutter along curbs.
            edge = interpolate(tri, bary, mesh.corners(data["carriageway"]))
            near = np.clip(1 - edge / 0.6, 0, 1) * np.clip(interpolate(tri, bary, mesh.corners(data["depth"]))
                                                          / roads.CURB, 0, 1)
            shade = shade * (1 - 0.25 * near) * wheel_tracks(edge)
        colors = np.concatenate([np.repeat(shade[..., None], 3, axis=-1), np.ones(shade.shape + (1,))], axis=-1)
    uvs = np.stack([x, -z], axis=-1)
    if angle is not None:
        c, s = np.cos(angle[tri])[:, None], np.sin(angle[tri])[:, None]
        uvs = np.stack([x * c + z * s, x * s - z * c], axis=-1)  # u along the angle, v across it.
    return orient(positions, normal, uvs, colors)


def wheel_tracks(edge):
    """Shade of asphalt by distance from the carriageway's edge: tyres darken two tracks per 3.25 m lane, and the
    strip between them stays lighter. Kept broad and soft, since it is sampled on the 0.5 m ground grid."""
    s = np.clip(edge, 0, None) % roads.LANE
    tracks = np.exp(-((s - 0.85) / 0.45) ** 2) + np.exp(-((s - 2.4) / 0.45) ** 2)
    middle = np.exp(-((s - 1.625) / 0.4) ** 2)
    fade = np.clip(edge / 0.5, 0, 1) * np.clip((BAND - 0.5 - edge) / 0.5, 0, 1)  # The field is clamped at BAND.
    return 1 - fade * (0.08 * tracks - 0.05 * middle)


def orient(positions, normals, uvs, colors=None):
    """Flatten (n, 3, …) triangles, flipping those wound clockwise against their normals, dropping slivers."""
    a, b, c = positions[:, 0], positions[:, 1], positions[:, 2]
    face = np.cross(b - a, c - a)
    area = np.linalg.norm(face, axis=1)
    facing = np.einsum("ij,ij->i", face, normals.mean(axis=1))
    keep = area > 1e-7
    flip = facing < 0
    order = np.where(flip[:, None], [0, 2, 1], [0, 1, 2])
    pick = lambda v: None if v is None else np.take_along_axis(v, order[..., None], 1)[keep].reshape(-1, v.shape[-1])
    return pick(positions), pick(normals), pick(uvs), pick(colors)


def noise(x, z, scale, seed=0):
    """Smooth value noise in 0–1 with features of about scale metres."""
    gx, gz = np.asarray(x) / scale, np.asarray(z) / scale
    ix, iz = np.floor(gx).astype(np.int64), np.floor(gz).astype(np.int64)
    fx, fz = gx - ix, gz - iz
    fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)

    def hash_(i, j):
        h = (i * 374761393 + j * 668265263 + seed * 982451653) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0

    top = hash_(ix, iz) * (1 - fx) + hash_(ix + 1, iz) * fx
    bottom = hash_(ix, iz + 1) * (1 - fx) + hash_(ix + 1, iz + 1) * fx
    return top * (1 - fz) + bottom * fz


def curb_faces(mesh, cuts):
    """Vertical granite faces from roadway level up to the sidewalk along the roadway's edges, facing the road."""
    data = mesh.ground.data
    tri = np.concatenate([segments[0] for segments, _, _ in cuts])
    ends = np.concatenate([segments[1] for segments, _, _ in cuts])
    # Each set's field gradient, times its sign, points into the road.
    gx, gz = (np.concatenate(values) for values in zip(*[
        (sign * gx_, sign * gz_) for (t, _), corner, sign in cuts
        for gx_, gz_ in [gradient(mesh, t, corner[t])]]))
    x = interpolate(tri, ends, mesh.x)
    z = interpolate(tri, ends, mesh.z)
    top = interpolate(tri, ends, mesh.corners(data["base"]))
    depth = interpolate(tri, ends, mesh.corners(data["depth"]))
    keep = (depth.max(axis=1) > 0.01) & (np.hypot(x[:, 1] - x[:, 0], z[:, 1] - z[:, 0]) > 1e-4)
    if not keep.any():
        return None
    x, z, top, depth, gx, gz = x[keep], z[keep], top[keep], depth[keep], gx[keep], gz[keep]
    along = np.stack([x[:, 1] - x[:, 0], z[:, 1] - z[:, 0]], axis=1)
    along /= np.linalg.norm(along, axis=1, keepdims=True)
    out = np.stack([-along[:, 1], along[:, 0]], axis=1)
    out *= np.where(out[:, 0] * gx + out[:, 1] * gz < 0, -1, 1)[:, None]
    bottom = top - depth
    p = lambda i, y: np.stack([x[:, i], y[:, i], z[:, i]], axis=-1)
    quads = np.stack([p(0, bottom), p(1, bottom), p(1, top), p(0, top)], axis=1)
    n = np.repeat(np.column_stack([out[:, 0], np.zeros(len(out)), out[:, 1]])[:, None], 4, axis=1)
    length = np.hypot(x[:, 1] - x[:, 0], z[:, 1] - z[:, 0])
    # u along the curb from world position so the stone pattern runs on across pieces; v up the face.
    u0 = x[:, 0] * along[:, 0] + z[:, 0] * along[:, 1]
    uv = np.stack([np.stack([u0, u0 + length, u0 + length, u0], 1), np.stack([bottom[:, 0], bottom[:, 1],
                                                                              top[:, 1], top[:, 0]], 1)], -1)
    colors = np.ones(quads.shape[:2] + (4,))
    return orient(*triangles(quads, n, uv, colors))


def gradient(mesh, tri, corner):
    """(dx, dz) gradient of a field linear on each triangle from its corner values."""
    x, z = mesh.x[tri], mesh.z[tri]
    e1x, e1z, e2x, e2z = x[:, 1] - x[:, 0], z[:, 1] - z[:, 0], x[:, 2] - x[:, 0], z[:, 2] - z[:, 0]
    d1, d2 = corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
    det = e1x * e2z - e2x * e1z
    return (d1 * e2z - d2 * e1z) / det, (e1x * d2 - e2x * d1) / det


def triangles(quads, normals, uvs, colors=None):
    """Two triangles per quad (n, 4, …) as (2n, 3, …) arrays."""
    split_ = lambda v: None if v is None else np.concatenate([v[:, [0, 1, 2]], v[:, [0, 2, 3]]])
    return split_(quads), split_(normals), split_(uvs), split_(colors)


def decals(quads, uvs, rect):
    """Prebuilt paint triangles whose centre lies in rect."""
    x0, z0, x1, z1 = rect
    centre = quads.mean(axis=1)
    keep = (centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 2] >= z0) & (centre[:, 2] < z1)
    if not keep.any():
        return None
    q = quads[keep]
    n = np.zeros(q.shape)
    n[..., 1] = 1
    return orient(*triangles(q, n, uvs[keep], np.ones(q.shape[:2] + (4,))))


def water_plane(mesh, polys):
    """A flat water surface a little past each water outline, so its edge tucks under the bank."""
    data = mesh.ground.data
    indices = [i for i, layer in enumerate(data["layers"]) if layer[0] == "water"]
    if not indices:
        return None
    f = mesh.corners(np.max(data["fields"][indices], axis=0)) + SHORE
    inside, _, _ = split(polys, f)
    tri, bary = fan(inside)
    if not len(tri):
        return None
    level = mesh.corners(np.nan_to_num(data["level"], nan=-1e3)).max(axis=1)
    x, z = interpolate(tri, bary, mesh.x), interpolate(tri, bary, mesh.z)
    y = np.repeat(level[tri][:, None], 3, axis=1)
    keep = level[tri] > -1e3
    positions = np.stack([x, y, z], axis=-1)[keep]
    n = np.zeros(positions.shape)
    n[..., 1] = 1
    return orient(positions, n, np.stack([x, -z], axis=-1)[keep])


def add(parts, material, chunk):
    if chunk is not None and len(chunk[0]):
        parts.setdefault(material, []).append(chunk)


def merge(material, chunks):
    """One indexed Part from unindexed triangle lists, welding vertices that share position and normal."""
    positions = np.concatenate([c[0] for c in chunks])
    normals_ = np.concatenate([c[1] for c in chunks])
    uvs = np.concatenate([c[2] for c in chunks])
    colors = None
    if all(c[3] is not None for c in chunks):
        colors = np.concatenate([c[3] for c in chunks])
    if not len(positions):
        return None
    key = np.column_stack([np.round(positions * 1000), np.round(normals_ * 100)]).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pick = lambda v: None if v is None else v[first].astype(np.float32)
    return Part(material, pick(positions), pick(normals_), pick(uvs), inverse.ravel().astype(np.uint32), pick(colors))


# Paint and curb colliders, prepared once.

def sampler(ground, values):
    return Raster(values, ground.height.x0, ground.height.z0, GROUND_CELL)


def paint(ground, sources, crossings):
    """Paint quads draped above the road and split into pieces of about a metre: {material: (quads, uvs)}."""
    lines = [s for s in sources if s.group == "road" and s.line is not None and s.curbed]
    field, junction, _ = carriageway_tests(ground, lines, crossings)
    painted = roads.markings(lines, crossings, field, junction)
    return {material: drape(ground, quads, uvs, True, LIFT[material], 1.0) for material, (quads, uvs) in painted.items()}


def carriageway_tests(ground, lines, crossings):
    """Functions of (n, 2) points: signed distance into the carriageway, whether points lie in a junction or on a
    crossing, and whether the carriageway there has a raised curb."""
    data = ground.data
    road = sampler(ground, data["carriageway"])
    # Ways of one street (same name or ref) count once, so lane lines run on where the street's ways join.
    streets = {}
    for s in lines:
        key = s.tags.get("name") or s.tags.get("ref") or s.id
        f = streets.setdefault(key, np.full(ground.height.shape, -BAND, np.float32))
        line_field(ground.height, s.line, s.half + 1.0, f)
    count = sum((f > 0).astype(np.int32) for f in streets.values()) if streets else np.zeros(ground.height.shape)
    blocked = np.full(ground.height.shape, -BAND, np.float32)
    for crossing in crossings:
        line_field(ground.height, crossing.line, crossing.width / 2 + 1.0, blocked)
    junction = sampler(ground, ((count >= 2) | (blocked > 0)).astype(np.float32))
    depth = sampler(ground, data["depth"])
    return (lambda p: road.sample(p[:, 0], p[:, 1]), lambda p: junction.sample(p[:, 0], p[:, 1]) > 0.01,
            lambda p: depth.sample(p[:, 0], p[:, 1]) > 0.5 * roads.CURB)


def drape(ground, quads, uvs, lowered, lift, size):
    """Flat quads (n, 4, 2) as pieces of at most size metres laid lift metres above the ground: (positions (m, 4, 3),
    uvs (m, 4, 2)); lowered quads lie on the roadway, the rest on the sidewalk level."""
    if not len(quads):
        return np.zeros((0, 4, 3), np.float32), np.zeros((0, 4, 2), np.float32)
    quads, uvs = tessellate(np.asarray(quads, float), np.asarray(uvs, float), size)
    data = ground.data
    x, z = quads[..., 0], quads[..., 1]
    y = sampler(ground, data["base"]).sample(x, z) + lift
    if lowered:
        y = y - sampler(ground, data["depth"]).sample(x, z)
    return np.stack([x, y, z], axis=-1).astype(np.float32), uvs.astype(np.float32)


def tessellate(quads, uvs, size):
    """Quads split along both directions into pieces of at most size metres, uvs interpolated."""
    quads, uvs = subdivide(quads, uvs, size)
    # Turning the corner order by one makes the across direction the along one for subdivide.
    turn = lambda v, k: np.roll(v, k, axis=1)
    quads, uvs = subdivide(turn(quads, -1), turn(uvs, -1), size)
    return turn(quads, 1), turn(uvs, 1)


def wear(ground, sources, crossings):
    """Decals prepared once: road wear from roads.wear, and grime along building bases."""
    lines = [s for s in sources if s.group == "road" and s.line is not None and s.curbed]
    field, junction, curbed = carriageway_tests(ground, lines, crossings)
    result = {material: drape(ground, quads, uvs, True, LIFT[material], 1.5)
              for material, (quads, uvs) in roads.wear(lines, field, junction, curbed).items()}
    quads, uvs = grime(ground, [s for s in sources if s.group == "core"])
    if len(quads):
        result["decal/grime"] = drape(ground, quads, uvs, False, LIFT["decal/grime"], 3.0)
    return result


def grime(ground, cores, inner=0.25, outer=0.75, piece=2.5, repeat=4.0):
    """Strips along building outlines, from inner metres under the wall to outer metres out, in pieces of about
    piece metres, where they lie on walked ground: (n, 4, 2) quads, a–b across from the wall outwards, and their
    uvs, u along the wall in repeats of repeat metres and v from 0 at the wall to 1 outside."""
    data = ground.data
    building = sampler(ground, np.max([f for f, layer in zip(data["fields"], data["layers"]) if layer[0] == "core"],
                                      axis=0) + RIM)
    road = sampler(ground, data["road"])
    water = sampler(ground, union(data["fields"], data["layers"], "water"))
    quads, uvs = [], []
    for source in cores:
        for ring in source.rings:
            a, b = ring, np.roll(ring, -1, axis=0)
            for p, q in zip(a, b):
                length = float(np.linalg.norm(q - p))
                if length < 0.5:
                    continue
                along = (q - p) / length
                normal = np.array([-along[1], along[0]])
                mid = (p + q) / 2
                # The outward side is where the building field falls.
                if building.sample(*(mid + normal * 0.4)) > building.sample(*(mid - normal * 0.4)):
                    normal = -normal
                n = max(1, round(length / piece))
                for k in range(n):
                    p0, p1 = p + along * length * k / n, p + along * length * (k + 1) / n
                    quads.append([p0 - normal * inner, p0 + normal * outer, p1 + normal * outer, p1 - normal * inner])
                    u0, u1 = length * k / n / repeat, length * (k + 1) / n / repeat
                    uvs.append([[u0, 0.0], [u0, 1.0], [u1, 1.0], [u1, 0.0]])
    quads, uvs = np.array(quads).reshape(-1, 4, 2), np.array(uvs).reshape(-1, 4, 2)
    if not len(quads):
        return quads, uvs
    # Outer corners must lie on walked ground: outside every building, off roads and water.
    outside = quads[:, [1, 2]].reshape(-1, 2)
    x, z = outside[:, 0], outside[:, 1]
    ok = (building.sample(x, z) < -0.05) & (road.sample(x, z) < -0.2) & (water.sample(x, z) < -0.2)
    ok = ok.reshape(-1, 2).all(axis=1)
    inside = quads[:, [0, 3]].reshape(-1, 2)
    ok &= (road.sample(inside[:, 0], inside[:, 1]) < -0.2).reshape(-1, 2).all(axis=1)
    return quads[ok], uvs[ok]


def late(ground):
    """Decals that need scatter's results, made on first use: tree pits and fallen leaves under street trees on
    paving, from ground.data["trees"], and oil stains under parked cars, from ground.data["parked"]."""
    data = ground.data
    trees, parked = data.get("trees", []), data.get("parked", [])
    key = (len(trees), len(parked))
    if data.get("late", (None, None))[0] != key:
        data["late"] = key, details(ground, trees, parked)
    return data["late"][1]


def details(ground, trees, parked):
    data = ground.data
    road, street = sampler(ground, data["road"]), sampler(ground, data["zone"])
    building = sampler(ground, np.max([f for f, layer in zip(data["fields"], data["layers"]) if layer[0] == "core"],
                                      axis=0) + RIM)
    codes = ground.surface
    paved = lambda p: np.isin(codes.values[codes.index(p[:, 0], p[:, 1])], [SURFACES.index("paving")])
    clear = lambda p: (road.sample(p[:, 0], p[:, 1]) < -0.15) & (building.sample(p[:, 0], p[:, 1]) < -0.1)
    found = {"ground/tree-grate": [], "decal/leaves": [], "decal/oil-stain": []}
    gradient = [sampler(ground, g) for g in np.gradient(data["road"], GROUND_CELL)[::-1]]  # d/dx, d/dz
    for x, z, crown in trees:
        rng = np.random.default_rng(zlib.crc32(f"pit {x:.1f} {z:.1f}".encode()))
        centre = np.array([x, z])
        if not paved(centre[None])[0]:
            continue  # Trees in lawns or mapped soil pits need no pit.
        # Pits line up with the curb: their sides run along the roadway field's level lines.
        gx, gz = (float(g.sample(x, z)) for g in gradient)
        along = np.array([-gz, gx]) / max(math.hypot(gx, gz), 1e-9) if math.hypot(gx, gz) > 0.2 else np.array([1.0, 0])
        if street.sample(x, z) > 0:
            size = 1.8 if rng.random() < 0.7 else 1.4
            pit = roads.rectangle(centre, along, size, size)
            if clear(pit).all() and paved(pit).all():
                found["ground/tree-grate"].append(pit)
        for _ in range(rng.integers(1, 4)):
            spot = centre + rng.normal(0, 0.45 * max(crown, 1.5), 2)
            size = rng.uniform(1.4, 2.6)
            leaves = roads.rectangle(spot, roads.rotate(along, rng.uniform(0, 2 * math.pi)), size, size)
            if clear(leaves).all() and paved(leaves).all():
                found["decal/leaves"].append(leaves)
    lanes = sampler(ground, data["carriageway"])
    for x, z, yaw, length in parked:
        rng = np.random.default_rng(zlib.crc32(f"oil {x:.1f} {z:.1f}".encode()))
        if rng.random() < 0.45:
            front = np.array([math.sin(yaw), math.cos(yaw)])
            spot = np.array([x, z]) + front * length * rng.uniform(0.1, 0.3)
            stain = roads.rectangle(spot, roads.rotate(front, rng.uniform(0, math.pi)), *rng.uniform(0.5, 1.2, 2))
            if (lanes.sample(stain[:, 0], stain[:, 1]) > 0.02).all():
                found["decal/oil-stain"].append(stain)
    result = {}
    for material, quads in found.items():
        if quads:
            quads = np.array(quads)
            lowered = material == "decal/oil-stain"
            result[material] = drape(ground, quads, np.broadcast_to(roads.DECAL_UVS, quads.shape), lowered,
                                     LIFT[material], 1.0)
    return result


def subdivide(quads, uvs, size):
    """Split quads (corners a, b, c, d; a–b across, b–c along) along their length into pieces of at most size."""
    length = np.linalg.norm(quads[:, 2] - quads[:, 1], axis=1)
    steps = np.maximum(1, np.ceil(length / size).astype(int))
    out_q, out_uv = [], []
    for n in np.unique(steps):
        q, w = quads[steps == n], uvs[steps == n]
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            lerp = lambda v, a, b, t: v[:, a] + (v[:, b] - v[:, a]) * t
            out_q.append(np.stack([lerp(q, 0, 3, t0), lerp(q, 1, 2, t0), lerp(q, 1, 2, t1), lerp(q, 0, 3, t1)], 1))
            out_uv.append(np.stack([lerp(w, 0, 3, t0), lerp(w, 1, 2, t0), lerp(w, 1, 2, t1), lerp(w, 0, 3, t1)], 1))
    return np.concatenate(out_q), np.concatenate(out_uv)


def curb_boxes(ground, tolerance=0.05, longest=4.0, deep=0.3):
    """Convex boxes along the raised curbs: (boxes (n, 8, 3), centres (n, 2)).

    The roadway outline comes from the same cuts as the curb faces, chained and simplified; each box spans from
    below the road to the sidewalk top, deep metres into the sidewalk, so the player steps up instead of snagging.
    """
    data = ground.data
    mesh = Cells(ground, None, 1)
    lowered = mesh.corners(ground.height.values < data["base"] - 0.01)
    mesh.keep(lowered.any(axis=1) & ~lowered.all(axis=1) & (mesh.corners(data["depth"]).max(axis=1) > 0.06))
    _, cuts = partition(mesh, mesh.polygons(), curbs=False)
    tri = np.concatenate([segments[0] for segments, _, _ in cuts])
    ends = np.concatenate([segments[1] for segments, _, _ in cuts])
    segments = np.stack([interpolate(tri, ends, mesh.x), interpolate(tri, ends, mesh.z)], axis=-1)
    base, depth = sampler(ground, data["base"]), sampler(ground, data["depth"])
    road = sampler(ground, (ground.height.values < data["base"] - 0.01).astype(np.float32))
    boxes = []
    for chain in chains(segments):
        points = simplify(chain, tolerance)
        for p, q in zip(points[:-1], points[1:]):
            n = max(1, math.ceil(np.linalg.norm(q - p) / longest))
            for k in range(n):
                box_ = curb_box(p + (q - p) * k / n, p + (q - p) * (k + 1) / n, base, depth, road, deep)
                if box_ is not None:
                    boxes.append(box_)
    boxes = np.array(boxes).reshape(-1, 8, 3)
    return boxes, boxes[:, :, [0, 2]].mean(axis=1)


def curb_box(p, q, base, depth, road, deep):
    along = q - p
    length = np.linalg.norm(along)
    d = depth.sample(np.array([p[0], q[0]]), np.array([p[1], q[1]]))
    if length < 0.2 or d.max() < 0.06:
        return None
    # side points into the sidewalk, away from the road.
    side = np.array([-along[1], along[0]]) / length
    middle = (p + q) / 2
    if road.sample(*(middle + side * 0.3)) > road.sample(*(middle - side * 0.3)):
        side = -side
    corners = []
    for point, drop in zip((p, q), d):
        top = float(base.sample(*point))
        for offset in (0.0, deep):
            x, z = point + side * offset
            corners += [(x, top - drop - 0.15, z), (x, top, z)]
    return np.array(corners)


def chains(segments):
    """Polylines joining segments (n, 2, 2) that share endpoints, to a millimetre."""
    keys = [tuple(k) for k in np.round(segments.reshape(-1, 2) * 1000).astype(np.int64)]
    ends = {}
    for i, key in enumerate(keys):
        ends.setdefault(key, []).append(i)
    used = np.zeros(len(segments), bool)
    result = []
    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        halves = []
        for end in (2 * start + 1, 2 * start):
            line = []
            while True:
                following = [j for j in ends[keys[end]] if not used[j // 2]]
                if not following:
                    break
                used[following[0] // 2] = True
                end = following[0] ^ 1  # The far end of the joined segment.
                line.append(segments[end // 2, end % 2])
            halves.append(line)
        result.append(np.array(halves[1][::-1] + [segments[start, 0], segments[start, 1]] + halves[0]))
    return result


def simplify(points, tolerance):
    """Douglas–Peucker simplification of a polyline."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    d = b - a
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        dist = np.linalg.norm(points - a, axis=1)
    else:
        dist = np.abs(d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0])) / norm
    i = int(np.argmax(dist))
    if dist[i] <= tolerance:
        return np.array([a, b])
    return np.concatenate([simplify(points[:i + 1], tolerance)[:-1], simplify(points[i:], tolerance)])
