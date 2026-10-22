"""Sculptures (statues, quadrigas, figures on monuments) remeshed from the reference mesh's shape.

The capture's mesh is a draped skin: right in shape at about the metre scale, often with thin open sheets (wings)
and noise. Only its triangles' positions are used, never its textures. The skin above a sculpture's support is
voxelised, thickened into a watertight shell, filled, blurred, and turned back into a smooth closed surface with
surface nets; Taubin smoothing removes the voxel terracing without shrinking it. Coarser voxels give the LODs.

All in numpy: no Blender needed.
"""

import math

import numpy as np

from . import roofs

BUDGET = 20000  # Triangles at LOD0.
VOXEL = 0.12  # Metres per voxel at LOD0 when the budget allows.
CORNERS = [(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if sum(abs(x - y) for x, y in zip(CORNERS[a], CORNERS[b])) == 1]


def select(reference, ring, y_min):
    """Reference triangles (as an (n, 3, 3) array) whose centroids lie inside ring and above y_min."""
    p, t = reference.positions, reference.triangles
    if not len(t):
        return np.zeros((0, 3, 3))
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    c = p[t].mean(axis=1)
    keep = (c[:, 1] > y_min) & (c[:, 0] >= lo[0]) & (c[:, 0] <= hi[0]) & (c[:, 2] >= lo[1]) & (c[:, 2] <= hi[1])
    if keep.any():
        idx = np.nonzero(keep)[0]
        keep[idx] = roofs.inside([ring], c[idx, 0], c[idx, 2])
    tris = p[t[keep]].astype(np.float64)
    # The support's own roof shows through round the sculpture's feet: flat skin just above y_min is dropped.
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flat = np.abs(n[:, 1]) > 0.85 * np.linalg.norm(n, axis=1)
    low = tris[:, :, 1].mean(axis=1) < y_min + 0.5
    return tris[~(flat & low)]


def level(reference, ring, floor):
    """Height of the support's top as the mesh shows it: the most common measured height inside ring above floor
    (the bare top round a sculpture outnumbers any one height of the sculpture), or None."""
    from .buildings import cells

    _, _, y = cells(reference.top, [ring], 0.0)
    y = y[np.isfinite(y) & (y > floor)]
    if len(y) < 4:
        return None
    bins = np.floor(y / 0.1).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    # Smooth the histogram over three bins so a tilted top still makes one peak.
    smooth = counts + np.r_[counts[1:], 0] * (np.r_[values[1:], 0] == values + 1) \
        + np.r_[0, counts[:-1]] * (np.r_[0, values[:-1]] == values - 1)
    best = values[np.argmax(smooth)]
    near = y[np.abs(bins - best) <= 1]
    return float(np.median(near))


def solid(tris, origin, h, shape, floor=None):
    """Occupancy grid of the volume the triangles enclose: the skin is rasterised, thickened by a voxel so it
    holds water, and everything the outside can't reach is solid. The grid's bottom face counts as closed, since
    the sculpture stands on it. floor: rows above which a column's lowest solid voxel is extended down to row 0,
    so feet and hooves meet their support."""
    skin = np.zeros(shape, bool)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)],
                     axis=0)
    steps = np.clip(np.ceil(longest / (0.5 * h)).astype(int), 1, 200)
    for n in np.unique(steps):
        sel = steps == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        ok = i + j <= n
        w1, w2 = i[ok] / n, j[ok] / n
        w0 = 1 - w1 - w2
        pts = (a[sel, None] * w0[None, :, None] + b[sel, None] * w1[None, :, None] + c[sel, None] * w2[None, :, None])
        idx = np.floor((pts.reshape(-1, 3) - origin) / h).astype(int)
        good = ((idx >= 0) & (idx < shape)).all(axis=1)
        idx = idx[good]
        skin[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    shell = dilate(skin, 1)
    outside = np.zeros(shape, bool)
    for axis in (0, 2):
        for end in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = end
            outside[tuple(sl)] = ~shell[tuple(sl)]
    outside[:, -1, :] = ~shell[:, -1, :]
    outside = flood(outside, ~shell)
    # Undo the thickening: the inside shrinks back by a voxel, the skin itself stays, so thin sheets survive.
    filled = ~dilate(outside, 1) | skin
    if floor is not None and floor > 0:
        # Columns whose lowest solid voxel is within floor rows of the bottom reach down to it.
        any_solid = filled.any(axis=1)
        lowest = np.where(any_solid, filled.argmax(axis=1), shape[1])
        reach = any_solid & (lowest <= floor)
        rows = np.arange(shape[1])[None, :, None]
        filled |= reach[:, None, :] & (rows <= lowest[:, None, :])
    return filled


def dilate(mask, r):
    """mask grown by r voxels in all 26 directions (a cube, done one axis at a time)."""
    out = mask.copy()
    for _ in range(r):
        for axis in range(3):
            grown = out.copy()
            grown[(slice(None),) * axis + (slice(1, None),)] |= out[(slice(None),) * axis + (slice(None, -1),)]
            grown[(slice(None),) * axis + (slice(None, -1),)] |= out[(slice(None),) * axis + (slice(1, None),)]
            out = grown
    return out


def _runs(free, axis):
    """Run keys along an axis: voxels share a key when a straight line of free voxels joins them."""
    f = np.moveaxis(free, axis, -1)
    n = f.shape[-1]
    lines = np.arange(int(np.prod(f.shape[:-1]))).reshape(f.shape[:-1])[..., None]
    key = lines * (n + 1) + np.cumsum(~f, axis=-1)
    return np.moveaxis(key, -1, axis)


def flood(seed, free):
    """seed grown through free voxels (6-connected) until it stops. Each step fills whole straight runs of free
    voxels along one axis, so it converges in a few sweeps."""
    region = seed & free
    keys = [_runs(free, axis) for axis in range(3)]
    size = max(int(k.max()) for k in keys) + 1
    stable = 0
    while stable < 3:
        for key in keys:
            hit = np.zeros(size, bool)
            hit[key[region]] = True
            grown = free & hit[key]
            stable = stable + 1 if grown.sum() == region.sum() else 0
            region = grown
            if stable >= 3:
                break
    return region


def components(mask, keep=0.03):
    """mask without the pieces smaller than keep times the largest (noise above the support). Labels spread as
    the largest label along straight runs, one axis at a time, until nothing changes."""
    if not mask.any():
        return mask
    keys = [_runs(mask, axis) for axis in range(3)]
    size = max(int(k.max()) for k in keys) + 1
    labels = np.where(mask, np.arange(mask.size).reshape(mask.shape) + 1, 0)
    stable = 0
    while stable < 3:
        for key in keys:
            best = np.zeros(size, np.int64)
            np.maximum.at(best, key[mask], labels[mask])
            spread = np.where(mask, best[key], 0)
            stable = stable + 1 if np.array_equal(spread, labels) else 0
            labels = spread
            if stable >= 3:
                break
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    big = ids[counts >= keep * counts.max()]
    return np.isin(labels, big)


def blur(field, passes=1):
    """Separable [1 2 1] / 4 smoothing along every axis, passes times."""
    f = field.astype(np.float64)
    for _ in range(passes):
        for axis in range(3):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            g = np.pad(f, pad, mode="edge")
            lo = (slice(None),) * axis + (slice(0, -2),)
            mid = (slice(None),) * axis + (slice(1, -1),)
            hi = (slice(None),) * axis + (slice(2, None),)
            f = 0.25 * g[lo] + 0.5 * g[mid] + 0.25 * g[hi]
    return f


def surface_nets(field, level, origin, h):
    """Closed surface where field crosses level: one vertex per cell the surface passes through, placed at the mean
    of its edge crossings, and a quad across every grid edge that changes sign. field must be below level on its
    border. Returns (positions, triangles)."""
    f = np.asarray(field, np.float64)
    inside = f > level
    nx, ny, nz = f.shape
    code = np.zeros((nx - 1, ny - 1, nz - 1), np.int32)
    for bit, (i, j, k) in enumerate(CORNERS):
        code |= inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k].astype(np.int32) << bit
    active = (code != 0) & (code != 255)
    cells = np.argwhere(active)
    if not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    index = np.full(code.shape, -1, np.int64)
    index[active] = np.arange(len(cells))
    total = np.zeros((len(cells), 3))
    count = np.zeros(len(cells))
    corner = np.array(CORNERS)
    for a, b in EDGES:
        pa, pb = cells + corner[a], cells + corner[b]
        fa, fb = f[pa[:, 0], pa[:, 1], pa[:, 2]], f[pb[:, 0], pb[:, 1], pb[:, 2]]
        hit = (fa > level) != (fb > level)
        t = np.where(hit, (level - fa) / np.where(hit, fb - fa, 1.0), 0.0)
        total += np.where(hit[:, None], pa + t[:, None] * (pb - pa), 0.0)
        count += hit
    verts = origin + (total / count[:, None] + 0.5) * h
    quads = []
    for d in range(3):
        u, w = [a for a in range(3) if a != d]
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[d], hi[d] = slice(0, -1), slice(1, None)
        change = inside[tuple(lo)] != inside[tuple(hi)]
        v = np.argwhere(change)
        if not len(v):
            continue
        sign = inside[v[:, 0], v[:, 1], v[:, 2]]  # Inside at the low end: the surface faces +d.
        eu, ew = np.eye(3, dtype=int)[u], np.eye(3, dtype=int)[w]
        c0 = v - eu - ew
        cs = [c0, c0 + eu, c0 + eu + ew, c0 + ew]
        q = np.stack([index[c[:, 0], c[:, 1], c[:, 2]] for c in cs], axis=1)
        # Order the quad so its normal (by the right-hand rule over u then w) points along +d, then flip.
        if np.cross(eu, ew) @ np.eye(3, dtype=int)[d] < 0:
            q = q[:, ::-1]
        q[~sign] = q[~sign][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads)
    quads = quads[(quads >= 0).all(axis=1)]
    p = verts
    d02 = np.linalg.norm(p[quads[:, 0]] - p[quads[:, 2]], axis=1)
    d13 = np.linalg.norm(p[quads[:, 1]] - p[quads[:, 3]], axis=1)
    split = d02 <= d13
    tris = np.concatenate([quads[split][:, [0, 1, 2]], quads[split][:, [0, 2, 3]],
                           quads[~split][:, [0, 1, 3]], quads[~split][:, [1, 2, 3]]])
    return verts, tris


def taubin(p, tris, passes=6, lam=0.5, mu=-0.53):
    """Smoothing that removes terracing without shrinking the shape."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(p)).astype(np.float64)
    degree = np.maximum(degree, 1)
    p = p.copy()
    for _ in range(passes):
        for factor in (lam, mu):
            total = np.zeros_like(p)
            np.add.at(total, edges[:, 0], p[edges[:, 1]])
            np.add.at(total, edges[:, 1], p[edges[:, 0]])
            p += factor * (total / degree[:, None] - p)
    return p


def vertex_normals(p, tris):
    face = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, tris[:, k], face)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def box_uvs(p, n, tris):
    """Split vertices by the dominant axis of each triangle's normal and project UVs on that plane, so every
    triangle gets an unstretched mapping (the seams are hidden by the metal's busy texture or a triplanar shader)."""
    face = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    axis = np.argmax(np.abs(face), axis=1)
    ps, ns, uvs, ts = [], [], [], []
    base = 0
    for a in range(3):
        sel = tris[axis == a]
        if not len(sel):
            continue
        used, inverse = np.unique(sel.ravel(), return_inverse=True)
        u, v = [(2, 1), (0, 2), (0, 1)][a]
        ps.append(p[used])
        ns.append(n[used])
        uvs.append(np.column_stack([p[used, u], p[used, v]]))
        ts.append(inverse.reshape(-1, 3) + base)
        base += len(used)
    return np.concatenate(ps), np.concatenate(ns), np.concatenate(uvs), np.concatenate(ts)


def remesh(tris, ring, level, top, h, floor=0.5):
    """Closed smooth surface of the skin `tris` above `level` inside ring, with voxels of h metres.
    Returns (positions, normals, uvs, triangles) or None."""
    x0, z0 = ring.min(axis=0) - 2 * h
    x1, z1 = ring.max(axis=0) + 2 * h
    origin = np.array([x0, level, z0])
    shape = (int(math.ceil((x1 - x0) / h)) + 1, int(math.ceil((top - level) / h)) + 3, int(math.ceil((z1 - z0) / h)) + 1)
    if np.prod(shape) > 40_000_000:
        return None
    occupied = solid(tris, origin, h, shape, floor=int(floor / h))
    occupied = components(occupied)
    if occupied.sum() < 8:
        return None
    field = blur(np.pad(occupied, 2).astype(np.float64), 1)
    p, t = surface_nets(field, 0.5, origin - 2 * h, h)
    if not len(t):
        return None
    p = taubin(p, t, passes=4)
    p[:, 1] = np.maximum(p[:, 1], level - 0.05)  # Nothing sinks through the support.
    n = vertex_normals(p, t)
    return box_uvs(p, n, t)


def build(reference, ring, level, budget=BUDGET, voxel=VOXEL, lods=3):
    """Sculpture LOD meshes [(positions, normals, uvs, triangles)…] from the reference skin above level inside ring,
    LOD0 within budget triangles and each further LOD at twice the voxel size; None when there is nothing."""
    tris = select(reference, ring, level + 0.12)
    if len(tris) < 4:
        return None
    top = float(tris[:, :, 1].max()) + 0.3
    out = []
    h = voxel
    for _ in range(4):  # Coarsen until LOD0 fits the budget.
        mesh = remesh(tris, ring, level, top, h)
        if mesh is None:
            return None
        if len(mesh[3]) <= budget:
            break
        h *= math.sqrt(len(mesh[3]) / budget) * 1.05
    out.append(mesh)
    for k in range(1, lods):
        mesh = remesh(tris, ring, level, top, h * (1.7, 2.8)[k - 1] if k <= 2 else h * 2 ** k)
        if mesh is None:
            break
        out.append(mesh)
    while len(out) < lods:
        out.append(out[-1])
    return out, h
