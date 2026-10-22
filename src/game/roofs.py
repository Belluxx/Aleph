"""Footprint geometry and roofs: polygon helpers, roof-shape fitting against the measured surface, and roof meshes.

A roof is y = eave + rise * s(x, z) with s = min over a few planes (0 at the eaves, 1 at the ridge). The minimum of
planes describes flat, skillion, gabled, hipped, pyramidal and mansard roofs alike, fits them by linear least
squares, and meshes exactly: splitting the footprint's triangles along the lines where two planes meet leaves every
piece on a single plane. Domes are rings scaled towards the centre instead.

Rings are (n, 2) [x, z] arrays, counter-clockwise seen from above for outlines and clockwise for holes.
"""

import math
from dataclasses import dataclass, field

import numpy as np

SHAPES = ("flat", "skillion", "gabled", "hipped", "pyramidal", "mansard", "dome")
# OSM roof:shape values mapped onto the shapes we build.
ALIASES = {"round": "dome", "onion": "dome", "half_dome": "dome", "cone": "pyramidal", "gambrel": "gabled",
           "half-hipped": "hipped", "saltbox": "gabled", "quadruple_saltbox": "hipped", "crosspitched": "hipped",
           "side_hipped": "hipped", "side_half-hipped": "hipped", "lean_to": "skillion", "sawtooth": "skillion",
           "butterfly": "gabled", "double_saltbox": "gabled"}
# Extra error, in metres, a shape must save over simpler ones to be chosen: it keeps noisy flat roofs flat.
PENALTY = {"flat": 0.0, "skillion": 0.2, "gabled": 0.15, "hipped": 0.15, "pyramidal": 0.25, "mansard": 0.3,
           "dome": 0.35}
MIN_RISE = 0.8  # Pitched fits rising less than this are flat roofs with clutter.
MIN_SLOPE = 0.15  # Nor are pitched fits flatter than this (about 8.5°): large roofs with a slight fall.
DOME_RINGS = 6


# Polygons.

def area(ring):
    """Signed area, positive for rings counter-clockwise seen from above (z points south)."""
    x, z = np.asarray(ring, float).T
    return -0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(np.roll(x, -1), z))


def orient(a, b, c):
    """Twice the signed area of triangle abc, positive when counter-clockwise seen from above."""
    return -((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def centroid(ring):
    p = np.asarray(ring, float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(axis=0)
    return np.array([((p[:, 0] + q[:, 0]) * cross).sum(), ((p[:, 1] + q[:, 1]) * cross).sum()]) / (6 * a)


def inside(rings, x, z):
    """Even-odd test of points against rings (outline and holes together)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    result = np.zeros(np.broadcast(x, z).shape, bool)
    for ring in rings:
        p = np.asarray(ring, float)
        for (ax, az), (bx, bz) in zip(p, np.roll(p, -1, axis=0)):
            crosses = (az > z) != (bz > z)
            with np.errstate(divide="ignore", invalid="ignore"):
                at = ax + (z - az) * (bx - ax) / (bz - az)
            result ^= crosses & (x < at)
    return result


def distance(rings, x, z):
    """Distance from points to the nearest ring edge."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    best = np.full(np.broadcast(x, z).shape, np.inf)
    for ring in rings:
        p = np.asarray(ring, float)
        for (ax, az), (bx, bz) in zip(p, np.roll(p, -1, axis=0)):
            dx, dz = bx - ax, bz - az
            t = np.clip(((x - ax) * dx + (z - az) * dz) / max(dx * dx + dz * dz, 1e-12), 0, 1)
            best = np.minimum(best, np.hypot(x - ax - t * dx, z - az - t * dz))
    return best


def simplify(ring, tolerance):
    """Douglas–Peucker on a closed ring, split at its two farthest-apart vertices."""
    p = np.asarray(ring, float)
    if len(p) <= 4:
        return p
    i = 0
    j = int(np.argmax(np.hypot(*(p - p[i]).T)))
    keep = np.zeros(len(p), bool)
    keep[[i, j]] = True

    def walk(a, b):
        ids = np.arange(a, b + 1) % len(p) if b >= a else np.r_[np.arange(a, len(p)), np.arange(0, b + 1)]
        stack = [(0, len(ids) - 1)]
        while stack:
            s, e = stack.pop()
            if e - s < 2:
                continue
            a, b = p[ids[s]], p[ids[e]]
            d = b - a
            n = math.hypot(*d)
            q = p[ids[s + 1:e]] - a
            dist = np.abs(q[:, 0] * d[1] - q[:, 1] * d[0]) / n if n > 1e-9 else np.hypot(*q.T)
            k = int(np.argmax(dist))
            if dist[k] > tolerance:
                keep[ids[s + 1 + k]] = True
                stack += [(s, s + 1 + k), (s + 1 + k, e)]

    walk(i, j)
    walk(j, i + len(p))
    return p[keep]


def clean(ring, shortest=0.25):
    """Drop repeated and collinear vertices and edges shorter than shortest."""
    p = [np.asarray(v, float) for v in ring]
    changed = True
    while changed and len(p) > 3:
        changed = False
        for k in range(len(p)):
            a, b, c = p[k - 1], p[k], p[(k + 1) % len(p)]
            if (math.hypot(*(b - a)) < shortest or abs(orient(a, b, c)) < 1e-3 * max(math.hypot(*(c - a)), 1e-9)
                    and np.dot(b - a, c - b) >= 0):
                del p[k]
                changed = True
                break
    return np.array(p)


def square(ring, tolerance=math.radians(8), shortest=1.0):
    """Snap edges within tolerance of the ring's dominant orientation (or 90° from it) onto it, keeping the area;
    edges under shortest keep their direction, so curves stay curves."""
    p = np.asarray(ring, float)
    d = np.roll(p, -1, axis=0) - p
    lengths = np.hypot(*d.T)
    angles = np.arctan2(d[:, 1], d[:, 0])
    # Edge-length-weighted mean of the angles folded into 90°.
    v = (lengths * np.exp(4j * angles)).sum()
    if abs(v) < 1e-9:
        return p
    main = np.angle(v) / 4
    lines = []
    for k in range(len(p)):
        off = (angles[k] - main) % (math.pi / 2)
        off = off - math.pi / 2 if off > math.pi / 4 else off
        a = angles[k] - off if abs(off) < tolerance and lengths[k] >= shortest else angles[k]
        mid = p[k] + d[k] / 2
        lines.append((mid, np.array([math.cos(a), math.sin(a)])))
    out = []
    for k in range(len(p)):
        (m1, u1), (m2, u2) = lines[k - 1], lines[k]
        det = u1[0] * -u2[1] - u1[1] * -u2[0]
        if abs(det) < 1e-6:  # Parallel neighbours: keep the original corner.
            out.append(p[k])
            continue
        t = ((m2 - m1)[0] * -u2[1] - (m2 - m1)[1] * -u2[0]) / det
        q = m1 + t * u1
        out.append(q if math.hypot(*(q - p[k])) < 1.0 else p[k])
    out = np.array(out)
    if abs(area(out) - area(p)) > 0.03 * abs(area(p)) or abs(area(out)) < 1e-6:
        return p
    return out


def offset(ring, distance, limit=3.0):
    """Ring moved outwards by distance (inwards when negative), mitred at corners; miters longer than limit times
    the distance are clamped. Outwards is to the right of travel, so holes move into the hole."""
    p = np.asarray(ring, float)
    d = np.roll(p, -1, axis=0) - p
    d /= np.maximum(np.hypot(*d.T), 1e-12)[:, None]
    n = np.column_stack([-d[:, 1], d[:, 0]])
    a, b = np.roll(n, 1, axis=0), n
    m = a + b
    m /= np.maximum(1 + (a * b).sum(axis=1), 1e-6)[:, None]
    length = np.hypot(*m.T)
    m *= (np.minimum(length, limit) / np.maximum(length, 1e-12))[:, None]
    return p + distance * m


def normals(ring):
    """Unit outward normals of each edge k (from vertex k to k + 1)."""
    p = np.asarray(ring, float)
    d = np.roll(p, -1, axis=0) - p
    d /= np.maximum(np.hypot(*d.T), 1e-12)[:, None]
    return np.column_stack([-d[:, 1], d[:, 0]])


def convex(ring):
    p = np.asarray(ring, float)
    return all(orient(p[k - 1], p[k], p[(k + 1) % len(p)]) >= -1e-9 for k in range(len(p)))


def ccw(ring, clockwise=False):
    """Ring turned counter-clockwise seen from above (clockwise for holes)."""
    ring = np.asarray(ring, float)
    return ring[::-1].copy() if (area(ring) < 0) != clockwise else ring


def triangulate(outer, holes=()):
    """Ear clipping: (points, triangles) with triangles counter-clockwise seen from above; holes are bridged in."""
    ring = [tuple(map(float, v)) for v in ccw(outer)]
    holes = [[tuple(map(float, v)) for v in ccw(hole, True)] for hole in holes]
    hole_edges = [(h[k], h[(k + 1) % len(h)]) for h in holes for k in range(len(h))]
    for hole in sorted(holes, key=lambda h: -max(v[0] for v in h)):
        h = max(range(len(hole)), key=lambda k: hole[k][0])
        hp = hole[h]
        edges = [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))] + hole_edges
        best = None
        for k in sorted(range(len(ring)), key=lambda k: math.dist(ring[k], hp)):
            if all(not _crosses(hp, ring[k], a, b) for a, b in edges):
                best = k
                break
        if best is None:
            continue
        hole_loop = hole[h:] + hole[:h + 1]
        ring = ring[:best + 1] + hole_loop + ring[best:]
    points = np.array(ring)
    index = list(range(len(ring)))
    triangles = []
    guard = 0
    while len(index) > 3 and guard < 10 * len(ring) ** 2:
        guard += 1
        n = len(index)
        found = False
        for k in range(n):
            i, j, l = index[k - 1], index[k], index[(k + 1) % n]
            a, b, c = ring[i], ring[j], ring[l]
            if orient(a, b, c) <= 1e-12:
                continue
            if any(_in_triangle(ring[m], a, b, c) for m in index if m not in (i, j, l)
                   and ring[m] != a and ring[m] != b and ring[m] != c):
                continue
            triangles.append((i, j, l))
            del index[k]
            found = True
            break
        if not found:  # Degenerate leftovers: clip the most convex corner.
            k = max(range(n), key=lambda k: orient(ring[index[k - 1]], ring[index[k]], ring[index[(k + 1) % n]]))
            i, j, l = index[k - 1], index[k], index[(k + 1) % n]
            if orient(ring[i], ring[j], ring[l]) > 1e-12:
                triangles.append((i, j, l))
            del index[k]
    if len(index) == 3 and orient(*(ring[i] for i in index)) > 1e-12:
        triangles.append(tuple(index))
    return points, np.array(triangles, int).reshape(-1, 3)


def _in_triangle(p, a, b, c):
    return orient(a, b, p) >= -1e-12 and orient(b, c, p) >= -1e-12 and orient(c, a, p) >= -1e-12


def _crosses(p, q, a, b):
    """Proper crossing of segments pq and ab, ignoring shared endpoints."""
    if p in (a, b) or q in (a, b):
        return False
    d1, d2 = orient(p, q, a), orient(p, q, b)
    d3, d4 = orient(a, b, p), orient(a, b, q)
    return d1 * d2 < 0 and d3 * d4 < 0


def pieces(points, triangles):
    """Convex pieces of a triangulated polygon (Hertel–Mehlhorn): lists of point indices, counter-clockwise."""
    polys = [list(t) for t in triangles]
    merged = True
    while merged:
        merged = False
        edges = {}
        for k, poly in enumerate(polys):
            for i in range(len(poly)):
                edges.setdefault((poly[i], poly[(i + 1) % len(poly)]), k)
        for (a, b), k in list(edges.items()):
            other = edges.get((b, a))
            if other is None or other == k:
                continue
            p, q = polys[k], polys[other]
            i, j = p.index(a), q.index(b)
            # Walk p from b round to a, then q from a round to b.
            joined = p[i + 1:] + p[:i + 1]
            joined = joined[:-1] + q[j + 1:] + q[:j + 1]
            joined = joined[:-1]
            if len(set(joined)) == len(joined) and convex(points[joined]):
                polys[k] = joined
                polys[other] = None
                polys = [poly for poly in polys if poly is not None]
                merged = True
                break
    return polys


def box(ring):
    """Minimum-area oriented rectangle: (centre, long axis, half length, half width)."""
    p = np.asarray(ring, float)
    best = None
    for (ax, az), (bx, bz) in zip(p, np.roll(p, -1, axis=0)):
        d = np.array([bx - ax, bz - az])
        n = math.hypot(*d)
        if n < 1e-6:
            continue
        u = d / n
        v = np.array([-u[1], u[0]])
        pu, pv = p @ u, p @ v
        a = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or a < best[0] - 1e-9:
            c = u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
            best = (a, c, u, (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2)
    _, c, u, hu, hv = best
    if hv > hu:
        u, hu, hv = np.array([-u[1], u[0]]), hv, hu
    return c, u, max(hu, 1e-3), max(hv, 1e-3)


# Roofs.

@dataclass
class Roof:
    shape: str
    eave: float  # y of the eaves.
    rise: float  # Metres from the eaves to the ridge.
    planes: list = field(default_factory=list)  # [(a, b, c)]: s = min(a x + b z + c); empty for flat roofs.
    centre: tuple = None  # Dome centre (x, z), axis, and half extents.
    axis: tuple = None
    half: tuple = None
    lines: list = None  # Crease lines over the footprint, from creases().

    def s(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        if self.shape == "dome":
            u, v = _local(self, x, z)
            return np.sqrt(np.clip(1 - (u / self.half[0]) ** 2 - (v / self.half[1]) ** 2, 0, 1))
        if not self.planes:
            return np.zeros(np.broadcast(x, z).shape)
        return np.min([a * x + b * z + c for a, b, c in self.planes], axis=0)

    def y(self, x, z):
        return self.eave + self.rise * self.s(x, z)

    @property
    def ridge(self):
        return self.eave + self.rise


def _local(roof, x, z):
    (cx, cz), (ax, az) = roof.centre, roof.axis
    return (x - cx) * ax + (z - cz) * az, -(x - cx) * az + (z - cz) * ax


def planes(shape, frame, direction=None, mansard=2.0):
    """Planes of s for a shape over the oriented box frame = (centre, axis, hu, hv); direction is a downhill unit
    vector for skillions, or "across" to run a gable's ridge along the short side."""
    c, a, hu, hv = frame
    b = np.array([-a[1], a[0]])

    def plane(g, k):  # s = g · (p - c) + k
        return float(g[0]), float(g[1]), float(k - g @ c)

    if shape == "skillion":
        down = np.asarray(direction if direction is not None and not isinstance(direction, str) else b, float)
        h = abs(down @ a) * hu + abs(down @ b) * hv
        return [plane(-down / (2 * h), 0.5)]
    if shape == "gabled":
        if isinstance(direction, str) and direction == "across":
            a, b, hu, hv = b, a, hv, hu
        return [plane(b / hv, 1), plane(-b / hv, 1)]
    if shape == "hipped":
        return [plane(b / hv, 1), plane(-b / hv, 1), plane(a / hv, hu / hv), plane(-a / hv, hu / hv)]
    if shape == "pyramidal":
        return [plane(b / hv, 1), plane(-b / hv, 1), plane(a / hu, 1), plane(-a / hu, 1)]
    if shape == "mansard":
        w = min(mansard, 0.45 * hv)
        return [plane(b / w, hv / w), plane(-b / w, hv / w), plane(a / w, hu / w), plane(-a / w, hu / w),
                (0.0, 0.0, 1.0)]
    return []


def roof(shape, ring, eave, rise, direction=None):
    """A Roof of shape over ring, oriented by its minimum box."""
    shape = ALIASES.get(shape, shape)
    if shape not in SHAPES:
        shape = "flat"
    frame = box(ring)
    if shape == "flat" or rise <= 0:
        return Roof("flat", eave, 0.0)
    if shape == "dome":
        c, a, hu, hv = frame
        return Roof("dome", eave, rise, centre=tuple(c), axis=tuple(a), half=(hu, hv))
    return Roof(shape, eave, rise, planes(shape, frame, direction))


def candidates(ring, direction=None):
    """Roofs worth trying for a footprint, with unit rise (eave and rise get fitted)."""
    frame = box(ring)
    c, a, hu, hv = frame
    b = np.array([-a[1], a[0]])
    found = [Roof("flat", 0, 0)]
    for down in ([direction] if direction is not None and not isinstance(direction, str) else [a, -a, b, -b]):
        found.append(Roof("skillion", 0, 1, planes("skillion", frame, down)))
    for across in [direction] if isinstance(direction, str) else (None, "across"):
        found.append(Roof("gabled", 0, 1, planes("gabled", frame, across)))
    for shape in ("hipped", "pyramidal", "mansard"):
        found.append(Roof(shape, 0, 1, planes(shape, frame)))
    found.append(Roof("dome", 0, 1, centre=tuple(c), axis=tuple(a), half=(hu, hv)))
    return found


def fit(x, z, y, ring, shape=None, direction=None):
    """Best roof for measured surface points (x, z, y) inside ring: (Roof, rms). With shape given, only its eave and
    rise are fitted. Robust to clutter: two passes, the second ignoring points far off the first fit."""
    x, z, y = (np.asarray(v, float) for v in (x, z, y))
    shape = ALIASES.get(shape, shape)
    if shape is not None and shape not in SHAPES:
        shape = None
    options = candidates(ring, direction)
    if shape is not None:
        options = [r for r in options if r.shape == shape] or [Roof("flat", 0, 0)]
    best = None
    for option in options:
        s = option.s(x, z)
        keep = np.ones(len(y), bool)
        for _ in range(2):
            eave, rise = _solve(s[keep], y[keep], option.shape != "flat")
            residual = y - eave - rise * s
            rms = math.sqrt(np.mean(residual[keep] ** 2)) if keep.any() else 0.0
            keep = np.abs(residual) <= max(1.0, 2.5 * rms)
        if option.shape != "flat" and shape is None and rise < max(MIN_RISE, MIN_SLOPE * _run(option, ring)):
            continue
        score = rms + PENALTY[option.shape]
        if best is None or score < best[0]:
            best = (score, Roof(option.shape, eave, rise, option.planes, option.centre, option.axis, option.half))
    result = best[1]
    if result.rise <= 0:
        result = Roof("flat", result.eave, 0.0)
    return result, math.sqrt(np.mean((y - result.y(x, z)) ** 2)) if len(y) else 0.0


def _run(roof, ring):
    """Horizontal distance from eaves to ridge of a unit-rise roof over ring."""
    _, _, hu, hv = box(ring)
    if roof.shape == "skillion":
        return 2 * hv
    if roof.shape == "mansard":
        return min(2.0, 0.45 * hv)
    return hv


def _solve(s, y, pitched):
    if len(y) == 0:
        return 0.0, 0.0
    if not pitched or np.ptp(s) < 1e-6:
        return float(np.median(y)), 0.0
    A = np.column_stack([np.ones_like(s), s])
    (eave, rise), *_ = np.linalg.lstsq(A, y, rcond=None)
    if rise < 0:
        return float(np.median(y)), 0.0
    return float(eave), float(rise)


# Meshes.

def surface(r, outer, holes=()):
    """Roof surface over a footprint: (m, 3, 3) triangles in world coordinates, counter-clockwise from above.

    Every triangle is cut by the full line of every crease, so neighbours share their cut points and the roof has
    no T-junctions; walls take their top breaks from the same lines (profile)."""
    if r.shape == "dome":
        return dome(r, outer)
    points, triangles = triangulate(outer, holes)
    tris = [points[t] for t in triangles]
    for line in creases(r, outer):
        tris = _split(tris, *line)
    out = []
    for t in tris:
        if abs(orient(*t)) < 1e-10:
            continue
        y = _plane_y(r, t)
        out.append(np.column_stack([t[:, 0], y, t[:, 1]]))
    return np.array(out).reshape(-1, 3, 3)


def creases(r, ring):
    """Lines a x + b z + c = 0 along which two of the roof's planes meet as its lowest, somewhere over ring."""
    if r.lines is not None:
        return r.lines
    p = np.asarray(ring, float)
    lo, hi = p.min(axis=0), p.max(axis=0)
    lines = []
    for i in range(len(r.planes)):
        for j in range(i + 1, len(r.planes)):
            a, b, c = np.subtract(r.planes[i], r.planes[j])
            if math.hypot(a, b) < 1e-9:
                continue
            # Points along the line across the footprint's bounding box.
            n = np.array([a, b]) / math.hypot(a, b)
            origin = -c / math.hypot(a, b) * n
            along = np.array([-n[1], n[0]])
            reach = np.linalg.norm(hi - lo) + abs(along @ ((lo + hi) / 2 - origin))
            t = along @ ((lo + hi) / 2 - origin) + np.linspace(-reach, reach, 801)
            x, z = (origin[:, None] + along[:, None] * t).reshape(2, -1)
            values = np.array([pa * x + pb * z + pc for pa, pb, pc in r.planes])
            low = values.min(axis=0)
            on = (values[i] - low < 1e-7) & (values[j] - low < 1e-7) & inside([p], x, z)
            if on.any():
                lines.append((float(a), float(b), float(c)))
    r.lines = lines
    return lines


def _plane_y(r, t):
    """Heights on the plane that is lowest at the centroid, so float noise near a crease can't bend a piece."""
    if not r.planes:
        return np.full(3, r.eave)
    m = t.mean(axis=0)
    k = int(np.argmin([a * m[0] + b * m[1] + c for a, b, c in r.planes]))
    a, b, c = r.planes[k]
    return r.eave + r.rise * (a * t[:, 0] + b * t[:, 1] + c)


def _split(tris, a, b, c):
    out = []
    for t in tris:
        d = t[:, 0] * a + t[:, 1] * b + c
        d = np.where(np.abs(d) < 1e-9, 0, d)
        if (d >= 0).all() or (d <= 0).all():
            out.append(t)
            continue
        # One vertex is alone on its side: cut the two edges leaving it.
        side = np.sign(d)
        k = next(k for k in range(3) if side[k] != 0 and side[(k + 1) % 3] != side[k] and side[(k + 2) % 3] != side[k])
        p, q, s = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
        dp, dq, ds = d[k], d[(k + 1) % 3], d[(k + 2) % 3]
        if dq == 0 or ds == 0:  # The line passes through a vertex: two triangles.
            o, do = (s, ds) if dq == 0 else (q, dq)
            m = p + (o - p) * dp / (dp - do)
            out += [np.array([p, q, m]), np.array([m, q, s])] if dq == 0 else [np.array([p, m, s]), np.array([m, q, s])]
            continue
        m1 = p + (q - p) * dp / (dp - dq)
        m2 = p + (s - p) * dp / (dp - ds)
        out += [np.array([p, m1, m2]), np.array([m1, q, s]), np.array([m1, s, m2])]
    return out


def profile(r, p, q, ring=None):
    """Roof height along a wall from p to q: [(t, y)] with t in [0, 1], broken where the roof's crease lines cross
    it, as surface breaks the roof (ring is the footprint, needed until the roof's creases are known)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if r.shape in ("flat", "dome"):  # Domes start from their outline at the eaves.
        return [(0.0, r.eave), (1.0, r.eave)]
    ts = {0.0, 1.0}
    if r.lines is None and ring is None:  # Unknown footprint: break at every line where two planes meet.
        lines = [tuple(np.subtract(r.planes[i], r.planes[j])) for i in range(len(r.planes))
                 for j in range(i + 1, len(r.planes))]
    else:
        lines = creases(r, ring)
    for a, b, c in lines:
        dp, dq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if abs(dp) > 1e-9 and abs(dq) > 1e-9 and dp * dq < 0:
            ts.add(float(dp / (dp - dq)))
    return [(t, float(r.y(*(p + t * (q - p))))) for t in sorted(ts)]


def dome(r, outer, rings=DOME_RINGS):
    """Dome over a footprint: its outline scaled towards the centre on a quarter-circle profile."""
    c = np.asarray(r.centre)
    p = np.asarray(outer, float)
    out = []
    levels = [(math.cos(k * math.pi / 2 / rings), math.sin(k * math.pi / 2 / rings)) for k in range(rings + 1)]
    for (t0, h0), (t1, h1) in zip(levels, levels[1:]):
        a0, a1 = c + t0 * (p - c), c + t1 * (p - c)
        y0, y1 = r.eave + r.rise * h0, r.eave + r.rise * h1
        for k in range(len(p)):
            j = (k + 1) % len(p)
            A, B = (*a0[k], y0), (*a0[j], y0)
            C, D = (*a1[j], y1), (*a1[k], y1)
            if t1 < 1e-9:
                out.append([A, B, C])
            else:
                out += [[A, B, C], [A, C, D]]
    out = np.array(out, float).reshape(-1, 3, 3)
    return out[:, :, [0, 2, 1]]
