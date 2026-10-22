"""Classical architecture kit for landmarks: columns, mouldings, entablatures, attics, pediments, balustrades, steps,
arches and reliefs as numpy geometry.

Everything is added to a Geo, which groups triangles by material. Surfaces are built as quad grids whose rows run up
a profile and whose columns run along an outline, counter-clockwise seen from above, so normals come from the grid
itself and creases are where one run of a profile ends and the next begins. Profiles are lists of runs, each a smooth
list of (out, y) points walked up the outside: out is metres outwards from an outline (or a radius for lathes).

UVs are in metres with v up, continuous round columns (the circumference is rounded to whole texture repeats) and
along mouldings. Vertex colours carry baked occlusion: flute grooves, undersides and grime near the ground.

detail: 0 is the full model (flutes, triglyph glyphs, guttae, reliefs, leaves), 1 about a third of it (coarser
lathes, no small parts), 2 the massing with the main mouldings only.
"""

import math

import numpy as np

from . import roofs
from .types import Part

TAU = 2 * math.pi
UP = np.array([0.0, 1.0, 0.0])
GRIME = 0.22  # Darkening at the foot of walls, fading over GRIME_HEIGHT metres.
GRIME_HEIGHT = 1.4
UNDERSIDE = 0.28  # Darkening of faces looking down, which the sky never lights.


class Geo:
    """Triangles grouped by material. Triangles are wound to agree with the normals given for their corners, so
    builders only need normals pointing the right way."""

    def __init__(self, ground=None, sizes=None):
        self.groups = {}
        self.ground = ground  # y of the ground, for grime; None for none.
        self.sizes = sizes or {}  # Material texture repeat in metres, for whole repeats round columns.

    def add(self, material, positions, normals, uvs, triangles, colors=None, occlusion=True):
        p = np.asarray(positions, np.float64).reshape(-1, 3)
        if not len(p):
            return
        n = np.asarray(normals, np.float64).reshape(-1, 3)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        t = np.asarray(triangles, np.int64).reshape(-1, 3)
        face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        keep = np.linalg.norm(face, axis=1) > 1e-10
        t, face = t[keep], face[keep]
        flip = (face * n[t].sum(axis=1)).sum(axis=1) < 0
        t[flip] = t[flip][:, ::-1]
        shade = np.ones(len(p)) if colors is None else np.asarray(colors, np.float64).reshape(-1)
        if occlusion:
            shade = shade * (1 - UNDERSIDE * np.clip(-n[:, 1], 0, 1))
            if self.ground is not None:
                shade = shade * (1 - GRIME * np.exp(-np.clip(p[:, 1] - self.ground, 0, None) / GRIME_HEIGHT))
        self.groups.setdefault(material, []).append((p, n, np.asarray(uvs, np.float64).reshape(-1, 2), t, shade))

    def count(self):
        return sum(len(item[3]) for items in self.groups.values() for item in items)

    def size(self, material):
        return self.sizes.get(material, 1.0) or 1.0

    def parts(self):
        out = []
        for material in sorted(self.groups):
            items = self.groups[material]
            base, tris = 0, []
            for p, _, _, t, _ in items:
                tris.append(t + base)
                base += len(p)
            if not base:
                continue
            shade = np.concatenate([i[4] for i in items])
            colors = np.column_stack([shade, shade, shade, np.ones_like(shade)])
            out.append(Part(material, np.concatenate([i[0] for i in items]).astype(np.float32),
                            np.concatenate([i[1] for i in items]).astype(np.float32),
                            np.concatenate([i[2] for i in items]).astype(np.float32),
                            np.concatenate(tris).astype(np.uint32).ravel(), colors.astype(np.float32)))
        return out


# Surfaces.

def grid(geo, material, p, uv, closed=False, shade=None, normals=None):
    """A quad grid p[rows, cols]: rows run up a profile, cols along an outline counter-clockwise seen from above
    (or along a face with its outside on the right), so cross(cols, rows) faces out. closed grids repeat their
    first column at the end."""
    rows, cols = p.shape[:2]
    if normals is None:
        dr = np.gradient(p, axis=0) if rows > 1 else np.zeros_like(p)
        if closed:
            q = p[:, :-1]
            dc = np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)
            dc = np.concatenate([dc, dc[:, :1]], axis=1)
        else:
            dc = np.gradient(p, axis=1)
        normals = np.cross(dc, dr)
    k = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = k[:-1, :-1], k[:-1, 1:], k[1:, 1:], k[1:, :-1]
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    geo.add(material, p, normals, uv, tris, None if shade is None else np.asarray(shade).reshape(-1))


def _arc(points):
    """Cumulative length along a polyline, starting at 0."""
    points = np.asarray(points, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def facets(points):
    """A polyline as runs of one segment each: every corner a crease."""
    return [[points[k], points[k + 1]] for k in range(len(points) - 1)]


def groove(phase, fillet=0.0):
    """Depth (0–1) across one flute at phase 0–1: a round-bottomed channel with fillet flat at the arrises."""
    s = (phase - fillet / 2) / max(1 - fillet, 1e-6)
    inside = (s >= 0) & (s <= 1)
    return np.where(inside, np.sqrt(np.clip(1 - (2 * s - 1) ** 2, 0, 1)), 0.0)


def lathe(geo, material, centre, runs, segments, flutes=0, depth=0.0, fluted=None, fillet=0.0, phase=0.0,
          reference=None):
    """Surface of revolution round the vertical through centre (x, z). runs: [[(r, y)…]] walked up the outside.
    flutes channels `depth` deep are cut into rows whose y lies inside fluted=(y0, y1), fading in and out at the
    ends; segments must then be a multiple of flutes. reference is the radius whose circumference sets u."""
    cx, cz = centre
    th = phase + TAU * np.arange(segments + 1) / segments
    cos, sin = np.cos(th), -np.sin(th)  # Counter-clockwise seen from above: from +x towards -z.
    all_r = [r for run in runs for r, _ in run]
    ref = reference or max(all_r)
    size = geo.size(material)
    repeats = max(1, round(TAU * ref / size)) * size
    u = (th - phase) / TAU * repeats
    if flutes:
        g = groove(((th - phase) * flutes / TAU) % 1.0, fillet)
        g[-1] = g[0]
    for run in runs:
        run = np.asarray(run, float)
        if len(run) < 2:
            continue
        if flutes and fluted is not None:
            # Rows where the flutes fade in and out, so they end in rounded tops however coarse the run is.
            fade = min(0.3, (fluted[1] - fluted[0]) / 4)
            ys = [y for y in (fluted[0] + fade, fluted[1] - fade) if run[0, 1] < y < run[-1, 1]]
            ys = sorted(set(run[:, 1]) | set(ys))
            run = np.column_stack([np.interp(ys, run[:, 1], run[:, 0]), ys])
        r, y = run[:, 0][:, None], run[:, 1][:, None]
        rr = np.repeat(r, segments + 1, axis=1)
        shade = np.ones_like(rr)
        if flutes and fluted is not None:
            y0, y1 = fluted
            fade = min(0.3, (y1 - y0) / 4)
            env = np.clip(np.minimum(y - y0, y1 - y) / max(fade, 1e-6), 0, 1)
            env = env * env * (3 - 2 * env)
            rr = rr - depth * env * g[None, :]
            shade = 1 - 0.3 * env * g[None, :]
        p = np.stack([cx + rr * cos, np.repeat(y, segments + 1, axis=1), cz + rr * sin], -1)
        v = run[0, 1] + _arc(run)
        uv = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), -1)
        normals = None
        if np.any(r[[0, -1]] < 1e-6):
            normals = _lathe_normals(run, cos, sin)
        grid(geo, material, p, uv, closed=True, shade=shade, normals=normals)


def _lathe_normals(run, cos, sin):
    """Profile normals for runs that touch the axis, where the grid's own normals vanish."""
    t = np.gradient(run, axis=0)
    n2 = np.column_stack([t[:, 1], -t[:, 0]])
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    return np.stack([n2[:, :1] * cos[None], np.repeat(n2[:, 1:], len(cos), 1), n2[:, :1] * sin[None]], -1)


def disc(geo, material, centre, radius, y, up=True, segments=32):
    """A flat circle facing up or down."""
    cx, cz = centre
    th = TAU * np.arange(segments) / segments
    p = np.column_stack([cx + radius * np.cos(th), np.full(segments, y), cz - radius * np.sin(th)])
    p = np.vstack([[cx, y, cz], p])
    tris = np.column_stack([np.zeros(segments, int), 1 + np.arange(segments), 1 + (np.arange(segments) + 1) % segments])
    n = np.tile([0.0, 1.0 if up else -1.0, 0.0], (len(p), 1))
    geo.add(material, p, n, p[:, [0, 2]] * [1, -1], tris)


def offset_path(path, distance, closed=True):
    """Points of a path moved outwards (to the right of travel) by distance, mitred; open paths keep square ends."""
    if closed:
        return roofs.offset(path, distance)
    p = np.asarray(path, float)
    d = np.diff(p, axis=0)
    d /= np.maximum(np.hypot(*d.T), 1e-12)[:, None]
    n = np.column_stack([-d[:, 1], d[:, 0]])
    m = np.vstack([n[:1], n[:-1] + n[1:], n[-1:]])
    m[1:-1] /= np.maximum(1 + (n[:-1] * n[1:]).sum(axis=1), 1e-6)[:, None]
    return p + distance * m


def sweep(geo, material, ring, runs, closed=True, smooth=False, start=0.0):
    """A moulding profile along a ring (or open path): runs [[(out, y)…]] walked up the outside. Faces are flat
    along each edge, with hard corners unless smooth (round outlines)."""
    ring = np.asarray(ring, float)
    n_pts = len(ring)
    edges = n_pts if closed else n_pts - 1
    a, b = ring[np.arange(edges)], ring[(np.arange(edges) + 1) % n_pts]
    d = b - a
    length = np.hypot(*d.T)
    d /= np.maximum(length, 1e-12)[:, None]
    en = np.column_stack([-d[:, 1], d[:, 0]])
    s0 = start + np.concatenate([[0.0], np.cumsum(length)[:-1]])
    if smooth:
        vn = np.vstack([en[-1:] if closed else en[:1], en])
        vn = vn[:-1] + vn[1:] if closed else np.vstack([en[:1], en[:-1] + en[1:], en[-1:]])[:n_pts]
        vn /= np.maximum(np.hypot(*vn.T), 1e-12)[:, None]
    for run in runs:
        run = np.asarray(run, float)
        if len(run) < 2:
            continue
        rows = len(run)
        t = np.gradient(run, axis=0)
        n2 = np.column_stack([t[:, 1], -t[:, 0]])
        n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
        o = np.stack([offset_path(ring, out, closed) for out in run[:, 0]])  # (rows, n_pts, 2)
        pa, pb = o[:, np.arange(edges)], o[:, (np.arange(edges) + 1) % n_pts]
        y = np.repeat(run[:, 1][:, None], edges, axis=1)
        p = np.stack([np.stack([pa[..., 0], y, pa[..., 1]], -1), np.stack([pb[..., 0], y, pb[..., 1]], -1)], 2)
        # u along the edge from the base ring's start, v up the profile.
        ua = s0[None] + ((pa - a[None]) * d[None]).sum(-1)
        ub = s0[None] + ((pb - a[None]) * d[None]).sum(-1)
        v = np.repeat((run[0, 1] + _arc(run))[:, None], edges, axis=1)
        uv = np.stack([np.stack([ua, v], -1), np.stack([ub, v], -1)], 2)
        if smooth:
            ha = vn[np.arange(edges)][None] * n2[:, None, :1]
            hb = vn[(np.arange(edges) + 1) % n_pts][None] * n2[:, None, :1]
            h = np.stack([ha, hb], 2)
        else:
            h = np.repeat((en[None] * n2[:, None, :1])[:, :, None], 2, axis=2)
        normals = np.stack([h[..., 0], np.broadcast_to(n2[:, None, None, 1], h.shape[:3]), h[..., 1]], -1)
        k = np.arange(rows * edges * 2).reshape(rows, edges, 2)
        q0, q1, q2, q3 = k[:-1, :, 0], k[:-1, :, 1], k[1:, :, 1], k[1:, :, 0]
        tris = np.concatenate([np.stack([q0, q1, q2], -1).reshape(-1, 3), np.stack([q0, q2, q3], -1).reshape(-1, 3)])
        geo.add(material, p, normals, uv, tris)


def cap(geo, material, ring, y, up=True, holes=()):
    """A flat polygon facing up or down."""
    points, tris = roofs.triangulate(ring, holes)
    points = np.asarray(points, float)
    if not len(tris):
        return
    p = np.column_stack([points[:, 0], np.full(len(points), y), points[:, 1]])
    n = np.tile([0.0, 1.0 if up else -1.0, 0.0], (len(p), 1))
    geo.add(material, p, n, points * [1, -1], np.asarray(tris))


def prism(geo, material, ring, y0, y1, top=True, bottom=False, smooth=False):
    sweep(geo, material, ring, [[(0.0, y0), (0.0, y1)]], smooth=smooth)
    if top:
        cap(geo, material, ring, y1, True)
    if bottom:
        cap(geo, material, ring, y0, False)


class Frame:
    """A vertical face frame: origin (x, z), unit direction d along the face; local (u, y, n) with n outwards,
    to the right of d."""

    def __init__(self, origin, d):
        self.o = np.asarray(origin, float)
        d = np.asarray(d, float)
        self.d = d / max(np.hypot(*d), 1e-12)
        self.n = np.array([-self.d[1], self.d[0]])

    @classmethod
    def edge(cls, p, q):
        return cls(p, np.asarray(q, float) - np.asarray(p, float))

    def at(self, u, y, n):
        u, y, n = np.broadcast_arrays(np.asarray(u, float), np.asarray(y, float), np.asarray(n, float))
        x = self.o[0] + u * self.d[0] + n * self.n[0]
        z = self.o[1] + u * self.d[1] + n * self.n[1]
        return np.stack([x, y, z], -1)

    def vector(self, du, dy, dn):
        return np.array([du * self.d[0] + dn * self.n[0], dy, du * self.d[1] + dn * self.n[1]])


# Box faces: corner (u, y, n) selectors and the face normal in (u, y, n).
_FACES = {
    "f": ([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], (0, 0, 1)),
    "k": ([(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)], (0, 0, -1)),
    "t": ([(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)], (0, 1, 0)),
    "b": ([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], (0, -1, 0)),
    "l": ([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (-1, 0, 0)),
    "r": ([(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)], (1, 0, 0)),
}


def boxes(geo, material, frame, spans, faces="fktblr", shade=1.0):
    """Boxes in a frame: spans (k, 6) of [u0, u1, y0, y1, n0, n1]. UVs in metres on every face."""
    spans = np.asarray(spans, float).reshape(-1, 6)
    if not len(spans):
        return
    lo = spans[:, [0, 2, 4]]
    hi = spans[:, [1, 3, 5]]
    ps, ns, uvs = [], [], []
    for f in faces:
        corners, normal = _FACES[f]
        sel = np.array(corners, float)  # (4, 3)
        local = lo[:, None, :] + sel[None] * (hi - lo)[:, None, :]  # (k, 4, 3) in (u, y, n)
        ps.append(frame.at(local[..., 0], local[..., 1], local[..., 2]))
        ns.append(np.broadcast_to(frame.vector(*normal), (len(spans), 4, 3)))
        axis = {"f": (0, 1), "k": (0, 1), "t": (0, 2), "b": (0, 2), "l": (2, 1), "r": (2, 1)}[f]
        uvs.append(local[..., list(axis)])
    p = np.concatenate(ps, axis=1).reshape(-1, 3)
    n = np.concatenate(ns, axis=1).reshape(-1, 3)
    uv = np.concatenate(uvs, axis=1).reshape(-1, 2)
    quads = np.arange(len(p)).reshape(-1, 4)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    geo.add(material, p, n, uv, tris, np.full(len(p), shade))


def box(geo, material, frame, u0, u1, y0, y1, n0, n1, faces="fktblr"):
    boxes(geo, material, frame, [[u0, u1, y0, y1, n0, n1]], faces)


def obox(geo, material, origin, ex, ey, ez, faces="fktblr"):
    """An oriented box spanned by edge vectors ex (u), ey (y), ez (n) from origin; for sloped parts."""
    origin, ex, ey, ez = (np.asarray(v, float) for v in (origin, ex, ey, ez))
    ps, ns, uvs = [], [], []
    lengths = [np.linalg.norm(v) for v in (ex, ey, ez)]
    for f in faces:
        corners, normal = _FACES[f]
        sel = np.array(corners, float)
        ps.append(origin + sel[:, :1] * ex + sel[:, 1:2] * ey + sel[:, 2:] * ez)
        nv = normal[0] * ex / lengths[0] + normal[1] * ey / lengths[1] + normal[2] * ez / lengths[2]
        ns.append(np.tile(nv, (4, 1)))
        axis = {"f": (0, 1), "k": (0, 1), "t": (0, 2), "b": (0, 2), "l": (2, 1), "r": (2, 1)}[f]
        uvs.append(sel[:, list(axis)] * [lengths[axis[0]], lengths[axis[1]]])
    p = np.concatenate(ps)
    quads = np.arange(len(p)).reshape(-1, 4)
    geo.add(material, p, np.concatenate(ns), np.concatenate(uvs),
            np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))


def polygon(geo, material, frame, points, n, facing=1):
    """A flat polygon in a face frame: points [(u, y)] at depth n, facing out (1) or in (-1)."""
    points = np.asarray(points, float)
    pts, tris = roofs.triangulate(points)
    pts = np.asarray(pts, float)
    if not len(tris):
        return
    p = frame.at(pts[:, 0], pts[:, 1], n)
    normal = frame.vector(0, 0, facing)
    geo.add(material, p, np.tile(normal, (len(p), 1)), pts, np.asarray(tris))


# Reliefs.

def relief(geo, material, frame, u0, u1, y0, y1, n, depth, rng, detail=0, figures=None, tone=1.0):
    """A sculpted panel at depth n: a heightfield of standing figures in low relief, flush at its border. Far away
    it reads as carving; detail 1 keeps a coarse version, detail 2 a flat panel."""
    if detail > 1:
        box(geo, material, frame, u0, u1, y0, y1, n - 0.01, n + depth * 0.3, "f")
        return
    w, h = u1 - u0, y1 - y0
    cell = 0.09 if detail == 0 else 0.3
    cols = int(np.clip(round(w / cell), 4, 56 if detail == 0 else 14))
    rows = int(np.clip(round(h / cell), 4, 28 if detail == 0 else 7))
    u = np.linspace(u0, u1, cols + 1)
    y = np.linspace(y0, y1, rows + 1)
    uu, yy = np.meshgrid(u, y)
    field = np.zeros_like(uu)
    count = figures or max(1, round(w / (0.55 * h)))
    for k in range(count):
        c = u0 + (k + 0.5 + rng.uniform(-0.25, 0.25)) * w / count
        lean = rng.uniform(-0.12, 0.12) * h
        tall = rng.uniform(0.75, 0.92) * h
        fy = (yy - y0) / max(tall, 1e-6)
        axis = c + lean * fy
        body = np.exp(-((uu - axis) / (0.11 * h)) ** 2 - ((fy - 0.45) / 0.38) ** 4)
        head = np.exp(-(((uu - axis - lean * 0.1) / (0.06 * h)) ** 2 + ((fy - 0.92) / 0.07) ** 2))
        arm = np.exp(-((uu - axis - rng.choice([-1, 1]) * 0.12 * h * (fy - 0.45) / 0.3) / (0.04 * h)) ** 2
                     - ((fy - 0.62) / 0.14) ** 2)
        field = np.maximum(field, np.maximum(body * 0.9, np.maximum(head, arm * 0.7)))
    border = np.minimum(np.minimum(uu - u0, u1 - uu), np.minimum(yy - y0, y1 - yy))
    field *= np.clip(border / 0.06, 0, 1)
    p = frame.at(uu, yy, n + depth * field)
    uv = np.stack([uu, yy], -1)
    grid(geo, material, p, uv, shade=tone * (0.8 + 0.2 * field))


# Profiles: (out, y) runs walked up the outside, for sweeps and lathes alike.

def ovolo(x0, y0, x1, y1, steps=4):
    """Convex quarter round from (x0, y0) out and up to (x1, y1): steep at first, flat at the top."""
    a = np.linspace(0, math.pi / 2, steps + 1)
    return [(x0 + (x1 - x0) * math.sin(t), y0 + (y1 - y0) * (1 - math.cos(t))) for t in a]


def echinus(x0, y0, x1, y1, steps=5):
    """A Doric echinus: flares out fast from the neck, then turns up under the abacus."""
    a = np.linspace(0, 1, steps + 1)
    return [(x0 + (x1 - x0) * (1 - (1 - t) ** 2.2), y0 + (y1 - y0) * t) for t in a]


def cavetto(x0, y0, x1, y1, steps=4):
    """Concave quarter round from (x0, y0) up and out to (x1, y1)."""
    a = np.linspace(0, math.pi / 2, steps + 1)
    return [(x0 + (x1 - x0) * (1 - math.cos(t)), y0 + (y1 - y0) * math.sin(t)) for t in a]


def cyma(x0, y0, x1, y1, steps=6, recta=True):
    """S-curve between two points: cyma recta (concave below, convex above) or reversa."""
    t = np.linspace(0, 1, steps + 1)
    s = (1 - np.cos(t * math.pi)) / 2 if recta else t + 0.12 * np.sin(t * TAU)
    f = t if recta else (1 - np.cos(t * math.pi)) / 2
    return [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * b) for a, b in zip(s, f)]


def torus(x, y0, y1, bulge, steps=4):
    """A half-round bead from y0 to y1 standing out by bulge from x."""
    a = np.linspace(-math.pi / 2, math.pi / 2, steps + 1)
    return [(x + bulge * math.cos(t), (y0 + y1) / 2 + (y1 - y0) / 2 * math.sin(t)) for t in a]


def base_profile(r, y, h, detail=0):
    """An Attic base above a plinth: torus, scotia, torus, then the shaft's radius r; height h."""
    if detail >= 2:
        return [[(r * 1.12, y), (r * 1.12, y + h * 0.5)], [(r * 1.12, y + h * 0.5), (r, y + h * 0.5)],
                [(r, y + h * 0.5), (r, y + h)]]
    steps = 3 if detail == 0 else 2
    t1 = torus(r * 1.08, y, y + 0.36 * h, r * 0.12, steps)
    sc = [(r * 1.06, y + 0.40 * h)] + [(r * 1.06 - r * 0.07 * math.sin(a), y + 0.40 * h + 0.24 * h * (1 - math.cos(a)) / 2)
                                       for a in np.linspace(0, math.pi, steps + 1)[1:]]
    t2 = torus(r * 1.02, y + 0.64 * h, y + 0.88 * h, r * 0.08, steps)
    return [[(r * 1.08, y), *t1[1:]], [t1[-1], (r * 1.06, y + 0.36 * h), (r * 1.06, y + 0.40 * h)], sc,
            [sc[-1], (r * 1.02, y + 0.64 * h)], t2, [t2[-1], (r, y + 0.88 * h)], cavetto(r, y + 0.88 * h, r, y + h, 1)]


def cornice_runs(h, overhang, y0=0.0, kind="doric", detail=0):
    """A cornice from y0 to y0 + h projecting overhang: bed moulding, soffit, corona and sima."""
    o, s = overhang, 3 if detail == 0 else 2
    bed = y0 + 0.28 * h
    corona = 0.78 * o
    if detail >= 2:
        return facets([(0.0, y0), (corona, y0 + 0.3 * h), (corona, y0 + 0.75 * h), (o, y0 + h)])
    runs = [cyma(0.0, y0, 0.22 * o, bed, s, recta=False)]
    runs += facets([(0.22 * o, bed), (corona, bed), (corona, y0 + 0.62 * h), (corona + 0.03, y0 + 0.62 * h),
                    (corona + 0.03, y0 + 0.68 * h)])
    runs.append(cyma(corona + 0.03, y0 + 0.68 * h, o, y0 + 0.96 * h, s + 1))
    runs.append([(o, y0 + 0.96 * h), (o, y0 + h)])  # The top is capped by the caller.
    return runs


def band_runs(y0, y1, out, detail=0):
    """A simple projecting band (string course or impost): fillet, ovolo, flat top."""
    h = y1 - y0
    if detail >= 2:
        return facets([(0.0, y0), (out, y0), (out, y1), (0.0, y1)])
    return [*facets([(0.0, y0), (0.0, y0 + 0.15 * h)]), ovolo(0.0, y0 + 0.15 * h, 0.55 * out, y0 + 0.45 * h, 3),
            *facets([(0.55 * out, y0 + 0.45 * h), (out, y0 + 0.5 * h), (out, y1), (0.0, y1)])]


def plinth_runs(y0, y1, out, detail=0):
    """A base course: a plain block stepping back through a small cyma to the wall."""
    h = y1 - y0
    if detail >= 2:
        return facets([(out, y0), (out, y1), (0.0, y1)])
    return [*facets([(out, y0), (out, y0 + 0.7 * h)]), cyma(out, y0 + 0.7 * h, 0.0, y1, 3, recta=False)]


# Orders.

ORDERS = ("tuscan", "doric", "ionic", "corinthian")


def column(geo, stone, carved, centre, y0, y1, diameter, order="doric", detail=0, base=True, angle=0.0, flutes=None,
           taper=0.14, foot=None):
    """A column standing on y0 with its capital's top at y1: plinth and Attic base, a shaft with entasis and
    flutes, and the order's capital. angle turns the square plinth and abacus (radians from +x). foot is where
    the base starts (the ground), when y0 lies below it; the plinth reaches down to y0."""
    d = diameter
    sunk = y0
    y0 = max(y0, foot) if foot is not None else y0
    r = d / 2
    frame = Frame(centre, (math.cos(angle), -math.sin(angle)))
    hc = {"tuscan": 0.45, "doric": 0.5, "ionic": 0.42, "corinthian": 1.1}[order] * d
    hb = (0.5 * d if base else 0.0)
    height = y1 - y0
    if height < hb + hc + d:
        scale = height / (hb + hc + d)
        hb, hc = hb * scale, hc * scale
    yb, yc = y0 + hb, y1 - hc
    segments = _segments(r, detail, flutes)
    n_flutes = (flutes if flutes is not None else {"tuscan": 0, "doric": 20, "ionic": 24, "corinthian": 24}[order])
    if detail >= 2:
        n_flutes = 0
    top = r * (1 - taper)
    # Plinth and base.
    if base:
        hp = 0.32 * hb
        box(geo, stone, frame, -0.66 * d, 0.66 * d, sunk, y0 + hp, -0.66 * d, 0.66 * d, "fktlr")
        lathe(geo, stone, centre, base_profile(r, y0 + hp, hb - hp, detail), segments)
    # Shaft with entasis: straight for the lower third, then curving in to the top radius.
    t = np.linspace(0, 1, 9 if detail == 0 else 5 if detail == 1 else 2)
    radius = r - (r - top) * np.clip((t - 1 / 3) / (2 / 3), 0, 1) ** 1.5
    shaft = [(float(a), float(yb + (yc - yb) * b)) for a, b in zip(radius, t)]
    if n_flutes and detail < 2:
        segments = n_flutes * (4 if detail == 0 else 2)
        lathe(geo, stone, centre, [shaft], segments, flutes=n_flutes, depth=0.035 * d, fluted=(yb, yc),
              fillet=0.18 if order in ("ionic", "corinthian") else 0.0, reference=r, phase=angle)
    else:
        lathe(geo, stone, centre, [shaft], segments, reference=r)
    capital(geo, stone, carved, frame, centre, yc, y1, top, d, order, detail, segments)


def _segments(r, detail, flutes=None):
    if detail >= 2:
        return 8
    return int(np.clip(round(TAU * r / (0.1 if detail == 0 else 0.3)), 16, 64 if detail == 0 else 24)) // 4 * 4


def capital(geo, stone, carved, frame, centre, y0, y1, top, d, order, detail, segments):
    h = y1 - y0
    abacus = 0.62 * d
    if order in ("doric", "tuscan"):
        ha = 0.4 * h
        if detail >= 2:
            lathe(geo, stone, centre, facets([(top, y0), (abacus * 0.9, y1 - ha)]), 8)
        else:
            # Necking, an annulet ring, then the echinus flaring out under the abacus.
            start = y0 + 0.2 * h
            neck = [(top, y0), (top, y0 + 0.1 * h), (top + 0.015 * d, y0 + 0.1 * h), (top + 0.015 * d, start)]
            runs = [*(facets(neck) if detail == 0 else [[(top, y0), (top, start)]]),
                    echinus(top + (0.015 * d if detail == 0 else 0), start, abacus - 0.01 * d, y1 - ha,
                            4 if detail == 0 else 2)]
            lathe(geo, carved, centre, runs, segments)
        box(geo, carved, frame, -abacus, abacus, y1 - ha, y1, -abacus, abacus, "fktblr")
        return
    if order == "ionic":
        ha = 0.18 * h
        lathe(geo, carved, centre, [ovolo(top, y0, top * 1.12, y0 + 0.4 * h, 3)], segments)
        box(geo, carved, frame, -0.62 * d, 0.62 * d, y0 + 0.4 * h, y1 - ha, -0.5 * d, 0.5 * d, "fktblr")
        box(geo, carved, frame, -0.66 * d, 0.66 * d, y1 - ha, y1, -0.58 * d, 0.58 * d, "fktblr")
        if detail <= 1:
            # Volutes: bolsters along the face's depth whose ends show stepped spirals.
            for side in (-1, 1):
                cu = side * 0.55 * d
                cy = y0 + 0.42 * h
                rv = 0.22 * d
                origin = frame.at(cu, cy, 0.0)
                axis = frame.vector(0, 0, 1)
                profile = [(0.4 * rv, -0.56 * d), (0.7 * rv, -0.56 * d), (0.7 * rv, -0.55 * d), (rv, -0.55 * d),
                           (0.82 * rv, 0.0), (rv, 0.55 * d), (0.7 * rv, 0.55 * d), (0.7 * rv, 0.56 * d),
                           (0.4 * rv, 0.56 * d)]
                revolve(geo, carved, origin, axis, frame.vector(1, 0, 0), [profile] if detail else facets(profile),
                        16 if detail == 0 else 8)
                for end in (-1, 1):
                    disc3(geo, carved, origin + axis * end * 0.56 * d, axis * end, 0.4 * rv, 12)
        return
    # Corinthian: a bell clad in two rows of acanthus leaves under a concave abacus with corner volutes.
    ha = 0.14 * h
    bell = [(top, y0), (top * 1.03, y0 + 0.5 * h), (top * 1.12, y0 + 0.78 * h), (abacus * 0.95, y1 - ha)]
    lathe(geo, carved, centre, [bell] if detail < 2 else facets([bell[0], bell[-1]]), segments)
    corners = [(math.cos(a), math.sin(a)) for a in np.linspace(0, TAU, 9)[:-1] + math.pi / 8]
    octagon = [frame.at(abacus * 1.05 * c, 0, abacus * 1.05 * s)[[0, 2]] for c, s in corners]
    octagon = roofs.ccw(np.array(octagon))
    prism(geo, carved, octagon, y1 - ha, y1, top=True, bottom=True)
    if detail == 0:
        for row, (frac, n_leaves, turn) in enumerate(((0.42, 8, 0.0), (0.72, 8, math.pi / 8))):
            for k in range(n_leaves):
                leaf(geo, carved, centre, y0, top * (1.04 + 0.04 * row), angle=turn + k * TAU / n_leaves,
                     height=frac * (h - ha), width=0.8 * TAU * top / n_leaves, curl=0.12 * d)
        for k in range(4):  # Corner volutes under the abacus.
            a = math.pi / 4 + k * math.pi / 2
            f = Frame(centre, (math.cos(a), -math.sin(a)))
            box(geo, carved, f, abacus * 0.75, abacus * 1.02, y1 - ha - 0.22 * h, y1 - ha, -0.06 * d, 0.06 * d)


def leaf(geo, material, centre, y0, r, angle, height, width, curl):
    """An acanthus leaf hugging a bell at radius r: a curved shell that curls out and over at its tip."""
    s = np.linspace(0, 1, 6)[:, None]
    w = np.linspace(-1, 1, 5)[None, :]
    half = width / 2 * np.sin(np.pi * np.clip(s * 0.9 + 0.1, 0, 1)) ** 0.6
    a = angle + w * half / r
    out = r + curl * s ** 2.5 + 0.015 * (1 - w ** 2) * (1 - s)
    y = y0 + height * (s - 0.18 * np.clip(s - 0.8, 0, 1) / 0.2 * s)
    cx, cz = centre
    p = np.stack([cx + out * np.cos(a), np.broadcast_to(y, a.shape), cz - out * np.sin(a)], -1)
    uv = np.stack([w * half + 0 * s, np.broadcast_to(y, a.shape)], -1)
    grid(geo, material, p, uv)
    back = p - np.stack([np.cos(a), 0 * a, -np.sin(a)], -1) * 0.012
    grid(geo, material, back[:, ::-1], uv[:, ::-1])


def revolve(geo, material, origin, axis, reference, runs, segments):
    """Surface of revolution round any axis: runs [[(r, t)…]] with t along axis from origin."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = np.asarray(reference, float)
    e1 = e1 - axis * (e1 @ axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    th = TAU * np.arange(segments + 1) / segments
    ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    for run in runs:
        run = np.asarray(run, float)
        p = origin + run[:, 1][:, None, None] * axis + run[:, 0][:, None, None] * ring[None]
        uv = np.stack(np.broadcast_arrays((th / TAU * TAU * run[:, :1].max())[None, :], _arc(run)[:, None]), -1)
        t = np.gradient(run, axis=0)
        n2 = np.column_stack([t[:, 1], -t[:, 0]])
        normals = n2[:, :1, None] * ring[None] + n2[:, 1:, None] * axis
        grid(geo, material, p, uv, closed=True, normals=normals)


def disc3(geo, material, centre, normal, radius, segments=12):
    """A flat circle facing along normal."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    e1 = np.cross(normal, [0.0, 1.0, 0.0] if abs(normal[1]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    th = TAU * np.arange(segments) / segments
    p = np.vstack([centre, centre + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)])
    tris = np.column_stack([np.zeros(segments, int), 1 + np.arange(segments), 1 + (np.arange(segments) + 1) % segments])
    uv = np.column_stack([p @ e1, p @ e2])
    geo.add(material, p, np.tile(normal, (len(p), 1)), uv, tris)


# Entablatures.

def triglyph_spots(length, axes, width, spacing):
    """Centres of triglyphs along an edge: over every column axis and the corners, with the bays between divided
    into near-square metopes."""
    ends = [width / 2, length - width / 2]
    anchors = sorted({round(a, 3) for a in [*ends, *(a for a in axes if ends[0] + width < a < ends[1] - width)]})
    spots = []
    for a, b in zip(anchors, anchors[1:]):
        k = max(1, round((b - a) / spacing))
        spots += [a + (b - a) * i / k for i in range(k)]
    return spots + [anchors[-1]]


def entablature(geo, stone, carved, ring, y0, y1, order="doric", detail=0, overhang=None, axes=None, rng=None,
                metopes=True, round_=False):
    """Architrave, frieze and cornice round a ring from y0 to y1. axes maps edge index to column axis positions
    along that edge, so triglyphs (Doric) stand over the columns."""
    ring = np.asarray(ring, float)
    h = y1 - y0
    ha, hf = 0.36 * h, 0.36 * h
    hc = h - ha - hf
    ya, yf = y0 + ha, y0 + ha + hf
    oh = overhang if overhang is not None else 0.8 * hc
    rng = rng or np.random.default_rng(0)
    taenia = 0.1 * ha
    if order == "doric":
        architrave = [[(0.0, y0), (0.0, ya - taenia)], *facets([(0.0, ya - taenia), (0.05, ya - taenia), (0.05, ya),
                                                                (0.0, ya)])]
    else:  # Three fasciae, each standing a little proud of the one below.
        f = [y0, y0 + 0.3 * ha, y0 + 0.62 * ha, ya - 0.08 * ha]
        architrave = facets([(0.0, f[0]), (0.0, f[1]), (0.02, f[1]), (0.02, f[2]), (0.04, f[2]), (0.04, f[3])])
        architrave.append(ovolo(0.04, f[3], 0.08, ya, 2) if detail < 2 else [(0.04, f[3]), (0.08, ya)])
        architrave += facets([(0.08, ya), (0.0, ya)])
    frieze = [[(0.0, ya), (0.0, yf)]]
    runs = architrave + frieze + cornice_runs(hc, oh, yf, order, detail)
    sweep(geo, stone, ring, runs, smooth=round_)
    cap(geo, stone, offset_path(ring, oh), y1, True)
    cap(geo, stone, ring, y0, False)  # The soffit: a gate's passage ceilings.
    if detail >= 2:
        return
    if detail == 0 and not round_:
        coffers(geo, stone, ring, y0)
    edges = len(ring)
    bed = yf + 0.28 * hc
    for k in range(edges):
        p, q = ring[k], ring[(k + 1) % edges]
        frame = Frame.edge(p, q)
        length = float(np.hypot(*(q - p)))
        if length < 0.5:
            continue
        if order == "doric" and not round_:
            wt = min(0.66 * hf, length / 3)
            spots = triglyph_spots(length, (axes or {}).get(k, []), wt, wt + hf * 0.95)
            spans, bars, regulae, guttae, mutules = [], [], [], [], []
            for c in spots:
                spans.append([c - wt / 2, c + wt / 2, ya, yf, 0.0, 0.05])
                if detail == 0:
                    for b in (1, 5, 9):  # Glyphs: three bars with grooves between, under a capital band.
                        bars.append([c - wt / 2 + b * wt / 12, c - wt / 2 + (b + 2) * wt / 12, ya, yf - 0.14 * hf,
                                     0.05, 0.085])
                    bars.append([c - wt / 2, c + wt / 2, yf - 0.14 * hf, yf, 0.05, 0.085])
                    regulae.append([c - wt / 2, c + wt / 2, ya - taenia - 0.06 * ha, ya - taenia, 0.0, 0.05])
                    for g in range(6):
                        gu = c - wt / 2 + (g + 0.5) * wt / 6
                        guttae.append([gu - wt / 30, gu + wt / 30, ya - taenia - 0.12 * ha, ya - taenia - 0.06 * ha,
                                       0.0, 0.035])
                mutules.append([c - wt / 2, c + wt / 2, bed - 0.04 * hc, bed, 0.22 * oh, 0.76 * oh])
            if len(spots) > 1:
                for a, b in zip(spots, spots[1:]):
                    mutules.append([(a + b) / 2 - wt / 2, (a + b) / 2 + wt / 2, bed - 0.04 * hc, bed, 0.22 * oh,
                                    0.76 * oh])
            boxes(geo, carved, frame, spans, "ftblr")
            boxes(geo, carved, frame, bars, "ftblr", shade=0.95)
            boxes(geo, carved, frame, regulae, "fblr")
            boxes(geo, carved, frame, guttae, "fblr", shade=0.9)
            boxes(geo, stone, frame, mutules if detail == 0 else [], "fblr", shade=0.9)
            if detail <= 1 and metopes:
                for a, b in zip(spots, spots[1:]):
                    if b - a - wt > 0.3:
                        relief(geo, carved, frame, a + wt / 2 + 0.04, b - wt / 2 - 0.04, ya + 0.04 * hf,
                               yf - 0.16 * hf, 0.0, 0.07, rng, detail, figures=1 + int(rng.random() < 0.5))
        elif detail == 0:
            # Dentils under the corona.
            dw = max(0.06, 0.12 * hc)
            n = int(length // (dw * 1.6))
            if n >= 2:
                c0 = (length - (n - 1) * dw * 1.6) / 2
                spans = [[c0 + i * dw * 1.6 - dw / 2, c0 + i * dw * 1.6 + dw / 2, bed - 0.18 * hc, bed - 0.02 * hc,
                          0.0, 0.2 * oh] for i in range(n)]
                boxes(geo, carved, frame, spans, "fblr")


def coffers(geo, stone, ring, y, spacing=1.5, rib=0.22, depth=0.16):
    """A grid of ribs under a soffit at y, so passage ceilings read as coffered."""
    c, axis, hu, hv = roofs.box(ring)
    frame = Frame(c, axis)
    nu, nv = max(1, round(2 * hu / spacing)), max(1, round(2 * hv / spacing))
    spans = [[-hu + 0.3, hu - 0.3, y - depth, y, -hv + 0.3 + k * (2 * hv - 0.6) / nv - rib / 2,
              -hv + 0.3 + k * (2 * hv - 0.6) / nv + rib / 2] for k in range(nv + 1)]
    spans += [[-hu + 0.3 + k * (2 * hu - 0.6) / nu - rib / 2, -hu + 0.3 + k * (2 * hu - 0.6) / nu + rib / 2,
               y - depth, y, -hv + 0.3, hv - 0.3] for k in range(nu + 1)]
    boxes(geo, stone, frame, spans, "fkblr", shade=0.85)


# Walls, attics and panels.

def panel(geo, stone, carved, frame, u0, u1, y0, y1, n=0.0, detail=0, rng=None, sculpted=False, frame_width=None):
    """A framed panel on a face: a raised moulding frame round a flat or sculpted field."""
    w = frame_width or min(0.18, 0.08 * min(u1 - u0, y1 - y0) + 0.04)
    if u1 - u0 < 3 * w or y1 - y0 < 3 * w:
        return
    if detail >= 2:
        return
    spans = [[u0, u1, y0, y0 + w, n, n + 0.06], [u0, u1, y1 - w, y1, n, n + 0.06],
             [u0, u0 + w, y0 + w, y1 - w, n, n + 0.06], [u1 - w, u1, y0 + w, y1 - w, n, n + 0.06]]
    boxes(geo, carved, frame, spans, "ftblr")
    if sculpted:
        relief(geo, carved, frame, u0 + w, u1 - w, y0 + w, y1 - w, n, min(0.12, 0.06 * (y1 - y0)),
               rng or np.random.default_rng(0), detail)


def window(geo, stone, glass, frame, u0, u1, y0, y1, arched=False, detail=0):
    """A window on a face, without cutting the wall: a glazed field just proud of it inside a raised stone
    surround with a sill; arched windows get a round head and a keystone."""
    w = u1 - u0
    cu = (u0 + u1) / 2
    spring = y1 - w / 2 if arched else y1
    steps = 8 if detail == 0 else 4
    a = np.linspace(math.pi, 0, steps + 1)
    head = [(cu + w / 2 * math.cos(t), spring + w / 2 * math.sin(t)) for t in a] if arched else [(u0, y1), (u1, y1)]
    outline = [(u0, y0), (u1, y0), *head[::-1]] if arched else [(u0, y0), (u1, y0), (u1, y1), (u0, y1)]
    polygon(geo, glass, frame, outline, 0.015, 1)
    if detail >= 1:
        return
    s = min(0.2, 0.12 * w + 0.04)
    boxes(geo, stone, frame, [[u0 - s, u0, y0, spring, 0.0, 0.07], [u1, u1 + s, y0, spring, 0.0, 0.07],
                              [u0 - s - 0.05, u1 + s + 0.05, y0 - 0.12, y0, 0.0, 0.12]], "ftblr")
    if arched:
        for t0, t1 in zip(a[:-1], a[1:]):
            r0, r1 = w / 2, w / 2 + s
            pts = [(cu + r0 * math.cos(t0), spring + r0 * math.sin(t0)), (cu + r1 * math.cos(t0), spring + r1 * math.sin(t0)),
                   (cu + r1 * math.cos(t1), spring + r1 * math.sin(t1)), (cu + r0 * math.cos(t1), spring + r0 * math.sin(t1))]
            polygon(geo, stone, frame, pts, 0.07, 1)
        box(geo, stone, frame, cu - 0.6 * s, cu + 0.6 * s, y1 - 0.05, y1 + s * 1.2, 0.0, 0.1)
    else:
        box(geo, stone, frame, u0 - s - 0.05, u1 + s + 0.05, y1, y1 + s, 0.0, 0.1)


def openings(geo, stone, glass, frame, length, y0, y1, kind, detail=0, margin=0.6, buttresses=False):
    """Windows along a face from y0 to y1: kind "arched" (one tall row, churches), "belfry" (paired openings
    near the top) or "rect" (a row per storey)."""
    h = y1 - y0
    if h < 2.0 or length < 2.5:
        return
    spacing = {"arched": 4.2, "belfry": max(2.0, length / 2), "rect": 3.2}[kind]
    n = max(1, int((length - 2 * margin) // spacing))
    step = (length - 2 * margin) / n
    rows = [(y0 + 0.25 * h, y0 + min(0.82 * h, 0.25 * h + 12.0))] if kind == "arched" else \
        [(y1 - 0.42 * h, y1 - 0.12 * h)] if kind == "belfry" else \
        [(y0 + k * 3.6 + 1.0, y0 + k * 3.6 + 3.0) for k in range(max(1, int((h - 0.6) // 3.6)))]
    for k in range(n):
        c = margin + (k + 0.5) * step
        w = min(0.42 * step, 2.4 if kind != "belfry" else 1.2)
        for a, b in rows:
            if kind == "belfry" and w > 0.6:
                window(geo, stone, glass, frame, c - w - 0.05, c - 0.05, a, b, True, detail)
                window(geo, stone, glass, frame, c + 0.05, c + w + 0.05, a, b, True, detail)
            else:
                window(geo, stone, glass, frame, c - w / 2, c + w / 2, a, b, kind != "rect", detail)
        if buttresses and 0 < k and detail < 2:
            u = margin + k * step
            # A stepped buttress: deep at the foot, shallower above.
            box(geo, stone, frame, u - 0.3, u + 0.3, y0, y0 + 0.45 * h, 0.0, 0.75, "ftlr")
            box(geo, stone, frame, u - 0.25, u + 0.25, y0 + 0.45 * h, y0 + 0.75 * h, 0.0, 0.4, "ftlr")


def block(geo, stone, carved, ring, y0, y1, detail=0, plinth=True, cornice=True, panels=True, rng=None,
          sculpted=False, ground=False, pilasters=False, cornice_out=None, smooth=False, windows=None, glass=None,
          buttresses=False):
    """A stone block: base course, die with framed panels (or windows of a kind, see openings) on long enough
    faces, and a cap moulding."""
    ring = np.asarray(ring, float)
    h = y1 - y0
    hp = min(0.6, 0.12 * h) if plinth else 0.0
    hk = min(0.8, 0.14 * h) if cornice else 0.0
    out = cornice_out if cornice_out is not None else min(0.35, 0.5 * hk + 0.05)
    if plinth:
        sweep(geo, stone, ring, plinth_runs(y0, y0 + hp, min(0.12, hp * 0.4), detail), smooth=smooth)
    sweep(geo, stone, ring, [[(0.0, y0 + hp), (0.0, y1 - hk)]], smooth=smooth)
    if cornice:
        sweep(geo, stone, ring, band_runs(y1 - hk, y1, out, detail), smooth=smooth)
    cap(geo, stone, ring, y1, True)
    if not ground:
        cap(geo, stone, ring, y0, False)
    if detail >= 2 or smooth:
        return
    rng = rng or np.random.default_rng(0)
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        length = float(np.hypot(*(q - p)))
        frame = Frame.edge(p, q)
        if pilasters and length > 3.0 and h > 4.0:
            pw = min(0.9, 0.12 * h)
            spans = [[0.05, 0.05 + pw, y0 + hp, y1 - hk, 0.0, 0.08], [length - 0.05 - pw, length - 0.05, y0 + hp,
                                                                       y1 - hk, 0.0, 0.08]]
            boxes(geo, stone, frame, spans, "flr")
            margin = 0.3 + pw
        else:
            margin = 0.3
        hh = y1 - hk - (y0 + hp)
        if windows and glass:
            openings(geo, stone, glass, frame, length, y0 + hp, y1 - hk, windows, detail, margin, buttresses)
        elif panels and length - 2 * margin > 1.0 and hh > 1.0:
            m = min(0.4, 0.15 * hh)
            panel(geo, stone, carved, frame, margin, length - margin, y0 + hp + m, y1 - hk - m, 0.0, detail, rng,
                  sculpted and length > 2.0)


def steps(geo, stone, ring, y0, y1, count, tread, detail=0, smooth=False, ground=False):
    """count steps from y0 up to y1, each set in from the one below by tread."""
    ring = np.asarray(ring, float)
    rise = (y1 - y0) / max(count, 1)
    for k in range(count):
        r = offset_path(ring, -k * tread) if k else ring
        a = y0 + k * rise
        prism(geo, stone, r, a, a + rise, top=True, bottom=False, smooth=smooth)
        if detail == 0 and rise > 0.12:  # A nosing shadow line under each tread's edge.
            sweep(geo, stone, r, facets([(0.0, a + rise * 0.82), (0.02, a + rise * 0.86), (0.02, a + rise)]),
                  smooth=smooth)


def balustrade(geo, stone, path, y0, height=1.0, detail=0, closed=False):
    """Plinth, turned balusters and a hand rail along a path (or round a ring)."""
    path = np.asarray(path, float)
    hp, hr = 0.15 * height, 0.12 * height
    edges = len(path) if closed else len(path) - 1
    for k in range(edges):
        p, q = path[k], path[(k + 1) % len(path)]
        frame = Frame.edge(p, q)
        length = float(np.hypot(*(q - p)))
        box(geo, stone, frame, 0, length, y0, y0 + hp, -0.14, 0.14, "fktlr")
        box(geo, stone, frame, -0.02, length + 0.02, y0 + height - hr, y0 + height, -0.16, 0.16)
        if detail >= 2:
            box(geo, stone, frame, 0, length, y0 + hp, y0 + height - hr, -0.08, 0.08, "fk")
            continue
        n = max(1, int(length / 0.32))
        for i in range(n):
            c = frame.at((i + 0.5) * length / n, 0, 0)
            b = y0 + hp
            hh = height - hp - hr
            profile = [(0.07, b), (0.05, b + 0.1 * hh), (0.1, b + 0.35 * hh), (0.045, b + 0.75 * hh),
                       (0.06, b + 0.9 * hh), (0.06, b + hh)]
            lathe(geo, stone, (c[0], c[2]), [profile], 8 if detail == 0 else 4)


def arch_wall(geo, stone, carved, frame, length, y0, y1, thickness, opening, detail=0):
    """A wall slab with a round-headed passage: opening = (u0, u1, spring) with the arch springing at y=spring.
    Builds both faces, the passage's sides and soffit, an archivolt and a keystone."""
    u0, u1, spring = opening
    r = (u1 - u0) / 2
    cu = (u0 + u1) / 2
    steps_ = 16 if detail == 0 else 8 if detail == 1 else 4
    a = np.linspace(math.pi, 0, steps_ + 1)
    arc = np.column_stack([cu + r * np.cos(a), spring + r * np.sin(a)])
    outline = np.vstack([[(0, y0), (u0, y0)], arc, [(u1, y0), (length, y0), (length, y1), (0, y1)]])
    for n, facing in ((thickness / 2, 1), (-thickness / 2, -1)):
        polygon(geo, stone, frame, outline if facing > 0 else outline, n, facing)
    # Ends and top.
    box(geo, stone, frame, 0, length, y0, y1, -thickness / 2, thickness / 2, "tlr")
    # Passage sides and soffit, facing into the passage.
    inner = np.vstack([[(u0, y0)], arc, [(u1, y0)]])
    for (ua, ya), (ub, yb) in zip(inner[:-1], inner[1:]):
        pa, pb = frame.at(ua, ya, 0.0), frame.at(ub, yb, 0.0)
        e = frame.vector(0, 0, 1) * thickness / 2
        quad = np.array([pa - e, pb - e, pb + e, pa + e])
        tangent = pb - pa
        normal = np.cross(tangent, frame.vector(0, 0, 1))
        normal *= -1 if normal @ (np.array([*frame.at(cu, spring, 0.0)]) - (pa + pb) / 2) < 0 else 1
        uv = np.array([[0, 0], [np.linalg.norm(tangent), 0], [np.linalg.norm(tangent), thickness], [0, thickness]])
        geo.add(stone, quad, np.tile(normal, (4, 1)), uv, [[0, 1, 2], [0, 2, 3]])
    if detail < 2:
        # Archivolt: a band of voussoirs round the arch on both faces, with a keystone.
        w = min(0.5, 0.18 * r)
        for side in (1, -1):
            n0 = side * thickness / 2
            for (a0, a1) in zip(a[:-1], a[1:]):
                pts = [(cu + r * math.cos(a0), spring + r * math.sin(a0)), (cu + (r + w) * math.cos(a0),
                                                                           spring + (r + w) * math.sin(a0)),
                       (cu + (r + w) * math.cos(a1), spring + (r + w) * math.sin(a1)),
                       (cu + r * math.cos(a1), spring + r * math.sin(a1))]
                polygon(geo, carved, frame, pts, n0 + side * 0.04, side)
            box(geo, carved, frame, cu - 0.3 * w, cu + 0.3 * w, spring + r - 0.05, spring + r + w * 1.3,
                *sorted((n0, n0 + side * 0.08)))
