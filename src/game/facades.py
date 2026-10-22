"""Facades: a mesh builder, wall-space geometry, and the facade grammar that places and details openings.

A wall runs from p to q along a footprint ring; its outward normal is to the right of travel. Wall space is
(u, v, w): u metres along the wall, v the world y, and w the depth behind the facade plane (negative in front).
Openings are holes in the wall closed at the back by glass, a door or a shopfront, so the shell stays closed. The
wall around them is a grid over every opening edge, which avoids T-junctions and cracks.

Depth is what makes a facade read as real under sun and ambient occlusion, so LOD0 builds real geometry: reveals,
frames with mullions and transoms, glass panes recessed behind them, sills, stucco surrounds, hoods and pediments,
balconies with railings, shopfronts with fascias and awnings, and portals with steps. LOD1 keeps the window rhythm
as flat quads on a plain wall, LOD2 only the glass.

Vertex colours carry the building tint and soft weathering; faces facing down or sideways are slightly darker, as
baked occlusion. Glass is never darkened: the viewer maps fake interiors onto it.
"""

import math
from dataclasses import dataclass

import numpy as np

from .roofs import offset, triangulate
from .types import Part

WHITE = (1.0, 1.0, 1.0, 1.0)
# Baked occlusion per face direction: fronts, tops, bottoms (soffits, undersides of sills) and sides.
TONE = dict(f=1.0, t=0.96, b=0.74, l=0.88, r=0.88, k=0.9)
FRAME = 0.06  # Width of window frame bars.
FRAME_DEPTH = 0.07  # How far frames stand in front of the glass.


def dim(colour, k):
    return (colour[0] * k, colour[1] * k, colour[2] * k, colour[3])


class Mesh:
    """Triangles grouped by material, added as flat polygons fanned from their first point (so the polygon must be
    star-shaped around it)."""

    def __init__(self, unit=frozenset()):
        self.groups = {}
        self.unit = unit  # Materials with 0–1 UVs instead of metres.

    def poly(self, material, points, normal, uvs, colors=WHITE, raw=False):
        """raw keeps the given UVs of a unit material (atlas cells) instead of stretching 0–1 over the polygon."""
        keep = [k for k in range(len(points)) if points[k] != points[k - 1]]
        if len(keep) < 3:
            return
        points, uvs = [points[k] for k in keep], [uvs[k] for k in keep]
        if not isinstance(colors, tuple):
            colors = [colors[k] for k in keep]
        # Newell's normal tells the winding, even with collinear corners.
        nx = ny = nz = 0.0
        for (ax, ay, az), (bx, by, bz) in zip(points, points[1:] + points[:1]):
            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)
        dot = nx * normal[0] + ny * normal[1] + nz * normal[2]
        if abs(nx) + abs(ny) + abs(nz) < 1e-10:
            return
        if material in self.unit and not raw:
            us, vs = [uv[0] for uv in uvs], [uv[1] for uv in uvs]
            du, dv = max(max(us) - min(us), 1e-9), max(max(vs) - min(vs), 1e-9)
            uvs = [((u - min(us)) / du, (v - min(vs)) / dv) for u, v in uvs]
        positions, normals, texcoords, tints, indices = self.groups.setdefault(material, ([], [], [], [], []))
        base = len(positions)
        positions += points
        normals += [normal] * len(points)
        texcoords += uvs
        tints += [colors] * len(points) if isinstance(colors, tuple) else colors
        for k in range(1, len(points) - 1):
            indices += (base, base + k, base + k + 1) if dot >= 0 else (base, base + k + 1, base + k)

    def count(self):
        return sum(len(group[4]) // 3 for group in self.groups.values())

    def parts(self):
        parts = []
        for material, (positions, normals, uvs, colors, indices) in sorted(self.groups.items()):
            if indices:
                parts.append(Part(material, np.asarray(positions, np.float32), np.asarray(normals, np.float32),
                                  np.asarray(uvs, np.float32), np.asarray(indices, np.uint32),
                                  np.asarray(colors, np.float32)))
        return parts


class Wall:
    """Wall space of one footprint edge; start is the perimeter distance of p, so texture u continues round corners.

    hidden lists (u0, u1, lo, hi): stretches where another piece stands in front of the wall, or shares its plane
    and wins, between heights lo and hi. Nothing is built there, so coincident walls never fight. blind lists
    stretches with a neighbour just in front: built, but without windows or mouldings."""

    def __init__(self, p, q, start=0.0, shade=None, hidden=(), blind=()):
        self.p = (float(p[0]), float(p[1]))
        dx, dz = float(q[0] - p[0]), float(q[1] - p[1])
        self.length = math.hypot(dx, dz)
        self.d = (dx / self.length, dz / self.length) if self.length > 1e-9 else (1.0, 0.0)
        self.n = (-self.d[1], self.d[0])
        self.start = start
        self.shade = shade or (lambda u, v: WHITE)
        self.hidden = list(hidden)
        self.blind = list(blind)
        self.detail = None  # Overrides the look's detail level for this wall's openings.

    def at(self, u, v, w=0.0):
        (px, pz), (dx, dz), (nx, nz) = self.p, self.d, self.n
        return (px + dx * u - nx * w, v, pz + dz * u - nz * w)

    def covered(self, u0, u1, v0, v1):
        """Whether a rectangle of the wall overlaps a hidden stretch."""
        return any(a < u1 and u0 < b and lo < v1 and v0 < hi for a, b, lo, hi in self.hidden)

    def face(self, mesh, material, u0, u1, v0, v1, w=0.0, facing=1, shade=True, colour=None):
        """Facade-parallel rectangle at depth w, facing out (facing=1) or in (-1)."""
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        points = [self.at(u, v, w) for u, v in corners]
        normal = (self.n[0] * facing, 0.0, self.n[1] * facing)
        uvs = [(self.start + u, v) for u, v in corners]
        colors = colour or ([self.shade(u, v) for u, v in corners] if shade else WHITE)
        mesh.poly(material, points, normal, uvs, colors)

    def side(self, mesh, material, u, v0, v1, w0, w1, facing, shade=False, colour=WHITE):
        """Rectangle across the wall at u, facing along the wall (facing=1) or back (-1)."""
        points = [self.at(u, v0, w0), self.at(u, v0, w1), self.at(u, v1, w1), self.at(u, v1, w0)]
        uvs = [(w0, v0), (w1, v0), (w1, v1), (w0, v1)]
        colors = [self.shade(u, v) for v in (v0, v0, v1, v1)] if shade else colour
        mesh.poly(material, points, (self.d[0] * facing, 0.0, self.d[1] * facing), uvs, colors)

    def flat(self, mesh, material, u0, u1, v, w0, w1, facing, shade=False, colour=WHITE):
        """Horizontal rectangle at height v, facing up (facing=1) or down (-1)."""
        points = [self.at(u0, v, w0), self.at(u1, v, w0), self.at(u1, v, w1), self.at(u0, v, w1)]
        uvs = [(self.start + u0, -w0), (self.start + u1, -w0), (self.start + u1, -w1), (self.start + u0, -w1)]
        colors = [self.shade(u, v) for u in (u0, u1, u1, u0)] if shade else colour
        mesh.poly(material, points, (0.0, float(facing), 0.0), uvs, colors)

    def box(self, mesh, material, u0, u1, v0, v1, w0, w1=0.0, faces="ftblr", colour=WHITE):
        """Box between depths w0 < w1 (usually in front of the wall, w1 = 0): front, top, bottom, left, right and
        back (k) faces, toned as baked occlusion."""
        if "f" in faces:
            self.face(mesh, material, u0, u1, v0, v1, w0, shade=False, colour=dim(colour, TONE["f"]))
        if "k" in faces:
            self.face(mesh, material, u0, u1, v0, v1, w1, -1, shade=False, colour=dim(colour, TONE["k"]))
        if "t" in faces:
            self.flat(mesh, material, u0, u1, v1, w0, w1, 1, colour=dim(colour, TONE["t"]))
        if "b" in faces:
            self.flat(mesh, material, u0, u1, v0, w0, w1, -1, colour=dim(colour, TONE["b"]))
        if "l" in faces:
            self.side(mesh, material, u0, v0, v1, w0, w1, -1, colour=dim(colour, TONE["l"]))
        if "r" in faces:
            self.side(mesh, material, u1, v0, v1, w0, w1, 1, colour=dim(colour, TONE["r"]))

    def hole(self, mesh, material, u0, u1, v0, v1, depth, sides="lrtb", w0=0.0):
        """The reveals of a recessed opening, facing into it, tinted like the wall and darker towards the back."""
        far = lambda u, v: dim(self.shade(u, v), 0.78)
        near = self.shade
        def tone(points):  # points: [(u, v, w)]
            return [near(u, v) if abs(w - w0) < 1e-9 else far(u, v) for u, v, w in points]
        if "l" in sides:
            pts = [(u0, v0, w0), (u0, v0, depth), (u0, v1, depth), (u0, v1, w0)]
            mesh.poly(material, [self.at(*p) for p in pts], (self.d[0], 0.0, self.d[1]),
                      [(w, v) for _, v, w in pts], tone(pts))
        if "r" in sides:
            pts = [(u1, v0, w0), (u1, v0, depth), (u1, v1, depth), (u1, v1, w0)]
            mesh.poly(material, [self.at(*p) for p in pts], (-self.d[0], 0.0, -self.d[1]),
                      [(w, v) for _, v, w in pts], tone(pts))
        if "b" in sides:
            pts = [(u0, v0, w0), (u1, v0, w0), (u1, v0, depth), (u0, v0, depth)]
            mesh.poly(material, [self.at(*p) for p in pts], (0.0, 1.0, 0.0),
                      [(self.start + u, -w) for u, _, w in pts], tone(pts))
        if "t" in sides:
            pts = [(u0, v1, w0), (u1, v1, w0), (u1, v1, depth), (u0, v1, depth)]
            mesh.poly(material, [self.at(*p) for p in pts], (0.0, -1.0, 0.0),
                      [(self.start + u, -w) for u, _, w in pts], tone(pts))

    def prism(self, mesh, material, outline, w0, w1, colour=WHITE, back=False):
        """A convex facade-parallel polygon [(u, v)] extruded from depth w0 (front) to w1: front face and sides."""
        front = [self.at(u, v, w0) for u, v in outline]
        mesh.poly(material, front, (self.n[0], 0.0, self.n[1]), [(self.start + u, v) for u, v in outline],
                  dim(colour, TONE["f"]))
        if back:
            mesh.poly(material, [self.at(u, v, w1) for u, v in outline], (-self.n[0], 0.0, -self.n[1]),
                      [(self.start + u, v) for u, v in outline], dim(colour, TONE["k"]))
        for (ua, va), (ub, vb) in zip(outline, outline[1:] + outline[:1]):
            du, dv = ub - ua, vb - va
            length = math.hypot(du, dv)
            if length < 1e-6:
                continue
            # Outward in wall space is (dv, -du) for a counter-clockwise outline (u right, v up).
            ou, ov = dv / length, -du / length
            normal = (self.d[0] * ou, ov, self.d[1] * ou)
            tone = TONE["t"] if ov > 0.5 else TONE["b"] if ov < -0.5 else TONE["l"]
            pts = [self.at(ua, va, w0), self.at(ub, vb, w0), self.at(ub, vb, w1), self.at(ua, va, w1)]
            mesh.poly(material, pts, normal, [(0.0, 0.0), (length, 0.0), (length, w1 - w0), (0.0, w1 - w0)],
                      dim(colour, tone))


@dataclass
class Opening:
    kind: str  # "window", "french", "door", "shop", or "curtain".
    u0: float
    u1: float
    v0: float
    v1: float
    balcony: bool = False
    floor: float = 0.0  # y of the floor the opening belongs to.
    level: int = 0  # Floor index, 0 at the bottom.
    head: str = ""  # Decoration above: "pediment", "segment", "hood", "keystone", or "".
    arch: bool = False  # Round-headed: the top half-width is a semicircle.
    rail: bool = False  # A French window's railing across the opening.
    cols: int = 1  # Panes across, split by mullions.
    transom: float = None  # Height of a transom bar above v0, or None.


def bays(length, width, margin):
    """Bay intervals across a wall, keeping margin clear at both ends; none when the wall is too short."""
    room = length - 2 * margin
    if room < 0.9:
        return []
    n = max(1, round(room / width))
    step = room / n
    return [(margin + k * step, margin + (k + 1) * step) for k in range(n)]


def layout(length, floors, look, rng, door=None, shop=False, blank=-math.inf, free=None):
    """Openings of one wall. floors are (y0, y1, kind) with kind "ground" or "upper"; door is the u the main door
    should be nearest to (None for no door); shop puts shopfronts in the other ground floor bays; openings below
    blank (a neighbour's roof) and those free(opening) rejects are left out."""
    margin = max(0.55, look["reveal"] + 0.35, look.get("margin", 0.0))
    spans = bays(length, look["bay"], margin)
    if not spans:
        return []
    group = max(1, int(look.get("group", 1)))
    if group > 1 and len(spans) >= group:
        # Several bays share one opening split by mullions: far fewer holes on very long facades.
        n = len(spans) // group
        merged = [(spans[k * group][0], spans[min((k + 1) * group, len(spans)) - 1][1]) for k in range(n)]
        if len(spans) % group:
            merged[-1] = (merged[-1][0], spans[-1][1])
        spans = merged
    openings = []
    door_bay = None
    if door is not None:
        door_bay = min(range(len(spans)), key=lambda k: abs((spans[k][0] + spans[k][1]) / 2 - door))
    balconies = set()
    if rng.random() < look["balconies"] and len(spans) >= 2:
        # Balconies stack on a few axes: the middle, the ends, or (short walls only) every other bay.
        n = len(spans)
        pattern = rng.choice(["centre", "ends", "axes"] + (["alternate"] if n <= 6 else []))
        balconies = ({n // 2} | ({(n - 1) // 2} if n % 2 == 0 else set()) if pattern == "centre"
                     else set(range(n % 2 == 0, n, 2)) if pattern == "alternate"
                     else {n // 4, n - 1 - n // 4} if pattern == "axes" else {0, n - 1})
    rails = rng.random() < look.get("rails", 0.0)
    curtain = look.get("curtain", False)
    classical = look.get("archetype") == "classical"
    uppers = [f for f in floors if f[2] == "upper"]
    for level, (y0, y1, kind) in enumerate(floors):
        if y0 < blank + 0.3:
            continue
        top = uppers and (y0, y1, kind) == uppers[-1] and len(uppers) >= 3
        for k, (b0, b1) in enumerate(spans):
            room = b1 - b0
            mid = (b0 + b1) / 2
            cols = max(1, round(room / look["bay"])) * (2 if classical or room > 2.2 else 1)
            if kind == "ground":
                if k == door_bay:
                    w = min(look["door_width"], room - 0.4)
                    h = min(look["door_height"], y1 - y0 - 0.6)
                    if w >= 0.8 and h >= 1.9:
                        openings.append(Opening("door", mid - w / 2, mid + w / 2, y0, y0 + h, floor=y0,
                                                level=level, arch=look.get("arches", False) and h > w / 2 + 2.0))
                    continue
                if shop:
                    w = room - (0.5 if not curtain else 0.15)
                    v1 = y1 - (0.75 if not curtain else 0.2)
                    if w >= 1.2 and v1 - y0 >= 2.2:
                        openings.append(Opening("shop", mid - w / 2, mid + w / 2, y0 + 0.4, v1, floor=y0,
                                                level=level, cols=max(1, round(w / 1.5)), transom=v1 - y0 - 0.85))
                    continue
                sill = look.get("ground_sill", 1.0)
            else:
                sill = look["sill"]
            if curtain:
                # Floor-to-ceiling glass between slab bands; the transom marks the spandrel panel at the slab.
                w = room - 0.12
                v0, v1 = y0 + 0.16, y1 - 0.16
                if w < 0.5 or v1 - v0 < 1.5:
                    continue
                openings.append(Opening("curtain", mid - w / 2, mid + w / 2, v0, v1, floor=y0, level=level,
                                        cols=max(1, round(room / look["bay"])), transom=min(0.75, v1 - v0 - 1.2)))
                continue
            w = min(look["window_width"] * (room / look["bay"] if group > 1 else 1.0), room - 0.5)
            if w < 0.55:
                continue
            french = kind == "upper" and k in balconies and y0 > floors[0][1] - 0.1 and not top
            rail = kind == "upper" and rails and not french and level == 1 and classical
            height = look["window_height"] * (0.82 if top and classical else 1.0)
            v0 = y0 + (0.1 if french or rail else sill)
            v1 = min(y0 + sill + height, y1 - 0.35 - (0.3 if classical and kind == "upper" else 0.0))
            if v1 - v0 < 0.7:
                continue
            arch = kind == "ground" and look.get("arches", False) and v1 - v0 > w / 2 + 0.9
            head = ""
            if classical and kind == "upper":
                heads = look.get("heads", ())
                head = heads[min(level - 1, len(heads) - 1)] if heads else ""
                if head == "pediment" and look.get("alternate") and k % 2:
                    head = "segment"  # Triangular and segmental pediments taking turns, as on many palazzi.
                if top:
                    head = ""
            elif classical and kind == "ground" and not arch:
                head = "keystone"
            transom = None
            if v1 - v0 > 1.65 and look.get("transoms", True):
                transom = (v1 - v0) * 0.72 if classical or french or rail else None
            openings.append(Opening("french" if french or rail else "window", mid - w / 2, mid + w / 2, v0, v1,
                                    balcony=french, floor=y0, level=level, head=head, arch=arch, rail=rail,
                                    cols=cols if w > 0.95 else 1, transom=transom))
    if free is not None:
        openings = [o for o in openings if free(o)]
    return openings


def extent(o, look):
    """The facade rectangle an opening's details take up: (u0, u1, v0, v1), for keeping walls' hidden parts and
    neighbouring pieces clear."""
    s = look.get("surround", 0.0)
    above = {"pediment": 0.75, "segment": 0.7, "hood": 0.45, "keystone": 0.12}.get(o.head, 0.0)
    if o.kind == "shop":
        return o.u0 - 0.15, o.u1 + 0.15, o.v0 - 0.4, o.v1 + 0.75
    if o.kind == "door":
        return o.u0 - 0.5, o.u1 + 0.5, o.v0, o.v1 + 0.7
    return o.u0 - s - 0.12, o.u1 + s + 0.12, o.v0 - 0.12, o.v1 + s + above


def wall(mesh, w, base, profile, openings, look, lod, plinth=None, levels=(), zones=()):
    """One wall: the surface from base up to profile [(t, y)] (t from 0 to 1 along the wall), with openings and
    their details. Every row and the sloped top share one set of u breaks, so no edge ends mid-way along another;
    levels are extra heights to break at, shared with the neighbouring walls so corners match. zones are
    (top, material) bands from the bottom (plinth, rusticated ground floor) under the main wall material.

    lod 0 cuts the openings and builds their details; lod 1 lays flat window quads on a plain wall; lod 2 only the
    glass."""
    L = w.length
    top = min(y for _, y in profile)
    openings = [o for o in openings if o.v1 <= top - 0.2 and o.u0 >= 0 and o.u1 <= L]
    zones = sorted(set(zones) | ({plinth} if plinth else set()))
    us = {0.0, L, *(t * L for t, _ in profile)}
    vs = {base, top, *(v for v in levels if base < v < top), *(z[0] for z in zones if base < z[0] < top)}
    for a, b, lo, hi in w.hidden:
        us |= {min(max(a, 0.0), L), min(max(b, 0.0), L)}
        vs |= {v for v in (lo, hi) if base < v < top}
    if lod == 0:
        us |= {o.u0 for o in openings} | {o.u1 for o in openings}
        vs |= {o.v0 for o in openings} | {o.v1 for o in openings}
    us = sorted(us)
    _surface(mesh, w, us, sorted(vs), openings if lod == 0 else [], look, zones)
    _gable(mesh, w, us, top, profile, look, levels)
    for o in openings:
        if lod == 0:
            _opening(mesh, w, o, look)
        elif lod == 1:
            _flat_opening(mesh, w, o, look)
        else:
            material = look["door"] if o.kind == "door" else look["glass"]
            w.face(mesh, material, o.u0, o.u1, o.v0, o.v1, -0.02, shade=False)


def _surface(mesh, w, us, vs, openings, look, zones):
    for v0, v1 in zip(vs, vs[1:]):
        vm = (v0 + v1) / 2
        material = next((m for t, m in zones if v1 <= t + 1e-6), look["wall"])
        for u0, u1 in zip(us, us[1:]):
            um = (u0 + u1) / 2
            if any(o.u0 < um < o.u1 and o.v0 < vm < o.v1 for o in openings):
                continue
            if any(a < um < b and lo < vm < hi for a, b, lo, hi in w.hidden):
                continue
            w.face(mesh, material, u0, u1, v0, v1)


def _gable(mesh, w, us, top, profile, look, levels=()):
    """The wall above the opening grid, following the roof: gables and sloped wall tops. Each segment's sides
    break at the shared levels too, zipped into triangles, so corners meet the next wall's rows exactly."""
    ts, ys = [t * w.length for t, _ in profile], [y for _, y in profile]
    for u0, u1 in zip(us, us[1:]):
        y0, y1 = float(np.interp(u0, ts, ys)), float(np.interp(u1, ts, ys))
        if max(y0, y1) - top < 1e-6 or u1 - u0 < 1e-6:
            continue
        um = (u0 + u1) / 2
        if any(a < um < b and lo <= top + 1e-6 and hi >= max(y0, y1) - 1e-6 for a, b, lo, hi in w.hidden):
            continue
        left = [top, *sorted(v for v in levels if top < v < y0 - 1e-6), y0]
        right = [top, *sorted(v for v in levels if top < v < y1 - 1e-6), y1]
        i = j = 0
        while i < len(left) - 1 or j < len(right) - 1:
            if j == len(right) - 1 or i < len(left) - 1 and left[i + 1] <= right[j + 1]:
                corners = [(u0, left[i]), (u1, right[j]), (u0, left[i + 1])]
                i += 1
            else:
                corners = [(u0, left[i]), (u1, right[j]), (u1, right[j + 1])]
                j += 1
            mesh.poly(look["wall"], [w.at(u, v) for u, v in corners], (w.n[0], 0.0, w.n[1]),
                      [(w.start + u, v) for u, v in corners], [w.shade(u, v) for u, v in corners])


# Openings at LOD0.

def _opening(mesh, w, o, look):
    detail = w.detail or look.get("detail", "full")
    if detail != look.get("detail", "full"):
        look = dict(look, detail=detail)
    full = detail == "full"
    lean = detail == "lean"
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    if o.kind in ("window", "french"):
        depth = look["reveal"]
    elif o.kind == "door":
        depth = look.get("door_reveal", 0.35)
    elif o.kind == "curtain":
        depth = look.get("curtain_reveal", 0.12)
    else:
        depth = 0.18
    spring = o.v1 - (o.u1 - o.u0) / 2 if o.arch else o.v1
    # Reveals: square openings get all four, arches the sides up to the springing and a curved soffit.
    w.hole(mesh, look["wall"], o.u0, o.u1, o.v0, spring, depth, "lrb" if o.arch else "lrtb")
    if o.arch:
        _arch(mesh, w, o, depth, look, spring)
    if o.kind == "door":
        _door(mesh, w, o, depth, look, spring)
    else:
        _glazing(mesh, w, o, depth, look, spring, lean)
    if o.kind == "shop":
        _shopfront(mesh, w, o, look, full)
        return
    if o.kind == "curtain":
        return
    surround = look.get("surround", 0.0)
    if o.kind in ("window", "french") and surround and not lean:
        _surround(mesh, w, o, surround, look, spring)
    if o.kind in ("window", "french") and not lean:
        _head(mesh, w, o, surround, look, full)
    if o.kind == "window" and look.get("sills", True) and not (look.get("sill_course") and o.level == 1):
        _sill(mesh, w, o, surround, look, full and look.get("archetype") == "classical" and o.level == 1)
    if o.kind == "window" and full and look.get("aprons") and o.level == 1 and surround:
        # A raised panel under the piano nobile windows.
        w.box(mesh, trim, o.u0 + 0.05, o.u1 - 0.05, o.floor + 0.2, o.v0 - 0.22, -0.025, 0.0, "ftblr", tc)
    if o.balcony:
        _balcony(mesh, w, o, look, full, lean)
    elif o.rail:
        _railing(mesh, w, o.u0 - 0.04, o.u1 + 0.04, o.v0 + 0.02, o.v0 + 0.95, -0.12, look, False, sides=False)


def _arch(mesh, w, o, depth, look, spring):
    """Round head of an arched opening: wall spandrels fanned from the upper corners and the curved soffit."""
    segments = 8 if look.get("detail") == "full" else 6
    r = (o.u1 - o.u0) / 2
    c = (o.u0 + o.u1) / 2
    arc = [(c - r * math.cos(a), spring + r * math.sin(a)) for a in np.linspace(0, math.pi, segments + 1)]
    half = segments // 2
    for corner, points in (((o.u0, o.v1), arc[:half + 1]), ((o.u1, o.v1), arc[half:][::-1])):
        pts = [corner, *points]
        mesh.poly(look["wall"], [w.at(u, v) for u, v in pts], (w.n[0], 0.0, w.n[1]),
                  [(w.start + u, v) for u, v in pts], [w.shade(u, v) for u, v in pts])
    for (ua, va), (ub, vb) in zip(arc, arc[1:]):
        pts = [w.at(ua, va), w.at(ub, vb), w.at(ub, vb, depth), w.at(ua, va, depth)]
        # The soffit faces in towards the arch's centre.
        mu, mv = (ua + ub) / 2 - c, (va + vb) / 2 - spring
        k = math.hypot(mu, mv) or 1.0
        normal = (-w.d[0] * mu / k, -mv / k, -w.d[1] * mu / k)
        colour = dim(w.shade(ua, va), 0.85)
        mesh.poly(look["wall"], pts, normal, [(ua, 0.0), (ub, 0.0), (ub, depth), (ua, depth)], colour)
    if look.get("surround") and look.get("detail") == "full":
        # An archivolt: a raised band round the arch, with a keystone.
        s, p = look["surround"], 0.05
        outer = [(c - (r + s) * math.cos(a), spring + (r + s) * math.sin(a)) for a in np.linspace(0, math.pi, segments + 1)]
        tc = look.get("trim_tint", WHITE)
        for (ia, oa), (ib, ob) in zip(zip(arc, outer), zip(arc[1:], outer[1:])):
            pts = [w.at(*ia, -p), w.at(*oa, -p), w.at(*ob, -p), w.at(*ib, -p)]
            mesh.poly(look["trim"], pts, (w.n[0], 0.0, w.n[1]), [(w.start + u, v) for u, v in (ia, oa, ob, ib)],
                      dim(tc, TONE["f"]))
            for (ua, va), (ub, vb) in ((oa, ob), (ib, ia)):
                pts = [w.at(ua, va, -p), w.at(ub, vb, -p), w.at(ub, vb), w.at(ua, va)]
                mu, mv = (ua + ub) / 2 - c, (va + vb) / 2 - spring
                k = (math.hypot(mu, mv) or 1.0) * (1 if (ua, va) == oa else -1)
                mesh.poly(look["trim"], pts, (w.d[0] * mu / k, mv / k, w.d[1] * mu / k),
                          [(ua, 0.0), (ub, 0.0), (ub, p), (ua, p)], dim(tc, TONE["l"]))
        _keystone(mesh, w, c, o.v1, s, look)


def _keystone(mesh, w, c, v, s, look):
    tc = look.get("trim_tint", WHITE)
    w.prism(mesh, look["trim"], [(c - 0.11, v - 0.04), (c + 0.11, v - 0.04), (c + 0.15, v + s + 0.1),
                                 (c - 0.15, v + s + 0.1)], -0.09, 0.0, tc)


def _panes(o, depth, spring, look):
    """Pane rectangles [(u0, u1, v0, v1)] inside the frame and the bars between them, as boxes."""
    f = look.get("frame_width", FRAME)
    u0, u1, v0, v1 = o.u0 + f, o.u1 - f, o.v0 + f, spring - (f if not o.arch else 0.0)
    cols = max(1, o.cols)
    m = 0.07 if o.kind != "curtain" else 0.06
    xs = [u0 + (u1 - u0) * k / cols for k in range(cols + 1)]
    rows = [v0]
    if o.transom is not None and v0 + 0.5 < o.v0 + o.transom < v1 - 0.3:
        rows.append(o.v0 + o.transom)
    rows.append(v1)
    panes, mullions, transoms = [], [], []
    for a, b in zip(rows, rows[1:]):
        for k, (x0, x1) in enumerate(zip(xs, xs[1:])):
            panes.append((x0 + (m / 2 if k else 0.0), x1 - (m / 2 if k < cols - 1 else 0.0),
                          a + (m / 2 if a != v0 else 0.0), b - (m / 2 if b != v1 else 0.0)))
    for x in xs[1:-1]:
        mullions.append((x - m / 2, x + m / 2, v0, v1))
    for y in rows[1:-1]:
        transoms.append((u0, u1, y - m / 2, y + m / 2))
    return panes, mullions, transoms


def _glazing(mesh, w, o, depth, look, spring, lean):
    """Glass panes at the back of the reveal, one quad each, behind a frame with mullions and transoms."""
    frame, fc = (look["frame"], look.get("frame_tint", WHITE)) if o.kind != "shop" else (look["shop_frame"], WHITE)
    if o.kind == "curtain":
        frame, fc = look["mullion"], WHITE
    fd = FRAME_DEPTH
    f = look.get("frame_width", FRAME)
    panes, mullions, transoms = _panes(o, depth, spring, look)
    spandrel = o.kind == "curtain" and o.transom is not None
    for k, (a, b, c, d) in enumerate(panes):
        # The lowest row of a curtain wall is the opaque spandrel in front of the slab.
        material = look["spandrel"] if spandrel and c < o.v0 + o.transom else look["glass"]
        w.face(mesh, material, a, b, c, d, depth, shade=False)
    if o.arch:
        _fanlight(mesh, w, o, depth, spring, look)
    fw = depth - fd
    # Frame bars: their outer faces touch the reveals, so only fronts and the inner edges are built.
    w.box(mesh, frame, o.u0, o.u0 + f, o.v0, spring, fw, depth, "fr", fc)
    w.box(mesh, frame, o.u1 - f, o.u1, o.v0, spring, fw, depth, "fl", fc)
    w.box(mesh, frame, o.u0 + f, o.u1 - f, o.v0, o.v0 + f, fw, depth, "ft", fc)
    if not o.arch:
        w.box(mesh, frame, o.u0 + f, o.u1 - f, spring - f, spring, fw, depth, "fb", fc)
    if lean and o.kind != "curtain":
        mullions = mullions[:1] if len(mullions) == 1 else []
    for a, b, c, d in mullions:
        w.box(mesh, frame, a, b, c, d, fw - (0.08 if o.kind == "curtain" else 0.0), depth, "flr", fc)
    for a, b, c, d in transoms:
        w.box(mesh, frame, a, b, c, d, fw, depth, "ftb", fc)


def _fanlight(mesh, w, o, depth, spring, look):
    segments = 8 if look.get("detail") == "full" else 6
    r = (o.u1 - o.u0) / 2
    c = (o.u0 + o.u1) / 2
    arc = [(c - r * math.cos(a), spring + r * math.sin(a)) for a in np.linspace(0, math.pi, segments + 1)]
    pts = [(c, spring), *arc]
    mesh.poly(look["glass"], [w.at(u, v, depth) for u, v in pts], (w.n[0], 0.0, w.n[1]),
              [(w.start + u, v) for u, v in pts])
    frame, fc = look["frame"], look.get("frame_tint", WHITE)
    w.box(mesh, frame, o.u0, o.u1, spring - 0.035, spring + 0.035, depth - FRAME_DEPTH, depth, "ftb", fc)


def _door(mesh, w, o, depth, look, spring):
    """A door leaf at the back of a deep reveal with a glazed fanlight above it; a portal and steps in front."""
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    leaf_top = min(spring, o.v0 + 2.25) if not o.arch else spring
    if leaf_top > spring - 0.3:
        leaf_top = spring  # No room for a fanlight: the leaf fills the opening.
    w.face(mesh, look["door"], o.u0, o.u1, o.v0, leaf_top, depth, shade=False)
    if leaf_top < spring:
        w.face(mesh, look["glass"], o.u0 + 0.06, o.u1 - 0.06, leaf_top + 0.08, spring - 0.06, depth, shade=False)
        w.box(mesh, look["frame"], o.u0, o.u1, leaf_top, leaf_top + 0.08, depth - 0.06, depth, "ftb",
              look.get("frame_tint", WHITE))
        w.box(mesh, look["frame"], o.u0, o.u1, spring - 0.06, spring, depth - 0.06, depth, "fb",
              look.get("frame_tint", WHITE))
    if o.arch:
        _fanlight(mesh, w, o, depth, spring, look)
    detail = look.get("detail", "full")
    if look.get("archetype") == "classical" and detail != "lean":
        # Portal: pilasters, an entablature and a cornice slab.
        p, s = 0.12, 0.28
        top = o.v1 + 0.08
        w.box(mesh, trim, o.u0 - s, o.u0, o.v0, top, -p, 0.0, "ftlr", tc)
        w.box(mesh, trim, o.u1, o.u1 + s, o.v0, top, -p, 0.0, "ftlr", tc)
        w.box(mesh, trim, o.u0, o.u1, o.v1, top, -p, 0.0, "fb", tc)
        w.box(mesh, trim, o.u0 - s - 0.05, o.u1 + s + 0.05, top, top + 0.42, -p - 0.03, 0.0, "fblr", tc)
        w.box(mesh, trim, o.u0 - s - 0.15, o.u1 + s + 0.15, top + 0.42, top + 0.56, -p - 0.2, 0.0, "ftblr", tc)
    elif look.get("archetype") in ("stone", "curtain") and detail != "lean":
        # A thin canopy slab over modern entrances.
        w.box(mesh, look["metal"], o.u0 - 0.6, o.u1 + 0.6, o.v1 + 0.25, o.v1 + 0.4, -1.4, 0.0, "ftblr")
    else:
        c = 0.12
        w.box(mesh, trim, o.u0 - c, o.u0, o.v0, o.v1 + c, -0.05, 0.0, "ftlr", tc)
        w.box(mesh, trim, o.u1, o.u1 + c, o.v0, o.v1 + c, -0.05, 0.0, "ftlr", tc)
        w.box(mesh, trim, o.u0, o.u1, o.v1, o.v1 + c, -0.05, 0.0, "ftb", tc)
    # Steps: buried below the floor level, so they meet uneven pavement.
    step = look.get("step", "stone/granite-dark")
    for k in range(look.get("steps", 2)):
        top = o.v0 - 0.16 * k
        w.box(mesh, step, o.u0 - 0.35 - 0.1 * k, o.u1 + 0.35 + 0.1 * k, top - 0.6, top, -0.36 * (k + 1), -0.36 * k,
              "ftlr")


def _surround(mesh, w, o, s, look, spring):
    """A stucco architrave round a window: a raised flat frame; its inner faces continue the reveal."""
    trim, tc, p = look["trim"], look.get("trim_tint", WHITE), 0.05
    if o.arch:
        return
    w.box(mesh, trim, o.u0 - s, o.u0, o.v0, o.v1 + s, -p, 0.0, "flr", tc)
    w.box(mesh, trim, o.u1, o.u1 + s, o.v0, o.v1 + s, -p, 0.0, "flr", tc)
    w.box(mesh, trim, o.u0, o.u1, o.v1, o.v1 + s, -p, 0.0, "fb", tc)
    w.flat(mesh, trim, o.u0 - s, o.u1 + s, o.v1 + s, -p, 0.0, 1, colour=dim(tc, TONE["t"]))
    if o.kind == "french" or not look.get("sills", True):
        w.flat(mesh, trim, o.u0 - s, o.u0, o.v0, -p, 0.0, -1, colour=dim(tc, TONE["b"]))
        w.flat(mesh, trim, o.u1, o.u1 + s, o.v0, -p, 0.0, -1, colour=dim(tc, TONE["b"]))


def _head(mesh, w, o, s, look, full):
    """Decoration over a window: a hood (frieze and cornice slab), a triangular or segmental pediment above the
    hood, or a keystone."""
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    top = o.v1 + s
    if o.head == "keystone":
        _keystone(mesh, w, (o.u0 + o.u1) / 2, o.v1, s, look)
        return
    if o.head not in ("hood", "pediment", "segment"):
        return
    a, b = o.u0 - s, o.u1 + s
    frieze = 0.18 if full else 0.0
    if frieze:
        w.box(mesh, trim, a, b, top, top + frieze, -0.035, 0.0, "flr", tc)
    y = top + frieze
    w.box(mesh, trim, a - 0.08, b + 0.08, y, y + 0.12, -0.16, 0.0, "ftblr", tc)
    y += 0.12
    if o.head == "pediment":
        c, h = (a + b) / 2, min(0.42, (b - a) * 0.22)
        w.prism(mesh, trim, [(a - 0.06, y), (b + 0.06, y), (c, y + h)], -0.14, 0.0, tc)
    elif o.head == "segment" and full:
        c, half = (a + b) / 2, (b - a) / 2 + 0.06
        rise = min(0.3, half * 0.35)
        r = (half ** 2 + rise ** 2) / (2 * rise)
        a0 = math.asin(half / r)
        # The arc runs from right to left over the top, so the outline is counter-clockwise.
        arc = [(c + r * math.sin(t), y + rise - r + r * math.cos(t)) for t in np.linspace(a0, -a0, 7)]
        w.prism(mesh, trim, [(c - half, y), *arc[:-1]], -0.14, 0.0, tc)


def _sill(mesh, w, o, s, look, brackets):
    """A stone sill under the window, wider than its surround, with a drip edge; consoles under it on the piano
    nobile."""
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    a, b = o.u0 - s - 0.06, o.u1 + s + 0.06
    out = look.get("sill_out", 0.09)
    w.box(mesh, trim, a, b, o.v0 - 0.08, o.v0, -out, 0.0, "ftblr", tc)
    if brackets:
        for u in (a + 0.06, b - 0.18):
            w.prism(mesh, trim, [(u, o.v0 - 0.08), (u + 0.12, o.v0 - 0.08), (u + 0.12, o.v0 - 0.3),
                                 (u, o.v0 - 0.3)][::-1], -out + 0.02, 0.0, tc)


def _balcony(mesh, w, o, look, full, lean):
    """A stone slab on consoles with a railing round its edge."""
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    u0, u1, y = o.u0 - 0.45, o.u1 + 0.45, o.floor
    deep = look.get("balcony_depth", 1.0)
    w.box(mesh, trim, u0, u1, y - 0.1, y + 0.08, -deep, 0.0, "ftblr", tc)
    w.box(mesh, trim, u0 - 0.04, u1 + 0.04, y - 0.16, y - 0.1, -deep - 0.04, 0.0, "fblr", tc)
    if not lean:
        for u in np.linspace(u0 + 0.1, u1 - 0.24, 3 if full and u1 - u0 > 2.0 else 2):
            w.prism(mesh, trim, [(u, y - 0.16), (u + 0.14, y - 0.16), (u + 0.14, y - 0.55), (u, y - 0.55)][::-1],
                    -deep * 0.7, 0.0, tc)
    _railing(mesh, w, u0 + 0.03, u1 - 0.03, y + 0.08, y + 1.0, -deep + 0.04, look, full and not lean, sides=True,
             lean=lean)


def _railing(mesh, w, u0, u1, v0, v1, front, look, full, sides, lean=False):
    """Metal railing: top and bottom rails with balusters, along the front and (for balconies) both sides."""
    metal = look["metal"]
    rail = 0.045
    # Balusters: thin bars whose fronts and sides are all that show; sparser as detail drops.
    gap = 0.13 if full else 0.3 if lean else 0.2
    w.box(mesh, metal, u0, u1, v1 - rail, v1, front, front + rail, "ftbk")
    w.box(mesh, metal, u0, u1, v0, v0 + rail, front, front + rail, "ftk")
    n = max(2, int((u1 - u0) / gap))
    for u in np.linspace(u0 + gap / 2, u1 - gap / 2, n):
        w.box(mesh, metal, u - 0.012, u + 0.012, v0 + rail, v1 - rail, front + 0.01, front + 0.034, "flr")
    if not sides:
        return
    for u in (u0, u1 - rail):
        w.box(mesh, metal, u, u + rail, v1 - rail, v1, front, 0.0, "tblr")
        w.box(mesh, metal, u, u + rail, v0, v0 + rail, front, 0.0, "tlr")
        n = max(1, int(-front / gap))
        for t in np.linspace(front + gap / 2, -gap / 2, n):
            w.box(mesh, metal, u + 0.01, u + 0.034, v0 + rail, v1 - rail, t - 0.012, t + 0.012, "lr")


def _shopfront(mesh, w, o, look, full):
    """Fascia with a sign panel over the shop window, and an awning on some buildings."""
    fascia = look.get("fascia", "trim/metal-dark")
    fc = look.get("fascia_tint", WHITE)
    a, b, y0, y1 = o.u0 - 0.1, o.u1 + 0.1, o.v1 + 0.12, o.v1 + 0.62
    w.box(mesh, fascia, a, b, y0, y1, -0.1, 0.0, "ftblr", fc)
    if look.get("sign") and full or look.get("sign") and look.get("detail") == "medium":
        # A sign panel from the shop sign atlas, a different cell per shop.
        cells = look.get("sign_cells", (1, 8))
        k = int(abs(o.u0 * 7.31 + o.v0 * 3.17 + w.start * 1.13) * 1000) % (cells[0] * cells[1])
        cu, cv = k % cells[0], k // cells[0]
        width = min(b - a - 0.4, (y1 - y0 - 0.1) * look.get("sign_aspect", 6.0))
        m = (a + b) / 2
        corners = [(m - width / 2, y0 + 0.05), (m + width / 2, y0 + 0.05), (m + width / 2, y1 - 0.05),
                   (m - width / 2, y1 - 0.05)]
        uvs = [((cu + x) / cells[0], 1 - (cv + 1 - y) / cells[1]) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        mesh.poly(look["sign"], [w.at(u, v, -0.106) for u, v in corners], (w.n[0], 0.0, w.n[1]), uvs, raw=True)
    if look.get("awning") and full or look.get("awning") and look.get("detail") == "medium":
        _awning(mesh, w, o.u0 - 0.05, o.u1 + 0.05, o.v1 + 0.05, look)


def _awning(mesh, w, u0, u1, v, look, out=1.3, drop=0.55, valance=0.22):
    """A sloped fabric awning with a valance and side cheeks; fabric is tinted by vertex colour."""
    material, colour = look["awning"], look.get("awning_tint", WHITE)
    top, low = (v, 0.0), (v - drop, -out)
    slope = (w.n[0] * drop, out, w.n[1] * drop)
    k = math.hypot(drop, out)
    normal = (slope[0] / k, slope[1] / k, slope[2] / k)
    pts = [w.at(u0, *top), w.at(u1, *top), w.at(u1, *low), w.at(u0, *low)]
    mesh.poly(material, pts, normal, [(0, 1), (1, 1), (1, 0), (0, 0)], colour)
    mesh.poly(material, pts[::-1], tuple(-c for c in normal), [(0, 0), (1, 0), (1, 1), (0, 1)], dim(colour, 0.6))
    w.face(mesh, material, u0, u1, v - drop - valance, v - drop, -out, colour=colour)
    w.face(mesh, material, u0, u1, v - drop - valance, v - drop, -out + 0.005, facing=-1, colour=dim(colour, 0.6))
    for u, facing in ((u0, -1), (u1, 1)):
        tri = [w.at(u, *top), w.at(u, *low), w.at(u, v - drop - valance, -out)]
        mesh.poly(material, tri, (w.d[0] * facing, 0.0, w.d[1] * facing), [(0, 1), (1, 0), (1, 0)],
                  dim(colour, 0.85))
        mesh.poly(material, tri[::-1], (-w.d[0] * facing, 0.0, -w.d[1] * facing), [(1, 0), (1, 0), (0, 1)],
                  dim(colour, 0.6))


# Openings at LOD1: flat quads just in front of a plain wall.

def _flat_opening(mesh, w, o, look):
    lean = look.get("detail") == "lean"
    if o.kind == "door":
        w.face(mesh, look["door"], o.u0, o.u1, o.v0, o.v1, -0.02, shade=False)
        return
    if o.kind == "shop":
        w.face(mesh, look.get("fascia", "trim/metal-dark"), o.u0 - 0.1, o.u1 + 0.1, o.v1 + 0.12, o.v1 + 0.62, -0.05,
               shade=False, colour=look.get("fascia_tint", WHITE))
    s = look.get("surround", 0.0)
    if s and not lean and o.kind in ("window", "french") and not o.arch:
        w.face(mesh, look["trim"], o.u0 - s, o.u1 + s, o.v0, o.v1 + s, -0.01, shade=False,
               colour=look.get("trim_tint", WHITE))
    if not lean:
        frame = look["mullion"] if o.kind == "curtain" else look["frame"]
        w.face(mesh, frame, o.u0, o.u1, o.v0, o.v1, -0.02, shade=False, colour=look.get("frame_tint", WHITE)
               if o.kind != "curtain" else WHITE)
    f = 0.06 if not lean else 0.0
    w.face(mesh, look["glass"], o.u0 + f, o.u1 - f, o.v0 + f, o.v1 - f, -0.03, shade=False)
    if o.kind == "window" and look.get("sills", True) and not lean:
        tc = look.get("trim_tint", WHITE)
        w.box(mesh, look["trim"], o.u0 - s - 0.06, o.u1 + s + 0.06, o.v0 - 0.08, o.v0, -0.09, 0.0, "ft", tc)
    if o.balcony:
        # Slab, rails and a few posts: a solid panel would read as a black box from afar.
        u0, u1, y = o.u0 - 0.45, o.u1 + 0.45, o.floor
        deep = look.get("balcony_depth", 1.0)
        w.box(mesh, look["trim"], u0, u1, y - 0.1, y + 0.08, -deep, 0.0, "ftblr", look.get("trim_tint", WHITE))
        w.box(mesh, look["metal"], u0, u1, y + 0.94, y + 1.0, -deep + 0.02, -deep + 0.07, "ftbk")
        w.box(mesh, look["metal"], u0, u1, y + 0.08, y + 0.14, -deep + 0.02, -deep + 0.07, "ftk")
        for u in np.linspace(u0 + 0.03, u1 - 0.07, 5):
            w.box(mesh, look["metal"], u, u + 0.04, y + 0.14, y + 0.94, -deep + 0.02, -deep + 0.06, "fk")


# Mouldings round a ring.

def moulding(mesh, ring, y0, profile, material, colour=WHITE, spans=None, start=0.0):
    """A horizontal moulding round a ring: profile [(out, y)] from bottom to top, out metres in front of the ring
    at height y0 + y, starting and ending at the wall (out 0). spans[k] lists the (t0, t1) parts of edge k to build
    (None: all); corners are mitred where both edges continue, and cut square with an end cap elsewhere."""
    ring = np.asarray(ring, float)
    n = len(ring)
    outs = sorted({o for o, _ in profile})
    rings = {o: offset(ring, o) for o in outs}
    d = np.roll(ring, -1, axis=0) - ring
    lengths = np.hypot(*d.T)
    normals = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(lengths, 1e-12)[:, None]
    spans = spans or [[(0.0, 1.0)] for _ in range(n)]
    # Arc length along the profile, so tops and soffits get unstretched texture.
    arc = [0.0]
    for (oa, ya), (ob, yb) in zip(profile, profile[1:]):
        arc.append(arc[-1] + math.hypot(ob - oa, yb - ya))
    s = start
    for k in range(n):
        j = (k + 1) % n
        if lengths[k] < 1e-6:
            continue
        for t0, t1 in spans[k]:
            if t1 - t0 < 1e-6:
                continue
            # Corners are mitred only where the neighbouring edge's moulding meets this one.
            mitre0 = t0 <= 1e-9 and any(b >= 1 - 1e-9 for _, b in spans[k - 1])
            mitre1 = t1 >= 1 - 1e-9 and any(a <= 1e-9 for a, _ in spans[j])

            def point(t, o, mitre, corner):
                if mitre:
                    return rings[o][corner]
                return ring[k] + t * d[k] + o * normals[k]
            ends = [(t0, mitre0, k), (t1, mitre1, j)]
            line = [[point(t, o, m, c) for t, m, c in ends] for o, _ in profile]
            nx, nz = normals[k]
            for i in range(len(profile) - 1):
                (oa, ya), (ob, yb) = profile[i], profile[i + 1]
                do, dy = ob - oa, yb - ya
                length = math.hypot(do, dy)
                if length < 1e-9:
                    continue
                ny, no = -do / length, dy / length
                normal = (nx * no, ny, nz * no)
                tone = TONE["t"] if ny > 0.5 else TONE["b"] if ny < -0.5 else 1.0
                (a0, a1), (b0, b1) = line[i], line[i + 1]
                pts = [(a0[0], y0 + ya, a0[1]), (a1[0], y0 + ya, a1[1]), (b1[0], y0 + yb, b1[1]),
                       (b0[0], y0 + yb, b0[1])]
                u0, u1 = s + t0 * lengths[k], s + t1 * lengths[k]
                mesh.poly(material, pts, normal, [(u0, y0 + arc[i]), (u1, y0 + arc[i]), (u1, y0 + arc[i + 1]),
                                                  (u0, y0 + arc[i + 1])], dim(colour, tone))
            for t, mitre, facing in ((t0, mitre0, -1), (t1, mitre1, 1)):
                if not mitre:
                    _cap(mesh, material, ring[k] + t * d[k], normals[k], d[k] / lengths[k] * facing, y0, profile,
                         colour)
        s += lengths[k]


def _cap(mesh, material, base, normal, along, y0, profile, colour):
    """The cross-section of a moulding where it stops: triangulated, since profiles are not convex."""
    section = list(profile)
    if section[0][0] != 0.0 or section[-1][0] != 0.0:
        section = [(0.0, profile[0][1]), *section, (0.0, profile[-1][1])]
    # (out, y) as a 2D ring; poly fixes the winding from the normal.
    points, triangles = triangulate(np.array(section, float))
    for t in triangles:
        tri = [(base[0] + normal[0] * points[i][0], y0 + points[i][1], base[1] + normal[1] * points[i][0])
               for i in t]
        mesh.poly(material, tri, (float(along[0]), 0.0, float(along[1])), [tuple(points[i]) for i in t],
                  dim(colour, TONE["l"]))


def band(mesh, ring, y0, y1, out, material, start=0.0, spans=None, colour=WHITE):
    """A plain rectangular moulding (string course, coping): front, top and bottom faces, mitred."""
    moulding(mesh, ring, y0, [(0.0, 0.0), (out, 0.0), (out, y1 - y0), (0.0, y1 - y0)], material, colour, spans,
             start)
