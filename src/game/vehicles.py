"""Procedural vehicles: cars, a van, a taxi, and a Berlin double-decker bus, as instanced library models.

Frame (as for every library model): metres, origin on the ground at the centre of the footprint, y up, +z the front
(the direction of travel). The vehicle's right-hand side, where bus doors are, is −x. Bodies are lofts of rounded
cross-sections along z, with wheel arches cut by raising the floor line over each axle; windows are the body's own
faces between the pillars, so glass sits flush. Paint is the near-white "vehicle/paint" with the body colour in the
vertex colours; each type also comes in a few colours ("vehicle/sedan-red"…, see VEHICLES).
"""

import math

import numpy as np

from .types import Collider

# Linear RGB body colours.
COLOURS = dict(silver=(0.42, 0.43, 0.45), black=(0.018, 0.019, 0.021), white=(0.78, 0.78, 0.76), blue=(0.02, 0.05, 0.15),
               red=(0.42, 0.025, 0.025), grey=(0.11, 0.115, 0.12), green=(0.03, 0.075, 0.05), ivory=(0.8, 0.65, 0.45),
               yellow=(0.9, 0.52, 0.0))
LAMP_FRONT, LAMP_REAR, LAMP_AMBER = (0.6, 0.62, 0.65, 1), (0.4, 0.01, 0.01, 1), (0.9, 0.35, 0.02, 1)
PLASTIC = "vehicle/plastic"

# Car bodies. Keypoints are (metres from the rear end, height): top is the centreline silhouette (bumper, deck or
# tailgate, rear window, roof, windscreen, bonnet, nose), belt the shoulder line under the side windows, bottom the
# sills and bumper undersides; width and roof are half-widths as fractions of W / 2. Zones in metres from the rear:
# rear window rw, windscreen ws (top, base), side glass, and B pillars; head, tail, and grille are height ranges.
TYPES = dict(
    # A Golf-sized hatchback.
    hatchback=dict(L=4.28, W=1.79, H=1.46, axles=(0.77, 3.39), r=0.32, tyre=0.225,
                   top=[(0, 0.56), (0.03, 0.93), (0.07, 1.04), (0.3, 1.39), (0.55, 1.45), (1.6, 1.46), (2.2, 1.43),
                        (2.45, 1.37), (3.25, 0.99), (3.6, 0.93), (3.95, 0.84), (4.15, 0.73), (4.25, 0.6), (4.28, 0.52)],
                   belt=[(0, 0.98), (0.4, 1.0), (2.0, 1.0), (3.25, 0.96), (4.28, 0.8)],
                   bottom=[(0, 0.36), (0.08, 0.24), (0.35, 0.18), (3.9, 0.17), (4.2, 0.25), (4.28, 0.31)],
                   width=[(0, 0.86), (0.07, 0.97), (0.3, 1.0), (3.9, 1.0), (4.15, 0.95), (4.28, 0.82)], roof=0.75,
                   rw=(0.07, 0.36), ws=(2.42, 3.22), glass=(0.6, 3.2), pillars=[2.0], head=(0.68, 0.8), tail=(0.86, 1.0),
                   grille=(0.36, 0.6)),
    # A Passat-sized saloon.
    sedan=dict(L=4.77, W=1.83, H=1.46, axles=(1.05, 3.84), r=0.33, tyre=0.235,
               top=[(0, 0.62), (0.03, 0.88), (0.12, 1.0), (0.45, 1.04), (0.62, 1.07), (1.22, 1.39), (1.5, 1.45),
                    (2.3, 1.46), (2.7, 1.43), (2.88, 1.38), (3.66, 0.99), (4.0, 0.93), (4.4, 0.86), (4.65, 0.74),
                    (4.74, 0.6), (4.77, 0.5)],
               belt=[(0, 0.98), (0.6, 1.04), (2.4, 1.0), (3.66, 0.95), (4.77, 0.8)],
               bottom=[(0, 0.37), (0.12, 0.23), (0.5, 0.17), (4.3, 0.16), (4.68, 0.24), (4.77, 0.3)],
               width=[(0, 0.86), (0.1, 0.96), (0.5, 1.0), (4.3, 1.0), (4.6, 0.95), (4.77, 0.82)], roof=0.73,
               rw=(0.62, 1.24), ws=(2.86, 3.64), glass=(1.3, 3.62), pillars=[2.42], head=(0.67, 0.8), tail=(0.86, 0.98),
               grille=(0.35, 0.58)),
    # A Tiguan-sized SUV with dark wheel-arch and sill cladding.
    suv=dict(L=4.51, W=1.86, H=1.67, axles=(0.9, 3.58), r=0.36, tyre=0.235,
             top=[(0, 0.74), (0.04, 1.08), (0.1, 1.17), (0.3, 1.57), (0.5, 1.64), (2.5, 1.65), (2.75, 1.6), (2.9, 1.54),
                  (3.45, 1.13), (3.8, 1.07), (4.2, 0.99), (4.4, 0.89), (4.48, 0.77), (4.51, 0.68)],
             belt=[(0, 1.14), (0.4, 1.17), (2.0, 1.15), (3.45, 1.1), (4.51, 0.95)],
             bottom=[(0, 0.44), (0.1, 0.3), (0.4, 0.23), (4.0, 0.22), (4.4, 0.31), (4.51, 0.38)],
             width=[(0, 0.88), (0.08, 0.97), (0.3, 1.0), (4.1, 1.0), (4.38, 0.95), (4.51, 0.84)], roof=0.78,
             rw=(0.1, 0.4), ws=(2.88, 3.43), glass=(0.45, 3.4), pillars=[1.95], head=(0.84, 0.96), tail=(1.0, 1.14),
             grille=(0.5, 0.78), cladding=0.2),
    # A Sprinter-sized panel van: windows in the cab doors only.
    van=dict(L=5.93, W=2.02, H=2.6, axles=(1.27, 4.93), r=0.36, tyre=0.235,
             top=[(0, 2.48), (0.04, 2.57), (0.12, 2.6), (4.2, 2.6), (4.42, 2.54), (4.55, 2.45), (5.2, 1.3), (5.5, 1.19),
                  (5.78, 1.06), (5.9, 0.92), (5.93, 0.82)],
             belt=[(0, 1.15), (4.3, 1.15), (5.2, 1.22), (5.93, 0.8)],
             bottom=[(0, 0.5), (0.08, 0.4), (0.3, 0.33), (5.6, 0.31), (5.88, 0.38), (5.93, 0.44)],
             width=[(0, 0.97), (0.05, 1.0), (5.5, 1.0), (5.8, 0.96), (5.93, 0.86)], roof=0.94, roof_radius=0.1,
             rw=None, ws=(4.5, 5.18), glass=(4.42, 5.16), pillars=[], head=(0.9, 1.04), tail=(0.75, 1.25),
             grille=(0.5, 0.86), cargo=True),
)
# An E-class-sized ivory taxi: the saloon stretched to 4.92 m, with a roof sign.
TYPES["taxi"] = dict(TYPES["sedan"], L=4.92, W=1.85, H=1.47, axles=(1.05, 3.99), sign=True)
for _key in ("top", "belt", "bottom", "width"):
    TYPES["taxi"][_key] = [(z * 4.92 / 4.77, y) for z, y in TYPES["sedan"][_key]]
for _key in ("rw", "ws", "glass"):
    TYPES["taxi"][_key] = tuple(z * 4.92 / 4.77 for z in TYPES["sedan"][_key])
TYPES["taxi"]["pillars"] = [2.5]

VEHICLES = {  # name: (type, colour)
    "vehicle/sedan": ("sedan", "silver"), "vehicle/hatchback": ("hatchback", "red"), "vehicle/suv": ("suv", "black"),
    "vehicle/van": ("van", "white"), "vehicle/taxi": ("taxi", "ivory"), "vehicle/bus": ("bus", "yellow"),
    **{f"vehicle/sedan-{c}": ("sedan", c) for c in ("black", "white", "blue", "red", "grey")},
    **{f"vehicle/hatchback-{c}": ("hatchback", c) for c in ("white", "grey", "blue", "silver", "green")},
    **{f"vehicle/suv-{c}": ("suv", c) for c in ("white", "grey", "silver", "blue")},
    **{f"vehicle/van-{c}": ("van", c) for c in ("silver", "grey")},
}
# assets imports this module at its end for VEHICLES, so its helpers come after the tables (no import cycle).
from .assets import Mesh, Model, _unit, box, lathe, quad  # noqa: E402

# Ring point counts per LOD: bottom flat, bottom corner, lower side, upper side (window), roof corner, roof.
RINGS = ((2, 2, 4, 3, 4, 4), (1, 1, 2, 2, 2, 2), (1, 1, 1, 1, 1, 1))


def _interp(points, s):
    xs, ys = zip(*points)
    return np.interp(s, xs, ys)


def _curve(points, s):
    """A smooth (Catmull-Rom) curve through keypoints, sampled at s."""
    p = np.asarray(points, float)
    p = np.concatenate(([2 * p[0] - p[1]], p, [2 * p[-1] - p[-2]]))
    t = np.linspace(0, 1, 24)[:, None]
    dense = []
    for a, b, c, d in zip(p, p[1:], p[2:], p[3:]):
        dense.append(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2 + (3 * b - a - 3 * c + d) * t ** 3))
    dense = np.concatenate(dense)
    order = np.argsort(dense[:, 0], kind="stable")
    return np.interp(s, dense[order, 0], dense[order, 1])


def _arc(cx, cy, r, a0, a1, count):
    a = np.linspace(a0, a1, count)
    return np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a)))


def _section(yb, belt, top, half, roof_half, roof_radius, counts):
    """Right half of a body cross-section from the bottom centre up the +x side to the top centre, and each point's
    zone: 0 lower body, 1 window side, 2 roof corner (rails and pillars), 3 roof (or windscreen, bonnet, deck)."""
    flat, corner, lower, side, rail, roof = counts
    rb = min(0.1, (belt - yb) * 0.3)
    x0 = half * 0.96
    points = [np.array([[0.0, yb]]), np.column_stack((np.linspace(0, x0 - rb, flat + 1)[1:], np.full(flat, yb))),
              _arc(x0 - rb, yb + rb, rb, -math.pi / 2, 0, corner + 1)[1:]]
    t = np.linspace(0, 1, lower + 1)[1:]
    points.append(np.column_stack((half * (0.96 + 0.04 * np.sin(np.pi * t * 0.85)), yb + rb + (belt - yb - rb) * t)))
    shoulder = half * (0.96 + 0.04 * math.sin(np.pi * 0.85))
    c = float(np.clip((top - belt) / 0.25, 0, 1))  # 0 on the bonnet or deck, 1 under the roof.
    edge = shoulder * 0.985 + (roof_half - shoulder * 0.985) * c
    rr = 0.03 + (roof_radius - 0.03) * c
    rise = max(top - belt - rr, 0.004 * side)
    t = np.linspace(0, 1, side + 1)[1:]
    points.append(np.column_stack((shoulder + (edge - shoulder) * t, belt + rise * t)))
    points.append(_arc(edge - rr, belt + rise, rr, 0, math.pi / 2, rail + 1)[1:])
    t = np.linspace(0, 1, roof + 1)[1:]
    crown = 0.015 + 0.03 * c
    points.append(np.column_stack(((edge - rr) * (1 - t), belt + rise + rr + crown * (1 - (1 - t) ** 2))))
    zones = [0] * (1 + flat + corner + lower) + [1] * side + [2] * rail + [3] * roof
    return np.concatenate(points), np.array(zones)


def _loft(stations, rings):
    """Grid positions (stations × ring points) and smooth vertex normals from the surrounding quads' areas."""
    p = np.array([[[x, y, z] for x, y in ring] for z, ring in zip(stations, rings)])
    ns, nr = p.shape[:2]
    normals = np.zeros_like(p)
    j1 = (np.arange(nr) + 1) % nr
    for i in range(ns - 1):
        a, b, c, d = p[i], p[i, j1], p[i + 1, j1], p[i + 1]
        face = np.cross(b - a, c - a) + np.cross(c - a, d - a)
        for target, cols in ((i, np.arange(nr)), (i, j1), (i + 1, np.arange(nr)), (i + 1, j1)):
            np.add.at(normals[target], cols, face)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    fallback = np.stack((p[..., 0], np.zeros_like(p[..., 0]), np.sign(p[..., 2]) * 0.01), -1)
    normals = np.where(length > 1e-9, normals / np.maximum(length, 1e-12), _unit(fallback + [0, 1e-3, 0]))
    return p, normals



def _ring_full(half):
    """Mirror a right half ring (bottom centre → top centre) into a closed ring: bottom centre, +x side, top, −x side."""
    left = half[-2:0:-1] * [-1, 1]
    return np.concatenate((half, left))


def _emit(mesh, material, positions, normals, uvs, quads, color=None):
    """Add the given grid quads (list of (a, b, c, d) flat vertex indices, counter-clockwise outside)."""
    if not len(quads):
        return
    quads = np.asarray(quads)
    used, inverse = np.unique(quads.ravel(), return_inverse=True)
    q = inverse.reshape(-1, 4)
    mesh.add(material, positions[used], normals[used], uvs[used], np.column_stack((q[:, [0, 1, 2]], q[:, [0, 2, 3]])).ravel(),
             color)


def _cap(mesh, material, ring, z, facing, color=None):
    """A flat end cap (fan) closing a ring at z, facing ±z."""
    centre = ring.mean(0)
    positions = np.column_stack((np.concatenate(([centre], ring)), np.full(len(ring) + 1, z)))
    count = len(ring)
    fan = np.array([(0, 1 + k, 1 + (k + 1) % count) for k in range(count)])
    if facing > 0:  # Ring runs counter-clockwise seen from +z.
        indices = fan
    else:
        indices = fan[:, [0, 2, 1]]
    mesh.add(material, positions, np.tile([0, 0, facing], (count + 1, 1)), positions[:, :2], indices.ravel(), color)


def _wheel(mesh, x, y, z, r, width, side, level, rim="vehicle/rim"):
    """A tyre with a dished rim and spokes; side +1 for the +x wheels (axle along +x outwards), −1 for −x."""
    rotation = np.array([[0, side, 0], [-side, 0, 0], [0, 0, 1]], float)  # Lathe axis (y) to ±x, outwards.
    sides = (20, 10, 8)[level]
    rr = r * 0.66
    at = (x - side * width, y, z)  # Inner face.
    lathe(mesh, "vehicle/rubber", [(rr, 0), (r * 0.93, 0), (r, 0.035), (r, width - 0.035), (r * 0.93, width), (rr, width)],
          sides, at=at, rotation=rotation)
    if level == 2:
        lathe(mesh, rim, [(rr, width - 0.01), (0, width - 0.02)], sides, at=at, rotation=rotation)
        return
    if level == 1:
        lathe(mesh, rim, [(rr, width - 0.004), (rr * 0.3, width - 0.05), (0, width - 0.02)], sides, at=at, rotation=rotation)
        return
    lathe(mesh, rim, [(rr, width - 0.004), (rr * 0.88, width - 0.03), (rr * 0.32, width - 0.055), (rr * 0.22, width - 0.02),
                      (0, width - 0.02)], sides, at=at, rotation=rotation)
    lathe(mesh, PLASTIC, [(0, 0.02), (rr, 0.02)], max(sides // 2, 6), at=at, rotation=rotation)
    if level == 0:
        for k in range(5):
            a = k * 2 * math.pi / 5
            radial = np.array([0, math.cos(a), math.sin(a)])
            axis = np.array([side, 0, 0])
            frame = np.column_stack((axis, radial, np.cross(axis, radial)))
            centre = np.array([x - side * 0.035, y, z]) + radial * rr * 0.55
            box(mesh, rim, centre, [0.03, rr * 0.62, 0.055], frame)


def car(kind, colour):
    """A car, van, or taxi of the given TYPES kind and COLOURS colour."""
    p = TYPES[kind]
    paint = (*COLOURS[colour], 1.0)
    L, W, H, r = p["L"], p["W"], p["H"], p["r"]
    arch = r * 1.16
    keys = [z for z, _ in p["top"]]
    zones = [*p["ws"], *p["glass"], *(p["rw"] or ())]
    pillars = [q + d for q in p["pillars"] for d in (-0.05, 0.05)]
    lods = []
    for level, counts in enumerate(RINGS):
        mesh = Mesh()
        if level == 2:
            z = np.unique(np.round(keys, 4))
        else:
            arches = [a + f * arch for a in p["axles"]
                      for f in ((-1, -0.75, -0.4, 0, 0.4, 0.75, 1) if level == 0 else (-1, 0, 1))]
            z = np.unique(np.round(np.concatenate((np.linspace(0, L, (14, 4)[level]), keys, zones, arches,
                                                  [0.012, L - 0.012, *pillars] if level == 0 else [])), 4))
        z = z[(z >= 0) & (z <= L)]
        top = _curve(p["top"], z)
        belt = np.minimum(_curve(p["belt"], z), top - 0.035)
        yb = _curve(p["bottom"], z)
        if level < 2:
            for axle in p["axles"]:
                d = np.abs(z - axle)
                yb = np.where(d < arch, np.maximum(yb, r + np.sqrt(np.clip(arch ** 2 - d ** 2, 0, None))), yb)
        yb = np.minimum(yb, belt - 0.12)
        half = W / 2 * _curve(p["width"], z)
        roof_half = W / 2 * p["roof"] * _interp([(0, 0.92), (0.15, 1.0), (L - 0.15, 1.0), (L, 0.92)], z)
        sections = [_section(*args, p.get("roof_radius", 0.16), counts) for args in zip(yb, belt, top, half, roof_half)]
        zone = sections[0][1]
        rank = np.concatenate((np.arange(len(zone)), np.arange(len(zone) - 2, 0, -1)))
        rings = [_ring_full(s) for s, _ in sections]
        grid, normals = _loft(z - L / 2, rings)
        ns, nr = grid.shape[:2]
        flat_p, flat_n = grid.reshape(-1, 3), normals.reshape(-1, 3)
        arc = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(rings[ns // 2], axis=0), axis=1))))
        uvs = np.column_stack((np.repeat(z, nr), np.tile(arc, ns)))
        groups = {}
        for i in range(ns - 1):
            zm = (z[i] + z[i + 1]) / 2
            for j in range(nr):
                k2 = (j + 1) % nr
                q = (i * nr + j, i * nr + k2, (i + 1) * nr + k2, (i + 1) * nr + j)
                cx, cy, _ = flat_p[list(q)].mean(0)
                kind_ = zone[max(rank[j], rank[k2])]
                groups.setdefault(_surface_kind(p, zm, cx, cy, kind_, yb[i], L, W), []).append(q)
        for key, quads in groups.items():
            material, color = key
            _emit(mesh, material, flat_p, flat_n, uvs, quads, paint if color == "paint" else color)
        _cap(mesh, "vehicle/paint", rings[0], -L / 2, -1, paint)
        _cap(mesh, "vehicle/paint", rings[-1], L / 2, 1, paint)
        _car_details(mesh, p, level, z, rings, grid, normals, yb, top, belt, half, paint, rank, zone)
        for axle in p["axles"]:
            for side in (1, -1):
                _wheel(mesh, side * (W / 2 * 0.985 - 0.015), r, axle - L / 2, r, p["tyre"], side, level)
        if level < 2:  # Dark wheel wells and underbody, so nothing shows through the arches.
            for axle in p["axles"]:
                box(mesh, PLASTIC, [0, r + arch * 0.4, axle - L / 2], [W - 2 * p["tyre"] - 0.12, arch * 1.2, arch * 1.9])
            box(mesh, PLASTIC, [0, float(yb.min()) + 0.08, 0], [W * 0.86, 0.16, L * 0.86])
        lods.append(mesh.parts())
        if level == 2:
            hull = np.concatenate([np.column_stack((ring, np.full(len(ring), zz - L / 2))) for ring, zz in zip(rings, z)])
    top_y = max(float(part.positions[:, 1].max()) for part in lods[0])
    return Model(lods, [Collider("convex", vertices=hull.astype(np.float32))], top_y, float(math.hypot(L / 2, W / 2)))


def _surface_kind(p, zm, cx, cy, zone, yb, L, W):
    """(material, vertex colour) of a body quad from where it lies: lamps, grille, cladding, glass, or paint."""
    ax = abs(cx) / (W / 2)
    if zm > L - 0.3 and p["head"][0] <= cy <= p["head"][1] and ax > 0.38:
        return "vehicle/lamp", LAMP_FRONT
    if zm < 0.25 and p["tail"][0] <= cy <= p["tail"][1] and ax > (0.72 if p.get("cargo") else 0.3):
        return "vehicle/lamp", LAMP_REAR
    if zm > L - 0.12 and ax < 0.42 and p["grille"][0] <= cy <= p["grille"][1]:
        return PLASTIC, None
    if (zm > L - 0.12 or zm < 0.12) and cy < p["grille"][0] - 0.04 and ax < 0.7:
        return PLASTIC, None
    if p.get("cladding") and zone == 0 and cy < yb + p["cladding"]:
        return PLASTIC, None
    if zone == 1 and p["glass"][0] <= zm <= p["glass"][1]:
        if any(abs(zm - q) < 0.05 for q in p["pillars"]):
            return PLASTIC, None  # Black B pillars, as on most current cars.
        return "vehicle/glass", None
    if zone == 3 and (p["ws"][0] <= zm <= p["ws"][1] or (p["rw"] and p["rw"][0] <= zm <= p["rw"][1])):
        return "vehicle/glass", None
    return "vehicle/paint", "paint"


def _strip(mesh, material, points, normals, width, color=None):
    """A thin strip along a polyline on the body surface, lifted 3 mm along the normals, width along z."""
    points = np.asarray(points) + np.asarray(normals) * 0.003
    across = np.array([0, 0, width / 2])
    positions = np.concatenate((points - across, points + across))
    n = len(points)
    quads = [(k, k + 1, n + k + 1, n + k) for k in range(n - 1)]
    tri = np.array([(a, b, c, a, c, d) for a, b, c, d in quads]).ravel()
    # Wind each quad so it faces along the surface normal.
    test = np.cross(positions[1] - positions[0], positions[n + 1] - positions[0])
    if np.dot(test, normals[0]) < 0:
        tri = tri.reshape(-1, 3)[:, ::-1].ravel()
    mesh.add(material, positions, np.concatenate((normals, normals)), positions[:, [2, 1]], tri, color)


def _car_details(mesh, p, level, z, rings, grid, normals, yb, top, belt, half, paint, rank, zone):
    """Number plates on the bumper faces, mirrors, door shut lines, van rear doors, and the taxi sign."""
    L, W = p["L"], p["W"]
    plate = (0, 1 - 0.22 / 2, 0.5, 1)  # Plate: top 22 % of atlas cell 0 (u 0–0.5, v 0.89–1).
    front_y = (yb[-1] + top[-1]) / 2
    rear_y = min((yb[0] + top[0]) / 2, yb[0] + 0.2)
    quad(mesh, "vehicle/details", [0, front_y, L / 2 + 0.006], [0.26, 0, 0], [0, 0.056, 0], plate)
    quad(mesh, "vehicle/details", [0, rear_y, -L / 2 - 0.006], [-0.26, 0, 0], [0, 0.056, 0], plate)
    if p.get("cargo"):  # Rear door windows and the gap between the doors.
        for x in (-1, 1):
            quad(mesh, "vehicle/glass", [x * W * 0.2, 1.9, -L / 2 - 0.006], [-x * W * 0.14, 0, 0], [0, 0.32, 0], (0, 0, 1, 1))
        quad(mesh, PLASTIC, [0, (yb[0] + top[0]) / 2, -L / 2 - 0.008], [-0.006, 0, 0], [0, (top[0] - yb[0]) / 2 - 0.1, 0],
             (0, 0, 1, 1))
    if level == 2:
        return
    # Mirrors on short stalks just behind the A-pillar base.
    zm = p["ws"][1] - 0.14
    b = float(np.interp(zm, z, belt))
    shoulder = float(np.interp(zm, z, half)) * 0.978
    for x in (-1, 1):
        box(mesh, "vehicle/paint", [x * (shoulder + 0.06), b + 0.1, zm - L / 2], [0.17, 0.11, 0.07], color=paint)
        box(mesh, PLASTIC, [x * (shoulder + 0.02), b + 0.05, zm - L / 2 + 0.02], [0.08, 0.04, 0.05])
    if level == 0:  # Door shut lines along the body surface.
        doors = [p["axles"][1] - r_ for r_ in (p["r"] * 1.35,)] + [q + 0.05 for q in p["pillars"]] + \
            [p["glass"][0] + 0.25 if not p.get("cargo") else 3.4]
        if p.get("cargo"):
            doors += [p["glass"][0] - 0.02, 2.3]
        nr = len(rings[0])
        for zd in doors:
            i = int(np.argmin(np.abs(z - zd)))
            for side in (1, -1):
                cols = [j for j in range(nr) if zone[rank[j]] == 0 and np.sign(grid[i, j, 0]) == side
                        and grid[i, j, 1] > yb[i] + 0.12]
                cols.sort(key=lambda j: grid[i, j, 1])
                if len(cols) > 1:
                    _strip(mesh, PLASTIC, grid[i, cols], normals[i, cols], 0.006)
    if p.get("sign"):
        y = float(np.interp(L / 2 - 0.2, z, top))
        box(mesh, PLASTIC, [0, y + 0.015, -0.2], [0.6, 0.03, 0.15])
        for facing in (1, -1):
            quad(mesh, "vehicle/details", [0, y + 0.09, -0.2 + facing * 0.061], [facing * 0.27, 0, 0], [0, 0.055, 0],
                 (0.5, 0.5 + 0.25 - 0.11, 1.0, 0.5 + 0.25 + 0.11))
        box(mesh, "vehicle/paint", [0, y + 0.09, -0.2], [0.56, 0.12, 0.12], color=(0.9, 0.55, 0.02, 1))


def bus(colour="yellow"):
    """A Berlin-style three-axle double-decker (13.7 m × 2.55 m × 4.06 m) in yellow with black window bands, two
    doors on the right (−x), a dot-matrix destination display, and twin rear tyres. No logos."""
    paint = (*COLOURS[colour], 1.0)
    L, W, H = 13.7, 2.55, 4.06
    r = 0.5
    axles = (L / 2 - 2.7, -L / 2 + 4.25, -L / 2 + 2.85)
    arch = r * 1.12
    clearance = 0.3
    lods = []
    for level in range(3):
        mesh = Mesh()
        if level == 2:
            stations = np.array([0, 0.01, 0.99, 1])
            corner, side = 1, 1
        else:
            count = (34, 14)[level]
            extra = [0.004, 0.996]
            for axle in axles:
                for f in ((-1, -0.75, -0.45, -0.15, 0.15, 0.45, 0.75, 1) if level == 0 else (-1, -0.5, 0, 0.5, 1)):
                    extra.append((axle + f * arch) / L + 0.5)
            stations = np.unique(np.round(np.concatenate((np.linspace(0, 1, count), extra)), 5))
            corner, side = ((4, 3), (2, 2))[level]
        z = (stations - 0.5) * L
        yb = np.full(len(stations), clearance)
        if level < 2:
            for axle in axles:
                d = np.abs(z - axle)
                yb = np.where(d < arch, np.maximum(yb, r + np.sqrt(np.clip(arch ** 2 - d ** 2, 0, None))), yb)
        taper = _interp([(0, 0.93), (0.004, 0.975), (0.012, 1), (0.988, 1), (0.996, 0.975), (1, 0.93)], stations)
        rings = []
        for b, t in zip(yb, taper):
            w = W / 2 * t
            rb, rt = 0.12, 0.3
            half = np.concatenate(([[0, b]], _arc(w - rb, b + rb, rb, -math.pi / 2, 0, corner + 1)[1:],
                                   np.column_stack((np.full(side * 2, w), np.linspace(b + rb, H - rt, side * 2 + 1)[1:])),
                                   _arc(w - rt, H - rt, rt, 0, math.pi / 2, corner + 2)[1:-1], [[0, H + 0.02]]))
            rings.append(_ring_full(half))
        grid, normals = _loft(z, rings)
        ns, nr = grid.shape[:2]
        uvs = np.column_stack((np.repeat(z, nr), np.tile(np.arange(nr) * 0.3, ns)))
        quads = [(i * nr + j, i * nr + (j + 1) % nr, (i + 1) * nr + (j + 1) % nr, (i + 1) * nr + j)
                 for i in range(ns - 1) for j in range(nr)]
        _emit(mesh, "vehicle/paint", grid.reshape(-1, 3), normals.reshape(-1, 3), uvs, quads, paint)
        _cap(mesh, "vehicle/paint", rings[0], z[0], -1, paint)
        _cap(mesh, "vehicle/paint", rings[-1], z[-1], 1, paint)
        front, rear = z[-1], z[0]
        fx = W / 2 * 0.93
        # Black window bands and glass: lower deck 1.0–2.05 m, upper deck 2.4–3.72 m.
        bands = [(1.0, 2.05), (2.4, 3.72)]
        for y0, y1 in bands:
            for x in (-1, 1):
                quad(mesh, PLASTIC, [x * (W / 2 + 0.004), (y0 + y1) / 2, 0], [0, 0, -x * (L / 2 - 0.35)], [0, (y1 - y0) / 2, 0],
                     (0, 0, 1, 1))
            if level < 2:
                panes = np.linspace(-L / 2 + 0.45, L / 2 - 0.45, 9 if level == 0 else 5)
                for x in (-1, 1):
                    for z0, z1 in zip(panes, panes[1:]):
                        if y0 < 2.2 and x < 0 and any(abs((z0 + z1) / 2 - d) < 0.8 for d in (L / 2 - 1.6, 0.6)):
                            continue  # Doors instead of lower-deck windows on the right.
                        quad(mesh, "vehicle/glass", [x * (W / 2 + 0.008), (y0 + y1) / 2, (z0 + z1) / 2],
                             [0, 0, -x * ((z1 - z0) / 2 - 0.05)], [0, (y1 - y0) / 2 - 0.06, 0], (0, 0, 1, 1))
        # Front: big lower windscreen, upper front window, destination display between them.
        quad(mesh, "vehicle/glass", [0, 1.42, front + 0.006], [fx * 0.97, 0, 0], [0, 0.68, 0], (0, 0, 1, 1))
        quad(mesh, PLASTIC, [0, 3.14, front + 0.004], [fx * 0.98, 0, 0], [0, 0.64, 0], (0, 0, 1, 1))
        quad(mesh, "vehicle/glass", [0, 3.16, front + 0.008], [fx * 0.94, 0, 0], [0, 0.56, 0], (0, 0, 1, 1))
        quad(mesh, PLASTIC, [0, 2.3, front + 0.004], [fx * 0.62, 0, 0], [0, 0.14, 0], (0, 0, 1, 1))
        quad(mesh, "vehicle/details", [0, 2.3, front + 0.008], [fx * 0.6, 0, 0], [0, 0.12, 0], (0.0, 0.0, 0.5, 0.5))
        quad(mesh, "vehicle/glass", [0, 3.1, rear - 0.006], [-fx * 0.8, 0, 0], [0, 0.45, 0], (0, 0, 1, 1))
        for x in (-1, 1):
            quad(mesh, "vehicle/lamp", [x * fx * 0.78, 0.65, front + 0.008], [0.16, 0, 0], [0, 0.07, 0], (0, 0, 1, 1),
                 color=LAMP_FRONT)
            quad(mesh, "vehicle/lamp", [-x * fx * 0.86, 0.9, rear - 0.008], [-0.08, 0, 0], [0, 0.22, 0], (0, 0, 1, 1),
                 color=LAMP_REAR)
        quad(mesh, PLASTIC, [0, clearance + 0.15, front + 0.004], [fx, 0, 0], [0, 0.13, 0], (0, 0, 1, 1))
        quad(mesh, "vehicle/details", [0, 0.55, rear - 0.01], [-0.26, 0, 0], [0, 0.055, 0], (0, 0.89, 0.5, 1))
        # Doors on the right (−x): glass leaves with dark frames.
        for d in (L / 2 - 1.6, 0.6):
            quad(mesh, PLASTIC, [-(W / 2 + 0.005), 1.2, d], [0, 0, 0.6], [0, 0.95, 0], (0, 0, 1, 1))
            if level < 2:
                for k in (-1, 1):
                    quad(mesh, "vehicle/glass", [-(W / 2 + 0.009), 1.25, d + k * 0.29], [0, 0, 0.26], [0, 0.82, 0], (0, 0, 1, 1))
        for axle in axles:
            twin = axle < 0
            for side_ in (1, -1):
                _wheel(mesh, side_ * (W / 2 - 0.03), r, axle, r, 0.5 if twin else 0.3, side_, level)
        if level < 2:
            for axle in axles:
                box(mesh, PLASTIC, [0, r + arch * 0.45, axle], [W - 1.1, arch * 0.9, arch * 1.9])
            box(mesh, PLASTIC, [0, clearance + 0.12, 0], [W * 0.92, 0.22, L * 0.94])
        lods.append(mesh.parts())
        if level == 2:
            hull = np.concatenate([np.column_stack((ring, np.full(len(ring), zz))) for ring, zz in zip(rings, z)])
    return Model(lods, [Collider("convex", vertices=hull.astype(np.float32))], H + 0.02, float(math.hypot(L / 2, W / 2)))


def vehicle(name):
    kind, colour = VEHICLES[name]
    return bus(colour) if kind == "bus" else car(kind, colour)
