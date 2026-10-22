"""Procedural library models: trees, bushes, and street props as numpy geometry, written by library.asset.

Every model's frame: metres, origin on the ground at the trunk or base centre, y up, and +z the front: where a lamp
arm reaches, a sign or signal faces, and a bench's sitter looks. Instances turn by yaw about +y, so the front points
along (sin yaw, cos yaw) in (x, z). Geometry starts slightly below y = 0 so it stays grounded on slopes.
"""

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
from PIL import Image

from .types import Collider, Part

GOLDEN = math.pi * (3 - math.sqrt(5))
# Linear RGB vertex colours for props painted with the near-white "prop/paint".
ORANGE, RED, DARK, WHITE = (0.85, 0.2, 0.012, 1), (0.6, 0.015, 0.015, 1), (0.035, 0.037, 0.04, 1), (0.9, 0.9, 0.9, 1)


@dataclass
class Model:
    lods: list  # [[Part]] for LOD0, LOD1, LOD2.
    colliders: list
    height: float
    radius: float
    impostor: tuple = None  # (material name, (colour, normal, orm) arrays) for a tree's LOD2 cards.


class Mesh:
    """Collects triangles per material."""

    def __init__(self):
        self.data = {}

    def add(self, material, positions, normals, uvs, indices, color=None):
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        entry = self.data.setdefault(material, dict(p=[], n=[], uv=[], i=[], c=[], count=0))
        colors = np.ones((len(positions), 4), np.float32)
        if color is not None:
            colors[:] = np.broadcast_to(np.asarray(color, np.float32).reshape(-1, 4), colors.shape)
        entry["p"].append(positions)
        entry["n"].append(np.asarray(normals, np.float32).reshape(-1, 3))
        entry["uv"].append(np.asarray(uvs, np.float32).reshape(-1, 2))
        entry["c"].append(colors)
        entry["i"].append(np.asarray(indices, np.uint32).ravel() + entry["count"])
        entry["count"] += len(positions)

    def parts(self):
        parts = []
        for material, e in sorted(self.data.items()):
            colors = np.concatenate(e["c"])
            parts.append(Part(material, np.concatenate(e["p"]), np.concatenate(e["n"]), np.concatenate(e["uv"]),
                              np.concatenate(e["i"]), None if np.all(colors == 1) else colors))
        return parts


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _grid(rows, cols):
    """Quad indices of a rows × cols vertex grid, counter-clockwise when rows advance along the outward-facing side."""
    r, c = np.mgrid[0:rows - 1, 0:cols - 1]
    a, b = r * cols + c, r * cols + c + 1
    d, e = a + cols, b + cols
    return np.stack((a, b, e, a, e, d), -1).ravel()


def tube(mesh, material, points, radii, sides, color=None):
    """A generalised cylinder along a polyline (parallel-transport frames); UVs in metres around and along. color is
    one RGBA for all, or one per point."""
    points = np.asarray(points, float)
    radii = np.broadcast_to(np.asarray(radii, float), len(points))
    tangent = _unit(np.gradient(points, axis=0))
    normal = _unit(np.cross(tangent[0], [0.0, 0, 1] if abs(tangent[0][2]) < 0.9 else [1.0, 0, 0]))
    frames = []
    for t in tangent:
        normal = _unit(normal - t * normal.dot(t))
        frames.append(normal)
    n = np.array(frames)
    b = np.cross(tangent, n)
    angle = np.linspace(0, 2 * np.pi, sides + 1)
    ring = np.cos(angle)[None, :, None] * n[:, None] + np.sin(angle)[None, :, None] * b[:, None]
    length = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    slope = np.gradient(radii) / np.maximum(np.gradient(length), 1e-9)
    normals = _unit(ring - tangent[:, None] * slope[:, None, None])
    positions = points[:, None] + radii[:, None, None] * ring
    u = np.broadcast_to(angle[None, :] * radii.mean(), (len(points), sides + 1))
    v = np.broadcast_to(length[:, None], u.shape)
    if color is not None and np.ndim(color) == 2:
        color = np.repeat(np.asarray(color, float), sides + 1, axis=0)
    mesh.add(material, positions, normals, np.stack((u, v), -1), _grid(len(points), sides + 1), color)


def lathe(mesh, material, profile, sides, color=None, at=(0, 0, 0), rotation=None, phase=0.0):
    """Revolve (radius, y) profile points, ordered bottom to top, around y; each profile segment shades flat."""
    profile = np.asarray(profile, float)
    angle = np.linspace(0, 2 * np.pi, sides + 1) + phase
    ring = np.stack((np.sin(angle), np.zeros_like(angle), np.cos(angle)), -1)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    v = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))))
    for (r0, y0), (r1, y1), v0, v1 in zip(profile, profile[1:], v, v[1:]):
        d = _unit([r1 - r0, y1 - y0])
        normal = ring * d[1] + np.array([0, -d[0], 0])  # Perpendicular to the profile, outwards.
        positions = np.concatenate((ring * r0 + [0, y0, 0], ring * r1 + [0, y1, 0]))
        normals = np.concatenate((normal, normal))
        u = np.tile(angle * max(profile[:, 0].max(), 0.01), 2)
        uvs = np.stack((u, np.repeat([v0, v1], sides + 1)), -1)
        mesh.add(material, positions @ rotation.T + at, normals @ rotation.T, uvs, _grid(2, sides + 1), color)


# Box faces: outward normal, then in-plane axes a and b with a × b = normal; b runs up on vertical faces.
FACES = [((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
         ((0, 0, -1), (-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]


def box(mesh, material, center, size, rotation=None, color=None):
    """An oriented box; UVs in metres along each face (v up on vertical faces)."""
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    half = np.asarray(size, float) / 2
    for out, a, b in (tuple(map(np.array, face)) for face in FACES):
        corners = np.array([(out + a * sa + b * sb) * half for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uvs = np.stack((corners @ a, corners @ b), -1) + [np.abs(a) @ half, np.abs(b) @ half]
        mesh.add(material, corners @ rotation.T + center, np.tile(out @ rotation.T, (4, 1)), uvs, [0, 1, 2, 0, 2, 3], color)


def quad(mesh, material, center, right, up, uv_rect, normals=None, color=None):
    """A card spanning center ± right ± up, front facing right × up; uv_rect is (u0, v0, u1, v1)."""
    center, right, up = (np.asarray(v, float) for v in (center, right, up))
    corners = np.array([center - right - up, center + right - up, center + right + up, center - right + up])
    u0, v0, u1, v1 = uv_rect
    normals = np.tile(_unit(np.cross(right, up)), (4, 1)) if normals is None else normals
    mesh.add(material, corners, normals, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], [0, 1, 2, 0, 2, 3], color)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _cylinder(radius, bottom, top, center=(0, 0)):
    return Collider("cylinder", params=dict(radius=radius, half_height=(top - bottom) / 2,
                                            center=[center[0], (top + bottom) / 2, center[1]]))


def _convex(lo, hi, rotation=None, offset=(0, 0, 0)):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    if rotation is not None:
        corners = corners @ np.asarray(rotation).T
    return Collider("convex", vertices=(corners + offset).astype(np.float32))


# Trees

# height: typical metres; ratio: crown radius / height; base: crown base / height; trunk: base radius; lobes: crown
# clumps at scaffold tips; rise: scaffold elevation (degrees); cards: LOD0 leaf cards; card: card size (m);
# tint: leaf vertex colour; leader: trunk runs to the top (excurrent) instead of forking into scaffolds.
TREES = dict(
    linden=dict(height=18.0, ratio=0.27, base=0.17, trunk=0.3, lobes=14, rise=(35, 70), cards=1900, card=1.25,
                tint=(1.04, 1.05, 0.82), leader=True, bark="veg/bark"),
    plane=dict(height=22.0, ratio=0.42, base=0.2, trunk=0.42, lobes=15, rise=(25, 55), cards=2100, card=1.4,
               tint=(0.95, 1.0, 0.82), leader=False, bark="veg/bark-plane"),
    maple=dict(height=15.0, ratio=0.33, base=0.15, trunk=0.26, lobes=13, rise=(30, 65), cards=1800, card=1.15,
               tint=(0.86, 0.95, 0.84), leader=False, bark="veg/bark"),
    chestnut=dict(height=16.0, ratio=0.36, base=0.14, trunk=0.32, lobes=13, rise=(25, 60), cards=1900, card=1.3,
                  tint=(0.74, 0.88, 0.72), leader=False, bark="veg/bark"),
    birch=dict(height=15.0, ratio=0.2, base=0.25, trunk=0.17, lobes=12, rise=(55, 75), cards=1300, card=1.0,
               tint=(1.1, 1.12, 0.8), leader=True, bark="veg/bark-birch"),
    pine=dict(height=20.0, ratio=0.22, base=0.62, trunk=0.27, lobes=9, rise=(-5, 20), cards=1500, card=1.2,
              tint=(1.0, 1.0, 1.0), leader=True, bark="veg/bark-pine", conifer=True),
)


def tree(species):
    """A tree: bark tubes for trunk and branches, crown clumps of leaf cards, and imposter cards for LOD2."""
    p = TREES[species]
    rng = np.random.default_rng(sum(map(ord, species)) * 7919)  # Deterministic per species.
    if p.get("conifer"):
        trunk, trunk_radii, branches, cards, bottom = _pine(p, rng)
    else:
        trunk, trunk_radii, branches, cards, bottom = _broadleaf(p, rng)
    lods = []
    for level, (sides, depth, share, grow) in enumerate([((10, 6, 4), 2, 1.0, 1.0), ((6, 4, 0), 1, 0.3, 1.65)]):
        mesh = Mesh()
        for path, radii, order, *color in branches:
            if order <= depth and sides[order]:
                tube(mesh, p["bark"], path, radii, sides[order], color[0] if color else None)
        _emit_cards(mesh, p, cards[: int(len(cards) * share)], grow, crossed=level == 0 and not p.get("conifer"))
        lods.append(mesh.parts())
    # Record the measured extent: scatter scales trees by height and crown radius.
    points = np.concatenate([part.positions for part in lods[0]])
    top, spread = float(points[:, 1].max()), float(np.percentile(np.hypot(points[:, 0], points[:, 2]), 97))
    name = f"veg/impostor-{species}"
    width = np.abs(points[:, [0, 2]]).max() * 2.04
    impostor = _impostor(lods[0], width, top * 1.02)
    mesh = Mesh()
    for angle in (0, math.pi / 2):
        right = _rotation_y(angle) @ [width / 2, 0, 0]
        quad(mesh, name, [0, top * 0.51, 0], right, [0, top * 0.51, 0], (0, 0, 1, 1))
    tube(mesh, p["bark"], trunk[[0, 6]], trunk_radii[[0, 6]], 4, branches[0][3][[0, 6]] if len(branches[0]) > 3 else None)
    lods.append(mesh.parts())
    colliders = [_cylinder(float(trunk_radii[4]), 0, max(bottom, 2.5))]
    return Model(lods, colliders, top, spread, (name, impostor))


def _broadleaf(p, rng):
    """Trunk, scaffold branches into crown clumps (lobes), twigs, and leaf cards of a broadleaf tree."""
    h, radius = p["height"], p["height"] * p["ratio"]
    bottom = h * p["base"]
    half = (h - bottom) / 2  # Vertical crown radius.
    centre = np.array([0, bottom + half, 0])
    lean = rng.normal(0, 0.012, 2)
    # Trunk: to the top for excurrent trees, else to where scaffolds fork.
    top = h * 0.92 if p["leader"] else bottom + 0.3 * (h - bottom)
    ts = np.linspace(-0.3, top, 12)
    wiggle = rng.normal(0, 0.04, (12, 2)).cumsum(0) * 0.5
    trunk = np.stack((lean[0] * ts + wiggle[:, 0] * (ts / top), ts, lean[1] * ts + wiggle[:, 1] * (ts / top)), -1)
    trunk_radii = p["trunk"] * (1 - 0.75 * np.clip(ts / top, 0, 1) ** 0.8) * np.where(ts < 0.6, 1 + 0.25 * (0.6 - ts), 1)
    branches = [(trunk, trunk_radii, 0)]
    lobes = []
    count = p["lobes"]
    for i in range(count):
        azimuth = i * GOLDEN + rng.normal(0, 0.2)
        fraction = (i + 0.5) / count
        # Scaffolds leave the trunk across the crown's lower part, lower ones reaching further out.
        start_y = bottom + (min(top, h * 0.85) - bottom) * (0.15 + 0.7 * fraction) if p["leader"] else top * rng.uniform(0.75, 1.0)
        k = np.argmin(np.abs(trunk[:, 1] - start_y))
        start = trunk[k]
        rise = math.radians(rng.uniform(*p["rise"]))
        polar = 0.35 + 1.75 * (1 - fraction) if p["leader"] else rng.uniform(0.3, 2.1)
        direction = np.array([math.sin(polar) * math.cos(azimuth), math.cos(polar), math.sin(polar) * math.sin(azimuth)])
        tip = centre + direction * np.array([radius, half, radius]) * rng.uniform(0.62, 0.82)
        bend = start + (tip - start) * 0.5 + np.array([0, (tip - start)[1] * 0.1 + math.sin(rise) * 0.8, 0])
        path = np.array([start, start + (bend - start) * 0.5, bend, bend + (tip - bend) * 0.5, tip])
        thickness = trunk_radii[k] * rng.uniform(0.45, 0.6)
        branches.append((path, thickness * np.linspace(1, 0.3, 5), 1))
        # Clumps: smaller and more distinct than the crown, so light gaps open between them.
        lobes.append((tip, radius * rng.uniform(0.3, 0.42), half * rng.uniform(0.28, 0.4)))
        for _ in range(3):  # Twigs into the lobe.
            s = path[rng.integers(1, 4)]
            end = tip + _unit(rng.normal(0, 1, 3)) * lobes[-1][1] * 0.8
            branches.append((np.array([s, (s + end) / 2 + [0, 0.3, 0], end]), thickness * np.array([0.35, 0.22, 0.08]), 2))
    if p["leader"]:
        lobes.append((np.array([trunk[-1][0], h - half * 0.45, trunk[-1][2]]), radius * 0.4, half * 0.45))
    cards = _crown_cards(rng, p, lobes, centre, radius, half)
    return trunk, trunk_radii, branches, cards, bottom


def _pine(p, rng):
    """A Scots pine: a tall, slightly curved bare trunk, grey-brown below and orange above, a few dead stubs, and a
    flat, irregular crown of needle pads at the ends of up-curving limbs."""
    h = p["height"]
    bottom = h * p["base"]
    top = h * 0.95
    ts = np.linspace(-0.3, top, 16)
    f = np.clip(ts / top, 0, 1)
    sweep = rng.normal(0, 1, 2)
    sweep = sweep / np.linalg.norm(sweep) * 0.6
    trunk = np.stack((sweep[0] * np.sin(f * np.pi * 1.1) + rng.normal(0, 0.03, 16).cumsum() * 0.3 * f, ts,
                      sweep[1] * np.sin(f * np.pi * 1.1) + rng.normal(0, 0.03, 16).cumsum() * 0.3 * f), -1)
    trunk_radii = p["trunk"] * (1 - 0.8 * f ** 1.1) * np.where(ts < 0.5, 1 + 0.35 * (0.5 - ts), 1)
    orange = np.clip((f - 0.35) / 0.3, 0, 1)[:, None]
    colors = np.hstack((np.array([0.78, 0.74, 0.72]) * (1 - orange) + np.array([1.0, 0.7, 0.48]) * orange, np.ones((16, 1))))
    branches = [(trunk, trunk_radii, 0, colors)]
    at = lambda y: trunk[np.argmin(np.abs(trunk[:, 1] - y))]
    radius_at = lambda y: float(np.interp(y, ts, trunk_radii))
    for k in range(7):  # Dead stubs below the crown.
        y = rng.uniform(0.3, 0.6) * h
        a = rng.uniform(0, 2 * np.pi)
        d = np.array([math.cos(a), rng.uniform(-0.4, 0.1), math.sin(a)])
        start = at(y)
        branches.append((np.array([start, start + d * rng.uniform(0.3, 0.9)]), radius_at(y) * np.array([0.25, 0.06]), 2,
                         np.tile([0.6, 0.55, 0.5, 1.0], (2, 1))))
    pads = []
    limbs = 9
    for i in range(limbs):
        fraction = i / (limbs - 1)
        y = bottom + (top - bottom - 1.5) * fraction ** 0.8
        a = i * GOLDEN + rng.normal(0, 0.25)
        horizontal = np.array([math.cos(a), 0, math.sin(a)])
        length = (4.2 - 2.4 * fraction) * rng.uniform(0.8, 1.15)
        start = at(y)
        rise = length * rng.uniform(0.35, 0.6)
        mid = start + horizontal * length * 0.45 + [0, rise * 0.75, 0]
        tip = start + horizontal * length + [0, rise, 0]
        path = np.array([start, (start + mid) / 2 + [0, 0.1, 0], mid, (mid + tip) / 2 + [0, 0.08, 0], tip])
        thickness = radius_at(y) * rng.uniform(0.4, 0.55)
        branches.append((path, thickness * np.linspace(1, 0.35, 5), 1, np.tile([0.95, 0.72, 0.55, 1.0], (5, 1))))
        pads.append((tip + [0, 0.25, 0], rng.uniform(1.4, 2.1) * (1.1 - 0.3 * fraction), rng.uniform(0.45, 0.7)))
        if fraction < 0.7:
            pads.append((mid + [0, 0.2, 0], rng.uniform(0.9, 1.3), rng.uniform(0.35, 0.5)))
        for _ in range(3):  # Twigs into the pad.
            s = path[rng.integers(2, 5)]
            end = pads[-1][0] + np.array([rng.normal(0, 0.8), rng.uniform(-0.1, 0.3), rng.normal(0, 0.8)])
            branches.append((np.array([s, (s + end) / 2 + [0, 0.15, 0], end]), thickness * np.array([0.3, 0.18, 0.06]), 2))
    for _ in range(3):  # The flat top.
        pads.append((trunk[-1] + [rng.normal(0, 0.9), rng.uniform(-0.5, 0.2), rng.normal(0, 0.9)], rng.uniform(1.6, 2.2),
                     rng.uniform(0.55, 0.8)))
    weights = np.array([r * r for _, r, _ in pads])
    choice = rng.choice(len(pads), p["cards"], p=weights / weights.sum())
    centres = np.array([pads[c][0] for c in choice])
    r = np.array([pads[c][1] for c in choice])
    rv = np.array([pads[c][2] for c in choice])
    a = rng.uniform(0, 2 * np.pi, len(choice))
    u = np.sqrt(rng.random(len(choice)))
    lift = np.clip(rng.normal(0.15, 0.45, len(choice)), -1, 1) * (1 - 0.6 * u ** 2)  # Domed pads, thin at the rim.
    points = centres + np.column_stack((np.cos(a) * r * u, lift * rv, np.sin(a) * r * u))
    out = _unit(np.column_stack((np.cos(a), np.full(len(a), 0.15), np.sin(a))))
    normals = _unit(np.array([0, 1.0, 0]) * 0.9 + out * 0.35 * u[:, None] + rng.normal(0, 0.3, (len(a), 3)))
    shade = (0.5 + 0.5 * (lift + 1) / 2) * (0.85 + 0.15 * u)
    shade *= 0.45 + 0.55 * _occlusion(points, _unit(out * 0.4 + [0, 1, 0]), 0.9)
    shade *= rng.uniform(0.9, 1.1, len(a))
    cards = list(zip(points, normals, out, shade, rng.integers(0, 4, len(a)), rng.uniform(0.8, 1.2, len(a)),
                     rng.uniform(0, 2 * np.pi, len(a))))
    rng.shuffle(cards)  # LOD1 keeps a random share.
    return trunk, trunk_radii, branches, cards, bottom


def _occlusion(points, directions, radius, steps=(0.6, 1.2, 1.9, 2.7)):
    """Per-point openness (1 open, towards 0 buried) from how many other points lie along each direction."""
    total = np.zeros(len(points))
    for step in steps:
        samples = points + directions * step
        for i in range(0, len(points), 512):
            d = np.linalg.norm(samples[i:i + 512, None] - points[None], axis=-1)
            total[i:i + 512] += (d < radius).sum(1)
    return np.exp(-0.06 * total)


def _crown_cards(rng, p, lobes, centre, radius, half):
    """Leaf card centres, normals, and shading: mostly near clump surfaces, some filling the crown shell."""
    count = p["cards"]
    points, outward = [], []
    for i in range(count):
        if i % 4 == 3:  # Crown shell between the clumps.
            d = _unit(rng.normal(0, 1, 3))
            d[1] *= 0.4 if p.get("conifer") else 1.0
            point = centre + _unit(d) * np.array([radius, half, radius]) * rng.uniform(0.6, 0.95)
        else:
            lobe, r, rv = lobes[rng.integers(len(lobes))]
            d = _unit(rng.normal(0, 1, 3))
            point = lobe + d * np.array([r, rv, r]) * rng.uniform(0.55, 1.0) ** 0.5
        points.append(point)
    points = np.array(points)
    out = _unit((points - centre) / np.array([radius, half, radius]))
    # Depth in the crown (0 centre, 1 surface) darkens inner and lower cards: cheap ambient occlusion.
    depth = np.clip(np.linalg.norm((points - centre) / np.array([radius, half, radius]), axis=1), 0, 1.2)
    height = np.clip((points[:, 1] - (centre[1] - half)) / (2 * half), 0, 1)
    shade = (0.5 + 0.5 * np.clip(depth, 0, 1) ** 1.5) * (0.8 + 0.2 * height) * rng.uniform(0.9, 1.1, len(points))
    # Self-shadowing: cards with many others between them and the sky (and outwards) are darker.
    shade *= 0.38 + 0.62 * _occlusion(points, _unit(out * 0.6 + [0, 0.8, 0]), 0.75)
    normals = _unit(out * 0.7 + rng.normal(0, 0.55, out.shape))
    flip = np.einsum("ij,ij->i", normals, out) < 0
    normals[flip] *= -1
    return list(zip(points, normals, out, shade, rng.integers(0, 4, len(points)), rng.uniform(0.75, 1.25, len(points)),
                    rng.uniform(0, 2 * np.pi, len(points))))


def _emit_cards(mesh, p, cards, grow, crossed=False):
    """Leaf cards; with crossed, every third card gets a second card at right angles for volume from the side."""
    material = "veg/leaves-conifer" if p.get("conifer") else "veg/leaves-broad"
    tint = np.asarray(p["tint"])
    for k, (point, normal, out, shade, cell, scale, roll) in enumerate(cards):
        size = p["card"] * scale * grow / 2
        # Twigs in the atlas point up the card: aim them outwards, rolled a little.
        up = out - normal * out.dot(normal)
        up = _unit(up if np.linalg.norm(up) > 1e-3 else np.cross(normal, [1, 0, 0]))
        up = _unit(up * math.cos(roll * 0.25) + np.cross(normal, up) * math.sin(roll * 0.25))
        right = np.cross(up, normal)
        u0, v0 = cell % 2 / 2, 1 - (cell // 2 + 1) / 2
        vertex_normal = _unit(out * 0.75 + normal * 0.25)
        color = (*np.clip(tint * shade, 0, 2), 1.0)
        quad(mesh, material, point, right * size, up * size, (u0, v0, u0 + 0.5, v0 + 0.5),
             normals=np.tile(vertex_normal, (4, 1)), color=color)
        if crossed and k % 3 == 0:
            quad(mesh, material, point, normal * size * 0.8, up * size * 0.8, (u0, v0, u0 + 0.5, v0 + 0.5),
                 normals=np.tile(vertex_normal, (4, 1)), color=color)


def _impostor(parts, width, height):
    """Render a tree's LOD0 from the front into colour (with alpha), normal, and ORM maps for its LOD2 cards."""
    cols = 512
    rows = 1024 if height > 1.4 * width else 512
    rgba, normal = render(parts, np.eye(3), (cols, rows), (-width / 2, width / 2, 0, height))
    solid = rgba[..., 3] > 0.5
    color = np.zeros((rows, cols, 4), np.uint8)
    srgb = np.clip(rgba[..., :3], 0, 1) ** (1 / 2.2) * 255
    color[..., :3] = np.where(solid[..., None], srgb, srgb[solid].mean(0) if solid.any() else 0)
    color[..., 3] = solid * 255
    encoded = np.where(solid[..., None], (normal + 1) * 127.5, [128, 128, 255]).astype(np.uint8)
    from .library import pack_orm

    return color, encoded, pack_orm(1.0, np.full((rows, cols), 0.75), 0.0)


def render(parts, view, size, bounds, light=None):
    """Orthographic software rasteriser: linear RGBA (alpha-tested textures × vertex colours) and view-space normals.

    view rotates model coordinates into the camera's (x right, y up, z towards the viewer); bounds is (x0, x1, y0, y1)
    in camera coordinates. With light (a view-space direction), colours are Lambert-shaded for previews.
    """
    from . import library

    cols, rows = size
    x0, x1, y0, y1 = bounds
    color = np.zeros((rows, cols, 4))
    normals = np.zeros((rows, cols, 3))
    depth = np.full((rows, cols), -np.inf)
    for part in parts:
        material = library.material(part.material)
        texture = np.asarray(Image.open(library.root() / material.base_color).convert("RGBA"), float) / 255
        texture[..., :3] **= 2.2
        th, tw = texture.shape[:2]
        p = np.asarray(part.positions, float) @ np.asarray(view).T
        n = _unit(np.asarray(part.normals, float) @ np.asarray(view).T)
        sx, sy = (p[:, 0] - x0) / (x1 - x0) * cols, (y1 - p[:, 1]) / (y1 - y0) * rows
        uv = np.asarray(part.uvs, float) / (material.size or 1)
        c = np.ones((len(p), 4)) if part.colors is None else np.asarray(part.colors, float)
        for tri in np.asarray(part.indices).reshape(-1, 3):
            xs, ys = sx[tri], sy[tri]
            c0, c1 = max(int(xs.min()), 0), min(int(math.ceil(xs.max())), cols)
            r0, r1 = max(int(ys.min()), 0), min(int(math.ceil(ys.max())), rows)
            if c0 >= c1 or r0 >= r1:
                continue
            gy, gx = np.mgrid[r0:r1, c0:c1] + 0.5
            det = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(det) < 1e-12:
                continue
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / det
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / det
            w = np.stack((w0, w1, 1 - w0 - w1), -1)
            inside = (w >= -1e-6).all(-1)
            if not inside.any():
                continue
            z = w @ p[tri, 2]
            t = w @ uv[tri]
            texel = texture[((1 - t[..., 1]) % 1 * th).astype(int) % th, (t[..., 0] % 1 * tw).astype(int) % tw]
            region = depth[r0:r1, c0:c1]
            ok = inside & (z > region) & ((texel[..., 3] > 0.5) | (material.alpha != "MASK"))
            if not ok.any():
                continue
            nn = _unit(w @ n[tri])
            facing = np.where((nn[..., 2:3] < 0) & material.double_sided, -nn, nn)
            rgb = texel[..., :3] * (w @ c[tri])[..., :3]
            if light is not None:
                rgb = rgb * (0.35 + 0.75 * np.clip(facing @ _unit(light), 0, 1))[..., None]
            region[ok] = z[ok]
            color[r0:r1, c0:c1][ok] = np.concatenate((rgb, np.ones(rgb.shape[:-1] + (1,))), -1)[ok]
            normals[r0:r1, c0:c1][ok] = facing[ok]
    return color, normals


# Bushes and hedges


def _cards_on(rng, points, out):
    """Leaf cards at points facing roughly along out."""
    mesh_cards = []
    for point, o in zip(points, out):
        normal = _unit(o * 0.7 + rng.normal(0, 0.5, 3))
        if normal.dot(o) < 0:
            normal = -normal
        mesh_cards.append((point, normal, o, rng.uniform(0.75, 1.05), rng.integers(0, 4), rng.uniform(0.8, 1.2),
                           rng.uniform(0, 2 * np.pi)))
    return mesh_cards


def bush():
    """A round shrub: an opaque foliage core wrapped in leaf cards."""
    rng = np.random.default_rng(31)
    height, radius = 1.3, 0.9
    centre = np.array([0, 0.62, 0])
    p = dict(card=0.55, tint=(0.82, 0.95, 0.75))
    lods = []
    for count, grow, sides in ((340, 1.0, 10), (110, 1.6, 6), (0, 1.0, 6)):
        mesh = Mesh()
        profile = [(0, -0.05)] + [(math.sin(a) * radius * 0.8, centre[1] - math.cos(a) * 0.58) for a in np.linspace(0.5, np.pi, 6)]
        lathe(mesh, "veg/foliage-solid", profile, sides)
        d = _unit(rng.normal(0, 1, (count, 3)) * [1, 0.7, 1] + [0, 0.25, 0])
        points = centre + d * np.array([radius, height - centre[1], radius]) * rng.uniform(0.75, 0.98, (count, 1))
        _emit_cards(mesh, p, _cards_on(rng, points, d), grow)
        lods.append(mesh.parts())
    return Model(lods, [_cylinder(0.6, 0, 1.0)], height, radius)


def hedge():
    """A 2 m hedge segment along x (from −1 to 1 m), 0.8 m deep and 1.5 m tall; chain them along hedge lines."""
    rng = np.random.default_rng(37)
    length, depth, height = 2.0, 0.8, 1.5
    p = dict(card=0.5, tint=(0.8, 0.92, 0.72))
    lods = []
    for count, grow in ((420, 1.0), (130, 1.6), (0, 1.0)):
        mesh = Mesh()
        box(mesh, "veg/foliage-solid", [0, height * 0.46, 0], [length * 0.99, height * 0.92, depth * 0.78])
        face = rng.choice(4, count, p=[0.3, 0.3, 0.3, 0.1])  # Front, back, top, ends.
        x = rng.uniform(-1, 1, count) * length / 2
        y = rng.uniform(0.1, 1, count) * height * 0.95
        out = np.zeros((count, 3))
        out[face == 0, 2], out[face == 1, 2], out[face == 2, 1] = 1, -1, 1
        out[face == 3, 0] = np.sign(x[face == 3])
        points = np.stack((x, y, rng.uniform(-1, 1, count) * depth * 0.4), -1)
        points[face == 0, 2], points[face == 1, 2] = depth * 0.42, -depth * 0.42
        points[face == 2, 1] = height * rng.uniform(0.9, 1.0, (face == 2).sum())
        points[face == 3, 0] = np.sign(x[face == 3]) * length * 0.48
        _emit_cards(mesh, p, _cards_on(rng, points, out), grow)
        lods.append(mesh.parts())
    return Model(lods, [_convex([-length / 2, 0, -depth / 2], [length / 2, height, depth / 2])], height, length / 2)


# Street props


def _levels(build):
    """Run a prop builder at three detail levels (0 = full) and collect its parts."""
    lods = []
    for level in range(3):
        mesh = Mesh()
        build(mesh, level)
        lods.append(mesh.parts())
    return lods


def street_lamp():
    """Modern street light: 8 m conical mast and an arm reaching 1.4 m towards the road (+z) with an LED head."""
    def build(mesh, level):
        sides = (12, 8, 5)[level]
        tube(mesh, "prop/metal-painted", [[0, -0.1, 0], [0, 4, 0], [0, 7.7, 0]], [0.085, 0.07, 0.05], sides)
        arm = [[0, 7.55, 0], [0, 7.85, 0.12], [0, 8.0, 0.45], [0, 8.05, 0.9], [0, 8.06, 1.25]]
        tube(mesh, "prop/metal-painted", arm, 0.035, (8, 6, 4)[level])
        box(mesh, "prop/metal-painted", [0, 8.05, 1.45], [0.28, 0.09, 0.62])
        box(mesh, "prop/lamp-glass", [0, 7.995, 1.47], [0.22, 0.02, 0.5])
        if level == 0:
            lathe(mesh, "prop/metal-painted", [(0, 0), (0.13, 0), (0.13, 0.35), (0.09, 0.4), (0, 0.4)], 12)
    return Model(_levels(build), [_cylinder(0.09, 0, 8.0)], 8.1, 1.8)


def street_lamp_berlin():
    """Classic Berlin lantern: cast mast with ladder bar and a four-sided glass lantern with pyramid roof, 4.4 m."""
    def build(mesh, level):
        sides = (14, 8, 5)[level]
        mast = [(0, -0.1), (0.16, -0.1), (0.16, 0.12), (0.13, 0.17), (0.13, 0.45), (0.1, 0.52), (0.08, 0.6), (0.068, 0.66),
                (0.062, 1.0), (0.052, 3.15), (0.068, 3.2), (0.068, 3.27), (0.048, 3.32), (0.048, 3.45), (0.1, 3.5),
                (0.1, 3.58), (0, 3.58)]
        lathe(mesh, "prop/metal-painted", mast if level < 2 else [(0, -0.1), (0.13, -0.1), (0.07, 0.6), (0.05, 3.5), (0, 3.55)],
              sides)
        if level < 2:
            tube(mesh, "prop/metal-painted", [[-0.32, 3.08, 0], [0, 3.08, 0], [0.32, 3.08, 0]], 0.014, 6)
        phase = math.pi / 4  # Flat sides face the axes.
        r = math.sqrt(2)
        lathe(mesh, "prop/lamp-glass", [(0.13 * r, 3.62), (0.22 * r, 4.08)], 4, phase=phase)
        lathe(mesh, "prop/metal-painted", [(0, 3.58), (0.14 * r, 3.58), (0.14 * r, 3.63), (0.12 * r, 3.63)], 4, phase=phase)
        lathe(mesh, "prop/metal-painted", [(0.21 * r, 4.07), (0.29 * r, 4.08), (0.29 * r, 4.12), (0.03, 4.36), (0, 4.36)], 4,
              phase=phase)
        if level < 2:
            lathe(mesh, "prop/metal-painted", [(0, 4.34), (0.03, 4.36), (0.04, 4.39), (0.02, 4.44), (0, 4.46)], 8)
            for k in range(4):  # Frame posts at the glass corners.
                a = k * math.pi / 2
                d0, d1 = np.array([math.sin(a), 0, math.cos(a)]) * 0.13 * r, np.array([math.sin(a), 0, math.cos(a)]) * 0.22 * r
                tube(mesh, "prop/metal-painted", [d0 + [0, 3.62, 0], d1 + [0, 4.08, 0]], 0.012, 4)
    return Model(_levels(build), [_cylinder(0.12, 0, 3.6)], 4.46, 0.42)


def bench():
    """Berlin park bench: wooden slats on two painted steel frames, 1.8 m long; the sitter faces +z."""
    tilt = math.radians(16)
    back = _rotation_x(-tilt)

    def build(mesh, level):
        for x in (-0.72, 0.72):
            box(mesh, "prop/metal-painted", [x, 0.2, 0.17], [0.05, 0.44, 0.05])
            box(mesh, "prop/metal-painted", [x, 0.41, 0.0], [0.05, 0.04, 0.46])
            box(mesh, "prop/metal-painted", [x, 0.42, -0.24], [0.05, 0.88, 0.05], back, None)
        slats = [-0.16, -0.055, 0.05, 0.155] if level < 2 else [0.0]
        for z in slats:
            box(mesh, "prop/wood", [0, 0.45, z], [1.8, 0.035, 0.095 if level < 2 else 0.42])
        for y in ([0.58, 0.7, 0.82] if level < 2 else [0.7]):
            centre = np.array([0, y, -0.24 - (y - 0.42) * math.tan(tilt) - 0.03])
            box(mesh, "prop/wood", centre, [1.8, 0.1 if level < 2 else 0.34, 0.03], back)
    colliders = [_convex([-0.9, 0, -0.24], [0.9, 0.47, 0.24]),
                 _convex([-0.9, 0.5, -0.05], [0.9, 0.88, 0.0], back, (0, 0, -0.26))]
    return Model(_levels(build), colliders, 0.9, 0.95)


def waste_basket():
    """Berlin orange street bin on a post; the opening faces up, the bin hangs on the +z side of the post."""
    def build(mesh, level):
        sides = (16, 10, 6)[level]
        tube(mesh, "prop/metal-galvanized", [[0, -0.1, -0.25], [0, 1.0, -0.25]], 0.03, (8, 6, 4)[level])
        profile = [(0, 0.35), (0.2, 0.35), (0.21, 0.37), (0.21, 0.92), (0.225, 0.93), (0.225, 0.96), (0.195, 0.96),
                   (0.195, 0.45), (0, 0.45)]
        lathe(mesh, "prop/paint", profile if level < 2 else [(0, 0.35), (0.21, 0.35), (0.21, 0.96), (0, 0.96)], sides, ORANGE)
        box(mesh, "prop/metal-galvanized", [0, 0.8, -0.215], [0.12, 0.08, 0.04])
    return Model(_levels(build), [_cylinder(0.23, 0, 1.0, (0, -0.02))], 1.0, 0.28)


def bollard():
    """Dark steel bollard with a white reflective band, 0.94 m."""
    def build(mesh, level):
        sides = (14, 8, 5)[level]
        lathe(mesh, "prop/metal-painted", [(0, -0.1), (0.08, -0.1), (0.08, 0.86), (0.065, 0.91), (0.035, 0.935), (0, 0.94)], sides)
        if level < 2:
            lathe(mesh, "prop/paint", [(0.082, 0.72), (0.082, 0.8)], sides, WHITE)
    return Model(_levels(build), [_cylinder(0.08, 0, 0.94)], 0.94, 0.08)


def traffic_signal():
    """Traffic light: galvanized mast with a dark three-lens head facing +z at 2.3–3.2 m."""
    face = _rotation_x(math.pi / 2)  # Lathe axis (y) to +z.

    def build(mesh, level):
        tube(mesh, "prop/metal-galvanized", [[0, -0.1, 0], [0, 3.6, 0]], 0.07, (12, 8, 5)[level])
        box(mesh, "prop/paint", [0, 2.75, 0.17], [0.3, 0.86, 0.2], color=DARK)
        if level < 2:
            box(mesh, "prop/paint", [0, 2.75, 0.06], [0.52, 1.06, 0.02], color=DARK)
            box(mesh, "prop/metal-galvanized", [0, 2.75, 0.03], [0.1, 0.5, 0.06])
        for k, color in enumerate([(1, 0.04, 0.02, 1), (1, 0.42, 0.02, 1), (0.05, 0.9, 0.35, 1)]):
            y = 3.03 - k * 0.28
            lathe(mesh, "prop/signal-lens", [(0, 0), (0.1, 0), (0.1, 0.01), (0.06, 0.03), (0, 0.035)], (12, 8, 6)[level], color,
                  at=(0, y, 0.27), rotation=face)
            if level == 0:
                box(mesh, "prop/paint", [0, y + 0.115, 0.34], [0.24, 0.015, 0.14], color=DARK)
    return Model(_levels(build), [_cylinder(0.08, 0, 3.6)], 3.6, 0.4)


def bicycle_stand():
    """Berlin bike stand ("Kreuzberger Bügel"): a galvanized hoop along x, 0.8 m wide and tall, with a crossbar."""
    def build(mesh, level):
        corner = [(0.4 - 0.15 + 0.15 * math.cos(a), 0.65 + 0.15 * math.sin(a)) for a in np.linspace(0, math.pi / 2, (5, 3, 2)[level])]
        right = [(0.4, -0.1)] + corner  # Up the right leg into the top corner.
        hoop = [(-x, y) for x, y in right] + right[::-1]
        tube(mesh, "prop/metal-galvanized", [(x, y, 0) for x, y in hoop], 0.024, (8, 6, 4)[level])
        if level < 2:
            tube(mesh, "prop/metal-galvanized", [[-0.4, 0.35, 0], [0.4, 0.35, 0]], 0.018, 6)
    return Model(_levels(build), [_convex([-0.43, 0, -0.03], [0.43, 0.82, 0.03])], 0.82, 0.43)


def hydrant():
    """Red above-ground hydrant (0.95 m) with side outlets and a larger front (+z) outlet."""
    sides_ = [np.array([[0, s, 0], [-s, 0, 0], [0, 0, 1]], float) for s in (1, -1)]  # Lathe axis (y) to ±x.
    front = _rotation_x(math.pi / 2)  # Lathe axis (y) to +z.

    def build(mesh, level):
        sides = (16, 10, 6)[level]
        profile = [(0, 0), (0.17, 0), (0.17, 0.06), (0.12, 0.08), (0.11, 0.7), (0.13, 0.72), (0.13, 0.76), (0.1, 0.78),
                   (0.08, 0.86), (0.03, 0.9), (0.03, 0.95), (0, 0.95)]
        lathe(mesh, "prop/paint", profile, sides, RED)
        if level < 2:
            outlet = [(0, 0.0), (0.04, 0.0), (0.04, 0.17), (0.05, 0.17), (0.05, 0.21), (0, 0.21)]
            for rotation in sides_:
                lathe(mesh, "prop/paint", outlet, sides // 2, RED, at=(0, 0.55, 0), rotation=rotation)
            lathe(mesh, "prop/paint", [(0, 0), (0.055, 0), (0.055, 0.19), (0.065, 0.19), (0.065, 0.23), (0, 0.23)], sides // 2,
                  RED, at=(0, 0.48, 0), rotation=front)
    return Model(_levels(build), [_cylinder(0.15, 0, 0.95)], 0.95, 0.25)


# Sign shapes: atlas cell (see library "prop/sign-faces"), outline in metres around the plate centre, centre height.
SIGNS = {
    "prop/sign": (0, [(0.3 * math.cos(a), 0.3 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)[:-1]], 2.5),
    "prop/sign-no-stopping": (1, [(0.3 * math.cos(a), 0.3 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)[:-1]], 2.5),
    "prop/sign-stop": (2, [(0.3 * math.cos(a), 0.3 * math.sin(a)) for a in np.arange(8) * math.pi / 4 + math.pi / 8], 2.5),
    "prop/sign-give-way": (3, [(0.52 * math.cos(a), 0.52 * math.sin(a)) for a in np.radians([270, 30, 150])], 2.45),
}


def sign(kind):
    """A traffic sign on a galvanized post; the face looks +z, its lower edge about 2.2 m up."""
    cell, outline, height = SIGNS[kind]
    outline = np.array(outline)
    # Atlas cells (0.5 UV wide) hold the face within a radius of 0.48 cell, 0.24 UV: map the outline's radius onto it.
    scale = 0.24 / np.abs(outline).max(axis=0).max()
    cx, cy = (cell % 2) / 2 + 0.25, 1 - (cell // 2) / 2 - 0.25

    def build(mesh, level):
        top = height + outline[:, 1].max()
        tube(mesh, "prop/metal-galvanized", [[0, -0.1, 0], [0, top, 0]], 0.03, (8, 6, 4)[level])
        z = 0.045
        n = len(outline)
        front = np.column_stack((outline[:, 0], outline[:, 1] + height, np.full(n, z + 0.005)))
        fan = np.array([(0, i, i + 1) for i in range(1, n - 1)]).ravel()
        uvs = np.column_stack((cx + outline[:, 0] * scale, cy + outline[:, 1] * scale))
        mesh.add("prop/sign-faces", front, np.tile([0, 0, 1], (n, 1)), uvs, fan)
        back = front - [0, 0, 0.01]
        mesh.add("prop/metal-galvanized", back, np.tile([0, 0, -1], (n, 1)), outline, fan.reshape(-1, 3)[:, ::-1].ravel())
        if level < 2:
            box(mesh, "prop/metal-galvanized", [0, height, 0.02], [0.08, 0.05, 0.04])
    return Model(_levels(build), [_cylinder(0.04, 0, height)], float(height + outline[:, 1].max()), float(np.abs(outline).max()))


def _band(mesh, material, radius, y0, y1, sides, color=None):
    """An open cylinder with unit UVs: u once around (from +z, clockwise seen from above, so text reads outside), v up."""
    angle = np.linspace(0, 2 * np.pi, sides + 1)
    ring = np.stack((np.sin(angle), np.zeros_like(angle), np.cos(angle)), -1)
    positions = np.concatenate((ring * radius + [0, y0, 0], ring * radius + [0, y1, 0]))
    uvs = np.column_stack((np.tile(angle / (2 * np.pi), 2), np.repeat([0.0, 1.0], sides + 1)))
    mesh.add(material, positions, np.concatenate((ring, ring)), uvs, _grid(2, sides + 1), color)


GREEN = (0.018, 0.06, 0.035, 1)


def advertising_column():
    """Berlin advertising column (Litfaßsäule): 1.3 m across, 3.7 m tall; a dark green plinth, a 2.4 m band of
    generic posters ("prop/poster", u once around), a moulded cornice, and a domed cap with a finial."""
    def build(mesh, level):
        sides = (32, 16, 8)[level]
        r = 0.65
        lathe(mesh, "prop/paint", [(0, -0.1), (0.74, -0.1), (0.74, 0.12), (0.7, 0.16), (0.7, 0.26), (0.66, 0.3), (r, 0.33)],
              sides, GREEN)
        _band(mesh, "prop/poster", r, 0.33, 2.73, sides)
        cornice = [(r, 2.73), (0.67, 2.76), (0.67, 2.82), (0.71, 2.86), (0.75, 2.92), (0.75, 2.99), (0.69, 3.02),
                   (0.66, 3.02), (0.66, 3.2), (0.7, 3.24), (0.7, 3.28)]
        lathe(mesh, "prop/paint", cornice if level < 2 else [(r, 2.73), (0.75, 2.95), (0.7, 3.28)], sides, GREEN)
        dome = [(0.7, 3.28)] + [(0.7 * math.cos(a), 3.28 + 0.32 * math.sin(a)) for a in np.linspace(0.25, math.pi / 2, (6, 3, 2)[level])]
        lathe(mesh, "prop/paint", dome, sides, GREEN)
        if level < 2:
            lathe(mesh, "prop/paint", [(0, 3.58), (0.05, 3.58), (0.03, 3.62), (0.06, 3.68), (0.02, 3.74), (0, 3.75)],
                  (12, 6)[level], GREEN)
    return Model(_levels(build), [_cylinder(0.72, 0, 3.6)], 3.75, 0.75)


def planter():
    """A 1.2 × 0.6 m concrete planter, 0.5 m tall, with soil and a small leafy shrub; the long side runs along x."""
    rng = np.random.default_rng(53)
    w, d, h, t = 1.2, 0.6, 0.5, 0.07
    p = dict(card=0.4, tint=(0.85, 0.98, 0.78))
    cards = []
    for _ in range(90):
        pt = np.array([rng.uniform(-0.5, 0.5) * w * 0.85, h + rng.uniform(0.05, 0.55), rng.uniform(-0.5, 0.5) * d * 0.8])
        out = _unit(pt - [0, h - 0.1, 0])
        cards.append((pt, out))
    def build(mesh, level):
        for sx in (-1, 1):
            box(mesh, "prop/concrete", [sx * (w - t) / 2, h / 2 - 0.05, 0], [t, h + 0.1, d])
            box(mesh, "prop/concrete", [0, h / 2 - 0.05, sx * (d - t) / 2], [w - 2 * t, h + 0.1, t])
        quad(mesh, "ground/dirt", [0, h - 0.06, 0], [(w - 2 * t) / 2, 0, 0], [0, 0, -(d - 2 * t) / 2], (0, 0, w - 2 * t, d - 2 * t))
        share = (1.0, 0.4, 0.15)[level]
        chosen = cards[: int(len(cards) * share)]
        _emit_cards(mesh, p, _cards_on(np.random.default_rng(7), [c[0] for c in chosen], [c[1] for c in chosen]),
                    (1.0, 1.4, 2.0)[level])
    return Model(_levels(build), [_convex([-w / 2, 0, -d / 2], [w / 2, h, d / 2])], 1.05, 0.68)


def bicycle():
    """A city bike (28" wheels, 1.75 m long) parked upright, front +z: black frame, rims, a few spokes, saddle,
    swept-back handlebar, mudguards, and a rear rack."""
    frame = (0.02, 0.02, 0.022, 1)
    r = 0.35
    rear, front = np.array([0, r, -0.56]), np.array([0, r, 0.56])
    bb = np.array([0, 0.3, -0.05])
    seat, head_top, head_bottom = np.array([0, 0.86, -0.22]), np.array([0, 0.92, 0.36]), np.array([0, 0.72, 0.41])

    def wheel(mesh, centre, level):
        segments = (24, 14, 8)[level]
        a = np.linspace(0, 2 * np.pi, segments + 1)
        circle = lambda radius: centre + np.column_stack((np.zeros_like(a), np.sin(a) * radius, np.cos(a) * radius))
        tube(mesh, "vehicle/rubber", circle(r - 0.018), 0.019, (6, 4, 3)[level])
        if level < 2:
            tube(mesh, "vehicle/rim", circle(r - 0.045), 0.009, 4)
            tube(mesh, "vehicle/rim", [centre - [0.05, 0, 0], centre + [0.05, 0, 0]], 0.02, 6)
        if level == 0:
            for k in range(12):
                b = k * np.pi / 6
                end = centre + np.array([0, math.sin(b), math.cos(b)]) * (r - 0.05)
                tube(mesh, "vehicle/rim", [centre + [(-1) ** k * 0.03, 0, 0], end], 0.0025, 3)

    def build(mesh, level):
        sides = (8, 5, 4)[level]
        wheel(mesh, rear, level)
        wheel(mesh, front, level)
        for a, b, radius in ((bb, seat, 0.017), (bb, head_bottom, 0.02), (seat + [0, -0.06, 0.02], head_top + [0, -0.05, 0], 0.016),
                             (head_bottom, head_top, 0.022)):
            tube(mesh, "prop/paint", [a, b], radius, sides, frame)
        for x in (-0.05, 0.05):
            for a, b in ((bb, rear), (seat + [0, -0.04, 0], rear)):
                tube(mesh, "prop/paint", [a + [x * 0.6, 0, 0], b + [x, 0, 0]], 0.009, (6, 4, 3)[level], frame)
            tube(mesh, "prop/paint", [head_bottom + [x * 0.5, 0, 0], head_bottom + [x, -0.15, 0.04], front + [x, 0, 0]], 0.011,
                 (6, 4, 3)[level], frame)
        # Seat post, saddle, stem, swept handlebar with grips.
        tube(mesh, "prop/metal-galvanized", [seat, seat + [0, 0.1, -0.03]], 0.012, sides)
        box(mesh, "vehicle/plastic", seat + [0, 0.12, -0.04], [0.13, 0.05, 0.24])
        tube(mesh, "prop/metal-galvanized", [head_top, head_top + [0, 0.12, -0.03]], 0.013, sides)
        bar = [head_top + [x, 0.12 + 0.02 * abs(x), -0.03 - 0.25 * x * x] for x in np.linspace(-0.3, 0.3, (7, 5, 3)[level])]
        tube(mesh, "prop/metal-galvanized", bar, 0.011, sides)
        for x in (-1, 1):
            tube(mesh, "vehicle/plastic", [bar[0 if x < 0 else -1], bar[0 if x < 0 else -1] + [x * 0.0, 0, -0.1]], 0.016, sides)
        if level < 2:
            # Mudguards: arcs over both wheels; a rear rack; chainring and cranks.
            for centre, a0, a1 in ((rear, -0.3, 2.2), (front, 0.9, 3.4)):
                arc = [centre + np.array([0, math.sin(t), math.cos(t)]) * (r + 0.03) for t in np.linspace(a0, a1, (10, 6)[level])]
                tube(mesh, "prop/metal-galvanized", arc, 0.022, 4)
            for x in (-0.07, 0.07):
                tube(mesh, "prop/paint", [rear + [x, 0, 0.02], rear + [x, 0.4, -0.02]], 0.007, 4, frame)
            box(mesh, "prop/paint", rear + [0, 0.42, 0.0], [0.15, 0.015, 0.4], color=frame)
            lathe(mesh, "prop/metal-galvanized", [(0.1, -0.005), (0.1, 0.005)], 16 if level == 0 else 8, at=bb + [0.06, 0, 0],
                  rotation=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float))
            for sx, sz in ((1, 1), (-1, -1)):
                crank = bb + [sx * 0.08, -sz * 0.08, sz * 0.12]
                tube(mesh, "prop/metal-galvanized", [bb + [sx * 0.08, 0, 0], crank], 0.01, 4)
                box(mesh, "vehicle/plastic", crank + [sx * 0.05, 0, 0], [0.1, 0.02, 0.06])
            box(mesh, "prop/paint", [0.05, 0.18, -0.25], [0.012, 0.012, 0.3], _rotation_x(0.6), frame)  # Kickstand.
    return Model(_levels(build), [_convex([-0.3, 0, -0.92], [0.3, 1.1, 0.92])], 1.1, 0.95)


def grass_clump():
    """A small grass tuft: three crossed alpha cards (two, one at lower LODs), 0.5 m across and 0.45 m tall."""
    def build(mesh, level):
        for k in range((3, 2, 1)[level]):
            a = k * math.pi / (3, 2, 1)[level] + 0.3
            right = np.array([math.cos(a), 0, math.sin(a)]) * 0.25
            quad(mesh, "veg/grass-blades", [0, 0.215, 0], right, [0, 0.225, 0], (0, 0, 1, 1), normals=np.tile([0, 1.0, 0], (4, 1)))
    return Model(_levels(build), [], 0.44, 0.25)


def flower_bed():
    """A 2 × 1 m flower bed along x: a low soil mound with clumps of flowers (crossed alpha cards, mixed colours)."""
    rng = np.random.default_rng(61)
    clumps = [(np.array([rng.uniform(-0.92, 0.92), 0.06, rng.uniform(-0.42, 0.42)]), rng.integers(0, 4), rng.uniform(0, math.pi),
               rng.uniform(0.8, 1.2)) for _ in range(34)]

    def build(mesh, level):
        box(mesh, "ground/dirt", [0, 0.0, 0], [2.04, 0.14, 1.04])
        for at, cell, a, scale in clumps[: (34, 14, 5)[level]]:
            u0, v0 = cell % 2 / 2, 1 - (cell // 2 + 1) / 2
            for k in range(2 if level < 2 else 1):
                right = np.array([math.cos(a + k * math.pi / 2), 0, math.sin(a + k * math.pi / 2)]) * 0.24 * scale
                quad(mesh, "veg/flowers", at + [0, 0.2 * scale, 0], right, [0, 0.2 * scale, 0], (u0, v0, u0 + 0.5, v0 + 0.5),
                     normals=np.tile([0, 1.0, 0], (4, 1)))
    return Model(_levels(build), [], 0.5, 1.1)


MODELS = {
    **{f"tree/{species}": partial(tree, species) for species in TREES},
    "bush/round": bush,
    "bush/hedge-segment": hedge,
    "prop/street-lamp": street_lamp,
    "prop/street-lamp-berlin": street_lamp_berlin,
    "prop/bench": bench,
    "prop/waste-basket": waste_basket,
    "prop/bollard": bollard,
    "prop/traffic-signal": traffic_signal,
    "prop/bicycle-stand": bicycle_stand,
    "prop/hydrant": hydrant,
    "prop/advertising-column": advertising_column,
    "prop/planter": planter,
    "prop/bicycle": bicycle,
    "grass/clump": grass_clump,
    "flowers/bed": flower_bed,
    **{name: partial(sign, name) for name in SIGNS},
}

from . import vehicles as _vehicles  # noqa: E402 (vehicles builds on the helpers above)



def _vehicle(name):
    return _vehicles.vehicle(name)


MODELS.update({name: partial(_vehicle, name) for name in _vehicles.VEHICLES})
