"""S4–S5: fit OSM footprints to the measured surface, then rebuild them as procedural buildings or landmarks.

The mesh is only measured: footprint shifts, heights and roof shapes come from the `top` raster; every surface is
regenerated with library materials. A building with building:part outlines is built from its parts, and its own
outline is dropped once the parts cover it. Each building becomes one Model with three LODs and convex colliders.

game.toml overrides, keyed by OSM id ("way/123" or "relation/45", the building or one of its parts):

    [buildings."way/123"]
    mode = "replace"            # replace | landmark | prism | building | drop
    file = "models/gate.glb"    # replace: the asset assembly places at the footprint (Model.extras["replace"])
    height = 26.0               # metres above the ground to the top
    style = "modern-office"     # a library style
    roof = "gabled"             # a roof:shape
    levels = 5
    material = "wall/brick-red"

"landmark" forces the landmark path (landmarks.py: classical columns, entablatures and porticos, see classical.py),
"prism" a plain flat-topped extrusion, "building" the procedural building path, and "drop" leaves the building out.
Gates, poorly fitting tagged landmarks and colonnades take the landmark path by themselves.

Ordinary buildings get facades by archetype (classical, stone, curtain, panel, plain; see facades.py): walls of
pieces that overlap or share a plane are left to one of them (wall_spans), so no two walls fight.
"""

import functools
import math
import random
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from . import facades, roofs
from .facades import TONE, WHITE, Mesh, Wall, dim
from .types import Collider, Model

SIMPLIFY = 0.3  # Douglas–Peucker tolerance for footprints, metres.
SHIFT = 2.0  # Largest footprint shift towards the measured edges, metres.
SHIFT_STEP = 0.25
LANDMARK_RMS = 1.5  # Roof fits worse than this send tagged landmarks to the landmark path.
TIER_RMS = 2.0  # Pieces fitting worse than this are split into height tiers.
TIERS = 5
# Triangles per LOD above which detail is shed: per m² of facade, at least the first and at most the second number.
# Small buildings stay well under the floor, so the median is far lower; very long facades hit the ceiling.
BUDGET = ((12.0, 20000, 60000), (1.6, 3000, 9000), (0.4, 800, 2400))
# Detail levels each LOD tries in turn until it fits its budget: (detail, bays per opening); 0 bays: no windows.
DETAILS = ([("full", 1), ("medium", 1), ("lean", 1), ("lean", 2), ("lean", 3), ("lean", 4)],
           [("medium", 1), ("lean", 1), ("lean", 2), ("lean", 3), ("lean", 4)],
           [("lean", 1), ("lean", 2), ("lean", 4), ("lean", 0)])
NEAR, FAR = 0.15, 0.5  # Probe distances off a wall: sharing its plane, and standing just in front of it.
TIER_TAGS = ("height", "min_height", "roof:", "building:levels", "building:min_level")
PART_COVER = 0.5  # Share of a building's outline its parts must cover to replace it.
LANDMARK_MATERIAL = "stone/sandstone-carved"
GATES = {"gate", "city_gate", "monument", "memorial", "obelisk", "triumphal_arch", "column", "statue"}
LANDMARKS = GATES | {"tower", "church", "cathedral", "chapel", "mosque", "temple", "synagogue", "shrine",
                     "bell_tower", "castle", "fort", "dome", "minaret", "lighthouse"}
COMMERCIAL = {"retail", "commercial", "kiosk", "supermarket", "shop", "mall"}
SHOP_AMENITIES = {"restaurant", "cafe", "bar", "pub", "fast_food", "bank", "pharmacy", "ice_cream", "post_office",
                  "bureau_de_change", "biergarten", "car_rental"}
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
         "living_street", "pedestrian", "footway", "primary_link", "secondary_link", "tertiary_link", "road"}
# Defaults for style keys a style file leaves out. archetype picks the facade grammar: classical (Gründerzeit
# stucco), stone (modern stone-clad), curtain (glass), panel (prefab slabs) or plain; ornaments are chances.
STYLE = dict(name="generic", archetype="plain", floor_height=[3.0, 3.4], ground_floor_height=[3.4, 4.0],
             bay_width=[2.8, 3.4],
             window=dict(width=[1.1, 1.3], height=[1.4, 1.7], reveal=0.18, sill=True, frame="trim/metal-dark"),
             walls={"wall/plaster-white": 1}, roofs=dict(flat="roof/membrane", pitched=["roof/tiles-red"]),
             trim="trim/stone", glass="glass/window", doors=["door/wood"], shop="shop/glass", ornament=0.2,
             parapet=[0.3, 0.8], plinth=dict(height=0.5, material="wall/concrete"), balconies=0.0, ornaments={})
# Material names newer than some libraries, with the nearest older one to use until the library has them.
FALLBACK = {"wall/ashlar-sandstone": "wall/sandstone", "wall/rustication": "wall/ashlar-sandstone",
            "wall/render-smooth": "wall/plaster-white", "stone/sandstone-smooth": "trim/stone",
            "stone/granite-dark": "ground/curb", "metal/railing": "trim/metal-dark",
            "roof/zinc-standing-seam": "roof/zinc", "awning/fabric": None, "sign/shop": None}
WALL_MATERIALS = {"brick": "wall/brick-red", "sandstone": "wall/ashlar-sandstone", "limestone": "wall/limestone",
                  "stone": "wall/limestone", "marble": "wall/limestone", "granite": "wall/limestone",
                  "concrete": "wall/concrete", "reinforced_concrete": "wall/concrete", "panel": "wall/concrete-panel",
                  "glass": "wall/glass-curtain", "mirror": "wall/glass-curtain", "metal": "wall/metal-panel",
                  "steel": "wall/metal-panel", "aluminium": "wall/metal-panel", "copper": "wall/metal-panel"}
PLASTERS = {"wall/plaster-white": (236, 234, 228), "wall/plaster-beige": (222, 205, 175),
            "wall/plaster-grey": (175, 175, 172), "wall/plaster-yellow": (230, 205, 130)}
ROOF_MATERIALS = {"roof_tiles": "roof/tiles-red", "tiles": "roof/tiles-red", "tile": "roof/tiles-red",
                  "slate": "roof/slate", "metal": "roof/zinc-standing-seam", "zinc": "roof/zinc-standing-seam",
                  "tin": "roof/zinc-standing-seam", "copper": "roof/copper-green", "gravel": "roof/gravel",
                  "concrete": "roof/membrane", "tar_paper": "roof/membrane", "asphalt": "roof/membrane",
                  "plastic": "roof/membrane", "grass": "roof/gravel", "glass": "glass/window", "stone": "roof/slate",
                  "eternit": "roof/slate"}
COLOURS = {"white": (255, 255, 255), "black": (30, 30, 30), "gray": (128, 128, 128), "grey": (128, 128, 128),
           "lightgrey": (211, 211, 211), "lightgray": (211, 211, 211), "darkgrey": (90, 90, 90),
           "darkgray": (90, 90, 90), "silver": (192, 192, 192), "red": (180, 50, 40), "darkred": (139, 0, 0),
           "maroon": (128, 0, 0), "brown": (140, 90, 50), "tan": (210, 180, 140), "beige": (225, 210, 180),
           "cream": (240, 230, 200), "ivory": (250, 245, 230), "yellow": (235, 210, 90), "orange": (230, 140, 50),
           "green": (70, 130, 70), "darkgreen": (30, 80, 30), "mediumaquamarine": (102, 205, 170),
           "blue": (60, 90, 170), "lightblue": (170, 200, 230), "sand": (215, 195, 150), "pink": (240, 180, 180),
           "terracotta": (190, 100, 70), "khaki": (195, 176, 145), "wheat": (245, 222, 179), "salmon": (250, 128, 114)}


@dataclass
class Piece:
    """One massing: a building or building:part outline with its fitted heights and roof."""

    feature: object
    tags: dict  # The feature's tags over its parent's.
    ring: np.ndarray = None
    holes: list = field(default_factory=list)
    base: float = 0.0  # y of the bottom of the walls.
    ground: float = 0.0  # y the floors count from (raised for parts that start above the ground).
    roof: roofs.Roof = None
    rms: float = None
    cells: tuple = None  # Measured (x, z, y) inside the eroded outline, for the fit report.
    raised: bool = False


@dataclass
class Group:
    id: str
    tags: dict
    pieces: list
    shift: tuple = (0.0, 0.0)
    override: dict = field(default_factory=dict)
    mode: str = "building"
    rms: float = None


def build(reference, ground, config, progress):
    quality = config.get("quality", "high")
    overrides = config.get("buildings", {}) or {}
    groups = [g for g in select(reference.osm, overrides) if _in_play(g, reference.play)]
    progress("Aligning footprints")
    align(groups, reference, ground)
    progress("Fitting buildings", 0, len(groups), "buildings")
    trust = trusted(reference.top)
    for done, group in enumerate(groups, 1):
        fit(group, reference, ground, trust)
        decide(group)
        progress("Fitting buildings", done, len(groups), "buildings")
    context = Context(reference, ground, [p for g in groups if g.mode != "drop" for p in g.pieces], quality)
    models = []
    progress("Generating buildings", 0, len(groups), "buildings")
    for done, group in enumerate(groups, 1):
        model = generate(group, context, config)
        if model is not None:
            models.append(model)
        progress("Generating buildings", done, len(groups), "buildings")
    return models


def accounting(reference, models, config=None):
    """What became of every OSM building and building:part, for report.json: feature counts per outcome.

    A building with parts is one Model whose extras list the parts; its own outline is left out (extras outline =
    False) once the parts cover it. Features outside the play area and those dropped in game.toml get no Model."""
    drops = {k for k, v in ((config or {}).get("buildings") or {}).items() if (v or {}).get("mode") == "drop"}
    built = {m.extras.get("osm"): m for m in models}
    parts = {p for m in models for p in m.extras.get("parts", [])}
    outcome = {}
    for f in reference.osm:
        if f.kind != "building":
            continue
        model = built.get(f.id)
        reason = ("dropped in game.toml" if f.id in drops else "part of a model" if f.id in parts
                  else "outside play area" if model is None
                  else "outline replaced by its parts" if not model.extras.get("outline", True)
                  else f"{model.extras['kind']} model")
        outcome[reason] = outcome.get(reason, 0) + 1
    kinds = {}
    for m in models:
        kind = "replaced" if "replace" in m.extras else m.extras.get("kind", "building")
        kinds[kind] = kinds.get(kind, 0) + 1
    return dict(models=kinds, features=outcome)


# Selection.

def _in_play(group, play):
    """Whether any of the group's outline lies inside the play area."""
    if play is None or len(play) < 3:
        return True
    points = np.concatenate([p.feature.points for p in group.pieces])
    return bool(roofs.inside([play], points[:, 0], points[:, 1]).any())


def select(features, overrides=None):
    """Groups of pieces: a building with its parts, or a lone building or part."""
    overrides = overrides or {}
    buildings = [f for f in features if f.kind == "building" and len(f.points) >= 3]
    parts = [f for f in buildings if f.tags.get("building:part", "no") != "no"]
    whole = [f for f in buildings if f.tags.get("building:part", "no") == "no"]
    areas = {f.id: abs(roofs.area(f.points)) for f in buildings}
    children = {f.id: [] for f in whole}
    lone = []
    for part in parts:
        c = roofs.centroid(part.points)
        owners = [f for f in whole if _bounds_hit(f.points, c)
                  and roofs.inside([f.points, *f.holes], c[0], c[1])]
        if owners:
            children[min(owners, key=lambda f: areas[f.id]).id].append(part)
        else:
            lone.append(part)
    groups = []
    for f in whole:
        kids = children[f.id]
        pieces = [Piece(k, {**_inherit(f.tags), **k.tags}) for k in kids]
        if not kids or coverage(f, kids) < PART_COVER:
            pieces.append(Piece(f, dict(f.tags)))
        groups.append(Group(f.id, dict(f.tags), pieces))
    groups += [Group(f.id, dict(f.tags), [Piece(f, dict(f.tags))]) for f in lone]
    for group in groups:
        group.override = dict(overrides.get(group.id) or {})
        for piece in group.pieces:
            group.override.update(overrides.get(piece.feature.id) or {})
        group.mode = group.override.get("mode", "building")
    return groups


def _inherit(tags):
    """Parent tags a part takes over: what the building is, not its size or shape."""
    return {k: v for k, v in tags.items() if not k.startswith(TIER_TAGS)}


def _bounds_hit(points, c):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]


def coverage(building, parts, step=1.0):
    """Share of the building's outline covered by its parts, sampled on a grid."""
    lo, hi = building.points.min(axis=0), building.points.max(axis=0)
    x, z = np.meshgrid(np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step))
    x, z = x.ravel(), z.ravel()
    mask = roofs.inside([building.points, *building.holes], x, z)
    if not mask.any():
        return 1.0
    x, z = x[mask], z[mask]
    covered = np.zeros(len(x), bool)
    for part in parts:
        covered |= roofs.inside([part.points, *part.holes], x, z)
    return float(covered.mean())


# Alignment.

def align(groups, reference, ground):
    """Shift each block of touching buildings by the offset (≤ SHIFT) whose footprint edges best separate high
    inside from low outside on the measured surface, then simplify the outlines."""
    for block in blocks(groups):
        points, inward = [], []
        for group in block:
            for piece in group.pieces:
                for ring in [piece.feature.points, *piece.feature.holes]:
                    p, n = _edge_samples(ring)
                    points.append(p)
                    inward.append(n)
        shift = (0.0, 0.0)
        if points:
            shift = best_shift(np.concatenate(points), np.concatenate(inward), reference.top, ground.height)
        for group in block:
            group.shift = shift
            for piece in group.pieces:
                piece.ring, piece.holes = outline(piece.feature, shift)


def blocks(groups, pad=1.0):
    """Groups whose bounding boxes come within pad of each other, joined transitively."""
    boxes = [(np.min([p.feature.points.min(axis=0) for p in g.pieces], axis=0) - pad,
              np.max([p.feature.points.max(axis=0) for p in g.pieces], axis=0) + pad) for g in groups]
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            (a0, a1), (b0, b1) = boxes[i], boxes[j]
            if (a0 <= b1).all() and (b0 <= a1).all():
                parent[find(i)] = find(j)
    found = {}
    for i, group in enumerate(groups):
        found.setdefault(find(i), []).append(group)
    return list(found.values())


def _edge_samples(ring, step=0.5, shortest=2.0):
    """Points every step along edges longer than shortest, with unit normals pointing into the building."""
    ring = np.asarray(ring, float)
    out, normals = [], []
    for a, b, n in zip(ring, np.roll(ring, -1, axis=0), roofs.normals(ring)):
        length = math.dist(a, b)
        if length < shortest:
            continue
        t = (np.arange(int(length / step)) + 0.5) / int(length / step)
        out.append(a + t[:, None] * (b - a))
        normals.append(np.repeat(-n[None], len(t), axis=0))
    if not out:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(out), np.concatenate(normals)


def best_shift(points, inward, top, ground, reach=0.75):
    """Shift maximising the mean height step across footprint edges; zero unless it clearly beats no shift."""
    steps = np.arange(-SHIFT, SHIFT + 1e-9, SHIFT_STEP)
    dx, dz = (v.ravel() for v in np.meshgrid(steps, steps))
    inner = points + reach * inward
    outer = points - reach * inward
    level = ground.sample(points[:, 0], points[:, 1])
    hi = top.sample(inner[None, :, 0] + dx[:, None], inner[None, :, 1] + dz[:, None])
    lo = top.sample(outer[None, :, 0] + dx[:, None], outer[None, :, 1] + dz[:, None])
    # Low ground stands in for missing mesh outside, so edges at the capture border still count.
    lo = np.where(np.isnan(lo), level[None, :], lo)
    with np.errstate(invalid="ignore"):
        step = np.clip(hi - lo, -10, 10)
    valid = np.isfinite(step)
    if valid.sum(axis=1).max(initial=0) < 40:  # Too little outline to trust a shift.
        return (0.0, 0.0)
    score = np.where(valid, step, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    zero = int(np.argmin(dx ** 2 + dz ** 2))
    k = int(np.argmax(score - 0.02 * np.hypot(dx, dz)))  # A slight pull towards no shift breaks ties.
    if score[k] - score[zero] < 0.25:
        return (0.0, 0.0)
    return (float(dx[k]), float(dz[k]))


def outline(feature, shift):
    """Simplified, squared outline and holes, moved by shift. Small outlines (columns, kiosks) keep finer detail."""
    def tidy(ring, hole=False):
        ring = roofs.ccw(np.asarray(ring, float) + shift, hole)
        tolerance = min(SIMPLIFY, 0.08 * math.sqrt(abs(roofs.area(ring))))
        simple = roofs.clean(roofs.simplify(ring, tolerance), tolerance)
        if len(simple) < 3 or abs(roofs.area(simple)) < 0.8 * abs(roofs.area(ring)):
            simple = ring
        return roofs.clean(roofs.square(simple), tolerance)
    ring = tidy(feature.points)
    holes = [h for h in (tidy(h, True) for h in feature.holes) if len(h) >= 3 and abs(roofs.area(h)) >= 4]
    return ring, holes


# Fitting.

def number(value):
    """Metres from an OSM value such as "12", "12.5 m" or "40'"; None when unreadable."""
    if value is None:
        return None
    match = re.match(r"\s*(-?\d+(?:[.,]\d+)?)\s*(m|ft|'|)?", str(value))
    if not match:
        return None
    n = float(match.group(1).replace(",", "."))
    return n * 0.3048 if match.group(2) in ("ft", "'") else n


def ground_level(ring, ground):
    """Lowest and median ground y just outside an outline, where the ground was measured rather than filled in."""
    p = roofs.offset(np.asarray(ring, float), 1.0)
    samples = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n = max(1, int(math.dist(a, b)))
        t = np.arange(n) / n
        samples.append(a + t[:, None] * (b - a))
    s = np.concatenate(samples)
    y = ground.height.sample(s[:, 0], s[:, 1])
    return float(np.nanmin(y)), float(np.nanmedian(y))


def cells(raster, rings, erode):
    """(x, z, value) of raster cells inside rings, at least erode from their edges."""
    p = np.concatenate(rings)
    lo, hi = p.min(axis=0), p.max(axis=0)
    r0, c0 = raster.index(lo[0], lo[1])
    r1, c1 = raster.index(hi[0], hi[1])
    x = raster.x0 + (np.arange(c0, c1 + 1) + 0.5) * raster.cell
    z = raster.z0 + (np.arange(r0, r1 + 1) + 0.5) * raster.cell
    x, z = np.meshgrid(x, z)
    v = raster.values[r0:r1 + 1, c0:c1 + 1]
    mask = roofs.inside(rings, x, z)
    if erode > 0 and mask.any():
        mask[mask] = roofs.distance(rings, x[mask], z[mask]) >= erode
    return x[mask], z[mask], v[mask]


def trusted(top, margin=2.0):
    """Cells of top with mesh all around them: the mesh's cut edges and holes carry skirts and stray heights."""
    ok = np.isfinite(top.values)
    r = int(round(margin / top.cell))
    padded = np.pad(ok, r + 1, constant_values=False).astype(np.int32)
    s = padded.cumsum(0).cumsum(1)
    k = 2 * r + 1
    window = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return window[:ok.shape[0], :ok.shape[1]] == k * k


def fit(group, reference, ground, trust=None):
    """Heights and roof of every piece: measured where the mesh shows its top, else from OSM tags. Pieces that no
    single roof fits are split into height tiers."""
    if trust is None:
        trust = trusted(reference.top)
    names = {"height": "height", "roof": "roof:shape", "levels": "building:levels"}
    for piece in group.pieces:
        piece.tags = {**piece.tags, **{names[k]: v for k, v in group.override.items() if k in names}}
    for piece in group.pieces:
        tags = piece.tags
        low, mid = ground_level(piece.ring, ground)
        piece.ground = mid
        min_height = number(tags.get("min_height"))
        if min_height is None and number(tags.get("building:min_level")):
            min_height = number(tags.get("building:min_level")) * 3.0
        piece.raised = bool(min_height and min_height > 0.5)
        piece.base = mid + min_height if piece.raised else low - 0.3
        shape = tags.get("roof:shape")
        direction = _direction(tags)
        x, z, y = _measure(piece, group.pieces, reference, trust)
        roof = rms = None
        tagged = number(tags.get("height")) is not None or number(tags.get("building:levels")) is not None
        # Narrow pieces catch their neighbours' walls in the raster: their tags know better.
        narrow = roofs.box(piece.ring)[3] < 2.5 or abs(roofs.area(piece.ring)) < 40
        if len(y) and "height" not in group.override and not (tagged and narrow):
            roof, rms = roofs.fit(x, z, y, piece.ring, shape, direction)
            if roof.ridge < piece.base + 1.0:
                roof = None
        if roof is None:
            roof = _osm_roof(piece, tags, shape, direction, number(tags.get("height")))
            if len(y):
                rms = math.sqrt(np.mean((y - roof.y(x, z)) ** 2))
        if _canopy(tags) and not piece.raised:
            piece.raised, piece.base = True, max(mid + 2.2, roof.eave - 0.5)
        # Ground pieces get at least a storey of wall; raised ones (roofs on columns, bridges) a sliver.
        least = 0.3 if piece.raised or _canopy(tags) else 2.0
        if roof.eave < piece.base + least:
            roof.eave = piece.base + least
        roofs.creases(roof, piece.ring)
        piece.roof, piece.rms, piece.cells = roof, rms, (x, z, y)
    if group.mode == "building" and "height" not in group.override:
        tiered = []
        for piece in group.pieces:
            poor = piece.rms is not None and piece.rms > TIER_RMS and not piece.raised and not _canopy(piece.tags)
            tiered += tiers(piece, reference, trust) if poor else [piece]
        group.pieces = tiered
    measured = [p.cells for p in group.pieces if p.cells is not None and len(p.cells[2])]
    if measured:
        x, z, y = (np.concatenate(v) for v in zip(*measured))
        group.rms = float(math.sqrt(np.nanmean((y - composite(group.pieces, x, z)) ** 2)))


def composite(pieces, x, z):
    """Top of the rebuilt massing at points: the highest roof over each, NaN outside every piece."""
    top = np.full(len(x), np.nan)
    for piece in pieces:
        hit = roofs.inside([piece.ring, *piece.holes], x, z)
        if hit.any():
            top[hit] = np.fmax(top[hit], piece.roof.y(x[hit], z[hit]))
    return top


def _canopy(tags):
    return tags.get("building") == "roof" or tags.get("building:part") == "roof"


def _small(piece):
    """Columns, posts and kiosks: too small for floors, windows or mouldings."""
    _, _, hu, hv = roofs.box(piece.ring)
    return hv < 1.2 or abs(roofs.area(piece.ring)) < 10


def _measure(piece, pieces, reference, trust):
    """Measured top inside the eroded outline, leaving out cells under pieces standing on or in this one; nothing
    when too little of the outline is measured."""
    rings = [piece.ring, *piece.holes]
    area = abs(roofs.area(piece.ring)) - sum(abs(roofs.area(h)) for h in piece.holes)
    perimeter = sum(_perimeter(r) for r in rings)
    for erode in (1.0, 0.4, 0.15):
        x, z, y = cells(reference.top, rings, erode)
        expected = (area - perimeter * erode) / reference.top.cell ** 2
        if expected >= 6:
            break
    if not len(y):
        return x, z, y
    row, col = reference.top.index(x, z)
    keep = np.isfinite(y) & trust[row, col]
    for other in pieces:
        if other is not piece and _above(other, piece) and _overlap(other.ring, x, z):
            keep &= ~roofs.inside([other.ring, *other.holes], x, z)
    if keep.sum() < max(6, 0.4 * expected):
        keep[:] = False
    return x[keep], z[keep], y[keep]


def _above(other, piece):
    """Whether other stands on or rises out of piece, hiding piece's own top where they overlap."""
    low, high = number(other.tags.get("min_height")) or 0, number(other.tags.get("height"))
    own_low, own_high = number(piece.tags.get("min_height")) or 0, number(piece.tags.get("height"))
    if low > own_low + 0.5:
        return True
    if own_low > low + 0.5:
        return False
    smaller = abs(roofs.area(other.ring)) < abs(roofs.area(piece.ring))
    return smaller and (high is None or own_high is None or high > own_high)


def _overlap(ring, x, z):
    if not len(x):
        return False
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    return lo[0] <= x.max() and hi[0] >= x.min() and lo[1] <= z.max() and hi[1] >= z.min()


def _direction(tags):
    """Downhill unit vector (x, z) from roof:direction, or "across" from roof:orientation."""
    value = tags.get("roof:direction")
    if value:
        compass = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
                   "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
        degrees = compass.get(value.upper().strip(), number(value))
        if degrees is not None:
            a = math.radians(degrees)
            return np.array([math.sin(a), -math.cos(a)])
    if tags.get("roof:orientation") == "across":
        return "across"
    return None


def _osm_roof(piece, tags, shape, direction, top):
    """Roof from OSM height tags, levels, or a three-floor default."""
    levels = number(tags.get("building:levels"))
    min_level = number(tags.get("building:min_level")) or 0
    roof_height = number(tags.get("roof:height"))
    if roof_height is None and number(tags.get("roof:levels")):
        roof_height = number(tags.get("roof:levels")) * 2.8
    shape = roofs.ALIASES.get(shape, shape) if shape else "flat"
    if shape not in roofs.SHAPES:
        shape = "flat"
    _, _, hu, hv = roofs.box(piece.ring)
    if shape == "flat":
        roof_height = 0.0
    elif roof_height is None:
        roof_height = min(hv * (0.7 if shape != "dome" else 1.0), 8.0)
    if top is not None:
        ridge = piece.ground + top
    elif levels:
        ridge = piece.ground + (levels + min_level) * 3.2 + roof_height
    else:
        ridge = max(piece.base, piece.ground) + 3 * 3.2 + roof_height
    if _canopy(tags):
        roof_height = 0.0
    return roofs.roof(shape, piece.ring, ridge - roof_height, roof_height, direction)


def decide(group):
    """Building, landmark, or one of the override modes."""
    if group.mode != "building" or "mode" in group.override:
        return
    kinds = {group.tags.get(k) for k in ("building", "historic", "man_made", "tower:type")}
    tagged = bool(kinds & LANDMARKS) or "tower:type" in group.tags
    if kinds & GATES:
        group.mode = "landmark"
    elif tagged and group.rms is not None and group.rms > LANDMARK_RMS:
        group.mode = "landmark"
    else:
        from . import landmarks  # landmarks imports this module, so the import waits until here.

        if landmarks.wants(group):  # Colonnades: the classical kit builds them, not the facade grammar.
            group.mode = "landmark"


# Generation.

class Context:
    """Scene-wide lookups for generation: neighbours for party walls, roads for doors, shops for shopfronts."""

    def __init__(self, reference, ground, pieces, quality):
        self.reference, self.ground, self.quality = reference, ground, quality
        self.pieces = pieces
        self.order = {id(p): k for k, p in enumerate(pieces)}
        self.bounds = np.array([np.r_[p.ring.min(axis=0), p.ring.max(axis=0)] for p in pieces]).reshape(-1, 4)
        segments = []
        for f in reference.osm:
            if f.kind == "line" and f.tags.get("highway") in ROADS and len(f.points) >= 2:
                segments += [(a, b) for a, b in zip(f.points[:-1], f.points[1:])]
        self.roads = np.array(segments, float).reshape(-1, 2, 2)
        self.shops = np.array([f.points[0] for f in reference.osm if f.kind == "node" and (
            "shop" in f.tags or f.tags.get("amenity") in SHOP_AMENITIES)], float).reshape(-1, 2)
        self.unit = _unit_materials()
        self.trust = trusted(reference.top)
        self.cache = {}

    def spans(self, piece, p, q):
        """(hidden, blind) stretches of the wall from p to q, cached as every LOD asks again."""
        key = (id(piece), float(p[0]), float(p[1]), float(q[0]), float(q[1]))
        if key not in self.cache:
            self.cache[key] = wall_spans(piece, p, q, self)
        return self.cache[key]

    def road(self, point):
        """Nearest road point to point, or None."""
        if not len(self.roads):
            return None
        a, b = self.roads[:, 0], self.roads[:, 1]
        d = b - a
        t = np.clip(((point - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-12), 0, 1)
        q = a + t[:, None] * d
        k = int(np.argmin(np.hypot(*(q - point).T)))
        return q[k]


def wall_spans(piece, p, q, context):
    """Where other pieces cover a wall: (hidden, blind) lists of (u0, u1, lo, hi).

    hidden: another piece stands within NEAR of the wall (in front, around it, or sharing its plane and winning
    because it is taller or larger), so this wall builds nothing there: two walls in one plane would fight and fill
    each other's window holes. blind: a neighbour stands a little further out, so the wall is built but gets no
    windows or mouldings. lo and hi are the other piece's bottom and roof height there."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.hypot(*(q - p)))
    if L < 1e-6 or not len(context.bounds):
        return [], []
    d = (q - p) / L
    n = np.array([-d[1], d[0]])
    lo, hi = np.minimum(p, q) - FAR - 0.1, np.maximum(p, q) + FAR + 0.1
    b = context.bounds
    near = np.nonzero((b[:, 0] <= hi[0]) & (b[:, 2] >= lo[0]) & (b[:, 1] <= hi[1]) & (b[:, 3] >= lo[1]))[0]
    others = [context.pieces[k] for k in near if context.pieces[k] is not piece]
    if not others:
        return [], []
    # Break the wall where any other outline crosses one of the probe lines, or has a corner next to the wall.
    cuts = {0.0, L}
    for other in others:
        for ring in [other.ring, *other.holes]:
            a = np.asarray(ring, float)
            c = np.roll(a, -1, axis=0)
            rel = a - p
            close = np.abs(rel @ n) <= FAR + 0.05
            cuts |= set(np.clip(rel[close] @ d, 0, L).tolist())
            for s in (FAR, NEAR, -NEAR):
                da, dc = rel @ n - s, (c - p) @ n - s
                cross = da * dc < 0
                if cross.any():
                    t = da[cross] / (da[cross] - dc[cross])
                    points = a[cross] + t[:, None] * (c[cross] - a[cross])
                    cuts |= set(np.clip((points - p) @ d, 0, L).tolist())
    cuts = sorted(cuts)
    own = abs(roofs.area(piece.ring))
    hidden, blind = [], []
    for u0, u1 in zip(cuts, cuts[1:]):
        if u1 - u0 < 0.02:
            continue
        m = p + d * (u0 + u1) / 2
        probes = np.array([m + FAR * n, m + NEAR * n, m - NEAR * n])
        mine = float(piece.roof.y(m[0], m[1]))
        cover, shut = None, None
        for other in others:
            hit = roofs.inside([other.ring, *other.holes], probes[:, 0], probes[:, 1])
            if not hit.any():
                continue
            # The other roof's height where it was hit: pitched planes don't extend past their outline.
            x, z = probes[int(np.argmax(hit))]
            top = float(other.roof.y(x, z))
            span = (other.base, top)
            area = abs(roofs.area(other.ring))
            wins = (top > mine + 0.05 or abs(top - mine) <= 0.05 and (
                area > own + 1e-6 or abs(area - own) <= 1e-6 and context.order[id(other)] < context.order[id(piece)]))
            if hit[1] or hit[2] and wins:
                cover = span if cover is None else (min(cover[0], span[0]), max(cover[1], span[1]))
            elif hit[0]:
                shut = span if shut is None else (min(shut[0], span[0]), max(shut[1], span[1]))
        if cover:
            hidden.append((u0, u1, *cover))
        if shut:
            blind.append((u0, u1, *shut))
    return _merge(hidden), _merge(blind)


def _merge(spans):
    out = []
    for span in spans:
        if out and abs(out[-1][1] - span[0]) < 1e-6 and out[-1][2:] == span[2:]:
            out[-1] = (out[-1][0], span[1], *span[2:])
        else:
            out.append(span)
    return out


def generate(group, context, config):
    if group.mode == "drop":
        return None
    first = group.pieces[0]
    rings = [p.ring for p in group.pieces]
    areas = [abs(roofs.area(r)) for r in rings]
    centre = tuple(float(v) for v in np.average([roofs.centroid(r) for r in rings], axis=0, weights=areas))
    top = max(p.roof.ridge for p in group.pieces)
    height = top - min(p.ground for p in group.pieces)
    parts = list(dict.fromkeys(p.feature.id for p in group.pieces if p.feature.id != group.id))
    extras = dict(osm=group.id, parts=parts, outline=any(p.feature.id == group.id for p in group.pieces),
                  height=round(height, 2), roof=max(group.pieces, key=lambda p: abs(roofs.area(p.ring))).roof.shape,
                  fit_rms=None if group.rms is None else round(group.rms, 2), shift=list(group.shift))
    if "name" in group.tags:
        extras["name"] = group.tags["name"]
    seed = zlib.crc32(group.id.encode())
    if group.mode == "replace":
        c, axis, hu, hv = roofs.box(first.ring)
        # The asset stands on base at center, its x axis along the footprint's long side (yaw counter-clockwise
        # from east, seen from above), and is scaled to size (x, y, z) by assembly.
        extras.update(kind="landmark", replace=group.override.get("file"),
                      base=round(min(p.base for p in group.pieces), 2), center=[float(c[0]), float(c[1])],
                      size=[round(2 * hu, 2), round(height, 2), round(2 * hv, 2)],
                      yaw=round(math.atan2(-axis[1], axis[0]), 4))
        colliders = [c for p in group.pieces for c in colliders_for(p, prism=True)]
        return Model("landmark/" + group.id.replace("/", "-"), centre, [[], [], []], colliders, extras)
    if group.mode in ("landmark", "prism"):
        return landmark(group, context, config, centre, extras, seed)
    style = choose_style(group, config)
    look = make_look(style, group, random.Random(seed))
    extras.update(kind="building", style=style.get("name", "generic"), wall=look["wall"],
                  archetype=look["archetype"])
    area = sum(_perimeter(r) * max(p.roof.eave - p.base, 0.0) for p in group.pieces for r in [p.ring, *p.holes])
    lods, details = [], []
    draft = context.quality == "draft"
    for lod, (density, least, most) in enumerate(BUDGET):
        limit = min(most, max(least, density * area))
        mesh, used = None, None
        for detail, grouped in DETAILS[lod]:
            mesh = Mesh(context.unit)
            # Each piece draws from its own generator, so every LOD and detail level lays out the same facades.
            for k, piece in enumerate(group.pieces):
                massing(mesh, piece, group, {**look, "detail": detail, "group": grouped}, context,
                        2 if draft else lod, random.Random(seed + 7 + k), draft)
            used = (detail, grouped)
            if draft or mesh.count() <= limit:
                break
        lods.append(mesh.parts())
        details.append(used)
    colliders = [c for p in group.pieces for c in colliders_for(p)]
    extras["triangles"] = [int(sum(len(p.indices) // 3 for p in parts)) for parts in lods]
    extras["detail"] = [f"{d}/{g}" for d, g in details]
    return Model("building/" + group.id.replace("/", "-"), centre, lods, colliders, extras)


# Styles and looks.

def choose_style(group, config):
    """The configured style, or for "auto" the library style whose match hints fit the tags best."""
    name = group.override.get("style") or config.get("style") or "auto"
    if name != "auto":
        return _style(name)
    tags = group.tags
    kind = tags.get("building", "yes")
    material = tags.get("building:material", "")
    year = _year(tags)
    levels = number(tags.get("building:levels")) or 0
    era = ("gdr" if year and 1960 <= year <= 1990 and kind in ("apartments", "residential", "dormitory") else
           "new" if (year and year >= 1950) or material in ("glass", "metal", "steel", "concrete") else
           "old" if (year and year < 1945) or material in ("sandstone", "brick", "plaster", "stone", "limestone")
           else None)
    best, score = "generic", 0.0
    for name in _style_names():
        style = _style(name)
        match = style.get("match") or {}
        s = 0.0
        if kind in (match.get("building") or []):
            s += 2
        if era and match.get("era") == era:
            s += 1.5
        if material and material in (match.get("material") or []):
            s += 2
        if match.get("levels") and levels and match["levels"][0] <= levels <= match["levels"][1]:
            s += 0.5
        if material == "glass" and name == "modern-office":
            s += 3
        if s > score:
            best, score = name, s
    return _style(best)


def _year(tags):
    return number(re.sub(r"^\D*", "", tags.get("start_date", ""))[:4] or None)


def _style_names():
    try:
        from . import library
        return list(library.styles())
    except Exception:
        return ["generic"]


@functools.cache
def _style(name):
    """A library style over the defaults, read once."""
    try:
        from . import library
        style = library.style(name)
    except Exception:
        style = {}
    merged = {**STYLE, **(style or {})}
    for key in ("window", "plinth", "roofs", "ornaments"):
        merged[key] = {**STYLE[key], **((style or {}).get(key) or {})}
    merged.setdefault("name", name)
    return merged


def _unit_materials():
    """Library materials mapped once per quad (size 0, such as doors) rather than tiled in metres. Read from the
    catalogue, so nothing is downloaded or generated here."""
    try:
        from .library import MATERIALS
    except Exception:
        return frozenset({"door/wood", "door/metal-glass", "awning/fabric", "sign/shop"})
    size = lambda spec: MATERIALS.get(spec["like"], spec).get("size", 1) if "like" in spec else spec.get("size", 1)
    return frozenset(name for name, spec in MATERIALS.items() if size(spec) == 0)


@functools.cache
def _catalogue():
    try:
        from .library import MATERIALS
        return frozenset(MATERIALS)
    except Exception:
        return None


def material(name):
    """name, or the nearest material the library has while newer ones are missing from it; None for none."""
    known, seen = _catalogue(), set()
    while name and known is not None and name not in known and name not in seen:
        seen.add(name)
        name = FALLBACK.get(name)
    return name


def _atlas(name):
    """(columns, rows) of a library atlas material such as the shop signs, row-major from the top-left."""
    try:
        from .library import MATERIALS
        return tuple(MATERIALS[name]["extras"]["atlas"])
    except Exception:
        return (2, 8)


def _pick(value, rng):
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return rng.uniform(*value)
    return value


def _weighted(choices, rng):
    if isinstance(choices, str):
        return choices
    if isinstance(choices, dict):
        names, weights = list(choices), list(choices.values())
        return rng.choices(names, weights)[0]
    return rng.choice(list(choices))


def colour(value):
    """sRGB 0–255 triple from an OSM colour, or None."""
    if not value:
        return None
    value = value.strip().lower().replace(" ", "")
    if re.fullmatch(r"#?[0-9a-f]{6}", value):
        value = value.lstrip("#")
        return tuple(int(value[k:k + 2], 16) for k in (0, 2, 4))
    if re.fullmatch(r"#[0-9a-f]{3}", value):
        return tuple(int(c * 2, 16) for c in value[1:])
    return COLOURS.get(value)


def tint(rgb, strength):
    """Linear RGBA multiplier nudging a material towards an sRGB colour, normalised so the brightest channel is 1."""
    if rgb is None:
        return (1.0, 1.0, 1.0, 1.0)
    top = max(max(rgb), 1)
    linear = [((c / top + 0.055) / 1.055) ** 2.4 for c in rgb]
    return tuple(1 - strength + strength * c for c in linear) + (1.0,)


def _scale(colour, k):
    return (min(colour[0] * k, 1.2), min(colour[1] * k, 1.2), min(colour[2] * k, 1.2), 1.0)


def make_look(style, group, rng):
    """Everything the facade grammar needs for one building, drawn once so all its walls agree."""
    tags = group.tags
    window = style["window"]
    ornaments = style["ornaments"]
    wall, wall_tint = _wall(tags, style, rng, group.override.get("material"))
    plinth = style.get("plinth") or {}
    archetype = style.get("archetype", "plain")
    if wall == "wall/glass-curtain":
        archetype = "curtain"
    elif archetype == "classical" and wall in ("wall/concrete", "wall/concrete-panel", "wall/metal-panel"):
        archetype = "plain"
    elif archetype == "stone" and wall in ("wall/metal-panel",):
        archetype = "curtain"
    look = dict(
        style=style, archetype=archetype, wall=wall, tint=wall_tint, trim=material(style.get("trim", "trim/stone")),
        trim_tint=WHITE, frame=material(window.get("frame", "trim/metal-dark")), frame_tint=WHITE,
        glass=material(style.get("glass", "glass/window")),
        door=material(_weighted(style.get("doors") or ["door/wood"], rng)), shop=material(style.get("shop", "shop/glass")),
        shop_frame=material("trim/metal-dark"), metal=material("metal/railing"), mullion=material("trim/metal-dark"),
        spandrel=material("wall/metal-panel"), step=material("stone/granite-dark"),
        sign=material("sign/shop"), sign_cells=_atlas("sign/shop"), sign_aspect=4.0, awning=material("awning/fabric"),
        fascia=material("trim/metal-dark"),
        reveal=min(max(float(window.get("reveal", 0.18)), 0.15), 0.35),
        sills=bool(window.get("sill", True)), sill=0.9, window_width=_pick(window["width"], rng),
        window_height=_pick(window["height"], rng), bay=_pick(style["bay_width"], rng),
        floor_height=_pick(style["floor_height"], rng), ground_floor_height=_pick(style["ground_floor_height"], rng),
        ornament=float(style.get("ornament", 0.2)), balconies=float(style.get("balconies", 0.0)),
        parapet=_pick(style.get("parapet", [0.0, 0.0]), rng),
        plinth=(float(plinth.get("height", 0.0)), material(plinth.get("material", wall))), door_width=1.2,
        door_height=2.4, curtain=archetype == "curtain", surround=0.0, cornice="simple", overhang=0.45,
        steps=2, ground_wall=None, eaves=material("trim/wood-painted"), eaves_tint=WHITE)
    look.update(_roof_look(tags, style, rng))
    chance = lambda key: rng.random() < float(ornaments.get(key, 0.0))
    rgb = colour(tags.get("building:colour"))
    if archetype == "classical":
        stone = "sandstone" in wall or "limestone" in wall or "ashlar" in wall
        brick = "brick" in wall
        # Stucco on render is the wall's own colour a shade lighter, or a contrasting cream; stone gets dressed
        # stone, brick gets sandstone trims.
        if stone:
            look.update(trim=material("stone/sandstone-smooth"), trim_tint=_scale(wall_tint, 1.04))
        elif brick:
            look.update(trim=material("stone/sandstone-smooth"), trim_tint=WHITE)
        elif rng.random() < 0.5:
            look.update(trim=wall, trim_tint=_scale(wall_tint, 1.07))
        else:
            look.update(trim=material("wall/render-smooth"), trim_tint=tint((238, 230, 210), 0.5))
        look.update(surround=rng.uniform(0.15, 0.2), cornice="classical", overhang=0.0,
                    heads=rng.choice([("pediment", "hood", "keystone"), ("hood", "hood", "keystone"),
                                      ("pediment", "hood", ""), ("segment", "hood", "keystone"), ("hood", "", "")]),
                    alternate=chance("alternate"), arches=chance("arches"), quoins=chance("quoins"),
                    pilasters=chance("pilasters"), sill_course=chance("sill_course"), aprons=chance("aprons"),
                    rails=float(ornaments.get("rails", 0.0)), ground_sill=1.15, margin=0.9)
        if chance("rustication"):
            look["ground_wall"] = material("wall/rustication") if not brick else wall
        frame_colours = [(1.0, 1.0, 1.0), (0.32, 0.22, 0.15), (0.16, 0.2, 0.16), (0.82, 0.78, 0.7)]
        look["frame_tint"] = (*rng.choices(frame_colours, [5, 2, 1, 2])[0], 1.0)
    elif archetype == "stone":
        stone = "sandstone" in wall or "limestone" in wall or "ashlar" in wall
        if rng.random() < 0.5:  # Anodised aluminium rather than dark bronze frames.
            look.update(frame=material("trim/wood-painted"), frame_tint=(0.62, 0.63, 0.64, 1.0))
        look.update(reveal=max(look["reveal"], 0.3), cornice="slab", sills=False, frame_width=0.05,
                    quoins=False, overhang=0.0, steps=1, trim=material("stone/sandstone-smooth") if stone else wall,
                    trim_tint=_scale(wall_tint, 1.02), transoms=rng.random() < 0.5, door_height=3.0, door_width=1.8)
    elif archetype == "curtain":
        look.update(wall=material("trim/metal-dark"), tint=WHITE, mullion=material("trim/metal-dark"),
                    spandrel=material("wall/metal-panel"), cornice="thin", reveal=0.15, sills=False,
                    curtain_reveal=0.14, overhang=0.0, steps=0, trim=material("trim/metal-dark"),
                    lod2_wall=material("wall/glass-curtain"), door_height=3.0,
                    door_width=2.0)
    elif archetype == "panel":
        look.update(cornice="thin", overhang=0.3, frame=material("trim/wood-painted"), steps=1,
                    trim=material("wall/concrete"), trim_tint=WHITE, sill_out=0.06, balcony_depth=1.2)
    else:
        if rng.random() < float(ornaments.get("surrounds", 0.5)):
            look["surround"] = rng.uniform(0.1, 0.14)
        look.update(cornice="simple", trim=wall if "plaster" in wall else look["trim"],
                    trim_tint=_scale(wall_tint, 1.06) if "plaster" in wall else WHITE, quoins=chance("quoins"))
    if archetype in ("classical", "plain"):
        palette = [(0.55, 0.12, 0.1), (0.12, 0.3, 0.18), (0.12, 0.16, 0.32), (0.75, 0.68, 0.55), (0.3, 0.3, 0.3)]
        look.update(awning_tint=(*rng.choice(palette), 1.0),
                    fascia_tint=(*rng.choice([(0.12, 0.12, 0.12), (0.2, 0.25, 0.2), (0.35, 0.1, 0.08),
                                              (0.85, 0.85, 0.82)]), 1.0))
        if rng.random() > float(ornaments.get("awnings", 0.5)):
            look["awning"] = None
        look["eaves_tint"] = (*rng.choice([(1.0, 1.0, 1.0), (0.45, 0.3, 0.2), (0.3, 0.3, 0.3)]), 1.0)
    else:
        look["awning"] = None
    look["dormers"] = chance("dormers") and archetype in ("classical", "plain")
    look["chimneys"] = float(ornaments.get("chimneys", 0.0)) if archetype in ("classical", "plain") else 0.0
    if look["ground_wall"] is None:
        look["ground_wall"] = wall if archetype != "curtain" else look["wall"]
    return look


def piece_look(look, piece, group):
    """The building's look with a part's own material and colour tags applied, and plain stone for columns."""
    own = {k: v for k, v in piece.tags.items() if k.startswith(("building:material", "building:colour", "roof:"))
           and group.tags.get(k) != v}
    if _small(piece) and ("sandstone" in look["wall"] or "limestone" in look["wall"] or "ashlar" in look["wall"]):
        look = dict(look, wall=material("stone/sandstone-smooth"))
    if not own or piece.feature is None:
        return look
    rng = random.Random(zlib.crc32(piece.feature.id.encode()))
    tags = {**group.tags, **piece.tags}
    look = dict(look)
    if any(k.startswith("building:") for k in own) and "material" not in group.override:
        wall, look["tint"] = _wall(tags, look["style"], rng)
        if wall == "wall/glass-curtain":
            look.update(archetype="curtain", curtain=True, wall=material("trim/metal-dark"), tint=WHITE,
                        lod2_wall=wall, cornice="thin", reveal=0.15, sills=False, curtain_reveal=0.14, surround=0.0,
                        ground_wall=material("trim/metal-dark"))
        else:
            look.update(wall=wall, ground_wall=wall if look["ground_wall"] == look["wall"] else look["ground_wall"])
    if any(k.startswith("roof:") for k in own):
        look.update(_roof_look(tags, look["style"], rng))
    return look


def _wall(tags, style, rng, chosen=None):
    """Wall material and tint from building:material and building:colour, else the style's weighted choice."""
    rgb = colour(tags.get("building:colour"))
    name = chosen
    if not name:
        tagged = tags.get("building:material", "").lower()
        walls = style["walls"] if isinstance(style["walls"], dict) else {}
        plasters = {k: v for k, v in walls.items() if "plaster" in k}
        if tagged in ("plaster", "stucco", "render"):
            name = _nearest_plaster(rgb) if rgb else _weighted(plasters or style["walls"], rng)
        elif tagged in WALL_MATERIALS:
            name = WALL_MATERIALS[tagged]
            if tagged == "brick" and rgb and rgb[1] > 0.75 * rgb[0]:
                name = "wall/brick-yellow"
        else:
            name = _weighted(style["walls"], rng)
            if rgb and "plaster" in name:
                name = _nearest_plaster(rgb)
    name = material(name)
    strength = 0.6 if "plaster" in name or "concrete" in name or "render" in name else 0.25
    shade = rng.uniform(0.94, 1.04)
    t = tint(rgb, strength)
    return name, (t[0] * shade, t[1] * shade, t[2] * shade, 1.0)


def _roof_look(tags, style, rng):
    """Roof materials (flat, pitched and steep mansard slopes) and tint from roof:material and roof:colour, else
    the style's."""
    rgb = colour(tags.get("roof:colour"))
    tagged = ROOF_MATERIALS.get(tags.get("roof:material", "").lower())
    flat = tagged or style["roofs"].get("flat", "roof/gravel")
    pitched = tagged or (_roof_by_colour(rgb) if rgb else None) or _weighted(
        style["roofs"].get("pitched") or ["roof/tiles-red"], rng)
    steep = tagged or style["roofs"].get("steep") or "roof/zinc-standing-seam"
    return dict(roof_flat=material(flat), roof_pitched=material(pitched), roof_steep=material(steep),
                roof_tint=tint(rgb, 0.35))


def _roof_by_colour(rgb):
    """A pitched roof material whose colour is nearest to the tagged one."""
    r, g, b = rgb
    if g > r * 1.1 and g >= b * 0.9:
        return "roof/copper-green"
    if r > g * 1.3 and r > b * 1.3:
        return "roof/tiles-red"
    return "roof/slate" if max(rgb) < 130 else "roof/zinc-standing-seam"


def _nearest_plaster(rgb):
    return min(PLASTERS, key=lambda k: sum((a - b) ** 2 for a, b in zip(PLASTERS[k], rgb)))


# Massing, facades and roofs.

def floors_of(piece, look, top):
    """(y0, y1, kind) floor bands from the piece's floor level up to top."""
    start = piece.base if piece.raised else piece.ground
    room = top - start
    if room < 2.2:
        return []
    levels = number(piece.tags.get("building:levels"))
    min_level = number(piece.tags.get("building:min_level")) or 0
    ground_floor = not piece.raised
    gf = look["ground_floor_height"] if ground_floor else look["floor_height"]
    fh = look["floor_height"]
    n = int(round(levels - min_level)) if levels and levels > min_level else 0
    if not n or not 2.4 <= room / n <= 5.5:
        n = max(1, int(round((room - gf) / fh)) + 1)
    nominal = gf + (n - 1) * fh
    scale = room / nominal
    out, y = [], start
    for k in range(n):
        h = (gf if k == 0 else fh) * scale
        out.append((y, y + h, "ground" if k == 0 and ground_floor else "upper"))
        y += h
    return out


# Height of the cornice zone above the top floor, and the top moulding's profile [(out, y)] per kind.
CORNICE = dict(
    classical=[(0.0, 0.0), (0.07, 0.0), (0.07, 0.16), (0.03, 0.16), (0.03, 0.46), (0.1, 0.5), (0.18, 0.56),
               (0.55, 0.6), (0.58, 0.6), (0.58, 0.78), (0.5, 0.86), (0.0, 0.92)],
    simple=[(0.0, 0.0), (0.05, 0.0), (0.05, 0.1), (0.12, 0.18), (0.26, 0.24), (0.26, 0.38), (0.2, 0.42),
            (0.0, 0.44)],
    slab=[(0.0, 0.0), (0.14, 0.0), (0.14, 0.32), (0.0, 0.32)],
    thin=[(0.0, 0.0), (0.05, 0.0), (0.05, 0.22), (0.0, 0.22)])
LOD1_CORNICE = dict(classical=[(0.0, 0.0), (0.12, 0.0), (0.2, 0.5), (0.5, 0.6), (0.5, 0.8), (0.0, 0.92)],
                    simple=[(0.0, 0.0), (0.2, 0.2), (0.25, 0.44), (0.0, 0.44)])
STRING = dict(classical=[(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.15, 0.16), (0.15, 0.27), (0.0, 0.32)],
              plain=[(0.0, 0.0), (0.06, 0.0), (0.06, 0.18), (0.0, 0.2)])


def massing(mesh, piece, group, look, context, lod, rng, draft=False):
    """Walls, facades, mouldings, roof and bottom of one piece. Small pieces (columns, kiosks) stay plain prisms."""
    look = piece_look(look, piece, group)
    roof = piece.roof
    flat = roof.shape == "flat"
    canopy = _canopy(piece.tags) or _small(piece)
    parapet = look["parapet"] if flat and not canopy and look["parapet"] >= 0.15 else 0.0
    inset = _inset(piece, 0.25) if parapet else None
    if inset is None:
        parapet = 0.0
    elif lod >= 2:
        inset = None
    wall_top = roof.eave + parapet if flat else None
    facade = not draft and not canopy
    kind = look["cornice"] if facade and lod < 2 else None
    profile = CORNICE.get(kind, [(0, 0), (0, 0)])
    cornice = profile[-1][1] if kind else 0.0
    floors = floors_of(piece, look, roof.eave - cornice - (0.25 if kind == "classical" else 0.0)) \
        if facade and not look.get("plain") else []
    rings = [piece.ring, *piece.holes]
    plinth = None
    if look["plinth"][0] > 0 and not piece.raised and not draft:
        plinth = (piece.ground + look["plinth"][0], look["plinth"][1])
    zones = []
    if floors and floors[0][2] == "ground" and look["ground_wall"] != look["wall"] and lod < 2:
        zones.append((floors[0][1], look["ground_wall"]))
    if lod == 2 and look.get("lod2_wall"):
        look = dict(look, wall=look["lod2_wall"])
    walls, start, edges = [], 0.0, []
    for r, ring in enumerate(rings):
        normals = roofs.normals(ring)
        row = []
        for k in range(len(ring)):
            p, q = ring[k], ring[(k + 1) % len(ring)]
            hidden, blind = context.spans(piece, p, q)
            wall = Wall(p, q, start, hidden=hidden, blind=blind)
            start += wall.length
            if wall.length < 1e-3:
                row.append(None)
                continue
            row.append(wall)
            prof = [(0.0, wall_top), (1.0, wall_top)] if flat else roofs.profile(roof, p, q)
            openings = []
            if floors:
                openings = _openings(piece, group, look, context, wall, normals[k], floors, r == 0, rng)
            if r and look.get("detail") == "full":
                wall.detail = "medium"  # Courtyard walls are seen far less than street fronts.
            wall.shade = _shade(look["tint"], piece.ground, openings, floors, roof.eave, k)
            walls.append((wall, prof, openings))
        edges.append(row)
    # Every wall breaks at every wall's heights, so vertical corner edges line up.
    levels = {roof.eave, *(plinth[:1] if plinth else ()), *(z[0] for z in zones),
              *(min(y for _, y in prof) for _, prof, _ in walls)}
    if lod == 0:
        levels |= {v for _, _, openings in walls for o in openings for v in (o.v0, o.v1)}
    for wall, prof, openings in walls:
        if lod == 2 and (look["group"] == 0 or look.get("lod2_wall")):
            openings = []  # Curtain walls' own texture shows their grid from afar.
        facades.wall(mesh, wall, piece.base, prof, openings, look, lod, plinth, levels, zones)
    if lod < 2 and floors:
        _bands(mesh, piece, look, floors, roof, parapet, lod, edges, kind, profile)
        if lod == 0 and look.get("detail") != "lean":
            _corners(mesh, piece, look, floors, roof, edges, cornice)
            if look.get("pilasters"):
                _pilasters(mesh, look, floors, roof, walls, cornice)
    _roof(mesh, piece, look, roof, parapet, inset, lod, rng, draft, edges, facade)
    _bottom(mesh, piece, look)


def _inset(piece, t):
    """Outline and holes moved inwards by t, or None when that folds the outline."""
    ring = roofs.offset(piece.ring, -t)
    holes = [roofs.offset(h, -t) for h in piece.holes]
    a0, a1 = roofs.area(piece.ring), roofs.area(ring)
    if a1 <= 0 or a1 < 0.5 * a0 or abs(a0 - a1) > 2 * t * _perimeter(piece.ring) + 1:
        return None
    for h, h0 in zip(holes, piece.holes):
        if roofs.area(h) >= 0 or abs(roofs.area(h)) > abs(roofs.area(h0)):
            return None
    return ring, holes


def _perimeter(ring):
    return float(np.hypot(*(np.roll(ring, -1, axis=0) - ring).T).sum())


def _blocked(wall):
    return wall.hidden + wall.blind


def _openings(piece, group, look, context, wall, normal, floors, outer, rng):
    """Facade grammar inputs for one wall: party walls, the door facing the road, shopfronts; openings whose
    details would reach into a neighbour are left out."""
    L = wall.length
    blocked = _blocked(wall)
    ground = floors[0][0]
    # Most of the wall behind neighbours up to its eaves: a party wall without windows.
    party = sum(b - a for a, b, lo, hi in blocked if lo <= ground + 0.5 and hi >= piece.roof.eave - 1.0)
    if party > 0.6 * L:
        return []
    low = sum(b - a for a, b, lo, hi in blocked if lo <= ground + 0.5)
    door, shop = None, False
    if outer and not piece.raised and L >= 3.0 and low < 0.3 * L:
        mid = np.array(wall.p) + 0.5 * L * np.array(wall.d)
        road = context.road(mid)
        if road is not None:
            to = road - mid
            dist = float(np.hypot(*to))
            if dist < 60 and dist > 0.5 and float(to @ normal) / dist > 0.4:
                door = float((road - np.array(wall.p)) @ np.array(wall.d))
                commercial = (group.tags.get("building") in COMMERCIAL or "shop" in group.tags
                              or group.tags.get("amenity") in SHOP_AMENITIES)
                if len(context.shops):
                    commercial |= bool((np.hypot(*(context.shops - mid).T) < max(8.0, L / 2)).any())
                shop = commercial and dist < 30
                if L < 6.0 and not shop:
                    door = None
    margin = 0.15

    def free(o):
        u0, u1, v0, v1 = facades.extent(o, look)
        return not any(a < u1 + margin and u0 - margin < b and lo < v1 and v0 < hi for a, b, lo, hi in blocked)
    return facades.layout(L, floors, look, rng, door, shop, free=free)


def _shade(tint, ground, openings, floors, eave, salt=0):
    """Vertex colour: building tint with soft weathering. Darker near the ground (splash and dirt), streaks below
    window sills and under the cornice where rain runs off, and a slight per-floor variation as patched render."""
    sills = [(o.u0, o.u1, o.v0) for o in openings if o.kind in ("window", "french")]
    bands = [(y0, y1, 1.0 + 0.03 * math.sin(7.3 * k + 1.7 * salt)) for k, (y0, y1, _) in enumerate(floors)]

    def shade(u, v):
        k = 0.8 + 0.2 * min(max((v - ground) / 2.0, 0.0), 1.0)
        for u0, u1, s in sills:
            if u0 - 1e-6 <= u <= u1 + 1e-6 and s - 1.2 <= v <= s + 1e-6:
                k *= 1 - 0.1 * (v - (s - 1.2)) / 1.2
        below = eave - v
        if 0 <= below < 1.6:
            k *= 1 - 0.07 * (1 - below / 1.6)
        for y0, y1, f in bands:
            if y0 <= v < y1:
                k *= f
                break
        return (tint[0] * k, tint[1] * k, tint[2] * k, 1.0)
    return shade


def _spans(row, ya, yb):
    """Per edge of a ring, the (t0, t1) parts free of hidden or blind stretches between heights ya and yb."""
    out = []
    for wall in row:
        if wall is None:
            out.append([])
            continue
        cuts = sorted((max(a, 0.0) / wall.length, min(b, wall.length) / wall.length)
                      for a, b, lo, hi in _blocked(wall) if lo < yb and ya < hi)
        free, t = [], 0.0
        for a, b in cuts:
            if a > t + 1e-6:
                free.append((t, a))
            t = max(t, b)
        if t < 1 - 1e-6:
            free.append((t, 1.0))
        out.append([(a, b) for a, b in free if (b - a) * wall.length > 0.05])
    return out


def _bands(mesh, piece, look, floors, roof, parapet, lod, edges, kind, profile):
    """String courses, slab bands of curtain walls, sill courses and the cornice with its modillions and dentils.
    Without a parapet the top moulding rises just above the eaves, covering the seam where the roof meets the
    wall's many edges."""
    e = roof.eave
    cover = 0.0 if parapet else 0.04
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    detail = look.get("detail", "full")
    height = profile[-1][1]
    y0 = e + cover - height
    for r, ring in enumerate([piece.ring, *piece.holes]):
        row = edges[r]
        if lod == 1:
            shape = LOD1_CORNICE.get(kind, profile)
            facades.moulding(mesh, ring, e + cover - shape[-1][1], shape, trim, tc, _spans(row, y0, e + cover))
        else:
            facades.moulding(mesh, ring, y0, profile, trim, tc, _spans(row, y0, e + cover))
            if kind == "classical" and detail != "lean" and r == 0:  # Courtyards make do without.
                _modillions(mesh, ring, row, y0, look, detail == "full")
        if look["curtain"]:
            for f0, f1, _ in floors[1:]:
                facades.band(mesh, ring, f0 - 0.18, f0 + 0.18, 0.05, look["mullion"],
                             spans=_spans(row, f0 - 0.18, f0 + 0.18))
            continue
        if len(floors) >= 2 and floors[0][2] == "ground" and look["archetype"] in ("classical", "plain", "stone"):
            shape = STRING["classical" if look["archetype"] == "classical" else "plain"]
            if look["archetype"] == "stone":
                shape = [(0.0, 0.0), (0.03, 0.0), (0.03, 0.12), (0.0, 0.12)]
            ya = floors[0][1] - 0.12
            facades.moulding(mesh, ring, ya, shape, trim, tc, _spans(row, ya, ya + shape[-1][1]))
        if lod == 0 and look.get("sill_course") and len(floors) >= 3:
            y = floors[1][0] + look["sill"] - 0.1
            facades.band(mesh, ring, y, y + 0.1, 0.09, trim, spans=_spans(row, y, y + 0.1), colour=tc)


def _modillions(mesh, ring, row, y0, look, dentils):
    """Consoles under the cornice's corona, and a row of dentils below them."""
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    spans = _spans(row, y0, y0 + 0.92)
    for wall, free in zip(row, spans):
        if wall is None:
            continue
        for t0, t1 in free:
            a, b = t0 * wall.length + 0.3, t1 * wall.length - 0.3
            if b - a < 0.5:
                continue
            n = max(1, int((b - a) / 0.65))
            for u in np.linspace(a, b, n + 1):
                wall.box(mesh, trim, u - 0.07, u + 0.07, y0 + 0.44, y0 + 0.6, -0.5, -0.03, "fblr", tc)
            if dentils:
                # Only fronts and bottoms: the frieze shows between them, and their sides are 7 cm deep.
                n = max(1, int((b - a) / 0.25))
                for u in np.linspace(a, b, n + 1):
                    wall.box(mesh, trim, u - 0.06, u + 0.06, y0 + 0.34, y0 + 0.45, -0.1, -0.03, "fb", tc)


def _corners(mesh, piece, look, floors, roof, edges, cornice):
    """Quoins (alternating long and short stone blocks) or plain corner strips at right-angled outside corners."""
    if not look.get("quoins"):
        return
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    full = look.get("detail") == "full"
    y0 = floors[0][1] if floors[0][2] == "ground" and len(floors) > 1 else floors[0][0] + 0.5
    y1 = roof.eave - cornice
    p = 0.04
    for row in edges:
        n = len(row)
        for k in range(n):
            a, b = row[k - 1], row[k]  # b starts at the corner where a ends.
            if a is None or b is None:
                continue
            cross = a.d[0] * b.d[1] - a.d[1] * b.d[0]
            if abs(a.d[0] * b.d[0] + a.d[1] * b.d[1]) > 0.2 or cross < 0:
                continue  # Not a right angle, or an inside corner.
            if _blocked(a) and any(hi > y0 and x1 > a.length - 1.0 for x0, x1, lo, hi in _blocked(a)):
                continue
            if _blocked(b) and any(hi > y0 and x0 < 1.0 for x0, x1, lo, hi in _blocked(b)):
                continue
            if not full:
                a.box(mesh, trim, a.length - 0.5, a.length + p, y0, y1, -p, 0.0, "fl", tc)
                b.box(mesh, trim, 0.0, 0.5, y0, y1, -p, 0.0, "fr", tc)
                continue
            courses = max(1, int((y1 - y0) / 0.55))
            h = (y1 - y0) / courses
            for c in range(courses):
                v0, v1 = y0 + c * h + 0.03, y0 + (c + 1) * h - 0.03
                long_a = c % 2 == 0
                # The block on one wall runs past the corner to meet the other wall's block.
                la, lb = (0.75, 0.4) if long_a else (0.4, 0.75)
                a.box(mesh, trim, a.length - la, a.length + p, v0, v1, -p, 0.0, "ftbl", tc)
                b.box(mesh, trim, 0.0, lb, v0, v1, -p, 0.0, "ftbr", tc)


def _pilasters(mesh, look, floors, roof, walls, cornice):
    """Flat pilasters between the window bays of the upper floors, with a base and a capital."""
    if len(floors) < 3:
        return
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    y0 = floors[1][0] + 0.25
    y1 = roof.eave - cornice
    s = look.get("surround", 0.0) + 0.1
    for wall, _, openings in walls:
        uppers = sorted((o for o in openings if o.level == 1), key=lambda o: o.u0)
        for left, right in zip(uppers, uppers[1:]):
            a, b = left.u1 + s, right.u0 - s
            if b - a < 0.55:
                continue
            m, half = (a + b) / 2, min(0.25, (b - a) / 2 - 0.05)
            if wall.covered(m - half - 0.1, m + half + 0.1, y0, y1) or any(
                    a2 < m + half and m - half < b2 and lo < y1 and y0 < hi for a2, b2, lo, hi in _blocked(wall)):
                continue
            wall.box(mesh, trim, m - half, m + half, y0 + 0.3, y1 - 0.35, -0.06, 0.0, "flr", tc)
            wall.box(mesh, trim, m - half - 0.05, m + half + 0.05, y0, y0 + 0.3, -0.1, 0.0, "ftlr", tc)
            wall.box(mesh, trim, m - half - 0.06, m + half + 0.06, y1 - 0.35, y1, -0.12, 0.0, "fblr", tc)


def _roof(mesh, piece, look, roof, parapet, inset, lod, rng, draft, edges, facade):
    material = look["roof_flat"] if roof.shape == "flat" else look["roof_pitched"]
    overhang = 0.0
    if roof.shape not in ("flat", "dome") and lod < 2 and not draft and facade:
        overhang = 0.15 if roof.shape == "mansard" else look.get("overhang", 0.4)
    if roof.shape == "flat" and parapet and inset is not None:
        ring, holes = inset
        top = roof.eave + parapet
        for outer, inner in zip([piece.ring, *piece.holes], [ring, *holes]):
            _parapet(mesh, outer, inner, roof.eave, top, look)
        tris = roofs.surface(roof, ring, holes)
    elif roof.shape == "flat" and parapet:
        tris = roofs.surface(roofs.Roof("flat", roof.eave + parapet, 0.0), piece.ring, piece.holes)
    elif overhang:
        tris = _eaves(mesh, piece, look, roof, edges, overhang)
    else:
        tris = roofs.surface(roof, piece.ring, piece.holes)
    for t in tris:
        _roof_triangle(mesh, _slope_material(look, material, t), t, look["roof_tint"])
    if draft or _canopy(piece.tags) or _small(piece) or lod == 2:
        return
    if roof.shape != "flat" and roof.shape != "dome" and look.get("dormers"):
        _dormers(mesh, piece, look, roof, edges, lod)
    if look.get("chimneys"):
        _chimneys(mesh, piece, look, roof, rng, parapet)
    if lod == 0 and roof.shape == "flat":
        _roof_props(mesh, piece, look, roof.eave, rng, parapet)


def _slope_material(look, material, t):
    """Steep slopes (mansards) get the steep roof material."""
    if material == look["roof_flat"]:
        return material
    n = np.cross(t[1] - t[0], t[2] - t[0])
    norm = np.linalg.norm(n)
    return look["roof_steep"] if norm > 1e-9 and abs(n[1]) / norm < 0.55 else material


def _edge_offset(ring, dists):
    """Ring with each edge k moved outwards by dists[k]; corners where the moved edges meet."""
    p = np.asarray(ring, float)
    d = np.roll(p, -1, axis=0) - p
    d /= np.maximum(np.hypot(*d.T), 1e-12)[:, None]
    n = np.column_stack([-d[:, 1], d[:, 0]])
    out = []
    for k in range(len(p)):
        i = k - 1
        a, b = n[i], n[k]
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) < 1e-3:
            out.append(p[k] + b * dists[k])
            continue
        c1, c2 = a @ p[k] + dists[i], b @ p[k] + dists[k]
        x = np.array([(c1 * b[1] - a[1] * c2) / det, (a[0] * c2 - c1 * b[0]) / det])
        reach = 3 * max(dists[i], dists[k], 1e-6)
        if np.hypot(*(x - p[k])) > reach:  # Sharp corners: clamp the mitre.
            x = p[k] + (x - p[k]) * reach / np.hypot(*(x - p[k]))
        out.append(x)
    return np.array(out)


def _eaves(mesh, piece, look, roof, edges, overhang, thick=0.16):
    """Pitched roof surface carried past the walls by the overhang where the eaves are free, with a soffit under
    the overhang and a fascia board along its edge. Returns the roof surface triangles."""
    rings = [piece.ring, *piece.holes]
    outs, dists = [], []
    for r, ring in enumerate(rings):
        row = edges[r]
        ds = []
        for k, wall in enumerate(row):
            top = roof.eave + 0.5
            free = wall is not None and not any(hi > roof.eave - 0.5 and lo < top for _, _, lo, hi in _blocked(wall))
            ds.append(overhang if free else 0.0)
        dists.append(ds)
        outs.append(_edge_offset(ring, ds))
    if not any(any(ds) for ds in dists):
        return roofs.surface(roof, piece.ring, piece.holes)
    try:
        tris = roofs.surface(roof, outs[0], outs[1:])
    except Exception:
        return roofs.surface(roof, piece.ring, piece.holes)
    eaves, ec = look["eaves"], look.get("eaves_tint", WHITE)
    for ring, out, ds in zip(rings, outs, dists):
        n = len(ring)
        for k in range(n):
            j = (k + 1) % n
            if not ds[k]:
                continue
            strip = np.array([ring[k], ring[j], out[j], out[k]])
            if abs(roofs.area(strip)) < 1e-4:
                continue
            for t in roofs.surface(roof, roofs.ccw(strip)):
                a, b, c = (np.asarray(v, float) - (0, thick, 0) for v in t)
                down = np.cross(b - a, c - a)
                down = -down / max(np.linalg.norm(down), 1e-12) * (1 if down[1] >= 0 else -1)
                mesh.poly(eaves, [tuple(a), tuple(c), tuple(b)], tuple(down),
                          [(a[0], a[2]), (c[0], c[2]), (b[0], b[2])], dim(ec, 0.8))
            prof = roofs.profile(roof, out[k], out[j])
            P, Q = out[k], out[j]
            d = Q - P
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            normal = (float(-d[1] / length), 0.0, float(d[0] / length))  # Outwards, right of travel.
            for (ta, ya), (tb, yb) in zip(prof, prof[1:]):
                A, B = P + ta * d, P + tb * d
                pts = [(A[0], ya - thick, A[1]), (B[0], yb - thick, B[1]), (B[0], yb, B[1]), (A[0], ya, A[1])]
                mesh.poly(eaves, pts, normal, [(ta * length, 0), (tb * length, 0), (tb * length, thick),
                                               (ta * length, thick)], ec)
    return tris


def _roof_triangle(mesh, material, t, colour):
    """A roof triangle with UVs in metres: u along the eaves, v up the slope."""
    a, b, c = (np.asarray(v, float) for v in t)
    n = np.cross(b - a, c - a)
    if n[1] < 0:
        b, c, n = c, b, -n
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return
    n /= norm
    slope = np.array([n[0], n[2]])
    if np.hypot(*slope) < 1e-4:
        uvs = [(p[0], -p[2]) for p in (a, b, c)]
    else:
        up = -slope / np.hypot(*slope)  # Uphill in (x, z).
        along = np.array([-up[1], up[0]])
        stretch = 1 / max(n[1], 1e-3)
        uvs = [(float(p[[0, 2]] @ along), float(p[[0, 2]] @ up) * stretch) for p in (a, b, c)]
    mesh.poly(material, [tuple(a), tuple(b), tuple(c)], tuple(n), uvs, colour)


def _parapet(mesh, outer, inner, eave, top, look, lip=0.05, thick=0.08):
    """Inner face of the parapet and a coping slab on top, overhanging the wall line by lip so the seam with the
    wall's many edges stays covered."""
    rim = roofs.offset(outer, lip)
    hi = top + thick
    trim, tc = look["trim"], look.get("trim_tint", WHITE)
    for k in range(len(outer)):
        j = (k + 1) % len(outer)
        a, b, A, B, P, Q = inner[k], inner[j], outer[k], outer[j], rim[k], rim[j]
        d = b - a
        length = math.hypot(*d)
        if length < 1e-6:
            continue
        inward = (d[1] / length, 0.0, -d[0] / length)
        mesh.poly(look["wall"], [(a[0], eave, a[1]), (b[0], eave, b[1]), (b[0], top, b[1]), (a[0], top, a[1])],
                  inward, [(0, eave), (length, eave), (length, top), (0, top)], dim(look["tint"], 0.85))
        mesh.poly(trim, [(a[0], top, a[1]), (b[0], top, b[1]), (b[0], hi, b[1]), (a[0], hi, a[1])],
                  inward, [(0, top), (length, top), (length, hi), (0, hi)], dim(tc, 0.9))
        mesh.poly(trim, [(P[0], hi, P[1]), (Q[0], hi, Q[1]), (b[0], hi, b[1]), (a[0], hi, a[1])],
                  (0.0, 1.0, 0.0), [(0, 0), (length, 0), (length, 0.3), (0, 0.3)], dim(tc, 0.92))
        out = (-inward[0], 0.0, -inward[2])
        mesh.poly(trim, [(P[0], top, P[1]), (Q[0], top, Q[1]), (Q[0], hi, Q[1]), (P[0], hi, P[1])],
                  out, [(0, top), (length, top), (length, hi), (0, hi)], tc)
        mesh.poly(trim, [(A[0], top, A[1]), (B[0], top, B[1]), (Q[0], top, Q[1]), (P[0], top, P[1])],
                  (0.0, -1.0, 0.0), [(0, 0), (length, 0), (length, lip), (0, lip)], dim(tc, 0.75))


def _place(piece, rng, sx, sz, margin, tries=40, placed=(), axis=None):
    """A random rectangle centre (cx, cz) inside the outline, margin from its edges and clear of placed ones;
    sides along axis (the footprint's long side). None when nothing fits."""
    rings = [piece.ring, *piece.holes]
    lo, hi = piece.ring.min(axis=0), piece.ring.max(axis=0)
    a = np.asarray(axis if axis is not None else (1.0, 0.0), float)
    b = np.array([-a[1], a[0]])
    for _ in range(tries):
        c = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])])
        corners = np.array([c + a * su * sx / 2 + b * sv * sz / 2 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        if not roofs.inside(rings, corners[:, 0], corners[:, 1]).all():
            continue
        if roofs.distance(rings, corners[:, 0], corners[:, 1]).min() < margin:
            continue
        if any(np.hypot(*(c - q[:2])) < (max(sx, sz) + q[2]) / 2 + 0.6 for q in placed):
            continue
        return c, corners
    return None, None


def _block(mesh, material, corners, y0, y1, colour=WHITE, top=True):
    """A prism over a counter-clockwise quad of corners [(x, z)]: four sides and the top."""
    ring = roofs.ccw(np.asarray(corners, float))
    start = 0.0
    for k in range(len(ring)):
        w = Wall(ring[k], ring[(k + 1) % len(ring)], start, lambda u, v: colour)
        start += w.length
        w.face(mesh, material, 0.0, w.length, y0, y1, colour=dim(colour, 0.92 if k % 2 else 1.0))
    if top:
        mesh.poly(material, [(x, y1, z) for x, z in ring], (0.0, 1.0, 0.0), [(x, -z) for x, z in ring],
                  dim(colour, TONE["t"]))


def _roof_props(mesh, piece, look, y, rng, parapet):
    """Flat roof clutter: air conditioning units with fan grilles, vent pipes, skylights, and a stair hut with a
    door on large roofs."""
    area = abs(roofs.area(piece.ring))
    _, axis, _, _ = roofs.box(piece.ring)
    wanted = min(8, int(area / 100))
    placed = []
    metal = look["mullion"]
    grey = (0.78, 0.8, 0.82, 1.0)
    if area > 300:
        c, corners = _place(piece, rng, 3.2, 2.6, 1.0, placed=placed, axis=axis)
        if c is not None:
            placed.append((*c, 3.2))
            _block(mesh, look["wall"], corners, y, y + 2.7, look["tint"])
            ring = roofs.ccw(corners)
            w = Wall(ring[0], ring[1])
            w.face(mesh, look["door"], w.length / 2 - 0.45, w.length / 2 + 0.45, y + 0.05, y + 2.1, -0.01,
                   shade=False)
            _block(mesh, look["trim"], roofs.offset(ring, 0.08), y + 2.7, y + 2.85, look.get("trim_tint", WHITE))
    for _ in range(wanted):
        kind = rng.random()
        if kind < 0.55:
            sx, sz, h = rng.uniform(1.0, 2.2), rng.uniform(0.8, 1.3), rng.uniform(0.8, 1.4)
            c, corners = _place(piece, rng, sx, sz, 1.0, placed=placed, axis=axis)
            if c is None:
                continue
            placed.append((*c, max(sx, sz)))
            _block(mesh, metal, roofs.offset(roofs.ccw(corners), -0.05), y, y + 0.15, (0.5, 0.5, 0.5, 1.0))
            _block(mesh, look["spandrel"], corners, y + 0.15, y + 0.15 + h, grey)
            # Two dark fan grilles on top.
            ring = roofs.ccw(corners)
            for f in (0.27, 0.73):
                m = ring[0] + (ring[1] - ring[0]) * f + (ring[3] - ring[0]) * 0.5
                r = min(sx, sz) * 0.32
                pts = [(m[0] + r * math.cos(t), y + 0.16 + h, m[1] + r * math.sin(t)) for t in
                       np.linspace(0, 2 * math.pi, 9)[:-1]]
                mesh.poly(metal, pts, (0.0, 1.0, 0.0), [(p[0], -p[2]) for p in pts], (0.25, 0.25, 0.25, 1.0))
        elif kind < 0.8:
            c, corners = _place(piece, rng, 0.25, 0.25, 0.8, placed=placed, axis=axis)
            if c is None:
                continue
            placed.append((*c, 0.4))
            h = rng.uniform(0.6, 1.4)
            _block(mesh, metal, corners, y, y + h, (0.6, 0.6, 0.6, 1.0))
            _block(mesh, metal, roofs.offset(roofs.ccw(corners), 0.06), y + h, y + h + 0.06, (0.5, 0.5, 0.5, 1.0))
        else:
            sx, sz = rng.uniform(1.2, 2.5), rng.uniform(1.0, 1.6)
            c, corners = _place(piece, rng, sx, sz, 1.2, placed=placed, axis=axis)
            if c is None:
                continue
            placed.append((*c, max(sx, sz)))
            _block(mesh, metal, corners, y, y + 0.35, (0.7, 0.7, 0.7, 1.0), top=False)
            ring = roofs.ccw(corners)
            mesh.poly(look["glass"], [(x, y + 0.35, z) for x, z in ring], (0.0, 1.0, 0.0),
                      [(x, -z) for x, z in ring])


def _chimneys(mesh, piece, look, roof, rng, parapet):
    """Brick or rendered chimney stacks with a cap, standing on the roof, higher than the ridge near them."""
    area = abs(roofs.area(piece.ring))
    if area < 40:
        return
    n = max(1, min(8, int(area / 110 * look["chimneys"] + rng.random())))
    _, axis, _, _ = roofs.box(piece.ring)
    brick = rng.random() < 0.6
    material = look["wall"] if not brick else "wall/brick-red"
    colour = look["tint"] if not brick else WHITE
    placed = []
    for _ in range(n):
        sx, sz = rng.uniform(0.5, 0.7), rng.uniform(0.7, 1.4)
        c, corners = _place(piece, rng, sx, sz, 1.0, placed=placed, axis=axis)
        if c is None:
            continue
        ys = roof.y(corners[:, 0], corners[:, 1])
        if roof.shape != "flat" and float(roof.y(*c)) < roof.eave + 0.45 * roof.rise:
            continue  # Chimneys rise near the ridge, not from the eaves.
        placed.append((*c, max(sx, sz)))
        y0 = float(ys.min()) - 0.3
        y1 = float(ys.max()) + (parapet if roof.shape == "flat" else 0.0) + rng.uniform(0.8, 1.5)
        _block(mesh, material, corners, y0, y1, colour, top=False)
        cap = roofs.offset(roofs.ccw(corners), 0.06)
        _block(mesh, look["trim"], cap, y1, y1 + 0.1, look.get("trim_tint", WHITE))
        for f in (0.3, 0.7):
            m = corners[0] + (corners[1] - corners[0]) * 0.5 + (corners[3] - corners[0]) * f
            pot = np.array([m + (dx, dz) for dx, dz in ((-0.09, -0.09), (0.09, -0.09), (0.09, 0.09), (-0.09, 0.09))])
            _block(mesh, "roof/tiles-red", pot, y1 + 0.1, y1 + 0.35, (0.8, 0.6, 0.5, 1.0))


def _dormers(mesh, piece, look, roof, edges, lod):
    """Gabled dormers on the long eaves of pitched roofs, each a small framed window in its cheeks and roof."""
    ring = piece.ring
    n = len(ring)
    steep = roof.shape == "mansard"
    for k, wall in enumerate(edges[0]):
        if wall is None or wall.length < 6.0 or _blocked(wall):
            continue
        prof = roofs.profile(roof, ring[k], ring[(k + 1) % n])
        if max(y for _, y in prof) - min(y for _, y in prof) > 0.05:
            continue  # Not an eaves wall.
        base = roof.eave

        def height(u, x):
            X, _, Z = wall.at(u, 0.0, x)
            return float(roof.y(X, Z))
        mid = wall.length / 2
        g = (height(mid, 0.6) - base) / 0.6
        if g < 0.35:
            continue
        sb = min(0.9, 0.45 / g) if not steep else 0.25
        vb = height(mid, sb)
        width, front = 1.5, 1.6
        vt = vb + front
        hr = 0.55
        need = vt + hr + 0.1
        depth = None
        for x in np.arange(sb + 0.5, 8.0, 0.25):
            if height(mid, x) >= need:
                depth = x - sb + 0.2
                break
        if depth is None or need > roof.ridge - 0.2:
            continue
        for u0, u1 in facades.bays(wall.length, max(look["bay"] * 2, 3.5), 1.6):
            c = (u0 + u1) / 2
            # The whole dormer must stand on one roof plane, clear of hips.
            ok = True
            for du in (-width / 2 - 0.3, width / 2 + 0.3):
                for x in (sb, sb + depth):
                    X, _, Z = wall.at(c + du, 0.0, x)
                    if abs(height(c + du, x) - height(c, x)) > 0.02:
                        ok = False
            if ok:
                _dormer(mesh, wall, c, sb, vb, vt, width, hr, depth, look, lod, steep)


def _dormer(mesh, wall, c, sb, vb, vt, width, hr, depth, look, lod, steep):
    cheek = look["roof_steep"] if steep else look["wall"]
    cc = look["roof_tint"] if steep else look["tint"]
    dw = Wall(wall.at(0.0, 0.0, sb)[::2], wall.at(wall.length, 0.0, sb)[::2], wall.start, lambda u, v: cc)
    a, b = c - width / 2, c + width / 2
    o = facades.Opening("window", c - 0.5, c + 0.5, vb + 0.3, vt - 0.2, cols=2, level=99)
    if lod == 0:
        us, vs = [a, o.u0, o.u1, b], [vb - 0.05, o.v0, o.v1, vt]
        for v0, v1 in zip(vs, vs[1:]):
            for u0, u1 in zip(us, us[1:]):
                if not (u0 == o.u0 and v0 == o.v0):
                    dw.face(mesh, cheek, u0, u1, v0, v1)
        dw.hole(mesh, cheek, o.u0, o.u1, o.v0, o.v1, 0.12)
        facades._glazing(mesh, dw, o, 0.12, look, o.v1, False)
    else:
        dw.face(mesh, cheek, a, b, vb - 0.05, vt)
        dw.face(mesh, look["glass"], o.u0, o.u1, o.v0, o.v1, -0.02, shade=False)
    dw.side(mesh, cheek, a, vb - 0.05, vt, 0.0, depth, -1, colour=dim(cc, 0.9))
    dw.side(mesh, cheek, b, vb - 0.05, vt, 0.0, depth, 1, colour=dim(cc, 0.9))
    gable = [dw.at(a, vt), dw.at(b, vt), dw.at(c, vt + hr - 0.03)]
    mesh.poly(cheek, gable, (dw.n[0], 0.0, dw.n[1]), [(a, vt), (b, vt), (c, vt + hr)], cc)
    roof_material = look["roof_steep"] if steep else look["roof_pitched"]
    for edge in (a - 0.1, b + 0.1):
        pts = [dw.at(edge, vt - 0.06, -0.15), dw.at(c, vt + hr, -0.15), dw.at(c, vt + hr, depth),
               dw.at(edge, vt - 0.06, depth)]
        p = np.array(pts)
        normal = np.cross(p[1] - p[0], p[3] - p[0])
        normal = normal / max(np.linalg.norm(normal), 1e-9) * (1 if normal[1] >= 0 else -1)
        slope = math.hypot(c - edge, hr + 0.06)
        mesh.poly(roof_material, pts, tuple(normal), [(0, 0), (slope, 0), (slope, depth + 0.15), (0, depth + 0.15)],
                  look["roof_tint"])
        mesh.poly(roof_material, pts[::-1], tuple(-normal), [(0, depth + 0.15), (slope, depth + 0.15), (slope, 0),
                                                             (0, 0)], dim(look["roof_tint"], 0.6))


def _bottom(mesh, piece, look):
    """Underside, so every piece is a closed shell (and raised parts show a soffit). Its outline carries the
    points where the roof creases meet the walls, as the walls' own edges do."""
    rings = [_creased(ring, piece.roof) for ring in [piece.ring, *piece.holes]]
    points, triangles = roofs.triangulate(rings[0], rings[1:])
    y = piece.base
    material = look["wall"] if piece.raised else look["plinth"][1] if look["plinth"][0] > 0 else look["wall"]
    colour = look["tint"] if piece.raised else (1.0, 1.0, 1.0, 1.0)
    for t in triangles:
        a, b, c = points[t]
        mesh.poly(material, [(a[0], y, a[1]), (c[0], y, c[1]), (b[0], y, b[1])], (0.0, -1.0, 0.0),
                  [(a[0], a[1]), (c[0], c[1]), (b[0], b[1])], colour)


def _creased(ring, roof):
    out = []
    for p, q in zip(ring, np.roll(ring, -1, axis=0)):
        out += [p + t * (q - p) for t, _ in roofs.profile(roof, p, q)[:-1]]
    return np.array(out)


# Colliders.

def colliders_for(piece, prism=False):
    """Convex hulls per convex piece of the outline, from the wall bottom up to the roof surface."""
    points, triangles = roofs.triangulate(piece.ring, piece.holes)
    roof = piece.roof if not prism else roofs.Roof("flat", piece.roof.ridge, 0.0)
    surface = np.zeros((0, 3))
    if roof.shape != "flat":
        surface = roofs.surface(roof, piece.ring, piece.holes).reshape(-1, 3)
    out = []
    for poly in roofs.pieces(points, triangles):
        p = points[poly]
        if abs(roofs.area(p)) < 0.05:
            continue
        verts = [(x, piece.base, z) for x, z in p]
        if roof.shape in ("flat", "dome"):  # Domes rise from their outline at the eaves.
            verts += [(x, roof.eave, z) for x, z in p]
        else:
            verts += [(x, float(roof.y(x, z)), z) for x, z in p]
        if len(surface):
            inner = surface[_in_convex(p, surface[:, [0, 2]])]
            verts += [tuple(v) for v in inner]
        out.append(Collider("convex", np.unique(np.round(np.array(verts, np.float32), 3), axis=0)))
    return out


def _in_convex(poly, points, tolerance=1e-6):
    p = np.asarray(poly, float)
    ok = np.ones(len(points), bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        ok &= -((b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])) >= -tolerance
    return ok


# Landmarks.

def landmark(group, context, config, centre, extras, seed):
    """Landmarks have their own module; it imports this one, so the import waits until a landmark is built."""
    from . import landmarks

    return landmarks.build(group, context, config, centre, extras, seed)


def _tier(mesh, piece, look):
    roof = piece.roof
    start = 0.0
    for ring in [piece.ring, *piece.holes]:
        for k in range(len(ring)):
            p, q = ring[k], ring[(k + 1) % len(ring)]
            wall = Wall(p, q, start)
            start += wall.length
            if wall.length > 1e-3:
                facades.wall(mesh, wall, piece.base, roofs.profile(roof, p, q), [], look, 2)
    for t in roofs.surface(roof, piece.ring, piece.holes):
        _roof_triangle(mesh, look["roof_flat"], t, look["roof_tint"])
    _bottom(mesh, piece, look)


def tiers(piece, reference, trust):
    """A piece no single roof fits, as flat-topped height tiers each standing on the one below."""
    found = clustered(piece, reference, trust)
    if len(found) <= 1:
        return [piece]
    tags = {k: v for k, v in piece.tags.items() if not k.startswith(TIER_TAGS)}
    out = []
    for k, (ring, holes, base, top, _) in enumerate(found):
        out.append(Piece(piece.feature, tags, ring, holes, piece.base if k == 0 else base, piece.ground,
                         roofs.Roof("flat", top, 0.0), raised=k > 0))
    out[0].cells = piece.cells
    return out


def clustered(piece, reference, trust, most=TIERS):
    """Tiers from clustering measured heights inside the outline: [(ring, holes, base, top, None)], nested so each
    tier stands on the one below."""
    x, z, y = cells(reference.top, [piece.ring, *piece.holes], 0.0)
    row, col = reference.top.index(x, z)
    ok = np.isfinite(y) & trust[row, col] & (y > piece.ground + 1.0)
    area = abs(roofs.area(piece.ring))
    cell = reference.top.cell
    smallest = max(10.0, 0.03 * area)
    if ok.sum() * cell ** 2 < smallest:
        return [(piece.ring, piece.holes, piece.base, piece.roof.ridge, None)]
    levels = _levels(y[ok], most, int(smallest / cell ** 2))
    out, below = [], piece.base
    for k, level in enumerate(levels):
        if k == 0:
            rings = [(piece.ring, piece.holes)]
        else:
            mask = ok & (y >= (levels[k - 1] + level) / 2)
            rings = [(_tidy(r), [_tidy(h) for h in hs]) for r, hs in trace(x[mask], z[mask], cell)
                     if abs(roofs.area(r)) >= smallest]
        for ring, holes in rings:
            if len(ring) >= 3 and roofs.area(ring) > 0:
                out.append((ring, [h for h in holes if len(h) >= 3 and abs(roofs.area(h)) >= 4], below, level, None))
        below = level
    return out


def _tidy(ring):
    return roofs.clean(roofs.square(roofs.clean(roofs.simplify(ring, 0.5))))


def _levels(values, most, smallest):
    """Up to most height levels (1D k-means), adding a level only while it cuts the error by over 0.5 m and every
    level keeps at least smallest values."""
    values = np.sort(values)
    best, last = [float(np.median(values))], math.sqrt(np.mean((values - np.median(values)) ** 2))
    for k in range(2, most + 1):
        centres = np.quantile(values, (np.arange(k) + 0.5) / k)
        for _ in range(20):
            label = np.argmin(np.abs(values[:, None] - centres[None]), axis=1)
            centres = np.array([values[label == j].mean() if (label == j).any() else centres[j] for j in range(k)])
        label = np.argmin(np.abs(values[:, None] - centres[None]), axis=1)
        rms = math.sqrt(np.mean((values - centres[label]) ** 2))
        if last - rms < 0.5 or min(np.bincount(label, minlength=k)) < smallest:
            break
        best, last = sorted(float(c) for c in centres), rms
    return best


def _open(grid, r=2):
    """Morphological opening with a (2r + 1)² square: drops specks and thin slivers."""
    def spread(mask, need):
        s = np.pad(mask, r + 1).astype(np.int32).cumsum(0).cumsum(1)
        k = 2 * r + 1
        window = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
        return window[:mask.shape[0], :mask.shape[1]] >= need
    eroded = spread(grid, (2 * r + 1) ** 2)
    return spread(eroded, 1) & grid


def trace(x, z, cell):
    """Outlines of a set of raster cells: [(ring, holes)] with rings counter-clockwise seen from above."""
    if not len(x):
        return []
    c = np.round((x - x.min()) / cell).astype(int)
    r = np.round((z - z.min()) / cell).astype(int)
    grid = np.zeros((r.max() + 3, c.max() + 3), bool)
    grid[r + 1, c + 1] = True
    grid = _open(grid)
    x0, z0 = x.min() - 1.5 * cell, z.min() - 1.5 * cell
    # Directed cell edges with the filled cell on the left of travel in (col, row) space.
    edges = {}
    rows, cols = np.nonzero(grid)
    for i, j in zip(rows, cols):
        if not grid[i - 1, j]:
            edges.setdefault((j + 1, i), []).append((j, i))
        if not grid[i + 1, j]:
            edges.setdefault((j, i + 1), []).append((j + 1, i + 1))
        if not grid[i, j - 1]:
            edges.setdefault((j, i), []).append((j, i + 1))
        if not grid[i, j + 1]:
            edges.setdefault((j + 1, i + 1), []).append((j + 1, i))
    loops = []
    while edges:
        start = next(iter(edges))
        loop, at = [start], start
        while True:
            nxt = edges[at].pop()
            if not edges[at]:
                del edges[at]
            if nxt == start:
                break
            loop.append(nxt)
            at = nxt
            if at not in edges:
                break
        ring = np.array(loop, float) * cell + (x0, z0)
        loops.append(ring)
    outers = [l for l in loops if roofs.area(l) > 0]
    holes = [l for l in loops if roofs.area(l) < 0]
    out = [(o, []) for o in outers]
    for h in holes:
        for o, hs in out:
            if roofs.inside([o], *h[0]):
                hs.append(h)
                break
    return out
