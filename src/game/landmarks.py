"""Landmarks: gates, monuments, memorial columns, towers and churches, rebuilt with the classical kit.

buildings.generate sends every landmark group here with the same group, context and extras it uses for buildings.

A landmark is read from its OSM parts, measured against the reference mesh. Every building:part becomes an Item
with a role, decided from its tags, size, shape and what it stands on:

    steps        building:part=foundation, or step_count: a stepped platform
    rod          a very thin part (cannon barrels, poles): a turned metal rod
    column       a small, compact part, tall for its width: base, fluted shaft with entasis, and capital
    drum         a round part, styled by its place in a stack of drums on one axis: fluted shaft drums, garland
                 rings, the capital at the top of a column, plain drums with mouldings
    entablature  a raised part resting on columns: architrave, frieze (triglyphs over the columns for Doric) and
                 cornice; a thin, slightly larger slab on top of it is taken as its cornice
    temple       a raised pitched roof resting on columns: entablature, pediments and a metal roof
    attic        a raised part above an entablature: base, framed (sculpted) panels and a cap
    wall         a thin slab standing on the ground (a gate's passage walls): plinth, antae, relief panel, impost
    block        anything else: base course, panelled die and cornice

A landmark without parts is split into tiers measured from the mesh, each a block, with its fitted roof on top.
Where the mesh rises clearly above the topmost part (a quadriga, a statue), that skin is remeshed into a smooth
closed sculpture (sculpt.py): shape only, never the mesh's pixels.

NAMED adds what OSM can't say about a few famous landmarks, keyed by wikidata id: the order of their columns and the
materials of their stone, sculpture and reliefs. Everything else comes from the data.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import classical, roofs, sculpt
from .buildings import LANDMARK_MATERIAL, LANDMARK_RMS, Piece, _tier, clustered, colliders_for, colour, number
from .classical import Frame, Geo
from .facades import Mesh
from .types import Collider, Model

STONE = "stone/sandstone-smooth"
# Library materials and what to use while the library doesn't have them yet.
FALLBACK = {"stone/sandstone-smooth": "stone/sandstone-carved", "stone/sandstone-carved": "trim/stone",
            "stone/granite-dark": "trim/stone", "stone/marble-white": "wall/limestone",
            "metal/bronze-triplanar": "metal/bronze", "metal/gold-triplanar": "metal/gold",
            "metal/copper-patina": "metal/bronze-triplanar", "metal/bronze": "trim/metal-dark",
            "metal/gold": "metal/bronze", "roof/copper-green": "roof/zinc"}
MATERIALS = {"sandstone": "stone/sandstone-smooth", "granite": "stone/granite-dark", "marble": "stone/marble-white",
             "limestone": "wall/limestone", "stone": "stone/sandstone-smooth", "bronze": "metal/bronze",
             "copper": "roof/copper-green", "gold": "metal/gold"}
CARVED = {"stone/sandstone-smooth": "stone/sandstone-carved"}
# Famous landmarks, keyed by wikidata: what the parts' tags can't tell.
NAMED = {
    # Brandenburger Tor: Doric with Attic bases (Langhans, 1791); the Quadriga is hammered copper.
    "Q82425": dict(order="doric", sculpture="metal/copper-patina"),
    # Siegessäule: polished granite base, hall and steps with bronze reliefs; a sandstone column with gilded cannon
    # barrels and garlands; the gilded Victoria on top.
    "Q154987": dict(order="doric", flutes=0, lower="stone/granite-dark", lower_top=18.05, reliefs="metal/bronze",
                    rings="metal/gold", sculpture="metal/gold-triplanar", capital="corinthian"),
}
SCULPTURE = "metal/bronze-triplanar"
CLASSICAL = {"classicism", "neoclassicism", "baroque", "renaissance", "historicism", "neo-renaissance",
             "neo-baroque", "palladian", "georgian", "empire"}
SCULPTED = {"monument", "memorial", "gate", "city_gate", "triumphal_arch", "column", "obelisk"}
WORSHIP = {"church", "cathedral", "chapel", "mosque", "synagogue", "temple", "shrine", "basilica"}
TOWERS = {"tower", "bell_tower", "minaret", "lighthouse", "campanile"}
MIN_SCULPTURE = 1.5  # Metres the mesh must rise above the topmost part to hold a sculpture.
SCULPTURE_BUDGET = 20000


@dataclass(eq=False)
class Item:
    """One part of a landmark with its role and the heights it spans."""

    piece: Piece
    ring: np.ndarray
    y0: float
    y1: float
    role: str = "block"
    round: bool = False
    centre: tuple = (0.0, 0.0)
    radius: float = 0.0
    width: float = 0.0  # Short side of the oriented box.
    length: float = 0.0
    axis: tuple = (1.0, 0.0)
    material: str = STONE
    data: dict = field(default_factory=dict)

    @property
    def tags(self):
        return self.piece.tags

    @property
    def height(self):
        return self.y1 - self.y0


def build(group, context, config, centre, extras, seed):
    if group.mode == "prism":
        return _prism(group, context, centre, extras)
    named = NAMED.get(group.tags.get("wikidata"), {})
    stone = material(group.override.get("material") or _stone(group.tags, STONE))
    items, ground = plan(group, context, named, stone)
    sculpture = _sculpture(group, items, context, named)
    sizes = {}
    lods = []
    details = (1, 2, 2) if context.quality == "draft" else (0, 1, 2)
    for lod, detail in enumerate(details):
        geo = Geo(ground=ground, sizes=sizes)
        _sizes(sizes, items, named)
        for k, it in enumerate(items):
            # Seeded per item, so reliefs look the same at every LOD and in every build.
            draw(geo, it, detail, np.random.default_rng(seed + 17 * k), items)
        if sculpture:
            p, n, uv, t = sculpture["lods"][lod]
            geo.add(sculpture["material"], p, n, uv, t)
        lods.append(geo.parts())
    colliders = [c for it in items for c in _colliders(it)]
    if sculpture:
        colliders.append(_hull(sculpture["lods"][-1][0]))
    roles = {}
    for it in items:
        roles[it.role] = roles.get(it.role, 0) + 1
    extras.update(kind="landmark", style="classical", material=stone, tiers=len(items), roles=roles,
                  triangles=[int(sum(len(p.indices) // 3 for p in parts)) for parts in lods])
    if named:
        extras["named"] = group.tags.get("wikidata")
    if sculpture:
        extras["sculpture"] = dict(material=sculpture["material"], voxel=round(sculpture["voxel"], 3),
                                   source=sculpture["source"])
    return Model("landmark/" + group.id.replace("/", "-"), centre, lods, colliders, extras)


def wants(group):
    """Whether a group headed for the building path is better built here: colonnades (four or more column-like
    parts), which the facade grammar can't describe. For buildings.decide, after fitting."""
    columns = 0
    for piece in group.pieces:
        if piece.ring is None or piece.feature.id == group.id:
            continue
        _, _, hu, hv = roofs.box(piece.ring)
        high, low = number(piece.tags.get("height")), number(piece.tags.get("min_height")) or 0.0
        tall = high - low if high is not None else (piece.roof.eave - piece.base if piece.roof else 0.0)
        if 2 * hu <= 2.5 and hu <= 1.35 * hv and tall >= max(2.0, 6 * hu):
            columns += 1
    return columns >= 4


def material(name):
    """name if the library has it, else the nearest material it does have."""
    from . import library

    seen = set()
    while name not in library.MATERIALS and name not in seen:
        seen.add(name)
        name = FALLBACK.get(name, LANDMARK_MATERIAL)
    return name


def _size(name):
    from . import library

    spec = library.MATERIALS.get(name, {})
    for _ in range(4):
        if "alias" in spec:
            spec = library.MATERIALS.get(spec["alias"], {})
        elif "like" in spec:
            spec = library.MATERIALS.get(spec["like"], {})
        else:
            break
    return spec.get("size", 1.0) or 1.0


def _sizes(sizes, items, named):
    for it in items:
        for name in (it.material, _carved(it.material), *it.data.get("materials", ())):
            if name not in sizes:
                sizes[name] = _size(name)


def _carved(name):
    return material(CARVED.get(name, name))


def _stone(tags, default):
    """Stone or metal from building:material or building:colour."""
    rgb = colour(tags.get("building:colour"))
    if rgb and rgb[0] > 200 and rgb[1] > 160 and rgb[2] < 110:
        return "metal/gold"
    return MATERIALS.get(tags.get("building:material"), default)


# Planning: items and their roles.

def plan(group, context, named, stone):
    """Items for the group's pieces, with roles, heights from one ground level, and materials."""
    grounds = [p.ground for p in group.pieces if not p.raised] or [p.ground for p in group.pieces]
    ground = float(np.median(grounds))
    pieces = group.pieces
    tagged = any(number(p.tags.get("height")) is not None for p in pieces)
    poor = len(pieces) == 1 and not tagged and group.rms is not None and group.rms > LANDMARK_RMS
    items = []
    if poor:
        # One outline the roof fit can't describe: height tiers measured from the mesh.
        for ring, holes, base, top, _ in clustered(pieces[0], context.reference, context.trust):
            piece = Piece(pieces[0].feature, pieces[0].tags, ring, holes, base, ground, roofs.Roof("flat", top, 0.0))
            items.append(_item(piece, ring, base, top))
    else:
        for piece in pieces:
            y0, y1 = _span(piece, ground)
            items.append(_item(piece, piece.ring, y0, y1))
    for it in items:
        it.material = material(group.override.get("material") or _stone(it.tags, stone))
        if named.get("lower") and it.y1 - ground <= named["lower_top"] and it.material == stone:
            it.material = material(named["lower"])
    _roles(items, ground, group, named)
    return items, ground


def _span(piece, ground):
    """Bottom and top of a piece from its tags over the landmark's ground, else from the fit."""
    low, high = number(piece.tags.get("min_height")), number(piece.tags.get("height"))
    if low is None and number(piece.tags.get("building:min_level")):
        low = number(piece.tags.get("building:min_level")) * 3.0
    y0 = ground + low if low and low > 0.3 else min(piece.base, ground - 0.3)
    if high is not None and high > (low or 0) + 0.05:
        y1 = ground + high
    else:
        y1 = piece.roof.eave if piece.roof.shape not in ("flat", "dome") else piece.roof.ridge
    return y0, max(y1, y0 + 0.1)


def _item(piece, ring, y0, y1):
    ring = roofs.ccw(np.asarray(ring, float))
    rise = 0.0
    if piece.roof is not None and piece.roof.shape == "dome":
        # The dome is drawn above the drum, so the drum stops at the eaves.
        rise = min(piece.roof.rise, max(0.0, y1 - y0 - 0.5))
        y1 -= rise
    c, axis, hu, hv = roofs.box(ring)
    it = Item(piece, ring, float(y0), float(y1), centre=(float(c[0]), float(c[1])), width=2 * hv, length=2 * hu,
              axis=(float(axis[0]), float(axis[1])))
    if rise > 0.3:
        it.data["dome"] = rise
    centroid = roofs.centroid(ring)
    r = np.hypot(*(ring - centroid).T)
    if len(ring) >= 8 and r.std() < 0.06 * r.mean() and hu < 1.15 * hv and abs(r.mean() - hu) < 0.12 * hu:
        it.round, it.centre, it.radius = True, (float(centroid[0]), float(centroid[1])), float(r.mean())
    return it


def _roles(items, ground, group, named):
    for it in items:
        raised = it.y0 > ground + 0.5
        tags = it.tags
        if tags.get("building:part") == "foundation" or "step_count" in tags:
            it.role = "steps"
        elif it.length < 0.45 and it.height > 4 * it.length:
            it.role = "rod"
        elif it.length <= 2.5 and it.length <= 1.35 * it.width and it.height >= max(2.0, 3 * it.length):
            it.role = "column"
        elif it.round:
            it.role = "drum"
        elif not raised and it.width <= 3.2 and it.length >= 1.8 * it.width and it.height >= 2.0:
            it.role = "wall"
        else:
            it.role = "block"
        it.data["sculpted"] = bool({group.tags.get("historic"), group.tags.get("building")} & SCULPTED)
        kinds = {group.tags.get(k) for k in ("building", "amenity", "historic", "man_made", "tower:type")}
        if kinds & WORSHIP or "place_of_worship" in kinds:
            it.data["windows"] = "arched"
        elif kinds & TOWERS or "tower:type" in group.tags:
            it.data["windows"] = "belfry"
        elif not it.data["sculpted"]:
            it.data["windows"] = "rect"
        it.data["order"] = named.get("order") or _order(group.tags)
        if "flutes" in named:
            it.data["flutes"] = named["flutes"]
        if named.get("reliefs"):
            it.data["reliefs"] = material(named["reliefs"])
    columns = [it for it in items if it.role == "column"]
    for it in columns:
        it.data["diameter"] = min(it.width, it.length)
        it.data["angle"] = _frame_angle(items)
    # Parts resting on columns are entablatures, or temple roofs when pitched.
    for e in items:
        if e.role not in ("block", "drum") or e.y0 < ground + 1.0:
            continue
        grown = classical.offset_path(e.ring, 1.0)
        held = [c for c in columns if abs(c.y1 - e.y0) <= 0.6 and roofs.inside([grown], *c.centre)]
        if len(held) < 2:
            continue
        under = [w for w in items if w.role in ("wall", "block") and w is not e and abs(w.y1 - e.y0) <= 0.6
                 and roofs.inside([e.ring], *roofs.centroid(w.ring))]
        e.data.update(columns=held, under=under)
        shape = roofs.ALIASES.get(e.tags.get("roof:shape"), e.tags.get("roof:shape"))
        # A raised roof with height on columns, its shape left out, is a temple's pitched roof.
        pitched = shape is None and ((number(e.tags.get("roof:height")) or 0) >= 0.8
                                     or e.piece.roof.shape in ("gabled", "hipped"))
        if shape in ("gabled", "hipped") or pitched:
            e.role = "temple"
            _temple_plan(e, ground)
        else:
            e.role = "entablature"
    # A thin, slightly larger slab on an entablature is its cornice.
    for e in [it for it in items if it.role == "entablature" and not it.round]:
        for s in items:
            if s.role != "block" or abs(s.y0 - e.y1) > 0.3 or s.height > 1.2:
                continue
            if roofs.inside([s.ring], *roofs.centroid(e.ring)) and s.length >= e.length and s.width >= e.width:
                e.data["cornice"] = s.ring
                e.y1 = s.y1
                s.role = "merged"
    for e in [it for it in items if it.role == "entablature" and not it.round]:
        _entablature_plan(e)
    # Blocks above an entablature are attics, kept within its architrave.
    for e in [it for it in items if it.role == "entablature" and not it.round]:
        for a in items:
            if a.role == "block" and a.y0 >= e.y1 - 0.3 and roofs.inside([e.ring], *roofs.centroid(a.ring)):
                a.role = "attic"
                a.ring = _clip(a.ring, e.data["rect"], 0.12)
    _stacks(items, columns, ground, named)
    _towers(items, ground)
    gate = {group.tags.get("building"), group.tags.get("historic")} & {"gate", "city_gate", "triumphal_arch"}
    if gate and not columns:
        # A gate mapped as plain blocks: a round-headed passage through each block standing on the ground.
        for it in items:
            if it.role == "block" and it.y0 <= ground + 0.3 and it.length >= 4.0 and it.height >= 4.0:
                it.role = "arch"


def _towers(items, ground):
    """Tall, slender blocks standing above the rest are towers: belfry openings near their top."""
    blocks = [it for it in items if it.role == "block"]
    if len(blocks) < 2:
        return
    lowest = min(it.y1 for it in blocks)
    for it in blocks:
        slender = max(it.length, it.width) <= 14 and it.y1 - ground >= 1.6 * max(it.length, it.width)
        if slender and it.y1 > lowest + 6 and it.data.get("windows"):
            it.data["windows"] = "belfry"

def _order(tags):
    if tags.get("building") in ("gate", "gatehouse") or tags.get("historic") in ("monument", "gate", "city_gate"):
        return "doric"
    if tags.get("building:architecture") in ("baroque", "renaissance", "neo-renaissance", "neo-baroque"):
        return "corinthian"
    return "ionic"


def _frame_angle(items):
    """Angle of the landmark's main axis, which square plinths and abaci follow."""
    big = max(items, key=lambda it: it.length * it.width)
    return math.atan2(-big.axis[1], big.axis[0])


def _rect(e, extra=0.0):
    """Oriented rectangle in e's frame round its supports: column tops and the walls under it."""
    c, axis, _, _ = roofs.box(e.ring)
    u = np.asarray(axis, float)
    v = np.array([-u[1], u[0]])
    pts = []
    for col in e.data["columns"]:
        r = col.data["diameter"] / 2 * 0.88
        p = np.asarray(col.centre)
        pts += [p + r * u, p - r * u, p + r * v, p - r * v]
    for w in e.data.get("under", []):
        pts += list(w.ring)
    pts = np.asarray(pts, float)
    pu, pv = (pts - c) @ u, (pts - c) @ v
    u0, u1, v0, v1 = pu.min() - extra, pu.max() + extra, pv.min() - extra, pv.max() + extra
    corners = [c + a * u + b * v for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    return dict(centre=c, u=u, v=v, bounds=(u0, u1, v0, v1), ring=roofs.ccw(np.array(corners)))


def _clip(ring, rect, inset):
    """ring with its points pulled inside rect less inset (in rect's frame)."""
    c, u, v = rect["centre"], rect["u"], rect["v"]
    u0, u1, v0, v1 = rect["bounds"]
    pu = np.clip((ring - c) @ u, u0 + inset, u1 - inset)
    pv = np.clip((ring - c) @ v, v0 + inset, v1 - inset)
    out = c + pu[:, None] * u + pv[:, None] * v
    out = roofs.clean(out, 0.05)
    return roofs.ccw(out) if len(out) >= 3 else ring


def _entablature_plan(e):
    """The architrave sits on the column tops, whatever the outline mapped; the mapped outline (or its cornice
    slab) only sets how far the cornice projects."""
    rect = _rect(e, 0.04)
    e.data["rect"] = rect
    outer = e.data.get("cornice", e.ring)
    c, axis, hu, hv = roofs.box(outer)
    u0, u1, v0, v1 = rect["bounds"]
    reach = min(hu - (u1 - u0) / 2, hv - (v1 - v0) / 2)
    e.data["overhang"] = float(np.clip(reach, 0.35, 0.9))
    e.ring = rect["ring"]
    e.data["axes"] = _axes(e.ring, e.data["columns"])


def _axes(ring, columns):
    """Column axis positions along each edge of ring, for triglyphs over the columns."""
    axes = {}
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        d = q - p
        length = float(np.hypot(*d))
        d = d / max(length, 1e-9)
        n = np.array([-d[1], d[0]])
        for col in columns:
            rel = np.asarray(col.centre) - p
            if abs(rel @ n) <= col.data["diameter"] * 0.9 and 0 <= rel @ d <= length:
                axes.setdefault(k, []).append(float(rel @ d))
    return axes


def _temple_plan(e, ground):
    """A pitched roof on columns: the columns and cella stop under a synthesised entablature."""
    cols = e.data["columns"]
    d = float(np.median([c.data["diameter"] for c in cols]))
    top = e.y0
    he = float(np.clip(1.9 * d, 1.0, 0.25 * (top - ground)))
    for c in cols:
        c.y1 = min(c.y1, top - he)
    for w in e.data["under"]:
        w.y1 = min(w.y1, top - he)
        w.data["covered"] = True
    rect = _rect(e, 0.04)
    _, axis, hu, hv = roofs.box(e.ring)
    u0, u1, v0, v1 = rect["bounds"]
    e.data.update(rect=rect, he=he, top=top, axes=_axes(rect["ring"], cols),
                  overhang=float(np.clip(min(hu - (u1 - u0) / 2, hv - (v1 - v0) / 2), 0.3, 0.8)),
                  roof=material(_roof_material(e.tags)))
    e.data["materials"] = (e.data["roof"],)


def _roof_material(tags):
    rgb = colour(tags.get("roof:colour"))
    m = tags.get("roof:material")
    if m in ("copper",) or (rgb and rgb[1] > rgb[0] + 30):
        return "roof/copper-green"
    if m in ("slate", "stone"):
        return "roof/slate"
    if m in ("metal", "zinc", "tin", "lead"):
        return "roof/zinc"
    return "roof/copper-green" if m is None else "roof/tiles-red" if "tile" in m else "roof/slate"


def _stacks(items, columns, ground, named):
    """Round parts sharing an axis, styled by their place in the stack: a memorial column's base, fluted shaft
    drums, garland rings, capital and statue pedestal; or a colonnade's platform and round entablature."""
    drums = [it for it in items if it.role in ("drum", "entablature") and it.round]
    stacks = []
    for it in sorted(drums, key=lambda it: it.y0):
        for s in stacks:
            if math.dist(s[0].centre, it.centre) < 0.6:
                s.append(it)
                break
        else:
            stacks.append([it])
    rods = [it for it in items if it.role == "rod"]
    for stack in stacks:
        for it in stack:
            if it.role == "entablature":
                it.data["style"] = "entablature"
            elif any(abs(c.y0 - it.y1) <= 0.6 and math.dist(c.centre, it.centre) < it.radius for c in columns):
                it.data["style"] = "platform"
            else:
                it.data["style"] = "plain"
        narrow = [it for it in stack if it.role == "drum" and it.radius <= 3.5 and it.height >= 2.0]
        if not narrow:
            continue
        # The shaft is the radius the most height of narrow drums shares.
        shaft_r = max((it.radius for it in narrow), key=lambda r: sum(
            o.height for o in narrow if abs(o.radius - r) <= 0.1 * r))
        body = [it for it in stack if it.role == "drum" and it.radius <= 1.4 * shaft_r and it.y0 >= narrow[0].y0 - 6]
        if sum(it.height for it in body) < 6 * shaft_r:
            continue
        shafts = [it for it in body if abs(it.radius - shaft_r) <= 0.06 * shaft_r and it.height >= 1.5]
        if not shafts:
            continue
        first, last = min(it.y0 for it in shafts), max(it.y1 for it in shafts)
        capital_done = False
        for it in stack:
            if it in shafts:
                it.data["style"] = "shaft"
                level = [r for r in rods if it.y0 - 0.1 <= r.y0 <= it.y1 and math.dist(r.centre, it.centre) < 2 * it.radius]
                if len(level) >= 8:
                    # Cannon barrels set into the flutes: as many flutes as barrels, lined up with them.
                    a = [math.atan2(-(r.centre[1] - it.centre[1]), r.centre[0] - it.centre[0]) for r in level]
                    it.data["flutes"] = len(level)
                    it.data["phase"] = float(min(x % (2 * math.pi / len(level)) for x in a)) - math.pi / len(level)
                    for r in level:
                        r.data["cannon"] = it
            elif it.y0 >= first - 4.0 and it.y1 <= first + 0.1 and it.radius <= 1.4 * shaft_r:
                it.data["style"] = "base"
            elif first < it.y0 <= last + 0.1 and it.height <= 1.3 and it.radius < 1.2 * shaft_r:
                it.data["style"] = "ring"
                if named.get("rings"):
                    it.material = material(named["rings"])
            elif it.y0 >= last - 0.1 and not capital_done and it.radius > shaft_r * 1.1:
                it.data["style"] = "capital"
                it.data["shaft"] = shaft_r
                it.data["capital"] = named.get("capital", "corinthian")
                capital_done = True


# Drawing.

def draw(geo, it, detail, rng, items):
    role = it.role
    stone, carved = it.material, _carved(it.material)
    if role == "merged":
        return
    if role == "steps":
        _steps(geo, it, items, detail)
    elif role == "rod":
        _rod(geo, it, detail)
    elif role == "column":
        classical.column(geo, stone, carved, it.centre, it.y0, it.y1, it.data["diameter"], it.data["order"], detail,
                         base=True, angle=it.data["angle"], flutes=it.data.get("flutes"), foot=geo.ground)
    elif role == "drum" or (role == "entablature" and it.round):
        _drum(geo, it, detail, rng)
    elif role == "entablature":
        classical.entablature(geo, stone, carved, it.ring, it.y0, it.y1, it.data["order"], detail,
                              it.data["overhang"], it.data["axes"], rng, metopes=it.data["sculpted"])
    elif role == "temple":
        _temple(geo, it, detail, rng)
    elif role == "attic":
        classical.block(geo, stone, carved, it.ring, it.y0, it.y1, detail, plinth=True, cornice=True, rng=rng,
                        sculpted=it.data["sculpted"], cornice_out=0.12)
    elif role == "wall":
        _wall(geo, it, detail, rng)
    elif role == "arch":
        _arch(geo, it, detail)
    else:
        _block(geo, it, detail, rng)


def _steps(geo, it, items, detail):
    count = int(number(it.tags.get("step_count")) or max(1, round(it.height / 0.17)))
    count = max(1, min(count, 12))
    above = [o for o in items if o is not it and o.role == "steps" and abs(o.y0 - it.y1) < 0.1]
    tread = 0.35
    if above:
        gap = min(_reach(it.ring, o.ring) for o in above)
        tread = float(np.clip(gap / count, 0.25, 0.45))
    ring = _round_ring(it, detail) if it.round else it.ring
    ground = geo.ground if geo.ground is not None else it.y0
    y0 = max(it.y0, ground - 0.02) if it.y0 < ground + 0.1 else it.y0
    if y0 > it.y0:
        classical.prism(geo, it.material, ring, it.y0, y0, top=False, smooth=it.round)
    classical.steps(geo, it.material, ring, y0, it.y1, count, tread, detail, smooth=it.round)


def _reach(outer, inner):
    """How far outer reaches beyond inner, roughly: the gap between their mean radii."""
    c = roofs.centroid(inner)
    return float(np.hypot(*(outer - c).T).mean() - np.hypot(*(inner - c).T).mean())


def _round_ring(it, detail):
    n = classical._segments(it.radius, detail) if detail < 2 else 16
    n = max(n, 24 if detail < 2 else 16)
    th = 2 * math.pi * np.arange(n) / n
    return np.column_stack([it.centre[0] + it.radius * np.cos(th), it.centre[1] - it.radius * np.sin(th)])


def _rod(geo, it, detail):
    """A cannon barrel (when set against a column shaft) or a plain metal rod."""
    r = max(it.length / 2, 0.11 if it.data.get("cannon") else 0.05)
    y0, y1 = it.y0, it.y1
    h = y1 - y0
    seg = 8 if detail == 0 else 6
    if detail >= 1:
        classical.lathe(geo, it.material, it.centre, [[(r, y0), (r, y1)]], 4)
        return
    if it.data.get("cannon"):
        # Muzzle up: a breech ring at the bottom, a taper, a reinforce ring and a swelling muzzle.
        profile = [[(0.55 * r, y0), (1.15 * r, y0 + 0.06 * h), (1.15 * r, y0 + 0.12 * h), (0.95 * r, y0 + 0.14 * h)],
                   [(0.95 * r, y0 + 0.14 * h), (0.8 * r, y0 + 0.86 * h), (0.95 * r, y0 + 0.95 * h), (0.9 * r, y1)]]
        classical.lathe(geo, it.material, it.centre, profile, seg)
        classical.disc(geo, it.material, it.centre, 0.9 * r, y1, True, seg)
    else:
        classical.lathe(geo, it.material, it.centre, [[(r, y0), (r, y1)]], seg)
        classical.disc(geo, it.material, it.centre, r, y1, True, seg)


def _drum(geo, it, detail, rng):
    """A round part, by style: fluted shaft, garland ring, base mouldings, capital, round entablature, platform or
    a plain drum."""
    style = it.data.get("style", "plain")
    stone, carved = it.material, _carved(it.material)
    c, r, y0, y1, h = it.centre, it.radius, it.y0, it.y1, it.height
    seg = max(24, classical._segments(r, detail)) if detail < 2 else 12
    if style == "shaft":
        flutes = it.data.get("flutes", 24)
        if detail < 2:
            # Four segments per flute up close, two further away.
            segments = flutes * (4 if detail == 0 else 2)
            classical.lathe(geo, stone, c, [[(r, y0), (r, y1)]], segments, flutes=flutes, depth=0.03 * 2 * r,
                            fluted=(y0, y1), phase=it.data.get("phase", 0.0), reference=r)
        else:
            classical.lathe(geo, stone, c, [[(r, y0), (r, y1)]], seg)
        return
    if style == "ring":
        if detail >= 2:
            classical.lathe(geo, stone, c, [[(r, y0), (r, y1)]], seg)
            return
        # A garland: a fat torus, leafy with fine vertical ribs, between two fillets.
        f = 0.12 * h
        runs = [[(r * 0.97, y0), (r * 0.97, y0 + f)], classical.torus(r * 0.93, y0 + f, y1 - f, r * 0.07, 6),
                [(r * 0.97, y1 - f), (r * 0.97, y1)]]
        classical.lathe(geo, stone, c, runs[:1] + runs[2:], seg)
        classical.lathe(geo, stone, c, runs[1:2], 96 if detail == 0 else 48, flutes=48 if detail == 0 else 0,
                        depth=0.04 * r / 2.5, fluted=(y0, y1))
        return
    if style == "base":
        classical.lathe(geo, stone, c, classical.base_profile(r * 0.92, y0, h, detail), seg)
        return
    if style == "capital":
        shaft = it.data["shaft"]
        frame = Frame(c, (1.0, 0.0))
        classical.capital(geo, stone, carved, frame, c, y0, y1, shaft, 2 * shaft, it.data["capital"], detail, seg)
        return
    if style == "entablature":
        ring = _round_ring(it, detail)
        classical.entablature(geo, stone, carved, ring, y0, y1, "ionic", detail, min(0.5, 0.25 * h), None, rng,
                              round_=True)
        return
    if it.data.get("dome") and not it.data.get("support"):
        _dome(geo, it, it.data["dome"], detail)
    # Platforms and plain drums: a base course and a cornice, or just a cap on short drums.
    if h >= 1.0:
        runs = classical.plinth_runs(y0, y0 + min(0.5, 0.2 * h), min(0.1, 0.03 * r + 0.03), detail)
        runs += [[(0.0, y0 + min(0.5, 0.2 * h)), (0.0, y1 - min(0.5, 0.2 * h))]]
        runs += classical.band_runs(y1 - min(0.5, 0.2 * h), y1, min(0.3, 0.05 * r + 0.05), detail)[:-1]
        ring = [[(r + a, y) for a, y in run] for run in runs]
        classical.lathe(geo, stone, c, ring, seg, reference=r)
        classical.disc(geo, stone, c, r, y1, True, seg)
        if style == "platform" or it.y0 > (geo.ground or 0) + 0.5:
            classical.disc(geo, stone, c, r, y0, False, seg)
    else:
        classical.lathe(geo, stone, c, [[(r, y0), (r, y1)]], seg)
        classical.disc(geo, stone, c, r, y1, True, seg)
        classical.disc(geo, stone, c, r, y0, False, seg)


def _dome(geo, it, rise, detail):
    """A ribbed metal dome on a drum, with a small lantern at its crown."""
    roof_mat = material(_roof_material(it.tags))
    geo.sizes.setdefault(roof_mat, _size(roof_mat))
    c, r, y = it.centre, it.radius, it.y1
    steps = 10 if detail == 0 else 5 if detail == 1 else 3
    a = np.linspace(0, math.pi / 2, steps + 1)[:-1]
    profile = [(r * 0.97 * math.cos(t) + 0.01, y + rise * math.sin(t)) for t in a]
    top = profile[-1]
    seg = max(24, classical._segments(r, detail)) if detail < 2 else 12
    classical.lathe(geo, roof_mat, c, [profile], seg if detail else 64, flutes=16 if detail == 0 else 0,
                    depth=0.03 * r, fluted=(y + 0.1, top[1] + 0.1))
    classical.disc(geo, roof_mat, c, top[0], top[1], True, seg)
    if detail < 2 and r > 2.0:
        lr = max(0.4, 0.12 * r)
        classical.lathe(geo, it.material, c, [[(lr, top[1] - 0.1), (lr, top[1] + 1.4 * lr)]], 12)
        classical.lathe(geo, roof_mat, c, [[(lr * 1.15, top[1] + 1.4 * lr), (0.05, top[1] + 2.4 * lr)]], 12)


def _wall(geo, it, detail, rng):
    """A gate's passage wall: plinth, a framed relief on each long face, antae at the ends and an impost."""
    stone, carved = it.material, _carved(it.material)
    ring, y0, y1, h = it.ring, it.y0, it.y1, it.height
    ground = geo.ground if geo.ground is not None else y0
    hp = min(0.8, 0.07 * h)
    hi = min(0.6, 0.05 * h)
    yb = max(y0, ground) + hp
    classical.sweep(geo, stone, ring, classical.plinth_runs(y0, yb, 0.1, detail))
    classical.sweep(geo, stone, ring, [[(0.0, yb), (0.0, y1 - hi)]])
    classical.sweep(geo, stone, ring, classical.band_runs(y1 - hi, y1, 0.12, detail)[:-1])
    classical.cap(geo, stone, classical.offset_path(ring, 0.12), y1, True)
    if detail >= 2:
        return
    longest = max(float(np.hypot(*(ring[(k + 1) % len(ring)] - ring[k]))) for k in range(len(ring)))
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        frame = Frame.edge(p, q)
        length = float(np.hypot(*(q - p)))
        if length >= 0.6 * longest:
            m = min(0.9, 0.15 * length)
            classical.panel(geo, stone, carved, frame, m, length - m, y0 + 0.33 * h, y0 + 0.66 * h, 0.0, detail, rng,
                            sculpted=it.data["sculpted"])
            classical.panel(geo, stone, carved, frame, m, length - m, y0 + 0.72 * h, y1 - hi - 0.35, 0.0, detail,
                            rng)
        elif length > 0.8:
            # Anta: a pilaster across the wall's end, with a small capital under the impost.
            classical.box(geo, stone, frame, 0.08, length - 0.08, yb, y1 - hi, 0.0, 0.07, "flr")
            classical.box(geo, carved, frame, 0.02, length - 0.02, y1 - hi - 0.35, y1 - hi, 0.0, 0.12, "fblr")


def _arch(geo, it, detail):
    """A gate block with a round-headed passage through its long side, under a cornice."""
    stone, carved = it.material, _carved(it.material)
    c, axis, hu, hv = roofs.box(it.ring)
    d = np.asarray(axis, float)
    frame = Frame(np.asarray(c) - d * hu, d)
    length, h = 2 * hu, it.height
    hk = min(0.8, 0.12 * h)
    w = float(np.clip(0.4 * length, 2.5, 8.0))
    spring = it.y0 + min(0.72 * (h - hk), 0.35 * (h - hk) + w) - w / 2
    if spring < it.y0 + 1.5:
        _block(geo, it, detail, None)
        return
    classical.arch_wall(geo, stone, carved, frame, length, it.y0, it.y1 - hk, 2 * hv,
                        (hu - w / 2, hu + w / 2, spring), detail)
    rect = it.ring
    classical.sweep(geo, stone, rect, classical.band_runs(it.y1 - hk, it.y1, min(0.35, 0.5 * hk), detail)[:-1])
    classical.cap(geo, stone, classical.offset_path(rect, min(0.35, 0.5 * hk)), it.y1, True)


def _block(geo, it, detail, rng):
    stone, carved = it.material, _carved(it.material)
    ground = geo.ground if geo.ground is not None else it.y0
    on_ground = it.y0 <= ground + 0.3
    tall = it.height > 4.0
    classic = it.tags.get("building:architecture") in CLASSICAL or it.data.get("sculpted")
    roof = it.piece.roof if it.piece.roof is not None and it.piece.roof.shape not in ("flat",) else None
    if it.data.get("covered"):
        # A cella under a temple's entablature: plain walls with a base course and antae at the corners.
        classical.block(geo, stone, carved, it.ring, it.y0, it.y1, detail, plinth=on_ground, cornice=False,
                        panels=True, rng=rng, ground=on_ground, pilasters=True)
        return
    reliefs = it.data.get("reliefs")
    if reliefs and on_ground is False and it.height > 3.0 and it.data.get("sculpted"):
        # A monument's base: framed bronze reliefs on its faces.
        classical.block(geo, stone, carved, it.ring, it.y0, it.y1, detail, plinth=True, cornice=True, panels=False,
                        rng=rng, ground=False)
        _reliefs(geo, it, reliefs, detail, rng)
    else:
        kind = it.data.get("windows")
        glass = material("glass/window") if kind else None
        if glass:
            geo.sizes.setdefault(glass, _size(glass))
        classical.block(geo, stone, carved, it.ring, it.y0, it.y1 if roof is None else roof.eave, detail,
                        plinth=on_ground or it.height > 2.0, cornice=True, panels=classic or it.height < 12,
                        rng=rng, ground=on_ground, pilasters=classic and tall and on_ground and kind != "arched",
                        sculpted=it.data.get("sculpted") and it.height < 8, windows=kind if it.height > 3 else None,
                        glass=glass, buttresses=kind == "arched" and on_ground)
    if roof is None and classic and tall and on_ground and detail < 2 and not it.data.get("sculpted"):
        # A balustrade round a classical flat roof; long ones get a solid parapet instead of balusters.
        ring = classical.offset_path(it.ring, -0.25)
        perimeter = float(np.hypot(*(np.roll(ring, -1, axis=0) - ring).T).sum())
        classical.balustrade(geo, stone, ring, it.y1, 1.0, detail if perimeter < 120 else 2, closed=True)
    if roof is not None:
        _roof(geo, it, roof, detail)


def _reliefs(geo, it, metal, detail, rng):
    ring, y0, y1, h = it.ring, it.y0, it.y1, it.height
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        length = float(np.hypot(*(q - p)))
        if length < 4.0:
            continue
        frame = Frame.edge(p, q)
        m = 0.12 * length
        a, b = y0 + 0.3 * h, y1 - 0.25 * h
        classical.box(geo, it.material, frame, m - 0.15, length - m + 0.15, a - 0.15, b + 0.15, 0.0, 0.05, "ftblr")
        # Bronze darkened to its patina, so it doesn't read as polished copper.
        classical.relief(geo, metal, frame, m, length - m, a, b, 0.05, min(0.2, 0.08 * (b - a)), rng, detail,
                         tone=0.6)


def _roof(geo, it, roof, detail):
    """A fitted pitched roof or dome on a block: gable walls up to the roof and the roof surface."""
    roof_mat = material(_roof_material(it.tags))
    geo.sizes.setdefault(roof_mat, _size(roof_mat))
    if roof.shape == "dome" and it.round:
        r, c = it.radius, it.centre
        steps = 8 if detail == 0 else 4 if detail == 1 else 2
        a = np.linspace(0, math.pi / 2, steps + 1)
        top = roof.eave + roof.rise
        profile = [(r * math.cos(t) + 0.02, roof.eave + roof.rise * math.sin(t)) for t in a[:-1]]
        seg = max(24, classical._segments(r, detail)) if detail < 2 else 12
        classical.lathe(geo, roof_mat, c, [profile], seg, flutes=16 if detail == 0 else 0, depth=0.04 * r,
                        fluted=(roof.eave + 0.2, top - 0.2))
        classical.disc(geo, roof_mat, c, profile[-1][0], profile[-1][1], True, seg)
        return
    surface = roofs.surface(roof, it.ring, it.piece.holes if it.piece.ring is it.ring else [])
    for t in surface:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        if n[1] < 0:
            n = -n
        geo.add(roof_mat, t, np.tile(n, (3, 1)), t[:, [0, 2]] * [1, -1] + [0, 0], [[0, 1, 2]])
    if roof.shape != "dome":
        for k in range(len(it.ring)):
            p, q = it.ring[k], it.ring[(k + 1) % len(it.ring)]
            prof = roofs.profile(roof, p, q, it.ring)
            length = float(np.hypot(*(q - p)))
            pts = [(0.0, roof.eave)] + [(t * length, y) for t, y in prof] + [(length, roof.eave)]
            if max(y for _, y in prof) - roof.eave > 0.05:
                classical.polygon(geo, it.material, Frame.edge(p, q), pts, 0.0, 1)


def _temple(geo, e, detail, rng):
    """Entablature, pediments at the gable ends and a metal roof over a colonnade."""
    stone, carved = e.material, _carved(e.material)
    rect = e.data["rect"]
    top, he, oh = e.data["top"], e.data["he"], e.data["overhang"]
    classical.entablature(geo, stone, carved, rect["ring"], top - he, top, e.data["order"], detail, oh,
                          e.data["axes"], rng, metopes=False)
    c, u, v = rect["centre"], rect["u"], rect["v"]
    u0, u1, v0, v1 = rect["bounds"]
    # The ridge runs along the longer side.
    if u1 - u0 < v1 - v0:  # Turn the frame a quarter so u runs along the ridge.
        u, v = v, -u
        u0, u1, v0, v1 = _bounds(rect, u, v)
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    cc = c + uc * u + vc * v
    lu, lv = (u1 - u0) / 2, (v1 - v0) / 2
    ridge = e.y1
    tc = min(0.45, 0.25 * he)
    s = max(0.12, (ridge - tc - top) / (lv + oh))
    roof_mat = e.data["roof"]

    def at(a, b, y):
        p = cc + a * u + b * v
        return np.array([p[0], y, p[1]])

    up = np.array([0.0, 1.0, 0.0])
    for side in (-1, 1):
        # Roof plane from the eave (side * (lv + oh)) up to the ridge, lying on the raking cornices.
        e0, e1 = at(-(lu + oh), side * (lv + oh), top + tc), at(lu + oh, side * (lv + oh), top + tc)
        r0, r1 = at(-(lu + oh), 0.0, top + tc + s * (lv + oh)), at(lu + oh, 0.0, top + tc + s * (lv + oh))
        quad = np.array([e0, e1, r1, r0])
        n = np.cross(e1 - e0, r0 - e0)
        n = n if n[1] > 0 else -n
        slope = float(np.linalg.norm(r0 - e0))
        uv = np.array([[0, 0], [2 * (lu + oh), 0], [2 * (lu + oh), slope], [0, slope]])
        geo.add(roof_mat, quad, np.tile(n, (4, 1)), uv, [[0, 1, 2], [0, 2, 3]])
        # Sima along the eave.
        f = Frame(np.array([cc[0], cc[1]]) + side * (lv + oh) * v - (lu + oh) * u, u * side)
        classical.box(geo, stone, f, 0.0 if side > 0 else -2 * (lu + oh), 2 * (lu + oh) if side > 0 else 0.0,
                      top, top + tc, -0.05, 0.03, "ftb")
    for end in (-1, 1):
        # Tympanum, set back in the architrave's plane, and the raking cornices above it.
        frame = Frame(np.array([cc[0], cc[1]]) + end * lu * u, -v * end)
        apex = top + s * (lv + oh)
        pts = [(-lv, top), (lv, top), (lv, top + s * oh), (0.0, apex), (-lv, top + s * oh)]
        classical.polygon(geo, stone, frame, pts, -0.12, 1)
        for side in (-1, 1):
            start = at(end * (lu + oh), side * (lv + oh), top)
            ridge_pt = at(end * (lu + oh), 0.0, top + s * (lv + oh))
            ex = ridge_pt - start
            ez = -end * u
            depth = np.array([ez[0], 0.0, ez[1]]) * (oh + 0.12)
            classical.obox(geo, carved, start, ex, up * tc, depth)
        if detail == 0:
            # Acroteria: plinths at the apex and the corners.
            for b, y in ((0.0, apex + tc), (lv + oh * 0.6, top + tc), (-lv - oh * 0.6, top + tc)):
                p = at(end * (lu + oh * 0.5), b, y)
                f = Frame((p[0], p[2]), u)
                classical.box(geo, carved, f, -0.25, 0.25, y - 0.05, y + 0.45, -0.25, 0.25, "fktlr")


def _bounds(rect, u, v):
    pts = rect["ring"] - rect["centre"]
    pu, pv = pts @ u, pts @ v
    return pu.min(), pu.max(), pv.min(), pv.max()


# Sculpture.

def _sculpture(group, items, context, named):
    """The mesh's shape above the topmost part, remeshed, when it rises clearly above it."""
    # Only monuments carry sculpture; elsewhere whatever rises above the top is clutter, aerials or trees.
    if not ({group.tags.get("historic"), group.tags.get("building"), group.tags.get("man_made"),
             group.tags.get("memorial")} & (SCULPTED | {"statue", "sculpture"})):
        return None
    candidates = [it for it in items if it.role not in ("rod", "merged", "column")]
    if not candidates or not len(context.reference.triangles):
        return None
    support = max(candidates, key=lambda it: (round(it.y1, 1), it.length * it.width))
    ring = _round_ring(support, 0) if support.round else support.ring
    level = sculpt.level(context.reference, ring, support.y0 + 0.3)
    if level is None:
        return None
    region_ring = classical.offset_path(ring, 1.0)
    x, z, y = _cells(context.reference, region_ring)
    if not len(y):
        return None
    peak = float(np.nanpercentile(y, 99.5))
    if peak - max(level, support.y1) < MIN_SCULPTURE:
        return None
    # The support reaches the top the mesh shows, so the sculpture stands on it.
    support.y1 = max(support.y0 + 0.3, level)
    support.data["support"] = True
    # Small supports (a statue's pedestal) carry figures much wider than themselves.
    margin = 1.0 if min(support.width, support.length) > 6 else max(1.0, 0.45 * (peak - level))
    region = classical.offset_path(ring, margin)
    result = sculpt.build(context.reference, region, level, budget=SCULPTURE_BUDGET)
    if result is None:
        return None
    lods, voxel = result
    name = named.get("sculpture") or SCULPTURE
    return dict(lods=lods, voxel=voxel, material=material(name), source="reference")


def _cells(reference, ring):
    from .buildings import cells

    x, z, y = cells(reference.top, [ring], 0.0)
    keep = np.isfinite(y)
    return x[keep], z[keep], y[keep]


# Colliders.

def _colliders(it):
    if it.role in ("rod", "merged"):
        return []
    if it.role == "column":
        r = it.data["diameter"] / 2
        th = np.linspace(0, 2 * math.pi, 9)[:-1]
        ring = np.column_stack([it.centre[0] + r * np.cos(th), it.centre[1] + r * np.sin(th)])
        return [_prism_hull(ring, it.y0, it.y1)]
    if it.role == "temple":
        ring = classical.offset_path(it.data["rect"]["ring"], it.data["overhang"])
        return [_prism_hull(ring, it.data["top"] - it.data["he"], it.y1)]
    if it.round:
        th = np.linspace(0, 2 * math.pi, 17)[:-1]
        ring = np.column_stack([it.centre[0] + it.radius * np.cos(th), it.centre[1] + it.radius * np.sin(th)])
        return [_prism_hull(ring, it.y0, it.y1)]
    out = []
    points, triangles = roofs.triangulate(it.ring)
    for poly in roofs.pieces(np.asarray(points), triangles):
        p = np.asarray(points)[poly]
        if abs(roofs.area(p)) >= 0.05:
            out.append(_prism_hull(p, it.y0, it.y1))
    return out


def _prism_hull(ring, y0, y1):
    verts = [(x, y0, z) for x, z in ring] + [(x, y1, z) for x, z in ring]
    return Collider("convex", np.unique(np.round(np.array(verts, np.float32), 3), axis=0))


def _hull(points):
    """A coarse convex collider round a sculpture: its extreme points in 26 directions."""
    dirs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
                    float)
    idx = np.unique(np.argmax(points @ dirs.T, axis=0))
    return Collider("convex", np.round(points[idx].astype(np.float32), 3))


# Plain prisms (game.toml mode = "prism").

def _prism(group, context, centre, extras):
    material_ = group.override.get("material") or LANDMARK_MATERIAL
    white = (1.0, 1.0, 1.0, 1.0)
    look = dict(wall=material_, roof_flat=material_, roof_pitched=material_, trim=material_, tint=white,
                roof_tint=white, plinth=(0.0, material_), parapet=0.0)
    lods, colliders = [], []
    for lod in range(3):
        mesh = Mesh(context.unit)
        for piece in group.pieces:
            flat = Piece(None, {}, piece.ring, piece.holes, piece.base, piece.base,
                         roofs.Roof("flat", piece.roof.ridge, 0.0), raised=True)
            _tier(mesh, flat, look)
            if lod == 0:
                colliders += colliders_for(flat)
        lods.append(mesh.parts())
    extras.update(kind="landmark", style="prism", material=material_, tiers=len(group.pieces),
                  triangles=[int(sum(len(p.indices) // 3 for p in parts)) for parts in lods])
    return Model("landmark/" + group.id.replace("/", "-"), centre, lods, colliders, extras)

