"""Write a tiny game folder for testing aleph_view without the Python pipeline.

    python viewer/fixtures/make_fixture.py OUT [GRID [TREES]]

GRID×GRID chunks (default 2×2) of 64 m with a rolling heightfield, box buildings with convex colliders, a ramp and stairs with trimesh
colliders, instanced trees with capsule colliders, and solid-colour textures it draws itself. LOD1 and LOD2 are tinted
blue and red on purpose, so switches are visible in screenshots.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.game.gltf import Node, write  # noqa: E402
from src.game.types import Collider, Material, Part  # noqa: E402

CHUNK, SPACING = 64.0, 0.5
LODS = (50, 120, 300)
LOD_TINT = ((1, 1, 1, 1), (0.6, 0.7, 1.0, 1), (1.0, 0.55, 0.5, 1))


def height(x, z):
    return 0.6 * np.sin(x / 15.0) + 0.4 * np.cos(z / 11.0) + 0.002 * x


def textures(library):
    """Draw tiling placeholder textures: brick, paving, grass, bark, leaves, roof."""
    def save(name, size, draw, ext="png"):
        image = Image.new("RGB", (size, size))
        if isinstance(draw, tuple):  # Noise around a base colour.
            rgb, amount, seed = draw
            noise = np.random.default_rng(seed).normal(0, amount, (size, size, 1))
            image = Image.fromarray(np.clip(np.array(rgb) + noise, 0, 255).astype(np.uint8))
        else:
            draw(ImageDraw.Draw(image), size)
        path = library / "materials" / name / f"color.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=92)
        return f"materials/{name}/color.{ext}"

    def brick(d, s):
        d.rectangle((0, 0, s, s), fill=(205, 200, 190))
        rows = 8
        for r in range(rows):
            shift = (r % 2) * s // 8
            for c in range(-1, 4):
                x = c * s // 4 + shift
                tone = 150 + (r * 7 + c * 13) % 40
                d.rectangle((x + 3, r * s // rows + 3, x + s // 4 - 3, (r + 1) * s // rows - 3), fill=(tone, 70, 50))

    def tiles(d, s):
        d.rectangle((0, 0, s, s), fill=(120, 120, 125))
        for i in range(4):
            for j in range(4):
                tone = 165 + (i * 5 + j * 11) % 25
                d.rectangle((i * s // 4 + 2, j * s // 4 + 2, (i + 1) * s // 4 - 2, (j + 1) * s // 4 - 2), fill=(tone,) * 3)

    return dict(brick=save("wall/brick", 256, brick, "jpg"), paving=save("ground/paving", 256, tiles),
                grass=save("ground/grass", 128, ((70, 120, 45), 18, 1), "jpg"),
                bark=save("nature/bark", 64, ((90, 65, 45), 12, 2)),
                leaves=save("nature/leaves", 64, ((55, 105, 40), 20, 3)),
                roof=save("roof/gravel", 128, ((95, 95, 100), 15, 4)))


def materials(tex):
    base = dict(
        grass=Material("ground/grass", 2.0, tex["grass"]),
        paving=Material("ground/paving", 2.0, tex["paving"], roughness=0.8),
        brick=Material("wall/brick", 2.0, tex["brick"], roughness=0.9),
        roof=Material("roof/gravel", 3.0, tex["roof"]),
        bark=Material("nature/bark", 1.0, tex["bark"]),
        leaves=Material("nature/leaves", 2.0, tex["leaves"], extras=dict(wind=True)),
        horizon=Material("horizon/grey", 0.0, factor=(0.55, 0.57, 0.6, 1.0)),
    )

    def lookup(lod):
        def get(name):
            m = base[name]
            tint = LOD_TINT[lod]
            return Material(m.name, m.size, m.base_color, factor=tuple(a * b for a, b in zip(m.factor, tint)),
                            roughness=m.roughness, extras=m.extras)
        return get
    return lookup


def box(material, lo, hi, color=None):
    """Axis-aligned box Part; uvs in metres along each face."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = [  # corners counter-clockwise seen from outside, normal
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
        ([(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)], (0, 0, -1)),
        ([(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)], (1, 0, 0)),
        ([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], (-1, 0, 0)),
        ([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], (0, 1, 0)),
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
    ]
    positions, normals, uvs, indices = [], [], [], []
    for corners, n in faces:
        start = len(positions)
        c = np.array(corners, float)
        u_axis = c[1] - c[0]
        u_len, v_len = np.linalg.norm(u_axis), np.linalg.norm(c[3] - c[0])
        positions += corners
        normals += [n] * 4
        uvs += [(0, 0), (u_len, 0), (u_len, v_len), (0, v_len)]
        indices += [start, start + 1, start + 2, start, start + 2, start + 3]
    colors = None if color is None else np.tile(np.array(color, "f4"), (len(positions), 1))
    return Part(material, np.array(positions, "f4"), np.array(normals, "f4"), np.array(uvs, "f4"),
                np.array(indices, "u4"), colors)


def grid(material, x0, z0, cols, rows, step, heights, center):
    """Heightfield mesh over heights[row, col], every step-th sample, relative to center."""
    h = heights[::step, ::step]
    r, c = h.shape
    xs = x0 + np.arange(c) * SPACING * step
    zs = z0 + np.arange(r) * SPACING * step
    gx, gz = np.meshgrid(xs, zs)
    positions = np.column_stack((gx.ravel() - center[0], h.ravel(), gz.ravel() - center[2])).astype("f4")
    dz, dx = np.gradient(h, SPACING * step)
    normals = np.column_stack((-dx.ravel(), np.ones(h.size), -dz.ravel())).astype("f4")
    uvs = np.column_stack((gx.ravel(), -gz.ravel())).astype("f4")
    i = np.arange(r * c).reshape(r, c)
    a, b, d, e = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    # Rows advance along +z (south), so a→d→b is counter-clockwise seen from above.
    indices = np.stack((a, d, b, b, d, e), -1).ravel().astype("u4")
    return Part(material, positions, normals, uvs, indices)


def tree_asset(library, mats):
    folder = library / "assets" / "tree" / "blob"
    for lod in range(3):
        crown_sides = (12, 8, 4)[lod]
        angles = np.linspace(0, 2 * math.pi, crown_sides, endpoint=False)
        # A double cone crown, faceted; v up in metres.
        ring = np.column_stack((2.5 * np.cos(angles), np.full(crown_sides, 6.0), 2.5 * np.sin(angles)))
        positions, normals, uvs, indices = [], [], [], []
        for k in range(crown_sides):
            p, q = ring[k], ring[(k + 1) % crown_sides]
            for apex in ((0, 9.5, 0), (0, 3.0, 0)):
                tri = [p, q, apex] if apex[1] < 5 else [q, p, apex]
                n = np.cross(np.subtract(tri[1], tri[0]), np.subtract(tri[2], tri[0]))
                start = len(positions)
                positions += tri
                normals += [n / np.linalg.norm(n)] * 3
                uvs += [(0, 0), (1.5, 0), (0.75, 2)]
                indices += [start, start + 1, start + 2]
        crown = Part("leaves", np.array(positions, "f4"), np.array(normals, "f4"), np.array(uvs, "f4"),
                     np.array(indices, "u4"))
        trunk = box("bark", (-0.2, 0, -0.2), (0.2, 4.0, 0.2))
        write(folder / f"lod{lod}.glb", [Node("tree", [trunk, crown])], mats(lod), library)
    return dict(lods=[f"library/assets/tree/blob/lod{i}.glb" for i in range(3)],
                colliders=[dict(shape="capsule", radius=0.25, half_height=1.8, center=[0, 2.0, 0])],
                height=9.5, radius=2.5)


def ramp(material, x, z, width, length, rise):
    """A wedge rising along -z; returns its Part and trimesh Collider (relative coordinates)."""
    p = np.array([(x, 0, z), (x + width, 0, z), (x + width, rise, z - length), (x, rise, z - length),
                  (x, 0, z - length), (x + width, 0, z - length)], "f4")
    tris = np.array([(0, 1, 2), (0, 2, 3), (4, 3, 2), (4, 2, 5), (0, 3, 4), (1, 5, 2), (0, 4, 5), (0, 5, 1)], "u4")
    pos = p[tris.ravel()]
    nrm = np.repeat(np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3]), 3, axis=0)
    uv = np.column_stack((pos[:, 0] + pos[:, 2], pos[:, 1] - pos[:, 2]))
    part = Part(material, pos, nrm, uv, np.arange(len(pos), dtype="u4"))
    return part, Collider("trimesh", p, tris)


def main(out, size=2, trees=4):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    library = out / "library"
    mats = materials(textures(library))
    assets = {"tree/blob": tree_asset(library, mats)}
    rng = np.random.default_rng(7)
    chunks = []
    for ci in range(size):
        for cj in range(size):
            x0, z0 = (ci - size / 2) * CHUNK, (cj - size / 2) * CHUNK
            center = (x0 + CHUNK / 2, 0.0, z0 + CHUNK / 2)
            n = int(CHUNK / SPACING) + 1
            gx, gz = np.meshgrid(x0 + np.arange(n) * SPACING, z0 + np.arange(n) * SPACING)
            heights = height(gx, gz).astype("<f4")
            cid = f"{ci}_{cj}"
            folder = out / "chunks" / cid
            folder.mkdir(parents=True, exist_ok=True)
            heights.tofile(folder / "ground.height")

            # Buildings: boxes on the ground, relative to the chunk center.
            buildings = []
            for k in range(3):
                bx, bz = x0 + 10 + k * 18 + rng.uniform(-2, 2), z0 + 12 + rng.uniform(0, 30)
                w, d, h = rng.uniform(6, 12), rng.uniform(6, 12), rng.uniform(6, 22)
                y = float(height(np.array([bx, bx + w]), np.array([bz, bz + d])).min()) - 0.5
                lo = (bx - center[0], y, bz - center[2])
                hi = (bx + w - center[0], y + h, bz + d - center[2])
                tint = (0.55, 0.75, 1.0, 1.0) if k == 1 else None  # COLOR_0 tint check: bluish brick.
                hull = np.array([(a, b, c) for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
                buildings.append(([box("brick", lo, (hi[0], hi[1] - 0.3, hi[2]), tint),
                                   box("roof", (lo[0], hi[1] - 0.3, lo[2]), hi)], [Collider("convex", hull)]))

            extra = []  # Ramps and stairs at the map centre test slope limit and step-up.
            if x0 == 0 and z0 == 0:
                for k, rise in enumerate((4.0, 9.0)):  # About 22° and 42°: both walkable.
                    part, col = ramp("paving", 4 + k * 6 - center[0], 20 - center[2], 4, 10, rise)
                    extra.append(Node(f"ramp/{k}", [part], colliders=[col]))
                part, col = ramp("paving", 16 - center[0], 20 - center[2], 4, 4, 9.0)  # 66°: too steep.
                extra.append(Node("ramp/steep", [part], colliders=[col]))
                steps = []
                for s in range(8):  # 0.25 m risers, 0.4 m treads.
                    lo = (24 - center[0], -0.5, 6 - s * 0.4 - center[2])
                    steps.append(box("paving", lo, (lo[0] + 3, (s + 1) * 0.25, lo[2] + 0.4)))
                extra.append(Node("stairs", steps, colliders=[Collider("convex", np.array(
                    [(a, b, c) for a in (st.positions[:, 0].min(), st.positions[:, 0].max())
                     for b in (st.positions[:, 1].min(), st.positions[:, 1].max())
                     for c in (st.positions[:, 2].min(), st.positions[:, 2].max())])) for st in steps]))

            for lod in range(3):
                nodes = [Node("ground", [grid("grass", x0, z0, n, n, (1, 2, 8)[lod], heights, center)])]
                for k, (parts, colliders) in enumerate(buildings):
                    nodes.append(Node(f"building/{cid}-{k}", parts, colliders=colliders if lod == 0 else []))
                if lod < 2:
                    nodes += [Node(e.name, e.parts, colliders=e.colliders if lod == 0 else []) for e in extra]
                write(folder / f"lod{lod}.glb", nodes, mats(lod), library)

            instances = []
            for _ in range(trees):
                tx, tz = rng.uniform(x0 + 4, x0 + CHUNK - 4), rng.uniform(z0 + (48 if trees <= 4 else 4), z0 + CHUNK - 4)  # Clear of buildings by default.
                instances.append(dict(asset="tree/blob", position=[tx, float(height(tx, tz)), tz],
                                      yaw=float(rng.uniform(0, 6.28)), scale=float(rng.uniform(0.8, 1.3)),
                                      tint=[1.0, 1.0, 1.0] if len(instances) % 2 else [1.25, 1.0, 0.45]))
            chunks.append(dict(id=cid, center=list(center), min=[x0, -2, z0], max=[x0 + CHUNK, 25, z0 + CHUNK],
                               lods=[f"chunks/{cid}/lod{i}.glb" for i in range(3)],
                               ground=dict(file=f"chunks/{cid}/ground.height", size=[n, n], spacing=SPACING,
                                           offset=[x0, z0]),
                               instances=instances))

    # A ring of grey blocks far outside the play area.
    blocks = []
    for k in range(48):
        a, r = k / 48 * 2 * math.pi, rng.uniform(250, 420)
        cx, cz, s, h = r * math.cos(a), r * math.sin(a), rng.uniform(15, 40), rng.uniform(8, 35)
        blocks.append(box("horizon", (cx - s / 2, -1, cz - s / 2), (cx + s / 2, h, cz + s / 2)))
    # One node per block, so the viewer can hide blocks nearer than horizon.from.
    write(out / "horizon.glb", [Node(f"horizon/{k}", [b]) for k, b in enumerate(blocks)], mats(0), library)

    b = size / 2 * CHUNK - 0.5
    scene = {"format": "aleph-game", "version": 1,
             "origin": {"latitude": 52.5163, "longitude": 13.3772, "base": 18.07},
             "attribution": ["Aleph test fixture"], "chunk_size": CHUNK, "lod_distances": list(LODS),
             "bounds": [[-b, b], [b, b], [b, -b], [-b, -b]],
             "spawn": {"position": [0.0, 1.7 + float(height(0.0, 0.0)), 0.0], "yaw": 0.0},
             "sun": {"azimuth": 205.0, "elevation": 32.0},
             "chunks": chunks, "assets": assets, "horizon": {"file": "horizon.glb", "from": 300}}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixture", *map(int, sys.argv[2:4]))
