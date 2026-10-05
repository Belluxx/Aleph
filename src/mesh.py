"""Google Earth 3D meshes: plan octree nodes, then join saved nodes into one glTF binary.

Earth Web streams photogrammetry as an octree of protobuf messages. BulkMetadata
lists nodes four levels at a time; NodeData holds a node's meshes with JPEG
textures. A finer node replaces one octant of its parent's mesh. Positions lie
on a sphere, onto which latitude and longitude map directly.
Protocol notes: https://github.com/retroplasma/earth-reverse-engineering
"""

import json
import math
import shutil
import struct
import tempfile

import numpy as np

from .common import atomic_path, fetch
from .pbf import fields, varints

BASE = "https://kh.google.com/rt/earth/"
LEAF, NODATA, USE_IMAGERY_EPOCH = 4, 8, 16
JPEG = 1
LOWEST, HIGHEST = -1000, 9000  # Heights searched for nodes, in meters from the sphere.
GLB_LIMIT = 2**32
TILES_PER_MESH = 64  # Unreal's Nanite allows 64 materials per mesh; Godot allows 256 surfaces.


def first(data):
    """Protobuf fields keeping the first value of repeated ones."""
    values = {}
    for number, value in fields(data):
        values.setdefault(number, value)
    return values


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def frame(lat, lon):
    """East, north, and up unit vectors."""
    la, lo = math.radians(lat), math.radians(lon)
    return ((-math.sin(lo), math.cos(lo), 0.0),
            (-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)),
            (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))


def search_box(area, radius):
    """An oriented box around the rectangle, from below sea level to above the highest peaks."""
    south, west, north, east = area
    lat, lon = (south + north) / 2, (west + east) / 2
    axes = frame(lat, lon)
    half_east = half_north = drop = 0
    for y in (south, north, lat, min(max(0, south), north)):
        for x in (west, east, lon):
            up = frame(y, x)[2]
            point = [radius * (a - b) for a, b in zip(up, axes[2])]
            half_east = max(half_east, abs(dot(point, axes[0])))
            half_north = max(half_north, abs(dot(point, axes[1])))
            drop = max(drop, -dot(point, axes[2]))
    middle = (HIGHEST + LOWEST - drop) / 2
    center = tuple((radius + middle) * a for a in axes[2])
    return center, (half_east + 1, half_north + 1, (HIGHEST - LOWEST + drop) / 2), axes


def overlaps(a, b):
    """Separating-axis test for two oriented boxes: (center, half sizes, unit axes)."""
    (c1, h1, r1), (c2, h2, r2) = a, b
    d = (c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2])
    axes = [*r1, *r2]
    for u in r1:
        for v in r2:
            w = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            if dot(w, w) > 1e-12:
                axes.append(w)
    return all(abs(dot(d, x)) <= sum(h * abs(dot(r, x)) for h, r in zip(h1 + h2, r1 + r2)) for x in axes)


def node_box(packed, head, texel):
    if len(packed) != 15:
        raise ValueError("Unexpected 3D mesh node bounds.")
    cx, cy, cz, ex, ey, ez, a0, a1, a2 = struct.unpack("<3h3B3H", packed)
    a0, a1, a2 = a0 * math.pi / 32768, a1 * math.pi / 65536, a2 * math.pi / 32768
    c0, s0, c1, s1, c2, s2 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)
    axes = ((c0 * c2 - c1 * s0 * s2, c1 * c0 * s2 + c2 * s0, s2 * s1),
            (-c0 * s2 - c2 * c1 * s0, c0 * c1 * c2 - s0 * s2, c2 * s1),
            (s1 * s0, -c0 * s1, c1))
    center = (head[0] + cx * texel, head[1] + cy * texel, head[2] + cz * texel)
    return center, (ex * texel, ey * texel, ez * texel), axes


def bulk(data, head, epoch):
    """Yield (path, flags, epoch, bulk epoch, imagery epoch, box) for up to four levels below head."""
    nodes, center, texels, imagery = [], None, (), None
    for number, value in fields(data):
        if number == 1:
            nodes.append(value)
        elif number == 2:
            epoch = dict(fields(value)).get(2, epoch)
        elif number == 3:
            center = struct.unpack("<3d", value)
        elif number == 4:
            texels = struct.unpack(f"<{len(value) // 4}f", value)
        elif number == 5:
            imagery = value
    for raw in nodes:
        node = dict(fields(raw))
        packed = node[1]
        depth = 1 + (packed & 3)
        packed >>= 2
        path = head
        for _ in range(depth):
            path += str(packed & 7)
            packed >>= 3
        texel = struct.unpack("<f", node[4])[0] if 4 in node else texels[depth - 1]
        box = node_box(node[3], center, texel) if 3 in node else None
        yield path, packed, node.get(2, epoch), node.get(5, epoch), node.get(7, imagery), box


def plan(client, area, level, workers, progress, *, allow_empty=False):
    """Nodes with data meeting the rectangle down to level, and the octants their children refine."""
    planet = dict(fields(client.get(BASE + "PlanetoidMetadata")))
    root = dict(fields(planet[1]))
    radius = struct.unpack("<f", planet[2])[0]
    target = search_box(area, radius)
    found, existing = [], set()
    pending, done = [("", root.get(5, root[2]))], 0
    progress("Finding 3D mesh nodes", done)
    while pending:
        batch, pending = pending, []
        addresses = [f"{BASE}BulkMetadata/pb=!1m2!1s{path}!2u{epoch}" for path, epoch in batch]
        for (head, epoch), data in zip(batch, fetch(client, addresses, workers, missing_ok=True)):
            done += 1
            progress("Finding 3D mesh nodes", done)
            if data is None:
                continue
            hit = set()
            for path, flags, node_epoch, bulk_epoch, imagery, box in sorted(bulk(data, head, epoch),
                                                                             key=lambda node: len(node[0])):
                if not flags & NODATA:
                    existing.add(path)
                # Parents come first; skip subtrees outside the area.
                if ((len(path) > len(head) + 1 and path[:-1] not in hit) or len(path) > level
                        or box is None or not overlaps(target, box)):
                    continue
                hit.add(path)
                if not flags & NODATA:
                    found.append((path, node_epoch, imagery if flags & USE_IMAGERY_EPOCH else None))
                if len(path) == len(head) + 4 and len(path) < level and not flags & LEAF:
                    pending.append((path, bulk_epoch))
    # Children replace their parent's octant, even outside the area where cropping removes them.
    # Skip parents with every octant replaced: they would add no triangles.
    nodes = []
    for path, epoch, imagery in sorted(found):
        mask = sum(1 << k for k in range(8) if f"{path}{k}" in existing) if len(path) < level else 0
        if mask != 255:
            nodes.append([path, epoch, imagery, mask])
    if not nodes and not allow_empty:
        raise ValueError("No Google Earth 3D data in this area.")
    return dict(mode="mesh", level=level, radius=radius, nodes=nodes, results=[])


def address(node):
    path, epoch, imagery, _ = node
    return (f"{BASE}NodeData/pb=!1m2!1s{path}!2u{epoch}!2e{JPEG}"
            + ("" if imagery is None else f"!3u{imagery}") + "!4b0")


def check(data):
    """Reject responses that are not a node with a 4 × 4 transform."""
    if len(first(data).get(1, b"")) != 128:
        raise ValueError("Unexpected 3D mesh node data.")


def counted(data, name):
    """Packed integers preceded by their count."""
    values = varints(data).astype(np.int64)
    if not values.size or values.size != values[0] + 1:
        raise ValueError(f"Unexpected 3D mesh {name}.")
    return values[1:]


def fitted(values, size):
    """Truncate or zero-pad to size."""
    return np.pad(values[:size], (0, max(0, size - len(values))))


def decode(data, mask, axes, radius, inside):
    """Yield (positions, texture coordinates, triangles, JPEG) per mesh, in local meters.

    Keeps the three base layers, without skirts, octants in mask, or triangles centered outside.
    """
    matrix, meshes = None, []
    for number, value in fields(data):
        if number == 1:
            matrix = struct.unpack("<16d", value)
        elif number == 2:
            meshes.append(value)
    if not meshes:
        return
    if matrix is None:
        raise ValueError("Unexpected 3D mesh node data.")
    # Fold the column-major globe transform into the move to east/north/up around the origin.
    rows = [[dot(axis, matrix[4 * j:4 * j + 3]) for j in range(4)] for axis in axes]
    rows[2][3] -= radius
    (a0, a1, a2, at), (b0, b1, b2, bt), (c0, c1, c2, ct) = rows
    for raw in meshes:
        mesh, offsets = {}, b""
        for number, value in fields(raw):
            if number == 10:
                offsets += value  # Four floats, packed or not.
            else:
                mesh.setdefault(number, value)
        vertices, uv = mesh.get(1, b""), mesh.get(7, b"")
        count = len(vertices) // 3
        if not count or len(vertices) != 3 * count or len(uv) != 4 + 4 * count:
            raise ValueError("Unexpected 3D mesh vertices.")
        texture = first(mesh.get(6, b""))
        if 1 not in texture or texture.get(2, JPEG) != JPEG:
            raise ValueError("Unexpected 3D mesh texture.")

        xs, ys, zs = np.cumsum(np.frombuffer(vertices, np.uint8).reshape(3, count), axis=1, dtype=np.uint8).astype(float)
        east = a0 * xs + a1 * ys + a2 * zs + at
        north = b0 * xs + b1 * ys + b2 * zs + bt
        up = c0 * xs + c1 * ys + c2 * zs + ct
        u_mod, v_mod = 1 + uv[0] + (uv[1] << 8), 1 + uv[2] + (uv[3] << 8)
        # Four byte planes: low u, low v, high u, high v.
        planes = np.frombuffer(uv, np.uint8, offset=4).reshape(4, count).astype(np.int64)
        if len(offsets) == 16:
            u_offset, v_offset, u_scale, v_scale = struct.unpack("<4f", offsets)
        else:
            u_offset, v_offset, u_scale, v_scale = 0.5, 0.5, 1 / u_mod, 1 / v_mod
        us = (np.cumsum(planes[0] | planes[2] << 8) % u_mod + u_offset) * u_scale
        vs = (np.cumsum(planes[1] | planes[3] << 8) % v_mod + v_offset) * v_scale

        # A zero index adds the next new vertex; others count back from the newest.
        values = counted(mesh.get(3, b""), "indices")
        new = values == 0
        strip = np.cumsum(new) - new - values
        if new.sum() > count or (strip.size and strip.min() < 0):
            raise ValueError("Unexpected 3D mesh indices.")
        # Runs cycle through eight octants per layer; trailing positions belong to no drawn layer.
        runs = counted(mesh.get(8, b""), "octants")
        layers = np.arange(runs.size)
        keep = fitted(np.repeat((layers < 24) & ~(mask >> (layers & 7) & 1).astype(bool), runs), strip.size)
        skirts = fitted(np.unpackbits(np.frombuffer(mesh.get(13, b""), np.uint8), bitorder="little"), strip.size)
        a, b, c = strip[:-2], strip[1:-1], strip[2:]
        drawn = (keep[2:].astype(bool) & ~skirts[:-2].astype(bool) & (a != b) & (b != c) & (c != a)
                 & inside((east[a] + east[b] + east[c]) / 3, (north[a] + north[b] + north[c]) / 3,
                          (up[a] + up[b] + up[c]) / 3))
        # Odd triangles in a strip are wound the other way.
        odd = np.arange(a.size) % 2 == 1
        triangles = np.stack((a, np.where(odd, c, b), np.where(odd, b, c)), axis=1)[drawn].ravel()
        if not triangles.size:
            continue
        # Vertices in order of first use.
        unique, first_use = np.unique(triangles, return_index=True)
        used = unique[np.argsort(first_use)]
        index = np.empty(count, np.int64)
        index[used] = np.arange(used.size)
        # glTF is Y-up: x east, y up, z south.
        positions = np.stack((east[used], up[used], -north[used]), axis=1).astype(np.float32)
        texcoords = np.stack((us[used], vs[used]), axis=1).astype(np.float32)
        indices = index[triangles].astype(np.uint16 if used.size <= 65536 else np.uint32)
        yield positions, texcoords, indices, texture[1]


def crop(area, axes, radius):
    """A test for local east/north/up points inside the rectangle's latitudes and longitudes."""
    south, west, north, east = (math.radians(value) for value in area)
    (ex, ey, ez), (nx, ny, nz), (ux, uy, uz) = axes
    low, high = math.sin(south), math.sin(north)
    west_x, west_y, east_x, east_y = -math.sin(west), math.cos(west), -math.sin(east), math.cos(east)

    def inside(e, n, u):
        r = radius + u
        x, y, z = ex * e + nx * n + ux * r, ey * e + ny * n + uy * r, ez * e + nz * n + uz * r
        length = np.sqrt(x * x + y * y + z * z)
        return ((low * length <= z) & (z <= high * length)
                & (x * west_x + y * west_y >= 0) & (x * east_x + y * east_y <= 0))

    return inside


def export(path, folder, stage, area, progress):
    """Join saved nodes into one glTF binary in a local east/north/up frame around the center.

    Each glTF mesh holds up to TILES_PER_MESH textured tiles from one octree block, named after
    its first tile: viewers import few objects, and Blender's import slows with many per mesh.
    """
    nodes, results = stage["nodes"], stage["results"]
    radius, block = stage["radius"], max(1, stage["level"] - 4)
    lat, lon = (area[0] + area[2]) / 2, (area[1] + area[3]) / 2
    axes = frame(lat, lon)
    inside = crop(area, axes, radius)
    # Keep parent octants whose child was unavailable, so gaps stay covered.
    missing = {node[0] for node, result in zip(nodes, results) if result.get("status") == "skipped"}
    saved = [(node, result["filename"]) for node, result in zip(nodes, results) if "filename" in result]
    gltf = dict(
        asset=dict(version="2.0", generator="Aleph", copyright="© Google"),
        scene=0, scenes=[dict(nodes=[])], nodes=[], meshes=[], materials=[], textures=[], images=[],
        accessors=[], bufferViews=[], samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
        extensionsUsed=["KHR_materials_unlit"],
    )
    progress("Building mesh.glb", 0, len(saved))
    with tempfile.TemporaryFile(dir=folder) as spool:
        def view(data, target=None):
            if isinstance(data, np.ndarray):
                data = data.astype(data.dtype.newbyteorder("<")).tobytes()
            gltf["bufferViews"].append(dict(buffer=0, byteOffset=spool.tell(), byteLength=len(data),
                                            **({"target": target} if target else {})))
            spool.write(data + b"\0" * (-len(data) % 4))
            if spool.tell() >= GLB_LIMIT:
                raise ValueError("The 3D mesh exceeds the 4 GiB glTF limit. Use a smaller area or lower mesh level.")
            return len(gltf["bufferViews"]) - 1

        # Nodes are sorted by path, so each block's nodes are consecutive.
        key = None
        for done, (node, filename) in enumerate(saved, 1):
            mask = node[3] & ~sum(1 << k for k in range(8) if f"{node[0]}{k}" in missing)
            data = (folder / filename).read_bytes()
            for part, (positions, texcoords, indices, jpeg) in enumerate(decode(data, mask, axes, radius, inside)):
                i, k = len(gltf["materials"]), len(gltf["accessors"])
                name = node[0] if not part else f"{node[0]}-{part}"
                gltf["accessors"] += [
                    dict(bufferView=view(positions, 34962), componentType=5126, count=len(positions),
                         type="VEC3", min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
                    dict(bufferView=view(texcoords, 34962), componentType=5126, count=len(texcoords),
                         type="VEC2"),
                    dict(bufferView=view(indices, 34963), componentType=5123 if indices.dtype == np.uint16 else 5125,
                         count=len(indices), type="SCALAR"),
                ]
                gltf["images"].append(dict(bufferView=view(jpeg), mimeType="image/jpeg"))
                gltf["textures"].append(dict(source=i, sampler=0))
                gltf["materials"].append(dict(name=name, doubleSided=True, extensions={"KHR_materials_unlit": {}},
                                              pbrMetallicRoughness=dict(baseColorTexture=dict(index=i),
                                                                        metallicFactor=0, roughnessFactor=1)))
                if node[0][:block] != key or len(gltf["meshes"][-1]["primitives"]) == TILES_PER_MESH:
                    key = node[0][:block]
                    gltf["meshes"].append(dict(name=name, primitives=[]))
                gltf["meshes"][-1]["primitives"].append(dict(
                    attributes=dict(POSITION=k, TEXCOORD_0=k + 1), indices=k + 2, material=i))
            progress("Building mesh.glb", done, len(saved))
        if not gltf["meshes"]:
            return
        gltf["nodes"] = [dict(name=mesh["name"], mesh=i) for i, mesh in enumerate(gltf["meshes"])]
        gltf["scenes"][0]["nodes"] = list(range(len(gltf["meshes"])))

        # Rest the lowest point on y = 0, so viewers orbit and zoom around the ground, not sea level.
        positions = [accessor for accessor in gltf["accessors"] if accessor["type"] == "VEC3"]
        base = min(accessor["min"][1] for accessor in positions)
        for accessor in positions:
            span = gltf["bufferViews"][accessor["bufferView"]]
            spool.seek(span["byteOffset"])
            values = np.frombuffer(spool.read(span["byteLength"]), "<f4").reshape(-1, 3).copy()
            # Subtract in double precision, then round once to float32.
            values[:, 1] = values[:, 1].astype(float) - base
            accessor["min"][1], accessor["max"][1] = float(values[:, 1].min()), float(values[:, 1].max())
            spool.seek(span["byteOffset"])
            spool.write(values.tobytes())
        spool.seek(0, 2)
        gltf["asset"]["extras"] = dict(
            latitude=lat, longitude=lon, base=base, radius=radius, bounds=list(area),
            frame="meters east (x), up (y), and south (z) of the center; y = 0 is the lowest point, base meters above sea level")
        size = spool.tell()
        gltf["buffers"] = [dict(byteLength=size)]
        header = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 4)
        total = 28 + len(header) + size
        if total >= GLB_LIMIT:
            raise ValueError("The 3D mesh exceeds the 4 GiB glTF limit. Use a smaller area or lower mesh level.")
        with atomic_path(path) as temporary, temporary.open("wb") as output:
            output.write(struct.pack("<III", 0x46546C67, 2, total))
            output.write(struct.pack("<II", len(header), 0x4E4F534A) + header)
            output.write(struct.pack("<II", size, 0x004E4942))
            spool.seek(0)
            shutil.copyfileobj(spool, output)
