"""S1–S2: read a capture into a Reference: frame, play polygon, welded mesh, rasters, OSM features, and DEM."""

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .. import terrain
from ..common import Client
from ..geo import MERCATOR_RADIUS
from ..layers import is_area, objects, rings
from .types import CELL, Feature, Frame, Raster, Reference

WELD = 1e-3  # Metres within which tile seam vertices are merged.
UP = 0.8  # Normal y above which a triangle faces up.
DEM_CELL = 2.0  # Metres per cell of the resampled terrain; zoom 14 terrain has about 3 m pixels here.
DEM_MARGIN = 16.0  # Metres of terrain kept around the play area for ground fitting at the edges.
SAMPLES = 4_000_000  # Cell samples per rasterization batch, which bounds memory.
RADIUS = 6_371_010.0  # Planet radius of Google's meshes, for captures without one.
COMPONENTS = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}


def build(folder, config, progress):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    lat, lon = np.asarray(manifest["polygon"], float).T
    progress("Reading mesh.glb")
    frame, positions, triangles = mesh(folder / "mesh.glb")
    # Without a mesh, place the frame as mesh.export would and rest y = 0 on the lowest terrain.
    meshless = frame is None
    if meshless:
        frame = Frame(float(lat.min() + lat.max()) / 2, float(lon.min() + lon.max()) / 2, 0.0, RADIUS)
    play = ccw(np.column_stack(frame.local(lat, lon)))
    x0, z0 = np.floor(play.min(axis=0) / CELL) * CELL
    cols, rows = np.ceil((play.max(axis=0) - (x0, z0)) / CELL).astype(int)
    progress("Measuring mesh", 0, len(triangles))
    top, bottom, up = rasterize(positions, triangles, x0, z0, cols, rows, CELL)
    progress("Measuring mesh", len(triangles), len(triangles))
    path = osm_path(folder, manifest["polygon"], config, progress)
    progress("Reading map.osm")
    features = read_osm(path, frame)
    dem = None
    if (folder / "terrain.tif").is_file():
        progress("Reading terrain.tif")
        dem = resample(folder / "terrain.tif", frame, play)
        if meshless and np.isfinite(dem.values).any():
            frame.base = float(np.nanmin(dem.values))
            dem.values -= frame.base
    return Reference(folder=folder, frame=frame, play=play, positions=positions, triangles=triangles,
                     top=Raster(top, x0, z0), bottom=Raster(bottom, x0, z0), up=Raster(up, x0, z0),
                     osm=features, dem=dem)


# Geometry helpers shared with the other stages.

def signed_area(points):
    """Shoelace area in (x, z): negative for rings counter-clockwise seen from above, because z points south."""
    x, z = np.asarray(points, float).T
    return float(np.dot(x, np.roll(z, -1)) - np.dot(np.roll(x, -1), z)) / 2


def ccw(points, clockwise=False):
    """The ring without repeated points, counter-clockwise seen from above (or clockwise)."""
    points = np.asarray(points, float).reshape(-1, 2)
    keep = np.any(points != np.roll(points, 1, axis=0), axis=1)
    points = points[keep] if keep.any() else points[:1]
    if (signed_area(points) > 0) != clockwise:
        points = points[::-1]
    return points


def inside(polygon, x, z):
    """Even-odd test of points against a ring of (x, z) pairs."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    result = np.zeros(np.broadcast(x, z).shape, bool)
    for (ax, az), (bx, bz) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if az != bz:
            result ^= ((az > z) != (bz > z)) & (x < ax + (z - az) * (bx - ax) / (bz - az))
    return result


# Mesh.

def mesh(path):
    """Frame, welded positions, and triangles of a capture's mesh.glb; no frame and empty arrays without one."""
    empty = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    if not path.is_file():
        return None, *empty
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path.name} is not a glTF binary.")
    length, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + length])
    binary = memoryview(data)[28 + length:] if len(data) > 20 + length else memoryview(b"")
    extras = gltf["asset"].get("extras", {})
    frame = Frame(extras["latitude"], extras["longitude"], extras["base"], extras["radius"])

    def array(index):
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        values = np.frombuffer(binary, COMPONENTS[accessor["componentType"]], accessor["count"] * width, start)
        return values.reshape(-1, width) if width > 1 else values

    positions, triangles, count = [], [], 0
    for item in gltf.get("meshes", []):
        for primitive in item["primitives"]:
            if primitive.get("mode", 4) != 4 or "indices" not in primitive:
                continue
            p = array(primitive["attributes"]["POSITION"])
            positions.append(p.astype(np.float32))
            triangles.append(array(primitive["indices"]).astype(np.int64).reshape(-1, 3) + count)
            count += len(p)
    if not positions:
        return frame, *empty
    positions, triangles = weld(np.concatenate(positions), np.concatenate(triangles))
    return frame, positions, triangles


def weld(positions, triangles):
    """Merge vertices within WELD and drop triangles that collapse."""
    q = np.round(positions.astype(np.float64) / WELD).astype(np.int64)
    q -= q.min(axis=0)
    span = q.max(axis=0) + 1
    if float(span[0]) * float(span[1]) * float(span[2]) < 2**62:
        key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    else:
        _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    triangles = inverse.ravel()[triangles]
    a, b, c = triangles.T
    triangles = triangles[(a != b) & (b != c) & (c != a)]
    welded = positions[first]
    v = welded[triangles].astype(np.float64)
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    return welded, triangles[area > 1e-8].astype(np.int32)


def rasterize(positions, triangles, x0, z0, cols, rows, cell):
    """Highest and lowest surface y and up-facing area fraction per cell, NaN where the mesh is absent.

    Cell centres inside a triangle's plan sample its plane. Vertices always count, and triangles too steep or small to
    hold a centre add their centroid and edge midpoints, so walls and slivers still mark their cells.
    """
    top = np.full(rows * cols, -np.inf)
    bottom = np.full(rows * cols, np.inf)
    area = np.zeros(rows * cols)
    upward = np.zeros(rows * cols)
    if not len(triangles):
        nan = np.full((rows, cols), np.nan)
        return nan, nan.copy(), nan.copy()

    def cells(x, z):
        col, row = np.floor((x - x0) / cell).astype(np.int64), np.floor((z - z0) / cell).astype(np.int64)
        valid = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        return row * cols + col, valid

    def mark(x, z, y):
        index, valid = cells(x, z)
        np.maximum.at(top, index[valid], y[valid])
        np.minimum.at(bottom, index[valid], y[valid])

    p = positions.astype(np.float64)
    mark(p[:, 0], p[:, 2], p[:, 1])
    v = p[triangles]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = v[:, 0].T, v[:, 1].T, v[:, 2].T
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    size = np.linalg.norm(normal, axis=1)
    weight = size / 2
    up = normal[:, 1] > UP * size
    # First vertex and edge vectors, gathered per batch; det is the doubled plan area, and steep triangles get no
    # centre samples.
    edges = np.stack((ax, az, ay, bx - ax, bz - az, by - ay, cx - ax, cz - az, cy - ay), axis=1)
    det = edges[:, 3] * edges[:, 7] - edges[:, 6] * edges[:, 4]
    flat = np.abs(det) > 1e-9

    # Range of cell centres inside each triangle's bounding box.
    lo = lambda m, origin, n: np.clip(np.ceil((m - origin) / cell - 0.5), 0, n).astype(np.int64)
    hi = lambda m, origin, n: np.clip(np.floor((m - origin) / cell - 0.5), -1, n - 1).astype(np.int64)
    c0, c1 = lo(np.minimum(np.minimum(ax, bx), cx), x0, cols), hi(np.maximum(np.maximum(ax, bx), cx), x0, cols)
    r0, r1 = lo(np.minimum(np.minimum(az, bz), cz), z0, rows), hi(np.maximum(np.maximum(az, bz), cz), z0, rows)
    w, h = np.where(flat, c1 - c0 + 1, 0).clip(0), np.where(flat, r1 - r0 + 1, 0).clip(0)
    samples = []  # (triangle, flat index, y) per batch.
    candidates = np.flatnonzero(w * h > 0)
    shapes = (w * 4096 + h)[candidates]
    order = np.argsort(shapes, kind="stable")
    candidates, shapes = candidates[order], shapes[order]
    starts = np.flatnonzero(np.diff(shapes, prepend=-1))
    for start, end in zip(starts, np.r_[starts[1:], len(shapes)]):
        wide, tall = divmod(int(shapes[start]), 4096)
        step = max(1, SAMPLES // (wide * tall))
        ox, oz = np.meshgrid(np.arange(wide), np.arange(tall))
        ox, oz = ox.ravel(), oz.ravel()
        for s in range(start, end, step):
            t = candidates[s:min(end, s + step)]
            col, row = c0[t, None] + ox, r0[t, None] + oz
            px, pz = x0 + (col + 0.5) * cell, z0 + (row + 0.5) * cell
            qx, qz, qy, ux, uz, uy, vx, vz, vy = edges[t].T[:, :, None]
            dx, dz, d = px - qx, pz - qz, det[t, None]
            l1 = (dx * vz - vx * dz) / d
            l2 = (ux * dz - dx * uz) / d
            hit = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
            y = qy + l1 * uy + l2 * vy
            samples.append((np.broadcast_to(t[:, None], hit.shape)[hit], (row * cols + col)[hit], y[hit]))
    if samples:
        owner, spot, height = (np.concatenate(values) for values in zip(*samples))
    else:
        owner, spot, height = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    hits = np.bincount(owner, minlength=len(triangles))

    # Triangles holding no centre: centroid and edge midpoints.
    missing = np.flatnonzero(hits == 0)
    m = v[missing]
    extra = np.concatenate([m.mean(axis=1), (m[:, 0] + m[:, 1]) / 2, (m[:, 1] + m[:, 2]) / 2, (m[:, 2] + m[:, 0]) / 2])
    mark(extra[:, 0], extra[:, 2], extra[:, 1])
    index, valid = cells(m[:, :, 0].mean(axis=1), m[:, :, 2].mean(axis=1))
    np.add.at(area, index[valid], weight[missing][valid])
    np.add.at(upward, index[valid], (weight * up)[missing][valid])

    np.maximum.at(top, spot, height)
    np.minimum.at(bottom, spot, height)
    share = weight[owner] / hits[owner]
    area += np.bincount(spot, share, rows * cols)
    upward += np.bincount(spot, share * up[owner], rows * cols)
    top[~np.isfinite(top)] = np.nan
    bottom[~np.isfinite(bottom)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(area > 0, upward / area, np.nan)
    return top.reshape(rows, cols), bottom.reshape(rows, cols), fraction.reshape(rows, cols)


# OSM.

def osm_path(folder, polygon, config, progress):
    """game/map.osm, else the capture's own map.osm, else a fresh extract saved as game/map.osm."""
    path = folder / "game" / "map.osm"
    for candidate in (path, folder / "map.osm"):
        if candidate.is_file():
            return candidate
    client = Client(cache_dir=config.get("cache_dir") or Client().cache_dir)
    client.maps.export(polygon, path, progress)
    return path


def read_osm(path, frame):
    """Features with every tag, in local coordinates."""
    ids, lats, lons, node_tags, ways, relations = [], [], [], {}, {}, []
    for element in objects(Path(path)):
        identity = int(element.attrib["id"])
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}
        if element.tag == "node":
            ids.append(identity)
            lats.append(float(element.attrib["lat"]))
            lons.append(float(element.attrib["lon"]))
            if tags:
                node_tags[identity] = tags
        elif element.tag == "way":
            ways[identity] = [int(nd.attrib["ref"]) for nd in element.findall("nd")], tags
        elif tags.get("type") == "multipolygon":
            members = [(int(m.attrib["ref"]), m.get("role", "")) for m in element.findall("member") if m.get("type") == "way"]
            relations.append((identity, tags, members))
    x, z = frame.local(lats, lons)
    xz = np.column_stack((x, z)).reshape(-1, 2)
    nodes = dict(zip(ids, range(len(ids))))

    features, consumed = [], set()
    for identity, tags, members in relations:
        outers = [ref for ref, role in members if role in ("", "outer")]
        meaningful = {key for key in tags if key != "type"}
        # Older multipolygons carry their tags on a single outer way.
        if not meaningful and len(outers) == 1 and outers[0] in ways:
            tags = dict(ways[outers[0]][1], **tags)
            consumed.add(outers[0])
        kind = classify(tags, True)
        if kind is None:
            continue
        shells, used = [], set()
        for ring, members_used in rings(outers, ways, nodes):
            shells.append([ccw(xz[ring[:-1]]), []])
            used.update(members_used)
        for ring, members_used in rings([ref for ref, role in members if role == "inner"], ways, nodes):
            hole = ccw(xz[ring[:-1]], clockwise=True)
            owners = [shell for shell in shells if inside(shell[0], *hole[0])]
            if owners:
                min(owners, key=lambda shell: -signed_area(shell[0]))[1].append(hole)
                used.update(members_used)
        for n, (shell, holes) in enumerate(shells):
            if len(shell) >= 3:
                features.append(Feature(f"relation/{identity}" + (f":{n + 1}" if n else ""), kind, tags, shell, holes))
        # Members tagged like the relation are the same object; others stand on their own too.
        consumed.update(ref for ref in used if all(tags.get(k) == v for k, v in ways[ref][1].items() if k != "type"))

    for identity, (refs, tags) in ways.items():
        if identity in consumed or not tags or len(refs) < 2 or any(ref not in nodes for ref in refs):
            continue
        closed = len(refs) >= 4 and refs[0] == refs[-1]
        kind = classify(tags, closed)
        if kind is None:
            continue
        if kind == "line":
            points = xz[[nodes[ref] for ref in refs]]
            points = points[np.r_[True, np.any(points[1:] != points[:-1], axis=1)]]
            if len(points) >= 2:
                features.append(Feature(f"way/{identity}", kind, tags, points))
        elif len(ring := ccw(xz[[nodes[ref] for ref in refs[:-1]]])) >= 3:
            features.append(Feature(f"way/{identity}", kind, tags, ring))

    for identity, tags in node_tags.items():
        features.append(Feature(f"node/{identity}", "node", tags, xz[nodes[identity]][None]))
    return features


def classify(tags, closed):
    """Feature kind of a way or multipolygon, or None for untagged or unclosed areas."""
    if tags.get("building", "no") != "no" or tags.get("building:part", "no") != "no":
        return "building" if closed else None
    # Closed walls, fences, and hedges stay lines unless tagged as areas.
    if closed and is_area(tags) and ("barrier" not in tags or tags.get("area") == "yes"):
        return "area"
    return "line" if any(key != "type" for key in tags) else None


# Terrain.

def resample(path, frame, play):
    """terrain.tif bilinearly sampled on a local DEM_CELL grid around the play area, in the mesh's y datum."""
    with path.open("rb") as stream:
        source = terrain.TIFF(stream)
        heights = decode(source)
        scale = source.values(33550)[0]
        tie = source.values(33922)
        point = source.keys.get(1025) == 2
    x0, z0 = np.floor((play.min(axis=0) - DEM_MARGIN) / DEM_CELL) * DEM_CELL
    x1, z1 = np.ceil((play.max(axis=0) + DEM_MARGIN) / DEM_CELL) * DEM_CELL
    grid = Raster(np.zeros((int(round((z1 - z0) / DEM_CELL)), int(round((x1 - x0) / DEM_CELL)))), x0, z0, DEM_CELL)
    x, z = grid.centres()
    lat, lon = degrees(frame, x, -z)
    mx = MERCATOR_RADIUS * np.radians(lon)
    my = MERCATOR_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    # Pixel centres sit half a pixel in from the tie point unless the raster marks points.
    shift = 0 if point else 0.5
    col, row = (mx - tie[3]) / scale - shift, (tie[4] - my) / scale - shift
    grid.values = bilinear(heights, col, row) - frame.base
    return grid


def degrees(frame, east, north):
    """Latitude and longitude of local ground points: Frame.local's inverse on the sphere."""
    (ex, ey, ez), (nx, ny, nz) = frame.axes()
    la, lo = math.radians(frame.latitude), math.radians(frame.longitude)
    ux, uy, uz = math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)
    e, n = east / frame.radius, north / frame.radius
    u = np.sqrt(np.maximum(0, 1 - e * e - n * n))
    px, py, pz = ex * e + nx * n + ux * u, ey * e + ny * n + uy * u, ez * e + nz * n + uz * u
    return np.degrees(np.arctan2(pz, np.hypot(px, py))), np.degrees(np.arctan2(py, px))


def bilinear(values, col, row):
    """Samples at fractional pixel positions, NaN outside or next to missing heights."""
    rows, cols = values.shape
    c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    valid = (c0 >= 0) & (r0 >= 0) & (c0 < cols - 1) & (r0 < rows - 1)
    c0, r0 = np.where(valid, c0, 0), np.where(valid, r0, 0)
    tx, tz = col - c0, row - r0
    result = ((values[r0, c0] * (1 - tx) + values[r0, c0 + 1] * tx) * (1 - tz)
              + (values[r0 + 1, c0] * (1 - tx) + values[r0 + 1, c0 + 1] * tx) * tz)
    return np.where(valid, result, np.nan)


def decode(source):
    """All heights of a tiled Int16 or Float32 terrain TIFF, NaN where missing."""
    width, height, size = source.value(256), source.value(257), source.value(322)
    across = -(-width // size)
    values = np.full((-(-height // size) * size, across * size), np.nan, np.float32)
    predictor = source.value(317, 1)
    for i, (offset, length) in enumerate(zip(source.values(324), source.values(325))):
        if not length:
            continue
        data = source.read(offset, length)
        if source.value(259) != 1:
            data = zlib.decompress(data)
        if source.float and predictor == 3:
            # Rows hold four byte planes, most significant first, with differences running on across planes.
            planes = np.cumsum(np.frombuffer(data, np.uint8).reshape(size, 4 * size), axis=1, dtype=np.uint8)
            block = planes.reshape(size, 4, size).transpose(0, 2, 1).copy().view(">f4").reshape(size, size)
        elif source.float:
            block = np.frombuffer(data, source.order + "f4").reshape(size, size)
        else:
            block = np.frombuffer(data, source.order + "u2").reshape(size, size)
            if predictor == 2:
                block = np.cumsum(block, axis=1, dtype=np.uint16)
            block = block.astype(np.int16)
        y, x = divmod(i, across)
        values[y * size:(y + 1) * size, x * size:(x + 1) * size] = block
    values = values[:height, :width]
    if 42113 in source.tags:
        values[values == float(b"".join(source.values(42113)).rstrip(b"\0"))] = np.nan
    values[~np.isfinite(values) | (np.abs(values) > 1e5)] = np.nan
    return values
