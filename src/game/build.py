"""S7–S8: run the game world stages and package chunks, scene.json, report.json, and the library for the viewer."""

import json
import math
import shutil
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

from ..common import RequestError, now, write_json
from . import buildings, chunks, gltf, ground, library, load, scatter
from .gltf import Node
from .types import GROUND_CELL, SURFACES

CHUNK = 64.0  # Default chunk size in metres.
LOD_DISTANCES = [50, 150, 400]
EYE = 1.7  # Spawn height above the ground.
SUN = 200.0, 35.0  # Azimuth and elevation used when the capture's sun was low.
LOW_SUN = 10.0
REPO = Path(__file__).resolve().parents[2]
OSM_CREDIT = "© OpenStreetMap contributors"


def command(args, progress):
    """The 'game' command: build FOLDER/game from FOLDER/game.toml and the command-line options."""
    folder = args.folder.expanduser().resolve()
    config = settings(folder, args.style, args.quality, args.chunk, args.cache_dir)
    report = run(folder, config, progress)
    return dict(command="game", status="complete", folder=str(folder / "game"), scene=str(folder / "game" / "scene.json"),
                chunks=report["counts"]["chunks"], warnings=report["warnings"])


def settings(folder, style=None, quality=None, chunk=None, cache_dir=None):
    """FOLDER/game.toml with the given options on top, as the stages expect it."""
    if not (folder / "manifest.json").is_file():
        raise ValueError(f"{folder} is not a capture folder: manifest.json is missing.")
    path = folder / "game.toml"
    config = tomllib.loads(path.read_text()) if path.is_file() else {}
    config["style"] = style if style and style != "auto" else config.get("style", "auto")
    config["quality"] = quality or config.get("quality", "high")
    if config["quality"] not in ("draft", "high"):
        raise ValueError("Quality must be draft or high.")
    config["chunk_size"] = float(chunk or config.get("chunk_size", CHUNK))
    config["cache_dir"] = cache_dir
    if config["chunk_size"] <= 0:
        raise ValueError("--chunk: use a positive number of metres.")
    return config


def viewer():
    """The command that starts aleph_view, before the game folder; needs cargo and the viewer workspace."""
    manifest = REPO / "viewer" / "Cargo.toml"
    if not manifest.is_file():
        raise RequestError("viewer_missing", f"The viewer workspace is missing: {manifest} not found.")
    if not shutil.which("cargo"):
        raise RequestError("viewer_missing", "cargo is not installed; get Rust from https://rustup.rs to run the viewer.")
    # The dev profile optimizes dependencies fully and builds in a fraction of the time of --release.
    return ["cargo", "run", "--manifest-path", str(manifest), "-p", "aleph_view", "--"]


def view(game):
    """Walk a built game world in aleph_view."""
    return subprocess.run([*viewer(), str(game)]).returncode


def run(folder, config, progress):
    """Build folder/game and return the report."""
    folder, timings, warnings = Path(folder), {}, []
    game = folder / "game"

    def stage(name, function, *args):
        started = time.monotonic()
        result = function(*args)
        timings[name] = round(time.monotonic() - started, 2)
        return result

    if config.get("cache_dir"):
        library.root(config["cache_dir"])
    reference = stage("load", load.build, folder, config, progress)
    if not len(reference.triangles):
        warnings.append("No mesh.glb: heights come from OSM tags and terrain only.")
    if reference.dem is None:
        warnings.append("No terrain.tif: ground holes are filled without a DEM.")
    terrain = stage("ground", ground.build, reference, config, progress)
    models = stage("buildings", buildings.build, reference, terrain, config, progress)
    instances = stage("scatter", scatter.build, reference, terrain, models, config, progress)
    warnings += [str(w) for w in terrain.data.get("warnings", [])]

    started = time.monotonic()
    size = float(config.get("chunk_size", CHUNK))
    cells = chunks.grid(reference.play, size)
    owners = chunks.assign(cells, [model.center for model in models])
    placed = chunks.assign(cells, [(i.position[0], i.position[2]) for i in instances])
    assets = {name: library.asset(name) for name in sorted({i.asset for i in instances})}
    materials, entries, triangles, draws = set(), [], {}, {}
    stale = {path.name for path in (game / "chunks").iterdir()} if (game / "chunks").is_dir() else set()
    for n, cell in enumerate(cells):
        progress("Writing chunks", n, len(cells))
        own = [model for model, k in zip(models, owners) if k == n]
        here = [instance for instance, k in zip(instances, placed) if k == n]
        entry, triangles[cell.id], draws[cell.id] = write_chunk(game, cell, terrain, own, here, assets, materials)
        entries.append(entry)
        stale.discard(cell.id)
    progress("Writing chunks", len(cells), len(cells))
    for name in stale:
        shutil.rmtree(game / "chunks" / name, ignore_errors=True)
    timings["chunks"] = round(time.monotonic() - started, 2)

    progress("Installing library")
    started = time.monotonic()
    library.install(sorted(materials | set(assets)), game / "library")
    timings["library"] = round(time.monotonic() - started, 2)

    frame = reference.frame
    manifest = json.loads((folder / "manifest.json").read_text())
    azimuth, elevation = sun(frame.latitude, frame.longitude, manifest.get("started_at"))
    if elevation < LOW_SUN:
        azimuth, elevation = SUN
    position, yaw = spawn(reference, terrain)
    scene = dict(
        format="aleph-game", version=1,
        origin=dict(latitude=frame.latitude, longitude=frame.longitude, base=frame.base),
        attribution=attribution(materials, assets),
        chunk_size=size, lod_distances=LOD_DISTANCES,
        bounds=np.round(reference.play, 3).tolist(),
        spawn=dict(position=position, yaw=yaw),
        sun=dict(azimuth=round(azimuth, 1), elevation=round(elevation, 1)),
        chunks=entries,
        assets={name: asset_entry(asset) for name, asset in assets.items()},
    )
    write_json(game / "scene.json", plain(scene))

    kinds = {}
    for feature in reference.osm:
        kinds[feature.kind] = kinds.get(feature.kind, 0) + 1
    budget = 3_000_000
    total = {lod: sum(counts[lod] for counts in triangles.values()) for lod in ("lod0", "lod1", "lod2")}
    primitives = {lod: sum(counts[lod] for counts in draws.values()) for lod in ("lod0", "lod1", "lod2")}
    if total["lod0"] > budget:
        warnings.append(f"LOD0 holds {total['lod0']:,} triangles, above the {budget:,} visible budget.")
    report = dict(
        format="aleph-game-report", version=1, built_at=now(), quality=config.get("quality"), style=config.get("style"),
        footprints=buildings.accounting(reference, models, config),
        counts=dict(chunks=len(cells), models=len(models),
                    buildings=sum(m.name.startswith("building/") for m in models),
                    landmarks=sum(m.name.startswith("landmark/") for m in models),
                    instances=len(instances), assets={name: sum(i.asset == name for i in instances) for name in assets},
                    osm=kinds, mesh_triangles=len(reference.triangles)),
        triangles=dict(total=total, chunks=triangles),
        primitives=dict(total=primitives, chunks=draws),
        materials=sorted(materials), timings=timings, warnings=warnings,
    )
    write_json(game / "report.json", plain(report))
    return report


def write_chunk(game, cell, terrain, models, instances, assets, materials):
    """Write one chunk's LOD files and heightfield; return its scene.json entry, and triangles and primitives per LOD."""
    folder = game / "chunks" / cell.id
    center = cell.center
    x0, z0, x1, z1 = cell.rect
    counts, draws, low, high = {}, {}, [x0, math.inf, z0], [x1, -math.inf, z1]
    for lod in range(3):
        nodes = [Node("ground", [chunks.shift(part, center) for part in ground.chunk(terrain, cell.rect, lod)],
                      colliders=[] if lod else [chunks.shift_collider(c, center) for c in ground.colliders(terrain, cell.rect)])]
        for model in models:
            parts = model.lods[min(lod, len(model.lods) - 1)] if model.lods else []
            nodes.append(Node(model.name, [chunks.shift(part, center) for part in parts], extras=model.extras or None,
                              colliders=[] if lod else [chunks.shift_collider(c, center) for c in model.colliders]))
        parts = [part for node in nodes for part in node.parts if len(part.indices)]
        materials.update(part.material for part in parts)
        counts[f"lod{lod}"] = sum(len(part.indices) // 3 for part in parts)
        draws[f"lod{lod}"] = len(parts)
        if lod == 0:
            for part in parts:
                positions = np.asarray(part.positions, float).reshape(-1, 3) + (center[0], 0, center[1])
                low = np.minimum(low, positions.min(axis=0)).tolist()
                high = np.maximum(high, positions.max(axis=0)).tolist()
        gltf.write(folder / f"lod{lod}.glb", nodes, library.material, game / "library")

    heights = chunks.heights(terrain.height, cell.rect)
    chunks.write_heights(folder / "ground.height", heights)
    low[1], high[1] = min(low[1], float(heights.min())), max(high[1], float(heights.max()))
    for instance in instances:
        x, y, z = (float(v) for v in instance.position)
        reach = assets[instance.asset].radius * instance.scale
        low = np.minimum(low, (x - reach, y, z - reach)).tolist()
        high = np.maximum(high, (x + reach, y + assets[instance.asset].height * instance.scale, z + reach)).tolist()
    entry = dict(
        id=cell.id, center=[center[0], 0.0, center[1]],
        min=[round(v, 3) for v in low], max=[round(v, 3) for v in high],
        lods=[f"chunks/{cell.id}/lod{lod}.glb" for lod in range(3)],
        ground=dict(file=f"chunks/{cell.id}/ground.height", size=[heights.shape[1], heights.shape[0]],
                    spacing=GROUND_CELL, offset=[x0, z0]),
        instances=[dict(asset=i.asset, position=[round(float(v), 3) for v in i.position], yaw=round(float(i.yaw), 4),
                        scale=round(float(i.scale), 4), tint=[round(float(v), 4) for v in i.tint]) for i in instances],
    )
    return entry, counts, draws


def asset_entry(asset):
    return dict(lods=[f"library/{path}" for path in asset.lods],
                colliders=[dict(c.params, shape=c.shape, **({} if c.vertices is None else
                                                            dict(vertices=np.asarray(c.vertices).tolist())))
                           for c in asset.colliders],
                height=asset.height, radius=asset.radius)


def attribution(materials, assets):
    """OSM's credit plus one per provider and licence that library materials and assets record in their extras."""
    credits = [OSM_CREDIT]
    sources = [library.material(name) for name in sorted(materials)] + list(assets.values())
    for source in sources:
        extras = getattr(source, "extras", None) or {}
        found = extras.get("attribution") or extras.get("credit")
        if not found and extras.get("source") and not extras["source"].startswith("Aleph"):
            # "ambientCG Asphalt010" credits ambientCG: drop trailing words naming the item.
            words = extras["source"].split()
            while len(words) > 1 and any(c.isdigit() for c in words[-1]):
                words.pop()
            found = " ".join(words) + (f" ({extras['license']})" if extras.get("license") else "")
        for credit in [found] if isinstance(found, str) else found or []:
            if credit not in credits:
                credits.append(credit)
    return credits


def spawn(reference, terrain):
    """A ground point near the play area's centre, outside buildings and water, facing north."""
    play = reference.play
    footprints = [(f.points, f.holes) for f in reference.osm if f.kind == "building"]
    blocked = {SURFACES.index(name) for name in ("building", "water") if name in SURFACES}
    cx, cz = (play.min(axis=0) + play.max(axis=0)) / 2
    reach = float(np.max(play.max(axis=0) - play.min(axis=0)))
    for radius in np.arange(0, reach / 2, 2.0):
        steps = max(1, int(2 * math.pi * radius / 2))
        for angle in np.arange(steps) * 2 * math.pi / steps:
            x, z = cx + radius * math.cos(angle), cz + radius * math.sin(angle)
            if not load.inside(play, x, z) or any(load.inside(ring, x, z) and not any(load.inside(h, x, z) for h in holes)
                                                  for ring, holes in footprints):
                continue
            row, col = terrain.surface.index(x, z)
            code = float(terrain.surface.values[row, col])
            y = float(terrain.height.sample(x, z))
            if (math.isfinite(code) and int(code) in blocked) or not math.isfinite(y):
                continue
            return [round(x, 3), round(y + EYE, 3), round(z, 3)], 0.0
    y = float(terrain.height.sample(cx, cz))
    return [round(float(cx), 3), round((y if math.isfinite(y) else 0.0) + EYE, 3), round(float(cz), 3)], 0.0


def sun(latitude, longitude, when):
    """Solar azimuth (degrees clockwise from north) and elevation at a UTC ISO time, by NOAA's approximation."""
    if not when:
        return SUN
    moment = datetime.fromisoformat(when)
    if moment.utcoffset():
        moment = (moment - moment.utcoffset()).replace(tzinfo=None)
    hours = moment.hour + moment.minute / 60 + moment.second / 3600
    g = 2 * math.pi / 365 * (moment.timetuple().tm_yday - 1 + (hours - 12) / 24)
    equation = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                         - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    declination = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
                   + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    angle = math.radians((hours * 60 + equation + 4 * longitude) / 4 - 180)
    lat = math.radians(latitude)
    cosine = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(angle)
    elevation = 90 - math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    azimuth = math.degrees(math.atan2(math.sin(angle), math.cos(angle) * math.sin(lat)
                                      - math.tan(declination) * math.cos(lat))) + 180
    return azimuth % 360, elevation


def plain(value):
    """JSON-ready copies of numpy scalars and arrays inside dicts and lists."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
