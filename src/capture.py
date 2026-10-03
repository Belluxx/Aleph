"""Download stages and keep their files and checkpoint together."""

import json
import math
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import satellite, streetview, terrain
from .common import MissingImagery, contained, now, number, open_image, positive, write_bytes, write_json
from .geo import bounds, collection, coordinate, distance, feature, grid, pixel, tile_ring, tiles

SOURCES = ("streetview", "satellite", "osm")
FORMAT_VERSION = 3
DEFAULT_OPTIONS = dict(
    include=list(SOURCES), step=30, fov=75, depth="roads", streetview_format="jpg",
    full_sphere=False, sphere_zoom=3,
    delay=0, satellite_zoom=18, satellite_format="jpg", terrain_zoom=14,
)

# Average connection speed / hardware timings, just to give an idea to the user, not precise
REQUEST_SECONDS = dict(metadata=0.17, photo=0.46, sphere_tile=0.27, satellite_tile=0.25, terrain_tile=1.39)
SPHERE_TILES = (1, 2, 8, 32, 128, 512)
OSM_EXPORT_SECONDS = 10

OSM_NOTES = {
    "map": "OSM XML from Geofabrik with original tags and topology; coordinates are WGS84. Contributor names, IDs and changeset IDs are omitted by Geofabrik.",
    "selection": "Selected ways are complete. Selected multipolygons include all available outer boundaries and holes, but members missing from the regional file remain unresolved. Other relations may also be incomplete. Objects may extend outside the rectangle; crossings without inside nodes and enclosing polygons may be absent.",
    "terrain": "Float32 heights in meters, EPSG:3857, without resampling. Full edge tiles extend beyond the rectangle. Tiles are checkpointed individually; terrain.tif is built after all terrain tiles are saved.",
    "quality": "Terrain resolution, dates, accuracy and vertical reference vary by source. Higher zoom does not guarantee more detail.",
}

README = b"""Aleph capture

streetview/photos/ contains photos; streetview/ holds GeoJSON and plan.svg.
satellite/patches/ contains patches; satellite/ holds GeoJSON and plan.svg.
Open satellite.tif for satellite imagery and map.osm for native map data.
Photo headings are clockwise from north; left/right follow OSM node order.
GeoJSON files record paths, photo locations, or satellite patch footprints.
satellite.tif is a lossless RGBA Cloud Optimized GeoTIFF in EPSG:3857.
It is north-up, cropped to enclosing pixels, with internal overviews.
satellite.png contains the same full-resolution pixels for easy viewing.
Transparent pixels mark missing imagery or unfinished downloads.
terrain.tif is Float32 meters in EPSG:3857, without resampling.
Keep terrain/tiles/ for resume and offline export.
Filenames in the manifest and GeoJSON are relative to this run folder.
See manifest.json for settings, sources, imagery dates, and coverage notes.

Keep this folder together. Continue with: alephgeo capture resume FOLDER
Rebuild images and metadata offline with: alephgeo capture export FOLDER
"""


def settings(values=None):
    """Validate settings shared by the CLI and dashboard."""
    values = {} if values is None else values
    if not isinstance(values, dict) or values.keys() - DEFAULT_OPTIONS.keys():
        raise ValueError("Unknown capture settings.")
    options = {**DEFAULT_OPTIONS, **values}
    include = options["include"]
    if not isinstance(include, list) or not include or any(source not in SOURCES for source in include):
        raise ValueError("Choose one or more sources: streetview, satellite, osm.")
    options["include"] = [source for source in SOURCES if source in include]
    positive(options["step"], "step")
    number(options["delay"], "delay", 0)
    for key, low, high in (("fov", 5, 175), ("sphere_zoom", 0, 5), ("satellite_zoom", 1, 21), ("terrain_zoom", 1, 14)):
        options[key] = number(options[key], key, low, high, whole=True)
    if type(options["full_sphere"]) is not bool:
        raise ValueError("full_sphere must be true or false.")
    if options["depth"] not in ("main", "roads", "all"):
        raise ValueError("Choose main, roads, or all for depth.")
    for key in ("streetview_format", "satellite_format"):
        if options[key] not in ("jpg", "png"):
            raise ValueError(f"{key}: choose jpg or png.")
    return options


def create_folder(run, parent):
    """Persist an already prepared capture without repeating planning requests."""
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(tempfile.mkdtemp(prefix=f"aleph-{stamp}-", dir=parent))
    write_json(folder / "manifest.json", run)
    return folder


def plan(client, area, options, progress):
    options = settings(options)
    stages = []
    if "streetview" in options["include"]:
        stages.append(streetview.plan(client, area, options, progress, allow_empty=len(options["include"]) > 1))
    if "satellite" in options["include"]:
        stages.append(dict(mode="satellite", grid=grid(area, options["satellite_zoom"]), results=[]))
    if "osm" in options["include"]:
        stages.append(dict(mode="osm", grid=grid(area, options["terrain_zoom"]), results=[], notes=OSM_NOTES))
    return dict(format="aleph-python", version=FORMAT_VERSION, bounds=area, options=options,
                started_at=now(), state="planned", stages=stages)


def total(stage):
    if stage["mode"] == "streetview":
        return len(stage["samples"]) * (1 if stage["full_sphere"] else 2)
    return stage["grid"]["rows"] * stage["grid"]["columns"] + (stage["mode"] == "osm")


def estimate(run):
    """Estimate remaining work from measured request costs and typical sphere sizes.

    Excludes planning, initial regional PBF downloads, and final merged exports.
    Partly downloaded spheres are counted in full; their cached tiles may save time.
    """
    counts = dict(streetview_photos=0, streetview_stops=0, satellite_tiles=0, terrain_tiles=0, osm_maps=0)
    requests = dict.fromkeys(REQUEST_SECONDS, 0)
    for stage in run["stages"]:
        remaining = total(stage) - len(stage["results"])
        if stage["mode"] == "streetview":
            sphere = stage["full_sphere"]
            stops = len(stage["samples"]) - len(stage["results"]) // (1 if sphere else 2)
            counts.update(streetview_photos=remaining, streetview_stops=stops)
            requests["metadata"] = stops  # Metadata is fetched again when resuming a run.
            if sphere:
                requests["sphere_tile"] = remaining * SPHERE_TILES[run["options"]["sphere_zoom"]]
            else:
                requests["photo"] = remaining
        elif stage["mode"] == "satellite":
            counts["satellite_tiles"] = requests["satellite_tile"] = remaining
        else:
            counts["osm_maps"] = int(not any(item["filename"] == "map.osm" for item in stage["results"]))
            counts["terrain_tiles"] = requests["terrain_tile"] = remaining - counts["osm_maps"]
    seconds = sum(count * REQUEST_SECONDS[kind] for kind, count in requests.items())
    seconds += counts["osm_maps"] * OSM_EXPORT_SECONDS
    seconds += max(0, sum(requests.values()) - 1) * run["options"]["delay"]
    return dict(counts, seconds=math.ceil(seconds))


def photos(run, after=0):
    return collection(
        feature("Point", [photo["lon"], photo["lat"]], photo)
        for stage in run["stages"] if stage["mode"] == "streetview"
        for photo in stage["results"][after:] if photo["status"] == "saved"
    )


def capture_streetview(stage, run, folder, client, progress):
    """Save both roadside photos per stop, or one full sphere."""
    options = run["options"]
    sphere = stage["full_sphere"]
    metadata = None
    progress("Street View", len(stage["results"]), total(stage))
    for index in range(len(stage["results"]), total(stage)):
        sample = stage["samples"][index if sphere else index // 2]
        part = stage["paths"][sample["path_index"]]
        photo = dict(sample, sequence=index + 1, path_id=part["id"], path_name=part["name"], osm_id=part["osm_id"],
                     highway=part["highway"], layer=part["layer"], road_heading=sample["heading"])
        try:
            # Both sides of a stop share one panorama.
            if metadata is None or metadata["pano_id"] != sample["pano_id"]:
                fetched = streetview.fetch_metadata(client, sample["pano_id"], full_sphere=sphere)
                if (fetched["pano_id"] != sample["pano_id"]
                        or distance((sample["lat"], sample["lon"]), (fetched["lat"], fetched["lon"])) > 0.1):
                    raise ValueError("The planned panorama identity or position changed. Plan a new run.")
                metadata = fetched
            photo.update(metadata)
            if sphere:
                del photo["heading"]  # A sphere has no single perspective camera heading.
                photo["filename"] = f"streetview/photos/{index + 1:06d}_sphere.{options['streetview_format']}"
                photo.update(streetview.save_sphere(client, metadata, options["sphere_zoom"], folder / photo["filename"]),
                             source_url=streetview.metadata_url(sample["pano_id"]))
            else:
                side = "left" if index % 2 == 0 else "right"
                photo.update(side=side, heading=(sample["heading"] + (-90 if side == "left" else 90)) % 360,
                             pitch=0, fov=options["fov"])
                photo["source_url"] = streetview.image_url(photo["pano_id"], photo["heading"], photo["fov"])
                data = client.get(photo["source_url"], missing_ok=True)
                photo["filename"] = "streetview/photos/" + streetview.photo_name(photo, options["streetview_format"])
                photo.update(streetview.save_photo(data, folder / photo["filename"]))
            photo.update(captured_at=now(), status="saved", streetview_url=streetview.maps_url(photo))
        except MissingImagery as error:
            photo.update(status="skipped", reason=str(error))
        stage["results"].append(photo)
        yield
        progress("Street View", index + 1, total(stage))


def capture_satellite(stage, run, folder, client, progress):
    progress("Satellite", len(stage["results"]), total(stage))
    for i, tile in enumerate(tiles(stage["grid"])):
        if i < len(stage["results"]):
            continue
        address = f"https://mt1.google.com/vt/lyrs=s&x={tile['x']}&y={tile['y']}&z={tile['zoom']}"
        extension = run["options"]["satellite_format"]
        try:
            data = client.get(address, missing_ok=True)
        except MissingImagery:
            # Persist the gap so resume and offline export keep it transparent.
            extension = "png"
            with BytesIO() as buffer, Image.new("RGBA", (256, 256)) as image:
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
        with open_image(data, (256, 256)) as image:
            target = "JPEG" if extension == "jpg" else "PNG"
            # Preserve the original bytes when they already have the chosen format.
            if image.format != target:
                with BytesIO() as buffer, image.convert("RGB" if extension == "jpg" else "RGBA") as converted:
                    converted.save(buffer, format=target, quality=80)
                    data = buffer.getvalue()
        name = (f"satellite/patches/satellite_z{tile['zoom']}_r{tile['row'] + 1:04d}_c{tile['column'] + 1:04d}"
                f"_x{tile['x']}_y{tile['y']}.{extension}")
        write_bytes(folder / name, data)
        stage["results"].append(dict(tile, filename=name, source_url=address, captured_at=now()))
        yield
        progress("Satellite", i + 1, total(stage))


def capture_osm(stage, run, folder, client, progress):
    failure = None
    if not any(item["filename"] == "map.osm" for item in stage["results"]):
        try:
            metadata = client.maps.export(run["bounds"], folder / "map.osm", progress)
            stage["results"].append(dict(filename="map.osm", captured_at=now(), **metadata))
            yield
        except (OSError, ValueError) as error:
            # Keep terrain usable if the map fails; report the error after saving it.
            failure = error
    saved = sum("x" in item for item in stage["results"])
    count = stage["grid"]["rows"] * stage["grid"]["columns"]
    label = "Downloading terrain tiles" if failure is None else f"Downloading terrain tiles (map failed: {failure})"
    progress(label, saved, count)
    for i, tile in enumerate(tiles(stage["grid"])):
        if i < saved:
            continue
        address = f"https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{tile['zoom']}/{tile['x']}/{tile['y']}.tif"
        data = client.get(address)
        terrain.validate(terrain.TIFF(BytesIO(data)), tile)
        name = f"terrain/tiles/terrain_z{tile['zoom']}_x{tile['x']}_y{tile['y']}.tif"
        write_bytes(folder / name, data)
        stage["results"].append(dict(tile, filename=name, source_url=address, captured_at=now()))
        yield
        progress(label, i + 1, count)
    if failure is not None:
        raise failure


STAGES = dict(streetview=capture_streetview, satellite=capture_satellite, osm=capture_osm)


def export(run, folder, progress, *, rebuild=True):
    """Also works offline, on partial runs, or after a failed final export."""
    # Commit capture progress before merging, which can fail or be interrupted.
    run["exports_saved"] = False
    write_json(folder / "manifest.json", run)
    for stage in run["stages"]:
        if stage["mode"] == "streetview":
            write_json(folder / "streetview/photos.geojson", photos(run))
        elif stage["mode"] == "satellite":
            write_json(folder / "satellite/patches.geojson",
                       collection(feature("Polygon", [tile_ring(p)], p) for p in stage["results"]))
            satellite.merge(stage, folder, progress)
        else:
            terrain_results = [item for item in stage["results"] if "x" in item]
            count = stage["grid"]["rows"] * stage["grid"]["columns"]
            if len(terrain_results) == count and (rebuild or not (folder / "terrain.tif").is_file()):
                terrain.merge(folder / "terrain.tif", folder, terrain_results, stage["grid"], progress)
    run["exports_saved"] = True
    write_json(folder / "manifest.json", run)


def save_preview(run, folder):
    """Write the README, planned street paths, and a north-up plan.svg per imagery stage."""
    if run["options"]["include"] == ["osm"]:
        return
    write_bytes(folder / "README.txt", README)
    area = run["bounds"]
    left, top = pixel((area[2], area[1]), 17)
    right, bottom = pixel((area[0], area[3]), 17)
    scale = min(850 / (right - left), 510 / (bottom - top))

    def xy(point):
        x, y = pixel(point, 17)
        return f"{25 + (x - left) * scale:.2f},{45 + (y - top) * scale:.2f}"

    def corner(x, y, zoom):
        return xy(coordinate(x * 256, y * 256, zoom))

    for stage in run["stages"]:
        if stage["mode"] == "streetview":
            write_json(folder / "streetview/paths.geojson", collection(
                feature("LineString", [[p[1], p[0]] for p in part["points"]],
                        {k: v for k, v in part.items() if k != "points"})
                for part in stage["paths"]))
            roads = "".join("M" + "L".join(xy(p) for p in part["points"]) for part in stage["paths"])
            gaps = "".join(f"M{xy(gap['start'])}L{xy(gap['end'])}" for gap in stage["coverage"]["gaps"])
            stops = "".join(f"M{xy((p['lat'], p['lon']))}h.1" for p in stage["samples"])
            drawing = [
                f'<path d="{roads}" stroke="#888" stroke-width="2"/>',
                f'<path d="{gaps}" stroke="#a32632" stroke-width="4" stroke-dasharray="3 5"/>',
                f'<path d="{stops}" stroke="#a32632" stroke-width="6" stroke-linecap="round"/>',
            ]
            caption = "Planned stops. Dotted red lines show spacing gaps. North is up."
        elif stage["mode"] == "satellite":
            g = stage["grid"]
            x0, y0, x1, y1, z = g["x0"], g["y0"], g["x0"] + g["columns"], g["y0"] + g["rows"], g["zoom"]
            lines = [f"M{corner(x, y0, z)}L{corner(x, y1, z)}" for x in range(x0, x1 + 1)]
            lines += [f"M{corner(x0, y, z)}L{corner(x1, y, z)}" for y in range(y0, y1 + 1)]
            drawing = [f'<path d="{"".join(lines)}" stroke="#888" stroke-width="1"/>']
            caption = f"Satellite zoom {g['zoom']}, {g['columns']} × {g['rows']} patches. North is up."
        else:
            continue
        drawing.append(f'<rect x="25" y="45" width="{(right - left) * scale}" height="{(bottom - top) * scale}" stroke="#a32632"/>')
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 580" role="img">'
            f'<title>{caption}</title><rect width="900" height="580" fill="#faf8f4"/>'
            f'<text x="25" y="25" font-family="sans-serif" font-size="14">{caption}</text>'
            f'<svg x="25" y="45" width="850" height="510" viewBox="25 45 850 510"><g fill="none">{"".join(drawing)}</g></svg></svg>'
        )
        write_bytes(folder / stage["mode"] / "plan.svg", svg.encode())


def download(run, folder, client, progress):
    """Resume every stage, then export; a failed run keeps its checkpoint and partial exports."""
    run.update(state="running", exports_saved=False)
    run.pop("error", None)
    run.pop("finished_at", None)
    errors = []
    try:
        write_json(folder / "manifest.json", run)
        saved_at, unsaved = time.monotonic(), 0
        for stage in run["stages"]:
            if len(stage["results"]) == total(stage):
                continue
            # Checkpoint slow items every time and quick ones in batches.
            slow = stage["mode"] == "osm" or stage.get("full_sphere")
            for _ in STAGES[stage["mode"]](stage, run, folder, client, progress):
                unsaved += 1
                if slow or unsaved >= 50 or time.monotonic() - saved_at >= 30:
                    write_json(folder / "manifest.json", run)
                    saved_at, unsaved = time.monotonic(), 0
            write_json(folder / "manifest.json", run)
    except (Exception, KeyboardInterrupt) as error:
        errors.append(error)
    try:
        export(run, folder, progress, rebuild=False)
    except (Exception, KeyboardInterrupt) as error:
        errors.append(error)
    if errors:
        run.update(state="stopped" if isinstance(errors[0], KeyboardInterrupt) else "failed",
                   error=" Export also failed: ".join(str(error) or "Interrupted" for error in errors))
    else:
        run.update(state="complete", finished_at=now())
    write_json(folder / "manifest.json", run)
    if errors:
        raise errors[0]


def load(folder, *, check_files=True):
    """Read a checkpoint; library browsing can skip checking every saved file."""
    run = json.loads(contained(folder, "manifest.json").read_text(encoding="utf-8"))
    if run.get("format") != "aleph-python" or run.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported capture format. Start a new capture with this version of Aleph.")
    bounds(run["bounds"], minimum=0)
    options = settings(run["options"])
    if [s["mode"] for s in run["stages"]] != options["include"]:
        raise ValueError("Invalid saved stages.")
    for stage in run["stages"]:
        if stage["mode"] != "streetview":
            zoom = options["satellite_zoom" if stage["mode"] == "satellite" else "terrain_zoom"]
            if stage["grid"] != grid(run["bounds"], zoom):
                raise ValueError("Invalid saved tile grid.")
        if len(stage["results"]) > total(stage):
            raise ValueError("Invalid saved progress.")
        if not check_files:
            continue
        for result in stage["results"]:
            if result.get("status") == "skipped":
                continue
            try:
                path = contained(folder, result["filename"])
            except FileNotFoundError as error:
                raise ValueError("Invalid output filename in checkpoint.") from error
            if not path.is_file():
                raise ValueError(f"Missing saved file: {result['filename']}. Restore it before resuming.")
    return run
