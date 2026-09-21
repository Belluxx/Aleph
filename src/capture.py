"""Download stages and keep their files and checkpoint together."""

import json
import math
import struct
import tempfile
import time
import zlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import streetview, terrain
from .common import (
    MissingImagery,
    atomic_path,
    contained,
    now,
    url,
    write_bytes,
    write_json,
)
from .geo import (
    bounds,
    collection,
    coordinate,
    distance,
    feature,
    grid,
    pixel,
    tile_ring,
    tiles,
)

SOURCES = ("streetview", "satellite", "osm")
FORMAT_VERSION = 3
DEFAULT_OPTIONS = dict(
    include=list(SOURCES), step=30, fov=75, depth="roads", image_format="jpg",
    delay=0, satellite_zoom=18, terrain_zoom=14,
)

OSM_NOTES = {
    "map": "OSM XML from Geofabrik with original tags and topology; coordinates are WGS84. Contributor names, IDs and changeset IDs are omitted by Geofabrik.",
    "selection": "Bounding-box selection with complete ways. Relations retain their original member references, but members outside the extract are not downloaded recursively. Objects may extend outside the rectangle; crossings without inside nodes and enclosing polygons may be absent.",
    "terrain": "Float32 heights in meters, EPSG:3857. Original compressed blocks retained without resampling. Full edge tiles extend beyond the rectangle. Tiles are checkpointed individually; terrain.tif is built after all terrain tiles are saved.",
    "quality": "Terrain resolution, dates, accuracy and vertical reference vary by source. Higher zoom does not guarantee more detail.",
}


def settings(values=None):
    """Validate settings shared by the CLI and dashboard."""
    values = {} if values is None else values
    if not isinstance(values, dict) or values.keys() - DEFAULT_OPTIONS.keys():
        raise ValueError("Unknown capture settings.")
    options = {**DEFAULT_OPTIONS, **values}
    if not isinstance(options["include"], list) or not all(
        isinstance(source, str) for source in options["include"]
    ):
        raise ValueError("Choose one or more capture sources.")
    selected = set(options["include"])
    if not selected or selected - set(SOURCES):
        raise ValueError("Choose one or more sources: streetview, satellite, osm.")
    options["include"] = [source for source in SOURCES if source in selected]
    for key in ("step", "delay"):
        value = options[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (key == "step" and value == 0)):
            requirement = "positive" if key == "step" else "nonnegative"
            raise ValueError(f"{key}: use a {requirement} finite number.")
    for key, low, high in (
        ("fov", 20, 120),
        ("satellite_zoom", 1, 21), ("terrain_zoom", 1, 14),
    ):
        value = options[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high or (key.endswith("zoom") and int(value) != value)):
            raise ValueError(f"{key}: use {'a whole number' if key.endswith('zoom') else 'a number'} from {low} to {high}.")
        if key.endswith("zoom"):
            options[key] = int(value)
    if options["depth"] not in ("main", "roads", "all"):
        raise ValueError("Choose main, roads, or all for depth.")
    if options["image_format"] not in ("jpg", "png"):
        raise ValueError("Choose jpg or png for image format.")
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
        stages.append(
            streetview.plan(
                client,
                area,
                options,
                progress,
                allow_empty=len(options["include"]) > 1,
            )
        )
    if "satellite" in options["include"]:
        stages.append(
            dict(mode="satellite", grid=grid(area, options["satellite_zoom"]), results=[])
        )
    if "osm" in options["include"]:
        stages.append(
            dict(mode="osm", grid=grid(area, options["terrain_zoom"]), results=[], notes=OSM_NOTES)
        )
    return dict(
        format="aleph-python",
        version=FORMAT_VERSION,
        bounds=area,
        options=options,
        started_at=now(),
        state="planned",
        stages=stages,
    )


def total(stage):
    if stage["mode"] == "streetview":
        return len(stage["samples"]) * 2
    return stage["grid"]["rows"] * stage["grid"]["columns"] + (stage["mode"] == "osm")


def estimate(run):
    """Estimate remaining downloads using observed zero-delay capture timings.

    2026-09-19 17:26:41 UTC: 66 Street View photos in 36 s,
    including panorama metadata. 2026-09-19 17:09:18 UTC:
    1,020 satellite tiles in 374 s and 9 terrain tiles in 10 s.
    2026-09-21: map exports with completed multipolygons took 17–29 s for
    10–1,000 km² from the 384 MB central Italy region. Region size affects this cost.
    Planning, initial regional downloads and final exports are excluded.
    """
    counts = dict(streetview_photos=0, streetview_stops=0, satellite_tiles=0,
                  terrain_tiles=0, osm_maps=0)
    metadata_requests = 0
    for stage in run["stages"]:
        remaining = total(stage) - len(stage["results"])
        if stage["mode"] == "streetview":
            counts["streetview_photos"] = remaining
            samples = stage["samples"][len(stage["results"]) // 2:]
            counts["streetview_stops"] = len(samples)
            previous = None  # Metadata is fetched again when resuming a run.
            for sample in samples:
                if sample["pano_id"] != previous:
                    metadata_requests += 1
                    previous = sample["pano_id"]
        elif stage["mode"] == "satellite":
            counts["satellite_tiles"] = remaining
        else:
            counts["osm_maps"] = int(not any(item["filename"] == "map.osm"
                                             for item in stage["results"]))
            counts["terrain_tiles"] = remaining - counts["osm_maps"]
    requests = (counts["streetview_photos"] + metadata_requests
                + counts["satellite_tiles"] + counts["terrain_tiles"])
    download_seconds = (counts["streetview_photos"] * 0.55
                        + counts["satellite_tiles"] * 0.37
                        + counts["terrain_tiles"] * 1.1 + counts["osm_maps"] * 30)
    delay_seconds = max(0, requests - 1) * run["options"]["delay"]
    return dict(counts, seconds=math.ceil(download_seconds + delay_seconds))


def photos(run, after=0):
    return collection(
        feature("Point", [photo["lon"], photo["lat"]], photo)
        for stage in run["stages"] if stage["mode"] == "streetview"
        for photo in stage["results"][after:] if photo["status"] == "saved"
    )


def open_image(data, size):
    image = Image.open(BytesIO(data))
    try:
        if image.format not in ("JPEG", "PNG") or image.size != size:
            raise ValueError(f"Expected a JPEG or PNG image of {size[0]} × {size[1]} pixels.")
        image.load()
        return image
    except BaseException:
        image.close()
        raise


def capture_streets(stage, run, folder, client, progress):
    options = run["options"]
    cached_id, metadata = None, None
    for index in range(len(stage["results"]), total(stage)):
        sample = stage["samples"][index // 2]
        part = stage["paths"][sample["path_index"]]
        side = "left" if index % 2 == 0 else "right"
        photo = dict(
            sample,
            side=side,
            sequence=index + 1,
            path_id=part["id"],
            path_name=part["name"],
            osm_id=part["osm_id"],
            highway=part["highway"],
            layer=part["layer"],
            road_heading=sample["heading"],
            heading=(sample["heading"] + (-90 if side == "left" else 90)) % 360,
            pitch=0,
            fov=options["fov"],
        )
        try:
            if cached_id != sample["pano_id"]:
                metadata = streetview.parse_metadata(
                    client.get(streetview.metadata_url(sample["pano_id"]))
                )
                if (
                    metadata["pano_id"] != sample["pano_id"]
                    or distance((sample["lat"], sample["lon"]), (metadata["lat"], metadata["lon"]))
                    > 0.1
                ):
                    raise ValueError(
                        "The planned panorama identity or position changed. Plan a new run."
                    )
                cached_id = sample["pano_id"]
            photo.update(metadata)
            photo["source_url"] = streetview.image_url(
                photo["pano_id"], photo["heading"], photo["fov"]
            )
            data = client.get(photo["source_url"], missing_ok=True)
            extension = options["image_format"]
            photo["filename"] = "streetview/photos/" + streetview.photo_name(photo, extension)
            with (
                open_image(data, (1024, 576)) as image,
                atomic_path(folder / photo["filename"]) as temporary,
            ):
                image.convert("RGB").save(
                    temporary, format="PNG" if extension == "png" else "JPEG", quality=80
                )
            photo.update(width=1024, height=576, captured_at=now(), status="saved")
            photo["streetview_url"] = url(
                "https://www.google.com/maps/@",
                api=1,
                map_action="pano",
                viewpoint=f"{photo['lat']},{photo['lon']}",
                heading=photo["heading"],
                pitch=0,
                fov=photo["fov"],
                pano=photo["pano_id"],
            )
        except MissingImagery as error:
            photo.update(status="skipped", reason=str(error))
        stage["results"].append(photo)
        yield
        progress("Street View", index + 1, total(stage))


def capture_satellite(stage, folder, client, progress):
    for i, tile in enumerate(tiles(stage["grid"])):
        if i < len(stage["results"]):
            continue
        address = f"https://mt1.google.com/vt/lyrs=s&x={tile['x']}&y={tile['y']}&z={tile['zoom']}"
        try:
            data = client.get(address, missing_ok=True)
        except MissingImagery:
            # Persist the gap so resume and offline export keep it transparent.
            with BytesIO() as buffer, Image.new("RGBA", (256, 256)) as image:
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
        with open_image(data, (256, 256)) as image:
            extension = "jpg" if image.format == "JPEG" else "png"
        name = (
            f"satellite/patches/satellite_z{tile['zoom']}_r{tile['row'] + 1:04d}_c{tile['column'] + 1:04d}"
            f"_x{tile['x']}_y{tile['y']}.{extension}"
        )
        write_bytes(folder / name, data)
        stage["results"].append(dict(tile, filename=name, source_url=address, captured_at=now()))
        yield
        progress("Satellite", i + 1, total(stage))


def capture_osm(stage, area, folder, client, progress):
    failure = None
    if not any(item["filename"] == "map.osm" for item in stage["results"]):
        try:
            metadata = client.maps.export(area, folder / "map.osm", progress)
            stage["results"].append(dict(filename="map.osm", captured_at=now(), **metadata))
            yield
        except (OSError, ValueError) as error:
            # Keep terrain usable if the map fails; report the error after saving it.
            failure = error
    saved = sum("x" in item for item in stage["results"])
    count = stage["grid"]["rows"] * stage["grid"]["columns"]
    label = "Downloading terrain tiles" if failure is None else "Downloading terrain tiles (OSM unavailable)"
    progress(label, saved, count)
    for i, tile in enumerate(tiles(stage["grid"])):
        if i < saved:
            continue
        address = f"https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{tile['zoom']}/{tile['x']}/{tile['y']}.tif"
        data = client.get(address)
        terrain.validate(terrain.TIFF(BytesIO(data)), tile)
        name = f"terrain/tiles/terrain_z{tile['zoom']}_x{tile['x']}_y{tile['y']}.tif"
        write_bytes(folder / name, data)
        stage["results"].append(
            dict(tile, filename=name, source_url=address, captured_at=now())
        )
        yield
        progress(label, i + 1, count)
    if failure is not None:
        raise failure


def merge_satellite(stage, folder):
    """Write RGBA PNG scanlines with at most one tile row decoded at a time."""
    mosaic = stage["grid"]
    width, height = mosaic["width"], mosaic["height"]
    bottom = mosaic["top"] + height
    with atomic_path(folder / "satellite.png") as temporary, temporary.open("wb") as output:
        def chunk(kind, data):
            output.write(struct.pack(">I", len(data)))
            output.write(kind)
            output.write(data)
            output.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind))))

        output.write(b"\x89PNG\r\n\x1a\n")
        chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
        compressor = zlib.compressobj()
        with Image.new("RGBA", (width, min(256, height))) as strip:
            for row in range(mosaic["rows"]):
                tile_top = (mosaic["y0"] + row) * 256
                top = max(mosaic["top"], tile_top)
                strip_height = min(bottom, tile_top + 256) - top
                strip.paste((0, 0, 0, 0), (0, 0, width, strip.height))
                # Saved patches are a row-major prefix, including transparent gaps.
                first = row * mosaic["columns"]
                for photo in stage["results"][first:first + mosaic["columns"]]:
                    with open_image((folder / photo["filename"]).read_bytes(), (256, 256)) as patch:
                        strip.paste(patch, (photo["x"] * 256 - mosaic["left"], tile_top - top))
                for y in range(strip_height):
                    with strip.crop((0, y, width, y + 1)) as scanline:
                        # Filter 0 keeps encoding simple and requires no previous row.
                        data = compressor.compress(b"\0" + scanline.tobytes())
                    if data:
                        chunk(b"IDAT", data)
        chunk(b"IDAT", compressor.flush())
        chunk(b"IEND", b"")


def export(run, folder, progress, *, rebuild=True):
    """Also works offline, on partial runs, or after a failed final export."""
    # Commit capture progress before merging, which can fail or be interrupted.
    run["exports_saved"] = False
    write_json(folder / "manifest.json", run)
    for stage in run["stages"]:
        if stage["mode"] == "streetview":
            write_json(folder / "streetview/photos.geojson", photos(run))
        elif stage["mode"] == "satellite":
            write_json(
                folder / "satellite/patches.geojson",
                collection(feature("Polygon", [tile_ring(p)], p) for p in stage["results"]),
            )
            progress("Merging satellite image")
            merge_satellite(stage, folder)
        elif stage["mode"] == "osm":
            terrain_results = [item for item in stage["results"] if "x" in item]
            count = stage["grid"]["rows"] * stage["grid"]["columns"]
            if len(terrain_results) == count and (rebuild or not (folder / "terrain.tif").is_file()):
                terrain.merge(folder / "terrain.tif", folder, terrain_results, stage["grid"], progress)
    run["exports_saved"] = True
    write_json(folder / "manifest.json", run)


def save_preview(run, folder):
    if run["options"]["include"] == ["osm"]:
        return
    write_bytes(
        folder / "README.txt",
        b"Aleph capture\n\n"
        b"streetview/photos/ contains photos; streetview/ holds GeoJSON and plan.svg.\n"
        b"satellite/patches/ contains patches; satellite/ holds GeoJSON and plan.svg.\n"
        b"Open satellite.png for the merged image and map.osm for native map data.\n"
        b"Photo headings are clockwise from north; left/right follow OSM node order.\n"
        b"GeoJSON files record paths, photo locations, or satellite patch footprints.\n"
        b"Satellite images are north-up Web Mercator, cropped to enclosing pixels.\n"
        b"terrain.tif is Float32 meters in EPSG:3857, with original source blocks.\n"
        b"Keep terrain/tiles/ for resume and offline export.\n"
        b"Filenames in the manifest and GeoJSON are relative to this run folder.\n"
        b"See manifest.json for settings, sources, imagery dates, and coverage notes.\n\n"
        b"Keep this folder together. Continue with: alephgeo capture resume FOLDER\n"
        b"Rebuild images and metadata offline with: alephgeo capture export FOLDER\n",
    )
    area = run["bounds"]
    left, top = pixel((area[2], area[1]), 17)
    right, bottom = pixel((area[0], area[3]), 17)
    scale = min(850 / (right - left), 510 / (bottom - top))

    def xy(point):
        x, y = pixel(point, 17)
        return f"{25 + (x - left) * scale:.2f},{45 + (y - top) * scale:.2f}"

    for stage in run["stages"]:
        if stage["mode"] == "osm":
            continue
        drawing = []
        if stage["mode"] == "streetview":
            write_json(
                folder / "streetview/paths.geojson",
                collection(
                    feature(
                        "LineString",
                        [[p[1], p[0]] for p in part["points"]],
                        {k: v for k, v in part.items() if k != "points"},
                    )
                    for part in stage["paths"]
                ),
            )
            roads = "".join(
                "M" + "L".join(xy(p) for p in part["points"]) for part in stage["paths"]
            )
            gaps = "".join(
                f"M{xy(gap['start'])}L{xy(gap['end'])}" for gap in stage["coverage"]["gaps"]
            )
            stops = "".join(f"M{xy((p['lat'], p['lon']))}h.1" for p in stage["samples"])
            drawing.extend(
                [
                    f'<path d="{roads}" stroke="#888" stroke-width="2"/>',
                    f'<path d="{gaps}" stroke="#a32632" stroke-width="4" stroke-dasharray="3 5"/>',
                    f'<path d="{stops}" stroke="#a32632" stroke-width="6" stroke-linecap="round"/>',
                ]
            )
            caption = "Planned stops. Dotted red lines show spacing gaps. North is up."
        else:
            g = stage["grid"]
            lines = []
            for x in range(g["x0"], g["x0"] + g["columns"] + 1):
                lines.append(
                    "M"
                    + xy(coordinate(x * 256, g["y0"] * 256, g["zoom"]))
                    + "L"
                    + xy(coordinate(x * 256, (g["y0"] + g["rows"]) * 256, g["zoom"]))
                )
            for y in range(g["y0"], g["y0"] + g["rows"] + 1):
                lines.append(
                    "M"
                    + xy(coordinate(g["x0"] * 256, y * 256, g["zoom"]))
                    + "L"
                    + xy(coordinate((g["x0"] + g["columns"]) * 256, y * 256, g["zoom"]))
                )
            drawing.append(f'<path d="{"".join(lines)}" stroke="#888" stroke-width="1"/>')
            caption = f"Satellite zoom {g['zoom']}, {g['columns']} × {g['rows']} patches. North is up."
        drawing.append(
            f'<rect x="25" y="45" width="{(right - left) * scale}" height="{(bottom - top) * scale}" stroke="#a32632"/>'
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 580" role="img">'
            f'<title>{caption}</title><rect width="900" height="580" fill="#faf8f4"/>'
            f'<text x="25" y="25" font-family="sans-serif" font-size="14">{caption}</text>'
            f'<svg x="25" y="45" width="850" height="510" viewBox="25 45 850 510"><g fill="none">{"".join(drawing)}</g></svg></svg>'
        )
        write_bytes(folder / stage["mode"] / "plan.svg", svg.encode())


def download(run, folder, client, progress):
    run.update(state="running", exports_saved=False)
    run.pop("error", None)
    run.pop("finished_at", None)
    saved_at, saved_count = time.monotonic(), 0
    failure = None
    try:
        write_json(folder / "manifest.json", run)
        for stage in run["stages"]:
            if len(stage["results"]) == total(stage):
                continue
            if stage["mode"] != "osm":
                label = "Street View" if stage["mode"] == "streetview" else "Satellite"
                progress(label, len(stage["results"]), total(stage))
            if stage["mode"] == "streetview":
                work = capture_streets(stage, run, folder, client, progress)
            elif stage["mode"] == "satellite":
                work = capture_satellite(stage, folder, client, progress)
            else:
                work = capture_osm(stage, run["bounds"], folder, client, progress)
            for _ in work:
                saved_count += 1
                if stage["mode"] == "osm" or saved_count >= 50 or time.monotonic() - saved_at >= 30:
                    write_json(folder / "manifest.json", run)
                    saved_at, saved_count = time.monotonic(), 0
            write_json(folder / "manifest.json", run)
    except (Exception, KeyboardInterrupt) as error:
        failure = error
        run.update(
            state="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error) or "Interrupted",
        )
    try:
        export(run, folder, progress, rebuild=False)
    except (Exception, KeyboardInterrupt) as error:
        if failure is None:
            failure = error
            run.update(state="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=str(error) or "Interrupted")
        else:
            run["error"] += f" Export also failed: {error}"
    if failure is None:
        run.update(state="complete", finished_at=now())
    write_json(folder / "manifest.json", run)
    if failure is not None:
        raise failure


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
            zoom = options[
                "satellite_zoom" if stage["mode"] == "satellite" else "terrain_zoom"
            ]
            if stage["grid"] != grid(run["bounds"], zoom):
                raise ValueError("Invalid saved tile grid.")
        if len(stage["results"]) > total(stage):
            raise ValueError("Invalid saved progress.")
        if not check_files:
            continue
        for result in stage["results"]:
            if result.get("status") == "skipped":
                continue
            name = result["filename"]
            try:
                path = contained(folder, name)
            except FileNotFoundError as error:
                raise ValueError("Invalid output filename in checkpoint.") from error
            if not path.is_file():
                raise ValueError(f"Missing saved file: {name}. Restore it before resuming.")
    return run
