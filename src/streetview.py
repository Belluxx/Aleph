"""Discover panoramas, match their roads, and select both roadside views."""

import json
import math
import re
import shutil
import unicodedata
from io import BytesIO
from itertools import pairwise
from pathlib import Path

from PIL import Image

from .common import MissingImagery, atomic_path, url, write_bytes
from .geo import Line, RoadIndex, clip, distance, grid, inside, tiles

EXCLUDED = re.compile(r"^(construction|proposed|planned|abandoned|razed|demolished)$")
MAIN = r"(motorway|trunk|primary|secondary|tertiary)(_link)?"
DEPTHS = {
    "main": re.compile(MAIN),
    "roads": re.compile(MAIN + r"|residential|unclassified|living_street|service|road"),
    "all": None,
}
PANO_ID = re.compile(r"[\w-]{10,200}", re.ASCII)


def panorama(identity, position):
    pano_id = identity[1]
    lat, lon = position[2:4]
    if identity[0] != 2 or not isinstance(pano_id, str) or not PANO_ID.fullmatch(pano_id):
        raise ValueError("Invalid panorama identity.")
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid panorama position.")
    return dict(pano_id=pano_id, lat=lat, lon=lon)


def google_json(data):
    return json.loads(re.sub(r"^\)\]\}'\s*", "", data.decode("utf-8")))


def parse_coverage(data):
    try:
        response = google_json(data)
        if response[1] is None or response[1] == []:
            return []
        found = []
        for row in response[1][1]:
            identity = row[0][0]
            if identity == 1 or (isinstance(identity, list) and identity[0] != 2):
                continue
            found.append(panorama(identity, row[0][2][0]))
        return found
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise ValueError(
            "Unreadable Google panorama coverage; the endpoint may have changed."
        ) from error


def parse_metadata(data, *, full_sphere=False):
    try:
        message = google_json(data)[1][0]
        code = message[0][0]
        if code == 2:
            raise MissingImagery("This planned panorama is no longer available.")
        if code not in (1, 3):
            raise ValueError("Unexpected response code.")
        result = panorama(message[1], message[5][0][1][0])
        date = (
            message[6][7]
            if len(message) > 6 and isinstance(message[6], list) and len(message[6]) > 7
            else None
        )
        result["imagery_date"] = None
        if (
            isinstance(date, list)
            and len(date) >= 2
            and all(type(v) is int for v in date[:2])
            and 1900 <= date[0] <= 2200
            and 1 <= date[1] <= 12
        ):
            result["imagery_date"] = f"{date[0]:04d}-{date[1]:02d}"
        if full_sphere:
            sizes, tile_size = message[2][3][:2]
            sizes = [[size[0][1], size[0][0]] for size in sizes]
            if (not sizes or len(sizes) > 6 or len(tile_size) != 2
                    or any(type(v) is not int or not 1 <= v <= 32768
                           for size in sizes + [tile_size] for v in size)
                    or any(w * h > 150_000_000 for w, h in sizes)):
                raise ValueError("Invalid panorama dimensions.")
            orientation = message[5][0][1][2][:3]
            if len(orientation) != 3:
                raise ValueError("Missing panorama orientation.")
            for angle in orientation:
                validate_angle(angle, "panorama orientation")
            result.update(image_sizes=sizes, tile_size=tile_size,
                          panorama_heading=orientation[0],
                          panorama_pitch=90 - orientation[1], panorama_roll=orientation[2])
        return result
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise ValueError(
            "Unreadable Google panorama metadata; resume to retry this view."
        ) from error


def metadata_url(pano_id):
    return url(
        "https://www.google.com/maps/photometa/v1",
        authuser=0,
        hl="en",
        gl="US",
        pb=f"!1m1!1smaps_sv.tactile!2m2!1sen!2sUS!3m3!1m2!1e2!2s{pano_id}!4m6!1e1!1e2!1e3!1e4!2m1!1e1",
    )


def validate_angle(value, name):
    # Google's pitch and yaw fields use protobuf TYPE_FLOAT.
    if (type(value) not in (int, float) or not math.isfinite(value)
            or abs(value) > float.fromhex("0x1.fffffep+127")):
        raise ValueError(f"{name}: use a finite 32-bit float in degrees.")
    return value


def validate_fov(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 5 <= value <= 175 or int(value) != value):
        raise ValueError("fov: use a whole number from 5 to 175.")
    return int(value)


def image_url(pano_id, heading, fov, pitch=0):
    return url(
        "https://streetviewpixels-pa.googleapis.com/v1/thumbnail",
        panoid=pano_id,
        cb_client="maps_sv.tactile",
        w=1024,
        h=576,
        yaw=validate_angle(heading, "heading"),
        # The thumbnail service uses positive-down; Aleph uses positive-up.
        pitch=-validate_angle(pitch, "pitch"),
        thumbfov=validate_fov(fov),
    )


def validate_sphere_zoom(zoom):
    if type(zoom) is not int or not 0 <= zoom <= 5:
        raise ValueError("sphere_zoom: use a whole number from 0 to 5.")
    return zoom


def tile_url(pano_id, zoom, x, y):
    return url("https://streetviewpixels-pa.googleapis.com/v1/tile",
               cb_client="maps_sv.tactile", panoid=pano_id, zoom=zoom, x=x, y=y)


def save_sphere(client, metadata, zoom, target, progress):
    """Assemble native panorama tiles; retain partial tiles for retry/resume."""
    zoom = min(validate_sphere_zoom(zoom), len(metadata["image_sizes"]) - 1)
    width, height = metadata["image_sizes"][zoom]
    tw, th = metadata["tile_size"]
    columns, rows = math.ceil(width / tw), math.ceil(height / th)
    target = Path(target)
    cache = target.parent / ".sphere-tiles" / metadata["pano_id"] / f"{zoom}-{width}x{height}-{tw}x{th}"
    count = columns * rows
    progress("Panorama tiles", 0, count)
    with Image.new("RGB", (width, height)) as sphere:
        for y in range(rows):
            for x in range(columns):
                client.check_cancel()
                path = cache / f"{x}-{y}.tile"
                data = path.read_bytes() if path.is_file() else client.get(
                    tile_url(metadata["pano_id"], zoom, x, y), missing_ok=True)
                with Image.open(BytesIO(data)) as tile:
                    if tile.format not in ("JPEG", "PNG") or tile.size != (tw, th):
                        raise ValueError(f"Expected a panorama tile of {tw} × {th} pixels.")
                    tile.load()
                    if not path.is_file():
                        write_bytes(path, data)
                    sphere.paste(tile, (x * tw, y * th))
                progress("Panorama tiles", y * columns + x + 1, count)
        client.check_cancel()
        with atomic_path(target) as temporary:
            sphere.save(temporary, format="PNG" if target.suffix == ".png" else "JPEG", quality=90)
    shutil.rmtree(cache)
    return dict(projection="equirectangular", width=width, height=height,
                sphere_zoom=zoom, tile_count=count)


def roads(ways, area, depth):
    parts, seen_edges = [], set()
    for way in sorted(ways, key=lambda way: way["id"]):
        tags = way["tags"]
        highway = tags.get("highway", "")
        if not highway or tags.get("area") == "yes" or EXCLUDED.fullmatch(highway):
            continue
        if DEPTHS[depth] and not DEPTHS[depth].fullmatch(highway):
            continue
        points, nodes = way["points"], way["nodes"]
        layer, run, part = str(tags.get("layer", "0")), [], 0

        def flush():
            nonlocal run, part
            if len(run) > 1:
                part += 1
                parts.append(
                    dict(
                        id=f"way-{way['id']}-{part}",
                        osm_id=way["id"],
                        name=tags.get("name", ""),
                        highway=highway,
                        layer=layer,
                        points=run,
                    )
                )
            run = []

        for i, (a, b) in enumerate(pairwise(points)):
            if distance(a, b) <= 0.01:
                continue
            segment = clip(a, b, area)
            edge = layer, tuple(sorted((nodes[i], nodes[i + 1])))
            if segment is None or edge in seen_edges:
                flush()
                continue
            seen_edges.add(edge)
            if run and distance(run[-1], segment[0]) > 0.01:
                flush()
            if not run:
                run.append(segment[0])
            run.append(segment[1])
            if not inside(b, area):
                flush()
        flush()
    return parts


def match_road(parts, lines, index, point, spacing):
    candidates = []
    for i, segments in index.near(point).items():
        meters, separation = lines[i].project(point, segments)
        if separation <= 30:
            candidates.append(
                dict(
                    path_index=i,
                    path_meters=meters,
                    road_distance=separation,
                    heading=lines[i].heading(meters, spacing),
                )
            )
    candidates.sort(key=lambda c: (c["road_distance"], c["path_index"]))
    if not candidates:
        return None
    nearest = candidates[0]
    plausible = [c for c in candidates if c["road_distance"] <= nearest["road_distance"] + 2]

    def endpoint(candidate):
        line = lines[candidate["path_index"]]
        meters = candidate["path_meters"]
        return min(meters, line.length - meters), line.points[
            0 if meters <= line.length / 2 else -1
        ]

    def connected(other, winner):
        a, b = other["path_index"], winner["path_index"]
        end_distance, end = endpoint(other)
        return other is winner or (
            parts[a]["layer"] == parts[b]["layer"]
            and end_distance < 5
            and lines[b].project(end)[1] <= 0.5
        )

    def joined(other):
        first, part = parts[nearest["path_index"]], parts[other["path_index"]]
        same_road = first["osm_id"] == part["osm_id"] or (
            first["name"] and first["name"] == part["name"] and first["highway"] == part["highway"]
        )
        a, start = endpoint(nearest)
        b, end = endpoint(other)
        angle = abs((nearest["heading"] - other["heading"] + 180) % 360 - 180)
        return other is nearest or (
            same_road
            and first["layer"] == part["layer"]
            and max(a, b) < 5
            and distance(start, end) <= 0.5
            and min(angle, 180 - angle) <= 15
        )

    method = "nearest-road"
    if len(plausible) > 1:
        through = [c for c in plausible if endpoint(c)[0] >= 5]
        if len(through) == 1 and all(connected(c, through[0]) for c in plausible):
            nearest, method = through[0], "through-road-at-connector"
        elif all(joined(c) for c in plausible):
            method = "joined-road-sections"
        else:
            return None
    return dict(nearest, match_method=method)


def select_stops(candidates, spacing):
    positions = []
    for candidate in sorted(candidates, key=lambda c: (c["path_meters"], c["pano_id"])):
        if not positions or candidate["path_meters"] - positions[-1]["path_meters"] >= 0.1:
            positions.append(candidate)
    index = 0
    while index < len(positions):
        yield positions[index]
        next_index = index + 1
        while next_index + 1 < len(positions) and positions[next_index + 1][
            "path_meters"
        ] - positions[index]["path_meters"] <= spacing * (1 + 1e-7):
            next_index += 1
        index = next_index


def plan(client, area, options, progress, *, allow_empty=False):
    progress("Finding roads")
    ways, metadata = client.maps.data(area, progress=progress)
    parts = roads(ways, area, options["depth"])
    if not parts and not allow_empty:
        raise ValueError("No mapped roads at this path depth. Choose another area or depth.")
    coverage_grid = grid(area, 17)
    total = coverage_grid["rows"] * coverage_grid["columns"]
    found = {}
    progress("Finding panoramas", 0, total)
    for i, tile in enumerate(tiles(coverage_grid), 1):
        address = url(
            "https://www.google.com/maps/photometa/ac/v1",
            pb=f"!1m1!1smaps_sv.tactile!6m3!1i{tile['x']}!2i{tile['y']}!3i17!8b1",
        )
        for view in parse_coverage(client.get(address)):
            if inside((view["lat"], view["lon"]), area):
                found[view["pano_id"]] = view
        progress("Finding panoramas", i, total)
    progress("Indexing roads")
    lines = [Line(part["points"]) for part in parts]
    index = RoadIndex(lines)
    groups = [[] for _ in parts]
    excluded, spacing = 0, options["step"]
    for i, view in enumerate(found.values(), 1):
        road = match_road(parts, lines, index, (view["lat"], view["lon"]), spacing)
        if road is None:
            excluded += 1
        else:
            groups[road["path_index"]].append(dict(view, **road))
        progress("Matching panoramas", i, len(found))
    samples, gaps = [], []
    for i, (line, group) in enumerate(zip(lines, groups)):
        selected = list(select_stops(group, spacing))
        positions = [0] + [view["path_meters"] for view in selected] + [line.length]
        for j, (start, end) in enumerate(pairwise(positions)):
            limit = spacing / 2 if j in (0, len(positions) - 2) else spacing
            if not selected or end - start > limit * (1 + 1e-7):
                gaps.append(
                    dict(
                        path_id=parts[i]["id"],
                        start=line.at(start),
                        end=line.at(end),
                        meters=end - start,
                    )
                )
        for stop, view in enumerate(selected, 1):
            samples.append(dict(view, stop_in_path=stop))
    return dict(
        mode="streetview",
        full_sphere=options["full_sphere"],
        paths=parts,
        samples=samples,
        results=[],
        coverage=dict(available=len(found), excluded=excluded, gaps=gaps, spacing=spacing),
        length=sum(line.length for line in lines),
        osm_data_at=metadata["osm_data_at"],
        osm_source_url=metadata["source_url"],
    )


def photo_name(photo, extension):
    slug = (
        unicodedata.normalize("NFKD", photo["path_name"] or photo["highway"])
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")[:36].rstrip("-") or "path"
    return (
        f"{photo['sequence']:06d}_{slug}_p{photo['path_index'] + 1:04d}_s{photo['stop_in_path']:04d}_"
        f"{photo['side']}_h{math.floor(photo['heading'] + 0.5) % 360:03d}.{extension}"
    )
