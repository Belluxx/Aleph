"""Discover panoramas, match their roads, and select both roadside views."""

import json
import math
import re
import shutil
import unicodedata
from itertools import pairwise
from pathlib import Path

from PIL import Image

from .common import MissingImagery, fetch, number, open_image, save_image, url, write_bytes
from .geo import Line, RoadIndex, clip, distance, grid, inside, tiles

EXCLUDED = re.compile(r"^(construction|proposed|planned|abandoned|razed|demolished)$")
MAIN = r"(motorway|trunk|primary|secondary|tertiary)(_link)?"
DEPTHS = {
    "main": re.compile(MAIN),
    "roads": re.compile(MAIN + r"|residential|unclassified|living_street|service|road"),
    "all": None,
}
PANO_ID = re.compile(r"[\w-]{10,200}", re.ASCII)
PHOTO_SIZE = (1024, 576)
VIEW_OFFSETS = {"forward": 0, "backward": 180, "left": -90, "right": 90}
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def angle(value, name):
    # Google's pitch and yaw fields use protobuf TYPE_FLOAT.
    return number(value, name, -FLOAT32_MAX, FLOAT32_MAX)


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
        if not response[1]:
            return []
        found = []
        for row in response[1][1]:
            identity = row[0][0]
            if identity == 1 or (isinstance(identity, list) and identity[0] != 2):
                continue
            found.append(panorama(identity, row[0][2][0]))
        return found
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise ValueError("Unreadable Google panorama coverage; the endpoint may have changed.") from error


def coverage(client, area, progress):
    """Every Google panorama listed in the zoom-17 tiles that cover the area."""
    coverage_grid = grid(area, 17)
    count = coverage_grid["rows"] * coverage_grid["columns"]
    found = {}
    progress("Finding panoramas", 0, count)
    addresses = (url("https://www.google.com/maps/photometa/ac/v1",
                     pb=f"!1m1!1smaps_sv.tactile!6m3!1i{tile['x']}!2i{tile['y']}!3i17!8b1")
                 for tile in tiles(coverage_grid))
    for i, data in enumerate(fetch(client, addresses), 1):
        for view in parse_coverage(data):
            found[view["pano_id"]] = view
        progress("Finding panoramas", i, count)
    return list(found.values())


def parse_metadata(data, *, full_sphere=False):
    try:
        message = google_json(data)[1][0]
        code = message[0][0]
        if code == 2:
            raise MissingImagery("This planned panorama is no longer available.")
        if code not in (1, 3):
            raise ValueError("Unexpected response code.")
        result = panorama(message[1], message[5][0][1][0])
        date = message[6][7] if len(message) > 6 and isinstance(message[6], list) and len(message[6]) > 7 else None
        result["imagery_date"] = None
        if (isinstance(date, list) and len(date) >= 2 and all(type(v) is int for v in date[:2])
                and 1900 <= date[0] <= 2200 and 1 <= date[1] <= 12):
            result["imagery_date"] = f"{date[0]:04d}-{date[1]:02d}"
        if full_sphere:
            sizes, tile_size = message[2][3][:2]
            sizes = [[size[0][1], size[0][0]] for size in sizes]
            if (not sizes or len(sizes) > 6 or len(tile_size) != 2
                    or any(type(v) is not int or not 1 <= v <= 32768 for size in sizes + [tile_size] for v in size)
                    or any(w * h > 150_000_000 for w, h in sizes)):
                raise ValueError("Invalid panorama dimensions.")
            orientation = message[5][0][1][2][:3]
            if len(orientation) != 3:
                raise ValueError("Missing panorama orientation.")
            for value in orientation:
                angle(value, "panorama orientation")
            result.update(image_sizes=sizes, tile_size=tile_size, panorama_heading=orientation[0],
                          panorama_pitch=90 - orientation[1], panorama_roll=orientation[2])
        return result
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise ValueError("Unreadable Google panorama metadata; resume to retry this view.") from error


def metadata_url(pano_id):
    return url("https://www.google.com/maps/photometa/v1", authuser=0, hl="en", gl="US",
               pb=f"!1m1!1smaps_sv.tactile!2m2!1sen!2sUS!3m3!1m2!1e2!2s{pano_id}!4m6!1e1!1e2!1e3!1e4!2m1!1e1")


def fetch_metadata(client, sample, *, full_sphere=False):
    """Refuse a panorama whose identity or position changed since it was found."""
    metadata = parse_metadata(client.get(metadata_url(sample["pano_id"])), full_sphere=full_sphere)
    if metadata["pano_id"] != sample["pano_id"] or (
            "lat" in sample and distance((sample["lat"], sample["lon"]), (metadata["lat"], metadata["lon"])) > 1):
        raise ValueError("The panorama identity or position changed. Plan again or rerun with --refresh.")
    return metadata


def image_url(pano_id, heading, fov, pitch=0):
    # The thumbnail service uses positive-down pitch; Aleph uses positive-up.
    return url("https://streetviewpixels-pa.googleapis.com/v1/thumbnail", panoid=pano_id,
               cb_client="maps_sv.tactile", w=PHOTO_SIZE[0], h=PHOTO_SIZE[1], yaw=heading, pitch=-pitch, thumbfov=fov)


def maps_url(photo):
    """A Google Maps link to the panorama, facing the photo's view when it has one."""
    params = dict(api=1, map_action="pano", pano=photo["pano_id"], viewpoint=f"{photo['lat']},{photo['lon']}")
    if "fov" in photo:
        # Maps links support 10–100°, unlike the thumbnail endpoint.
        params.update(heading=photo["heading"], pitch=photo["pitch"], fov=max(10, min(100, photo["fov"])))
    return url("https://www.google.com/maps/@", **params)


def save_photo(client, pano_id, heading, fov, pitch, path):
    address = image_url(pano_id, heading, fov, pitch)
    with open_image(client.get(address, missing_ok=True), PHOTO_SIZE) as image, image.convert("RGB") as rgb:
        save_image(rgb, path, quality=80)
    return dict(width=PHOTO_SIZE[0], height=PHOTO_SIZE[1], source_url=address)


def save_sphere(client, metadata, zoom, target):
    """Assemble native panorama tiles; retain partial tiles for retry/resume."""
    zoom = min(zoom, len(metadata["image_sizes"]) - 1)
    width, height = metadata["image_sizes"][zoom]
    tw, th = metadata["tile_size"]
    columns, rows = math.ceil(width / tw), math.ceil(height / th)
    target = Path(target)
    cache = target.parent / ".sphere-tiles" / metadata["pano_id"] / f"{zoom}-{width}x{height}-{tw}x{th}"
    positions = [(x, y) for y in range(rows) for x in range(columns)]
    missing = [(x, y) for x, y in positions if not (cache / f"{x}-{y}.tile").is_file()]
    addresses = (url("https://streetviewpixels-pa.googleapis.com/v1/tile", cb_client="maps_sv.tactile",
                     panoid=metadata["pano_id"], zoom=zoom, x=x, y=y) for x, y in missing)
    with Image.new("RGB", (width, height)) as sphere:
        for (x, y), data in zip(missing, fetch(client, addresses, missing_ok=True)):
            if data is None:
                raise MissingImagery("The requested image is not available.")
            with open_image(data, (tw, th)) as tile:
                write_bytes(cache / f"{x}-{y}.tile", data)
                sphere.paste(tile, (x * tw, y * th))
        for x, y in set(positions) - set(missing):
            client.check_cancel()
            with open_image((cache / f"{x}-{y}.tile").read_bytes(), (tw, th)) as tile:
                sphere.paste(tile, (x * tw, y * th))
        client.check_cancel()
        save_image(sphere, target, quality=90)
    shutil.rmtree(cache)
    return dict(projection="equirectangular", width=width, height=height, sphere_zoom=zoom,
                tile_count=columns * rows, source_url=metadata_url(metadata["pano_id"]))


def roads(ways, area, depth):
    """Clip mapped roads to the area, splitting them wherever they leave it or repeat an edge."""
    parts, seen_edges = [], set()
    for way in sorted(ways, key=lambda way: way["id"]):
        tags = way["tags"]
        highway = tags.get("highway", "")
        if not highway or tags.get("area") == "yes" or EXCLUDED.fullmatch(highway):
            continue
        if DEPTHS[depth] and not DEPTHS[depth].fullmatch(highway):
            continue
        points, nodes = way["points"], way["nodes"]
        layer, runs = str(tags.get("layer", "0")), [[]]
        for i, (a, b) in enumerate(pairwise(points)):
            if distance(a, b) <= 0.01:
                continue
            segment = clip(a, b, area)
            edge = layer, tuple(sorted((nodes[i], nodes[i + 1])))
            if segment is None or edge in seen_edges:
                runs.append([])
                continue
            seen_edges.add(edge)
            if runs[-1] and distance(runs[-1][-1], segment[0]) > 0.01:
                runs.append([])
            if not runs[-1]:
                runs[-1].append(segment[0])
            runs[-1].append(segment[1])
            if not inside(b, area):
                runs.append([])
        for part, run in enumerate((run for run in runs if len(run) > 1), 1):
            parts.append(dict(id=f"way-{way['id']}-{part}", osm_id=way["id"], name=tags.get("name", ""),
                              highway=highway, layer=layer, points=run))
    return parts


def match_road(parts, lines, index, point, spacing):
    candidates = []
    for i, segments in index.near(point).items():
        meters, separation = lines[i].project(point, segments)
        if separation <= 30:
            candidates.append(dict(path_index=i, path_meters=meters, road_distance=separation,
                                   heading=lines[i].heading(meters, spacing)))
    candidates.sort(key=lambda c: (c["road_distance"], c["path_index"]))
    if not candidates:
        return None
    nearest = candidates[0]
    plausible = [c for c in candidates if c["road_distance"] <= nearest["road_distance"] + 2]

    def endpoint(candidate):
        line = lines[candidate["path_index"]]
        meters = candidate["path_meters"]
        return min(meters, line.length - meters), line.points[0 if meters <= line.length / 2 else -1]

    def connected(other, winner):
        a, b = other["path_index"], winner["path_index"]
        end_distance, end = endpoint(other)
        return other is winner or (
            parts[a]["layer"] == parts[b]["layer"] and end_distance < 5 and lines[b].project(end)[1] <= 0.5)

    def joined(other):
        first, part = parts[nearest["path_index"]], parts[other["path_index"]]
        same_road = first["osm_id"] == part["osm_id"] or (
            first["name"] and first["name"] == part["name"] and first["highway"] == part["highway"])
        a, start = endpoint(nearest)
        b, end = endpoint(other)
        turn = abs((nearest["heading"] - other["heading"] + 180) % 360 - 180)
        return other is nearest or (
            same_road and first["layer"] == part["layer"] and max(a, b) < 5
            and distance(start, end) <= 0.5 and min(turn, 180 - turn) <= 15)

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


def distinct(candidates):
    """Coverage can include different panorama IDs at the same physical position."""
    unique = []
    for candidate in sorted(candidates, key=lambda c: (c["path_meters"], c["road_distance"], c["pano_id"])):
        if not unique or candidate["path_meters"] - unique[-1]["path_meters"] >= 0.1:
            unique.append(candidate)
    return unique


def select_stops(candidates, spacing):
    positions = distinct(candidates)
    index = 0
    while index < len(positions):
        yield positions[index]
        following = index + 1
        while (following + 1 < len(positions) and positions[following + 1]["path_meters"]
               - positions[index]["path_meters"] <= spacing * (1 + 1e-7)):
            following += 1
        index = following


def plan(client, area, options, progress, *, allow_empty=False):
    progress("Finding roads")
    ways, metadata = client.maps.data(area, progress=progress)
    parts = roads(ways, area, options["depth"])
    if not parts and not allow_empty:
        raise ValueError("No mapped roads at this path depth. Choose another area or depth.")
    found = [view for view in coverage(client, area, progress) if inside((view["lat"], view["lon"]), area)]
    progress("Indexing roads")
    lines = [Line(part["points"]) for part in parts]
    index = RoadIndex(lines)
    groups = [[] for _ in parts]
    excluded, spacing = 0, options["step"]
    for i, view in enumerate(found, 1):
        road = match_road(parts, lines, index, (view["lat"], view["lon"]), spacing)
        if road is None:
            excluded += 1
        else:
            groups[road["path_index"]].append(dict(view, **road))
        progress("Matching panoramas", i, len(found))
    samples, gaps = [], []
    for part, line, group in zip(parts, lines, groups):
        selected = list(select_stops(group, spacing))
        positions = [0] + [view["path_meters"] for view in selected] + [line.length]
        for j, (start, end) in enumerate(pairwise(positions)):
            limit = spacing / 2 if j in (0, len(positions) - 2) else spacing
            if not selected or end - start > limit * (1 + 1e-7):
                gaps.append(dict(path_id=part["id"], start=line.at(start), end=line.at(end), meters=end - start))
        samples.extend(dict(view, stop_in_path=stop) for stop, view in enumerate(selected, 1))
    return dict(
        mode="streetview", full_sphere=options["full_sphere"], paths=parts, samples=samples, results=[],
        coverage=dict(available=len(found), excluded=excluded, gaps=gaps, spacing=spacing),
        length=sum(line.length for line in lines),
        osm_data_at=metadata["osm_data_at"], osm_source_url=metadata["source_url"],
    )


def photo_name(photo, extension):
    slug = unicodedata.normalize("NFKD", photo["path_name"] or photo["highway"]).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")[:36].rstrip("-") or "path"
    return (f"{photo['sequence']:06d}_{slug}_p{photo['path_index'] + 1:04d}_s{photo['stop_in_path']:04d}_"
            f"{photo['side']}_h{math.floor(photo['heading'] + 0.5) % 360:03d}.{extension}")
