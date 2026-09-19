"""Quick Street View and satellite requests."""

import tempfile
from pathlib import Path

from . import capture, places, routes, streetview
from .common import MissingImagery, RequestError, atomic_path, now, url, write_json
from .geo import bearing, bounds, coordinate, distance, grid, tiles

MAX_TILES = 128
MAX_PIXELS = 16_000_000


def coverage(client, area, progress):
    g = grid(area, 17)
    count = g["rows"] * g["columns"]
    if count > MAX_TILES:
        raise RequestError("request_too_large", f"Quick requests allow at most {MAX_TILES} coverage tiles.")
    found = {}
    for i, tile in enumerate(tiles(g), 1):
        address = url("https://www.google.com/maps/photometa/ac/v1",
                      pb=f"!1m1!1smaps_sv.tactile!6m3!1i{tile['x']}!2i{tile['y']}!3i17!8b1")
        try:
            views = streetview.parse_coverage(client.get(address))
        except (OSError, ValueError) as error:
            raise RequestError("provider_unavailable", f"Street View coverage request failed: {error}") from error
        for view in views:
            found[view["pano_id"]] = view
        progress("Finding panoramas", i, count)
    return list(found.values())


def location(client, *, at=None, place=None, match=None, endpoint=places.GEOCODER):
    if at is not None:
        return places.point(at), None
    selected = places.choose(client, place, match=match, endpoint=endpoint)
    return (selected["lat"], selected["lon"]), selected


def street_photos(client, output, progress, *, at=None, place=None, pano_id=None,
                  street=None, match=None, endpoint=places.GEOCODER, route=None,
                  reverse=False, stops=10, view="forward", heading=0, look_at=None,
                  pitch=0, fov=75, radius=50, image_format="jpg"):
    places.number(heading, "heading", 0, 360)
    places.number(pitch, "pitch", -90, 90)
    places.number(fov, "fov", 20, 120)
    places.number(radius, "radius", 1, 1000)
    places.number(stops, "stops", 1, 100)
    if look_at is not None:
        places.point(look_at)
    selected_place, selected_route = None, None
    gaps = []
    if street is not None:
        selected_place = places.choose(client, street, match=match, street=True, endpoint=endpoint)
        selected_route = routes.resolve(client, selected_place, route=route, reverse=reverse)
        points = selected_route["points"]
        south, north = min(p[0] for p in points), max(p[0] for p in points)
        west, east = min(p[1] for p in points), max(p[1] for p in points)
        lower, upper = places.around((south, west), 80), places.around((north, east), 80)
        area = bounds((lower[0], lower[1], upper[2], upper[3]))
        views = coverage(client, area, progress)
        identities = ",".join(map(str, selected_route["way_ids"]))
        data = places.osm_data(client, f'[out:json][timeout:25];way(id:{identities})->.street;'
                              'way(around.street:40)["highway"];out body geom;')
        context = streetview.roads(data, area, "all")
        samples, gaps = routes.match_views(views, selected_route, context, stops)
    elif pano_id:
        if not streetview.PANO_ID.fullmatch(pano_id):
            raise ValueError("Invalid panorama ID.")
        samples = [dict(pano_id=pano_id, stop=1)]
    else:
        requested, selected_place = location(client, at=at, place=place, match=match, endpoint=endpoint)
        views = coverage(client, places.around(requested, radius * 2), progress)
        nearest = min(views, key=lambda v: (distance(requested, (v["lat"], v["lon"])), v["pano_id"]), default=None)
        samples = []
        if nearest and distance(requested, (nearest["lat"], nearest["lon"])) <= radius:
            samples = [dict(nearest, stop=1, requested_location=list(requested))]
    if not samples:
        raise RequestError("no_coverage", "No matching Street View panoramas were found.", gaps=gaps)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="aleph-streetview-", dir=output))
    result = dict(command="streetview", status="partial", folder=str(folder), photos=[], gaps=gaps,
                  requested_stops=stops if street else 1, saved_stops=0,
                  place=selected_place, source="Google Street View", captured_at=now())
    if selected_route:
        result["route"] = {key: value for key, value in selected_route.items() if key != "points"}
    offsets = {"forward": 0, "backward": 180, "left": -90, "right": 90}
    directions = ("left", "right") if view == "both" else (view,)
    try:
        for index, sample in enumerate(samples, 1):
            try:
                metadata = streetview.parse_metadata(client.get(streetview.metadata_url(sample["pano_id"])))
                if metadata["pano_id"] != sample["pano_id"]:
                    raise ValueError("The provider returned a different panorama.")
                actual = (metadata["lat"], metadata["lon"])
                if "lat" in sample and distance(actual, (sample["lat"], sample["lon"])) > 1:
                    raise MissingImagery("Panorama position changed; retry with --refresh.")
                for direction in directions if street else (None,):
                    angle = (sample["heading"] + offsets[direction]) % 360 if street else heading % 360
                    if look_at is not None:
                        angle = bearing(actual, look_at)
                    address = streetview.image_url(sample["pano_id"], angle, fov, pitch)
                    data = client.get(address, missing_ok=True)
                    name = f"{len(result['photos']) + 1:03d}.{image_format}"
                    with capture.open_image(data, (1024, 576)) as image, atomic_path(folder / name) as temporary:
                        image.convert("RGB").save(temporary, format="PNG" if image_format == "png" else "JPEG", quality=80)
                    photo = {**sample, **metadata}
                    photo.update(heading=angle, pitch=pitch, fov=fov,
                                 path=str(folder / name), source_url=address, width=1024, height=576)
                    if street:
                        photo["view"] = direction
                    if "requested_location" in sample:
                        photo["distance_m"] = round(distance(sample["requested_location"], actual), 2)
                    result["photos"].append(photo)
                result["saved_stops"] += 1
            except MissingImagery as error:
                gaps.append(dict(stop=sample["stop"], reason="no_coverage", message=str(error)))
            progress("Street View", index, len(samples))
        if not result["photos"]:
            raise RequestError("no_coverage", "The selected panoramas are no longer available.", folder=str(folder))
        result["status"] = "partial" if gaps else "complete"
    except ValueError as error:
        raise RequestError("provider_unavailable", str(error), folder=str(folder)) from error
    except OSError as error:
        raise RequestError("request_failed", str(error), folder=str(folder)) from error
    except KeyboardInterrupt as error:
        raise RequestError("interrupted", "Stopped. Rerun this request to reuse cached responses.", folder=str(folder)) from error
    finally:
        write_json(folder / "result.json", result)
    return result


def satellite(client, output, progress, *, at=None, place=None, match=None,
              endpoint=places.GEOCODER, bbox=None, tile=None, size=200, zoom=19):
    if tile is not None:
        zoom, x, y = tile
    places.number(zoom, "zoom", 1, 21)
    selected_place = None
    if tile is not None:
        places.number(x, "tile x", 0, 2 ** zoom - 1)
        places.number(y, "tile y", 0, 2 ** zoom - 1)
        north, west = coordinate(x * 256, y * 256, zoom)
        south, east = coordinate((x + 1) * 256, (y + 1) * 256, zoom)
        area = (south, west, north, east)
    elif bbox is not None:
        area = bounds(bbox)
    else:
        center, selected_place = location(client, at=at, place=place, match=match, endpoint=endpoint)
        area = places.around(center, size)
    g = grid(area, zoom)
    if g["rows"] * g["columns"] > MAX_TILES or g["width"] * g["height"] > MAX_PIXELS:
        raise RequestError("request_too_large", "Use a smaller area or lower zoom (128 tiles / 16 megapixels maximum).")
    run = capture.plan(client, area, dict(include=["satellite"], satellite_zoom=zoom), progress)
    folder = capture.create_folder(run, output)
    try:
        capture.download(run, folder, client, progress)
    except OSError as error:
        raise RequestError("request_failed", str(error), folder=str(folder)) from error
    except KeyboardInterrupt as error:
        raise RequestError("interrupted", "Stopped. Use 'capture resume' to continue this capture.", folder=str(folder)) from error
    # Pixel rounding means actual bounds can extend slightly beyond the request.
    north, west = coordinate(g["left"], g["top"], zoom)
    south, east = coordinate(g["left"] + g["width"], g["top"] + g["height"], zoom)
    result = dict(command="satellite", status="complete", folder=str(folder),
                  path=str(folder / "satellite.png"), manifest=str(folder / "manifest.json"),
                  requested_bounds=list(area), bounds=[south, west, north, east],
                  width=g["width"], height=g["height"], zoom=zoom, crs="EPSG:3857",
                  source="Google satellite", imagery_date=None, place=selected_place)
    if tile is not None:
        result["tile"] = dict(zoom=zoom, x=x, y=y)
    write_json(folder / "result.json", result)
    return result
