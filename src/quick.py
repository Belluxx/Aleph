"""Quick Street View and satellite requests."""

import tempfile
from pathlib import Path

from . import capture, places, routes, streetview
from .common import MissingImagery, RequestError, atomic_path, now, url, write_json
from .geo import bearing, bounds, coordinate, distance, extent, grid, tiles


def coverage(client, area, progress):
    g = grid(area, 17)
    count = g["rows"] * g["columns"]
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


def location(client, *, at=None, place=None, match=None, best_match=False, endpoint=places.GEOCODER):
    if at is not None:
        return places.point(at), None
    selected = places.choose(client, place, match=match, best_match=best_match, endpoint=endpoint)
    return (selected["lat"], selected["lon"]), selected


def street_photos(client, output, progress, *, at=None, place=None, pano_id=None,
                  street=None, match=None, best_match=False, endpoint=places.GEOCODER, route=None,
                  reverse=False, stops=None, step=None, view="forward", heading=0, look_at=None,
                  pitch=0, fov=75, radius=50, streetview_format="jpg", full_sphere=False, sphere_zoom=3):
    streetview.validate_sphere_zoom(sphere_zoom)
    streetview.validate_angle(heading, "heading")
    streetview.validate_angle(pitch, "pitch")
    fov = streetview.validate_fov(fov)
    places.positive(radius, "radius")
    if stops is not None and (type(stops) is not int or stops < 1):
        raise ValueError("stops must be a positive whole number.")
    if step is not None:
        places.positive(step, "step")
    if look_at is not None:
        places.point(look_at)
    selected_place, selected_routes = None, []
    requested_stops = 1
    gaps = []
    if street is not None:
        selected_place = places.choose(client, street, match=match, best_match=best_match, street=True, endpoint=endpoint)
        selected_routes = routes.resolve(client, selected_place, route=route, reverse=reverse, progress=progress)
        points = [point for section in selected_routes for point in section["points"]]
        south, west, north, east = extent(points)
        lower, upper = places.around((south, west), 80), places.around((north, east), 80)
        area = bounds((lower[0], lower[1], upper[2], upper[3]))
        views = coverage(client, area, progress)
        ways, _ = client.maps.data(area, progress=progress)
        context = streetview.roads(ways, area, "all")
        samples, requested_stops = [], 0
        counts = routes.allocate_stops(selected_routes, stops) if stops is not None else [None] * len(selected_routes)
        for section, count in zip(selected_routes, counts):
            if count == 0:
                continue
            spacing, targets = routes.sampling(section, step=step, stops=count)
            section_samples, section_gaps = routes.match_views(views, section, context, spacing, targets)
            for item in section_samples + section_gaps:
                item["route"] = section["route"]
                item["stop"] += requested_stops
            samples.extend(section_samples)
            gaps.extend(section_gaps)
            requested_stops += len(targets)
    elif pano_id:
        if not streetview.PANO_ID.fullmatch(pano_id):
            raise ValueError("Invalid panorama ID.")
        samples = [dict(pano_id=pano_id, stop=1)]
    else:
        requested, selected_place = location(client, at=at, place=place, match=match,
                                             best_match=best_match, endpoint=endpoint)
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
                  requested_stops=requested_stops, saved_stops=0,
                  place=selected_place, source="Google Street View", captured_at=now())
    if street and step is not None:
        result["requested_step_m"] = step
    if selected_routes:
        result["routes"] = [{key: value for key, value in section.items() if key != "points"}
                            for section in selected_routes]
    offsets = {"forward": 0, "backward": 180, "left": -90, "right": 90}
    directions = ("left", "right") if view == "both" else (view,)
    try:
        progress("Street View", 0, len(samples))
        for index, sample in enumerate(samples, 1):
            try:
                metadata = streetview.parse_metadata(client.get(streetview.metadata_url(sample["pano_id"])),
                                                     full_sphere=full_sphere)
                if metadata["pano_id"] != sample["pano_id"]:
                    raise ValueError("The provider returned a different panorama.")
                actual = (metadata["lat"], metadata["lon"])
                if "lat" in sample and distance(actual, (sample["lat"], sample["lon"])) > 1:
                    raise MissingImagery("Panorama position changed; retry with --refresh.")
                for direction in directions if street and not full_sphere else (None,):
                    if full_sphere:
                        name = f"{len(result['photos']) + 1:03d}_sphere.{streetview_format}"
                        photo = {**sample, **metadata}
                        if "heading" in photo:
                            photo["road_heading"] = photo.pop("heading")
                        photo.update(streetview.save_sphere(client, metadata, sphere_zoom, folder / name))
                        photo.update(path=str(folder / name), source_url=streetview.metadata_url(sample["pano_id"]))
                        result["photos"].append(photo)
                        continue
                    angle = (sample["heading"] + offsets[direction]) % 360 if street else heading
                    if look_at is not None:
                        angle = bearing(actual, look_at)
                    address = streetview.image_url(sample["pano_id"], angle, fov, pitch)
                    data = client.get(address, missing_ok=True)
                    name = f"{len(result['photos']) + 1:03d}.{streetview_format}"
                    with capture.open_image(data, (1024, 576)) as image, atomic_path(folder / name) as temporary:
                        image.convert("RGB").save(temporary, format="PNG" if streetview_format == "png" else "JPEG", quality=80)
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
                gap = dict(stop=sample["stop"], reason="no_coverage", message=str(error))
                if street:
                    gap["route"] = sample["route"]
                gaps.append(gap)
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


def satellite(client, output, progress, *, at=None, place=None, match=None, best_match=False,
              endpoint=places.GEOCODER, bbox=None, tile=None, size=200, zoom=19, satellite_format="jpg"):
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
        center, selected_place = location(client, at=at, place=place, match=match,
                                          best_match=best_match, endpoint=endpoint)
        area = places.around(center, size)
    g = grid(area, zoom)
    run = capture.plan(client, area, dict(include=["satellite"], satellite_zoom=zoom,
                                        satellite_format=satellite_format), progress)
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
                  path=str(folder / "satellite.tif"), manifest=str(folder / "manifest.json"),
                  png_path=str(folder / "satellite.png"),
                  requested_bounds=list(area), bounds=[south, west, north, east],
                  width=g["width"], height=g["height"], zoom=zoom, crs="EPSG:3857",
                  source="Google satellite", imagery_date=None, place=selected_place)
    if tile is not None:
        result["tile"] = dict(zoom=zoom, x=x, y=y)
    write_json(folder / "result.json", result)
    return result
