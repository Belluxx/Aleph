"""Quick Street View and satellite requests."""

import tempfile
from pathlib import Path

from . import capture, places, routes, streetview
from .common import WORKERS, MissingImagery, RequestError, now, number, positive, write_json
from .geo import around, bearing, bounds, coordinate, distance, extent, point

def coverage(client, area, progress, workers):
    try:
        return streetview.coverage(client, area, progress, workers)
    except (OSError, ValueError) as error:
        raise RequestError("provider_unavailable", f"Street View coverage request failed: {error}") from error


def location(client, *, at=None, place=None, match=None, best_match=False, endpoint=places.GEOCODER):
    if at is not None:
        return point(at), None
    selected = places.choose(client, place, match=match, best_match=best_match, endpoint=endpoint)
    return (selected["lat"], selected["lon"]), selected


def street_samples(client, sections, stops, step, progress, workers):
    """Match panoramas to target positions along every selected street section."""
    south, west, north, east = extent([p for section in sections for p in section["points"]])
    lower, upper = around((south, west), 80), around((north, east), 80)
    area = bounds((lower[0], lower[1], upper[2], upper[3]))
    views = coverage(client, area, progress, workers)
    ways, _ = client.maps.data(area, progress=progress)
    context = streetview.roads(ways, area, "all")
    samples, gaps, requested = [], [], 0
    counts = routes.allocate_stops(sections, stops) if stops is not None else [None] * len(sections)
    for section, count in zip(sections, counts):
        if count == 0:
            continue
        spacing, targets = routes.sampling(section, step=step, stops=count)
        section_samples, section_gaps = routes.match_views(views, section, context, spacing, targets)
        for item in section_samples + section_gaps:
            item.update(route=section["route"], stop=item["stop"] + requested)
        samples += section_samples
        gaps += section_gaps
        requested += len(targets)
    return samples, gaps, requested


def street_photos(client, output, progress, *, at=None, place=None, pano_id=None, street=None, match=None,
                  best_match=False, endpoint=places.GEOCODER, route=None, reverse=False, stops=None, step=None,
                  view="forward", heading=0, look_at=None, pitch=0, fov=75, radius=50, streetview_format="jpg",
                  full_sphere=False, sphere_zoom=3, workers=WORKERS):
    number(sphere_zoom, "sphere_zoom", 0, 5, whole=True)
    number(workers, "workers", 1, 64, whole=True)
    streetview.angle(heading, "heading")
    streetview.angle(pitch, "pitch")
    fov = number(fov, "fov", 5, 175, whole=True)
    positive(radius, "radius")
    if stops is not None:
        number(stops, "stops", 1, whole=True)
    if step is not None:
        positive(step, "step")
    if look_at is not None:
        point(look_at)

    selected_place, sections, gaps, requested_stops = None, [], [], 1
    if street is not None:
        selected_place = places.choose(client, street, match=match, best_match=best_match, street=True, endpoint=endpoint)
        sections = routes.resolve(client, selected_place, route=route, reverse=reverse, progress=progress)
        samples, gaps, requested_stops = street_samples(client, sections, stops, step, progress, workers)
    elif pano_id:
        if not streetview.PANO_ID.fullmatch(pano_id):
            raise ValueError("Invalid panorama ID.")
        samples = [dict(pano_id=pano_id, stop=1)]
    else:
        requested, selected_place = location(client, at=at, place=place, match=match,
                                             best_match=best_match, endpoint=endpoint)
        views = coverage(client, around(requested, radius * 2), progress, workers)
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
    if sections:
        result["routes"] = [{key: value for key, value in section.items() if key != "points"} for section in sections]
    directions = ("left", "right") if view == "both" else (view,)
    try:
        progress("Street View", 0, len(samples))
        for index, sample in enumerate(samples, 1):
            try:
                metadata = streetview.fetch_metadata(client, sample, full_sphere=full_sphere)
                actual = (metadata["lat"], metadata["lon"])
                if full_sphere:
                    path = folder / f"{len(result['photos']) + 1:03d}_sphere.{streetview_format}"
                    photo = {**sample, **metadata}
                    if "heading" in photo:
                        photo["road_heading"] = photo.pop("heading")
                    photo.update(streetview.save_sphere(client, metadata, sphere_zoom, path, workers), path=str(path))
                    result["photos"].append(photo)
                else:
                    for direction in directions if street else (None,):
                        angle = (sample["heading"] + streetview.VIEW_OFFSETS[direction]) % 360 if street else heading
                        if look_at is not None:
                            angle = bearing(actual, look_at)
                        path = folder / f"{len(result['photos']) + 1:03d}.{streetview_format}"
                        photo = {**sample, **metadata,
                                 **streetview.save_photo(client, sample["pano_id"], angle, fov, pitch, path)}
                        photo.update(heading=angle, pitch=pitch, fov=fov, path=str(path))
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
              endpoint=places.GEOCODER, bbox=None, tile=None, size=200, zoom=19, satellite_format="jpg",
              workers=WORKERS):
    if tile is not None:
        zoom, x, y = tile
    number(zoom, "zoom", 1, 21, whole=True)
    selected_place = None
    if tile is not None:
        number(x, "tile x", 0, 2 ** zoom - 1)
        number(y, "tile y", 0, 2 ** zoom - 1)
        north, west = coordinate(x * 256, y * 256, zoom)
        south, east = coordinate((x + 1) * 256, (y + 1) * 256, zoom)
        area = (south, west, north, east)
    elif bbox is not None:
        area = bounds(bbox)
    else:
        center, selected_place = location(client, at=at, place=place, match=match,
                                          best_match=best_match, endpoint=endpoint)
        area = around(center, size)
    run = capture.plan(client, area, dict(include=["satellite"], satellite_zoom=zoom,
                                        satellite_format=satellite_format, satellite_workers=workers), progress)
    folder = capture.create_folder(run, output)
    try:
        capture.download(run, folder, client, progress)
    except OSError as error:
        raise RequestError("request_failed", str(error), folder=str(folder)) from error
    except KeyboardInterrupt as error:
        raise RequestError("interrupted", "Stopped. Use 'capture resume' to continue this capture.", folder=str(folder)) from error
    # Pixel rounding means actual bounds can extend slightly beyond the request.
    g = run["stages"][0]["grid"]
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
