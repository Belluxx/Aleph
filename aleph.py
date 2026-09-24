"""Find places, request imagery, and manage area captures."""

import json
import math
import os
import shutil
import sys
import textwrap
from pathlib import Path

from src import capture, places, quick
from src.cli import parser
from src.common import DEFAULT_CACHE, CachedClient, Client, Progress, RequestError, now
from src.geo import MERCATOR_RADIUS, bounds


def style(text, code="1"):
    if sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        return f"\033[{code}m{text}\033[0m"
    return text


def describe(run):
    estimate = capture.estimate(run)
    print(style("Capture plan"), file=sys.stderr)
    for stage in run["stages"]:
        mode = stage["mode"]
        if mode == "streetview":
            label = "Street View"
            kind = "spheres" if stage.get("full_sphere") else "photos"
            message = f"{estimate['streetview_photos']:,} {kind} at {estimate['streetview_stops']:,} stops"
        elif mode == "satellite":
            label = "Satellite"
            g = stage["grid"]
            latitude = math.radians((run["bounds"][0] + run["bounds"][2]) / 2)
            scale = 2 * math.pi * MERCATOR_RADIUS * math.cos(latitude) / (256 * 2 ** g["zoom"])
            message = f"{estimate['satellite_tiles']:,} tiles, {g['width']:,} × {g['height']:,} px at about {scale:.2f} m/px"
        else:
            label = "Map & terrain"
            maps = estimate["osm_maps"]
            message = f"{maps:,} OSM {'map' if maps == 1 else 'maps'} and {estimate['terrain_tiles']:,} terrain tiles"
        print(f"  {label:<15}{message}", file=sys.stderr)
    if estimate["seconds"] is None:
        print("  Download time depends on panorama tile counts; progress is shown during capture.", file=sys.stderr)
        return
    minutes, seconds = divmod(estimate["seconds"], 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours} hr {minutes} min" if hours else f"{minutes} min {seconds} sec" if minutes else f"{seconds} sec"
    print(f"  Estimated download time: {style('~' + duration)}", file=sys.stderr)


def proceed():
    print(file=sys.stderr)
    while True:
        print(f"{style('Proceed with capture?')} [y/N] ", end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip().lower()
        except EOFError:
            print(file=sys.stderr)
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please enter y or n.", file=sys.stderr)


def query(args, progress):
    """Dispatch quick requests; reject conflicting options before network access."""
    client = CachedClient(args.cache_dir, delay=args.delay, refresh=args.refresh)
    if args.command == "resolve":
        if args.nearby and args.at is None:
            raise ValueError("--nearby requires --at LAT LON.")
        if args.radius is not None and not args.nearby:
            raise ValueError("--radius requires --nearby.")
        if args.streets and args.query is None:
            raise ValueError("--streets requires a place name.")
        if args.nearby:
            results = places.nearby(client, args.at, radius=100 if args.radius is None else args.radius,
                                    limit=args.limit, progress=progress)
            mode = "nearby"
        else:
            results = places.geocode(client, query=args.query, at=args.at, limit=args.limit if args.query is not None else 1,
                                     street=args.streets, endpoint=args.geocoder)
            mode = "search" if args.query is not None else "reverse"
        result = dict(command="resolve", mode=mode, status="complete", results=results)
    else:
        if (args.match or args.best_match) and not (args.place or getattr(args, "street", None)):
            raise ValueError("--match and --best-match require --place or --street.")
        keys = ("at", "place", "match", "best_match")
        if args.command == "streetview":
            if args.street is None and (args.stops is not None or args.step is not None
                                       or args.view is not None
                                       or args.route is not None or args.reverse):
                raise ValueError("--stops, --step, --view, --route, and --reverse require --street.")
            if args.street is not None and args.stops is None and args.step is None:
                raise ValueError("--street requires --stops or --step.")
            if args.street is not None and (args.heading is not None or args.look_at is not None or args.radius is not None):
                raise ValueError("Use --view for a street sequence; --heading, --look-at, and --radius are for single views.")
            if args.pano_id and args.radius is not None:
                raise ValueError("--radius applies to coordinates or a place name.")
            keys += ("pano_id", "street", "route", "reverse", "stops", "step", "view", "heading",
                     "look_at", "pitch", "fov", "radius", "streetview_format", "full_sphere", "sphere_zoom")
            operation = quick.street_photos
        else:
            if args.size is not None and (args.bbox is not None or args.tile is not None):
                raise ValueError("--size applies to coordinates or a place name.")
            if args.tile is not None and args.zoom is not None:
                raise ValueError("--tile already includes its zoom; omit --zoom.")
            keys += ("bbox", "tile", "size", "zoom", "satellite_format")
            operation = quick.satellite
        values = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
        result = operation(client, args.output, progress, endpoint=args.geocoder, **values)
    result["cache"] = client.stats()
    return result


def rows(headers, values, stream):
    """Align identifiers and wrap descriptions without dropping choices."""
    values = [headers, *values]
    width = max(len(str(key)) for key, _ in values)
    columns = min(100, shutil.get_terminal_size().columns)
    for key, description in values:
        prefix = f"  {key:<{width}}  "
        print(textwrap.fill(" ".join(description.split()), width=max(columns, width + 24),
                            initial_indent=prefix, subsequent_indent=" " * len(prefix)), file=stream)


def coordinates(point):
    return f"{point[0]:.5f}, {point[1]:.5f}"


def place_rows(items, stream, *, locations=False):
    values = []
    for item in items:
        name = item.get("label") or item.get("name") or "Unnamed place"
        categories = [item["category"]] if item.get("category") else item.get("categories", [])
        kind = ", ".join(category.split(":", 1)[-1].replace("_", " ") for category in categories)
        description = name + (f" ({kind})" if kind else "")
        if locations:
            description += f" · {coordinates((item['lat'], item['lon']))}"
            if "distance_m" in item:
                description += f" · {item['distance_m']:.0f} m away"
        values.append((item["id"], description))
    rows(("ID", "PLACE"), values, stream)


def error_details(code, details):
    candidates = details.get("candidates", [])
    if code == "ambiguous_place" and candidates:
        place_rows(candidates, sys.stderr)
    if details.get("gaps"):
        count = len(details["gaps"])
        print(f"  Missing coverage at {count} {'stop' if count == 1 else 'stops'}.", file=sys.stderr)
    if details.get("folder"):
        print(f"Saved files: {details['folder']}", file=sys.stderr)


def emit(result, as_json):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif result.get("folder"):
        if place := result.get("place"):
            print(f"Selected: {place['label']} ({place['id']})")
        if result["command"] == "satellite":
            print(f"Saved satellite image · {result['width']} × {result['height']} px")
            print(result["path"])
        elif result["command"] == "streetview":
            count = len(result["photos"])
            print(f"Saved {count} {'photo' if count == 1 else 'photos'}"
                  f" · {result['saved_stops']}/{result['requested_stops']} stops")
            print(result["photos"][0]["path"] if count == 1 else result["folder"])
            if result["status"] == "partial":
                print(f"Missing stops: see {Path(result['folder']) / 'result.json'}", file=sys.stderr)
        else:
            label = "Capture exported" if result.get("action") == "export" else f"Capture {result['status']}"
            print(f"{label}\n{result['folder']}")
    elif "results" in result:
        if result["results"]:
            place_rows(result["results"], sys.stdout, locations=True)
        else:
            print("No matches.", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    folder = None
    try:
        command = parser()
        if not argv:
            command.print_help()
            return 0
        args = command.parse_args(argv)
        if args.command == "dashboard":
            if not 1 <= args.port <= 65535:
                command.error("--port must be from 1 to 65535")
            from src.dashboard import serve

            serve(args.root, args.port, open_browser=not args.no_browser)
            return 0
        if args.command in ("resolve", "streetview", "satellite"):
            with Progress() as progress:
                result = query(args, progress)
            if as_json and result["status"] == "partial":
                print(f"Saved {result['saved_stops']} of {result['requested_stops']} requested stops; see result.json for gaps.", file=sys.stderr)
            emit(result, as_json)
            return 0
        action = args.action
        if action in ("resume", "export"):
            folder = args.folder.expanduser().resolve()
            run = capture.load(folder)
            client = Client(run["options"]["delay"], cache_dir=getattr(args, "cache_dir", DEFAULT_CACHE),
                            refresh=getattr(args, "refresh", False))
        else:
            area = bounds(args.bbox)
            options = {key: getattr(args, key) for key in capture.DEFAULT_OPTIONS}
            client = Client(args.delay, cache_dir=args.cache_dir, refresh=args.refresh)
            with Progress() as progress:
                run = capture.plan(client, area, options, progress)
        save_only = action == "create" and args.plan
        if action != "export":
            describe(run)
            if not (save_only or args.no_plan) and not proceed():
                print("Cancelled.", file=sys.stderr)
                if as_json:
                    emit(dict(command="capture", action=action, status="cancelled"), True)
                return 0
        if folder is None:
            run["started_at"] = now()
            folder = capture.create_folder(run, args.output)
        capture.save_preview(run, folder)
        if action == "export":
            with Progress() as progress:
                capture.export(run, folder, progress)
        elif save_only:
            print("Plan saved. Use 'capture resume' to download it.", file=sys.stderr)
        else:
            with Progress() as progress:
                capture.download(run, folder, client, progress)
            skipped = sum(
                p.get("status") == "skipped" for stage in run["stages"] for p in stage["results"]
            )
            if skipped:
                print(style(f"Skipped {skipped:,} unavailable Street View {'photo' if skipped == 1 else 'photos'}.", "33"),
                      file=sys.stderr)
        result = dict(command="capture", action=action, status=run["state"], folder=str(folder),
                      manifest=str(folder / "manifest.json"))
        if save_only:
            result["estimate"] = capture.estimate(run)
        emit(result, as_json)
        return 0
    except KeyboardInterrupt:
        print(
            "Stopped."
            + (
                f" Resume with: alephgeo capture resume '{folder}'"
                if folder
                else " Planning can be started again."
            ),
            file=sys.stderr,
        )
        if as_json:
            emit(dict(status="error", error=dict(code="interrupted", message="Stopped."),
                      folder=str(folder) if folder else None), True)
        return 130
    except Exception as error:
        code = error.code if isinstance(error, RequestError) else (
            "invalid_arguments" if isinstance(error, ValueError) else "request_failed")
        details = error.details if isinstance(error, RequestError) else {}
        if folder:
            details = dict(details, folder=str(folder))
        print(f"{style('alephgeo:', '31')} {error}", file=sys.stderr)
        if folder and as_json:
            print(
                f"Saved run: {folder}\nUse 'capture resume' to retry, or 'capture export' to rebuild saved outputs.",
                file=sys.stderr,
            )
        if as_json:
            emit(dict(status="error", error=dict(code=code, message=str(error), **details)), True)
        else:
            error_details(code, details)
            if folder:
                print("Use 'capture resume' to retry, or 'capture export' to rebuild saved outputs.",
                      file=sys.stderr)
        return {"interrupted": 130, "invalid_arguments": 2}.get(code, 1)


if __name__ == "__main__":
    raise SystemExit(main())
