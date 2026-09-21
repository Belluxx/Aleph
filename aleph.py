"""Find places, request imagery, and manage area captures."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from src import capture, places, quick
from src.common import CachedClient, Client, Progress, RequestError, now
from src.geo import MERCATOR_RADIUS, bounds


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise RequestError("invalid_arguments", message)


def tile_coordinates(value):
    try:
        values = tuple(int(part) for part in value.split("/"))
        if len(values) == 3:
            return values
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use Z/X/Y, for example 19/280337/194891")


def parser():
    root = Parser(prog="alephgeo", description="Download chunks of the earth with 'capture', get specific data with 'streetview' and 'satellite', search for places with 'resolve'.")
    commands = root.add_subparsers(dest="command", required=True)

    def output(command):
        command.add_argument("-o", "--output", type=Path, default=Path("."), help="parent output directory")

    def json_output(command):
        command.add_argument("--json", action="store_true", help="one JSON result on stdout; progress on stderr")

    def network(command):
        command.add_argument("--geocoder", default=os.environ.get("ALEPH_GEOCODER_URL", places.GEOCODER),
                             metavar="URL", help="Photon server URL (or ALEPH_GEOCODER_URL)")
        command.add_argument("--cache-dir", type=Path,
                             default=Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "aleph",
                             help="response cache directory (default: %(default)s)")
        command.add_argument("--refresh", action="store_true", help="fetch fresh responses instead of cached responses")
        json_output(command)

    def locations(command):
        group = command.add_mutually_exclusive_group(required=True)
        group.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="coordinates")
        group.add_argument("--place", help="place name, including city or country")
        return group

    resolve = commands.add_parser("resolve", help="Find places, reverse geocode, or list nearby POIs")
    source = resolve.add_mutually_exclusive_group(required=True)
    source.add_argument("query", nargs="?", help="place name, including city or country")
    source.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"))
    resolve.add_argument("--nearby", action="store_true", help="list nearby POIs instead of reverse geocoding")
    resolve.add_argument("--radius", type=float, help="nearby POI radius in meters (default: 100; max: 5000)")
    resolve.add_argument("--limit", type=int, default=10, help="maximum results (default: 10; max: 50)")
    resolve.add_argument("--streets", action="store_true", help="restrict a name search to streets")
    network(resolve)

    street = commands.add_parser("streetview", help="Get one view or an ordered street sequence")
    source = locations(street)
    source.add_argument("--pano-id", help="exact panorama ID")
    source.add_argument("--street", help="named street to follow from end to end")
    street.add_argument("--match", metavar="TYPE/ID", help="choose an OSM ID returned for an ambiguous name")
    street.add_argument("--stops", type=int, help="street positions (default: 10)")
    street.add_argument("--view", choices=("forward", "backward", "left", "right", "both"),
                        help="street orientation; both saves left and right (default: forward)")
    street.add_argument("--route", type=int, help="choose a returned branch of a street")
    street.add_argument("--reverse", action="store_true", help="reverse the street traversal order")
    angle = street.add_mutually_exclusive_group()
    angle.add_argument("--heading", type=float, help="single-view heading clockwise from north (default: 0)")
    angle.add_argument("--look-at", nargs=2, type=float, metavar=("LAT", "LON"), help="turn a single view toward this point")
    street.add_argument("--pitch", type=float, default=0, help="vertical angle in degrees (default: 0)")
    street.add_argument("--fov", type=float, default=75, help="horizontal field of view (default: 75)")
    street.add_argument("--radius", type=float, help="maximum panorama search distance in meters (default: 50)")
    street.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    output(street)
    network(street)

    satellite = commands.add_parser("satellite", help="Get a satellite patch or an exact tile")
    source = locations(satellite)
    source.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    source.add_argument("--tile", type=tile_coordinates, metavar="Z/X/Y")
    satellite.add_argument("--match", metavar="TYPE/ID", help="choose an OSM ID returned for an ambiguous name")
    satellite.add_argument("--size", type=float, help="square width in meters around a point (default: 200)")
    satellite.add_argument("--zoom", type=int, help="satellite zoom, 1–21 (default: 19)")
    output(satellite)
    network(satellite)

    captures = commands.add_parser("capture", help="Create, resume, or export an area capture")
    actions = captures.add_subparsers(dest="action", required=True)
    area = actions.add_parser("create", help="Plan and download an area with multiple sources")
    area.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("S", "W", "N", "E"))
    area.add_argument("--sources", dest="include", nargs="+", choices=capture.SOURCES,
                      help="sources to download (default: all; osm includes terrain)")
    output(area)
    json_output(area)
    area.add_argument("--delay", type=float, help="pause between requests in seconds")
    planning = area.add_mutually_exclusive_group()
    planning.add_argument("--plan", action="store_true", help="save a plan without downloading imagery")
    no_plan_help = "start without the confirmation prompt"
    planning.add_argument("--no-plan", action="store_true", help=no_plan_help)
    area.add_argument("--step", type=float, help="target Street View spacing in meters (default: 30)")
    area.add_argument("--fov", type=float, help="horizontal field of view (default: 75)")
    area.add_argument("--depth", choices=("main", "roads", "all"), help="road/path selection (default: roads)")
    area.add_argument("--image-format", choices=("jpg", "png"))
    area.add_argument("--satellite-zoom", type=int, help="satellite zoom, 1–21 (default: 18)")
    area.add_argument("--terrain-zoom", type=int, help="terrain zoom, 1–14 (default: 14)")
    area.set_defaults(**capture.DEFAULT_OPTIONS)
    dashboard = commands.add_parser("dashboard", help="Open the local capture dashboard")
    dashboard.add_argument("--root", type=Path, default=Path("."), help="capture directory (default: current directory)")
    dashboard.add_argument("--port", type=int, metavar="PORT", default=8100,
                           help="local port (default: 8100)")
    dashboard.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    for name, help_text in (
        ("resume", "Continue a saved or planned run"),
        ("export", "Rebuild images and metadata from saved files, offline"),
    ):
        command = actions.add_parser(name, help=help_text, description=help_text)
        command.add_argument(
            "folder", type=Path, help="timestamped run directory containing manifest.json"
        )
        if name == "resume":
            command.add_argument("--no-plan", action="store_true", help=no_plan_help)
        json_output(command)
    return root


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
            message = f"{estimate['streetview_photos']:,} photos at {estimate['streetview_stops']:,} stops"
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
    client = CachedClient(args.cache_dir, refresh=args.refresh)
    if args.command == "resolve":
        if args.nearby and args.at is None:
            raise ValueError("--nearby requires --at LAT LON.")
        if args.radius is not None and not args.nearby:
            raise ValueError("--radius requires --nearby.")
        if args.streets and args.query is None:
            raise ValueError("--streets requires a place name.")
        if args.nearby:
            results = places.nearby(client, args.at, radius=100 if args.radius is None else args.radius, limit=args.limit)
            mode = "nearby"
        else:
            results = places.geocode(client, query=args.query, at=args.at, limit=args.limit if args.query is not None else 1,
                                     street=args.streets, endpoint=args.geocoder)
            mode = "search" if args.query is not None else "reverse"
        result = dict(command="resolve", mode=mode, status="complete", results=results)
    else:
        if args.match and not (args.place or getattr(args, "street", None)):
            raise ValueError("--match requires --place or --street.")
        keys = ("at", "place", "match")
        if args.command == "streetview":
            if args.street is None and (args.stops is not None or args.view is not None
                                       or args.route is not None or args.reverse):
                raise ValueError("--stops, --view, --route, and --reverse require --street.")
            if args.street is not None and (args.heading is not None or args.look_at is not None or args.radius is not None):
                raise ValueError("Use --view for a street sequence; --heading, --look-at, and --radius are for single views.")
            if args.pano_id and args.radius is not None:
                raise ValueError("--radius applies to coordinates or a place name.")
            keys += ("pano_id", "street", "route", "reverse", "stops", "view", "heading",
                     "look_at", "pitch", "fov", "radius", "image_format")
            operation = quick.street_photos
        else:
            if args.size is not None and (args.bbox is not None or args.tile is not None):
                raise ValueError("--size applies to coordinates or a place name.")
            if args.tile is not None and args.zoom is not None:
                raise ValueError("--tile already includes its zoom; omit --zoom.")
            keys += ("bbox", "tile", "size", "zoom")
            operation = quick.satellite
        values = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
        result = operation(client, args.output, progress, endpoint=args.geocoder, **values)
    result["cache"] = client.stats()
    return result


def emit(result, as_json):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif result.get("folder"):
        print(result["folder"])
    elif "results" in result:
        for item in result["results"]:
            separation = f"  {item['distance_m']:.0f} m" if "distance_m" in item else ""
            print(f"{item['id']}  {item.get('label') or item.get('name') or 'Unnamed POI'}"
                  f"  ({item['lat']:.6f}, {item['lon']:.6f}){separation}")
        if not result["results"]:
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
            if result["status"] == "partial":
                print(f"Saved {result['saved_stops']} of {result['requested_stops']} requested stops; see result.json for gaps.", file=sys.stderr)
            emit(result, as_json)
            return 0
        action = args.action
        if action in ("resume", "export"):
            folder = args.folder.expanduser().resolve()
            run = capture.load(folder)
            client = Client(run["options"]["delay"])
        else:
            area = bounds(args.bbox)
            options = {key: getattr(args, key) for key in capture.DEFAULT_OPTIONS}
            client = Client(args.delay)
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
        if folder:
            print(
                f"Saved run: {folder}\nUse 'capture resume' to retry, or 'capture export' to rebuild saved outputs.",
                file=sys.stderr,
            )
        if as_json:
            emit(dict(status="error", error=dict(code=code, message=str(error), **details)), True)
        elif details:
            print(json.dumps(details, ensure_ascii=False, indent=2), file=sys.stderr)
        return {"interrupted": 130, "invalid_arguments": 2}.get(code, 1)


if __name__ == "__main__":
    raise SystemExit(main())
