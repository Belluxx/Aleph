"""Command-line arguments, grouped by command and scope."""

import argparse
import os
from pathlib import Path

from . import capture, places
from .common import DEFAULT_CACHE, RequestError


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
    _resolve_command(commands)
    _streetview_command(commands)
    _satellite_command(commands)
    _capture_command(commands)
    _dashboard_command(commands)
    return root


def _output_options(command, *, directory=False):
    if directory:
        command.add_argument("-o", "--output", type=Path, default=Path("."), help="parent output directory")
    command.add_argument("--json", action="store_true", help="one JSON result on stdout; progress on stderr")


def _cache_options(command):
    command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="data cache directory (default: %(default)s)")
    command.add_argument("--refresh", action="store_true", help="refresh responses and Geofabrik regional files")


def _network_options(command):
    command.add_argument("--geocoder", default=os.environ.get("ALEPH_GEOCODER_URL", places.GEOCODER), metavar="URL", help="Photon server URL (or ALEPH_GEOCODER_URL)")
    command.add_argument("--delay", type=float, default=0, help="pause between requests in seconds (default: 0)")
    _cache_options(command)


def _locations(command):
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="coordinates")
    group.add_argument("--place", help="place name, including city or country")

    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--match", metavar="TYPE/ID", help="choose an OSM ID returned for an ambiguous name")
    selection.add_argument("--best-match", action="store_true", help="automatically select the geocoder's highest-ranked place")
    return group


def _resolve_command(commands):
    resolve = commands.add_parser("resolve", help="Find places, reverse geocode, or list nearby POIs")
    source = resolve.add_mutually_exclusive_group(required=True)
    source.add_argument("query", nargs="?", help="place name, including city or country")
    source.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"))

    # Search filters and result limits.
    resolve.add_argument("--streets", action="store_true", help="restrict a name search to streets")
    resolve.add_argument("--nearby", action="store_true", help="list nearby POIs instead of reverse geocoding")
    resolve.add_argument("--radius", type=float, help="nearby POI radius in meters (default: 100; max: 5000)")
    resolve.add_argument("--limit", type=int, default=10, help="maximum results (default: 10; max: 50)")

    _network_options(resolve)
    _output_options(resolve)


def _streetview_command(commands):
    street = commands.add_parser("streetview", help="Get one view or an ordered street sequence")
    source = _locations(street)
    source.add_argument("--pano-id", help="exact panorama ID")
    source.add_argument("--street", help="named street to follow from end to end")

    # Single-view search and orientation.
    street.add_argument("--radius", type=float, help="maximum panorama search distance in meters (default: 50)")
    angle = street.add_mutually_exclusive_group()
    angle.add_argument("--heading", type=float, help="single-view heading clockwise from north, any finite float32 degrees (default: 0)")
    angle.add_argument("--look-at", nargs=2, type=float, metavar=("LAT", "LON"), help="turn a single view toward this point")

    # Street route, sampling, and orientation.
    street.add_argument("--route", type=int, help="choose a returned branch of a street")
    street.add_argument("--reverse", action="store_true", help="reverse the street traversal order")
    street.add_argument("--stops", type=int, help="street positions (default: 10)")
    street.add_argument("--view", choices=("forward", "backward", "left", "right", "both"), help="street orientation; both saves left and right (default: forward)")

    # Camera settings shared by single views and street sequences.
    street.add_argument("--pitch", type=float, default=0, help="vertical angle, positive up and negative down; any finite float32 degrees (default: 0)")
    street.add_argument("--fov", type=int, default=75, help="horizontal field of view, 5–175 whole degrees (default: 75)")

    _network_options(street)
    street.add_argument("--streetview-format", choices=("jpg", "png"), default="jpg")
    _output_options(street, directory=True)


def _satellite_command(commands):
    satellite = commands.add_parser("satellite", help="Get a satellite patch or an exact tile")
    source = _locations(satellite)
    source.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    source.add_argument("--tile", type=tile_coordinates, metavar="Z/X/Y")

    # Image coverage and resolution.
    satellite.add_argument("--size", type=float, help="square width in meters around a point (default: 200)")
    satellite.add_argument("--zoom", type=int, help="satellite zoom, 1–21 (default: 19)")

    _network_options(satellite)
    satellite.add_argument("--satellite-format", choices=("jpg", "png"), default="jpg", help="saved tile format; merged PNG and COG are always produced (default: jpg)")
    _output_options(satellite, directory=True)


def _capture_command(commands):
    captures = commands.add_parser("capture", help="Create, resume, or export an area capture")
    actions = captures.add_subparsers(dest="action", required=True)
    area = actions.add_parser("create", help="Plan and download an area with multiple sources")
    area.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("S", "W", "N", "E"))
    area.add_argument("--sources", dest="include", nargs="+", choices=capture.SOURCES, help="sources to download (default: all; osm includes terrain)")

    # Planning and confirmation.
    planning = area.add_mutually_exclusive_group()
    planning.add_argument("--plan", action="store_true", help="save a plan without downloading imagery")
    no_plan_help = "start without the confirmation prompt"
    planning.add_argument("--no-plan", action="store_true", help=no_plan_help)

    # Street View coverage, camera, and photo format.
    area.add_argument("--depth", choices=("main", "roads", "all"), help="road/path selection (default: roads)")
    area.add_argument("--step", type=float, help="target Street View spacing in meters (default: 30)")
    area.add_argument("--fov", type=int, help="horizontal field of view, 5–175 whole degrees (default: 75)")
    area.add_argument("--streetview-format", choices=("jpg", "png"), help="Street View photo format (default: jpg)")

    # Satellite imagery.
    area.add_argument("--satellite-zoom", type=int, help="satellite zoom, 1–21 (default: 18)")
    area.add_argument("--satellite-format", choices=("jpg", "png"), help="saved satellite tile format; merged PNG and COG are always produced (default: jpg)")

    # Terrain resolution.
    area.add_argument("--terrain-zoom", type=int, help="terrain zoom, 1–14 (default: 14)")

    # Requests, cache, and output shared by all capture sources.
    area.add_argument("--delay", type=float, help="pause between requests in seconds (default: 0)")
    _cache_options(area)
    _output_options(area, directory=True)
    area.set_defaults(**capture.DEFAULT_OPTIONS)

    resume_help = "Continue a saved or planned run"
    resume = actions.add_parser("resume", help=resume_help, description=resume_help)
    resume.add_argument("folder", type=Path, help="timestamped run directory containing manifest.json")
    resume.add_argument("--no-plan", action="store_true", help=no_plan_help)
    _cache_options(resume)
    _output_options(resume)

    export_help = "Rebuild images and metadata from saved files, offline"
    export = actions.add_parser("export", help=export_help, description=export_help)
    export.add_argument("folder", type=Path, help="timestamped run directory containing manifest.json")
    _output_options(export)


def _dashboard_command(commands):
    dashboard = commands.add_parser("dashboard", help="Open the local capture dashboard")
    dashboard.add_argument("--root", type=Path, default=Path("."), help="capture directory (default: current directory)")
    dashboard.add_argument("--port", type=int, metavar="PORT", default=8100, help="local port (default: 8100)")
    dashboard.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
