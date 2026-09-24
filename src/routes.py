"""Join named OSM ways into ordered, unbranched street routes."""

import math
from collections import defaultdict
from itertools import pairwise

from . import streetview
from .common import RequestError
from .geo import Line, RoadIndex, distance


def chains(ways):
    """Split at real OSM junctions; never join roads merely because they cross."""
    points, edges, neighbors = {}, {}, defaultdict(set)
    for way in sorted(ways, key=lambda item: item["id"]):
        nodes = way["nodes"]
        points.update(zip(nodes, way["points"]))
        for a, b in pairwise(nodes):
            if a == b or distance(points[a], points[b]) < 0.01:
                continue
            edge = tuple(sorted((a, b)))
            edges.setdefault(edge, set()).add(way["id"])
            neighbors[a].add(b)
            neighbors[b].add(a)
    remaining = set(edges)
    routes = []

    def walk(start, following):
        nodes, ids = [start], set()
        previous, current = start, following
        while True:
            edge = tuple(sorted((previous, current)))
            if edge not in remaining:
                break
            remaining.remove(edge)
            ids.update(edges[edge])
            nodes.append(current)
            if current == start or len(neighbors[current]) != 2:
                break
            following = next(node for node in neighbors[current] if node != previous)
            previous, current = current, following
        coordinates = [points[node] for node in nodes]
        if coordinates[-1] < coordinates[0]:
            coordinates.reverse()
        routes.append(dict(points=coordinates, way_ids=sorted(ids), length_m=Line(coordinates).length))

    for start in sorted(neighbors, key=lambda node: (points[node], node)):
        if len(neighbors[start]) != 2:
            for following in sorted(neighbors[start]):
                if tuple(sorted((start, following))) in remaining:
                    walk(start, following)
    while remaining:  # Closed loops have no endpoints.
        a, b = min(remaining, key=lambda edge: (points[edge[0]], points[edge[1]]))
        walk(a, b)
    routes.sort(key=lambda route: (route["points"][0], route["points"][-1], route["way_ids"]))
    return routes


def describe(route, index):
    return dict(route=index, start=list(route["points"][0]), end=list(route["points"][-1]),
                length_m=round(route["length_m"], 2), way_ids=route["way_ids"],
                closed=route["points"][0] == route["points"][-1])


def sampling(route, *, step=None, stops=None):
    """Normalize a sampling request to meter spacing and target distances."""
    line = Line(route["points"])
    closed = line.points[0] == line.points[-1]
    if stops == 1:
        return line.length, [line.length / 2]
    if stops is not None:
        intervals = stops if closed else stops - 1
    else:
        # Fit complete intervals without exceeding the requested step.
        intervals = max(1, math.ceil(line.length / step))
    step = line.length / intervals
    targets = [(i + 0.5) * step for i in range(intervals)] if closed else [
        i * step for i in range(intervals)] + [line.length]
    return step, targets


def resolve(client, place, *, route=None, reverse=False, progress=lambda *args: None):
    if not place["id"].startswith("way/"):
        raise RequestError("place_not_found", "Choose a street represented by an OSM way.")
    name, ways = client.maps.street(place, progress)
    ways = [way for way in ways if way["tags"].get("area") != "yes"
            and not streetview.EXCLUDED.fullmatch(way["tags"]["highway"])]
    options = chains(ways)
    if not options:
        raise RequestError("place_not_found", "No usable street geometry was found.")
    if route is None and len(options) > 1:
        raise RequestError("ambiguous_route", "The street branches. Choose an ordered section with --route N.",
                           candidates=[describe(item, i) for i, item in enumerate(options, 1)])
    choice = 1 if route is None else route
    if not 1 <= choice <= len(options):
        raise ValueError(f"--route must be from 1 to {len(options)}.")
    selected = options[choice - 1]
    if reverse:
        selected["points"].reverse()
    selected.update(describe(selected, choice), name=name, scope="connected_same_name_ways")
    return selected


def match_views(views, route, context, step, targets):
    """Keep distinct panoramas on the selected street, in target-distance order."""
    line = Line(route["points"])
    lines = [Line(part["points"]) for part in context]
    index = RoadIndex(lines)
    candidates = []
    for view in views:
        location = (view["lat"], view["lon"])
        road = streetview.match_road(context, lines, index, location, step)
        if road is None or context[road["path_index"]]["osm_id"] not in route["way_ids"]:
            continue
        meters, separation = line.project(location)
        if separation <= 30:
            candidates.append(dict(view, path_meters=meters, road_distance_m=separation,
                                   heading=line.heading(meters, step)))
    # Coverage can include different panorama IDs at the same physical position.
    unique = []
    for candidate in sorted(candidates, key=lambda item: (item["path_meters"], item["road_distance_m"], item["pano_id"])):
        if not unique or candidate["path_meters"] - unique[-1]["path_meters"] >= 0.1:
            unique.append(candidate)
    selected, gaps, used = [], [], set()
    for stop, target in enumerate(targets, 1):
        eligible = [view for view in unique if view["pano_id"] not in used
                    and abs(view["path_meters"] - target) <= step / 2 + 0.01]
        if eligible:
            chosen = min(eligible, key=lambda view: (abs(view["path_meters"] - target),
                                                    view["road_distance_m"], view["pano_id"]))
            used.add(chosen["pano_id"])
            selected.append(dict(chosen, stop=stop, target_meters=target,
                                 requested_location=list(line.at(target))))
        else:
            gaps.append(dict(stop=stop, target_meters=target, reason="no_coverage"))
    selected.sort(key=lambda view: view["path_meters"])
    return selected, gaps
