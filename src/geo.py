"""Spherical road geometry and Web Mercator tile grids. Points are (lat, lon)."""

import math
from bisect import bisect_right
from collections import defaultdict
from itertools import pairwise

RADIUS = 6_371_008.8
MERCATOR_RADIUS = 6_378_137
MAX_LATITUDE = 85.0511287798066


def clamp(value, low, high):
    return max(low, min(high, value))


def ring_contains(ring, point):
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in pairwise(ring):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def signed_area(ring):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in pairwise(ring)) / 2


def extent(points):
    """Bounding box in south/west/north/east order, including degenerate boxes."""
    latitudes, longitudes = zip(*points)
    return min(latitudes), min(longitudes), max(latitudes), max(longitudes)


def distance(a, b):
    lat1, lat2 = map(math.radians, (a[0], b[0]))
    delta = math.radians(b[1] - a[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta / 2) ** 2
    )
    return 2 * RADIUS * math.asin(math.sqrt(min(1, h)))


def bearing(a, b):
    lat1, lat2 = map(math.radians, (a[0], b[0]))
    delta = math.radians(b[1] - a[1])
    return (
        math.degrees(
            math.atan2(
                math.sin(delta) * math.cos(lat2),
                math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta),
            )
        )
        % 360
    )


def destination(point, heading, meters):
    lat, lon = map(math.radians, point)
    angle, heading = meters / RADIUS, math.radians(heading)
    end = math.asin(
        clamp(
            math.sin(lat) * math.cos(angle) + math.cos(lat) * math.sin(angle) * math.cos(heading),
            -1,
            1,
        )
    )
    delta = math.atan2(
        math.sin(heading) * math.sin(angle) * math.cos(lat),
        math.cos(angle) - math.sin(lat) * math.sin(end),
    )
    return math.degrees(end), (math.degrees(lon + delta) + 180) % 360 - 180


def bounds(values, *, minimum=1):
    lat1, lon1, lat2, lon2 = values
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite numbers.")
    south, north = sorted((lat1, lat2))
    west, east = sorted((lon1, lon2))
    if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
        raise ValueError("Use latitude −90…90 and longitude −180…180.")
    if east - west > 180:
        raise ValueError("Split areas crossing the date line into two rectangles.")
    middle = (south + north) / 2
    size = min(distance((middle, west), (middle, east)), distance((south, west), (north, west)))
    if size <= 0 or size < minimum:
        raise ValueError(f"The rectangle must be nonempty and at least {minimum} meter(s) wide and high.")
    return south, west, north, east


def inside(point, area):
    south, west, north, east = area
    return south - 1e-9 <= point[0] <= north + 1e-9 and west - 1e-9 <= point[1] <= east + 1e-9


def clip(a, b, area):
    """Liang–Barsky clipping also retains segments with both ends outside."""
    south, west, north, east = area
    dy, dx = b[0] - a[0], b[1] - a[1]
    enter, leave = 0, 1
    for p, q in ((-dx, a[1] - west), (dx, east - a[1]), (-dy, a[0] - south), (dy, north - a[0])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            enter = max(enter, q / p)
        else:
            leave = min(leave, q / p)
        if enter > leave:
            return None
    start, end = [
        (clamp(a[0] + t * dy, south, north), clamp(a[1] + t * dx, west, east))
        for t in (enter, leave)
    ]
    return (start, end) if distance(start, end) > 0.01 else None


class Line:
    def __init__(self, points):
        self.points = points
        self.cumulative = [0]
        self.headings = []
        for a, b in pairwise(points):
            self.cumulative.append(self.cumulative[-1] + distance(a, b))
            self.headings.append(bearing(a, b))
        self.length = self.cumulative[-1]

    def at(self, meters):
        meters = clamp(meters, 0, self.length)
        i = min(bisect_right(self.cumulative, meters) - 1, len(self.headings) - 1)
        return destination(self.points[i], self.headings[i], meters - self.cumulative[i])

    def heading(self, meters, spacing):
        window = min(5, spacing / 4, self.length / 4)
        before, after = self.at(meters - window), self.at(meters + window)
        if distance(before, after) < 0.01:
            before = self.at(meters)
        return bearing(before, after)

    def project(self, point, segments=None):
        nearest = None
        for i in range(len(self.headings)) if segments is None else sorted(segments):
            angle = distance(self.points[i], point) / RADIUS
            delta = math.radians(bearing(self.points[i], point) - self.headings[i])
            offset = RADIUS * math.atan2(math.sin(angle) * math.cos(delta), math.cos(angle))
            meters = clamp(self.cumulative[i] + offset, self.cumulative[i], self.cumulative[i + 1])
            separation = distance(self.at(meters), point)
            if (
                nearest is None
                or separation < nearest[1] - 0.1
                or (abs(separation - nearest[1]) <= 0.1 and meters < nearest[0])
            ):
                nearest = meters, separation
        return nearest


class RoadIndex:
    """A small spatial grid; long segments live in a separate overflow list."""

    def __init__(self, lines, radius=30):
        self.size = math.degrees(max(50, radius) / RADIUS)
        self.cells = defaultdict(list)
        self.wide = []
        for road, line in enumerate(lines):
            for segment, heading in enumerate(line.headings):
                length = line.cumulative[segment + 1] - line.cumulative[segment]
                lat, lon = destination(line.points[segment], heading, length / 2)
                angle = (length / 2 + radius + 0.2) / RADIUS
                dy = math.degrees(angle)
                dx = (
                    180
                    if abs(lat) + dy >= 90
                    else math.degrees(
                        math.asin(clamp(math.sin(angle) / math.cos(math.radians(lat)), -1, 1))
                    )
                )
                x0, x1 = (math.floor(x / self.size) for x in (lon - dx, lon + dx))
                y0, y1 = (math.floor(y / self.size) for y in (lat - dy, lat + dy))
                entry = road, segment
                if (x1 - x0 + 1) * (y1 - y0 + 1) > 256 or lon - dx < -180 or lon + dx > 180:
                    self.wide.append(entry)
                else:
                    for x in range(x0, x1 + 1):
                        for y in range(y0, y1 + 1):
                            self.cells[x, y].append(entry)

    def near(self, point):
        key = math.floor(point[1] / self.size), math.floor(point[0] / self.size)
        groups = defaultdict(list)
        for road, segment in self.cells.get(key, []) + self.wide:
            groups[road].append(segment)
        return groups


def pixel(point, zoom):
    size = 256 * 2**zoom
    latitude = math.radians(clamp(point[0], -MAX_LATITUDE, MAX_LATITUDE))
    return (point[1] + 180) / 360 * size, clamp(
        (1 - math.asinh(math.tan(latitude)) / math.pi) / 2 * size, 0, size
    )


def coordinate(x, y, zoom):
    size = 256 * 2**zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / size)))), x / size * 360 - 180


def grid(area, zoom):
    south, west, north, east = area
    if south < -MAX_LATITUDE or north > MAX_LATITUDE:
        raise ValueError("Imagery areas must stay within latitudes −85.0511…85.0511.")

    # Projection roundoff grows with the world pixel size at high zooms.
    tolerance = max(1e-7, 8 * math.ulp(256 * 2**zoom))

    def snap(value):
        return round(value) if abs(value - round(value)) < tolerance else value

    left, top = (math.floor(snap(v)) for v in pixel((north, west), zoom))
    right, bottom = (math.ceil(snap(v)) for v in pixel((south, east), zoom))
    return dict(
        zoom=zoom,
        left=left,
        top=top,
        width=right - left,
        height=bottom - top,
        x0=left // 256,
        y0=top // 256,
        columns=math.ceil(right / 256) - left // 256,
        rows=math.ceil(bottom / 256) - top // 256,
    )


def tiles(grid):
    for row in range(grid["rows"]):
        for column in range(grid["columns"]):
            yield dict(
                x=grid["x0"] + column, y=grid["y0"] + row, zoom=grid["zoom"], row=row, column=column
            )


def tile_ring(tile):
    north, west = coordinate(tile["x"] * 256, tile["y"] * 256, tile["zoom"])
    south, east = coordinate((tile["x"] + 1) * 256, (tile["y"] + 1) * 256, tile["zoom"])
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def feature(geometry, coordinates, properties):
    return dict(
        type="Feature", geometry=dict(type=geometry, coordinates=coordinates), properties=properties
    )


def collection(features):
    return dict(type="FeatureCollection", features=list(features))
