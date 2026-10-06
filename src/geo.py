"""Spherical road geometry and Web Mercator tile grids. Points are (lat, lon)."""

import math
from bisect import bisect_right
from collections import defaultdict
from itertools import pairwise

import numpy as np

from .common import number, positive

RADIUS = 6_371_008.8
MERCATOR_RADIUS = 6_378_137
MAX_LATITUDE = 85.0511287798066
MAX_VERTICES = 1000
DETAILED_GRID = 64  # Rows and columns up to which lines() draws every tile.


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
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta / 2) ** 2
    return 2 * RADIUS * math.asin(math.sqrt(min(1, h)))


def bearing(a, b):
    lat1, lat2 = map(math.radians, (a[0], b[0]))
    delta = math.radians(b[1] - a[1])
    y = math.sin(delta) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta)
    return math.degrees(math.atan2(y, x)) % 360


def destination(point, heading, meters):
    lat, lon = map(math.radians, point)
    angle, heading = meters / RADIUS, math.radians(heading)
    end = math.asin(clamp(math.sin(lat) * math.cos(angle) + math.cos(lat) * math.sin(angle) * math.cos(heading), -1, 1))
    delta = math.atan2(math.sin(heading) * math.sin(angle) * math.cos(lat),
                       math.cos(angle) - math.sin(lat) * math.sin(end))
    return math.degrees(end), (math.degrees(lon + delta) + 180) % 360 - 180


def point(values):
    lat, lon = values
    return number(lat, "latitude", -90, 90), number(lon, "longitude", -180, 180)


def bounds(values, *, minimum=1):
    lat1, lon1, lat2, lon2 = values
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite numbers.")
    south, north = sorted((lat1, lat2))
    west, east = sorted((lon1, lon2))
    if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
        raise ValueError("Use latitude −90…90 and longitude −180…180.")
    if east - west > 180:
        raise ValueError("Split areas crossing the date line in two.")
    middle = (south + north) / 2
    size = min(distance((middle, west), (middle, east)), distance((south, west), (north, west)))
    if size <= 0 or size < minimum:
        raise ValueError(f"The area must be nonempty and at least {minimum} meter(s) wide and high.")
    return south, west, north, east


def corners(area):
    """A rectangle as a polygon, counterclockwise from its southwest corner."""
    south, west, north, east = area
    return [(south, west), (south, east), (north, east), (north, west)]


def crosses(a, b, c, d):
    """Whether segments a–b and c–d cross at a point inside both."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def vertices(values, *, minimum=1):
    """A simple polygon of (lat, lon) points, without repeating the first one at the end."""
    polygon = []
    for value in values:
        vertex = point(value)
        if not polygon or vertex != polygon[-1]:
            polygon.append(vertex)
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()
    if not 3 <= len(polygon) <= MAX_VERTICES:
        raise ValueError(f"A polygon needs 3 to {MAX_VERTICES} distinct points.")
    bounds(extent(polygon), minimum=minimum)
    ring = [*polygon, polygon[0]]
    sides = list(pairwise(ring))
    for i, (a, b) in enumerate(sides):
        # Neighbors share a point, including the last and first sides.
        for c, d in sides[i + 2:len(sides) - (i == 0)]:
            if crosses(a, b, c, d):
                raise ValueError("The polygon's edges must not cross.")
    if not signed_area(ring):
        raise ValueError("The polygon must enclose an area.")
    return polygon


def within(polygon, lats, lons):
    """ring_contains for arrays of latitudes and longitudes."""
    inside = np.zeros(np.shape(lats), bool)
    for (alat, alon), (blat, blon) in pairwise([*polygon, polygon[0]]):
        if alat != blat:
            inside ^= ((alat > lats) != (blat > lats)) & (lons < alon + (lats - alat) * (blon - alon) / (blat - alat))
    return inside


def pieces(a, b, polygon):
    """Parts of segment a–b inside the polygon, in order from a."""
    ring = [*polygon, polygon[0]]
    dlat, dlon = b[0] - a[0], b[1] - a[1]
    cuts = [0, 1]
    for c, d in pairwise(ring):
        elat, elon = d[0] - c[0], d[1] - c[1]
        if denominator := dlat * elon - dlon * elat:
            t = ((c[0] - a[0]) * elon - (c[1] - a[1]) * elat) / denominator
            u = ((c[0] - a[0]) * dlon - (c[1] - a[1]) * dlat) / denominator
            if 0 < t < 1 and 0 <= u <= 1:
                cuts.append(t)

    def at(t):
        return a if t == 0 else b if t == 1 else (a[0] + t * dlat, a[1] + t * dlon)

    parts = []
    for start, end in pairwise(sorted(cuts)):
        if start < end and ring_contains(ring, at((start + end) / 2)):
            if parts and parts[-1][1] == start:
                parts[-1][1] = end
            else:
                parts.append([start, end])
    return [(at(start), at(end)) for start, end in parts if distance(at(start), at(end)) > 0.01]


def offset(point, origin):
    """Meters east and north of a nearby origin, on a plane."""
    scale = math.radians(RADIUS)
    return (((point[1] - origin[1] + 180) % 360 - 180) * scale * math.cos(math.radians(origin[0])),
            (point[0] - origin[0]) * scale)


def near(polygon, point, meters):
    """Whether a point lies inside the polygon or within a short distance of its edges."""
    ring = [*polygon, polygon[0]]
    if ring_contains(ring, point):
        return True
    for (ax, ay), (bx, by) in pairwise([offset(p, point) for p in ring]):
        dx, dy = bx - ax, by - ay
        t = clamp(-(ax * dx + ay * dy) / (dx * dx + dy * dy), 0, 1) if dx or dy else 0
        if math.hypot(ax + t * dx, ay + t * dy) <= meters:
            return True
    return False


def measure(polygon):
    """Approximate ground area in square meters and perimeter in meters."""
    origin = sum(p[0] for p in polygon) / len(polygon), polygon[0][1]
    ring = [offset(p, origin) for p in [*polygon, polygon[0]]]
    return abs(signed_area(ring)), sum(math.dist(a, b) for a, b in pairwise(ring))


def around(center, size):
    """A square in ground meters, returned in south/west/north/east order."""
    lat, lon = point(center)
    positive(size, "size")
    dy = math.degrees(size / (2 * RADIUS))
    if abs(lat) + dy >= 90:
        raise ValueError("The area cannot cross a pole.")
    dx = dy / math.cos(math.radians(lat))
    if lon - dx < -180 or lon + dx > 180:
        raise ValueError("Split areas crossing the date line into two rectangles.")
    return bounds((lat - dy, lon - dx, lat + dy, lon + dx), minimum=0)


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
    start, end = [(clamp(a[0] + t * dy, south, north), clamp(a[1] + t * dx, west, east)) for t in (enter, leave)]
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
            if (nearest is None or separation < nearest[1] - 0.1
                    or (abs(separation - nearest[1]) <= 0.1 and meters < nearest[0])):
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
                dx = 180 if abs(lat) + dy >= 90 else math.degrees(
                    math.asin(clamp(math.sin(angle) / math.cos(math.radians(lat)), -1, 1)))
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
    return (point[1] + 180) / 360 * size, clamp((1 - math.asinh(math.tan(latitude)) / math.pi) / 2 * size, 0, size)


def coordinate(x, y, zoom):
    size = 256 * 2**zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / size)))), x / size * 360 - 180


def grid(polygon, zoom):
    """Tiles meeting the polygon and the pixels enclosing it; a rectangle meets every tile of its grid.

    Spans list those tiles in download order: [row, first column, column after the last, tiles in earlier spans].
    """
    south, west, north, east = extent(polygon)
    if south < -MAX_LATITUDE or north > MAX_LATITUDE:
        raise ValueError("Imagery areas must stay within latitudes −85.0511…85.0511.")

    # Projection roundoff grows with the world pixel size at high zooms.
    tolerance = max(1e-7, 8 * math.ulp(256 * 2**zoom))

    def snap(value):
        return round(value) if abs(value - round(value)) < tolerance else value

    left, top = (math.floor(snap(v)) for v in pixel((north, west), zoom))
    right, bottom = (math.ceil(snap(v)) for v in pixel((south, east), zoom))
    x0, y0 = left // 256, top // 256
    columns, rows = math.ceil(right / 256) - x0, math.ceil(bottom / 256) - y0
    points = [tuple(snap(v) / 256 - origin for v, origin in zip(pixel(p, zoom), (x0, y0))) for p in polygon]
    spans, count = [], 0
    for row in range(rows):
        for start, end in meeting(points, row, columns):
            spans.append([row, start, end, count])
            count += end - start
    return dict(zoom=zoom, left=left, top=top, width=right - left, height=bottom - top,
                x0=x0, y0=y0, columns=columns, rows=rows, spans=spans, count=count)


def meeting(points, row, columns):
    """Column ranges of a row's tiles that meet a polygon in tile units, ignoring touches along tile edges."""
    ranges, crossings = [], []
    for (ax, ay), (bx, by) in pairwise([*points, points[0]]):
        # Part of the edge within the row.
        if ay == by:
            if row < ay < row + 1:
                ranges.append((min(ax, bx), max(ax, bx)))
            continue
        low, high = sorted(((row - ay) / (by - ay), (row + 1 - ay) / (by - ay)))
        low, high = max(low, 0), min(high, 1)
        if low < high:
            ranges.append(tuple(sorted((ax + low * (bx - ax), ax + high * (bx - ax)))))
        # Tiles with no edges inside lie between edges crossing the row's middle.
        if (ay > row + 0.5) != (by > row + 0.5):
            crossings.append(ax + (row + 0.5 - ay) * (bx - ax) / (by - ay))
    crossings.sort()
    ranges += zip(crossings[::2], crossings[1::2])
    runs = []
    for start, end in sorted((max(math.floor(low), 0), min(math.ceil(high), columns)) for low, high in ranges):
        if start >= end:
            continue
        if runs and start <= runs[-1][1]:
            runs[-1] = runs[-1][0], max(end, runs[-1][1])
        else:
            runs.append((start, end))
    return runs


def tiles(grid):
    for row, start, end, _ in grid["spans"]:
        for column in range(start, end):
            yield dict(x=grid["x0"] + column, y=grid["y0"] + row, zoom=grid["zoom"], row=row, column=column)


def lines(grid):
    """Edges of a grid's tiles as ((x, y), (x, y)) in tile units, or just their outline for large grids."""
    detailed = grid["rows"] <= DETAILED_GRID and grid["columns"] <= DETAILED_GRID
    x0, y0 = grid["x0"], grid["y0"]
    rows = [[] for _ in range(grid["rows"] + 2)]  # Empty rows above and below.
    segments = []
    for row, start, end, _ in grid["spans"]:
        rows[row + 1].append((start, end))
        for column in range(start, end + 1) if detailed else (start, end):
            segments.append(((x0 + column, y0 + row), (x0 + column, y0 + row + 1)))
    for row, (above, below) in enumerate(pairwise(rows)):
        # Every tile's top and bottom, or only where coverage changes between rows.
        ends = sorted({x for run in above + below for x in run})
        for low, high in pairwise(ends):
            depth = sum(start <= low < end for start, end in above + below)
            if depth == 1 or detailed and depth:
                segments.append(((x0 + low, y0 + row), (x0 + high, y0 + row)))
    return segments


def tile_ring(tile):
    north, west = coordinate(tile["x"] * 256, tile["y"] * 256, tile["zoom"])
    south, east = coordinate((tile["x"] + 1) * 256, (tile["y"] + 1) * 256, tile["zoom"])
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def feature(geometry, coordinates, properties):
    return dict(type="Feature", geometry=dict(type=geometry, coordinates=coordinates), properties=properties)


def collection(features):
    return dict(type="FeatureCollection", features=list(features))
