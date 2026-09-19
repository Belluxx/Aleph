"""Place search, reverse lookup, and nearby OSM points of interest."""

import json
import math
from urllib.parse import urlsplit

from .common import RequestError, url
from .geo import RADIUS, bounds, distance

GEOCODER = "https://photon.komoot.io"
POI_KEYS = ("amenity", "tourism", "shop", "leisure", "historic", "office")


def number(value, name, low, high):
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be from {low} to {high}.")
    return value


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def point(values):
    lat, lon = values
    number(lat, "latitude", -90, 90)
    number(lon, "longitude", -180, 180)
    return lat, lon


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


def osm_data(client, query):
    try:
        data = json.loads(client.overpass(query))
    except (OSError, ValueError) as error:
        raise RequestError("provider_unavailable", f"Overpass request failed: {error}") from error
    if data.get("remark") or not isinstance(data.get("elements"), list):
        raise RequestError("provider_unavailable", "Overpass returned incomplete data; retry later.")
    return data


def geocode(client, *, query=None, at=None, limit=5, street=False, endpoint=GEOCODER):
    number(limit, "limit", 1, 50)
    endpoint = endpoint.rstrip("/")
    address = urlsplit(endpoint)
    if address.scheme not in ("http", "https") or not address.netloc:
        raise ValueError("The geocoder URL must use HTTP or HTTPS.")
    params = dict(limit=limit)
    if query is not None:
        if not query.strip():
            raise ValueError("Enter a nonempty place name.")
        params["q"] = query
        if street:
            params["layer"] = "street"
        path = "/api/"
    else:
        params["lat"], params["lon"] = point(at)
        path = "/reverse/"
    try:
        data = json.loads(client.get(url(endpoint + path, **params), user_agent="Aleph/1.0"))
    except (OSError, ValueError) as error:
        raise RequestError("provider_unavailable", f"Geocoder request failed: {error}") from error
    if not isinstance(data.get("features"), list):
        raise RequestError("provider_unavailable", "The geocoder returned an unreadable response.")
    results, seen = [], set()
    for feature in data["features"]:
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"][:2]
        point((lat, lon))
        kind = {"N": "node", "W": "way", "R": "relation"}[props["osm_type"]]
        identity = f"{kind}/{int(props['osm_id'])}"
        if identity in seen:
            continue
        seen.add(identity)
        name = props.get("name") or props.get("street") or "Unnamed place"
        label = ", ".join(dict.fromkeys(str(v) for v in (
            name, props.get("housenumber"), props.get("city"),
            props.get("state"), props.get("country"),
        ) if v))
        result = dict(id=identity, name=name, label=label, lat=lat, lon=lon,
                      category=f"{props.get('osm_key', 'place')}:{props.get('osm_value', 'unknown')}",
                      source="OpenStreetMap", source_url=f"https://www.openstreetmap.org/{identity}")
        if at is not None:
            result["distance_m"] = round(distance(at, (lat, lon)), 2)
        results.append(result)
    return results


def choose(client, query, *, match=None, street=False, endpoint=GEOCODER):
    candidates = geocode(client, query=query, street=street, endpoint=endpoint, limit=10)
    if not candidates:
        raise RequestError("place_not_found", "No matching place was found. Include a city or country.")
    if match is None and len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate["id"] == match:
            return candidate
    raise RequestError("ambiguous_place", "Choose a returned place with --match TYPE/ID.", candidates=candidates)


def nearby(client, at, *, radius=100, limit=10):
    lat, lon = point(at)
    number(radius, "radius", 1, 5000)
    number(limit, "limit", 1, 50)
    selection = "".join(f'nwr(around:{radius},{lat},{lon})["{key}"];' for key in POI_KEYS)
    data = osm_data(client, f"[out:json][timeout:25];({selection});out center tags;")
    results = []
    for item in data["elements"]:
        tags = item.get("tags", {})
        center = item if item["type"] == "node" else item.get("center")
        if not center:
            continue
        separation = distance(at, point((center["lat"], center["lon"])))
        # Ways/relations are represented by their bounding-box center, not an entrance.
        if separation > radius:
            continue
        identity = f"{item['type']}/{item['id']}"
        categories = [f"{key}:{tags[key]}" for key in POI_KEYS if key in tags]
        results.append(dict(id=identity, name=tags.get("name"), categories=categories,
                            lat=center["lat"], lon=center["lon"], distance_m=round(separation, 2),
                            location_type="point" if item["type"] == "node" else "bbox_center",
                            source="OpenStreetMap", source_url=f"https://www.openstreetmap.org/{identity}"))
    results.sort(key=lambda item: (item["distance_m"], item["id"]))
    return results[:limit]
