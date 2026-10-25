# CLI reference

```sh
alephgeo COMMAND [OPTIONS]
```

| Command | What it does |
| --- | --- |
| [`resolve`](#resolve) | Search places, reverse geocode coordinates, list nearby points of interest |
| [`streetview`](#streetview) | Download one Street View photo or a sequence along a street |
| [`satellite`](#satellite) | Download a satellite image of a point, rectangle, or tile |
| [`capture`](#capture) | Download everything inside an area: Street View, satellite, OSM, terrain, 3D mesh |
| [`dashboard`](#dashboard) | Draw, capture, and explore areas in the browser |

Run `alephgeo COMMAND --help` for the full option list. See the [README](../README.md#install) to install.

## Basics

- **Coordinates** are `LAT LON` in decimal degrees. Rectangles are `S W N E`. Distances are in meters, angles in degrees clockwise from north.
- **Place names** work anywhere `--place` or `--street` is accepted. Include a city or country. If the name is ambiguous, Aleph lists the matches; pick one with `--match TYPE/ID` (for example `--match relation/1834818`) or take the top result with `--best-match`.
- **Output**: `-o DIR` sets the parent folder (default: current directory). Every download goes into a new `aleph-*` folder inside it.
- **Scripting**: `--json` prints one JSON result on stdout. See [Scripting](#scripting).

## resolve

```sh
alephgeo resolve "Colosseum, Rome"                          # search by name
alephgeo resolve "Via del Corso, Rome" --streets            # streets only
alephgeo resolve --at 41.8902 12.4922                       # reverse geocode
alephgeo resolve --at 41.8902 12.4922 --nearby --radius 200 # nearby POIs
```

| Option | Description |
| --- | --- |
| `QUERY` or `--at LAT LON` | Name to search, or coordinates to look up |
| `--streets` | Only return streets (name search) |
| `--nearby` | List points of interest around `--at` instead of the address |
| `--radius M` | Nearby search radius, up to 5000 (default: 100) |
| `--limit N` | Maximum results, up to 50 (default: 10) |

Results show the ID to pass to `--match`.

## streetview

### Single view

```sh
alephgeo streetview --place "Colosseum, Rome" --best-match
alephgeo streetview --at 41.8909 12.4919 --heading 180 --pitch 10 --fov 60
alephgeo streetview --at 41.8909 12.4919 --look-at 41.8902 12.4922
alephgeo streetview --pano-id PANO_ID
```

Aleph picks the nearest panorama within `--radius` (default: 50 m) of the point.

| Option | Description |
| --- | --- |
| `--at`, `--place`, `--pano-id` | Where to look: coordinates, place name, or exact panorama ID |
| `--radius M` | Maximum distance to the panorama (default: 50) |
| `--heading DEG` | Camera direction: 0 north, 90 east, 180 south, 270 west (default: 0) |
| `--look-at LAT LON` | Turn the camera toward a point instead of using `--heading` |

### Along a street

```sh
# 10 photos spread along the whole street, looking forward
alephgeo streetview --street "Via del Corso, Rome" --best-match --stops 10

# A stop every 10 m, photographing both sides
alephgeo streetview --street "Via del Corso, Rome" --best-match --step 10 --view both
```

| Option | Description |
| --- | --- |
| `--street NAME` | Street to follow from end to end |
| `--stops N` or `--step M` | Total number of stops, or target spacing between them (one is required) |
| `--view` | `forward` (default), `backward`, `left`, `right`, or `both` (left and right) |
| `--route N` | Only capture section `N`; sections are numbered in `result.json` (default: all) |
| `--reverse` | Walk each section in the opposite direction |

Stops without coverage are skipped and listed as gaps in `result.json`.

### Camera and format

These apply to both modes.

| Option | Description |
| --- | --- |
| `--pitch DEG` | Vertical angle, positive up (default: 0) |
| `--fov DEG` | Horizontal field of view, 5–175 (default: 75) |
| `--full-sphere` | Save the full 360° panorama instead of a photo; heading, pitch, and fov are ignored |
| `--sphere-zoom 0–5` | Panorama resolution, capped at the highest available (default: 3) |
| `--streetview-format` | `jpg` (default) or `png` |
| `--workers N` | Parallel requests, 1–64 (default: 8) |

Output: numbered photos (`001.jpg`, …) and `result.json` with each photo's panorama ID, location, date, and camera settings.

## satellite

```sh
alephgeo satellite --place "Colosseum, Rome" --best-match           # 200 m square
alephgeo satellite --at 41.8902 12.4922 --size 500 --zoom 20        # 500 m square
alephgeo satellite --bbox 41.8895 12.4910 41.8910 12.4940           # rectangle
alephgeo satellite --tile 19/280337/194891                          # one map tile
```

| Option | Description |
| --- | --- |
| `--at`, `--place`, `--bbox S W N E`, `--tile Z/X/Y` | Area to download |
| `--size M` | Width of the square around `--at` or `--place` (default: 200) |
| `--zoom 1–21` | Detail level; 19 is about 0.3 m/px at the equator (default: 19). Not used with `--tile` |
| `--tile-format` | Format of the saved tiles: `jpg` (default) or `png` |
| `--workers N` | Parallel requests, 1–64 (default: 8) |

Output: `satellite.png`, `satellite.tif` (lossless Cloud Optimized GeoTIFF, EPSG:3857), `result.json` with the exact bounds, and the source tiles. Missing imagery is transparent.

## capture

Download several sources for one area into a single folder that can be resumed and rebuilt.

### capture create

```sh
# Rectangle with the default sources
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures

# Any polygon: corners in order as LAT LON pairs (edges must not cross)
alephgeo capture create --polygon 41.8895 12.4910 41.8910 12.4915 41.8905 12.4940 41.8893 12.4932

# Only the 3D mesh and satellite image
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 --sources mesh satellite
```

Aleph prints a plan with a time estimate and asks before downloading. `--yes` skips the prompt; `--plan` saves the plan without downloading, to start later with `capture resume`.

`--sources` picks any of:

| Source | Output | Settings |
| --- | --- | --- |
| `streetview` | Photos along the roads in the area, both sides | `--depth`: `main` roads, `roads` (default), or `all` paths<br>`--step M`: spacing (default: 30)<br>`--fov`, `--full-sphere`, `--sphere-zoom`, `--streetview-format` as in [streetview](#camera-and-format) |
| `satellite` | `satellite.png` and `satellite.tif` | `--satellite-zoom 1–21` (default: 18)<br>`--tile-format jpg\|png` |
| `osm` | `map.osm`: buildings, roads, water, and all other OpenStreetMap data | — |
| `terrain` | `terrain.tif`: elevation in meters, EPSG:3857 | `--terrain-zoom 1–14` (default: 14) |
| `mesh` | `mesh.glb`: Google Earth's textured 3D model | `--mesh-level 1–22` (default: 21) |

The first four are the default; `mesh` must be requested.

Everything stays inside the polygon (satellite pixels outside it are transparent) except terrain, which covers the bounding box.

**Mesh details.** Each level halves the texel size: 22 is about 4 cm per texel and needs about four times as many downloads as 21. Coordinates are meters from the center of the bounding box (x east, y up, z south), with the lowest point at y = 0. The model is split into objects of up to 64 tiles, each keeping its original texture.

**Speed.** `--streetview-workers`, `--satellite-workers`, `--terrain-workers`, and `--mesh-workers` set parallel requests per source, 1–64 (default: 8). `--delay SECONDS` pauses between requests and makes them sequential.

### capture resume

```sh
alephgeo capture resume captures/aleph-20261025T101500Z-abc123
```

Continues a stopped capture or starts a saved plan. Already downloaded files are kept. Accepts `--yes`.

### capture export

```sh
alephgeo capture export captures/aleph-20261025T101500Z-abc123
```

Rebuilds the merged images, mesh, and metadata from the downloaded files, offline.

### Capture folder

```
aleph-20261025T101500Z-abc123/
├── manifest.json       settings, sources, imagery dates, coverage notes
├── README.txt          file guide and data credits
├── satellite.png/.tif  merged satellite image
├── terrain.tif         merged elevation
├── mesh.glb            3D model
├── map.osm             OpenStreetMap data
├── streetview/         photos/, photos.geojson, paths.geojson, plan.svg
├── satellite/          patches/, patches.geojson, plan.svg
├── terrain/tiles/      source tiles, kept for resume and export
└── mesh/nodes/         source nodes, kept for resume and export
```

## dashboard

```sh
alephgeo dashboard -o captures
```

Opens `http://127.0.0.1:8100` in your browser. Draw a rectangle or polygon, choose sources and settings, check the preview and estimate, and start the capture. Open any capture in `captures`, including ones made with `capture create`, to toggle layers, explore the mesh or terrain in 3D, or resume it. Quitting during a capture stops it so it can be resumed later.

| Option | Description |
| --- | --- |
| `-o DIR` | Folder holding captures (default: current directory) |
| `--port N` | Local port (default: 8100) |
| `--no-browser` | Don't open a browser tab |

## Network and cache

| Option | Commands | Description |
| --- | --- | --- |
| `--cache-dir DIR` | all except `capture export` | Response cache (default: `~/.cache/aleph`, or `$XDG_CACHE_HOME/aleph`) |
| `--refresh` | all except `capture export` | Ignore cached responses and redownload OSM regional files |
| `--delay SECONDS` | `resolve`, `streetview`, `satellite`, `capture create` | Pause between requests (default: 0) |
| `--geocoder URL` | `resolve`, `streetview`, `satellite` | Photon geocoder server, also set with `ALEPH_GEOCODER_URL` |

OSM data comes from Geofabrik regional extracts, which are cached and reused across requests.

## Scripting

`--json` prints a single JSON object on stdout; progress and messages go to stderr.

```sh
alephgeo satellite --at 41.8902 12.4922 -o captures --json > result.json
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures --yes --json
```

On failure the object is `{"status": "error", "error": {"code": ..., "message": ...}}`. Codes include `invalid_arguments`, `place_not_found`, `ambiguous_place` (with `candidates`), `no_coverage`, `provider_unavailable`, `request_failed`, and `interrupted`. If files were already saved, `folder` points to them.

Exit codes: `0` success, `1` request failed, `2` invalid arguments, `130` interrupted.

Set `NO_COLOR` to disable colored output.
