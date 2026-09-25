# CLI guide

[Install Aleph](../README.md#install)

## Download a photo or satellite image

```sh
# Street View near a named place
alephgeo streetview --place "Colosseum, Rome" --best-match -o captures

# A satellite image covering a 200 × 200 meter square
alephgeo satellite --place "Colosseum, Rome" --best-match --size 200 -o captures
```

`--best-match` picks the first search result. Omit it to list matches, then choose with `--match TYPE/ID`.

`-o captures` saves downloads under `captures/` instead of the current directory.

Satellite captures produce `satellite.png` and `satellite.tif` (Cloud Optimized GeoTIFF), with transparent gaps for missing imagery.

- `--tile-format jpg|png` sets the saved satellite tile format (default: `jpg`); both merged outputs are always generated.
- `--streetview-format jpg|png` sets the Street View photo format (default: `jpg`).
- `--full-sphere` saves one 360° panorama per stop. `--sphere-zoom 0–5` sets resolution (default: `3`).

These options also work with `capture create`.

## Use coordinates

Coordinates are `LAT LON`; distances are in meters.

```sh
# Street View facing east (0 = north, 90 = east, 180 = south, 270 = west)
alephgeo streetview --at 41.8902 12.4922 --heading 90 -o captures

# Aim the camera at a specific point
alephgeo streetview --at 41.8902 12.4922 --look-at 41.8903 12.4924 -o captures

# Satellite image around a point
alephgeo satellite --at 41.8902 12.4922 --size 500 -o captures

# Satellite image of a rectangle: south west north east
alephgeo satellite --bbox 41.8895 12.4910 41.8910 12.4940 -o captures
```

Street View searches within 50 meters; use `--radius 200` to search farther.

## Find places

```sh
# Look up a name and its coordinates
alephgeo resolve "Colosseum, Rome"

# Find the address or place at these coordinates
alephgeo resolve --at 41.8902 12.4922

# List nearby points of interest
alephgeo resolve --at 41.8902 12.4922 --nearby --radius 200
```

## Take photos along a street

```sh
# Ten stops total, looking forward
alephgeo streetview --street "Via del Corso, Rome" --best-match --stops 10 -o captures

# Both sides, targeting 10-meter spacing
alephgeo streetview --street "Via del Corso, Rome" --best-match --step 10 --view both -o captures
```

Use `--stops N` for a total count or `--step METERS` for spacing. All street sections are included; `--route N` selects one by its number in `result.json`.

`--view` accepts `forward`, `backward`, `left`, `right`, or `both`. Add `--reverse` to reverse each section's direction.

## Download an area

Capture Street View, satellite imagery, OSM data, and terrain inside a rectangle:

```sh
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures
```

The command shows an estimate and asks before downloading. Add `--yes` to skip the prompt, or `--plan` to save a plan for later.

Use `--sources` to select `streetview`, `satellite`, or `osm` (including terrain), or a combination.

Resume a capture or start a saved plan using its printed run folder:

```sh
alephgeo capture resume captures/RUN_FOLDER
```

Rebuild outputs from downloaded files, offline:

```sh
alephgeo capture export captures/RUN_FOLDER
```

## Use in scripts

Add `--json` for one JSON result on stdout; progress stays on stderr.

```sh
alephgeo satellite --at 41.8902 12.4922 -o captures --json > result.json
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures --yes --json
```

For all options, run `alephgeo COMMAND --help`, such as `alephgeo capture create --help`.
