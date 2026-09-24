# CLI guide

[Install Aleph](../README.md#install)

## Download a photo or satellite image

```sh
# Street View near a named place
alephgeo streetview --place "Colosseum, Rome" --best-match -o captures

# A satellite image covering a 200 × 200 meter square
alephgeo satellite --place "Colosseum, Rome" --best-match --size 200 -o captures
```

`--best-match` picks the first search result. Omit it to see a list of matching places, then choose it with `--match TYPE/ID`.

`-o captures` saves each download in the `captures` dir. Otherwise the results folders are created in the current directory.

Satellite captures produce `satellite.tif`, a lossless RGBA Cloud Optimized GeoTIFF (COG). Missing imagery is transparent. Large outputs automatically use BigTIFF.

Each capture also includes `satellite.png` with identical full-resolution pixels and transparency. 

Choose the saved satellite tile format with `satellite --satellite-format png`
or `satellite --satellite-format jpg` (default). For area captures, use
`capture create --satellite-format png`; the dashboard has the same choice
under **Advanced → Satellite → Tile format**. This only changes files in
`satellite/patches/`: both `satellite.png` and `satellite.tif` are always generated.
Missing tiles remain transparent PNGs, including when JPEG is selected.

Choose Street View photo formats with `streetview --streetview-format png`
or `capture create --streetview-format png` (default: `jpg`).

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

Street View searches within 50 meters. Add `--radius 200` to search farther away if no panorama is found.

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
# Request photos at ten stops, looking forward along the street
alephgeo streetview --street "Via del Corso, Rome" --best-match --stops 10 -o captures

# Photograph both sides at each stop
alephgeo streetview --street "Via del Corso, Rome" --best-match --stops 10 --view both -o captures
```

`--view` accepts `forward`, `backward`, `left`, `right`, or `both`. Add `--reverse` to travel in the opposite direction. If the street has multiple branches, repeat with `--route N` using a returned route number. Stops without coverage are skipped.

## Download an area

Capture all available data inside a rectangle (Street View, satellite imagery, OSM data, and terrain):

```sh
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures
```

The command shows an estimate and asks before downloading. Add `--no-plan` to skip the prompt, or `--plan` to save a plan for later.

To download only map data and terrain:

```sh
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 --sources osm -o captures
```

`--sources` accepts one or more of `streetview`, `satellite`, and `osm` (which includes terrain).

Resume an interrupted capture or start a saved plan; replace `captures/RUN_FOLDER` with the printed folder containing `manifest.json`:

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
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 -o captures --no-plan --json
```

For all options, run `alephgeo COMMAND --help` (for example, `alephgeo streetview --help` or `alephgeo capture create --help`).
