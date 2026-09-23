# CLI guide

[Back to the README](../README.md)

Each operation has its own command. Run `alephgeo COMMAND --help` for its options. Coordinates are always `LAT LON`, distances are in meters, and headings are clockwise from north.

Without `--json`, output is a compact summary: place choices show IDs, names, and types; route choices show lengths and endpoints; downloads show what was saved and its path. `resolve` also shows coordinates and nearby distances. Add `--json` for full metadata, source links, and error details.

| Command | Purpose |
| --- | --- |
| `resolve` | Search for places, reverse geocode, or list nearby POIs |
| `streetview` | Save one view or an ordered sequence along a named street |
| `satellite` | Save a satellite patch or an exact tile |
| `capture` | Create, resume, or export an area capture |
| `dashboard` | Open the visual interface |

## Find a place

```sh
alephgeo resolve "Colosseum, Rome" --json
alephgeo resolve --at 41.8902 12.4922 --json
alephgeo resolve --at 41.8902 12.4922 --nearby --radius 100 --limit 10 --json
```

Reverse lookup returns the nearest address or named place. Nearby search lists places by distance, using the center of buildings and other areas.

Name lookup uses [Photon](https://github.com/komoot/photon). Set `ALEPH_GEOCODER_URL` or `--geocoder URL` to use another Photon server. Nearby places and streets use Geofabrik map data.

## Quick imagery

```sh
alephgeo streetview --at 41.8902 12.4922 --heading 90 -o captures --json
alephgeo streetview --at 41.8902 12.4922 --look-at 41.8903 12.4924 --json
alephgeo streetview --pano-id PANORAMA_ID --pitch 10 --fov 75 --json
alephgeo satellite --at 41.8902 12.4922 --size 200 --zoom 19 -o captures --json
alephgeo satellite --tile 19/280337/194891 -o captures --json
```

Both commands accept `--place "Colosseum, Rome"` instead of `--at LAT LON`. If several places match, they are listed. Repeat the command with `--match` followed by your chosen result's `id`.

Street View finds the nearest panorama within 50 meters; change this with `--radius`. Results include the camera position, distance, viewing direction, and photo date when available.

For satellite images, `--size 200` requests a square 200 meters wide around your location. Use `--bbox SOUTH WEST NORTH EAST` for a rectangle.

## Follow a street

```sh
alephgeo resolve "Via del Corso, Rome" --streets --json
alephgeo streetview --street "Via del Corso, Rome" --stops 10 --view forward -o captures --json
```

For example, `--stops 10` requests ten positions along the street. Choose a viewing direction with `--view forward`, `backward`, `left`, or `right`; `both` takes left and right photos at each stop. Use `--reverse` to go in the opposite direction.

If several results match, use `--match ID` to pick a place or `--route N` to pick a street branch.

> [!TIP]
> Images and Photon results are cached for one week. Geofabrik files are kept until refreshed. Use `--refresh` to fetch new data or `--cache-dir PATH` to choose the cache folder (default: `$XDG_CACHE_HOME/aleph` or `~/.cache/aleph`).

## Capture an area

Use the capture feature when you need to download a large piece of land. It supports many square kilometers of land.

```sh
alephgeo capture create --bbox 41.8895 12.4910 41.8910 12.4940 --no-plan -o captures
```

`--bbox` takes two opposite corners: `lat1 lon1 lat2 lon2`. All sources are included; use `--sources satellite osm` to download only satellite imagery, map data, and terrain.

`capture create` and `capture resume` also accept `--cache-dir PATH` and `--refresh`. Time estimates do not include the first Geofabrik download.

`--no-plan` starts downloading without confirmation (not recommended, as it may take a lot of time). Omit it to review the estimate first, or use `--plan` to save a plan without downloading. See `alephgeo capture create --help` for resolution and spacing options.
