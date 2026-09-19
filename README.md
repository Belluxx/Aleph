![Aleph](resources/readme-banner.svg)

Aleph does two things:

- Lets you draw a rectangle on the map and download everything about it (buildings, elevation, HD satellite imagery, Street View photos).
- Provides any of the above information for specific points on Earth, on the fly.

This tool was born as a way to let LLMs quickly gather information from the physical world.

> [!WARNING]  
> This tool uses undocumented APIs; for this reason, it may break or you may experience rate limits.

## Open the dashboard

Requires Python 3.11+. Pillow is the only dependency. No API keys required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
alephgeo dashboard --root captures
```

The dashboard opens in your browser. Your captures will be saved in `captures`.

## Inside the folder

Each capture lives in its own `aleph-TIMESTAMP-ID` directory. Files depend on the sources you chose. Here is an overview:

| File | Contents |
| --- | --- |
| `streetview/` | Street photos, positions and headings in GeoJSON |
| `satellite.png` | Stitched satellite imagery (original tiles in `satellite/`) |
| `map.osm` | OpenStreetMap roads, buildings, water, and other features |
| `terrain.tif` | Elevation as a GeoTIFF (original tiles in `terrain/`) |

## From the terminal (for scripts and agents)

See the [CLI docs](docs/cli.md) for examples and details.
