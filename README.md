<p align="center">
  <img src="resources/readme-banner.svg" alt="Aleph" width="600">
</p>

<p align="center">
  Download a piece of the world<br>
  for you (<em>or your agent</em>)
</p>

Aleph does two things:

- Lets you draw a rectangle on the map and downloads **everything** inside it (buildings, elevation, HD satellite imagery, Street View photos).
- Provides any of the above information for **specific points** on Earth, on the fly, via the CLI.

This was born as a way to let LLMs quickly gather information from the physical world, but you can just use it to have piece of the Earth on your laptop and do whatever you want with it.

> [!WARNING]  
> This tool uses undocumented APIs; for this reason, it may break or you may experience rate limits.

## Install

Requires Python 3.11+. Pillow is the only dependency. No API keys are required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Then test with

```sh
alephgeo --help
```

## Open the dashboard

From the dashboard you can select an area on the map and download all the data you need.
You can later give the resulting directory as context for the agent to work on.
If you prefer using Aleph from the CLI or want to let an agent interface with directly see [CLI docs](docs/cli.md).

```sh
alephgeo dashboard --root captures
```

The dashboard opens in your browser at `http://127.0.0.1:8100`. Your captures will be saved in `captures`.

## Inside the folder

Here is an overview of the structure of the output dir after a capture:

- `streetview/`: Street photos, positions and headings in GeoJSON
- `satellite.png`: Stitched satellite imagery (original tiles in `satellite/`)
- `map.osm`: OpenStreetMap roads, buildings, water, and other features
- `terrain.tif`: Elevation as a GeoTIFF (original tiles in `terrain/`)

Map data comes from [Geofabrik](https://download.geofabrik.de/). The first use will download a large regional file. It will be reused for later requests.

## From the terminal (for scripts and agents)

See the [CLI docs](docs/cli.md) for examples and details.
