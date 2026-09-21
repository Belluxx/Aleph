# OSM provider benchmark — 21 September 2026

Aleph now uses the Python `osmium` dependency for Geofabrik extraction and local road geometry. The native Osmium CLI measurements below describe the earlier implementation, which has been replaced. No Overpass service is used by either Geofabrik implementation.

## Method

Both selections are centered on the Colosseum in Rome, at 41.8902° N, 12.4922° E. They are squares approximately 3.162 km and 31.623 km wide, with spherical areas of 10.000 and 999.999 km². Tests ran on macOS 15.7.1 ARM64 with Python 3.12.12. The original CLI comparison used Osmium 1.19.1; the current Python implementation uses `osmium` 4.3.1.

The baseline is commit `38aa50015fe347a71bcd1c0906376a8a46e3779d`. Its road and map queries, memory budgets, HTTP retries, and three-server fallback were unchanged. Each network operation had an outer 600-second benchmark deadline. A deadline result means no usable response arrived within the test budget, not that the client exhausted every possible retry. Imagery and terrain downloads were excluded.

These are single live trials, so network timings depend on server load. Local files were available in the filesystem cache. Software installation time is excluded.

## Current Python Osmium results

The Python implementation installs through `pip install .` and uses no external Osmium executable. It reads PBF files directly, preserves complete ways and parent relations, and completes nested POI relations. Roads and nearby places are read directly into Python without intermediate XML files.

Both exports reused the same cached central Italy regional file, with network access disabled in the client:

| Area | Map export | Map XML size |
| --- | ---: | ---: |
| 10 km² | 161.65 s | 41.3 MB |
| 1,000 km² | 173.88 s | 571.4 MB |

These single-trial timings exclude downloading the regional file and subsequent validation. Python scans the entire regional file to select the rectangle, so the regional file's size matters even for small selections. Capture estimates now allow about three minutes for local map extraction, based on this region.

Both exports have identical node, way, and relation IDs to the earlier CLI exports, and both pass way-reference validation through Python Osmium. The 10 km² export's 12,412 highway ways match the earlier road geometry exactly. Nearby results around the Colosseum and Via del Corso's 11 connected ways also matched when read from the new export. All 44 tests pass with the native tool absent from `PATH`, including cancellation, nested relations, offline reuse, and preserving previous files on failure.

## Earlier native Osmium CLI results

Both selections use [central Italy's extract](https://download.geofabrik.de/europe/italy/centro.html), containing data through `2026-09-20T20:22:06Z`. The regional PBF is 383,641,858 bytes (383.6 MB). Its initial standalone download took 13.59 seconds.

The earlier CLI-backed application path also completed a fresh index download, regional download, and 10 km² map export in **15.26 seconds total**. Its request log contained exactly two GETs, both to Geofabrik. Subsequent operations reused the same regional file:

| Area | Map export | Road loading | Map XML size | Highway ways |
| --- | ---: | ---: | ---: | ---: |
| 10 km² | 3.39 s | 3.64 s | 41.3 MB | 12,412 |
| 1,000 km² | 3.66 s | 11.72 s | 571.4 MB | 163,195 |

Map timings include selecting the region and writing the selected rectangle as XML. Road timings include extracting the rectangle, filtering highways, and decoding geometry and OSM node IDs into Python. The larger area reused the first area's regional download; it did not require downloading a larger file. Each operation creates temporary local files, which are removed afterward. Only the regional PBF and region index are retained.

## Baseline results

The 10 km² road query reached the 600-second deadline after repeated HTTP 504 overload responses and a socket timeout. The 1,000 km² road query also reached the 600-second deadline after repeated HTTP 504 responses. Both reached the third endpoint before their deadlines.

Both map queries succeeded:

| Area | Download, including retries | Original validation | XML size |
| --- | ---: | ---: | ---: |
| 10 km² | 7.84 s | 1.79 s | 52.4 MB |
| 1,000 km² | 62.23 s | 113.68 s | 769.4 MB |

The smaller map succeeded on its first attempt. The larger map succeeded on its second attempt, after an HTTP 504 response. The recorded 10.69 and 226.08 second benchmark totals also include an additional XML parse for statistics, so they should not be presented as application timings. The new exporter streams Osmium's output to disk and uses its complete-way extraction, avoiding the original full-tree Python validation.

## Data checks

Both Geofabrik map exports passed `osmium check-refs`, with zero missing nodes in ways. Their counts were:

| Area | Nodes | Ways | Relations |
| --- | ---: | ---: | ---: |
| 10 km² | 177,525 | 32,787 | 3,219 |
| 1,000 km² | 3,172,834 | 469,155 | 21,004 |

All highway way IDs matched between the baseline map and Geofabrik at both sizes: 12,412 and 163,195 respectively. The snapshots are approximately 16 hours apart, and Geofabrik omits contributor metadata. Osmium also retains parent relations, so the files are not expected to have identical object counts or bytes. Distant relation members may remain unresolved; local ways are complete.

A real nearby search returned the Colosseum relation and surrounding POIs from the local file. Resolving Via del Corso's connected street geometry took 2.69 seconds. Offline fixtures cover boundary-crossing ways, disconnected streets with the same name, relation centers, empty nearby results, cancellation, and preserving existing files when refresh or extraction fails.

All 44 tests passed with `python -m unittest discover -s tests -v`. [Machine-readable measurements](osm-benchmark.json) retain the recorded timings, bounds, source dates, and attempt outcomes.

## Reproduce the historical CLI extraction

With the central Italy PBF downloaded as `centro.osm.pbf`:

```sh
osmium extract centro.osm.pbf -s complete_ways --set-bounds \
  -b 12.473098695094054,41.875980496522295,12.511301304905947,41.904419503477705 \
  -o rome-10.osm
osmium extract centro.osm.pbf -s complete_ways --set-bounds \
  -b 12.301186950940533,41.74800496522297,12.683213049059468,42.03239503477703 \
  -o rome-1000.osm
osmium check-refs rome-10.osm
osmium check-refs rome-1000.osm
```
