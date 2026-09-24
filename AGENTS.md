# AGENTS.md

## Project

Aleph is a Python CLI for capturing Google Street View photos, satellite imagery, and OSM data for AI agents. It requires no browser or API key.

- `aleph.py`: command-line entry point
- `src/`: capture, networking, geometry, Street View, and terrain code
- `tests/`: unit tests
- `README.md`: setup, usage, and troubleshooting.
- `resources/`: assets and resources

## Changes

- Keep the CLI minimal and simple
- Completely ignore backward compatibility
- Do not add tests unless they really matter to avoid future regressions
