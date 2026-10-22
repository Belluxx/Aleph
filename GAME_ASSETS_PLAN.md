# Plan: recreate captured places as game-ready Bevy worlds

Status, 2026-10-22: M0–M3 are implemented and work on the Berlin test captures (`alephgeo game FOLDER`, the
dashboard's "Game world" block, and `viewer/`). Still open:
- landmarks are rebuilt as stone tiers, not remeshed through Blender;
- there is no horizon mesh, no wind or water shader, and no stage caching (`build.json`);
- materials are not packaged as KTX2.

Use a capture's rough 3D model and its OSM data as a *reference*: where the ground is, where buildings stand, how
tall they are, and what shape their roofs have. From that, **recreate** the place as a clean, textured game level:
procedural buildings with real window and door geometry, roads with curbs and markings, instanced trees and street
props, all using authored tiling PBR materials. No photo pixels are copied into the output. Bevy renders the result,
and you walk it in first person.

---

## 1. Goals and non-goals

**Goals**

- One command turns a capture into a walkable first-person Bevy level of that place: correct layout, heights,
  massing and roof shapes, and consistent game-quality materials.
- Surfaces are sharp at player height (1.7 m eyes). Every texture is an authored, tiling material at 512 px/m or
  more, instead of a 10 px/m photo.
- Clean, budgeted geometry with LODs and colliders, packaged as glTF plus a scene manifest.
- A Bevy plugin (`AlephScenePlugin`) to load it into any Bevy game, and a viewer (`aleph_view`) to walk it.
- Automatic by default; styles and per-building fixes go in a small config file.

**Non-goals**

- A photoreal copy of the real facades. Facades are *plausible* for the building's type and region, not the actual
  ones.
- Gameplay: weapons, AI, networking.
- Interiors. Buildings are closed shells; enterable buildings are a later option.
- Worlds larger than a few km². The output is chunked so streaming can come later.

---

## 2. What the capture gives us (measured)

Measured on two real captures at mesh level 21:

| | Siegessäule (09:58) | Brandenburger Tor (10:13) |
|---|---|---|
| Area | 364 m × 330 m | 295 m × 361 m |
| Triangles | 838 k | 681 k |
| Texel density | 10.6 px/m (≈ 9 cm/px) | 10.6 px/m |
| Median triangle edge | 0.87 m | 0.89 m |

- **The mesh** is accurate in *shape* at about the 1 m scale: ground height, building height, roof form and tree
  canopy size. That is exactly what we use it for. It is a single draped skin with baked lighting, so it is useless
  as a surface. We never render it.
- **OSM (`map.osm`)** gives meaning:
  - building footprints, `building=*` type, `building:levels`, `roof:shape`, `building:material`, `roof:material`
    and colour tags;
  - roads with type, lanes and surface;
  - footways, crossings, parking, water, parks;
  - trees, street lamps, benches, traffic signals, bollards.
- **Terrain (`terrain.tif`)** gives coarse ground height for filling holes.
- **Satellite and Street View** are not used by default (see section 13, decision 2).

Frame of `mesh.glb`: metres, x east, y up, z south, centred on the capture. `asset.extras` holds `latitude`,
`longitude` and `base`. This matches Bevy's Y-up, right-handed convention, with north along −Z, so no axis
conversion is needed.

---

## 3. Core idea: measure, then rebuild

| World part | Measured from | Rebuilt as |
|---|---|---|
| Ground | Lowest mesh surface, DEM for holes | Heightfield terrain with blended PBR layers (asphalt, paving, grass, dirt, gravel) |
| Roads and sidewalks | OSM lines and areas, cut into the ground | Road meshes with lane markings, curbs, crossings, drains |
| Ordinary buildings | OSM footprint + height and roof shape fitted to the mesh | Procedural buildings: floors, window bays, doors, shopfronts, cornices, parapets, roofs, with modular trim sheets |
| Landmarks (gates, towers, monuments, domes) | Mesh inside the footprint | Cleaned, remeshed shape with procedural stone or metal materials, or a hand-made replacement asset |
| Trees | Canopy height model from the mesh + OSM trees | Instanced tree models from a species library, scaled to the measured height and crown |
| Street props | OSM nodes (lamps, benches, signals, bins, bollards) + rules (lamps along roads) | Instanced prop models |
| Water | OSM water areas | Water plane with a water shader |
| Out-of-bounds horizon | Mesh outside the play area | Low-poly silhouette with flat stylised materials, fogged |

---

## 4. Quality targets for an FPS

| Property | Target |
|---|---|
| Texel density near the player | ≥ 512 px/m through tiling materials and trim sheets; no stretched or blurry surfaces |
| Geometry | Clean, closed shells; real depth on windows (0.15–0.3 m reveals), doors, cornices and curbs |
| Triangles, standard building | 2–20 k at LOD0 (windows and trims are real geometry); 200–1 000 at LOD2 |
| Triangles, landmark | 20–80 k at LOD0 |
| Triangles, tree | 2–8 k at LOD0, instanced; card at far LOD |
| Visible budget | ≤ 3 M triangles and ≤ 1 500 draw calls at 60 fps on a mid-range desktop GPU |
| Materials | Full PBR (base colour, normal, roughness/metal/AO), shared across the scene: about 30–60 materials in total, not one per tile |
| Variation | Per-building colour tint, dirt and wear masks so repeated materials don't look tiled |
| LODs | LOD0/1/2 per chunk and per instanced asset; horizon silhouette beyond the play area |
| Collision | Heightfield ground; box or convex shapes per building part; capsules for trees and poles; boundary walls |
| Scale | Real metres; doors 2.1–2.4 m, floors about 3–3.5 m, so the player's scale feels right |

---

## 5. Pipeline overview

```
capture folder
 ├─ mesh.glb ──────┐  reference only (never rendered)
 ├─ map.osm ───────┤
 └─ terrain.tif ───┘
        │
  S1 Load + common frame            (Python, numpy)
  S2 Measure: rasters + segmentation (Python, numpy)
  S3 Ground + roads                  (Python → meshes)
  S4 Buildings: fit + generate       (Python → meshes)
  S5 Landmarks                       (Blender worker, optional)
  S6 Vegetation + props placement    (Python → instance tables)
  S7 Chunk, LODs, colliders          (Python)
  S8 Package                         (Python, optional toktx)
        │
        ▼
  capture/game/  scene.json, chunks/*.glb, report.json
  + shared asset library (materials, trims, trees, props) in Aleph's cache
        │
        ▼
  Bevy: AlephScenePlugin + aleph_view
```

Every stage records its results in `game/build.json`, using the same resumable-checkpoint pattern as captures.
Rerunning after changing a style or `game.toml` only redoes the affected stages.

---

## 6. Architecture and tooling

### 6.1 Python generator inside Aleph

- A new package `src/game/`: `load.py`, `measure.py`, `ground.py`, `roads.py`, `buildings.py` (fitting),
  `facades.py` (generator), `roofs.py`, `scatter.py` (trees and props), `chunks.py`, `gltf.py` (writer), `build.py`
  (orchestration).
- It stays on numpy and Pillow, like the rest of Aleph:
  - Most of the work is grids, polygons and procedural mesh generation, all of which numpy handles well.
  - Aleph already writes GLB files ([src/mesh.py](src/mesh.py) `export`), so the glTF writer can be shared.
- It reuses `mesh.frame()` for lat/lon → local ENU, OSM parsing from [src/osm.py](src/osm.py) and
  [src/layers.py](src/layers.py), GeoTIFF reading from [src/terrain.py](src/terrain.py), and progress reporting and
  atomic writes from [src/common.py](src/common.py).

### 6.2 Blender only for landmarks (optional)

- Voxel remeshing, decimation and UV unwrapping of irregular landmark shapes are the one job numpy does badly.
  Aleph calls Blender headless (`blender --background --python src/game/blender/landmark.py -- job.json`) for those
  shapes only.
- Without Blender, landmarks fall back to a fitted prism, or to a replacement asset from `game.toml`. The rest of
  the pipeline doesn't need Blender.

### 6.3 Shared asset library (not per capture)

- **Materials**: CC0 PBR sets (ambientCG, Poly Haven) downloaded once into Aleph's cache. For example: brick
  (several), plaster (several), sandstone, limestone, concrete, glass, painted metal, roof tiles, slate, zinc,
  asphalt, paving, cobblestone, grass, dirt, gravel.
- **Trim sheets**: one or two of our own sheets, made once in Blender and kept in the repo or released as a
  download. They cover window frames, sills, cornices, door frames, shop signs and gutters, so details are cheap
  geometry with shared UVs.
- **Modules**: parametric window, door and shopfront generators in Python (no fixed meshes). Unique props use small
  CC0 or own glTF models: street lamp, bench, bin, bollard, traffic light, sign, tree species (4–8), bushes.
- **Styles**: TOML presets that pick materials, floor heights, window rhythm, ornament level and roof types, for
  example `berlin-gruenderzeit`, `modern-office`, `plattenbau`, `suburban`, `industrial`. A building's style comes
  from OSM tags and region, with `game.toml` overrides.
- Every asset records its licence, and `scene.json` lists the required attributions.

### 6.4 Bevy runtime

- `viewer/` at the repo root: a Cargo workspace with `aleph_scene` (plugin library) and `aleph_view` (binary).
- Pin one Bevy release, the latest stable one when M1 starts.
- `avian3d` for physics and the character controller.

### 6.5 Interchange format: glTF 2.0 plus `extras`

Plain glTF keeps the output usable in Blender, Godot, Unity and Unreal too. Aleph-specific meaning goes in
`extras.aleph` on nodes and materials (colliders, material variation, instancing). Bevy exposes these as
`GltfExtras` and `GltfMaterialExtras`.

---

## 7. Stages in detail

### S1 Load and frame

- Read `mesh.glb` positions only, into numpy. Textures are ignored.
- Weld seam vertices and drop degenerate triangles.
- Bring OSM and the DEM into the mesh's ENU frame using `asset.extras` and `mesh.frame()`.

### S2 Measure: rasters and segmentation

0.25 m rasters over the area:

- **DSM top and bottom**: highest and lowest mesh surface per cell.
- **Up-facing fraction** per cell.
- **Footprint ids**: OSM buildings, buffered 0.5 m.
- **Masks** for road, sidewalk, parking, water and park, from OSM. Roads are buffered by `width`, or lanes ×
  3.25 m, or a default per type.

Classify each mesh triangle as `ground`, `building:<id>`, `vegetation` (> 1.5 m above ground, outside footprints,
rough canopy-like surface or OSM tree areas) or `other`.

**Footprint alignment**: OSM footprints can be shifted by 0.5–2 m from the mesh. Find the 2D offset per block that
best aligns footprint edges with DSM height edges, and apply it.

### S3 Ground and roads

- **Ground heightfield** (0.5 m):
  - Take the lowest robust surface outside buildings and vegetation; a percentile filter removes cars.
  - Inpaint holes, constrained by the DEM after fitting its offset.
  - Smooth to remove photogrammetry noise. The target is a gentle, realistic surface, not the mesh's bumps.
- **Roads**:
  - Build OSM highway lines into a road graph.
  - Generate road surfaces with correct widths, intersections and smooth corner radii.
  - Lane markings follow `lanes`, `oneway` and `highway` type, as decals or a marking mesh with an alpha-tested
    paint material.
  - Crossings come from `highway=crossing`.
- **Curbs and sidewalks**: a 0.12–0.15 m curb mesh with a stone trim along road edges; sidewalks are raised and
  paved. Parking areas and squares get their own paving.
- **Terrain layers**: a splat map (asphalt, paving, grass, dirt, gravel) from OSM areas and land use. Edges are
  noise-blended so they aren't razor-straight.
- **Water**: a plane at the measured water level, a bank mesh, and a water shader (normal-map flow, Fresnel,
  depth tint).

### S4 Buildings: fit and generate

**Fit** for each footprint:

1. **Eave and ridge height**: percentiles of `DSM top − ground` inside the footprint, eroded by 1 m. Fall back to
   OSM `height`, then `building:levels × 3.2 m`.
2. **Floors**: from OSM `building:levels`, else height ÷ style floor height, rounded. The ground floor is taller
   (3.8–4.5 m) for shops in urban styles.
3. **Roof shape**:
   - Use OSM `roof:shape` if present.
   - Otherwise fit candidates against the DSM heights inside the footprint and keep the lowest error: flat (with
     parapet), gabled, hipped, mansard, pyramidal, dome, skillion.
   - Inner courtyards and height steps come from footprint parts (`building:part`) or from clustering roof heights
     into tiers.
4. **Landmark test**: if no candidate fits within 1.5 m RMS, or the footprint is tagged as a monument, tower,
   church or similar with a poor fit, send the building to S5.

**Generate** for each normal building, using its style:

- **Massing**: a simplified footprint (Douglas–Peucker 0.3 m, right angles snapped), extruded per tier.
- **Facade layout**: split each wall into bays (2.5–3.5 m by style) and floors. Run a small facade grammar:
  - ground floor: shopfronts on commercial streets (from OSM `shop=*`/`amenity=*` nearby), entrance doors facing
    the street, plinth;
  - upper floors: windows per bay, with variants for balconies, French windows and pilasters;
  - string courses between floors, a cornice at the top, a parapet or eaves.
- **Real geometry**: windows have recessed reveals, frames from the trim sheet, sills and glass panes with a
  reflective glass material. Doors are recessed with frames. Cornices are extruded profiles.
- **Corners and edges**: corner pilasters or quoins by style; bays shrink to fit the wall length. Walls shorter than
  one bay stay blank, with a smaller window.
- **Materials**: wall material from `building:material` and colour tags, or a style-weighted random choice seeded by
  OSM id. Per-building tint is in vertex colour or material extras. Dirt and wear masks (darker plinth, streaks
  under sills) come from simple rules and ambient occlusion computed on the generated mesh.
- **Roofs**: roof mesh from the fitted shape with tiles, slate or zinc. Flat roofs get gravel or membrane and a few
  props (AC units, vents, stair huts). Chimneys and dormers on pitched roofs follow style rules.
- **LODs**:
  - LOD1 replaces window geometry with flat quads using baked normal and AO (baked from our own LOD0, not from
    photos).
  - LOD2 is a textured box with a facade "imposter" texture rendered from LOD0 at build time.

Every building is deterministic: seeded by OSM id, so rebuilds look the same.

### S5 Landmarks

- **Automatic**:
  - Take the mesh triangles in the footprint plus a buffer.
  - In Blender: voxel remesh (0.1–0.15 m), remove fragments, decimate to the landmark budget, unwrap UVs.
  - Apply procedural materials by surface: triplanar stone, metal or bronze, chosen from OSM tags or a per-landmark
    override. Add AO and curvature-based edge wear, computed on the mesh. No photo colour.
  - Result: correct silhouette and proportions with clean material. Fine carving stays soft.
- **Replacement**: `game.toml` can map an OSM id to a hand-made or downloaded glTF (for example a modelled
  Brandenburger Tor), placed at the footprint and scaled to the measured size. This is how a level gets truly good
  hero pieces.

### S6 Vegetation and props

- **Trees**:
  - Find canopy peaks in `DSM top − ground` inside the vegetation mask, with minimum separation by height; add OSM
    `natural=tree` and tree rows.
  - For each tree, keep height, crown radius and position.
  - Choose a species from OSM `genus`/`leaf_type` if tagged, otherwise from the region's style mix.
  - Place instances with random yaw and scale jitter.
- **Bushes and hedges**: OSM `barrier=hedge`, plus low vegetation areas scattered with bush instances.
- **Grass detail**: an optional instanced grass clump scatter near the player in park areas (a GPU instancing
  shader in Bevy, culled by distance).
- **Props**:
  - Placed from OSM nodes: street lamps, benches, waste baskets, traffic signals, bollards, post boxes, bicycle
    parking, fountains.
  - Rule-based fills where OSM is empty: lamps every 25–30 m along roads, signs at intersections, bollards at
    pedestrian areas.
  - Parked cars as an optional scatter along `parking:lane` tags or measured car-sized bumps in the ground layer.
- Everything is an instance table (`asset id, position, yaw, scale, tint`), not baked geometry, so Bevy batches it.

### S7 Chunks, LODs and colliders

- Split the play area into 64 m chunks (configurable).
  - Ground and roads are cut at chunk borders.
  - Buildings and landmarks belong to the chunk of their centroid and are never cut.
  - Instances are listed per chunk.
- **LOD switch distances** of about 50 m, 150 m and 400 m, stored in `scene.json` and used with Bevy's
  `VisibilityRange` for crossfades.
- **Horizon**: mesh outside the play polygon, or a ring around it, decimated to about 50 k triangles, with flat
  stylised colours per class (building grey, tree green, ground). It is drawn only far away and in fog.
- **Colliders**:

| Part | Collider |
|---|---|
| Ground | Heightfield per chunk (`.height`, raw little-endian f32) |
| Curbs, steps | Thin boxes (so the player steps up instead of snagging) |
| Building | Box or convex hull per tier; roof slopes as convex parts |
| Landmark | Decimated trimesh, ≤ 5 k triangles |
| Tree, lamp, pole, bollard | Capsule or cylinder per asset type, defined once in the library |
| Play-area edge | Boundary walls along the capture polygon |

### S8 Package

- Shared library materials go to KTX2 through `toktx`, if present and verified in M0 (UASTC + zstd for normals and
  ORM, ETC1S for colour). Otherwise use PNG.
- Per-capture glTFs reference library textures by relative path, so 50 buildings don't duplicate the same brick
  texture.
- Write `report.json` (section 10).

---

## 8. Output format

```
capture/game/
  scene.json            # manifest
  build.json            # stage checkpoints, input hashes, tool and library versions
  report.json           # budgets, fit statistics, warnings
  chunks/
    0_0/lod0.glb lod1.glb lod2.glb ground.height
    …
  landmarks/<osm id>.glb
  horizon.glb
  library/              # copied or linked shared materials, trims, trees and props used by this scene
```

`scene.json` (sketch):

```json
{
  "format": "aleph-game", "version": 1,
  "origin": {"latitude": 52.5163, "longitude": 13.3777, "base": 18.07},
  "frame": "metres; x east, y up, z south",
  "style": "berlin-gruenderzeit",
  "attribution": ["© OpenStreetMap contributors", "ambientCG (CC0)", "Poly Haven (CC0)"],
  "chunk_size": 64,
  "lod_distances": [50, 150, 400],
  "chunks": [{"id": "0_0", "min": [-160, 0, -192], "max": [-96, 42, -128],
              "lods": ["chunks/0_0/lod0.glb", "chunks/0_0/lod1.glb", "chunks/0_0/lod2.glb"],
              "ground": {"file": "chunks/0_0/ground.height", "size": [129, 129], "spacing": 0.5, "offset": [-160, -128]},
              "instances": [{"asset": "tree/linden", "position": [12.4, 3.1, -40.2], "yaw": 1.3, "scale": 1.1},
                            {"asset": "prop/street-lamp-berlin", "position": [3.0, 3.2, -35.0], "yaw": 0.0, "scale": 1.0}]}],
  "landmarks": [{"id": "way/518071791", "file": "landmarks/way-518071791.glb", "collider": "trimesh"}],
  "horizon": {"file": "horizon.glb", "from": 400},
  "bounds": [[-147.5, -180.1], [147.5, -180.1], [147.5, 181.6], [-147.5, 181.6]],
  "spawn": {"position": [0, 1.7, 0], "yaw": 0},
  "sun": {"azimuth": 205, "elevation": 32}
}
```

---

## 9. Bevy runtime

### `aleph_scene` (library)

- Loads `scene.json` and spawns chunks: three `SceneRoot` LODs with `VisibilityRange`, the ground mesh, and a
  heightfield collider.
- Spawns instances from the library and lets Bevy batch identical meshes and materials automatically.
- Adds colliders from `GltfExtras` and from asset definitions, using `avian3d`.
- A material extension (`ExtendedMaterial<StandardMaterial, …>`) adds:
  - per-instance and per-building tint;
  - dirt and wear blending from a mask;
  - triplanar mapping for landmarks;
  - wind sway for tree leaves (vertex shader).
- A water material.
- **Lighting and atmosphere**:
  - `DirectionalLight` with cascaded shadows; sun position from a configurable time of day (default: a pleasant
    afternoon).
  - Physical atmosphere or skybox, distance and height fog, an environment map for reflections.
  - SSAO, TAA, optional screen-space reflections and bloom, filmic tonemapping, and a colour grading preset per
    style.
- Boundary walls and spawn.

### `aleph_view` (binary)

- `aleph_view <capture>/game`, or `alephgeo game FOLDER --view`.
- FPS controller:
  - Kinematic capsule (0.3 m radius, 1.8 m tall) on `avian3d`.
  - WASD, mouse look, Shift to sprint, Space to jump, C to crouch.
  - Step-up of 0.35 m, slope limit of 45°.
  - Footstep surface detection from the splat layer.
- `V` toggles noclip flight.
- Time-of-day slider.
- Debug keys:
  - F3: stats (FPS, triangles, draw calls, chunks).
  - F4: wireframe.
  - F5: colliders.
  - F6: LOD tint.
  - F7: show the reference mesh as a ghost overlay to check fit.
  - F12: screenshot.
- `--bench` flies a fixed path and reports frame-time percentiles, which are checked against the section 4 budgets.

### Later

- WebAssembly/WebGPU build embedded in the dashboard.
- Baked indirect lighting: irradiance volumes or light probes generated at build time, if Bevy's support matures
  enough for this scene type.

---

## 10. CLI, dashboard and checks

**CLI (one subcommand):**

```sh
alephgeo game FOLDER [--style auto|NAME] [--chunk 64] [--quality draft|high] [--blender PATH] [--view]
```

- `draft`: massing, roofs and ground, without window geometry or props. Gives a quick look at layout and heights.
- `high`: everything.
- It is resumable; rerun after editing `game.toml` or changing `--style`.
- If OSM or terrain is missing from the capture, `game` fetches it for the capture's polygon with the existing
  stages first. OSM is required, because without footprints and roads there is little to rebuild.

**Dashboard:**

- A "Game world" block on the capture page: style picker, Build button (job with progress), key report numbers,
  "Open in viewer".
- The 2D map can overlay building classes (generated, landmark, replaced, dropped) for quick review.

**`report.json`:**

- Counts: buildings generated, landmarks, replacements and drops (with reasons); trees; props.
- Fit statistics: per-building height and roof RMS against the mesh. Outliers are listed so they can be fixed in
  `game.toml`.
- Budgets: triangles and draw calls per LOD and chunk, texture memory; pass or warn against section 4.
- Fixed-viewpoint screenshots from `aleph_view --shots`, to compare builds over time.

**Tests** (only where they prevent regressions, following AGENTS.md):

- numpy unit tests on synthetic inputs: footprint alignment, roof-shape fitting, facade bay splitting (no overlaps,
  windows within walls), road widths and intersections, tree peak detection, chunk assignment, `scene.json` writing.
- One optional smoke test runs a tiny fixture end to end and is skipped without `cargo`.

---

## 11. Milestones

Sizes are relative (S < M < L < XL).

**M0: Spikes (S).**

- Bevy: `GltfExtras` and `GltfMaterialExtras`, `VisibilityRange` crossfades, `ExtendedMaterial` on glTF
  materials, KTX2 in glTF, and instancing performance with about 5 k tree and prop instances.
- `avian3d`: heightfield + convex + capsule character at our scale.
- Material library sourcing and licence check. Make the first trim sheet.
- Acceptance: a short note with answers and numbers; the plan adjusted where reality differs.

**M1: Walkable blockout with real materials (M).**

- S1–S3 (ground, roads, curbs, sidewalks).
- S4 massing only: extruded footprints at fitted heights, flat roofs, walls with tiling materials, no windows yet.
- Colliders, chunks, `scene.json`.
- `aleph_scene` and `aleph_view` with the FPS controller, sun, fog and debug keys.
- Acceptance: walk the Brandenburger Tor area at 60 fps or more. Layout and heights match the reference (F7
  overlay), and surfaces are sharp up close.

**M2: Procedural buildings (L).**

- Roof-shape fitting.
- Facade grammar with real window, door and shopfront geometry.
- Trim sheet, styles, per-building variation, dirt and wear masks, building LODs with self-baked imposters.
- Acceptance: at least 85% of OSM buildings in both test captures fit within 1.5 m RMS. Facades have no visible
  overlaps or stretched UVs. LOD0 budget is met.

**M3: Nature, props and atmosphere (M).**

- Tree detection and species library, bushes, grass scatter.
- OSM and rule-based props, water.
- Time of day, colour grading, horizon silhouette.
- Acceptance: the area reads as a lived-in street at eye level; draw-call budget is met with all instances.

**M4: Landmarks and integration (M).**

- S5 Blender landmark path with triplanar procedural materials.
- Replacement assets through `game.toml`.
- CLI flags finalised, dashboard block, README and `docs/` pages.
- Acceptance: Siegessäule and Brandenburger Tor appear as clean, recognizable landmarks, or as replacements when
  provided.

**M5: Optional upgrades (XL, pick individually).**

- More styles and regions.
- Enterable ground floors with simple interiors.
- Navmesh for AI.
- Streaming for multi-km² areas.
- WebGPU viewer in the dashboard.
- Optional analysis-only hints from photos to choose wall colour and material class (never copying pixels; off by
  default).

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Generic facades make every city look alike | Styles per region and era, OSM material and colour tags, seeded variation, room for hand-made modules per style |
| OSM footprints missing, misaligned or outdated | Footprint alignment to DSM edges; mesh-only building blobs (tall `other` areas without footprints) get a footprint traced from the DSM; `game.toml` overrides |
| Landmarks look soft or wrong | Clear landmark test; replacement assets as the quality path; triplanar materials hide UV issues |
| Procedural geometry exceeds the triangle budget (thousands of windows) | LOD1 swaps window geometry for normal-mapped quads; instanced window modules if needed; budget checks in the report |
| Blender dependency | Only for landmarks, and optional; fallback to a fitted prism |
| Bevy API churn | Pin a Bevy release; keep the plugin small; deliberate upgrades |
| Asset licences | CC0 or own assets only; licence recorded per asset; attributions in `scene.json` |
| Measurements derived from Google's mesh | Only numbers (heights, roof shape, tree positions) are taken, never pixels or mesh surfaces (except the Blender landmark remesh). An `--sources osm` mode builds from OSM and the DEM alone, for output you want to ship |
| Float precision | Local ENU frame keeps coordinates within a few km, which f32 handles fine |

---

## 13. Decisions to confirm before M1

1. **Recreate, don't restore**: the photogrammetry is only a reference, and facades are plausible rather than exact.
   (This plan assumes yes.)
2. **Photos**: none in the output. Should satellite or Street View be used even as analysis-only hints for colour
   and material choice (an M5 option, off by default)?
3. **Target look**: realistic PBR (as planned), or deliberately stylised (simpler materials, stronger colours), which
   is cheaper and hides generic facades better?
4. **Target hardware**: mid-range desktop GPU at 60 fps, or also laptops or the web?
5. **Landmarks**: automatic remesh through Blender (optional dependency), or prisms plus hand-made replacements
   only?
6. **Where the Bevy code lives**: `viewer/` in this repo (proposed), or a separate repository.
