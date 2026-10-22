# Aleph viewer

Bevy runtime for Aleph game worlds (`capture/game/` with `scene.json`):

- `aleph_scene`: library crate with `AlephScenePlugin`, which loads a game folder into any Bevy app.
- `aleph_view`: first-person viewer binary.

Pinned: **Bevy 0.20.0**, **avian3d 0.8.0** (the latest stable releases on 2026-10-22), Rust 2024 edition.

## Run

```sh
cd viewer
cargo build                                    # dev profile: dependencies at opt-level 3, fast enough to play
target/debug/aleph_view ../captures/<id>/game  # walk it (opens a window)
target/debug/aleph_view GAME --shots DIR       # headless: render 6 fixed views to DIR/*.png and exit
target/debug/aleph_view GAME --bench 20        # headless: fly 20 s of a 1080p loop (1/60 s per frame), print frame times
target/debug/aleph_view GAME --shots DIR --views FILE  # headless: render named cameras [{"name", "eye", "at"}]
target/debug/aleph_view GAME --max-texture 1024        # drop texture mips above 1024 px to save GPU memory
```

A Brandenburger Tor world peaks at about 3.7 GB of GPU memory (3.1 GB with `--max-texture 1024`). Out of Memory
errors at start-up mean another program holds the GPU's memory (check `nvidia-smi`).

`--shots` and `--bench` render into an offscreen image with no window, so they work on a busy desktop or over SSH
(Vulkan still needs a GPU). Shots: `eye-000/090/180/270.png` at the spawn eye position (yaw in degrees,
counter-clockwise from north), `overview.png` from high in the south, and `overview-lod.png` with the LOD tint.

Controls: click to grab the mouse, Esc to release; WASD, Shift sprint, Space jump, C crouch, V noclip flight (Space
and C go up and down). F3 stats, F4 wireframe, F5 collider gizmos, F6 LOD tint (white, blue, red), F12 screenshot
into the game folder.

Test data without the Python pipeline:

```sh
python viewer/fixtures/make_fixture.py /tmp/fixture          # 2×2 chunks, boxes, ramps, stairs, 16 trees
python viewer/fixtures/make_fixture.py /tmp/fixture-big 8 80 # 64 chunks, 5 120 trees
```

## Using the plugin

```rust
App::new()
    .add_plugins(AlephScenePlugin { folder: "path/to/game".into() }) // before DefaultPlugins: registers aleph://
    .add_plugins((DefaultPlugins, AlephRenderPlugin, PhysicsPlugins::default())) // after: materials, bounce light
    .add_systems(Startup, |mut c: Commands| { c.spawn((Camera3d::default(), AlephCamera)); })
    .run();
```

`AlephCamera` adds the sky, fog, and post-processing to a camera. `AlephWorld` holds the parsed scene and ground
heights (`ground_height(x, z)`); `AlephStatus.loaded` turns true once every glTF and texture is in.

## How the scene maps to Bevy

- Every game-folder path loads through the `aleph://` asset source rooted at the folder, so relative texture URIs
  (`../../library/...`) resolve from each glb's own location. The library must sit inside the game folder (or be
  symlinked there).
- Each glTF is flattened once into (transform, mesh, material) pieces plus colliders and shared by all its instances,
  so identical trees batch. Collider nodes (`collider/...`) are never rendered.
- LODs: `VisibilityRange` per mesh with ±4 % crossfades. Each node's meshes are recentred in place on the node's
  bounds (the offset moves into the transform), and the switch distances grow by the node's bounding radius. A 200 m
  building keeps LOD0 while you stand at either end, and the LODs of one node switch together because they share
  about the same bounds. Bevy's `use_aabb` isn't used: CPU culling would measure from the AABB centre but the GPU
  crossfade dither from the mesh origin, which for chunk meshes is the chunk centre, so off-centre buildings dithered
  out. Instances scale the
  distances by asset size (`max(height, 2·radius)·scale / 10 m`, clamped to 0.25–1), so bushes and bollards
  switch and vanish sooner.
- Colliders (avian3d, static): chunk LOD0 `collider/` nodes (convex hull from POINTS, trimesh, capsule, cylinder);
  a heightfield per chunk from `ground.height`; asset colliders from `scene.json` per instance (scaled, glb collider
  nodes in assets are ignored); 20 m boundary walls along `bounds`.
- Instance tint multiplies the base colour, through material copies shared by tints in 1/32 steps.
- Materials (`material.rs`, `shaders/aleph.wesl`): every mesh uses `AlephMaterial`, a bindless `ExtendedMaterial` over
  the glTF's StandardMaterial, one per library material name (so meshes of all glTFs batch together). Its
  `extras.aleph` turn on, per material (unknown keys are ignored):
  - `interior`: glass becomes a near-mirror (Fresnel, env reflections) over a fake room (interior mapping): ceiling,
    floor, walls, a back-wall cupboard and a furniture band, colours hashed per room, 18 % of rooms lit, curtains,
    blinds or nets behind some panes. Each connected glass piece up to 6 m is a pane with its own room (centred on
    it, floor below a sill, written to UV_1 and COLOR_0 at load); larger glass (curtain walls) uses a 3 m grid.
    Smooth non-metal texels only, so frames painted into door and curtain-wall textures stay opaque. Glass names
    from the library are the fallback.
  - `wind`: vertex sway of the crown and leaf flutter, in the main pass, prepass, shadows and motion vectors
    (`veg/leaves-*` as fallback).
  - `triplanar`: world-space projection at `size` metres per repeat, with matching tangents.
  - `height` + `parallax`: relief mapping (4 steps over 12 layers) from `library/<height>`, inverted to Bevy's depth
    convention at load; `parallax` is metres of relief.
  - `decal`: constant depth bias in front of the surface below and no shadow casting; alpha as the glTF says.
- LOD crossfade: the shader discards with interleaved gradient noise that moves every frame, which TAA resolves into
  a smooth blend (Bevy's fixed 4×4 pattern showed as dots on trees).
- Light: a sun from `sun` with three shadow cascades to 120 m, Bevy's physical atmosphere with the Mie terms of its
  earth medium corrected (scattering, not absorption; ×1.5 for city haze), a pale squared-falloff haze (clear near,
  opaque at about 6 km), SSAO (1.5 m radius), TAA, ACES tonemapping at 0.9 saturation, EV100 13.
- Image-based light (`bounce.rs`): Bevy fills the atmosphere's environment map from the sky only, so the lower
  hemisphere is black and shade turned saturated blue. Each frame, before Bevy filters it, a compute pass replaces
  the lower hemisphere with sunlit ground (albedo 0.2, 65 % in sun, lit by the transmitted sun and the integrated
  sky), blends lit facades over the sky up to about 15° and halves the sky's saturation. Shade reads neutral and
  metals (gold, bronze) reflect warm surroundings instead of blue and black.
- Instances under 1 m (grass clumps, small bushes) cast no shadows.
- Beyond the bounds: a ground skirt out to 6 km, whose inner edge follows the heightfields along the bounds and eases
  to a low edge percentile within 200 m. It uses `library/materials/ground/grass` at 25 m per repeat, muted, if the
  game folder has it, otherwise a plain field colour, and fades into the haze.
- Horizon glb (optional): drawn without shadows; each glTF node shows once its bounds centre is farther than
  `horizon.from`, so a horizon split into tiles hides the tiles near the player.
- Mipmaps: Bevy loads PNG and JPEG with one mip level, so the plugin builds mip chains on the CPU at load
  (sRGB-correct box filter) and turns on 16× anisotropic filtering. This was worth 30–60 % frame rate on the
  Siegessäule capture and removes distant shimmer. KTX2 textures with their own mips are left alone.
- Player: a kinematic capsule (0.3 m radius, 1.8 m tall, eyes at 1.7 m) moved with avian's `MoveAndSlide` shape
  casts, with step-up of 0.35 m and slopes steeper than 45° acting as walls. `spawn.position` is the eye; it's
  lifted if the feet would be below the ground.

## Verified (M0)

- glTF node extras (`GltfNode.extras`) and material extras (`GltfPrimitive.material_extras`) are both readable.
- External relative texture URIs load from an arbitrary folder; JPEG and PNG; missing files only log errors.
- Vertex colours (COLOR_0) multiply base colour and texture in `StandardMaterial` by default.
- `VisibilityRange` crossfades work (see `overview-lod.png`).
- Heightfield (orientation checked by walking: feet follow the generated height function), convex, trimesh,
  capsule, cylinder colliders line up with the meshes (F5 gizmos).
- Controller, walked headlessly on the fixture: 0.25 m stairs climbed, 22° and 42° ramps climbed, a 66° ramp
  blocks, buildings and boundary walls block, jump apex 1.27 m, crouch and noclip work.
- Headless rendering into an image (`--shots`, `--bench`) with no window.
- Performance on a GTX 1080 Ti at 1080p (dev build, `--bench 20`, pipelines compiled before measuring, desktop
  sharing the GPU): Brandenburger Tor (30 chunks) ~105 fps, p99 ~23 ms; Siegessäule (36 chunks, 6 174 grass clumps,
  1 169 bushes) 66 or 30 fps depending on the run (see Gaps). GPU-bound on alpha-tested foliage in the shadow pass,
  the depth prepass, and the main pass.

## Gaps

- The interactive window path is built and shares all systems with the headless modes, but it wasn't opened during
  development (no display to spare); F3's overlay text was checked by printing it.
- No water shader and no screen-space reflections (they need deferred rendering); glass reflects the environment
  map only, not the buildings opposite.
- Alpha-tested foliage makes the frame time bimodal on dense parks: the same Siegessäule run takes 15 or 33 ms
  (shadow pass 3 or 10 ms) depending on the draw order Bevy happens to pick at startup. With opaque leaves it is a
  steady 12 ms.
- Rarely, wgpu fails to create the device at startup when the GPU is busy; rerun.
