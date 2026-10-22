//! Loads an Aleph game folder (scene.json, chunk and library glTFs, ground heightfields) into a Bevy world.
//!
//! Add [`AlephScenePlugin`] before `DefaultPlugins`, since it registers the `aleph://` asset source that points at the
//! game folder, and [`AlephRenderPlugin`] after them, since it needs the renderer. Add avian3d's `PhysicsPlugins`
//! yourself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use avian3d::parry::{shape::SharedShape, utils::Array2};
use avian3d::prelude::*;
use bevy::asset::io::{AssetSource, AssetSourceBuilder};
use bevy::asset::{AssetPath, LoadState};
use bevy::camera::visibility::VisibilityRange;
use bevy::gltf::{Gltf, GltfMaterial, GltfMesh, GltfNode};
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;

mod bounce;
pub mod format;
pub mod material;
pub mod light;
mod mips;
mod skirt;
pub use format::*;
pub use light::{sun_direction, AlephCamera};
pub use material::AlephMaterial;

/// Relative crossfade width around each LOD switch distance. Bevy dithers crossfades in screen space, which TAA can't
/// smooth, so keep the band narrow.
const FADE: f32 = 0.04;
const WALL_HEIGHT: f32 = 20.0;
/// Instances smaller than this (grass clumps, small bushes) cast no shadows: in alpha-tested foliage they were the
/// largest cost of the shadow pass, for shadows the cascades blur away and SSAO already hints at.
const SHADOW_SIZE: f32 = 1.0;
/// Instances this tall or wide keep the scene's full LOD distances.
const SIZE_REFERENCE: f32 = 10.0;

pub struct AlephScenePlugin {
    pub folder: PathBuf,
    /// Largest texture side kept on the GPU; bigger textures drop their top mip levels. 0 keeps every level.
    pub max_texture: u32,
}

impl Plugin for AlephScenePlugin {
    fn build(&self, app: &mut App) {
        let folder = self.folder.canonicalize().unwrap_or_else(|_| self.folder.clone());
        let scene = SceneFile::read(&folder).unwrap_or_else(|e| panic!("{}: {e}", folder.join("scene.json").display()));
        let ground = scene.chunks.iter().filter_map(|c| c.ground.as_ref()).filter_map(|g| match Heights::read(&folder, g) {
            Ok(h) => Some(h),
            Err(e) => {
                error!("{}: {e}", g.file);
                None
            }
        });
        let ground = ground.collect();
        let reader = AssetSource::get_default_reader(folder.to_string_lossy().into_owned());
        app.register_asset_source("aleph", AssetSourceBuilder::new(reader))
            .insert_resource(AlephWorld { folder, scene, ground })
            .insert_resource(mips::MaxTexture(self.max_texture))
            .init_resource::<AlephStatus>()
            .init_resource::<Models>()
            .init_resource::<material::DepthMaps>()
            .add_systems(Startup, (spawn_world, light::spawn_sun, skirt::spawn))
            .add_systems(Update, ((spawn_models, track_loading).chain(), light::setup_camera, (material::invert_heights, mips::generate).chain()));
    }
}

/// Scene materials and the bounce light pass; add after `DefaultPlugins`.
pub struct AlephRenderPlugin;

impl Plugin for AlephRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((bounce::BouncePlugin, material::MaterialsPlugin));
    }
}

/// The loaded scene.json and ground heights, for games that need them (spawn, minimaps…).
#[derive(Resource)]
pub struct AlephWorld {
    pub folder: PathBuf,
    pub scene: SceneFile,
    pub ground: Vec<Heights>,
}

impl AlephWorld {
    pub fn ground_height(&self, x: f32, z: f32) -> Option<f32> {
        self.ground.iter().find_map(|h| h.sample(x, z))
    }

    /// Bounds' bounding rectangle as (min, max) in x, z.
    pub fn extent(&self) -> (Vec2, Vec2) {
        let points = self.scene.bounds.iter().map(|p| Vec2::from(*p));
        let points = points.chain(self.scene.chunks.iter().flat_map(|c| [vec2(c.min[0], c.min[2]), vec2(c.max[0], c.max[2])]));
        points.fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)))
    }
}

/// `loaded` turns true once every glTF and texture is in and spawned.
#[derive(Resource, Default)]
pub struct AlephStatus {
    pub loaded: bool,
    pub models: usize,
    pub failed: usize,
}

/// Marks a chunk's root entity.
#[derive(Component)]
pub struct Chunk {
    pub index: usize,
}

/// LOD of a spawned mesh, for debug tinting.
#[derive(Component, Clone, Copy)]
pub struct Lod(pub u8);

/// A glTF to spawn under this entity once it has loaded.
#[derive(Component)]
struct LoadModel {
    path: String,
    lod: Option<u8>,
    /// LOD switch distances; None draws at any distance.
    distances: Option<[f32; 3]>,
    /// Instance scale, which grows the meshes' bounding radii.
    scale: f32,
    colliders: bool,
    tint: Option<[f32; 3]>,
    /// Draw pieces only beyond this distance (the horizon's `from`).
    from: Option<f32>,
    /// Cast shadows; small instances don't.
    shadows: bool,
}

/// A glTF flattened into meshes and colliders relative to its root, shared by every instance.
struct Model {
    pieces: Vec<Piece>,
    colliders: Vec<(Transform, Collider)>,
}

/// A mesh of a flattened glTF.
struct Piece {
    transform: Transform,
    mesh: Handle<Mesh>,
    material: Handle<AlephMaterial>,
    /// Bounding radius of the glTF node holding it.
    radius: f32,
    /// Decals cast no shadows.
    decal: bool,
}

#[derive(Resource, Default)]
struct Models {
    handles: HashMap<String, Handle<Gltf>>,
    flat: HashMap<String, Arc<Model>>,
    /// Scene materials by library name, shared by every glTF that uses them so their meshes batch together.
    materials: HashMap<String, (Handle<AlephMaterial>, bool)>,
    /// Tinted material copies, by original and tint in 1/32 steps so similar tints share one material.
    tinted: HashMap<(AssetId<AlephMaterial>, [u8; 3]), Handle<AlephMaterial>>,
}

impl Models {
    fn load(&mut self, server: &AssetServer, path: &str) -> Handle<Gltf> {
        self.handles.entry(path.to_owned()).or_insert_with(|| server.load(AssetPath::from(path.to_owned()).with_source("aleph"))).clone()
    }
}

/// Visibility range of LOD `lod` (0–2) given the switch distances, measured from the mesh origin, which `flatten`
/// puts at its glTF node's bounds centre.
pub fn lod_range(distances: [f32; 3], lod: usize) -> VisibilityRange {
    let edge = |d: f32| d * (1.0 - FADE)..d * (1.0 + FADE);
    VisibilityRange {
        start_margin: if lod == 0 { 0.0..0.0 } else { edge(distances[lod - 1]) },
        end_margin: edge(distances[lod]),
        // Not the AABB: Bevy's CPU culling would use its centre but the GPU crossfade dither the mesh origin.
        use_aabb: false,
    }
}

fn spawn_world(mut commands: Commands, world: Res<AlephWorld>, mut models: ResMut<Models>, server: Res<AssetServer>) {
    let scene = &world.scene;
    // In a fixed order: load order decides how meshes and materials are packed on the GPU.
    let mut assets: Vec<_> = scene.assets.iter().collect();
    assets.sort_by_key(|(name, _)| *name);
    for path in scene.chunks.iter().flat_map(|c| &c.lods).chain(assets.iter().flat_map(|(_, a)| &a.lods)) {
        models.load(&server, path);
    }
    for (index, chunk) in scene.chunks.iter().enumerate() {
        // Static body at the chunk centre; collider nodes and LOD meshes hang below it.
        commands
            .spawn((Name::new(format!("chunk {}", chunk.id)), Chunk { index }, RigidBody::Static,
                    Transform::from_translation(Vec3::from(chunk.center)), Visibility::default()))
            .with_children(|parent| {
                for (lod, path) in chunk.lods.iter().enumerate().take(3) {
                    parent.spawn((Transform::default(), Visibility::default(), LoadModel {
                        path: path.clone(), lod: Some(lod as u8), distances: Some(scene.lod_distances), scale: 1.0, colliders: lod == 0,
                        tint: None, from: None, shadows: true,
                    }));
                }
            });
        for instance in &chunk.instances {
            let Some(asset) = scene.assets.get(&instance.asset) else {
                warn!("chunk {}: unknown asset {}", chunk.id, instance.asset);
                continue;
            };
            // Small things switch LODs and vanish sooner, as they cover fewer pixels: a 10 m tree keeps the scene's
            // distances, a 1.5 m bush a quarter of them.
            let extent = asset.height.max(2.0 * asset.radius) * instance.scale;
            let size = (extent / SIZE_REFERENCE).clamp(0.25, 1.0);
            let transform = Transform::from_translation(Vec3::from(instance.position))
                .with_rotation(Quat::from_rotation_y(instance.yaw))
                .with_scale(Vec3::splat(instance.scale));
            commands
                .spawn((RigidBody::Static, transform, Visibility::default()))
                .with_children(|parent| {
                    for (lod, path) in asset.lods.iter().enumerate().take(3) {
                        parent.spawn((Transform::default(), Visibility::default(), LoadModel {
                            path: path.clone(), lod: Some(lod as u8), distances: Some(scene.lod_distances.map(|d| d * size)), scale: instance.scale,
                            // glTF collider nodes only stand in when scene.json defines none for the asset.
                            colliders: lod == 0 && asset.colliders.is_empty(),
                            tint: instance.tint.filter(|t| t.iter().any(|&c| (c - 1.0).abs() > 0.01)),
                            from: None,
                            shadows: extent >= SHADOW_SIZE,
                        }));
                    }
                    for c in &asset.colliders {
                        if let Some((collider, centre)) = asset_collider(c) {
                            parent.spawn((collider, Transform::from_translation(centre)));
                        }
                    }
                });
        }
    }
    if let Some(horizon) = &scene.horizon {
        // Each glTF node of the horizon shows once its bounds centre is beyond `from`; split it into tiles to use that.
        let from = (horizon.from > 0.0).then_some(horizon.from);
        commands.spawn((Name::new("horizon"), Transform::default(), Visibility::default(),
                        LoadModel { path: horizon.file.clone(), lod: None, distances: None, scale: 1.0, colliders: false, tint: None, from,
                                    shadows: false }));
    }
    for heights in &world.ground {
        commands.spawn((Name::new("ground"), RigidBody::Static, heights.collider(), heights.transform()));
    }
    spawn_walls(&mut commands, &world);
}

/// Invisible walls along the play area polygon, from below the lowest ground to WALL_HEIGHT above the highest.
fn spawn_walls(commands: &mut Commands, world: &AlephWorld) {
    let bounds = &world.scene.bounds;
    let (lo, hi) = world.ground.iter().flat_map(|h| h.data.iter().copied()).filter(|v| v.is_finite())
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let (lo, hi) = if lo > hi { (0.0, 0.0) } else { (lo, hi) };
    let (bottom, top) = (lo - 5.0, hi + WALL_HEIGHT);
    for (i, a) in bounds.iter().enumerate() {
        let (a, b) = (Vec2::from(*a), Vec2::from(bounds[(i + 1) % bounds.len()]));
        let edge = b - a;
        if edge.length() < 1e-3 {
            continue;
        }
        let mid = (a + b) / 2.0;
        commands.spawn((Name::new("wall"), RigidBody::Static, Collider::cuboid(edge.length() + 1.0, top - bottom, 0.5),
                        Transform::from_xyz(mid.x, (bottom + top) / 2.0, mid.y)
                            .with_rotation(Quat::from_rotation_y(-edge.y.atan2(edge.x)))));
    }
}

/// A scene.json asset collider and its offset from the asset origin.
fn asset_collider(c: &ColliderDef) -> Option<(Collider, Vec3)> {
    let points = || c.vertices.iter().map(|&p| Vec3::from(p)).collect::<Vec<_>>();
    match c.shape.as_str() {
        "convex" => Collider::convex_hull(&points()).map(|h| (h, Vec3::ZERO)),
        "trimesh" if c.vertices.len() >= 3 => {
            Some((Collider::trimesh(points(), c.indices.as_chunks::<3>().0.to_vec()), Vec3::ZERO))
        }
        _ => shape(&c.shape, c.radius, c.half_height).map(|s| (s, Vec3::from(c.center))),
    }
}

fn shape(kind: &str, radius: f32, half_height: f32) -> Option<Collider> {
    // half_height is half the total height, caps included.
    match kind {
        "capsule" => Some(Collider::capsule(radius, (2.0 * (half_height - radius)).max(0.0))),
        "cylinder" => Some(Collider::cylinder(radius, 2.0 * half_height)),
        _ => {
            warn!("unsupported collider shape {kind}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_models(
    mut commands: Commands,
    pending: Query<(Entity, &LoadModel)>,
    mut models: ResMut<Models>,
    mut status: ResMut<AlephStatus>,
    server: Res<AssetServer>,
    gltfs: Res<Assets<Gltf>>,
    nodes: Res<Assets<GltfNode>>,
    meshes: Res<Assets<GltfMesh>>,
    gltf_materials: Res<Assets<GltfMaterial>>,
    mut mesh_data: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AlephMaterial>>,
    mut depth_maps: ResMut<material::DepthMaps>,
) {
    for (entity, load) in &pending {
        let handle = models.load(&server, &load.path);
        if let Some(LoadState::Failed(e)) = server.get_load_state(&handle) {
            error!("{}: {e}", load.path);
            status.failed += 1;
            commands.entity(entity).remove::<LoadModel>();
            continue;
        }
        let Some(gltf) = gltfs.get(&handle) else { continue };
        let model = match models.flat.get(&load.path) {
            Some(model) => model.clone(),
            None => {
                let mut assets = FlattenAssets {
                    nodes: &nodes, meshes: &meshes, gltf_materials: &gltf_materials, data: &mut mesh_data,
                    materials: &mut materials, depth_maps: &mut depth_maps, server: &server,
                };
                let model = Arc::new(flatten(&load.path, gltf, &mut assets, &mut models.materials));
                models.flat.insert(load.path.clone(), model.clone());
                model
            }
        };
        let pieces: Vec<_> = model.pieces.iter().map(|piece| {
            let Piece { transform, mesh, material, radius, decal } = piece;
            let material = match load.tint {
                Some(tint) => tinted(&mut models.tinted, &mut materials, material, tint),
                None => material.clone(),
            };
            // Switch LODs by distance to the node's bounding sphere rather than its centre, so a 200 m building keeps
            // its LOD0 while you stand at either end. All LODs of a node share about the same bounds, so they stay in step.
            let range = load.distances.zip(load.lod).map(|(d, lod)| lod_range(d.map(|d| d + radius * load.scale), lod as usize))
                .or(load.from.map(|from| VisibilityRange {
                    start_margin: from * (1.0 - FADE)..from * (1.0 + FADE),
                    end_margin: skirt::RADIUS * 2.0..skirt::RADIUS * 2.0,
                    use_aabb: false,
                }));
            (*transform, mesh.clone(), material, range, *decal)
        }).collect();
        let mut root = commands.entity(entity);
        root.remove::<LoadModel>();
        root.with_children(|parent| {
            for (transform, mesh, material, range, decal) in pieces {
                let mut piece = parent.spawn((transform, Mesh3d(mesh), MeshMaterial3d(material)));
                if let Some(range) = range {
                    piece.insert(range);
                }
                if decal || !load.shadows {
                    piece.insert(NotShadowCaster);
                }
                match load.lod {
                    Some(lod) => piece.insert(Lod(lod)),
                    None => piece.insert((NotShadowCaster, NotShadowReceiver)),
                };
            }
            if load.colliders {
                for (transform, collider) in &model.colliders {
                    parent.spawn((*transform, collider.clone()));
                }
            }
        });
        status.models += 1;
    }
}

fn tinted(
    cache: &mut HashMap<(AssetId<AlephMaterial>, [u8; 3]), Handle<AlephMaterial>>,
    materials: &mut Assets<AlephMaterial>,
    material: &Handle<AlephMaterial>,
    tint: [f32; 3],
) -> Handle<AlephMaterial> {
    let key = tint.map(|c| (c.clamp(0.0, 2.0) * 32.0).round() as u8);
    cache.entry((material.id(), key)).or_insert_with(|| {
        let mut copy = materials.get(material).cloned().unwrap_or_default();
        let [r, g, b] = key.map(|c| c as f32 / 32.0);
        let c = copy.base.base_color.to_linear();
        copy.base.base_color = LinearRgba::new(c.red * r, c.green * g, c.blue * b, c.alpha).into();
        materials.add(copy)
    }).clone()
}

/// What `flatten` reads and adds to.
struct FlattenAssets<'a> {
    nodes: &'a Assets<GltfNode>,
    meshes: &'a Assets<GltfMesh>,
    gltf_materials: &'a Assets<GltfMaterial>,
    data: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<AlephMaterial>,
    depth_maps: &'a mut material::DepthMaps,
    server: &'a AssetServer,
}

/// Every mesh primitive and collider node of the glTF's scene, with transforms relative to its root.
///
/// Meshes are recentred in place on their node's bounds, so LOD distances measure from where the geometry is rather
/// than from the chunk centre all chunk vertices are relative to.
fn flatten(path: &str, gltf: &Gltf, assets: &mut FlattenAssets, materials: &mut HashMap<String, (Handle<AlephMaterial>, bool)>) -> Model {
    let FlattenAssets { nodes, meshes, gltf_materials, data, .. } = assets;
    let names: HashMap<AssetId<GltfMaterial>, &str> = gltf.named_materials.iter().map(|(name, h)| (h.id(), &**name)).collect();
    let mut model = Model { pieces: vec![], colliders: vec![] };
    let mut shifted = HashMap::new();
    let mut panes = std::collections::HashSet::new();
    let all: Vec<&GltfNode> = gltf.nodes.iter().filter_map(|h| nodes.get(h)).collect();
    let children: std::collections::HashSet<usize> =
        all.iter().flat_map(|n| n.children.iter().filter_map(|c| nodes.get(c)).map(|c| c.index)).collect();
    let mut stack: Vec<(&GltfNode, Transform)> =
        all.iter().filter(|n| !children.contains(&n.index)).map(|n| (*n, n.transform)).collect();
    while let Some((node, transform)) = stack.pop() {
        for child in node.children.iter().filter_map(|c| nodes.get(c)) {
            stack.push((child, transform.mul_transform(child.transform)));
        }
        let primitives = node.mesh.as_ref().and_then(|m| meshes.get(m)).map(|m| m.primitives.as_slice()).unwrap_or(&[]);
        if node.name.starts_with("collider/") {
            // Helper nodes: never rendered, only read for their shape.
            let extras: serde_json::Value = node.extras.as_ref().and_then(|e| serde_json::from_str(&e.value).ok()).unwrap_or_default();
            let aleph = &extras["aleph"];
            let kind = aleph["collider"].as_str().unwrap_or("");
            let float = |key: &str| aleph[key].as_f64().unwrap_or(0.0) as f32;
            let positions: Vec<Vec3> = primitives.iter().filter_map(|p| data.get(&p.mesh)).flat_map(positions).collect();
            let collider = match kind {
                "convex" => Collider::convex_hull(&positions),
                "trimesh" => {
                    let indices: Vec<u32> = primitives.iter().filter_map(|p| data.get(&p.mesh)).flat_map(|m| match m.indices() {
                        Some(Indices::U16(i)) => i.iter().map(|&v| v as u32).collect(),
                        Some(Indices::U32(i)) => i.clone(),
                        None => (0..m.count_vertices() as u32).collect(),
                    }).collect();
                    (positions.len() >= 3).then(|| Collider::trimesh(positions, indices.as_chunks::<3>().0.to_vec()))
                }
                _ => shape(kind, float("radius"), float("half_height")),
            };
            match collider {
                Some(collider) => model.colliders.push((transform, collider)),
                None => warn!("{path}: bad collider node {}", node.name),
            }
            continue;
        }
        // Bounds of the whole node, shared by its primitives so they switch LODs together.
        let (lo, hi) = primitives.iter().filter_map(|p| data.get(&p.mesh)).flat_map(positions)
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
        let (centre, radius) = if lo.x > hi.x { (Vec3::ZERO, 0.0) } else {
            ((lo + hi) / 2.0, transform.transform_point(hi).distance(transform.transform_point(lo)) / 2.0)
        };
        for primitive in primitives {
            // A mesh used by several nodes is shifted once, by the first node's centre.
            let centre = *shifted.entry(primitive.mesh.id()).or_insert_with(|| {
                if let Some(mut mesh) = data.get_mut(&primitive.mesh)
                    && let Some(VertexAttributeValues::Float32x3(p)) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
                {
                    p.iter_mut().for_each(|v| *v = (Vec3::from(*v) - centre).to_array());
                }
                centre
            });
            let transform = transform.mul_transform(Transform::from_translation(centre));
            let name = primitive.material.as_ref().and_then(|m| names.get(&m.id())).copied().unwrap_or("");
            let extras = material::Extras::read(name, primitive.material_extras.as_ref().map(|e| e.value.as_str()));
            if extras.interior && panes.insert(primitive.mesh.id())
                && let Some(mut mesh) = data.get_mut(&primitive.mesh)
            {
                material::add_panes(&mut mesh, name);
            }
            let (material, decal) = materials.entry(name.to_owned()).or_insert_with(|| {
                let base = primitive.material.as_ref().and_then(|m| gltf_materials.get(m))
                    .map(bevy::pbr::gltf::standard_material_from_gltf_material).unwrap_or_default();
                let built = material::build(base, &extras, assets.server, assets.depth_maps);
                (assets.materials.add(built), extras.decal)
            }).clone();
            model.pieces.push(Piece { transform, mesh: primitive.mesh.clone(), material, radius, decal });
        }
    }
    model
}

fn positions(mesh: &Mesh) -> Vec<Vec3> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p.iter().map(|&v| Vec3::from(v)).collect(),
        _ => vec![],
    }
}

/// Flags the world loaded once every glTF is spawned and its textures are in.
fn track_loading(mut status: ResMut<AlephStatus>, pending: Query<(), With<LoadModel>>, models: Res<Models>, server: Res<AssetServer>) {
    if status.loaded || !pending.is_empty() {
        return;
    }
    let done = models.handles.values().all(|h| {
        server.is_loaded_with_dependencies(h) || matches!(server.get_load_state(h), Some(LoadState::Failed(_)))
            || server.get_recursive_dependency_load_state(h).is_some_and(|s| s.is_failed())
    });
    if done {
        info!("scene loaded: {} models spawned, {} failed", status.models, status.failed);
        status.loaded = true;
    }
}

/// A chunk's ground heights from its `.height` file.
pub struct Heights {
    pub cols: usize,
    pub rows: usize,
    pub spacing: f32,
    pub offset: Vec2,
    /// Row-major, rows advancing along +z.
    pub data: Vec<f32>,
}

impl Heights {
    pub fn read(folder: &Path, ground: &GroundDef) -> Result<Self, String> {
        let bytes = std::fs::read(folder.join(&ground.file)).map_err(|e| e.to_string())?;
        let [cols, rows] = ground.size;
        if bytes.len() != cols * rows * 4 || cols < 2 || rows < 2 {
            return Err(format!("{} bytes for {cols}×{rows} heights", bytes.len()));
        }
        let data = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok(Self { cols, rows, spacing: ground.spacing, offset: Vec2::from(ground.offset), data })
    }

    pub fn size(&self) -> Vec2 {
        vec2((self.cols - 1) as f32, (self.rows - 1) as f32) * self.spacing
    }

    /// Bilinear height at a point inside the grid.
    pub fn sample(&self, x: f32, z: f32) -> Option<f32> {
        let f = (vec2(x, z) - self.offset) / self.spacing;
        if f.x < 0.0 || f.y < 0.0 || f.x > (self.cols - 1) as f32 || f.y > (self.rows - 1) as f32 {
            return None;
        }
        let (i, j) = ((f.x as usize).min(self.cols - 2), (f.y as usize).min(self.rows - 2));
        let (tx, tz) = (f.x - i as f32, f.y - j as f32);
        let h = |i: usize, j: usize| self.data[j * self.cols + i];
        Some((h(i, j) * (1.0 - tx) + h(i + 1, j) * tx) * (1.0 - tz) + (h(i, j + 1) * (1.0 - tx) + h(i + 1, j + 1) * tx) * tz)
    }

    pub fn collider(&self) -> Collider {
        // parry wants a column-major (z, x) matrix, centred on the collider's origin.
        let mut zx = Vec::with_capacity(self.data.len());
        for i in 0..self.cols {
            zx.extend((0..self.rows).map(|j| self.data[j * self.cols + i]));
        }
        let size = self.size();
        SharedShape::heightfield(Array2::new(self.rows, self.cols, zx), vec3(size.x, 1.0, size.y)).into()
    }

    pub fn transform(&self) -> Transform {
        let c = self.offset + self.size() / 2.0;
        Transform::from_xyz(c.x, 0.0, c.y)
    }
}
