//! Scene materials: Bevy's StandardMaterial from the glTF, extended by the features its `extras.aleph` asks for.
//!
//! - `interior`: window glass with fake rooms behind it (interior mapping) under a Fresnel reflection;
//! - `wind`: vertex sway for foliage;
//! - `triplanar`: world-space texture projection for stone without good UVs;
//! - `height` + `parallax`: parallax occlusion mapping from a height map;
//! - `decal`: alpha-blended overlays drawn just in front of the surface below.
//!
//! Every scene mesh uses [`AlephMaterial`], also when no feature is on, so all of them share the shader's temporal
//! LOD crossfade (see `shaders/aleph.wesl`).

use std::collections::HashSet;

use bevy::asset::{embedded_asset, AssetPath};
use bevy::image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexAttributeValues};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline, ParallaxMappingMethod};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError};
use bevy::shader::{ShaderDefVal, ShaderRef};

pub type AlephMaterial = ExtendedMaterial<StandardMaterial, AlephExtension>;

const SHADER: &str = "embedded://aleph_scene/shaders/aleph.wesl";
/// Parallax layers; POM is the most expensive per-pixel effect, so keep it modest.
pub const PARALLAX_LAYERS: f32 = 12.0;
/// Constant depth bias of decals over the surface below, on top of the lift the Python side gives them.
const DECAL_BIAS: f32 = 64.0;
/// Glass and leaves named by the library before materials carried `interior` and `wind` extras.
const GLASS: [&str; 4] = ["glass/window", "shop/glass", "door/metal-glass", "wall/glass-curtain"];

// Bindless like StandardMaterial: otherwise every material needs its own bind group and draws stop batching, which
// cost more than half the frame rate.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
#[data(50, AlephParams, binding_array(101))]
#[bindless(index_table(range(50..51), binding(100)))]
#[bind_group_data(AlephKey)]
pub struct AlephExtension {
    pub params: AlephParams,
    /// Feature bits, compiled into the shader per material.
    pub features: u32,
}

impl From<&AlephExtension> for AlephParams {
    fn from(e: &AlephExtension) -> Self {
        e.params
    }
}

#[derive(ShaderType, Reflect, Debug, Clone, Copy, Default)]
pub struct AlephParams {
    /// Metres per texture repeat, for triplanar mapping.
    pub size: f32,
    /// Room height of glass without pane data (curtain walls), metres.
    pub storey: f32,
    pub _pad: Vec2,
}

pub const INTERIOR: u32 = 1;
pub const WIND: u32 = 2;
pub const TRIPLANAR: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlephKey(u32);

impl From<&AlephExtension> for AlephKey {
    fn from(e: &AlephExtension) -> Self {
        Self(e.features)
    }
}

impl MaterialExtension for AlephExtension {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn specialize(
        _: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let bits = key.bind_group_data.0;
        let defs: Vec<ShaderDefVal> = [(INTERIOR, "ALEPH_INTERIOR"), (WIND, "ALEPH_WIND"), (TRIPLANAR, "ALEPH_TRIPLANAR")]
            .into_iter()
            .filter(|(bit, _)| bits & bit != 0)
            .map(|(_, name)| name.into())
            .collect();
        descriptor.vertex.shader_defs.extend(defs.iter().cloned());
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs.extend(defs);
        }
        Ok(())
    }
}

pub(crate) struct MaterialsPlugin;

impl Plugin for MaterialsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/aleph.wesl");
        app.add_plugins(MaterialPlugin::<AlephMaterial>::default());
    }
}

/// Height maps to turn into Bevy depth maps (1 − height) as they load.
#[derive(Resource, Default)]
pub(crate) struct DepthMaps(pub HashSet<AssetId<Image>>);

/// What a glTF material's extras ask for.
#[derive(Default)]
pub(crate) struct Extras {
    pub size: f32,
    pub interior: bool,
    pub wind: bool,
    pub triplanar: bool,
    pub decal: bool,
    pub height: Option<String>,
    pub parallax: f32,
}

impl Extras {
    /// From `extras.aleph`, falling back on library names for glass and leaves. Unknown keys are ignored.
    pub fn read(name: &str, json: Option<&str>) -> Self {
        let value: serde_json::Value = json.and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default();
        let a = &value["aleph"];
        let flag = |key: &str| a[key].as_bool().unwrap_or(false);
        Self {
            size: a["size"].as_f64().unwrap_or(0.0) as f32,
            interior: flag("interior") || GLASS.contains(&name),
            wind: flag("wind") || name.starts_with("veg/leaves-"),
            triplanar: flag("triplanar"),
            decal: flag("decal"),
            height: a["height"].as_str().map(str::to_owned),
            parallax: a["parallax"].as_f64().unwrap_or(0.0) as f32,
        }
    }
}

/// The scene material for a glTF StandardMaterial and its extras.
pub(crate) fn build(mut base: StandardMaterial, extras: &Extras, server: &AssetServer, depth_maps: &mut DepthMaps) -> AlephMaterial {
    if let Some(height) = &extras.height && extras.parallax > 0.0 {
        let path = AssetPath::from(format!("library/{height}")).with_source("aleph").into_owned();
        let image: Handle<Image> = server.load_builder().with_settings(|s: &mut ImageLoaderSettings| {
            s.is_srgb = false;
            s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                ..ImageSamplerDescriptor::linear()
            });
        }).load(path);
        depth_maps.0.insert(image.id());
        base.depth_map = Some(image);
        // `parallax` is metres of relief; Bevy measures depth in texture repeats.
        base.parallax_depth_scale = if extras.size > 0.0 { extras.parallax / extras.size } else { extras.parallax };
        base.parallax_mapping_method = ParallaxMappingMethod::Relief { max_steps: 4 };
        base.max_parallax_layer_count = PARALLAX_LAYERS;
    }
    if extras.decal {
        // Blended or masked as the glTF says; only pulled in front of the surface below.
        base.depth_bias = DECAL_BIAS;
    }
    let mut features = 0;
    if extras.interior {
        features |= INTERIOR;
    }
    if extras.wind {
        features |= WIND;
    }
    if extras.triplanar {
        features |= TRIPLANAR;
    }
    let params = AlephParams { size: if extras.size > 0.0 { extras.size } else { 2.0 }, storey: 3.0, _pad: Vec2::ZERO };
    ExtendedMaterial { base, extension: AlephExtension { params, features } }
}

/// Room data for interior-mapped glass, per vertex.
///
/// Each connected piece of glass up to MAX_PANE wide is a pane with its own room behind it: UV_1 holds the position
/// along the facade from the pane's centre and the height above the room's floor, COLOR_0 the pane's top above the
/// floor, the room width, the pane's half width, and 1. Larger glass (curtain walls) gets absolute facade
/// coordinates and alpha 0, and the shader lays rooms out on a grid.
pub(crate) fn add_panes(mesh: &mut Mesh, name: &str) {
    const MAX_PANE: f32 = 6.0;
    let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { return };
    let positions: Vec<Vec3> = positions.iter().map(|&p| Vec3::from(p)).collect();
    let normals: Vec<Vec3> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(n)) => n.iter().map(|&n| Vec3::from(n)).collect(),
        _ => vec![Vec3::Z; positions.len()],
    };
    let indices: Vec<usize> = match mesh.indices() {
        Some(i) => i.iter().collect(),
        None => (0..positions.len()).collect(),
    };
    // Union-find over shared vertices: one component per pane.
    let mut parent: Vec<usize> = (0..positions.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for tri in indices.chunks_exact(3) {
        let a = root(&mut parent, tri[0]);
        for &v in &tri[1..] {
            let b = root(&mut parent, v);
            parent[b] = a;
        }
    }
    // Along-facade axis from the normal, as the shader computes it.
    let along = |n: Vec3| vec3(n.z, 0.0, -n.x).normalize_or(Vec3::X);
    let mut bounds: std::collections::HashMap<usize, (Vec2, Vec2)> = Default::default();
    for (i, p) in positions.iter().enumerate() {
        let r = root(&mut parent, i);
        let q = vec2(p.dot(along(normals[i])), p.y);
        let e = bounds.entry(r).or_insert((q, q));
        *e = (e.0.min(q), e.1.max(q));
    }
    // Sill height under the pane: shopfronts and doors reach nearly to the floor, windows sit on a parapet.
    let sill = |height: f32| match name {
        n if n.starts_with("shop/") => 0.4,
        n if n.starts_with("door/") => 0.0,
        _ if height > 2.4 => 0.15,
        _ => 0.9,
    };
    let mut uvs = Vec::with_capacity(positions.len());
    let mut colors = Vec::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        let (lo, hi) = bounds[&root(&mut parent, i)];
        let size = hi - lo;
        let u = p.dot(along(normals[i]));
        if size.x > MAX_PANE || size.y > MAX_PANE || normals[i].y.abs() > 0.5 {
            uvs.push([u, p.y]);
            colors.push([3.0, 3.0, 0.0, 0.0]);
            continue;
        }
        let floor = lo.y - sill(size.y);
        // One room per pane up to 4.5 m wide, else rooms about 4 m wide side by side.
        let room = if size.x <= 4.5 { 4.0f32.max(size.x + 0.6) } else { size.x / (size.x / 4.0).round() };
        uvs.push([u - (lo.x + hi.x) / 2.0, p.y - floor]);
        colors.push([hi.y - floor, room, size.x / 2.0, 1.0]);
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

/// Turns height maps into depth maps (1 − height, as RGBA8 so `mips::generate`, which runs next, builds their mips).
pub(crate) fn invert_heights(mut events: MessageReader<AssetEvent<Image>>, mut images: ResMut<Assets<Image>>, maps: Res<DepthMaps>) {
    use bevy::render::render_resource::TextureFormat;
    for event in events.read() {
        let AssetEvent::Added { id } = event else { continue };
        if !maps.0.contains(id) {
            continue;
        }
        let Some(mut image) = images.get_mut(*id) else { continue };
        let Some(data) = image.data.take() else { continue };
        let heights: Vec<u8> = match image.texture_descriptor.format {
            TextureFormat::R8Unorm => data,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => data.chunks_exact(4).map(|p| p[0]).collect(),
            TextureFormat::R16Unorm => data.chunks_exact(2).map(|p| p[1]).collect(),
            TextureFormat::Rgba16Unorm => data.chunks_exact(8).map(|p| p[1]).collect(),
            other => {
                warn!("height map format {other:?} not supported");
                image.data = Some(data);
                continue;
            }
        };
        image.data = Some(heights.iter().flat_map(|&h| [255 - h, 255 - h, 255 - h, 255]).collect());
        image.texture_descriptor.format = TextureFormat::Rgba8Unorm;
    }
}
