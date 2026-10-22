//! Land beyond the play area: a ring from the bounds out to the horizon, so the world doesn't end in the
//! atmosphere's planet ground. Its inner edge follows the ground heights along the bounds, then it eases to a flat
//! level within BLEND metres.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::AlephWorld;

/// Outer radius, metres from the play area's centre.
pub const RADIUS: f32 = 6000.0;
const BLEND: f32 = 200.0;
/// Sits a little below the play-area ground so chunk ground overlapping the bounds always wins.
const DROP: f32 = 0.15;
/// Metres one texture repeat covers: large, so the tiling reads as fields from afar.
const TILE: f32 = 25.0;
const GRASS: &str = "library/materials/ground/grass/color.jpg";

pub(crate) fn spawn(
    mut commands: Commands,
    world: Res<AlephWorld>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    server: Res<AssetServer>,
) {
    let Some(mesh) = ring(&world) else { return };
    let texture = world.folder.join(GRASS).is_file().then(|| {
        server.load_builder().with_settings(|s: &mut ImageLoaderSettings| {
            s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                ..ImageSamplerDescriptor::linear()
            });
        }).load(format!("aleph://{GRASS}"))
    });
    // Muted, so it reads as distant fields rather than lawn.
    let base_color = if texture.is_some() { Color::srgb(0.8, 0.8, 0.74) } else { Color::srgb(0.27, 0.31, 0.21) };
    let material = materials.add(StandardMaterial { base_color, base_color_texture: texture, perceptual_roughness: 0.95, ..default() });
    commands.spawn((Name::new("skirt"), Mesh3d(meshes.add(mesh)), MeshMaterial3d(material), NotShadowCaster, Transform::default()));
}

fn ring(world: &AlephWorld) -> Option<Mesh> {
    let bounds: Vec<Vec2> = world.scene.bounds.iter().map(|p| Vec2::from(*p)).collect();
    if bounds.len() < 3 {
        return None;
    }
    let centre = bounds.iter().copied().sum::<Vec2>() / bounds.len() as f32;
    // Points every 4 m along the bounds, with the ground height just inside them.
    let mut edge = vec![];
    for (i, &a) in bounds.iter().enumerate() {
        let b = bounds[(i + 1) % bounds.len()];
        let steps = ((b - a).length() / 4.0).ceil().max(1.0) as usize;
        for k in 0..steps {
            let p = a.lerp(b, k as f32 / steps as f32);
            let inside = p + (centre - p).normalize_or_zero() * 0.5;
            edge.push((p, world.ground_height(inside.x, inside.y)));
        }
    }
    // The far level: a low percentile of the edge, so the land never floats above a sunken edge.
    let mut heights: Vec<f32> = edge.iter().filter_map(|(_, h)| *h).collect();
    heights.sort_by(f32::total_cmp);
    let level = heights.get(heights.len() / 10).copied().unwrap_or(0.0) - DROP;

    // Three rings: the edge, BLEND metres out, and RADIUS from the centre.
    let mut positions = vec![];
    for &(p, h) in &edge {
        let out = (p - centre).normalize_or_zero();
        let mid = p + out * BLEND;
        let far = centre + out * RADIUS.max((p - centre).length() + 2.0 * BLEND);
        positions.extend([vec3(p.x, h.map_or(level, |h| h - DROP), p.y), vec3(mid.x, level, mid.y), vec3(far.x, level, far.y)]);
    }
    let n = edge.len() as u32;
    let mut indices = vec![];
    for i in 0..n {
        let j = (i + 1) % n;
        for r in 0..2 {
            let (a, b, c, d) = (i * 3 + r, j * 3 + r, j * 3 + r + 1, i * 3 + r + 1);
            for tri in [[a, b, c], [a, c, d]] {
                // Face up whichever way the bounds wind.
                let [p0, p1, p2] = tri.map(|k| positions[k as usize]);
                let up = (p1 - p0).cross(p2 - p0).y > 0.0;
                indices.extend(if up { tri } else { [tri[0], tri[2], tri[1]] });
            }
        }
    }
    let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p.x / TILE, p.z / TILE]).collect();
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    let positions: Vec<[f32; 3]> = positions.iter().map(|p| p.to_array()).collect();
    Some(Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices)))
}
