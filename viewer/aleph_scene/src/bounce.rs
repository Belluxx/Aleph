//! Bounce light for the atmosphere's environment map.
//!
//! Bevy fills the environment map from the sky alone, so everything below the horizon is black: shade gets only blue
//! skylight and turns strongly blue, and rough surfaces at grazing angles reflect pure sky. Outdoors, about half the
//! light reaching a wall in shade comes back up from sunlit ground and facades, which is warm and neutral. Before
//! Bevy filters the map each frame, this pass replaces the lower hemisphere with sunlit ground and blends a band of
//! lit facades and trees over the sky near the horizon.

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::pbr::generate::{downsampling_system, RenderEnvironmentMap};
use bevy::prelude::*;
use bevy::render::extract_resource::{extract_resource, ExtractResource};
use bevy::render::render_resource::binding_types::{texture_2d_array, texture_storage_2d_array, uniform_buffer_sized};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::TextureCache;
use bevy::render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

/// Mean albedo of the ground seen from street level: paving, asphalt, and some grass.
pub const GROUND_ALBEDO: [f32; 3] = [0.20, 0.19, 0.17];
/// Share of that ground in sunlight.
pub const GROUND_SUNLIT: f32 = 0.65;
/// Mean albedo of what stands around the horizon: plaster, stone, glass, and trees.
pub const SURROUND_ALBEDO: [f32; 3] = [0.30, 0.28, 0.24];
/// How much of the sky the surroundings hide at the horizon; it fades out by about 15° up.
pub const SURROUND_COVER: f32 = 0.7;

/// Saturation of skylight in the environment map. Bevy's clear-sky skylight is far bluer than daylight in photos,
/// where haze, the city's own bounce light, and white balance pull shade toward neutral.
pub const SKY_SATURATION: f32 = 0.5;

/// The sun as the bounce pass needs it, set by `light::spawn_sun`.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Bounce {
    /// Illuminance on a surface facing the sun at ground level, after the atmosphere (lux per channel).
    pub sun: Vec3,
    /// Sine of the sun's elevation.
    pub sin_elevation: f32,
}

pub(crate) struct BouncePlugin;

impl Plugin for BouncePlugin {
    fn build(&self, _: &mut App) {}

    // The render app may only exist once every plugin is built.
    fn finish(&self, app: &mut App) {
        embedded_asset!(app, "shaders/bounce.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
        render_app.add_systems(ExtractSchedule, extract_resource::<Bounce, RenderApp, ()>).add_systems(RenderStartup, init).add_systems(
            Render,
            bounce.after(RenderSystems::PrepareBindGroups).before(downsampling_system).before(RenderSystems::Render),
        );
    }
}

#[derive(Resource)]
struct BouncePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init(mut commands: Commands, cache: Res<PipelineCache>, server: Res<AssetServer>) {
    let layout = BindGroupLayoutDescriptor::new(
        "aleph_bounce_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                texture_storage_2d_array(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer_sized(false, None),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("aleph_bounce".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(server.as_ref(), "shaders/bounce.wesl"),
        ..default()
    });
    commands.insert_resource(BouncePipeline { layout, pipeline });
}

fn bounce(
    maps: Query<&RenderEnvironmentMap>,
    params: Option<Res<Bounce>>,
    pipeline: Option<Res<BouncePipeline>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    mut textures: ResMut<TextureCache>,
    mut ctx: RenderContext,
) {
    let (Some(params), Some(pipeline)) = (params, pipeline) else { return };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else { return };
    let layout = cache.get_bind_group_layout(&pipeline.layout);
    let g = GROUND_ALBEDO;
    let s = SURROUND_ALBEDO;
    let values = [
        params.sun.x, params.sun.y, params.sun.z, params.sin_elevation,
        g[0], g[1], g[2], GROUND_SUNLIT,
        s[0], s[1], s[2], SURROUND_COVER,
        SKY_SATURATION, 0.0, 0.0, 0.0,
    ];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("aleph_bounce_params"),
        contents: &bytes,
        usage: BufferUsages::UNIFORM,
    });
    for map in &maps {
        let target = &map.environment_map.texture;
        let size = map.environment_map.texture_descriptor.size;
        // The pass reads the sky from a copy and writes the result back over the original.
        let sky = textures.get(&device, TextureDescriptor {
            label: Some("aleph_bounce_sky"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let array = TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), mip_level_count: Some(1), ..default() };
        let sky_view = sky.texture.create_view(&array);
        let target_view = target.create_view(&array);
        let bind_group = device.create_bind_group(
            "aleph_bounce",
            &layout,
            &BindGroupEntries::sequential((&sky_view, &target_view, uniform.as_entire_binding())),
        );
        let encoder = ctx.command_encoder();
        encoder.copy_texture_to_texture(target.as_image_copy(), sky.texture.as_image_copy(), size);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("aleph_bounce"), timestamp_writes: None });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 6);
    }
}
