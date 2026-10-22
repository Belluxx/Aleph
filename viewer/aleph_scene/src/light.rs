//! Sun, sky, and camera effects.

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::{Exposure, Hdr};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::light::{light_consts::lux, Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, SunDisk};
use bevy::pbr::{AtmosphereSettings, ScreenSpaceAmbientOcclusion};
use bevy::prelude::*;
use bevy::render::view::{ColorGrading, ColorGradingGlobal};

use crate::bounce::Bounce;
use crate::{skirt, AlephWorld};

/// Camera exposure: a sunny-16 day.
pub const EV100: f32 = 13.0;
/// Saturation after tonemapping.
pub const SATURATION: f32 = 0.9;
/// SSAO sampling radius, metres.
pub const AO_RADIUS: f32 = 1.5;
/// Aerosol density relative to Hillaire's clear-sky reference.
const AEROSOLS: f32 = 1.5;
/// Distance at which haze leaves 5 % contrast.
const HAZE: f32 = 6000.0;

/// Add to a `Camera3d` to give it the scene's sky, aerial haze, and post-processing.
#[derive(Component, Default)]
pub struct AlephCamera;

/// Unit vector toward the sun: azimuth clockwise from north (-Z), elevation above the horizon.
pub fn sun_direction(azimuth: f32, elevation: f32) -> Vec3 {
    let (a, e) = (azimuth.to_radians(), elevation.to_radians());
    vec3(a.sin() * e.cos(), e.sin(), -a.cos() * e.cos())
}

pub(crate) fn spawn_sun(mut commands: Commands, world: Res<AlephWorld>, mut media: ResMut<Assets<ScatteringMedium>>) {
    let sun = &world.scene.sun;
    let toward = sun_direction(sun.azimuth, sun.elevation.max(1.0));
    commands.spawn((
        Name::new("sun"),
        DirectionalLight { illuminance: lux::RAW_SUNLIGHT, shadow_maps_enabled: true, ..default() },
        // Three cascades to 120 m: crisp near the player, and alpha-masked foliage in every cascade is the main GPU cost
        // (four cascades to 200 m cost ~30% more frame time on a dense park).
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            first_cascade_far_bound: 12.0,
            maximum_distance: 120.0,
            ..default()
        }
        .build(),
        Transform::from_translation(toward).looking_at(Vec3::ZERO, Vec3::Y),
        SunDisk::EARTH,
    ));
    commands.spawn((Name::new("atmosphere"), Atmosphere::earth(media.add(medium()))));
    commands.insert_resource(Bounce { sun: lux::RAW_SUNLIGHT * transmittance(toward.y), sin_elevation: toward.y });
}

/// Bevy's earth medium with its Mie terms the right way round: aerosols mostly scatter (Hillaire 2020: scattering
/// 3.996e-6, absorption 0.444e-6 per metre), times AEROSOLS for a city's haze. As shipped they mostly absorb, which
/// leaves a deep blue sky without the pale horizon of a real one, and skylight that turns shade saturated blue.
fn medium() -> ScatteringMedium {
    let mut medium = ScatteringMedium::earth(256, 256);
    let mie = &mut medium.terms[1];
    mie.scattering = Vec3::splat(3.996e-6 * AEROSOLS);
    mie.absorption = Vec3::splat(0.444e-6 * AEROSOLS);
    medium
}

/// Sunlight left after crossing the atmosphere at sin(elevation) `mu`: optical depths of `ScatteringMedium::earth`'s
/// Rayleigh, Mie and ozone terms at sea level, with Kasten and Young's air mass.
fn transmittance(mu: f32) -> Vec3 {
    let depth = vec3(0.0562, 0.1365, 0.2661) + 0.0053 * AEROSOLS;
    let elevation = mu.asin().to_degrees().max(0.0);
    let air_mass = 1.0 / (mu.max(0.0) + 0.50572 * (elevation + 6.07995).powf(-1.6364));
    (-depth * air_mass).exp()
}

pub(crate) fn setup_camera(mut commands: Commands, cameras: Query<Entity, Added<AlephCamera>>) {
    for camera in &cameras {
        commands.entity(camera).insert((
            Hdr,
            Msaa::Off,
            AtmosphereSettings { aerial_view_lut_max_distance: skirt::RADIUS, ..default() },
            // A pale haze matched to the horizon sky, so the skirt melts into it; the atmosphere's own aerial
            // perspective is too clear at a few km. Keep it neutral: a saturated blue tints dark roofs within 200 m.
            DistanceFog {
                color: Color::linear_rgb(1.7, 1.85, 2.1),
                directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
                directional_light_exponent: 30.0,
                falloff: FogFalloff::from_visibility_squared(HAZE),
            },
            AtmosphereEnvironmentMapLight::default(),
            Exposure { ev100: EV100 },
            // Filmic contrast; slightly desaturated, since ACES pushes sunlit sandstone and brick toward orange.
            // TonyMcMapface and AgX looked flat and grey next to photos of the same places.
            Tonemapping::AcesFitted,
            ColorGrading { global: ColorGradingGlobal { post_saturation: SATURATION, ..default() }, ..default() },
            // A wider radius than Bevy's 0.73 m, so building corners, column bases and window reveals darken.
            ScreenSpaceAmbientOcclusion { radius: AO_RADIUS, constant_object_thickness: 0.5, ..default() },
            TemporalAntiAliasing::default(),
            Projection::Perspective(PerspectiveProjection { fov: 75f32.to_radians(), near: 0.05, far: skirt::RADIUS * 1.5, ..default() }),
        ));
    }
}
