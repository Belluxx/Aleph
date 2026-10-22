//! Headless modes: --shots renders fixed viewpoints to PNGs, --bench flies a path and prints frame times.

use std::path::PathBuf;
use std::time::Instant;

use aleph_scene::{AlephCamera, AlephStatus, AlephWorld};
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};

use crate::debug;
use crate::player::{spawn_eye, EYE};

pub enum Mode {
    /// Output folder, and optionally a JSON list of {"name", "eye": [x, y, z], "at": [x, y, z]} views.
    Shots(PathBuf, Option<PathBuf>),
    Bench(f32),
}

pub struct AutoPlugin {
    pub mode: Mode,
}

/// Frames to let TAA, SSAO and the atmosphere settle after the world loads and after each camera jump.
const SETTLE: u32 = 40;
/// Give up waiting for assets after this long and render what is there.
const LOAD_TIMEOUT: f32 = 120.0;

type View = (String, Transform, bool);

#[derive(Resource)]
struct Auto {
    /// Output folder and views: file name, camera, LOD tint.
    shots: Option<(PathBuf, Vec<View>)>,
    bench: f32,
    target: Handle<Image>,
    frame: u32,
    started: Instant,
    loaded_at: Option<f32>,
    taken: usize,
    saved: usize,
    times: Vec<f32>,
}

impl Plugin for AutoPlugin {
    fn build(&self, app: &mut App) {
        let (shots, bench) = match &self.mode {
            Mode::Shots(dir, file) => (Some((dir.clone(), file.as_deref().map(read_views).unwrap_or_default())), 0.0),
            Mode::Bench(seconds) => (None, *seconds),
        };
        app.insert_resource(Auto {
            shots, bench, target: Handle::default(), frame: 0, started: Instant::now(), loaded_at: None, taken: 0, saved: 0,
            times: vec![],
        })
        .add_systems(Startup, setup)
        .add_systems(Update, run);
    }
}

fn setup(mut commands: Commands, mut auto: ResMut<Auto>, mut images: ResMut<Assets<Image>>, world: Res<AlephWorld>) {
    let (width, height) = if auto.shots.is_some() { (1600, 900) } else { (1920, 1080) };
    let mut target = Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
    target.asset_usage = RenderAssetUsages::RENDER_WORLD;
    auto.target = images.add(target);
    let eye = spawn_eye(&world);
    let yaw = world.scene.spawn.yaw;
    if let Some((dir, views)) = &mut auto.shots {
        std::fs::create_dir_all(&*dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        // Views from --views replace the default ones.
        if views.is_empty() {
            for k in 0..4 {
                let yaw = yaw + k as f32 * std::f32::consts::FRAC_PI_2;
                let name = format!("eye-{:03}.png", (yaw.to_degrees().rem_euclid(360.0)).round() as i32);
                views.push((name, Transform::from_translation(eye).with_rotation(Quat::from_rotation_y(yaw)), false));
            }
            // Overview from the south, high enough to see the whole play area and the LOD bands.
            let (lo, hi) = world.extent();
            let centre = (lo + hi) / 2.0;
            let size = (hi - lo).max_element();
            let ground = world.ground_height(centre.x, centre.y).unwrap_or(0.0);
            let look = vec3(centre.x, ground, centre.y);
            let overview = Transform::from_translation(look + vec3(0.0, 0.45 * size, 0.6 * size)).looking_at(look, Vec3::Y);
            views.push(("overview.png".into(), overview, false));
            views.push(("overview-lod.png".into(), overview, true));
        }
    }
    commands.spawn((Camera3d::default(), AlephCamera, RenderTarget::Image(auto.target.clone().into()),
                    Transform::from_translation(eye).with_rotation(Quat::from_rotation_y(yaw))));
}

#[derive(serde::Deserialize)]
struct ViewFile {
    name: String,
    eye: [f32; 3],
    at: [f32; 3],
}

/// Named cameras for --views, so the same close-ups can be compared between builds.
fn read_views(path: &std::path::Path) -> Vec<View> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let views: Vec<ViewFile> = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    views.into_iter().map(|v| {
        let name = if v.name.ends_with(".png") { v.name } else { format!("{}.png", v.name) };
        (name, Transform::from_translation(Vec3::from(v.eye)).looking_at(Vec3::from(v.at), Vec3::Y), false)
    }).collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    mut commands: Commands,
    mut auto: ResMut<Auto>,
    status: Res<AlephStatus>,
    world: Res<AlephWorld>,
    mut camera: Single<&mut Transform, With<AlephCamera>>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
    mut tints: ResMut<debug::Tints>,
    mut materials: ResMut<Assets<aleph_scene::AlephMaterial>>,
    mut pieces: Query<debug::Piece>,
    meshes: Query<&ViewVisibility, With<Mesh3d>>,
) {
    let elapsed = auto.started.elapsed().as_secs_f32();
    if auto.loaded_at.is_none() {
        if !status.loaded && elapsed < LOAD_TIMEOUT {
            return;
        }
        if !status.loaded {
            warn!("assets still loading after {LOAD_TIMEOUT} s; rendering anyway");
        }
        info!("loaded in {elapsed:.1} s");
        auto.loaded_at = Some(elapsed);
        auto.frame = 0;
    }
    auto.frame += 1;
    let frame = auto.frame;

    if let Some((dir, views)) = auto.shots.clone() {
        // Shot k: jump the camera at k * SETTLE, capture SETTLE frames later.
        let k = (frame / SETTLE) as usize;
        if frame.is_multiple_of(SETTLE) && k >= 1 && k <= views.len() && auto.taken < k {
            let (name, _, _) = &views[k - 1];
            let path = dir.join(name);
            auto.taken = k;
            commands.spawn(Screenshot::image(auto.target.clone())).observe(save_to_disk(path.clone())).observe(
                move |_: On<ScreenshotCaptured>, mut auto: ResMut<Auto>| {
                    println!("{}", path.display());
                    auto.saved += 1;
                },
            );
        }
        if frame % SETTLE == 1 && k < views.len() {
            let (_, transform, tint) = &views[k];
            **camera = *transform;
            if *tint != tints.on {
                tints.on = *tint;
                debug::set_tint(&mut commands, &mut tints, &mut materials, &mut pieces);
            }
        }
        if auto.saved == views.len() || elapsed > LOAD_TIMEOUT * 2.0 {
            exit.write(if auto.saved == views.len() { AppExit::Success } else { AppExit::error() });
        }
        return;
    }

    // Bench: a loop around the play area at eye height, one lap per 60 s, after SETTLE warm-up frames. The path
    // advances a fixed 1/60 s per frame, so every run renders the same views however fast it goes (stalls such as
    // pipeline compiles would otherwise move the camera to another part of the lap).
    let (lo, hi) = world.extent();
    let centre = (lo + hi) / 2.0;
    let radius = 0.3 * (hi - lo).min_element();
    let t = frame.saturating_sub(SETTLE) as f32 / 60.0;
    let angle = t / 60.0 * std::f32::consts::TAU;
    let p = centre + radius * vec2(angle.cos(), angle.sin());
    let ground = world.ground_height(p.x, p.y).unwrap_or(0.0);
    camera.translation = vec3(p.x, ground + EYE, p.y);
    camera.rotation = Quat::from_rotation_y(-angle);
    if frame > SETTLE {
        auto.times.push(time.delta_secs() * 1000.0);
    }
    if t > auto.bench || elapsed > LOAD_TIMEOUT + auto.bench + 60.0 {
        let mut times = auto.times.clone();
        times.sort_by(f32::total_cmp);
        let pick = |q: f32| times.get(((times.len() as f32 * q) as usize).min(times.len().saturating_sub(1))).copied().unwrap_or(0.0);
        let mean = times.iter().sum::<f32>() / times.len().max(1) as f32;
        println!("frames {}  mean {mean:.2} ms ({:.0} fps)  p50 {:.2} ms  p95 {:.2} ms  p99 {:.2} ms  max {:.2} ms",
                 times.len(), 1000.0 / mean.max(1e-3), pick(0.5), pick(0.95), pick(0.99), pick(1.0));
        println!("meshes {} ({} visible at the end)", meshes.iter().count(), meshes.iter().filter(|v| v.get()).count());
        exit.write(AppExit::Success);
    }
}
