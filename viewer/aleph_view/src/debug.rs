//! Debug keys: F3 stats, F4 wireframe, F5 colliders, F6 LOD tint, F12 screenshot.

use std::collections::HashMap;

use aleph_scene::{AlephCamera, AlephMaterial, AlephWorld, Chunk, Lod};
use avian3d::prelude::PhysicsGizmos;
use bevy::diagnostic::{DiagnosticPath, DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::wireframe::WireframeConfig;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PrimaryWindow;

use crate::player::Player;

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tints>()
            .add_systems(Startup, setup)
            .add_systems(Update, (keys, stats));
    }
}

#[derive(Component)]
struct Stats;

/// F6 state: tinted copies of materials, and the originals to restore.
#[derive(Resource, Default)]
pub struct Tints {
    pub on: bool,
    copies: HashMap<(AssetId<AlephMaterial>, u8), Handle<AlephMaterial>>,
}

#[derive(Component)]
pub struct Untinted(Handle<AlephMaterial>);

pub type Piece = (Entity, &'static Lod, &'static mut MeshMaterial3d<AlephMaterial>, Option<&'static Untinted>);

fn setup(mut commands: Commands, mut gizmos: ResMut<GizmoConfigStore>, windows: Query<(), With<PrimaryWindow>>) {
    gizmos.config_mut::<PhysicsGizmos>().0.enabled = false;
    if !windows.is_empty() {
        commands.spawn((Stats, Text::default(), TextFont { font_size: FontSize::Px(14.0), ..default() },
                        TextShadow::default(), Visibility::Hidden,
                        Node { position_type: PositionType::Absolute, top: px(8), left: px(8), ..default() }));
    }
}

#[allow(clippy::too_many_arguments)]
fn keys(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut stats: Query<&mut Visibility, With<Stats>>,
    mut wireframe: ResMut<WireframeConfig>,
    mut gizmos: ResMut<GizmoConfigStore>,
    mut tints: ResMut<Tints>,
    mut materials: ResMut<Assets<AlephMaterial>>,
    mut pieces: Query<Piece>,
    world: Res<AlephWorld>,
) {
    if input.just_pressed(KeyCode::F3) {
        for mut v in &mut stats {
            *v = if *v == Visibility::Hidden { Visibility::Inherited } else { Visibility::Hidden };
        }
    }
    if input.just_pressed(KeyCode::F4) {
        wireframe.global = !wireframe.global;
    }
    if input.just_pressed(KeyCode::F5) {
        let config = &mut gizmos.config_mut::<PhysicsGizmos>().0;
        config.enabled = !config.enabled;
    }
    if input.just_pressed(KeyCode::F6) {
        tints.on = !tints.on;
        set_tint(&mut commands, &mut tints, &mut materials, &mut pieces);
    }
    if input.just_pressed(KeyCode::F12) {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let path = world.folder.join(format!("screenshot-{stamp}.png"));
        info!("screenshot {}", path.display());
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    }
}

/// LOD0 white, LOD1 blue, LOD2 red.
pub fn set_tint(
    commands: &mut Commands,
    tints: &mut Tints,
    materials: &mut Assets<AlephMaterial>,
    pieces: &mut Query<Piece>,
) {
    const COLORS: [Color; 3] = [Color::srgb(1.0, 1.0, 1.0), Color::srgb(0.15, 0.35, 1.0), Color::srgb(1.0, 0.15, 0.1)];
    for (entity, lod, mut material, untinted) in pieces {
        match (tints.on, untinted) {
            (true, None) => {
                let original = material.0.clone();
                let key = (original.id(), lod.0);
                let copy = tints.copies.entry(key).or_insert_with(|| {
                    let mut m = materials.get(&original).cloned().unwrap_or_default();
                    // Multiply over the texture, which alpha-masked leaves need for their shape.
                    m.base.base_color = COLORS[lod.0.min(2) as usize];
                    materials.add(m)
                });
                material.0 = copy.clone();
                commands.entity(entity).insert(Untinted(original));
            }
            (false, Some(Untinted(original))) => {
                material.0 = original.clone();
                commands.entity(entity).remove::<Untinted>();
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stats(
    mut text: Query<(&mut Text, &Visibility), With<Stats>>,
    diagnostics: Res<DiagnosticsStore>,
    meshes: Query<(&ViewVisibility, Option<&Lod>, &ChildOf), With<Mesh3d>>,
    parents: Query<&ChildOf>,
    chunks: Query<(), With<Chunk>>,
    camera: Query<&GlobalTransform, With<AlephCamera>>,
    player: Query<&Player>,
    time: Res<Time>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    let Ok((mut text, visibility)) = text.single_mut() else { return };
    if *visibility == Visibility::Hidden || *timer < 0.25 {
        return;
    }
    *timer = 0.0;
    let value = |path: &DiagnosticPath| diagnostics.get(path).and_then(|d| d.smoothed()).unwrap_or(0.0);
    let fps = value(&FrameTimeDiagnosticsPlugin::FPS);
    let ms = value(&FrameTimeDiagnosticsPlugin::FRAME_TIME);
    let entities = value(&EntityCountDiagnosticsPlugin::ENTITY_COUNT);
    // Visible meshes per LOD, and chunks with anything visible (mesh → LOD root → chunk).
    let (mut lods, mut visible, mut seen) = ([0; 3], 0, std::collections::HashSet::new());
    for (view, lod, parent) in &meshes {
        if !view.get() {
            continue;
        }
        visible += 1;
        if let Some(lod) = lod {
            lods[lod.0.min(2) as usize] += 1;
        }
        if let Ok(chunk) = parents.get(parent.parent()).map(|c| c.parent()) && chunks.contains(chunk) {
            seen.insert(chunk);
        }
    }
    let eye = camera.single().map_or(Vec3::ZERO, |t| t.translation());
    let state = player.single().map_or(String::new(), |p| {
        format!("{}{}{}", if p.noclip { "noclip " } else { "" }, if p.grounded { "grounded " } else { "" },
                if p.crouched { "crouched" } else { "" })
    });
    text.0 = format!(
        "{fps:.0} fps  {ms:.2} ms\nentities {entities:.0}\nmeshes {visible} visible / {}  (lod0 {} lod1 {} lod2 {})\n\
         chunks {} visible / {}\npos {:.1} {:.1} {:.1}  {state}",
        meshes.iter().count(), lods[0], lods[1], lods[2], seen.len(), chunks.iter().count(), eye.x, eye.y, eye.z
    );
}
