//! aleph_view GAME_FOLDER [--shots DIR | --bench SECONDS]: walk an Aleph game world in first person.

mod auto;
mod debug;
mod player;

use std::path::PathBuf;
use std::time::Duration;

use aleph_scene::{AlephRenderPlugin, AlephScenePlugin};
use avian3d::prelude::*;
use bevy::app::ScheduleRunnerPlugin;
use bevy::diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy::render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

const USAGE: &str = "usage: aleph_view GAME_FOLDER [--max-texture PIXELS] [--shots DIR [--views FILE] | --bench SECONDS]";

fn main() -> AppExit {
    let mut args = std::env::args().skip(1);
    let (mut folder, mut mode, mut views) = (None, None, None);
    let mut max_texture = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--shots" => mode = args.next().map(|d| auto::Mode::Shots(PathBuf::from(d), None)),
            "--views" => views = args.next().map(PathBuf::from),
            // 1024 roughly halves texture memory, for GPUs shared with other programs.
            "--max-texture" => max_texture = args.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--bench" => mode = args.next().and_then(|s| s.parse().ok()).map(auto::Mode::Bench),
            "-h" | "--help" => {
                println!("{USAGE}");
                return AppExit::Success;
            }
            _ if folder.is_none() && !arg.starts_with('-') => folder = Some(PathBuf::from(arg)),
            _ => folder = None,
        }
    }
    if let (Some(auto::Mode::Shots(_, file)), Some(path)) = (&mut mode, views) {
        *file = Some(path);
    }
    let Some(folder) = folder.filter(|f| f.join("scene.json").is_file()) else {
        eprintln!("{USAGE}\nGAME_FOLDER must contain scene.json");
        return AppExit::error();
    };

    let mut app = App::new();
    app.add_plugins(AlephScenePlugin { folder: folder.clone(), max_texture });
    let render = RenderPlugin {
        render_creation: WgpuSettings { features: WgpuFeatures::POLYGON_MODE_LINE, ..default() }.into(),
        // Shots and benches must not catch materials whose pipelines are still compiling: those draw nothing.
        synchronous_pipeline_compilation: mode.is_some(),
        ..default()
    };
    if let Some(mode) = mode {
        // Headless: render into an image, no window on the desktop.
        app.add_plugins((
            DefaultPlugins
                .set(WindowPlugin { primary_window: None, exit_condition: ExitCondition::DontExit, ..default() })
                .set(render)
                .disable::<WinitPlugin>(),
            ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        ));
        app.add_plugins(auto::AutoPlugin { mode });
    } else {
        let window = Window { title: format!("Aleph – {}", folder.display()), ..default() };
        app.add_plugins(DefaultPlugins.set(WindowPlugin { primary_window: Some(window), ..default() }).set(render));
        app.add_plugins(player::PlayerPlugin);
    }
    app.add_plugins((
        AlephRenderPlugin,
        PhysicsPlugins::default(),
        PhysicsDebugPlugin,
        WireframePlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        EntityCountDiagnosticsPlugin::default(),
        debug::DebugPlugin,
    ));
    app.run()
}
