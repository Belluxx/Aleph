//! First-person controller: a kinematic capsule moved with avian's shape casts (move and slide).

use aleph_scene::{AlephCamera, AlephStatus, AlephWorld};
use avian3d::prelude::*;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

pub const RADIUS: f32 = 0.3;
pub const HEIGHT: f32 = 1.8;
pub const EYE: f32 = 1.7;
const CROUCH_HEIGHT: f32 = 1.2;
const CROUCH_EYE: f32 = 1.1;
const STEP: f32 = 0.35;
/// cos 45°: steeper surfaces act as walls.
const WALKABLE: f32 = std::f32::consts::FRAC_1_SQRT_2;
const WALK: f32 = 4.5;
const SPRINT: f32 = 8.0;
const CROUCH: f32 = 2.0;
const FLY: f32 = 12.0;
const JUMP: f32 = 5.0;
const GRAVITY: f32 = 9.81;
const LOOK: f32 = 0.002;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(Update, (grab_cursor, look, walk).chain());
    }
}

#[derive(Component)]
pub struct Player {
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: Vec3,
    pub grounded: bool,
    pub noclip: bool,
    pub crouched: bool,
    /// Feet position.
    pub feet: Vec3,
}

impl Player {
    fn height(&self) -> f32 {
        if self.crouched { CROUCH_HEIGHT } else { HEIGHT }
    }

    pub fn eye(&self) -> Vec3 {
        self.feet + Vec3::Y * if self.crouched { CROUCH_EYE } else { EYE }
    }
}

fn capsule(height: f32) -> Collider {
    Collider::capsule(RADIUS, height - 2.0 * RADIUS)
}

/// Spawn eye position, lifted so the feet are not below the ground.
pub fn spawn_eye(world: &AlephWorld) -> Vec3 {
    let p = Vec3::from(world.scene.spawn.position);
    let ground = world.ground_height(p.x, p.z).unwrap_or(f32::MIN);
    vec3(p.x, (p.y - EYE).max(ground) + EYE, p.z)
}

fn spawn(mut commands: Commands, world: Res<AlephWorld>) {
    let eye = spawn_eye(&world);
    let yaw = world.scene.spawn.yaw;
    commands.spawn((Name::new("player"), Player {
        yaw, pitch: 0.0, velocity: Vec3::ZERO, grounded: false, noclip: false, crouched: false, feet: eye - Vec3::Y * EYE,
    }));
    commands.spawn((Name::new("camera"), Camera3d::default(), AlephCamera,
                    Transform::from_translation(eye).with_rotation(Quat::from_rotation_y(yaw))));
}

fn grab_cursor(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>, mouse: Res<ButtonInput<MouseButton>>, keys: Res<ButtonInput<KeyCode>>) {
    let Ok(mut cursor) = cursor.single_mut() else { return };
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

fn look(mut player: Single<&mut Player>, motion: Res<AccumulatedMouseMotion>, cursor: Query<&CursorOptions, With<PrimaryWindow>>) {
    if cursor.single().is_ok_and(|c| c.grab_mode == CursorGrabMode::None) {
        return;
    }
    player.yaw -= motion.delta.x * LOOK;
    player.pitch = (player.pitch - motion.delta.y * LOOK).clamp(-1.54, 1.54);
}

fn walk(
    mut player: Single<&mut Player>,
    mut camera: Single<&mut Transform, With<AlephCamera>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    status: Res<AlephStatus>,
    casts: MoveAndSlide,
) {
    let p = &mut **player;
    let dt = time.delta_secs().min(0.05);
    let rotation = Quat::from_euler(EulerRot::YXZ, p.yaw, p.pitch, 0.0);
    let axis = |a: KeyCode, b: KeyCode| keys.pressed(a) as i32 as f32 - keys.pressed(b) as i32 as f32;
    let input = vec3(axis(KeyCode::KeyD, KeyCode::KeyA), 0.0, axis(KeyCode::KeyS, KeyCode::KeyW));
    let sprint = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    if keys.just_pressed(KeyCode::KeyV) {
        p.noclip = !p.noclip;
        p.velocity = Vec3::ZERO;
    }

    if p.noclip {
        let up = axis(KeyCode::Space, KeyCode::KeyC);
        let speed = FLY * if sprint { 4.0 } else { 1.0 };
        p.feet += (rotation * input.normalize_or_zero() + Vec3::Y * up) * speed * dt;
    } else if status.loaded {
        step(p, &casts, &keys, input, sprint, dt);
    }
    camera.translation = p.eye();
    camera.rotation = rotation;
}

/// One kinematic step: crouch, horizontal move with step-up, then gravity and ground snapping.
fn step(p: &mut Player, casts: &MoveAndSlide, keys: &ButtonInput<KeyCode>, input: Vec3, sprint: bool, dt: f32) {
    let filter = SpatialQueryFilter::default();
    let config = MoveAndSlideConfig::default();
    let skin = config.skin_width;
    // The capsule's centre sits half its height above the feet.
    let centre = |p: &Player| p.feet + Vec3::Y * p.height() / 2.0;

    let crouch = keys.pressed(KeyCode::KeyC);
    if crouch != p.crouched {
        // Stand up only with headroom above the crouched capsule.
        let blocked = !crouch && casts.cast_move(&capsule(CROUCH_HEIGHT), centre(p), Quat::IDENTITY,
                                                 Vec3::Y * (HEIGHT - CROUCH_HEIGHT), skin, &filter).is_some();
        if !blocked {
            p.crouched = crouch;
        }
    }
    let shape = capsule(p.height());
    let half = Vec3::Y * p.height() / 2.0;

    let speed = if p.crouched { CROUCH } else if sprint { SPRINT } else { WALK };
    let wish = Quat::from_rotation_y(p.yaw) * input.normalize_or_zero() * speed;
    // Snappy on the ground, little control in the air.
    let blend = if p.grounded { 1.0 - (-15.0 * dt).exp() } else { 1.0 - (-1.5 * dt).exp() };
    let horizontal = p.velocity.with_y(0.0).lerp(wish, blend);
    p.velocity = horizontal.with_y(p.velocity.y);

    let slide = |from: Vec3, velocity: Vec3, walls: bool| {
        casts.move_and_slide(&shape, from, Quat::IDENTITY, velocity, std::time::Duration::from_secs_f32(dt), &config, &filter, |hit| {
            // Too-steep slopes block like walls instead of being climbed.
            if walls && hit.normal.y < WALKABLE && hit.normal.y > -0.1
                && let Ok(flat) = Dir3::new(hit.normal.with_y(0.0))
            {
                *hit.normal = flat;
            }
            MoveAndSlideHitResponse::Accept
        })
    };

    let mut position = centre(p);
    if horizontal.length_squared() > 1e-6 {
        let plain = slide(position, horizontal, true).position;
        let mut best = plain;
        if p.grounded {
            // Step up: lift, move, drop back down; keep it if that got further on walkable ground.
            let lift = casts.cast_move(&shape, position, Quat::IDENTITY, Vec3::Y * STEP, skin, &filter).map_or(STEP, |h| h.distance);
            let raised = slide(position + Vec3::Y * lift, horizontal, true).position;
            if let Some(hit) = casts.cast_move(&shape, raised, Quat::IDENTITY, -Vec3::Y * (lift + 0.05), skin, &filter)
                && walkable(casts, &filter, hit.point1, raised)
            {
                let landed = raised - Vec3::Y * hit.distance;
                if (landed - position).xz().length() > (plain - position).xz().length() + 0.01 {
                    best = landed;
                }
            }
        }
        position = best;
    }

    if p.grounded && keys.just_pressed(KeyCode::Space) {
        p.velocity.y = JUMP;
        p.grounded = false;
    }
    let walkable_below = |from: Vec3, distance: f32| {
        casts.cast_move(&shape, from, Quat::IDENTITY, -Vec3::Y * distance, skin, &filter).filter(|h| walkable(casts, &filter, h.point1, from))
    };
    if p.grounded {
        // Stick to the ground down slopes and off small steps.
        match walkable_below(position, STEP) {
            Some(hit) => {
                position -= Vec3::Y * hit.distance;
                p.velocity.y = 0.0;
            }
            None => p.grounded = false,
        }
    }
    if !p.grounded {
        p.velocity.y -= GRAVITY * dt;
        let out = slide(position, Vec3::Y * p.velocity.y, false);
        position = out.position;
        p.velocity.y = out.projected_velocity.y;
        if p.velocity.y <= 0.0 && walkable_below(position, 0.05).is_some() {
            p.grounded = true;
            p.velocity.y = 0.0;
        }
    }
    p.feet = position - half;
}

/// Whether the surface under a contact is flat enough to stand on. The cast's own normal is unreliable on step edges,
/// where the capsule touches a corner, so probe the surface just past the contact, away from the capsule's centre.
fn walkable(casts: &MoveAndSlide, filter: &SpatialQueryFilter, point: Vec3, centre: Vec3) -> bool {
    let outward = (point - centre).with_y(0.0).normalize_or_zero() * 0.02;
    casts.spatial_query.cast_ray(point + outward + Vec3::Y * 0.05, Dir3::NEG_Y, 0.15, true, filter)
        .is_some_and(|hit| hit.normal.y >= WALKABLE)
}
