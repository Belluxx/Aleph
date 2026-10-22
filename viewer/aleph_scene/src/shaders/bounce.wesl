// Replaces the black lower hemisphere of the atmosphere's environment map with lit ground, and hides the sky near
// the horizon behind lit surroundings. Works in cube map space, whose y is up as in the world.
import bevy_pbr::render::utils::{sample_cube_dir, dir_to_cube_uv};

struct Params {
    // rgb: sun illuminance at the ground on a surface facing it; a: sine of its elevation.
    sun: vec4<f32>,
    // rgb: ground albedo; a: share of the ground in sunlight.
    ground: vec4<f32>,
    // rgb: albedo of facades and trees around; a: share of the horizon they hide.
    surround: vec4<f32>,
    // x: saturation of skylight.
    sky: vec4<f32>,
}

@group(0) @binding(0) var sky: texture_2d_array<f32>;
@group(0) @binding(1) var output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265;

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let c = dir_to_cube_uv(dir);
    let size = textureDimensions(sky);
    let p = min(vec2<u32>(c.uv * vec2<f32>(size)), size - 1u);
    let color = textureLoad(sky, p, c.face, 0).rgb;
    return mix(vec3<f32>(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722))), color, params.sky.x);
}

// Skylight on an upward-facing surface: cosine-distributed samples of the upper hemisphere.
fn sky_irradiance() -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        for (var j = 0u; j < 16u; j++) {
            let u = (f32(i) + 0.5) / 8.0;
            let phi = (f32(j) + 0.5) / 16.0 * 2.0 * PI;
            let r = sqrt(u);
            sum += sky_radiance(vec3<f32>(r * cos(phi), sqrt(1.0 - u), r * sin(phi)));
        }
    }
    return sum / 128.0 * PI;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y || id.z >= 6u) {
        return;
    }
    let dir = sample_cube_dir((vec2<f32>(id.xy) + 0.5) / vec2<f32>(size), id.z);
    let sky_e = sky_irradiance();
    let sun = params.sun.rgb;
    // Lambertian ground, partly shaded, lit by the sun and the sky.
    let ground = params.ground.rgb / PI * (sun * params.sun.a * params.ground.a + sky_e);
    // Walls around: on average lit by the sun at cos(elevation) / π, and seeing half sky, half ground.
    let cos_elevation = sqrt(max(1.0 - params.sun.a * params.sun.a, 0.0));
    let surround = params.surround.rgb / PI * (sun * cos_elevation / PI + 0.5 * sky_e + 0.5 * PI * ground);
    var color: vec3<f32>;
    if (dir.y < 0.0) {
        // Below the horizon, walls give way to ground.
        color = mix(surround, ground, smoothstep(0.0, 0.3, -dir.y));
    } else {
        let cover = params.surround.a * (1.0 - smoothstep(0.0, 0.26, dir.y));
        color = mix(sky_radiance(dir), surround, cover);
    }
    textureStore(output, vec2<i32>(id.xy), i32(id.z), vec4<f32>(color, 1.0));
}
