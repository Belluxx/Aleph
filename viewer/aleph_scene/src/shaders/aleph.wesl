// Scene material: StandardMaterial plus temporal LOD crossfade, wind (ALEPH_WIND), interior-mapped glass
// (ALEPH_INTERIOR), and triplanar projection (ALEPH_TRIPLANAR). One file serves the main pass and the prepass
// (depth, normals, motion vectors, and shadows), which must agree on wind and crossfade.

import bevy_render::globals::Globals;
import bevy_pbr::render::{
    mesh_bindings::mesh,
    mesh_functions,
    pbr_functions,
    pbr_types,
    view_transformations::position_world_to_clip,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex};
@if(PREPASS_PIPELINE)
import bevy_pbr::render::{pbr_prepass_functions, pbr_bindings, mesh_view_bindings::view};
@if(PREPASS_PIPELINE && BINDLESS)
import bevy_pbr::render::pbr_bindings::material_indices;
@if(PREPASS_PIPELINE && BINDLESS)
import bevy_render::bindless::{bindless_samplers_filtering, bindless_textures_2d};

@if(!PREPASS_PIPELINE)
import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    mesh_view_bindings::globals,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
};

// The prepass binds globals at its own slot.
@if(PREPASS_PIPELINE)
@group(0) @binding(1) var<uniform> globals: Globals;

struct AlephParams {
    size: f32,
    storey: f32,
    pad: vec2<f32>,
}

@if(BINDLESS) {
struct AlephIndices {
    params: u32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<storage> aleph_indices: array<AlephIndices>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage> aleph_params: array<AlephParams>;
}
@else {
@group(constants::MATERIAL_BIND_GROUP) @binding(50) var<uniform> aleph_params: AlephParams;
}

fn params(in: VertexOutput) -> AlephParams {
@if(BINDLESS) {
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
    return aleph_params[aleph_indices[slot].params];
}
@else {
    return aleph_params;
}
}

const PI: f32 = 3.14159265;

// Wind ---------------------------------------------------------------------------------------------------------

// Offset of a foliage vertex: a slow sway of the whole crown, stronger higher up, and a quick flutter of leaves.
fn wind(world: vec3<f32>, origin: vec3<f32>, t: f32) -> vec3<f32> {
    let phase = dot(origin.xz, vec2<f32>(0.21, 0.17));
    let gust = 0.65 + 0.35 * sin(t * 0.31 + phase * 0.3);
    let lift = clamp(1.0 + (world.y - origin.y) * 0.15, 0.3, 2.0);
    let sway = (sin(t * 1.05 + phase) + 0.4 * sin(t * 2.3 + phase * 1.7)) * 0.035 * gust * lift;
    let flutter = sin(t * 7.0 + dot(world, vec3<f32>(1.9, 2.7, 2.3))) * 0.012 * gust;
    return vec3<f32>(0.8, 0.0, 0.6) * sway + vec3<f32>(0.3, 1.0, -0.2) * flutter;
}

// LOD crossfade ------------------------------------------------------------------------------------------------

// Bevy's crossfade discards with a fixed 4×4 pattern, which TAA cannot resolve. This uses interleaved gradient noise
// that moves every frame, so TAA averages it into a smooth blend. Fading-out and fading-in LODs get complementary
// coverage (dither d and d − 16), as in Bevy.
fn lod_fade(frag: vec4<f32>, dither: i32) {
    if dither == 0 {
        return;
    }
    if dither <= -16 || dither >= 16 {
        discard;
    }
    let p = frag.xy + 5.588238 * f32(globals.frame_count % 64u);
    let n = fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
    if (dither > 0 && n >= f32(16 - dither) / 16.0) || (dither < 0 && n < f32(-dither) / 16.0) {
        discard;
    }
}

// Vertex -------------------------------------------------------------------------------------------------------

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let vertex = decompress_vertex(vertex_in, vertex_in.instance_index);
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
@if(ALEPH_WIND)
    out.world_position = vec4<f32>(out.world_position.xyz + wind(out.world_position.xyz, world_from_local[3].xyz, globals.time), 1.0);
    out.position = position_world_to_clip(out.world_position.xyz);

@if(PREPASS_PIPELINE && UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
}

@if(VERTEX_UVS_A)
    out.uv = vertex.uv;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;

@if(VERTEX_NORMALS && (!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS))
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex_in.instance_index);
@if(VERTEX_TANGENTS && (!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS))
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(world_from_local, vertex.tangent, vertex_in.instance_index);

@if(VERTEX_COLORS)
    out.color = vertex.color;

@if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS) {
    let previous = mesh_functions::get_previous_world_from_local(vertex_in.instance_index);
    var before = mesh_functions::mesh_position_local_to_world(previous, vec4<f32>(vertex.position, 1.0));
@if(ALEPH_WIND)
    before = vec4<f32>(before.xyz + wind(before.xyz, previous[3].xyz, globals.time - globals.delta_time), 1.0);
    out.previous_world_position = before;
}

@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(vertex_in.instance_index, world_from_local[3]);
    return out;
}

// Prepass fragment: Bevy's pbr_prepass with the crossfade above --------------------------------------------------

@if(PREPASS_PIPELINE && PREPASS_FRAGMENT)
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
@if(VISIBILITY_RANGE_DITHER)
    lod_fade(in.position, in.visibility_range_dither);
    pbr_prepass_functions::prepass_sample_color_and_alpha_discard(in);
    var out: FragmentOutput;

@if(UNCLIPPED_DEPTH_ORTHO_EMULATION)
    out.frag_depth = in.unclipped_depth;

@if(NORMAL_PREPASS) {
@if(BINDLESS)
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
@if(BINDLESS)
    let flags = pbr_bindings::material_array[material_indices[slot].material].flags;
@else
    let flags = pbr_bindings::material.flags;
    let double_sided = (flags & pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let is_logically_front = pbr_functions::winding_corrected_front_facing(mesh[in.instance_index].flags, is_front);
    var normal = pbr_functions::prepare_world_normal(in.world_normal, double_sided, is_logically_front);
@if(VERTEX_UVS && VERTEX_TANGENTS && STANDARD_MATERIAL_NORMAL_MAP && !ALEPH_TRIPLANAR) {
@if(BINDLESS)
    let uv_transform = pbr_bindings::material_array[material_indices[slot].material].uv_transform;
@else
    let uv_transform = pbr_bindings::material.uv_transform;
@if(BINDLESS)
    let texture = bindless_textures_2d[material_indices[slot].normal_map_texture];
@else
    let texture = pbr_bindings::normal_map_texture;
@if(BINDLESS)
    let sampler_ = bindless_samplers_filtering[material_indices[slot].normal_map_sampler];
@else
    let sampler_ = pbr_bindings::normal_map_sampler;
    let uv = (uv_transform * vec3<f32>(in.uv, 1.0)).xy;
    let Nt = textureSampleBias(texture, sampler_, uv, view.mip_bias).rgb;
    let TBN = pbr_functions::calculate_tbn_mikktspace(normal, in.world_tangent);
    normal = pbr_functions::apply_normal_mapping(flags, TBN, double_sided, is_logically_front, Nt);
}
    out.normal = vec4<f32>(normal * 0.5 + vec3<f32>(0.5), 1.0);
}

@if(MOTION_VECTOR_PREPASS)
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(in.world_position, in.previous_world_position);
    return out;
}

@if(PREPASS_PIPELINE && !PREPASS_FRAGMENT)
@fragment
fn fragment(in: VertexOutput) {
@if(VISIBILITY_RANGE_DITHER)
    lod_fade(in.position, in.visibility_range_dither);
    pbr_prepass_functions::prepass_sample_color_and_alpha_discard(in);
}

// Main pass fragment -------------------------------------------------------------------------------------------

@if(!PREPASS_PIPELINE)
@fragment
fn fragment(vertex_output: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var in = vertex_output;
@if(VISIBILITY_RANGE_DITHER)
    lod_fade(in.position, in.visibility_range_dither);

@if(ALEPH_INTERIOR && VERTEX_COLORS)
    let pane = in.color;
@if(ALEPH_INTERIOR && VERTEX_COLORS)
    in.color = vec4<f32>(1.0);

@if(ALEPH_TRIPLANAR && VERTEX_UVS_A)
    var pbr_input = triplanar(in, is_front);
@else
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    pbr_input.material.base_color = alpha_discard(pbr_input.material.flags, pbr_input.material.alpha_cutoff, pbr_input.material.base_color);

@if(ALEPH_INTERIOR && VERTEX_COLORS && VERTEX_UVS_B)
    glass(&pbr_input, in, pane);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}

// Triplanar ----------------------------------------------------------------------------------------------------

// Three projections of the material along the world axes, blended by the normal. Tangents follow each projection
// so normal maps keep their sense.
@if(!PREPASS_PIPELINE && ALEPH_TRIPLANAR && VERTEX_UVS_A)
fn triplanar(in: VertexOutput, is_front: bool) -> pbr_types::PbrInput {
    let n = normalize(in.world_normal);
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= w.x + w.y + w.z;
    let p = in.world_position.xyz / params(in).size;
    let s = select(vec3<f32>(-1.0), vec3<f32>(1.0), n >= vec3<f32>(0.0));
    var ix = in;
    var iy = in;
    var iz = in;
    ix.uv = vec2<f32>(-p.z * s.x, -p.y);
    iy.uv = vec2<f32>(p.x, p.z * s.y);
    iz.uv = vec2<f32>(p.x * s.z, -p.y);
@if(VERTEX_TANGENTS) {
    ix.world_tangent = vec4<f32>(0.0, 0.0, -s.x, 1.0);
    iy.world_tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    iz.world_tangent = vec4<f32>(s.z, 0.0, 0.0, 1.0);
}
    var a = pbr_input_from_standard_material(ix, is_front);
    let b = pbr_input_from_standard_material(iy, is_front);
    let c = pbr_input_from_standard_material(iz, is_front);
    a.material.base_color = a.material.base_color * w.x + b.material.base_color * w.y + c.material.base_color * w.z;
    a.material.perceptual_roughness = a.material.perceptual_roughness * w.x + b.material.perceptual_roughness * w.y
        + c.material.perceptual_roughness * w.z;
    a.material.metallic = a.material.metallic * w.x + b.material.metallic * w.y + c.material.metallic * w.z;
    a.diffuse_occlusion = a.diffuse_occlusion * w.x + b.diffuse_occlusion * w.y + c.diffuse_occlusion * w.z;
    a.N = normalize(a.N * w.x + b.N * w.y + c.N * w.z);
    return a;
}

// Interior-mapped glass ----------------------------------------------------------------------------------------

const ROOM_DEPTH: f32 = 5.0;
// Daylight on surfaces just inside a window on a clear day, falling off into the room (lux).
const DAYLIGHT: f32 = 8000.0;
// Daylight on curtains and blinds right behind the glass (lux).
const CURTAIN_LIGHT: f32 = 6000.0;
// Lit rooms: ceiling lamp luminance (nits) and the warm light they add (lux).
const LAMP: f32 = 3000.0;
const LAMP_LIGHT: f32 = 400.0;
// Share of rooms with the lights on.
const LIT_ROOMS: f32 = 0.18;

fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// Turns glass pixels (smooth, non-metal) into a mirror-like layer over a fake room. `pane` and `in.uv_b` carry the
// room layout from `material::add_panes`.
@if(!PREPASS_PIPELINE && ALEPH_INTERIOR && VERTEX_COLORS && VERTEX_UVS_B)
fn glass(pbr: ptr<function, pbr_types::PbrInput>, in: VertexOutput, pane: vec4<f32>) {
    if (*pbr).material.perceptual_roughness > 0.2 || (*pbr).material.metallic > 0.5 {
        return; // Frames and mullions painted into curtain-wall and door textures.
    }
    let n = normalize(in.world_normal);
    let NdotV = max(dot(n, (*pbr).V), 0.0);
    let inside = room(in, pane, n, -(*pbr).V);
    let fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
    (*pbr).material.base_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    (*pbr).material.metallic = 0.0;
    (*pbr).material.perceptual_roughness = 0.06;
    (*pbr).material.reflectance = vec3<f32>(0.7);
    (*pbr).N = n;
    // Slightly green float glass, transmitting about 80 %.
    (*pbr).material.emissive = vec4<f32>(inside * vec3<f32>(0.78, 0.82, 0.8) * (1.0 - fresnel), 1.0);
}

// Radiance of the room seen along `ray` through the glass at this fragment.
@if(!PREPASS_PIPELINE && ALEPH_INTERIOR && VERTEX_COLORS && VERTEX_UVS_B)
fn room(in: VertexOutput, pane: vec4<f32>, n: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    if abs(n.y) > 0.5 {
        return vec3<f32>(0.0); // Skylights: no room model.
    }
    let along = normalize(vec3<f32>(n.z, 0.0, -n.x));
    // Room frame: x along the facade, y up from the floor, z into the building.
    var d = vec3<f32>(dot(ray, along), ray.y, -dot(ray, n));
    if d.z < 1e-3 {
        return vec3<f32>(0.0);
    }
    d = select(d, sign(d) * 1e-4, abs(d) < vec3<f32>(1e-4));
    var x: f32;
    var y: f32;
    var width: f32;
    var height: f32;
    var top: f32;
    var half: f32;
    if pane.a > 0.5 {
        // One pane: rooms centred on it, the floor below its sill, the ceiling above its head.
        width = pane.g;
        let cell = floor(in.uv_b.x / width + 0.5);
        x = in.uv_b.x - cell * width;
        y = in.uv_b.y;
        top = pane.r;
        half = pane.b;
        height = max(3.2, top + 0.35);
    } else {
        // Curtain wall: storeys and bays on a grid of absolute facade coordinates.
        width = 3.0;
        height = params(in).storey;
        x = in.uv_b.x - (floor(in.uv_b.x / width) + 0.5) * width;
        y = in.uv_b.y - floor(in.uv_b.y / height) * height;
        top = height - 0.1;
        half = width * 0.5;
    }
    // A point fixed per room, rounded so every fragment of the room agrees on its hash.
    let id = round((in.world_position.xyz - along * x - vec3<f32>(0.0, y, 0.0)) * 4.0) * 0.25;
    let r = hash3(id + 0.37);
    let r2 = hash3(r * 91.7 + 3.1);
    let rc = hash3(r2 * 57.3 + 7.9);

    let o = vec3<f32>(x, y, 0.0);
    let tx = (select(-0.5 * width, 0.5 * width, d.x > 0.0) - x) / d.x;
    let ty = (select(0.0, height, d.y > 0.0) - y) / d.y;
    let tz = ROOM_DEPTH / d.z;
    let t = min(tx, min(ty, tz));
    let h = o + d * t;

    let lit = r.z < LIT_ROOMS;
    // White to cream walls, wood, grey, or pale floors.
    let wall = mix(vec3<f32>(0.80, 0.78, 0.74), vec3<f32>(0.74, 0.66, 0.52), r.x) * mix(0.7, 1.0, r2.x);
    let floor_color = select(select(vec3<f32>(0.34, 0.22, 0.12), vec3<f32>(0.28, 0.28, 0.29), r.y > 0.45),
                             vec3<f32>(0.52, 0.47, 0.40), r.y > 0.8);
    var light = DAYLIGHT * (0.15 + 0.85 * exp(-h.z / 1.6));
    var albedo: vec3<f32>;
    var glow = vec3<f32>(0.0);
    if t == tz {
        albedo = wall * 0.9;
        // A cupboard, shelf, or picture against the back wall.
        if abs(h.x - (r2.y - 0.5) * width * 0.5) < 0.4 + 0.6 * r2.z && h.y < 0.7 + 1.4 * r2.z {
            albedo = vec3<f32>(0.10, 0.08, 0.07) + 0.25 * r2;
        }
    } else if t == ty {
        if d.y > 0.0 {
            albedo = vec3<f32>(0.82);
            light *= 0.5;
            if lit && abs(h.x) < 0.5 && abs(h.z - 0.5 * ROOM_DEPTH) < 0.6 {
                glow = vec3<f32>(LAMP);
            }
        } else {
            albedo = floor_color;
        }
    } else {
        albedo = wall * 0.85;
    }
    var color = albedo * light / PI + glow;
    if lit {
        color += albedo * LAMP_LIGHT * vec3<f32>(1.0, 0.82, 0.6) / PI;
    }

    // Furniture: a band of dark shapes at desk height partway into the room.
    let tf = ROOM_DEPTH * (0.25 + 0.3 * r2.y) / d.z;
    if tf < t {
        let q = o + d * tf;
        if q.y < 0.75 + 0.5 * r2.x && abs(q.x - (r.y - 0.5) * width * 0.6) < 0.4 + 0.6 * r.x {
            color = vec3<f32>(0.10, 0.09, 0.08) * mix(1.0, 2.5, r2.z) * (DAYLIGHT * 0.6 + select(0.0, LAMP_LIGHT, lit)) / PI;
        }
    }

    // Curtains and blinds 10 cm behind the glass.
    let q = o + d * (0.1 / d.z);
    let cloth = mix(vec3<f32>(0.78, 0.76, 0.72), vec3<f32>(0.62, 0.55, 0.45), rc.z) * CURTAIN_LIGHT / PI;
    if rc.x < 0.2 {
        // Side curtains, drawn partly open.
        if abs(q.x) > half * mix(0.35, 0.85, rc.y) && q.y < top + 0.1 {
            color = cloth * (0.8 + 0.2 * sin(q.x * 40.0));
        }
    } else if rc.x < 0.32 {
        // A roller blind down from the head of the window.
        if q.y > top - mix(0.2, 1.4, rc.y) {
            color = cloth * (0.92 + 0.08 * sin(q.y * 60.0));
        }
    } else if rc.x < 0.4 {
        // Sheer net curtain over the whole window.
        color = mix(color, cloth * 0.8, 0.65);
    }
    return color;
}
