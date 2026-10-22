//! scene.json, as the Python side writes it. Metres; x east, y up, z south (north is -Z).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct SceneFile {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub attribution: Vec<String>,
    pub chunk_size: f32,
    pub lod_distances: [f32; 3],
    #[serde(default)]
    pub bounds: Vec<[f32; 2]>,
    pub spawn: Spawn,
    pub sun: Sun,
    pub chunks: Vec<ChunkDef>,
    #[serde(default)]
    pub assets: HashMap<String, AssetDef>,
    #[serde(default)]
    pub horizon: Option<Horizon>,
}

impl SceneFile {
    pub fn read(folder: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(folder.join("scene.json")).map_err(|e| e.to_string())?;
        let scene: Self = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if scene.format != "aleph-game" || scene.version != 1 {
            return Err(format!("unsupported format {} version {}", scene.format, scene.version));
        }
        Ok(scene)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Spawn {
    /// Eye position.
    pub position: [f32; 3],
    /// Radians counter-clockwise seen from above; 0 looks north (-Z).
    #[serde(default)]
    pub yaw: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Sun {
    /// Degrees clockwise from north.
    pub azimuth: f32,
    /// Degrees above the horizon.
    pub elevation: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChunkDef {
    pub id: String,
    pub center: [f32; 3],
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// LOD0, LOD1, LOD2 glTFs; vertices relative to `center`.
    pub lods: Vec<String>,
    #[serde(default)]
    pub ground: Option<GroundDef>,
    #[serde(default)]
    pub instances: Vec<InstanceDef>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GroundDef {
    /// Raw little-endian f32, row-major, rows advancing along +z.
    pub file: String,
    /// [cols, rows]
    pub size: [usize; 2],
    pub spacing: f32,
    /// World (x, z) of sample (0, 0).
    pub offset: [f32; 2],
}

#[derive(Deserialize, Debug, Clone)]
pub struct InstanceDef {
    pub asset: String,
    pub position: [f32; 3],
    #[serde(default)]
    pub yaw: f32,
    #[serde(default = "one")]
    pub scale: f32,
    #[serde(default)]
    pub tint: Option<[f32; 3]>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AssetDef {
    pub lods: Vec<String>,
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub height: f32,
    #[serde(default)]
    pub radius: f32,
}

#[derive(Deserialize, Debug, Clone)]
/// capsule and cylinder use radius, half_height, and center; convex and trimesh use vertices (and indices).
pub struct ColliderDef {
    pub shape: String,
    #[serde(default)]
    pub radius: f32,
    /// Half the total height, caps included.
    #[serde(default)]
    pub half_height: f32,
    #[serde(default)]
    pub center: [f32; 3],
    /// Hull points (convex) or triangle vertices (trimesh), in the asset's frame.
    #[serde(default)]
    pub vertices: Vec<[f32; 3]>,
    /// Triangle indices (trimesh).
    #[serde(default)]
    pub indices: Vec<u32>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Horizon {
    pub file: String,
    #[serde(default)]
    pub from: f32,
}

fn one() -> f32 {
    1.0
}
