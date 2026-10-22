//! Mipmaps for PNG and JPEG textures, which Bevy loads with a single level. Without them, distant tiling textures
//! shimmer and sampling 2k textures on far geometry is slow. KTX2 files with their own mips are left alone.

use bevy::image::{ImageFilterMode, ImageSampler};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};

/// Largest texture side to upload, so the world fits next to other GPU users; 0 means no limit.
#[derive(Resource)]
pub(crate) struct MaxTexture(pub u32);

pub(crate) fn generate(mut events: MessageReader<AssetEvent<Image>>, mut images: ResMut<Assets<Image>>,
                       limit: Res<MaxTexture>) {
    for event in events.read() {
        let AssetEvent::Added { id } = event else { continue };
        // get_mut marks the image modified, so only take it mutably when there is work to do.
        let Some(image) = images.get(*id) else { continue };
        let d = &image.texture_descriptor;
        let srgb = match d.format {
            TextureFormat::Rgba8UnormSrgb => true,
            TextureFormat::Rgba8Unorm => false,
            _ => continue,
        };
        if d.mip_level_count > 1 || d.size.depth_or_array_layers > 1 || d.usage.contains(TextureUsages::RENDER_ATTACHMENT)
            || image.data.is_none() || d.size.width.max(d.size.height) < 4
        {
            continue;
        }
        let Some(mut image) = images.get_mut(*id) else { continue };
        let (mut w, mut h) = (image.texture_descriptor.size.width as usize, image.texture_descriptor.size.height as usize);
        let mut data = image.data.take().unwrap_or_default();
        // Average in linear light for colour textures; normals and ORM are already linear.
        let to_linear: Vec<f32> = (0..256).map(|v| if srgb { srgb_to_linear(v as f32 / 255.0) } else { v as f32 / 255.0 }).collect();
        let mut level = data.clone();
        let mut levels = 1;
        // Levels larger than the limit are built (each comes from the one above) but not kept.
        let mut kept = (w, h);
        let too_big = |w: usize, h: usize| limit.0 > 0 && w.max(h) > limit.0 as usize;
        if too_big(w, h) {
            data.clear();
            levels = 0;
        }
        while w > 1 || h > 1 {
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = vec![0u8; nw * nh * 4];
            for y in 0..nh {
                for x in 0..nw {
                    for c in 0..4 {
                        let mut sum = 0.0;
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                            let v = level[(sy * w + sx) * 4 + c];
                            sum += if c == 3 { v as f32 / 255.0 } else { to_linear[v as usize] };
                        }
                        let v = sum / 4.0;
                        let v = if c != 3 && srgb { linear_to_srgb(v) } else { v };
                        next[(y * nw + x) * 4 + c] = (v * 255.0 + 0.5) as u8;
                    }
                }
            }
            if !too_big(nw, nh) {
                if levels == 0 {
                    kept = (nw, nh);
                }
                data.extend_from_slice(&next);
                levels += 1;
            }
            (level, w, h) = (next, nw, nh);
        }
        image.data = Some(data);
        image.texture_descriptor.mip_level_count = levels;
        image.texture_descriptor.size.width = kept.0 as u32;
        image.texture_descriptor.size.height = kept.1 as u32;
        // Anisotropic filtering keeps ground textures sharp at grazing angles; wgpu allows it only with all-linear filters.
        if let ImageSampler::Descriptor(s) = &mut image.sampler
            && [s.mag_filter, s.min_filter, s.mipmap_filter].iter().all(|f| *f == ImageFilterMode::Linear)
        {
            s.anisotropy_clamp = 16;
        }
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}
