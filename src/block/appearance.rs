//! Block appearance seam: a [`Visual`] becomes one texture-array layer.
//!
//! Core fallback is [`FlatAppearance`] (flat `rgb` + `alpha`). The world's
//! texture cache asks [`Mods::appearance`](crate::mods::Mods::appearance) —
//! first enabled appearance mod, else this fallback. A `revision` change
//! rebuilds every layer; otherwise the cache is append-only per descriptor.

use material::Visual;
use voxel_engine::{MaterialDesc, MATERIAL_FLAG_PROCEDURAL};

use crate::block::registry::BlockRegistry;

/// Edge length of a CPU appearance layer, in texels.
pub const TEXTURE_SIZE: u32 = 16;

/// RGBA8 bytes in one [`TEXTURE_SIZE`] layer.
pub const LAYER_BYTES: usize = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;

/// Turns a render descriptor into pixels (and, optionally, GPU material
/// descriptors). Implementors live in mods; core only ships [`FlatAppearance`].
pub trait BlockAppearance {
    fn layer(&self, vis: &Visual, out: &mut [u8; 16 * 16 * 4]);
    fn revision(&self) -> u32;
    fn wants_gpu_descriptors(&self) -> bool;
}

/// Core fallback: every texel is `vis.rgb` with `vis.alpha`.
pub struct FlatAppearance;

/// Process-lifetime fallback when no appearance mod is enabled.
pub static FLAT: FlatAppearance = FlatAppearance;

impl BlockAppearance for FlatAppearance {
    fn layer(&self, vis: &Visual, out: &mut [u8; LAYER_BYTES]) {
        for texel in out.chunks_exact_mut(4) {
            texel[0] = vis.rgb[0];
            texel[1] = vis.rgb[1];
            texel[2] = vis.rgb[2];
            texel[3] = vis.alpha;
        }
    }

    fn revision(&self) -> u32 {
        0
    }

    fn wants_gpu_descriptors(&self) -> bool {
        false
    }
}

/// Visual bytes mapped 1:1 onto an engine descriptor with the procedural flag.
pub fn procedural_material_desc(vis: &Visual) -> MaterialDesc {
    MaterialDesc {
        rgb: vis.rgb,
        rgb2: vis.rgb2,
        frequency: vis.frequency,
        roughness: vis.roughness,
        alpha: vis.alpha,
        glow: vis.glow,
        flags: MATERIAL_FLAG_PROCEDURAL,
        _pad: [0; 4],
    }
}

/// Fill one descriptor layer. Layer 0 is all white — the engine's immediate
/// cubes and wires always sample it.
pub fn fill_descriptor_layer(
    appearance: &dyn BlockAppearance,
    registry: &BlockRegistry,
    layer: u16,
    out: &mut [u8; LAYER_BYTES],
) {
    if layer == 0 {
        out.fill(255);
        return;
    }
    appearance.layer(&registry.descriptor(layer), out);
}

/// 1×1 RGBA8 placeholder so a GPU-descriptor layer still has an array index.
pub fn placeholder_layer(vis: &Visual, layer: u16) -> Vec<u8> {
    if layer == 0 {
        vec![255, 255, 255, 255]
    } else {
        vec![vis.rgb[0], vis.rgb[1], vis.rgb[2], vis.alpha]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mods::Mods;
    use material::{visual, Configuration, Element, Law};

    fn vis(e: [u8; 4]) -> Visual {
        visual(&Law::v0(), &Configuration::single(Element::new(e)))
    }

    #[test]
    fn flat_fills_rgb_with_alpha() {
        let vis = Visual {
            rgb: [10, 20, 30],
            rgb2: [200, 0, 0],
            frequency: 255,
            roughness: 128,
            alpha: 80,
            glow: 40,
        };
        let mut out = [0u8; LAYER_BYTES];
        FlatAppearance.layer(&vis, &mut out);
        for texel in out.chunks_exact(4) {
            assert_eq!(texel, &[10, 20, 30, 80]);
        }
        assert_eq!(FlatAppearance.revision(), 0);
        assert!(!FlatAppearance.wants_gpu_descriptors());
    }

    #[test]
    fn empty_mods_use_flat_appearance() {
        let mods = Mods::empty();
        let a = mods.appearance();
        assert_eq!(a.revision(), 0);
        assert!(!a.wants_gpu_descriptors());
        let vis = Visual {
            rgb: [1, 2, 3],
            rgb2: [9, 8, 7],
            frequency: 1,
            roughness: 2,
            alpha: 4,
            glow: 5,
        };
        let mut out = [0u8; LAYER_BYTES];
        a.layer(&vis, &mut out);
        assert_eq!(&out[..4], &[1, 2, 3, 4]);
        assert!(out.chunks_exact(4).all(|t| t == [1, 2, 3, 4]));
    }

    #[test]
    fn layer_zero_is_all_white_regardless_of_visual() {
        let reg = BlockRegistry::with_builtins();
        let mut out = [0u8; LAYER_BYTES];
        fill_descriptor_layer(&FLAT, &reg, 0, &mut out);
        assert!(out.iter().all(|&b| b == 255));
        let air = vis_from_void();
        assert_ne!(air.alpha, 255, "air's visual is not opaque white");
    }

    fn vis_from_void() -> Visual {
        visual(&Law::v0(), &Configuration::void())
    }

    #[test]
    fn placeholder_layer_is_one_texel() {
        let v = Visual {
            rgb: [4, 5, 6],
            rgb2: [0, 0, 0],
            frequency: 0,
            roughness: 0,
            alpha: 7,
            glow: 0,
        };
        assert_eq!(placeholder_layer(&v, 0), vec![255, 255, 255, 255]);
        assert_eq!(placeholder_layer(&v, 1), vec![4, 5, 6, 7]);
    }

    #[test]
    fn procedural_material_desc_is_visual_bytes_plus_flag() {
        let v = vis([40, 80, 120, 160]);
        let d = procedural_material_desc(&v);
        assert_eq!(d.rgb, v.rgb);
        assert_eq!(d.rgb2, v.rgb2);
        assert_eq!(d.frequency, v.frequency);
        assert_eq!(d.roughness, v.roughness);
        assert_eq!(d.alpha, v.alpha);
        assert_eq!(d.glow, v.glow);
        assert_eq!(d.flags, MATERIAL_FLAG_PROCEDURAL);
        assert_eq!(d._pad, [0; 4]);
        assert_eq!(std::mem::size_of_val(&d), 16);
    }
}
