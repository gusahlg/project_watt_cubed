//! GPU material-descriptor appearance: the engine paints the two-colour
//! pattern from a 16-byte [`MaterialDesc`] per layer. Off by default and
//! not in Essentials — enable it (and disable `procedural_textures`) to
//! replace the CPU generator.

use material::Visual;

use crate::block::appearance::{BlockAppearance, FlatAppearance, LAYER_BYTES};
use crate::mods::Mod;
#[cfg(test)]
use crate::block::appearance::procedural_material_desc;
#[cfg(test)]
use voxel_engine::MaterialDesc;

/// Engine `Engine::set_material_descs` / `append_material_descs` landed with
/// the `wt/material` branch (16-byte [`MaterialDesc`], capacity 16384).
pub const ENGINE_HAS_MATERIAL_DESCS: bool = true;

/// GPU-side procedural appearance. The texture cache uploads a 1×1
/// placeholder per descriptor; the shader reads this table instead.
pub struct GpuMaterialsMod;

impl GpuMaterialsMod {
    pub fn new() -> Self {
        Self
    }

    /// Visual bytes mapped 1:1 with [`crate::block::appearance::procedural_material_desc`].
    #[cfg(test)]
    pub fn desc(vis: &Visual) -> MaterialDesc {
        procedural_material_desc(vis)
    }
}

impl Default for GpuMaterialsMod {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockAppearance for GpuMaterialsMod {
    fn layer(&self, vis: &Visual, out: &mut [u8; LAYER_BYTES]) {
        // The array still needs a layer index; the shader ignores the texels.
        FlatAppearance.layer(vis, out);
    }

    fn revision(&self) -> u32 {
        0
    }

    fn wants_gpu_descriptors(&self) -> bool {
        ENGINE_HAS_MATERIAL_DESCS
    }
}

impl Mod for GpuMaterialsMod {
    fn name(&self) -> &str {
        "GPU materials"
    }

    fn id(&self) -> &'static str {
        "gpu_materials"
    }

    fn description(&self) -> &str {
        "Upload per-layer MaterialDesc and a 1×1 placeholder; the GPU paints the pattern."
    }

    fn appearance(&self) -> Option<&dyn BlockAppearance> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mods::Mods;
    use voxel_engine::MATERIAL_FLAG_PROCEDURAL;

    #[test]
    fn engine_exposes_material_descs() {
        assert!(
            ENGINE_HAS_MATERIAL_DESCS,
            "engine API is present; this const must stay true"
        );
    }

    #[test]
    fn desc_is_visual_plus_procedural_flag() {
        let vis = Visual {
            rgb: [1, 2, 3],
            rgb2: [4, 5, 6],
            frequency: 7,
            roughness: 8,
            alpha: 9,
            glow: 10,
        };
        let d = GpuMaterialsMod::desc(&vis);
        assert_eq!(d.rgb, [1, 2, 3]);
        assert_eq!(d.rgb2, [4, 5, 6]);
        assert_eq!(d.frequency, 7);
        assert_eq!(d.roughness, 8);
        assert_eq!(d.alpha, 9);
        assert_eq!(d.glow, 10);
        assert_eq!(d.flags, MATERIAL_FLAG_PROCEDURAL);
        assert_eq!(d._pad, [0; 4]);
        assert_eq!(std::mem::size_of::<MaterialDesc>(), 16);
    }

    #[test]
    fn default_off_and_not_essentials() {
        let mods = Mods::with_defaults();
        let i = (0..mods.len())
            .find(|&i| mods.id(i) == "gpu_materials")
            .expect("installed");
        assert!(!mods.is_enabled(i));
        assert_eq!(mods.group(i), "");
        assert!(!mods.appearance().wants_gpu_descriptors());
    }

    #[test]
    fn wins_the_seam_when_procedural_is_off() {
        let mut mods = Mods::with_defaults();
        mods.set_enabled("procedural_textures", false);
        mods.set_enabled("gpu_materials", true);
        let a = mods.appearance();
        assert!(a.wants_gpu_descriptors());
        let vis = Visual {
            rgb: [9, 8, 7],
            rgb2: [0, 0, 0],
            frequency: 0,
            roughness: 0,
            alpha: 255,
            glow: 0,
        };
        let mut out = [0u8; LAYER_BYTES];
        a.layer(&vis, &mut out);
        assert_eq!(&out[..4], &[9, 8, 7, 255]);
    }
}
