//! Centralized render lane configuration, replacing env vars with explicit types.
//! Lanes are split: `lod2` and `occlusion` control world geometry generation;
//! `blocklight` and others map to engine `RenderFlags`; `clouds` and `weather`
//! are per-frame look toggles. All read at construction except the look lanes.

use voxel_engine::RenderFlags;

/// Render lane toggles. `Copy` to thread freely.
#[derive(Clone, Copy)]
pub struct RenderConfig {
    /// Occlusion culling.
    pub occlusion: bool,
    /// Column-section far field; `false` is near-only mode.
    pub lod2: bool,
    /// Torch/candle block light.
    pub blocklight: bool,
    /// Auto-exposure metering.
    pub exposure: bool,
    /// HDR bloom effect.
    pub bloom: bool,
    /// Volumetric clouds (per-frame lookup, no engine gate).
    pub clouds: bool,
    /// Weather coverage (per-frame lookup, no engine gate).
    pub weather: bool,
    /// Night starfield (`RenderFlags::stars`): off skips the sky pass's
    /// per-pixel hash-grid star evaluation.
    pub stars: bool,
    /// Day/night cycle (per-frame lookup, no engine gate): off freezes the
    /// sky clock at the current time of day.
    pub day_night: bool,
    /// Screen-space sun god rays.
    pub godrays: bool,
    /// Temporal AA with camera jitter (always coupled).
    pub taa: bool,
    pub fog: bool,
    /// Ambient lighting floor (off = black caves).
    pub ambient: bool,
    /// Sun and sky light (also affects sky halo).
    pub sunlight: bool,
    /// Cascade shadow map (off = fully lit).
    pub shadows: bool,
    pub sky: bool,
    /// Variable-rate shading (`RenderFlags::vrs`): depth-classified coarse
    /// fragment shading on distant/flat regions. Off shades full-rate everywhere.
    pub vrs: bool,
    /// Water surface animation (`RenderFlags::water_anim`). Off freezes the
    /// phase — water renders, but still.
    pub water_anim: bool,
    pub vignette: bool,
}

impl Default for RenderConfig {
    /// Shipped defaults; exposure off by design.
    fn default() -> Self {
        Self {
            occlusion: true,
            lod2: true,
            blocklight: false,
            exposure: false,
            bloom: true,
            godrays: true,
            clouds: true,
            weather: true,
            stars: true,
            day_night: true,
            // Keep these identical to the previous shipped look.
            taa: false,
            fog: false,
            ambient: false,
            sunlight: true,
            shadows: false,
            sky: true,
            vrs: true,
            water_anim: true,
            vignette: false,
        }
    }
}

impl RenderConfig {
    /// Golden harness config: defaults with blocklight and exposure for proper lighting.
    pub fn golden() -> Self {
        Self { blocklight: true, exposure: true, ..Self::default() }
    }

    /// Map to engine RenderFlags (total mapping, no fallthrough).
    pub fn engine_flags(self) -> RenderFlags {
        RenderFlags {
            blocklight: self.blocklight,
            exposure: self.exposure,
            bloom: self.bloom,
            godrays: self.godrays,
            taa: self.taa,
            fog: self.fog,
            ambient: self.ambient,
            sunlight: self.sunlight,
            shadows: self.shadows,
            sky: self.sky,
            vrs: self.vrs,
            water_anim: self.water_anim,
            vignette: self.vignette,
            stars: self.stars,
        }
    }
}
