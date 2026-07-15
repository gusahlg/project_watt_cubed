//! Centralized render lane configuration, replacing env vars with explicit types.
//! Lanes are split: `lod2` and `occlusion` control world-derived geometry/state;
//! `blocklight` and others map to engine `RenderFlags`; `clouds` and `weather`
//! are per-frame look toggles. The game's single apply path transitions all of
//! them live and tears down/rebuilds derived world state when required.

use std::ops::RangeInclusive;

use voxel_engine::RenderFlags;

/// Number of active far-field LOD rings. Eight rings with the coarsest-detail
/// guard below is the largest hierarchy the current section key can use.
pub const LOD_LEVELS_RANGE: RangeInclusive<u8> = 1..=8;
/// Finest section detail (`2^detail` metres per cell).
pub const LOD_DETAIL_RANGE: RangeInclusive<u8> = 2..=6;
/// The section hierarchy is deliberately capped here: coordinates and shifts
/// throughout the quadtree assume a small, consecutive base-2 ladder.
const LOD_COARSEST_DETAIL: u8 = 9;

/// Render lane toggles. `Copy` to thread freely.
#[derive(Clone, Copy)]
pub struct RenderConfig {
    /// Occlusion culling.
    pub occlusion: bool,
    /// Column-section far field; `false` is near-only mode.
    pub lod2: bool,
    /// Number of consecutive base-2 section levels in the far-field ladder.
    pub lod_levels: u8,
    /// Finest far-field detail (`2^lod_detail` metres per cell).
    pub lod_detail: u8,
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
            lod_levels: 7,
            lod_detail: 2,
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
        Self {
            blocklight: true,
            exposure: true,
            ..Self::default()
        }
    }

    /// Clamp the requested LOD ladder to the supported ranges and shorten it
    /// when necessary so its coarsest level never exceeds detail 9.
    pub fn normalized_lod(self) -> (u8, u8) {
        let detail = self
            .lod_detail
            .clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
        let levels = self
            .lod_levels
            .clamp(*LOD_LEVELS_RANGE.start(), *LOD_LEVELS_RANGE.end());
        let max_levels = LOD_COARSEST_DETAIL - detail + 1;
        (levels.min(max_levels), detail)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_defaults_and_normalization_preserve_detail_cap() {
        assert_eq!(RenderConfig::default().normalized_lod(), (7, 2));
        assert_eq!(
            RenderConfig {
                lod_levels: u8::MAX,
                lod_detail: u8::MAX,
                ..RenderConfig::default()
            }
            .normalized_lod(),
            (4, 6),
            "detail 6 can expose only levels 6 through 9"
        );
        assert_eq!(
            RenderConfig {
                lod_levels: 0,
                lod_detail: 0,
                ..RenderConfig::default()
            }
            .normalized_lod(),
            (1, 2)
        );
    }
}
