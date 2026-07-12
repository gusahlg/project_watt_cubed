//! The typed set of render lanes a world is built and drawn with — the ONE
//! source that replaced the former ambient `WATT_*`/`VOXEL_*` env vars (and the
//! engine's process-global `OnceLock` flag cache). A world/capture now names its
//! lanes explicitly: the game builds [`RenderConfig::default`] from its shipped
//! defaults, and the golden harness builds [`RenderConfig::golden`] per stage, so
//! "captured with tiles on" is a compile-time property of the shot rather than an
//! env var an operator must remember to set.
//!
//! Two lanes are app-side world geometry ([`World::with_config`](crate::world::World::with_config)):
//! `lod2` (the column-section far field), `occlusion`. `blocklight` is an engine lighting lane, mapped
//! to [`voxel_engine::RenderFlags`] via [`engine_flags`](RenderConfig::engine_flags)
//! and handed to the engine through `Config::flags`. `clouds`/`weather` are
//! app-side per-frame look lanes read by [`compose`](crate::frame_snapshot::compose)
//! (they replaced `WATT_CLOUDS`/`WATT_WEATHER`); all lanes here are read at
//! construction only, but the two look lanes are per-frame-safe if live toggling
//! is ever wanted.

use voxel_engine::RenderFlags;

/// Which render lanes are enabled. `Copy` so it threads freely with no ceremony.
#[derive(Clone, Copy)]
pub struct RenderConfig {
    /// Occlusion cull gate (was `VOXEL_OCCLUSION`).
    pub occlusion: bool,
    /// The column-section quadtree far field — the sole far renderer. `false`
    /// enables near-only mode where the section lane stays dormant and free.
    pub lod2: bool,
    /// Torch/candle block light — an engine lane (was `WATT_BLOCKLIGHT`).
    pub blocklight: bool,
    /// Auto-exposure metering — an engine lane (`RenderFlags::exposure`). Off
    /// pins the exposure multiplier at 1.0; on runs the metering + tonemap curve.
    pub exposure: bool,
    /// HDR bloom — an engine lane (`RenderFlags::bloom`). On thresholds the HDR
    /// offscreen, downsamples it, and adds the spill in the tonemap pass; off skips
    /// the compute and clears the bloom target so the composite is a no-op.
    pub bloom: bool,
    /// Volumetric cloud slab in `sky.frag` (was `WATT_CLOUDS`). Off pushes the
    /// slab's camera height (`anim.w`) to a sentinel so the march early-outs at
    /// zero cost — an app-side look lane read per frame by
    /// [`compose`](crate::frame_snapshot::compose); no engine gate.
    pub clouds: bool,
    /// Weather coverage feeding the direct-light mute/desaturate (was
    /// `WATT_WEATHER`). Off reads coverage as zero for clear-sky direct light —
    /// an app-side look lane read per frame by `compose`; no engine gate. (Fog has
    /// no app lane: it rides the engine `RenderFlags::fog` gate alone.)
    pub weather: bool,
    /// Screen-space godrays — an engine lane (`RenderFlags::godrays`). On marches
    /// the tonemap pass toward the sun's screen position for a sunward veil; off
    /// pushes a strength-0 gate so the march is skipped.
    pub godrays: bool,
    /// Temporal AA + camera jitter (`RenderFlags::taa`), always coupled.
    pub taa: bool,
    /// Horizon density fade (`RenderFlags::fog`).
    pub fog: bool,
    /// Omnidirectional ambient floor (`RenderFlags::ambient`); off means black caves.
    pub ambient: bool,
    /// Sun/skylight (`RenderFlags::sunlight`); off also kills the sky halo glow.
    pub sunlight: bool,
    /// Cascade shadow map (`RenderFlags::shadows`); off is fully lit.
    pub shadows: bool,
    /// Off shows the clear colour (`RenderFlags::sky`).
    pub sky: bool,
    /// Variable-rate shading (`RenderFlags::vrs`): depth-classified coarse
    /// fragment shading on distant/flat regions. Off shades full-rate everywhere.
    pub vrs: bool,
    /// Water surface animation (`RenderFlags::water_anim`). Off freezes the
    /// phase — water renders, but still.
    pub water_anim: bool,
}

impl Default for RenderConfig {
    /// Shipped defaults; exposure off to keep it a deliberate look-activation.
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
            // The former `RenderFlags::default()` values these lanes inherited when
            // `engine_flags` left them unmapped — kept identical so the shipped look
            // is unchanged.
            taa: false,
            fog: false,
            ambient: false,
            sunlight: true,
            shadows: false,
            sky: true,
            vrs: true,
            water_anim: true,
        }
    }
}

impl RenderConfig {
    /// The golden harness config: shipped defaults plus `blocklight` (needed for
    /// cave interiors) and `exposure` (forces metering+tonemap path rather than
    /// hardcoded 1.0). Clouds/weather inherit `default()` true since goldens were
    /// captured with those unset.
    pub fn golden() -> Self {
        Self { blocklight: true, exposure: true, ..Self::default() }
    }

    /// The engine feature flags this config implies — every engine lane is now a
    /// named field here, so this is a total mapping (no `..default()` fallthrough).
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
        }
    }
}
