//! Centralized render lane configuration, replacing env vars with explicit types.
//! Lanes are split: `lod2` and `occlusion` control world geometry generation;
//! `blocklight` and others map to engine `RenderFlags`; `clouds` and `weather`
//! are per-frame look toggles. All read at construction except the look lanes.

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

/// Auto enables VRS only when the render extent (window × render scale) has
/// at least this many pixels. RTX 4060 @ 1920×1080 (2.07 Mpx) and RTX 3070 @
/// 3440×1440 (4.95 Mpx) still lose with VRS on; it pays at 4K-class extents
/// (3440×1440 at 200% render scale = 19.8 Mpx).
pub const VRS_AUTO_MIN_PIXELS: u64 = 8_000_000;

/// User choice for variable-rate shading. [`vrs_effective`] turns this into
/// the engine bool; Auto keys off the live render extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VrsChoice {
    Auto,
    On,
    Off,
}

impl VrsChoice {
    /// Persistence word (`auto` / `on` / `off`).
    pub fn code(self) -> &'static str {
        match self {
            VrsChoice::Auto => "auto",
            VrsChoice::On => "on",
            VrsChoice::Off => "off",
        }
    }

    /// Parse a persisted or console word. Legacy `true`/`false` keep their
    /// forced On/Off choice so existing `settings.cfg` files stay put.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(VrsChoice::Auto),
            "on" | "true" => Some(VrsChoice::On),
            "off" | "false" => Some(VrsChoice::Off),
            _ => None,
        }
    }

    /// Capitalized display name for the menu row and confirm line.
    pub fn label(self) -> &'static str {
        match self {
            VrsChoice::Auto => "Auto",
            VrsChoice::On => "On",
            VrsChoice::Off => "Off",
        }
    }
}

/// Engine VRS flag for a user choice at a render extent.
pub fn vrs_effective(choice: VrsChoice, render_w: u32, render_h: u32) -> bool {
    match choice {
        VrsChoice::On => true,
        VrsChoice::Off => false,
        VrsChoice::Auto => {
            (render_w as u64).saturating_mul(render_h as u64) >= VRS_AUTO_MIN_PIXELS
        }
    }
}

/// Fancy presentation groups owned by default-enabled visual mods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualGroup {
    Atmosphere,
    Post,
    Lighting,
}

impl VisualGroup {
    pub fn mod_name(self) -> &'static str {
        match self {
            Self::Atmosphere => "Atmosphere",
            Self::Post => "Post",
            Self::Lighting => "Lighting",
        }
    }
}

/// Which visual-mod group owns a settings/`/gfx` lane key, if any.
pub fn lane_group(key: &str) -> Option<VisualGroup> {
    match key {
        "clouds" | "weather" | "stars" | "day_night" | "fog" | "sky" | "water_anim" => {
            Some(VisualGroup::Atmosphere)
        }
        "bloom" | "godrays" | "taa" | "exposure" | "vignette" | "vrs" => Some(VisualGroup::Post),
        "shadows" | "ambient" | "blocklight" => Some(VisualGroup::Lighting),
        _ => None,
    }
}

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
    /// Variable-rate shading (`RenderFlags::vrs`): the resolved engine flag
    /// from [`vrs_effective`]. Off shades full-rate everywhere.
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
            shadows: true,
            sky: true,
            vrs: false,
            water_anim: true,
            vignette: false,
        }
    }
}

impl RenderConfig {
    /// Core look: sunlight on readable terrain, every fancy lane off.
    /// Visual groups are stripped via [`strip_group`] so the two cannot drift.
    pub fn core() -> Self {
        let mut cfg = Self {
            occlusion: false,
            lod2: false,
            lod_levels: 1,
            lod_detail: 6,
            ..Self::default()
        };
        cfg.strip_group(VisualGroup::Atmosphere);
        cfg.strip_group(VisualGroup::Post);
        cfg.strip_group(VisualGroup::Lighting);
        cfg
    }

    /// Golden harness config: defaults with blocklight and exposure for proper lighting.
    pub fn golden() -> Self {
        Self { blocklight: true, exposure: true, ..Self::default() }
    }

    /// Drop one visual group; used when that group's mod is disabled.
    pub fn strip_group(&mut self, group: VisualGroup) {
        match group {
            VisualGroup::Atmosphere => {
                self.clouds = false;
                self.weather = false;
                self.stars = false;
                self.day_night = false;
                self.fog = false;
                self.sky = false;
                self.water_anim = false;
            }
            VisualGroup::Post => {
                self.bloom = false;
                self.godrays = false;
                self.taa = false;
                self.exposure = false;
                self.vignette = false;
                self.vrs = false;
            }
            VisualGroup::Lighting => {
                self.shadows = false;
                self.ambient = false;
                self.blocklight = false;
            }
        }
    }

    /// Clamp the requested LOD ladder to the supported ranges and shorten it
    /// when necessary so its coarsest level never exceeds detail
    /// [`LOD_COARSEST_DETAIL`]. Returns `(levels, detail)`.
    pub fn normalized_lod(self) -> (u8, u8) {
        let detail = self.lod_detail.clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
        let levels = self.lod_levels.clamp(*LOD_LEVELS_RANGE.start(), *LOD_LEVELS_RANGE.end());
        (levels.min(max_lod_levels(detail)), detail)
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

/// The longest ladder a given finest detail supports before its coarsest
/// level would exceed [`LOD_COARSEST_DETAIL`]. Shared with the settings
/// stepper so the menu and the normalization can never disagree.
pub fn max_lod_levels(detail: u8) -> u8 {
    let detail = detail.clamp(*LOD_DETAIL_RANGE.start(), *LOD_DETAIL_RANGE.end());
    (LOD_COARSEST_DETAIL - detail + 1).min(*LOD_LEVELS_RANGE.end())
}

/// Percent of live free device-local bytes ([`DeviceCaps::available_device_bytes`])
/// reserved for render targets. The rest is for meshes, textures, and the swapchain.
pub const VRAM_AVAILABLE_SAFETY_FRACTION: u64 = 85;

/// Percent of [`DeviceCaps::device_local_memory_bytes`] reserved for render
/// targets when the live budget is unavailable. The rest is for the driver,
/// mesh arenas, and other processes.
pub const VRAM_SAFETY_FRACTION: u64 = 60;

/// Floor used when dropping `render_scale` to fit the VRAM budget.
pub const VRAM_SCALE_FLOOR: f32 = 0.5;
const VRAM_SCALE_STEP: f32 = 0.25;

/// Sample counts the engine will actually create, descending.
const MSAA_STEPS: &[u32] = &[8, 4, 2, 1];

/// Conservative colour bytes/pixel. Engine colour targets are
/// `R16G16B16A16_SFLOAT` (8 B/px in `voxel-engine/src/vk/targets.rs`); the
/// budget uses 16 B/px so image-memory rounding, padding, and uncounted
/// attachments stay inside ±20%.
const HDR_COLOR_BYTES: u64 = 16;
/// Engine depth is `D32_SFLOAT` (or a 4-byte packed fallback).
const DEPTH_BYTES: u64 = 4;
/// `FRAMES_IN_FLIGHT` in `voxel-engine/src/vk/buffers.rs`.
const FRAMES_IN_FLIGHT: u64 = 2;
/// `SHADOW_RESOLUTION` / `SHADOW_CASCADES` / `D32_SFLOAT` in targets.rs.
const SHADOW_RESOLUTION: u64 = 2048;
const SHADOW_CASCADES: u64 = 2;
/// `SKY_CLOUD_LUT_SIZE` (RGBA16F, budgeted at [`HDR_COLOR_BYTES`]).
const SKY_CLOUD_LUT: u64 = 256;
/// `BLOOM_MAX_MIPS` in targets.rs (half-res base + two more lods).
const BLOOM_MIPS: u32 = 3;

/// GPU facts probed once at startup (Vulkan heaps + framebuffer samples).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCaps {
    pub device_local_memory_bytes: Option<u64>,
    /// `heapBudget - heapUsage` on device-local heaps when `VK_EXT_memory_budget`
    /// is present; `None` falls back to the heap-size rule.
    pub available_device_bytes: Option<u64>,
    pub max_msaa: u32,
}

impl Default for DeviceCaps {
    fn default() -> Self {
        Self {
            device_local_memory_bytes: None,
            available_device_bytes: None,
            max_msaa: 8,
        }
    }
}

impl DeviceCaps {
    /// Live free × [`VRAM_AVAILABLE_SAFETY_FRACTION`], else heap ×
    /// [`VRAM_SAFETY_FRACTION`]. `None` skips the session VRAM guard.
    pub fn render_target_budget_bytes(self) -> Option<u64> {
        if let Some(available) = self.available_device_bytes {
            Some(available.saturating_mul(VRAM_AVAILABLE_SAFETY_FRACTION) / 100)
        } else {
            self.device_local_memory_bytes
                .map(|bytes| bytes.saturating_mul(VRAM_SAFETY_FRACTION) / 100)
        }
    }
}

/// Bytes the engine is expected to spend on swapchain-sized render targets
/// (plus the fixed shadow map and sky LUT when those lanes are on).
///
/// Internal extent `W×H = (width·scale)×(height·scale)`:
/// - 1× MSAA colour image at `msaa` samples (`HDR_COLOR_BYTES` each) when `msaa>1`
/// - `FRAMES_IN_FLIGHT` depth images at `msaa` samples (`DEPTH_BYTES`)
/// - `FRAMES_IN_FLIGHT` single-sample depth resolves when `msaa>1`
/// - `FRAMES_IN_FLIGHT` single-sample HDR offscreen/history colour images
/// - two extra HDR history images when `lanes.taa`
/// - per-slot bloom pyramid (`BLOOM_MIPS` of a half-res RGBA16F image) when `lanes.bloom`
/// - 2048² × 2 cascade D32 shadow map when `lanes.shadows`
/// - 256² HDR cloud LUT × slots when `lanes.clouds` or `lanes.sky`
pub fn render_target_bytes(
    width: u32,
    height: u32,
    render_scale: f32,
    msaa: u32,
    lanes: RenderConfig,
) -> u64 {
    let scale = render_scale.max(0.0);
    let w = ((width as f32 * scale) as u64).max(1);
    let h = ((height as f32 * scale) as u64).max(1);
    let pixels = w.saturating_mul(h);
    let samples = msaa.max(1) as u64;

    let mut bytes = 0u64;
    if samples > 1 {
        bytes = bytes.saturating_add(pixels.saturating_mul(HDR_COLOR_BYTES).saturating_mul(samples));
        bytes = bytes.saturating_add(
            pixels.saturating_mul(DEPTH_BYTES).saturating_mul(FRAMES_IN_FLIGHT),
        );
    }
    bytes = bytes.saturating_add(
        pixels
            .saturating_mul(DEPTH_BYTES)
            .saturating_mul(samples)
            .saturating_mul(FRAMES_IN_FLIGHT),
    );
    bytes = bytes.saturating_add(
        pixels
            .saturating_mul(HDR_COLOR_BYTES)
            .saturating_mul(FRAMES_IN_FLIGHT),
    );
    if lanes.taa {
        bytes = bytes.saturating_add(pixels.saturating_mul(HDR_COLOR_BYTES).saturating_mul(2));
    }
    if lanes.bloom {
        let mut mw = w.div_ceil(2).max(1);
        let mut mh = h.div_ceil(2).max(1);
        for _ in 0..BLOOM_MIPS {
            bytes = bytes.saturating_add(
                mw.saturating_mul(mh)
                    .saturating_mul(HDR_COLOR_BYTES)
                    .saturating_mul(FRAMES_IN_FLIGHT),
            );
            if mw == 1 && mh == 1 {
                break;
            }
            mw = mw.div_ceil(2).max(1);
            mh = mh.div_ceil(2).max(1);
        }
    }
    if lanes.shadows {
        bytes = bytes.saturating_add(
            SHADOW_RESOLUTION
                .saturating_mul(SHADOW_RESOLUTION)
                .saturating_mul(SHADOW_CASCADES)
                .saturating_mul(DEPTH_BYTES),
        );
    }
    if lanes.clouds || lanes.sky {
        bytes = bytes.saturating_add(
            SKY_CLOUD_LUT
                .saturating_mul(SKY_CLOUD_LUT)
                .saturating_mul(HDR_COLOR_BYTES)
                .saturating_mul(FRAMES_IN_FLIGHT),
        );
    }
    bytes
}

/// Session-only graphics after the VRAM guard (never written to disk).
#[derive(Clone, Debug, PartialEq)]
pub struct SessionGraphics {
    pub msaa: u32,
    pub render_scale: f32,
    pub notice: Option<String>,
}

/// Settings-screen line when the engine allocated less than the session request.
pub fn engine_applied_notice(msaa: u32, render_scale: f32) -> String {
    let pct = (render_scale * 100.0).round() as i32;
    format!("the renderer could only allocate {msaa}x MSAA at {pct}% scale this session")
}

/// True when the engine's applied MSAA / scale differ from the session request.
pub fn engine_applied_differs(requested: &SessionGraphics, msaa: u32, render_scale: f32) -> bool {
    msaa != requested.msaa || (render_scale - requested.render_scale).abs() > 1e-3
}

/// Drop MSAA to the next supported count, then `render_scale` in 0.25 steps
/// (not below [`VRAM_SCALE_FLOOR`]), until [`render_target_bytes`] fits the
/// budget from [`DeviceCaps::render_target_budget_bytes`]. If even the floor
/// does not fit, start at 1× MSAA and [`VRAM_SCALE_FLOOR`] and say so.
pub fn fit_render_targets(
    width: u32,
    height: u32,
    render_scale: f32,
    msaa: u32,
    lanes: RenderConfig,
    caps: DeviceCaps,
) -> SessionGraphics {
    let requested_scale = render_scale;
    let Some(budget) = caps.render_target_budget_bytes() else {
        return SessionGraphics {
            msaa: msaa.min(caps.max_msaa).max(1),
            render_scale: requested_scale,
            notice: None,
        };
    };
    let requested_msaa = snap_msaa(msaa, caps.max_msaa);
    let needed = render_target_bytes(width, height, requested_scale, requested_msaa, lanes);
    if needed <= budget {
        return SessionGraphics {
            msaa: requested_msaa,
            render_scale: requested_scale,
            notice: None,
        };
    }

    let mut chosen_msaa = requested_msaa;
    let mut chosen_scale = requested_scale;
    loop {
        let cost = render_target_bytes(width, height, chosen_scale, chosen_msaa, lanes);
        if cost <= budget {
            break;
        }
        if let Some(next) = MSAA_STEPS.iter().copied().find(|&n| n < chosen_msaa) {
            chosen_msaa = next;
            continue;
        }
        let snapped = (chosen_scale / VRAM_SCALE_STEP).round() * VRAM_SCALE_STEP;
        let next_scale = snapped - VRAM_SCALE_STEP;
        if next_scale + 1e-4 < VRAM_SCALE_FLOOR {
            chosen_msaa = 1;
            chosen_scale = VRAM_SCALE_FLOOR;
            break;
        }
        chosen_scale = next_scale.max(VRAM_SCALE_FLOOR);
    }

    let chosen_cost = render_target_bytes(width, height, chosen_scale, chosen_msaa, lanes);
    SessionGraphics {
        msaa: chosen_msaa,
        render_scale: chosen_scale,
        notice: Some(vram_notice(
            requested_msaa,
            requested_scale,
            needed,
            chosen_msaa,
            chosen_scale,
            chosen_cost,
            chosen_cost > budget,
            budget,
            caps,
        )),
    }
}

fn snap_msaa(requested: u32, max_msaa: u32) -> u32 {
    let cap = requested.min(max_msaa).max(1);
    MSAA_STEPS.iter().copied().find(|&n| n <= cap).unwrap_or(1)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000_000.0
}

fn vram_notice(
    req_msaa: u32,
    req_scale: f32,
    needed: u64,
    run_msaa: u32,
    run_scale: f32,
    run_cost: u64,
    floor_exceeded: bool,
    budget: u64,
    caps: DeviceCaps,
) -> String {
    let need_gb = gb(needed);
    let req_pct = (req_scale * 100.0).round() as i32;
    let run_pct = (run_scale * 100.0).round() as i32;
    let head = match (caps.available_device_bytes, caps.device_local_memory_bytes) {
        (Some(available), Some(heap)) => {
            let held = heap.saturating_sub(available);
            format!(
                "graphics: {req_msaa}x MSAA at {req_pct}% scale needs ~{need_gb:.1} GB; {avail:.1} GB of {heap:.1} GB is free (other processes hold {held:.1} GB)",
                avail = gb(available),
                heap = gb(heap),
                held = gb(held),
            )
        }
        _ => format!(
            "graphics: {req_msaa}x MSAA at {req_pct}% scale needs ~{need_gb:.1} GB of VRAM for render targets"
        ),
    };
    if floor_exceeded {
        return format!(
            "{head}; {run_msaa}x MSAA at {run_pct}% scale still needs ~{run:.1} GB; starting at that floor anyway",
            run = gb(run_cost),
        );
    }
    let running = if run_msaa != req_msaa && (run_scale - req_scale).abs() > 1e-3 {
        format!("{run_msaa}x MSAA, {run_pct}% scale")
    } else if run_msaa != req_msaa {
        format!("{run_msaa}x MSAA")
    } else {
        format!("{run_pct}% scale")
    };
    match caps.available_device_bytes {
        Some(_) => format!("{head}; running at {running} this session"),
        None => format!(
            "{head}; running at {running} this session (budget {budget:.1} GB)",
            budget = gb(budget),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vrs_effective_auto_turns_on_at_eight_million_pixels() {
        assert!(!vrs_effective(VrsChoice::Auto, 0, 0));
        assert!(!vrs_effective(VrsChoice::Auto, 1, 1));
        assert!(!vrs_effective(VrsChoice::Auto, 1920, 1080));
        assert!(!vrs_effective(VrsChoice::Auto, 3440, 1440));
        assert!(!vrs_effective(
            VrsChoice::Auto,
            (VRS_AUTO_MIN_PIXELS - 1) as u32,
            1
        ));
        assert!(vrs_effective(
            VrsChoice::Auto,
            VRS_AUTO_MIN_PIXELS as u32,
            1
        ));
        assert!(vrs_effective(VrsChoice::Auto, 3840, 2160));
        assert!(vrs_effective(VrsChoice::Auto, 6880, 2880));
        assert!(vrs_effective(VrsChoice::On, 1, 1));
        assert!(!vrs_effective(VrsChoice::Off, 6880, 2880));
    }

    #[test]
    fn engine_applied_notice_names_allocated_msaa_and_scale() {
        assert_eq!(
            engine_applied_notice(2, 1.5),
            "the renderer could only allocate 2x MSAA at 150% scale this session"
        );
        let requested = SessionGraphics {
            msaa: 8,
            render_scale: 1.5,
            notice: None,
        };
        assert!(engine_applied_differs(&requested, 2, 1.5));
        assert!(!engine_applied_differs(&requested, 8, 1.5));
    }

    #[test]
    fn core_is_default_with_visual_groups_stripped() {
        let mut expected = RenderConfig {
            occlusion: false,
            lod2: false,
            lod_levels: 1,
            lod_detail: 6,
            ..RenderConfig::default()
        };
        expected.strip_group(VisualGroup::Atmosphere);
        expected.strip_group(VisualGroup::Post);
        expected.strip_group(VisualGroup::Lighting);
        let core = RenderConfig::core();
        assert!(!core.bloom && !expected.bloom);
        assert!(!core.clouds && !expected.clouds);
        assert!(!core.shadows && !expected.shadows);
        assert!(core.sunlight && expected.sunlight);
        assert_eq!(core.occlusion, expected.occlusion);
        assert_eq!(core.lod2, expected.lod2);
        assert_eq!(core.normalized_lod(), expected.normalized_lod());
        assert_eq!(core.vrs, expected.vrs);
        assert_eq!(core.water_anim, expected.water_anim);
        assert_eq!(core.blocklight, expected.blocklight);
    }

    #[test]
    fn lod_defaults_and_normalization_preserve_detail_cap() {
        assert_eq!(RenderConfig::default().normalized_lod(), (7, 2));
        assert_eq!(
            RenderConfig { lod_levels: u8::MAX, lod_detail: u8::MAX, ..RenderConfig::default() }
                .normalized_lod(),
            (4, 6),
            "detail 6 can expose only levels 6 through 9"
        );
        assert_eq!(
            RenderConfig { lod_levels: 0, lod_detail: 0, ..RenderConfig::default() }
                .normalized_lod(),
            (1, 2)
        );
    }

    fn user_ultrawide_lanes() -> RenderConfig {
        RenderConfig { taa: true, bloom: true, exposure: true, ..RenderConfig::default() }
    }

    #[test]
    fn user_ultrawide_exceeds_48_gb_budget() {
        let bytes = render_target_bytes(3440, 1440, 2.0, 8, user_ultrawide_lanes());
        assert!(
            bytes > 4_800_000_000,
            "3440×1440 scale 2 8×MSAA TAA+bloom must exceed 4.8 GB, got {bytes}"
        );
    }

    #[test]
    fn full_hd_fits_48_gb_budget() {
        let bytes = render_target_bytes(1920, 1080, 1.0, 4, RenderConfig::default());
        assert!(bytes < 4_800_000_000, "1080p must fit 4.8 GB, got {bytes}");
    }

    fn heap_caps(heap: u64, max_msaa: u32) -> DeviceCaps {
        DeviceCaps {
            device_local_memory_bytes: Some(heap),
            available_device_bytes: None,
            max_msaa,
        }
    }

    fn live_caps(heap: u64, available: u64, max_msaa: u32) -> DeviceCaps {
        DeviceCaps {
            device_local_memory_bytes: Some(heap),
            available_device_bytes: Some(available),
            max_msaa,
        }
    }

    #[test]
    fn degrade_drops_msaa_before_scale_and_respects_floor() {
        let lanes = user_ultrawide_lanes();
        let needed_8 = render_target_bytes(3440, 1440, 2.0, 8, lanes);
        let needed_4 = render_target_bytes(3440, 1440, 2.0, 4, lanes);
        assert!(needed_8 > needed_4);

        // Heap-only: 60% of heap sits in [needed_4, needed_8), so 4× fits and 8× does not.
        let heap_for_4 = needed_4.div_ceil(VRAM_SAFETY_FRACTION).saturating_mul(100);
        assert!(
            heap_caps(heap_for_4, 8).render_target_budget_bytes().unwrap() >= needed_4
                && heap_caps(heap_for_4, 8).render_target_budget_bytes().unwrap() < needed_8
        );
        let msaa_only = fit_render_targets(3440, 1440, 2.0, 8, lanes, heap_caps(heap_for_4, 8));
        assert_eq!(msaa_only.msaa, 4);
        assert!((msaa_only.render_scale - 2.0).abs() < 1e-4, "scale stays until MSAA is 1");
        let n = msaa_only.notice.as_ref().unwrap();
        assert!(n.contains("4x MSAA"));
        assert!(!n.contains("% scale this session"));
        assert!(n.contains("budget"));

        let tiny = fit_render_targets(3440, 1440, 2.0, 8, lanes, heap_caps(1, 8));
        assert_eq!(tiny.msaa, 1);
        assert!((tiny.render_scale - VRAM_SCALE_FLOOR).abs() < 1e-4);
        let n = tiny.notice.unwrap();
        assert!(n.contains("1x MSAA"));
        assert!(n.contains("50% scale"));
        assert!(n.contains("starting at that floor anyway"));
    }

    #[test]
    fn live_available_budget_beats_heap_size() {
        let lanes = user_ultrawide_lanes();
        let caps = live_caps(8_000_000_000, 2_000_000_000, 8);
        let budget = caps.render_target_budget_bytes().unwrap();
        assert_eq!(budget, 2_000_000_000 * VRAM_AVAILABLE_SAFETY_FRACTION / 100);
        assert_eq!(budget, 1_700_000_000);

        let needed_8 = render_target_bytes(3440, 1440, 2.0, 8, lanes);
        assert!(needed_8 > budget, "8× 200% must miss a 1.7 GB live budget, got {needed_8}");

        let fitted = fit_render_targets(3440, 1440, 2.0, 8, lanes, caps);
        let cost = render_target_bytes(3440, 1440, fitted.render_scale, fitted.msaa, lanes);
        assert!(
            cost <= budget,
            "fitted {fitted:?} costs {cost}, budget {budget}"
        );
        let n = fitted.notice.as_ref().expect("live over-budget request prints a notice");
        assert!(
            n.contains("2.0 GB of 8.0 GB is free (other processes hold 6.0 GB)"),
            "{n}"
        );
        assert!(n.contains("8x MSAA at 200% scale needs"), "{n}");
        assert!(n.contains("running at"), "{n}");
        assert!(!n.contains("budget "), "{n}");
    }

    #[test]
    fn heap_only_fallback_keeps_sixty_percent_rule() {
        let lanes = user_ultrawide_lanes();
        let caps = heap_caps(8_000_000_000, 8);
        assert_eq!(caps.render_target_budget_bytes(), Some(4_800_000_000));
        let fitted = fit_render_targets(3440, 1440, 2.0, 8, lanes, caps);
        let cost = render_target_bytes(3440, 1440, fitted.render_scale, fitted.msaa, lanes);
        assert!(cost <= 4_800_000_000, "heap fallback fitted cost {cost}");
        let n = fitted.notice.as_ref().expect("8× 200% exceeds 4.8 GB");
        assert!(n.contains("budget 4.8 GB"), "{n}");
        assert!(!n.contains("is free"), "{n}");
    }
}
