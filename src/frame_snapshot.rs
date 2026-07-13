//! `FrameSnapshot` — the single CPU source of per-frame lighting state.
//! The `From<&FrameSnapshot>` impl below bridges to the GPU (FrameUniformsGpu).
//!
//! Colours stay linear and unclamped here; `Rgb::to_srgb8` is the only quantization point.
use voxel_engine::skeleton::{Exposure, FrameUniformsGpu, JitterOffset};
use voxel_engine::{genconst, DVec3, Vec3};

use crate::render_config::RenderConfig;
use crate::sky::palette::{Palette, Rgb, Role, RAIN_HORIZON, RAIN_ZENITH};
use crate::sky::Sky;

/// Minimum ambient luma: shadowed/indoor scenes floor here instead of pure black,
/// preserving the old day-night look when sky was centralized.
const AVOID_DARK_LEVEL: f32 = 0.030;

/// Base exponential fog density in clear weather.
/// Tuned so terrain fades at the view horizon instead of cutting off abruptly.
const FOG_BASE: f32 = 0.00023;

/// Per-frame dither phase; gains are shader-side.
#[derive(Clone, Copy, Debug, Default)]
pub struct DitherPhase(pub f32);

/// Per-frame dither phase from the generated constant table; shared with shaders.
pub fn dither_at(frame_index: u64) -> DitherPhase {
    let table = &genconst::DITHER_PHASE_16;
    DitherPhase(table[(frame_index % table.len() as u64) as usize])
}

/// Per-frame rendering state (linear colour, unclamped).
pub struct FrameSnapshot {
    pub frame_index: u64,
    pub sun_dir: Vec3,
    /// Sun elevation in radians; drives palette blending for day/night.
    pub elevation: f32,
    pub day_night_mix: f32,
    pub light: Rgb,
    pub zenith: Rgb,
    pub horizon: Rgb,
    pub candle: Rgb,
    pub ambient_floor: f32,
    pub fog_density: f32,
    pub turbidity: f32,
    /// Live autoexposure value from the render thread.
    pub exposure: Exposure,
    pub dither: DitherPhase,
    /// `JitterOffset::ZERO` until Phase E.
    pub jitter: JitterOffset,
    /// World time wrapped into [0, ANIM_PERIOD) for shader animation phase (unused).
    pub anim_time: f32,
    /// Camera XZ wrapped to [0,1) in f64 before downcast, preserving f32 precision at distance.
    pub anim_uv: [f32; 2],
    /// Camera altitude; tells shader where to position the cloud slab.
    /// Pass f32::MAX when clouds are disabled to skip rendering.
    pub camera_y: f32,
}

/// Compute per-frame lighting state from sky conditions and time.
/// Single source for direct light, ambient, and fog; combines sky, weather, and time.
pub fn compose(
    sky: &Sky,
    cam_world: DVec3,
    frame_index: u64,
    exposure: Exposure,
    render: &RenderConfig,
) -> FrameSnapshot {
    let clock = &sky.clock;
    let atm = &sky.atmosphere;
    let weather = &sky.weather;

    let sun_dir = clock.sun_dir();
    let elev = clock.sun_elevation();
    let daylight = clock.daylight();

    // Direct light, scaled/desaturated by overcast.
    let mut light = atm.palette.at(Role::Light, elev);
    // Disable weather here (not in shader) to allow debug captures without branch overhead.
    let coverage = if render.weather { weather.coverage } else { 0.0 };
    let overcast = coverage * 0.5;
    let light_luma = light.luma();
    light = light
        .lerp(Rgb::linear(light_luma, light_luma, light_luma), overcast)
        .scale(1.0 - overcast * 0.4);

    // Ambient: tinted shadow floor so caves don't render pure black.
    // Keep zenith RAW for re-tinting on GPU; `ambient_floor` records the floored luma.
    // Rain desaturates sky (zenith/horizon only; direct light already muted above).
    let rain = weather.rain_strength();
    let zenith = atm.palette.at(Role::Zenith, elev).rain_override(RAIN_ZENITH, rain);
    let amt = 0.10 + 0.12 * daylight;
    let ambient = zenith.scale(amt);
    let al = ambient.luma();
    let ambient_floor = if al > 0.0 && al < AVOID_DARK_LEVEL { AVOID_DARK_LEVEL } else { al };

    // Horizon colour; shader's sky_radiance reads this as gradient base.
    let horizon = atm.palette.at(Role::Horizon, elev).rain_override(RAIN_HORIZON, rain);
    // Fog density (passed in horizon.w). Engine RenderFlags::fog gate controls it downstream;
    // currently disabled by default, so fog is inert until that flag is turned on.
    let fog_density = FOG_BASE + weather.fog_bonus();

    // Wrap time and camera position in f64 before downcast to preserve f32 phase precision at distance.
    let period = genconst::ANIM_PERIOD as f64;
    let anim_time = (clock.day() * sky.day_length.0).rem_euclid(period) as f32;
    let anim_uv = [
        (cam_world.x / period).rem_euclid(1.0) as f32,
        (cam_world.z / period).rem_euclid(1.0) as f32,
    ];

    FrameSnapshot {
        frame_index,
        sun_dir,
        elevation: elev,
        day_night_mix: Palette::day_night_mix(elev),
        light,
        zenith,
        horizon,
        // Blocklight (candle) color, tuned to match torch/lantern light in-game.
        candle: Rgb::linear(0.27475, 0.17392, 0.0899),
        ambient_floor,
        fog_density,
        turbidity: atm.turbidity,
        exposure,
        dither: dither_at(frame_index),
        jitter: JitterOffset::ZERO,
        anim_time,
        anim_uv,
        // When clouds are off, f32::MAX makes sky.frag early-out at no cost.
        camera_y: if render.clouds { cam_world.y as f32 } else { f32::MAX },
    }
}

impl From<&FrameSnapshot> for FrameUniformsGpu {
    /// Bridge to GPU. Colours stay linear and unclamped; quantization happens at `to_srgb8`.
    fn from(s: &FrameSnapshot) -> FrameUniformsGpu {
        FrameUniformsGpu {
            sun_dir_elev: [s.sun_dir.x, s.sun_dir.y, s.sun_dir.z, s.elevation],
            light: [s.light.r(), s.light.g(), s.light.b(), s.day_night_mix],
            zenith: [s.zenith.r(), s.zenith.g(), s.zenith.b(), s.turbidity],
            horizon: [s.horizon.r(), s.horizon.g(), s.horizon.b(), s.fog_density],
            candle: [s.candle.r(), s.candle.g(), s.candle.b(), s.ambient_floor],
            exposure_dither: [s.exposure.0, s.dither.0, s.jitter.0.x, s.jitter.0.y],
            reserved: [0.0; 4],
            anim: [s.anim_time, s.anim_uv[0], s.anim_uv[1], s.camera_y],
        }
    }
}
