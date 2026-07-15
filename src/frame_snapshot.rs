//! `FrameSnapshot` — the single CPU source of per-frame lighting state.
//! The `From<&FrameSnapshot>` impl below bridges to the GPU (FrameUniformsGpu).
//!
//! Colours stay linear and unclamped here; `Rgb::to_srgb8` is the only quantization point.
use voxel_engine::skeleton::{Exposure, FrameUniformsGpu, JitterOffset};
use voxel_engine::{genconst, DVec3, Vec3};

use crate::render_config::RenderConfig;
use crate::sky::palette::{Palette, Rgb, Role, RAIN_HORIZON, RAIN_ZENITH};
use crate::sky::{Sky, SkyFrame};

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

/// Wrapped camera XZ consumed by water/cloud shaders. Kept separate so a
/// frozen lighting packet can update this precision-preserving lane only when
/// the camera actually moves.
pub(crate) fn animation_uv(cam_world: DVec3) -> [f32; 2] {
    let period = genconst::ANIM_PERIOD as f64;
    [
        (cam_world.x / period).rem_euclid(1.0) as f32,
        (cam_world.z / period).rem_euclid(1.0) as f32,
    ]
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
    compose_at(sky, sky.frame(), cam_world, frame_index, exposure, render)
}

/// Compose from a clock sample shared with clear-colour and sky-geometry
/// consumers. The game uses this path so one frame performs one sun sample.
pub fn compose_at(
    sky: &Sky,
    sky_frame: SkyFrame,
    cam_world: DVec3,
    frame_index: u64,
    exposure: Exposure,
    render: &RenderConfig,
) -> FrameSnapshot {
    compose_at_with_uv(
        sky,
        sky_frame,
        cam_world,
        animation_uv(cam_world),
        frame_index,
        exposure,
        render,
    )
}

/// Game hot path: accepts the separately cached camera-XZ animation anchor.
/// Public compose helpers retain their self-contained contract above.
pub(crate) fn compose_at_with_uv(
    sky: &Sky,
    sky_frame: SkyFrame,
    cam_world: DVec3,
    anim_uv: [f32; 2],
    frame_index: u64,
    exposure: Exposure,
    render: &RenderConfig,
) -> FrameSnapshot {
    let clock = &sky.clock;
    let atm = &sky.atmosphere;
    let weather = &sky.weather;

    let sun_dir = sky_frame.sun_dir;
    let elev = sky_frame.elevation;
    let daylight = sky_frame.daylight;

    // Direct light, scaled/desaturated by overcast.
    let mut light = atm.palette.at(Role::Light, elev);
    // Disable weather here (not in shader) to allow stripped profiles and debug
    // captures to skip every weather-derived lane without a shader branch.
    let (coverage, rain, fog_bonus) = if render.weather {
        (weather.coverage, weather.rain_strength(), weather.fog_bonus())
    } else {
        (0.0, 0.0, 0.0)
    };
    let overcast = coverage * 0.5;
    let light_luma = light.luma();
    light = light
        .lerp(Rgb::linear(light_luma, light_luma, light_luma), overcast)
        .scale(1.0 - overcast * 0.4);

    // Ambient: tinted shadow floor so caves don't render pure black.
    // Keep zenith RAW for re-tinting on GPU; `ambient_floor` records the floored luma.
    // Rain desaturates sky (zenith/horizon only; direct light already muted above).
    let zenith = atm.palette.at(Role::Zenith, elev).rain_override(RAIN_ZENITH, rain);
    let amt = 0.10 + 0.12 * daylight;
    let ambient = zenith.scale(amt);
    let al = ambient.luma();
    let ambient_floor = if al > 0.0 && al < AVOID_DARK_LEVEL { AVOID_DARK_LEVEL } else { al };

    // Horizon colour; shader's sky_radiance reads this as gradient base.
    let horizon = atm.palette.at(Role::Horizon, elev).rain_override(RAIN_HORIZON, rain);
    // Fog density (passed in horizon.w). Engine RenderFlags::fog gate controls it downstream;
    // currently disabled by default, so fog is inert until that flag is turned on.
    let fog_density = FOG_BASE + fog_bonus;

    // Freeze time when neither animation consumer is enabled. Camera XZ must
    // remain world-anchored even for still water, so its wrapped coordinates
    // are retained (the game caches this whole packet between camera moves).
    let period = genconst::ANIM_PERIOD as f64;
    let anim_time = if render.water_anim || render.clouds {
        (clock.day() * sky.day_length.0).rem_euclid(period) as f32
    } else {
        0.0
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::Precip;

    fn channels(rgb: Rgb) -> [f32; 3] {
        [rgb.r(), rgb.g(), rgb.b()]
    }

    #[test]
    fn disabled_weather_removes_every_weather_derived_lane() {
        let mut sky = Sky::new();
        let render = RenderConfig {
            weather: false,
            clouds: false,
            water_anim: false,
            ..RenderConfig::default()
        };
        sky.weather.coverage = 1.0;
        sky.weather.precip = Precip::Rain;
        sky.weather.wetness = 1.0;
        let storm = compose(&sky, DVec3::ZERO, 0, Exposure::DEFAULT, &render);

        sky.weather.coverage = 0.0;
        sky.weather.precip = Precip::Clear;
        sky.weather.wetness = 0.0;
        let clear = compose(&sky, DVec3::ZERO, 0, Exposure::DEFAULT, &render);

        assert_eq!(channels(storm.light), channels(clear.light));
        assert_eq!(channels(storm.zenith), channels(clear.zenith));
        assert_eq!(channels(storm.horizon), channels(clear.horizon));
        assert_eq!(storm.fog_density, clear.fog_density);
    }

    #[test]
    fn disabled_animation_consumers_freeze_time_but_keep_world_anchoring() {
        let sky = Sky::new();
        let render = RenderConfig {
            clouds: false,
            water_anim: false,
            ..RenderConfig::default()
        };
        let snapshot = compose(
            &sky,
            DVec3::new(12_345.0, 80.0, -54_321.0),
            9,
            Exposure::DEFAULT,
            &render,
        );
        assert_eq!(snapshot.anim_time, 0.0);
        assert_ne!(snapshot.anim_uv, [0.0; 2]);
        assert_eq!(snapshot.camera_y, f32::MAX);
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
            // x = stars gain: always composed ON; the engine's RenderFlags::stars
            // gate (frame::gate_uniforms) zeroes it, like every other lane gate.
            extras: [1.0, 0.0, 0.0, 0.0],
            anim: [s.anim_time, s.anim_uv[0], s.anim_uv[1], s.camera_y],
        }
    }
}
