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

/// Per-frame rendering state (linear colour, unclamped).
pub struct FrameSnapshot {
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

/// Camera XZ wrapped to [0,1) in f64 before downcast, preserving f32 phase
/// precision at distance. Split out so the game can cache it by exact camera
/// XZ bits: translation-free frames skip both `rem_euclid` divisions.
pub fn animation_uv(cam_world: DVec3) -> [f32; 2] {
    let period = genconst::ANIM_PERIOD as f64;
    [(cam_world.x / period).rem_euclid(1.0) as f32, (cam_world.z / period).rem_euclid(1.0) as f32]
}

/// Compute per-frame lighting state from sky conditions and time.
/// Single source for direct light, ambient, and fog; combines sky, weather, and time.
#[cfg(test)]
pub fn compose(
    sky: &Sky,
    cam_world: DVec3,
    exposure: Exposure,
    render: &RenderConfig,
) -> FrameSnapshot {
    compose_at(sky, sky.frame(), cam_world, animation_uv(cam_world), exposure, render)
}

/// [`compose`] against an already-sampled clock frame and animation UV, so the
/// game's per-frame caches (clock sample by day value, UV by camera XZ bits)
/// feed the one composition path instead of a parallel one.
pub fn compose_at(
    sky: &Sky,
    frame: SkyFrame,
    cam_world: DVec3,
    anim_uv: [f32; 2],
    exposure: Exposure,
    render: &RenderConfig,
) -> FrameSnapshot {
    let clock = &sky.clock;
    let atm = &sky.atmosphere;
    let weather = &sky.weather;

    let SkyFrame { sun_dir, elevation: elev, daylight } = frame;

    // Disabled weather removes EVERY weather-derived input before composition —
    // coverage, rain palette overrides, and the fog bonus — so it can never
    // leave storm tint or weather fog active. Gated here (not in shader) to
    // allow debug captures without branch overhead.
    let (coverage, rain, fog_bonus) = if render.weather {
        (weather.coverage, weather.rain_strength(), weather.fog_bonus())
    } else {
        (0.0, 0.0, 0.0)
    };

    // Direct light, scaled/desaturated by overcast.
    let mut light = atm.palette.at(Role::Light, elev);
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

    // Wrap time in f64 before downcast to preserve f32 phase precision.
    let period = genconst::ANIM_PERIOD as f64;
    let anim_time = (clock.day() * sky.day_length.0).rem_euclid(period) as f32;

    FrameSnapshot {
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
            // .y is a reserved zero (post-effect dither removed); .zw carry TAA jitter.
            exposure_dither: [s.exposure.0, 0.0, s.jitter.0.x, s.jitter.0.y],
            // x = stars gain: always composed ON; the engine's RenderFlags::stars
            // gate (frame::gate_uniforms) zeroes it, like every other lane gate.
            extras: [1.0, 0.0, 0.0, 0.0],
            anim: [s.anim_time, s.anim_uv[0], s.anim_uv[1], s.camera_y],
        }
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
        let render =
            RenderConfig { weather: false, clouds: false, water_anim: false, ..RenderConfig::default() };
        sky.weather.coverage = 1.0;
        sky.weather.precip = Precip::Rain;
        sky.weather.wetness = 1.0;
        let storm = compose(&sky, DVec3::ZERO, Exposure::DEFAULT, &render);

        sky.weather.coverage = 0.0;
        sky.weather.precip = Precip::Clear;
        sky.weather.wetness = 0.0;
        let clear = compose(&sky, DVec3::ZERO, Exposure::DEFAULT, &render);

        assert_eq!(channels(storm.light), channels(clear.light));
        assert_eq!(channels(storm.zenith), channels(clear.zenith));
        assert_eq!(channels(storm.horizon), channels(clear.horizon));
        assert_eq!(storm.fog_density, clear.fog_density);
    }

    #[test]
    fn composed_snapshot_keeps_world_anchoring_when_clouds_are_off() {
        let sky = Sky::new();
        let render = RenderConfig { clouds: false, water_anim: false, ..RenderConfig::default() };
        let snapshot =
            compose(&sky, DVec3::new(12_345.0, 80.0, -54_321.0), Exposure::DEFAULT, &render);
        assert_ne!(snapshot.anim_uv, [0.0; 2], "wrapped camera-XZ anchoring survives");
        assert_eq!(snapshot.camera_y, f32::MAX, "clouds-off sentinel early-outs the shader");
    }
}
