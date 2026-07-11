//! `FrameSnapshot` — the ONE place per-frame lighting state is computed.
//! It absorbs the old `SkyEnv::resolve` and `Sky::fog` bodies,
//! so there is a single CPU truth for the frame's sky/lighting; the wire form is
//! [`voxel_engine::skeleton::FrameUniformsGpu`] and the `From<&FrameSnapshot>`
//! impl below is the ONE crossing to the GPU.
//!
//! Semantics are typed, linear, and unclamped: colours never quantise here
//! (that is `Rgb::to_srgb8`'s sole job). `SkyEnv::resolve` is DELETED in the
//! same change that lands this so two sources of truth cannot compile.
use voxel_engine::skeleton::{Exposure, FrameUniformsGpu, JitterOffset};
use voxel_engine::{genconst, DVec3, Vec3};

use crate::render_config::RenderConfig;
use crate::sky::palette::{Palette, Rgb, Role, RAIN_HORIZON, RAIN_ZENITH};
use crate::sky::Sky;

/// Minimum ambient luma — the `AVOID_DARK_LEVEL` floor: scenes
/// with no sky access never fall to pure black, they floor at a dim version of
/// the ambient tint. Re-derived to match the old `env.rs` night floor so the
/// default look is preserved. (Moved here from `env.rs` with `SkyEnv::resolve`.)
const AVOID_DARK_LEVEL: f32 = 0.030;

/// Base exponential fog density in clear weather (was `sky/mod.rs::FOG_BASE`).
/// Tuned so terrain fades into the horizon near the edge of a typical view
/// radius rather than cutting off.
const FOG_BASE: f32 = 0.0009;

/// Per-frame element of the shared stochastic sequence — the PHASE only; gains
/// are shader-side generated constants.
#[derive(Clone, Copy, Debug, Default)]
pub struct DitherPhase(pub f32);

/// Entry `frame % TEMPORAL_SEQ_LEN` of the shuffled k/16 table from the generated
/// constants — single source with the shaders.
pub fn dither_at(frame_index: u64) -> DitherPhase {
    let table = &genconst::DITHER_PHASE_16;
    DitherPhase(table[(frame_index % table.len() as u64) as usize])
}

/// SEMANTIC per-frame state: typed, linear, unclamped, CPU truth.
pub struct FrameSnapshot {
    pub frame_index: u64,
    pub sun_dir: Vec3,
    /// Sun elevation in radians (the palette blend axis).
    pub elevation: f32,
    pub day_night_mix: f32,
    pub light: Rgb,
    pub zenith: Rgb,
    pub horizon: Rgb,
    pub candle: Rgb,
    pub ambient_floor: f32,
    pub fog_density: f32,
    pub turbidity: f32,
    /// Last METERED+smoothed exposure (the live autoexposure value the render
    /// thread published, read via `Engine::exposure_for_compose`).
    pub exposure: Exposure,
    pub dither: DitherPhase,
    /// `JitterOffset::ZERO` until Phase E.
    pub jitter: JitterOffset,
    /// World-time seconds folded into `[0, ANIM_PERIOD)` — the phase animated
    /// shaders read (foliage/water). No shader samples it yet.
    pub anim_time: f32,
    /// `fract(camera_world.xz / ANIM_PERIOD)` — camera UV wrapped CPU-side in f64
    /// so f32 keeps phase precision arbitrarily far from the origin.
    pub anim_uv: [f32; 2],
    /// Camera world-y (metres). The cloud slab (sky.frag) sits at a fixed world
    /// altitude; the shader works camera-relative, so it needs the camera height
    /// to place the slab planes. World-y is bounded (voxel column height), so a
    /// plain f32 narrow keeps full precision — no wrap needed, unlike xz.
    pub camera_y: f32,
}

/// The ONE place per-frame lighting state is computed. Absorbs the bodies
/// of `SkyEnv::resolve` (direct light + ambient) and `Sky::fog` (horizon colour +
/// density). Weather comes through `sky.weather`; day/night mixers through
/// `sky.clock` + palette.
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

    // Direct light scaled/desaturated by overcast (the rain rule, reduced).
    let mut light = atm.palette.at(Role::Light, elev);
    // `RenderConfig::weather` off zeroes coverage here, before it scales direct
    // light — a clear-sky bless/debug capture with no shader-side branch.
    let coverage = if render.weather { weather.coverage } else { 0.0 };
    let overcast = coverage * 0.5;
    let light_luma = light.luma();
    light = light
        .lerp(Rgb::linear(light_luma, light_luma, light_luma), overcast)
        .scale(1.0 - overcast * 0.4);

    // Ambient: a dim, sky-tinted floor so shadowed faces and caves are not pure
    // black. Tint follows the zenith; amount follows daylight; floored at
    // AVOID_DARK_LEVEL luma so nothing renders unreadably black. `zenith` is kept
    // RAW (the palette colour, into the UBO `zenith` lane) and `ambient_floor`
    // records the floored luma (UBO `candle.w`) — consumers reconstruct the
    // tinted ambient from the two (e.g. engine `KeyLight` for avatar shading).
    // Rain desaturates sky; only zenith/horizon (direct light already muted above).
    let rain = weather.rain_strength();
    let zenith = atm.palette.at(Role::Zenith, elev).rain_override(RAIN_ZENITH, rain);
    let amt = 0.10 + 0.12 * daylight;
    let ambient = zenith.scale(amt);
    let al = ambient.luma();
    let ambient_floor = if al > 0.0 && al < AVOID_DARK_LEVEL { AVOID_DARK_LEVEL } else { al };

    // Horizon palette lane, straight into the UBO: the shader's `sky_radiance`
    // is the only sky/fog formula, and it reads this lane as the gradient base.
    let horizon = atm.palette.at(Role::Horizon, elev).rain_override(RAIN_HORIZON, rain);
    // Fog density written to `horizon.w`. There is no app-side fog toggle: the
    // engine `RenderFlags::fog` gate (`frame::gate_uniforms`) is the single fog
    // switch and zeroes this lane downstream when off — which it is by default,
    // so terrain distance fog is currently inert everywhere until that flag is
    // enabled. The old `WATT_FOG` was a redundant second gate on the same lane.
    let fog_density = FOG_BASE + weather.fog_bonus();

    // `anim` lane: world time and camera position both wrapped to [0, ANIM_PERIOD)
    // in f64 BEFORE narrowing, so the f32 the shader eventually reads never loses
    // phase precision far into a day or far from the world origin.
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
        // Sentinel when clouds are toggled off (`RenderConfig::clouds`): a camera
        // far above the slab makes the sky.frag march early-out at zero cost.
        camera_y: if render.clouds { cam_world.y as f32 } else { f32::MAX },
    }
}

impl From<&FrameSnapshot> for FrameUniformsGpu {
    /// Layout per `FrameUniformsGpu` docs. Colours pass through LINEAR and
    /// unclamped — quantisation is `to_srgb8`'s job, never this one's.
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
