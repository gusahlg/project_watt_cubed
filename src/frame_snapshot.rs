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
use voxel_engine::{genconst, Vec3};

use crate::sky::palette::{Palette, Rgb, Role};
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
    /// Last METERED exposure, or `Exposure::DEFAULT` until Phase C lands.
    pub exposure: Exposure,
    pub dither: DitherPhase,
    /// `JitterOffset::ZERO` until Phase E.
    pub jitter: JitterOffset,
}

/// The ONE place per-frame lighting state is computed. Absorbs the bodies
/// of `SkyEnv::resolve` (direct light + ambient) and `Sky::fog` (horizon colour +
/// density). Weather comes through `sky.weather`; day/night mixers through
/// `sky.clock` + palette.
pub fn compose(sky: &Sky, frame_index: u64, exposure: Exposure) -> FrameSnapshot {
    let clock = &sky.clock;
    let atm = &sky.atmosphere;
    let weather = &sky.weather;

    let sun_dir = clock.sun_dir();
    let elev = clock.sun_elevation();
    let daylight = clock.daylight();

    // Direct light: the palette Light role at this elevation. Overcast mutes and
    // desaturates it toward its own luma (MakeUp's rain rule, reduced).
    let mut light = atm.palette.at(Role::Light, elev);
    let overcast = weather.coverage * 0.5;
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
    let zenith = atm.palette.at(Role::Zenith, elev);
    let amt = 0.10 + 0.12 * daylight;
    let ambient = zenith.scale(amt);
    let al = ambient.luma();
    let ambient_floor = if al > 0.0 && al < AVOID_DARK_LEVEL { AVOID_DARK_LEVEL } else { al };

    // Horizon colour terrain fog fades toward — the same sample `Sky::fog` used.
    let horizon = atm.radiance_rgb(Vec3::new(0.0, 0.02, 1.0).normalize(), sun_dir);
    let fog_density = FOG_BASE + weather.fog_bonus();

    FrameSnapshot {
        frame_index,
        sun_dir,
        elevation: elev,
        day_night_mix: Palette::day_night_mix(elev),
        light,
        zenith,
        horizon,
        // Blocklight (candle) colour: a warm placeholder for now
        // (no shader reads the candle lane yet — the per-frame UBO is neutral).
        candle: Rgb::linear(1.0, 0.85, 0.6),
        ambient_floor,
        fog_density,
        turbidity: atm.turbidity,
        exposure,
        dither: dither_at(frame_index),
        jitter: JitterOffset::ZERO,
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
        }
    }
}
