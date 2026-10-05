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

/// Camera altitudes over which the atmosphere thins into space: the sky darkens, the stars come
/// out by day and the haze clears.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpaceFade(pub f64, pub f64);

impl SpaceFade {
    /// A flat world: from just above the highest mountains to a little above the floor of the space
    /// realm (`crate::world::terrain::SPACE_FLOOR`).
    pub const FLAT: Self = Self(500.0, 820.0);
    /// A body of the cosmos: from above its relief band to the top of its air.
    pub const BODY: Self = Self(2_500.0, crate::world::terrain::cosmos::AIR_TOP);
}
/// The sky overhead in space, and the atmosphere's limb below the horizon (where far ground fades).
const SPACE_ZENITH: Rgb = Rgb::linear(0.0004, 0.0006, 0.0016);
const SPACE_HORIZON: Rgb = Rgb::linear(0.0015, 0.003, 0.009);
/// Turbidity in space: the sun's halo is atmospheric scatter, so it all but vanishes
/// (the engine scales the halo by `0.5 + turbidity`).
const SPACE_TURBIDITY: f32 = -0.46;

/// How far into space an altitude above the local surface datum is: 0 in the
/// atmosphere, 1 above it (smoothstep). The `+Y` face datum is world `y = 0`.
pub fn space_factor(altitude: f64, fade: SpaceFade) -> f32 {
    let t = ((altitude - fade.0) / (fade.1 - fade.0)).clamp(0.0, 1.0) as f32;
    t * t * (3.0 - 2.0 * t)
}

/// The viewer's local sky frame for one composed frame.
///
/// `up` is the body frame before roll (`ViewPose::up`).
/// `altitude` is metres above the local surface datum (`Game::sky_altitude`).
#[derive(Clone, Copy, Debug)]
pub struct SkyContext {
    pub up: DVec3,
    pub altitude: f64,
    /// Where this world's atmosphere ends.
    pub fade: SpaceFade,
}

/// Per-frame rendering state (linear colour, unclamped).
pub struct FrameSnapshot {
    pub sun_dir: Vec3,
    /// Sun elevation above the local horizon, `dot(sun_dir, up)` in `[-1, 1]`.
    pub elevation: f32,
    pub day_night_mix: f32,
    pub light: Rgb,
    pub zenith: Rgb,
    pub horizon: Rgb,
    pub candle: Rgb,
    pub ambient_floor: f32,
    pub fog_density: f32,
    pub turbidity: f32,
    /// The least night factor the starfield renders at: 0 in the atmosphere, 1 in space.
    pub star_floor: f32,
    /// Live autoexposure value from the render thread.
    pub exposure: Exposure,
    /// `JitterOffset::ZERO` until Phase E.
    pub jitter: JitterOffset,
    /// World time wrapped into [0, ANIM_PERIOD) for shader animation phase (unused).
    pub anim_time: f32,
    /// Tangent-plane camera position wrapped to [0,1) in f64 before downcast.
    /// World XZ when up is `+Y`.
    pub anim_uv: [f32; 2],
    /// Altitude above the local datum; tells the shader where the cloud slab sits.
    /// Pass f32::MAX when clouds are disabled to skip rendering.
    pub camera_y: f32,
}

/// Wrap two tangent-plane metres into `[0, 1)` in f64 before the f32 downcast.
/// The game caches by the unwrapped plane coords so a still eye skips both divisions.
pub fn wrap_plane(u: f64, v: f64) -> [f32; 2] {
    let period = genconst::ANIM_PERIOD as f64;
    [(u / period).rem_euclid(1.0) as f32, (v / period).rem_euclid(1.0) as f32]
}

/// Camera position projected onto the local tangent plane, then wrapped.
/// For up `+Y` the basis is world X and Z, so this is the old world-XZ wrap.
#[cfg(test)]
pub fn animation_uv(cam: DVec3, tangent: DVec3, bitangent: DVec3) -> [f32; 2] {
    wrap_plane(cam.dot(tangent), cam.dot(bitangent))
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
    let (t, _, b) = voxel_engine::local_sky_basis(Vec3::Y);
    compose_at(
        sky,
        sky.frame(),
        SkyContext { up: DVec3::Y, altitude: cam_world.y, fade: SpaceFade::FLAT },
        animation_uv(
            cam_world,
            DVec3::new(t.x as f64, t.y as f64, t.z as f64),
            DVec3::new(b.x as f64, b.y as f64, b.z as f64),
        ),
        exposure,
        render,
    )
}

/// [`compose`] against an already-sampled clock frame, local sky context, and
/// animation UV. The caller built `frame` with `ctx.up`, so elevation is that
/// dot and is not recomputed here. Caches key the clock on day and up, and the
/// UV on the tangent-plane projection.
pub fn compose_at(
    sky: &Sky,
    frame: SkyFrame,
    ctx: SkyContext,
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

    // Above the atmosphere the sky goes black and starry and the haze clears. Only the sky lanes
    // fade: the ambient above was taken from the atmosphere's zenith, and the GPU luma-matches
    // its zenith tints, so planets stay lit by the sun and the near-sky bounce. Fog reaches 0 at
    // the top of the air: a mapped body is fogged like terrain, and that haze is the air shell.
    let space = space_factor(ctx.altitude, ctx.fade);
    // Nor is there night out there: the sun shines from where it is, however the viewer is turned.
    let light = light.lerp(atm.palette.at(Role::Light, 1.0), space);
    let day_night_mix = Palette::day_night_mix(elev) + (1.0 - Palette::day_night_mix(elev)) * space;
    let zenith = zenith.lerp(SPACE_ZENITH, space);
    let horizon = horizon.lerp(SPACE_HORIZON, space);
    let fog_density = fog_density * (1.0 - space);

    // Wrap time in f64 before downcast to preserve f32 phase precision.
    let period = genconst::ANIM_PERIOD as f64;
    let anim_time = (clock.day() * sky.day_length.0).rem_euclid(period) as f32;

    FrameSnapshot {
        sun_dir,
        elevation: elev,
        day_night_mix,
        light,
        zenith,
        horizon,
        // Blocklight (candle) color, tuned to match torch/lantern light in-game.
        candle: Rgb::linear(0.27475, 0.17392, 0.0899),
        ambient_floor,
        fog_density,
        turbidity: atm.turbidity + (SPACE_TURBIDITY - atm.turbidity) * space,
        star_floor: space,
        exposure,
        jitter: JitterOffset::ZERO,
        anim_time,
        anim_uv,
        // When clouds are off, f32::MAX makes sky.frag early-out at no cost.
        camera_y: if render.clouds { ctx.altitude as f32 } else { f32::MAX },
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
            // .y is the star floor (stars by day in space); .zw carry TAA jitter.
            exposure_dither: [s.exposure.0, s.star_floor, s.jitter.0.x, s.jitter.0.y],
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

    /// On a body of the cosmos the sky stays blue over mountains and turns to space at the top of
    /// the air; a flat world keeps its low space realm.
    #[test]
    fn the_sky_turns_to_space_where_the_air_ends() {
        assert_eq!(space_factor(1_500.0, SpaceFade::BODY), 0.0, "a mountain top under a blue sky");
        let mid = space_factor(11_000.0, SpaceFade::BODY);
        assert!(mid > 0.2 && mid < 0.8, "half way up the air the sky darkens: {mid}");
        assert_eq!(space_factor(crate::world::terrain::cosmos::AIR_TOP, SpaceFade::BODY), 1.0);
        assert_eq!(space_factor(400.0, SpaceFade::FLAT), 0.0);
        assert_eq!(space_factor(900.0, SpaceFade::FLAT), 1.0);
    }

    /// Mapped impostors are fogged like terrain, so the haze has to belong to the air shell.
    #[test]
    fn fog_clears_at_the_top_of_the_air() {
        let sky = Sky::new();
        let render = RenderConfig { weather: false, ..RenderConfig::default() };
        let frame = sky.frame_at_day(0.5, Vec3::Y);
        let fog_at = |altitude: f64| {
            compose_at(
                &sky,
                frame,
                SkyContext { up: DVec3::Y, altitude, fade: SpaceFade::BODY },
                [0.0; 2],
                Exposure::DEFAULT,
                &render,
            )
            .fog_density
        };
        let ground = fog_at(0.0);
        let mid = fog_at(11_000.0);
        let top = fog_at(crate::world::terrain::cosmos::AIR_TOP);
        let above = fog_at(crate::world::terrain::cosmos::AIR_TOP + 50_000.0);
        assert!(ground > 0.0, "the ground still has haze: {ground}");
        assert!(mid > 0.0 && mid < ground, "mid-air haze sits between: mid {mid} ground {ground}");
        assert_eq!(top, 0.0, "no haze at the top of the air");
        assert_eq!(above, 0.0, "no haze above the air");
    }
    use crate::sky::Precip;

    fn channels(rgb: Rgb) -> [f32; 3] {
        [rgb.r(), rgb.g(), rgb.b()]
    }

    #[test]
    fn disabled_weather_removes_every_weather_derived_lane() {
        let mut sky = Sky::new();
        let render = RenderConfig { weather: false, clouds: false, ..RenderConfig::default() };
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
        let render = RenderConfig { clouds: false, ..RenderConfig::default() };
        let snapshot =
            compose(&sky, DVec3::new(12_345.0, 80.0, -54_321.0), Exposure::DEFAULT, &render);
        assert_ne!(snapshot.anim_uv, [0.0; 2], "wrapped camera-XZ anchoring survives");
        assert_eq!(snapshot.camera_y, f32::MAX, "clouds-off sentinel early-outs the shader");
    }

    fn basis_dvec(v: Vec3) -> DVec3 {
        DVec3::new(v.x as f64, v.y as f64, v.z as f64)
    }

    #[test]
    fn plus_y_animation_uv_is_the_world_xz_wrap() {
        let cam = DVec3::new(12_345.0, 80.0, -54_321.0);
        let (t, _, b) = voxel_engine::local_sky_basis(Vec3::Y);
        let uv = animation_uv(cam, basis_dvec(t), basis_dvec(b));
        assert_eq!(uv, wrap_plane(cam.x, cam.z));
    }

    #[test]
    fn plus_x_animation_uv_uses_the_tangent_plane() {
        let cam = DVec3::new(10.0, 20.0, 30.0);
        let (t, u, b) = voxel_engine::local_sky_basis(Vec3::X);
        assert_eq!((t, u, b), (Vec3::Y, Vec3::X, -Vec3::Z));
        let uv = animation_uv(cam, basis_dvec(t), basis_dvec(b));
        assert_eq!(uv, wrap_plane(cam.y, -cam.z));
        assert_ne!(uv, wrap_plane(cam.x, cam.z));
    }

    #[test]
    fn compose_follows_local_up_and_altitude() {
        let mut sky = Sky::new();
        sky.clock.set_day(0.5);
        let render = RenderConfig { clouds: true, ..RenderConfig::default() };
        let frame = sky.frame_at_day(0.5, Vec3::X);
        let ground = compose_at(
            &sky,
            frame,
            SkyContext { up: DVec3::X, altitude: 0.0, fade: SpaceFade::FLAT },
            [0.0; 2],
            Exposure::DEFAULT,
            &render,
        );
        assert!((ground.elevation - frame.sun_dir.dot(Vec3::X)).abs() < 1e-6);
        assert_ne!(ground.elevation, frame.sun_dir.y);
        assert_eq!(ground.star_floor, 0.0);
        assert_eq!(ground.camera_y, 0.0);

        let space = compose_at(
            &sky,
            frame,
            SkyContext { up: DVec3::X, altitude: 900.0, fade: SpaceFade::FLAT },
            [0.0; 2],
            Exposure::DEFAULT,
            &render,
        );
        assert_eq!(space.star_floor, 1.0);
        assert!((space.camera_y - 900.0).abs() < 1e-3);
    }
}
