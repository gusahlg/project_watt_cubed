//! The sky: a day/night clock, an atmosphere colour function, weather, and the
//! lighting edge into voxel shading — the systems the [skybox plan](crate)
//! collapses the feature list into.
//!
//! Only two things are inputs the world shares: [`SkyClock`] (the "when") and
//! [`Weather`]. Everything else derives. [`SkyEnv`] is the sole output edge into
//! the mesh pipeline. Drawing a real gradient dome, sun disc, stars, and clouds
//! is deferred behind [`Sky::draw`], which is a no-op until the engine grows a
//! background pass — the seams here do not change when it lands.
mod atmosphere;
mod clock;
mod env;
mod weather;

pub use atmosphere::Atmosphere;
pub use clock::{DayLength, SkyClock};
pub use env::SkyEnv;
pub use weather::{Precip, Weather};

use voxel_engine::{Color, Frame3D, SkyDesc, Vec3};

/// The warm horizon glow smeared toward the sun. It is strongest at low sun
/// (sunrise/sunset) and cools toward pale daylight as the sun climbs, so a
/// midday sky glows only faintly while dawn/dusk flare orange.
fn sun_tint(daylight: f32) -> Color {
    let warm = Color::rgb(240, 150, 70);
    let pale = Color::rgb(210, 205, 200);
    let t = daylight.clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t) as u8;
    Color::rgb(
        mix(warm.r, pale.r),
        mix(warm.g, pale.g),
        mix(warm.b, pale.b),
    )
}

/// Base exponential fog density in clear weather. Tuned so terrain fades into the
/// horizon near the edge of a typical view radius rather than cutting off.
const FOG_BASE: f32 = 0.0016;

/// The whole sky state, owned by the game.
#[derive(Default)]
pub struct Sky {
    pub clock: SkyClock,
    pub atmosphere: Atmosphere,
    pub weather: Weather,
    /// How long a full day/night cycle lasts, in real seconds.
    pub day_length: DayLength,
}

impl Sky {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the clock by one frame's `dt`.
    pub fn tick(&mut self, dt: f64) {
        self.clock.advance(dt, self.day_length);
    }

    /// The flat clear colour for [`Engine::begin_frame`](voxel_engine::Engine::begin_frame).
    pub fn clear(&self) -> Color {
        self.atmosphere.clear(self.clock.sun_dir())
    }

    /// This frame's contribution to voxel lighting.
    pub fn env(&self) -> SkyEnv {
        SkyEnv::resolve(&self.clock, &self.atmosphere, &self.weather)
    }

    /// Horizon fog colour and density (weather thickens it).
    pub fn fog(&self) -> (Color, f32) {
        (
            self.atmosphere.horizon(self.clock.sun_dir()),
            FOG_BASE + self.weather.fog_bonus(),
        )
    }

    /// Push the frame's sky lighting and fog into an active 3D scope. Call once
    /// per frame inside `begin_3d`, before or after world geometry.
    pub fn apply(&self, f: &mut Frame3D) {
        let env = self.env();
        let (fog_color, fog_density) = self.fog();
        f.set_sky_light(env.sun_light, env.ambient, fog_color, fog_density);
    }

    /// Draw the procedural sky: a zenith→horizon gradient, a horizon glow
    /// toward the sun, and a sun disc. The two anchor colours come straight from
    /// [`Atmosphere::radiance`] (the single source of truth); the engine's
    /// fullscreen pass interpolates and adds the high-frequency features.
    pub fn draw(&self, f: &mut Frame3D) {
        let sun = self.clock.sun_dir();
        let daylight = self.clock.daylight();
        f.set_sky(SkyDesc {
            sun_dir: sun,
            zenith: self.atmosphere.radiance(Vec3::Y, sun),
            horizon: self.atmosphere.horizon(sun),
            sun_tint: sun_tint(daylight),
            exposure: 0.6 + 0.4 * daylight,
            sun_angular_radius: 0.03,
        });
    }
}
