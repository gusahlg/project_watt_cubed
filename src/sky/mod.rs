//! The sky: a day/night clock, an atmosphere colour function, weather, and the
//! lighting edge into voxel shading — the systems the [skybox plan](crate)
//! collapses the feature list into.
//!
//! Only two things are inputs the world shares: [`SkyClock`] (the "when") and
//! [`Weather`]. Everything else derives. The lighting edge into voxel shading is
//! [`crate::frame_snapshot::compose`] → the per-frame UBO; this module
//! owns [`Sky::clear`] (flat clear) and [`Sky::draw`] (the procedural
//! background pass).
mod atmosphere;
mod clock;
pub mod palette;
mod weather;

pub use atmosphere::Atmosphere;
pub use clock::{DayLength, SkyClock, SkyFrame};
pub use weather::{Precip, Weather};

use voxel_engine::{Frame3D, LinearRgb, SkyDesc};

use crate::sky::palette::Rgb;

/// The warm sun-disc tint. Authored as display-space sRGB literals, decoded to
/// linear, and handed to the engine boundary UNCHANGED (`to_linear`, no clamp) —
/// warm orange at low sun (sunrise/sunset), cooling toward pale as it climbs.
fn sun_tint(daylight: f32) -> LinearRgb {
    let t = daylight.clamp(0.0, 1.0);
    Rgb::from_srgb8(240, 150, 70)
        .lerp(Rgb::from_srgb8(210, 205, 200), t)
        .to_linear()
}

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

    /// Sample the clock once for every sun consumer this frame.
    pub fn frame(&self) -> SkyFrame {
        self.clock.frame()
    }

    /// The clock sample at a pinned day fraction (stripped profiles render
    /// fixed noon without mutating the authoritative clock).
    pub fn frame_at_day(&self, day: f64) -> SkyFrame {
        let mut clock = self.clock;
        clock.set_day(day);
        clock.frame()
    }

    /// The flat clear colour for [`Engine::begin_frame`](voxel_engine::Engine::begin_frame).
    pub fn clear(&self) -> LinearRgb {
        self.clear_at(self.frame())
    }

    /// [`clear`](Self::clear) against an already-sampled clock frame.
    pub fn clear_at(&self, frame: SkyFrame) -> LinearRgb {
        self.atmosphere.clear(frame.sun_dir)
    }

    /// Draw the procedural sky. Only sun geometry + disc tint cross here; the
    /// gradient/glow colours are read GPU-side from the shared per-frame UBO (the
    /// same linear source the terrain fog reads), so the sky and the fog
    /// it blends into cannot diverge.
    pub fn draw(&self, f: &mut Frame3D, frame: SkyFrame) {
        f.set_sky(SkyDesc {
            sun_dir: frame.sun_dir,
            sun_tint: sun_tint(frame.daylight),
            sun_angular_radius: 0.03,
        });
    }
}
