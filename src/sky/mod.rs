//! The sky: a day/night clock, an atmosphere colour function, weather, and the
//! lighting edge into voxel shading — the systems the [skybox plan](crate)
//! collapses the feature list into.
//!
//! Only two things are inputs the world shares: [`SkyClock`] (the "when") and
//! [`Weather`]. Everything else derives. The lighting edge into voxel shading is
//! [`crate::frame_snapshot::compose`] → the per-frame UBO; this module
//! owns [`Sky::clear_at`] (flat clear) and [`Sky::draw`] (the procedural
//! background pass).
mod atmosphere;
mod bodies;
mod clock;
pub mod palette;
mod rocks;
mod weather;

pub use atmosphere::Atmosphere;
pub use clock::{DayLength, SkyClock, SkyFrame};
#[cfg(test)]
pub use weather::Precip;
pub use weather::Weather;

use voxel_engine::{DVec3, Frame3D, LinearRgb, SkyDesc, Vec3};

use crate::sky::palette::Rgb;
use crate::world::generation::TerrainGenerator;

/// Half-extent, in blocks, of a chunk view of `view_chunks` rings. The eye sits
/// inside its own chunk, so one extra chunk covers the far block of the ring.
pub(crate) fn chunk_view_blocks(view_chunks: i32) -> f64 {
    rocks::chunk_view_blocks(view_chunks)
}

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
    /// Reused list of planets, moons and the home cube for the sky pass.
    far: bodies::FarBodies,
    /// Reused distant-asteroid boxes.
    rocks: rocks::DistantRocks,
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
    #[cfg(test)]
    pub fn frame(&self) -> SkyFrame {
        self.clock.frame(Vec3::Y)
    }

    /// The clock sample at a pinned day fraction against local `up` (stripped
    /// profiles render fixed noon without mutating the authoritative clock).
    pub fn frame_at_day(&self, day: f64, up: Vec3) -> SkyFrame {
        let mut clock = self.clock;
        clock.set_day(day);
        clock.frame(up)
    }

    /// Flat clear colour against an already-sampled clock frame and local up.
    pub fn clear_at(&self, frame: SkyFrame, up: Vec3) -> LinearRgb {
        self.atmosphere.clear(frame.sun_dir, up)
    }

    /// Sky-pass descriptor for `frame`. Same value `draw` would push.
    pub fn desc(&self, frame: SkyFrame) -> SkyDesc {
        SkyDesc {
            sun_dir: frame.sun_dir,
            sun_tint: sun_tint(frame.daylight),
            sun_angular_radius: 0.03,
        }
    }

    /// Draw the procedural sky, the far-body impostors and distant asteroids.
    /// Only sun geometry + disc tint cross here; the gradient/glow colours are
    /// read GPU-side from the shared per-frame UBO (the same linear source the
    /// terrain fog reads), so the sky and the fog it blends into cannot diverge.
    /// The engine clears its draw lists every frame, so all of them are pushed
    /// every frame. `view_blocks` is the chunk view's reach: rocks inside it are
    /// voxels, and [`chunk_view_blocks`] turns a render distance into one.
    pub fn draw(
        &mut self,
        f: &mut Frame3D,
        frame: SkyFrame,
        eye: DVec3,
        generator: &dyn TerrainGenerator,
        view_blocks: f64,
    ) {
        f.set_sky(self.desc(frame));
        f.set_far_bodies(self.far.update(generator, eye));
        self.rocks.draw(f, generator, eye, view_blocks);
    }
}
