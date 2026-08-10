//! `Atmosphere` — the atmospheric parameters (turbidity + the time-of-day
//! colour table) the sky reads.
//!
//! There is no CPU sky formula: the shader's `sky_radiance` (common.slang) is
//! the single source of truth for the sky background, the distance fog, and the
//! water reflection. Time-of-day colour comes from the typed [`Palette`]
//! (`Role × Anchor`, blended on sun elevation); the UBO carries the linear
//! palette lanes and the GPU evaluates the gradient, so the sky, the fog, and
//! (via [`compose`](crate::frame_snapshot::compose)) the voxel lighting can
//! never drift apart.
use voxel_engine::{LinearRgb, Vec3};

use super::palette::{Palette, Role, NEW_SHOKA};

/// Atmospheric parameters. `turbidity` scales the shader's horizon sun halo
/// (`zenith.w` lane).
#[derive(Clone, Copy)]
pub struct Atmosphere {
    pub turbidity: f32,
    /// The time-of-day colour table (data, not code — swap for a new look).
    pub palette: Palette,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self { turbidity: 0.2, palette: NEW_SHOKA }
    }
}

impl Atmosphere {
    /// The flat frame-clear colour: the horizon palette lane at the sun's
    /// elevation. This is only the background the sky pass repaints; it matches
    /// the shader's `ray.y <= 0` limit (fog/sky clamp to the horizon colour) so
    /// the seam between the clear and the drawn sky is the same colour. Handed
    /// to the engine boundary UNCHANGED (`to_linear`, no clamp/quantise); the
    /// tonemap owns the OETF.
    pub fn clear(&self, sun: Vec3) -> LinearRgb {
        self.palette.at(Role::Horizon, sun.y).to_linear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_is_dark_at_night_and_bright_by_day() {
        let atm = Atmosphere::default();
        let day = atm.clear(Vec3::new(0.0, 1.0, 0.0));
        let night = atm.clear(Vec3::new(0.0, -1.0, 0.0));
        // Day horizon blue far exceeds night's.
        assert!(day.0[2] > night.0[2] + 0.05, "day horizon far brighter than night");
    }

    #[test]
    fn clear_warms_at_sunset() {
        let atm = Atmosphere::default();
        // Sun on the horizon → the Sunset anchor dominates: red >> blue.
        let sunset = atm.clear(Vec3::new(1.0, 0.0, 0.0).normalize());
        assert!(sunset.0[0] > sunset.0[2] + 0.03, "sunset horizon reads warm: {sunset:?}");
    }
}
