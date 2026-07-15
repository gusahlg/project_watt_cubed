//! `SkyClock` — the single source of "when". Everything visual about the sky
//! derives from one wrapped fraction of the day; there are no other time inputs.
use std::f64::consts::TAU;

use voxel_engine::Vec3;

/// Length of a full day/night cycle, in real seconds. A [`Settings`] value, not
/// sky state — it controls how fast [`SkyClock`] advances, nothing more.
///
/// [`Settings`]: crate::settings::Settings
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DayLength(pub f64);

impl DayLength {
    /// Clamp to a sane range: fast enough to demo, never zero (which would stall
    /// or divide-by-zero the advance).
    pub fn clamped(secs: f64) -> Self {
        Self(secs.clamp(10.0, 86_400.0))
    }
}

impl Default for DayLength {
    fn default() -> Self {
        Self(600.0) // ten-minute day out of the box
    }
}

/// Fractional time of day in `[0, 1)`: `0.0` = midnight, `0.25` = sunrise,
/// `0.5` = noon, `0.75` = sunset. The one degree of freedom the whole sky reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyClock {
    day: f64,
}

/// The clock-derived values consumed together by a rendered frame. Sampling
/// them as one unit prevents lighting, clear colour, and sky geometry from
/// independently repeating the same trigonometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyFrame {
    pub sun_dir: Vec3,
    pub elevation: f32,
    pub daylight: f32,
}

impl Default for SkyClock {
    fn default() -> Self {
        Self { day: 0.3 } // start a little after sunrise
    }
}

impl SkyClock {
    /// Advance by `dt` seconds, wrapping at the end of the day.
    pub fn advance(&mut self, dt: f64, len: DayLength) {
        self.day = (self.day + dt / len.0).rem_euclid(1.0);
    }

    /// The current fraction in `[0, 1)`.
    pub fn day(&self) -> f64 {
        self.day
    }

    /// Set the fraction directly (from a `/time` command or a network sync),
    /// wrapping into range so callers never have to.
    pub fn set_day(&mut self, day: f64) {
        self.day = day.rem_euclid(1.0);
    }

    fn direction(&self) -> Vec3 {
        let a = TAU * (self.day - 0.25); // 0 at sunrise, π/2 at noon
        Vec3::new(a.cos() as f32, a.sin() as f32, 0.2).normalize()
    }

    /// All clock-derived render values with one sun-direction evaluation.
    pub fn frame(&self) -> SkyFrame {
        let sun_dir = self.direction();
        let elevation = sun_dir.y;
        SkyFrame {
            sun_dir,
            elevation,
            daylight: smoothstep(-0.12, 0.18, elevation),
        }
    }

    /// Unit direction toward the sun. Rises in the east (`+x`), peaks overhead at
    /// noon, sets in the west; a small `z` tilt keeps it off a perfect great
    /// circle so the arc reads as a path rather than a line.
    pub fn sun_dir(&self) -> Vec3 {
        self.direction()
    }

    /// Sun elevation above the horizon, `[-1, 1]` (`sun_dir().y`). The single
    /// scalar the atmosphere and lighting blend day↔night on.
    pub fn sun_elevation(&self) -> f32 {
        self.sun_dir().y
    }

    /// Daylight amount in `[0, 1]`: 0 through the night, 1 in full day, with a
    /// smooth twilight either side of the horizon crossing.
    pub fn daylight(&self) -> f32 {
        self.frame().daylight
    }
}

/// Hermite smoothstep from `edge0`→`edge1`, clamped outside the range.
pub(crate) fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_wraps_and_scales_by_day_length() {
        let mut c = SkyClock { day: 0.9 };
        c.advance(60.0, DayLength(600.0)); // +0.1 of a day
        assert!((c.day() - 0.0).abs() < 1e-9, "0.9 + 0.1 wraps to 0.0: {}", c.day());
    }

    #[test]
    fn sun_is_up_at_noon_and_down_at_midnight() {
        let noon = SkyClock { day: 0.5 };
        let midnight = SkyClock { day: 0.0 };
        assert!(noon.sun_elevation() > 0.9, "noon sun overhead");
        assert!(midnight.sun_elevation() < -0.9, "midnight sun below");
        assert!(noon.daylight() > 0.99 && midnight.daylight() < 0.01);
    }

    #[test]
    fn frame_sample_matches_individual_clock_lanes() {
        let clock = SkyClock { day: 0.37 };
        let frame = clock.frame();
        assert_eq!(frame.sun_dir, clock.sun_dir());
        assert_eq!(frame.elevation, clock.sun_elevation());
        assert_eq!(frame.daylight, clock.daylight());
    }
}
