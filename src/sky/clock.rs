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

/// One clock sample every sun consumer shares: direction, elevation, and
/// daylight from a single trig evaluation, instead of each consumer
/// re-deriving them per frame. Cached by the game keyed on the day value, so
/// a frozen clock (day/night off, stripped profiles) performs no
/// steady-frame sun trigonometry at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyFrame {
    /// Unit direction toward the sun (see [`SkyClock::sun_dir`]).
    pub sun_dir: Vec3,
    /// Sun elevation above the horizon, `[-1, 1]` (`sun_dir.y`).
    pub elevation: f32,
    /// Daylight amount in `[0, 1]` with smooth twilight (see [`SkyClock::daylight`]).
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

    /// Unit direction toward the sun. Rises in the east (`+x`), peaks overhead at
    /// noon, sets in the west; a small `z` tilt keeps it off a perfect great
    /// circle so the arc reads as a path rather than a line.
    pub fn sun_dir(&self) -> Vec3 {
        let a = TAU * (self.day - 0.25); // 0 at sunrise, π/2 at noon
        Vec3::new(a.cos() as f32, a.sin() as f32, 0.2).normalize()
    }

    /// Sun elevation above the horizon, `[-1, 1]` (`sun_dir().y`). The single
    /// scalar the atmosphere and lighting blend day↔night on.
    #[cfg(test)]
    pub fn sun_elevation(&self) -> f32 {
        self.sun_dir().y
    }

    /// Daylight amount in `[0, 1]`: 0 through the night, 1 in full day, with a
    /// smooth twilight either side of the horizon crossing.
    #[cfg(test)]
    pub fn daylight(&self) -> f32 {
        smoothstep(-0.12, 0.18, self.sun_elevation())
    }

    /// Sample direction, elevation, and daylight once — the per-frame form
    /// every consumer shares (lighting compose, clear colour, sky geometry).
    pub fn frame(&self) -> SkyFrame {
        let sun_dir = self.sun_dir();
        let elevation = sun_dir.y;
        SkyFrame { sun_dir, elevation, daylight: smoothstep(-0.12, 0.18, elevation) }
    }
}

/// Hermite smoothstep from `edge0`→`edge1`, clamped outside the range.
pub(crate) fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    crate::math::smooth_between(edge0, edge1, x)
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
}
