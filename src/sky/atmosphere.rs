//! `Atmosphere` — the pure "what colour is the sky in direction `d`" function.
//!
//! It is deliberately a function of direction, not a renderer: today it is
//! sampled at just two directions (overhead for the frame clear, the horizon for
//! fog), but the exact same [`Atmosphere::radiance`] can later be evaluated
//! per-fragment for a real gradient dome with no change to this contract.
use voxel_engine::{Color, Vec3};

/// Linear RGB in `[0, 1]` — the working space for blending; converted to the
/// engine's 8-bit [`Color`] only at the boundary.
#[derive(Clone, Copy)]
struct Rgb(f32, f32, f32);

impl Rgb {
    const fn new(r: f32, g: f32, b: f32) -> Self {
        Rgb(r, g, b)
    }
    fn lerp(self, o: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        Rgb(
            self.0 + (o.0 - self.0) * t,
            self.1 + (o.1 - self.1) * t,
            self.2 + (o.2 - self.2) * t,
        )
    }
    fn color(self) -> Color {
        let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
        Color::rgb(c(self.0), c(self.1), c(self.2))
    }
}

/// Atmospheric parameters. `turbidity` thickens the horizon haze; `ground` is
/// the colour reflected up from below the horizon.
#[derive(Clone, Copy)]
pub struct Atmosphere {
    pub turbidity: f32,
    pub ground: Color,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self { turbidity: 0.2, ground: Color::rgb(46, 42, 38) }
    }
}

// Palette anchors (linear RGB). Kept as consts so the whole look lives in one place.
const DAY_ZENITH: Rgb = Rgb::new(0.28, 0.50, 0.88);
const DAY_HORIZON: Rgb = Rgb::new(0.66, 0.80, 0.94);
const NIGHT_ZENITH: Rgb = Rgb::new(0.02, 0.03, 0.09);
const NIGHT_HORIZON: Rgb = Rgb::new(0.05, 0.07, 0.15);
const SUNSET: Rgb = Rgb::new(0.92, 0.46, 0.24);

impl Atmosphere {
    /// Sky radiance looking along `view` with the sun at `sun` (both unit-ish).
    /// The single function every consumer calls; the flat clear and the fog
    /// colour are just this sampled overhead and at the horizon.
    pub fn radiance(&self, view: Vec3, sun: Vec3) -> Color {
        let daylight = smoothstep(-0.12, 0.18, sun.y);
        let up = view.y.clamp(0.0, 1.0);

        let day = DAY_HORIZON.lerp(DAY_ZENITH, up);
        let night = NIGHT_HORIZON.lerp(NIGHT_ZENITH, up);
        let mut sky = night.lerp(day, daylight);

        // Warm glow banded at the horizon while the sun is near it (sunrise/set).
        let sun_low = 1.0 - (sun.y.abs() / 0.3).min(1.0);
        let glow = (1.0 - up) * sun_low;
        sky = sky.lerp(SUNSET, glow * 0.6);

        // Turbidity washes the low sky toward a pale haze.
        let haze = (1.0 - up) * self.turbidity;
        sky = sky.lerp(Rgb::new(0.75, 0.78, 0.82), haze * daylight);

        sky.color()
    }

    /// The sky colour used for the frame clear — sampled part-way up so a flat
    /// fill reads as an average sky rather than the pale horizon or dark zenith.
    pub fn clear(&self, sun: Vec3) -> Color {
        self.radiance(Vec3::new(0.0, 0.45, 0.9).normalize(), sun)
    }

    /// The horizon colour terrain fog fades toward, so distant geometry melts
    /// seamlessly into the sky.
    pub fn horizon(&self, sun: Vec3) -> Color {
        self.radiance(Vec3::new(0.0, 0.02, 1.0).normalize(), sun)
    }
}

/// Hermite smoothstep, clamped outside `[edge0, edge1]`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn night_sky_is_dark_and_day_sky_is_bright() {
        let atm = Atmosphere::default();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let day = atm.radiance(up, Vec3::new(0.0, 1.0, 0.0));
        let night = atm.radiance(up, Vec3::new(0.0, -1.0, 0.0));
        assert!(day.b > night.b + 100, "day zenith far brighter than night");
    }
}
