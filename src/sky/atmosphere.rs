//! `Atmosphere` — the pure "what colour is the sky in direction `d`" function.
//!
//! It is deliberately a function of direction, not a renderer: today it is
//! sampled at just two directions (overhead for the frame clear, the horizon for
//! fog), but the exact same [`Atmosphere::radiance`] can later be evaluated
//! per-fragment for a real gradient dome with no change to this contract.
//!
//! Time-of-day colour now comes from the typed [`Palette`] (`Role × Anchor`,
//! blended on sun elevation) instead of local day/night consts — one data
//! table drives the sky, the fog, and (via
//! [`compose`](crate::frame_snapshot::compose)) the voxel lighting, so they
//! can never drift apart.
use voxel_engine::{Color, Vec3};

use super::palette::{Palette, Rgb, Role, CLASSIC};

/// Atmospheric parameters. `turbidity` thickens the horizon haze; `ground` is
/// the colour reflected up from below the horizon.
#[derive(Clone, Copy)]
pub struct Atmosphere {
    pub turbidity: f32,
    pub ground: Color,
    /// The time-of-day colour table (data, not code — swap for a new look).
    pub palette: Palette,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self { turbidity: 0.2, ground: Color::rgb(46, 42, 38), palette: CLASSIC }
    }
}

impl Atmosphere {
    /// Sky radiance looking along `view` with the sun at `sun` (both unit-ish),
    /// as a display `Color` — used only for the FLAT frame clear (background
    /// pixels the sky pass repaints anyway). The `to_srgb8_legacy` exit is a
    /// deliberate U31a look-freeze: it quantises the linear value to UNORM /255
    /// with no OETF, matching the clear's linear-consistent path; the LIVE sky
    /// and fog read the linear palette straight from the UBO, not this.
    pub fn radiance(&self, view: Vec3, sun: Vec3) -> Color {
        self.radiance_rgb(view, sun).to_srgb8_legacy()
    }

    /// Linear-RGB radiance — the palette-space form, for consumers that keep
    /// compositing (the palette table stays the single source of truth).
    pub fn radiance_rgb(&self, view: Vec3, sun: Vec3) -> Rgb {
        let elev = sun.y;
        let up = view.y.clamp(0.0, 1.0);

        // Palette handles the sunset→day→night time blend per role; this
        // function only does geometry: elevation gradient + directional haze.
        let horizon = self.palette.at(Role::Horizon, elev);
        let zenith = self.palette.at(Role::Zenith, elev);
        let mut sky = horizon.lerp(zenith, up);

        // Turbidity washes the low sky toward a pale haze, daylight only.
        let daylight = smoothstep(-0.12, 0.18, elev);
        let haze = (1.0 - up) * self.turbidity;
        sky = sky.lerp(Rgb::linear(0.75, 0.78, 0.82), haze * daylight);

        sky
    }

    /// The sky colour used for the frame clear — sampled part-way up so a flat
    /// fill reads as an average sky rather than the pale horizon or dark zenith.
    pub fn clear(&self, sun: Vec3) -> Color {
        self.radiance(Vec3::new(0.0, 0.45, 0.9).normalize(), sun)
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

    #[test]
    fn horizon_warms_at_sunset() {
        let atm = Atmosphere::default();
        let toward_horizon = Vec3::new(0.0, 0.02, 1.0).normalize();
        // Sun exactly on the horizon → the Sunset anchor dominates: red >> blue.
        let sunset = atm.radiance(toward_horizon, Vec3::new(1.0, 0.0, 0.0).normalize());
        assert!(sunset.r > sunset.b + 60, "sunset horizon reads warm: {sunset:?}");
    }
}
