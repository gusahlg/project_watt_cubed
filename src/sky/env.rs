//! `SkyEnv` — the one edge from the sky into voxel shading. Resolves the clock,
//! atmosphere, and weather into the two colours the mesh pipeline consumes:
//! `sun_light` (tints the sky-lit term) and `ambient` (the unlit floor).
use voxel_engine::Color;

use super::atmosphere::Atmosphere;
use super::clock::SkyClock;
use super::weather::Weather;

/// The sky's contribution to voxel lighting for one frame.
#[derive(Clone, Copy)]
pub struct SkyEnv {
    pub sun_light: Color,
    pub ambient: Color,
}

impl SkyEnv {
    /// Derive the frame's lighting from the current sky state.
    pub fn resolve(clock: &SkyClock, atm: &Atmosphere, weather: &Weather) -> Self {
        let d = clock.daylight();
        let elev = clock.sun_elevation();

        // Sun colour warms toward the horizon and whitens overhead; brightness
        // scales with daylight. A faint blue moon term keeps night navigable.
        let warm = smoothstep(0.0, 0.35, elev);
        let sun = [
            1.0,
            lerp(0.72, 0.98, warm),
            lerp(0.42, 0.92, warm),
        ];
        // Overcast mutes and cools direct sun.
        let overcast = 1.0 - weather.coverage * 0.5;
        let moon = 0.16 * (1.0 - d);
        let sun_light = rgb(
            sun[0] * d * overcast + moon * 0.8,
            sun[1] * d * overcast + moon * 0.9,
            sun[2] * d * overcast + moon,
        );

        // Ambient: a dim, sky-tinted floor so shadowed faces and caves are not
        // pure black. Kept low so it reads as fill, not a second sun.
        let sky = atm.clear(clock.sun_dir());
        let amt = 0.10 + 0.12 * d;
        let ambient = Color::rgb(
            scale(sky.r, amt),
            scale(sky.g, amt),
            scale(sky.b, amt),
        );

        Self { sun_light, ambient }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb(r: f32, g: f32, b: f32) -> Color {
    let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    Color::rgb(c(r), c(g), c(b))
}

fn scale(channel: u8, factor: f32) -> u8 {
    (channel as f32 * factor).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn night_sun_light_is_much_dimmer_than_day() {
        let atm = Atmosphere::default();
        let w = Weather::default();
        let mut c = SkyClock::default();
        c.set_day(0.5); // noon
        let day = SkyEnv::resolve(&c, &atm, &w);
        c.set_day(0.0); // midnight
        let night = SkyEnv::resolve(&c, &atm, &w);
        assert!(day.sun_light.r > night.sun_light.r + 100);
    }
}
