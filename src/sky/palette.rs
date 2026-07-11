//! Typed sky palette — the whole "what colour is the world's light" surface in
//! one exhaustively-matched table, replacing the reference shader's ~200
//! `#define` sprawl with `Role × Anchor` enums.
//!
//! Three roles (what the colour is *for*) × three anchors (what time it
//! belongs to). Blending between anchors keys off **sun elevation**, not a
//! clock fraction: the reference shader's quadratic-in-worldTime mixers were an equilibrium
//! of Minecraft's clock; elevation is the quantity the blend is really *about*,
//! and it stays correct if day length or the sun's arc ever changes.
use voxel_engine::{Color, Vec3};

/// A colour in **linear** RGB — the palette's working space and an invariant of
/// the type, not just a convention. Fields are private so a value can only be
/// built through the blessed constructors below, every one of which lands in
/// linear space:
///
/// - [`Rgb::linear`] — components already in linear space (the explicit ctor).
/// - [`Rgb::from_srgb8`] / [`Rgb::from_srgb_hex`] — decode author-space sRGB.
///
/// All the arithmetic here (`lerp`, `scale`, `luma`) is therefore linear by
/// construction. The space is **unclamped**: HDR palettes are legitimate
/// (the New shoka palette carries channels > 1.0). Quantisation and the clamp to displayable
/// range happen in exactly one place, [`Rgb::to_srgb8`], the sole exit to 8-bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(f32, f32, f32);

impl Rgb {
    /// Construct from components that are already in **linear** space.
    pub const fn linear(r: f32, g: f32, b: f32) -> Rgb {
        Rgb(r, g, b)
    }

    /// Decode an 8-bit sRGB (author-space) colour into linear, via the
    /// generated decode table — the same table the shaders `#include`, so the
    /// authoring boundary cannot drift from the GPU's.
    ///
    /// The authoring boundary *in*: `from_srgb8(255, 255, 255) == linear(1, 1, 1)`.
    pub const fn from_srgb8(r: u8, g: u8, b: u8) -> Rgb {
        let t = &voxel_engine::genconst::SRGB8_TO_LINEAR;
        Rgb(t[r as usize], t[g as usize], t[b as usize])
    }

    /// `0xRRGGBB` convenience over [`from_srgb8`](Rgb::from_srgb8).
    pub const fn from_srgb_hex(rgb: u32) -> Rgb {
        Rgb::from_srgb8(
            ((rgb >> 16) & 0xff) as u8,
            ((rgb >> 8) & 0xff) as u8,
            (rgb & 0xff) as u8,
        )
    }

    /// The **one** exit to 8-bit: clamp to displayable range, encode to sRGB,
    /// quantise. This is the only place HDR values are clamped and the only
    /// place a linear `Rgb` becomes a `Color`.
    // TODO: switch to the generated encode table.
    pub fn to_srgb8(self) -> Color {
        Color::rgb(
            srgb_encode(self.0),
            srgb_encode(self.1),
            srgb_encode(self.2),
        )
    }

    /// The SECOND blessed exit: hand the linear components to the engine
    /// boundary UNCHANGED — no quantisation, no clamp, no OETF. For HDR tints
    /// that keep compositing in linear light on the GPU (the sky disc), where
    /// `to_srgb8`'s clamp+encode would crush the value. `to_srgb8` stays the
    /// sole 8-bit display exit; this is the sole linear engine-boundary exit.
    pub fn to_linear(self) -> voxel_engine::LinearRgb {
        voxel_engine::LinearRgb([self.0, self.1, self.2])
    }

    /// Linear component accessors (for consumers outside this module).
    pub fn r(self) -> f32 {
        self.0
    }
    pub fn g(self) -> f32 {
        self.1
    }
    pub fn b(self) -> f32 {
        self.2
    }

    /// Rec. 709 relative luminance of the linear colour.
    pub fn luma(self) -> f32 {
        0.2126 * self.0 + 0.7152 * self.1 + 0.0722 * self.2
    }

    pub fn lerp(self, o: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        Rgb(
            self.0 + (o.0 - self.0) * t,
            self.1 + (o.1 - self.1) * t,
            self.2 + (o.2 - self.2) * t,
        )
    }
    pub fn scale(self, f: f32) -> Rgb {
        Rgb(self.0 * f, self.1 * f, self.2 * f)
    }

    /// Desaturate toward a rain tint by `strength` [0,1]: blend toward
    /// `rain · luma(self)` — the rain colour at this colour's own brightness — so
    /// a rainy sky greys out without changing overall exposure. (The rain
    /// rule; see [`RAIN_ZENITH`]/[`RAIN_HORIZON`].)
    pub fn rain_override(self, rain: Rgb, strength: f32) -> Rgb {
        self.lerp(rain.scale(self.luma()), strength)
    }
}

/// Rain sky overrides, imported as linear color. See [`Rgb::rain_override`].
pub const RAIN_ZENITH: Rgb = Rgb::linear(0.7, 0.85, 1.0);
pub const RAIN_HORIZON: Rgb = Rgb::linear(0.35, 0.425, 0.5);

/// sRGB OETF: clamp a linear channel to [0, 1], encode, and quantise to 8-bit
/// (round-to-nearest, so `from_srgb8`→`to_srgb8` is the identity on all 256
/// codes).
fn srgb_encode(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// A clamped Hermite smoothstep over one elevation band — the ONE shape every
/// elevation→scalar blend uses, so all the named instances below stay
/// comparable at a glance. A curve a shader must share migrates its edges
/// into the generated constants table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    pub edge0: f32,
    pub edge1: f32,
}

impl Curve {
    pub const fn new(edge0: f32, edge1: f32) -> Curve {
        Curve { edge0, edge1 }
    }

    pub fn eval(self, x: f32) -> f32 {
        let t = ((x - self.edge0) / (self.edge1 - self.edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Sunset→Day palette blend band: full Day once the sun clears horizon effects.
pub const DAY_BLEND: Curve = Curve::new(0.05, 0.35);
/// Sunset→Night palette blend band (evaluated on `-elev`): full Night past a
/// civil-twilight-ish cutoff.
pub const NIGHT_BLEND: Curve = Curve::new(0.05, 0.25);
/// Sunset-glow widening band: the sky sun-halo exponent lerps from a wide
/// golden-hour halo at low sun to the tight noon halo above this band. CPU-side
/// documentation/parity for the shader; the edges are mirrored into the generated
/// `GLOW_EDGE0`/`GLOW_EDGE1` constants (with `GLOW_POW_SUNSET`/`GLOW_POW_DAY`) that
/// `sky_radiance` consumes. Axis is sun elevation `sun_dir().y`.
pub const GLOW: Curve = Curve::new(0.0, 0.4);
/// Sun↔moon light-source mix band: crosses 0.5 exactly at the horizon; narrow
/// so the flip hides inside the sunset colour wash. (The `dayNightMix`
/// mixer re-derived onto elevation.)
pub const DAY_NIGHT_MIX: Curve = Curve::new(-0.08, 0.08);

/// What a palette colour is used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Direct sun/moon light — tints the sky-lit term of voxel shading.
    Light,
    /// Sky overhead — zenith of the gradient, ambient tint source.
    Zenith,
    /// Sky at the horizon — gradient base and the colour fog fades toward.
    Horizon,
}

/// Which time-of-day a colour anchors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Sunset,
    Day,
    Night,
}

/// A complete sky palette: one colour per `Role × Anchor`.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    colors: [[Rgb; 3]; 3], // [role][anchor]
}

impl Palette {
    pub const fn new(colors: [[Rgb; 3]; 3]) -> Self {
        Self { colors }
    }

    pub fn get(&self, role: Role, anchor: Anchor) -> Rgb {
        let r = match role {
            Role::Light => 0,
            Role::Zenith => 1,
            Role::Horizon => 2,
        };
        let a = match anchor {
            Anchor::Sunset => 0,
            Anchor::Day => 1,
            Anchor::Night => 2,
        };
        self.colors[r][a]
    }

    /// The palette colour for `role` at sun elevation `elev` (`sun_dir().y`,
    /// [-1, 1]): Sunset at the horizon, blended to Day above ([`DAY_BLEND`])
    /// and Night below ([`NIGHT_BLEND`]), derivative-continuous at both joins.
    pub fn at(&self, role: Role, elev: f32) -> Rgb {
        let sunset = self.get(role, Anchor::Sunset);
        let day = self.get(role, Anchor::Day);
        let night = self.get(role, Anchor::Night);
        if elev >= 0.0 {
            sunset.lerp(day, DAY_BLEND.eval(elev))
        } else {
            sunset.lerp(night, NIGHT_BLEND.eval(-elev))
        }
    }

    /// Sun↔moon light-source mix in [0, 1]: 1 = sun is the light source,
    /// 0 = moon. See [`DAY_NIGHT_MIX`].
    pub fn day_night_mix(elev: f32) -> f32 {
        DAY_NIGHT_MIX.eval(elev)
    }
}

/// Default palette: "New shoka", imported as LINEAR light (L10/L1 verdict: linear
/// import + sigmoid tonemap beat PBR-Neutral and EOTF-decode). Changing the default
/// look is a data edit here, nowhere else. Note the HDR horizon-day blue (1.3),
/// which only survives because the sky path now carries linear f32 end-to-end (L9).
pub const NEW_SHOKA: Palette = Palette::new([
    [Rgb::linear(1.0, 0.588, 0.3555), Rgb::linear(0.90, 0.84, 0.79), Rgb::linear(0.048, 0.052, 0.061)],
    [Rgb::linear(0.143, 0.244, 0.365), Rgb::linear(0.143, 0.244, 0.365), Rgb::linear(0.014, 0.019, 0.025)],
    [Rgb::linear(1.0, 0.648, 0.378), Rgb::linear(0.65, 0.91, 1.3), Rgb::linear(0.021, 0.031, 0.039)],
]);

/// Convenience: elevation from a sun direction (`sun_dir().y`).
pub fn elevation(sun: Vec3) -> f32 {
    sun.y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Rgb, b: Rgb) -> bool {
        // lerp(t=1) is a+(b-a)·1, not b bit-exactly — compare with an epsilon.
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5 && (a.2 - b.2).abs() < 1e-5
    }

    #[test]
    fn anchors_are_pure_at_their_elevations() {
        for role in [Role::Light, Role::Zenith, Role::Horizon] {
            assert!(close(NEW_SHOKA.at(role, 0.9), NEW_SHOKA.get(role, Anchor::Day)));
            assert!(close(NEW_SHOKA.at(role, 0.0), NEW_SHOKA.get(role, Anchor::Sunset)));
            assert!(close(NEW_SHOKA.at(role, -0.9), NEW_SHOKA.get(role, Anchor::Night)));
        }
    }

    #[test]
    fn blend_is_continuous_across_the_horizon() {
        // Approaching elev=0 from both sides converges to the Sunset anchor.
        let above = NEW_SHOKA.at(Role::Horizon, 0.001);
        let below = NEW_SHOKA.at(Role::Horizon, -0.001);
        let sunset = NEW_SHOKA.get(Role::Horizon, Anchor::Sunset);
        for (got, want) in [(above, sunset), (below, sunset)] {
            assert!((got.0 - want.0).abs() < 0.02, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn srgb_round_trips_on_all_gray_codes() {
        // from_srgb8 → to_srgb8 is the identity on every 8-bit code.
        for v in 0u8..=255 {
            let c = Rgb::from_srgb8(v, v, v).to_srgb8();
            assert_eq!((c.r, c.g, c.b), (v, v, v), "round-trip broke at {v}");
        }
    }

    #[test]
    fn white_srgb_decodes_to_unit_linear() {
        assert_eq!(Rgb::from_srgb8(255, 255, 255), Rgb::linear(1.0, 1.0, 1.0));
    }

    #[test]
    fn day_night_mix_flips_at_the_horizon() {
        assert!(Palette::day_night_mix(0.3) > 0.99);
        assert!(Palette::day_night_mix(-0.3) < 0.01);
        assert!((Palette::day_night_mix(0.0) - 0.5).abs() < 0.01);
    }
}
