//! Typed sky palette — the whole "what colour is the world's light" surface in
//! one exhaustively-matched table, replacing MakeUp's ~200 `#define` sprawl
//! with `Role × Anchor` enums.
//!
//! Three roles (what the colour is *for*) × three anchors (what time it
//! belongs to). Blending between anchors keys off **sun elevation**, not a
//! clock fraction: MakeUp's quadratic-in-worldTime mixers were an equilibrium
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

    /// Decode an 8-bit sRGB (author-space) colour into linear.
    ///
    /// The authoring boundary *in*: `from_srgb8(255, 255, 255) == linear(1, 1, 1)`.
    // TODO: switch to the generated 256-entry decode table.
    pub fn from_srgb8(r: u8, g: u8, b: u8) -> Rgb {
        Rgb(srgb_decode(r), srgb_decode(g), srgb_decode(b))
    }

    /// `0xRRGGBB` convenience over [`from_srgb8`](Rgb::from_srgb8).
    // TODO: becomes `const fn` once the decode table is a const array.
    pub fn from_srgb_hex(rgb: u32) -> Rgb {
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

    /// LEGACY look-compatibility exit — reproduces the pre-linear-invariant
    /// display pipeline, which quantized linear values without sRGB encoding.
    /// Every call site is a deliberate look-freeze.
    // PROVISIONAL(A): retire by switching call sites to to_srgb8 under
    // golden-diff once the harness lands (the switch is a real, global look
    // change that must be evaluated, not inherited).
    pub fn to_srgb8_legacy(self) -> Color {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
        Color::rgb(q(self.0), q(self.1), q(self.2))
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

    /// Decode a *display-space* float literal to linear via the sRGB EOTF,
    /// extended with pow(x, 2.4) above 1.0 so HDR channels decode monotonically
    /// instead of clamping.
    pub fn from_display(r: f32, g: f32, b: f32) -> Rgb {
        fn d(c: f32) -> f32 {
            if c <= 0.04045 {
                c / 12.92
            } else if c <= 1.0 {
                ((c + 0.055) / 1.055).powf(2.4)
            } else {
                c.powf(2.4)
            }
        }
        Rgb(d(r), d(g), d(b))
    }
}

/// sRGB EOTF: decode one 8-bit author-space channel to linear.
fn srgb_decode(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

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
    /// [-1, 1]). The blend shape is documented so the numbers have a "why":
    /// - `elev ≥ 0.35`: full Day (sun clear of horizon effects).
    /// - `elev ≈ 0`: full Sunset band (the eye's golden hour).
    /// - `elev ≤ -0.25`: full Night (civil-twilight-ish cutoff).
    /// Smoothsteps between, so the derivative is continuous at both joins.
    pub fn at(&self, role: Role, elev: f32) -> Rgb {
        let sunset = self.get(role, Anchor::Sunset);
        let day = self.get(role, Anchor::Day);
        let night = self.get(role, Anchor::Night);
        if elev >= 0.0 {
            sunset.lerp(day, smoothstep(0.05, 0.35, elev))
        } else {
            sunset.lerp(night, smoothstep(0.05, 0.25, -elev))
        }
    }

    /// Sun↔moon light-source mix in [0, 1]: 1 = sun is the light source,
    /// 0 = moon. Crosses 0.5 exactly at the horizon; the narrow band keeps
    /// the flip invisible inside the sunset colour wash. (MakeUp's
    /// `dayNightMix` re-derived onto elevation.)
    pub fn day_night_mix(elev: f32) -> f32 {
        smoothstep(-0.08, 0.08, elev)
    }
}

/// The atmosphere colour table with its literal matrix re-read as
/// display-space MakeUp constants and EOTF-decoded via [`Rgb::from_display`].
pub fn new_shoka_v2() -> Palette {
    Palette { colors: NEW_SHOKA.colors.map(|row| row.map(|c| Rgb::from_display(c.0, c.1, c.2))) }
}

/// Default palette: the engine's existing look, verbatim — `atmosphere.rs`
/// anchors for Zenith/Horizon (Sunset row synthesized from its SUNSET glow
/// constant) and `env.rs`'s warm/pale sun ramp for Light. Changing the
/// default look is a data edit here, nowhere else.
pub const CLASSIC: Palette = Palette::new([
    // Light: sunset warm → day pale → night faint blue moon
    [Rgb::linear(1.0, 0.72, 0.42), Rgb::linear(1.0, 0.98, 0.92), Rgb::linear(0.13, 0.14, 0.16)],
    // Zenith
    [Rgb::linear(0.30, 0.28, 0.35), Rgb::linear(0.28, 0.50, 0.88), Rgb::linear(0.02, 0.03, 0.09)],
    // Horizon
    [Rgb::linear(0.92, 0.46, 0.24), Rgb::linear(0.66, 0.80, 0.94), Rgb::linear(0.05, 0.07, 0.15)],
]);

/// MakeUp "New shoka" palette, converted from its sRGB-ish constants — an
/// alternative preset proving the table is data, not code.
pub const NEW_SHOKA: Palette = Palette::new([
    [Rgb::linear(1.0, 0.588, 0.3555), Rgb::linear(0.90, 0.84, 0.79), Rgb::linear(0.048, 0.052, 0.061)],
    [Rgb::linear(0.143, 0.244, 0.365), Rgb::linear(0.143, 0.244, 0.365), Rgb::linear(0.014, 0.019, 0.025)],
    [Rgb::linear(1.0, 0.648, 0.378), Rgb::linear(0.65, 0.91, 1.3), Rgb::linear(0.021, 0.031, 0.039)],
]);

/// Convenience: elevation from a sun direction (`sun_dir().y`).
pub fn elevation(sun: Vec3) -> f32 {
    sun.y
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
            assert!(close(CLASSIC.at(role, 0.9), CLASSIC.get(role, Anchor::Day)));
            assert!(close(CLASSIC.at(role, 0.0), CLASSIC.get(role, Anchor::Sunset)));
            assert!(close(CLASSIC.at(role, -0.9), CLASSIC.get(role, Anchor::Night)));
        }
    }

    #[test]
    fn blend_is_continuous_across_the_horizon() {
        // Approaching elev=0 from both sides converges to the Sunset anchor.
        let above = CLASSIC.at(Role::Horizon, 0.001);
        let below = CLASSIC.at(Role::Horizon, -0.001);
        let sunset = CLASSIC.get(Role::Horizon, Anchor::Sunset);
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
