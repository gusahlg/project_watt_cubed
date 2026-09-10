//! Operational readings of a configuration: the response of the material to the law's fixed probe
//! elements. These are cached observations of the physics (spec §3.3), never authored causes.

use crate::configuration::Configuration;
use crate::element::{Element, D};
use crate::kernel::raw_influence;
use crate::law::Law;

/// The acoustic class the audio director keys on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Acoustic {
    /// Nothing there.
    Void = 0,
    /// Reads as liquid.
    Liquid = 1,
    /// Soft solid.
    Soft = 2,
    /// Hard solid.
    Hard = 3,
}

/// What the world needs to know operationally about a configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Observation {
    /// Blocks movement.
    pub solid: bool,
    /// Flows and floats things (the flow response reached the law's threshold).
    pub liquid: bool,
    /// 0 opaque … 255 fully transparent (the light response).
    pub transparency: u8,
    /// Block-light level 0..15 (the glow response above threshold).
    pub emission: u8,
    /// 255 − contact response: high = stable under contact.
    pub hardness: u8,
    /// The friction response.
    pub friction: u8,
    /// The raw flow response (buoyancy strength for liquids).
    pub flow: u8,
    /// Sound class.
    pub acoustic: Acoustic,
}

impl Observation {
    /// What the void reads as: passable, transparent, silent.
    pub const AIR: Observation = Observation {
        solid: false,
        liquid: false,
        transparency: 255,
        emission: 0,
        hardness: 0,
        friction: 0,
        flow: 0,
        acoustic: Acoustic::Void,
    };
}

/// Per-element response magnitude that reads as 255 (the sum over axes of the largest knot response
/// after mixing is ~16 under law v0's band-limited curve); larger responses saturate.
const RESPONSE_FULL: i32 = 16;

/// Mean per-element response magnitude of `c` to `probe`, 0..255. Void → 0.
pub(crate) fn response(law: &Law, c: &Configuration, probe: Element) -> u8 {
    if c.is_void() {
        return 0;
    }
    let origin = Configuration::single(probe);
    let strength = law.probes.strength as i32;
    let mut total = 0i64;
    for b in c.elements() {
        if let Some(raw) = raw_influence(law, &origin, *b) {
            let mut mag = 0i32;
            for i in 0..D {
                mag += (raw[i] * strength / 256).unsigned_abs() as i32;
            }
            total += mag as i64;
        }
    }
    let mean = total / c.len() as i64;
    (mean * 255 / RESPONSE_FULL as i64).min(255) as u8
}

/// Take the standardized readings of `c` under `law`.
pub fn observe(law: &Law, c: &Configuration) -> Observation {
    if c.is_void() {
        return Observation::AIR;
    }
    let p = &law.probes;
    let contact = response(law, c, p.contact);
    let light = response(law, c, p.light);
    let flow = response(law, c, p.flow);
    let glow = response(law, c, p.glow);
    let friction = response(law, c, p.friction);
    let liquid = flow >= p.liquid_min;
    let transparency = if light > p.transparent_min {
        ((light - p.transparent_min) as u32 * 255 / (255 - p.transparent_min) as u32) as u8
    } else {
        0
    };
    let emission = if glow >= p.glow_min {
        (1 + (glow - p.glow_min) as u32 * 14 / (255 - p.glow_min) as u32).min(15) as u8
    } else {
        0
    };
    let hardness = 255 - contact;
    let acoustic = if liquid {
        Acoustic::Liquid
    } else if hardness < 96 {
        Acoustic::Soft
    } else {
        Acoustic::Hard
    };
    Observation {
        solid: !liquid,
        liquid,
        transparency,
        emission,
        hardness,
        friction,
        flow,
        acoustic,
    }
}
