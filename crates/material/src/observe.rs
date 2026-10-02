//! Operational readings of a configuration, taken with the law's own quantity: the fit function.
//! Hardness is how strongly the occurrences hold each other (the internal support the law caches);
//! clarity, glow and grip are how strongly the configuration attracts the law's probe elements. These
//! are cached observations of the physics, never authored causes.

use crate::kernel::{fit_raw, Block, QUANTUM};
use crate::law::Law;
use crate::element::Element;

/// The acoustic class the audio director keys on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Acoustic {
    /// Nothing there.
    Void = 0,
    /// Weakly held matter.
    Soft = 2,
    /// Strongly held matter.
    Hard = 3,
}

/// What the world needs to know operationally about a configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Observation {
    /// Blocks movement (every non-empty configuration).
    pub solid: bool,
    /// 0 opaque … 255 fully transparent (attraction to the light probe).
    pub transparency: u8,
    /// Block-light level 0..15 (attraction to the glow probe).
    pub emission: u8,
    /// How firmly the occurrences hold each other, 0..255 (mining time and impact read this).
    pub hardness: u8,
    /// Grip, 0..255 (attraction to the grip probe).
    pub friction: u8,
    /// Mean internal pair fit in 1/256 of `QUANTUM` (negative: the mixture wants to come apart).
    pub cohesion: i16,
    /// Sound class.
    pub acoustic: Acoustic,
}

impl Observation {
    /// What the void reads as: passable, transparent, silent.
    pub const AIR: Observation = Observation {
        solid: false,
        transparency: 255,
        emission: 0,
        hardness: 0,
        friction: 0,
        cohesion: 0,
        acoustic: Acoustic::Void,
    };
}

/// Mean fit of `probe` with the block's occurrences, in 1/256 of `QUANTUM` (−2048 ..= 1152).
pub fn probe_response(block: &Block, probe: Element) -> i32 {
    let n = block.len() as i64;
    if n == 0 {
        return 0;
    }
    let sum: i64 = block.elements().iter().map(|&e| i64::from(fit_raw(probe, e))).sum();
    (sum * 256 / (n * i64::from(QUANTUM))) as i32
}

/// Mean internal pair fit in 1/256 of `QUANTUM`; 0 for fewer than two occurrences.
pub fn cohesion(block: &Block) -> i32 {
    let n = block.len() as i64;
    if n < 2 {
        return 0;
    }
    let pairs = n * (n - 1) / 2;
    (block.internal_fit() * 256 / (pairs * i64::from(QUANTUM))) as i32
}

/// Probe response (Q8 of `QUANTUM`) where clarity begins, and the span over which it saturates.
const CLEAR_FROM: i32 = 2 * 256;
const CLEAR_SPAN: i32 = 5 * 128;
/// Probe response where glow begins (1.75 Q), and its span to level 15 (1.5 Q).
const GLOW_FROM: i32 = 7 * 64;
const GLOW_SPAN: i32 = 6 * 64;

/// Take the standardized readings of a configuration under `law`.
pub fn observe(law: &Law, block: &Block) -> Observation {
    if block.is_empty() {
        return Observation::AIR;
    }
    let n = block.len() as i32;
    let coh = cohesion(block);
    // A lone occurrence has no companions: it reads as middling. Mixtures read by how well they hold
    // together, and amount adds a little bulk.
    let hardness = (110 + coh * 40 / 256 + 3 * (n - 1)).clamp(1, 255) as u8;
    let light = probe_response(block, law.probes.light);
    let transparency = ((light - CLEAR_FROM) * 255 / CLEAR_SPAN).clamp(0, 255) as u8;
    let glow = probe_response(block, law.probes.glow);
    let emission = if glow > GLOW_FROM { (1 + (glow - GLOW_FROM) * 14 / GLOW_SPAN).min(15) as u8 } else { 0 };
    let grip = probe_response(block, law.probes.grip);
    let friction = (128 + grip * 24 / 256).clamp(0, 255) as u8;
    Observation {
        solid: true,
        transparency,
        emission,
        hardness,
        friction,
        cohesion: coh.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        acoustic: if hardness < 96 { Acoustic::Soft } else { Acoustic::Hard },
    }
}
