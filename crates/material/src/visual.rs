//! Presentation: a smooth map from a configuration to a render descriptor. Colour is periodic noise
//! over the resource torus (no axis means a colour); a configuration takes the colour at its
//! centroid, so a block that gains or loses constituents visibly drifts toward or away from them,
//! and its least-held occurrence lends the accent. Texture mods turn the descriptor (and the
//! configuration itself) into pixels.

use crate::element::{Element, D};
use crate::kernel::Block;
use crate::law::Law;
use crate::observe::{observe, Observation};

/// The render descriptor of a configuration.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Visual {
    /// Base colour: the colour at the configuration's centroid on the lattice torus.
    pub rgb: [u8; 3],
    /// Accent colour: the least-held occurrence (the impurity the eye should find).
    pub rgb2: [u8; 3],
    /// Pattern frequency: grows with the number of distinct elements.
    pub frequency: u8,
    /// Grain: weakly held matter is rough.
    pub roughness: u8,
    /// Opacity: 255 opaque … 0 fully transparent.
    pub alpha: u8,
    /// Emissive strength 0..255.
    pub glow: u8,
}

impl Visual {
    /// What the void looks like (layer 0 is white for the engine's immediate geometry).
    pub const VOID: Visual = Visual { rgb: [255; 3], rgb2: [255; 3], frequency: 0, roughness: 0, alpha: 0, glow: 0 };
}

/// 32-bit integer mixer (murmur3 finalizer over a seeded combination of the inputs).
fn hash32(seed: u32, coords: [u32; D], salt: u32) -> u32 {
    let mut h = seed ^ salt.wrapping_mul(0x9E37_79B1);
    for (i, c) in coords.iter().enumerate() {
        h ^= c.wrapping_mul([0x85EB_CA77, 0xC2B2_AE3D, 0x27D4_EB2F, 0x1656_67B1][i % 4]);
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    }
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// Smoothstep of a 0..256 fraction, in 0..256.
fn smooth_q8(t: u32) -> u32 {
    (t * t * (768 - 2 * t)) >> 16
}

/// Lattice cells per axis of the colour noise: 256 / CELLS units per cell, periodic like the law.
const CELLS: u32 = 8;
const CELL: u32 = 256 / CELLS;

/// 4-D periodic value noise at a lattice position in 1/256 units (`p[i] / 256` is the coordinate),
/// 0..255. Multilinear over 16 corners with smoothstep weights, all integer.
fn noise4(seed: u32, p: [u32; D], salt: u32) -> u32 {
    let cell_q8 = CELL * 256;
    let mut base = [0u32; D];
    let mut w = [0u32; D];
    for i in 0..D {
        let q = p[i] % (256 * 256);
        base[i] = q / cell_q8;
        w[i] = smooth_q8((q % cell_q8) * 256 / cell_q8);
    }
    let mut vals = [0u32; 1 << D];
    for (c, v) in vals.iter_mut().enumerate() {
        let mut coords = [0u32; D];
        for i in 0..D {
            coords[i] = (base[i] + ((c >> i) & 1) as u32) % CELLS;
        }
        *v = (hash32(seed, coords, salt) >> 24) * 256;
    }
    let mut n = 1 << D;
    for i in (0..D).rev() {
        n /= 2;
        for c in 0..n {
            vals[c] = (vals[c] * (256 - w[i]) + vals[c + n] * w[i]) / 256;
        }
    }
    vals[0] / 256
}

/// The centroid of a multiset of elements on the torus, in 1/256 lattice units: offsets from the
/// smallest element taken the short way round, averaged per axis. Deterministic and smooth: one
/// added or removed occurrence moves it by a fraction of the way to that occurrence.
pub fn centroid_q8(elements: &[Element]) -> Option<[u32; D]> {
    let reference = *elements.iter().min()?;
    let n = elements.len() as i64;
    let mut out = [0u32; D];
    for (i, o) in out.iter_mut().enumerate() {
        let sum: i64 = elements.iter().map(|e| e.0[i].wrapping_sub(reference.0[i]) as i8 as i64).sum();
        let mean_q8 = reference.0[i] as i64 * 256 + sum * 256 / n;
        *o = mean_q8.rem_euclid(256 * 256) as u32;
    }
    Some(out)
}

/// Integer HSV → RGB: hue `0..1536` (six sectors of 256), saturation and value `0..=255`.
fn hsv(h: u32, s: u32, v: u32) -> [u8; 3] {
    let (sector, f) = ((h / 256) % 6, h % 256);
    let p = v * (255 - s) / 255;
    let q = v * (255 - s * f / 255) / 255;
    let t = v * (255 - s * (255 - f) / 255) / 255;
    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r as u8, g as u8, b as u8]
}

/// The colour at a lattice position (1/256 units): hue, saturation and value from three periodic
/// noise channels, so nearby positions have related colours and the whole wheel is reachable.
pub fn colour_at(law: &Law, p: [u32; D]) -> [u8; 3] {
    let n = |salt| noise4(law.visual_seed, p, salt);
    // Value noise clusters around the middle: folding the hue channel round the wheel and
    // stretching saturation and value (clamped) makes pale, deep, grey and vivid all common.
    let spread = |v: u32| (v as i32 * 2 - 128).clamp(0, 255) as u32;
    let hue = (n(1) * 12) % 1536;
    let sat = 8 + spread(n(2)) * 236 / 255;
    let val = 40 + spread(n(3)) * 215 / 255;
    hsv(hue, sat, val)
}

/// The colour of one element.
pub fn element_colour(law: &Law, e: Element) -> [u8; 3] {
    colour_at(law, e.0.map(|c| c as u32 * 256))
}

/// The presentation law V(C), from the configuration's kernel record.
pub fn visual(law: &Law, block: &Block) -> Visual {
    visual_with(law, block, &observe(law, block))
}

/// [`visual`] when the observation is already at hand.
pub fn visual_with(law: &Law, block: &Block, obs: &Observation) -> Visual {
    if block.is_empty() {
        return Visual::VOID;
    }
    let mut accent = (i32::MAX, Element::default());
    let mut distinct = 0u32;
    let mut sorted = [Element::default(); crate::configuration::CAPACITY];
    sorted[..block.len()].copy_from_slice(block.elements());
    sorted[..block.len()].sort_unstable();
    for (i, (&e, &h)) in block.elements().iter().zip(block.holding()).enumerate() {
        if (h, e) < accent {
            accent = (h, e);
        }
        if i == 0 || sorted[i] != sorted[i - 1] {
            distinct += 1;
        }
    }
    let rgb = colour_at(law, centroid_q8(block.elements()).expect("non-empty"));
    // The accent stays in the material's family: its own colour, darkened, with a third of the
    // least-held occurrence's hue mixed in (an impurity reads as a tint, not a clash).
    let dark = rgb.map(|v| v as u32 * 170 / 256);
    let rgb2 = if distinct == 1 {
        dark.map(|v| v as u8)
    } else {
        let a = element_colour(law, accent.1);
        std::array::from_fn(|k| ((dark[k] * 2 + a[k] as u32) / 3) as u8)
    };
    Visual {
        rgb,
        rgb2,
        frequency: (distinct * 24).min(255) as u8,
        roughness: 255 - obs.hardness,
        alpha: 255 - obs.transparency,
        glow: (obs.emission as u32 * 17) as u8,
    }
}
