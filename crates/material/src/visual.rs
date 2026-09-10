//! Presentation: a smooth map from a configuration to a render descriptor. Nearby configurations look
//! alike because the colours are value noise over the resource lattice; no axis means a colour and no
//! table names one. A texture or material mod turns the descriptor into pixels or shader parameters.

use crate::configuration::Configuration;
use crate::element::D;
use crate::law::Law;
use crate::observe::{observe, response};

/// The render descriptor of a configuration (the engine's `MaterialDesc` carries the same bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Visual {
    /// Base colour.
    pub rgb: [u8; 3],
    /// Secondary colour (the pattern blends toward it).
    pub rgb2: [u8; 3],
    /// Pattern frequency: 0 for a uniform material, higher for configurations whose elements are
    /// spread out in resource space.
    pub frequency: u8,
    /// Grain: the contact response (stable materials are smooth).
    pub roughness: u8,
    /// Opacity: 255 opaque … 0 fully transparent.
    pub alpha: u8,
    /// Emissive strength 0..255.
    pub glow: u8,
}

/// The quantized key a descriptor is interned by (rgb 5-6-5, rgb2 5-6-5, frequency 3 bits,
/// roughness 3, alpha 4, glow 4 = 46 bits). Many configurations share one key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DescriptorKey(pub u64);

fn q565(c: [u8; 3]) -> u64 {
    ((c[0] as u64 >> 3) << 11) | ((c[1] as u64 >> 2) << 5) | (c[2] as u64 >> 3)
}

fn dq565(v: u64) -> [u8; 3] {
    let r = ((v >> 11) & 31) as u8;
    let g = ((v >> 5) & 63) as u8;
    let b = (v & 31) as u8;
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

impl Visual {
    /// Quantize into the intern key.
    pub fn quantize(&self) -> DescriptorKey {
        let k = q565(self.rgb)
            | (q565(self.rgb2) << 16)
            | (((self.frequency >> 5) as u64) << 32)
            | (((self.roughness >> 5) as u64) << 35)
            | (((self.alpha >> 4) as u64) << 38)
            | (((self.glow >> 4) as u64) << 42);
        DescriptorKey(k)
    }

    /// The representative descriptor of a key (the centre of its quantization cell).
    pub fn dequantize(key: DescriptorKey) -> Visual {
        let k = key.0;
        let f = ((k >> 32) & 7) as u8;
        let r = ((k >> 35) & 7) as u8;
        let a = ((k >> 38) & 15) as u8;
        let g = ((k >> 42) & 15) as u8;
        Visual {
            rgb: dq565(k & 0xffff),
            rgb2: dq565((k >> 16) & 0xffff),
            frequency: (f << 5) | (f << 2) | (f >> 1),
            roughness: (r << 5) | (r << 2) | (r >> 1),
            alpha: (a << 4) | a,
            glow: (g << 4) | g,
        }
    }
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
    // t²(3 − 2t) with t in Q8: (t*t*(768 - 2t)) >> 16
    (t * t * (768 - 2 * t)) >> 16
}

/// 4-D value noise over the lattice at Q8 coordinates (1/256 lattice units), cell size in lattice units.
/// Output 0..255. Multilinear over the 16 corners with smoothstep weights, all integer.
fn noise4(seed: u32, p: [u32; D], cell: u32, salt: u32) -> u32 {
    let cell_q8 = cell * 256;
    let mut base = [0u32; D];
    let mut w = [0u32; D];
    for i in 0..D {
        base[i] = p[i] / cell_q8;
        w[i] = smooth_q8((p[i] % cell_q8) * 256 / cell_q8);
    }
    // 16 corner values, then collapse one axis at a time (Q8 lerps).
    let mut vals = [0u32; 1 << D];
    for (c, v) in vals.iter_mut().enumerate() {
        let mut coords = [0u32; D];
        for i in 0..D {
            coords[i] = base[i] + ((c >> i) & 1) as u32;
        }
        *v = (hash32(seed, coords, salt) >> 24) * 256; // Q8 value 0..65280
    }
    // Collapse the highest axis first: pairs (c, c + n) differ in bit log2(n) = axis index.
    let mut n = 1 << D;
    for i in (0..D).rev() {
        n /= 2;
        for c in 0..n {
            let a = vals[c];
            let b = vals[c + n];
            vals[c] = (a * (256 - w[i]) + b * w[i]) / 256;
        }
    }
    vals[0] / 256
}

/// Cell size of the colour noise in lattice units: colours drift slowly across resource space.
const COLOUR_CELL: u32 = 32;

/// The presentation law V(C).
pub fn visual(law: &Law, c: &Configuration) -> Visual {
    let obs = observe(law, c);
    let Some(mean) = c.mean_q8() else {
        return Visual { rgb: [255; 3], rgb2: [255; 3], frequency: 0, roughness: 0, alpha: 0, glow: 0 };
    };
    let seed = law.visual_seed;
    let rgb = [
        noise4(seed, mean, COLOUR_CELL, 1) as u8,
        noise4(seed, mean, COLOUR_CELL, 2) as u8,
        noise4(seed, mean, COLOUR_CELL, 3) as u8,
    ];
    // Secondary colour: sample toward the element farthest from the mean (24 lattice units along
    // that direction), or a darker base when the configuration has no spread.
    let spread = c.spread_q8();
    let rgb2 = if spread == 0 {
        rgb.map(|v| (v as u32 * 205 / 256) as u8)
    } else {
        let far = c
            .elements()
            .iter()
            .max_by_key(|e| (0..D).map(|i| (e.0[i] as i32 * 256 - mean[i] as i32).unsigned_abs()).sum::<u32>())
            .copied()
            .expect("non-void");
        let mut dir = [0i64; D];
        let mut len = 0i64;
        for i in 0..D {
            dir[i] = far.0[i] as i64 * 256 - mean[i] as i64;
            len += dir[i].abs();
        }
        let len = len.max(1);
        let mut p = [0u32; D];
        for i in 0..D {
            let off = dir[i] * 24 * 256 / len;
            p[i] = (mean[i] as i64 + off).clamp(0, 255 * 256) as u32;
        }
        [
            noise4(seed, p, COLOUR_CELL, 1) as u8,
            noise4(seed, p, COLOUR_CELL, 2) as u8,
            noise4(seed, p, COLOUR_CELL, 3) as u8,
        ]
    };
    Visual {
        rgb,
        rgb2,
        frequency: (spread / 64).min(255) as u8,
        roughness: response(law, c, law.probes.contact),
        alpha: 255 - obs.transparency,
        glow: (obs.emission as u32 * 17) as u8,
    }
}
