//! Neural textures: every configuration paints its own 32×32 layer with a CPPN (a compositional
//! pattern-producing network — the small generative neural net behind a long line of "AI art"
//! pattern work). There is no training data and no table of looks: the network is *grown from the
//! configuration*. Each distinct element contributes one first-layer neuron whose weights, frequency
//! and activation are hashed from its four coordinates, and whose output gain follows its
//! multiplicity; the second layer is wired from the whole configuration's digest. So a block that
//! gains a constituent grows a neuron, configurations that share elements share structure, and every
//! distinct configuration gets a texture of its own.
//!
//! The palette comes from the law's presentation (element colours, the impurity accent); hardness
//! draws crystalline veins, weak cohesion draws grain, glow lights the pattern's crests, clarity sets
//! alpha. All inputs are periodic in the face coordinates, so textures tile seamlessly.

use std::f32::consts::TAU;

use material::{element_colour, Element};

use crate::block::appearance::{AppearanceSource, BlockAppearance, LAYER_BYTES, TEXTURE_SIZE};
use crate::mods::{Knob, Mod};

/// Periodic input features per texel.
const INPUTS: usize = 10;
/// Largest first layer (one neuron per distinct element, capped).
const MAX_HIDDEN: usize = 12;
/// Smallest first layer: sparse configurations get extra digest-seeded neurons.
const MIN_HIDDEN: usize = 4;
/// Second layer width.
const MIX: usize = 6;
/// The minimum opacity a translucent layer renders at.
const MIN_ALPHA: u8 = 40;
const KNOB_MAX: i32 = 20;

#[derive(Clone, Copy)]
enum Act {
    Sin,
    Gauss,
    Tanh,
    Ridge,
    Saw,
    Soft,
}

impl Act {
    fn of(byte: u8) -> Act {
        [Act::Sin, Act::Gauss, Act::Tanh, Act::Ridge, Act::Saw, Act::Soft][byte as usize % 6]
    }

    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Act::Sin => x.sin(),
            Act::Gauss => 2.0 * (-x * x).exp() - 1.0,
            Act::Tanh => x.tanh(),
            Act::Ridge => 1.0 - 2.0 * x.sin().abs(),
            Act::Saw => {
                let f = x * 0.5 / std::f32::consts::PI;
                2.0 * (f - f.floor()) - 1.0
            }
            Act::Soft => x / (1.0 + x.abs()),
        }
    }
}

/// A small deterministic generator (splitmix64) for weights.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn element_seed(e: Element, salt: u32) -> u64 {
    let c = u32::from_be_bytes(e.0) as u64;
    (c << 32 | salt as u64) ^ 0xD6E8_FEB8_6659_FD93
}

#[derive(Clone, Copy)]
struct Neuron<const N: usize> {
    w: [f32; N],
    bias: f32,
    act: Act,
    gain: f32,
}

impl<const N: usize> Neuron<N> {
    fn grow(rng: &mut Rng, scale: f32, act: Act, gain: f32) -> Self {
        let mut w = [0.0; N];
        for x in w.iter_mut() {
            *x = rng.signed() * scale;
        }
        Neuron { w, bias: rng.signed() * 1.5, act, gain }
    }

    #[inline]
    fn fire(&self, input: &[f32; N]) -> f32 {
        let mut s = self.bias;
        for (w, x) in self.w.iter().zip(input) {
            s += w * x;
        }
        self.act.apply(s) * self.gain
    }
}

/// The network grown from one configuration.
struct Cppn {
    hidden: Vec<Neuron<INPUTS>>,
    mix: [Neuron<MAX_HIDDEN>; MIX],
    out: [[f32; MIX]; 3],
}

impl Cppn {
    fn grow(src: &AppearanceSource, detail: f32) -> Cppn {
        let law = src.law;
        let config = src.block.configuration();
        let digest = config.digest() ^ law.visual_seed as u64;
        let mut hidden = Vec::with_capacity(MAX_HIDDEN);
        // The most abundant elements first: they define the texture's character.
        let mut counts: Vec<(Element, usize)> = config.counts().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let total = config.len().max(1) as f32;
        for &(e, m) in counts.iter().take(MAX_HIDDEN) {
            let mut rng = Rng(element_seed(e, law.visual_seed));
            // Coordinate 1 sets the spatial frequency; coordinate 0 the activation.
            let freq = (0.6 + e.0[1] as f32 / 255.0 * 2.4) * detail;
            let gain = 0.6 + 1.4 * (m as f32 / total).sqrt();
            hidden.push(Neuron::grow(&mut rng, freq, Act::of(e.0[0]), gain));
        }
        let mut rng = Rng(digest);
        while hidden.len() < MIN_HIDDEN {
            let act = Act::of((rng.next() & 0xFF) as u8);
            hidden.push(Neuron::grow(&mut rng, 1.4 * detail, act, 0.7));
        }
        let width = hidden.len();
        let mix = std::array::from_fn(|_| {
            let act = Act::of((rng.next() & 0xFF) as u8);
            let mut n = Neuron::<MAX_HIDDEN>::grow(&mut rng, 1.8 / (width as f32).sqrt(), act, 1.0);
            for w in n.w[width..].iter_mut() {
                *w = 0.0;
            }
            n
        });
        let out = std::array::from_fn(|_| std::array::from_fn(|_| rng.signed() * 1.2));
        Cppn { hidden, mix, out }
    }

    #[inline]
    fn eval(&self, u: f32, v: f32) -> [f32; 3] {
        let (a, b) = (u * TAU, v * TAU);
        let input = [
            a.sin(),
            a.cos(),
            b.sin(),
            b.cos(),
            (2.0 * a).sin(),
            (2.0 * a).cos(),
            (2.0 * b).sin(),
            (2.0 * b).cos(),
            (a + b).sin(),
            (a - b).cos(),
        ];
        let mut h = [0.0f32; MAX_HIDDEN];
        for (i, n) in self.hidden.iter().enumerate() {
            h[i] = n.fire(&input);
        }
        let mut m = [0.0f32; MIX];
        for (i, n) in self.mix.iter().enumerate() {
            m[i] = n.fire(&h);
        }
        self.out.map(|row| {
            let mut s = 0.0;
            for i in 0..MIX {
                s += row[i] * m[i];
            }
            s.tanh()
        })
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// The palette: base mix, accent, and the colours of the most abundant elements, ordered dark to
/// light so the network's first output walks a gradient.
fn palette(src: &AppearanceSource, contrast: f32) -> Vec<[f32; 3]> {
    let f = |c: [u8; 3]| c.map(|v| v as f32);
    let config = src.block.configuration();
    let mut counts: Vec<(Element, usize)> = config.counts().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let base = f(src.visual.rgb);
    let mut stops = vec![base, f(src.visual.rgb2)];
    // Highlights and shadows of the material's own colour carry the pattern; each abundant
    // element adds a tint of its hue so mixtures shimmer without losing the material's colour.
    stops.push(base.map(|v| (v * 1.22 + 10.0).min(255.0)));
    stops.push(base.map(|v| v * 0.72));
    for &(e, _) in counts.iter().take(3) {
        stops.push(lerp3(base, f(element_colour(src.law, e)), 0.22));
    }
    let mid = base;
    for c in stops.iter_mut() {
        for k in 0..3 {
            c[k] = (mid[k] + (c[k] - mid[k]) * contrast).clamp(0.0, 255.0);
        }
    }
    stops.sort_by(|a, b| lum(*a).total_cmp(&lum(*b)));
    stops.dedup_by(|a, b| (lum(*a) - lum(*b)).abs() < 1.0 && (a[0] - b[0]).abs() < 1.0);
    stops
}

fn gradient(stops: &[[f32; 3]], t: f32) -> [f32; 3] {
    if stops.len() == 1 {
        return stops[0];
    }
    let x = ((t + 1.0) * 0.5).clamp(0.0, 0.9999) * (stops.len() - 1) as f32;
    let i = x as usize;
    // Smooth the steps a little so bands read as strata, not hard posterization.
    let f = x - i as f32;
    let f = f * f * (3.0 - 2.0 * f);
    lerp3(stops[i], stops[i + 1], f)
}

fn hash01(seed: u64, x: u32, y: u32) -> f32 {
    let mut r = Rng(seed ^ ((x as u64) << 32 | y as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
    (r.next() >> 40) as f32 / (1u64 << 24) as f32
}

/// Paint one layer.
pub fn paint(src: &AppearanceSource, detail: f32, contrast: f32, out: &mut [u8; LAYER_BYTES]) {
    let net = Cppn::grow(src, detail);
    let stops = palette(src, contrast);
    let obs = src.obs;
    let grain_seed = src.block.configuration().digest();
    let hard = obs.hardness as f32 / 255.0;
    // Crystalline veins on strongly held matter; grain on weakly held matter.
    let vein_width = (hard - 0.55).max(0.0) * 0.35;
    let grain = 0.03 + (1.0 - hard) * 0.14;
    let glow = obs.emission as f32 / 15.0;
    let alpha = src.visual.alpha;
    let size = TEXTURE_SIZE as usize;
    for y in 0..size {
        for x in 0..size {
            let (u, v) = ((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
            let [t, shade, vein] = net.eval(u, v);
            let mut c = gradient(&stops, t);
            let light = 1.0 + shade * (0.10 + 0.12 * (1.0 - hard));
            let noise = 1.0 + (hash01(grain_seed, x as u32, y as u32) * 2.0 - 1.0) * grain;
            for ch in c.iter_mut() {
                *ch *= light * noise;
            }
            if vein_width > 0.0 && vein.abs() < vein_width {
                let k = 1.0 - vein.abs() / vein_width;
                c = lerp3(c, [255.0, 250.0, 240.0], 0.35 * k);
            }
            if glow > 0.0 {
                let crest = ((t + shade) * 0.5).max(0.0);
                let k = glow * (0.35 + 0.65 * crest);
                let peak = c[0].max(c[1]).max(c[2]);
                c = lerp3(c, [255.0f32, 244.0, 214.0].map(|w| w.max(peak)), 0.5 * k);
            }
            // A faint bevel at the face border keeps block edges legible.
            let edge = x == 0 || y == 0 || x == size - 1 || y == size - 1;
            if edge {
                for ch in c.iter_mut() {
                    *ch *= 0.9;
                }
            }
            let i = (y * size + x) * 4;
            out[i] = c[0].round().clamp(0.0, 255.0) as u8;
            out[i + 1] = c[1].round().clamp(0.0, 255.0) as u8;
            out[i + 2] = c[2].round().clamp(0.0, 255.0) as u8;
            out[i + 3] = if alpha == 255 {
                255
            } else {
                // Frosted clarity: the pattern clouds translucent matter a little.
                ((alpha as f32 + shade * 24.0).clamp(0.0, 255.0) as u8).max(MIN_ALPHA)
            };
        }
    }
}

/// The Essentials appearance mod.
pub struct NeuralTexturesMod {
    detail: f32,
    contrast: f32,
    revision: u32,
}

impl NeuralTexturesMod {
    pub fn new() -> Self {
        Self { detail: 1.0, contrast: 1.0, revision: 1 }
    }

    fn set_knob(&mut self, which: usize, value: f32) {
        let value = ((value * 10.0).round() as i32).clamp(1, KNOB_MAX) as f32 / 10.0;
        let slot = match which {
            0 => &mut self.detail,
            1 => &mut self.contrast,
            _ => return,
        };
        if *slot != value {
            *slot = value;
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

impl Default for NeuralTexturesMod {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockAppearance for NeuralTexturesMod {
    fn layer(&self, src: &AppearanceSource, out: &mut [u8; LAYER_BYTES]) {
        paint(src, self.detail, self.contrast, out);
    }

    fn revision(&self) -> u32 {
        self.revision
    }
}

impl Mod for NeuralTexturesMod {
    fn name(&self) -> &str {
        "Neural textures"
    }

    fn id(&self) -> &'static str {
        "neural_textures"
    }

    fn description(&self) -> &str {
        "Each configuration grows a pattern network from its own elements: a unique 32×32 texture per material."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn appearance(&self) -> Option<&dyn BlockAppearance> {
        Some(self)
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob { label: "Detail", value: format!("{:.1}", self.detail), hint: "0.1..2.0".to_string() },
            Knob { label: "Contrast", value: format!("{:.1}", self.contrast), hint: "0.1..2.0".to_string() },
        ]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        match index {
            0 => self.set_knob(0, self.detail + delta as f32 * 0.1),
            1 => self.set_knob(1, self.contrast + delta as f32 * 0.1),
            _ => {}
        }
    }

    fn save_choice_state(&self) -> Option<String> {
        Some(format!("detail={:.1},contrast={:.1}", self.detail, self.contrast))
    }

    fn load_choice_state(&mut self, data: &str) {
        for part in data.split(',') {
            let Some((k, v)) = part.split_once('=') else { continue };
            let Ok(n) = v.trim().parse::<f32>() else { continue };
            match k.trim() {
                "detail" => self.set_knob(0, n),
                "contrast" => self.set_knob(1, n),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use material::{observe, visual, Block, Configuration, Law};

    fn layer(elems: &[[u8; 4]]) -> [u8; LAYER_BYTES] {
        let law = Law::current();
        let c = Configuration::new(elems.iter().map(|&e| Element::new(e)).collect::<Vec<_>>()).unwrap();
        let block = Block::of(&c);
        let obs = observe(&law, &block);
        let vis = visual(&law, &block);
        let src = AppearanceSource { law: &law, block: &block, visual: &vis, obs: &obs };
        let mut out = [0u8; LAYER_BYTES];
        NeuralTexturesMod::new().layer(&src, &mut out);
        out
    }

    #[test]
    fn deterministic_and_unique_per_configuration() {
        let a = layer(&[[10, 20, 30, 40], [90, 20, 200, 7]]);
        assert_eq!(a, layer(&[[90, 20, 200, 7], [10, 20, 30, 40]]), "storage order is not meaning");
        let b = layer(&[[10, 20, 30, 40], [90, 20, 200, 7], [90, 20, 200, 7]]);
        let c = layer(&[[10, 20, 30, 40], [91, 20, 200, 7]]);
        assert_ne!(a, b, "multiplicity changes the texture");
        assert_ne!(a, c, "a one-unit coordinate change changes the texture");
    }

    #[test]
    fn textures_have_structure_not_a_flat_fill() {
        let t = layer(&[[200, 140, 30, 77], [12, 99, 180, 3], [55, 66, 77, 88]]);
        let mut lums: Vec<u32> = t.chunks_exact(4).map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32).collect();
        lums.sort_unstable();
        let spread = lums[lums.len() * 9 / 10] - lums[lums.len() / 10];
        assert!(spread > 30, "10-90 % luminance spread {spread}");
    }

    #[test]
    fn opaque_matter_is_fully_opaque() {
        let law = Law::current();
        for i in 0..40u8 {
            let e = [[i.wrapping_mul(37), 11, 200, i], [9, i.wrapping_mul(13), 70, 140]];
            let c = Configuration::new(e.iter().map(|&e| Element::new(e)).collect::<Vec<_>>()).unwrap();
            let obs = observe(&law, &Block::of(&c));
            let t = layer(&e);
            if obs.transparency == 0 {
                assert!(t.chunks_exact(4).all(|p| p[3] == 255));
            } else {
                assert!(t.chunks_exact(4).all(|p| p[3] >= MIN_ALPHA));
            }
        }
    }

    #[test]
    #[ignore]
    fn paint_cost_per_layer() {
        let t0 = std::time::Instant::now();
        let n = 200;
        for i in 0..n {
            let i = i as u8;
            std::hint::black_box(layer(&[[i, 3, 9, 200], [i ^ 0x55, 77, 1, 2], [5, i, 8, 9], [1, 2, i, 4]]));
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!("neural texture: {us:.1} µs per 32×32 layer");
    }
}
