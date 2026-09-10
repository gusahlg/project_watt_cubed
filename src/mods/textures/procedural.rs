//! Procedural block textures: one 16×16 RGBA8 layer per *render descriptor*.
//!
//! Essentials appearance mod (`procedural_textures`). The same two-colour
//! value-noise pattern as the former core generator, with `grain` and
//! `contrast` knobs (1.0 = bit-identical to that generator). Layer 0 (air)
//! is painted all white by the appearance seam, not here.

use material::Visual;

use crate::block::appearance::{BlockAppearance, LAYER_BYTES, TEXTURE_SIZE};
use crate::mods::{Knob, Mod};

/// Soft transition band around the two-colour cutoff, in noise units.
const BLEND: f32 = 0.06;

/// ~16 % — the minimum opacity a translucent layer renders at.
const MIN_ALPHA: u8 = 40;

/// Channel index reserved for the brightness jitter hash (octaves use 0/1).
const JITTER_CHANNEL: u32 = 0xdead_beef;

const GRAIN_DEFAULT: f32 = 1.0;
const CONTRAST_DEFAULT: f32 = 1.0;
const KNOB_MAX: i32 = 20;

/// CPU procedural appearance. Defaults keep the pre-mod generator bytes.
pub struct ProceduralTexturesMod {
    grain: f32,
    contrast: f32,
    revision: u32,
}

impl ProceduralTexturesMod {
    pub fn new() -> Self {
        Self {
            grain: GRAIN_DEFAULT,
            contrast: CONTRAST_DEFAULT,
            revision: 1,
        }
    }

    fn set_knob(&mut self, which: usize, value: f32) {
        let value = snap_knob(value);
        let slot = match which {
            0 => &mut self.grain,
            1 => &mut self.contrast,
            _ => return,
        };
        if *slot != value {
            *slot = value;
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

impl Default for ProceduralTexturesMod {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockAppearance for ProceduralTexturesMod {
    fn layer(&self, vis: &Visual, out: &mut [u8; LAYER_BYTES]) {
        paint(vis, self.grain, self.contrast, out);
    }

    fn revision(&self) -> u32 {
        self.revision
    }

    fn wants_gpu_descriptors(&self) -> bool {
        false
    }
}

impl Mod for ProceduralTexturesMod {
    fn name(&self) -> &str {
        "Procedural textures"
    }

    fn id(&self) -> &'static str {
        "procedural_textures"
    }

    fn description(&self) -> &str {
        "Two-colour value-noise 16×16 layers from each render descriptor."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn appearance(&self) -> Option<&dyn BlockAppearance> {
        Some(self)
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![
            Knob {
                label: "Grain",
                value: format!("{:.2}", self.grain),
                hint: "0.00..2.00".to_string(),
            },
            Knob {
                label: "Contrast",
                value: format!("{:.2}", self.contrast),
                hint: "0.00..2.00".to_string(),
            },
        ]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        match index {
            0 => self.set_knob(0, self.grain + delta as f32 * 0.1),
            1 => self.set_knob(1, self.contrast + delta as f32 * 0.1),
            _ => {}
        }
    }

    fn save_choice_state(&self) -> Option<String> {
        Some(format!(
            "grain={:.2},contrast={:.2}",
            self.grain, self.contrast
        ))
    }

    fn load_choice_state(&mut self, data: &str) {
        let mut grain = self.grain;
        let mut contrast = self.contrast;
        for part in data.split(',') {
            let Some((k, v)) = part.split_once('=') else {
                continue;
            };
            match k.trim() {
                "grain" => {
                    if let Ok(n) = v.trim().parse() {
                        grain = n;
                    }
                }
                "contrast" => {
                    if let Ok(n) = v.trim().parse() {
                        contrast = n;
                    }
                }
                _ => {}
            }
        }
        self.set_knob(0, grain);
        self.set_knob(1, contrast);
    }
}

fn snap_knob(v: f32) -> f32 {
    let t = (v * 10.0).round() as i32;
    t.clamp(0, KNOB_MAX) as f32 / 10.0
}

fn paint(vis: &Visual, grain: f32, contrast: f32, out: &mut [u8; LAYER_BYTES]) {
    let seed = seed_of(vis);
    let (c0, c1) = contrast_colors(
        vis.rgb.map(|c| c as f32),
        vis.rgb2.map(|c| c as f32),
        contrast,
    );
    let colors = [c0, c1];
    let cuts = [0.5_f32, 1.0];
    let cell = noise_cell(vis.frequency);
    let jitter_amp = if grain == GRAIN_DEFAULT {
        vis.roughness as f32 / 255.0 * 0.16
    } else {
        vis.roughness as f32 / 255.0 * 0.16 * grain
    };
    let lift = vis.glow / 4;
    let alpha = texel_alpha(vis.alpha);

    let mut i = 0;
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let n = tile_noise(seed, cell, x as f32 + 0.5, y as f32 + 0.5);
            let mut rgb = pick_color(&colors, &cuts, n);
            let jitter = 1.0 + (hash01(seed, JITTER_CHANNEL, x, y) * 2.0 - 1.0) * jitter_amp;
            for c in rgb.iter_mut() {
                *c = (*c * jitter).clamp(0.0, 255.0);
                out[i] = ((*c).round() as u8).saturating_add(lift);
                i += 1;
            }
            out[i] = alpha;
            i += 1;
        }
    }
}

fn contrast_colors(c0: [f32; 3], c1: [f32; 3], contrast: f32) -> ([f32; 3], [f32; 3]) {
    if contrast == CONTRAST_DEFAULT {
        return (c0, c1);
    }
    let mid = [
        (c0[0] + c1[0]) * 0.5,
        (c0[1] + c1[1]) * 0.5,
        (c0[2] + c1[2]) * 0.5,
    ];
    let adj = |c: [f32; 3]| {
        [
            (mid[0] + (c[0] - mid[0]) * contrast).clamp(0.0, 255.0),
            (mid[1] + (c[1] - mid[1]) * contrast).clamp(0.0, 255.0),
            (mid[2] + (c[2] - mid[2]) * contrast).clamp(0.0, 255.0),
        ]
    };
    (adj(c0), adj(c1))
}

fn texel_alpha(alpha: u8) -> u8 {
    if alpha == 255 {
        255
    } else {
        alpha.max(MIN_ALPHA)
    }
}

/// Frequency 0 → cell 16 (one cell across the tile); 255 → cell 2.
fn noise_cell(frequency: u8) -> f32 {
    16.0 - frequency as f32 * 14.0 / 255.0
}

fn seed_of(vis: &Visual) -> u32 {
    crate::hash::fnv1a_32(&[
        vis.rgb[0],
        vis.rgb[1],
        vis.rgb[2],
        vis.rgb2[0],
        vis.rgb2[1],
        vis.rgb2[2],
        vis.frequency,
        vis.roughness,
        vis.alpha,
        vis.glow,
    ])
}

/// Pick a colour for a noise value, blending between the two colours near the cut.
fn pick_color(colors: &[[f32; 3]], cuts: &[f32], n: f32) -> [f32; 3] {
    let i = cuts
        .iter()
        .position(|&c| n < c)
        .unwrap_or(colors.len() - 1);
    let lo = if i == 0 { 0.0 } else { cuts[i - 1] };
    let hi = cuts[i];
    if i > 0 && n - lo < BLEND {
        lerp3(colors[i - 1], colors[i], (n - (lo - BLEND)) / (2.0 * BLEND))
    } else if i + 1 < colors.len() && hi - n < BLEND {
        lerp3(colors[i], colors[i + 1], (n - (hi - BLEND)) / (2.0 * BLEND))
    } else {
        colors[i]
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Tiling noise that repeats seamlessly across texture boundaries.
fn tile_noise(seed: u32, cell: f32, x: f32, y: f32) -> f32 {
    (octave_noise(seed, 0, cell, x, y) + 0.5 * octave_noise(seed, 1, cell * 0.5, x, y)) / 1.5
}

fn octave_noise(seed: u32, octave: u32, cell: f32, x: f32, y: f32) -> f32 {
    let cell = cell.max(1.0);
    let period = (TEXTURE_SIZE as f32 / cell).round().max(1.0) as u32;
    let cell = TEXTURE_SIZE as f32 / period as f32;
    let (fx, fy) = (x / cell, y / cell);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (smoothstep(fx - x0), smoothstep(fy - y0));
    let (ix, iy) = (x0 as u32, y0 as u32);
    let v00 = lattice(seed, octave, period, ix, iy);
    let v10 = lattice(seed, octave, period, ix + 1, iy);
    let v01 = lattice(seed, octave, period, ix, iy + 1);
    let v11 = lattice(seed, octave, period, ix + 1, iy + 1);
    let a = v00 + (v10 - v00) * tx;
    let b = v01 + (v11 - v01) * tx;
    a + (b - a) * ty
}

fn lattice(seed: u32, octave: u32, period: u32, lx: u32, ly: u32) -> f32 {
    hash01(seed, octave, lx % period, ly % period)
}

fn smoothstep(t: f32) -> f32 {
    crate::math::smooth(t)
}

fn hash01(seed: u32, a: u32, b: u32, c: u32) -> f32 {
    let h = mix(
        seed ^ mix(
            a.wrapping_mul(0x9e37_79b9) ^ mix(b.wrapping_mul(0x85eb_ca6b) ^ mix(c.wrapping_mul(0xc2b2_ae35))),
        ),
    );
    (h >> 8) as f32 / (1u32 << 24) as f32
}

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::appearance::{fill_descriptor_layer, LAYER_BYTES};
    use crate::block::registry::BlockRegistry;
    use crate::hash::fnv1a_32;
    use crate::mods::Mods;
    use material::{visual, Configuration, Element, Law};

    fn vis(e: [u8; 4]) -> Visual {
        visual(&Law::v0(), &Configuration::single(Element::new(e)))
    }

    fn layer_of(m: &ProceduralTexturesMod, v: &Visual) -> [u8; LAYER_BYTES] {
        let mut out = [0u8; LAYER_BYTES];
        m.layer(v, &mut out);
        out
    }

    #[test]
    fn build_is_deterministic() {
        let m = ProceduralTexturesMod::new();
        let v = vis([40, 80, 120, 160]);
        assert_eq!(layer_of(&m, &v), layer_of(&m, &v));
    }

    #[test]
    fn layer_is_the_right_size() {
        let v = vis([10, 20, 30, 40]);
        assert_eq!(layer_of(&ProceduralTexturesMod::new(), &v).len(), LAYER_BYTES);
    }

    #[test]
    fn air_descriptor_layer_is_all_white() {
        let reg = BlockRegistry::with_builtins();
        let mut layer = [0u8; LAYER_BYTES];
        fill_descriptor_layer(&ProceduralTexturesMod::new(), &reg, 0, &mut layer);
        assert!(
            layer.iter().all(|&b| b == 255),
            "layer 0 must satisfy the engine's layer-0-white contract"
        );
    }

    #[test]
    fn translucent_visual_carries_sub_opaque_alpha() {
        let m = ProceduralTexturesMod::new();
        let mut glass = vis([200, 10, 180, 40]);
        glass.alpha = 80;
        let mut stone = vis([120, 130, 140, 150]);
        stone.alpha = 255;
        assert!(layer_of(&m, &glass)[3] < 255);
        assert_eq!(layer_of(&m, &stone)[3], 255);
        let mut clear = glass;
        clear.alpha = 0;
        assert_eq!(layer_of(&m, &clear)[3], MIN_ALPHA);
    }

    #[test]
    fn two_colours_both_appear() {
        let v = Visual {
            rgb: [200, 40, 40],
            rgb2: [40, 200, 40],
            frequency: 180,
            roughness: 40,
            alpha: 255,
            glow: 0,
        };
        let layer = layer_of(&ProceduralTexturesMod::new(), &v);
        let mut redish = 0;
        let mut greenish = 0;
        for texel in layer.chunks_exact(4) {
            if texel[0] > texel[1] + 40 {
                redish += 1;
            }
            if texel[1] > texel[0] + 40 {
                greenish += 1;
            }
        }
        assert!(redish >= 5, "expected red-dominant texels, got {redish}");
        assert!(greenish >= 5, "expected green-dominant texels, got {greenish}");
    }

    #[test]
    fn noise_lattice_wraps_at_the_tile_period() {
        for seed in [0u32, 0xabcd_ef01, 42] {
            for cell in [16.0, 8.0, 4.0, 2.0] {
                for i in 0..=32 {
                    let t = i as f32 * 0.5;
                    assert_eq!(
                        tile_noise(seed, cell, 0.0, t),
                        tile_noise(seed, cell, 16.0, t),
                        "x seam, seed {seed:#x}, cell {cell}, t {t}"
                    );
                    assert_eq!(
                        tile_noise(seed, cell, t, 0.0),
                        tile_noise(seed, cell, t, 16.0),
                        "y seam, seed {seed:#x}, cell {cell}, t {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn pick_color_is_continuous_across_a_cutoff() {
        let colors = [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]];
        let cuts = [0.5, 1.0];
        let at_cutoff = pick_color(&colors, &cuts, cuts[0]);
        for (c, want) in at_cutoff.iter().zip([50.0, 50.0, 50.0]) {
            assert!((c - want).abs() < 1e-4, "expected midpoint at cutoff, got {at_cutoff:?}");
        }
    }

    #[test]
    fn same_visual_same_bytes_regardless_of_descriptor_id() {
        let m = ProceduralTexturesMod::new();
        let mut a = BlockRegistry::with_builtins();
        let mut b = BlockRegistry::with_builtins();
        let c = Configuration::single(Element::new([90, 100, 110, 120]));
        let ia = a.intern(&c).unwrap();
        b.intern(&Configuration::single(Element::new([1, 2, 3, 4]))).unwrap();
        let ib = b.intern(&c).unwrap();
        let la = a.render_layer(ia);
        let lb = b.render_layer(ib);
        let mut oa = [0u8; LAYER_BYTES];
        let mut ob = [0u8; LAYER_BYTES];
        fill_descriptor_layer(&m, &a, la, &mut oa);
        fill_descriptor_layer(&m, &b, lb, &mut ob);
        assert_eq!(oa, ob);
    }

    /// Eight fixed descriptors, hashed once. A mismatch means the default
    /// generator (grain=1, contrast=1) moved.
    #[test]
    fn procedural_layer_byte_pin() {
        let m = ProceduralTexturesMod::new();
        let pins: [(Visual, u32); 8] = [
            (
                Visual {
                    rgb: [255, 255, 255],
                    rgb2: [255, 255, 255],
                    frequency: 0,
                    roughness: 0,
                    alpha: 255,
                    glow: 0,
                },
                0x422f51c5,
            ),
            (
                Visual {
                    rgb: [200, 40, 40],
                    rgb2: [40, 200, 40],
                    frequency: 180,
                    roughness: 40,
                    alpha: 255,
                    glow: 0,
                },
                0x8f3ec301,
            ),
            (
                Visual {
                    rgb: [30, 60, 90],
                    rgb2: [90, 60, 30],
                    frequency: 0,
                    roughness: 0,
                    alpha: 255,
                    glow: 0,
                },
                0xfbe1c1c5,
            ),
            (
                Visual {
                    rgb: [10, 20, 30],
                    rgb2: [40, 50, 60],
                    frequency: 255,
                    roughness: 255,
                    alpha: 255,
                    glow: 0,
                },
                0xe264d943,
            ),
            (
                Visual {
                    rgb: [80, 80, 80],
                    rgb2: [160, 160, 160],
                    frequency: 64,
                    roughness: 128,
                    alpha: 80,
                    glow: 0,
                },
                0xff3fbdcf,
            ),
            (
                Visual {
                    rgb: [16, 32, 48],
                    rgb2: [200, 180, 20],
                    frequency: 32,
                    roughness: 16,
                    alpha: 0,
                    glow: 0,
                },
                0xaf5b8ec9,
            ),
            (
                Visual {
                    rgb: [240, 200, 40],
                    rgb2: [40, 40, 240],
                    frequency: 120,
                    roughness: 200,
                    alpha: 255,
                    glow: 64,
                },
                0xd9587c43,
            ),
            (
                Visual {
                    rgb: [8, 16, 24],
                    rgb2: [24, 16, 8],
                    frequency: 8,
                    roughness: 8,
                    alpha: 40,
                    glow: 255,
                },
                0x02462a38,
            ),
        ];
        for (i, (v, want)) in pins.iter().enumerate() {
            let got = fnv1a_32(&layer_of(&m, v));
            assert_eq!(got, *want, "descriptor {i} hash {got:#010x} != {want:#010x}");
        }
    }

    #[test]
    fn default_knobs_are_identity_and_bump_revision() {
        let mut m = ProceduralTexturesMod::new();
        assert_eq!(m.grain, 1.0);
        assert_eq!(m.contrast, 1.0);
        let v = vis([40, 80, 120, 160]);
        let before = layer_of(&m, &v);
        let rev = m.revision();
        m.step_knob(0, 1);
        assert_eq!(m.grain, 1.1);
        assert_eq!(m.revision(), rev + 1);
        assert_ne!(layer_of(&m, &v), before);
        m.step_knob(0, -1);
        assert_eq!(m.grain, 1.0);
        assert_eq!(layer_of(&m, &v), before);
    }

    #[test]
    fn knobs_persist_like_other_mod_state() {
        let mut m = ProceduralTexturesMod::new();
        m.step_knob(0, 5);
        m.step_knob(1, -4);
        let text = m.save_choice_state().unwrap();
        assert_eq!(text, "grain=1.50,contrast=0.60");
        let mut fresh = ProceduralTexturesMod::new();
        let rev = fresh.revision();
        fresh.load_choice_state(&text);
        assert_eq!(fresh.grain, 1.5);
        assert_eq!(fresh.contrast, 0.6);
        assert!(fresh.revision() > rev);
        fresh.load_choice_state("grain=9,contrast=-1");
        assert_eq!(fresh.grain, 2.0);
        assert_eq!(fresh.contrast, 0.0);
    }

    #[test]
    fn defaults_install_this_mod_as_the_appearance() {
        let mods = Mods::with_defaults();
        assert!(!mods.appearance().wants_gpu_descriptors());
        assert_eq!(mods.appearance().revision(), 1);
        let mut off = Mods::with_defaults();
        off.set_enabled("procedural_textures", false);
        assert_eq!(off.appearance().revision(), 0);
        assert!(!off.appearance().wants_gpu_descriptors());
    }

}
