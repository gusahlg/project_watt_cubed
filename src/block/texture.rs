//! Procedural block textures: one 16×16 RGBA8 layer per *render descriptor*.
//!
//! Layer index == descriptor id == engine texture-array layer, which the mesher
//! carries as a dedicated per-vertex layer index. Many configurations share one
//! descriptor, so the 14-bit vertex field never caps the number of materials.
//! Layer 0 (air's descriptor) is all white, satisfying the engine's
//! layer-0-white contract (immediate cubes and flat-colored vertices sample it).
//!
//! Everything here is a *deterministic function of the [`Visual`]* — two colours
//! blended by tiling value noise, grain from roughness, alpha from the visual
//! (floor 40 for translucent, 255 opaque), glow a lightening of `glow/4`.
use material::Visual;

use crate::block::registry::BlockRegistry;

/// Edge length of every block texture layer, in texels.
pub const TEXTURE_SIZE: u32 = 16;

/// Soft transition band around the two-colour cutoff, in noise units.
const BLEND: f32 = 0.06;

const BYTES_PER_LAYER: usize = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;

/// ~16 % — the minimum opacity a translucent layer renders at.
const MIN_ALPHA: u8 = 40;

/// Channel index reserved for the brightness jitter hash (octaves use 0/1).
const JITTER_CHANNEL: u32 = 0xdead_beef;

/// Build the 16×16 RGBA8 layer for one visual. Deterministic in `vis`.
pub fn build_layer(vis: &Visual) -> Vec<u8> {
    let seed = seed_of(vis);
    let colors = [
        vis.rgb.map(|c| c as f32),
        vis.rgb2.map(|c| c as f32),
    ];
    let cuts = [0.5_f32, 1.0];
    let cell = noise_cell(vis.frequency);
    let jitter_amp = vis.roughness as f32 / 255.0 * 0.16;
    let lift = vis.glow / 4;
    let alpha = texel_alpha(vis.alpha);

    let mut out = Vec::with_capacity(BYTES_PER_LAYER);
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let n = tile_noise(seed, cell, x as f32 + 0.5, y as f32 + 0.5);
            let mut rgb = pick_color(&colors, &cuts, n);
            let jitter = 1.0 + (hash01(seed, JITTER_CHANNEL, x, y) * 2.0 - 1.0) * jitter_amp;
            for c in rgb.iter_mut() {
                *c = (*c * jitter).clamp(0.0, 255.0);
                out.push(((*c).round() as u8).saturating_add(lift));
            }
            out.push(alpha);
        }
    }
    out
}

/// Build the texture layer for one render descriptor. Layer 0 is all white.
pub fn build_descriptor_texture(registry: &BlockRegistry, layer: u16) -> Vec<u8> {
    if layer == 0 {
        vec![255u8; BYTES_PER_LAYER]
    } else {
        build_layer(&registry.descriptor(layer))
    }
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
    use crate::block::registry::BlockRegistry;
    use material::{visual, Configuration, Element, Law};

    fn vis(e: [u8; 4]) -> Visual {
        visual(&Law::v0(), &Configuration::single(Element::new(e)))
    }

    #[test]
    fn build_is_deterministic() {
        let v = vis([40, 80, 120, 160]);
        assert_eq!(build_layer(&v), build_layer(&v));
    }

    #[test]
    fn layer_is_the_right_size() {
        let v = vis([10, 20, 30, 40]);
        assert_eq!(build_layer(&v).len(), BYTES_PER_LAYER);
    }

    #[test]
    fn air_descriptor_layer_is_all_white() {
        let reg = BlockRegistry::with_builtins();
        let layer = build_descriptor_texture(&reg, 0);
        assert!(
            layer.iter().all(|&b| b == 255),
            "layer 0 must satisfy the engine's layer-0-white contract"
        );
    }

    #[test]
    fn translucent_visual_carries_sub_opaque_alpha() {
        let mut glass = vis([200, 10, 180, 40]);
        glass.alpha = 80;
        let mut stone = vis([120, 130, 140, 150]);
        stone.alpha = 255;
        assert!(build_layer(&glass)[3] < 255);
        assert_eq!(build_layer(&stone)[3], 255);
        let mut clear = glass;
        clear.alpha = 0;
        assert_eq!(build_layer(&clear)[3], MIN_ALPHA);
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
        let layer = build_layer(&v);
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
        let mut a = BlockRegistry::with_builtins();
        let mut b = BlockRegistry::with_builtins();
        let c = Configuration::single(Element::new([90, 100, 110, 120]));
        let ia = a.intern(&c).unwrap();
        b.intern(&Configuration::single(Element::new([1, 2, 3, 4]))).unwrap();
        let ib = b.intern(&c).unwrap();
        let la = a.render_layer(ia);
        let lb = b.render_layer(ib);
        assert_eq!(
            build_descriptor_texture(&a, la),
            build_descriptor_texture(&b, lb)
        );
    }
}
