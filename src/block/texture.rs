//! Procedural block textures: one 16x16 RGBA8 layer per block id, blended
//! from the colours of the elements in the block's composition.
//!
//! Layer index == block id == engine texture-array layer, so the mesher can
//! stamp `BlockId` straight into `Vertex::color.a`. Layer 0 (air) is all
//! white, satisfying the engine's layer-0-white contract (immediate cubes and
//! flat-colored vertices sample it).
//!
//! Everything here is a *deterministic function of the composition*, never of
//! the block id, so multiplayer clients whose palettes grew in different
//! orders still render identical materials. Textures are computed once per
//! palette growth (world entry, crafting a new block type) — never per frame.
use crate::block::composition::Composition;
use crate::block::element::ElementId;
use crate::block::registry::{BlockId, BlockRegistry};

/// Edge length of every block texture layer, in texels.
pub const TEXTURE_SIZE: u32 = 16;

/// Half-width of the soft transition band around each element cutoff, in
/// noise units: texels whose noise value lands within this distance of a
/// cutoff lerp between the two adjacent elements' colours.
const BLEND: f32 = 0.06;
/// Maximum per-texel brightness jitter, as a +/- fraction.
const JITTER: f32 = 0.08;
/// Neutral gray base for compositions with no elements (Computational).
const NEUTRAL_GRAY: [f32; 3] = [140.0, 140.0, 140.0];

const BYTES_PER_LAYER: usize = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;

/// Build one 16x16 RGBA8 texture layer per registered block, indexed by block
/// id. Block 0 (air) is pure white; every other layer is the element blend of
/// that block's composition. Feed straight to `Engine::set_block_textures`.
pub fn build_block_textures(registry: &BlockRegistry) -> Vec<Vec<u8>> {
    (0..registry.block_count())
        .map(|i| {
            if i == 0 {
                vec![255u8; BYTES_PER_LAYER] // air: engine's layer-0-white contract
            } else {
                layer_for(registry, BlockId(i as u16))
            }
        })
        .collect()
}

/// One block's layer: resolve the composition to sorted element fractions,
/// seed the noise from them, then blend per texel.
fn layer_for(registry: &BlockRegistry, id: BlockId) -> Vec<u8> {
    let parts = parts(&registry.block(id).composition);
    let seed = seed_of(&parts);
    let (colors, cuts): (Vec<[f32; 3]>, Vec<f32>) = if parts.is_empty() {
        // Computational blocks (and any degenerate empty composition beyond
        // air) get a neutral gray base with speckle only.
        (vec![NEUTRAL_GRAY], vec![1.0])
    } else {
        let colors = parts
            .iter()
            .map(|&(e, _)| {
                let c = registry.elements().get(e).color;
                [c.r as f32, c.g as f32, c.b as f32]
            })
            .collect();
        // Cumulative weights partition the noise range [0,1) among elements;
        // the last cutoff is forced to 1.0 to absorb float drift.
        let mut acc = 0.0;
        let mut cuts: Vec<f32> = parts
            .iter()
            .map(|&(_, w)| {
                acc += w;
                acc
            })
            .collect();
        *cuts.last_mut().expect("parts is non-empty") = 1.0;
        (colors, cuts)
    };

    let mut out = Vec::with_capacity(BYTES_PER_LAYER);
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            // Smooth tiling noise picks the texel's element...
            let n = tile_noise(seed, x as f32 + 0.5, y as f32 + 0.5);
            let rgb = pick_color(&colors, &cuts, n);
            // ...and an uncorrelated per-texel hash speckles the brightness.
            // (A pure per-texel hash has no spatial correlation, so the
            // repeat seam is invisible by construction.)
            let jitter = 1.0 + (hash01(seed, JITTER_CHANNEL, x, y) * 2.0 - 1.0) * JITTER;
            for c in rgb {
                out.push((c * jitter).clamp(0.0, 255.0).round() as u8);
            }
            out.push(255);
        }
    }
    out
}

/// The composition as `(element, fraction)` pairs: duplicates aggregated,
/// fractions summing to 1, sorted by element id so the seed and the cutoff
/// order are canonical. Natural blocks come out as equal fractions
/// (`weights()` gives 1 per occurrence), mixtures as their percentages.
fn parts(composition: &Composition) -> Vec<(ElementId, f32)> {
    let mut acc: Vec<(ElementId, f32)> = Vec::new();
    for &(e, w) in composition.weights().iter() {
        match acc.iter_mut().find(|(id, _)| *id == e) {
            Some(entry) => entry.1 += w as f32,
            None => acc.push((e, w as f32)),
        }
    }
    let total: f32 = acc.iter().map(|&(_, w)| w).sum();
    if total > 0.0 {
        for p in &mut acc {
            p.1 /= total;
        }
    }
    acc.sort_unstable_by_key(|&(e, _)| e);
    acc
}

/// FNV-1a over the sorted `(element id, whole percentage)` pairs. Composition
/// -> seed, so identical materials look identical everywhere.
fn seed_of(parts: &[(ElementId, f32)]) -> u32 {
    const PRIME: u32 = 0x0100_0193;
    let mut h: u32 = 0x811c_9dc5;
    for &(e, frac) in parts {
        for b in e.0.to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(PRIME);
        }
        let pct = (frac * 100.0).round() as u8;
        h = (h ^ pct as u32).wrapping_mul(PRIME);
    }
    h
}

/// Map a noise value through the cumulative element cutoffs. Inside an
/// element's band the texel is that element's colour; within [`BLEND`] of a
/// cutoff it lerps toward the adjacent element for a soft transition
/// (continuous across the cutoff: t hits 0.5 exactly on it).
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

// ---- tiling value noise -------------------------------------------------

/// Octave 0: a 4x4 random lattice across the tile (one cell = 4 texels).
const OCTAVE0_PERIOD: u32 = 4;
/// Octave 1: an 8x8 lattice at half amplitude (one cell = 2 texels).
const OCTAVE1_PERIOD: u32 = 8;
/// Channel index reserved for the brightness jitter hash (octaves use 0/1).
const JITTER_CHANNEL: u32 = 0xdead_beef;

/// 2-octave smooth value noise in [0,1), tiling with period [`TEXTURE_SIZE`]
/// in both axes so REPEAT sampling is seamless.
fn tile_noise(seed: u32, x: f32, y: f32) -> f32 {
    (octave_noise(seed, 0, OCTAVE0_PERIOD, x, y) + 0.5 * octave_noise(seed, 1, OCTAVE1_PERIOD, x, y))
        / 1.5
}

/// One octave: bilinear interpolation (smoothstep-faded) of a random lattice.
/// Lattice coordinates are taken modulo `period`, so x == 16 lands on the
/// same lattice points as x == 0 — that is what makes the tile seamless.
fn octave_noise(seed: u32, octave: u32, period: u32, x: f32, y: f32) -> f32 {
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
    t * t * (3.0 - 2.0 * t)
}

/// Hash `(seed, a, b, c)` to a uniform float in [0, 1). A few rounds of a
/// xorshift-multiply mixer — cheap, stateless, and fully deterministic.
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
    use crate::block::element::El;

    #[test]
    fn build_is_deterministic() {
        let mut reg = BlockRegistry::with_builtins();
        reg.natural(&[El::Iron.id(), El::Sulfur.id()]);
        let a = build_block_textures(&reg);
        let b = build_block_textures(&reg);
        assert_eq!(a, b, "two builds over the same palette are identical");
    }

    #[test]
    fn one_layer_per_block_of_the_right_size() {
        let reg = BlockRegistry::with_builtins();
        let layers = build_block_textures(&reg);
        assert_eq!(layers.len(), reg.block_count());
        assert!(layers.iter().all(|l| l.len() == BYTES_PER_LAYER));
    }

    #[test]
    fn air_layer_is_all_white() {
        let reg = BlockRegistry::with_builtins();
        let layers = build_block_textures(&reg);
        assert!(
            layers[0].iter().all(|&b| b == 255),
            "layer 0 must satisfy the engine's layer-0-white contract"
        );
    }

    #[test]
    fn two_element_natural_block_shows_both_element_colors() {
        // Stone (128,128,128) + Organic (86,176,0): gray texels have high
        // blue relative to organic's zero, green texels dominate in G.
        let mut reg = BlockRegistry::with_builtins();
        let id = reg.natural(&[El::Stone.id(), El::Organic.id()]);
        let layers = build_block_textures(&reg);
        let layer = &layers[id.0 as usize];

        let mut stoneish = 0;
        let mut organicish = 0;
        for texel in layer.chunks_exact(4) {
            let (r, g, b) = (texel[0] as i32, texel[1] as i32, texel[2] as i32);
            if b >= 100 && (r - g).abs() <= 30 {
                stoneish += 1;
            }
            if g >= 140 && b <= 50 {
                organicish += 1;
            }
        }
        assert!(stoneish >= 5, "expected stone-dominant texels, got {stoneish}");
        assert!(organicish >= 5, "expected organic-dominant texels, got {organicish}");
    }

    #[test]
    fn noise_lattice_wraps_at_the_tile_period() {
        // Seamless REPEAT tiling: the noise at x=0 must use the same lattice
        // points x=16 would (and likewise for y), for every octave at once.
        for seed in [0u32, 0xabcd_ef01, seed_of(&[(El::Clay.id(), 1.0)])] {
            for i in 0..=32 {
                let t = i as f32 * 0.5;
                assert_eq!(
                    tile_noise(seed, 0.0, t),
                    tile_noise(seed, 16.0, t),
                    "x seam, seed {seed:#x}, t {t}"
                );
                assert_eq!(
                    tile_noise(seed, t, 0.0),
                    tile_noise(seed, t, 16.0),
                    "y seam, seed {seed:#x}, t {t}"
                );
            }
        }
    }

    #[test]
    fn seed_depends_on_composition_not_registration_order() {
        // The same material registered in two registries (different ids if
        // other blocks landed first) must produce byte-identical layers.
        let mut reg_a = BlockRegistry::with_builtins();
        let id_a = reg_a.natural(&[El::Copper.id(), El::Glass.id()]);

        let mut reg_b = BlockRegistry::with_builtins();
        reg_b.natural(&[El::Sulfur.id()]); // shift subsequent ids
        let id_b = reg_b.natural(&[El::Glass.id(), El::Copper.id()]); // order-independent

        assert_ne!(id_a, id_b, "test relies on differing ids");
        let layers_a = build_block_textures(&reg_a);
        let layers_b = build_block_textures(&reg_b);
        assert_eq!(layers_a[id_a.0 as usize], layers_b[id_b.0 as usize]);
    }
}
