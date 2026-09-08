//! InfiniteDiffusion-style fields: overlapping denoising windows over an
//! unbounded integer lattice.
//!
//! A [`Score`] predicts a clean sample from a noisy one. [`InfiniteField`]
//! runs that score through phased, overlapping tiles with linear blending so
//! the result is seed-consistent, order-independent, and lazily cached.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// How tiles are laid out and how many denoising phases run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    /// World seed mixed into the noise lattice.
    pub seed: u64,
    /// Tile edge in samples (e.g. 32 or 64).
    pub tile: u32,
    /// Tile origin stride; `tile/2` is 50% overlap.
    pub stride: u32,
    /// Denoising phases (paper: more phases ⇒ stronger global consistency).
    pub phases: u32,
    /// Planar channels stored per sample.
    pub channels: u32,
}

impl Spec {
    /// A compact default: 32² tiles, 50% overlap, 2 phases, 4 channels.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            tile: 32,
            stride: 16,
            phases: 2,
            channels: 4,
        }
    }

    fn valid(self) -> bool {
        self.tile >= 4
            && self.stride >= 1
            && self.stride <= self.tile
            && self.phases >= 1
            && self.phases <= 12
            && self.channels >= 1
            && self.channels <= 16
    }
}

/// Predicts a clean value from the current noisy sample at one cell.
pub trait Score: Send + Sync {
    fn predict(&self, channel: u32, x: i32, z: i32, phase: u32, phases: u32, current: f32) -> f32;
}

/// Hash-structure score: each phase pulls toward value-noise whose cell size
/// halves as the phase index rises (coarse structure first, then detail).
#[derive(Clone, Copy, Debug)]
pub struct HashScore {
    pub seed: u64,
}

impl Score for HashScore {
    fn predict(&self, channel: u32, x: i32, z: i32, phase: u32, phases: u32, current: f32) -> f32 {
        let remaining = phases.saturating_sub(1).saturating_sub(phase);
        let cell = 8u32 << remaining.min(8);
        let n = value_noise(self.seed ^ (channel as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), x, z, cell);
        let alpha = 0.28 + 0.12 * (phase as f32 / phases.max(1) as f32);
        current + (n - current) * alpha
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TileKey {
    phase: u32,
    tx: i32,
    tz: i32,
}

/// Finished tiles kept around; values are pure, so eviction is only a
/// recompute. Far-LOD queries would otherwise grow this without bound.
pub const TILE_CACHE_CAP: usize = 4096;

/// Lazy, thread-safe infinite field.
pub struct InfiniteField<S: Score> {
    spec: Spec,
    score: S,
    cache: RwLock<HashMap<TileKey, Arc<[f32]>>>,
}

impl<S: Score> InfiniteField<S> {
    pub fn new(spec: Spec, score: S) -> Self {
        assert!(spec.valid(), "InfiniteField spec out of range");
        Self {
            spec,
            score,
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn spec(&self) -> Spec {
        self.spec
    }

    /// One channel at one lattice point.
    pub fn sample(&self, channel: u32, x: i32, z: i32) -> f32 {
        let phase = self.spec.phases.saturating_sub(1);
        let loaded = self.load_covering(phase, x, z);
        blend_loaded(&self.spec, &loaded, channel, x, z)
    }

    /// Fill `out[z * w + x]` with channel 0 over `[x0, x0+w) × [z0, z0+h)`.
    pub fn fill_ch0(&self, x0: i32, z0: i32, w: u32, h: u32, out: &mut [f32]) {
        assert_eq!(out.len(), (w * h) as usize);
        let mut buf = vec![0.0f32; (w * h * self.spec.channels) as usize];
        self.fill_all(x0, z0, w, h, &mut buf);
        for i in 0..(w * h) as usize {
            out[i] = buf[i * self.spec.channels as usize];
        }
    }

    /// All channels at one point, written into `out`.
    pub fn sample_all(&self, x: i32, z: i32, out: &mut [f32]) {
        let n = self.spec.channels as usize;
        assert!(out.len() >= n);
        let phase = self.spec.phases.saturating_sub(1);
        let loaded = self.load_covering(phase, x, z);
        for c in 0..self.spec.channels {
            out[c as usize] = blend_loaded(&self.spec, &loaded, c, x, z);
        }
    }

    /// All channels over a rectangle. `out` is `(z * w + x) * channels + c`.
    /// Loads each covering tile once, then blends — same values as
    /// [`sample_all`](Self::sample_all) at every cell.
    pub fn fill_all(&self, x0: i32, z0: i32, w: u32, h: u32, out: &mut [f32]) {
        let ch = self.spec.channels;
        assert_eq!(out.len(), (w * h * ch) as usize);
        if w == 0 || h == 0 {
            return;
        }
        let phase = self.spec.phases.saturating_sub(1);
        let tile = self.spec.tile as i32;
        let stride = self.spec.stride as i32;
        let x1 = x0 + w as i32 - 1;
        let z1 = z0 + h as i32 - 1;
        let map = self.load_rect(phase, x0, z0, x1, z1);
        for dz in 0..h as i32 {
            for dx in 0..w as i32 {
                let x = x0 + dx;
                let z = z0 + dz;
                let base = ((dz as u32 * w + dx as u32) * ch) as usize;
                let tx0 = div_floor(x - tile + 1, stride);
                let tx1 = div_floor(x, stride);
                let tz0 = div_floor(z - tile + 1, stride);
                let tz1 = div_floor(z, stride);
                for c in 0..ch {
                    let mut sum = 0.0f32;
                    let mut wsum = 0.0f32;
                    for tz in tz0..=tz1 {
                        for tx in tx0..=tx1 {
                            let Some(raw) = map.get(&(tx, tz)) else { continue };
                            let ox = tx * stride;
                            let oz = tz * stride;
                            let lx = x - ox;
                            let lz = z - oz;
                            if lx < 0 || lz < 0 || lx >= tile || lz >= tile {
                                continue;
                            }
                            let wt = kernel(lx, tile) * kernel(lz, tile);
                            let i = index(ch, self.spec.tile, c, lx as u32, lz as u32);
                            sum += raw[i] * wt;
                            wsum += wt;
                        }
                    }
                    out[base + c as usize] = if wsum < 1e-6 {
                        tiled_gaussian(self.spec.seed, c, x, z)
                    } else {
                        sum / wsum
                    };
                }
            }
        }
    }

    pub fn cached_tiles(&self) -> usize {
        self.cache.read().expect("field cache").len()
    }

    fn covering_range(spec: Spec, x: i32, z: i32) -> (i32, i32, i32, i32) {
        let tile = spec.tile as i32;
        let stride = spec.stride as i32;
        (
            div_floor(x - tile + 1, stride),
            div_floor(x, stride),
            div_floor(z - tile + 1, stride),
            div_floor(z, stride),
        )
    }

    fn load_covering(&self, phase: u32, x: i32, z: i32) -> Vec<(i32, i32, Arc<[f32]>)> {
        let (tx0, tx1, tz0, tz1) = Self::covering_range(self.spec, x, z);
        self.load_range(phase, tx0, tx1, tz0, tz1)
    }

    fn load_rect(
        &self,
        phase: u32,
        x0: i32,
        z0: i32,
        x1: i32,
        z1: i32,
    ) -> HashMap<(i32, i32), Arc<[f32]>> {
        let tile = self.spec.tile as i32;
        let stride = self.spec.stride as i32;
        let tx0 = div_floor(x0 - tile + 1, stride);
        let tx1 = div_floor(x1, stride);
        let tz0 = div_floor(z0 - tile + 1, stride);
        let tz1 = div_floor(z1, stride);
        self.load_range(phase, tx0, tx1, tz0, tz1)
            .into_iter()
            .map(|(tx, tz, raw)| ((tx, tz), raw))
            .collect()
    }

    fn load_range(
        &self,
        phase: u32,
        tx0: i32,
        tx1: i32,
        tz0: i32,
        tz1: i32,
    ) -> Vec<(i32, i32, Arc<[f32]>)> {
        let mut keys = Vec::new();
        for tz in tz0..=tz1 {
            for tx in tx0..=tx1 {
                keys.push(TileKey { phase, tx, tz });
            }
        }
        let mut out = Vec::with_capacity(keys.len());
        let mut missing = Vec::new();
        {
            let cache = self.cache.read().expect("field cache");
            for key in keys {
                if let Some(hit) = cache.get(&key) {
                    out.push((key.tx, key.tz, Arc::clone(hit)));
                } else {
                    missing.push(key);
                }
            }
        }
        for key in missing {
            out.push((key.tx, key.tz, self.raw_tile(key.phase, key.tx, key.tz)));
        }
        out
    }

    fn raw_tile(&self, phase: u32, tx: i32, tz: i32) -> Arc<[f32]> {
        let key = TileKey { phase, tx, tz };
        if let Some(hit) = self.cache.read().expect("field cache").get(&key) {
            return Arc::clone(hit);
        }
        let built = self.build_tile(phase, tx, tz);
        let mut cache = self.cache.write().expect("field cache");
        if cache.len() >= TILE_CACHE_CAP {
            cache.clear();
        }
        Arc::clone(cache.entry(key).or_insert(built))
    }

    fn build_tile(&self, phase: u32, tx: i32, tz: i32) -> Arc<[f32]> {
        let spec = self.spec;
        let tile = spec.tile;
        let stride = spec.stride as i32;
        let ox = tx * stride;
        let oz = tz * stride;
        let n = (spec.channels * tile * tile) as usize;
        let mut buf = vec![0.0f32; n];
        if phase == 0 {
            for lz in 0..tile {
                for lx in 0..tile {
                    for c in 0..spec.channels {
                        let i = index(spec.channels, tile, c, lx, lz);
                        buf[i] = tiled_gaussian(spec.seed, c, ox + lx as i32, oz + lz as i32);
                    }
                }
            }
        } else {
            let x1 = ox + tile as i32 - 1;
            let z1 = oz + tile as i32 - 1;
            let prev = self.load_rect(phase - 1, ox, oz, x1, z1);
            let stride = spec.stride as i32;
            let t = spec.tile as i32;
            for lz in 0..tile {
                for lx in 0..tile {
                    let x = ox + lx as i32;
                    let z = oz + lz as i32;
                    let tx0 = div_floor(x - t + 1, stride);
                    let tx1 = div_floor(x, stride);
                    let tz0 = div_floor(z - t + 1, stride);
                    let tz1 = div_floor(z, stride);
                    for c in 0..spec.channels {
                        let i = index(spec.channels, tile, c, lx, lz);
                        let mut sum = 0.0f32;
                        let mut wsum = 0.0f32;
                        for tz in tz0..=tz1 {
                            for tx in tx0..=tx1 {
                                let Some(raw) = prev.get(&(tx, tz)) else { continue };
                                let lx2 = x - tx * stride;
                                let lz2 = z - tz * stride;
                                if lx2 < 0 || lz2 < 0 || lx2 >= t || lz2 >= t {
                                    continue;
                                }
                                let wt = kernel(lx2, t) * kernel(lz2, t);
                                let j = index(spec.channels, spec.tile, c, lx2 as u32, lz2 as u32);
                                sum += raw[j] * wt;
                                wsum += wt;
                            }
                        }
                        buf[i] = if wsum < 1e-6 {
                            tiled_gaussian(spec.seed, c, x, z)
                        } else {
                            sum / wsum
                        };
                    }
                }
            }
        }
        for lz in 0..tile {
            for lx in 0..tile {
                let x = ox + lx as i32;
                let z = oz + lz as i32;
                for c in 0..spec.channels {
                    let i = index(spec.channels, tile, c, lx, lz);
                    buf[i] = self.score.predict(c, x, z, phase, spec.phases, buf[i]);
                }
            }
        }
        Arc::from(buf)
    }
}

fn blend_loaded(spec: &Spec, tiles: &[(i32, i32, Arc<[f32]>)], channel: u32, x: i32, z: i32) -> f32 {
    let tile = spec.tile as i32;
    let stride = spec.stride as i32;
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for &(tx, tz, ref raw) in tiles {
        let ox = tx * stride;
        let oz = tz * stride;
        let lx = x - ox;
        let lz = z - oz;
        if lx < 0 || lz < 0 || lx >= tile || lz >= tile {
            continue;
        }
        let w = kernel(lx, tile) * kernel(lz, tile);
        let i = index(spec.channels, spec.tile, channel, lx as u32, lz as u32);
        sum += raw[i] * w;
        wsum += w;
    }
    if wsum < 1e-6 {
        tiled_gaussian(spec.seed, channel, x, z)
    } else {
        sum / wsum
    }
}

fn index(channels: u32, tile: u32, c: u32, x: u32, z: u32) -> usize {
    ((z * tile + x) * channels + c) as usize
}

fn div_floor(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

/// Separable linear weight, ~1 at the tile centre, ~0 at the edge.
fn kernel(local: i32, tile: i32) -> f32 {
    if tile <= 1 {
        return 1.0;
    }
    let mid = (tile - 1) as f32 * 0.5;
    1.0 - 0.999 * (local as f32 - mid).abs() / mid
}

fn splitmix(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// Deterministic Gaussian-ish in (-ish [0,1) mapped through inverse-erf-lite).
/// Column `x` depends only on `(seed, channel, x.div_euclid(tile_noise))`.
fn tiled_gaussian(seed: u64, channel: u32, x: i32, z: i32) -> f32 {
    const NOISE_TILE: i32 = 64;
    let tx = x.div_euclid(NOISE_TILE);
    let tz = z.div_euclid(NOISE_TILE);
    let lx = x.rem_euclid(NOISE_TILE);
    let lz = z.rem_euclid(NOISE_TILE);
    let h = splitmix(
        seed
            ^ (channel as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (tx as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (tz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F),
    );
    let cell = splitmix(h ^ (lx as u32 as u64) << 32 ^ lz as u32 as u64);
    // 24-bit uniform in [0,1). Good enough as the diffusion prior.
    (cell >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

fn value_noise(seed: u64, x: i32, z: i32, cell: u32) -> f32 {
    let cell = cell.max(1) as i32;
    let gx = x.div_euclid(cell);
    let gz = z.div_euclid(cell);
    let fx = x.rem_euclid(cell) as f32 / cell as f32;
    let fz = z.rem_euclid(cell) as f32 / cell as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let n = |ix: i32, iz: i32| {
        let h = splitmix(
            seed
                ^ (ix as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (iz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F),
        );
        (h >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    };
    let a = n(gx, gz);
    let b = n(gx + 1, gz);
    let c = n(gx, gz + 1);
    let d = n(gx + 1, gz + 1);
    let u = a + (b - a) * sx;
    let v = c + (d - c) * sx;
    u + (v - u) * sz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(seed: u64) -> InfiniteField<HashScore> {
        InfiniteField::new(Spec::new(seed), HashScore { seed })
    }

    #[test]
    fn seed_consistency() {
        let a = field(7);
        let b = field(7);
        let c = field(8);
        for z in -3..5 {
            for x in 10..18 {
                assert_eq!(a.sample(0, x, z).to_bits(), b.sample(0, x, z).to_bits());
                assert_ne!(a.sample(0, x, z).to_bits(), c.sample(0, x, z).to_bits());
            }
        }
    }

    #[test]
    fn query_order_does_not_change_values() {
        let a = field(11);
        let b = field(11);
        let mut seq_ab = Vec::new();
        for z in 0..40 {
            for x in 0..40 {
                seq_ab.push(a.sample(1, x, z).to_bits());
            }
        }
        let mut seq_ba = Vec::new();
        for z in (0..40).rev() {
            for x in (0..40).rev() {
                seq_ba.push(b.sample(1, x, z).to_bits());
            }
        }
        seq_ba.reverse();
        assert_eq!(seq_ab, seq_ba);
    }

    #[test]
    fn overlapping_queries_agree() {
        let f = field(3);
        let mut left = vec![0.0; 16 * 8];
        let mut right = vec![0.0; 16 * 8];
        f.fill_ch0(0, 0, 16, 8, &mut left);
        f.fill_ch0(8, 0, 16, 8, &mut right);
        for z in 0..8 {
            for x in 0..8 {
                let a = left[(z * 16 + (x + 8)) as usize];
                let b = right[(z * 16 + x) as usize];
                assert_eq!(a.to_bits(), b.to_bits(), "at {x},{z}");
            }
        }
    }

    #[test]
    fn fill_all_matches_sample_all() {
        let f = field(19);
        let (x0, z0, w, h) = (-4, 7, 16u32, 8u32);
        let ch = f.spec().channels;
        let mut buf = vec![0.0f32; (w * h * ch) as usize];
        f.fill_all(x0, z0, w, h, &mut buf);
        let mut point = vec![0.0f32; ch as usize];
        for dz in 0..h as i32 {
            for dx in 0..w as i32 {
                f.sample_all(x0 + dx, z0 + dz, &mut point);
                let base = ((dz as u32 * w + dx as u32) * ch) as usize;
                for c in 0..ch as usize {
                    assert_eq!(
                        buf[base + c].to_bits(),
                        point[c].to_bits(),
                        "at {},{} ch {c}",
                        x0 + dx,
                        z0 + dz
                    );
                }
            }
        }
    }

    #[test]
    fn cache_stays_capped() {
        let f = InfiniteField::new(
            Spec {
                seed: 1,
                tile: 8,
                stride: 8,
                phases: 1,
                channels: 1,
            },
            HashScore { seed: 1 },
        );
        for z in (0..2048).step_by(8) {
            for x in (0..2048).step_by(8) {
                let _ = f.sample(0, x, z);
            }
        }
        assert!(f.cached_tiles() <= TILE_CACHE_CAP);
    }

    #[test]
    fn samples_stay_finite() {
        let f = field(1);
        for z in [-1000, -1, 0, 17, 4096] {
            for x in [-2048, 0, 33] {
                let v = f.sample(0, x, z);
                assert!(v.is_finite());
            }
        }
    }
}
