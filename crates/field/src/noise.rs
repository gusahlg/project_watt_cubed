//! Integer (Q16.16) noise for field priors: the v2 `inoise` (ec10f71^) lifted to three dimensions.
//! Every formula is wrapping 32-bit integer math with i64 multiply transients, so every machine (and
//! a shader) computes the same bits.

/// 1.0 in Q16.16.
pub const ONE: i32 = 1 << 16;
/// 0.5 in Q16.16.
pub const HALF: i32 = 1 << 15;

/// Q16.16 product: arithmetic shift of the i64 product (rounds toward −∞).
#[inline]
pub fn mul_q16(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// Floor division for a positive divisor (0 for `b <= 0`).
#[inline]
pub fn div_floor(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return 0;
    }
    let (q, r) = (a / b, a % b);
    if r < 0 { q.wrapping_sub(1) } else { q }
}

/// Non-negative remainder paired with [`div_floor`]: `0..b` for `b > 0`.
#[inline]
pub fn rem_floor(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return 0;
    }
    let r = a % b;
    if r < 0 { r.wrapping_add(b) } else { r }
}

#[inline]
fn mix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// 32-bit hash of a 3-D lattice point. Each axis is folded in with a rotate-multiply so planes of
/// equal `x ^ y ^ z` products do not alias.
#[inline]
pub fn hash32_3(seed: u32, x: i32, y: i32, z: i32, salt: u32) -> u32 {
    let mut h = seed.wrapping_add(0x9E37_79B9).wrapping_add(salt.wrapping_mul(0x7F4A_7C15));
    h ^= (x as u32).wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    h ^= (y as u32).wrapping_mul(0x27D4_EB2F);
    h = h.rotate_left(11).wrapping_mul(9).wrapping_add(0x1656_67B1);
    h ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    mix32(h)
}

/// Low 16 bits of a hash as a Q16 fraction in `0..=65535`.
#[inline]
pub fn uniform_q16(hash: u32) -> i32 {
    (hash & 0xFFFF) as i32
}

/// `a + (b − a) · t` in Q16.
#[inline]
pub fn lerp_q16(a: i32, b: i32, t: i32) -> i32 {
    a.wrapping_add(mul_q16(b.wrapping_sub(a), t))
}

/// Cubic Hermite smoothstep on the unit interval (input clamped).
#[inline]
pub fn smoothstep_q16(t: i32) -> i32 {
    let t = t.clamp(0, ONE);
    mul_q16(mul_q16(t, t), 3 * ONE - (t + t))
}

/// Ridged transform `1 − |2n − 1|`: peaks where `n` is one half.
#[inline]
pub fn ridged_q16(n: i32) -> i32 {
    ONE - (n + n - ONE).abs()
}

/// Floor of `sqrt(n)`, digit by digit (32 steps).
pub fn isqrt_u64(n: u64) -> u32 {
    let (mut op, mut res, mut one) = (n, 0u64, 1u64 << 62);
    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }
    res as u32
}

/// Lattice cell and smoothstep weights of `p` for cell size `c`.
#[inline]
fn cell_weights(p: [i32; 3], c: i32) -> ([i32; 3], [i32; 3]) {
    (p.map(|v| div_floor(v, c)), p.map(|v| smoothstep_q16((((rem_floor(v, c) as i64) << 16) / c as i64) as i32)))
}

/// Trilinear blend of the eight corner values `n(dx, dy, dz)`.
#[inline]
fn blend(n: impl Fn(i32, i32, i32) -> i32, s: [i32; 3]) -> i32 {
    let x00 = lerp_q16(n(0, 0, 0), n(1, 0, 0), s[0]);
    let x10 = lerp_q16(n(0, 1, 0), n(1, 1, 0), s[0]);
    let x01 = lerp_q16(n(0, 0, 1), n(1, 0, 1), s[0]);
    let x11 = lerp_q16(n(0, 1, 1), n(1, 1, 1), s[0]);
    lerp_q16(lerp_q16(x00, x10, s[1]), lerp_q16(x01, x11, s[1]), s[2])
}

/// Trilinear value noise with smoothstep weights, output in `0..1` Q16.
pub fn value_noise3_q16(seed: u32, p: [i32; 3], cell: i32, salt: u32) -> i32 {
    let (g, s) = cell_weights(p, cell.max(1));
    blend(|dx, dy, dz| uniform_q16(hash32_3(seed, g[0].wrapping_add(dx), g[1].wrapping_add(dy), g[2].wrapping_add(dz), salt)), s)
}

/// Fractal sum of [`value_noise3_q16`], at most 4 octaves, normalised back to `0..1` Q16.
pub fn fbm3_q16(seed: u32, p: [i32; 3], cell: i32, octaves: u32, gain: i32, salt: u32) -> i32 {
    let (mut sum, mut norm, mut amp, mut c) = (0i64, 0i64, ONE, cell.max(1));
    for o in 0..octaves.min(4) {
        sum += mul_q16(value_noise3_q16(seed, p, c, salt.wrapping_add(o)), amp) as i64;
        norm += amp as i64;
        amp = mul_q16(amp, gain);
        c = (c / 2).max(1);
    }
    if norm == 0 { 0 } else { ((sum << 16) / norm) as i32 }
}

/// [`value_noise3_q16`] over a bounded box with its lattice hashed once: [`NoiseBox::at`] equals
/// the free function for every point of the box, at the cost of eight loads.
pub struct NoiseBox {
    cell: i32,
    lo: [i32; 3],
    n: [usize; 3],
    values: Vec<i32>,
}

impl NoiseBox {
    /// The lattice of cell size `cell` covering the points `min ..= max`.
    pub fn new(seed: u32, cell: i32, salt: u32, min: [i32; 3], max: [i32; 3]) -> Self {
        let cell = cell.max(1);
        let lo = min.map(|v| div_floor(v, cell));
        let n: [usize; 3] = std::array::from_fn(|a| (div_floor(max[a], cell) - lo[a] + 2) as usize);
        let mut values = Vec::with_capacity(n[0] * n[1] * n[2]);
        for z in 0..n[2] as i32 {
            for y in 0..n[1] as i32 {
                for x in 0..n[0] as i32 {
                    values.push(uniform_q16(hash32_3(seed, lo[0] + x, lo[1] + y, lo[2] + z, salt)));
                }
            }
        }
        Self { cell, lo, n, values }
    }

    /// The noise at `p` (inside the box).
    #[inline]
    pub fn at(&self, p: [i32; 3]) -> i32 {
        let (g, s) = cell_weights(p, self.cell);
        let base = [0, 1, 2].map(|a| (g[a] - self.lo[a]) as usize);
        let (sx, sxy) = (self.n[0], self.n[0] * self.n[1]);
        let at = base[0] + sx * base[1] + sxy * base[2];
        blend(|dx, dy, dz| self.values[at + dx as usize + sx * dy as usize + sxy * dz as usize], s)
    }
}

/// [`fbm3_q16`] over a bounded box: one [`NoiseBox`] per octave.
pub struct Fbm {
    octaves: Vec<NoiseBox>,
    gain: i32,
}

impl Fbm {
    /// The fractal noise of `seed` with base cell `cell` over `min ..= max`.
    pub fn new(seed: u32, cell: i32, octaves: u32, gain: i32, salt: u32, min: [i32; 3], max: [i32; 3]) -> Self {
        let mut c = cell.max(1);
        let octaves = (0..octaves.min(4))
            .map(|o| {
                let b = NoiseBox::new(seed, c, salt.wrapping_add(o), min, max);
                c = (c / 2).max(1);
                b
            })
            .collect();
        Self { octaves, gain }
    }

    /// The noise at `p` (inside the box).
    #[inline]
    pub fn at(&self, p: [i32; 3]) -> i32 {
        let (mut sum, mut norm, mut amp) = (0i64, 0i64, ONE);
        for b in &self.octaves {
            sum += mul_q16(b.at(p), amp) as i64;
            norm += amp as i64;
            amp = mul_q16(amp, self.gain);
        }
        if norm == 0 { 0 } else { ((sum << 16) / norm) as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_stays_in_range_and_is_continuous() {
        let mut prev = value_noise3_q16(9, [0, 5, -3], 16, 1);
        for x in 1..200 {
            let v = value_noise3_q16(9, [x, 5, -3], 16, 1);
            assert!((0..ONE).contains(&v));
            assert!((v - prev).abs() < ONE / 4, "jump at {x}");
            prev = v;
        }
        let f = fbm3_q16(3, [i32::MAX - 40, i32::MIN + 7, 0], 16, 4, HALF, 0);
        assert!((0..ONE).contains(&f));
    }

    #[test]
    fn boxed_noise_equals_the_free_functions() {
        let (min, max) = ([-300, -5, 1000], [260, 77, 1400]);
        let b = NoiseBox::new(5, 37, 3, min, max);
        let f = Fbm::new(5, 64, 4, HALF, 9, min, max);
        for k in 0..2000i32 {
            let p = [min[0] + k * 7 % 561, min[1] + k * 13 % 83, min[2] + k * 31 % 401];
            assert_eq!(b.at(p), value_noise3_q16(5, p, 37, 3), "{p:?}");
            assert_eq!(f.at(p), fbm3_q16(5, p, 64, 4, HALF, 9), "{p:?}");
        }
    }

    #[test]
    fn small_helpers() {
        assert_eq!(isqrt_u64(15), 3);
        assert_eq!(isqrt_u64(1 << 32), 1 << 16);
        assert_eq!(isqrt_u64(u64::MAX), u32::MAX);
        assert_eq!(smoothstep_q16(HALF), HALF);
        assert_eq!(ridged_q16(HALF), ONE);
        for b in [1, 7, 32] {
            for a in [-40, -1, 0, 31, 100] {
                assert_eq!((div_floor(a, b), rem_floor(a, b)), (a.div_euclid(b), a.rem_euclid(b)));
            }
        }
    }
}
