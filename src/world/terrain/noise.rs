//! Deterministic gradient noise for the generator.
//!
//! Every peer regenerates terrain from the seed, so the arithmetic must be bit-identical on every
//! machine: only IEEE basic operations (`+ - * /`, `floor`, `sqrt`) on values computed in a fixed
//! order — no `sin`, `exp` or `powf`, whose results vary between math libraries. Lattice cells are
//! split in `f64` so coordinates near ±10⁹ stay exact; the smooth part runs in `f32`.

/// 32-bit mix of a lattice point (2-D).
#[inline]
pub fn hash2(seed: u32, x: i32, z: i32) -> u32 {
    let mut h = seed ^ 0x9E37_79B9;
    h ^= (x as u32).wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    h ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    fmix(h)
}

/// 32-bit mix of a lattice point (3-D).
#[inline]
pub fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let mut h = seed ^ 0x7F4A_7C15;
    h ^= (x as u32).wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xE654_6B64);
    h ^= (y as u32).wrapping_mul(0x27D4_EB2F);
    h = h.rotate_left(11).wrapping_mul(9).wrapping_add(0x1656_67B1);
    h ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    fmix(h)
}

#[inline]
fn fmix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// A hash as a fraction in `[0, 1)`.
#[inline]
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Sixteen unit gradients around the circle (constants, not computed with trigonometry).
const GRAD2: [(f32, f32); 16] = [
    (1.0, 0.0),
    (0.923_879_5, 0.382_683_43),
    (0.707_106_77, 0.707_106_77),
    (0.382_683_43, 0.923_879_5),
    (0.0, 1.0),
    (-0.382_683_43, 0.923_879_5),
    (-0.707_106_77, 0.707_106_77),
    (-0.923_879_5, 0.382_683_43),
    (-1.0, 0.0),
    (-0.923_879_5, -0.382_683_43),
    (-0.707_106_77, -0.707_106_77),
    (-0.382_683_43, -0.923_879_5),
    (0.0, -1.0),
    (0.382_683_43, -0.923_879_5),
    (0.707_106_77, -0.707_106_77),
    (0.923_879_5, -0.382_683_43),
];

/// Perlin's twelve cube-edge gradients (plus four repeats so a nibble indexes them).
const GRAD3: [(f32, f32, f32); 16] = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.0, -1.0, 1.0),
    (0.0, -1.0, -1.0),
];

#[inline]
fn split(p: f64) -> (i32, f32) {
    let f = p.floor();
    (f as i64 as i32, (p - f) as f32)
}

#[inline]
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn quintic_d(t: f32) -> f32 {
    30.0 * t * t * (t * (t - 2.0) + 1.0)
}

/// 2-D gradient noise at `(x, z)` (lattice units) with its analytic derivatives:
/// `(value ≈ −1..1, ∂/∂x, ∂/∂z)`.
pub fn perlin2_d(seed: u32, x: f64, z: f64) -> (f32, f32, f32) {
    let (ix, fx) = split(x);
    let (iz, fz) = split(z);
    let g = |dx: i32, dz: i32| GRAD2[(hash2(seed, ix.wrapping_add(dx), iz.wrapping_add(dz)) & 15) as usize];
    let (g00, g10, g01, g11) = (g(0, 0), g(1, 0), g(0, 1), g(1, 1));
    let v00 = g00.0 * fx + g00.1 * fz;
    let v10 = g10.0 * (fx - 1.0) + g10.1 * fz;
    let v01 = g01.0 * fx + g01.1 * (fz - 1.0);
    let v11 = g11.0 * (fx - 1.0) + g11.1 * (fz - 1.0);
    let (u, w) = (quintic(fx), quintic(fz));
    let (du, dw) = (quintic_d(fx), quintic_d(fz));
    let k0 = v00;
    let k1 = v10 - v00;
    let k2 = v01 - v00;
    let k3 = v11 - v01 - v10 + v00;
    let value = k0 + k1 * u + k2 * w + k3 * u * w;
    // Derivative of the gradient terms plus the interpolation weights.
    let gx = g00.0 + (g10.0 - g00.0) * u + (g01.0 - g00.0) * w + (g11.0 - g01.0 - g10.0 + g00.0) * u * w;
    let gz = g00.1 + (g10.1 - g00.1) * u + (g01.1 - g00.1) * w + (g11.1 - g01.1 - g10.1 + g00.1) * u * w;
    let dx = gx + du * (k1 + k3 * w);
    let dz = gz + dw * (k2 + k3 * u);
    (value * 1.414, dx * 1.414, dz * 1.414)
}

/// 2-D gradient noise, value only.
#[inline]
pub fn perlin2(seed: u32, x: f64, z: f64) -> f32 {
    perlin2_d(seed, x, z).0
}

/// 3-D gradient noise at `(x, y, z)` in lattice units, ≈ −1..1.
pub fn perlin3(seed: u32, x: f64, y: f64, z: f64) -> f32 {
    let (ix, fx) = split(x);
    let (iy, fy) = split(y);
    let (iz, fz) = split(z);
    let corner = |dx: i32, dy: i32, dz: i32| {
        let g = GRAD3[(hash3(seed, ix.wrapping_add(dx), iy.wrapping_add(dy), iz.wrapping_add(dz)) & 15) as usize];
        g.0 * (fx - dx as f32) + g.1 * (fy - dy as f32) + g.2 * (fz - dz as f32)
    };
    let (u, v, w) = (quintic(fx), quintic(fy), quintic(fz));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 1.1
}

/// Fractal sum of 2-D noise: `octaves` layers, each `lacunarity` × the frequency and `gain` × the
/// amplitude of the last, rotated so lattice artefacts do not line up. Normalized to ≈ −1..1.
pub fn fbm2(seed: u32, x: f64, z: f64, octaves: u32, gain: f32) -> f32 {
    let (mut px, mut pz) = (x, z);
    let (mut sum, mut amp, mut norm) = (0.0f32, 1.0f32, 0.0f32);
    for o in 0..octaves {
        sum += amp * perlin2(seed.wrapping_add(o.wrapping_mul(0x632B_E5AB)), px, pz);
        norm += amp;
        amp *= gain;
        let (rx, rz) = (0.8 * px - 0.6 * pz, 0.6 * px + 0.8 * pz);
        px = rx * 2.0 + 17.3;
        pz = rz * 2.0 - 41.1;
    }
    sum / norm
}

/// Eroded fractal terrain: octaves whose slope accumulates damp the finer ones, so slopes stay
/// smooth and gullied while peaks and valleys keep detail (the derivative-weighted fBm known from
/// real-time terrain work). Returns ≈ −1..1.
pub fn eroded2(seed: u32, x: f64, z: f64, octaves: u32) -> f32 {
    let (mut px, mut pz) = (x, z);
    let (mut sum, mut amp, mut norm) = (0.0f32, 1.0f32, 0.0f32);
    let (mut dx, mut dz) = (0.0f32, 0.0f32);
    for o in 0..octaves {
        let (n, nx, nz) = perlin2_d(seed.wrapping_add(o.wrapping_mul(0x2C1B_3C6D)), px, pz);
        dx += nx;
        dz += nz;
        sum += amp * n / (1.0 + dx * dx + dz * dz);
        norm += amp;
        amp *= 0.5;
        let (rx, rz) = (0.8 * px - 0.6 * pz, 0.6 * px + 0.8 * pz);
        px = rx * 2.0 + 9.7;
        pz = rz * 2.0 + 3.1;
    }
    sum / norm * 1.6
}

/// Ridged multifractal: sharp crests where the noise crosses zero, each octave weighted by the
/// previous crest so ridges branch along ridges. Returns 0..1 (1 = crest).
pub fn ridged2(seed: u32, x: f64, z: f64, octaves: u32) -> f32 {
    let (mut px, mut pz) = (x, z);
    let (mut sum, mut amp, mut norm) = (0.0f32, 1.0f32, 0.0f32);
    let mut weight = 1.0f32;
    for o in 0..octaves {
        let n = perlin2(seed.wrapping_add(o.wrapping_mul(0x5851_F42D)), px, pz);
        let mut r = 1.0 - n.abs();
        r *= r;
        r *= weight;
        weight = (r * 1.8).clamp(0.0, 1.0);
        sum += r * amp;
        norm += amp;
        amp *= 0.5;
        let (rx, rz) = (0.8 * px - 0.6 * pz, 0.6 * px + 0.8 * pz);
        px = rx * 2.0 - 7.9;
        pz = rz * 2.0 + 13.3;
    }
    sum / norm
}

/// Cubic smoothstep of `t` between `a` and `b`.
#[inline]
pub fn smoothstep(a: f32, b: f32, t: f32) -> f32 {
    let x = ((t - a) / (b - a)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_bounded_continuous_and_deterministic() {
        let mut max = 0.0f32;
        for i in 0..20_000 {
            let x = i as f64 * 0.137 - 900.0;
            let z = i as f64 * 0.071 + 33.0;
            let v = perlin2(7, x, z);
            assert_eq!(v, perlin2(7, x, z));
            max = max.max(v.abs());
            let a = perlin3(9, x, z * 0.5, -x);
            assert!(a.abs() <= 1.2, "{a}");
            let near = perlin2(7, x + 1e-3, z);
            assert!((near - v).abs() < 0.02, "continuity");
        }
        assert!(max > 0.5 && max <= 1.2, "{max}");
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for i in 0..400 {
            let (x, z) = (i as f64 * 0.31 + 0.13, i as f64 * -0.17 + 4.4);
            let (v, dx, dz) = perlin2_d(3, x, z);
            let h = 1e-3;
            let fx = (perlin2(3, x + h, z) - v) / h as f32;
            let fz = (perlin2(3, x, z + h) - v) / h as f32;
            assert!((fx - dx).abs() < 0.05 && (fz - dz).abs() < 0.05, "{dx} vs {fx}, {dz} vs {fz}");
        }
    }

    #[test]
    fn far_coordinates_stay_finite() {
        for &c in &[1e9f64, -1e9, 2_147_483_000.0] {
            assert!(eroded2(1, c, c, 6).is_finite());
            assert!(ridged2(1, c, -c, 5).is_finite());
            assert!(perlin3(1, c, 10.0, c).is_finite());
        }
    }
}
