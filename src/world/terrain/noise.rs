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

/// Feature points stay in `[0.3, 0.7)` of their cell, so anything nearer than this
/// (squared, lattice units) cannot lie outside the 3×3×3 around the query.
const CELL_INNER: f64 = 1.29 * 1.29;

/// 3-D cellular noise at `p` in lattice units. `(F1, F2, id1, id2)` are the distances to the
/// nearest and second-nearest feature points and the ids of the cells that own them.
/// Feature points sit in the inner 40 % of each cell, so the nearest is always inside the
/// 3×3×3; the ring one cell further out is searched only when the second-nearest might be there.
pub fn cellular3(seed: u32, p: [f64; 3]) -> (f32, f32, u32, u32) {
    let (ix, iy, iz) = (floor_i(p[0]), floor_i(p[1]), floor_i(p[2]));
    let (mut best, mut second) = (Cand::NONE, Cand::NONE);
    NEAR.with_borrow_mut(|slots| {
        let key = (seed, ix, iy, iz);
        let slot = &mut slots[(hash3(seed, ix, iy, iz) & (NEAR_SLOTS as u32 - 1)) as usize];
        if slot.as_ref().is_none_or(|n| n.key != key) {
            *slot = Some(Neighbourhood::new(key));
        }
        let Some(near) = slot.as_mut() else { return };
        scan(&near.inner, p, &mut best, &mut second);
        if second.d2 > CELL_INNER {
            let ring = near.ring.get_or_insert_with(|| {
                let mut ring = Box::new([Feat::NONE; RING]);
                cells(seed, ix, iy, iz, 2, 1, &mut ring[..]);
                ring
            });
            scan(&ring[..], p, &mut best, &mut second);
        }
    });
    (best.d2.sqrt() as f32, second.d2.sqrt() as f32, best.id, second.id)
}

/// Cells around the query cell: the 3×3×3, and the ring out to 5×5×5.
const INNER: usize = 27;
const RING: usize = 125 - INNER;

/// Cached neighbourhoods per thread, direct-mapped. Province and region lookups alternate.
const NEAR_SLOTS: usize = 4;

/// One lattice cell's id and feature point.
#[derive(Clone, Copy)]
struct Feat {
    id: u32,
    at: [f64; 3],
}

impl Feat {
    const NONE: Self = Self { id: 0, at: [0.0; 3] };
}

/// The cells around one lattice cell in search order: the 3×3×3, then the ring once a query needs
/// it. A chunk's or a section's columns share their lattice cell, so its hashes are paid once.
struct Neighbourhood {
    key: (u32, i32, i32, i32),
    inner: [Feat; INNER],
    ring: Option<Box<[Feat; RING]>>,
}

impl Neighbourhood {
    fn new((seed, ix, iy, iz): (u32, i32, i32, i32)) -> Self {
        let mut inner = [Feat::NONE; INNER];
        // `-1` skips nothing: the centre cell is a candidate.
        cells(seed, ix, iy, iz, 1, -1, &mut inner);
        Self { key: (seed, ix, iy, iz), inner, ring: None }
    }
}

thread_local! {
    static NEAR: std::cell::RefCell<[Option<Neighbourhood>; NEAR_SLOTS]> = const { std::cell::RefCell::new([const { None }; NEAR_SLOTS]) };
}

#[derive(Clone, Copy)]
struct Cand {
    d2: f64,
    id: u32,
}

impl Cand {
    const NONE: Self = Self { d2: f64::MAX, id: u32::MAX };
    fn nearer(self, d2: f64, id: u32) -> bool {
        d2 < self.d2 || (d2 == self.d2 && id < self.id)
    }
}

fn floor_i(p: f64) -> i32 {
    p.floor() as i64 as i32
}

/// The cells within `reach` of the query cell and outside `inner` of it, in a fixed order.
fn cells(seed: u32, ix: i32, iy: i32, iz: i32, reach: i32, inner: i32, out: &mut [Feat]) {
    let mut n = 0;
    for dz in -reach..=reach {
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                if dx.abs() <= inner && dy.abs() <= inner && dz.abs() <= inner {
                    continue;
                }
                let (x, y, z) = (ix.wrapping_add(dx), iy.wrapping_add(dy), iz.wrapping_add(dz));
                out[n] = Feat { id: hash3(seed ^ 0xCE11_1D00, x, y, z), at: feature(seed, x, y, z) };
                n += 1;
            }
        }
    }
    debug_assert_eq!(n, out.len());
}

/// Fold `cells` into the nearest and second-nearest candidates, in order.
fn scan(cells: &[Feat], p: [f64; 3], best: &mut Cand, second: &mut Cand) {
    for c in cells {
        let (ax, ay, az) = (c.at[0] - p[0], c.at[1] - p[1], c.at[2] - p[2]);
        let d2 = ax * ax + ay * ay + az * az;
        if best.nearer(d2, c.id) {
            *second = *best;
            *best = Cand { d2, id: c.id };
        } else if c.id != best.id && second.nearer(d2, c.id) {
            *second = Cand { d2, id: c.id };
        }
    }
}

/// Jittered feature point of one lattice cell, in lattice units. The jitter stays inside
/// `[0.3, 0.7)`, which is what makes the 3×3×3 search exact for the nearest point.
fn feature(seed: u32, x: i32, y: i32, z: i32) -> [f64; 3] {
    let j = |salt: u32| f64::from(0.3 + 0.4 * unit(hash3(seed ^ salt, x, y, z)));
    [f64::from(x) + j(0xA11C_E001), f64::from(y) + j(0xB011_D002), f64::from(z) + j(0xC0DE_D003)]
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-cache search, verbatim: every cell hashed per query.
    #[allow(clippy::too_many_arguments)]
    fn search(seed: u32, p: [f64; 3], ix: i32, iy: i32, iz: i32, reach: i32, inner: i32, best: &mut Cand, second: &mut Cand) {
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    if dx.abs() <= inner && dy.abs() <= inner && dz.abs() <= inner {
                        continue;
                    }
                    let (x, y, z) = (ix.wrapping_add(dx), iy.wrapping_add(dy), iz.wrapping_add(dz));
                    let id = hash3(seed ^ 0xCE11_1D00, x, y, z);
                    let feat = feature(seed, x, y, z);
                    let (ax, ay, az) = (feat[0] - p[0], feat[1] - p[1], feat[2] - p[2]);
                    let d2 = ax * ax + ay * ay + az * az;
                    if best.nearer(d2, id) {
                        *second = *best;
                        *best = Cand { d2, id };
                    } else if id != best.id && second.nearer(d2, id) {
                        *second = Cand { d2, id };
                    }
                }
            }
        }
    }

    fn cellular3_reference(seed: u32, p: [f64; 3]) -> (f32, f32, u32, u32) {
        let (ix, iy, iz) = (floor_i(p[0]), floor_i(p[1]), floor_i(p[2]));
        let (mut best, mut second) = (Cand::NONE, Cand::NONE);
        search(seed, p, ix, iy, iz, 1, -1, &mut best, &mut second);
        if second.d2 > CELL_INNER {
            search(seed, p, ix, iy, iz, 2, 1, &mut best, &mut second);
        }
        (best.d2.sqrt() as f32, second.d2.sqrt() as f32, best.id, second.id)
    }

    /// The cache returns bit for bit what hashing every cell returns, across interleaved seeds,
    /// neighbouring and distant cells, and points near ±10⁹.
    #[test]
    fn cached_cellular_matches_the_reference() {
        for i in 0..200_000u32 {
            let seed = [7, 0x51E0_0003, 0xDEAD_BEEF][(i % 3) as usize];
            let t = f64::from(i);
            let p = match i % 4 {
                0 => [t * 0.0137, -t * 0.0071, t * 0.0029],
                1 => [1e9 + t * 0.37, -1e9 + t * 0.11, 5e8 - t * 0.23],
                2 => [(t * 0.61).sin() * 40.0, t * 1e-3, -(t * 0.17).cos() * 40.0],
                _ => [t * 3.1, t * -2.7, t * 1.3],
            };
            let (a, b) = (cellular3(seed, p), cellular3_reference(seed, p));
            assert!(
                a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits() && a.2 == b.2 && a.3 == b.3,
                "{seed:#x} {p:?}: {a:?} vs {b:?}"
            );
        }
    }

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
            let (f1, f2, _, _) = cellular3(1, [c / 5_000.0, 0.4, -c / 5_000.0]);
            assert!(f1.is_finite() && f2.is_finite() && f1 <= f2 + 1e-4);
        }
    }

    #[test]
    fn cellular_is_ordered_deterministic_and_continuous() {
        let mut differed = false;
        for i in 0..800 {
            let p = [i as f64 * 0.173 - 40.0, i as f64 * 0.047 + 2.2, i as f64 * -0.091 - 7.0];
            let (f1, f2, a, b) = cellular3(11, p);
            assert!(f1 >= 0.0 && f1 <= f2 + 1e-5, "{f1} {f2}");
            assert_eq!((f1, f2, a, b), cellular3(11, p));
            differed |= a != b;
            let (g1, _, _, _) = cellular3(11, [p[0] + 1.0e-3, p[1], p[2]]);
            assert!((g1 - f1).abs() < 2.0e-3, "F1 lipschitz {f1} vs {g1}");
        }
        assert!(differed, "nearest and second share an id everywhere");
    }

    /// The bounded search agrees with an exhaustive ±3 walk (the proof's neighbourhood).
    #[test]
    fn cellular_matches_a_wide_walk() {
        fn wide(seed: u32, p: [f64; 3]) -> (f64, f64, u32, u32) {
            let (ix, iy, iz) = (super::floor_i(p[0]), super::floor_i(p[1]), super::floor_i(p[2]));
            let mut best = (f64::MAX, u32::MAX);
            let mut second = (f64::MAX, u32::MAX);
            for dz in -3..=3 {
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let (x, y, z) = (ix.wrapping_add(dx), iy.wrapping_add(dy), iz.wrapping_add(dz));
                        let id = hash3(seed ^ 0xCE11_1D00, x, y, z);
                        let f = super::feature(seed, x, y, z);
                        let (ax, ay, az) = (f[0] - p[0], f[1] - p[1], f[2] - p[2]);
                        let d2 = ax * ax + ay * ay + az * az;
                        let nearer = |cur: (f64, u32)| d2 < cur.0 || (d2 == cur.0 && id < cur.1);
                        if nearer(best) {
                            second = best;
                            best = (d2, id);
                        } else if id != best.1 && nearer(second) {
                            second = (d2, id);
                        }
                    }
                }
            }
            (best.0.sqrt(), second.0.sqrt(), best.1, second.1)
        }
        for i in 0..200 {
            let p = [i as f64 * 0.31 - 8.0, i as f64 * -0.17 + 3.3, i as f64 * 0.09 - 1.0];
            let (f1, f2, a, b) = cellular3(3, p);
            let (g1, g2, c, d) = wide(3, p);
            assert_eq!(a, c, "id1 at {p:?}");
            assert_eq!(b, d, "id2 at {p:?}");
            assert!((f64::from(f1) - g1).abs() < 1e-4 && (f64::from(f2) - g2).abs() < 1e-4);
        }
    }
}
