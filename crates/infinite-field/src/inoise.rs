//! Integer (Q16.16) noise primitives for the v2 field. Every formula is a
//! 32-bit wrapping integer expression so a SPIR-V mirror can transliterate it.
//!
//! Q16.16: `ONE = 1 << 16` is 1.0. Multiplies use a transient i64, then
//! arithmetic-shift back (`(a as i64 * b as i64) >> 16` as i32). `%` is Rust
//! `SRem` (SPIR-V `OpSRem`); `/` is trunc-toward-zero (SPIR-V `OpSDiv`).
//! Shift amounts are constants in `0..32`. Division is guarded against 0.
//!
//! | name | inputs | exact integer expression |
//! |---|---|---|
//! | `ONE` | — | `65536` |
//! | `HALF` | — | `32768` |
//! | `mul_q16(a,b)` | i32, i32 | `((a as i64 * b as i64) >> 16) as i32` (ASR) |
//! | `div_floor(a,b)` | i32, i32 (`b>0`) | `q=a/b; r=a%b; r<0 ? q-1 : q` |
//! | `rem_floor(a,b)` | i32, i32 (`b>0`) | `r=a%b; r<0 ? r+b : r` |
//! | `mix32(h)` | u32 | `h^=h>>16; h*=0x7FEB352D; h^=h>>15; h*=0x846CA68B; h^=h>>16` |
//! | `hash32` | seed, x, z, salt | `h=seed+0x9E3779B9+salt*0x7F4A7C15; h^=(x as u32)*0x85EBCA6B; h^=(z as u32)*0xC2B2AE35; mix32(h)` |
//! | `uniform_q16(h)` | u32 | `(h & 0xFFFF) as i32`  (range `0..=65535`) |
//! | `clamp_q16(v,lo,hi)` | i32×3 | `v<lo ? lo : (v>hi ? hi : v)` |
//! | `lerp_q16(a,b,t)` | i32×3 | `a + mul_q16(b-a, t)` |
//! | `smoothstep_q16(t)` | i32 | `t=clamp(t,0,ONE); mul_q16(mul_q16(t,t), 3*ONE - (t+t))` |
//! | `value_noise_q16` | seed,x,z,cell,salt | lattice `gx,gz=div_floor(x,c)`, `fx=((rx<<16)/c)` with `c=cell<=0?1:cell`; bilinear of four `uniform_q16(hash32(seed,gx(+1),gz(+1),salt))` with `sx=smoothstep(fx)` |
//! | `fbm_q16` | …, octaves≤4, gain, salt | `sum=0; amp=ONE; c=cell; o=0..min(octaves,4): sum+=mul_q16(value_noise(seed,x,z,c,salt+o),amp); amp=mul_q16(amp,gain); c=c<=1?1:c/2` |
//! | `ridged_q16(n)` | i32 | `ONE - abs(n+n-ONE)` |
//! | `isqrt_u64(n)` | u64 | digit-by-digit, 32 steps, `one=1<<62`; `floor(sqrt(n))` as u32 |
//! | `cellular_q16` | seed,x,z,cell,salt | 3×3 jittered lattice; feature `(dx*ONE+jx, dz*ONE+jz)`; Euclidean `isqrt(ddx*ddx+ddz*ddz)` in Q16; F1 nearest, F2 second, `id=hash32` of winner; ties: first in `dz,dx ∈ [-1,1]` |
//! | `warp_q16` | seed,x,z,cell,amp,salt | `(mul_q16(vn(salt)-HALF, amp), mul_q16(vn(salt+1)-HALF, amp))` — Q16 offsets |
//! | `gradient_q16` | f,x,z,step | `s=step==0?1:step; d=s+s; d==0?(0,0): ((f(x+s,z)-f(x-s,z))/d, (f(x,z+s)-f(x,z-s))/d)` (SDiv) |

/// 1.0 in Q16.16.
pub const ONE: i32 = 1 << 16;
/// 0.5 in Q16.16.
pub const HALF: i32 = 1 << 15;

/// Q16.16 product: arithmetic-shift of the i64 product (rounds toward −∞).
#[inline]
pub fn mul_q16(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// Floor division for a positive divisor. Matches Euclidean `div_euclid` for
/// `b > 0`; expressed with `OpSDiv`/`OpSRem` so a shader can copy it.
#[inline]
pub fn div_floor(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return 0;
    }
    let q = a / b;
    let r = a % b;
    if r < 0 {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// Non-negative remainder paired with [`div_floor`]: `0..b` for `b > 0`.
#[inline]
pub fn rem_floor(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return 0;
    }
    let r = a % b;
    if r < 0 {
        r.wrapping_add(b)
    } else {
        r
    }
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

/// Splitmix/murmur-family 32-bit mixer. Wrapping; no platform variants.
#[inline]
pub fn hash32(seed: u32, x: i32, z: i32, salt: u32) -> u32 {
    let mut h = seed.wrapping_add(0x9E37_79B9);
    h = h.wrapping_add(salt.wrapping_mul(0x7F4A_7C15));
    h ^= (x as u32).wrapping_mul(0x85EB_CA6B);
    h ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    mix32(h)
}

/// Low 16 bits of a hash as a Q16 fraction in `0..=65535` (never quite 1.0).
#[inline]
pub fn uniform_q16(hash: u32) -> i32 {
    (hash & 0xFFFF) as i32
}

/// Clamp `v` into `[lo, hi]`. If `lo > hi`, returns `lo`.
#[inline]
pub fn clamp_q16(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `a + (b − a) · t` in Q16.
#[inline]
pub fn lerp_q16(a: i32, b: i32, t: i32) -> i32 {
    a.wrapping_add(mul_q16(b.wrapping_sub(a), t))
}

/// Cubic Hermite smoothstep on the unit interval.
#[inline]
pub fn smoothstep_q16(t: i32) -> i32 {
    let t = clamp_q16(t, 0, ONE);
    let t2 = mul_q16(t, t);
    let inner = ONE.wrapping_mul(3).wrapping_sub(t.wrapping_add(t));
    mul_q16(t2, inner)
}

#[inline]
fn cell_or_one(cell: i32) -> i32 {
    if cell <= 0 {
        1
    } else {
        cell
    }
}

/// Bilinear value noise, output in `0..1` Q16.
pub fn value_noise_q16(seed: u32, x: i32, z: i32, cell: i32, salt: u32) -> i32 {
    let c = cell_or_one(cell);
    let gx = div_floor(x, c);
    let gz = div_floor(z, c);
    let fx = (((rem_floor(x, c) as i64) << 16) / (c as i64)) as i32;
    let fz = (((rem_floor(z, c) as i64) << 16) / (c as i64)) as i32;
    let sx = smoothstep_q16(fx);
    let sz = smoothstep_q16(fz);
    let n00 = uniform_q16(hash32(seed, gx, gz, salt));
    let n10 = uniform_q16(hash32(seed, gx.wrapping_add(1), gz, salt));
    let n01 = uniform_q16(hash32(seed, gx, gz.wrapping_add(1), salt));
    let n11 = uniform_q16(hash32(seed, gx.wrapping_add(1), gz.wrapping_add(1), salt));
    let u = lerp_q16(n00, n10, sx);
    let v = lerp_q16(n01, n11, sx);
    lerp_q16(u, v, sz)
}

/// Fractal Brownian motion, at most 4 octaves. Un-normalised (amp starts at 1).
pub fn fbm_q16(
    seed: u32,
    x: i32,
    z: i32,
    cell: i32,
    octaves: u32,
    gain_q16: i32,
    salt: u32,
) -> i32 {
    let n = if octaves > 4 { 4 } else { octaves };
    let mut sum = 0i32;
    let mut amp = ONE;
    let mut c = cell_or_one(cell);
    let mut o = 0u32;
    while o < n {
        let v = value_noise_q16(seed, x, z, c, salt.wrapping_add(o));
        sum = sum.wrapping_add(mul_q16(v, amp));
        amp = mul_q16(amp, gain_q16);
        c = if c <= 1 { 1 } else { c / 2 };
        o = o.wrapping_add(1);
    }
    sum
}

/// Ridged transform: `1 − |2n − 1|`.
#[inline]
pub fn ridged_q16(n: i32) -> i32 {
    let t = n.wrapping_add(n).wrapping_sub(ONE);
    let a = if t < 0 { t.wrapping_neg() } else { t };
    ONE.wrapping_sub(a)
}

/// Floor of `sqrt(n)`. Digit-by-digit, 32 steps, `one` starts at `1 << 62`.
pub fn isqrt_u64(n: u64) -> u32 {
    let mut op = n;
    let mut res: u64 = 0;
    let mut one: u64 = 1 << 62;
    let mut i = 0u32;
    while i < 32 {
        let t = res.wrapping_add(one);
        if op >= t {
            op = op.wrapping_sub(t);
            res = (res >> 1).wrapping_add(one);
        } else {
            res >>= 1;
        }
        one >>= 2;
        i = i.wrapping_add(1);
    }
    res as u32
}

/// Jittered-grid Voronoi over a 3×3 neighbourhood.
///
/// Distances are Euclidean in Q16 cell-units (1.0 = one lattice cell).
/// `id` is `hash32` of the nearest cell. Ties keep the first cell in
/// `dz, dx ∈ [-1, 1]` order.
pub fn cellular_q16(seed: u32, x: i32, z: i32, cell: i32, salt: u32) -> (i32, i32, u32) {
    let c = cell_or_one(cell);
    let gx = div_floor(x, c);
    let gz = div_floor(z, c);
    let lx = (((rem_floor(x, c) as i64) << 16) / (c as i64)) as i32;
    let lz = (((rem_floor(z, c) as i64) << 16) / (c as i64)) as i32;
    let mut f1 = i32::MAX;
    let mut f2 = i32::MAX;
    let mut id = 0u32;
    let mut dz = -1i32;
    while dz <= 1 {
        let mut dx = -1i32;
        while dx <= 1 {
            let cx = gx.wrapping_add(dx);
            let cz = gz.wrapping_add(dz);
            let h = hash32(seed, cx, cz, salt);
            let jx = uniform_q16(h);
            let jz = uniform_q16(hash32(seed, cx, cz, salt.wrapping_add(1)));
            let px = dx.wrapping_mul(ONE).wrapping_add(jx);
            let pz = dz.wrapping_mul(ONE).wrapping_add(jz);
            let ddx = px.wrapping_sub(lx) as i64;
            let ddz = pz.wrapping_sub(lz) as i64;
            let dist2 = ddx * ddx + ddz * ddz;
            let dist = isqrt_u64(if dist2 < 0 { 0 } else { dist2 as u64 }) as i32;
            if dist < f1 {
                f2 = f1;
                f1 = dist;
                id = h;
            } else if dist < f2 {
                f2 = dist;
            }
            dx = dx.wrapping_add(1);
        }
        dz = dz.wrapping_add(1);
    }
    (f1, f2, id)
}

/// Two value-noises as a Q16 domain-warp offset: `(n − 0.5) · amp` on each axis.
pub fn warp_q16(seed: u32, x: i32, z: i32, cell: i32, amp_q16: i32, salt: u32) -> (i32, i32) {
    let nx = value_noise_q16(seed, x, z, cell, salt);
    let nz = value_noise_q16(seed, x, z, cell, salt.wrapping_add(1));
    (
        mul_q16(nx.wrapping_sub(HALF), amp_q16),
        mul_q16(nz.wrapping_sub(HALF), amp_q16),
    )
}

/// Central differences of any Q16 function. `step == 0` is treated as 1.
pub fn gradient_q16<F: Fn(i32, i32) -> i32>(f: F, x: i32, z: i32, step: i32) -> (i32, i32) {
    let s = if step == 0 { 1 } else { step };
    let d = s.wrapping_add(s);
    if d == 0 {
        return (0, 0);
    }
    let dx = f(x.wrapping_add(s), z).wrapping_sub(f(x.wrapping_sub(s), z));
    let dz = f(x, z.wrapping_add(s)).wrapping_sub(f(x, z.wrapping_sub(s)));
    (dx / d, dz / d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shader-contract inputs. Cell is the lattice size for spatial noises.
    const GOLDEN_IN: [(u32, i32, i32, u32, i32); 8] = [
        (0, 0, 0, 0, 8),
        (1, 0, 0, 0, 8),
        (42, 16, -8, 7, 16),
        (0x1234_5678, 100, 200, 1, 32),
        (u32::MAX, -1, -1, u32::MAX, 8),
        (7, 1_000_000, -1_000_000, 3, 64),
        (99, i32::MAX, 0, 5, 8),
        (13, -40, 40, 2, 24),
    ];

    /// Pins: `(hash32, value_noise_q16, cellular f1, f2, id, fbm_q16)`.
    /// FBM uses 4 octaves and gain = HALF. Shader contract — do not regenerate
    /// casually; W7's SPIR-V mirror must match these bits.
    const GOLDEN_OUT: [(u32, i32, i32, i32, u32, i32); 8] = [
        (33350994, 58706, 40258, 43357, 2147850067, 83168),
        (2672842292, 22068, 12984, 54734, 2525366996, 51614),
        (4003907969, 62763, 29796, 44277, 4093167359, 107492),
        (3692607777, 15243, 48703, 48979, 1207603441, 42126),
        (3492534658, 12555, 20152, 27899, 1124150939, 43066),
        (3387476348, 55161, 10261, 26275, 3445741547, 77974),
        (2775162672, 42170, 49644, 61398, 2014281720, 60904),
        (3042848206, 26129, 53348, 54033, 2864127302, 57742),
    ];

    fn eval_golden(seed: u32, x: i32, z: i32, salt: u32, cell: i32) -> (u32, i32, i32, i32, u32, i32) {
        let h = hash32(seed, x, z, salt);
        let vn = value_noise_q16(seed, x, z, cell, salt);
        let (f1, f2, id) = cellular_q16(seed, x, z, cell, salt);
        let fbm = fbm_q16(seed, x, z, cell, 4, HALF, salt);
        (h, vn, f1, f2, id, fbm)
    }

    #[test]
    fn inoise_golden() {
        let mut got = [(0u32, 0i32, 0i32, 0i32, 0u32, 0i32); 8];
        for (i, &(seed, x, z, salt, cell)) in GOLDEN_IN.iter().enumerate() {
            got[i] = eval_golden(seed, x, z, salt, cell);
        }
        if got != GOLDEN_OUT {
            for (i, g) in got.iter().enumerate() {
                eprintln!(
                    "GOLDEN[{i}] hash={} vn={} f1={} f2={} id={} fbm={}",
                    g.0, g.1, g.2, g.3, g.4, g.5
                );
            }
            panic!("update GOLDEN_OUT pins from the dump above");
        }
    }

    #[test]
    fn hash32_is_deterministic() {
        for &(seed, x, z, salt, _) in &GOLDEN_IN {
            assert_eq!(hash32(seed, x, z, salt), hash32(seed, x, z, salt));
        }
    }

    #[test]
    fn uniform_q16_range() {
        assert_eq!(uniform_q16(0), 0);
        assert_eq!(uniform_q16(0xFFFF), 65535);
        assert_eq!(uniform_q16(0x1_0000), 0);
    }

    #[test]
    fn smoothstep_endpoints() {
        assert_eq!(smoothstep_q16(0), 0);
        assert_eq!(smoothstep_q16(ONE), ONE);
        assert_eq!(smoothstep_q16(HALF), HALF);
        assert_eq!(smoothstep_q16(-ONE), 0);
        assert_eq!(smoothstep_q16(2 * ONE), ONE);
    }

    #[test]
    fn lerp_endpoints() {
        assert_eq!(lerp_q16(10, 90, 0), 10);
        assert_eq!(lerp_q16(10, 90, ONE), 90);
    }

    #[test]
    fn ridged_peaks_at_half() {
        assert_eq!(ridged_q16(0), 0);
        assert_eq!(ridged_q16(HALF), ONE);
        assert_eq!(ridged_q16(ONE), 0);
    }

    #[test]
    fn isqrt_known() {
        assert_eq!(isqrt_u64(0), 0);
        assert_eq!(isqrt_u64(1), 1);
        assert_eq!(isqrt_u64(3), 1);
        assert_eq!(isqrt_u64(4), 2);
        assert_eq!(isqrt_u64(15), 3);
        assert_eq!(isqrt_u64(1 << 32), 1 << 16);
        assert_eq!(isqrt_u64(ONE as u64 * ONE as u64), ONE as u32);
    }

    #[test]
    fn div_floor_matches_euclid_for_positive_b() {
        for b in [1i32, 2, 7, 16, 32] {
            for a in [-40i32, -17, -1, 0, 1, 15, 16, 31, 100, i32::MAX - 40] {
                assert_eq!(div_floor(a, b), a.div_euclid(b), "div {a}/{b}");
                assert_eq!(rem_floor(a, b), a.rem_euclid(b), "rem {a}/{b}");
            }
        }
        assert_eq!(div_floor(5, 0), 0);
        assert_eq!(rem_floor(5, -3), 0);
    }

    #[test]
    fn value_noise_far_coords_do_not_panic() {
        let _ = value_noise_q16(1, i32::MAX - 40, -1_000_000_000, 8, 0);
        let _ = cellular_q16(1, 1_000_000_000, -1_000_000_000, 32, 3);
        let _ = fbm_q16(1, i32::MAX - 40, 0, 8, 4, HALF, 0);
        let _ = warp_q16(1, i32::MAX - 40, 0, 8, ONE, 0);
    }

    #[test]
    fn gradient_of_linear_is_one() {
        // f(x,z) = x; step 1 → (f(x+1)-f(x-1))/2 = 1.
        let (gx, gz) = gradient_q16(|x, _z| x, 10, 3, 1);
        assert_eq!(gx, 1);
        assert_eq!(gz, 0);
        let (gx, gz) = gradient_q16(|_x, z| z, 10, 3, 2);
        assert_eq!(gx, 0);
        assert_eq!(gz, 1);
        let _ = gradient_q16(|x, z| x.wrapping_add(z), i32::MAX - 40, 0, 0);
    }

    #[test]
    fn fbm_zero_octaves_is_zero() {
        assert_eq!(fbm_q16(1, 0, 0, 8, 0, HALF, 0), 0);
    }
}
