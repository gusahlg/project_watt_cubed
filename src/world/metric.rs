//! Distance metrics for LOD selection. Invariants (finite, ordered, pre-clamped)
//! are enforced in constructors to avoid scattered validation. The altitude term
//! `dy` may be zero (when the eye is level with terrain) or nonzero (when raised
//! above or below).

use voxel_engine::DVec3;

use super::section::SectionPos;

/// A player→cell distance in metres: always finite and `>= 0`. Non-finite or
/// negative inputs collapse to 0 (safe fallback for invalid eye data).
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub(in crate::world) struct EyeDist(f32);

impl EyeDist {
    pub fn new(d: f32) -> EyeDist {
        EyeDist(if d.is_finite() && d >= 0.0 { d } else { 0.0 })
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// A cell's nearest and farthest distance to the eye (`near <= far` by construction).
/// Ranges allow band membership to straddle boundaries.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct DistRange {
    near: EyeDist,
    far: EyeDist,
}

impl DistRange {
    pub fn new(near: EyeDist, far: EyeDist) -> DistRange {
        debug_assert!(near.0 <= far.0, "DistRange near {} > far {}", near.0, far.0);
        DistRange { near, far }
    }
    pub fn near(self) -> EyeDist {
        self.near
    }
    pub fn far(self) -> EyeDist {
        self.far
    }
}

/// A cell's vertical extent `[lo, hi]` in world-Y — the reference for altitude measurements.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct HeightEnvelope {
    lo: f32,
    hi: f32,
}

impl HeightEnvelope {
    pub fn new(lo: f32, hi: f32) -> HeightEnvelope {
        debug_assert!(lo <= hi, "HeightEnvelope lo {lo} > hi {hi}");
        HeightEnvelope { lo, hi }
    }
}

/// Altitude clamp: caps `dy` to prevent the coarsest ring's annulus from collapsing.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct DyCap(f32);

impl DyCap {
    pub fn new(outer_m: f32, base: f32) -> DyCap {
        DyCap(outer_m * (1.0 - 1.0 / base))
    }
}

/// Ground-plane projection of a distance band's outer radius. `None` when the band
/// sits entirely overhead (contributes no terrain to LOD selection).
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct XzAnnulus {
    hi: f32,
}

impl XzAnnulus {
    pub fn hi(self) -> f32 {
        self.hi
    }
}

/// Per-frame eye position and altitude offset for LOD selection. Combines horizontal
/// coordinates with a clamped vertical offset for distance-based LOD decisions.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct EyeMetric {
    ex: f64,
    ez: f64,
    /// Raw eye altitude, kept for per-cell altitude offset calculations.
    ey: f64,
    /// Altitude offset against the global height envelope. Used by range selection.
    dy: f32,
    /// Altitude clamp applied to both global and per-cell altitude offsets.
    cap: f32,
}

impl EyeMetric {
    /// Computes altitude offset `dy` from eye to global height envelope, clamped.
    /// `dy == 0` when the eye is inside the envelope.
    pub fn new(eye: DVec3, env: HeightEnvelope, cap: DyCap) -> EyeMetric {
        let y = eye.y;
        let raw = if !y.is_finite() {
            0.0
        } else if y < env.lo as f64 {
            (env.lo as f64 - y) as f32
        } else if y > env.hi as f64 {
            (y - env.hi as f64) as f32
        } else {
            0.0
        };
        EyeMetric { ex: eye.x, ez: eye.z, ey: y, dy: raw.min(cap.0), cap: cap.0 }
    }

    /// Altitude offset from eye to a per-cell height envelope, clamped.
    /// Used so a high-flying eye sees coarser LOD over terrain below it.
    fn dy_in(&self, env: HeightEnvelope) -> f32 {
        if !self.ey.is_finite() {
            return 0.0;
        }
        let raw = if self.ey < env.lo as f64 {
            (env.lo as f64 - self.ey) as f32
        } else if self.ey > env.hi as f64 {
            (self.ey - env.hi as f64) as f32
        } else {
            0.0
        };
        raw.min(self.cap)
    }

    /// Integer grid cell containing the eye's XZ position.
    pub fn anchor(&self) -> (i32, i32) {
        (self.ex.floor() as i32, self.ez.floor() as i32)
    }

    /// Eye-to-point distance in 3D, combining horizontal distance with altitude offset.
    pub fn point(&self, x: f64, z: f64) -> EyeDist {
        let dx = (x - self.ex) as f32;
        let dz = (z - self.ez) as f32;
        EyeDist::new(dx.hypot(dz).hypot(self.dy))
    }

    /// Nearest and farthest eye-to-cell distances, using global altitude offset.
    pub fn range(&self, s: SectionPos) -> DistRange {
        self.range_with_dy(s, self.dy)
    }

    /// Nearest and farthest distances, but using the cell's own height envelope.
    /// Allows a high eye to see coarser detail over terrain with varied elevation.
    pub fn range_in(&self, s: SectionPos, env: HeightEnvelope) -> DistRange {
        self.range_with_dy(s, self.dy_in(env))
    }

    /// Shared implementation of `range` and `range_in` with an explicit altitude offset.
    fn range_with_dy(&self, s: SectionPos, dy: f32) -> DistRange {
        let span = s.span() as f64;
        let axis = |lo: f64, e: f64| -> (f64, f64) {
            let hi = lo + span;
            let near = if e < lo {
                lo - e
            } else if e >= hi {
                e - hi + 1.0
            } else {
                0.0
            };
            let far = (e - lo).abs().max((e - (hi - 1.0)).abs());
            (near, far)
        };
        let (nx, fx) = axis(s.min_x() as f64, self.ex);
        let (nz, fz) = axis(s.min_z() as f64, self.ez);
        let near_xz = ((nx * nx + nz * nz) as f32).sqrt();
        let far_xz = ((fx * fx + fz * fz) as f32).sqrt();
        DistRange::new(EyeDist::new(near_xz.hypot(dy)), EyeDist::new(far_xz.hypot(dy)))
    }

    /// Ground-plane projection of a distance band's outer radius. Returns `None` if
    /// the band sits entirely overhead (altitude offset >= radius).
    pub fn xz_annulus(&self, _lo: f32, hi: f32) -> Option<XzAnnulus> {
        if self.dy >= hi {
            return None;
        }
        let hi = if self.dy == 0.0 || !hi.is_finite() {
            hi
        } else {
            (hi * hi - self.dy * self.dy).max(0.0).sqrt()
        };
        Some(XzAnnulus { hi })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eyedist_is_total_and_nonnegative() {
        assert_eq!(EyeDist::new(3.5).get(), 3.5);
        assert_eq!(EyeDist::new(f32::NAN).get(), 0.0);
        assert_eq!(EyeDist::new(-2.0).get(), 0.0);
        assert_eq!(EyeDist::new(f32::INFINITY).get(), 0.0);
    }

    #[test]
    fn dy_zero_inside_envelope_and_capped_outside() {
        let env = HeightEnvelope::new(0.0, 512.0);
        let cap = DyCap::new(4096.0, 2.0); // 2048
        // Eye inside envelope → dy is 0.
        let m = EyeMetric::new(DVec3::new(10.0, 300.0, -5.0), env, cap);
        assert_eq!(m.point(10.0, -5.0).get(), 0.0);
        // Invalid eye data → dy is 0.
        let m = EyeMetric::new(DVec3::new(0.0, f64::NAN, 0.0), env, cap);
        assert_eq!(m.point(0.0, 0.0).get(), 0.0);
    }

    #[test]
    fn xz_annulus_passes_through_at_dy_zero_and_empties_when_overhead() {
        let m = EyeMetric::new(
            DVec3::new(0.0, 0.0, 0.0),
            HeightEnvelope::new(0.0, 0.0),
            DyCap::new(4096.0, 2.0),
        );
        // When altitude offset is zero, outer radius unchanged.
        assert_eq!(m.xz_annulus(96.0, 384.0).unwrap().hi(), 384.0);
        // Coarsest band (hi = infinity) survives unchanged.
        assert!(m.xz_annulus(96.0, f32::INFINITY).unwrap().hi().is_infinite());
    }

    #[test]
    fn xz_annulus_projects_and_empties_at_altitude() {
        let m = EyeMetric::new(
            DVec3::new(0.0, 300.0, 0.0),
            HeightEnvelope::new(0.0, 100.0), // dy = 200
            DyCap::new(1_000_000.0, 2.0),    // cap out of the way
        );
        // Outer radius projects onto ground plane.
        let hi = m.xz_annulus(96.0, 500.0).unwrap().hi();
        assert!((hi - (500.0f32 * 500.0 - 200.0 * 200.0).sqrt()).abs() < 1e-3);
        // Band entirely overhead contributes nothing.
        assert!(m.xz_annulus(96.0, 150.0).is_none());
    }
}
