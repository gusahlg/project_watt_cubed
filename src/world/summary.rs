//! Screen-space error budget for LOD selection: measured geometric error per cell
//! and a test to decide if that error fits within the pixel tolerance at a given
//! distance. Tested early; wired into LOD selection later.
#![allow(dead_code)]

use super::metric::{EyeDist, HeightEnvelope};

/// Geometric error in metres when drawing a cell coarse — an over-approximation
/// until baking refines it. Worst case: 2^detail (the implicit error from the old ladder).
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub(in crate::world) struct CellError(f32);

impl CellError {
    /// Error for a cell with no summary: 2^detail metres (matches the old ladder).
    pub fn worst_case(detail: u8) -> CellError {
        CellError((1u32 << detail) as f32)
    }
    /// A measured error in metres. Still conservative: caller passes the full relief range
    /// which bounds any column's vertical displacement when drawn coarse.
    pub fn from_metres(m: f32) -> CellError {
        debug_assert!(m >= 0.0, "cell error is non-negative, got {m}");
        CellError(m.max(0.0))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Cell height envelope and its recorded error. Both are worst-case until baking refines them.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct CellSummary {
    pub env: HeightEnvelope,
    pub err: CellError,
}

/// Screen-space error budget for LOD selection: converts world-space error to pixels
/// via the camera's field of view. Built once at startup.
#[derive(Clone, Copy, Debug)]
pub(in crate::world) struct SseBudget {
    tau_px: f32,
    k: f32,
}

impl SseBudget {
    pub fn new(tau_px: f32, k: f32) -> SseBudget {
        SseBudget { tau_px, k }
    }

    /// Calibrate the budget to match the old radial ladder exactly at worst case
    /// (error = 2^detail). This ensures worst-case terrain stays bit-identical,
    /// while flatter terrain can coarsen further. Backward-compatible by design.
    pub fn ladder(k: f32, unit: f32, finest: u8) -> SseBudget {
        SseBudget { tau_px: k * (1u32 << finest) as f32 / unit, k }
    }

    /// Test if error at distance d fits within budget. Cross-multiplied to handle d=0 safely.
    pub fn coarse_ok(&self, err: CellError, d: EyeDist) -> bool {
        err.get() * self.k <= self.tau_px * d.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_case_is_two_to_the_detail() {
        assert_eq!(CellError::worst_case(0).get(), 1.0);
        assert_eq!(CellError::worst_case(2).get(), 4.0);
        assert_eq!(CellError::worst_case(5).get(), 32.0);
    }

    #[test]
    fn coarse_ok_is_monotone_in_distance_and_total_at_zero() {
        let b = SseBudget::new(2.0, 500.0);
        let err = CellError::worst_case(4); // 16 m
        // Near: error too large for the budget.
        assert!(!b.coarse_ok(err, EyeDist::new(1.0)));
        // Far enough: fits.
        assert!(b.coarse_ok(err, EyeDist::new(100_000.0)));
        // d = 0 is well-defined (no panic): a positive error never fits at 0.
        assert!(!b.coarse_ok(err, EyeDist::new(0.0)));
    }

    #[test]
    fn cell_summary_carries_envelope_and_error() {
        let s = CellSummary { env: HeightEnvelope::new(0.0, 512.0), err: CellError::worst_case(3) };
        assert_eq!(s.err.get(), 8.0);
    }
}
