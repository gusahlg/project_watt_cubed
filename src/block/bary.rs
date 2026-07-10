//! A weighted-combination operation shared by three different value types.
//!
//! Core bytes ([`Core`]), RGBA color ([`Color`]), and special effects
//! ([`SparseSpecials`]) all need the same logic: combine multiple values
//! using weights. This module implements that once via [`barycenter`], and
//! each value type supplies its own math via the [`Bary`] trait.
//!
//! Integer results are truncated (not rounded), and zero total weight falls
//! back to the default value.
use voxel_engine::Color;

use crate::block::element::{CORE_FIELD_COUNT, Core, SpecialKind};

/// A value type that can be combined with weights. `Acc` collects contributions;
/// [`axpy`](Bary::axpy) adds a weighted copy to it, and [`finish`](Bary::finish)
/// computes the final result (or returns the default when total is zero).
pub trait Bary: Sized {
    /// The accumulator that sums contributions before the final divide.
    type Acc;
    /// Initialize the accumulator.
    fn zero() -> Self::Acc;
    /// Add a weighted copy of this value to the accumulator.
    fn axpy(&self, weight: u32, acc: &mut Self::Acc);
    /// Compute the final result from the accumulated values and total weight.
    fn finish(acc: Self::Acc, total: u32) -> Self;
}

/// Combine values using their weights, or return the default if total weight is zero.
pub fn barycenter<V: Bary>(parts: impl IntoIterator<Item = (V, u32)>) -> V {
    let mut acc = V::zero();
    let mut total = 0u32;
    for (v, w) in parts {
        v.axpy(w, &mut acc);
        debug_assert!(total.checked_add(w).is_some(), "barycenter weight sum overflow");
        total += w;
    }
    V::finish(acc, total)
}

impl Bary for Core {
    type Acc = [u32; CORE_FIELD_COUNT];

    fn zero() -> [u32; CORE_FIELD_COUNT] {
        [0u32; CORE_FIELD_COUNT]
    }

    fn axpy(&self, weight: u32, acc: &mut [u32; CORE_FIELD_COUNT]) {
        for i in 0..CORE_FIELD_COUNT {
            debug_assert!(
                acc[i].checked_add(self.0[i] as u32 * weight).is_some(),
                "barycenter channel overflow"
            );
            acc[i] += self.0[i] as u32 * weight;
        }
    }

    fn finish(acc: [u32; CORE_FIELD_COUNT], total: u32) -> Core {
        if total == 0 {
            return Core::default();
        }
        Core(std::array::from_fn(|i| (acc[i] / total) as u8))
    }
}

impl Bary for Color {
    type Acc = [u32; 4];

    fn zero() -> [u32; 4] {
        [0u32; 4]
    }

    fn axpy(&self, weight: u32, acc: &mut [u32; 4]) {
        let channels = [self.r, self.g, self.b, self.a];
        for i in 0..4 {
            debug_assert!(
                acc[i].checked_add(channels[i] as u32 * weight).is_some(),
                "barycenter channel overflow"
            );
            acc[i] += channels[i] as u32 * weight;
        }
    }

    fn finish(acc: [u32; 4], total: u32) -> Color {
        if total == 0 {
            return Color::new(0, 0, 0, 0);
        }
        Color::new(
            (acc[0] / total) as u8,
            (acc[1] / total) as u8,
            (acc[2] / total) as u8,
            (acc[3] / total) as u8,
        )
    }
}

/// A sparse set of special effects with their strengths. Each entry is merged
/// by kind during accumulation, sorted before the final result, or empty if
/// total weight is zero.
pub struct SparseSpecials(pub Vec<(SpecialKind, u8)>);

impl Bary for SparseSpecials {
    type Acc = Vec<(SpecialKind, u32)>;

    fn zero() -> Vec<(SpecialKind, u32)> {
        Vec::new()
    }

    fn axpy(&self, weight: u32, acc: &mut Vec<(SpecialKind, u32)>) {
        for &(kind, strength) in &self.0 {
            let contribution = strength as u32 * weight;
            match acc.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, a)) => *a += contribution,
                None => acc.push((kind, contribution)),
            }
        }
    }

    fn finish(mut acc: Vec<(SpecialKind, u32)>, total: u32) -> SparseSpecials {
        if total == 0 {
            return SparseSpecials(Vec::new());
        }
        acc.sort_by_key(|&(k, _)| k);
        SparseSpecials(acc.into_iter().map(|(k, a)| (k, (a / total) as u8)).collect())
    }
}
