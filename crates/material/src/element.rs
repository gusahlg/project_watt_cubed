//! A point of the resource lattice. Its coordinates ARE its identity; no table says what it means.

/// Dimensions of the resource lattice. Part of every law stamp (a world records it).
pub const D: usize = 4;

/// One element: a position in the D-dimensional resource lattice, one unsigned byte per axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Element(pub [u8; D]);

impl Element {
    /// Build an element from its coordinates.
    pub const fn new(coords: [u8; D]) -> Self {
        Self(coords)
    }

    /// L1 (Manhattan) distance in lattice units; the similarity metric of the model.
    pub fn distance(self, other: Element) -> u32 {
        let mut d = 0u32;
        for i in 0..D {
            d += (self.0[i] as i32 - other.0[i] as i32).unsigned_abs();
        }
        d
    }

    /// Largest single-axis separation.
    pub fn max_axis_distance(self, other: Element) -> u32 {
        (0..D)
            .map(|i| (self.0[i] as i32 - other.0[i] as i32).unsigned_abs())
            .max()
            .unwrap_or(0)
    }
}
