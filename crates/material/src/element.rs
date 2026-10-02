//! A point of the resource lattice. Its coordinates ARE its identity; no table says what it means.

/// Dimensions of the resource lattice.
pub const D: usize = 4;

/// One element: a position on the four periodic resource axes (255 is adjacent to 0). Elements are
/// conserved by the law: a reaction moves occurrences between blocks, it never changes coordinates.
///
/// `Ord` is the lexicographic order of the four bytes, which is exactly the tie order of the law.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
pub struct Element(pub [u8; D]);

impl Element {
    /// Build an element from its coordinates.
    pub const fn new(coords: [u8; D]) -> Self {
        Self(coords)
    }

    /// Per-axis periodic separation (the short way round the ring), summed over the axes. A
    /// presentation and worldgen metric only; the law reads [`crate::fit_raw`].
    pub fn ring_distance(self, other: Element) -> u32 {
        let mut d = 0u32;
        for i in 0..D {
            let s = self.0[i].wrapping_sub(other.0[i]);
            d += s.min(s.wrapping_neg()) as u32;
        }
        d
    }
}
