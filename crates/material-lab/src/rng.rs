//! Deterministic xorshift64. Integer only; the same seed yields the same stream on every machine.

use material::{Configuration, Element, D};

/// Xorshift64* high-half generator. Seed 0 is mapped to a non-zero constant so the stream is not stuck.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Mix `seed` into a non-zero state.
    pub fn new(seed: u64) -> Self {
        let s = seed ^ 0xA076_1D64_78BD_642F;
        Self {
            state: if s == 0 { 0xA076_1D64_78BD_642F } else { s },
        }
    }

    /// Next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 32) as u32
    }

    /// Uniform `0..n` (n > 0).
    pub fn index(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u32() as usize) % n
    }

    /// Inclusive integer range. `lo` must be ≤ `hi`.
    pub fn inc(&mut self, lo: i32, hi: i32) -> i32 {
        debug_assert!(lo <= hi);
        let span = (hi as i64 - lo as i64 + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }

    /// A random lattice point.
    pub fn element(&mut self) -> Element {
        let mut c = [0u8; D];
        for x in c.iter_mut() {
            *x = self.next_u32() as u8;
        }
        Element(c)
    }

    /// A random non-void configuration of 1..=`max_len` elements.
    pub fn config(&mut self, max_len: usize) -> Configuration {
        let n = 1 + self.index(max_len);
        Configuration::new((0..n).map(|_| self.element()).collect::<Vec<_>>()).expect("n ≤ max_len ≤ CONFIG_MAX")
    }
}
