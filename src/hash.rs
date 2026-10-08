//! Small deterministic hash primitives, single-sourced so every call site that
//! depends on their exact bit output (terrain generation, procedural block
//! textures, peer colours, fingerprints and cache keys) shares one implementation
//! and can never drift. FNV-1a lives in the material crate, which hashes the law.

pub use material::{Fnv32, Fnv64};

/// FNV-1a over a byte slice (32-bit). Shared by peer-colour and texture-seed so
/// both use the same algorithm.
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    Fnv32::new().bytes(bytes).finish()
}

/// The splitmix64 finisher: mixes up a 64-bit hash state using xor and
/// multiply operations. Shared by terrain and ore generation for consistency.
pub fn splitmix_finish(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// One splitmix64 step: add the golden ratio, then finish.
#[inline]
pub fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    splitmix_finish(*state)
}

