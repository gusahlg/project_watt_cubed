//! Small deterministic hash primitives, single-sourced so every call site that
//! depends on their exact bit output (terrain generation, procedural block
//! textures, peer colours) shares one implementation and can never drift.

/// FNV / FNV1a hash over a byte slice (32-bit). Shared by peer-colour and
/// texture-seed so both use the same algorithm.
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    h
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
