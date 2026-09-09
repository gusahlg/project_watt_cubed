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

/// One splitmix64 step: add the golden ratio, then finish.
#[inline]
pub fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    splitmix_finish(*state)
}

/// Three-multiplier lattice mix used by terrain hashes.
#[inline]
pub fn mix3(seed: u64, x: i32, y: i32, z: i32) -> u64 {
    seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
}
