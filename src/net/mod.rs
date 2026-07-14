//! Multiplayer: an authoritative headless [`server`] and the thin [`client`]
//! [`Connection`](client::Connection) the game talks to, speaking the compact
//! binary [`protocol`].
//!
//! The whole design leans on one fact from [`world`](crate::world): terrain is
//! *procedural*, regenerated identically from a seed, so the network never carries
//! voxel data. A join transfers only the seed and the sparse overlay of player
//! *edits* — the same portable, name-keyed form [`save`](crate::save) already uses.
//! Everything else on the wire is small and frequent (positions, chat), which is
//! what the framing and interest management here are tuned for.
//!
//! **Trust:** the server is authoritative and never trusts a client. Frames are
//! length-capped, joins are password-gated, and every edit and move is validated
//! and rate-limited server-side ([`server`]).
pub mod client;
pub mod protocol;

// The `.lock().unwrap()` painpoint in `server` is now the ONE
// `LockRecover::lock_recover` helper (server.rs); deny any regression back to
// a bare `.unwrap()` on the production paths. Scoped to this module only; the
// tests module carries its own `#![allow]` (setup unwraps there are loud test
// failures, which is the desired behavior).
#[deny(clippy::unwrap_used)]
pub mod server;

/// The protocol revision. Client and server must match exactly, checked at join.
/// v2: positions are 3x f64 on the wire (far-coordinate correctness).
/// v4: element-first worldgen (worldgen v2) — chunk materials are a pure
/// function of (seed, worldgen), so any worldgen change MUST bump this: mixed
/// peers would silently desync on terrain contents otherwise.
/// v5: worldgen v3 (alien pass — trees gone, per-biome crust, surface glow).
/// v6: authoritative edits (request id + expected cell revision + ack),
/// content fingerprint in `Hello`, explicit `Teleport`, `Position`
/// corrections, peer visibility exits, and synchronized day length.
pub const PROTOCOL_VERSION: u32 = 6;

/// The default TCP port a server listens on and a client dials.
pub const DEFAULT_PORT: u16 = 5555;

/// Hard cap on a single wire frame (bytes). A frame claiming more is rejected
/// before a byte of its body is read, so a hostile peer can't force a huge alloc.
pub const MAX_FRAME: usize = 64 * 1024;

/// Longest accepted player name.
pub const MAX_NAME: usize = 24;

/// Longest accepted chat line.
pub const MAX_CHAT: usize = 256;

/// Longest accepted block spec string (an edit's portable composition).
pub const MAX_SPEC: usize = 256;

/// A stable 64-bit digest of everything that determines what a seed GENERATES:
/// the worldgen version, the element table, and the full compiled placement
/// palette (canonical portable spec per block). Seed-only multiplayer never
/// ships voxels, so two builds whose generation differs in ANY of these would
/// silently build different worlds from one seed — the handshake compares
/// fingerprints and rejects the join instead. Protocol changes are versioned
/// separately by [`PROTOCOL_VERSION`].
///
/// FNV-1a (not the std hasher) so the value is identical across platforms,
/// architectures, and Rust releases. The placement compile is seed-invariant,
/// so one fingerprint speaks for every world a build can generate.
pub fn content_fingerprint() -> u64 {
    let mut registry = crate::block::BlockRegistry::with_builtins();
    let _ = crate::world::placement::builtin().compile(&mut registry);
    fingerprint_of(&registry)
}

/// The fingerprint of an already-compiled registry — the server hashes the
/// one it built for spawn heights instead of compiling twice.
pub fn fingerprint_of(registry: &crate::block::BlockRegistry) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    eat(&crate::world::placement::WORLDGEN_VERSION.to_le_bytes());
    let elements = registry.elements();
    eat(&(elements.len() as u32).to_le_bytes());
    for i in 0..elements.len() {
        let element = elements.get(crate::block::ElementId(i as u16));
        eat(element.name.as_bytes());
        eat(&[0]); // name terminator so "ab"+"c" != "a"+"bc"
    }
    eat(&(registry.block_count() as u32).to_le_bytes());
    for i in 0..registry.block_count() {
        eat(crate::save::registry_block_spec(registry, crate::block::BlockId(i as u16)).as_bytes());
        eat(&[0]);
    }
    hash
}

/// Chat channels. Local is proximity-limited; global reaches everyone.
pub mod chat {
    /// Proximity chat: only players within [`RADIUS`] world units hear it.
    pub const LOCAL: u8 = 0;
    /// Global chat: reaches every connected player.
    pub const GLOBAL: u8 = 1;
    /// How far local (proximity) chat carries: 48 m, in world units. `f64`
    /// like all position math server-side.
    pub const RADIUS: f64 = 48.0 * crate::math::PER_METER;
}
