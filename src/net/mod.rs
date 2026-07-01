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
pub mod server;

/// The protocol revision. Client and server must match exactly, checked at join.
pub const PROTOCOL_VERSION: u32 = 1;

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

/// Chat channels. Local is proximity-limited; global reaches everyone.
pub mod chat {
    /// Proximity chat: only players within [`RADIUS`] world units hear it.
    pub const LOCAL: u8 = 0;
    /// Global chat: reaches every connected player.
    pub const GLOBAL: u8 = 1;
    /// How far local (proximity) chat carries, in world units.
    pub const RADIUS: f32 = 48.0;
}
