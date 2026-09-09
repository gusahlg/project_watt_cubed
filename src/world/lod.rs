//! Detail-level geometry: everything a level's cell size implies derives from
//! `k`, the engine's signed [`Detail`]. One cell is `2^k` metres; a chunk section
//! is `16·2^k`; a section is `32·2^k`.
use crate::ident::Detail;

/// Metres per cell (`2^k`) as an integer. `k` must be non-negative — sub-block
/// levels (`k < 0`) have a fractional cell size and use [`Detail::scale`] instead.
pub(in crate::world) fn cell(d: Detail) -> i32 {
    debug_assert!(d.0 >= 0, "integer cell size needs k >= 0, got {}", d.0);
    1 << d.0
}
