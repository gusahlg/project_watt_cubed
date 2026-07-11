//! LOD level k: cells are `2^k` meters; all sizing derives from k.
use super::chunk::CHUNK_SIZE;

/// LOD level carrier: everything about a level's cell size derives from `k`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Lod(pub u8);

impl Lod {
    /// Metres per cell (`2^k`); also a section block's `draw_mesh` scale.
    pub const fn cell(self) -> i32 {
        1 << self.0
    }
    /// Metres per tile side (`16·2^k`).
    pub const fn span(self) -> i32 {
        (CHUNK_SIZE as i32) << self.0
    }
    /// Chunk columns spanned per side (`2^k`).
    pub const fn chunks_per_side(self) -> i32 {
        1 << self.0
    }
}
