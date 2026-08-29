//! Shared identity vocabulary used by stored blocks and world-detail addressing.

use crate::block::BlockId;

pub mod codec;

/// Canonical definition lives in the engine (`producer::Detail`, alongside the
/// GPU-boundary biased encoding it shares with the app) — one type, not two
/// structurally identical ones.
pub use voxel_engine::producer::Detail;

/// Block value = id + variant/orientation state. State 0 = today's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockState {
    pub id: BlockId,
    pub state: u16,
}
