//! Test fixtures shared by the world's test modules.

use super::SectionState;

/// A resident section with no slabs: what a headless upload lands as.
pub(in crate::world) fn ready_section() -> SectionState {
    SectionState::Ready { meshes: Vec::new(), cages: Vec::new(), last_style: None }
}
