//! Universe-level layout: how patches of cells are embedded in physical space (face frames of
//! axis-aligned bodies, curved cube-sphere charts of round ones). Layout is world state; physics
//! never reads it.

pub mod atlas;
pub mod chart;

/// The six signed axes (cube faces, face frames, collision axes).
pub use crate::coord::Face;
