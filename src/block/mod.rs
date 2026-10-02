//! The world's material table: voxels store a compact [`BlockId`], the [`registry`] maps it to a
//! configuration observed under the law. Presentation: a texture layer per configuration painted by
//! an appearance mod ([`appearance`]) and names given by a naming mod ([`naming`]).
pub mod appearance;
pub mod naming;
pub mod registry;

#[allow(unused_imports)] // re-exported crate API
pub use registry::{AIR, BlockId, BlockRegistry, HotTables, SoundClass, MAX_BLOCK_TYPES, MAX_DESCRIPTORS};
#[allow(unused_imports)] // re-exported crate API
pub use material::{Configuration, Element, Law, Observation, Visual};
