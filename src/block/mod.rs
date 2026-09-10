//! The world's material table: voxels store a compact [`BlockId`], the
//! [`registry`] maps it to a configuration observed under the law, and
//! [`regions`] name the worldgen starting families. Presentation is a
//! [`Visual`] turned into a texture layer by an appearance mod
//! ([`appearance`]).
pub mod appearance;
pub mod regions;
pub mod registry;

#[allow(unused_imports)] // re-exported crate API
pub use registry::{
    AIR, BlockId, BlockRegistry, HotTables, SoundClass, MAX_BLOCK_TYPES, MAX_DESCRIPTORS,
};
#[allow(unused_imports)] // re-exported crate API
pub use material::{Configuration, Element, Law, Observation, Visual};
