//! The element/block core: the minimal, powerful foundation the rest of the game
//! builds on. The world is made of [`element`]s with physical properties; a block
//! is a [`composition`] of elements, and everything observable about it is
//! [`derive`]d from that composition (plus any [`reaction`] it triggers). The
//! [`registry`] resolves a composition to a compact [`BlockId`] and is the single
//! source of truth voxels index into.
//!
//! Submodules, roughly in dependency order:
//! - [`element`] — element properties, the built-in element table, [`ElementId`].
//! - [`composition`] — the four block tiers and the `(element, weight)` view of them.
//! - [`derive`] — composition → solidity, colour, core properties, specials.
//! - [`reaction`] — emergent properties from element combinations.
//! - [`registry`] — the [`BlockRegistry`], hot/cold split, and the crafting API.
pub mod composition;
pub mod derive;
pub mod element;
pub mod reaction;
pub mod registry;

pub use composition::{Composition, Mix, MixError};
pub use element::{CoreProperties, Element, ElementId, ElementRegistry, SpecialKind, SpecialProperty};
pub use reaction::{ActiveReaction, Reaction, ReactionEffect, ReactionRegistry};
pub use registry::{AIR, Block, BlockId, BlockRegistry};
