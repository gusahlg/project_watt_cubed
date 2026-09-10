//! Emergent material model. An element is a point of an abstract D-dimensional resource lattice; a
//! block holds a configuration (an ordered list of elements, multiplicity kept); one universal
//! integer law turns neighbouring configurations into new configurations. Nothing here knows a
//! material name. Everything is a pure function of (law, inputs) in 32-bit integer arithmetic so every
//! peer computes the same bytes.
//!
//! Layers, in causal order: [`Element`] / [`Configuration`] (state) → [`Law`] (the physics as a value:
//! kernel, event strengths, probes, visual seed) → [`interact`] (the reaction) → [`observe`] (cached
//! operational readings taken with the law's probe elements) → [`visual`] (presentation, a smooth map
//! from configurations to a render descriptor).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod configuration;
mod element;
mod kernel;
mod law;
mod observe;
mod visual;

pub use configuration::{Configuration, ConfigError, DecodeError, Encoding, CONFIG_MAX};
pub use element::{Element, D};
pub use kernel::{element_influence, interact, interact_many, Delta, ReactionResult};
pub use law::{
    Boundary, EventKind, EventStrengths, Kernel, Law, LawError, Probes, EVENT_KINDS, KNOTS, STAMP_LEN,
};
pub use observe::{observe, Acoustic, Observation};
pub use visual::{visual, DescriptorKey, Visual};

#[cfg(test)]
mod tests;
