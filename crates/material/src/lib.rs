//! The material model. An element is a point of a four-dimensional periodic resource lattice; a block
//! holds a configuration (a multiset of element occurrences, at most [`CAPACITY`]); one universal
//! integer law — selective transfer v1 — moves occurrences between face-adjacent blocks when the move
//! makes both groupings fit better. Elements are conserved: matter is regrouped, never minted. Nothing
//! here knows a material name, and every decision is integer arithmetic over a committed table, so
//! every peer computes the same bytes.
//!
//! Layers, in causal order: [`Element`] / [`Configuration`] (state) → [`Block`] (the configuration with
//! its cached internal supports) → [`Contact`] / [`react_once`] (the law) → [`observe`] (readings taken
//! with the law's own fit function) → [`visual`] (presentation).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod configuration;
mod element;
#[rustfmt::skip]
mod fit_table;
mod fnv;
mod kernel;
mod law;
mod observe;
mod visual;

pub use configuration::{ConfigError, Configuration, DecodeError, Encoding, CAPACITY};
pub use element::{Element, D};
pub use fnv::{Fnv32, Fnv64};
pub use kernel::{fit_raw, react_once, Block, Change, Contact, Operation, QUANTUM};
pub use law::{Law, LawError, Probes, LAW_ID, STAMP_LEN};
pub use observe::{cohesion, observe, probe_response, Acoustic, Observation};
pub use visual::{centroid_q8, colour_at, element_colour, visual, visual_with, Visual};

#[cfg(test)]
mod tests;
