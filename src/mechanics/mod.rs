//! Mechanics: how matter deforms under its own gravity (guide §8; plan of record
//! `documentation/notes/WARP-ARCHITECTURE-2026-10-04.md`).
//!
//! - [`lattice`]: a body's deformation map φ (reference → physical), the authoritative geometry.
//! - [`material`]: constitutive parameters and the prototype mapping from matter to them.
//! - [`selfgrav`]: self-gravity of deformed matter (Barnes–Hut).
//! - [`solver`]: dynamic relaxation with J2 viscoplasticity.
//!
//! Headless: nothing here touches the renderer.

pub mod genesis;
pub mod lattice;
pub mod material;
pub mod selfgrav;
pub mod solver;
