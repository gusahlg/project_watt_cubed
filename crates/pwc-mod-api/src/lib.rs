//! Compatibility shim: the module API (`pwc-module-api`) under its old name.
//!
//! It exists so packages written before the rename (`use pwc_mod_api::{Mod, ModRegistrar}`) still
//! compile. New code depends on `pwc-module-api`.

pub use pwc_module_api::*;
