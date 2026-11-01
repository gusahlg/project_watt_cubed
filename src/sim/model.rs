//! Native simulation implementations register a factory; the core owns authority, storage and
//! commits. Dynamical state belongs to a model instance, never to material intern identities.
use super::reactions::{Budget, CellStore, Contact, Mutation, Pos};
use crate::block::{BlockId, BlockRegistry};

/// Migration identity of worlds saved before optional simulation providers existed.
pub const LEGACY_ID: &str = "pwc.physics-selective";

/// A compiled model's stable identity and constructor. Bump `version` whenever its law changes.
#[derive(Clone, Copy)]
pub struct ModelSpec {
    pub id: &'static str,
    pub version: u32,
    pub create: fn() -> Box<dyn Model>,
}

impl std::fmt::Debug for ModelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSpec")
            .field("id", &self.id)
            .field("version", &self.version)
            .finish()
    }
}

impl ModelSpec {
    /// Portable compatibility identity; no function address or installation order enters it.
    pub fn fingerprint(self) -> u64 {
        let mut h = crate::hash::Fnv64::new();
        h.bytes(b"pwc-simulation-1\0");
        h.bytes(self.id.as_bytes());
        h.bytes(&self.version.to_le_bytes());
        h.finish()
    }
}

/// Small readout for experimental HUDs and headless laboratories.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub labels: [&'static str; 4],
    pub values: [i64; 4],
}

/// Work/state counters. `pending` can count a persistent active domain, not just queued contacts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub turns: u64,
    pub operations: u64,
    pub dropped: u64,
    pub pending: usize,
}

/// One optional, authoritative law. Tick decisions must be deterministic and budgeted.
/// A model may advance sparse fields without changing any block configurations.
pub trait Model: Send {
    fn id(&self) -> &'static str;
    fn wake_cell(&mut self, pos: Pos);
    fn tick(&mut self, cells: &mut dyn CellStore, budget: Budget, committed: &mut Vec<Mutation>);
    fn stats(&self) -> Stats;
    fn save(&self) -> String;
    fn load(&mut self, state: &str) -> Result<(), String>;

    /// A held-material operation. The caller commits its returned cell and accounts for holdings.
    fn tool(
        &mut self,
        _registry: &mut BlockRegistry,
        _cell: BlockId,
        _tool: BlockId,
    ) -> Option<(BlockId, BlockId)> {
        None
    }

    /// Explicit experimental input. It is external work/source matter, not spontaneous creation.
    fn perturb(&mut self, pos: Pos, _amount: i64) {
        self.wake_cell(pos);
    }

    fn sample(&self, _pos: Pos) -> Sample {
        Sample::default()
    }

    /// The old save's pending-contact field, supported only by the extracted legacy model.
    fn legacy_snapshot(&self) -> Vec<(u32, Contact)> {
        Vec::new()
    }

    fn restore_legacy(&mut self, pending: &[(u32, Contact)]) -> Result<(), String> {
        if pending.is_empty() {
            Ok(())
        } else {
            Err("this model cannot read selective-transfer contacts".into())
        }
    }
}

// Historical core fixtures remain standalone. Production packages and their integration tests
// belong to the package-manager repository; this fixture is never part of a production build.
#[cfg(test)]
#[allow(dead_code)]
#[path = "test_selective.rs"]
mod selective_fixture;

#[cfg(test)]
pub(crate) fn test_default() -> Option<ModelSpec> {
    Some(ModelSpec {
        id: selective_fixture::ID,
        version: 1,
        create: || Box::new(selective_fixture::SelectiveTransfer::new()),
    })
}

#[cfg(test)]
pub(crate) fn test_with_capacity(capacity: usize) -> Box<dyn Model> {
    Box::new(selective_fixture::SelectiveTransfer::with_capacity(
        capacity,
    ))
}
