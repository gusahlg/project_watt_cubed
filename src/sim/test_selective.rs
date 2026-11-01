// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test-only legacy fixture so core tests remain independent of the package-manager checkout.
//! Production law and package integration tests live in pwc-package-manager/modules/physics-selective.
use pwc_module_api::block::{BlockId, BlockRegistry};
use pwc_module_api::sim::{
    model::{Model, Sample, Stats},
    reactions::{Budget, CellStore, Contact, DEFAULT_CAPACITY, Mutation, Pos},
};
use std::collections::{BTreeSet, HashSet};

pub const ID: &str = "pwc.physics-selective";

/// The scheduler state: active contacts in processing order, and counters.
pub struct SelectiveTransfer {
    queue: BTreeSet<(u64, Contact)>,
    active: HashSet<Contact>,
    capacity: usize,
    /// Turns run so far.
    pub turns: u64,
    /// Law operations committed so far.
    pub operations: u64,
    /// Contacts refused because the queue was full.
    pub dropped: u64,
}

impl Default for SelectiveTransfer {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl SelectiveTransfer {
    /// An empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty scheduler holding at most `capacity` active contacts.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: BTreeSet::new(),
            active: HashSet::new(),
            capacity,
            turns: 0,
            operations: 0,
            dropped: 0,
        }
    }

    fn enqueue(&mut self, arrival: u64, c: Contact) {
        if self.active.contains(&c) {
            return;
        }
        if self.active.len() >= self.capacity {
            self.dropped += 1;
            return;
        }
        self.active.insert(c);
        self.queue.insert((arrival, c));
    }

    /// Something changed at `p` (placed, removed, configuration changed): wake its six contacts.
    pub fn wake_cell(&mut self, p: Pos) {
        for c in Contact::around(p) {
            self.enqueue(self.turns, c);
        }
    }

    /// A block moved from `from` to `to`: both locations' contacts.
    pub fn wake_move(&mut self, from: Pos, to: Pos) {
        self.wake_cell(from);
        self.wake_cell(to);
    }

    /// Active contacts waiting for a turn.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// The active contacts in processing order, each with its age in turns (how long it has been
    /// waiting): the pending work a save carries. Order and ages restore exactly.
    pub fn snapshot(&self) -> Vec<(u32, Contact)> {
        self.queue
            .iter()
            .map(|&(t, c)| (self.turns.saturating_sub(t).min(u32::MAX as u64) as u32, c))
            .collect()
    }

    /// Restore saved pending work with its order, without inventing events.
    pub fn restore(&mut self, contacts: &[(u32, Contact)]) {
        let oldest = contacts
            .iter()
            .map(|&(age, _)| age as u64)
            .max()
            .unwrap_or(0);
        self.turns = self.turns.max(oldest);
        for &(age, c) in contacts {
            self.enqueue(self.turns - age as u64, c);
        }
    }

    /// Run one turn: attempt up to `budget.contacts_per_turn` contacts that arrived before this turn,
    /// oldest first. A changed contact commits both cells and queues every contact of both cells for
    /// the next turn; an unchanged one goes dormant. Returns the commits in order.
    pub fn tick<S: CellStore + ?Sized>(&mut self, store: &mut S, budget: Budget) -> Vec<Mutation> {
        let mut committed = Vec::new();
        self.tick_into(store, budget, &mut committed);
        committed
    }

    /// [`tick`](Self::tick), appending the commits to `committed`, so a caller that ticks
    /// every frame reuses one buffer.
    pub fn tick_into<S: CellStore + ?Sized>(
        &mut self,
        store: &mut S,
        budget: Budget,
        committed: &mut Vec<Mutation>,
    ) {
        let this_turn = self.turns;
        let next = this_turn + 1;
        for _ in 0..budget.contacts_per_turn {
            let Some(&(arrival, c)) = self.queue.first() else {
                break;
            };
            if arrival > this_turn {
                break;
            }
            self.queue.pop_first();
            self.active.remove(&c);
            let (lo, hi) = (store.resolve(c.lo), store.resolve(c.hi()));
            if lo == hi {
                continue;
            }
            let (lo, hi) = if lo < hi { (lo, hi) } else { (hi, lo) };
            let (Some(a), Some(b)) = (store.block_at(lo), store.block_at(hi)) else {
                continue;
            };
            if a == b {
                // Identical configurations (and two voids) have no candidate: dormant.
                continue;
            }
            let Some((_, na, nb)) = store.registry_mut().react(a, b) else {
                continue;
            };
            // Admission is atomic: no product survives a refused partner write.
            if !store.set_pair(lo, a, na, hi, b, nb) {
                self.enqueue(next, c);
                continue;
            }
            committed.push(Mutation {
                pos: lo,
                from: a,
                to: na,
            });
            committed.push(Mutation {
                pos: hi,
                from: b,
                to: nb,
            });
            self.operations += 1;
            for p in [lo, hi] {
                for w in Contact::around(p) {
                    self.enqueue(next, w);
                }
            }
        }
        self.turns = next;
    }
}

impl Model for SelectiveTransfer {
    fn id(&self) -> &'static str {
        ID
    }
    fn wake_cell(&mut self, pos: Pos) {
        SelectiveTransfer::wake_cell(self, pos);
    }
    fn tick(&mut self, cells: &mut dyn CellStore, budget: Budget, out: &mut Vec<Mutation>) {
        self.tick_into(cells, budget, out);
    }
    fn stats(&self) -> Stats {
        Stats {
            turns: self.turns,
            operations: self.operations,
            dropped: self.dropped,
            pending: self.pending(),
        }
    }
    fn tool(
        &mut self,
        registry: &mut BlockRegistry,
        cell: BlockId,
        tool: BlockId,
    ) -> Option<(BlockId, BlockId)> {
        registry.react(cell, tool).map(|(_, a, b)| (a, b))
    }
    fn sample(&self, _: Pos) -> Sample {
        Sample {
            labels: ["contacts", "turns", "changes", "dropped"],
            values: [
                self.pending() as i64,
                self.turns as i64,
                self.operations as i64,
                self.dropped as i64,
            ],
        }
    }
    fn save(&self) -> String {
        let mut text = format!("{} {} {}\n", self.turns, self.operations, self.dropped);
        for (age, c) in self.snapshot() {
            text.push_str(&format!(
                "{} {} {} {} {}\n",
                age, c.lo.0, c.lo.1, c.lo.2, c.axis
            ));
        }
        text
    }
    fn load(&mut self, text: &str) -> Result<(), String> {
        let mut lines = text.lines();
        let header: Vec<_> = lines
            .next()
            .ok_or("missing selective-transfer header")?
            .split_whitespace()
            .collect();
        if header.len() != 3 {
            return Err("invalid selective-transfer header".into());
        }
        let turns = header[0].parse::<u64>().map_err(|_| "invalid turn count")?;
        let operations = header[1]
            .parse::<u64>()
            .map_err(|_| "invalid operation count")?;
        let dropped = header[2].parse::<u64>().map_err(|_| "invalid drop count")?;
        let mut pending = Vec::new();
        for line in lines {
            let v: Vec<_> = line.split_whitespace().collect();
            if v.len() != 5 || pending.len() >= self.capacity {
                return Err("invalid/oversized selective-transfer queue".into());
            }
            let age = v[0].parse::<u32>().map_err(|_| "invalid contact age")?;
            let x = v[1].parse::<i32>().map_err(|_| "invalid contact x")?;
            let y = v[2].parse::<i32>().map_err(|_| "invalid contact y")?;
            let z = v[3].parse::<i32>().map_err(|_| "invalid contact z")?;
            let axis = v[4].parse::<u8>().map_err(|_| "invalid contact axis")?;
            if axis > 2 || [x, y, z].iter().any(|c| c.unsigned_abs() >= 2_050_000_000) {
                return Err("contact outside supported cells".into());
            }
            pending.push((
                age,
                Contact {
                    lo: (x, y, z),
                    axis,
                },
            ));
        }
        self.queue.clear();
        self.active.clear();
        self.turns = turns;
        self.operations = operations;
        self.dropped = dropped;
        self.restore(&pending);
        Ok(())
    }
    fn legacy_snapshot(&self) -> Vec<(u32, Contact)> {
        self.snapshot()
    }
    fn restore_legacy(&mut self, contacts: &[(u32, Contact)]) -> Result<(), String> {
        self.restore(contacts);
        Ok(())
    }
}
