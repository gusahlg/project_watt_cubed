//! Selective transfer v1 (`watt-selective-transfer-v1`): a deterministic, passive, element-conserving
//! law. One reaction moves ONE occurrence between two face-adjacent blocks (or, when a block is at
//! capacity, exchanges one occurrence each way) when the move improves the summed internal fit by more
//! than `1/32` of the contact's size. Integer scores, no heap allocation, no floating point decisions.
//!
//! Specification: `guides/reaction-guide/selective-transfer-v1.md` (§§ 3-10). The arithmetic below is the
//! reference implementation's, unchanged; only the element type is the crate's [`Element`].

use crate::configuration::{Configuration, CAPACITY};
use crate::element::Element;
use crate::fit_table::FIT;

/// Fixed-point scale of the fit table: `fit_raw / (4 · QUANTUM)` is the normalized pair fit.
pub const QUANTUM: i32 = 1 << 20;
/// Most occurrences one contact can hold (two full blocks).
const MAX_TOTAL: usize = 2 * CAPACITY;

/// `4 · QUANTUM` times the dimension-averaged pair fit of two elements, rounded per axis by the
/// committed table. Positive favours grouping, negative favours separation; identical elements fit 0.
/// Four byte differences, four table reads, three additions.
#[inline]
pub fn fit_raw(a: Element, b: Element) -> i32 {
    let (a, b) = (a.0, b.0);
    FIT[a[0].wrapping_sub(b[0]) as usize]
        + FIT[a[1].wrapping_sub(b[1]) as usize]
        + FIT[a[2].wrapping_sub(b[2]) as usize]
        + FIT[a[3].wrapping_sub(b[3]) as usize]
}

/// The law's operational record of one configuration: its occurrences plus each occurrence's cached
/// raw internal support `h[i] = Σ_{j≠i} fit_raw(e_i, e_j)`. Array order is storage only. Build it once
/// per distinct configuration (the game's intern table keeps one per id) and reuse it.
#[derive(Clone, Debug)]
pub struct Block {
    elements: [Element; CAPACITY],
    holding: [i32; CAPACITY],
    len: u8,
}

impl Default for Block {
    fn default() -> Self {
        Self { elements: [Element::default(); CAPACITY], holding: [0; CAPACITY], len: 0 }
    }
}

impl Block {
    /// Build the support cache from occurrences: one pass over the unordered internal pairs.
    pub fn new(elements: &[Element]) -> Option<Self> {
        if elements.len() > CAPACITY {
            return None;
        }
        let mut block = Self::default();
        let n = elements.len();
        block.len = n as u8;
        block.elements[..n].copy_from_slice(elements);
        for i in 0..n {
            for j in (i + 1)..n {
                let k = fit_raw(block.elements[i], block.elements[j]);
                block.holding[i] += k;
                block.holding[j] += k;
            }
        }
        Some(block)
    }

    /// The kernel record of a configuration (never fails: configurations respect [`CAPACITY`]).
    pub fn of(c: &Configuration) -> Self {
        Self::new(c.elements()).expect("a configuration never exceeds CAPACITY")
    }

    /// The occurrences, in storage order.
    pub fn elements(&self) -> &[Element] {
        &self.elements[..self.len as usize]
    }

    /// Each occurrence's raw internal support, aligned with [`elements`](Self::elements).
    pub fn holding(&self) -> &[i32] {
        &self.holding[..self.len as usize]
    }

    /// Number of occurrences.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// True for the void.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sum of fit over internal unordered pairs (`Φ` of this block), exact in `i64`.
    pub fn internal_fit(&self) -> i64 {
        self.holding().iter().map(|&h| i64::from(h)).sum::<i64>() / 2
    }

    /// Sort storage into canonical element order, keeping each support attached to its element.
    pub fn canonicalize(&mut self) {
        for i in 1..self.len as usize {
            let (element, holding) = (self.elements[i], self.holding[i]);
            let mut j = i;
            while j > 0 && self.elements[j - 1] > element {
                self.elements[j] = self.elements[j - 1];
                self.holding[j] = self.holding[j - 1];
                j -= 1;
            }
            self.elements[j] = element;
            self.holding[j] = holding;
        }
    }

    /// The canonical configuration of this block.
    pub fn configuration(&self) -> Configuration {
        Configuration::new(self.elements().to_vec()).expect("a block never exceeds CAPACITY")
    }
}

/// What one accepted operation did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// One occurrence of `element` left block `from` (0 = A, 1 = B) for the other block.
    Transfer {
        /// The giving block: 0 = A, 1 = B.
        from: u8,
        /// The moved element.
        element: Element,
    },
    /// An atomic exchange at capacity: `from_a` went A→B and `from_b` went B→A.
    Swap {
        /// The element that left A.
        from_a: Element,
        /// The element that left B.
        from_b: Element,
    },
}

/// One accepted operation of the law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operation {
    /// What moved.
    pub change: Change,
    /// Exact increase of the summed internal fit (always `> total_elements · QUANTUM / 8`).
    pub raw_gain: i32,
    /// Occurrences in the contact (both blocks).
    pub total_elements: u8,
}

impl Operation {
    /// Diagnostic conversion only; no floating-point result controls the law.
    pub fn normalized_gain(self) -> f64 {
        f64::from(self.raw_gain) / (4.0 * f64::from(QUANTUM) * f64::from(self.total_elements))
    }
}

#[derive(Clone, Copy)]
enum Move {
    Transfer(usize),
    Swap(usize, usize),
}

#[derive(Clone, Copy)]
struct Decision {
    movement: Move,
    gain: i32,
}

/// Scratch state for ONE two-block contact: a snapshot of both blocks, an owner bit per occurrence,
/// supports and outgoing gains. Reuse it only while these are still the complete current contents of
/// both blocks; rebuild after anything else changes either block.
#[derive(Clone)]
pub struct Contact {
    elements: [Element; MAX_TOTAL],
    owner: [u8; MAX_TOTAL],
    gain: [i32; MAX_TOTAL],
    holding: [i32; MAX_TOTAL],
    len: usize,
    counts: [usize; 2],
}

impl Contact {
    /// Scores for the contact of `a` and `b` (in the caller's canonical physical order): one traversal
    /// of the cross-block pairs on top of the cached internal supports.
    pub fn new(a: &Block, b: &Block) -> Self {
        let (na, nb) = (a.len(), b.len());
        let mut state = Self {
            elements: [Element::default(); MAX_TOTAL],
            owner: [0; MAX_TOTAL],
            gain: [0; MAX_TOTAL],
            holding: [0; MAX_TOTAL],
            len: na + nb,
            counts: [na, nb],
        };
        state.elements[..na].copy_from_slice(a.elements());
        state.elements[na..state.len].copy_from_slice(b.elements());
        state.owner[na..state.len].fill(1);
        state.holding[..na].copy_from_slice(a.holding());
        state.holding[na..state.len].copy_from_slice(b.holding());
        for i in 0..state.len {
            state.gain[i] = -state.holding[i];
        }
        // Each cross-block pair contributes its attraction once to each endpoint's outgoing gain.
        for i in 0..na {
            for j in na..state.len {
                let k = fit_raw(state.elements[i], state.elements[j]);
                state.gain[i] += k;
                state.gain[j] += k;
            }
        }
        state
    }

    /// Occurrences currently owned by A and by B.
    pub fn counts(&self) -> [usize; 2] {
        self.counts
    }

    /// The strict threshold `G > 1/32` without division: `raw_gain > total · QUANTUM / 8`.
    fn threshold(&self) -> i32 {
        self.len as i32 * (QUANTUM / 8)
    }

    fn tie_key(&self, movement: Move) -> (u8, u8, Element, Element) {
        match movement {
            Move::Transfer(i) => (0, self.owner[i], self.elements[i], Element::default()),
            Move::Swap(i, j) => (1, 0, self.elements[i], self.elements[j]),
        }
    }

    fn consider(&self, best: &mut Option<Decision>, candidate: Decision) {
        if candidate.gain <= self.threshold() {
            return;
        }
        let replace = match *best {
            None => true,
            Some(old) => {
                candidate.gain > old.gain
                    || (candidate.gain == old.gain
                        && self.tie_key(candidate.movement) < self.tie_key(old.movement))
            }
        };
        if replace {
            *best = Some(candidate);
        }
    }

    fn choose(&self) -> Option<Decision> {
        let mut best = None;
        let capacity_limited = self.counts.contains(&CAPACITY);
        let mut leaders: [Option<usize>; 2] = [None, None];
        for i in 0..self.len {
            let side = self.owner[i] as usize;
            if capacity_limited
                && leaders[side].is_none_or(|old| {
                    self.gain[i] > self.gain[old]
                        || (self.gain[i] == self.gain[old] && self.elements[i] < self.elements[old])
                })
            {
                leaders[side] = Some(i);
            }
            if self.counts[1 - side] < CAPACITY {
                self.consider(&mut best, Decision { movement: Move::Transfer(i), gain: self.gain[i] });
            }
        }
        // At capacity some transfers are impossible: check EVERY swap, including constituents whose
        // individual transfer scores are negative. Below capacity on both sides swaps are not part of
        // this version of the law.
        if capacity_limited {
            let (Some(lead_a), Some(lead_b)) = (leaders[0], leaders[1]) else { return best };
            let gain = self.gain[lead_a] + self.gain[lead_b]
                - 2 * fit_raw(self.elements[lead_a], self.elements[lead_b]);
            self.consider(&mut best, Decision { movement: Move::Swap(lead_a, lead_b), gain });
            // Exact pruning, not a heuristic: fit_raw ≥ −8·QUANTUM, so the best possible correction
            // −2·fit_raw is at most 16·QUANTUM. Candidates able to tie stay eligible.
            let maximum_correction = 16 * QUANTUM;
            for i in 0..self.len {
                if self.owner[i] != 0 {
                    continue;
                }
                let cutoff = best.map_or(self.threshold() + 1, |d| d.gain);
                if self.gain[i] + self.gain[lead_b] + maximum_correction < cutoff {
                    continue;
                }
                for j in 0..self.len {
                    if self.owner[j] != 1 {
                        continue;
                    }
                    let cutoff = best.map_or(self.threshold() + 1, |d| d.gain);
                    if self.gain[i] + self.gain[j] + maximum_correction < cutoff {
                        continue;
                    }
                    let gain =
                        self.gain[i] + self.gain[j] - 2 * fit_raw(self.elements[i], self.elements[j]);
                    self.consider(&mut best, Decision { movement: Move::Swap(i, j), gain });
                }
            }
        }
        best
    }

    fn describe(&self, decision: Decision) -> Operation {
        let change = match decision.movement {
            Move::Transfer(i) => Change::Transfer { from: self.owner[i], element: self.elements[i] },
            Move::Swap(i, j) => Change::Swap { from_a: self.elements[i], from_b: self.elements[j] },
        };
        Operation { change, raw_gain: decision.gain, total_elements: self.len as u8 }
    }

    /// The operation [`step`](Self::step) would commit, without committing it.
    pub fn peek(&self) -> Option<Operation> {
        self.choose().map(|d| self.describe(d))
    }

    /// Commit at most one transfer or atomic swap, updating every cached score exactly in one scan.
    /// `None` means quiescent for these two exact configurations.
    pub fn step(&mut self) -> Option<Operation> {
        let decision = self.choose()?;
        let operation = self.describe(decision);
        match decision.movement {
            Move::Transfer(p) => {
                let from = self.owner[p];
                for i in 0..self.len {
                    if i == p {
                        continue;
                    }
                    let k = fit_raw(self.elements[i], self.elements[p]);
                    let holding_delta = if self.owner[i] == from { -k } else { k };
                    self.holding[i] += holding_delta;
                    self.gain[i] -= 2 * holding_delta;
                }
                self.holding[p] += self.gain[p];
                self.gain[p] = -self.gain[p];
                self.owner[p] = 1 - from;
                self.counts[from as usize] -= 1;
                self.counts[(1 - from) as usize] += 1;
            }
            Move::Swap(p, q) => {
                // p belongs to A, q to B. Both changes in one transaction.
                let pq = fit_raw(self.elements[p], self.elements[q]);
                for i in 0..self.len {
                    if i == p || i == q {
                        continue;
                    }
                    let difference =
                        fit_raw(self.elements[i], self.elements[p]) - fit_raw(self.elements[i], self.elements[q]);
                    let holding_delta = if self.owner[i] == 0 { -difference } else { difference };
                    self.holding[i] += holding_delta;
                    self.gain[i] -= 2 * holding_delta;
                }
                self.holding[p] += self.gain[p] - pq;
                self.holding[q] += self.gain[q] - pq;
                self.gain[p] = -self.gain[p] + 2 * pq;
                self.gain[q] = -self.gain[q] + 2 * pq;
                self.owner[p] = 1;
                self.owner[q] = 0;
            }
        }
        Some(operation)
    }

    /// Both blocks as they are now, with valid support caches (storage order, not canonical).
    pub fn blocks(&self) -> [Block; 2] {
        let mut result = [Block::default(), Block::default()];
        for i in 0..self.len {
            let block = &mut result[self.owner[i] as usize];
            let n = block.len as usize;
            block.elements[n] = self.elements[i];
            block.holding[n] = self.holding[i];
            block.len += 1;
        }
        result
    }
}

/// Attempt ONE operation between `a` and `b` (canonical physical order). On success both blocks are
/// updated (canonicalized, caches valid) and the operation returned; on `None` neither changed. The
/// caller commits A and B together and wakes the other contacts of both.
pub fn react_once(a: &mut Block, b: &mut Block) -> Option<Operation> {
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let mut contact = Contact::new(a, b);
    let operation = contact.step()?;
    let [mut next_a, mut next_b] = contact.blocks();
    next_a.canonicalize();
    next_b.canonicalize();
    *a = next_a;
    *b = next_b;
    Some(operation)
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
pub(crate) mod tests;
