//! What a block is *made of*. Composition is the input to property derivation;
//! everything observable about a block (solidity, colour, the nine core
//! properties, specials, reactions) falls out of it.
//!
//! The four variants mirror the documented block hierarchy, from cheapest to most
//! expressive. Only [`Natural`](Composition::Natural) and
//! [`Mixture`](Composition::Mixture) are wired end-to-end today; `Configuration`
//! and `Computational` exist so the type — and every call site keyed on
//! [`BlockId`](crate::block::BlockId) — stays stable while their interiors are
//! filled in later.
use crate::block::element::ElementId;

/// Each distinct element paired with its total weight, sorted by
/// [`ElementId`], with no duplicate keys.
///
/// The inner field is private to this module; the only constructor is
/// [`Composition::weights`], which always sorts and merges duplicates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weights(Box<[(ElementId, u32)]>);

impl Weights {
    /// The `(element, total weight)` pairs, sorted by id with duplicates merged.
    pub fn parts(&self) -> &[(ElementId, u32)] {
        &self.0
    }

    /// Whether the multiset carries no elements (only air / computational).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The sum of every element's weight — the divisor for the weighted mean.
    pub fn total(&self) -> u32 {
        self.0.iter().map(|&(_, w)| w).sum()
    }

    /// The total weight of one element, or `0` if it is absent.
    pub fn weight_of(&self, id: ElementId) -> u32 {
        self.0
            .binary_search_by_key(&id, |&(e, _)| e)
            .map(|i| self.0[i].1)
            .unwrap_or(0)
    }
}

/// An exact element mixture: each element paired with a whole-percent share. The
/// shares always sum to 100 (enforced by [`Mix::new`]). Shared by `Mixture` and
/// `Configuration`, which differ only in whether spatial layout matters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mix(Box<[(ElementId, u8)]>);

/// Why a [`Mix`] was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixError {
    /// The percentages did not add up to exactly 100.
    NotHundred(u32),
    /// A mixture needs at least one element.
    Empty,
    /// The block palette is already at its capacity cap.
    Full,
}

impl std::fmt::Display for MixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixError::NotHundred(got) => write!(f, "mixture percentages sum to {got}, not 100"),
            MixError::Empty => write!(f, "mixture must contain at least one element"),
            MixError::Full => write!(f, "block palette is full"),
        }
    }
}

impl Mix {
    /// Build a mixture, validating that the shares sum to exactly 100.
    pub fn new(parts: &[(ElementId, u8)]) -> Result<Self, MixError> {
        // Normalize first: merge duplicate ids (summing their shares), drop
        // zero shares, and sort by id. Validation runs on the normalized form.
        let mut norm: Vec<(ElementId, u8)> = Vec::with_capacity(parts.len());
        for &(e, p) in parts {
            if p == 0 {
                continue;
            }
            match norm.iter_mut().find(|(ne, _)| *ne == e) {
                Some((_, np)) => *np = np.saturating_add(p),
                None => norm.push((e, p)),
            }
        }
        norm.sort_by_key(|&(e, _)| e);
        if norm.is_empty() {
            return Err(MixError::Empty);
        }
        let sum: u32 = norm.iter().map(|&(_, p)| p as u32).sum();
        if sum != 100 {
            return Err(MixError::NotHundred(sum));
        }
        Ok(Mix(norm.into_boxed_slice()))
    }

    /// The `(element, whole-percent share)` pairs, guaranteed to sum to 100.
    pub fn parts(&self) -> &[(ElementId, u8)] {
        &self.0
    }
}

/// Opaque placeholder for a configuration block's spatial arrangement of elements.
/// Arrangement only affects routing (e.g. directing electricity), which is
/// deferred; derived *properties* ignore it, so today it carries no data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout;

/// Opaque placeholder for a computational block's logic-gate graph. Built in a
/// special crafter; the gate model and signal routing are deferred.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Computer;

/// What a block is made of, in increasing order of expressiveness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Composition {
    /// Elements in unspecified, equal parts. Cheapest and what terrain generation
    /// emits. An empty set is the one and only non-solid block (air).
    Natural(Box<[ElementId]>),
    /// Exact element percentages. The first craftable tier; supports specials and
    /// reactions.
    Mixture(Mix),
    /// Exact percentages *and* a spatial arrangement. Derives like a mixture today;
    /// `layout` drives routing later.
    Configuration { mix: Mix, layout: Layout },
    /// A graph of logic-gate components. Interior deferred.
    Computational(Computer),
}

impl Composition {
    /// Convenience constructor for an equal-parts natural block.
    pub fn natural(elements: &[ElementId]) -> Self {
        Composition::Natural(Box::from(elements))
    }

    /// Convenience constructor for a validated mixture.
    pub fn mixture(parts: &[(ElementId, u8)]) -> Result<Self, MixError> {
        Mix::new(parts).map(Composition::Mixture)
    }

    /// The [`Weights`] multiset driving derivation: each distinct
    /// element with its total weight. Natural blocks weight every element
    /// equally (`1` each); mixtures weight by percentage. The caller divides by
    /// the weight sum, so the units only need to be consistent.
    pub fn weights(&self) -> Weights {
        let mut raw: Vec<(ElementId, u32)> = match self {
            Composition::Natural(els) => els.iter().map(|&e| (e, 1u32)).collect(),
            Composition::Mixture(mix) | Composition::Configuration { mix, .. } => {
                mix.0.iter().map(|&(e, p)| (e, p as u32)).collect()
            }
            // Computational blocks have no element composition to average yet.
            Composition::Computational(_) => Vec::new(),
        };
        raw.sort_by_key(|&(e, _)| e);
        // Merge adjacent equal ids, summing their weights.
        raw.dedup_by(|&mut (e, w), &mut (pe, ref mut pw)| {
            if e == pe {
                *pw += w;
                true
            } else {
                false
            }
        });
        Weights(raw.into_boxed_slice())
    }

    /// The distinct elements present, regardless of amount.
    pub fn elements(&self) -> Box<[ElementId]> {
        self.weights().parts().iter().map(|&(e, _)| e).collect()
    }

    /// Whether the composition carries no material at all. Only air is empty, and
    /// only air is non-solid.
    pub fn is_empty(&self) -> bool {
        match self {
            Composition::Natural(els) => els.is_empty(),
            Composition::Mixture(mix) | Composition::Configuration { mix, .. } => mix.0.is_empty(),
            // A computational block is a built object — solid even with no elements.
            Composition::Computational(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_must_sum_to_one_hundred() {
        let a = ElementId(0);
        let b = ElementId(1);
        assert_eq!(Mix::new(&[(a, 70), (b, 30)]).is_ok(), true);
        assert_eq!(Mix::new(&[(a, 70), (b, 20)]), Err(MixError::NotHundred(90)));
        assert_eq!(Mix::new(&[]), Err(MixError::Empty));
    }

    #[test]
    fn natural_weights_are_one_each() {
        let comp = Composition::natural(&[ElementId(3), ElementId(5)]);
        assert_eq!(comp.weights().parts(), &[(ElementId(3), 1), (ElementId(5), 1)]);
        assert!(!comp.is_empty());
    }

    #[test]
    fn weights_aggregate_duplicates() {
        // A natural block listing the same element more than once reduces to one
        // entry with the summed weight, sorted by id.
        let (copper, iron) = (ElementId(5), ElementId(2));
        let comp = Composition::natural(&[copper, copper, iron]);
        let w = comp.weights();
        assert_eq!(w.parts(), &[(iron, 1), (copper, 2)], "deduped and sorted by id");
        assert_eq!(w.weight_of(copper), 2, "both occurrences summed");
        assert_eq!(w.weight_of(iron), 1);
        assert_eq!(w.weight_of(ElementId(99)), 0, "absent element weighs 0");
        assert_eq!(w.total(), 3);
    }

    #[test]
    fn mix_normalizes_duplicate_ids() {
        let a = ElementId(1);
        let b = ElementId(0);
        // Duplicate `a` (30 + 30) and an out-of-order `b`: normalizes to sorted,
        // merged form and passes the post-merge sum check.
        let mix = Mix::new(&[(a, 30), (b, 40), (a, 30)]).unwrap();
        assert_eq!(mix.parts(), &[(b, 40), (a, 60)], "merged and sorted");

        // A zero share is dropped before validation.
        let dropped = Mix::new(&[(a, 100), (b, 0)]).unwrap();
        assert_eq!(dropped.parts(), &[(a, 100)]);

        // elements() reports exactly the weights() keys, in the same order.
        let comp = Composition::Mixture(mix);
        let weight_keys: Box<[ElementId]> =
            comp.weights().parts().iter().map(|&(e, _)| e).collect();
        assert_eq!(comp.elements(), weight_keys, "elements() == weights() keys");
    }
}
