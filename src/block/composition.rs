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

/// An exact element mixture: each element paired with a whole-percent share. The
/// shares always sum to 100 (enforced by [`Mix::new`]). Shared by `Mixture` and
/// `Configuration`, which differ only in whether spatial layout matters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mix(pub Box<[(ElementId, u8)]>);

/// Why a [`Mix`] was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixError {
    /// The percentages did not add up to exactly 100.
    NotHundred(u32),
    /// A mixture needs at least one element.
    Empty,
}

impl std::fmt::Display for MixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixError::NotHundred(got) => write!(f, "mixture percentages sum to {got}, not 100"),
            MixError::Empty => write!(f, "mixture must contain at least one element"),
        }
    }
}

impl Mix {
    /// Build a mixture, validating that the shares sum to exactly 100.
    pub fn new(parts: &[(ElementId, u8)]) -> Result<Self, MixError> {
        if parts.is_empty() {
            return Err(MixError::Empty);
        }
        let sum: u32 = parts.iter().map(|&(_, p)| p as u32).sum();
        if sum != 100 {
            return Err(MixError::NotHundred(sum));
        }
        Ok(Mix(Box::from(parts)))
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

    /// The (element, weight) pairs that drive derivation. Natural blocks weight
    /// every element equally (`1` each) so the average is exact in element count
    /// rather than lossy percentages; mixtures weight by percentage. The caller
    /// divides by the weight sum, so the units only need to be consistent.
    pub fn weights(&self) -> Box<[(ElementId, u16)]> {
        match self {
            Composition::Natural(els) => els.iter().map(|&e| (e, 1u16)).collect(),
            Composition::Mixture(mix) | Composition::Configuration { mix, .. } => {
                mix.0.iter().map(|&(e, p)| (e, p as u16)).collect()
            }
            // Computational blocks have no element composition to average yet.
            Composition::Computational(_) => Box::from([]),
        }
    }

    /// The distinct elements present, regardless of amount.
    pub fn elements(&self) -> Box<[ElementId]> {
        match self {
            Composition::Natural(els) => els.clone(),
            Composition::Mixture(mix) | Composition::Configuration { mix, .. } => {
                mix.0.iter().map(|&(e, _)| e).collect()
            }
            Composition::Computational(_) => Box::from([]),
        }
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
        assert_eq!(&*comp.weights(), &[(ElementId(3), 1), (ElementId(5), 1)]);
        assert!(!comp.is_empty());
    }

    #[test]
    fn empty_natural_is_empty() {
        assert!(Composition::natural(&[]).is_empty());
    }
}
