//! Reactions are properties that *emerge* from combining elements — alloys that
//! are harder than either ingredient, mixes that gain a behaviour neither element
//! has alone. A reaction activates whenever its reagent elements are all present,
//! and its strength peaks when their ratios match the reaction's optimum, tapering
//! off as the mix drifts away.
//!
//! Like derivation, reaction matching runs once per block at registration, so the
//! naive "test every reaction" scan here never touches the frame budget.
use crate::block::composition::Composition;
use crate::block::element::{CoreProperties, ElementId};

/// An emergent property a reaction can grant that isn't one of the nine core
/// properties. A tagged seam for behaviours later systems will read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmergentKind {
    /// Conducts signals along a controlled path.
    Superconductive,
    /// Generates power on its own.
    Reactive,
}

/// What an active reaction does to the block it fires in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReactionEffect {
    /// Add these core-property bonuses, each scaled by the reaction's strength.
    CoreBonus(CoreProperties),
    /// Grant an emergent property at the given peak value, scaled by strength.
    Emergent(EmergentKind, u8),
}

/// A recipe for an emergent property: the reagent elements with the ratios that
/// give the strongest effect, and what that effect is.
#[derive(Clone, Debug)]
pub struct Reaction {
    pub name: Box<str>,
    /// Reagent elements paired with their optimal whole-percent share. The shares
    /// are relative to one another (they sum to 100 across the reagents).
    pub reagents: Box<[(ElementId, u8)]>,
    pub effect: ReactionEffect,
}

/// A reaction that fired in a particular block, with the strength it reached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveReaction {
    pub name: Box<str>,
    /// `0..=255`: peaks at `255` when the block's reagent ratios match the optimum.
    pub strength: u8,
    pub effect: ReactionEffect,
}

/// Owns the reaction recipes and matches them against compositions. Built once;
/// mods add recipes through [`register`](Self::register).
pub struct ReactionRegistry {
    reactions: Vec<Reaction>,
}

impl ReactionRegistry {
    /// A registry preloaded with the built-in reactions.
    pub fn with_builtins() -> Self {
        Self {
            reactions: builtin_reactions(),
        }
    }

    /// Add a reaction recipe.
    pub fn register(&mut self, reaction: Reaction) {
        self.reactions.push(reaction);
    }

    /// Every reaction whose reagents are all present in `comp`, each with its
    /// computed strength. The strength compares the block's reagent ratios (the
    /// reagents renormalised to 100% among themselves, ignoring inert filler) to
    /// the reaction's optimum via L1 distance: identical ratios give `255`, and it
    /// falls linearly to `0` at the maximum possible divergence.
    pub fn active_for(&self, comp: &Composition) -> Box<[ActiveReaction]> {
        let weights = comp.weights();
        let mut active = Vec::new();

        for reaction in &self.reactions {
            // Sum the weights of just this reaction's reagents; skip if any are absent.
            let reagent_total: u32 = reaction
                .reagents
                .iter()
                .map(|&(id, _)| weight_of(&weights, id))
                .sum();
            let all_present = reaction
                .reagents
                .iter()
                .all(|&(id, _)| weight_of(&weights, id) > 0);
            if !all_present || reagent_total == 0 {
                continue;
            }

            // L1 distance between actual reagent ratios and the optimum, in percent.
            // Two distributions over the same support differ by at most 200.
            let distance: u32 = reaction
                .reagents
                .iter()
                .map(|&(id, optimal)| {
                    let actual = weight_of(&weights, id) * 100 / reagent_total;
                    actual.abs_diff(optimal as u32)
                })
                .sum();
            let strength = (255 * (200u32.saturating_sub(distance)) / 200) as u8;

            active.push(ActiveReaction {
                name: reaction.name.clone(),
                strength,
                effect: reaction.effect,
            });
        }

        active.into_boxed_slice()
    }
}

/// The weight of a single element within a composition's weight list (`0` if absent).
fn weight_of(weights: &[(ElementId, u16)], id: ElementId) -> u32 {
    weights
        .iter()
        .find(|&&(e, _)| e == id)
        .map(|&(_, w)| w as u32)
        .unwrap_or(0)
}

/// Fold every active reaction's effect into a block's derived core properties.
/// `CoreBonus` adds (saturating) each field scaled by `strength/255`; `Emergent`
/// effects don't touch core properties (they live on the block record for now).
pub fn apply_reactions(mut core: CoreProperties, active: &[ActiveReaction]) -> CoreProperties {
    for reaction in active {
        if let ReactionEffect::CoreBonus(bonus) = reaction.effect {
            let scale = |v: u8| ((v as u32 * reaction.strength as u32) / 255) as u8;
            core.durability = core.durability.saturating_add(scale(bonus.durability));
            core.hardness = core.hardness.saturating_add(scale(bonus.hardness));
            core.conductivity = core.conductivity.saturating_add(scale(bonus.conductivity));
            core.thermal_conductivity = core
                .thermal_conductivity
                .saturating_add(scale(bonus.thermal_conductivity));
            core.density = core.density.saturating_add(scale(bonus.density));
            core.temperature_resistance = core
                .temperature_resistance
                .saturating_add(scale(bonus.temperature_resistance));
            core.friction = core.friction.saturating_add(scale(bonus.friction));
            core.light_emission = core.light_emission.saturating_add(scale(bonus.light_emission));
            core.transparency = core.transparency.saturating_add(scale(bonus.transparency));
        }
    }
    core
}

/// The built-in reaction table.
fn builtin_reactions() -> Vec<Reaction> {
    use crate::block::element::El;
    vec![
        // Bronze-like alloy: copper + iron, best near 60/40, yields a hardness and
        // durability boost neither metal reaches alone.
        Reaction {
            name: "Alloy".into(),
            reagents: Box::from([(El::Copper.id(), 60), (El::Iron.id(), 40)]),
            effect: ReactionEffect::CoreBonus(CoreProperties {
                hardness: 60,
                durability: 40,
                ..Default::default()
            }),
        },
        // Charged crystal: sulfur + copper become a self-generating power source.
        Reaction {
            name: "Cell".into(),
            reagents: Box::from([(El::Sulfur.id(), 50), (El::Copper.id(), 50)]),
            effect: ReactionEffect::Emergent(EmergentKind::Reactive, 200),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::element::El;

    #[test]
    fn fires_only_when_all_reagents_present() {
        let reg = ReactionRegistry::with_builtins();
        // Copper alone: the alloy needs iron too, so nothing fires.
        let copper_only = Composition::natural(&[El::Copper.id()]);
        assert!(reg.active_for(&copper_only).is_empty());

        // Copper + iron: the alloy fires.
        let alloy = Composition::natural(&[El::Copper.id(), El::Iron.id()]);
        let active = reg.active_for(&alloy);
        assert!(active.iter().any(|r| r.name.as_ref() == "Alloy"));
    }

    #[test]
    fn strength_peaks_at_optimal_ratio() {
        let reg = ReactionRegistry::with_builtins();
        // Exact 60/40 copper/iron hits the optimum: full strength.
        let optimal =
            Composition::mixture(&[(El::Copper.id(), 60), (El::Iron.id(), 40)]).unwrap();
        let peak = reg.active_for(&optimal);
        let peak = peak.iter().find(|r| r.name.as_ref() == "Alloy").unwrap();
        assert_eq!(peak.strength, 255);

        // A lopsided 90/10 mix activates but is weaker.
        let off = Composition::mixture(&[(El::Copper.id(), 90), (El::Iron.id(), 10)]).unwrap();
        let off = reg.active_for(&off);
        let off = off.iter().find(|r| r.name.as_ref() == "Alloy").unwrap();
        assert!(off.strength < peak.strength);
    }

    #[test]
    fn core_bonus_is_scaled_by_strength() {
        // At full strength the bonus applies in full.
        let bonus = CoreProperties { hardness: 60, ..Default::default() };
        let active = vec![ActiveReaction {
            name: "x".into(),
            strength: 255,
            effect: ReactionEffect::CoreBonus(bonus),
        }];
        let core = apply_reactions(CoreProperties { hardness: 10, ..Default::default() }, &active);
        assert_eq!(core.hardness, 70);

        // At half strength, half the bonus.
        let half = vec![ActiveReaction {
            name: "x".into(),
            strength: 127,
            effect: ReactionEffect::CoreBonus(bonus),
        }];
        let core = apply_reactions(CoreProperties::default(), &half);
        assert_eq!(core.hardness, ((60u32 * 127) / 255) as u8);
    }
}
