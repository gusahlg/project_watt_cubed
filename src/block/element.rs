//! Elements are the atoms of the world: every block is a combination of them, and
//! a block's behaviour is *derived* from the properties of the elements it contains
//! (see [`derive`](crate::block::derive)). This module defines an element's
//! properties, the built-in element table, and the registry that hands out stable
//! [`ElementId`]s.
//!
//! The registry is built once at startup and read-only afterwards. Element records
//! are cold data — the per-voxel render/collision path never touches them; it only
//! reads the precomputed arrays on the [`BlockRegistry`](crate::block::BlockRegistry).
use voxel_engine::Color;

use crate::macros::elements;

/// A stable handle to an element in the [`ElementRegistry`]. The built-in elements
/// keep the same ids forever (declaration order); mods append after them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ElementId(pub u16);

/// Declares [`CoreProperties`] and its bidirectional [`Core`] array conversions
/// from a single ordered field list — the one place the field order and the
/// name-to-index mapping live. Reordering a line here reorders the struct and
/// both `From` impls together.
macro_rules! core_properties {
    ($($(#[$doc:meta])* $field:ident),* $(,)?) => {
        /// The nine properties every element has. A block inherits each one as the
        /// weighted average of its elements (see [`derive_core`](crate::block::derive::derive_core)).
        ///
        /// All values are `0..=255` except `transparency`, which reads as a percentage
        /// (`0..=100`). `Default` is all-zero, which is what an empty (air) block derives.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct CoreProperties {
            $($(#[$doc])* pub $field: u8,)*
        }

        impl From<CoreProperties> for Core {
            fn from(c: CoreProperties) -> Self {
                Core([$(c.$field),*])
            }
        }

        impl From<Core> for CoreProperties {
            fn from(f: Core) -> Self {
                let [$($field),*] = f.0;
                CoreProperties { $($field),* }
            }
        }

        /// The number of core properties, derived from the field list so it can't
        /// drift out of sync with the struct.
        pub const CORE_FIELD_COUNT: usize = {
            const FIELDS: &[&str] = &[$(stringify!($field)),*];
            FIELDS.len()
        };
    };
}

core_properties! {
    /// Damage a block absorbs before it breaks.
    durability,
    /// Damage floor: attacks weaker than this do nothing.
    hardness,
    /// How readily electricity flows through it.
    conductivity,
    /// How readily heat flows through it.
    thermal_conductivity,
    /// Weight; heavier blocks are harder to move and push others less.
    density,
    /// Highest temperature it survives before degrading.
    temperature_resistance,
    /// Grip on neighbours: if this exceeds an adjacent block's density it drags it along.
    friction,
    /// Light output, `0` (none) to `255` (brightest).
    light_emission,
    /// Light transmission as a percentage, `0` (opaque) to `100` (clear).
    transparency,
}

/// The nine core fields as a fixed-order array, so property math can loop instead
/// of spelling out every field. Used by
/// [`derive_core`](crate::block::derive::derive_core) and
/// [`apply_reactions`](crate::block::reaction::apply_reactions).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Core(pub [u8; CORE_FIELD_COUNT]);

impl Core {
    /// The weighted mean of several cores. `parts` yields `(core, weight)`; a zero
    /// total weight (an empty block) blends to all-zero.
    pub fn blend(parts: impl IntoIterator<Item = (Core, u32)>) -> Core {
        crate::block::bary::barycenter(parts)
    }

    /// Add `bonus` to each field (saturating), first scaling it by `strength/255`.
    pub fn add_scaled(&mut self, bonus: Core, strength: u8) {
        for i in 0..CORE_FIELD_COUNT {
            let scaled = (bonus.0[i] as u32 * strength as u32 / 255) as u8;
            self.0[i] = self.0[i].saturating_add(scaled);
        }
    }
}

/// Extra behaviours only some elements carry. Each holds an intrinsic strength
/// (`0..=255`); a block scales it by how much of the element it contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialProperty {
    /// Releases an explosion when the block breaks.
    ExplosionAtBreakage(u8),
    /// Attracts/repels along magnetic lines.
    Magnetism(u8),
    /// Eats away at adjacent blocks over time.
    Corrosion(u8),
    /// Converts a heat differential into electricity.
    HeatToElectricity(u8),
    /// Acts as a battery.
    ElectricityStorage(u8),
}

/// The tag of a [`SpecialProperty`] without its strength — used to group and
/// average contributions from several elements that share the same behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpecialKind {
    ExplosionAtBreakage,
    Magnetism,
    Corrosion,
    HeatToElectricity,
    ElectricityStorage,
}

impl SpecialProperty {
    /// Which behaviour this is, ignoring strength.
    pub fn kind(self) -> SpecialKind {
        match self {
            SpecialProperty::ExplosionAtBreakage(_) => SpecialKind::ExplosionAtBreakage,
            SpecialProperty::Magnetism(_) => SpecialKind::Magnetism,
            SpecialProperty::Corrosion(_) => SpecialKind::Corrosion,
            SpecialProperty::HeatToElectricity(_) => SpecialKind::HeatToElectricity,
            SpecialProperty::ElectricityStorage(_) => SpecialKind::ElectricityStorage,
        }
    }

    /// The intrinsic strength of this behaviour.
    pub fn strength(self) -> u8 {
        match self {
            SpecialProperty::ExplosionAtBreakage(v)
            | SpecialProperty::Magnetism(v)
            | SpecialProperty::Corrosion(v)
            | SpecialProperty::HeatToElectricity(v)
            | SpecialProperty::ElectricityStorage(v) => v,
        }
    }
}

/// One element: a name, its physical properties, a tint that blocks blend by
/// composition, and any special behaviours. This is cold, build-once data.
#[derive(Clone, Debug)]
pub struct Element {
    pub name: Box<str>,
    pub core: CoreProperties,
    /// The colour this element lends to blocks containing it.
    pub color: Color,
    pub specials: Box<[SpecialProperty]>,
}

/// Owns every known element and maps [`ElementId`] to its record. Built-ins are
/// registered first so their ids match the [`El`] constants; mods append afterwards.
pub struct ElementRegistry {
    elements: Vec<Element>,
}

impl ElementRegistry {
    /// A registry preloaded with the built-in element table.
    pub fn with_builtins() -> Self {
        Self {
            elements: builtin_elements(),
        }
    }

    /// Add an element and return its fresh id. The modding entry point for new
    /// materials.
    pub fn register(&mut self, element: Element) -> ElementId {
        let id = ElementId(
            u16::try_from(self.elements.len()).expect("element palette exceeded u16::MAX"),
        );
        self.elements.push(element);
        id
    }

    /// The record for an id. Panics on an unknown id — ids only come from this
    /// registry, so an unknown one is a bug, not user input.
    pub fn get(&self, id: ElementId) -> &Element {
        &self.elements[id.0 as usize]
    }

    /// Find an element by its (case-sensitive) name. Cold path: used when loading
    /// saved inventories/edits that reference elements by name for portability.
    pub fn id_by_name(&self, name: &str) -> Option<ElementId> {
        self.elements
            .iter()
            .position(|e| e.name.as_ref() == name)
            .map(|i| ElementId(i as u16))
    }

    /// Number of registered elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether the registry holds no elements (only possible before built-ins load).
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

// The built-in element palette. Adding a material is one row here; the macro
// generates the `El` id constants (declaration order) and `builtin_elements()`.
//
// Values are illustrative starting points, not balanced game design — the point
// is a representative spread so derivation, specials, and reactions have something
// to chew on. `core { .. }` lists only the non-zero properties; the rest default to 0.
elements! {
    // Heavy, hard, inert — the bedrock of the world.
    Stone => {
        color: (128, 128, 128),
        core: { durability: 120, hardness: 180, density: 210, temperature_resistance: 230, friction: 90, thermal_conductivity: 35 },
    },
    // Loose earth: soft, gritty, middling everything.
    Soil => {
        color: (121, 85, 58),
        core: { durability: 30, hardness: 20, density: 110, temperature_resistance: 120, friction: 70, thermal_conductivity: 25 },
    },
    // Binds soil; a touch denser and stickier.
    Clay => {
        color: (150, 100, 70),
        core: { durability: 45, hardness: 35, density: 130, temperature_resistance: 150, friction: 80, thermal_conductivity: 20 },
    },
    // Living matter: light, vivid green, flammable in spirit (low temp resistance).
    Organic => {
        color: (86, 176, 0),
        core: { durability: 15, hardness: 8, density: 40, temperature_resistance: 60, friction: 65, thermal_conductivity: 15, light_emission: 0 },
    },
    // The conductor: routes electricity and heat readily; harvests heat too.
    Copper => {
        color: (184, 115, 51),
        core: { durability: 90, hardness: 60, density: 160, temperature_resistance: 200, friction: 50, conductivity: 240, thermal_conductivity: 220 },
        specials: [HeatToElectricity(200)],
    },
    // Strong and magnetic; rusts (corrodes) and the base of alloys.
    Iron => {
        color: (130, 110, 100),
        core: { durability: 200, hardness: 150, density: 200, temperature_resistance: 220, friction: 55, conductivity: 120, thermal_conductivity: 160 },
        specials: [Magnetism(180), Corrosion(40)],
    },
    // Volatile yellow mineral: stores charge, blows up when broken.
    Sulfur => {
        color: (220, 200, 40),
        core: { durability: 25, hardness: 30, density: 90, temperature_resistance: 80, friction: 60, conductivity: 10 },
        specials: [ExplosionAtBreakage(160), ElectricityStorage(140)],
    },
    // Clear, brittle, lets light through.
    Glass => {
        color: (210, 235, 240),
        core: { durability: 20, hardness: 50, density: 100, temperature_resistance: 180, friction: 20, transparency: 90 },
    },
    // Glowing crystal: a built-in light source.
    Lumin => {
        color: (255, 240, 150),
        core: { durability: 40, hardness: 30, density: 70, temperature_resistance: 160, friction: 40, light_emission: 230, transparency: 30 },
    },
    // Compressed ancient organics: soft, light, and eager to burn.
    Coal => {
        color: (40, 35, 30),
        core: { durability: 20, hardness: 15, density: 65, temperature_resistance: 50, friction: 55, conductivity: 30, thermal_conductivity: 20 },
    },
    // Loose grains: barely holds itself together, let alone anything else.
    Sand => {
        color: (220, 200, 150),
        core: { durability: 15, hardness: 12, density: 120, temperature_resistance: 170, friction: 30, thermal_conductivity: 20 },
    },
    // Frozen water: slick underfoot and gone at the first warm breeze.
    Ice => {
        color: (180, 220, 240),
        core: { durability: 25, hardness: 20, density: 85, temperature_resistance: 20, friction: 5, thermal_conductivity: 40, transparency: 60 },
    },
    // The hoard metal: soft, absurdly heavy, conducts everything but restraint.
    Gold => {
        color: (255, 200, 60),
        core: { durability: 60, hardness: 40, density: 250, temperature_resistance: 210, friction: 30, conductivity: 255, thermal_conductivity: 250 },
    },
    // Dense, dull, dependable — all shielding, none of the shine.
    Lead => {
        color: (90, 90, 110),
        core: { durability: 110, hardness: 70, density: 255, temperature_resistance: 190, friction: 45, conductivity: 60, thermal_conductivity: 80 },
    },
    // Piezo crystal: squeeze it and it keeps the charge for later.
    Quartz => {
        color: (235, 225, 235),
        core: { durability: 100, hardness: 170, density: 150, temperature_resistance: 200, friction: 30, transparency: 40 },
        specials: [ElectricityStorage(90)],
    },
    // Volcanic glass: fears no furnace and very few tools.
    Obsidian => {
        color: (30, 25, 45),
        core: { durability: 240, hardness: 220, density: 190, temperature_resistance: 255, friction: 45, thermal_conductivity: 15 },
    },
    // The overachiever metal: lighter than stone, tougher than iron.
    Titan => {
        color: (170, 180, 190),
        core: { durability: 255, hardness: 240, density: 140, temperature_resistance: 245, friction: 40, conductivity: 80, thermal_conductivity: 90 },
    },
    // Sky-metal that barely believes in gravity — the reason islands fly.
    Aerium => {
        color: (150, 220, 255),
        core: { durability: 35, hardness: 25, density: 5, temperature_resistance: 140, friction: 35, transparency: 20 },
        specials: [Magnetism(60)],
    },
    // Glows all night and goes out with a bang.
    Phosphor => {
        color: (120, 255, 140),
        core: { durability: 20, hardness: 10, density: 60, temperature_resistance: 70, friction: 40, light_emission: 200 },
        specials: [ExplosionAtBreakage(60)],
    },
    // Deep blue and heavy; lets some light through, so blocks made of it are
    // translucent solids (the glass path) — the world's oceans, rivers, and lakes.
    Water => {
        color: (40, 90, 200),
        core: { durability: 5, hardness: 5, density: 200, temperature_resistance: 100, friction: 20, thermal_conductivity: 60, transparency: 55 },
    },
    // Fresh snow: soft, pale, and slick underfoot — the biome frosting on cold or
    // high ground.
    Snow => {
        color: (240, 245, 255),
        core: { durability: 8, hardness: 5, density: 40, temperature_resistance: 20, friction: 10, thermal_conductivity: 30 },
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_field_count_is_nine() {
        assert_eq!(CORE_FIELD_COUNT, 9);
    }

    #[test]
    fn builtin_ids_match_declaration_order() {
        let reg = ElementRegistry::with_builtins();
        assert_eq!(El::Stone.id(), ElementId(0));
        assert_eq!(reg.get(El::Stone.id()).name.as_ref(), "Stone");
        assert_eq!(reg.get(El::Copper.id()).core.conductivity, 240);
    }

    #[test]
    fn specials_carry_kind_and_strength() {
        let reg = ElementRegistry::with_builtins();
        let copper = reg.get(El::Copper.id());
        assert_eq!(copper.specials.len(), 1);
        assert_eq!(copper.specials[0].kind(), SpecialKind::HeatToElectricity);
        assert_eq!(copper.specials[0].strength(), 200);
    }

    #[test]
    fn registering_appends_after_builtins() {
        let mut reg = ElementRegistry::with_builtins();
        let before = reg.len();
        let id = reg.register(Element {
            name: "Test".into(),
            core: CoreProperties::default(),
            color: Color::new(1, 2, 3, 255),
            specials: Box::from([]),
        });
        assert_eq!(id, ElementId(before as u16));
        assert_eq!(reg.len(), before + 1);
    }
}
