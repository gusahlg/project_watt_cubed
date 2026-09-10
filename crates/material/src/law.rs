//! The physics of the universe as a value. A world records its law; peers compare stamps; nothing in
//! the law names a material. Every "open question" of the model is a parameter here with the
//! provisional v0 value, never an architectural assumption.

use crate::element::{Element, D};

/// Number of knots of the per-axis response curve.
pub const KNOTS: usize = 9;
/// Number of interaction event kinds.
pub const EVENT_KINDS: usize = 4;

/// How coordinates behave at the lattice edge (spec §2: unsettled; v0 uses `Clamp`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Boundary {
    /// Coordinates saturate at 0 and 255; differences are plain `a - b`.
    Clamp,
    /// The lattice is a torus: coordinates wrap and differences take the short way round.
    Wrap,
}

/// What kind of world event caused an interaction (spec §1.9: the provisional whitelist).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EventKind {
    /// A block moved and now has (some) new neighbours.
    Moved = 0,
    /// A block was placed against another.
    NewContact = 1,
    /// Blocks collided with force.
    Collision = 2,
    /// A neighbour changed for an external reason (broken, replaced).
    ExternallyChanged = 3,
}

impl EventKind {
    /// All kinds, in stamp order.
    pub const ALL: [EventKind; EVENT_KINDS] = [
        EventKind::Moved,
        EventKind::NewContact,
        EventKind::Collision,
        EventKind::ExternallyChanged,
    ];
}

/// The elementary law F(δ) → Δ: a per-axis odd response curve over |δ| (piecewise linear through
/// integer knots) mixed across axes by a Q4 matrix, bounded per event by `max_step`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Kernel {
    /// `(distance, response)` knots, distances ascending from 0 to 255. Response < 0 pushes the target
    /// away from the origin element, > 0 pulls it closer.
    pub knots: [(u8, i16); KNOTS],
    /// Cross-axis mixing, Q4 (16 = 1.0): `Δ_i = Σ_j mixing[i][j] · r_j / 16`.
    pub mixing: [[i8; D]; D],
    /// Largest coordinate change one event may cause on one axis.
    pub max_step: u8,
}

/// Per-event-kind strength, Q8 (256 = 1.0), indexed by `EventKind as usize`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EventStrengths(pub [u8; EVENT_KINDS]);

/// The reference elements used to take standardized readings of a configuration (spec §3.3: cached
/// observations of the law, not authored stats). Constants of the universe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Probes {
    /// Contact probe: a low response reads as hard/stable.
    pub contact: Element,
    /// Light probe: the response reads as transparency.
    pub light: Element,
    /// Flow probe: a response at or above `liquid_min` reads as liquid.
    pub flow: Element,
    /// Glow probe: a response at or above `glow_min` reads as light emission.
    pub glow: Element,
    /// Friction probe.
    pub friction: Element,
    /// Strength (Q8) all probes interact with.
    pub strength: u8,
    /// Flow response threshold for "liquid".
    pub liquid_min: u8,
    /// Glow response threshold for "emits light".
    pub glow_min: u8,
    /// Light response below which a material is fully opaque; transparency rises linearly above it.
    pub transparent_min: u8,
}

/// The universe's physics as data. `version` changes whenever any field's MEANING changes; two peers
/// with different stamps do not share a world.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Law {
    /// Law version: bumped whenever the stamp's fields OR any value-path constant of this crate
    /// changes meaning (e.g. the probe response scale), so two peers computing different readings from
    /// the same bytes can never share a world. History: 0 = initial; 1 = response scale ×4 (2026-09-10).
    pub version: u16,
    /// Lattice edge behaviour.
    pub boundary: Boundary,
    /// The elementary interaction law.
    pub kernel: Kernel,
    /// Strength per event kind.
    pub events: EventStrengths,
    /// The observation probes.
    pub probes: Probes,
    /// Seed of the presentation noise (colours are a fixed function of this seed and the lattice).
    pub visual_seed: u32,
    /// Coordinate grid applied at commit time (1 = none). A proliferation valve, measured before use.
    pub quantum: u8,
}

/// Why a stamp did not decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LawError {
    /// Wrong length for the stamp version.
    Length(usize),
    /// Unknown version.
    Version(u16),
    /// A field held an impossible value (knots not ascending, zero max_step, unknown boundary).
    Invalid(&'static str),
}

/// Bytes of a law stamp (version 0).
pub const STAMP_LEN: usize = 2 + 1 + KNOTS * 3 + D * D + 1 + EVENT_KINDS + 5 * D + 4 + 4 + 1;

impl Law {
    /// The provisional law of version 0. Per axis: a DEAD ZONE up to 10 units (near-identical matter
    /// does not react — families and drifted variants stay themselves), repulsion 10..24 (peak at 16),
    /// attraction 24..64 (peak at 40) and NOTHING beyond 64 (very different matter is inert, so a world
    /// of well-separated regions is at rest); identity mixing plus a quarter cyclic coupling, six units
    /// of maximum step, clamped boundaries, no quantization. Reactions therefore happen between
    /// moderately similar materials — chemistry among relatives — and products settle in the flat
    /// band 22..26 where repulsion and attraction balance (a zero-slope band, so integer steps cannot
    /// oscillate across it: pairs come to rest instead of hopping around the ring forever).
    pub const fn v0() -> Law {
        Law {
            version: 1,
            boundary: Boundary::Clamp,
            kernel: Kernel {
                knots: [
                    (0, 0),
                    (10, 0),
                    (16, -8),
                    (22, 0),
                    (26, 0),
                    (40, 10),
                    (56, 4),
                    (64, 0),
                    (255, 0),
                ],
                mixing: [[16, 4, 0, 0], [0, 16, 4, 0], [0, 0, 16, 4], [4, 0, 0, 16]],
                max_step: 6,
            },
            events: EventStrengths([128, 96, 255, 64]),
            probes: Probes {
                contact: Element::new([24, 200, 88, 152]),
                light: Element::new([232, 40, 176, 64]),
                flow: Element::new([96, 96, 232, 24]),
                glow: Element::new([160, 240, 32, 120]),
                friction: Element::new([64, 128, 192, 240]),
                strength: 255,
                liquid_min: 172,
                glow_min: 180,
                transparent_min: 140,
            },
            visual_seed: 0x5ee_d5eed,
            quantum: 1,
        }
    }

    /// Validate the invariants a stamp relies on.
    pub fn validate(&self) -> Result<(), LawError> {
        if self.kernel.max_step == 0 {
            return Err(LawError::Invalid("max_step"));
        }
        if self.kernel.knots[0].0 != 0 || self.kernel.knots[KNOTS - 1].0 != 255 {
            return Err(LawError::Invalid("knot range"));
        }
        for w in self.kernel.knots.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(LawError::Invalid("knots ascending"));
            }
        }
        if self.quantum == 0 {
            return Err(LawError::Invalid("quantum"));
        }
        Ok(())
    }

    /// Canonical bytes of the law: what saves, `Welcome` and the content fingerprint carry.
    pub fn stamp(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(STAMP_LEN);
        v.extend_from_slice(&self.version.to_le_bytes());
        v.push(match self.boundary {
            Boundary::Clamp => 0,
            Boundary::Wrap => 1,
        });
        for (d, r) in self.kernel.knots {
            v.push(d);
            v.extend_from_slice(&r.to_le_bytes());
        }
        for row in self.kernel.mixing {
            for m in row {
                v.push(m as u8);
            }
        }
        v.push(self.kernel.max_step);
        v.extend_from_slice(&self.events.0);
        for e in [
            self.probes.contact,
            self.probes.light,
            self.probes.flow,
            self.probes.glow,
            self.probes.friction,
        ] {
            v.extend_from_slice(&e.0);
        }
        v.push(self.probes.strength);
        v.push(self.probes.liquid_min);
        v.push(self.probes.glow_min);
        v.push(self.probes.transparent_min);
        v.extend_from_slice(&self.visual_seed.to_le_bytes());
        v.push(self.quantum);
        debug_assert_eq!(v.len(), STAMP_LEN);
        v
    }

    /// Inverse of [`Law::stamp`].
    pub fn from_stamp(bytes: &[u8]) -> Result<Law, LawError> {
        if bytes.len() != STAMP_LEN {
            return Err(LawError::Length(bytes.len()));
        }
        let mut i = 0usize;
        let mut take = |n: usize| {
            let s = &bytes[i..i + n];
            i += n;
            s
        };
        let version = u16::from_le_bytes([take(1)[0], take(1)[0]]);
        if version != 1 {
            return Err(LawError::Version(version));
        }
        let boundary = match take(1)[0] {
            0 => Boundary::Clamp,
            1 => Boundary::Wrap,
            _ => return Err(LawError::Invalid("boundary")),
        };
        let mut knots = [(0u8, 0i16); KNOTS];
        for k in knots.iter_mut() {
            let d = take(1)[0];
            let r = i16::from_le_bytes([take(1)[0], take(1)[0]]);
            *k = (d, r);
        }
        let mut mixing = [[0i8; D]; D];
        for row in mixing.iter_mut() {
            for m in row.iter_mut() {
                *m = take(1)[0] as i8;
            }
        }
        let max_step = take(1)[0];
        let mut events = [0u8; EVENT_KINDS];
        events.copy_from_slice(take(EVENT_KINDS));
        let mut probe = || {
            let mut c = [0u8; D];
            c.copy_from_slice(take(D));
            Element(c)
        };
        let contact = probe();
        let light = probe();
        let flow = probe();
        let glow = probe();
        let friction = probe();
        let strength = take(1)[0];
        let liquid_min = take(1)[0];
        let glow_min = take(1)[0];
        let transparent_min = take(1)[0];
        let mut seed = [0u8; 4];
        seed.copy_from_slice(take(4));
        let visual_seed = u32::from_le_bytes(seed);
        let quantum = take(1)[0];
        let law = Law {
            version,
            boundary,
            kernel: Kernel { knots, mixing, max_step },
            events: EventStrengths(events),
            probes: Probes { contact, light, flow, glow, friction, strength, liquid_min, glow_min, transparent_min },
            visual_seed,
            quantum,
        };
        law.validate()?;
        Ok(law)
    }

    /// A 64-bit fingerprint of the stamp (FNV-1a), for the content handshake.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.stamp() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}
