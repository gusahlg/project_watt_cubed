//! The physics of the universe as a value. A world records its law's stamp; peers compare stamps;
//! nothing in the law names a material.
//!
//! The reaction function itself (selective transfer v1: the committed fit table, capacity 32, the
//! strict `1/32` threshold, the tie order) is fixed code in [`crate::kernel`]; the stamp folds a digest
//! of the table so a changed table can never silently share a world with the old one. What remains a
//! value here is how the world is *observed*: the probe elements whose fit with a configuration reads
//! as its operational properties, and the seed of its presentation.

use crate::configuration::CAPACITY;
use crate::element::{Element, D};
use crate::fit_table::FIT;

/// The identifier of the reaction function this crate implements.
pub const LAW_ID: &str = "watt-selective-transfer-v1";

/// Reference elements whose fit with a configuration is read as an operational property (spec: cached
/// observations of the law, never authored stats). Constants of the universe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Probes {
    /// Strong attraction to this element reads as transparency.
    pub light: Element,
    /// Strong attraction to this element reads as light emission.
    pub glow: Element,
    /// Attraction to this element reads as grip (friction).
    pub grip: Element,
}

/// The universe as data. `version` changes whenever any field's MEANING changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Law {
    /// Law version. History: 0-1 = the coordinate-moving response-curve law (2026-09-10);
    /// 2 = selective transfer v1 (2026-10-02).
    pub version: u16,
    /// The observation probes.
    pub probes: Probes,
    /// Seed of the presentation noise.
    pub visual_seed: u32,
}

/// Why a stamp did not decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LawError {
    /// Wrong length.
    Length(usize),
    /// Unknown version.
    Version(u16),
    /// The stamp was made with a different reaction function (capacity, table).
    Function,
}

/// Bytes of a law stamp: version, capacity, fit-table digest, three probes, visual seed.
pub const STAMP_LEN: usize = 2 + 1 + 8 + 3 * D + 4;

/// FNV-1a over the committed fit table: part of every stamp.
fn table_digest() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in FIT {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Law {
    /// The current law: selective transfer v1 observed through three probes.
    pub const fn current() -> Law {
        Law {
            version: 2,
            probes: Probes {
                light: Element::new([232, 40, 176, 64]),
                glow: Element::new([160, 240, 32, 120]),
                grip: Element::new([64, 128, 192, 240]),
            },
            visual_seed: 0x5ee_d5eed,
        }
    }

    /// Canonical bytes of the law: what saves, `Welcome` and the content fingerprint carry.
    pub fn stamp(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(STAMP_LEN);
        v.extend_from_slice(&self.version.to_le_bytes());
        v.push(CAPACITY as u8);
        v.extend_from_slice(&table_digest().to_le_bytes());
        for e in [self.probes.light, self.probes.glow, self.probes.grip] {
            v.extend_from_slice(&e.0);
        }
        v.extend_from_slice(&self.visual_seed.to_le_bytes());
        debug_assert_eq!(v.len(), STAMP_LEN);
        v
    }

    /// Inverse of [`Law::stamp`]. Refuses stamps of another version or another reaction function.
    pub fn from_stamp(bytes: &[u8]) -> Result<Law, LawError> {
        if bytes.len() != STAMP_LEN {
            return Err(LawError::Length(bytes.len()));
        }
        let version = u16::from_le_bytes([bytes[0], bytes[1]]);
        if version != 2 {
            return Err(LawError::Version(version));
        }
        let mut digest = [0u8; 8];
        digest.copy_from_slice(&bytes[3..11]);
        if bytes[2] as usize != CAPACITY || u64::from_le_bytes(digest) != table_digest() {
            return Err(LawError::Function);
        }
        let probe = |at: usize| {
            let mut c = [0u8; D];
            c.copy_from_slice(&bytes[at..at + D]);
            Element(c)
        };
        let mut seed = [0u8; 4];
        seed.copy_from_slice(&bytes[11 + 3 * D..]);
        Ok(Law {
            version,
            probes: Probes { light: probe(11), glow: probe(11 + D), grip: probe(11 + 2 * D) },
            visual_seed: u32::from_le_bytes(seed),
        })
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
