//! A block's material content: an ordered list of elements with multiplicity. The representation keeps
//! every bit of information (order and repeats) because the model has not yet proved either irrelevant.

use crate::element::{Element, D};

/// Maximum number of elements one configuration may hold.
pub const CONFIG_MAX: usize = 16;

/// A configuration of elements. `[A, B] != [B, A]` and `[A, A] != [A]` by design (see the spec §3.2).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Configuration(Box<[Element]>);

/// Why a configuration could not be built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConfigError {
    /// More than [`CONFIG_MAX`] elements.
    TooLarge(usize),
}

/// Why bytes did not decode into a configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// Empty input.
    Empty,
    /// The length byte exceeds [`CONFIG_MAX`].
    TooLarge(usize),
    /// Fewer bytes than the length byte promises.
    Truncated,
    /// More bytes than the length byte promises.
    Trailing,
}

/// The canonical bytes of a configuration: `len` then `len × D` coordinates. The intern key, the save
/// and wire form. Equal configurations have equal encodings and vice versa.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Encoding(Box<[u8]>);

impl Encoding {
    /// The bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Configuration {
    /// The empty configuration: void / air. Observes as passable and transparent.
    pub fn void() -> Self {
        Self(Box::new([]))
    }

    /// A configuration of exactly one element.
    pub fn single(e: Element) -> Self {
        Self(Box::new([e]))
    }

    /// Build from elements, in the given order, keeping repeats.
    pub fn new(elements: impl Into<Box<[Element]>>) -> Result<Self, ConfigError> {
        let elements = elements.into();
        if elements.len() > CONFIG_MAX {
            return Err(ConfigError::TooLarge(elements.len()));
        }
        Ok(Self(elements))
    }

    /// The elements, in order.
    pub fn elements(&self) -> &[Element] {
        &self.0
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True for the void configuration.
    pub fn is_void(&self) -> bool {
        self.0.is_empty()
    }

    /// Canonical bytes.
    pub fn encode(&self) -> Encoding {
        let mut v = Vec::with_capacity(1 + self.0.len() * D);
        v.push(self.0.len() as u8);
        for e in self.0.iter() {
            v.extend_from_slice(&e.0);
        }
        Encoding(v.into_boxed_slice())
    }

    /// Inverse of [`Configuration::encode`]; rejects malformed input instead of guessing.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (&len, rest) = bytes.split_first().ok_or(DecodeError::Empty)?;
        let len = len as usize;
        if len > CONFIG_MAX {
            return Err(DecodeError::TooLarge(len));
        }
        if rest.len() < len * D {
            return Err(DecodeError::Truncated);
        }
        if rest.len() > len * D {
            return Err(DecodeError::Trailing);
        }
        let mut elements = Vec::with_capacity(len);
        for chunk in rest.chunks_exact(D) {
            let mut c = [0u8; D];
            c.copy_from_slice(chunk);
            elements.push(Element(c));
        }
        Ok(Self(elements.into_boxed_slice()))
    }

    /// Mean coordinate in 1/256 units (`None` for the void). Smooth in every element.
    pub fn mean_q8(&self) -> Option<[u32; D]> {
        if self.0.is_empty() {
            return None;
        }
        let n = self.0.len() as u32;
        let mut sum = [0u32; D];
        for e in self.0.iter() {
            for i in 0..D {
                sum[i] += e.0[i] as u32;
            }
        }
        Some(sum.map(|s| s * 256 / n))
    }

    /// Mean L1 distance of the elements to their mean, in 1/256 lattice units (0 for ≤ 1 element).
    pub fn spread_q8(&self) -> u32 {
        let Some(mean) = self.mean_q8() else { return 0 };
        if self.0.len() < 2 {
            return 0;
        }
        let mut acc = 0u32;
        for e in self.0.iter() {
            for i in 0..D {
                acc += (e.0[i] as i32 * 256 - mean[i] as i32).unsigned_abs();
            }
        }
        acc / self.0.len() as u32
    }
}
