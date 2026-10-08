//! A block's material content: an unordered multiset of element occurrences. Storage is sorted, so
//! equal multisets have equal bytes; multiplicity is kept (`[x, x, y]` is not `[x, y]`).

use crate::element::{Element, D};

/// At most this many occurrences fit in one voxel (law constant).
pub const CAPACITY: usize = 32;

/// A configuration of elements: the canonical (sorted) multiset. Empty is the void (air).
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Configuration(Box<[Element]>);

/// Why a configuration could not be built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConfigError {
    /// More than [`CAPACITY`] occurrences.
    TooLarge(usize),
}

/// Why bytes did not decode into a configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// Empty input.
    Empty,
    /// The length byte exceeds [`CAPACITY`].
    TooLarge(usize),
    /// Fewer bytes than the length byte promises.
    Truncated,
    /// More bytes than the length byte promises.
    Trailing,
}

/// The canonical bytes of a configuration: `len` then `len × D` coordinates in sorted order. The intern
/// key, the save and the wire form.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Encoding(Box<[u8]>);

impl Encoding {
    /// The bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Configuration {
    /// The empty configuration: void / air.
    pub fn void() -> Self {
        Self(Box::new([]))
    }

    /// A configuration of exactly one occurrence.
    pub fn single(e: Element) -> Self {
        Self(Box::new([e]))
    }

    /// Build from occurrences in any order (storage order is not meaning).
    pub fn new(elements: impl Into<Box<[Element]>>) -> Result<Self, ConfigError> {
        let mut elements = elements.into();
        if elements.len() > CAPACITY {
            return Err(ConfigError::TooLarge(elements.len()));
        }
        elements.sort_unstable();
        Ok(Self(elements))
    }

    /// The occurrences, sorted.
    pub fn elements(&self) -> &[Element] {
        &self.0
    }

    /// Number of occurrences.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True for the void configuration (no occurrences).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Distinct elements with their multiplicities, in element order.
    pub fn counts(&self) -> impl Iterator<Item = (Element, usize)> + '_ {
        let mut i = 0;
        std::iter::from_fn(move || {
            let e = *self.0.get(i)?;
            let start = i;
            while i < self.0.len() && self.0[i] == e {
                i += 1;
            }
            Some((e, i - start))
        })
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

    /// Inverse of [`Configuration::encode`]. Occurrence order in the input does not matter (it is
    /// re-sorted); malformed input is rejected instead of guessed.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (&len, rest) = bytes.split_first().ok_or(DecodeError::Empty)?;
        let len = len as usize;
        if len > CAPACITY {
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
        elements.sort_unstable();
        Ok(Self(elements.into_boxed_slice()))
    }

    /// A 64-bit hash of the canonical bytes (FNV-1a): a stable seed for presentation and naming.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        eat(self.0.len() as u8);
        for e in self.0.iter() {
            for b in e.0 {
                eat(b);
            }
        }
        h
    }
}

#[cfg(test)]
mod pins {
    use super::*;

    /// Saves and the wire carry the encoding; presentation and naming seed from the digest.
    #[test]
    fn the_encoding_and_digest_are_pinned() {
        let c = Configuration::new(vec![Element::new([9, 200, 31, 4]), Element::new([1, 2, 3, 4]), Element::new([1, 2, 3, 4])])
            .unwrap();
        assert_eq!(c.encode().as_bytes(), [3, 1, 2, 3, 4, 1, 2, 3, 4, 9, 200, 31, 4], "the save and wire form");
        assert_eq!(Configuration::void().encode().as_bytes(), [0]);
        assert_eq!(Configuration::void().digest(), 0xaf63_bd4c_8601_b7df);
        assert_eq!(c.digest(), 0xf6c4_6a7e_784f_db28);
    }
}
