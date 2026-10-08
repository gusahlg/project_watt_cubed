//! FNV-1a, the one implementation behind every hash that is persisted or sent: the law stamp's table
//! digest and fingerprint, the join id, the palette's fingerprints and the planet-map cache key.
//! Changing it changes those bytes.

macro_rules! fnv {
    ($name:ident, $word:ty, $offset:expr, $prime:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name($word);

        impl $name {
            /// The hash of nothing (the offset basis).
            pub const fn new() -> Self {
                Self($offset)
            }

            /// Feed `bytes` in order.
            #[inline]
            pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
                for &b in bytes {
                    self.0 = (self.0 ^ b as $word).wrapping_mul($prime);
                }
                self
            }

            /// The hash of everything fed so far.
            pub const fn finish(&self) -> $word {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

fnv!(Fnv64, u64, 0xcbf2_9ce4_8422_2325, 0x0000_0100_0000_01b3, "FNV-1a, 64-bit.");
fnv!(Fnv32, u32, 0x811c_9dc5, 0x0100_0193, "FNV-1a, 32-bit.");

#[cfg(test)]
mod tests {
    use super::*;

    /// The published FNV-1a test vectors.
    #[test]
    fn known_answers() {
        assert_eq!(Fnv64::new().finish(), 0xcbf2_9ce4_8422_2325);
        assert_eq!(Fnv64::new().bytes(b"a").finish(), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(Fnv64::new().bytes(b"foobar").finish(), 0x8594_4171_f739_67e8);
        assert_eq!(Fnv64::new().bytes(b"foo").bytes(b"bar").finish(), 0x8594_4171_f739_67e8);
        assert_eq!(Fnv32::default().finish(), 0x811c_9dc5);
        assert_eq!(Fnv32::new().bytes(b"a").finish(), 0xe40c_292c);
        assert_eq!(Fnv32::new().bytes(b"foobar").finish(), 0xbf9c_f968);
    }
}
