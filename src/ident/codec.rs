//! Canonical binary codec for shared primitive values.
//!
//! `save/format.rs` and `net/protocol.rs` frame this codec instead of hand-rolling
//! their own bit-twiddling for the same primitives (integers, length-prefixed
//! strings, position vectors) — real duplication collapses into one implementation
//! here, with each caller keeping only what's genuinely framing-specific (save's
//! fixed-width header/salvage layout; net's big-endian frame-length envelope).
//!
//! CANONICAL ENDIANNESS: little (matches save, the larger surface). Decoding never
//! panics and never yields a partial value: every getter returns a typed
//! [`CodecError`] on truncated or malformed input (parse-don't-validate).

use voxel_engine::DVec3;

/// A decode failure. Never a panic: every path through [`Reader`] returns this
/// instead of indexing past the buffer or trusting an attacker-controlled length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    Truncated,
}

/// The wire-shared subset of player state: position + orientation. Save's
/// `flying`/`noclip` and net's `Stance` are framing-specific (persistence mode
/// flags vs. a transient presence signal) and stay with their own callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct Writer(Vec<u8>);

impl Writer {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }

    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    /// Raw bytes, no length prefix — caller manages framing.
    pub fn raw(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }

    /// Clamped to `u16::MAX` bytes so the length prefix always stays honest.
    pub fn str16(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let len = bytes.len().min(u16::MAX as usize);
        self.u16(len as u16);
        self.raw(&bytes[..len]);
    }

    /// A position: 3x f64 — bit-exact at any in-world distance from the origin.
    pub fn vec3(&mut self, v: DVec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }

    pub fn pose(&mut self, p: Pose) {
        self.vec3(p.pos);
        self.f32(p.yaw);
        self.f32(p.pitch);
    }
}

macro_rules! le_scalars {
    ($($name:ident : $ty:ty),+ $(,)?) => {
        impl Writer {
            $(
                pub fn $name(&mut self, v: $ty) {
                    self.0.extend_from_slice(&v.to_le_bytes());
                }
            )+
        }
        impl Reader<'_> {
            $(
                pub fn $name(&mut self) -> Result<$ty, CodecError> {
                    Ok(<$ty>::from_le_bytes(
                        self.take(core::mem::size_of::<$ty>())?.try_into().unwrap(),
                    ))
                }
            )+
        }
    };
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounds-checked little-endian cursor shared by every framing. Every getter
/// returns [`CodecError`] rather than panicking on a short or hostile buffer.
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Start the cursor partway into `bytes` — for callers (e.g. save's header)
    /// that already know a fixed prefix to skip.
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    pub fn finished(&self) -> bool {
        self.pos == self.bytes.len()
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        if self.remaining() < n {
            return Err(CodecError::Truncated);
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    /// Lossy UTF8: a `u16`-length-prefixed string that never fails on invalid
    /// bytes (net's convention — a garbled string shouldn't fail an otherwise
    /// valid decode).
    pub fn str16_lossy(&mut self) -> Result<String, CodecError> {
        let len = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    pub fn vec3(&mut self) -> Result<DVec3, CodecError> {
        Ok(DVec3::new(self.f64()?, self.f64()?, self.f64()?))
    }

    pub fn pose(&mut self) -> Result<Pose, CodecError> {
        Ok(Pose {
            pos: self.vec3()?,
            yaw: self.f32()?,
            pitch: self.f32()?,
        })
    }
}

le_scalars!(u16: u16, u32: u32, u64: u64, i32: i32, i64: i64, f32: f32, f64: f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_round_trips_at_world_border() {
        let poses = [
            Pose {
                pos: DVec3::new(0.0, 0.0, 0.0),
                yaw: 0.0,
                pitch: 0.0,
            },
            Pose {
                pos: DVec3::new(1.0e9 + 0.123456789, -3_000.25, -(1.0e9 - 0.75)),
                yaw: 1.25,
                pitch: -0.5,
            },
        ];
        for p in poses {
            let mut w = Writer::new();
            w.pose(p);
            let bytes = w.into_inner();
            let mut r = Reader::new(&bytes);
            let got = r.pose().unwrap();
            assert_eq!(got.pos.x.to_bits(), p.pos.x.to_bits());
            assert_eq!(got.pos.y.to_bits(), p.pos.y.to_bits());
            assert_eq!(got.pos.z.to_bits(), p.pos.z.to_bits());
            assert_eq!(got.yaw, p.yaw);
            assert_eq!(got.pitch, p.pitch);
        }
    }

    #[test]
    fn truncated_input_errors_instead_of_panicking() {
        assert_eq!(Reader::new(&[]).u8(), Err(CodecError::Truncated));
        assert_eq!(Reader::new(&[1, 2, 3]).u64(), Err(CodecError::Truncated));
    }

    #[test]
    fn trailing_bytes_are_visible_via_finished() {
        let mut w = Writer::new();
        w.u8(7);
        let mut bytes = w.into_inner();
        bytes.push(0xa5);
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u8(), Ok(7));
        assert!(
            !r.finished(),
            "reader should report the stray trailing byte"
        );
    }
}
