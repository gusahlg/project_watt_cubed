//! Canonical binary codec for the ident vocabulary.
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
//!
//! `Edit`/`Stamped`'s full domain-type encoding (this module's `edit`/`stamped`
//! methods) is not yet wired into either `save/format.rs` (its v5 edit record is
//! fixed-width, which its truncation-salvage arithmetic depends on; `Edit`'s `Data`
//! variant is variable-width) or `net/protocol.rs` (its `Edit` messages carry a
//! portable spec *string*, constructed in read-only `net/client.rs`/`server.rs`,
//! not a numeric `BlockState`) — a future scope for the follow-up integration.

use voxel_engine::DVec3;

use crate::block::BlockId;

use super::{
    BlockState, CellPos, DataKey, Edit, EditSeq, EditSource, FieldKind, PlayerId, Stamped,
    EDIT_TAG_CELL, EDIT_TAG_DATA, EDIT_TAG_FILL, EDIT_TAG_SPHERE,
};

/// A decode failure. Never a panic: every path through [`Reader`] returns this
/// instead of indexing past the buffer or trusting an attacker-controlled length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    Truncated,
    UnknownTag(u8),
    /// Rejected before the byte count is trusted enough to allocate for.
    DataTooLarge(u32),
}

/// Sanity cap on a single `Edit::Data` payload, so a corrupt length prefix can't
/// balloon memory before the bytes are even read.
pub const MAX_DATA_LEN: u32 = 1 << 20;

// EditSource wire tags. Local to the codec (not part of the frozen `EDIT_TAG_*`
// set, which is reserved for `Edit`'s own tagged union) — changing an existing
// one is forbidden by the same "extend, never mutate" rule.
const SOURCE_TAG_PLAYER: u8 = 0;
const SOURCE_TAG_SIM: u8 = 1;
const SOURCE_TAG_SYSTEM: u8 = 2;

const FIELD_TAG_THERMAL: u8 = 0;
const FIELD_TAG_ELECTRICAL: u8 = 1;

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
    pub fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
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

    pub fn cell_pos(&mut self, p: CellPos) {
        self.i64(p.x);
        self.i32(p.y);
        self.i64(p.z);
    }

    pub fn block_state(&mut self, b: BlockState) {
        self.u16(b.id.0);
        self.u16(b.state);
    }

    pub fn source(&mut self, s: &EditSource) {
        match s {
            EditSource::Player(PlayerId(id)) => {
                self.u8(SOURCE_TAG_PLAYER);
                self.u32(*id);
            }
            EditSource::Sim(field) => {
                self.u8(SOURCE_TAG_SIM);
                self.u8(match field {
                    FieldKind::Thermal => FIELD_TAG_THERMAL,
                    FieldKind::Electrical => FIELD_TAG_ELECTRICAL,
                });
            }
            EditSource::System => self.u8(SOURCE_TAG_SYSTEM),
        }
    }

    pub fn edit(&mut self, e: &Edit) {
        match e {
            Edit::Cell { pos, block } => {
                self.u8(EDIT_TAG_CELL);
                self.cell_pos(*pos);
                self.block_state(*block);
            }
            Edit::Fill { min, max, block } => {
                self.u8(EDIT_TAG_FILL);
                self.cell_pos(*min);
                self.cell_pos(*max);
                self.block_state(*block);
            }
            Edit::Sphere { center, radius_cells, block } => {
                self.u8(EDIT_TAG_SPHERE);
                self.cell_pos(*center);
                self.u32(*radius_cells);
                self.block_state(*block);
            }
            Edit::Data { pos, key, value } => {
                self.u8(EDIT_TAG_DATA);
                self.cell_pos(*pos);
                self.u32(key.0);
                self.u32(value.len() as u32);
                self.raw(value);
            }
        }
    }

    pub fn stamped(&mut self, s: &Stamped) {
        self.u64(s.seq.0);
        self.source(&s.source);
        self.edit(&s.edit);
    }
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

    pub fn pos(&self) -> usize {
        self.pos
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
    pub fn u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn i32(&mut self) -> Result<i32, CodecError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn i64(&mut self) -> Result<i64, CodecError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn f64(&mut self) -> Result<f64, CodecError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Strict UTF8: a `u16`-length-prefixed string that fails on invalid bytes
    /// (save's convention — corrupt text is a corrupt file, not a lossy repair).
    pub fn str16(&mut self) -> Result<String, CodecError> {
        let len = self.u16()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| CodecError::Truncated)
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
        Ok(Pose { pos: self.vec3()?, yaw: self.f32()?, pitch: self.f32()? })
    }

    pub fn cell_pos(&mut self) -> Result<CellPos, CodecError> {
        Ok(CellPos { x: self.i64()?, y: self.i32()?, z: self.i64()? })
    }

    pub fn block_state(&mut self) -> Result<BlockState, CodecError> {
        Ok(BlockState { id: BlockId(self.u16()?), state: self.u16()? })
    }

    pub fn source(&mut self) -> Result<EditSource, CodecError> {
        match self.u8()? {
            SOURCE_TAG_PLAYER => Ok(EditSource::Player(PlayerId(self.u32()?))),
            SOURCE_TAG_SIM => Ok(EditSource::Sim(match self.u8()? {
                FIELD_TAG_THERMAL => FieldKind::Thermal,
                FIELD_TAG_ELECTRICAL => FieldKind::Electrical,
                other => return Err(CodecError::UnknownTag(other)),
            })),
            SOURCE_TAG_SYSTEM => Ok(EditSource::System),
            other => Err(CodecError::UnknownTag(other)),
        }
    }

    pub fn edit(&mut self) -> Result<Edit, CodecError> {
        match self.u8()? {
            EDIT_TAG_CELL => Ok(Edit::Cell { pos: self.cell_pos()?, block: self.block_state()? }),
            EDIT_TAG_FILL => {
                let min = self.cell_pos()?;
                let max = self.cell_pos()?;
                Ok(Edit::Fill { min, max, block: self.block_state()? })
            }
            EDIT_TAG_SPHERE => {
                let center = self.cell_pos()?;
                let radius_cells = self.u32()?;
                Ok(Edit::Sphere { center, radius_cells, block: self.block_state()? })
            }
            EDIT_TAG_DATA => {
                let pos = self.cell_pos()?;
                let key = DataKey(self.u32()?);
                let len = self.u32()?;
                if len > MAX_DATA_LEN {
                    return Err(CodecError::DataTooLarge(len));
                }
                let value = self.take(len as usize)?.to_vec().into_boxed_slice();
                Ok(Edit::Data { pos, key, value })
            }
            other => Err(CodecError::UnknownTag(other)),
        }
    }

    pub fn stamped(&mut self) -> Result<Stamped, CodecError> {
        let seq = EditSeq(self.u64()?);
        let source = self.source()?;
        let edit = self.edit()?;
        Ok(Stamped { seq, source, edit })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockId;

    fn block(id: u16, state: u16) -> BlockState {
        BlockState { id: BlockId(id), state }
    }

    fn cell(x: i64, y: i32, z: i64) -> CellPos {
        CellPos { x, y, z }
    }

    #[test]
    fn cell_pos_round_trips_extreme_coords() {
        for p in [
            cell(0, 0, 0),
            cell(i64::MIN, i32::MIN, i64::MAX),
            cell(i64::MAX, i32::MAX, i64::MIN),
            cell(-1_000_000_000, -2048, 1_000_000_000),
        ] {
            let mut w = Writer::new();
            w.cell_pos(p);
            let bytes = w.into_inner();
            let mut r = Reader::new(&bytes);
            assert_eq!(r.cell_pos().unwrap(), p);
            assert!(r.finished());
        }
    }

    #[test]
    fn block_state_round_trips() {
        for b in [block(0, 0), block(u16::MAX, u16::MAX), block(1, 0), block(0, 1)] {
            let mut w = Writer::new();
            w.block_state(b);
            let bytes = w.into_inner();
            let mut r = Reader::new(&bytes);
            assert_eq!(r.block_state().unwrap(), b);
        }
    }

    #[test]
    fn pose_round_trips_at_world_border() {
        let poses = [
            Pose { pos: DVec3::new(0.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0 },
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

    fn edit_corpus() -> Vec<Edit> {
        vec![
            Edit::Cell { pos: cell(1, 60, -1), block: block(3, 0) },
            Edit::Cell { pos: cell(i64::MIN, i32::MIN, i64::MAX), block: block(0, 0) },
            Edit::Fill { min: cell(0, 0, 0), max: cell(15, 15, 15), block: block(7, 2) },
            Edit::Sphere { center: cell(-5, 10, 5), radius_cells: 8, block: block(9, 0) },
            Edit::Data { pos: cell(2, 1, 2), key: DataKey(5), value: Box::new([]) },
            Edit::Data { pos: cell(2, 1, 2), key: DataKey(u32::MAX), value: Box::new([1, 2, 3, 4, 5]) },
        ]
    }

    #[test]
    fn edit_round_trips_every_variant() {
        for e in edit_corpus() {
            let mut w = Writer::new();
            w.edit(&e);
            let bytes = w.into_inner();
            let mut r = Reader::new(&bytes);
            assert_eq!(r.edit().unwrap(), e);
            assert!(r.finished(), "decode left trailing bytes for {e:?}");
        }
    }

    #[test]
    fn stamped_round_trips_every_source() {
        let sources = [
            EditSource::Player(PlayerId(0)),
            EditSource::Player(PlayerId(u32::MAX)),
            EditSource::Sim(FieldKind::Thermal),
            EditSource::Sim(FieldKind::Electrical),
            EditSource::System,
        ];
        for (i, source) in sources.into_iter().enumerate() {
            let s = Stamped { seq: EditSeq(i as u64), source, edit: edit_corpus()[i % 6].clone() };
            let mut w = Writer::new();
            w.stamped(&s);
            let bytes = w.into_inner();
            let mut r = Reader::new(&bytes);
            assert_eq!(r.stamped().unwrap(), s);
        }
    }

    #[test]
    fn unknown_and_reserved_edit_tags_are_rejected_cleanly() {
        for tag in [0x00u8, 0x05, 0x0f, 0x10, 0x7f, 0xff] {
            let bytes = [tag];
            let mut r = Reader::new(&bytes);
            assert_eq!(r.edit(), Err(CodecError::UnknownTag(tag)));
        }
    }

    #[test]
    fn oversize_data_payload_is_rejected_before_reading_bytes() {
        let mut w = Writer::new();
        w.u8(EDIT_TAG_DATA);
        w.cell_pos(cell(0, 0, 0));
        w.u32(0); // key
        w.u32(MAX_DATA_LEN + 1); // length prefix, but no actual payload follows
        let bytes = w.into_inner();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.edit(), Err(CodecError::DataTooLarge(MAX_DATA_LEN + 1)));
    }

    #[test]
    fn truncated_input_errors_instead_of_panicking() {
        assert_eq!(Reader::new(&[]).u8(), Err(CodecError::Truncated));
        assert_eq!(Reader::new(&[1, 2, 3]).u64(), Err(CodecError::Truncated));
        assert_eq!(Reader::new(&[EDIT_TAG_CELL]).edit(), Err(CodecError::Truncated));
    }

    #[test]
    fn trailing_bytes_are_visible_via_finished() {
        let mut w = Writer::new();
        w.edit(&Edit::Cell { pos: cell(0, 0, 0), block: block(1, 0) });
        let mut bytes = w.into_inner();
        bytes.push(0xa5);
        let mut r = Reader::new(&bytes);
        r.edit().unwrap();
        assert!(!r.finished(), "reader should report the stray trailing byte");
    }
}
