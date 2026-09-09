//! Pure save codec: `SaveDoc` ↔ bytes. No game types — the bridge converts
//! `World`/`Player`/`Mods` to and from `SaveDoc`, so this whole module is
//! testable without constructing a world.
//!
//! Layout, version 7 (all integers little-endian):
//!
//! ```text
//! header (fixed 126 bytes, peekable without the body):
//!   magic        b"WATT"                                       4
//!   version      u16 = 7                                       2
//!   name         u8 len + 64-byte field (utf8, zero padded)   65
//!   seed         i64                                           8
//!   created      u64 unix secs                                 8
//!   last_played  u64 unix secs                                 8
//!   playtime     u64 secs                                      8
//!   edit_count   u32                                           4
//!   worldgen     u16 (v5+; a v4 header ends here, worldgen 1)   2
//!   kind         u8  (v6+; 0 = classic, 1 = diffusion)         1
//!   tile         u32 (v6+; diffusion knobs, ignored classic)   4
//!   stride       u32                                           4
//!   phases       u32                                           4
//!   relief       f32                                           4
//! player         pos f64 x3, yaw f32, pitch f32,
//!                flags u8 (bit 0 = fly, bit 1 = noclip)       33
//!                stash (v7+): u16 len + utf8 "Name=count,..."  variable
//! spec table     u16 count, then per spec: u16 len + utf8
//! edits          edit_count records of i32 x, i32 y, i32 z, u16 spec index
//! mods           u8 count, then per mod: u8 name-len + utf8,
//!                                        u32 state-len + utf8
//! ```
//!
//! Version 4 files (no worldgen stamp) still decode — the field defaults to
//! worldgen 1 and the loader WARNS rather than rejects: the seed regenerates
//! terrain fine, but its materials may have moved under the edits.
//!
//! Version 5 files (no kind/knobs) still decode — kind defaults to classic
//! and the diffusion knobs to their shipped defaults. A v5 diffusion world
//! cannot exist: InfiniteDiffusion landed with v6.
//!
//! Version 6 files (no player stash) still decode — `PlayerState::stash` is
//! `None`. The bridge then migrates an old Inventory mod-state line into the
//! core stash.
//!
//! The edit count lives only in the header (no body prefix), and each record is
//! a fixed 14 bytes, so a truncated file still yields its longest valid prefix
//! of edits: decoding degrades to [`Decoded::Salvaged`] instead of failing.

use crate::ident::codec;

use super::slot::{SaveError, SaveMeta};

pub const MAGIC: &[u8; 4] = b"WATT";
pub const VERSION: u16 = 7;

const NAME_FIELD: usize = 64;
/// Offset of the name length byte, for in-place renames via [`set_name`].
const NAME_OFF: usize = 6;
/// The version-4 header, which the v5 header extends by the worldgen stamp.
const HEADER_LEN_V4: usize = NAME_OFF + 1 + NAME_FIELD + 8 + 8 + 8 + 8 + 4;
/// The version-5 header, which the v6 header extends by kind + diffusion knobs.
const HEADER_LEN_V5: usize = HEADER_LEN_V4 + 2;
/// kind u8 + tile/stride/phases u32 + relief f32.
const WORLDGEN_STAMP_LEN: usize = 1 + 4 + 4 + 4 + 4;
pub const HEADER_LEN: usize = HEADER_LEN_V5 + WORLDGEN_STAMP_LEN;

/// Header length for a supported on-disk version, or `BadVersion`.
fn header_len(version: u16) -> Result<usize, SaveError> {
    match version {
        4 => Ok(HEADER_LEN_V4),
        5 => Ok(HEADER_LEN_V5),
        6 | 7 => Ok(HEADER_LEN),
        v => Err(SaveError::BadVersion(v)),
    }
}

/// Pose (3×f64 + 2×f32) plus the flying/noclip flags byte.
const PLAYER_POSE_LEN: usize = 32 + 1;

const EDIT_BYTES: usize = 14;

/// Sanity caps while reading, so a corrupt length prefix can't balloon memory.
/// Specs track the block palette cap — a long-played world can legitimately
/// reference one spec per registered block type.
const MAX_SPECS: usize = 16_384;
const MAX_EDITS: u32 = 50_000_000;
const MAX_MOD_STATE: u32 = 16 * 1024 * 1024;

/// Invariants enforced at encode/decode boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveDoc {
    pub meta: SaveMeta,
    /// Which worldgen semantics generated this world's terrain — see
    /// `placement::WORLDGEN_VERSION`. A mismatch on load WARNS (the seed still
    /// regenerates, but materials under old edits may have moved).
    pub worldgen_version: u16,
    /// Generator kind and diffusion knobs captured from the live world.
    /// `kind` is 0 = classic, 1 = diffusion (unknown values load as classic).
    pub worldgen: WorldgenStamp,
    pub player: PlayerState,
    /// Deduplicated block-spec table; edits reference it by index.
    pub specs: Vec<String>,
    pub edits: Vec<Edit>,
    /// Per-mod `(name, state)` in each mod's own string format.
    pub mods: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlayerState {
    /// f64 so far-out positions (the game plays to ±1e9 blocks) restore bit-exactly.
    pub pos: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub flying: bool,
    pub noclip: bool,
    /// Held elements as `(name, count)` in first-seen order.
    /// `None` means the field was absent (pre-v7); the bridge then migrates
    /// from the Inventory mod-state line.
    pub stash: Option<Vec<(String, u32)>>,
}

/// On-disk worldgen identity. Kept as raw integers so this codec stays free
/// of game types; the bridge maps to `WorldgenKind` / `DiffusionCfg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldgenStamp {
    pub kind: u8,
    pub tile: u32,
    pub stride: u32,
    pub phases: u32,
    pub relief: f32,
}

impl Default for WorldgenStamp {
    fn default() -> Self {
        Self {
            kind: 0,
            tile: 32,
            stride: 16,
            phases: 2,
            relief: 1.0,
        }
    }
}

fn encode_stash_payload(items: &[(String, u32)]) -> String {
    let mut s = String::new();
    for (i, (name, count)) in items.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(name);
        s.push('=');
        s.push_str(&count.to_string());
    }
    s
}

fn parse_stash_payload(s: &str) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    for entry in s.split(',').filter(|e| !e.is_empty()) {
        let Some((name, count)) = entry.split_once('=') else {
            continue;
        };
        let Ok(count) = count.parse::<u32>() else {
            continue;
        };
        if name.is_empty() || count == 0 {
            continue;
        }
        out.push((name.to_string(), count));
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edit {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub spec: u16,
}

/// A decode can't silently lie: a partial read comes back as `Salvaged`, which
/// the caller has to match on.
#[derive(Debug)]
pub enum Decoded {
    Intact(SaveDoc),
    /// File was truncated; `doc` holds the longest valid prefix.
    Salvaged { doc: SaveDoc, recovered: u32, expected: u32 },
}

/// Clamp to UTF-8 char boundary to avoid splitting multibyte sequences.
fn clamp_name(name: &str) -> &str {
    if name.len() <= NAME_FIELD {
        return name;
    }
    let mut end = NAME_FIELD;
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

pub fn encode(doc: &SaveDoc) -> Result<Vec<u8>, SaveError> {
    let edit_count = u32::try_from(doc.edits.len())
        .ok()
        .filter(|&n| n <= MAX_EDITS)
        .ok_or(SaveError::Corrupt("too many edits to save"))?;
    if doc.specs.len() > MAX_SPECS {
        return Err(SaveError::Corrupt("too many distinct block specs to save"));
    }

    let mut out = Vec::with_capacity(HEADER_LEN + 33 + doc.edits.len() * EDIT_BYTES + 256);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let name = clamp_name(&doc.meta.name);
    out.push(name.len() as u8);
    let mut field = [0u8; NAME_FIELD];
    field[..name.len()].copy_from_slice(name.as_bytes());
    out.extend_from_slice(&field);
    out.extend_from_slice(&doc.meta.seed.to_le_bytes());
    out.extend_from_slice(&doc.meta.created.to_le_bytes());
    out.extend_from_slice(&doc.meta.last_played.to_le_bytes());
    out.extend_from_slice(&doc.meta.playtime_secs.to_le_bytes());
    out.extend_from_slice(&edit_count.to_le_bytes());
    out.extend_from_slice(&doc.worldgen_version.to_le_bytes());
    out.push(doc.worldgen.kind);
    out.extend_from_slice(&doc.worldgen.tile.to_le_bytes());
    out.extend_from_slice(&doc.worldgen.stride.to_le_bytes());
    out.extend_from_slice(&doc.worldgen.phases.to_le_bytes());
    out.extend_from_slice(&doc.worldgen.relief.to_le_bytes());
    debug_assert_eq!(out.len(), HEADER_LEN);

    let mut pw = codec::Writer::new();
    pw.pose(codec::Pose {
        pos: voxel_engine::DVec3::new(doc.player.pos[0], doc.player.pos[1], doc.player.pos[2]),
        yaw: doc.player.yaw,
        pitch: doc.player.pitch,
    });
    out.extend_from_slice(&pw.into_inner());
    out.push(doc.player.flying as u8 | (doc.player.noclip as u8) << 1);
    debug_assert_eq!(out.len(), HEADER_LEN + PLAYER_POSE_LEN);
    let stash = encode_stash_payload(doc.player.stash.as_deref().unwrap_or(&[]));
    if u16::try_from(stash.len()).is_err() {
        return Err(SaveError::Corrupt("player stash too long to save"));
    }
    let mut sw = codec::Writer::new();
    sw.str16(&stash);
    out.extend_from_slice(&sw.into_inner());

    out.extend_from_slice(&(doc.specs.len() as u16).to_le_bytes());
    for spec in &doc.specs {
        if u16::try_from(spec.len()).is_err() {
            return Err(SaveError::Corrupt("block spec too long to save"));
        }
        let mut sw = codec::Writer::new();
        sw.str16(spec);
        out.extend_from_slice(&sw.into_inner());
    }

    for edit in &doc.edits {
        if usize::from(edit.spec) >= doc.specs.len() {
            return Err(SaveError::Corrupt("edit references a spec outside the table"));
        }
        out.extend_from_slice(&edit.x.to_le_bytes());
        out.extend_from_slice(&edit.y.to_le_bytes());
        out.extend_from_slice(&edit.z.to_le_bytes());
        out.extend_from_slice(&edit.spec.to_le_bytes());
    }

    let count = u8::try_from(doc.mods.len())
        .map_err(|_| SaveError::Corrupt("too many mod states to save"))?;
    out.push(count);
    for (name, data) in &doc.mods {
        let name_len =
            u8::try_from(name.len()).map_err(|_| SaveError::Corrupt("mod name too long to save"))?;
        out.push(name_len);
        out.extend_from_slice(name.as_bytes());
        let data_len = u32::try_from(data.len())
            .ok()
            .filter(|&n| n <= MAX_MOD_STATE)
            .ok_or(SaveError::Corrupt("mod state too long to save"))?;
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(data.as_bytes());
    }

    Ok(out)
}

/// Read the metadata alone. Needs only the first [`HEADER_LEN`] bytes, so the
/// slot list never touches file bodies.
pub fn peek_meta(bytes: &[u8]) -> Result<SaveMeta, SaveError> {
    if bytes.len() < NAME_OFF {
        return Err(SaveError::Corrupt("save file is shorter than its header"));
    }
    if &bytes[..4] != MAGIC {
        return Err(SaveError::Corrupt("not a watt-cubed save (bad magic)"));
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    if bytes.len() < header_len(version)? {
        return Err(SaveError::Corrupt("save file is shorter than its header"));
    }
    let name_len = bytes[NAME_OFF] as usize;
    if name_len > NAME_FIELD {
        return Err(SaveError::Corrupt("header name length out of range"));
    }
    let name = std::str::from_utf8(&bytes[NAME_OFF + 1..NAME_OFF + 1 + name_len])
        .map_err(|_| SaveError::Corrupt("invalid UTF-8 in header name"))?
        .to_string();
    let word = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
    let base = NAME_OFF + 1 + NAME_FIELD;
    Ok(SaveMeta {
        name,
        seed: word(base) as i64,
        created: word(base + 8),
        last_played: word(base + 16),
        playtime_secs: word(base + 24),
        edit_count: u32::from_le_bytes(bytes[base + 32..base + 36].try_into().unwrap()),
    })
}

/// Rewrite the display name in an already-encoded save, in place. The fixed
/// header layout makes this a pure byte patch — used by duplicate/rename so
/// they never need to decode a body.
pub fn set_name(bytes: &mut [u8], name: &str) -> Result<(), SaveError> {
    peek_meta(&bytes[..bytes.len().min(HEADER_LEN)])?;
    let name = clamp_name(name);
    bytes[NAME_OFF] = name.len() as u8;
    let field = &mut bytes[NAME_OFF + 1..NAME_OFF + 1 + NAME_FIELD];
    field.fill(0);
    field[..name.len()].copy_from_slice(name.as_bytes());
    Ok(())
}

/// A truncated read is the one error this format ever reports for it — every
/// [`codec::CodecError`] a save can hit collapses to that (a decode never
/// sees `UnknownTag`/`DataTooLarge`: this file's records are all fixed-width).
fn truncated<T>(_: codec::CodecError) -> Result<T, SaveError> {
    Err(SaveError::Corrupt("save file is truncated"))
}

/// Thin framing over the shared little-endian cursor ([`codec::Reader`]):
/// same bounds-checked primitives as `net/protocol.rs`, plus save's own
/// strict-UTF-8 string convention (corrupt text is a corrupt file, not a
/// lossy repair).
struct Reader<'a>(codec::Reader<'a>);

impl<'a> Reader<'a> {
    fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self(codec::Reader::with_pos(bytes, pos))
    }

    fn remaining(&self) -> usize {
        self.0.remaining()
    }

    fn u8(&mut self) -> Result<u8, SaveError> {
        self.0.u8().or_else(truncated)
    }
    fn u16(&mut self) -> Result<u16, SaveError> {
        self.0.u16().or_else(truncated)
    }
    fn u32(&mut self) -> Result<u32, SaveError> {
        self.0.u32().or_else(truncated)
    }
    fn i32(&mut self) -> Result<i32, SaveError> {
        self.0.i32().or_else(truncated)
    }
    fn pose(&mut self) -> Result<codec::Pose, SaveError> {
        self.0.pose().or_else(truncated)
    }
    fn string(&mut self, len: usize) -> Result<String, SaveError> {
        String::from_utf8(self.0.take(len).or_else(truncated)?.to_vec())
            .map_err(|_| SaveError::Corrupt("invalid UTF-8 in save file"))
    }
}

pub fn decode(bytes: &[u8]) -> Result<Decoded, SaveError> {
    let meta = peek_meta(bytes)?;
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    // v4 predates the worldgen stamp: those worlds came from the legacy picker.
    let worldgen_version = if version >= 5 {
        u16::from_le_bytes(bytes[HEADER_LEN_V5 - 2..HEADER_LEN_V5].try_into().unwrap())
    } else {
        1
    };
    // v5 predates kind + diffusion knobs: those worlds are classic.
    let worldgen = if version >= 6 {
        let off = HEADER_LEN_V5;
        WorldgenStamp {
            kind: bytes[off],
            tile: u32::from_le_bytes(bytes[off + 1..off + 5].try_into().unwrap()),
            stride: u32::from_le_bytes(bytes[off + 5..off + 9].try_into().unwrap()),
            phases: u32::from_le_bytes(bytes[off + 9..off + 13].try_into().unwrap()),
            relief: f32::from_le_bytes(bytes[off + 13..off + 17].try_into().unwrap()),
        }
    } else {
        WorldgenStamp::default()
    };
    let mut r = Reader::with_pos(bytes, header_len(version)?);

    // Header through spec table must be intact — there's no way to regenerate
    // a partial spec table, and everything after depends on it.
    let pose = r.pose()?;
    let player = PlayerState {
        pos: [pose.pos.x, pose.pos.y, pose.pos.z],
        yaw: pose.yaw,
        pitch: pose.pitch,
        flying: false,
        noclip: false,
        stash: None,
    };
    let flags = r.u8()?;
    let stash = if version >= 7 {
        let len = r.u16()? as usize;
        Some(parse_stash_payload(&r.string(len)?))
    } else {
        None
    };
    let player = PlayerState {
        flying: flags & 1 != 0,
        noclip: flags & 2 != 0,
        stash,
        ..player
    };
    // Raw float bit patterns are not all valid game states: NaN/Infinity would
    // poison camera/physics on load, and a position outside the border breaks
    // the clamp every continuous writer maintains. Reject rather than repair —
    // the slot store then falls back to the intact backup file.
    if player.pos.iter().any(|v| !v.is_finite() || v.abs() > crate::math::WORLD_BORDER)
        || !player.yaw.is_finite()
        || !player.pitch.is_finite()
    {
        return Err(SaveError::Corrupt("player state is non-finite or out of world"));
    }

    let spec_count = r.u16()? as usize;
    if spec_count > MAX_SPECS {
        return Err(SaveError::Corrupt("spec table too large"));
    }
    let mut specs = Vec::with_capacity(spec_count);
    for _ in 0..spec_count {
        let len = r.u16()? as usize;
        specs.push(r.string(len)?);
    }

    let expected = meta.edit_count;
    if expected > MAX_EDITS {
        return Err(SaveError::Corrupt("edit count too large"));
    }
    // Fixed-width records: a truncated tail is still a valid prefix. A spec
    // index outside the table also truncates — past that point the bytes
    // aren't trustworthy.
    let avail = (r.remaining() / EDIT_BYTES) as u32;
    let n = expected.min(avail);
    let mut clean = n == expected;
    let mut edits = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let (x, y, z) = (r.i32()?, r.i32()?, r.i32()?);
        let spec = r.u16()?;
        if usize::from(spec) >= specs.len() {
            clean = false;
            break;
        }
        edits.push(Edit { x, y, z, spec });
    }
    let recovered = edits.len() as u32;

    // Mod state is best-effort once edits are intact: keep whole entries
    // until one fails to parse.
    let mut mods = Vec::new();
    if clean {
        clean = (|| -> Result<(), SaveError> {
            let count = r.u8()?;
            for _ in 0..count {
                let name_len = r.u8()? as usize;
                let name = r.string(name_len)?;
                let data_len = r.u32()?;
                if data_len > MAX_MOD_STATE {
                    return Err(SaveError::Corrupt("mod state too large"));
                }
                let data = r.string(data_len as usize)?;
                mods.push((name, data));
            }
            Ok(())
        })()
        .is_ok();
    }

    let doc = SaveDoc { meta, worldgen_version, worldgen, player, specs, edits, mods };
    Ok(if clean {
        Decoded::Intact(doc)
    } else {
        Decoded::Salvaged { doc, recovered, expected }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SaveDoc {
        SaveDoc {
            worldgen_version: 2,
            worldgen: WorldgenStamp::default(),
            meta: SaveMeta {
                name: "My World".to_string(),
                seed: -4242,
                created: 1_770_000_000,
                last_played: 1_770_001_234,
                playtime_secs: 3600,
                edit_count: 3,
            },
            player: PlayerState {
                pos: [1.0e8 + 0.123456789, 61.5, -(1.0e9 - 42.25)],
                yaw: 1.25,
                pitch: -0.5,
                flying: true,
                noclip: true,
                stash: Some(vec![("Stone".into(), 2), ("Iron".into(), 1)]),
            },
            specs: vec!["air".to_string(), "natural:Stone".to_string()],
            edits: vec![
                Edit { x: 1, y: 60, z: -1, spec: 0 },
                Edit { x: 2, y: 61, z: -2, spec: 1 },
                Edit { x: 3, y: 62, z: -3, spec: 0 },
            ],
            mods: vec![("inventory".to_string(), "Stone,Iron".to_string())],
        }
    }

    fn expect_intact(d: Decoded) -> SaveDoc {
        match d {
            Decoded::Intact(doc) => doc,
            Decoded::Salvaged { recovered, expected, .. } => {
                panic!("expected intact, got salvage {recovered}/{expected}")
            }
        }
    }

    /// Drop the v7 stash blob so a current encode can be spliced into an older
    /// version whose player record ends at the flags byte.
    fn strip_stash(bytes: &[u8]) -> Vec<u8> {
        let start = HEADER_LEN + PLAYER_POSE_LEN;
        let len = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()) as usize;
        let mut out = Vec::with_capacity(bytes.len() - 2 - len);
        out.extend_from_slice(&bytes[..start]);
        out.extend_from_slice(&bytes[start + 2 + len..]);
        out
    }

    #[test]
    fn round_trip_is_identity() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        assert_eq!(expect_intact(decode(&bytes).unwrap()), doc);
    }

    #[test]
    fn diffusion_stamp_round_trips() {
        let mut doc = sample();
        doc.worldgen = WorldgenStamp {
            kind: 1,
            tile: 64,
            stride: 8,
            phases: 6,
            relief: 1.5,
        };
        let bytes = encode(&doc).unwrap();
        assert_eq!(expect_intact(decode(&bytes).unwrap()), doc);
    }

    #[test]
    fn far_positions_round_trip_bit_exactly() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        let loaded = expect_intact(decode(&bytes).unwrap());
        for i in 0..3 {
            assert_eq!(loaded.player.pos[i].to_bits(), doc.player.pos[i].to_bits());
        }
    }

    #[test]
    fn peek_reads_meta_from_header_prefix_alone() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        assert_eq!(peek_meta(&bytes[..HEADER_LEN]).unwrap(), doc.meta);
    }

    #[test]
    fn truncated_edits_salvage_the_prefix() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        // Cut mid-way through the third edit record.
        let cut = bytes.len() - (1 + 1 + "inventory".len() + 4 + "Stone,Iron".len()) - 7;
        match decode(&bytes[..cut]).unwrap() {
            Decoded::Salvaged { doc: got, recovered, expected } => {
                assert_eq!((recovered, expected), (2, 3));
                assert_eq!(got.edits, doc.edits[..2]);
                assert!(got.mods.is_empty(), "mods after the break are dropped");
            }
            Decoded::Intact(_) => panic!("truncation must not decode as intact"),
        }
    }

    #[test]
    fn truncated_mods_keep_all_edits() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        match decode(&bytes[..bytes.len() - 3]).unwrap() {
            Decoded::Salvaged { doc: got, recovered, expected } => {
                assert_eq!((recovered, expected), (3, 3), "only mod state was lost");
                assert_eq!(got.edits, doc.edits);
            }
            Decoded::Intact(_) => panic!("truncation must not decode as intact"),
        }
    }

    #[test]
    fn out_of_table_spec_index_truncates_there() {
        let doc = sample();
        let mut bytes = encode(&doc).unwrap();
        // Corrupt the second edit's spec index (last 2 of its 14 bytes).
        let mods_len = 1 + 1 + "inventory".len() + 4 + "Stone,Iron".len();
        let edit2_spec = bytes.len() - mods_len - EDIT_BYTES - 2;
        bytes[edit2_spec..edit2_spec + 2].copy_from_slice(&999u16.to_le_bytes());
        match decode(&bytes).unwrap() {
            Decoded::Salvaged { recovered, .. } => assert_eq!(recovered, 1),
            Decoded::Intact(_) => panic!("bad spec index must not decode as intact"),
        }
    }

    #[test]
    fn truncated_header_or_spec_table_is_fatal() {
        let doc = sample();
        let bytes = encode(&doc).unwrap();
        assert!(decode(&bytes[..HEADER_LEN - 1]).is_err());
        assert!(decode(&bytes[..HEADER_LEN + 10]).is_err());
    }

    #[test]
    fn bad_magic_and_version_are_errors() {
        let doc = sample();
        let mut bytes = encode(&doc).unwrap();
        bytes[0] = b'N';
        assert!(matches!(decode(&bytes), Err(SaveError::Corrupt(_))));
        bytes[0] = b'W';
        bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SaveError::BadVersion(3))));
        bytes[4..6].copy_from_slice(&8u16.to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SaveError::BadVersion(8))));
    }

    #[test]
    fn version_4_files_still_decode_with_the_legacy_worldgen_stamp() {
        // A v4 file is a current file minus the v5 worldgen stamp, the v6
        // kind/knobs, and the v7 player stash: splice those out and patch the
        // version. It must decode INTACT with worldgen_version defaulting to 1
        // and kind classic.
        let doc = sample();
        let current = encode(&doc).unwrap();
        let body = strip_stash(&current);
        let mut v4 = Vec::with_capacity(body.len() - (HEADER_LEN - HEADER_LEN_V4));
        v4.extend_from_slice(&body[..HEADER_LEN_V4]);
        v4.extend_from_slice(&body[HEADER_LEN..]);
        v4[4..6].copy_from_slice(&4u16.to_le_bytes());

        let got = expect_intact(decode(&v4).unwrap());
        assert_eq!(got.worldgen_version, 1, "v4 files predate the stamp");
        assert_eq!(got.worldgen, WorldgenStamp::default());
        assert_eq!(got.meta, doc.meta);
        let mut player = doc.player.clone();
        player.stash = None;
        assert_eq!(got.player, player);
        assert_eq!(got.specs, doc.specs);
        assert_eq!(got.edits, doc.edits);
        assert_eq!(got.mods, doc.mods);
        // And the peek path (slot lists) accepts the shorter header too.
        assert_eq!(peek_meta(&v4).unwrap().name, doc.meta.name);
    }

    #[test]
    fn version_5_files_still_decode_as_classic() {
        // A v5 file is a v6 file minus the kind + diffusion knobs. Kind
        // defaults to classic so pre-InfiniteDiffusion worlds keep their
        // generator; knobs take the shipped defaults.
        let mut doc = sample();
        doc.worldgen = WorldgenStamp {
            kind: 1,
            tile: 64,
            stride: 32,
            phases: 4,
            relief: 2.0,
        };
        let v7 = encode(&doc).unwrap();
        let body = strip_stash(&v7);
        let mut v5 = Vec::with_capacity(body.len() - WORLDGEN_STAMP_LEN);
        v5.extend_from_slice(&body[..HEADER_LEN_V5]);
        v5.extend_from_slice(&body[HEADER_LEN..]);
        v5[4..6].copy_from_slice(&5u16.to_le_bytes());

        let got = expect_intact(decode(&v5).unwrap());
        assert_eq!(got.worldgen_version, doc.worldgen_version);
        assert_eq!(got.worldgen, WorldgenStamp::default(), "v5 files predate kind");
        assert_eq!(got.meta, doc.meta);
        let mut player = doc.player.clone();
        player.stash = None;
        assert_eq!(got.player, player);
        assert_eq!(got.specs, doc.specs);
        assert_eq!(got.edits, doc.edits);
        assert_eq!(got.mods, doc.mods);
        assert_eq!(peek_meta(&v5).unwrap().name, doc.meta.name);
    }

    #[test]
    fn version_6_files_still_decode_without_stash() {
        let doc = sample();
        let v7 = encode(&doc).unwrap();
        let mut v6 = strip_stash(&v7);
        v6[4..6].copy_from_slice(&6u16.to_le_bytes());

        let got = expect_intact(decode(&v6).unwrap());
        assert_eq!(got.worldgen, doc.worldgen);
        let mut player = doc.player.clone();
        player.stash = None;
        assert_eq!(got.player, player, "v6 files predate the player stash field");
        assert_eq!(got.specs, doc.specs);
        assert_eq!(got.edits, doc.edits);
        assert_eq!(got.mods, doc.mods);
    }

    #[test]
    fn set_name_patches_in_place() {
        let doc = sample();
        let mut bytes = encode(&doc).unwrap();
        set_name(&mut bytes, "Renamed").unwrap();
        let got = expect_intact(decode(&bytes).unwrap());
        assert_eq!(got.meta.name, "Renamed");
        assert_eq!(got.edits, doc.edits, "body untouched");
    }

    #[test]
    fn overlong_names_clamp_on_a_char_boundary() {
        let mut doc = sample();
        doc.meta.name = format!("{}é", "x".repeat(63)); // é straddles the 64-byte cut
        let bytes = encode(&doc).unwrap();
        assert_eq!(peek_meta(&bytes).unwrap().name, "x".repeat(63));
    }

    #[test]
    fn non_finite_or_out_of_world_player_state_is_rejected() {
        let doc = sample();
        let base = encode(&doc).unwrap();

        // NaN into pos.x (first f64 after the header).
        let mut bytes = base.clone();
        bytes[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SaveError::Corrupt(_))));

        // Infinity into yaw (after the three f64 position words).
        let mut bytes = base.clone();
        bytes[HEADER_LEN + 24..HEADER_LEN + 28].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SaveError::Corrupt(_))));

        // A finite position far outside the world border is corrupt too.
        let mut bytes = base;
        bytes[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&3.0e9f64.to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SaveError::Corrupt(_))));
    }

    #[test]
    fn encode_rejects_edit_with_missing_spec() {
        let mut doc = sample();
        doc.edits.push(Edit { x: 0, y: 0, z: 0, spec: 7 });
        doc.meta.edit_count = doc.edits.len() as u32;
        assert!(encode(&doc).is_err());
    }
}
