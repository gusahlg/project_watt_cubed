//! The wire protocol: the message enums the client and server exchange, a
//! tiny hand-rolled binary codec for them, and the length-prefixed framing.
//!
//! Binary on purpose: the hot message is [`ClientMessage::Move`] /
//! [`ServerMessage::PeerMove`] at tick rate for every player, so each is a
//! fixed handful of bytes rather than a line of text. Variable data (names,
//! chat, block specs) is length-prefixed and bounded by the caps in the
//! [parent module](super).
//!
//! Positions travel as 3x f64 (24 bytes): the game plays out to ±1e9 blocks,
//! where f32 cannot even represent adjacent positions.
use std::io;
#[cfg(test)]
use std::io::{Read, Write};
use std::sync::Arc;

use quinn::{RecvStream, SendStream};
use glam::DQuat;
use voxel_engine::{DVec3, Vec3};

use crate::coord::Face;
use crate::ident::codec;
use crate::presence::Stance;
use crate::world::terrain::TerrainCfg;
use crate::world::generation::WorldgenKind;

use super::{MAX_FRAME, MAX_VOICE_PAYLOAD};

pub(crate) fn law_stamp() -> [u8; material::STAMP_LEN] {
    let v = material::Law::current().stamp();
    let mut a = [0u8; material::STAMP_LEN];
    a.copy_from_slice(&v);
    a
}

/// `Ok` iff `stamp` is this build's law. Anything else (an older law, another reaction function,
/// garbage) is refused with [`ServerMessage::Reject`]; never panics.
pub fn handshake_law(stamp: &[u8]) -> Result<(), ServerMessage> {
    if stamp == material::Law::current().stamp() {
        return Ok(());
    }
    let reason: Arc<str> = match material::Law::from_stamp(stamp) {
        Ok(_) => "law stamp differs from this game's law".into(),
        Err(material::LawError::Version(v)) => format!("world uses law version {v}; this game runs {}", material::LAW_ID).into(),
        Err(_) => "law stamp is not this game's law".into(),
    };
    Err(ServerMessage::Reject { reason })
}

/// One field's wire codec: how it is written to and read back from a message
/// payload. The [`messages!`] table below pairs every enum field with exactly
/// one of these impls, so the field's Rust type IS its wire format — encode
/// and decode can never disagree on layout, and a new message is one table row.
trait Wire: Sized {
    fn put(&self, w: &mut codec::Writer);
    /// `None` on malformed or truncated input (the whole message is rejected).
    fn get(r: &mut codec::Reader) -> Option<Self>;
}

impl Wire for [u8; material::STAMP_LEN] {
    fn put(&self, w: &mut codec::Writer) {
        w.raw(self);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let s = r.take(material::STAMP_LEN).ok()?;
        let mut a = [0u8; material::STAMP_LEN];
        a.copy_from_slice(s);
        Some(a)
    }
}

/// Plain fixed-width fields whose `Writer`/`Reader` method pair share a name.
macro_rules! wire_scalar {
    ($($t:ty => $m:ident),* $(,)?) => {$(
        impl Wire for $t {
            fn put(&self, w: &mut codec::Writer) {
                w.$m(*self);
            }
            fn get(r: &mut codec::Reader) -> Option<Self> {
                r.$m().ok()
            }
        }
    )*};
}
wire_scalar!(u8 => u8, u32 => u32, u64 => u64, i32 => i32, i64 => i64, f32 => f32, f64 => f64, DVec3 => vec3);

/// Strings travel as `Arc<str>` end to end: the sender can broadcast one
/// interned name/spec as a refcount bump per recipient, and the receiver
/// stores the very allocation the decoder produced (peer rosters, edit
/// ledgers) instead of cloning it onward.
impl Wire for Arc<str> {
    fn put(&self, w: &mut codec::Writer) {
        w.str16(self);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        r.str16_lossy().ok().map(Arc::from)
    }
}

impl Wire for Stance {
    fn put(&self, w: &mut codec::Writer) {
        w.u8(self.wire());
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        Stance::from_wire(r.u8().ok()?)
    }
}

impl Wire for WorldgenKind {
    fn put(&self, w: &mut codec::Writer) {
        w.u8(self.wire());
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        WorldgenKind::from_wire(r.u8().ok()?)
    }
}

impl Wire for DQuat {
    fn put(&self, w: &mut codec::Writer) {
        w.quat(*self);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        r.quat().ok()
    }
}

impl Wire for Vec3 {
    fn put(&self, w: &mut codec::Writer) {
        w.f32(self.x);
        w.f32(self.y);
        w.f32(self.z);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        Some(Vec3::new(r.f32().ok()?, r.f32().ok()?, r.f32().ok()?))
    }
}

impl Wire for Face {
    fn put(&self, w: &mut codec::Writer) {
        w.u8(*self as u8);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        Face::from_index(r.u8().ok()?)
    }
}

impl Wire for TerrainCfg {
    fn put(&self, w: &mut codec::Writer) {
        for v in self.to_wire() {
            w.u32(v as u32);
        }
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let mut v = [0u16; 8];
        for x in v.iter_mut() {
            *x = u16::try_from(r.u32().ok()?).ok()?;
        }
        Some(TerrainCfg::from_wire(v))
    }
}

/// One byte, strictly `0`/`1` — any other value rejects the whole message
/// rather than silently mapping to `true`.
impl Wire for bool {
    fn put(&self, w: &mut codec::Writer) {
        w.u8(*self as u8);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        match r.u8().ok()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

/// Raw audio bytes already known to fit [`MAX_VOICE_PAYLOAD`] — the bound is
/// checked once, in `TryFrom<Vec<u8>>` below, so nothing downstream (encode,
/// relay) needs to re-check or trust a caller.
#[derive(Clone, Debug, PartialEq)]
pub struct VoicePayload(Vec<u8>);

impl VoicePayload {
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn into_boxed_slice(self) -> Box<[u8]> {
        self.0.into_boxed_slice()
    }
}

/// `Err` if `bytes` exceeds [`MAX_VOICE_PAYLOAD`] — the only place that bound
/// is enforced; every `VoicePayload` in the system is provably in range.
impl TryFrom<Vec<u8>> for VoicePayload {
    type Error = ();
    fn try_from(bytes: Vec<u8>) -> Result<Self, ()> {
        (bytes.len() <= MAX_VOICE_PAYLOAD).then_some(Self(bytes)).ok_or(())
    }
}

/// u16 length prefix, then the bytes. Decode rejects a length prefix past the
/// cap before the bytes are trusted — a hostile peer can't smuggle an
/// over-cap frame past the codec.
impl Wire for VoicePayload {
    fn put(&self, w: &mut codec::Writer) {
        w.u16(self.0.len() as u16);
        w.raw(&self.0);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let len = r.u16().ok()? as usize;
        if len > MAX_VOICE_PAYLOAD {
            return None;
        }
        Some(Self(r.take(len).ok()?.to_vec()))
    }
}

/// The snapshot edit list: u32 count, then each cell's coord, revision, and
/// spec. The pre-reserve is clamped so a forged count can't balloon memory
/// before the per-entry reads fail on truncation.
impl Wire for Vec<(i32, i32, i32, u32, Arc<str>)> {
    fn put(&self, w: &mut codec::Writer) {
        w.u32(self.len() as u32);
        for (x, y, z, rev, spec) in self {
            w.i32(*x);
            w.i32(*y);
            w.i32(*z);
            w.u32(*rev);
            w.str16(spec);
        }
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let count = r.u32().ok()? as usize;
        let mut edits = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            edits.push((
                r.i32().ok()?,
                r.i32().ok()?,
                r.i32().ok()?,
                r.u32().ok()?,
                r.str16_lossy().ok()?.into(),
            ));
        }
        Some(edits)
    }
}

/// Define one direction's message enum AND its codec from a single table:
/// `Variant = TAG { field: Type, .. }`. Declaration order of the fields is the
/// wire order; each type's [`Wire`] impl is its byte format. Generates the
/// enum (docs preserved), `encode` (tag byte + fields), and a total `decode`
/// that rejects malformed, truncated, and trailing-byte payloads.
macro_rules! messages {
    (
        $(#[$enum_meta:meta])*
        pub enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $tag:path $( { $( $field:ident : $ty:ty ),+ $(,)? } )?
            ),* $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub enum $name {
            $( $(#[$variant_meta])* $variant $( { $( $field : $ty ),+ } )? ),*
        }

        impl $name {
            /// Serialise to a frame payload (tag byte + fields, in declared order).
            pub fn encode(&self) -> Vec<u8> {
                let mut w = codec::Writer::new();
                match self {
                    $( $name::$variant $( { $( $field ),+ } )? => {
                        w.u8($tag);
                        $( $( Wire::put($field, &mut w); )+ )?
                    } )*
                }
                w.into_inner()
            }

            /// Parse a frame payload. `None` on any malformed or truncated input.
            pub fn decode(bytes: &[u8]) -> Option<Self> {
                let mut r = codec::Reader::new(bytes);
                let message = match r.u8().ok()? {
                    $( t if t == $tag => $name::$variant $( { $( $field: Wire::get(&mut r)? ),+ } )?, )*
                    _ => return None,
                };
                r.finished().then_some(message)
            }
        }
    };
}

// Message type tags. Client and server tag spaces are independent.
mod tag {
    pub const HELLO: u8 = 0;
    pub const MOVE: u8 = 1;
    pub const EDIT: u8 = 2;
    pub const CHAT: u8 = 3;
    pub const SET_TIME: u8 = 4;
    pub const SWING: u8 = 5;
    pub const PING: u8 = 6;
    pub const TELEPORT: u8 = 7;
    pub const VOICE: u8 = 8;
    pub const TOOL_USE: u8 = 9;
    pub const CRUISE: u8 = 10;

    pub const WELCOME: u8 = 0;
    pub const REJECT: u8 = 1;
    pub const SNAPSHOT: u8 = 2;
    pub const PEER_JOINED: u8 = 3;
    pub const PEER_LEFT: u8 = 4;
    pub const PEER_MOVE: u8 = 5;
    pub const S_EDIT: u8 = 6;
    pub const S_CHAT: u8 = 7;
    pub const S_TIME: u8 = 8;
    pub const PEER_SWING: u8 = 9;
    pub const PONG: u8 = 10;
    pub const EDIT_ACK: u8 = 11;
    pub const POSITION: u8 = 12;
    pub const PEER_EXITED: u8 = 13;
    pub const PEER_VOICE: u8 = 14;
    pub const TOOL_RESULT: u8 = 15;
}

messages! {
    /// A message from a client to the server.
    pub enum ClientMessage {
        /// Content parts only: generator version, gravity, material law, palette.
        /// Worldgen kind and terrain knobs arrive in [`ServerMessage::Welcome`].
        /// `protocol` stays the first field so a peer can be told "server vX, client vY"
        /// before the rest of the payload is decoded.
        Hello = tag::HELLO {
            protocol: u32,
            worldgen: u32,
            gravity: u64,
            law: u64,
            palette: u64,
            name: Arc<str>,
            password: Arc<str>,
        },
        /// Declare cruise. `speed` 0 ends it; otherwise the movement envelope may
        /// follow up to `min(speed, CRUISE_MAX)`. Sent when the cruise state changes.
        Cruise = tag::CRUISE { speed: f64 },
        /// Client simulates its own player; server-side this is plausibility-checked
        /// (movement envelope + border) — discontinuities must go through
        /// [`Teleport`](Self::Teleport).
        Move = tag::MOVE { pos: DVec3, yaw: f32, pitch: f32, frame: DQuat, velocity: Vec3, up: Face, stance: Stance },
        /// Exempt from the movement envelope, but the server may refuse it
        /// (configuration) and answer with a [`ServerMessage::Position`] snap-back.
        Teleport = tag::TELEPORT { pos: DVec3 },
        Swing = tag::SWING,
        /// The server echoes `nonce` back in [`ServerMessage::Pong`].
        Ping = tag::PING { nonce: u32 },
        /// `req` identifies this request in the sender's [`ServerMessage::EditAck`];
        /// `expect` is the cell revision the sender believes is current (0 = never
        /// edited), so racing edits on one cell resolve to exactly one winner.
        Edit = tag::EDIT { req: u32, x: i32, y: i32, z: i32, expect: u32, spec: Arc<str> },
        Chat = tag::CHAT { channel: u8, text: Arc<str> },
        /// `day` is a `[0,1)` fraction.
        SetTime = tag::SET_TIME { day: f32 },
        /// Part of a loss-tolerant journal: `seq` orders the sender's own stream so
        /// the receiver's jitter buffer can reorder and detect gaps. The server
        /// stamps speaker id + epoch on relay; the client never mints those.
        /// `payload` is bounded by [`MAX_VOICE_PAYLOAD`](super::MAX_VOICE_PAYLOAD).
        Voice = tag::VOICE { seq: u32, payload: VoicePayload },
        /// Use the held configuration `tool_spec` as a tool on the cell: the server runs ONE
        /// operation of the law between the cell (A) and the tool (B) and answers with
        /// [`ServerMessage::ToolResult`]. `req`/`expect` as for [`Edit`](Self::Edit).
        ToolUse = tag::TOOL_USE { req: u32, x: i32, y: i32, z: i32, expect: u32, tool_spec: Arc<str> },
    }
}

messages! {
    /// A message from the server to a client.
    pub enum ServerMessage {
        Welcome = tag::WELCOME {
            player_id: u32,
            seed: i64,
            spawn: DVec3,
            worldgen: WorldgenKind,
            terrain: TerrainCfg,
            law: [u8; material::STAMP_LEN],
        },
        /// The stream closes after this (bad password, version mismatch, server full).
        Reject = tag::REJECT { reason: Arc<str> },
        /// Sent once right after [`Welcome`](Self::Welcome). Each cell carries its
        /// authoritative revision so the joiner's future edit expectations line up.
        Snapshot = tag::SNAPSHOT { edits: Vec<(i32, i32, i32, u32, Arc<str>)> },
        /// Roster only — a peer's pose arrives via [`PeerMove`](Self::PeerMove) once
        /// they are inside interest range.
        PeerJoined = tag::PEER_JOINED { id: u32, name: Arc<str> },
        PeerLeft = tag::PEER_LEFT { id: u32 },
        /// Also the "entered interest range" signal.
        PeerMove = tag::PEER_MOVE { id: u32, pos: DVec3, yaw: f32, pitch: f32, frame: DQuat, velocity: Vec3, up: Face, stance: Stance },
        /// A peer left interest range: hide their avatar instead of drawing a
        /// frozen ghost at the last heard pose. They re-appear on the next
        /// [`PeerMove`](Self::PeerMove) for that id.
        PeerExited = tag::PEER_EXITED { id: u32 },
        PeerSwing = tag::PEER_SWING { id: u32 },
        /// Echo of a [`ClientMessage::Ping`], carrying its `nonce` unchanged.
        Pong = tag::PONG { nonce: u32 },
        /// Sent to everyone except the editor (who gets the ack).
        Edit = tag::S_EDIT { x: i32, y: i32, z: i32, rev: u32, spec: Arc<str> },
        /// `accepted` with the committed revision, or rejected (stale expectation,
        /// out of reach, invalid spec, or a server-mod `Deny`) — the signal
        /// prediction rolls back on. A hook Deny does not advance the cell, so
        /// the client's `restore` is the same as a lost race.
        EditAck = tag::EDIT_ACK { req: u32, accepted: bool, rev: u32 },
        /// Refused teleport or implausible movement: snap to it. Carries the
        /// body frame and the up-axis face the server last accepted.
        Position = tag::POSITION { pos: DVec3, frame: DQuat, up: Face },
        Chat = tag::S_CHAT { from_id: u32, from_name: Arc<str>, channel: u8, text: Arc<str> },
        /// `day` is a `[0,1)` fraction and `day_secs` the shared real-seconds
        /// length of a full cycle, so every clock advances in step.
        Time = tag::S_TIME { day: f32, day_secs: f32 },
        /// `id` is the speaker's server-assigned player id (the runtime's
        /// `SessionKey`); `epoch` distinguishes reconnections under a reused id —
        /// constant `0` here because the server never reuses ids. `seq` and
        /// `payload` are the sender's own [`ClientMessage::Voice`] values, unchanged.
        PeerVoice = tag::PEER_VOICE { id: u32, epoch: u32, seq: u32, payload: VoicePayload },
        /// The outcome of the sender's [`ClientMessage::ToolUse`] `req`: whether the law moved
        /// anything, the cell's committed revision, and both configurations afterwards (unchanged
        /// specs when nothing reacted or the request was refused).
        ToolResult = tag::TOOL_RESULT { req: u32, reacted: bool, rev: u32, cell_spec: Arc<str>, tool_spec: Arc<str> },
    }
}

/// What the first bytes of a client frame say, before a full [`ClientMessage`] decode.
pub(crate) enum HelloPeek {
    /// Tag byte is missing or is not `Hello`.
    NotHello,
    /// Tag is `Hello` but the protocol `u32` is not all there.
    Truncated,
    /// Little-endian protocol number sitting immediately after the tag.
    Protocol(u32),
}

/// Tag, then the protocol number. A v12 `Hello` laid the same two fields first,
/// so a mismatched peer is named without parsing the rest of its payload.
pub(crate) fn peek_hello(frame: &[u8]) -> HelloPeek {
    match frame.first() {
        Some(&tag::HELLO) if frame.len() >= 5 => {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&frame[1..5]);
            HelloPeek::Protocol(u32::from_le_bytes(bytes))
        }
        Some(&tag::HELLO) => HelloPeek::Truncated,
        _ => HelloPeek::NotHello,
    }
}

fn frame_header(payload: &[u8]) -> io::Result<[u8; 4]> {
    if payload.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    Ok((payload.len() as u32).to_be_bytes())
}

fn frame_len(header: [u8; 4]) -> io::Result<usize> {
    let len = u32::from_be_bytes(header) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame exceeds cap"));
    }
    Ok(len)
}

/// Refuses to emit an over-cap frame so both ends share one hard size bound.
#[cfg(test)]
pub fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    w.write_all(&frame_header(payload)?)?;
    w.write_all(payload)
}

/// `buf` is caller-owned scratch, reused so steady-state traffic never
/// allocates per frame. Rejects a length past [`MAX_FRAME`] before growing
/// the buffer, so a malicious header can't trigger a huge or endless read.
#[cfg(test)]
pub fn read_frame<R: Read>(r: &mut R, buf: &mut Vec<u8>) -> io::Result<()> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    buf.resize(frame_len(len_bytes)?, 0);
    r.read_exact(buf)
}

/// Async twin of [`write_frame`] over a QUIC send stream. No explicit flush
/// and no `finish` — quinn transmits on its own, and finishing would close
/// the multiplexed stream.
pub async fn write_frame_async(s: &mut SendStream, payload: &[u8]) -> io::Result<()> {
    s.write_all(&frame_header(payload)?).await.map_err(io::Error::other)?;
    s.write_all(payload).await.map_err(io::Error::other)
}

/// Async twin of [`read_frame`] over a QUIC recv stream.
pub async fn read_frame_async(r: &mut RecvStream, buf: &mut Vec<u8>) -> io::Result<()> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes).await.map_err(io::Error::other)?;
    buf.resize(frame_len(len_bytes)?, 0);
    r.read_exact(buf).await.map_err(io::Error::other)
}

// The frame-length envelope (write_frame/read_frame) stays big-endian,
// independent of the little-endian message-payload codec.

#[cfg(test)]
mod tests {
    use super::*;

    fn client_cases() -> Vec<ClientMessage> {
        vec![
            ClientMessage::Hello {
                protocol: 1,
                worldgen: 10,
                gravity: 0xDEAD_BEEF_1234_5678,
                law: 0x1111,
                palette: 0x2222,
                name: "player".into(),
                password: "hunter2".into(),
            },
            ClientMessage::Cruise { speed: 1.5e8 },
            ClientMessage::Cruise { speed: 0.0 },
            ClientMessage::Move {
                pos: DVec3::new(1.5, -2.0, 3.25),
                yaw: 0.5,
                pitch: -0.25,
                frame: DQuat::from_xyzw(0.0, 1.0, 0.0, 0.0),
                velocity: Vec3::new(1.5, -2.25, 0.5),
                up: Face::PosX,
                stance: Stance::Sneaking,
            },
            ClientMessage::Teleport { pos: DVec3::new(1.0e8, -40.0, 3.5) },
            ClientMessage::Swing,
            ClientMessage::Ping { nonce: 7 },
            ClientMessage::Edit {
                req: 12,
                x: -4,
                y: 7,
                z: 900,
                expect: 3,
                spec: "natural:Stone".into(),
            },
            ClientMessage::Chat { channel: 1, text: "hello world".into() },
            ClientMessage::SetTime { day: 0.5 },
            ClientMessage::Voice { seq: 5, payload: vec![1, 2, 3, 4].try_into().unwrap() },
            ClientMessage::Voice { seq: 0, payload: Vec::new().try_into().unwrap() },
            ClientMessage::ToolUse { req: 3, x: 5, y: -60, z: 9, expect: 2, tool_spec: "c:0101020304".into() },
        ]
    }

    fn server_cases() -> Vec<ServerMessage> {
        vec![
            ServerMessage::Welcome {
                player_id: 42,
                seed: -9_999,
                spawn: DVec3::new(0.5, 40.0, 0.5),
                worldgen: WorldgenKind::Flat,
                terrain: TerrainCfg::default(),
                law: law_stamp(),
            },
            ServerMessage::Welcome {
                player_id: 7,
                seed: 11,
                spawn: DVec3::new(1.0, 20.0, 2.0),
                worldgen: WorldgenKind::Diffusion,
                terrain: TerrainCfg { relief: 150, caves: 50, mines: 0, space: 200, ..Default::default() },
                law: law_stamp(),
            },
            ServerMessage::Reject { reason: "bad password".into() },
            ServerMessage::Snapshot {
                edits: vec![
                    (1, 2, 3, 1, "air".into()),
                    (-5, 6, -7, 9, "mixture:Soil=70;Clay=30".into()),
                ],
            },
            ServerMessage::PeerJoined { id: 3, name: "friend".into() },
            ServerMessage::PeerLeft { id: 3 },
            ServerMessage::PeerMove {
                id: 3,
                pos: DVec3::new(9.0, 8.0, 7.0),
                yaw: 1.0,
                pitch: 0.1,
                frame: DQuat::from_xyzw(1.0, 0.0, 0.0, 0.0),
                velocity: Vec3::new(0.25, 0.0, -1.5),
                up: Face::NegY,
                stance: Stance::Sneaking,
            },
            ServerMessage::PeerExited { id: 3 },
            ServerMessage::PeerSwing { id: 3 },
            ServerMessage::Pong { nonce: 7 },
            ServerMessage::Edit { x: 0, y: 0, z: 0, rev: 4, spec: "air".into() },
            ServerMessage::EditAck { req: 12, accepted: true, rev: 4 },
            ServerMessage::EditAck { req: 13, accepted: false, rev: 4 },
            ServerMessage::Position {
                pos: DVec3::new(-1.0e9, 2.0, 3.0),
                frame: DQuat::from_xyzw(0.0, 0.0, 1.0, 0.0),
                up: Face::PosZ,
            },
            ServerMessage::Chat {
                from_id: 3,
                from_name: "friend".into(),
                channel: 0,
                text: "hi".into(),
            },
            ServerMessage::Time { day: 0.75, day_secs: 600.0 },
            ServerMessage::PeerVoice { id: 3, epoch: 0, seq: 5, payload: vec![9, 8, 7].try_into().unwrap() },
            ServerMessage::PeerVoice { id: 1, epoch: 2, seq: 0, payload: Vec::new().try_into().unwrap() },
            ServerMessage::ToolResult {
                req: 3,
                reacted: true,
                rev: 7,
                cell_spec: "air".into(),
                tool_spec: "c:0201020304aabbccdd".into(),
            },
        ]
    }

    #[test]
    fn spec_round_trips_through_the_wire() {
        let mut r = crate::block::BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut r);
        let id = r.id_by_label("rock").unwrap();
        let spec = r.spec(id);
        let msg = ClientMessage::Edit {
            req: 1,
            x: 0,
            y: 1,
            z: 2,
            expect: 0,
            spec: spec.clone().into(),
        };
        match ClientMessage::decode(&msg.encode()) {
            Some(ClientMessage::Edit { spec: got, .. }) => assert_eq!(&*got, spec),
            other => panic!("bad decode: {other:?}"),
        }
        let mut r2 = crate::block::BlockRegistry::with_builtins();
        let id2 = r2.parse_spec(&spec).unwrap();
        assert_eq!(r2.configuration(id2), r.configuration(id));
    }

    #[test]
    fn spec_round_trips_through_save_and_wire_for_random_configs() {
        use crate::save::format::{self, PlayerState, SaveDoc, WorldgenStamp};
        use crate::save::slot::SaveMeta;
        let mut r = crate::block::BlockRegistry::with_builtins();
        let mut s = 0xDEAD_BEEFu64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut ids = Vec::new();
        for _ in 0..40 {
            let n = 1 + (next() as usize % 4);
            let elems: Vec<_> = (0..n)
                .map(|_| {
                    let x = next();
                    material::Element::new([
                        x as u8,
                        (x >> 8) as u8,
                        (x >> 16) as u8,
                        (x >> 24) as u8,
                    ])
                })
                .collect();
            let c = material::Configuration::new(elems).unwrap();
            ids.push(r.intern(&c).unwrap());
        }
        for id in ids {
            let spec = r.spec(id);
            let msg = ClientMessage::Edit {
                req: 9,
                x: -4,
                y: 20,
                z: 7,
                expect: 1,
                spec: spec.clone().into(),
            };
            match ClientMessage::decode(&msg.encode()) {
                Some(ClientMessage::Edit { spec: got, .. }) => assert_eq!(&*got, spec),
                other => panic!("wire lost spec: {other:?}"),
            }
            let doc = SaveDoc {
                meta: SaveMeta {
                    name: "t".into(),
                    seed: 1,
                    created: 0,
                    last_played: 0,
                    playtime_secs: 0,
                    edit_count: 1,
                },
                worldgen_version: crate::world::terrain::WORLDGEN_VERSION,
                worldgen: WorldgenStamp::default(),
                law_stamp: material::Law::current().stamp(),
                player: PlayerState {
                    pos: [0.0, 0.0, 0.0],
                    yaw: 0.0,
                    pitch: 0.0,
                    frame: DQuat::IDENTITY,
                    velocity: [0.0; 3],
                    up: Face::PosY as u8,
                    legacy_pose: false,
                    flying: false,
                    noclip: false,
                    inventory: Some(vec![(spec.clone(), 1)]),
                },
                specs: vec![spec.clone()],
                edits: vec![format::Edit { x: 1, y: 2, z: 3, spec: 0 }],
                mods: vec![],
                pending: vec![],
            };
            let bytes = format::encode(&doc).unwrap();
            let back = match format::decode(&bytes).unwrap() {
                format::Decoded::Intact(d) => d,
                other => panic!("save lost spec: {other:?}"),
            };
            assert_eq!(back.specs, vec![spec.clone()]);
            assert_eq!(back.player.inventory.unwrap()[0].0, spec);
            let mut r2 = crate::block::BlockRegistry::with_builtins();
            let id2 = r2.parse_spec(&spec).unwrap();
            assert_eq!(r2.configuration(id2), r.configuration(id));
        }
    }

    #[test]
    fn a_client_cannot_send_reaction_results() {
        // Multi-cell Snapshot (the server's reaction broadcast) is not a ClientMessage.
        let snap = ServerMessage::Snapshot {
            edits: vec![
                (1, 2, 3, 4, "c:0101020304".into()),
                (5, 6, 7, 8, "c:0101020304".into()),
            ],
        };
        assert_eq!(
            ClientMessage::decode(&snap.encode()),
            None,
            "a reaction Snapshot must not decode as a client edit"
        );
    }

    #[test]
    fn messages_obey_codec_contract() {
        for message in client_cases() {
            let mut payload = message.encode();
            assert_eq!(ClientMessage::decode(&payload), Some(message.clone()));
            payload.push(0xa5);
            assert_eq!(ClientMessage::decode(&payload), None, "accepted suffix after {message:?}");
        }
        for message in server_cases() {
            let mut payload = message.encode();
            assert_eq!(ServerMessage::decode(&payload), Some(message.clone()));
            payload.push(0x5a);
            assert_eq!(ServerMessage::decode(&payload), None, "accepted suffix after {message:?}");
        }
    }

    #[test]
    fn edit_ack_rejects_non_boolean_accepted_bytes() {
        let mut payload = ServerMessage::EditAck { req: 1, accepted: true, rev: 2 }.encode();
        // The `accepted` byte sits right after the tag and req.
        payload[5] = 2;
        assert_eq!(ServerMessage::decode(&payload), None);
    }

    #[test]
    fn positions_round_trip_bit_exactly_at_far_coordinates() {
        // The reason positions are f64 on the wire: at 1e8 the fractional
        // part below survives exactly; an f32 wire would quantise it to a
        // multiple of 8. Round-trip both directions of the hot path.
        let pos = DVec3::new(1.0e8 + 0.123456789, -3_000.25, -(1.0e9 - 0.75));
        let mv = ClientMessage::Move {
            pos,
            yaw: 1.0,
            pitch: -0.5,
            frame: DQuat::IDENTITY,
            velocity: Vec3::ZERO,
            up: Face::PosY,
            stance: Stance::Standing,
        };
        match ClientMessage::decode(&mv.encode()) {
            Some(ClientMessage::Move { pos: got, .. }) => {
                assert_eq!(got.x.to_bits(), pos.x.to_bits());
                assert_eq!(got.y.to_bits(), pos.y.to_bits());
                assert_eq!(got.z.to_bits(), pos.z.to_bits());
            }
            other => panic!("bad decode: {other:?}"),
        }
        let pm = ServerMessage::PeerMove {
            id: 7,
            pos,
            yaw: 0.0,
            pitch: 0.0,
            frame: DQuat::IDENTITY,
            velocity: Vec3::ZERO,
            up: Face::PosY,
            stance: Stance::Standing,
        };
        assert_eq!(ServerMessage::decode(&pm.encode()), Some(pm));
        let wl = ServerMessage::Welcome {
            player_id: 1,
            seed: 3,
            spawn: pos,
            worldgen: WorldgenKind::Diffusion,
            terrain: TerrainCfg::default(),
            law: law_stamp(),
        };
        assert_eq!(ServerMessage::decode(&wl.encode()), Some(wl));
    }

    #[test]
    fn handshake_refuses_a_perturbed_law_without_panic() {
        assert!(handshake_law(&law_stamp()).is_ok());
        let mut law = material::Law::current();
        law.probes.glow = material::Element::new([1, 2, 3, 4]);
        let mut old_version = law_stamp();
        old_version[0] = 1;
        for stamp in [law.stamp(), old_version.to_vec(), vec![0u8; 3]] {
            let encoded = match handshake_law(&stamp) {
                Err(ServerMessage::Reject { reason }) => {
                    assert!(!reason.is_empty(), "reject names the reason");
                    ServerMessage::Reject { reason }.encode()
                }
                other => panic!("expected Reject, got {other:?}"),
            };
            match ServerMessage::decode(&encoded) {
                Some(ServerMessage::Reject { reason }) => assert!(!reason.is_empty()),
                other => panic!("Reject must round-trip, got {other:?}"),
            }
        }
    }

    #[test]
    fn welcome_carries_every_terrain_knob() {
        let terrain = TerrainCfg {
            relief: 175,
            caves: 25,
            mines: 200,
            space: 0,
            variety: 125,
            features: 50,
            structures: 175,
            deep: 0,
        };
        let wl = ServerMessage::Welcome {
            player_id: 1,
            seed: 3,
            spawn: DVec3::ZERO,
            worldgen: WorldgenKind::Diffusion,
            terrain,
            law: law_stamp(),
        };
        match ServerMessage::decode(&wl.encode()) {
            Some(ServerMessage::Welcome { terrain: got, .. }) => assert_eq!(got, terrain),
            other => panic!("bad decode: {other:?}"),
        }
    }

    #[test]
    fn protocol_12_body_frame_round_trips() {
        let frame = DQuat::from_xyzw(0.0, 1.0, 0.0, 0.0);
        let velocity = Vec3::new(1.5, -2.25, 0.5);
        let mv = ClientMessage::Move {
            pos: DVec3::new(4.0, 5.0, 6.0),
            yaw: 0.25,
            pitch: -0.5,
            frame,
            velocity,
            up: Face::PosX,
            stance: Stance::Standing,
        };
        match ClientMessage::decode(&mv.encode()) {
            Some(ClientMessage::Move { frame: got_f, velocity: got_v, up, .. }) => {
                assert_eq!(got_f, frame);
                assert_eq!(got_v, velocity);
                assert_eq!(up, Face::PosX);
            }
            other => panic!("bad decode: {other:?}"),
        }
        let pm = ServerMessage::PeerMove {
            id: 3,
            pos: DVec3::new(1.0, 2.0, 3.0),
            yaw: 0.0,
            pitch: 0.0,
            frame,
            velocity,
            up: Face::NegZ,
            stance: Stance::Sneaking,
        };
        assert_eq!(ServerMessage::decode(&pm.encode()), Some(pm));
        let pos = ServerMessage::Position { pos: DVec3::new(8.0, 9.0, 10.0), frame, up: Face::PosZ };
        assert_eq!(ServerMessage::decode(&pos.encode()), Some(pos));

        // A non-finite frame repairs to identity; an unknown face rejects the message.
        let mut payload = mv.encode();
        let quat_at = 1 + 24 + 4 + 4;
        payload[quat_at..quat_at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        match ClientMessage::decode(&payload) {
            Some(ClientMessage::Move { frame: got, up, .. }) => {
                assert_eq!(got, DQuat::IDENTITY);
                assert_eq!(up, Face::PosX);
            }
            other => panic!("a bad quaternion must still decode, got {other:?}"),
        }
        let mut payload = mv.encode();
        let up_at = quat_at + 16 + 12;
        payload[up_at] = 9;
        assert_eq!(ClientMessage::decode(&payload), None);
    }

    #[test]
    fn welcome_rejects_unknown_worldgen_kind() {
        let mut payload = ServerMessage::Welcome {
            player_id: 1,
            seed: 3,
            spawn: DVec3::ZERO,
            worldgen: WorldgenKind::Flat,
            terrain: TerrainCfg::default(),
            law: law_stamp(),
        }
        .encode();
        // kind sits after tag, player_id, seed, spawn (1+4+8+24 = 37).
        payload[37] = 9;
        assert_eq!(ServerMessage::decode(&payload), None);
    }

    #[test]
    fn truncated_frame_decodes_to_none() {
        let full =
            ClientMessage::Edit { req: 1, x: 1, y: 2, z: 3, expect: 0, spec: "air".into() }
                .encode();
        // Chop the payload short: the reader must report failure, not panic.
        assert_eq!(ClientMessage::decode(&full[..full.len() - 2]), None);
        assert_eq!(ClientMessage::decode(&[]), None);
    }

    #[test]
    fn voice_payload_at_the_cap_round_trips_both_directions() {
        let payload: VoicePayload = (0..MAX_VOICE_PAYLOAD).map(|i| i as u8).collect::<Vec<u8>>().try_into().unwrap();
        let cm = ClientMessage::Voice { seq: 99, payload: payload.clone() };
        assert_eq!(ClientMessage::decode(&cm.encode()), Some(cm));
        let sm = ServerMessage::PeerVoice { id: 7, epoch: 0, seq: 99, payload };
        assert_eq!(ServerMessage::decode(&sm.encode()), Some(sm));
    }

    /// A frame whose length prefix claims more than [`MAX_VOICE_PAYLOAD`] is
    /// refused by the decoder before the bytes are trusted — the encode path
    /// can't build one (`VoicePayload::try_from` refuses it), so the frame is
    /// forged directly, exactly as a hostile peer would.
    #[test]
    fn oversized_voice_frame_is_rejected_both_directions() {
        let over = vec![0u8; MAX_VOICE_PAYLOAD + 1];

        let mut w = codec::Writer::new();
        w.u8(super::tag::VOICE);
        w.u32(1);
        w.u16(over.len() as u16);
        w.raw(&over);
        assert_eq!(ClientMessage::decode(&w.into_inner()), None);

        let mut w = codec::Writer::new();
        w.u8(super::tag::PEER_VOICE);
        w.u32(2); // id
        w.u32(0); // epoch
        w.u32(1); // seq
        w.u16(over.len() as u16);
        w.raw(&over);
        assert_eq!(ServerMessage::decode(&w.into_inner()), None);
    }

    #[test]
    fn frame_round_trips_through_a_pipe() {
        let payload = ServerMessage::PeerLeft { id: 7 }.encode();
        let mut buf = Vec::new();
        write_frame(&mut buf, &payload).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let mut read = Vec::new();
        read_frame(&mut cursor, &mut read).unwrap();
        assert_eq!(ServerMessage::decode(&read), Some(ServerMessage::PeerLeft { id: 7 }));
    }

    #[test]
    fn oversize_frame_is_refused() {
        let big = vec![0u8; MAX_FRAME + 1];
        let mut buf = Vec::new();
        assert!(write_frame(&mut buf, &big).is_err());

        // A header claiming a huge body is rejected before the body is read.
        let mut hostile = ((MAX_FRAME as u32) + 1).to_be_bytes().to_vec();
        hostile.push(0);
        let mut scratch = Vec::new();
        assert!(read_frame(&mut std::io::Cursor::new(hostile), &mut scratch).is_err());
    }

    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
        fn len(&mut self, max_incl: usize) -> usize {
            (self.next() as usize) % (max_incl + 1)
        }
    }

    fn decode_must_not_panic(bytes: &[u8]) {
        let client = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ClientMessage::decode(bytes)));
        let server = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ServerMessage::decode(bytes)));
        assert!(client.is_ok(), "ClientMessage::decode panicked on {bytes:?}");
        assert!(server.is_ok(), "ServerMessage::decode panicked on {bytes:?}");
    }

    #[test]
    fn random_frames_never_panic_the_decoder() {
        let mut rng = XorShift::new(0xC0FF_EE42_D00D);
        let mut buf = vec![0u8; MAX_FRAME];
        for b in buf.iter_mut() {
            *b = rng.byte();
        }
        for _ in 0..100_000 {
            let len = rng.len(MAX_FRAME);
            for _ in 0..16 {
                let i = rng.len(MAX_FRAME.saturating_sub(1));
                buf[i] = rng.byte();
            }
            decode_must_not_panic(&buf[..len]);
        }
    }

    fn encoding_side_rejects_trailing_bytes<T>(
        frame: Vec<u8>,
        extra: u8,
        decode: fn(&[u8]) -> Option<T>,
        rng: &mut XorShift,
    ) {
        decode_must_not_panic(&frame);
        for n in 0..frame.len() {
            decode_must_not_panic(&frame[..n]);
        }
        let mut grown = frame.clone();
        grown.push(extra);
        assert!(decode(&grown).is_none(), "encoding side must reject a trailing byte");
        decode_must_not_panic(&grown);
        if frame.is_empty() {
            return;
        }
        let i = rng.len(frame.len() - 1);
        let mut flipped = frame.clone();
        flipped[i] ^= rng.byte() | 1;
        decode_must_not_panic(&flipped);
    }

    #[test]
    fn structural_flips_and_truncations_never_panic_and_reject_trailing_bytes() {
        let mut rng = XorShift::new(0xA11C_EDED);
        for message in client_cases() {
            encoding_side_rejects_trailing_bytes(message.encode(), rng.byte(), ClientMessage::decode, &mut rng);
        }
        for message in server_cases() {
            encoding_side_rejects_trailing_bytes(message.encode(), rng.byte(), ServerMessage::decode, &mut rng);
        }
    }

    #[test]
    fn strings_at_the_cap_round_trip_and_overlong_invalid_and_nuls_never_panic() {
        let cap_name: Arc<str> = "n".repeat(super::super::MAX_NAME).into();
        let hello = ClientMessage::Hello {
            protocol: 1,
            worldgen: 0,
            gravity: 0,
            law: 0,
            palette: 0,
            name: cap_name.clone(),
            password: "".into(),
        };
        assert_eq!(ClientMessage::decode(&hello.encode()), Some(hello));

        let over: Arc<str> = "n".repeat(super::super::MAX_NAME + 1).into();
        let hello_over = ClientMessage::Hello {
            protocol: 1,
            worldgen: 0,
            gravity: 0,
            law: 0,
            palette: 0,
            name: over,
            password: "p".repeat(super::super::MAX_NAME + 1).into(),
        };
        decode_must_not_panic(&hello_over.encode());

        let cap_spec: Arc<str> = "s".repeat(super::super::MAX_SPEC).into();
        let edit = ClientMessage::Edit {
            req: 1,
            x: 0,
            y: 0,
            z: 0,
            expect: 0,
            spec: cap_spec,
        };
        assert_eq!(ClientMessage::decode(&edit.encode()), Some(edit.clone()));
        let mut over_spec = edit.clone();
        if let ClientMessage::Edit { spec, .. } = &mut over_spec {
            *spec = "s".repeat(super::super::MAX_SPEC + 1).into();
        }
        decode_must_not_panic(&over_spec.encode());

        let nuls = ClientMessage::Chat { channel: 0, text: "ok\0still".into() };
        match ClientMessage::decode(&nuls.encode()) {
            Some(ClientMessage::Chat { text, .. }) => assert!(text.contains('\0') || text.contains("ok")),
            other => panic!("nul chat must decode or reject, got {other:?}"),
        }

        // Invalid UTF-8 in a length-prefixed string: forge the bytes.
        let mut w = codec::Writer::new();
        w.u8(super::tag::CHAT);
        w.u8(0);
        w.u16(2);
        w.raw(&[0xff, 0xfe]);
        decode_must_not_panic(&w.into_inner());
    }

    #[test]
    fn frame_helpers_reject_hostile_lengths_and_accept_empty_and_cap() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &[]).unwrap();
        let mut got = Vec::new();
        read_frame(&mut std::io::Cursor::new(&buf), &mut got).unwrap();
        assert!(got.is_empty());

        let payload = vec![0x5a; MAX_FRAME];
        buf.clear();
        write_frame(&mut buf, &payload).unwrap();
        got.clear();
        read_frame(&mut std::io::Cursor::new(&buf), &mut got).unwrap();
        assert_eq!(got, payload);

        let mut hostile = u32::MAX.to_be_bytes().to_vec();
        hostile.extend_from_slice(&[1, 2, 3, 4]);
        assert!(read_frame(&mut std::io::Cursor::new(hostile), &mut got).is_err());

        let mut at_cap = (MAX_FRAME as u32).to_be_bytes().to_vec();
        at_cap.push(1); // body short of the claimed length
        assert!(read_frame(&mut std::io::Cursor::new(at_cap), &mut got).is_err());

        let mut zero = 0u32.to_be_bytes().to_vec();
        got.clear();
        read_frame(&mut std::io::Cursor::new(&zero), &mut got).unwrap();
        assert!(got.is_empty());
        zero.extend_from_slice(&[9]); // trailing unread bytes are the caller's problem
    }
}
