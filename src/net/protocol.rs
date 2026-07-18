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
use std::io::{self, Read, Write};

use quinn::{RecvStream, SendStream};
use voxel_engine::DVec3;

use crate::ident::codec;
use crate::presence::Stance;

use super::{MAX_FRAME, MAX_VOICE_PAYLOAD};

/// A message from a client to the server.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientMessage {
    /// `fingerprint` is the sender's [`content_fingerprint`](super::content_fingerprint);
    /// the server rejects a mismatch so two builds that would generate
    /// different worlds from one seed never silently join.
    Hello { protocol: u32, fingerprint: u64, name: String, password: String },
    /// Client simulates its own player; server-side this is plausibility-checked
    /// (movement envelope + border) — discontinuities must go through
    /// [`Teleport`](Self::Teleport).
    Move { pos: DVec3, yaw: f32, pitch: f32, stance: Stance },
    /// Exempt from the movement envelope, but the server may refuse it
    /// (configuration) and answer with a [`ServerMessage::Position`] snap-back.
    Teleport { pos: DVec3 },
    Swing,
    /// The server echoes `nonce` back in [`ServerMessage::Pong`].
    Ping { nonce: u32 },
    /// `req` identifies this request in the sender's [`ServerMessage::EditAck`];
    /// `expect` is the cell revision the sender believes is current (0 = never
    /// edited), so racing edits on one cell resolve to exactly one winner.
    Edit { req: u32, x: i32, y: i32, z: i32, expect: u32, spec: String },
    Chat { channel: u8, text: String },
    /// `day` is a `[0,1)` fraction.
    SetTime { day: f32 },
    /// Part of a loss-tolerant journal: `seq` orders the sender's own stream so
    /// the receiver's jitter buffer can reorder and detect gaps. The server
    /// stamps speaker id + epoch on relay; the client never mints those.
    /// `payload` is bounded by [`MAX_VOICE_PAYLOAD`](super::MAX_VOICE_PAYLOAD).
    Voice { seq: u32, payload: Vec<u8> },
}

/// A message from the server to a client.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerMessage {
    Welcome { player_id: u32, seed: i64, spawn: DVec3 },
    /// The stream closes after this (bad password, version mismatch, server full).
    Reject { reason: String },
    /// Sent once right after [`Welcome`](Self::Welcome). Each cell carries its
    /// authoritative revision so the joiner's future edit expectations line up.
    Snapshot { edits: Vec<(i32, i32, i32, u32, String)> },
    /// Roster only — a peer's pose arrives via [`PeerMove`](Self::PeerMove) once
    /// they are inside interest range.
    PeerJoined { id: u32, name: String },
    PeerLeft { id: u32 },
    /// Also the "entered interest range" signal.
    PeerMove { id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance },
    /// A peer left interest range: hide their avatar instead of drawing a
    /// frozen ghost at the last heard pose. They re-appear on the next
    /// [`PeerMove`](Self::PeerMove) for that id.
    PeerExited { id: u32 },
    PeerSwing { id: u32 },
    /// Echo of a [`ClientMessage::Ping`], carrying its `nonce` unchanged.
    Pong { nonce: u32 },
    /// Sent to everyone except the editor (who gets the ack).
    Edit { x: i32, y: i32, z: i32, rev: u32, spec: String },
    /// `accepted` with the committed revision, or rejected (stale expectation,
    /// out of reach, invalid spec) — the signal prediction rolls back on.
    EditAck { req: u32, accepted: bool, rev: u32 },
    /// Refused teleport or implausible movement: snap to it.
    Position { pos: DVec3 },
    Chat { from_id: u32, from_name: String, channel: u8, text: String },
    /// `day` is a `[0,1)` fraction and `day_secs` the shared real-seconds
    /// length of a full cycle, so every clock advances in step.
    Time { day: f32, day_secs: f32 },
    /// `id` is the speaker's server-assigned player id (the runtime's
    /// `SessionKey`); `epoch` distinguishes reconnections under a reused id —
    /// constant `0` here because the server never reuses ids. `seq` and
    /// `payload` are the sender's own [`ClientMessage::Voice`] values, unchanged.
    PeerVoice { id: u32, epoch: u32, seq: u32, payload: Vec<u8> },
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
}

/// Callers must have already refused an over-cap payload (client `send_voice`,
/// server relay); the assert catches one that forgot.
fn write_voice_payload(w: &mut codec::Writer, payload: &[u8]) {
    debug_assert!(
        payload.len() <= MAX_VOICE_PAYLOAD,
        "voice payload {} exceeds MAX_VOICE_PAYLOAD; callers must guard",
        payload.len()
    );
    w.u16(payload.len() as u16);
    w.raw(payload);
}

/// Rejects a length prefix past [`MAX_VOICE_PAYLOAD`] before the bytes are
/// trusted — a hostile peer can't smuggle an over-cap frame past the codec.
fn read_voice_payload(r: &mut codec::Reader) -> Option<Vec<u8>> {
    let len = r.u16().ok()? as usize;
    if len > MAX_VOICE_PAYLOAD {
        return None;
    }
    Some(r.take(len).ok()?.to_vec())
}

impl ClientMessage {
    /// Serialise to a frame payload (tag byte + fields).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = codec::Writer::new();
        match self {
            ClientMessage::Hello { protocol, fingerprint, name, password } => {
                w.u8(tag::HELLO);
                w.u32(*protocol);
                w.u64(*fingerprint);
                w.str16(name);
                w.str16(password);
            }
            ClientMessage::Move { pos, yaw, pitch, stance } => {
                w.u8(tag::MOVE);
                w.vec3(*pos);
                w.f32(*yaw);
                w.f32(*pitch);
                w.u8(stance.wire());
            }
            ClientMessage::Teleport { pos } => {
                w.u8(tag::TELEPORT);
                w.vec3(*pos);
            }
            ClientMessage::Swing => w.u8(tag::SWING),
            ClientMessage::Ping { nonce } => {
                w.u8(tag::PING);
                w.u32(*nonce);
            }
            ClientMessage::Edit { req, x, y, z, expect, spec } => {
                w.u8(tag::EDIT);
                w.u32(*req);
                w.i32(*x);
                w.i32(*y);
                w.i32(*z);
                w.u32(*expect);
                w.str16(spec);
            }
            ClientMessage::Chat { channel, text } => {
                w.u8(tag::CHAT);
                w.u8(*channel);
                w.str16(text);
            }
            ClientMessage::SetTime { day } => {
                w.u8(tag::SET_TIME);
                w.f32(*day);
            }
            ClientMessage::Voice { seq, payload } => {
                w.u8(tag::VOICE);
                w.u32(*seq);
                write_voice_payload(&mut w, payload);
            }
        }
        w.into_inner()
    }

    /// Parse a frame payload. `None` on any malformed or truncated input.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = codec::Reader::new(bytes);
        let message = match r.u8().ok()? {
            tag::HELLO => ClientMessage::Hello {
                protocol: r.u32().ok()?,
                fingerprint: r.u64().ok()?,
                name: r.str16_lossy().ok()?,
                password: r.str16_lossy().ok()?,
            },
            tag::MOVE => ClientMessage::Move {
                pos: r.vec3().ok()?,
                yaw: r.f32().ok()?,
                pitch: r.f32().ok()?,
                stance: Stance::from_wire(r.u8().ok()?)?,
            },
            tag::TELEPORT => ClientMessage::Teleport { pos: r.vec3().ok()? },
            tag::SWING => ClientMessage::Swing,
            tag::PING => ClientMessage::Ping { nonce: r.u32().ok()? },
            tag::EDIT => ClientMessage::Edit {
                req: r.u32().ok()?,
                x: r.i32().ok()?,
                y: r.i32().ok()?,
                z: r.i32().ok()?,
                expect: r.u32().ok()?,
                spec: r.str16_lossy().ok()?,
            },
            tag::CHAT => ClientMessage::Chat {
                channel: r.u8().ok()?,
                text: r.str16_lossy().ok()?,
            },
            tag::SET_TIME => ClientMessage::SetTime { day: r.f32().ok()? },
            tag::VOICE => ClientMessage::Voice {
                seq: r.u32().ok()?,
                payload: read_voice_payload(&mut r)?,
            },
            _ => return None,
        };
        r.finished().then_some(message)
    }
}

impl ServerMessage {
    /// Serialise to a frame payload (tag byte + fields).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = codec::Writer::new();
        match self {
            ServerMessage::Welcome { player_id, seed, spawn } => {
                w.u8(tag::WELCOME);
                w.u32(*player_id);
                w.i64(*seed);
                w.vec3(*spawn);
            }
            ServerMessage::Reject { reason } => {
                w.u8(tag::REJECT);
                w.str16(reason);
            }
            ServerMessage::Snapshot { edits } => {
                w.u8(tag::SNAPSHOT);
                w.u32(edits.len() as u32);
                for (x, y, z, rev, spec) in edits {
                    w.i32(*x);
                    w.i32(*y);
                    w.i32(*z);
                    w.u32(*rev);
                    w.str16(spec);
                }
            }
            ServerMessage::PeerJoined { id, name } => {
                w.u8(tag::PEER_JOINED);
                w.u32(*id);
                w.str16(name);
            }
            ServerMessage::PeerLeft { id } => {
                w.u8(tag::PEER_LEFT);
                w.u32(*id);
            }
            ServerMessage::PeerMove { id, pos, yaw, pitch, stance } => {
                w.u8(tag::PEER_MOVE);
                w.u32(*id);
                w.vec3(*pos);
                w.f32(*yaw);
                w.f32(*pitch);
                w.u8(stance.wire());
            }
            ServerMessage::PeerExited { id } => {
                w.u8(tag::PEER_EXITED);
                w.u32(*id);
            }
            ServerMessage::PeerSwing { id } => {
                w.u8(tag::PEER_SWING);
                w.u32(*id);
            }
            ServerMessage::Pong { nonce } => {
                w.u8(tag::PONG);
                w.u32(*nonce);
            }
            ServerMessage::Edit { x, y, z, rev, spec } => {
                w.u8(tag::S_EDIT);
                w.i32(*x);
                w.i32(*y);
                w.i32(*z);
                w.u32(*rev);
                w.str16(spec);
            }
            ServerMessage::EditAck { req, accepted, rev } => {
                w.u8(tag::EDIT_ACK);
                w.u32(*req);
                w.u8(*accepted as u8);
                w.u32(*rev);
            }
            ServerMessage::Position { pos } => {
                w.u8(tag::POSITION);
                w.vec3(*pos);
            }
            ServerMessage::Chat { from_id, from_name, channel, text } => {
                w.u8(tag::S_CHAT);
                w.u32(*from_id);
                w.str16(from_name);
                w.u8(*channel);
                w.str16(text);
            }
            ServerMessage::Time { day, day_secs } => {
                w.u8(tag::S_TIME);
                w.f32(*day);
                w.f32(*day_secs);
            }
            ServerMessage::PeerVoice { id, epoch, seq, payload } => {
                w.u8(tag::PEER_VOICE);
                w.u32(*id);
                w.u32(*epoch);
                w.u32(*seq);
                write_voice_payload(&mut w, payload);
            }
        }
        w.into_inner()
    }

    /// Parse a frame payload. `None` on any malformed or truncated input.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = codec::Reader::new(bytes);
        let message = match r.u8().ok()? {
            tag::WELCOME => ServerMessage::Welcome {
                player_id: r.u32().ok()?,
                seed: r.i64().ok()?,
                spawn: r.vec3().ok()?,
            },
            tag::REJECT => ServerMessage::Reject { reason: r.str16_lossy().ok()? },
            tag::SNAPSHOT => {
                let count = r.u32().ok()? as usize;
                let mut edits = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    edits.push((r.i32().ok()?, r.i32().ok()?, r.i32().ok()?, r.u32().ok()?, r.str16_lossy().ok()?));
                }
                ServerMessage::Snapshot { edits }
            }
            tag::PEER_JOINED => ServerMessage::PeerJoined {
                id: r.u32().ok()?,
                name: r.str16_lossy().ok()?,
            },
            tag::PEER_LEFT => ServerMessage::PeerLeft { id: r.u32().ok()? },
            tag::PEER_MOVE => ServerMessage::PeerMove {
                id: r.u32().ok()?,
                pos: r.vec3().ok()?,
                yaw: r.f32().ok()?,
                pitch: r.f32().ok()?,
                stance: Stance::from_wire(r.u8().ok()?)?,
            },
            tag::PEER_EXITED => ServerMessage::PeerExited { id: r.u32().ok()? },
            tag::PEER_SWING => ServerMessage::PeerSwing { id: r.u32().ok()? },
            tag::PONG => ServerMessage::Pong { nonce: r.u32().ok()? },
            tag::S_EDIT => ServerMessage::Edit {
                x: r.i32().ok()?,
                y: r.i32().ok()?,
                z: r.i32().ok()?,
                rev: r.u32().ok()?,
                spec: r.str16_lossy().ok()?,
            },
            tag::EDIT_ACK => ServerMessage::EditAck {
                req: r.u32().ok()?,
                accepted: match r.u8().ok()? {
                    0 => false,
                    1 => true,
                    _ => return None,
                },
                rev: r.u32().ok()?,
            },
            tag::POSITION => ServerMessage::Position { pos: r.vec3().ok()? },
            tag::S_CHAT => ServerMessage::Chat {
                from_id: r.u32().ok()?,
                from_name: r.str16_lossy().ok()?,
                channel: r.u8().ok()?,
                text: r.str16_lossy().ok()?,
            },
            tag::S_TIME => ServerMessage::Time { day: r.f32().ok()?, day_secs: r.f32().ok()? },
            tag::PEER_VOICE => ServerMessage::PeerVoice {
                id: r.u32().ok()?,
                epoch: r.u32().ok()?,
                seq: r.u32().ok()?,
                payload: read_voice_payload(&mut r)?,
            },
            _ => return None,
        };
        r.finished().then_some(message)
    }
}

/// Refuses to emit an over-cap frame so both ends share one hard size bound.
pub fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    if payload.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    w.write_all(&(payload.len() as u32).to_be_bytes())?;
    w.write_all(payload)
}

/// `buf` is caller-owned scratch, reused so steady-state traffic never
/// allocates per frame. Rejects a length past [`MAX_FRAME`] before growing
/// the buffer, so a malicious header can't trigger a huge or endless read.
pub fn read_frame<R: Read>(r: &mut R, buf: &mut Vec<u8>) -> io::Result<()> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame exceeds cap"));
    }
    buf.resize(len, 0);
    r.read_exact(buf)
}

/// Async twin of [`write_frame`] over a QUIC send stream. No explicit flush
/// and no `finish` — quinn transmits on its own, and finishing would close
/// the multiplexed stream.
pub async fn write_frame_async(s: &mut SendStream, payload: &[u8]) -> io::Result<()> {
    if payload.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    s.write_all(&(payload.len() as u32).to_be_bytes()).await.map_err(io::Error::other)?;
    s.write_all(payload).await.map_err(io::Error::other)
}

/// Async twin of [`read_frame`] over a QUIC recv stream.
pub async fn read_frame_async(r: &mut RecvStream, buf: &mut Vec<u8>) -> io::Result<()> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes).await.map_err(io::Error::other)?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame exceeds cap"));
    }
    buf.resize(len, 0);
    r.read_exact(buf).await.map_err(io::Error::other)
}

// The frame-length envelope (write_frame/read_frame) stays big-endian,
// independent of the little-endian message-payload codec.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_messages_round_trip() {
        let cases = [
            ClientMessage::Hello {
                protocol: 1,
                fingerprint: 0xDEAD_BEEF_1234_5678,
                name: "player".into(),
                password: "hunter2".into(),
            },
            ClientMessage::Move {
                pos: DVec3::new(1.5, -2.0, 3.25),
                yaw: 0.5,
                pitch: -0.25,
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
            ClientMessage::Voice { seq: 5, payload: vec![1, 2, 3, 4] },
            ClientMessage::Voice { seq: 0, payload: Vec::new() },
        ];
        for msg in cases {
            assert_eq!(ClientMessage::decode(&msg.encode()), Some(msg));
        }
    }

    #[test]
    fn server_messages_round_trip() {
        let cases = [
            ServerMessage::Welcome {
                player_id: 42,
                seed: -9_999,
                spawn: DVec3::new(0.5, 40.0, 0.5),
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
                stance: Stance::Swimming,
            },
            ServerMessage::PeerExited { id: 3 },
            ServerMessage::PeerSwing { id: 3 },
            ServerMessage::Pong { nonce: 7 },
            ServerMessage::Edit { x: 0, y: 0, z: 0, rev: 4, spec: "air".into() },
            ServerMessage::EditAck { req: 12, accepted: true, rev: 4 },
            ServerMessage::EditAck { req: 13, accepted: false, rev: 4 },
            ServerMessage::Position { pos: DVec3::new(-1.0e9, 2.0, 3.0) },
            ServerMessage::Chat {
                from_id: 3,
                from_name: "friend".into(),
                channel: 0,
                text: "hi".into(),
            },
            ServerMessage::Time { day: 0.75, day_secs: 600.0 },
            ServerMessage::PeerVoice { id: 3, epoch: 0, seq: 5, payload: vec![9, 8, 7] },
            ServerMessage::PeerVoice { id: 1, epoch: 2, seq: 0, payload: Vec::new() },
        ];
        for msg in cases {
            assert_eq!(ServerMessage::decode(&msg.encode()), Some(msg));
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
        let mv = ClientMessage::Move { pos, yaw: 1.0, pitch: -0.5, stance: Stance::Standing };
        match ClientMessage::decode(&mv.encode()) {
            Some(ClientMessage::Move { pos: got, .. }) => {
                assert_eq!(got.x.to_bits(), pos.x.to_bits());
                assert_eq!(got.y.to_bits(), pos.y.to_bits());
                assert_eq!(got.z.to_bits(), pos.z.to_bits());
            }
            other => panic!("bad decode: {other:?}"),
        }
        let pm = ServerMessage::PeerMove { id: 7, pos, yaw: 0.0, pitch: 0.0, stance: Stance::Standing };
        assert_eq!(ServerMessage::decode(&pm.encode()), Some(pm));
        let wl = ServerMessage::Welcome { player_id: 1, seed: 3, spawn: pos };
        assert_eq!(ServerMessage::decode(&wl.encode()), Some(wl));
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
    fn trailing_bytes_are_rejected_for_every_message_direction() {
        let client_cases = [
            ClientMessage::Hello {
                protocol: 1,
                fingerprint: 7,
                name: "player".into(),
                password: String::new(),
            },
            ClientMessage::Move {
                pos: DVec3::new(1.0, 2.0, 3.0),
                yaw: 0.25,
                pitch: -0.5,
                stance: Stance::Standing,
            },
            ClientMessage::Teleport { pos: DVec3::new(1.0, 2.0, 3.0) },
            ClientMessage::Swing,
            ClientMessage::Ping { nonce: 9 },
            ClientMessage::Edit { req: 1, x: 1, y: 2, z: 3, expect: 0, spec: "air".into() },
            ClientMessage::Chat { channel: 0, text: "hi".into() },
            ClientMessage::SetTime { day: 0.25 },
            ClientMessage::Voice { seq: 3, payload: vec![7, 7] },
        ];
        for message in client_cases {
            let mut payload = message.encode();
            payload.push(0xa5);
            assert_eq!(ClientMessage::decode(&payload), None, "accepted suffix after {message:?}");
        }

        let server_cases = [
            ServerMessage::Welcome {
                player_id: 1,
                seed: 2,
                spawn: DVec3::new(3.0, 4.0, 5.0),
            },
            ServerMessage::Reject { reason: "no".into() },
            ServerMessage::Snapshot { edits: vec![(1, 2, 3, 1, "air".into())] },
            ServerMessage::PeerJoined { id: 2, name: "peer".into() },
            ServerMessage::PeerLeft { id: 2 },
            ServerMessage::PeerMove {
                id: 2,
                pos: DVec3::new(6.0, 7.0, 8.0),
                yaw: 0.5,
                pitch: -0.25,
                stance: Stance::Sneaking,
            },
            ServerMessage::PeerExited { id: 2 },
            ServerMessage::PeerSwing { id: 2 },
            ServerMessage::Pong { nonce: 9 },
            ServerMessage::Edit { x: 1, y: 2, z: 3, rev: 1, spec: "air".into() },
            ServerMessage::EditAck { req: 4, accepted: false, rev: 0 },
            ServerMessage::Position { pos: DVec3::new(1.0, 2.0, 3.0) },
            ServerMessage::Chat {
                from_id: 2,
                from_name: "peer".into(),
                channel: 0,
                text: "hi".into(),
            },
            ServerMessage::Time { day: 0.5, day_secs: 600.0 },
            ServerMessage::PeerVoice { id: 2, epoch: 1, seq: 4, payload: vec![5, 5] },
        ];
        for message in server_cases {
            let mut payload = message.encode();
            payload.push(0x5a);
            assert_eq!(ServerMessage::decode(&payload), None, "accepted suffix after {message:?}");
        }
    }

    #[test]
    fn voice_payload_at_the_cap_round_trips_both_directions() {
        let payload: Vec<u8> = (0..MAX_VOICE_PAYLOAD).map(|i| i as u8).collect();
        let cm = ClientMessage::Voice { seq: 99, payload: payload.clone() };
        assert_eq!(ClientMessage::decode(&cm.encode()), Some(cm));
        let sm = ServerMessage::PeerVoice { id: 7, epoch: 0, seq: 99, payload };
        assert_eq!(ServerMessage::decode(&sm.encode()), Some(sm));
    }

    /// A frame whose length prefix claims more than [`MAX_VOICE_PAYLOAD`] is
    /// refused by the decoder before the bytes are trusted — the encode path
    /// can't build one (callers guard + `debug_assert`), so the frame is forged
    /// directly, exactly as a hostile peer would.
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
}
