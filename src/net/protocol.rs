//! The wire protocol: the two message enums the client and server exchange, a tiny
//! hand-rolled binary codec for them, and the length-prefixed framing that carries
//! them over a TCP stream.
//!
//! Binary and hand-written on purpose (the "optimisation ahead of readability"
//! mandate, and zero dependencies): the hot message is [`ClientMessage::Move`] /
//! [`ServerMessage::PeerMove`] at tick rate for every player, so each is a fixed
//! handful of bytes rather than a line of text. Variable data (names, chat, block
//! specs) is length-prefixed and bounded by the caps in the [parent module](super).
//!
//! Positions travel as 3x f64 (24 bytes) since protocol v2: the game plays out
//! to ±1e9 blocks, where f32 cannot even represent adjacent positions.
use std::io::{self, Read, Write};

use voxel_engine::DVec3;

use crate::presence::Stance;

use super::MAX_FRAME;

/// A message from a client to the server.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientMessage {
    /// First frame after connecting: identify and authenticate.
    Hello { protocol: u32, name: String, password: String },
    /// The client's own player state this tick (client simulates its own player).
    Move { pos: DVec3, yaw: f32, pitch: f32, stance: Stance },
    /// Visual-only event broadcast to other players.
    Swing,
    /// Latency probe; the server echoes `nonce` back in [`ServerMessage::Pong`].
    Ping { nonce: u32 },
    /// The client changed a block, described by portable spec (see [`save`](crate::save)).
    Edit { x: i32, y: i32, z: i32, spec: String },
    /// A chat line on the given [`channel`](super::chat).
    Chat { channel: u8, text: String },
    /// The player set the world time (via `/time`); `day` is a `[0,1)` fraction.
    SetTime { day: f32 },
}

/// A message from the server to a client.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerMessage {
    /// Join accepted: the assigned id, the world seed to generate from, and where
    /// to spawn.
    Welcome { player_id: u32, seed: i64, spawn: DVec3 },
    /// Join refused (bad password, version mismatch, server full); the stream closes.
    Reject { reason: String },
    /// The full current edit overlay, sent once right after [`Welcome`](Self::Welcome).
    Snapshot { edits: Vec<(i32, i32, i32, String)> },
    /// Another player joined.
    PeerJoined { id: u32, name: String },
    /// Another player disconnected.
    PeerLeft { id: u32 },
    /// Another player moved.
    PeerMove { id: u32, pos: DVec3, yaw: f32, pitch: f32, stance: Stance },
    /// Another player swung their arm.
    PeerSwing { id: u32 },
    /// Echo of a [`ClientMessage::Ping`], carrying its `nonce` unchanged.
    Pong { nonce: u32 },
    /// A block changed somewhere in the world (from a peer or the server).
    Edit { x: i32, y: i32, z: i32, spec: String },
    /// A chat line to display.
    Chat { from_id: u32, from_name: String, channel: u8, text: String },
    /// The shared world time changed (a peer's `/time`, or the current value sent
    /// to a joiner); `day` is a `[0,1)` fraction.
    Time { day: f32 },
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
}

impl ClientMessage {
    /// Serialise to a frame payload (tag byte + fields).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            ClientMessage::Hello { protocol, name, password } => {
                w.u8(tag::HELLO);
                w.u32(*protocol);
                w.str(name);
                w.str(password);
            }
            ClientMessage::Move { pos, yaw, pitch, stance } => {
                w.u8(tag::MOVE);
                w.vec3(*pos);
                w.f32(*yaw);
                w.f32(*pitch);
                w.u8(stance.wire());
            }
            ClientMessage::Swing => w.u8(tag::SWING),
            ClientMessage::Ping { nonce } => {
                w.u8(tag::PING);
                w.u32(*nonce);
            }
            ClientMessage::Edit { x, y, z, spec } => {
                w.u8(tag::EDIT);
                w.i32(*x);
                w.i32(*y);
                w.i32(*z);
                w.str(spec);
            }
            ClientMessage::Chat { channel, text } => {
                w.u8(tag::CHAT);
                w.u8(*channel);
                w.str(text);
            }
            ClientMessage::SetTime { day } => {
                w.u8(tag::SET_TIME);
                w.f32(*day);
            }
        }
        w.into_inner()
    }

    /// Parse a frame payload. `None` on any malformed or truncated input.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        let message = match r.u8()? {
            tag::HELLO => ClientMessage::Hello {
                protocol: r.u32()?,
                name: r.str()?,
                password: r.str()?,
            },
            tag::MOVE => ClientMessage::Move {
                pos: r.vec3()?,
                yaw: r.f32()?,
                pitch: r.f32()?,
                stance: Stance::from_wire(r.u8()?)?,
            },
            tag::SWING => ClientMessage::Swing,
            tag::PING => ClientMessage::Ping { nonce: r.u32()? },
            tag::EDIT => ClientMessage::Edit {
                x: r.i32()?,
                y: r.i32()?,
                z: r.i32()?,
                spec: r.str()?,
            },
            tag::CHAT => ClientMessage::Chat {
                channel: r.u8()?,
                text: r.str()?,
            },
            tag::SET_TIME => ClientMessage::SetTime { day: r.f32()? },
            _ => return None,
        };
        r.finished().then_some(message)
    }
}

impl ServerMessage {
    /// Serialise to a frame payload (tag byte + fields).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            ServerMessage::Welcome { player_id, seed, spawn } => {
                w.u8(tag::WELCOME);
                w.u32(*player_id);
                w.i64(*seed);
                w.vec3(*spawn);
            }
            ServerMessage::Reject { reason } => {
                w.u8(tag::REJECT);
                w.str(reason);
            }
            ServerMessage::Snapshot { edits } => {
                w.u8(tag::SNAPSHOT);
                w.u32(edits.len() as u32);
                for (x, y, z, spec) in edits {
                    w.i32(*x);
                    w.i32(*y);
                    w.i32(*z);
                    w.str(spec);
                }
            }
            ServerMessage::PeerJoined { id, name } => {
                w.u8(tag::PEER_JOINED);
                w.u32(*id);
                w.str(name);
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
            ServerMessage::PeerSwing { id } => {
                w.u8(tag::PEER_SWING);
                w.u32(*id);
            }
            ServerMessage::Pong { nonce } => {
                w.u8(tag::PONG);
                w.u32(*nonce);
            }
            ServerMessage::Edit { x, y, z, spec } => {
                w.u8(tag::S_EDIT);
                w.i32(*x);
                w.i32(*y);
                w.i32(*z);
                w.str(spec);
            }
            ServerMessage::Chat { from_id, from_name, channel, text } => {
                w.u8(tag::S_CHAT);
                w.u32(*from_id);
                w.str(from_name);
                w.u8(*channel);
                w.str(text);
            }
            ServerMessage::Time { day } => {
                w.u8(tag::S_TIME);
                w.f32(*day);
            }
        }
        w.into_inner()
    }

    /// Parse a frame payload. `None` on any malformed or truncated input.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        let message = match r.u8()? {
            tag::WELCOME => ServerMessage::Welcome {
                player_id: r.u32()?,
                seed: r.i64()?,
                spawn: r.vec3()?,
            },
            tag::REJECT => ServerMessage::Reject { reason: r.str()? },
            tag::SNAPSHOT => {
                let count = r.u32()? as usize;
                let mut edits = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    edits.push((r.i32()?, r.i32()?, r.i32()?, r.str()?));
                }
                ServerMessage::Snapshot { edits }
            }
            tag::PEER_JOINED => ServerMessage::PeerJoined {
                id: r.u32()?,
                name: r.str()?,
            },
            tag::PEER_LEFT => ServerMessage::PeerLeft { id: r.u32()? },
            tag::PEER_MOVE => ServerMessage::PeerMove {
                id: r.u32()?,
                pos: r.vec3()?,
                yaw: r.f32()?,
                pitch: r.f32()?,
                stance: Stance::from_wire(r.u8()?)?,
            },
            tag::PEER_SWING => ServerMessage::PeerSwing { id: r.u32()? },
            tag::PONG => ServerMessage::Pong { nonce: r.u32()? },
            tag::S_EDIT => ServerMessage::Edit {
                x: r.i32()?,
                y: r.i32()?,
                z: r.i32()?,
                spec: r.str()?,
            },
            tag::S_CHAT => ServerMessage::Chat {
                from_id: r.u32()?,
                from_name: r.str()?,
                channel: r.u8()?,
                text: r.str()?,
            },
            tag::S_TIME => ServerMessage::Time { day: r.f32()? },
            _ => return None,
        };
        r.finished().then_some(message)
    }
}

/// Write a length-prefixed frame: a `u32` big-endian length followed by `payload`.
/// Refuses to emit an over-cap frame so both ends share one hard size bound.
/// Deliberately does not flush: on a raw `TcpStream` flush is a no-op anyway, and
/// the server's buffered writer flushes once per drained batch, not per message.
pub fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    if payload.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    w.write_all(&(payload.len() as u32).to_be_bytes())?;
    w.write_all(payload)
}

/// Read one length-prefixed frame into `buf`, a caller-owned scratch buffer that
/// reader loops reuse so steady-state traffic never allocates per frame. Rejects a
/// length past [`MAX_FRAME`] before growing the buffer, so a malicious header can't
/// trigger a huge or endless read. On error `buf`'s contents are unspecified.
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

/// A minimal big-endian byte writer for the codec above.
struct Writer(Vec<u8>);

impl Writer {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn into_inner(self) -> Vec<u8> {
        self.0
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_bits().to_be_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_be_bytes());
    }
    /// A position: 3x f64, 24 bytes — bit-exact at any distance from origin.
    fn vec3(&mut self, v: DVec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }
    /// A `u16`-length-prefixed UTF-8 string. Callers cap lengths before sending;
    /// anything longer than `u16::MAX` is clamped so the prefix stays honest.
    fn str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let len = bytes.len().min(u16::MAX as usize);
        self.0.extend_from_slice(&(len as u16).to_be_bytes());
        self.0.extend_from_slice(&bytes[..len]);
    }
}

/// The reader half: every getter is bounds-checked and returns `None` past the end,
/// so a truncated or hostile frame decodes to `None` instead of panicking.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn finished(&self) -> bool {
        self.pos == self.bytes.len()
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_be_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(u32::from_be_bytes(self.take(4)?.try_into().ok()?)))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(u64::from_be_bytes(self.take(8)?.try_into().ok()?)))
    }
    fn vec3(&mut self) -> Option<DVec3> {
        Some(DVec3::new(self.f64()?, self.f64()?, self.f64()?))
    }
    fn str(&mut self) -> Option<String> {
        let len = u16::from_be_bytes(self.take(2)?.try_into().ok()?) as usize;
        let bytes = self.take(len)?;
        // Lossy so a garbled string can't fail an otherwise valid decode; the caps
        // that bound length are enforced by callers, not here.
        Some(String::from_utf8_lossy(bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_messages_round_trip() {
        let cases = [
            ClientMessage::Hello {
                protocol: 1,
                name: "player".into(),
                password: "hunter2".into(),
            },
            ClientMessage::Move {
                pos: DVec3::new(1.5, -2.0, 3.25),
                yaw: 0.5,
                pitch: -0.25,
                stance: Stance::Sneaking,
            },
            ClientMessage::Swing,
            ClientMessage::Ping { nonce: 7 },
            ClientMessage::Edit { x: -4, y: 7, z: 900, spec: "natural:Stone".into() },
            ClientMessage::Chat { channel: 1, text: "hello world".into() },
            ClientMessage::SetTime { day: 0.5 },
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
                    (1, 2, 3, "air".into()),
                    (-5, 6, -7, "mixture:Soil=70;Clay=30".into()),
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
            ServerMessage::PeerSwing { id: 3 },
            ServerMessage::Pong { nonce: 7 },
            ServerMessage::Edit { x: 0, y: 0, z: 0, spec: "air".into() },
            ServerMessage::Chat {
                from_id: 3,
                from_name: "friend".into(),
                channel: 0,
                text: "hi".into(),
            },
            ServerMessage::Time { day: 0.75 },
        ];
        for msg in cases {
            assert_eq!(ServerMessage::decode(&msg.encode()), Some(msg));
        }
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
        let full = ClientMessage::Edit { x: 1, y: 2, z: 3, spec: "air".into() }.encode();
        // Chop the payload short: the reader must report failure, not panic.
        assert_eq!(ClientMessage::decode(&full[..full.len() - 2]), None);
        assert_eq!(ClientMessage::decode(&[]), None);
    }

    #[test]
    fn trailing_bytes_are_rejected_for_every_message_direction() {
        let client_cases = [
            ClientMessage::Hello {
                protocol: 1,
                name: "player".into(),
                password: String::new(),
            },
            ClientMessage::Move {
                pos: DVec3::new(1.0, 2.0, 3.0),
                yaw: 0.25,
                pitch: -0.5,
                stance: Stance::Standing,
            },
            ClientMessage::Swing,
            ClientMessage::Ping { nonce: 9 },
            ClientMessage::Edit { x: 1, y: 2, z: 3, spec: "air".into() },
            ClientMessage::Chat { channel: 0, text: "hi".into() },
            ClientMessage::SetTime { day: 0.25 },
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
            ServerMessage::Snapshot { edits: vec![(1, 2, 3, "air".into())] },
            ServerMessage::PeerJoined { id: 2, name: "peer".into() },
            ServerMessage::PeerLeft { id: 2 },
            ServerMessage::PeerMove {
                id: 2,
                pos: DVec3::new(6.0, 7.0, 8.0),
                yaw: 0.5,
                pitch: -0.25,
                stance: Stance::Sneaking,
            },
            ServerMessage::PeerSwing { id: 2 },
            ServerMessage::Pong { nonce: 9 },
            ServerMessage::Edit { x: 1, y: 2, z: 3, spec: "air".into() },
            ServerMessage::Chat {
                from_id: 2,
                from_name: "peer".into(),
                channel: 0,
                text: "hi".into(),
            },
            ServerMessage::Time { day: 0.5 },
        ];
        for message in server_cases {
            let mut payload = message.encode();
            payload.push(0x5a);
            assert_eq!(ServerMessage::decode(&payload), None, "accepted suffix after {message:?}");
        }
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
