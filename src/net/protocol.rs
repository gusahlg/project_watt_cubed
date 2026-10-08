//! The wire protocol: the message enums the client and server exchange, a
//! tiny hand-rolled binary codec for them, and the length-prefixed framing.
//!
//! Binary on purpose: the hot messages are [`ClientMessage::Move`] and
//! [`ServerMessage::PeerPoses`] at tick rate for every player, so each is a
//! fixed handful of bytes rather than a line of text. Variable data (names,
//! chat, block specs) is length-prefixed and bounded by the caps in the
//! [parent module](super).
//!
//! Positions travel as 3x f64 (24 bytes): the game plays out to ±1e9 blocks,
//! where f32 cannot even represent adjacent positions. Peer poses are the
//! exception: i16 offsets from the recipient's own position.
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

use super::{MAX_FRAME, MAX_MOD_BYTES, MAX_SPEC};

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

/// A mod channel name: 1..=16 bytes of UTF-8. The length prefix is a `u8`.
/// Empty, over-long, and non-UTF-8 names reject the whole message.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Channel(Arc<str>);

impl Channel {
    pub const MAX_LEN: usize = 16;

    /// `None` when `name` is empty or longer than [`Self::MAX_LEN`].
    pub fn parse(name: &str) -> Option<Self> {
        (1..=Self::MAX_LEN).contains(&name.len()).then(|| Self(Arc::from(name)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn share(&self) -> Arc<str> {
        Arc::clone(&self.0)
    }
}

impl Wire for Channel {
    fn put(&self, w: &mut codec::Writer) {
        w.u8(self.0.len() as u8);
        w.raw(self.0.as_bytes());
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let len = r.u8().ok()? as usize;
        if !(1..=Self::MAX_LEN).contains(&len) {
            return None;
        }
        let bytes = r.take(len).ok()?;
        let name = std::str::from_utf8(bytes).ok()?;
        Some(Self(Arc::from(name)))
    }
}

/// Opaque bytes on a mod channel, already known to fit [`MAX_MOD_BYTES`].
/// Empty is legal. The bound is checked once, in `TryFrom` and in [`Wire::get`].
#[derive(Clone, Debug, PartialEq)]
pub struct ModBytes(Vec<u8>);

impl ModBytes {
    #[cfg(test)]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

/// `Err` if `bytes` exceeds [`MAX_MOD_BYTES`].
impl TryFrom<Vec<u8>> for ModBytes {
    type Error = ();
    fn try_from(bytes: Vec<u8>) -> Result<Self, ()> {
        (bytes.len() <= MAX_MOD_BYTES).then_some(Self(bytes)).ok_or(())
    }
}

/// One enabled mod an honest client reports at join: the build's package id
/// and version. A modified client can put anything here. The server still
/// enforces teleport, the speed cap, time permission, edit reach, and the
/// movement envelope; this list is not a security boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModOffer {
    pub id: Arc<str>,
    pub version: Arc<str>,
}

/// How many mod ids one Hello or ModsDenied may carry.
const MAX_MOD_OFFERS: usize = 128;
/// Longest accepted mod id or version, in bytes.
const MAX_MOD_ID: usize = 64;

impl Wire for ModOffer {
    fn put(&self, w: &mut codec::Writer) {
        w.str16(&self.id);
        w.str16(&self.version);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let id = bounded_mod_str(r)?;
        let version = bounded_mod_str(r)?;
        Some(Self { id, version })
    }
}

/// The enabled packages a `Hello` carries: at most [`MAX_MOD_OFFERS`], each id and version
/// 1..=[`MAX_MOD_ID`] bytes, so the encoder never writes a list its decoder refuses. Also how
/// many packages were left out.
pub(crate) fn hello_offers(mods: &[(String, String)]) -> (Vec<ModOffer>, usize) {
    let fits = |text: &str| (1..=MAX_MOD_ID).contains(&text.len());
    let offers: Vec<ModOffer> = mods
        .iter()
        .filter(|(id, version)| fits(id) && fits(version))
        .take(MAX_MOD_OFFERS)
        .map(|(id, version)| ModOffer { id: id.as_str().into(), version: version.as_str().into() })
        .collect();
    let dropped = mods.len() - offers.len();
    (offers, dropped)
}

fn bounded_mod_str(r: &mut codec::Reader) -> Option<Arc<str>> {
    let s = r.str16_lossy().ok()?;
    if s.is_empty() || s.len() > MAX_MOD_ID {
        return None;
    }
    Some(Arc::from(s))
}

impl Wire for Vec<ModOffer> {
    fn put(&self, w: &mut codec::Writer) {
        w.u16(self.len() as u16);
        for offer in self {
            offer.put(w);
        }
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let count = r.u16().ok()? as usize;
        if count > MAX_MOD_OFFERS {
            return None;
        }
        let mut offers = Vec::with_capacity(count);
        for _ in 0..count {
            offers.push(ModOffer::get(r)?);
        }
        Some(offers)
    }
}

impl Wire for Vec<Arc<str>> {
    fn put(&self, w: &mut codec::Writer) {
        w.u16(self.len() as u16);
        for s in self {
            w.str16(s);
        }
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let count = r.u16().ok()? as usize;
        if count > MAX_MOD_OFFERS {
            return None;
        }
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(bounded_mod_str(r)?);
        }
        Some(ids)
    }
}

/// u16 length prefix, then the bytes. Decode rejects a length prefix past the
/// cap before the bytes are trusted.
impl Wire for ModBytes {
    fn put(&self, w: &mut codec::Writer) {
        w.u16(self.0.len() as u16);
        w.raw(&self.0);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let len = r.u16().ok()? as usize;
        if len > MAX_MOD_BYTES {
            return None;
        }
        Some(Self(r.take(len).ok()?.to_vec()))
    }
}

/// One peer's pose in a [`ServerMessage::PeerPoses`] frame. The wire form is
/// quantised: see [`PoseBody`] and [`POSE_STEP`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeerPose {
    pub id: u32,
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub frame: DQuat,
    pub velocity: Vec3,
    pub up: Face,
    pub stance: Stance,
}

/// Positions travel as i16 multiples of this from the frame's origin, so a pose
/// reaches [`POSE_REACH`] blocks from it.
pub(crate) const POSE_STEP: f64 = 1.0 / 128.0;
pub(crate) const POSE_REACH: f64 = i16::MAX as f64 * POSE_STEP;

/// Peer poses relative to `origin`, the recipient's own position.
#[derive(Clone, Debug, PartialEq)]
pub struct Poses {
    pub origin: DVec3,
    pub list: Vec<PeerPose>,
}

const FACE_BITS: u8 = 0b111;
const SNEAKING: u8 = 1 << 3;
const FRAMED: u8 = 1 << 4;
const MOVING: u8 = 1 << 5;
/// The shortest record: a one-byte id, flags, yaw, pitch, and the position.
const POSE_MIN: usize = 1 + 1 + 2 + 2 + 6;
/// The longest body: flags, yaw, pitch, the frame, and the velocity.
const BODY_MAX: usize = 1 + 2 + 2 + 4 + 6;
/// The longest record: a five-byte id, the body, and the position.
pub(crate) const POSE_MAX: usize = 5 + BODY_MAX + 6;
/// Tag, origin, and record count.
pub(crate) const POSES_HEAD: usize = 1 + 24 + 2;

/// A pose without its id and position, in wire form: flags (up face, sneaking,
/// frame present, velocity present), yaw and pitch (16 bits each), the frame
/// as smallest-three (32 bits) unless it is identity, and the velocity as three
/// f16 unless it is zero. Encoded once per move and copied into every
/// recipient's frame. A record is the id (LEB128), this body, then the position.
#[derive(Clone, Copy)]
pub(crate) struct PoseBody {
    bytes: [u8; BODY_MAX],
    len: u8,
}

impl PoseBody {
    pub(crate) fn new(yaw: f32, pitch: f32, frame: DQuat, velocity: Vec3, up: Face, stance: Stance) -> Self {
        let mut body = Self { bytes: [0; BODY_MAX], len: 0 };
        let framed = frame != DQuat::IDENTITY;
        let moving = velocity != Vec3::ZERO;
        let mut flags = up as u8;
        if stance == Stance::Sneaking {
            flags |= SNEAKING;
        }
        if framed {
            flags |= FRAMED;
        }
        if moving {
            flags |= MOVING;
        }
        body.put(&[flags]);
        body.put(&yaw_bits(yaw).to_le_bytes());
        body.put(&pitch_bits(pitch).to_le_bytes());
        if framed {
            body.put(&pack_frame(frame).to_le_bytes());
        }
        if moving {
            for c in [velocity.x, velocity.y, velocity.z] {
                body.put(&f16_bits(c).to_le_bytes());
            }
        }
        body
    }

    fn put(&mut self, b: &[u8]) {
        let at = self.len as usize;
        self.bytes[at..at + b.len()].copy_from_slice(b);
        self.len += b.len() as u8;
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

fn offset_bits(v: f64) -> i16 {
    (v / POSE_STEP).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

fn yaw_bits(yaw: f32) -> u16 {
    use std::f32::consts::TAU;
    ((yaw.rem_euclid(TAU) / TAU * 65536.0).round() as u32) as u16
}

fn yaw_of(bits: u16) -> f32 {
    bits as f32 / 65536.0 * std::f32::consts::TAU
}

fn pitch_bits(pitch: f32) -> i16 {
    ((pitch / std::f32::consts::PI).clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn pitch_of(bits: i16) -> f32 {
    bits as f32 / i16::MAX as f32 * std::f32::consts::PI
}

/// Smallest-three: the index of the largest component, then the other three in
/// 10 bits each over ±1/√2. The largest is made positive (q and -q are one rotation).
fn pack_frame(q: DQuat) -> u32 {
    let c = [q.x, q.y, q.z, q.w];
    let mut big = 0;
    for i in 1..4 {
        if c[i].abs() > c[big].abs() {
            big = i;
        }
    }
    let sign = if c[big] < 0.0 { -1.0 } else { 1.0 };
    let mut bits = big as u32;
    let mut shift = 2;
    for (i, v) in c.iter().enumerate() {
        if i == big {
            continue;
        }
        let unit = (v * sign * std::f64::consts::SQRT_2 + 1.0) * 0.5;
        bits |= ((unit * 1023.0).round().clamp(0.0, 1023.0) as u32) << shift;
        shift += 10;
    }
    bits
}

fn unpack_frame(bits: u32) -> DQuat {
    let big = (bits & 3) as usize;
    let mut c = [0.0f64; 4];
    let mut sum = 0.0;
    let mut shift = 2;
    for (i, slot) in c.iter_mut().enumerate() {
        if i == big {
            continue;
        }
        let v = (((bits >> shift) & 1023) as f64 / 1023.0 * 2.0 - 1.0) / std::f64::consts::SQRT_2;
        *slot = v;
        sum += v * v;
        shift += 10;
    }
    c[big] = (1.0 - sum).max(0.0).sqrt();
    codec::quat_from_f32(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32)
}

/// f32 to IEEE half, rounding to nearest even. Past the half range it saturates
/// at ±65504 rather than becoming infinite.
fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7bff;
    }
    let (half, rem, mid) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1))
    } else {
        (((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000)
    };
    let rounded = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
    if rounded >= 0x7c00 { sign | 0x7bff } else { sign | rounded as u16 }
}

fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    let magnitude = match exp {
        0 => man * (1.0 / 16_777_216.0),
        0x1f => if man == 0.0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    };
    sign * magnitude
}

/// LEB128: seven bits per byte, low bits first.
fn var_into(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push(v as u8 | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn put_var(w: &mut codec::Writer, v: u64) {
    let mut bytes = Vec::with_capacity(10);
    var_into(&mut bytes, v);
    w.raw(&bytes);
}

/// `None` when truncated or past 64 bits.
fn get_var(r: &mut codec::Reader) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = r.u8().ok()?;
        let low = u64::from(b & 0x7f);
        if shift == 63 && low > 1 {
            return None;
        }
        v |= low << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

impl Wire for Poses {
    fn put(&self, w: &mut codec::Writer) {
        w.vec3(self.origin);
        w.u16(self.list.len() as u16);
        for p in &self.list {
            put_var(w, u64::from(p.id));
            w.raw(PoseBody::new(p.yaw, p.pitch, p.frame, p.velocity, p.up, p.stance).as_bytes());
            let d = p.pos - self.origin;
            for c in [d.x, d.y, d.z] {
                w.u16(offset_bits(c) as u16);
            }
        }
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let origin = r.vec3().ok()?;
        let count = r.u16().ok()? as usize;
        let mut list = Vec::with_capacity(count.min(r.remaining() / POSE_MIN));
        for _ in 0..count {
            let id = u32::try_from(get_var(r)?).ok()?;
            let flags = r.u8().ok()?;
            if flags & !(FACE_BITS | SNEAKING | FRAMED | MOVING) != 0 {
                return None;
            }
            let up = Face::from_index(flags & FACE_BITS)?;
            let stance = if flags & SNEAKING != 0 { Stance::Sneaking } else { Stance::Standing };
            let yaw = yaw_of(r.u16().ok()?);
            let pitch = pitch_of(r.u16().ok()? as i16);
            let frame = if flags & FRAMED != 0 { unpack_frame(r.u32().ok()?) } else { DQuat::IDENTITY };
            let velocity = if flags & MOVING != 0 {
                Vec3::new(f16_value(r.u16().ok()?), f16_value(r.u16().ok()?), f16_value(r.u16().ok()?))
            } else {
                Vec3::ZERO
            };
            let mut pos = origin;
            for axis in 0..3 {
                pos[axis] += f64::from(r.u16().ok()? as i16) * POSE_STEP;
            }
            list.push(PeerPose { id, pos, yaw, pitch, frame, velocity, up, stance });
        }
        Some(Self { origin, list })
    }
}

/// Builds one [`ServerMessage::PeerPoses`] payload a record at a time, with the
/// same bytes as its [`Wire`] form. The buffer is reused from frame to frame.
pub(crate) struct PosesWriter {
    buf: Vec<u8>,
    origin: DVec3,
    count: u16,
}

impl PosesWriter {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new(), origin: DVec3::ZERO, count: 0 }
    }

    pub(crate) fn begin(&mut self, origin: DVec3) {
        self.buf.clear();
        self.buf.push(tag::PEER_POSES);
        for c in [origin.x, origin.y, origin.z] {
            self.buf.extend_from_slice(&c.to_le_bytes());
        }
        self.buf.extend_from_slice(&[0, 0]);
        self.origin = origin;
        self.count = 0;
    }

    pub(crate) fn push(&mut self, id: u32, body: &PoseBody, pos: DVec3) {
        var_into(&mut self.buf, u64::from(id));
        self.buf.extend_from_slice(body.as_bytes());
        let d = pos - self.origin;
        for c in [d.x, d.y, d.z] {
            self.buf.extend_from_slice(&offset_bits(c).to_le_bytes());
        }
        self.count += 1;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub(crate) fn frame(&mut self) -> Arc<[u8]> {
        self.buf[POSES_HEAD - 2..POSES_HEAD].copy_from_slice(&self.count.to_le_bytes());
        Arc::from(self.buf.as_slice())
    }

    /// A frame with one pose.
    pub(crate) fn single(origin: DVec3, id: u32, body: &PoseBody, pos: DVec3) -> Arc<[u8]> {
        let mut w = Self::new();
        w.begin(origin);
        w.push(id, body, pos);
        w.frame()
    }
}

/// One snapshot cell: the zigzag delta from the previous cell on each axis, the
/// revision, and the palette index, each LEB128.
fn cell_into(buf: &mut Vec<u8>, last: &mut (i32, i32, i32), at: (i32, i32, i32), rev: u32, index: u32) {
    for (a, b) in [(at.0, last.0), (at.1, last.1), (at.2, last.2)] {
        let d = i64::from(a) - i64::from(b);
        var_into(buf, ((d << 1) ^ (d >> 63)) as u64);
    }
    var_into(buf, u64::from(rev));
    var_into(buf, u64::from(index));
    *last = at;
}

/// Longest cell: three five-byte deltas, a five-byte revision and index.
const CELL_MAX: usize = 5 * 5;
/// Shortest cell: one byte for each of its five fields.
const CELL_MIN: usize = 5;
/// Tag, palette count, and cell count at their longest.
const SNAPSHOT_HEAD: usize = 1 + 5 + 10;

/// The snapshot edit list: a palette of the specs it names (count, then each
/// spec), then the cells (count, then each as [`cell_into`] writes it). Every
/// allocation is bounded by the bytes left, so a forged count cannot balloon memory.
impl Wire for Vec<(i32, i32, i32, u32, Arc<str>)> {
    fn put(&self, w: &mut codec::Writer) {
        let mut names: Vec<&str> = Vec::new();
        let mut cells = Vec::new();
        let mut last = (0, 0, 0);
        for (x, y, z, rev, spec) in self {
            let index = names.iter().position(|n| *n == spec.as_ref()).unwrap_or_else(|| {
                names.push(spec);
                names.len() - 1
            });
            cell_into(&mut cells, &mut last, (*x, *y, *z), *rev, index as u32);
        }
        put_var(w, names.len() as u64);
        for name in names {
            w.str16(name);
        }
        put_var(w, self.len() as u64);
        w.raw(&cells);
    }
    fn get(r: &mut codec::Reader) -> Option<Self> {
        let names = usize::try_from(get_var(r)?).ok()?;
        if names > r.remaining() / 2 {
            return None;
        }
        let mut palette: Vec<Arc<str>> = Vec::with_capacity(names);
        for _ in 0..names {
            let spec = r.str16_lossy().ok()?;
            if spec.len() > MAX_SPEC {
                return None;
            }
            palette.push(spec.into());
        }
        let count = usize::try_from(get_var(r)?).ok()?;
        let mut edits = Vec::with_capacity(count.min(r.remaining() / CELL_MIN));
        let mut at = [0i32; 3];
        for _ in 0..count {
            for c in at.iter_mut() {
                let z = get_var(r)?;
                let d = (z >> 1) as i64 ^ -((z & 1) as i64);
                *c = i32::try_from(i64::from(*c).checked_add(d)?).ok()?;
            }
            let rev = u32::try_from(get_var(r)?).ok()?;
            let spec = palette.get(usize::try_from(get_var(r)?).ok()?)?.clone();
            edits.push((at[0], at[1], at[2], rev, spec));
        }
        Some(edits)
    }
}

/// Builds [`ServerMessage::Snapshot`] payloads from borrowed cells, in the bytes
/// of the [`Wire`] form: each frame stays within [`MAX_FRAME`] and carries its
/// own palette. `key` names a spec (the server's block id), so a cell never
/// compares strings.
pub(crate) struct SnapshotWriter {
    /// Palette index of each key in this frame; `u32::MAX` when absent.
    slots: Vec<u32>,
    used: Vec<u16>,
    palette: Vec<u8>,
    cells: Vec<u8>,
    count: u64,
    last: (i32, i32, i32),
}

impl SnapshotWriter {
    pub(crate) fn new() -> Self {
        Self { slots: Vec::new(), used: Vec::new(), palette: Vec::new(), cells: Vec::new(), count: 0, last: (0, 0, 0) }
    }

    /// Add one cell, first handing `emit` the frame so far when this cell would not fit.
    pub(crate) fn push(&mut self, at: (i32, i32, i32), rev: u32, key: u16, spec: &str, emit: &mut impl FnMut(Arc<[u8]>)) {
        let k = usize::from(key);
        if k >= self.slots.len() {
            self.slots.resize(k + 1, u32::MAX);
        }
        let novel = self.slots[k] == u32::MAX;
        let grow = CELL_MAX + if novel { 2 + spec.len() } else { 0 };
        if self.count > 0 && SNAPSHOT_HEAD + self.palette.len() + self.cells.len() + grow > MAX_FRAME {
            self.finish(emit);
        }
        if self.slots[k] == u32::MAX {
            self.slots[k] = self.used.len() as u32;
            self.used.push(key);
            let len = spec.len().min(u16::MAX as usize);
            self.palette.extend_from_slice(&(len as u16).to_le_bytes());
            self.palette.extend_from_slice(&spec.as_bytes()[..len]);
        }
        cell_into(&mut self.cells, &mut self.last, at, rev, self.slots[k]);
        self.count += 1;
    }

    /// Hand `emit` the frame being built, if it holds a cell, and start the next.
    pub(crate) fn finish(&mut self, emit: &mut impl FnMut(Arc<[u8]>)) {
        if self.count == 0 {
            return;
        }
        let mut frame = Vec::with_capacity(SNAPSHOT_HEAD + self.palette.len() + self.cells.len());
        frame.push(tag::SNAPSHOT);
        var_into(&mut frame, self.used.len() as u64);
        frame.extend_from_slice(&self.palette);
        var_into(&mut frame, self.count);
        frame.extend_from_slice(&self.cells);
        emit(frame.into());
        for key in self.used.drain(..) {
            self.slots[usize::from(key)] = u32::MAX;
        }
        self.palette.clear();
        self.cells.clear();
        self.count = 0;
        self.last = (0, 0, 0);
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
    pub const MOD_DATA: u8 = 8;
    pub const TOOL_USE: u8 = 9;
    pub const CRUISE: u8 = 10;

    pub const WELCOME: u8 = 0;
    pub const REJECT: u8 = 1;
    pub const SNAPSHOT: u8 = 2;
    pub const PEER_JOINED: u8 = 3;
    pub const PEER_LEFT: u8 = 4;
    pub const PEER_POSES: u8 = 5;
    pub const S_EDIT: u8 = 6;
    pub const S_CHAT: u8 = 7;
    pub const S_TIME: u8 = 8;
    pub const PEER_SWING: u8 = 9;
    pub const PONG: u8 = 10;
    pub const EDIT_ACK: u8 = 11;
    pub const POSITION: u8 = 12;
    pub const PEER_EXITED: u8 = 13;
    pub const PEER_MOD_DATA: u8 = 14;
    pub const TOOL_RESULT: u8 = 15;
    pub const MODS_DENIED: u8 = 16;
    pub const SNAPSHOT_END: u8 = 17;
}

messages! {
    /// A message from a client to the server.
    pub enum ClientMessage {
        /// Content parts only: generator version, gravity, material law, palette.
        /// Worldgen kind and terrain knobs arrive in [`ServerMessage::Welcome`].
        /// `protocol` stays the first field so a peer can be told "server vX, client vY"
        /// before the rest of the payload is decoded. `mods` follows the password:
        /// enabled package ids and versions. That list is the client's own report.
        Hello = tag::HELLO {
            protocol: u32,
            worldgen: u32,
            gravity: u64,
            law: u64,
            palette: u64,
            name: Arc<str>,
            password: Arc<str>,
            mods: Vec<ModOffer>,
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
        /// Bytes on a named channel. `seq` orders the sender's own stream. The server
        /// stamps the sender on relay; the client never names itself. `bytes` is
        /// bounded by [`MAX_MOD_BYTES`](super::MAX_MOD_BYTES). Rides the reliable stream.
        ModData = tag::MOD_DATA { channel: Channel, seq: u32, bytes: ModBytes },
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
        /// The join overlay, in frames after [`Welcome`](Self::Welcome), and each
        /// reaction batch. Each cell carries its authoritative revision so the
        /// joiner's future edit expectations line up.
        Snapshot = tag::SNAPSHOT { edits: Vec<(i32, i32, i32, u32, Arc<str>)> },
        /// The join overlay is complete. Later [`Snapshot`](Self::Snapshot) frames are
        /// reaction batches and do not reopen the loading screen.
        SnapshotEnd = tag::SNAPSHOT_END,
        /// Roster only — a peer's pose arrives via [`PeerPoses`](Self::PeerPoses) once
        /// they are inside interest range.
        PeerJoined = tag::PEER_JOINED { id: u32, name: Arc<str> },
        PeerLeft = tag::PEER_LEFT { id: u32 },
        /// The poses of visible peers that moved, one frame per recipient per tick.
        /// A pose is also the "entered interest range" signal.
        PeerPoses = tag::PEER_POSES { poses: Poses },
        /// A peer left interest range: hide their avatar instead of drawing a
        /// frozen ghost at the last heard pose. They re-appear with their next
        /// [`PeerPoses`](Self::PeerPoses) record.
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
        /// A [`ClientMessage::ModData`] relayed to the sender's visible interest set.
        /// `sender` is stamped by the server from the authenticated player id.
        /// `seq` and `bytes` are the sender's own values, unchanged.
        PeerModData = tag::PEER_MOD_DATA { channel: Channel, sender: u32, seq: u32, bytes: ModBytes },
        /// The outcome of the sender's [`ClientMessage::ToolUse`] `req`: whether the law moved
        /// anything, the cell's committed revision, and both configurations afterwards (unchanged
        /// specs when nothing reacted or the request was refused).
        ToolResult = tag::TOOL_RESULT { req: u32, reacted: bool, rev: u32, cell_spec: Arc<str>, tool_spec: Arc<str> },
        /// The stream closes after this. `ids` are the enabled mods this server
        /// refuses. An honest client disables them for the session and joins once more.
        ModsDenied = tag::MODS_DENIED { ids: Vec<Arc<str>> },
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

/// A [`ServerMessage::PeerSwing`] or [`ServerMessage::PeerPoses`] payload. Cosmetic: a
/// joining backlog may drop it, since a peer entering range sends both poses afresh.
pub(crate) fn is_cosmetic(frame: &[u8]) -> bool {
    matches!(frame.first().copied(), Some(tag::PEER_SWING | tag::PEER_POSES))
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

/// Append `payload` with its length header, so many frames go out in one write.
pub(crate) fn put_frame(buf: &mut Vec<u8>, payload: &[u8]) -> io::Result<()> {
    buf.extend_from_slice(&frame_header(payload)?);
    buf.extend_from_slice(payload);
    Ok(())
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
                mods: vec![ModOffer { id: "pwc.hotbar".into(), version: "0.1.0".into() }],
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
            ClientMessage::ModData {
                channel: Channel::parse("voice").unwrap(),
                seq: 5,
                bytes: vec![1, 2, 3, 4].try_into().unwrap(),
            },
            ClientMessage::ModData {
                channel: Channel::parse("voice").unwrap(),
                seq: 0,
                bytes: Vec::new().try_into().unwrap(),
            },
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
            ServerMessage::PeerPoses {
                poses: Poses {
                    origin: DVec3::new(1.0e8, -40.0, 3.5),
                    list: vec![
                        PeerPose {
                            id: 3,
                            pos: DVec3::new(1.0e8 + 9.5, -32.25, 10.0),
                            yaw: std::f32::consts::PI,
                            pitch: 0.0,
                            frame: DQuat::IDENTITY,
                            velocity: Vec3::new(0.25, 0.0, -1.5),
                            up: Face::NegY,
                            stance: Stance::Sneaking,
                        },
                        PeerPose {
                            id: 300,
                            pos: DVec3::new(1.0e8 - 100.0, -40.0, 3.5),
                            yaw: 0.0,
                            pitch: 0.0,
                            frame: DQuat::IDENTITY,
                            velocity: Vec3::ZERO,
                            up: Face::PosY,
                            stance: Stance::Standing,
                        },
                    ],
                },
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
            ServerMessage::PeerModData {
                channel: Channel::parse("voice").unwrap(),
                sender: 3,
                seq: 5,
                bytes: vec![9, 8, 7].try_into().unwrap(),
            },
            ServerMessage::PeerModData {
                channel: Channel::parse("voice").unwrap(),
                sender: 1,
                seq: 0,
                bytes: Vec::new().try_into().unwrap(),
            },
            ServerMessage::ToolResult {
                req: 3,
                reacted: true,
                rev: 7,
                cell_spec: "air".into(),
                tool_spec: "c:0201020304aabbccdd".into(),
            },
            ServerMessage::ModsDenied { ids: vec!["pwc.dev-toolkit".into()] },
            ServerMessage::SnapshotEnd,
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
    fn mod_bytes_at_the_cap_round_trip_both_directions() {
        let bytes: ModBytes = (0..MAX_MOD_BYTES).map(|i| i as u8).collect::<Vec<u8>>().try_into().unwrap();
        let channel = Channel::parse("voice").unwrap();
        let cm = ClientMessage::ModData { channel: channel.clone(), seq: 99, bytes: bytes.clone() };
        assert_eq!(ClientMessage::decode(&cm.encode()), Some(cm));
        let sm = ServerMessage::PeerModData { channel, sender: 7, seq: 99, bytes };
        assert_eq!(ServerMessage::decode(&sm.encode()), Some(sm));
    }

    /// A frame whose length prefix claims more than [`MAX_MOD_BYTES`] is refused
    /// before the bytes are trusted. The encode path can't build one
    /// (`ModBytes::try_from` refuses it), so the frame is forged, channel prefix included.
    #[test]
    fn oversized_mod_frame_is_rejected_both_directions() {
        let over = vec![0u8; MAX_MOD_BYTES + 1];

        let mut w = codec::Writer::new();
        w.u8(super::tag::MOD_DATA);
        w.u8(5);
        w.raw(b"voice");
        w.u32(1);
        w.u16(over.len() as u16);
        w.raw(&over);
        assert_eq!(ClientMessage::decode(&w.into_inner()), None);

        let mut w = codec::Writer::new();
        w.u8(super::tag::PEER_MOD_DATA);
        w.u8(5);
        w.raw(b"voice");
        w.u32(2); // sender
        w.u32(1); // seq
        w.u16(over.len() as u16);
        w.raw(&over);
        assert_eq!(ServerMessage::decode(&w.into_inner()), None);
    }

    #[test]
    fn a_bad_channel_name_rejects_the_message() {
        fn forged(tag: u8, len: u8, name: &[u8]) -> Vec<u8> {
            let mut w = codec::Writer::new();
            w.u8(tag);
            w.u8(len);
            w.raw(name);
            w.u32(1);
            w.u16(0);
            w.into_inner()
        }
        assert_eq!(ClientMessage::decode(&forged(super::tag::MOD_DATA, 0, b"")), None);
        assert_eq!(ClientMessage::decode(&forged(super::tag::MOD_DATA, 17, &[b'a'; 17])), None);
        assert_eq!(ClientMessage::decode(&forged(super::tag::MOD_DATA, 1, &[0xff])), None);
        assert_eq!(ServerMessage::decode(&forged(super::tag::PEER_MOD_DATA, 0, b"")), None);
        assert!(Channel::parse("").is_none());
        assert!(Channel::parse(&"a".repeat(17)).is_none());
        assert!(Channel::parse("voice").is_some());
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

    /// Frames split at the cap, each with its own palette, and decode back to the
    /// cells in the order they were pushed.
    #[test]
    fn snapshot_frames_stay_within_the_cap_and_keep_their_order() {
        let mut writer = SnapshotWriter::new();
        let mut frames: Vec<Arc<[u8]>> = Vec::new();
        let mut emit = |frame| frames.push(frame);
        let spec = |key: u16| format!("{key:0>width$}", width = MAX_SPEC);
        let cell = |i: i32| (i * 7 - 50_000, (i % 40) - 20, 1_100_000_000 + i / 3);
        for i in 0..30_000 {
            let key = (i % 300) as u16;
            writer.push(cell(i), i as u32, key, &spec(key), &mut emit);
        }
        writer.finish(&mut emit);
        writer.finish(&mut emit);
        assert!(frames.len() > 1);
        let mut i = 0;
        for frame in &frames {
            assert!(frame.len() <= MAX_FRAME, "{}", frame.len());
            let Some(ServerMessage::Snapshot { edits }) = ServerMessage::decode(frame) else {
                panic!("a snapshot frame must decode");
            };
            for (x, y, z, rev, s) in edits {
                assert_eq!(((x, y, z), rev, s.as_ref()), (cell(i), i as u32, spec((i % 300) as u16).as_str()));
                i += 1;
            }
        }
        assert_eq!(i, 30_000);
    }

    #[test]
    fn the_snapshot_writer_matches_the_wire_form() {
        let edits: Vec<(i32, i32, i32, u32, Arc<str>)> = vec![
            (5, -3, 1_100_000_007, 1, "air".into()),
            (6, -3, 1_100_000_007, 4, "c:0101020304".into()),
            (i32::MIN, i32::MAX, 0, u32::MAX, "air".into()),
        ];
        let mut writer = SnapshotWriter::new();
        let mut frames: Vec<Arc<[u8]>> = Vec::new();
        let mut emit = |frame| frames.push(frame);
        for (x, y, z, rev, spec) in &edits {
            writer.push((*x, *y, *z), *rev, if spec.as_ref() == "air" { 0 } else { 9 }, spec, &mut emit);
        }
        writer.finish(&mut emit);
        assert_eq!(frames.len(), 1);
        assert_eq!(&*frames[0], ServerMessage::Snapshot { edits }.encode().as_slice());
    }

    #[test]
    fn malformed_snapshots_are_rejected_without_a_large_allocation() {
        let frame = |palette: u64, specs: &[&str], count: u64, cells: &[u8]| {
            let mut w = codec::Writer::new();
            w.u8(super::tag::SNAPSHOT);
            put_var(&mut w, palette);
            for s in specs {
                w.str16(s);
            }
            put_var(&mut w, count);
            w.raw(cells);
            ServerMessage::decode(&w.into_inner())
        };
        assert!(frame(1, &["air"], 1, &[0, 0, 0, 1, 0]).is_some());
        assert_eq!(frame(u64::MAX, &[], 0, &[]), None, "a forged palette count");
        assert_eq!(frame(1, &["air"], u64::MAX, &[0, 0, 0, 1, 0]), None, "a forged cell count");
        assert_eq!(frame(1, &["air"], 1, &[0, 0, 0, 1, 1]), None, "an index past the palette");
        assert_eq!(frame(1, &[&"s".repeat(MAX_SPEC + 1)], 0, &[]), None, "a spec past the cap");
        assert_eq!(frame(0, &[], 1, &[0xfe, 0xff, 0xff, 0xff, 0x0f, 0, 0, 1, 0]), None, "a coordinate past i32");
        assert_eq!(frame(1, &["air"], 1, &[0x80; 11]), None, "a varint past 64 bits");
    }

    /// Poses come back within their quantisation: 1/128 block, 16-bit angles,
    /// a smallest-three frame, and f16 velocity.
    #[test]
    fn poses_round_trip_within_their_steps() {
        let origin = DVec3::new(1.1e9 + 0.3, -7.0e5, 42.0);
        let frame = DQuat::from_axis_angle(glam::DVec3::new(1.0, 2.0, -0.5).normalize(), 2.1);
        let pose = PeerPose {
            id: 1_000_000,
            pos: origin + DVec3::new(-180.123, 77.7, 0.004),
            yaw: -2.5,
            pitch: -1.2,
            frame,
            velocity: Vec3::new(3.3, -9.81, 7000.0),
            up: Face::PosZ,
            stance: Stance::Sneaking,
        };
        let msg = ServerMessage::PeerPoses { poses: Poses { origin, list: vec![pose] } };
        let Some(ServerMessage::PeerPoses { poses }) = ServerMessage::decode(&msg.encode()) else {
            panic!("poses must decode");
        };
        let got = poses.list[0];
        assert_eq!((got.id, got.up, got.stance), (pose.id, pose.up, pose.stance));
        assert!(got.pos.distance(pose.pos) <= POSE_STEP, "{:?}", got.pos - pose.pos);
        let turn = (got.yaw - pose.yaw).rem_euclid(std::f32::consts::TAU);
        assert!(turn.min(std::f32::consts::TAU - turn) < 1e-4, "yaw {}", got.yaw);
        assert!((got.pitch - pose.pitch).abs() < 1e-4);
        assert!(got.frame.dot(frame).abs() > 0.9999, "{:?}", got.frame);
        for (a, b) in [(got.velocity.x, 3.3), (got.velocity.y, -9.81), (got.velocity.z, 7000.0f32)] {
            assert!((a - b).abs() <= b.abs() / 1024.0, "{a} vs {b}");
        }
        assert_eq!(f16_value(f16_bits(1.0e9)), 65504.0, "past the half range saturates");
        assert_eq!(f16_value(f16_bits(-0.0)).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_value(f16_bits(6.0e-8)), 5.960_464_5e-8, "the smallest subnormal");
    }

    #[test]
    fn the_pose_writer_matches_the_wire_form() {
        let origin = DVec3::new(10.0, 20.0, 30.0);
        let list = vec![
            PeerPose {
                id: 5,
                pos: DVec3::new(12.0, 20.5, 29.0),
                yaw: 1.0,
                pitch: 0.5,
                frame: DQuat::from_rotation_x(0.3),
                velocity: Vec3::new(1.0, 0.0, 0.0),
                up: Face::PosX,
                stance: Stance::Standing,
            },
            PeerPose {
                id: 200,
                pos: DVec3::new(-100.0, 20.0, 30.0),
                yaw: 0.0,
                pitch: 0.0,
                frame: DQuat::IDENTITY,
                velocity: Vec3::ZERO,
                up: Face::NegX,
                stance: Stance::Sneaking,
            },
        ];
        let mut w = PosesWriter::new();
        w.begin(origin);
        assert!(w.is_empty());
        for p in &list {
            w.push(p.id, &PoseBody::new(p.yaw, p.pitch, p.frame, p.velocity, p.up, p.stance), p.pos);
        }
        let expected = ServerMessage::PeerPoses { poses: Poses { origin, list } }.encode();
        assert_eq!(&*w.frame(), expected.as_slice());
    }

    #[test]
    fn malformed_poses_are_rejected_without_a_large_allocation() {
        let mut w = codec::Writer::new();
        w.u8(super::tag::PEER_POSES);
        w.vec3(DVec3::ZERO);
        w.u16(u16::MAX);
        w.u8(1);
        w.u8(0b1100_0000);
        assert_eq!(ServerMessage::decode(&w.into_inner()), None, "reserved flag bits");
        let mut w = codec::Writer::new();
        w.u8(super::tag::PEER_POSES);
        w.vec3(DVec3::ZERO);
        w.u16(1);
        w.u8(1);
        w.u8(6);
        w.raw(&[0; 10]);
        assert_eq!(ServerMessage::decode(&w.into_inner()), None, "an unknown face");
        let mut w = codec::Writer::new();
        w.u8(super::tag::PEER_POSES);
        w.vec3(DVec3::ZERO);
        w.u16(1);
        w.raw(&[0xff, 0xff, 0xff, 0xff, 0x7f]);
        w.raw(&[0; 11]);
        assert_eq!(ServerMessage::decode(&w.into_inner()), None, "an id past 32 bits");
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
            mods: vec![],
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
            mods: vec![],
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

    /// The packages a client reports always decode on the server: over-long ones and those past
    /// the count are left out and counted, and ones at the limits stay.
    #[test]
    fn hello_offers_stay_within_the_decoder_limits() {
        let mut mods: Vec<(String, String)> = (0..=MAX_MOD_OFFERS).map(|i| (format!("pkg.{i}"), "1.0.0".into())).collect();
        mods.insert(0, ("pkg.long".into(), "1".repeat(MAX_MOD_ID + 1)));
        mods.insert(1, ("x".repeat(MAX_MOD_ID), "v".repeat(MAX_MOD_ID)));
        let hello = |mods: Vec<ModOffer>| ClientMessage::Hello {
            protocol: 1,
            worldgen: 0,
            gravity: 0,
            law: 0,
            palette: 0,
            name: "ada".into(),
            password: "".into(),
            mods,
        };
        let raw = mods.iter().map(|(id, version)| ModOffer { id: id.as_str().into(), version: version.as_str().into() }).collect();
        assert_eq!(ClientMessage::decode(&hello(raw).encode()), None, "the unfitted list is refused");
        let (offers, dropped) = hello_offers(&mods);
        assert_eq!((offers.len(), dropped), (MAX_MOD_OFFERS, mods.len() - MAX_MOD_OFFERS));
        assert_eq!(offers[0].id.len(), MAX_MOD_ID, "an id and version at the limit are kept");
        let sent = hello(offers);
        assert_eq!(ClientMessage::decode(&sent.encode()), Some(sent));
    }
}
