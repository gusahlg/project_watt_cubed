//! The audio service: gait, the acoustic window, capture and the voice sessions.
//! Mods decide which cue plays and who is heard; this only carries out those decisions.
//!
//! Footsteps stay here. A step is a π crossing of the same stride phase the wire already
//! carries, so the core emits [`GameEvent::Footstep`] and a mod only picks the cue.

use std::sync::Arc;

use glam::DQuat;
use voxel_engine::{DVec3, IVec3};

use crate::block::registry::{BlockId, SoundClass};
use crate::modding::{NoticeLevel, Notices};
use crate::coord::Face;
use crate::modding::ActionSet;
use crate::net::client::Connection;
use crate::presence::STRIDE_FREQ;
use crate::world::World;

use super::acoustics::{AcousticWindow, Listener, Response, WindowFrame};
use super::backend::recording::{Intent, Recorder, RecordingBackend};
use super::capture::{Capture, CaptureConfig};
use super::content::{Catalog, CueId, CueMode, CueSymbols, Loop, OneShot};
use super::frame::{AudioFrame, Emitter, EmitterId, Occurrence, OccurrenceId, MAX_OCCURRENCES};
use super::runtime::{MixChange, SoundSystem};
use super::voice::{Epoch, Seq, SessionKey, VoicePacket, MAX_VOICE_PAYLOAD};

/// Occlusion uses `OCCL_K = 0.08`; past ~16 m gain is at most `e^{-1.3}`.
/// Radius 19 (dim 39, ~59k cells) covers that and is 7.7× cheaper than 38.
const ACOUSTIC_RADIUS: u32 = 19;

/// Squared metres. Below this the listener is treated as still, so an idle
/// frame can skip the mixer without missing a footstep (those need >0.5 m/s).
const LISTENER_STILL_EPS2: f64 = 1e-6;

/// One fact the core cannot derive. Positions are in physical space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameEvent {
    BlockBroken { at: DVec3, block: BlockId, local: bool },
    BlockPlaced { at: DVec3, block: BlockId, local: bool },
    /// A tool reaction on `block` (the cell before the law ran).
    ToolReacted { at: DVec3, block: BlockId },
    /// The local player swung. No cue of its own: a tool or a block event carries the sound.
    Swing { at: DVec3 },
    PeerSwing { at: DVec3, peer: u32 },
    Footstep { at: DVec3, ground: BlockId },
    UiNavigate,
    UiConfirm,
    VoiceTest,
    EnterWorld,
    LeaveWorld,
}

/// A peer as the per-frame audio hook sees them.
#[derive(Clone, Copy, Debug)]
pub struct PeerAudio {
    pub id: u32,
    pub at: DVec3,
    pub feet: DVec3,
    pub visible: bool,
    /// Gait phase (∫ speed dt), the same value the wire carries.
    pub gait: f32,
    pub speed: f32,
}

/// What a block sounds like: the acoustic class and its absorption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSound {
    pub class: SoundClass,
    pub absorption: u8,
}

/// Listener and roster for one frame, menus included. Built on the stack.
pub struct AudioView<'a> {
    pub dt: f32,
    pub pos: DVec3,
    pub peers: &'a [PeerAudio],
    pub in_world: bool,
    /// The inverse of deafen: the saved "hear voice" row.
    pub hear_voice: bool,
    pub actions: ActionSet,
    pub ids: &'a [&'static str],
}

impl AudioView<'_> {
    /// True when the action fired this frame, or is held if it was declared `held`.
    pub fn action(&self, id: &str) -> bool {
        self.ids.iter().enumerate().any(|(i, name)| *name == id && self.actions.contains(i))
    }
}

/// One captured opus frame. The bytes are opaque; the mod does not decode them.
pub struct CapturedFrame {
    pub seq: u32,
    pub bytes: Vec<u8>,
}

/// One frame a mod channel delivered. `sender` is the server-stamped player id.
pub struct ModFrame {
    pub sender: u32,
    pub seq: u32,
    pub bytes: Vec<u8>,
}

/// A cue the service accepted. Only a bench that asked for a log sees these.
#[derive(Clone, Debug, PartialEq)]
pub struct Play {
    pub cue: String,
    pub at: Option<DVec3>,
    pub gain: f32,
}

/// The link a mod uses to send and receive on a named channel. Idle when there is no server.
pub struct ModLink<'a> {
    net: Option<&'a mut Connection>,
}

impl<'a> ModLink<'a> {
    pub(crate) fn new(net: Option<&'a mut Connection>) -> Self {
        Self { net }
    }

    /// No connection. A menu frame and a disabled voice mod both look like this.
    pub fn idle() -> Self {
        Self { net: None }
    }

    pub fn connected(&self) -> bool {
        self.net.as_ref().is_some_and(|net| net.is_alive())
    }

    /// Send `bytes` on `channel`. False when the link is down, the channel name is illegal,
    /// or the payload exceeds the cap. The server stamps the sender.
    pub fn send(&mut self, channel: &str, seq: u32, bytes: &[u8]) -> bool {
        match self.net.as_deref_mut() {
            Some(net) => net.send_channel(channel, seq, bytes),
            None => false,
        }
    }

    /// True when `channel` has a frame waiting. False does not allocate.
    pub fn pending(&self, channel: &str) -> bool {
        self.net.as_ref().is_some_and(|net| net.channel_pending(channel))
    }

    /// Append waiting frames. A channel with nothing queued leaves `out` untouched
    /// and allocates nothing.
    pub fn drain(&mut self, channel: &str, out: &mut Vec<ModFrame>) {
        let Some(net) = self.net.as_deref_mut() else { return };
        for (sender, seq, bytes) in net.drain_channel(channel) {
            out.push(ModFrame { sender, seq, bytes });
        }
    }
}

/// The local pose the gait integrator reads. Not part of the mod surface.
pub(crate) struct StepPose {
    pub feet: DVec3,
    pub velocity: DVec3,
    pub on_ground: bool,
    pub up: Face,
}

struct Gait {
    phase: f64,
    prev: f64,
}

impl Gait {
    fn new() -> Self {
        Self { phase: 0.0, prev: 0.0 }
    }

    /// Advance by this frame's travel and report whether a π boundary was crossed.
    fn advance(&mut self, speed: f64, dt: f32) -> bool {
        self.phase += speed * dt as f64 * STRIDE_FREQ;
        let crossed = (self.phase / std::f64::consts::PI).floor() != (self.prev / std::f64::consts::PI).floor();
        self.prev = self.phase;
        crossed
    }
}

fn phase_crossed(prev: f32, now: f32) -> bool {
    (now as f64 / std::f64::consts::PI).floor() != (prev as f64 / std::f64::consts::PI).floor()
}

struct WindowCache {
    window: Option<Arc<AcousticWindow>>,
    edit_gen: u64,
    cell: IVec3,
    timer: f32,
}

impl WindowCache {
    fn new() -> Self {
        Self {
            window: None,
            edit_gen: u64::MAX,
            cell: IVec3::new(i32::MIN, i32::MIN, i32::MIN),
            timer: 0.0,
        }
    }

    fn refresh(&mut self, world: &World, pos: DVec3, dt: f32, needed: bool) -> Option<Arc<AcousticWindow>> {
        // An idle frame with nothing to trace must not sample the world.
        if !needed && self.window.is_none() {
            self.timer += dt;
            return None;
        }
        self.timer += dt;
        let at = world.stream_eye(pos);
        let cell = IVec3::new(at.x.floor() as i32, at.y.floor() as i32, at.z.floor() as i32);
        let edit_gen = world.edit_generation();
        let stale = self.window.is_none() || edit_gen != self.edit_gen || cell != self.cell || self.timer >= 0.5;
        if stale && needed {
            let reuse = self.window.take().and_then(|arc| Arc::try_unwrap(arc).ok()).map(AcousticWindow::into_cells);
            self.window = Some(world.capture_acoustic_window_reuse(cell, ACOUSTIC_RADIUS, reuse, window_frame(world, pos, at)));
            self.edit_gen = edit_gen;
            self.cell = cell;
            self.timer = 0.0;
        }
        self.window.clone()
    }
}

struct VoiceWish {
    id: u32,
    at: Option<DVec3>,
    present: bool,
}

struct AmbientBed {
    cue: CueId<Loop>,
    gain: f32,
}

/// World-scoped audio memory that lives on App across frames. Occurrence ids are
/// process-monotone and survive [`Self::enter_world`].
pub struct AudioService {
    gait: Gait,
    /// Previous gait phase per peer. A vec, not a map: the roster is a handful of ids.
    peer_phase: Vec<(u32, f32)>,
    window: WindowCache,
    capture: Option<Capture>,
    /// Set before the device is opened, so a failed open is still observable.
    capture_requested: bool,
    capture_faulted: bool,
    wishes: Vec<VoiceWish>,
    ambient: Option<AmbientBed>,
    journal: Vec<Occurrence>,
    next_occurrence: u64,
    last_commit: Option<DVec3>,
    dirty: bool,
}

impl AudioService {
    pub fn new() -> Self {
        Self {
            gait: Gait::new(),
            peer_phase: Vec::new(),
            window: WindowCache::new(),
            capture: None,
            capture_requested: false,
            capture_faulted: false,
            wishes: Vec::new(),
            ambient: None,
            journal: Vec::new(),
            next_occurrence: 0,
            last_commit: None,
            dirty: false,
        }
    }

    /// Drop world-scoped memory and the microphone. The occurrence counter stays.
    pub fn enter_world(&mut self) {
        self.gait = Gait::new();
        self.peer_phase.clear();
        self.window = WindowCache::new();
        self.capture = None;
        self.capture_requested = false;
        self.capture_faulted = false;
        self.wishes.clear();
        self.ambient = None;
        self.journal.clear();
        self.last_commit = None;
        self.dirty = false;
    }

    /// True when the listener has not moved since the last commit, nothing is
    /// sounding, and the caller reports the frame as idle (no events, no held action).
    /// The first frame is never idle: the gait needs a real pose.
    pub(crate) fn can_skip(&self, sound: &SoundSystem, idle: bool, listener: DVec3) -> bool {
        if !idle || self.dirty || sound.has_live_sources() || sound.ui_pending() {
            return false;
        }
        let Some(last) = self.last_commit else {
            return false;
        };
        (listener - last).length_squared() <= LISTENER_STILL_EPS2
    }

    pub(crate) fn dirty(&self) -> bool {
        self.dirty
    }

    /// Append footstep events. Local speed drops the up-axis component; a peer steps
    /// when their wire phase crosses π and they are moving.
    pub(crate) fn footsteps(
        &mut self,
        pose: StepPose,
        world: &World,
        dt: f32,
        peers: &[PeerAudio],
        ups: &[Face],
        out: &mut Vec<GameEvent>,
    ) {
        let v = pose.velocity;
        let speed = match pose.up.axis() {
            0 => (v.y * v.y + v.z * v.z).sqrt(),
            1 => (v.x * v.x + v.z * v.z).sqrt(),
            _ => (v.x * v.x + v.y * v.y).sqrt(),
        };
        if self.gait.advance(speed, dt) && speed > 0.5 && pose.on_ground {
            out.push(GameEvent::Footstep { at: pose.feet, ground: world.ground_block(pose.feet, pose.up) });
        }
        for (peer, up) in peers.iter().zip(ups.iter().copied()) {
            let prev = self.note_phase(peer.id, peer.gait);
            if let Some(prev) = prev
                && phase_crossed(prev, peer.gait)
                && peer.speed > 0.5
            {
                out.push(GameEvent::Footstep { at: peer.feet, ground: world.ground_block(peer.feet, up) });
            }
        }
        self.peer_phase.retain(|entry| peers.iter().any(|peer| peer.id == entry.0));
    }

    fn note_phase(&mut self, id: u32, phase: f32) -> Option<f32> {
        if let Some(slot) = self.peer_phase.iter_mut().find(|entry| entry.0 == id) {
            let prev = slot.1;
            slot.1 = phase;
            return Some(prev);
        }
        self.peer_phase.push((id, phase));
        None
    }

    fn mint(&mut self) -> OccurrenceId {
        let id = OccurrenceId(self.next_occurrence);
        self.next_occurrence = self.next_occurrence.wrapping_add(1);
        id
    }

    pub(crate) fn api<'a>(
        &'a mut self,
        sound: &'a mut SoundSystem,
        symbols: &'a CueSymbols,
        world: Option<&'a World>,
        notices: Option<&'a mut Notices>,
    ) -> AudioApi<'a> {
        AudioApi { sound, symbols, service: self, world, notices, log: None }
    }

    /// Submit the journal the mods just filled, then surface faults. An empty idle
    /// journal does not sample the acoustic window.
    pub(crate) fn finish(
        &mut self,
        sound: &mut SoundSystem,
        world: &World,
        listener: Listener,
        dt: f32,
        notices: &mut Notices,
    ) {
        let bed = self.ambient.as_ref().map(|bed| Emitter { id: EmitterId(1), cue: bed.cue, at: listener.pos, gain: bed.gain });
        let emitters = bed.as_slice();
        let needed = sound.has_live_sources() || !self.journal.is_empty() || !emitters.is_empty();
        let window = self.window.refresh(world, listener.pos, dt, needed);
        let frame_dt = dt.clamp(1e-4, 0.5);
        if let Ok(frame) = AudioFrame::new(frame_dt, listener, &self.journal, emitters, window) {
            sound.submit(frame);
        }
        self.journal.clear();
        let mut seen: Option<std::collections::HashSet<std::mem::Discriminant<super::Fault>>> = None;
        for fault in sound.drain_faults() {
            let set = seen.get_or_insert_with(std::collections::HashSet::new);
            if set.insert(std::mem::discriminant(&fault)) {
                notices.push(NoticeLevel::Error, format!("* audio: {fault:?}"));
            }
        }
        if let Some(error) = self.capture.as_mut().and_then(|cap| cap.poll_fault()) {
            notices.push(NoticeLevel::Error, format!("* voice capture error: {error}"));
            self.capture = None;
            self.capture_faulted = true;
        }
        self.last_commit = Some(listener.pos);
        self.dirty = false;
    }

    /// A menu hook played UI cues immediately. Drop the dirty bit when nothing was
    /// queued for the next world frame; a pending UI voice still blocks the skip.
    pub(crate) fn settle_menu(&mut self) {
        if self.journal.is_empty() && self.ambient.is_none() {
            self.dirty = false;
        }
    }
}

impl Default for AudioService {
    fn default() -> Self {
        Self::new()
    }
}

/// What a mod may ask the audio service to do. It does not expose the device, the codecs,
/// or the mixer internals.
pub struct AudioApi<'a> {
    sound: &'a mut SoundSystem,
    symbols: &'a CueSymbols,
    service: &'a mut AudioService,
    world: Option<&'a World>,
    notices: Option<&'a mut Notices>,
    log: Option<&'a mut Vec<Play>>,
}

impl AudioApi<'_> {
    pub fn master(&self) -> f32 {
        self.sound.mix().master
    }
    pub fn effects(&self) -> f32 {
        self.sound.mix().effects
    }
    pub fn voice(&self) -> f32 {
        self.sound.mix().voice
    }
    pub fn muted(&self) -> bool {
        self.sound.mix().muted
    }
    pub fn deafened(&self) -> bool {
        self.sound.mix().deafen
    }

    /// Replace the mix until the next settings apply. The saved rows stay the source of
    /// truth: a settings change writes them back over this.
    pub fn set_mix(&mut self, master: f32, effects: f32, voice: f32, muted: bool, deafened: bool) {
        self.sound.set_mix(MixChange { master, effects, voice, muted, deafen: deafened });
    }

    /// The block's acoustic class. Menus (no world) read as open air.
    pub fn block_sound(&self, id: BlockId) -> BlockSound {
        let Some(world) = self.world else {
            return BlockSound { class: SoundClass::Open, absorption: 0 };
        };
        let class = world.registry().sound(id);
        BlockSound { class, absorption: class.absorption() }
    }

    /// Play a world-response one-shot at `at`. False when the cue is missing or the wrong kind.
    pub fn play_at(&mut self, cue: &str, at: DVec3, gain: f32) -> bool {
        let Some(gain) = finite_gain(gain) else { return false };
        if !at.is_finite() || self.service.journal.len() >= MAX_OCCURRENCES {
            return false;
        }
        let Some(id) = self.cue::<OneShot>(cue, Response::World) else { return false };
        let occ = self.service.mint();
        self.service.journal.push(Occurrence { id: occ, cue: id, at: Some(at), gain });
        self.note(cue, Some(at), gain);
        true
    }

    /// Play a UI one-shot now. Menus have no frame journal.
    pub fn play_ui(&mut self, cue: &str, gain: f32) -> bool {
        let Some(gain) = finite_gain(gain) else { return false };
        let Some(id) = self.cue::<OneShot>(cue, Response::Ui) else { return false };
        self.sound.play_ui(id, gain);
        self.note(cue, None, gain);
        true
    }

    /// Remember one ambient bed and resubmit it each frame until replaced or stopped.
    pub fn play_ambient(&mut self, cue: &str, gain: f32) -> bool {
        let Some(gain) = finite_gain(gain) else { return false };
        let Some(id) = self.cue::<Loop>(cue, Response::Ambient) else { return false };
        self.service.ambient = Some(AmbientBed { cue: id, gain });
        self.note(cue, None, gain);
        true
    }

    pub fn stop_ambient(&mut self) {
        if self.service.ambient.take().is_some() {
            self.service.dirty = true;
        }
    }

    /// Remember that `peer` should have a session. The first packet opens it.
    pub fn open_voice(&mut self, peer: u32) {
        self.wish(peer).present = true;
        self.sound.set_session_present(SessionKey(peer), true, None);
    }

    /// Place `peer`. `audible` is the interest bit; the acoustics kernel does distance.
    pub fn position_voice(&mut self, peer: u32, at: DVec3, audible: bool) {
        let wish = self.wish(peer);
        wish.at = Some(at);
        wish.present = audible;
        self.sound.set_session_present(SessionKey(peer), audible, Some(at));
    }

    pub fn close_voice(&mut self, peer: u32) {
        self.service.wishes.retain(|wish| wish.id != peer);
        self.sound.close_session(SessionKey(peer));
    }

    /// Feed one encoded frame. A stored position is applied after the session opens.
    pub fn push_voice(&mut self, peer: u32, seq: u32, bytes: &[u8]) {
        if bytes.is_empty() || bytes.len() > MAX_VOICE_PAYLOAD {
            return;
        }
        let placed = self.service.wishes.iter().find(|wish| wish.id == peer).map(|wish| (wish.present, wish.at));
        self.sound.ingest_voice(VoicePacket {
            session: SessionKey(peer),
            epoch: Epoch(0),
            seq: Seq(seq),
            payload: bytes.to_vec().into_boxed_slice(),
        });
        if let Some((present, at)) = placed {
            self.sound.set_session_present(SessionKey(peer), present, at);
        }
        self.service.dirty = true;
    }

    /// Open the microphone. The request is recorded before the device, so a missing
    /// device still counts as an attempt. A second call just resumes transmit.
    pub fn start_capture(&mut self) {
        self.service.capture_requested = true;
        self.service.dirty = true;
        if self.service.capture.is_some() {
            if let Some(cap) = self.service.capture.as_mut() {
                cap.set_transmitting(true);
            }
            return;
        }
        if self.service.capture_faulted {
            return;
        }
        match Capture::new(CaptureConfig { device: None }) {
            Ok(mut cap) => {
                cap.set_transmitting(true);
                self.service.capture = Some(cap);
            }
            Err(error) => {
                self.service.capture_faulted = true;
                if let Some(notices) = self.notices.as_deref_mut() {
                    notices.push(NoticeLevel::Error, format!("* voice capture unavailable: {error}"));
                }
            }
        }
    }

    pub fn stop_capture(&mut self) {
        self.service.capture_faulted = false;
        if let Some(cap) = self.service.capture.as_mut() {
            cap.set_transmitting(false);
            cap.drain().for_each(drop);
        }
    }

    pub fn capturing(&self) -> bool {
        self.service.capture.is_some()
    }

    /// Append encoded frames. Does nothing — and allocates nothing — when capture is closed.
    pub fn drain_capture(&mut self, out: &mut Vec<CapturedFrame>) {
        let Some(cap) = self.service.capture.as_mut() else { return };
        for frame in cap.drain() {
            out.push(CapturedFrame { seq: frame.seq.0, bytes: frame.payload.into_vec() });
        }
    }

    fn wish(&mut self, peer: u32) -> &mut VoiceWish {
        if let Some(index) = self.service.wishes.iter().position(|wish| wish.id == peer) {
            return &mut self.service.wishes[index];
        }
        self.service.wishes.push(VoiceWish { id: peer, at: None, present: false });
        let last = self.service.wishes.len() - 1;
        &mut self.service.wishes[last]
    }

    fn note(&mut self, cue: &str, at: Option<DVec3>, gain: f32) {
        self.service.dirty = true;
        if let Some(log) = self.log.as_deref_mut() {
            log.push(Play { cue: cue.to_string(), at, gain });
        }
    }

    /// The cue named `name`, when it has mode `M` and plays in `response`.
    fn cue<M: CueMode>(&self, name: &str, response: Response) -> Option<CueId<M>> {
        let catalog = self.sound.catalog();
        let cue = catalog.typed::<M>(self.symbols, name)?;
        (catalog.cue(cue).response == response).then_some(cue)
    }
}

fn finite_gain(gain: f32) -> Option<f32> {
    gain.is_finite().then_some(gain.clamp(0.0, 4.0))
}

/// Headless driver for mod tests: a recording backend plus the cue log.
pub struct AudioBench {
    sound: SoundSystem,
    symbols: CueSymbols,
    service: AudioService,
    recorder: Recorder,
    log: Vec<Play>,
}

const BENCH_CATALOG: &str = r#"
[cues.break_default]
response = "world"
[[cues.break_default.layers]]
variants = ["a"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"

[cues.place_default]
response = "world"
[[cues.place_default.layers]]
variants = ["b"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"

[cues.step_default]
response = "world"
[[cues.step_default.layers]]
variants = ["c"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"

[cues.swing]
response = "world"
[[cues.swing.layers]]
variants = ["d"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"

[cues.menu_click]
response = "ui"
[[cues.menu_click.layers]]
variants = ["e"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"

[cues.voicetest]
response = "ui"
[[cues.voicetest.layers]]
variants = ["f"]
gain = [1.0]
pitch = [0.0]
delay = [0.0]
mode = "one_shot"
"#;

impl AudioBench {
    /// A system whose catalog has the six cues the sounds mod plays, and whose backend
    /// records every clip and stream.
    pub fn recording() -> Self {
        let (mut backend, recorder) = RecordingBackend::new();
        let mut resolve = |_name: &str| -> Result<Vec<u8>, std::path::PathBuf> { Ok(vec![0]) };
        let (catalog, symbols) = Catalog::from_manifest(BENCH_CATALOG, &mut resolve, &mut backend)
            .expect("bench catalog");
        let mut cfg = super::assets::SoundConfig::silent();
        cfg.max_voices = 32;
        let sound = SoundSystem::assemble_recording(Box::new(backend), catalog, cfg);
        Self { sound, symbols, service: AudioService::new(), recorder, log: Vec::new() }
    }

    pub fn api<'a>(&'a mut self, world: Option<&'a World>) -> AudioApi<'a> {
        AudioApi {
            sound: &mut self.sound,
            symbols: &self.symbols,
            service: &mut self.service,
            world,
            notices: None,
            log: Some(&mut self.log),
        }
    }

    pub fn finish(&mut self, world: &World) {
        let listener = Listener { pos: DVec3::ZERO, yaw: 0.0, pitch: 0.0, frame: DQuat::IDENTITY };
        let mut notices = Notices::default();
        self.service.finish(&mut self.sound, world, listener, 1.0 / 60.0, &mut notices);
    }

    pub fn plays(&self) -> &[Play] {
        &self.log
    }

    pub fn clip_count(&self) -> usize {
        self.recorder.intents().iter().filter(|intent| matches!(intent, Intent::PlayClip { .. })).count()
    }

    pub fn stream_count(&self) -> usize {
        self.recorder.intents().iter().filter(|intent| matches!(intent, Intent::PlayStream { .. })).count()
    }

    pub fn capture_requested(&self) -> bool {
        self.service.capture_requested
    }
}

/// The map from physical positions near `pos` into the cells streamed around `at`.
fn window_frame(world: &World, pos: DVec3, at: DVec3) -> WindowFrame {
    if at == pos {
        return WindowFrame::IDENTITY;
    }
    let mut cols = [DVec3::ZERO; 3];
    for (axis, col) in cols.iter_mut().enumerate() {
        let e = DVec3::AXES[axis];
        let fwd = world.stream_eye(pos + e) - at;
        let bwd = at - world.stream_eye(pos - e);
        let (forward, back) = ((fwd.length() - 1.0).abs(), (bwd.length() - 1.0).abs());
        *col = if forward < 0.5 && back < 0.5 {
            (fwd + bwd) * 0.5
        } else if forward <= back {
            fwd
        } else {
            bwd
        };
    }
    WindowFrame { phys_at: pos, cell_at: at, to_cells: glam::DMat3::from_cols(cols[0], cols[1], cols[2]) }
}

#[cfg(test)]
fn sound_class_at_feet(world: &World, feet: DVec3, up: Face) -> &'static str {
    world.registry().sound_class(world.ground_block(feet, up))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::SoundSystem;
    use crate::world::World;

    #[test]
    fn acoustic_radius_matches_occlusion_falloff() {
        assert_eq!(ACOUSTIC_RADIUS, 19);
        assert_eq!(2 * ACOUSTIC_RADIUS + 1, 39);
    }

    #[test]
    fn skip_commit_waits_for_the_first_frame_and_a_still_listener() {
        let (sound, _) = SoundSystem::mute();
        let svc = AudioService::new();
        assert!(!svc.can_skip(&sound, true, DVec3::ZERO));
    }

    #[test]
    fn skip_commit_after_a_still_silent_frame() {
        let (mut sound, symbols) = SoundSystem::mute();
        let mut svc = AudioService::new();
        let world = World::generate();
        let pos = DVec3::new(0.5, 80.0, 0.5);
        let listener = Listener { pos, yaw: 0.0, pitch: 0.0, frame: DQuat::IDENTITY };
        let mut notices = Notices::default();
        {
            let _api = svc.api(&mut sound, &symbols, Some(&world), Some(&mut notices));
        }
        svc.finish(&mut sound, &world, listener, 1.0 / 60.0, &mut notices);
        assert!(svc.can_skip(&sound, true, pos));
        assert!(!svc.can_skip(&sound, true, pos + DVec3::X * 0.01), "a centimetre of travel must re-enable the commit");
        assert!(!svc.can_skip(&sound, false, pos), "a pending event or a held action must re-enable the commit");
    }

    #[test]
    fn the_window_frame_inverts_the_chart_embedding_near_the_listener() {
        use crate::render_config::RenderConfig;
        use crate::world::generation::WorldgenKind;
        let world = World::with_kind(42, RenderConfig::default(), WorldgenKind::Diffusion, false);
        let pos = DVec3::new(0.5, 51.6, 0.5);
        let at = world.stream_eye(pos);
        assert!(at.x > 1.0e9, "the start world is charted: {at}");
        let frame = window_frame(&world, pos, at);
        assert!((frame.map(pos) - at).length() < 1e-9);
        let mut worst = 0.0f64;
        for d in [(3, -2, 1), (-5, -1, 4), (0, -3, 0), (7, 0, -6), (-20, -4, 15), (30, -2, -30)] {
            let cell = (at.x.floor() as i32 + d.0, at.y.floor() as i32 + d.1, at.z.floor() as i32 + d.2);
            let phys = crate::space::atlas::embed_cell(world.atlases(), cell).expect("a storage cell");
            assert!((phys - pos).length() < 60.0, "the block is next to the listener: {phys}");
            let centre = DVec3::new(cell.0 as f64 + 0.5, cell.1 as f64 + 0.5, cell.2 as f64 + 0.5);
            let err = (frame.map(phys) - centre).length() / (phys - pos).length().max(1.0);
            worst = worst.max(err);
        }
        assert!(worst < 0.03, "the frame carries nearby blocks onto their cells: {worst} blocks per block");
        let flat = World::new(1);
        assert_eq!(window_frame(&flat, pos, flat.stream_eye(pos)), WindowFrame::IDENTITY);
    }

    #[test]
    fn footstep_block_follows_the_up_axis_and_a_chart() {
        let mut world = World::with_config_lazy(1, crate::render_config::RenderConfig::default());
        world.ensure_around(DVec3::new(10.5, 5.0, 3.5));
        let rock = world.registry().id_by_label("rock").unwrap();
        let air = crate::block::AIR;
        world.set_block(10, 5, 3, air);
        world.set_block(10, 4, 3, rock);
        let feet = DVec3::new(10.5, 5.0, 3.5);
        assert_eq!(world.ground_block(feet, Face::PosX), air);
        assert_eq!(world.ground_block(feet, Face::PosY), rock);
        assert_eq!(sound_class_at_feet(&world, feet, Face::PosX), "open");
        assert_eq!(sound_class_at_feet(&world, feet, Face::PosY), world.registry().sound_class(rock));

        use crate::space::atlas::{Atlas, Patch};
        let centre = DVec3::new(2.0e7, 3.0e7, -1.0e7);
        let r = 3_000i64;
        let atlas = std::sync::Arc::new(Atlas::new(centre, r, r + 64, false, crate::space::atlas::STORAGE_X0));
        world.set_atlases(vec![atlas.clone()]);
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let b = atlas.bands[0];
        let (k, mid) = (r - b.r_lo - 1, b.n / 2);
        let feet = atlas.embed(top, DVec3::new(mid as f64 + 0.5, k as f64 + 1.05, mid as f64 + 0.5));
        world.ensure_around(feet);
        let s = atlas.storage(top, [mid, k, mid]);
        world.set_block(s[0] as i32, s[1] as i32, s[2] as i32, rock);
        assert_eq!(world.ground_block(feet, Face::PosX), rock);
        assert_eq!(sound_class_at_feet(&world, feet, Face::PosX), world.registry().sound_class(rock));
        let physical = (feet.x.floor() as i32, (feet.y - 0.1).floor() as i32, feet.z.floor() as i32);
        assert_eq!(
            world.registry().sound_class(world.block_at(physical.0, physical.1, physical.2)),
            "open",
            "a world-Y probe is not the storage cell under the feet"
        );
    }

    /// A cue plays only in the response class it was authored for. The old palette rejected a
    /// mode-correct cue whose response did not match the role; the service does the same at play.
    #[test]
    fn a_cue_plays_only_in_its_response_class() {
        let mut bench = AudioBench::recording();
        {
            let mut api = bench.api(None);
            assert!(!api.play_at("menu_click", DVec3::ZERO, 1.0), "a ui cue is not a world cue");
            assert!(!api.play_ui("break_default", 1.0), "a world cue is not a ui cue");
            assert!(!api.play_at("missing", DVec3::ZERO, 1.0));
            assert!(!api.play_at("swing", DVec3::ZERO, f32::NAN));
            assert!(api.play_ui("voicetest", 1.0));
            assert!(api.play_at("swing", DVec3::ZERO, 1.0));
        }
        let names: Vec<_> = bench.plays().iter().map(|play| play.cue.as_str()).collect();
        assert_eq!(names, ["voicetest", "swing"]);
        assert_eq!(bench.clip_count(), 1, "ui plays now; the world cue waits for a frame");
    }
}
