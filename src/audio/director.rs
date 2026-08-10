//! Gameplay reports facts, the director decides sounds. It keeps just enough
//! per-frame state (previous medium, previous speed, previous roster, ...) to
//! derive cues like splash (medium changed), footsteps (speed crossed a
//! threshold), underwater bed (current medium), and session join/leave (roster
//! diff) without gameplay having to push explicit events for them. Only facts
//! that can't be derived this way cross as events (`SoundEvent`).
//!
//! Lives on App beside `SoundSystem`; App feeds it one `AudioCtx` per frame.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::{DVec3, IVec3};

use crate::block::registry::BlockId;
use crate::console::Console;
use crate::math::PER_METER;
use crate::net::client::Connection;
use crate::presence::STRIDE_FREQ;
use crate::world::World;

use super::acoustics::AcousticWindow;
use super::capture::{Capture, CaptureConfig};
use super::content::OneShot;
use super::frame::MAX_OCCURRENCES;
use super::palette::{CuePalette, Sfx, UiSound};
use super::{
    AudioFrame, Emitter, EmitterId, Epoch, Fault, Listener, Medium, Occurrence, OccurrenceId, Seq,
    SessionKey, SoundSystem, VoicePacket,
};

/// Radius of the acoustic window, authored as 32 metres and converted to whole
/// world cells. Round outward so the advertised range is never truncated.
const ACOUSTIC_RADIUS: u32 = (32.0 * PER_METER) as u32 + 1;

/// The unrecoverable facts: everything else the director derives from
/// `AudioCtx`. Closed — its fold is one `match`, no bus/trait indirection.
pub enum SoundEvent {
    BlockBroken { at: DVec3, block: BlockId },
    BlockPlaced { at: DVec3, block: BlockId },
    PeerSwing { at: DVec3 },
    Ui(UiSound),
}

/// The local listener pose, built once per frame from `Player`.
pub struct PlayerPose {
    pub pos: DVec3,
    pub feet: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: DVec3,
    pub on_ground: bool,
}

/// The one per-frame peer sample: built from `Connection::peers().sample(now)`
/// once and fed to both the director and `peer_draws`. `speed` mirrors the
/// local footstep gate; `phase` is the peer's gait phase (already ∫ speed dt on
/// the wire) for crossing detection.
pub struct PeerPose {
    pub id: u32,
    pub at: DVec3,
    pub feet: DVec3,
    pub visible: bool,
    pub phase: f32,
    pub speed: f32,
}

/// Everything the director derives or caches this frame, built by borrowing from
/// the frame inputs; `events` is the one owned/drained field.
pub struct AudioCtx<'a> {
    pub dt: f32,
    pub player: PlayerPose,
    pub ptt: bool,
    pub voice_enabled: bool,
    pub events: Vec<SoundEvent>,
    pub peers: &'a [PeerPose],
    pub world: &'a World,
    pub net: Option<&'a mut Connection>,
    pub console: &'a mut Console,
}

/// The walk-cycle integrator: a footstep fires on each half-cycle (π) phase
/// crossing, so the step count is invariant to dt (crossing, not per-frame sampling).
struct Gait {
    phase: f64,
    prev: f64,
}

impl Gait {
    fn new() -> Self {
        Self {
            phase: 0.0,
            prev: 0.0,
        }
    }

    /// Advance by this frame's travel and report whether a π boundary was crossed.
    fn advance(&mut self, speed: f64, dt: f32) -> bool {
        self.phase += speed * dt as f64 * STRIDE_FREQ;
        let crossed = (self.phase / std::f64::consts::PI).floor()
            != (self.prev / std::f64::consts::PI).floor();
        self.prev = self.phase;
        crossed
    }
}

/// Whether a peer's gait phase crossed a π boundary between the last sample and this.
fn phase_crossed(prev: f32, now: f32) -> bool {
    (now as f64 / std::f64::consts::PI).floor() != (prev as f64 / std::f64::consts::PI).floor()
}

/// The memoized acoustic window: recapture only when the world changed, the
/// listener crossed a cell, or ~500 ms elapsed.
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

    fn refresh(&mut self, world: &World, pos: DVec3, dt: f32) -> Option<Arc<AcousticWindow>> {
        self.timer += dt;
        let cell = IVec3::new(
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        );
        let edit_gen = world.edit_generation();
        let stale = self.window.is_none()
            || edit_gen != self.edit_gen
            || cell != self.cell
            || self.timer >= 0.5;
        if stale {
            self.window = Some(world.capture_acoustic_window(cell, ACOUSTIC_RADIUS));
            self.edit_gen = edit_gen;
            self.cell = cell;
            self.timer = 0.0;
        }
        self.window.clone()
    }
}

/// The mic lifecycle: the device is created lazily on the first transmit, a
/// failure is surfaced once, and its own error stream is polled each frame.
struct CaptureLane {
    capture: Option<Capture>,
    faulted: bool,
}

impl CaptureLane {
    fn new() -> Self {
        Self {
            capture: None,
            faulted: false,
        }
    }

    fn service(&mut self, want_tx: bool, net: Option<&mut Connection>, console: &mut Console) {
        // One failure is reported per press. Releasing PTT rearms discovery so a
        // hot-plugged/default device can recover without reloading the world.
        if !want_tx {
            self.faulted = false;
        }
        if want_tx && self.capture.is_none() && !self.faulted {
            match Capture::new(CaptureConfig { device: None }) {
                Ok(c) => self.capture = Some(c),
                Err(e) => {
                    self.faulted = true;
                    console.print(format!("* voice capture unavailable: {e}"));
                }
            }
        }
        let mut runtime_fault = None;
        if let Some(cap) = self.capture.as_mut() {
            cap.set_transmitting(want_tx);
            match (want_tx, net) {
                (true, Some(net)) => {
                    for f in cap.drain() {
                        net.send_voice(f.seq.0, &f.payload);
                    }
                }
                _ => {
                    // Release means stop now; do not send an already-encoded tail.
                    cap.drain().for_each(drop);
                }
            }
            runtime_fault = cap.poll_fault();
        }
        if let Some(error) = runtime_fault {
            console.print(format!("* voice capture error: {error}"));
            self.capture = None;
            self.faulted = true;
        }
    }
}

/// The audio director: the closed field set plus the process-monotone
/// occurrence mint — the journal needs an id source that never resets across
/// worlds, while `SoundSystem::enter_world` resets its high-water to 0.
pub struct AudioDirector {
    palette: CuePalette,
    prev_medium: Option<Medium>,
    gait: Gait,
    peer_gait: HashMap<u32, f32>,
    voice_open: HashSet<u32>,
    window: WindowCache,
    capture: CaptureLane,
    next_occurrence: u64,
}

impl AudioDirector {
    pub fn new(palette: CuePalette) -> Self {
        Self {
            palette,
            prev_medium: None,
            gait: Gait::new(),
            peer_gait: HashMap::new(),
            voice_open: HashSet::new(),
            window: WindowCache::new(),
            capture: CaptureLane::new(),
            next_occurrence: 0,
        }
    }

    /// Reset the world-scoped derivation memory on a world change. The director
    /// persists on App across worlds, so its trace-derivative state must be
    /// cleared explicitly — and the mic released. `next_occurrence` is
    /// process-monotone and deliberately survives (coupled with
    /// `SoundSystem::enter_world`, which resets its high-water to 0).
    pub fn enter_world(&mut self) {
        self.prev_medium = None;
        self.gait = Gait::new();
        self.peer_gait.clear();
        self.voice_open.clear();
        self.window = WindowCache::new();
        self.capture = CaptureLane::new();
    }

    fn mint(&mut self) -> OccurrenceId {
        let id = OccurrenceId(self.next_occurrence);
        self.next_occurrence = self.next_occurrence.wrapping_add(1);
        id
    }

    /// Push one occurrence, bounded like `Game::emit` so `AudioFrame::new` never
    /// rejects the whole frame for overflow. Ids are minted in push order, so the
    /// journal is strictly increasing by construction.
    fn push(
        &mut self,
        journal: &mut Vec<Occurrence>,
        at: Option<DVec3>,
        medium: Medium,
        sfx: Sfx<OneShot>,
    ) {
        if journal.len() >= MAX_OCCURRENCES {
            return;
        }
        let id = self.mint();
        journal.push(Occurrence {
            id,
            cue: sfx.cue,
            at,
            medium,
            gain: sfx.gain,
        });
    }

    pub fn frame(&mut self, mut ctx: AudioCtx<'_>, sound: &mut SoundSystem) {
        let dt = ctx.dt;
        let medium = medium_at(ctx.world, ctx.player.pos);
        let listener = Listener {
            pos: ctx.player.pos,
            yaw: ctx.player.yaw,
            pitch: ctx.player.pitch,
            medium,
        };
        let window = self.window.refresh(ctx.world, ctx.player.pos, dt);

        let mut journal: Vec<Occurrence> = Vec::new();

        // --- Fold the drained events (one match) ---
        for ev in &ctx.events {
            match ev {
                SoundEvent::BlockBroken { at, block } => {
                    if let Some(sfx) = self
                        .palette
                        .break_block(ctx.world.registry().sound_class(*block))
                    {
                        self.push(&mut journal, Some(*at), medium_at(ctx.world, *at), sfx);
                    }
                }
                SoundEvent::BlockPlaced { at, block } => {
                    if let Some(sfx) = self.palette.place(ctx.world.registry().sound_class(*block))
                    {
                        self.push(&mut journal, Some(*at), medium_at(ctx.world, *at), sfx);
                    }
                }
                SoundEvent::PeerSwing { at } => {
                    if let Some(sfx) = self.palette.swing() {
                        self.push(&mut journal, Some(*at), medium_at(ctx.world, *at), sfx);
                    }
                }
                // UI is fire-and-forget outside the journal (the menus path).
                SoundEvent::Ui(sound_id) => {
                    if let Some(sfx) = self.palette.ui(*sound_id) {
                        sound.play_ui(sfx.cue);
                    }
                }
            }
        }

        // --- Derived: local footstep (∫ speed dt crossing) ---
        let v = ctx.player.velocity;
        let speed = (v.x * v.x + v.z * v.z).sqrt();
        if self.gait.advance(speed, dt) && speed > 0.5 && ctx.player.on_ground {
            let feet = ctx.player.feet;
            if let Some(sfx) = self.palette.step(sound_class_at_feet(ctx.world, feet)) {
                self.push(
                    &mut journal,
                    Some(feet),
                    medium_at(ctx.world, ctx.player.pos),
                    sfx,
                );
            }
        }

        // --- Derived: remote footsteps (per-peer phase crossing) ---
        // Roster diffs scan the (small) peer slice directly instead of building
        // a per-frame `HashSet`: O(n·m) over single-digit counts beats an
        // allocation plus hashing every multiplayer frame.
        for peer in ctx.peers {
            let prev = self.peer_gait.insert(peer.id, peer.phase);
            if let Some(prev) = prev
                && phase_crossed(prev, peer.phase)
                && peer.speed > 0.5
                && let Some(sfx) = self.palette.step(sound_class_at_feet(ctx.world, peer.feet))
            {
                self.push(
                    &mut journal,
                    Some(peer.feet),
                    medium_at(ctx.world, peer.at),
                    sfx,
                );
            }
        }
        let peers = ctx.peers;
        self.peer_gait
            .retain(|id, _| peers.iter().any(|p| p.id == *id));

        // --- Derived: splash on a listener medium transition (both directions) ---
        if let Some(prev) = self.prev_medium
            && matches!(prev, Medium::Water) != matches!(medium, Medium::Water)
            && let Some(sfx) = self.palette.splash()
        {
            self.push(&mut journal, Some(ctx.player.pos), medium, sfx);
        }
        self.prev_medium = Some(medium);

        // --- Derived: emitter table (latest-wins), one underwater bed iff submerged ---
        let mut emitters: Vec<Emitter> = Vec::new();
        if matches!(medium, Medium::Water)
            && let Some(sfx) = self.palette.underwater_loop()
        {
            emitters.push(Emitter {
                id: EmitterId(0),
                cue: sfx.cue,
                at: ctx.player.pos,
                medium: Medium::Water,
                gain: sfx.gain,
            });
        }

        // --- Voice sessions: present per visible peer, close on roster exit.
        // Consumes the one peer sample instead of resampling net. ---
        if ctx.net.is_some() {
            for peer in ctx.peers {
                sound.set_session_present(
                    SessionKey(peer.id),
                    peer.visible,
                    Some(peer.at),
                    medium_at(ctx.world, peer.at),
                );
            }
            self.voice_open.retain(|&id| {
                let present = peers.iter().any(|p| p.id == id);
                if !present {
                    sound.close_session(SessionKey(id));
                }
                present
            });
        }

        // A rejected frame is a construction bug: debug-assert, never panic in release.
        if let Some(window) = window {
            match AudioFrame::new(dt.clamp(1e-4, 0.5), listener, journal, emitters, window) {
                Ok(frame) => sound.submit(frame),
                Err(e) => debug_assert!(false, "audio frame rejected: {e:?}"),
            }
        }

        // --- Ingest peer voice; record the key so the close-diff can retire it ---
        if let Some(net) = ctx.net.as_deref_mut() {
            for (id, epoch, seq, payload) in net.drain_voice() {
                sound.ingest_voice(VoicePacket {
                    session: SessionKey(id),
                    epoch: Epoch(epoch),
                    seq: Seq(seq),
                    payload: payload.into_boxed_slice(),
                });
                self.voice_open.insert(id);
            }
        }

        // --- Capture / push-to-talk ---
        let want_tx = ctx.ptt && ctx.voice_enabled && ctx.net.is_some();
        self.capture
            .service(want_tx, ctx.net.as_deref_mut(), ctx.console);

        // --- Surface faults to the console, deduped once per Fault kind ---
        let mut seen: HashSet<std::mem::Discriminant<Fault>> = HashSet::new();
        for fault in sound.drain_faults() {
            if seen.insert(std::mem::discriminant(&fault)) {
                ctx.console.print(format!("* audio: {fault:?}"));
            }
        }
    }
}

/// The listener's medium, read from the block occupying the eye cell (buoyant
/// blocks are water for acoustics).
fn medium_at(world: &World, pos: DVec3) -> Medium {
    let id = world.block_at(
        pos.x.floor() as i32,
        pos.y.floor() as i32,
        pos.z.floor() as i32,
    );
    if world.registry().buoyancy(id) > 0 {
        Medium::Water
    } else {
        Medium::Air
    }
}

/// The sound class of the block just under a foot position (the 0.1 m probe below
/// the eye/feet pos), for footstep cue selection.
fn sound_class_at_feet(world: &World, feet: DVec3) -> &'static str {
    let below = world.block_at(
        feet.x.floor() as i32,
        (feet.y - 0.1).floor() as i32,
        feet.z.floor() as i32,
    );
    world.registry().sound_class(below)
}
