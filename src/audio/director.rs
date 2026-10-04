//! Gameplay reports facts, the director decides sounds. It keeps just enough
//! per-frame state (gait phase, previous roster, ...) to derive cues like
//! footsteps (gait phase crossed a stride boundary) and session join/leave
//! (roster diff) without gameplay having to push explicit events for them. Only
//! facts that can't be derived this way cross as events (`SoundEvent`).
//!
//! Lives on App beside `SoundSystem`; App feeds it one `AudioCtx` per frame.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::DQuat;
use voxel_engine::{DVec3, IVec3};

use crate::block::registry::BlockId;
use crate::console::Console;
use crate::coord::Face;
use crate::net::client::Connection;
use crate::presence::STRIDE_FREQ;
use crate::world::World;

use super::acoustics::{AcousticWindow, WindowFrame};
use super::capture::{Capture, CaptureConfig};
use super::content::OneShot;
use super::frame::MAX_OCCURRENCES;
use super::palette::{CuePalette, Sfx, UiSound};
use super::{
    AudioFrame, Epoch, Fault, Listener, Occurrence, OccurrenceId, Seq, SessionKey, SoundSystem,
    VoicePacket,
};

/// Occlusion uses `OCCL_K = 0.08`; past ~16 m gain is at most `e^{-1.3}`.
/// Radius 19 (dim 39, ~59k cells) covers that and is 7.7× cheaper than 38.
const ACOUSTIC_RADIUS: u32 = 19;

/// Squared metres. Below this the listener is treated as still, so an idle
/// director can skip the frame without missing a footstep (those need >0.5 m/s).
const LISTENER_STILL_EPS2: f64 = 1e-6;

/// The unrecoverable facts: everything else the director derives from
/// `AudioCtx`. Closed — its fold is one `match`, no bus/trait indirection.
pub enum SoundEvent {
    BlockBroken { at: DVec3, block: BlockId },
    BlockPlaced { at: DVec3, block: BlockId },
    PeerSwing { at: DVec3 },
    Ui(UiSound),
}

/// The local listener pose, built once per frame from `Player`.
#[derive(Clone, Copy)]
pub struct PlayerPose {
    pub pos: DVec3,
    pub feet: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: DVec3,
    pub on_ground: bool,
    /// Collision axis the gait speed is measured across.
    pub up: Face,
    /// View frame. Yaw and pitch are relative to it.
    pub frame: DQuat,
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
    /// Collision axis the footstep probe walks along.
    pub up: Face,
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

/// The memoized acoustic window: recapture only when something will read it
/// and the world changed, the listener crossed a cell, or ~500 ms elapsed.
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

    fn refresh(
        &mut self,
        world: &World,
        pos: DVec3,
        dt: f32,
        needed: bool,
    ) -> Option<Arc<AcousticWindow>> {
        self.timer += dt;
        // Loaded chunks on a round body are storage cells. The window samples those, and carries
        // the map from physical positions (listener, sources) into them.
        let at = world.stream_eye(pos);
        let cell = IVec3::new(
            at.x.floor() as i32,
            at.y.floor() as i32,
            at.z.floor() as i32,
        );
        let edit_gen = world.edit_generation();
        let stale = self.window.is_none()
            || edit_gen != self.edit_gen
            || cell != self.cell
            || self.timer >= 0.5;
        if stale && needed {
            let reuse = self
                .window
                .take()
                .and_then(|arc| Arc::try_unwrap(arc).ok())
                .map(AcousticWindow::into_cells);
            self.window = Some(world.capture_acoustic_window_reuse(
                cell,
                ACOUSTIC_RADIUS,
                reuse,
                window_frame(world, pos, at),
            ));
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
    gait: Gait,
    peer_gait: HashMap<u32, f32>,
    voice_open: HashSet<u32>,
    window: WindowCache,
    capture: CaptureLane,
    next_occurrence: u64,
    /// Listener position of the last committed frame. `None` until the first
    /// commit, which always runs so the gait starts from a real pose.
    last_commit: Option<DVec3>,
}

impl AudioDirector {
    pub fn new(palette: CuePalette) -> Self {
        Self {
            palette,
            gait: Gait::new(),
            peer_gait: HashMap::new(),
            voice_open: HashSet::new(),
            window: WindowCache::new(),
            capture: CaptureLane::new(),
            next_occurrence: 0,
            last_commit: None,
        }
    }

    /// Reset the world-scoped derivation memory on a world change. The director
    /// persists on App across worlds, so its trace-derivative state must be
    /// cleared explicitly — and the mic released. `next_occurrence` is
    /// process-monotone and deliberately survives (coupled with
    /// `SoundSystem::enter_world`, which resets its high-water to 0).
    pub fn enter_world(&mut self) {
        self.gait = Gait::new();
        self.peer_gait.clear();
        self.voice_open.clear();
        self.window = WindowCache::new();
        self.capture = CaptureLane::new();
        self.last_commit = None;
    }

    /// True when this frame would produce an empty journal,
    /// nothing is sounding, the listener has not moved, and no UI cue is waiting
    /// — Game can skip building a pose and submitting an [`AudioFrame`].
    pub fn can_skip_commit(
        &self,
        sound: &SoundSystem,
        events: &[SoundEvent],
        listener_pos: DVec3,
        ptt: bool,
    ) -> bool {
        if ptt || !events.is_empty() || sound.has_live_sources() || sound.ui_pending() {
            return false;
        }
        let Some(last) = self.last_commit else {
            return false;
        };
        (listener_pos - last).length_squared() <= LISTENER_STILL_EPS2
    }

    fn mint(&mut self) -> OccurrenceId {
        let id = OccurrenceId(self.next_occurrence);
        self.next_occurrence = self.next_occurrence.wrapping_add(1);
        id
    }

    /// Push one occurrence, bounded like `Game::emit` so `AudioFrame::new` never
    /// rejects the whole frame for overflow. Ids are minted in push order, so the
    /// journal is strictly increasing by construction.
    fn push(&mut self, journal: &mut Vec<Occurrence>, at: Option<DVec3>, sfx: Sfx<OneShot>) {
        if journal.len() >= MAX_OCCURRENCES {
            return;
        }
        let id = self.mint();
        journal.push(Occurrence {
            id,
            cue: sfx.cue,
            at,
            gain: sfx.gain,
        });
    }

    pub fn frame(&mut self, mut ctx: AudioCtx<'_>, sound: &mut SoundSystem) {
        let dt = ctx.dt;
        let listener = Listener {
            pos: ctx.player.pos,
            yaw: ctx.player.yaw,
            pitch: ctx.player.pitch,
            frame: ctx.player.frame,
        };

        let mut journal: Vec<Occurrence> = Vec::new();

        // --- Fold the drained events (one match) ---
        for ev in &ctx.events {
            match ev {
                SoundEvent::BlockBroken { at, block } => {
                    if let Some(sfx) = self
                        .palette
                        .break_block(ctx.world.registry().sound_class(*block))
                    {
                        self.push(&mut journal, Some(*at), sfx);
                    }
                }
                SoundEvent::BlockPlaced { at, block } => {
                    if let Some(sfx) = self.palette.place(ctx.world.registry().sound_class(*block))
                    {
                        self.push(&mut journal, Some(*at), sfx);
                    }
                }
                SoundEvent::PeerSwing { at } => {
                    if let Some(sfx) = self.palette.swing() {
                        self.push(&mut journal, Some(*at), sfx);
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
        // PosY is exactly the old XZ speed. Other faces drop the up-axis component.
        let speed = match ctx.player.up.axis() {
            0 => (v.y * v.y + v.z * v.z).sqrt(),
            1 => (v.x * v.x + v.z * v.z).sqrt(),
            _ => (v.x * v.x + v.y * v.y).sqrt(),
        };
        if self.gait.advance(speed, dt) && speed > 0.5 && ctx.player.on_ground {
            let feet = ctx.player.feet;
            if let Some(sfx) = self.palette.step(sound_class_at_feet(ctx.world, feet, ctx.player.up)) {
                self.push(&mut journal, Some(feet), sfx);
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
                && let Some(sfx) = self.palette.step(sound_class_at_feet(ctx.world, peer.feet, peer.up))
            {
                self.push(&mut journal, Some(peer.feet), sfx);
            }
        }
        let peers = ctx.peers;
        self.peer_gait
            .retain(|id, _| peers.iter().any(|p| p.id == *id));

        // --- Voice sessions: present per visible peer, close on roster exit.
        // Consumes the one peer sample instead of resampling net. ---
        if ctx.net.is_some() {
            for peer in ctx.peers {
                sound.set_session_present(SessionKey(peer.id), peer.visible, Some(peer.at));
            }
            self.voice_open.retain(|&id| {
                let present = peers.iter().any(|p| p.id == id);
                if !present {
                    sound.close_session(SessionKey(id));
                }
                present
            });
        }

        let needed = sound.has_live_sources() || !journal.is_empty();
        let window = self.window.refresh(ctx.world, ctx.player.pos, dt, needed);

        // A rejected frame is a construction bug: debug-assert, never panic in release.
        // Gameplay derives no looping emitters, so the emitter table is empty.
        match AudioFrame::new(dt.clamp(1e-4, 0.5), listener, journal, Vec::new(), window) {
            Ok(frame) => sound.submit(frame),
            Err(e) => debug_assert!(false, "audio frame rejected: {e:?}"),
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

        self.last_commit = Some(ctx.player.pos);
    }
}

/// The sound class of the block just under a foot, along `up` (storage +Y on a chart).
/// The map from physical positions near `pos` into the frame of the cells streamed around it
/// (`at = world.stream_eye(pos)`): identity off round worlds; on a chart, the inverse embedding
/// linearised by one-block differences. Central where both sides stay on one chart (the datum has
/// kinks, e.g. at a face centre, which a one-sided slope misreads on the other side); one-sided
/// where the other side crosses a seam (a one-block step moves about one storage cell, a seam jumps
/// by a chart's width).
fn window_frame(world: &World, pos: DVec3, at: DVec3) -> WindowFrame {
    if at == pos {
        return WindowFrame::IDENTITY;
    }
    let mut cols = [DVec3::ZERO; 3];
    for (a, col) in cols.iter_mut().enumerate() {
        let e = DVec3::AXES[a];
        let fwd = world.stream_eye(pos + e) - at;
        let bwd = at - world.stream_eye(pos - e);
        let (f, b) = ((fwd.length() - 1.0).abs(), (bwd.length() - 1.0).abs());
        *col = if f < 0.5 && b < 0.5 {
            (fwd + bwd) * 0.5
        } else if f <= b {
            fwd
        } else {
            bwd
        };
    }
    WindowFrame {
        phys_at: pos,
        cell_at: at,
        to_cells: glam::DMat3::from_cols(cols[0], cols[1], cols[2]),
    }
}

fn sound_class_at_feet(world: &World, feet: DVec3, up: Face) -> &'static str {
    world.registry().sound_class(world.ground_block(feet, up))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::SoundSystem;
    use crate::console::Console;
    use crate::coord::Face;
    use crate::world::World;
    use glam::DQuat;
    use voxel_engine::DVec3;

    fn pose(pos: DVec3) -> PlayerPose {
        PlayerPose {
            pos,
            feet: DVec3::new(pos.x, pos.y - 1.6, pos.z),
            yaw: 0.0,
            pitch: 0.0,
            velocity: DVec3::ZERO,
            on_ground: true,
            up: Face::PosY,
            frame: DQuat::IDENTITY,
        }
    }

    /// On the round start world the acoustic window samples storage cells while every sound plays
    /// at a physical position: the window's frame carries the physical centre of a nearby block back
    /// onto that block's storage cell, so its sound is traced through the cells really between.
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
        // The datum's kink under spawn (the face centre) bends the embedding by one or two hundredths of a
        // block per block; occlusion cells are one block.
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
        // Off a round world nothing moves.
        let flat = World::new(1);
        assert_eq!(window_frame(&flat, pos, flat.stream_eye(pos)), WindowFrame::IDENTITY);
    }

    fn commit(dir: &mut AudioDirector, sound: &mut SoundSystem, world: &World, pos: DVec3) {
        let mut console = Console::new();
        dir.frame(
            AudioCtx {
                dt: 1.0 / 60.0,
                player: pose(pos),
                ptt: false,
                voice_enabled: false,
                events: Vec::new(),
                peers: &[],
                world,
                net: None,
                console: &mut console,
            },
            sound,
        );
    }

    #[test]
    fn acoustic_radius_matches_occlusion_falloff() {
        assert_eq!(super::ACOUSTIC_RADIUS, 19);
        assert_eq!(2 * super::ACOUSTIC_RADIUS + 1, 39);
    }

    #[test]
    fn skip_commit_waits_for_the_first_frame() {
        let (sound, symbols) = SoundSystem::mute();
        let (palette, _) = CuePalette::build(&symbols, sound.catalog());
        let dir = AudioDirector::new(palette);
        assert!(!dir.can_skip_commit(&sound, &[], DVec3::ZERO, false));
    }

    #[test]
    fn skip_commit_after_a_still_silent_frame() {
        let (mut sound, symbols) = SoundSystem::mute();
        let (palette, _) = CuePalette::build(&symbols, sound.catalog());
        let mut dir = AudioDirector::new(palette);
        let world = World::generate();
        let pos = DVec3::new(0.5, 80.0, 0.5);
        commit(&mut dir, &mut sound, &world, pos);
        assert!(dir.can_skip_commit(&sound, &[], pos, false));
        assert!(
            !dir.can_skip_commit(&sound, &[], pos + DVec3::X * 0.01, false),
            "a centimetre of travel must re-enable the commit"
        );
        assert!(
            !dir.can_skip_commit(&sound, &[SoundEvent::PeerSwing { at: pos }], pos, false),
            "a pending event must re-enable the commit"
        );
        assert!(
            !dir.can_skip_commit(&sound, &[], pos, true),
            "push-to-talk must keep capture serviced"
        );
    }

    /// The footstep probe walks the up axis, and a chart walks storage −Y.
    #[test]
    fn footstep_block_follows_the_up_axis_and_a_chart() {
        let mut world = World::with_config_lazy(1, crate::render_config::RenderConfig::default());
        world.ensure_around(DVec3::new(10.5, 5.0, 3.5));
        let rock = world.registry().id_by_label("rock").unwrap();
        let air = crate::block::AIR;
        // (10, 5, 3) is one step along −X from the foot; (10, 4, 3) is one step along −Y.
        world.set_block(10, 5, 3, air);
        world.set_block(10, 4, 3, rock);
        let feet = DVec3::new(10.5, 5.0, 3.5);
        assert_eq!(world.ground_block(feet, Face::PosX), air);
        assert_eq!(world.ground_block(feet, Face::PosY), rock);
        assert_eq!(sound_class_at_feet(&world, feet, Face::PosX), "open");
        assert_eq!(
            sound_class_at_feet(&world, feet, Face::PosY),
            world.registry().sound_class(rock)
        );

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
        let physical = (
            feet.x.floor() as i32,
            (feet.y - 0.1).floor() as i32,
            feet.z.floor() as i32,
        );
        assert_eq!(
            world.registry().sound_class(world.block_at(physical.0, physical.1, physical.2)),
            "open",
            "a world-Y probe is not the storage cell under the feet"
        );
    }
}
