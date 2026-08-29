//! `SoundSystem` — the client's single audio committer.
//!
//! One `&mut self` object owns every bit of client audio continuation: the loaded
//! catalog, the backend vtable, the three voice-group maps (clip occurrences,
//! looping emitters, live voice sessions), and the fault queue. Nothing else on
//! the client commits audio state — Game hands this object an [`AudioFrame`]
//! snapshot per frame ([`submit`](SoundSystem::submit)) and forwards voice packets
//! ([`ingest_voice`](SoundSystem::ingest_voice)); the pure logic (acoustics,
//! allocation) is defined elsewhere.
//!
//! Every public method after construction returns `()` or a value, never a
//! `Result`: device/decoder death degrades to silence and a drained `Fault`, never
//! a panic across the game seam.

pub mod acoustics;
pub mod assets;
pub mod capture;
pub mod content;
pub mod director;
pub mod frame;
pub mod palette;
pub mod voice;

pub(crate) mod backend;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use glam::DVec3;

use acoustics::{AcousticWindow, Coords, Dsp, SmoothedCoords, audibility, respond, trace};
use backend::kira::KiraBackend;
use backend::null::NullBackend;
use backend::{Backend, BackendVoice, ClipId, ClipStore};
use content::{Catalog, CatalogError, DrawKind, Layer, draw, draw_raw};
use voice::{PACKET_QUEUE_DEPTH, Session};

// Re-exports so callers name these through `crate::audio::*` (the seam surface).
pub use acoustics::{Listener, Medium, Response};
pub use assets::{AudioAssets, SoundConfig};
pub use capture::{Capture, CaptureConfig, CaptureError, EncodedFrame};
pub use content::{CueId, CueMode, CueSymbols, Loop, OneShot};
pub use director::{AudioCtx, AudioDirector, PeerPose, PlayerPose, SoundEvent};
pub use frame::{AudioFrame, Emitter, EmitterId, FrameError, Occurrence, OccurrenceId};
pub use palette::{CuePalette, Sfx, UiSound};
pub use voice::{Epoch, Seq, SessionKey, VoicePacket};

/// Control snapshot. `muted` silences all output at the master; `deafen` mutes
/// incoming voice only, leaving decoders running.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixChange {
    pub master: f32,
    pub effects: f32,
    pub voice: f32,
    pub muted: bool,
    pub deafen: bool,
}

impl Default for MixChange {
    fn default() -> Self {
        Self {
            master: 1.0,
            effects: 1.0,
            voice: 1.0,
            muted: false,
            deafen: false,
        }
    }
}

impl MixChange {
    fn normalized(mut self) -> Self {
        fn gain(value: f32) -> f32 {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        self.master = gain(self.master);
        self.effects = gain(self.effects);
        self.voice = gain(self.voice);
        self
    }
}

/// Every failure the runtime reports (drained by Game).
#[derive(Debug)]
pub enum Fault {
    BackendLost,
    Starved {
        session: SessionKey,
    },
    BadUiCue {
        cue: CueId<OneShot>,
    },
    /// A one-shot occurrence reached its `max_duration` with layers that never
    /// sounded (lost the voice budget for its whole life): an audible drop, logged
    /// rather than silently discarded.
    OccurrenceDropped {
        id: OccurrenceId,
        pending: usize,
    },
}

/// Only construction can fail: a dead backend degrades in place, but a
/// malformed catalog has no silent fallback.
#[derive(Debug)]
pub enum SoundInitError {
    Catalog(CatalogError),
}

impl std::fmt::Display for SoundInitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SoundInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Catalog(error) => Some(error),
        }
    }
}

// --- coordinate smoothing ---
//   k = 1 - 0.5^(dt / halflife); each field s += (target - s) * k.
// The first observation initializes fields directly (no fade-in from zero).
pub(crate) struct Smoothed {
    distance: f32,
    occlusion: f32,
    medium: Medium,
    init: bool,
}

impl Smoothed {
    pub(crate) fn new() -> Self {
        Self {
            distance: 0.0,
            occlusion: 0.0,
            medium: Medium::Air,
            init: false,
        }
    }
    /// Advance toward `raw` and mint the smoothed value. The returned
    /// [`SmoothedCoords`] is the only coords `respond`/`audibility`/`dsp_for` accept,
    /// so ranking a source is what earns the right to voice it.
    fn observe(&mut self, raw: &Coords, k: f32) -> SmoothedCoords {
        if !self.init {
            self.distance = raw.distance;
            self.occlusion = raw.occlusion;
            self.init = true;
        } else {
            self.distance += (raw.distance - self.distance) * k;
            self.occlusion += (raw.occlusion - self.occlusion) * k;
        }
        self.medium = raw.medium;
        SmoothedCoords::new(self.distance, self.occlusion, raw.medium)
    }
    /// The already-integrated value without advancing — the apply passes read this
    /// after ranking observed once this frame.
    fn current(&self) -> SmoothedCoords {
        SmoothedCoords::new(self.distance, self.occlusion, self.medium)
    }
}

/// One realized occurrence group — an allocation unit. Its layers are allocated
/// all-or-nothing; each starts against its own delay deadline.
struct GroupState {
    cue: CueId<OneShot>,
    at: Option<DVec3>,
    medium: Medium,
    gain: f32,
    smooth: Smoothed,
    layers: Box<[LayerState]>,
    age: f32, // seconds since realization; ≥ cue.max_duration ⇒ Released
}

/// Every randomized layer parameter is drawn together exactly once. One-shots,
/// looping emitters, and UI playback all use this same realization path, keeping
/// variant/gain/pitch/delay semantics from drifting between sinks.
#[derive(Clone, Copy)]
struct LayerDraw {
    clip: ClipId,
    gain: f32,
    rate: f32,
    delay: f32,
}

enum LayerState {
    Pending { draw: LayerDraw },
    Sounding { voice: BackendVoice, gain: f32 },
    Done,
}

struct EmitterVoice {
    cue: CueId<Loop>,
    smooth: Smoothed,
    age: f32,
    layers: Box<[EmitterLayer]>,
}

struct EmitterLayer {
    draw: LayerDraw,
    voice: Option<BackendVoice>,
}

/// The allocation ranking key — unifies the three state maps under one budget.
/// The derived `Ord` gives the deterministic tiebreak (Clip < Emitter < Voice, then
/// inner id ascending).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum GroupKey {
    Clip(OccurrenceId),
    Emitter(EmitterId),
    Voice(SessionKey),
}

pub struct SoundSystem {
    catalog: Catalog,
    backend: Box<dyn Backend>,
    cfg: SoundConfig,
    mix: MixChange,
    /// Idempotence: occurrences at or below this id were already realized.
    /// `None` is the only correct fresh sentinel because id zero is valid.
    high_water: Option<OccurrenceId>,
    clip_voices: HashMap<OccurrenceId, GroupState>,
    emitter_voices: HashMap<EmitterId, EmitterVoice>,
    sessions: HashMap<SessionKey, Session>,
    /// Per-session starvation mirror (decode thread sets it; submit polls it
    /// edge-triggered). Value pairs the shared flag with the last reported state.
    session_starved: HashMap<SessionKey, (Arc<AtomicBool>, bool)>,
    faults: VecDeque<Fault>,
    /// Latest listener, used to open voice streams that arrive between frames.
    last_listener: Listener,
    /// Monotone id feeding the deterministic draw for fire-and-forget UI cues.
    ui_counter: u64,
    /// Live UI voices with their expiry: `play_ui` bypasses the group budget, but
    /// the handles are retained here so they can be stopped on world exit and
    /// pruned once their finite one-shot has elapsed — never silently leaked.
    ui_voices: Vec<(BackendVoice, Instant)>,
    /// Whether `Fault::BackendLost` has already been reported (edge-trigger).
    reported_lost: bool,
}

impl SoundSystem {
    pub fn new(mut cfg: SoundConfig) -> Result<(Self, CueSymbols), SoundInitError> {
        cfg.normalize();
        let mut faults = VecDeque::new();
        let mut reported_lost = false;
        // A missing/failed device degrades to a silent NullBackend rather
        // than failing construction — the game still runs, just mute.
        let mut backend: Box<dyn Backend> = match KiraBackend::new() {
            Some(b) => Box::new(b),
            None => {
                faults.push_back(Fault::BackendLost);
                reported_lost = true;
                Box::new(NullBackend::new())
            }
        };
        // Catalog decoding folds bytes through the backend's ClipStore.
        // `Backend: ClipStore`, so `&mut dyn Backend` upcasts directly.
        let (catalog, symbols) = {
            let clips: &mut dyn ClipStore = &mut *backend;
            Catalog::load(&cfg.content_dir, clips).map_err(SoundInitError::Catalog)?
        };

        Ok((
            Self::assemble(backend, catalog, faults, reported_lost, cfg),
            symbols,
        ))
    }

    /// A guaranteed-silent system: a `NullBackend` over an empty catalog, with no
    /// device probe and no disk access. Headless golden runs use this so they need
    /// neither an audio device nor the `assets/sounds` tree. Every cue lookup
    /// yields None, so the palette resolves every role to silence. `reported_lost`
    /// starts set: the silence is deliberate, not a device loss to surface.
    pub fn mute() -> (Self, CueSymbols) {
        let (catalog, symbols) = Catalog::empty();
        let system = Self::assemble(
            Box::new(NullBackend::new()),
            catalog,
            VecDeque::new(),
            true,
            SoundConfig::silent(),
        );
        (system, symbols)
    }

    /// Try the real device+catalog; on any init error, report it once to stderr and
    /// fall back to [`mute`](Self::mute). App uses this so a missing or corrupt
    /// catalog can never panic the client — it degrades to silence instead.
    pub fn with_graceful_degradation(cfg: SoundConfig) -> (Self, CueSymbols) {
        match Self::new(cfg) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("audio: init failed ({e}); running muted");
                Self::mute()
            }
        }
    }

    /// The one struct-literal site shared by [`new`](Self::new) and
    /// [`mute`](Self::mute): backend + catalog are the only inputs that differ.
    fn assemble(
        backend: Box<dyn Backend>,
        catalog: Catalog,
        faults: VecDeque<Fault>,
        reported_lost: bool,
        cfg: SoundConfig,
    ) -> Self {
        Self {
            catalog,
            backend,
            mix: MixChange::default(),
            high_water: None,
            clip_voices: HashMap::new(),
            emitter_voices: HashMap::new(),
            sessions: HashMap::new(),
            session_starved: HashMap::new(),
            faults,
            last_listener: Listener {
                pos: DVec3::ZERO,
                yaw: 0.0,
                pitch: 0.0,
                medium: Medium::Air,
            },
            ui_counter: 0,
            ui_voices: Vec::new(),
            reported_lost,
            cfg,
        }
    }

    /// The loaded catalog, so App can build the [`CuePalette`] once at startup
    /// (the palette resolves every game fact's mode-typed cue up front).
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Reset all world-scoped state (voices, sessions, high-water → empty). Idempotent.
    /// Game's `next_occurrence` counter is process-monotone and never resets, so a
    /// fresh high-water accepts every valid id, including zero.
    pub fn enter_world(&mut self) {
        for (_, group) in self.clip_voices.drain() {
            for layer in group.layers.iter() {
                if let LayerState::Sounding { voice, .. } = layer {
                    self.backend.stop(*voice);
                }
            }
        }
        for (_, ev) in self.emitter_voices.drain() {
            stop_emitter(&mut *self.backend, ev);
        }
        for (_, session) in self.sessions.drain() {
            if let Session::Streaming { voice, .. } = session {
                self.backend.stop(voice);
            }
        }
        for (voice, _) in self.ui_voices.drain(..) {
            self.backend.stop(voice); // menu cues must not bleed across a world load
        }
        self.session_starved.clear();
        self.high_water = None;
    }

    pub fn leave_world(&mut self) {
        self.enter_world();
    }

    pub fn submit(&mut self, frame: AudioFrame) {
        self.poll_starvation();

        if !self.backend.alive() {
            // A dead backend degrades to a total no-op. Still advance the
            // high-water so ids never replay as fresh if a backend returns.
            self.report_lost_once();
            let (_, listener, occurrences, _, _) = frame.into_parts();
            self.last_listener = listener;
            for o in &occurrences {
                self.high_water = Some(self.high_water.map_or(o.id, |seen| seen.max(o.id)));
            }
            return;
        }

        let (dt, listener, occurrences, emitters, window) = frame.into_parts();
        self.last_listener = listener;
        let k = smoothing_factor(dt, self.cfg.smoothing_halflife_s);

        // --- Drop ≤ high_water, realize new occurrence groups ---
        let mut max_id = self.high_water;
        for group in self.clip_voices.values_mut() {
            group.age += dt;
        }
        for emitter in self.emitter_voices.values_mut() {
            emitter.age += dt;
        }
        for occ in &occurrences {
            if self.high_water.is_some_and(|seen| occ.id <= seen) {
                continue; // idempotence
            }
            max_id = Some(max_id.map_or(occ.id, |seen| seen.max(occ.id)));
            let cue = self.catalog.cue(occ.cue);
            let layers: Box<[LayerState]> = cue
                .layers
                .iter()
                .enumerate()
                .map(|(li, layer)| LayerState::Pending {
                    draw: draw_layer(layer, occ.id, li as u16),
                })
                .collect();
            self.clip_voices.insert(
                occ.id,
                GroupState {
                    cue: occ.cue,
                    at: occ.at,
                    medium: occ.medium,
                    gain: occ.gain,
                    smooth: Smoothed::new(),
                    layers,
                    age: 0.0,
                },
            );
        }
        self.high_water = max_id;

        // --- Emitter reconcile (latest-wins snapshot) ---
        let table: HashMap<EmitterId, Emitter> = emitters.iter().map(|e| (e.id, *e)).collect();
        // Release emitters absent from the table, or whose stable id was reused
        // for a different cue. Keeping the old decoded loop in the latter case
        // would make the snapshot lie about what is sounding.
        let gone: Vec<EmitterId> = self
            .emitter_voices
            .iter()
            .filter_map(|(id, state)| {
                table
                    .get(id)
                    .is_none_or(|emitter| emitter.cue != state.cue)
                    .then_some(*id)
            })
            .collect();
        for id in gone {
            if let Some(ev) = self.emitter_voices.remove(&id) {
                stop_emitter(&mut *self.backend, ev);
            }
        }
        // Realize deterministic layer data for new loop groups. Actual backend
        // voices open only after ranking grants the whole group a budget.
        for e in &emitters {
            if self.emitter_voices.contains_key(&e.id) {
                continue;
            }
            let cue = self.catalog.cue(e.cue);
            let draw_id = OccurrenceId(e.id.0 ^ EMITTER_DRAW_SALT);
            let layers = cue
                .layers
                .iter()
                .enumerate()
                .map(|(index, layer)| EmitterLayer {
                    draw: draw_layer(layer, draw_id, index as u16),
                    voice: None,
                })
                .collect();
            self.emitter_voices.insert(
                e.id,
                EmitterVoice {
                    cue: e.cue,
                    smooth: Smoothed::new(),
                    age: 0.0,
                    layers,
                },
            );
        }

        // --- Trace coords, rank by audibility, allocate whole groups ---
        let mut cands: Vec<(f32, GroupKey, usize)> = Vec::new();

        for (id, group) in self.clip_voices.iter_mut() {
            let raw = raw_coords(&window, &listener, group.at, group.medium);
            let sc = group.smooth.observe(&raw, k);
            let cue = self.catalog.cue(group.cue);
            // Rank with the group's loudest possible layer gain so the bound covers
            // every layer's `lgain`: audibility ≥ applied for all of them.
            let max_lgain = cue
                .layers
                .iter()
                .map(|l| *l.gain.end())
                .fold(0.0f32, f32::max);
            let aud = audibility(cue.response, sc, group.gain * max_lgain);
            // Cost is the voices the group will actually hold — Done layers hold none.
            let cost = group
                .layers
                .iter()
                .filter(|l| !matches!(l, LayerState::Done))
                .count();
            cands.push((aud, GroupKey::Clip(*id), cost));
        }
        for (id, ev) in self.emitter_voices.iter_mut() {
            let e = table[id];
            let raw = raw_coords(&window, &listener, Some(e.at), e.medium);
            let sc = ev.smooth.observe(&raw, k);
            let cue = self.catalog.cue(ev.cue);
            let max_lgain = ev
                .layers
                .iter()
                .map(|layer| layer.draw.gain)
                .fold(0.0, f32::max);
            let aud = audibility(cue.response, sc, e.gain * max_lgain);
            cands.push((aud, GroupKey::Emitter(*id), ev.layers.len()));
        }
        // Sessions smooth like every other source: a per-session `Smoothed`
        // cell, observed here, read again in apply_sessions.
        for (id, session) in self.sessions.iter_mut() {
            if let Session::Streaming {
                present: true,
                last_at,
                last_medium,
                smooth,
                ..
            } = session
            {
                let raw = raw_coords(&window, &listener, *last_at, *last_medium);
                let sc = smooth.observe(&raw, k);
                let aud = audibility(Response::Voice, sc, 1.0);
                cands.push((aud, GroupKey::Voice(*id), 1));
            }
        }

        // Rank by audibility desc, key asc (deterministic tiebreak).
        cands.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut remaining = self.cfg.max_voices;
        let mut winners: std::collections::HashSet<GroupKey> = std::collections::HashSet::new();
        for (_, key, cost) in &cands {
            if *cost <= remaining {
                remaining -= *cost;
                winners.insert(*key);
            }
            // Groups that do not fit are skipped whole (never a partial layer).
        }

        // --- respond() and drive the backend ---
        self.apply_clip_groups(&winners, &listener);
        self.apply_emitters(&winners, &table, &listener);
        self.apply_sessions(&winners, &listener);

        // Release finished one-shot groups (max_duration fold).
        let done: Vec<OccurrenceId> = self
            .clip_voices
            .iter()
            .filter(|(_, g)| g.age >= self.catalog.cue(g.cue).max_duration)
            .map(|(id, _)| *id)
            .collect();
        for id in done {
            if let Some(group) = self.clip_voices.remove(&id) {
                let mut pending = 0;
                for layer in group.layers.iter() {
                    match layer {
                        LayerState::Sounding { voice, .. } => self.backend.stop(*voice),
                        LayerState::Pending { .. } => pending += 1,
                        LayerState::Done => {}
                    }
                }
                if pending > 0 {
                    self.faults
                        .push_back(Fault::OccurrenceDropped { id, pending });
                }
            }
        }
    }

    fn apply_clip_groups(
        &mut self,
        winners: &std::collections::HashSet<GroupKey>,
        listener: &Listener,
    ) {
        for (id, group) in self.clip_voices.iter_mut() {
            let allocated = winners.contains(&GroupKey::Clip(*id));
            let cue = self.catalog.cue(group.cue);
            let response = cue.response;
            let sc = group.smooth.current();
            for layer in group.layers.iter_mut() {
                match layer {
                    LayerState::Pending { draw } => {
                        if allocated && group.age >= draw.delay {
                            let dsp = dsp_for(
                                response,
                                sc,
                                listener,
                                group.at,
                                group.gain * draw.gain,
                                self.mix.effects,
                            );
                            // A clip group's cue is `CueId<OneShot>` by type, so it never loops.
                            match self
                                .backend
                                .play_clip(draw.clip, dsp, draw.rate, false, group.at, listener)
                            {
                                Some(voice) => {
                                    *layer = LayerState::Sounding {
                                        voice,
                                        gain: draw.gain,
                                    }
                                }
                                None => *layer = LayerState::Done,
                            }
                        }
                    }
                    LayerState::Sounding { voice, gain } => {
                        if allocated {
                            let dsp = dsp_for(
                                response,
                                sc,
                                listener,
                                group.at,
                                group.gain * *gain,
                                self.mix.effects,
                            );
                            self.backend.update(*voice, dsp, group.at, listener);
                        } else {
                            // A losing sounding clip is stopped (no resume).
                            self.backend.stop(*voice);
                            *layer = LayerState::Done;
                        }
                    }
                    LayerState::Done => {}
                }
            }
        }
    }

    fn apply_emitters(
        &mut self,
        winners: &std::collections::HashSet<GroupKey>,
        table: &HashMap<EmitterId, Emitter>,
        listener: &Listener,
    ) {
        for (id, ev) in self.emitter_voices.iter_mut() {
            let e = table[id];
            let sc = ev.smooth.current();
            let allocated = winners.contains(&GroupKey::Emitter(*id));
            let response = self.catalog.cue(ev.cue).response;
            // A losing emitter is muted (loop preserved), never stopped — mirrors the
            // streaming-session rule and, unlike a stop, avoids the per-frame respawn
            // churn a still-tabled emitter would cause.
            let bus = if allocated { self.mix.effects } else { 0.0 };
            for layer in ev.layers.iter_mut() {
                let dsp = dsp_for(
                    response,
                    sc,
                    listener,
                    Some(e.at),
                    e.gain * layer.draw.gain,
                    bus,
                );
                if let Some(voice) = layer.voice {
                    self.backend.update(voice, dsp, Some(e.at), listener);
                } else if allocated && ev.age >= layer.draw.delay {
                    layer.voice = self.backend.play_clip(
                        layer.draw.clip,
                        dsp,
                        layer.draw.rate,
                        true,
                        Some(e.at),
                        listener,
                    );
                }
            }
        }
    }

    fn apply_sessions(
        &mut self,
        winners: &std::collections::HashSet<GroupKey>,
        listener: &Listener,
    ) {
        for (id, session) in self.sessions.iter() {
            let Session::Streaming {
                present,
                last_at,
                voice,
                smooth,
                ..
            } = session
            else {
                continue;
            };
            let allocated = winners.contains(&GroupKey::Voice(*id));
            // detach/deafen/allocation-loss all mute (gain 0), never stop.
            let audible = *present && allocated && !self.mix.deafen;
            // The smoothed coords were integrated in ranking; read, don't re-trace.
            let sc = smooth.current();
            let bus = if audible { self.mix.voice } else { 0.0 };
            let dsp = dsp_for(Response::Voice, sc, listener, *last_at, 1.0, bus);
            self.backend.update(*voice, dsp, *last_at, listener);
        }
    }

    /// Teardown order: on epoch bump, `stop` the old voice BEFORE dropping the
    /// old producer and opening the new stream.
    pub fn ingest_voice(&mut self, pkt: VoicePacket) {
        // Decide first (immutable peek), then act — so the epoch-bump teardown
        // can `remove`/`open` without holding a `get_mut` borrow of `sessions`.
        // `floor` is the epoch to preserve as a Tombstone if the (re)open fails, so a
        // stale old-epoch packet can never later read as `OpenNew`.
        enum Act {
            Drop,
            Feed,
            OpenNew { floor: Option<Epoch> },
            Bump { old: Epoch },
        }
        let act = match self.sessions.get(&pkt.session) {
            Some(Session::Tombstone { epoch }) => {
                if pkt.epoch > *epoch {
                    Act::OpenNew {
                        floor: Some(*epoch),
                    }
                } else {
                    Act::Drop
                }
            }
            Some(Session::Streaming { epoch, .. }) => {
                if pkt.epoch < *epoch {
                    Act::Drop
                } else if pkt.epoch == *epoch {
                    Act::Feed
                } else {
                    Act::Bump { old: *epoch }
                }
            }
            None => Act::OpenNew { floor: None },
        };
        match act {
            Act::Drop => {}
            Act::Feed => {
                if let Some(Session::Streaming { feed, .. }) = self.sessions.get_mut(&pkt.session) {
                    let _ = feed.push(pkt); // queue-full drops
                }
            }
            Act::OpenNew { floor } => self.open_session(pkt, floor),
            Act::Bump { old } => {
                // Stop the old voice BEFORE dropping the old producer and opening the
                // new stream. The old epoch is the floor to keep if the reopen fails
                // (the new epoch still reopens: it > old).
                let key = pkt.session;
                if let Some(Session::Streaming { voice, .. }) = self.sessions.remove(&key) {
                    self.backend.stop(voice);
                }
                self.session_starved.remove(&key);
                self.open_session(pkt, Some(old));
            }
        }
    }

    /// Open a fresh `Streaming` session for `pkt`'s epoch: an rtrb pair, a starved
    /// flag, an immediate muted `play_stream`, then feed the triggering packet. On
    /// failure (no device or backend refusal) it leaves `floor` as a Tombstone so the
    /// epoch floor survives independent of session-entry success.
    fn open_session(&mut self, pkt: VoicePacket, floor: Option<Epoch>) {
        let key = pkt.session;
        let epoch = pkt.epoch;
        if !self.backend.alive() {
            if let Some(f) = floor {
                self.sessions.insert(key, Session::Tombstone { epoch: f });
            }
            return; // no device, no session
        }
        let (mut producer, consumer) = rtrb::RingBuffer::<VoicePacket>::new(PACKET_QUEUE_DEPTH);
        let starved = Arc::new(AtomicBool::new(false));
        match self.backend.play_stream(
            consumer,
            self.cfg.jitter_target_ms,
            starved.clone(),
            None,
            &self.last_listener,
        ) {
            Some(voice) => {
                let _ = producer.push(pkt); // seed the buffer with its first frame
                self.sessions.insert(
                    key,
                    Session::Streaming {
                        epoch,
                        feed: producer,
                        voice,
                        present: false,
                        last_at: None,
                        last_medium: self.last_listener.medium,
                        smooth: Smoothed::new(),
                    },
                );
                self.session_starved.insert(key, (starved, false));
            }
            None => {
                // Backend refused: preserve the epoch floor so stale packets stay
                // rejected; a packet at `epoch` (> floor) still retries via OpenNew.
                if let Some(f) = floor {
                    self.sessions.insert(key, Session::Tombstone { epoch: f });
                }
            }
        }
    }

    /// Presentation control, decoupled from the decoder lifetime. Never stops.
    pub fn set_session_present(
        &mut self,
        session: SessionKey,
        present: bool,
        at: Option<DVec3>,
        medium: Medium,
    ) {
        if let Some(Session::Streaming {
            present: p,
            last_at,
            last_medium,
            ..
        }) = self.sessions.get_mut(&session)
        {
            *p = present;
            if at.is_some() {
                *last_at = at;
                *last_medium = medium;
            }
        }
    }

    /// Terminal close (peer left the server): `Streaming` → `Tombstone(epoch)`.
    pub fn close_session(&mut self, session: SessionKey) {
        if let Some(Session::Streaming { epoch, voice, .. }) = self.sessions.remove(&session) {
            self.backend.stop(voice);
            self.sessions.insert(session, Session::Tombstone { epoch });
        }
        self.session_starved.remove(&session);
    }

    pub fn set_mix(&mut self, mix: MixChange) {
        let mix = mix.normalized();
        self.mix = mix;
        let master = if mix.muted { 0.0 } else { mix.master };
        self.backend.set_master(master);
        // Deafen/undeafen take effect on the next submit (voices muted via update).
    }

    pub fn drain_faults(&mut self) -> impl Iterator<Item = Fault> + '_ {
        self.faults.drain(..)
    }

    /// App-wide maintenance tick. It is intentionally independent of
    /// `AudioFrame`, so menus and loading screens reclaim finished UI tracks and
    /// future music/device recovery has one lifecycle hook.
    pub fn service(&mut self) {
        let now = Instant::now();
        let mut index = 0;
        while index < self.ui_voices.len() {
            if self.ui_voices[index].1 <= now {
                let (voice, _) = self.ui_voices.swap_remove(index);
                self.backend.stop(voice);
            } else {
                index += 1;
            }
        }
        if !self.backend.alive() {
            self.report_lost_once();
        }
    }

    /// UI one-shot outside the frame journal (menus have no AudioFrame cadence).
    /// Non-Ui cue ⇒ `Fault::BadUiCue` + no-op (fire-and-forget).
    pub fn play_ui(&mut self, cue: CueId<OneShot>) {
        let now = Instant::now();
        self.service();

        let cue_ref = self.catalog.cue(cue);
        if !matches!(cue_ref.response, Response::Ui) {
            self.faults.push_back(Fault::BadUiCue { cue });
            return;
        }
        if !self.backend.alive() {
            return;
        }
        let id = OccurrenceId(self.ui_counter);
        self.ui_counter = self.ui_counter.wrapping_add(1);
        let expiry = now + Duration::from_secs_f32(cue_ref.max_duration.max(0.0));
        // Ui is non-spatial: coincident smoothed coords (nothing to integrate).
        let ui_dsp = respond(
            Response::Ui,
            SmoothedCoords::coincident(Medium::Air),
            &self.last_listener,
            None,
        );
        for (li, layer) in cue_ref.layers.iter().enumerate() {
            let draw = draw_layer(layer, id, li as u16);
            debug_assert_eq!(draw.delay, 0.0, "UI delay is rejected by the catalog");
            let dsp = Dsp {
                gain: (ui_dsp.gain * draw.gain * self.mix.effects).clamp(0.0, 4.0),
                ..ui_dsp
            };
            // Budget-exempt (menus have no frame cadence) but retained, not leaked.
            if let Some(voice) =
                self.backend
                    .play_clip(draw.clip, dsp, draw.rate, false, None, &self.last_listener)
            {
                self.ui_voices.push((voice, expiry));
            }
        }
    }

    /// Edge-trigger `Fault::Starved` once per starvation burst per session.
    fn poll_starvation(&mut self) {
        for (key, (flag, last)) in self.session_starved.iter_mut() {
            let now = flag.load(Ordering::Relaxed);
            if now && !*last {
                self.faults.push_back(Fault::Starved { session: *key });
            }
            *last = now;
        }
    }

    fn report_lost_once(&mut self) {
        if !self.reported_lost {
            self.faults.push_back(Fault::BackendLost);
            self.reported_lost = true;
        }
    }
}

// --- free helpers (no `&self`, so they never conflict with field borrows) ---

fn smoothing_factor(dt: f32, halflife_s: f32) -> f32 {
    1.0 - 0.5f32.powf(dt / halflife_s.max(1e-4))
}

/// Raw (unsmoothed) coords for a source. A non-spatial source is coincident.
fn raw_coords(
    window: &AcousticWindow,
    listener: &Listener,
    at: Option<DVec3>,
    medium: Medium,
) -> Coords {
    match at {
        Some(pos) => trace(window, listener.pos, pos, medium),
        None => Coords {
            distance: 0.0,
            occlusion: 0.0,
            medium: listener.medium,
        },
    }
}

/// Fold `respond`'s spatial gain with the authored scale and the mixer bus. Takes
/// [`SmoothedCoords`], so the smoothing obligation is discharged by the type.
fn dsp_for(
    response: Response,
    sc: SmoothedCoords,
    listener: &Listener,
    source: Option<DVec3>,
    authored: f32,
    bus: f32,
) -> Dsp {
    let mut d = respond(response, sc, listener, source);
    d.gain = (d.gain * authored * bus).clamp(0.0, 4.0);
    d
}

/// Map a deterministic draw bucket onto a `[lo, hi]` range (no RNG state).
fn draw_range(
    id: OccurrenceId,
    layer: u16,
    kind: DrawKind,
    range: &std::ops::RangeInclusive<f32>,
) -> f32 {
    let (lo, hi) = (*range.start(), *range.end());
    if lo == hi {
        return lo;
    }
    // Center each of the 2^32 buckets in (0, 1): (raw + 0.5) / 2^32 is uniform and
    // never hits either endpoint, unlike the old `% (2^32 - 1)` fold.
    let u = (draw_raw(id, layer, kind) as f64 + 0.5) / 2f64.powi(32);
    lo + (u as f32) * (hi - lo)
}

fn draw_layer(layer: &Layer, id: OccurrenceId, index: u16) -> LayerDraw {
    let variant = draw(id, index, DrawKind::Variant, layer.variants.len() as u32) as usize;
    LayerDraw {
        clip: layer.variants[variant],
        gain: draw_range(id, index, DrawKind::Gain, &layer.gain),
        rate: 2f32.powf(draw_range(id, index, DrawKind::Pitch, &layer.pitch) / 12.0),
        delay: draw_range(id, index, DrawKind::Delay, &layer.delay),
    }
}

/// Salt (a reserved bit above the layer/kind fields) mixed into an emitter's draw
/// id so emitter `n` and occurrence `n` never share a variant/pitch draw.
const EMITTER_DRAW_SALT: u64 = 1 << 58;

fn stop_emitter(backend: &mut dyn Backend, emitter: EmitterVoice) {
    for layer in emitter.layers {
        if let Some(voice) = layer.voice {
            backend.stop(voice);
        }
    }
}

/// Backend-boundary tests: the assertions read the `RecordingBackend`'s intent log
/// (what the runtime actually told the backend), not internal state that's already
/// guaranteed by the type system (e.g. "occurrences are OneShot" needs no test —
/// the compiler owns that). Each test names its reachable failure state.
#[cfg(test)]
mod seam_tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use glam::{DVec3, IVec3, UVec3};

    use super::acoustics::{
        AcousticWindow, Cell, Dsp, Listener, Medium, Response, SmoothedCoords, respond,
    };
    use super::backend::ClipId;
    use super::backend::recording::{Intent, Recorder, RecordingBackend};
    use super::content::{Catalog, CueSymbols, Loop, OneShot};
    use super::{
        AudioFrame, Emitter, EmitterId, Occurrence, OccurrenceId, SoundConfig, SoundSystem,
    };

    // Shared catalog fixture: one one-shot, one loop bed, and a loud/quiet pair whose
    // only difference is layer gain (authored gain ≠ 1.0).
    const MANIFEST: &str = r#"
        [cues.oneshot]
        response = "world"
        [[cues.oneshot.layers]]
        variants = ["a"]
        gain = [1.0]
        pitch = [0.0]
        delay = [0.0]
        mode = "one_shot"

        [cues.bed]
        response = "ambient"
        [[cues.bed.layers]]
        variants = ["b"]
        gain = [1.0]
        pitch = [0.0]
        delay = [0.0]
        mode = "loop"

        [cues.layered_bed]
        response = "world"
        [[cues.layered_bed.layers]]
        variants = ["e"]
        gain = [0.5]
        pitch = [0.0]
        delay = [0.0]
        mode = "loop"
        [[cues.layered_bed.layers]]
        variants = ["f"]
        gain = [1.5]
        pitch = [0.0]
        delay = [0.2]
        mode = "loop"

        [cues.loud]
        response = "world"
        [[cues.loud.layers]]
        variants = ["c"]
        gain = [2.0]
        pitch = [0.0]
        delay = [0.0]
        mode = "one_shot"

        [cues.quiet]
        response = "world"
        [[cues.quiet.layers]]
        variants = ["d"]
        gain = [0.5]
        pitch = [0.0]
        delay = [0.0]
        mode = "one_shot"

        [cues.ui]
        response = "ui"
        [[cues.ui.layers]]
        variants = ["g"]
        gain = [1.0]
        pitch = [0.0]
        delay = [0.0]
        mode = "one_shot"
    "#;

    /// A `SoundSystem` over a `RecordingBackend` + the in-memory catalog, plus the
    /// test-side `Recorder`. `max_voices` lets a test create budget pressure.
    fn system(max_voices: usize) -> (SoundSystem, CueSymbols, Recorder) {
        let (mut backend, rec) = RecordingBackend::new();
        let mut resolve = |_name: &str| -> Result<Vec<u8>, std::path::PathBuf> { Ok(vec![0u8]) };
        let (catalog, syms) = Catalog::from_manifest(MANIFEST, &mut resolve, &mut backend).unwrap();
        let cfg = SoundConfig {
            max_voices,
            ..SoundConfig::default()
        };
        let sys = SoundSystem::assemble(Box::new(backend), catalog, VecDeque::new(), false, cfg);
        (sys, syms, rec)
    }

    /// An all-`Open` window spanning x ∈ [-8, 88): any axis ray between in-range points
    /// reads occlusion 0, so trace distance is the plain euclidean gap.
    fn open_window() -> Arc<AcousticWindow> {
        let size = UVec3::new(96, 16, 16);
        let cells = vec![Cell::Open; (96 * 16 * 16) as usize].into_boxed_slice();
        Arc::new(AcousticWindow::new(IVec3::new(-8, -8, -8), size, cells).unwrap())
    }

    fn origin_listener() -> Listener {
        Listener {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            medium: Medium::Air,
        }
    }

    fn source(x: f64) -> DVec3 {
        DVec3::new(x, 0.0, 0.0)
    }

    fn count_play_clips(rec: &Recorder) -> usize {
        rec.intents()
            .iter()
            .filter(|i| matches!(i, Intent::PlayClip { .. }))
            .count()
    }

    fn played_clips(rec: &Recorder) -> Vec<ClipId> {
        rec.intents()
            .into_iter()
            .filter_map(|i| match i {
                Intent::PlayClip { clip, .. } => Some(clip),
                _ => None,
            })
            .collect()
    }

    fn played_dsps(rec: &Recorder) -> Vec<Dsp> {
        rec.intents()
            .into_iter()
            .filter_map(|intent| match intent {
                Intent::PlayClip { dsp, .. } => Some(dsp),
                _ => None,
            })
            .collect()
    }

    fn emitter_gains(rec: &Recorder) -> Vec<f32> {
        rec.intents()
            .into_iter()
            .filter_map(|intent| match intent {
                Intent::PlayClip {
                    dsp, looped: true, ..
                }
                | Intent::Update { dsp, .. } => Some(dsp.gain),
                _ => None,
            })
            .collect()
    }

    // A source jumping 40 m: the recorded Dsp.gain must be damped, not snapped,
    // between frames. Reachable failure: coordinate smoothing is bypassed (raw
    // trace feeds respond), so frame 2's gain equals the unsmoothed gain at 50 m
    // instead of sitting above it.
    #[test]
    fn a8_moving_source_gain_is_smoothed() {
        let (mut sound, syms, rec) = system(32);
        let bed = sound.catalog().typed::<Loop>(&syms, "bed").unwrap();
        let emit = |x| Emitter {
            id: EmitterId(0),
            cue: bed,
            at: source(x),
            medium: Medium::Air,
            gain: 1.0,
        };

        sound.submit(
            AudioFrame::new(
                0.1,
                origin_listener(),
                vec![],
                vec![emit(10.0)],
                open_window(),
            )
            .unwrap(),
        );
        sound.submit(
            AudioFrame::new(
                0.1,
                origin_listener(),
                vec![],
                vec![emit(50.0)],
                open_window(),
            )
            .unwrap(),
        );

        let gains = emitter_gains(&rec);
        assert_eq!(gains.len(), 2, "one emitter Update per frame");
        let (g1, g2) = (gains[0], gains[1]);
        // What a NO-smoothing path would emit at the new 50 m distance.
        let unsmoothed = respond(
            Response::Ambient,
            SmoothedCoords::new(50.0, 0.0, Medium::Air),
            &origin_listener(),
            Some(source(50.0)),
        )
        .gain;
        assert!(g2 < g1, "receding source must get quieter: g1={g1} g2={g2}");
        assert!(
            g2 > unsmoothed + 1e-4,
            "smoothing must damp the jump: g2={g2} unsmoothed={unsmoothed}"
        );
    }

    // With room for exactly one voice, the louder group wins, decided by audibility
    // from the recorded log (which clip actually played), not by the id tiebreak.
    // `loud` (gain 2.0) gets the higher id, `quiet` (gain 0.5) the lower; both at the
    // same position. Reachable failure: ranking ignores layer gain (`max_lgain`
    // dropped), the two audibilities tie, and the id tiebreak voices `quiet` (lower
    // id) — an admissibility bug, observable as the wrong clip.
    #[test]
    fn a9_louder_group_wins_the_only_voice() {
        let (mut sound, syms, rec) = system(1);
        let loud = sound.catalog().typed::<OneShot>(&syms, "loud").unwrap();
        let quiet = sound.catalog().typed::<OneShot>(&syms, "quiet").unwrap();
        let loud_clip = sound.catalog().cue(loud).layers[0].variants[0];

        let at = Some(source(12.0));
        let occs = vec![
            Occurrence {
                id: OccurrenceId(1),
                cue: quiet,
                at,
                medium: Medium::Air,
                gain: 1.0,
            },
            Occurrence {
                id: OccurrenceId(2),
                cue: loud,
                at,
                medium: Medium::Air,
                gain: 1.0,
            },
        ];
        sound.submit(AudioFrame::new(0.1, origin_listener(), occs, vec![], open_window()).unwrap());

        let clips = played_clips(&rec);
        assert_eq!(clips.len(), 1, "budget of 1 must voice exactly one group");
        assert_eq!(
            clips[0], loud_clip,
            "the louder group (higher layer gain) must win, not the lower id"
        );
    }

    #[test]
    fn occurrence_source_medium_reaches_cross_liquid_response() {
        let (mut sound, syms, rec) = system(32);
        let cue = sound.catalog().typed::<OneShot>(&syms, "oneshot").unwrap();
        let mut listener = origin_listener();
        listener.medium = Medium::Liquid;
        let at = source(4.0);
        let occurrence = Occurrence {
            id: OccurrenceId(0),
            cue,
            at: Some(at),
            medium: Medium::Air,
            gain: 1.0,
        };
        sound.submit(
            AudioFrame::new(0.1, listener, vec![occurrence], vec![], open_window()).unwrap(),
        );

        let actual = played_dsps(&rec)[0].gain;
        let expected = respond(
            Response::World,
            SmoothedCoords::new(4.0 * crate::math::BLOCK_METERS as f32, 0.0, Medium::Air),
            &listener,
            Some(at),
        )
        .gain;
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual={actual} expected={expected}"
        );
    }

    // The journal is idempotent: an occurrence id at or below the high-water mark
    // must not re-realize. Reachable failure: the id/high-water guard is dropped
    // and the re-submitted occurrence plays a second clip.
    #[test]
    fn a3_occurrence_zero_is_realized_exactly_once() {
        let (mut sound, syms, rec) = system(32);
        let cue = sound.catalog().typed::<OneShot>(&syms, "oneshot").unwrap();
        let occ = || Occurrence {
            id: OccurrenceId(0),
            cue,
            at: Some(source(4.0)),
            medium: Medium::Air,
            gain: 1.0,
        };

        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![occ()], vec![], open_window()).unwrap(),
        );
        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![occ()], vec![], open_window()).unwrap(),
        );

        assert_eq!(
            count_play_clips(&rec),
            1,
            "id ≤ high_water must not re-realize"
        );
    }

    #[test]
    fn expired_ui_voice_is_stopped_before_the_next_click() {
        let (mut sound, syms, rec) = system(32);
        let cue = sound.catalog().typed::<OneShot>(&syms, "ui").unwrap();
        sound.ui_voices.push((
            super::BackendVoice(777),
            std::time::Instant::now() - std::time::Duration::from_millis(1),
        ));

        sound.play_ui(cue);

        assert!(
            rec.intents()
                .iter()
                .any(|intent| matches!(intent, Intent::Stop { v } if v.0 == 777)),
            "expired UI tracks must be explicitly reclaimed"
        );
        assert_eq!(count_play_clips(&rec), 1);
    }

    // The emitter table is a latest-wins snapshot: an emitter absent from this
    // frame's table has ceased and must be stopped. Reachable failure: the
    // absent-emitter release is skipped and no Stop is recorded (a leaked voice).
    #[test]
    fn a2_absent_emitter_is_stopped() {
        let (mut sound, syms, rec) = system(32);
        let bed = sound.catalog().typed::<Loop>(&syms, "bed").unwrap();
        let e = Emitter {
            id: EmitterId(0),
            cue: bed,
            at: source(5.0),
            medium: Medium::Air,
            gain: 1.0,
        };

        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![], vec![e], open_window()).unwrap(),
        );
        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![], vec![], open_window()).unwrap(),
        );

        let stops = rec
            .intents()
            .iter()
            .filter(|i| matches!(i, Intent::Stop { .. }))
            .count();
        assert_eq!(stops, 1, "an emitter absent from the table must be stopped");
    }

    #[test]
    fn layered_emitter_honors_every_layer_delay_gain_and_response() {
        let (mut sound, syms, rec) = system(32);
        let cue = sound.catalog().typed::<Loop>(&syms, "layered_bed").unwrap();
        let emitter = Emitter {
            id: EmitterId(4),
            cue,
            at: source(4.0),
            medium: Medium::Air,
            gain: 1.0,
        };

        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![], vec![emitter], open_window()).unwrap(),
        );
        assert_eq!(played_dsps(&rec).len(), 1, "zero-delay layer starts first");

        sound.submit(
            AudioFrame::new(0.2, origin_listener(), vec![], vec![emitter], open_window()).unwrap(),
        );
        let played = played_dsps(&rec);
        assert_eq!(played.len(), 2, "delayed second loop layer must start");

        let spatial = respond(
            Response::World,
            SmoothedCoords::new(4.0 * crate::math::BLOCK_METERS as f32, 0.0, Medium::Air),
            &origin_listener(),
            Some(source(4.0)),
        )
        .gain;
        assert!((played[0].gain - spatial * 0.5).abs() < 1e-5);
        assert!((played[1].gain - spatial * 1.5).abs() < 1e-5);
    }

    #[test]
    fn reusing_an_emitter_id_for_a_new_cue_stops_the_old_group() {
        let (mut sound, syms, rec) = system(32);
        let bed = sound.catalog().typed::<Loop>(&syms, "bed").unwrap();
        let layered = sound.catalog().typed::<Loop>(&syms, "layered_bed").unwrap();
        let make = |cue| Emitter {
            id: EmitterId(9),
            cue,
            at: source(3.0),
            medium: Medium::Air,
            gain: 1.0,
        };

        sound.submit(
            AudioFrame::new(
                0.1,
                origin_listener(),
                vec![],
                vec![make(bed)],
                open_window(),
            )
            .unwrap(),
        );
        sound.submit(
            AudioFrame::new(
                0.1,
                origin_listener(),
                vec![],
                vec![make(layered)],
                open_window(),
            )
            .unwrap(),
        );

        assert_eq!(count_play_clips(&rec), 2);
        assert_eq!(
            rec.intents()
                .iter()
                .filter(|intent| matches!(intent, Intent::Stop { .. }))
                .count(),
            1,
            "cue replacement must retire the prior loop"
        );
    }

    // A dead backend is a total no-op, yet the high-water mark still advances so a
    // returning backend never replays stale ids. Dead: id 5 records nothing.
    // Revived: id 5 (already past the water mark) stays silent while a fresh id 6
    // plays. Reachable failure: submit records on a dead backend, OR the
    // high-water is not advanced while dead and id 5 replays once the backend
    // returns.
    #[test]
    fn a12_dead_backend_no_ops_yet_advances_high_water() {
        let (mut sound, syms, rec) = system(32);
        let cue = sound.catalog().typed::<OneShot>(&syms, "oneshot").unwrap();
        let occ = |id| Occurrence {
            id: OccurrenceId(id),
            cue,
            at: Some(source(4.0)),
            medium: Medium::Air,
            gain: 1.0,
        };

        rec.set_alive(false);
        sound.submit(
            AudioFrame::new(0.1, origin_listener(), vec![occ(5)], vec![], open_window()).unwrap(),
        );
        assert!(
            rec.intents().is_empty(),
            "a dead backend records no intents"
        );

        rec.set_alive(true);
        sound.submit(
            AudioFrame::new(
                0.1,
                origin_listener(),
                vec![occ(5), occ(6)],
                vec![],
                open_window(),
            )
            .unwrap(),
        );
        assert_eq!(
            count_play_clips(&rec),
            1,
            "only the fresh id 6 plays; id 5 was past the dead-frame water mark"
        );
    }
}
