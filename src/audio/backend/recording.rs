//! Bisimulation witness backend. Records intents against a monotone step counter
//! — no wall clock, no real device — so the runtime's ordering, atomicity, and
//! containment properties are observable in unit tests.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::DVec3;

use super::super::acoustics::{Dsp, Listener};
use super::super::voice::VoicePacket;
use super::{Backend, BackendVoice, ClipId, ClipStore, StoredClip};

/// One recorded backend command. `at` collapses spatial position to the observable
/// datum; `listener` is intentionally omitted (the runtime already applied it via `Dsp`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Intent {
    PlayClip {
        clip: ClipId,
        dsp: Dsp,
        rate: f32,
        looped: bool,
        at: Option<DVec3>,
    },
    PlayStream {
        at: Option<DVec3>,
        jitter_target_ms: u32,
    },
    Update {
        v: BackendVoice,
        dsp: Dsp,
        at: Option<DVec3>,
    },
    Stop {
        v: BackendVoice,
    },
    SetMaster {
        master: f32,
    },
}

/// The backend the `SoundSystem` owns; its recorded intents and its `alive` flag are
/// shared (`Arc`) with a [`Recorder`] the test keeps, so the test observes the seam
/// after driving the system and can flip `alive` to simulate device loss.
pub(crate) struct RecordingBackend {
    log: Arc<Mutex<Vec<Intent>>>,
    alive: Arc<AtomicBool>,
    next_voice: u64,
    next_clip: u32,
}

/// Test-side observer of a `RecordingBackend`: reads the intent log (in emission
/// order) and controls the `alive` flag.
pub(crate) struct Recorder {
    log: Arc<Mutex<Vec<Intent>>>,
    alive: Arc<AtomicBool>,
}

impl Recorder {
    pub fn intents(&self) -> Vec<Intent> {
        self.log.lock().unwrap().clone()
    }
    pub fn set_alive(&self, alive: bool) {
        self.alive.store(alive, Ordering::Relaxed);
    }
}

impl RecordingBackend {
    pub fn new() -> (Self, Recorder) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let backend = Self { log: log.clone(), alive: alive.clone(), next_voice: 0, next_clip: 0 };
        (backend, Recorder { log, alive })
    }

    fn record(&mut self, intent: Intent) {
        self.log.lock().unwrap().push(intent);
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    fn mint_voice(&mut self) -> BackendVoice {
        let v = BackendVoice(self.next_voice);
        self.next_voice += 1;
        v
    }
}

impl ClipStore for RecordingBackend {
    fn store(&mut self, _bytes: &[u8]) -> Result<StoredClip, String> {
        let id = ClipId(self.next_clip);
        self.next_clip += 1;
        Ok(StoredClip { id, duration_s: 1.0 })
    }
}

impl Backend for RecordingBackend {
    fn play_clip(
        &mut self,
        clip: ClipId,
        dsp: Dsp,
        rate: f32,
        looped: bool,
        spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        if !self.is_alive() {
            return None; // dead backend is a total no-op
        }
        self.record(Intent::PlayClip { clip, dsp, rate, looped, at: spatial });
        Some(self.mint_voice())
    }

    fn play_stream(
        &mut self,
        _feed: rtrb::Consumer<VoicePacket>,
        jitter_target_ms: u32,
        _starved: Arc<AtomicBool>,
        spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        if !self.is_alive() {
            return None;
        }
        self.record(Intent::PlayStream { at: spatial, jitter_target_ms });
        Some(self.mint_voice())
    }

    fn update(&mut self, v: BackendVoice, dsp: Dsp, spatial: Option<DVec3>, _listener: &Listener) {
        if !self.is_alive() {
            return;
        }
        self.record(Intent::Update { v, dsp, at: spatial });
    }

    fn stop(&mut self, v: BackendVoice) {
        if !self.is_alive() {
            return;
        }
        self.record(Intent::Stop { v });
    }

    fn set_master(&mut self, master: f32) {
        if !self.is_alive() {
            return;
        }
        self.record(Intent::SetMaster { master });
    }

    fn alive(&self) -> bool {
        self.is_alive()
    }
}
