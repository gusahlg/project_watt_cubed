//! Device-less backend.
//!
//! It never plays, but it still decodes every catalog clip. That distinction
//! matters on headless CI: a missing audio device must not let corrupt packaged
//! assets pass validation merely because there is nowhere to play them.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use voxel_engine::DVec3;
use kira::sound::static_sound::StaticSoundData;

use super::{Backend, BackendVoice, ClipId, ClipStore, StoredClip};
use crate::audio::acoustics::{Dsp, Listener};
use crate::audio::voice::VoicePacket;

pub(crate) struct NullBackend {
    next_clip: u32,
}

impl NullBackend {
    pub(crate) fn new() -> Self {
        Self { next_clip: 0 }
    }
}

impl ClipStore for NullBackend {
    fn store(&mut self, bytes: &[u8]) -> Result<StoredClip, String> {
        let data = StaticSoundData::from_cursor(Cursor::new(bytes.to_vec()))
            .map_err(|error| format!("clip decode: {error:?}"))?;
        let stored = StoredClip {
            id: ClipId(self.next_clip),
            duration_s: data.duration().as_secs_f32(),
        };
        self.next_clip = self
            .next_clip
            .checked_add(1)
            .ok_or_else(|| "too many decoded audio clips".to_owned())?;
        Ok(stored)
    }
}

impl Backend for NullBackend {
    fn play_clip(
        &mut self,
        _clip: ClipId,
        _dsp: Dsp,
        _rate: f32,
        _looped: bool,
        _spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        None
    }

    fn play_stream(
        &mut self,
        _feed: rtrb::Consumer<VoicePacket>,
        _jitter_target_ms: u32,
        _starved: Arc<AtomicBool>,
        _spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        None
    }

    fn update(
        &mut self,
        _voice: BackendVoice,
        _dsp: Dsp,
        _spatial: Option<DVec3>,
        _listener: &Listener,
    ) {
    }

    fn stop(&mut self, _voice: BackendVoice) {}

    fn set_master(&mut self, _master: f32) {}

    fn alive(&self) -> bool {
        false
    }
}
