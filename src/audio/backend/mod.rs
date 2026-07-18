//! Audio backend seam. The backend earns a vtable, not authority: all semantic
//! state stays in `SoundSystem`; kira/oddio/a mock implement `Backend`.

pub(crate) mod kira;
#[cfg(test)]
pub(crate) mod recording;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use glam::DVec3;

use super::acoustics::{Dsp, Listener};
use super::voice::VoicePacket;

/// Handle to a live backend voice (clip layer or stream): a bare monotone counter
/// minted per voice and never reused. Staleness needs no generation field — an id
/// whose voice was stopped is simply absent from the backend's live map, so a stale
/// handle looks up nothing and is inert.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct BackendVoice(pub u64);

/// Load-time decode target. `content.rs` folds catalog bytes into `ClipId`s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ClipId(pub u32);

/// A decoded, retained clip. `duration_s` is the `max_duration` fold's clip term
/// (content.rs computes `duration / min_rate`); without it zero-delay one-shots would
/// fold to `max_duration = 0` and release instantly.
pub(crate) struct StoredClip {
    pub id: ClipId,
    pub duration_s: f32,
}

/// Decode-and-retain a clip's PCM; returns the stored clip or a human-readable reason.
pub(crate) trait ClipStore {
    fn store(&mut self, bytes: &[u8]) -> Result<StoredClip, String>;
}

/// The playback surface. Every method is total and non-panicking across the seam:
/// device/decoder death degrades to `alive() == false`, never a panic.
pub(crate) trait Backend: ClipStore {
    /// The returned voice MUST be retained (registry or state cell) so it can be
    /// stopped; dropping it silently leaks the voice past the budget.
    #[must_use]
    fn play_clip(
        &mut self,
        clip: ClipId,
        dsp: Dsp,
        rate: f32,
        looped: bool,
        spatial: Option<DVec3>,
        listener: &Listener,
    ) -> Option<BackendVoice>;

    /// Spatial stream fed by a packet queue. The `VoiceDecoder` (jitter + PLC + opus) is
    /// built here and pulled on the backend's decode thread. Opened immediately when a
    /// session opens (initially muted via `Dsp{gain: 0.0}`) so the decoder outlives
    /// presentation toggles. `starved` is the runtime-owned flag the decoder raises on
    /// PLC-budget exhaustion and clears on recovery — the runtime keeps a clone and
    /// polls it per frame to raise `Fault::Starved` across the decode-thread seam.
    fn play_stream(
        &mut self,
        feed: rtrb::Consumer<VoicePacket>,
        jitter_target_ms: u32,
        starved: Arc<AtomicBool>,
        spatial: Option<DVec3>,
        listener: &Listener,
    ) -> Option<BackendVoice>;

    fn update(&mut self, v: BackendVoice, dsp: Dsp, spatial: Option<DVec3>, listener: &Listener);
    fn stop(&mut self, v: BackendVoice);
    fn set_master(&mut self, master: f32);
    /// Whether this backend can produce sound. `false` for the `NullBackend` chosen
    /// when no device was available at construction, so the runtime degrades to
    /// silence. NOTE: the kira backend cannot observe *runtime* device loss — kira
    /// 0.12 owns cpal internally and exposes no device-error callback — so its
    /// `alive()` stays `true` for its lifetime (needs a kira API upstream).
    fn alive(&self) -> bool;
}
