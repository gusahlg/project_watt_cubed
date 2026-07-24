//! kira 0.12 backend — the operational half of the audio path.
//!
//! Two responsibilities: [`KiraBackend`] realizes the [`Backend`] trait onto kira's
//! mixer, and [`VoiceDecoder`] implements kira's streaming [`Decoder`] to turn an rtrb
//! packet feed into gapless PCM (jitter reorder → opus decode/PLC → silence/finish).
//!
//! Load-bearing divergences from kira's assumed shape:
//!  * kira's `Decoder` is finite-oriented (`num_frames()` bounds a transport). A live
//!    voice stream reports `num_frames() == usize::MAX` and signals end by returning
//!    `Err` from `decode()`, which kira treats as "stop the sound" (verified against
//!    kira's `streaming_sound_stops_on_error` test). This is how an abandoned feed frees
//!    the voice with no leak.
//!  * kira spatializes per *track* (listener + position → attenuation/pan), but the
//!    whole nonlinearity lives in `respond()` upstream. So each voice gets a plain
//!    (non-spatial) sub-track and we drive `Dsp` directly: gain → track volume, lowpass →
//!    a `Filter` effect, pan → a `PanningControl` effect. `spatial`/`listener` args are
//!    accepted for trait conformance but unused here (they matter to a spatializing
//!    fallback backend).
//!  * Streaming/static sound *handles* expose no live volume/pan setters in 0.12, which
//!    is why modulation lives on the per-voice track + effects rather than the sound.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::DVec3;
use kira::{
    AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Frame, Panning, Tween,
    effect::{
        filter::{FilterBuilder, FilterHandle, FilterMode},
        panning_control::{PanningControlBuilder, PanningControlHandle},
    },
    sound::{
        static_sound::{StaticSoundData, StaticSoundHandle},
        streaming::{Decoder, StreamingSoundData, StreamingSoundHandle},
    },
    track::{TrackBuilder, TrackHandle},
};
use rtrb::Consumer;

use super::super::acoustics::{Dsp, Listener};
use super::super::voice::{
    JitterBuffer, PlayoutStep, VOICE_FRAME_SAMPLES, VOICE_SAMPLE_RATE, VoicePacket,
};
use super::{Backend, BackendVoice, ClipId, ClipStore, StoredClip};

/// Anti-zipper gauge tween on backend `Dsp` outputs — declared ≤ 10 ms; never a
/// substitute for coordinate smoothing, which happens upstream.
const GAUGE_TWEEN: Tween = Tween {
    start_time: kira::StartTime::Immediate,
    duration: std::time::Duration::from_millis(8),
    easing: kira::Easing::Linear,
};

pub(crate) struct KiraBackend {
    manager: AudioManager,
    clips: Vec<StaticSoundData>,
    voices: HashMap<u64, LiveVoice>,
    next_voice: u64,
    /// Factory for the per-stream opus decoder (see [`OpusDecode`]).
    opus_factory: fn() -> Box<dyn OpusDecode>,
    alive: bool,
}

/// A realized backend voice: its own sub-track plus the effect handles that carry `Dsp`.
/// The sound handle is retained so the voice can be stopped and so dropping it does not
/// end the sound prematurely.
struct LiveVoice {
    track: TrackHandle,
    filter: FilterHandle,
    pan: PanningControlHandle,
    sound: SoundHandle,
}

enum SoundHandle {
    Clip(StaticSoundHandle),
    Stream(StreamingSoundHandle<VoiceFinished>),
}

impl KiraBackend {
    pub fn new() -> Option<Self> {
        // No kira listener/spatial track: the full spatial nonlinearity lives in
        // `respond()` upstream, so voices ride plain sub-tracks driven by `Dsp`. This also
        // keeps the backend free of glam→kira coupling (kira's mint APIs want glam 0.33).
        let manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).ok()?;
        Some(Self {
            manager,
            clips: Vec::new(),
            voices: HashMap::new(),
            next_voice: 0,
            opus_factory: make_opus,
            alive: true,
        })
    }

    fn mint(&mut self) -> u64 {
        let id = self.next_voice;
        self.next_voice += 1;
        id
    }

    /// A fresh sub-track carrying the initial `Dsp` as track volume + lowpass + pan.
    fn open_track(
        &mut self,
        dsp: &Dsp,
    ) -> Option<(TrackHandle, FilterHandle, PanningControlHandle)> {
        let mut builder = TrackBuilder::new();
        let pan = builder.add_effect(PanningControlBuilder(pan_scalar(dsp).into()));
        let filter = builder.add_effect(
            FilterBuilder::new()
                .mode(FilterMode::LowPass)
                .cutoff(dsp.lowpass_hz as f64),
        );
        let builder = builder.volume(gain_db(dsp.gain));
        let track = self.manager.add_sub_track(builder).ok()?;
        Some((track, filter, pan))
    }
}

impl ClipStore for KiraBackend {
    fn store(&mut self, bytes: &[u8]) -> Result<StoredClip, String> {
        let data = StaticSoundData::from_cursor(Cursor::new(bytes.to_vec()))
            .map_err(|e| format!("clip decode: {e:?}"))?;
        let duration_s = data.duration().as_secs_f32();
        let id = ClipId(
            u32::try_from(self.clips.len())
                .map_err(|_| "too many decoded audio clips".to_owned())?,
        );
        self.clips.push(data);
        Ok(StoredClip { id, duration_s })
    }
}

impl Backend for KiraBackend {
    fn play_clip(
        &mut self,
        clip: ClipId,
        dsp: Dsp,
        rate: f32,
        looped: bool,
        _spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        if !self.alive {
            return None;
        }
        let data = self.clips.get(clip.0 as usize)?.clone();
        let (mut track, filter, pan) = self.open_track(&dsp)?;
        let data = data.playback_rate(rate as f64);
        let data = if looped {
            data.loop_region(0.0..)
        } else {
            data
        };
        let sound = track.play(data).ok()?;
        let id = self.mint();
        self.voices.insert(
            id,
            LiveVoice {
                track,
                filter,
                pan,
                sound: SoundHandle::Clip(sound),
            },
        );
        Some(BackendVoice(id))
    }

    fn play_stream(
        &mut self,
        feed: Consumer<VoicePacket>,
        jitter_target_ms: u32,
        starved: Arc<AtomicBool>,
        _spatial: Option<DVec3>,
        _listener: &Listener,
    ) -> Option<BackendVoice> {
        if !self.alive {
            return None;
        }
        // Opened muted: the runtime unmutes via update() once the peer is presented.
        let dsp = Dsp {
            gain: 0.0,
            lowpass_hz: 20_000.0,
            pan: None,
        };
        let (mut track, filter, pan) = self.open_track(&dsp)?;
        let decoder = VoiceDecoder {
            feed,
            jitter: JitterBuffer::new(jitter_target_ms),
            opus: (self.opus_factory)(),
            starved,
        };
        let sound = track.play(StreamingSoundData::from_decoder(decoder)).ok()?;
        let id = self.mint();
        self.voices.insert(
            id,
            LiveVoice {
                track,
                filter,
                pan,
                sound: SoundHandle::Stream(sound),
            },
        );
        Some(BackendVoice(id))
    }

    fn update(&mut self, v: BackendVoice, dsp: Dsp, _spatial: Option<DVec3>, _listener: &Listener) {
        if let Some(voice) = self.voices.get_mut(&v.0) {
            voice.track.set_volume(gain_db(dsp.gain), GAUGE_TWEEN);
            voice.filter.set_cutoff(dsp.lowpass_hz as f64, GAUGE_TWEEN);
            voice.pan.set_panning(pan_scalar(&dsp), GAUGE_TWEEN);
        }
    }

    fn stop(&mut self, v: BackendVoice) {
        if let Some(mut voice) = self.voices.remove(&v.0) {
            match &mut voice.sound {
                SoundHandle::Clip(h) => h.stop(GAUGE_TWEEN),
                SoundHandle::Stream(h) => h.stop(GAUGE_TWEEN),
            }
            // Track/effect handles drop here; the sub-track is reclaimed once silent.
        }
    }

    fn set_master(&mut self, master: f32) {
        self.manager
            .main_track()
            .set_volume(gain_db(master), GAUGE_TWEEN);
    }

    fn alive(&self) -> bool {
        self.alive
    }
}

/// Amplitude (0..=1 nominal) → decibels; 0 collapses to kira's silence floor.
fn gain_db(gain: f32) -> Decibels {
    if gain <= 1e-4 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * gain.log10())
    }
}

/// Collapse `Dsp`'s 3-vector pan to kira's scalar L/R panning (x = listener-right axis).
fn pan_scalar(dsp: &Dsp) -> Panning {
    match dsp.pan {
        Some([x, _, _]) => Panning(x.clamp(-1.0, 1.0)),
        None => Panning::CENTER,
    }
}

/// The decode step, traited out so [`JitterBuffer`] logic (in `voice.rs`) stays testable
/// without a real opus decoder and so the concrete codec can be swapped without touching
/// the jitter/kira wiring. Output is mono f32 PCM (`VOICE_FRAME_SAMPLES` samples), the
/// native form of both opus_rs and kira's `Frame`, so no lossy i16 round-trip.
pub(crate) trait OpusDecode: Send {
    /// Decode one 20 ms opus frame into mono f32 PCM.
    fn decode(&mut self, payload: &[u8], out: &mut Vec<f32>);
    /// Packet-loss concealment for one missing 20 ms frame.
    fn conceal(&mut self, out: &mut Vec<f32>);
}

/// Production decoder factory: a real opus_rs decoder, falling back to silence if the
/// decoder cannot be constructed (never a panic across the seam). A free `fn` so it
/// coerces to the `opus_factory` function pointer with no captures. Tests substitute
/// [`SilentOpus`] via the factory field.
fn make_opus() -> Box<dyn OpusDecode> {
    match RealOpus::new() {
        Ok(d) => Box::new(d),
        Err(_) => Box::new(SilentOpus),
    }
}

/// Silence decoder — used as the construction-failure fallback and by tests that need
/// a decoder without a real codec. Every frame is `VOICE_FRAME_SAMPLES` zeros.
struct SilentOpus;

impl OpusDecode for SilentOpus {
    fn decode(&mut self, _payload: &[u8], out: &mut Vec<f32>) {
        out.clear();
        out.resize(VOICE_FRAME_SAMPLES, 0.0);
    }
    fn conceal(&mut self, out: &mut Vec<f32>) {
        out.clear();
        out.resize(VOICE_FRAME_SAMPLES, 0.0);
    }
}

/// Real pure-Rust opus decode via `opus_rs` (48 kHz mono).
///
/// `OpusDecoder::decode(input, frame_size, output: &mut [f32]) -> Result<usize,
/// &'static str>` rejects an empty packet — it offers no null-packet PLC entry
/// (unlike libopus). So concealment is synthesized here:
/// the last good frame is replayed with a per-conceal energy decay (halving), which masks
/// short gaps far better than hard silence and decays to silence over a burst. The
/// `JitterBuffer` already bounds consecutive conceals (`PLC_MAX_CONSECUTIVE`) upstream.
struct RealOpus {
    dec: opus_rs::OpusDecoder,
    /// Last successfully decoded frame, retained for synthesized concealment.
    last: Vec<f32>,
}

impl RealOpus {
    fn new() -> Result<Self, &'static str> {
        Ok(Self {
            dec: opus_rs::OpusDecoder::new(VOICE_SAMPLE_RATE as i32, 1)?,
            last: Vec::new(),
        })
    }
}

impl OpusDecode for RealOpus {
    fn decode(&mut self, payload: &[u8], out: &mut Vec<f32>) {
        out.clear();
        out.resize(VOICE_FRAME_SAMPLES, 0.0);
        match self.dec.decode(payload, VOICE_FRAME_SAMPLES, out) {
            Ok(n) => {
                out.truncate(n);
                self.last.clear();
                self.last.extend_from_slice(out);
            }
            // Decode fault degrades to silence, never a panic across the seam.
            Err(_) => out.iter_mut().for_each(|s| *s = 0.0),
        }
    }

    fn conceal(&mut self, out: &mut Vec<f32>) {
        out.clear();
        if self.last.is_empty() {
            out.resize(VOICE_FRAME_SAMPLES, 0.0);
            return;
        }
        out.extend_from_slice(&self.last);
        // Decay the retained frame so successive conceals fade toward silence (standard
        // cheap decay-repeat PLC; opus_rs offers no decoder-side concealment).
        self.last.iter_mut().for_each(|s| *s *= 0.7);
    }
}

/// Reported from `decode()` when the feed is abandoned and drained; kira stops the sound
/// on a decoder error, which frees the voice instead of leaving it playing forever.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct VoiceFinished;

/// kira streaming decoder wrapping the jitter buffer and opus decode step. One
/// `decode()` call yields exactly one 20 ms chunk pulled from the playout clock.
pub(crate) struct VoiceDecoder {
    feed: Consumer<VoicePacket>,
    jitter: JitterBuffer,
    opus: Box<dyn OpusDecode>,
    /// Cross-thread mirror of the jitter buffer's starvation state (runtime polls it).
    starved: Arc<AtomicBool>,
}

impl Decoder for VoiceDecoder {
    type Error = VoiceFinished;

    fn sample_rate(&self) -> u32 {
        VOICE_SAMPLE_RATE
    }

    fn num_frames(&self) -> usize {
        usize::MAX // open-ended live stream; end is signalled via decode() Err
    }

    fn decode(&mut self) -> Result<Vec<Frame>, VoiceFinished> {
        while let Ok(pkt) = self.feed.pop() {
            self.jitter.push(pkt);
        }
        let abandoned = self.feed.is_abandoned();

        let mut pcm: Vec<f32> = Vec::with_capacity(VOICE_FRAME_SAMPLES);
        let step = self.jitter.pull(abandoned);
        let recovered = matches!(step, PlayoutStep::Decode(_));
        match step {
            PlayoutStep::Decode(payload) => self.opus.decode(payload, &mut pcm),
            PlayoutStep::Conceal => self.opus.conceal(&mut pcm),
            PlayoutStep::Silence => pcm.resize(VOICE_FRAME_SAMPLES, 0.0),
            PlayoutStep::Finished => return Err(VoiceFinished),
        }
        // Raise on PLC-budget exhaustion; clear once real audio flows again (Relaxed: a
        // per-frame status bit, not a synchronization edge).
        if self.jitter.take_starved() {
            self.starved.store(true, Ordering::Relaxed);
        } else if recovered {
            self.starved.store(false, Ordering::Relaxed);
        }
        Ok(pcm.iter().map(|&s| Frame::from_mono(s)).collect())
    }

    fn seek(&mut self, index: usize) -> Result<usize, VoiceFinished> {
        Ok(index) // voice is not seekable; kira issues no seeks without loop/seek commands
    }
}
