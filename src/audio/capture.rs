//! Capture local: mic → 48 kHz mono → 20 ms Opus frames.
//!
//! Capture knows nothing about sessions, epochs, or peers. It mints an ordered
//! journal of `EncodedFrame`s carrying only a `Seq`; the net layer stamps
//! id/epoch downstream. The pipeline is a strict SPSC chain so the realtime audio
//! callback never encodes, allocates, or locks:
//!
//!   cpal callback  --rtrb SPSC f32-->  encoder thread  --OutRing-->  drain() (main)
//!
//! The encoder thread owns all conversion (channel downmix already done in the
//! callback as cheap arithmetic), linear resampling, 20 ms framing, and Opus
//! encoding. All non-device logic is factored into pure structs (`Resampler`,
//! `EncodeCore`, `OutRing`) that the in-file tests exercise without a real device.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};

use super::voice::{MAX_VOICE_PAYLOAD, Seq};

/// Opus decode/encode operate at a fixed 48 kHz internally; we resample to it.
const DST_HZ: u32 = 48_000;
/// 20 ms @ 48 kHz mono = 960 samples per Opus frame.
const FRAME_SAMPLES: usize = 960;
/// Voice-appropriate constant target; one 20 ms frame stays well under
/// `MAX_VOICE_PAYLOAD` (400 B). Inband FEC is enabled for loss tolerance.
const BITRATE_BPS: i32 = 24_000;
/// Bound on the callback→encoder PCM ring in source-rate mono samples
/// (~0.1 s at 48 kHz). Overflow drops newest input (fire-and-forget); a full
/// ring means the encoder thread stalled, which the fault slot will surface.
const PCM_RING_SAMPLES: usize = 4800;
/// Bound on encoded output frames retained for `drain()` (~1 s of 20 ms frames).
const OUT_RING_FRAMES: usize = 50;
/// Encoder-thread idle nap when the PCM ring is momentarily empty. Far below the
/// 20 ms frame cadence, so it never delays a ready frame; rtrb offers no blocking
/// pop, so a short park is the simplest correct wait.
const ENCODER_NAP: Duration = Duration::from_millis(5);

#[derive(Clone, Copy, Debug)]
struct PcmSample {
    value: f32,
    epoch: u32,
}

/// One 20 ms Opus frame. `payload` is opaque codec bytes, `≤ MAX_VOICE_PAYLOAD`.
pub struct EncodedFrame {
    pub seq: Seq,
    pub payload: Box<[u8]>,
}

/// `None` device = system default input.
#[derive(Default)]
pub struct CaptureConfig {
    pub device: Option<String>,
}

#[derive(Debug)]
pub enum CaptureError {
    NoDevice,
    Stream(String),
    Encoder(String),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => formatter.write_str("no matching input device"),
            Self::Stream(reason) => write!(formatter, "input stream: {reason}"),
            Self::Encoder(reason) => write!(formatter, "Opus encoder: {reason}"),
        }
    }
}

impl std::error::Error for CaptureError {}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub struct Capture {
    // Declared first, but dropped explicitly-first in `Drop` so the callback
    // stops writing the PCM ring before the encoder thread is joined.
    stream: Option<cpal::Stream>,
    encoder: Option<JoinHandle<()>>,
    out: Arc<OutRing<EncodedFrame>>,
    /// Zero while closed; each PTT rising edge publishes a fresh non-zero epoch.
    /// Callback samples carry this tag so pre-PTT and prior-burst PCM cannot be
    /// encoded into a later talk burst.
    transmit_epoch: Arc<AtomicU32>,
    next_epoch: u32,
    shutdown: Arc<AtomicBool>,
    /// Detailed encoder-thread faults (off the RT path — a worker thread, not the
    /// audio callback).
    fault: Arc<Mutex<Option<CaptureError>>>,
    /// Device-loss latch set by cpal's error callback. A preallocated `AtomicBool`
    /// so that callback — invoked on the audio thread — neither allocates nor locks;
    /// the error detail is generic because the flag carries no payload.
    lost: Arc<AtomicBool>,
}

impl Capture {
    pub fn new(cfg: CaptureConfig) -> Result<Self, CaptureError> {
        let host = crate::audio::host::host();
        let device = match cfg.device {
            Some(ref want) => host
                .input_devices()
                .map_err(|e| CaptureError::Stream(e.to_string()))?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == want)
                        .unwrap_or(false)
                })
                .ok_or(CaptureError::NoDevice)?,
            None => host.default_input_device().ok_or(CaptureError::NoDevice)?,
        };

        let supported = device
            .default_input_config()
            .map_err(|e| CaptureError::Stream(e.to_string()))?;
        let src_hz = supported.sample_rate();
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        let config = supported.config();

        let (producer, consumer) = rtrb::RingBuffer::<PcmSample>::new(PCM_RING_SAMPLES);
        let fault = Arc::new(Mutex::new(None));
        let out = Arc::new(OutRing::new(OUT_RING_FRAMES));
        let transmit_epoch = Arc::new(AtomicU32::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let lost = Arc::new(AtomicBool::new(false));

        // Realtime path: downmix to mono f32 and push; no alloc, no lock, no encode.
        // cpal runs the error callback on the audio thread too, so it must be lock- and
        // alloc-free — just latch a preallocated flag; poll_fault reads it.
        let err_lost = lost.clone();
        let error_cb = move |_e: cpal::Error| {
            err_lost.store(true, Ordering::Relaxed);
        };
        let stream = match format {
            SampleFormat::F32 => build_stream::<f32>(
                &device,
                &config,
                channels,
                producer,
                transmit_epoch.clone(),
                error_cb,
            ),
            SampleFormat::I16 => build_stream::<i16>(
                &device,
                &config,
                channels,
                producer,
                transmit_epoch.clone(),
                error_cb,
            ),
            SampleFormat::U16 => build_stream::<u16>(
                &device,
                &config,
                channels,
                producer,
                transmit_epoch.clone(),
                error_cb,
            ),
            SampleFormat::I32 => build_stream::<i32>(
                &device,
                &config,
                channels,
                producer,
                transmit_epoch.clone(),
                error_cb,
            ),
            SampleFormat::F64 => build_stream::<f64>(
                &device,
                &config,
                channels,
                producer,
                transmit_epoch.clone(),
                error_cb,
            ),
            other => {
                return Err(CaptureError::Stream(format!(
                    "unsupported sample format {other:?}"
                )));
            }
        }
        .map_err(|e| CaptureError::Stream(e.to_string()))?;
        stream
            .play()
            .map_err(|e| CaptureError::Stream(e.to_string()))?;

        let encoder = spawn_encoder(
            src_hz,
            consumer,
            out.clone(),
            transmit_epoch.clone(),
            shutdown.clone(),
            fault.clone(),
        )?;

        Ok(Self {
            stream: Some(stream),
            encoder: Some(encoder),
            out,
            transmit_epoch,
            next_epoch: 0,
            shutdown,
            fault,
            lost,
        })
    }

    /// PTT edge; while `false` the encoder idles (no frames minted, `Seq` frozen).
    pub fn set_transmitting(&mut self, on: bool) {
        let active = self.transmit_epoch.load(Ordering::Acquire) != 0;
        match (on, active) {
            (true, false) => {
                self.next_epoch = self.next_epoch.wrapping_add(1).max(1);
                self.transmit_epoch
                    .store(self.next_epoch, Ordering::Release);
            }
            (false, true) => self.transmit_epoch.store(0, Ordering::Release),
            _ => {}
        }
    }

    /// Drain all pending frames; the bounded ring already dropped oldest on
    /// overflow (fire-and-forget), so callers see only the freshest journal tail.
    pub fn drain(&mut self) -> impl Iterator<Item = EncodedFrame> + '_ {
        self.out.drain().into_iter()
    }

    pub fn poll_fault(&mut self) -> Option<CaptureError> {
        // Device-loss latch first (set lock-free by the audio-thread error callback),
        // then the encoder thread's detailed fault.
        if self.lost.swap(false, Ordering::Relaxed) {
            return Some(CaptureError::Stream("input stream error".into()));
        }
        lock_recover(&self.fault).take()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // Stop the callback first so the encoder thread reaches a clean ring end,
        // then signal shutdown and join it.
        self.stream.take();
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.encoder.take() {
            let _ = h.join();
        }
    }
}

/// Monomorphized per device sample format; converts each interleaved frame to a
/// single mono `f32` and pushes it. Ring-full drops the sample (fire-and-forget).
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut producer: rtrb::Producer<PcmSample>,
    transmit_epoch: Arc<AtomicU32>,
    error_cb: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let inv = 1.0 / channels.max(1) as f32;
    device.build_input_stream(
        *config,
        move |data: &[T], _| {
            let epoch = transmit_epoch.load(Ordering::Acquire);
            if epoch == 0 {
                return;
            }
            for frame in data.chunks_exact(channels) {
                let mono: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() * inv;
                let _ = producer.push(PcmSample { value: mono, epoch });
            }
        },
        error_cb,
        None,
    )
}

fn spawn_encoder(
    src_hz: u32,
    mut consumer: rtrb::Consumer<PcmSample>,
    out: Arc<OutRing<EncodedFrame>>,
    transmit_epoch: Arc<AtomicU32>,
    shutdown: Arc<AtomicBool>,
    fault: Arc<Mutex<Option<CaptureError>>>,
) -> Result<JoinHandle<()>, CaptureError> {
    let mut encoder = opus_rs::OpusEncoder::new(DST_HZ as i32, 1, opus_rs::Application::Voip)
        .map_err(|e| CaptureError::Encoder(e.to_string()))?;
    encoder.bitrate_bps = BITRATE_BPS;
    encoder.use_inband_fec = true;

    let handle = std::thread::Builder::new()
        .name("voice-encoder".into())
        .spawn(move || {
            let mut core = EncodeCore::new(src_hz, DST_HZ);
            let mut pcm: Vec<f32> = Vec::with_capacity(PCM_RING_SAMPLES);
            let mut obuf = [0u8; MAX_VOICE_PAYLOAD];
            let mut encoding_epoch = 0;
            while !shutdown.load(Ordering::Relaxed) {
                let active_epoch = transmit_epoch.load(Ordering::Acquire);
                if active_epoch != encoding_epoch {
                    core.reset();
                    encoding_epoch = active_epoch;
                }
                pcm.clear();
                drain_epoch(&mut consumer, active_epoch, &mut pcm);
                if pcm.is_empty() {
                    std::thread::sleep(ENCODER_NAP);
                    continue;
                }
                core.feed(&pcm, true, |seq, frame| {
                    match encoder.encode(frame, FRAME_SAMPLES, &mut obuf) {
                        Ok(n) => out.push(EncodedFrame {
                            seq: Seq(seq),
                            payload: obuf[..n].to_vec().into_boxed_slice(),
                        }),
                        Err(e) => {
                            *lock_recover(&fault) = Some(CaptureError::Encoder(e.to_string()));
                        }
                    }
                });
            }
        })
        .map_err(|e| CaptureError::Encoder(e.to_string()))?;
    Ok(handle)
}

fn drain_epoch(consumer: &mut rtrb::Consumer<PcmSample>, active_epoch: u32, pcm: &mut Vec<f32>) {
    while let Ok(sample) = consumer.pop() {
        if active_epoch != 0 && sample.epoch == active_epoch {
            pcm.push(sample.value);
        }
    }
}

/// Continuous linear resampler (stateful across buffers via `frac`/`prev`).
///
/// Linear interpolation; sinc/polyphase is deferred because the audible gain
/// over voice-band mono at 48 kHz does not justify the cost/complexity now.
struct Resampler {
    /// Input samples consumed per output sample (`src / dst`).
    step: f64,
    /// Position within the current inter-sample interval, in [0, 1).
    frac: f64,
    /// Last input sample seen; the left anchor of the next interpolation.
    prev: f32,
}

impl Resampler {
    fn new(src_hz: u32, dst_hz: u32) -> Self {
        Self {
            step: src_hz as f64 / dst_hz as f64,
            frac: 0.0,
            prev: 0.0,
        }
    }

    fn reset(&mut self) {
        self.frac = 0.0;
        self.prev = 0.0;
    }

    /// Append resampled output for `input` to `out`. Each input sample advances
    /// the read cursor by one unit; outputs are emitted every `step` units by
    /// interpolating between `prev` and the incoming sample.
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        for &s in input {
            while self.frac < 1.0 {
                out.push(self.prev + (s - self.prev) * self.frac as f32);
                self.frac += self.step;
            }
            self.frac -= 1.0;
            self.prev = s;
        }
    }
}

/// Resample + 20 ms framing + `Seq` minting; the pure heart of the encoder loop.
/// The Opus encode itself is injected via the `emit` closure so this is testable
/// without a codec or device.
struct EncodeCore {
    resampler: Resampler,
    /// Resampled samples not yet forming a full 960-sample frame.
    buf: Vec<f32>,
    /// Wrapping frame counter; advances only when a frame is actually minted.
    seq: u32,
}

impl EncodeCore {
    fn new(src_hz: u32, dst_hz: u32) -> Self {
        Self {
            resampler: Resampler::new(src_hz, dst_hz),
            buf: Vec::with_capacity(FRAME_SAMPLES * 2),
            seq: 0,
        }
    }

    fn reset(&mut self) {
        self.resampler.reset();
        self.buf.clear();
    }

    /// Feed source-rate mono PCM. While `!transmitting`, input is discarded and
    /// all partial state is reset so the next talk burst starts clean and `Seq`
    /// does not advance. While transmitting, emit each completed frame.
    fn feed(&mut self, pcm: &[f32], transmitting: bool, mut emit: impl FnMut(u32, &[f32])) {
        if !transmitting {
            self.reset();
            return;
        }
        self.resampler.process(pcm, &mut self.buf);
        let mut frame = [0.0f32; FRAME_SAMPLES];
        while self.buf.len() >= FRAME_SAMPLES {
            frame.copy_from_slice(&self.buf[..FRAME_SAMPLES]);
            self.buf.drain(..FRAME_SAMPLES);
            let seq = self.seq;
            self.seq = self.seq.wrapping_add(1);
            emit(seq, &frame);
        }
    }
}

/// Bounded MPSC-free hand-off between the encoder thread (single producer) and
/// `drain()` (single consumer). Drop-oldest on overflow keeps latency bounded;
/// neither side is the realtime callback, so a mutex is acceptable here (rtrb's
/// SPSC ring cannot drop from the producer side).
struct OutRing<T> {
    q: Mutex<VecDeque<T>>,
    cap: usize,
}

impl<T> OutRing<T> {
    fn new(cap: usize) -> Self {
        Self {
            q: Mutex::new(VecDeque::with_capacity(cap)),
            cap,
        }
    }

    fn push(&self, v: T) {
        let mut q = lock_recover(&self.q);
        if q.len() >= self.cap {
            q.pop_front();
        }
        q.push_back(v);
    }

    fn drain(&self) -> VecDeque<T> {
        std::mem::take(&mut *lock_recover(&self.q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_doubles_sample_count_on_2x_upsample() {
        let mut r = Resampler::new(24_000, 48_000);
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut out = Vec::new();
        r.process(&input, &mut out);
        // 2x upsample yields ~2 outputs per input; allow ±2 for edge warm-up.
        assert!((out.len() as isize - 2000).abs() <= 2, "len={}", out.len());
        // A monotonic ramp stays monotonic through linear interpolation.
        assert!(out.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn resample_halves_sample_count_on_2x_downsample() {
        let mut r = Resampler::new(48_000, 24_000);
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut out = Vec::new();
        r.process(&input, &mut out);
        assert!((out.len() as isize - 500).abs() <= 2, "len={}", out.len());
    }

    #[test]
    fn framing_accumulates_across_odd_chunks() {
        // Identity rate so output count equals input count; framing must not
        // depend on how the input is chunked.
        let mut core = EncodeCore::new(48_000, 48_000);
        let mut frames = 0usize;
        let total = FRAME_SAMPLES * 3 + 100;
        let mut fed = 0;
        for chunk in [7usize, 13, 101, 960, 519] // odd, boundary-crossing sizes
            .iter()
            .cycle()
        {
            let n = (*chunk).min(total - fed);
            let pcm = vec![0.5f32; n];
            core.feed(&pcm, true, |_, _| frames += 1);
            fed += n;
            if fed >= total {
                break;
            }
        }
        // Identity resampler delays by one sample, so full frames = floor(total/960)
        // once warmed; 3*960+100 gives exactly 3.
        assert_eq!(frames, 3);
    }

    #[test]
    fn seq_freezes_and_resets_while_not_transmitting() {
        let mut core = EncodeCore::new(48_000, 48_000);
        let mut minted: Vec<u32> = Vec::new();

        // Not transmitting: heavy input, nothing minted, seq frozen.
        core.feed(&vec![1.0; 5000], false, |s, _| minted.push(s));
        assert!(minted.is_empty());

        // Turning on: a clean burst starts at seq 0 with no leftover from above.
        core.feed(&vec![1.0; FRAME_SAMPLES * 3], true, |s, _| minted.push(s));
        assert_eq!(minted, vec![0, 1, 2]);
    }

    #[test]
    fn prior_ptt_epoch_samples_never_enter_the_next_burst() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::new(8);
        producer
            .push(PcmSample {
                value: -1.0,
                epoch: 1,
            })
            .unwrap();
        producer
            .push(PcmSample {
                value: 0.25,
                epoch: 2,
            })
            .unwrap();
        producer
            .push(PcmSample {
                value: -2.0,
                epoch: 1,
            })
            .unwrap();
        producer
            .push(PcmSample {
                value: 0.5,
                epoch: 2,
            })
            .unwrap();

        let mut pcm = Vec::new();
        drain_epoch(&mut consumer, 2, &mut pcm);
        assert_eq!(pcm, vec![0.25, 0.5]);
    }

    #[test]
    fn out_ring_drops_oldest_on_overflow() {
        let ring = OutRing::<u32>::new(3);
        for v in 1..=5 {
            ring.push(v);
        }
        let got: Vec<u32> = ring.drain().into_iter().collect();
        assert_eq!(got, vec![3, 4, 5]);
    }
}
