//! Placeholder audio asset generator. Writes short mono 48 kHz 16-bit WAVs into
//! `assets/sounds/`, one family per cue, so the catalog loader has real files to
//! decode before a proper audition track replaces them. Idempotent: every run
//! overwrites.
//!
//! Run: `cargo run --bin gen_tones`.

use std::f32::consts::TAU;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 48_000;

fn main() {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "assets", "sounds"]
        .iter()
        .collect();
    std::fs::create_dir_all(&dir).expect("create assets/sounds");

    let assets: Vec<(&str, Vec<i16>)> = vec![
        ("break_default_1.wav", noise_burst(0.15, 0.35, 1)),
        ("break_default_2.wav", noise_burst(0.15, 0.35, 2)),
        ("place_default_1.wav", tone_burst(140.0, 0.12, 0.30)),
        ("step_default_1.wav", noise_burst(0.07, 0.20, 3)),
        ("step_default_2.wav", noise_burst(0.07, 0.20, 4)),
        ("splash_1.wav", noise_burst(0.30, 0.18, 5)),
        ("swing_1.wav", sweep_burst(600.0, 180.0, 0.20)),
        ("menu_click_1.wav", tone_burst(1000.0, 0.04, 0.60)),
        ("voicetest_1.wav", tone_burst(440.0, 0.40, 0.40)),
        ("underwater_loop_1.wav", loop_noise(2.0, 0.20)),
    ];

    for (name, samples) in &assets {
        let path = dir.join(name);
        std::fs::write(&path, wav_bytes(samples)).expect("write wav");
        println!("wrote {} ({} samples)", path.display(), samples.len());
    }
}

/// Deterministic white noise in [-1, 1] from a splitmix-style step (no rng dep).
fn noise(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = (*state ^ (*state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn len_samples(secs: f32) -> usize {
    (secs * SAMPLE_RATE as f32) as usize
}

/// Attack-then-exponential-decay envelope, so bursts start clean and never click.
fn envelope(i: usize, n: usize) -> f32 {
    let t = i as f32 / n as f32;
    let attack = (t / 0.02).min(1.0); // ~first 2% ramps up
    attack * (-5.0 * t).exp()
}

fn tone_burst(freq: f32, secs: f32, amp: f32) -> Vec<i16> {
    let n = len_samples(secs);
    (0..n)
        .map(|i| {
            let phase = TAU * freq * i as f32 / SAMPLE_RATE as f32;
            quantize(amp * envelope(i, n) * phase.sin())
        })
        .collect()
}

fn noise_burst(secs: f32, amp: f32, seed: u64) -> Vec<i16> {
    let n = len_samples(secs);
    let mut state = seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(1);
    (0..n)
        .map(|i| quantize(amp * envelope(i, n) * noise(&mut state)))
        .collect()
}

/// Linear frequency glide — a crude whoosh for swing cues.
fn sweep_burst(f0: f32, f1: f32, secs: f32) -> Vec<i16> {
    let n = len_samples(secs);
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let freq = f0 + (f1 - f0) * t;
            phase += TAU * freq / SAMPLE_RATE as f32;
            quantize(0.30 * envelope(i, n) * phase.sin())
        })
        .collect()
}

/// Constant-amplitude low-passed noise with no end fades, so the buffer loops
/// seamlessly (ambient beds are pulled as `ClipMode::Loop`).
fn loop_noise(secs: f32, amp: f32) -> Vec<i16> {
    let n = len_samples(secs);
    let mut state = 0xABCD_1234_5678_9F01u64;
    let mut lp = 0.0f32; // one-pole low-pass for a muffled underwater timbre
    (0..n)
        .map(|i| {
            lp += 0.02 * (noise(&mut state) - lp);
            // Blend the two ends over a short window to hide the seam entirely.
            let cross = 4_800.min(n / 4);
            let mix = if i < cross {
                0.5 + 0.5 * (i as f32 / cross as f32)
            } else {
                1.0
            };
            quantize(amp * lp * mix)
        })
        .collect()
}

fn quantize(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// 44-byte canonical PCM WAV header + interleaved (mono) little-endian samples.
fn wav_bytes(samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let byte_rate = SAMPLE_RATE * 2;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align (mono 16-bit)
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}
