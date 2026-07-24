//! Placeholder audio asset generator. Writes short mono 48 kHz 16-bit WAVs into
//! `assets/sounds/generated/`, one family per cue, so the catalog loader has real
//! files to decode before proper recordings replace them.
//!
//! Existing files are protected unless `--force` is present.
//!
//! Run: `cargo run --features dev-tools --bin gen_tones -- --force`

use std::f32::consts::TAU;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 48_000;

fn main() {
    let (dir, force) = arguments();
    std::fs::create_dir_all(&dir).expect("create placeholder sound directory");

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
        ("underwater_loop_1.wav", loop_bed(2.0, 0.20)),
    ];

    for (name, samples) in &assets {
        let path = dir.join(name);
        if path.exists() && !force {
            println!("kept {} (pass --force to replace)", path.display());
            continue;
        }
        std::fs::write(&path, wav_bytes(samples)).expect("write wav");
        println!("wrote {} ({} samples)", path.display(), samples.len());
    }
}

fn arguments() -> (PathBuf, bool) {
    let mut force = false;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--force") => force = true,
            Some("--out") => {
                output = Some(
                    args.next()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| usage("missing path after --out")),
                );
            }
            _ => usage(&format!("unknown argument `{}`", arg.to_string_lossy())),
        }
    }
    let default: PathBuf = [env!("CARGO_MANIFEST_DIR"), "assets", "sounds", "generated"]
        .iter()
        .collect();
    (output.unwrap_or(default), force)
}

fn usage(reason: &str) -> ! {
    eprintln!("{reason}");
    eprintln!("usage: gen_tones [--force] [--out DIRECTORY]");
    std::process::exit(2);
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

/// A periodic low bed. Every component completes an integer number of cycles,
/// including its amplitude modulation, so the last→first jump is just another
/// sample step rather than a random-noise discontinuity.
fn loop_bed(secs: f32, amp: f32) -> Vec<i16> {
    let n = len_samples(secs);
    (0..n)
        .map(|i| {
            let phase = TAU * i as f32 / n as f32;
            let carrier = 0.55 * (phase * 74.0).sin()
                + 0.30 * (phase * 107.0 + 0.4).sin()
                + 0.15 * (phase * 151.0 + 1.1).sin();
            let swell = 0.78 + 0.22 * (phase * 2.0).sin();
            quantize(amp * swell * carrier)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looping_bed_has_no_boundary_click() {
        let samples = loop_bed(2.0, 0.2);
        let boundary = (i32::from(samples[0]) - i32::from(samples[samples.len() - 1])).abs();
        let largest_step = samples
            .windows(2)
            .map(|pair| (i32::from(pair[1]) - i32::from(pair[0])).abs())
            .max()
            .unwrap();
        assert!(
            boundary <= largest_step + 1,
            "loop boundary jump {boundary} exceeds ordinary step {largest_step}"
        );
    }

    #[test]
    fn wav_header_matches_payload() {
        let wav = wav_bytes(&[1, -2, 3]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(wav.len(), 50);
    }
}
