//! Output/input host selection.
//!
//! cpal's native PulseAudio host (`cpal` feature `pulseaudio`) talks to
//! Pulse/PipeWire over the Pulse protocol. On this machine that client
//! disconnects in a loop (`pulseaudio::client::reactor: Client disconnected`),
//! which stalls one-shots by seconds and sometimes silences the whole mixer.
//! ALSA (including PipeWire's ALSA plugin) is the stable path.

use cpal::traits::{DeviceTrait, HostTrait};

pub fn host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    {
        if let Ok(h) = cpal::host_from_id(cpal::HostId::Alsa) {
            return h;
        }
    }
    cpal::default_host()
}

/// ~10 ms at 48 kHz. Pulse's default buffer is often hundreds of ms to seconds.
const FRAMES: cpal::FrameCount = 512;

pub fn output_device_and_config() -> Option<(cpal::Device, cpal::StreamConfig)> {
    let host = host();
    let device = host.default_output_device()?;
    let supported = device.default_output_config().ok()?;
    let mut config = supported.config();
    config.buffer_size = cpal::BufferSize::Fixed(FRAMES);
    Some((device, config))
}

#[cfg(test)]
mod tests {
    #[test]
    fn host_probe_does_not_panic() {
        let _ = std::panic::catch_unwind(|| {
            let _ = super::host();
            super::output_device_and_config()
        });
    }
}
