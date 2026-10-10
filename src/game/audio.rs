//! The frame's audio commit: every peer sampled once for the frame, footsteps, the mods' audio
//! hooks and the mixer.
use voxel_engine::DVec3;

use super::{FrameInput, Game};
use crate::audio::{AudioService, AudioView, CueSymbols, GameEvent, ModLink, PeerAudio, SoundSystem, StepPose};
use crate::modding::Mods;
use crate::settings::Settings;

/// Facts for the audio hook. `events` is what this frame already knows; footsteps are added inside.
pub(super) struct AudioPhase<'a> {
    pub(super) dt: f32,
    pub(super) input: &'a FrameInput,
    pub(super) sound: &'a mut SoundSystem,
    pub(super) audio: &'a mut AudioService,
    pub(super) cues: &'a CueSymbols,
    pub(super) settings: &'a Settings,
    pub(super) events: Vec<GameEvent>,
    pub(super) active: bool,
    pub(super) mods: &'a mut Mods,
    pub(super) ids: &'a [&'static str],
}

/// One peer as sampled once this frame: audio and drawing read the same pose.
pub(super) struct PeerFrame {
    pub(super) id: u32,
    pub(super) rendered: crate::net::client::Rendered,
    pub(super) visible: bool,
}

impl Game {
    /// Sample every peer's pose once, at one instant, for this frame's audio and drawing.
    pub(super) fn sample_peers(&mut self) {
        self.peer_frames.clear();
        let Some(net) = &self.net else { return };
        let now = crate::sched::now();
        self.peer_frames.extend(net.peers().map(|peer| PeerFrame {
            id: peer.id(),
            rendered: peer.sample(now),
            visible: peer.visible(),
        }));
    }

    /// Hand this frame to the mods. A still singleplayer frame still runs the hook,
    /// on stack data, and skips the mixer unless the hook queued work.
    pub(super) fn commit_audio(&mut self, phase: AudioPhase<'_>) {
        let AudioPhase {
            dt,
            input,
            sound,
            audio,
            cues,
            settings,
            mut events,
            active,
            mods,
            ids,
        } = phase;
        let idle = events.is_empty() && input.actions.is_empty();
        if self.net.is_none() && audio.can_skip(sound, idle, self.player.position) {
            self.dispatch_audio(dt, input, sound, audio, cues, settings, mods, ids, &[], &events);
            if audio.dirty() {
                self.submit_audio(dt, sound, audio);
            } else {
                sound.poll_starvation();
            }
        } else {
            let mut peers = std::mem::take(&mut self.peer_pose_scratch);
            let mut ups = std::mem::take(&mut self.peer_up_scratch);
            peers.clear();
            ups.clear();
            for frame in &self.peer_frames {
                let r = &frame.rendered;
                peers.push(PeerAudio {
                    id: frame.id,
                    at: r.pos.0,
                    feet: r.pos.feet(r.stance, r.up).0,
                    visible: frame.visible,
                    gait: r.phase,
                    speed: r.speed,
                });
                ups.push(r.up);
            }
            // A frame a mod's text capture owns does not step the player, so a stale walk speed
            // must not fire a footstep.
            let velocity = if active { self.player.velocity() } else { DVec3::ZERO };
            audio.footsteps(
                StepPose {
                    feet: self.player.feet(),
                    velocity,
                    on_ground: self.player.on_ground(),
                    up: self.player.up_axis,
                },
                &self.world,
                dt,
                &peers,
                &ups,
                &mut events,
            );
            self.dispatch_audio(dt, input, sound, audio, cues, settings, mods, ids, &peers, &events);
            self.submit_audio(dt, sound, audio);
            self.peer_pose_scratch = peers;
            self.peer_up_scratch = ups;
        }
        events.clear();
        self.events_scratch = events;
    }

    fn dispatch_audio(
        &mut self,
        dt: f32,
        input: &FrameInput,
        sound: &mut SoundSystem,
        audio: &mut AudioService,
        cues: &CueSymbols,
        settings: &Settings,
        mods: &mut Mods,
        ids: &[&'static str],
        peers: &[PeerAudio],
        events: &[GameEvent],
    ) {
        let pos = self.player.position;
        let view = AudioView {
            dt,
            pos,
            peers,
            in_world: true,
            voice_enabled: settings.voice_enabled,
            hear_voice: settings.voice_incoming,
            actions: input.actions,
            ids,
        };
        let mut link = ModLink::new(self.net.as_mut());
        let mut api = audio.api(sound, cues, Some(&self.world), Some(&mut self.notices));
        for event in events {
            mods.on_game_event(event, &mut api);
        }
        mods.on_audio(&view, &mut api, &mut link);
    }

    fn submit_audio(&mut self, dt: f32, sound: &mut SoundSystem, audio: &mut AudioService) {
        let listener = crate::audio::Listener {
            pos: self.player.position,
            yaw: self.player.orientation.yaw,
            pitch: self.player.orientation.pitch,
            frame: self.player.orientation.frame,
        };
        audio.finish(sound, &self.world, listener, dt, &mut self.notices);
    }
}
