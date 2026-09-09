//! The Game → runtime snapshot (`AudioFrame`) plus the occurrence and emitter
//! records it bundles. Construction is the sole validation site: a well-typed
//! `AudioFrame` is already checked (parse-don't-validate), so `SoundSystem::submit`
//! can trust its contents and move the Vecs out without a clone.

use std::sync::Arc;

use voxel_engine::DVec3;

use super::acoustics::AcousticWindow;
use super::content::{CueId, Loop, OneShot};

pub use super::acoustics::{Listener, Medium};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OccurrenceId(pub u64); // monotone per world session; minted by Game
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EmitterId(pub u64); // stable while the emitter exists; Ord for GroupKey ranking

#[derive(Clone, Copy, Debug)]
pub struct Occurrence {
    pub id: OccurrenceId,
    pub cue: CueId<OneShot>, // one-shot by type: max_duration is finite, so the
    // journal release guard can never leak a never-ending voice
    pub at: Option<DVec3>, // None = non-spatial (UI)
    pub medium: Medium,
    pub gain: f32, // authored scale, finite, [0, 4]
}

#[derive(Clone, Copy, Debug)]
pub struct Emitter {
    pub id: EmitterId,
    pub cue: CueId<Loop>, // Loop-capable by type, not a submit-time check
    pub at: DVec3,
    pub medium: Medium,
    pub gain: f32,
}

pub struct AudioFrame {
    dt: f32,
    listener: Listener,
    occurrences: Vec<Occurrence>, // ordered; ids strictly increasing within the vec
    emitters: Vec<Emitter>,       // complete table; absence = ceased
    /// `None` when nothing this frame (and no live voices) will trace it.
    window: Option<Arc<AcousticWindow>>,
}

pub const MAX_OCCURRENCES: usize = 256;
pub const MAX_EMITTERS: usize = 256;

#[derive(Debug)]
pub enum FrameError {
    NonFinite,
    OutOfRange,
    UnorderedOccurrences,
    DuplicateEmitter,
    OverLimit,
}

const MAX_GAIN: f32 = 4.0;

// Range check only; callers test `is_finite` first so NaN reports NonFinite, not
// OutOfRange (error precedence).
fn gain_in_range(g: f32) -> bool {
    (0.0..=MAX_GAIN).contains(&g)
}

impl AudioFrame {
    /// Sole constructor. Checks: dt finite & (0, 0.5]; all positions/gains finite;
    /// gains in [0, 4]; occurrence ids strictly increasing; len bounds; no duplicate
    /// EmitterId. `window` may be `None` when nothing will trace it. Returns
    /// `FrameError` otherwise.
    pub fn new(
        dt: f32,
        listener: Listener,
        occurrences: Vec<Occurrence>,
        emitters: Vec<Emitter>,
        window: Option<Arc<AcousticWindow>>,
    ) -> Result<Self, FrameError> {
        if !dt.is_finite() {
            return Err(FrameError::NonFinite);
        }
        if dt <= 0.0 || dt > 0.5 {
            return Err(FrameError::OutOfRange);
        }
        if occurrences.len() > MAX_OCCURRENCES || emitters.len() > MAX_EMITTERS {
            return Err(FrameError::OverLimit);
        }

        if !listener.pos.is_finite() || !listener.yaw.is_finite() || !listener.pitch.is_finite() {
            return Err(FrameError::NonFinite);
        }

        let mut prev: Option<OccurrenceId> = None;
        for o in &occurrences {
            if let Some(at) = o.at
                && !at.is_finite()
            {
                return Err(FrameError::NonFinite);
            }
            if !o.gain.is_finite() {
                return Err(FrameError::NonFinite);
            }
            if !gain_in_range(o.gain) {
                return Err(FrameError::OutOfRange);
            }
            if let Some(p) = prev
                && o.id <= p
            {
                return Err(FrameError::UnorderedOccurrences);
            }
            prev = Some(o.id);
        }

        let mut seen: Vec<EmitterId> = Vec::with_capacity(emitters.len());
        for e in &emitters {
            if !e.at.is_finite() {
                return Err(FrameError::NonFinite);
            }
            if !e.gain.is_finite() {
                return Err(FrameError::NonFinite);
            }
            if !gain_in_range(e.gain) {
                return Err(FrameError::OutOfRange);
            }
            if seen.contains(&e.id) {
                return Err(FrameError::DuplicateEmitter);
            }
            seen.push(e.id);
        }

        Ok(Self {
            dt,
            listener,
            occurrences,
            emitters,
            window,
        })
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }
    pub fn listener(&self) -> &Listener {
        &self.listener
    }
    pub fn occurrences(&self) -> &[Occurrence] {
        &self.occurrences
    }
    pub fn emitters(&self) -> &[Emitter] {
        &self.emitters
    }
    pub fn window(&self) -> Option<&Arc<AcousticWindow>> {
        self.window.as_ref()
    }

    /// Consume the frame, moving the owned journal and table out without cloning
    /// — the committer takes ownership at submit.
    pub fn into_parts(
        self,
    ) -> (
        f32,
        Listener,
        Vec<Occurrence>,
        Vec<Emitter>,
        Option<Arc<AcousticWindow>>,
    ) {
        (
            self.dt,
            self.listener,
            self.occurrences,
            self.emitters,
            self.window,
        )
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;
    use voxel_engine::IVec3;

    use super::super::acoustics::{Cell, Listener, Medium};
    use super::*;

    fn window() -> Option<Arc<AcousticWindow>> {
        Some(Arc::new(
            AcousticWindow::new(IVec3::ZERO, UVec3::ONE, vec![Cell::Open].into_boxed_slice())
                .unwrap(),
        ))
    }

    fn listener() -> Listener {
        Listener {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            medium: Medium::Air,
        }
    }

    fn occ(id: u64, gain: f32) -> Occurrence {
        Occurrence {
            id: OccurrenceId(id),
            cue: CueId::TEST,
            at: None,
            medium: Medium::Air,
            gain,
        }
    }

    fn emitter(id: u64, gain: f32) -> Emitter {
        Emitter {
            id: EmitterId(id),
            cue: CueId::TEST,
            at: DVec3::ZERO,
            medium: Medium::Air,
            gain,
        }
    }

    #[test]
    fn accepts_valid_frame() {
        let f = AudioFrame::new(
            0.016,
            listener(),
            vec![occ(1, 1.0), occ(2, 0.5)],
            vec![emitter(1, 1.0)],
            window(),
        );
        assert!(f.is_ok());
    }

    #[test]
    fn rejects_non_finite() {
        let f = AudioFrame::new(f32::NAN, listener(), vec![], vec![], window());
        assert!(matches!(f, Err(FrameError::NonFinite)));

        let bad = Occurrence {
            id: OccurrenceId(1),
            cue: CueId::TEST,
            at: Some(DVec3::new(f64::NAN, 0.0, 0.0)),
            medium: Medium::Air,
            gain: 1.0,
        };
        let f = AudioFrame::new(0.016, listener(), vec![bad], vec![], window());
        assert!(matches!(f, Err(FrameError::NonFinite)));
    }

    #[test]
    fn rejects_out_of_range() {
        // dt out of (0, 0.5].
        let f = AudioFrame::new(1.0, listener(), vec![], vec![], window());
        assert!(matches!(f, Err(FrameError::OutOfRange)));

        // gain outside [0, 4].
        let f = AudioFrame::new(0.016, listener(), vec![occ(1, 5.0)], vec![], window());
        assert!(matches!(f, Err(FrameError::OutOfRange)));
    }

    #[test]
    fn rejects_unordered_occurrences() {
        let f = AudioFrame::new(
            0.016,
            listener(),
            vec![occ(2, 1.0), occ(2, 1.0)],
            vec![],
            window(),
        );
        assert!(matches!(f, Err(FrameError::UnorderedOccurrences)));

        let f = AudioFrame::new(
            0.016,
            listener(),
            vec![occ(3, 1.0), occ(1, 1.0)],
            vec![],
            window(),
        );
        assert!(matches!(f, Err(FrameError::UnorderedOccurrences)));
    }

    #[test]
    fn rejects_duplicate_emitter() {
        let f = AudioFrame::new(
            0.016,
            listener(),
            vec![],
            vec![emitter(1, 1.0), emitter(1, 1.0)],
            window(),
        );
        assert!(matches!(f, Err(FrameError::DuplicateEmitter)));
    }

    #[test]
    fn rejects_over_limit() {
        let occs: Vec<_> = (0..(MAX_OCCURRENCES as u64 + 1))
            .map(|i| occ(i, 1.0))
            .collect();
        let f = AudioFrame::new(0.016, listener(), occs, vec![], window());
        assert!(matches!(f, Err(FrameError::OverLimit)));
    }

    #[test]
    fn accepts_absent_window_when_nothing_traces() {
        let f = AudioFrame::new(0.016, listener(), vec![], vec![], None);
        assert!(f.is_ok());
        assert!(f.unwrap().window().is_none());
    }
}
