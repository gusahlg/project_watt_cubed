//! Player-presence types: one representation for how any player — local or
//! remote — is observed and presented each frame.
//!
//! Local and remote players differ only in authority (where the pose comes
//! from), never in representation. Both produce a RenderPose and the avatar
//! draws from that alone. Stance is a closed enum, not flag booleans, so
//! invalid combinations are unrepresentable. Tag visibility is computed fresh
//! each frame from state; no mutated flags. Anything derivable from pose
//! trajectory (speed, gait, body yaw) is derived client-side, never networked.
//! Only non-derivable signal is a discrete action sent as a WireAction.
use voxel_engine::{DVec3, Vec3};

use crate::player::{self, Player};

/// A gait cycle advances this many radians per world unit of horizontal
/// travel, so limbs swing at a cadence tied to distance, not frame rate.
/// At the 6 u/s base walk speed this is ~1.9 swing cycles per second.
pub const STRIDE_FREQ: f64 = 2.0;

/// Movement stance, networked per snapshot. Distinct from
/// [`player::Stance`] (which owns collision/eye heights): this is the closed
/// *broadcast* set, including swimming, which the local player models as
/// [`Motion::Swimming`](crate::player::Motion) rather than a stance.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Stance {
    #[default]
    Standing,
    Sneaking,
    Swimming,
}

impl Stance {
    pub fn of_player(p: &Player) -> Self {
        if p.swimming() {
            Stance::Swimming
        } else if p.stance == player::Stance::Sneaking {
            Stance::Sneaking
        } else {
            Stance::Standing
        }
    }

    pub fn height_scale(self) -> f32 {
        match self {
            Stance::Standing | Stance::Swimming => 1.0,
            Stance::Sneaking => 0.82,
        }
    }

    pub fn prone(self) -> bool {
        matches!(self, Stance::Swimming)
    }

    /// Eye height above the feet for a broadcast stance. Reuses [`player::Stance`]'s
    /// offset so the eye/feet gap can't drift from the local player's; swimming
    /// keeps the standing eye height (the local swimmer's box is still upright).
    pub fn eye_offset(self) -> f64 {
        match self {
            Stance::Sneaking => player::Stance::Sneaking,
            Stance::Standing | Stance::Swimming => player::Stance::Standing,
        }
        .eye_offset()
    }

    /// Wire codec: one byte, closed set. `from_wire` rejects unknown values so
    /// a hostile byte can't smuggle an out-of-enum stance.
    pub fn wire(self) -> u8 {
        match self {
            Stance::Standing => 0,
            Stance::Sneaking => 1,
            Stance::Swimming => 2,
        }
    }
    pub fn from_wire(v: u8) -> Option<Self> {
        Some(match v {
            0 => Stance::Standing,
            1 => Stance::Sneaking,
            2 => Stance::Swimming,
            _ => return None,
        })
    }
}

/// Walk-cycle observation. Amplitude is clamped at construction so consumers
/// never re-clamp.
#[derive(Clone, Copy)]
pub struct Gait {
    pub phase: f32,
    amp: f32,
}

impl Gait {
    pub const MAX_AMP: f32 = 0.9;

    pub fn new(phase: f32, speed: f32) -> Self {
        Self { phase, amp: (speed * 0.22).min(Self::MAX_AMP) }
    }

    pub fn amp(self) -> f32 {
        self.amp
    }
}

/// A world-space position anchored at the eye (camera height). What the wire
/// carries, and the authoritative origin for the server's reach checks. Distinct
/// type from [`Feet`] so the two anchors can't be swapped by accident — the class
/// of bug where a remote avatar renders eye-high instead of on the ground.
#[derive(Clone, Copy)]
pub struct Eye(pub DVec3);

/// A world-space position anchored at the feet — the avatar rig's origin. The only
/// way to reach it from an [`Eye`] is [`Eye::feet`], which *requires* a stance, so
/// the eye-height drop can never be silently skipped.
#[derive(Clone, Copy)]
pub struct Feet(pub DVec3);

impl Eye {
    /// Drop to the feet for the given stance.
    pub fn feet(self, stance: Stance) -> Feet {
        Feet(self.0 - DVec3::new(0.0, stance.eye_offset(), 0.0))
    }
}

/// The single per-frame pose the avatar renders from. Remote authority
/// produces it by snapshot interpolation; local authority directly from
/// [`Player`]. `feet` is already camera-relative (world minus eye, subtracted
/// in f64 then narrowed) — safe for the f32 immediate draws.
#[derive(Clone, Copy)]
pub struct RenderPose {
    pub feet: Vec3,
    /// Head yaw; body yaw tracked separately in [`RigParams`].
    pub yaw: f32,
    pub pitch: f32,
    pub stance: Stance,
    pub gait: Gait,
}

impl RenderPose {
    /// Build a pose from a world-space [`Feet`] position, narrowed relative to the
    /// camera. Taking a typed `Feet` (never a bare vector) is the guard: callers
    /// must convert an eye position through [`Eye::feet`] first, so an eye can't be
    /// mistaken for feet.
    pub fn new(feet: Feet, camera: Eye, yaw: f32, pitch: f32, stance: Stance, gait: Gait) -> Self {
        Self { feet: (feet.0 - camera.0).as_vec3(), yaw, pitch, stance, gait }
    }
}

/// Name-tag visibility with fade baked in: no `(visible, alpha)` pair to
/// disagree, and near-zero alpha normalizes to `Hidden`.
#[derive(Clone, Copy, PartialEq)]
pub enum TagVisibility {
    Hidden,
    Visible { alpha: f32 },
}

impl TagVisibility {
    /// Peers past this distance get no floating name tag (unreadable anyway).
    pub const RANGE: f64 = 90.0;
    /// Fade band before the cutoff, so tags dissolve instead of popping.
    const FADE: f64 = 15.0;
    /// Alpha when terrain occludes the head: a faint hint instead of a
    /// full-strength tag drawn through walls.
    const OCCLUDED_ALPHA: f32 = 0.25;

    pub fn of(distance: f64, occluded: bool) -> Self {
        if distance > Self::RANGE {
            return Self::Hidden;
        }
        let fade = ((Self::RANGE - distance) / Self::FADE).min(1.0) as f32;
        let alpha = fade * if occluded { Self::OCCLUDED_ALPHA } else { 1.0 };
        if alpha <= f32::EPSILON { Self::Hidden } else { Self::Visible { alpha } }
    }
}

/// The one action that must cross the wire: an interact swing is not
/// derivable from the movement trajectory.
#[derive(Clone, Copy)]
pub enum WireAction {
    Swing,
}

/// Everything the avatar rig needs beyond the raw [`RenderPose`]. Pure output
/// of [`Animator::step`]; holds no state.
#[derive(Clone, Copy)]
pub struct RigParams {
    /// Torso/limb orientation. The head uses `RenderPose.yaw`; the difference
    /// is guaranteed within ±[`Animator::MAX_TWIST`].
    pub body_yaw: f32,
    /// Blend toward the target stance (0–1).
    pub stance_blend: f32,
    /// Interact swing amplitude envelope (0–1).
    pub action_swing: f32,
}

/// Per-player animation state: updates from (pose, dt) to produce rig parameters.
/// Local and remote players get one each, the same authority-agnostic boundary
/// as [`RenderPose`]. Only persists three floats and an Option.
#[derive(Default)]
pub struct Animator {
    body_yaw: f32,
    stance_blend: f32,
    prev_stance: Stance,
    /// Seconds since a [`WireAction::Swing`]; `None` when idle, normalized
    /// back to `None` past the envelope so "swinging forever" is
    /// unrepresentable.
    swing_age: Option<f32>,
}

impl Animator {
    /// Max head-over-body twist before the body follows (radians).
    pub const MAX_TWIST: f32 = 0.87;
    /// Body re-align rate (rad/s).
    const TWIST_RATE_IDLE: f32 = 3.0;
    const TWIST_RATE_MOVING: f32 = 10.0;
    /// Stance blend time constant (seconds).
    const STANCE_TAU: f32 = 0.12;
    /// Swing envelope duration (seconds).
    const SWING_LEN: f32 = 0.55;

    pub fn on_action(&mut self, action: WireAction) {
        match action {
            WireAction::Swing => self.swing_age = Some(0.0),
        }
    }

    /// Update body yaw and stance toward their targets, unfold the swing envelope.
    pub fn step(&mut self, pose: &RenderPose, dt: f32) -> RigParams {
        // Body follows head: dead within ±MAX_TWIST while idle, tracks
        // continuously while moving.
        let moving = pose.gait.amp() > 0.05;
        let err = wrap_pi(pose.yaw - self.body_yaw);
        let rate = if moving { Self::TWIST_RATE_MOVING } else { Self::TWIST_RATE_IDLE };
        let chase = if moving || err.abs() > Self::MAX_TWIST { rate * dt } else { 0.0 };
        self.body_yaw = wrap_pi(self.body_yaw + err.clamp(-chase, chase));
        // Never let the head exceed the twist limit even mid-chase.
        let err = wrap_pi(pose.yaw - self.body_yaw);
        if err.abs() > Self::MAX_TWIST {
            self.body_yaw = wrap_pi(pose.yaw - err.signum() * Self::MAX_TWIST);
        }

        // Stance: reset blend on stance change, then exponentially approach the target.
        if pose.stance != self.prev_stance {
            self.prev_stance = pose.stance;
            self.stance_blend = 0.0;
        }
        let target = if pose.stance == Stance::Standing { 0.0 } else { 1.0 };
        let k = 1.0 - (-dt / Self::STANCE_TAU).exp();
        self.stance_blend += (target - self.stance_blend) * k;

        // Swing: apply half-sine envelope, then idle.
        let action_swing = match self.swing_age.take() {
            Some(age) if age < Self::SWING_LEN => {
                self.swing_age = Some(age + dt);
                ((age / Self::SWING_LEN) * std::f32::consts::PI).sin()
            }
            _ => 0.0,
        };

        RigParams { body_yaw: self.body_yaw, stance_blend: self.stance_blend, action_swing }
    }
}

/// Wrap an angle into [-pi, pi].
pub fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_to_feet_uses_the_broadcast_stance_height() {
        let eye = Eye(DVec3::new(17.25, 93.0, -8.5));
        for stance in [Stance::Standing, Stance::Sneaking, Stance::Swimming] {
            let feet = eye.feet(stance);
            assert_eq!(feet.0.x.to_bits(), eye.0.x.to_bits());
            assert_eq!(feet.0.z.to_bits(), eye.0.z.to_bits());
            assert_eq!(feet.0.y, eye.0.y - stance.eye_offset());
        }
    }

    #[test]
    fn remote_pose_keeps_sub_block_offsets_at_far_coordinates() {
        // Subtract in f64 before narrowing. Narrowing each absolute position first
        // would erase these offsets hundreds of millions of units from the origin.
        let camera = Eye(DVec3::new(900_000_000.25, 71.5, -800_000_000.75));
        let remote_eye = Eye(camera.0 + DVec3::new(3.125, 2.0, -4.375));
        let stance = Stance::Sneaking;
        let pose = RenderPose::new(
            remote_eye.feet(stance),
            camera,
            0.0,
            0.0,
            stance,
            Gait::new(0.0, 0.0),
        );

        assert_eq!(
            pose.feet,
            Vec3::new(3.125, (2.0 - stance.eye_offset()) as f32, -4.375)
        );
    }

    #[test]
    fn wire_stance_is_a_closed_round_trip() {
        for stance in [Stance::Standing, Stance::Sneaking, Stance::Swimming] {
            assert_eq!(Stance::from_wire(stance.wire()), Some(stance));
        }
        assert_eq!(Stance::from_wire(3), None);
        assert_eq!(Stance::from_wire(u8::MAX), None);
    }
}
