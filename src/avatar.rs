//! Six-box humanoid for remote players. Animation lives in the type (each part
//! carries a [`Swing`] rule), so resolution is a single loop with no per-part
//! name matching.
use voxel_engine::{Color, Frame3D, Mat3, Vec3};

use crate::presence::{RenderPose, RigParams, wrap_pi};

enum Swing {
    None,
    /// Head tracks yaw/pitch separately from body.
    Look,
    /// Limbs swing with gait phase; phase_offset staggers left/right; action_arm responds to input.
    Limb { phase_offset: f32, action_arm: bool },
}

/// tint: darkens limbs vs head/torso so player colour still identifies the avatar.
struct Part {
    pivot: Vec3,
    rest: Vec3,
    half: Vec3,
    tint: f32,
    swing: Swing,
}

use std::f32::consts::PI;

/// Index of the head part in [`RIG`]; omitted for the local first-person body.
const HEAD: usize = 0;

const RIG: [Part; 6] = [
    // Head.
    Part {
        pivot: Vec3::new(0.0, 1.5, 0.0),
        rest: Vec3::new(0.0, 1.7, 0.0),
        half: Vec3::new(0.2, 0.2, 0.2),
        tint: 1.0,
        swing: Swing::Look,
    },
    // Torso.
    Part {
        pivot: Vec3::new(0.0, 1.15, 0.0),
        rest: Vec3::new(0.0, 1.15, 0.0),
        half: Vec3::new(0.25, 0.35, 0.15),
        tint: 1.0,
        swing: Swing::None,
    },
    // Left arm.
    Part {
        pivot: Vec3::new(-0.34, 1.5, 0.0),
        rest: Vec3::new(-0.34, 1.1, 0.0),
        half: Vec3::new(0.09, 0.4, 0.09),
        tint: 0.75,
        swing: Swing::Limb { phase_offset: PI, action_arm: false },
    },
    // Right arm.
    Part {
        pivot: Vec3::new(0.34, 1.5, 0.0),
        rest: Vec3::new(0.34, 1.1, 0.0),
        half: Vec3::new(0.09, 0.4, 0.09),
        tint: 0.75,
        swing: Swing::Limb { phase_offset: 0.0, action_arm: true },
    },
    // Left leg.
    Part {
        pivot: Vec3::new(-0.12, 0.8, 0.0),
        rest: Vec3::new(-0.12, 0.4, 0.0),
        half: Vec3::new(0.1, 0.4, 0.1),
        tint: 0.7,
        swing: Swing::Limb { phase_offset: 0.0, action_arm: false },
    },
    // Right leg.
    Part {
        pivot: Vec3::new(0.12, 0.8, 0.0),
        rest: Vec3::new(0.12, 0.4, 0.0),
        half: Vec3::new(0.1, 0.4, 0.1),
        tint: 0.7,
        swing: Swing::Limb { phase_offset: PI, action_arm: false },
    },
];

pub struct Pose {
    feet: Vec3,
    parts: [(Vec3, Mat3); 6],
}

/// Half-width of the contact-shadow blob, and its darkness (alpha over terrain).
const SHADOW_RADIUS: f32 = 0.4 * SCALE;
const SHADOW_COLOR: Color = Color::new(0, 0, 0, 90);
/// Lifts shadow above feet plane to avoid z-fighting.
const SHADOW_LIFT: f32 = 0.02;

/// Metres → world units for the rig: the whole model is authored in metres
/// (head at 1.7, hip at 0.9) and scaled once where body-local space meets the
/// world, so the avatar always matches the player's collision height.
const SCALE: f32 = crate::math::PER_METER as f32;

impl Pose {
    /// Head-top height above the feet, so name tags anchor to the model.
    pub const HEAD_TOP: f32 = 1.9 * SCALE;

    pub fn resolve(pose: &RenderPose, rig: &RigParams) -> Self {
        /// Tips body forward in prone stance.
        const PRONE_ANGLE: f32 = -1.3;
        const ACTION_AMP: f32 = 1.6;
        const HIP: Vec3 = Vec3::new(0.0, 0.9, 0.0);

        // Negate yaw to flip rotation sense; -pi/2 offset aligns body-local -Z forward with world +X.
        let body_rot = Mat3::from_rotation_y(-rig.body_yaw - PI / 2.0);
        let head_yaw = Mat3::from_rotation_y(-wrap_pi(pose.yaw - rig.body_yaw));
        let h = 1.0 + (pose.stance.height_scale() - 1.0) * rig.stance_blend;
        let squash = |v: Vec3| Vec3::new(v.x, v.y * h, v.z);
        let prone_rot = if pose.stance.prone() {
            Mat3::from_rotation_x(PRONE_ANGLE * rig.stance_blend)
        } else {
            Mat3::IDENTITY
        };

        let (phase, amp) = (pose.gait.phase, pose.gait.amp());
        let mut parts = [(Vec3::ZERO, Mat3::IDENTITY); 6];
        for (i, part) in RIG.iter().enumerate() {
            let swing_rot = match part.swing {
                Swing::None => Mat3::IDENTITY,
                Swing::Look => head_yaw * Mat3::from_rotation_x(pose.pitch),
                Swing::Limb { phase_offset, action_arm } => {
                    let mut angle = (phase + phase_offset).sin() * amp;
                    if action_arm {
                        angle += rig.action_swing * ACTION_AMP;
                    }
                    Mat3::from_rotation_x(angle)
                }
            };
            let pivot = squash(part.pivot);
            let local = pivot + swing_rot * (squash(part.rest) - pivot);
            let hip = squash(HIP);
            let local = hip + prone_rot * (local - hip);
            parts[i] = (pose.feet + body_rot * (local * SCALE), body_rot * prone_rot * swing_rot);
        }
        Self { feet: pose.feet, parts }
    }

    /// `include_head` is false for the local player in first-person view: the
    /// camera sits inside the head box, so drawing it would clip the near plane
    /// when pitching down. The rest of the body still renders — visible when the
    /// player looks down, and casting a shadow like any other avatar.
    pub fn draw(&self, f3: &mut Frame3D, color: Color, include_head: bool) {
        let ground = self.feet + Vec3::new(0.0, SHADOW_LIFT, 0.0);
        f3.draw_shadow(ground, SHADOW_RADIUS, SHADOW_COLOR);
        for (i, part) in RIG.iter().enumerate() {
            if i == HEAD && !include_head {
                continue;
            }
            let (center, rot) = self.parts[i];
            f3.draw_box(center, part.half * SCALE, rot, tint(color, part.tint));
        }
    }
}

fn tint(color: Color, factor: f32) -> Color {
    let s = |v: u8| (v as f32 * factor).round().clamp(0.0, 255.0) as u8;
    Color::new(s(color.r), s(color.g), s(color.b), color.a)
}
