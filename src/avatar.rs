//! A minimal six-box humanoid for remote players. The rig is described once as a
//! `const` table in body-local space (origin = feet, +Y up, -Z forward); `Pose`
//! resolves it into world-space boxes given facing, look, and gait, and draws them
//! with the engine's oriented, face-shaded [`Frame3D::draw_box`].
//!
//! The design keeps all animation in the type: each part carries a [`Swing`] rule,
//! so `resolve` is a single loop with no per-part name matching.
use voxel_engine::{Color, Frame3D, Mat3, Vec3};

/// How a part responds to motion.
enum Swing {
    /// Faces the body only (torso).
    None,
    /// Tracks the look pitch (head).
    Look,
    /// Swings about its pivot; `phase_offset` puts limbs in antiphase.
    Limb { phase_offset: f32 },
}

/// One rigid box in body-local space. `pivot` is where it rotates; `rest` is the
/// box centre at rest; `tint` darkens limbs vs. head/torso so the player colour
/// still identifies them.
struct Part {
    pivot: Vec3,
    rest: Vec3,
    half: Vec3,
    tint: f32,
    swing: Swing,
}

use std::f32::consts::PI;

const RIG: [Part; 6] = [
    // Head — tracks look pitch, pivots at the neck.
    Part {
        pivot: Vec3::new(0.0, 1.5, 0.0),
        rest: Vec3::new(0.0, 1.7, 0.0),
        half: Vec3::new(0.2, 0.2, 0.2),
        tint: 1.0,
        swing: Swing::Look,
    },
    // Torso — faces only.
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
        swing: Swing::Limb { phase_offset: PI },
    },
    // Right arm.
    Part {
        pivot: Vec3::new(0.34, 1.5, 0.0),
        rest: Vec3::new(0.34, 1.1, 0.0),
        half: Vec3::new(0.09, 0.4, 0.09),
        tint: 0.75,
        swing: Swing::Limb { phase_offset: 0.0 },
    },
    // Left leg.
    Part {
        pivot: Vec3::new(-0.12, 0.8, 0.0),
        rest: Vec3::new(-0.12, 0.4, 0.0),
        half: Vec3::new(0.1, 0.4, 0.1),
        tint: 0.7,
        swing: Swing::Limb { phase_offset: 0.0 },
    },
    // Right leg.
    Part {
        pivot: Vec3::new(0.12, 0.8, 0.0),
        rest: Vec3::new(0.12, 0.4, 0.0),
        half: Vec3::new(0.1, 0.4, 0.1),
        tint: 0.7,
        swing: Swing::Limb { phase_offset: PI },
    },
];

/// Fully-resolved, ready to draw: world-space centre + rotation per part, plus
/// the feet position so the contact shadow anchors to the model.
pub struct Pose {
    feet: Vec3,
    parts: [(Vec3, Mat3); 6],
}

/// Half-width of the contact-shadow blob, and its darkness (alpha over terrain).
const SHADOW_RADIUS: f32 = 0.4;
const SHADOW_COLOR: Color = Color::new(0, 0, 0, 90);
/// Nudge the shadow just above the feet plane so it doesn't z-fight the ground.
const SHADOW_LIFT: f32 = 0.02;

impl Pose {
    /// Head-top height above the feet, so name tags anchor to the model.
    pub const HEAD_TOP: f32 = 1.9;

    /// `yaw` faces the body, `pitch` tilts the head, `phase` drives limb swing,
    /// `amp` scales it by speed (0 = idle → no swing).
    pub fn resolve(feet: Vec3, yaw: f32, pitch: f32, phase: f32, amp: f32) -> Self {
        let yaw_rot = Mat3::from_rotation_y(yaw);
        let mut parts = [(Vec3::ZERO, Mat3::IDENTITY); 6];
        for (i, part) in RIG.iter().enumerate() {
            let swing_rot = match part.swing {
                Swing::None => Mat3::IDENTITY,
                Swing::Look => Mat3::from_rotation_x(pitch),
                Swing::Limb { phase_offset } => {
                    Mat3::from_rotation_x((phase + phase_offset).sin() * amp)
                }
            };
            let local = part.pivot + swing_rot * (part.rest - part.pivot);
            parts[i] = (feet + yaw_rot * local, yaw_rot * swing_rot);
        }
        Self { feet, parts }
    }

    pub fn draw(&self, f3: &mut Frame3D, color: Color) {
        let ground = self.feet + Vec3::new(0.0, SHADOW_LIFT, 0.0);
        f3.draw_shadow(ground, SHADOW_RADIUS, SHADOW_COLOR);
        for (i, part) in RIG.iter().enumerate() {
            let (center, rot) = self.parts[i];
            f3.draw_box(center, part.half, rot, tint(color, part.tint));
        }
    }
}

/// Scale a colour's RGB toward black by `factor`, leaving alpha untouched.
fn tint(color: Color, factor: f32) -> Color {
    let s = |v: u8| (v as f32 * factor).round().clamp(0.0, 255.0) as u8;
    Color::new(s(color.r), s(color.g), s(color.b), color.a)
}
