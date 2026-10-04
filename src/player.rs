//! Player state: position, orientation, and the elements they carry. Positions
//! use `f64` for precision out to world borders; view angles use `f32` since
//! rotation doesn't accumulate magnitude (see [`math`](crate::math)).
use voxel_engine::DVec3;

use crate::camera::{Orientation, rotate};
use crate::coord::Face;
use crate::math::{Aabb, Bounded, PER_METER};
use crate::stash::{ElementStash, START_CAPACITY};

/// The player's collision half-width across the two axes perpendicular to the collision axis.
/// The extent along it is not a constant — it derives from [`Stance::height`] — so there is no
/// third value here to fall out of sync with the stance.
pub const PLAYER_HALF_WIDTH: f64 = 0.3 * PER_METER;

/// A fresh player's base ground walk speed, units/second. It lives on the player
/// (see [`Player::speed`]) rather than in the movement module so it can vary per
/// player; this is only the starting value.
pub const DEFAULT_WALK_SPEED: f64 = 6.0 * PER_METER;

/// A fresh player's flying speed, units/second. Lives on the player (see
/// [`Player::fly_speed`]) for the same reason [`DEFAULT_WALK_SPEED`] does — so it
/// can vary per player; this is only the starting value.
pub const DEFAULT_FLY_SPEED: f64 = 14.0 * PER_METER;

/// A fresh player's health, and the ceiling it's created at. Health is an intrinsic
/// property the player carries but nothing yet reads or changes — see
/// [`Player::health`].
pub const MAX_HEALTH: f32 = 20.0;

/// The reference gravity, units/s²: what the designed start planet pulls at its spawn face
/// centre, and the scale the zero-g thresholds are measured against.
pub const STANDARD_GRAVITY: f64 = 24.0 * PER_METER;

/// Below this fraction of [`STANDARD_GRAVITY`] a spawn or teleport has no up to stand on.
const STAND_MIN: f64 = 0.02;

/// Body frame and collision axis that `gravity` defines for a fresh spawn.
/// `(IDENTITY, PosY)` when gravity is too weak to define an up, and when that up is exactly +Y
/// (the +Y face centre): snapping an identity frame onto +Y is a no-op.
pub fn standing_pose(gravity: DVec3) -> (glam::DQuat, Face) {
    let Some(up) = crate::gravity::Sample::uniform(gravity).up(STAND_MIN * STANDARD_GRAVITY) else {
        return (glam::DQuat::IDENTITY, Face::PosY);
    };
    let mut orientation = Orientation::new(0.0, 0.0);
    orientation.snap(up);
    (orientation.frame, Face::from_dominant(up))
}

/// How tall the player stands and how high their eye sits, as a function of what
/// they're doing. Geometry is a pure function of the stance — box height and eye
/// height both derive from one [`height`](Stance::height), so there is no free
/// float to drift and invalid heights are unrepresentable.
///
/// The eye is anchored to the *feet*, not to the box centre: [`eye_offset`] is the
/// eye's height above the feet, fixed at 90% of the stance height so the eyes sit
/// just below the crown. `Standing` is 1.8 m tall; `Sneaking` shrinks the box *and*
/// drops the eye proportionally, so crouching lowers both the head and the camera.
///
/// [`eye_offset`]: Stance::eye_offset
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    Standing,
    Sneaking,
}

impl Stance {
    /// Full standing (or crouching) height (1.8 m / 1.5 m in world units) — the
    /// primitive from which the box half-extent and eye height both derive, so
    /// they can't drift apart. Sneaking lowers it.
    pub fn height(self) -> f64 {
        match self {
            Stance::Standing => 1.8 * PER_METER,
            Stance::Sneaking => 1.5 * PER_METER,
        }
    }

    /// Eye height above the feet: 90% of the stance's full height, so the eyes sit
    /// just below the crown and sneaking lowers the camera along with the box.
    pub fn eye_offset(self) -> f64 {
        self.height() * 0.9
    }
}

/// How the player is moving through the world. A sum type instead of three loose
/// fields (`velocity_y` / `on_ground` / `fly`) so the nonsense combinations —
/// flying while grounded, flying with an accumulated fall velocity — are simply
/// unrepresentable. Both variants carry a *full* velocity vector: walking drives
/// `.xz` by inertia and `.y` by the gravity/jump integrator, flying drives all
/// three toward a directly-commanded target.
#[derive(Clone, Copy)]
pub enum Motion {
    /// On foot: subject to gravity, jumping, and ground contact.
    Walking { velocity: DVec3, on_ground: bool },
    /// Free flight: no gravity, no ground, velocity chases input on every axis of the body frame.
    /// `noclip` additionally skips collision, letting the player pass through
    /// solid geometry — meaningful only in flight, so it rides on this variant
    /// rather than being a loose flag that could contradict walking.
    Flying { velocity: DVec3, noclip: bool },
}

impl Motion {
    /// The current velocity, whichever mode we're in.
    pub fn velocity(self) -> DVec3 {
        match self {
            Motion::Walking { velocity, .. } | Motion::Flying { velocity, .. } => velocity,
        }
    }
}

/// The player: where they are, where they're looking, and what they carry.
pub struct Player {
    /// Eye position in world space.
    pub position: DVec3,
    /// Body frame and view angles — the one orientation; every camera mode is a function of it.
    pub orientation: Orientation,
    /// The grid axis the collision box stands along: the signed axis nearest the body's up,
    /// switched with hysteresis.
    pub up_axis: Face,
    /// The gravity vector the last physics step applied (the camera aligns to it between steps).
    pub gravity: DVec3,
    /// How the player is moving — walking (with gravity) or flying.
    pub motion: Motion,
    /// Standing or sneaking — drives the player's height and eye offset.
    pub stance: Stance,
    /// Base ground walk speed in units/second — an intrinsic the movement code
    /// reads to scale the walking target (sprint still multiplies on top).
    pub speed: f64,
    /// Flying speed in units/second — an intrinsic the movement code reads to
    /// scale the flying target, mirroring [`Player::speed`] for walking.
    pub fly_speed: f64,
    /// Current health. Intrinsic and carried on the player, but *not wired*: no
    /// system reads or mutates it yet, so it simply holds [`MAX_HEALTH`].
    pub health: f32,
    /// Elements this player holds. Core-owned; mods present and spend it.
    pub stash: ElementStash,
}

impl Player {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            orientation: Orientation::new(0.0, 0.0),
            up_axis: Face::PosY,
            gravity: DVec3::new(0.0, -STANDARD_GRAVITY, 0.0),
            motion: Motion::Walking { velocity: DVec3::ZERO, on_ground: false },
            stance: Stance::Standing,
            speed: DEFAULT_WALK_SPEED,
            fly_speed: DEFAULT_FLY_SPEED,
            health: MAX_HEALTH,
            stash: ElementStash::new(START_CAPACITY),
        }
    }

    /// The player's current velocity.
    pub fn velocity(&self) -> DVec3 {
        self.motion.velocity()
    }

    /// Whether the player is standing on solid ground this frame (never while flying).
    pub fn on_ground(&self) -> bool {
        matches!(self.motion, Motion::Walking { on_ground: true, .. })
    }

    /// Whether the player is in free flight.
    pub fn flying(&self) -> bool {
        matches!(self.motion, Motion::Flying { .. })
    }

    /// Whether the player is flying with collision disabled (passing through
    /// solid geometry). False whenever not flying.
    pub fn noclip(&self) -> bool {
        matches!(self.motion, Motion::Flying { noclip: true, .. })
    }

    /// The velocity with its component along the collision axis removed.
    fn level_velocity(&self) -> DVec3 {
        let mut v = self.velocity();
        v[self.up_axis.axis()] = 0.0;
        v
    }

    /// Enter or leave flight. Momentum across the ground carries over the switch, but the
    /// component along the up axis is cleared so the player neither keeps falling into the new
    /// mode nor launches when leaving it.
    pub fn set_flying(&mut self, flying: bool) {
        let velocity = self.level_velocity();
        self.motion = if flying {
            Motion::Flying { velocity, noclip: false }
        } else {
            Motion::Walking { velocity, on_ground: false }
        };
    }

    /// Toggle walking and ordinary flight. Never enters noclip. Momentum across the ground
    /// carries over; the component along up is cleared, as in [`Player::set_flying`].
    pub fn toggle_fly(&mut self) {
        self.set_flying(!self.flying());
    }

    /// Toggle noclip flight: from walking or ordinary flight into noclip, and from noclip back
    /// to walking. Same velocity carry as [`Player::set_flying`].
    pub fn toggle_noclip(&mut self) {
        let velocity = self.level_velocity();
        self.motion = if self.noclip() {
            Motion::Walking { velocity, on_ground: false }
        } else {
            Motion::Flying { velocity, noclip: true }
        };
    }

    /// Drop any accumulated velocity along the up axis (e.g. after a teleport, so the player
    /// doesn't rocket down on arrival).
    pub fn cancel_fall(&mut self) {
        let a = self.up_axis.axis();
        match &mut self.motion {
            Motion::Walking { velocity, .. } | Motion::Flying { velocity, .. } => velocity[a] = 0.0,
        }
    }

    /// Where the feet are: the eye dropped by the stance's eye offset along the up axis.
    pub fn feet(&self) -> DVec3 {
        feet_of(self.position, self.stance, self.up_axis)
    }

    /// The body's up direction (smoothed toward the local −gravity).
    pub fn up(&self) -> DVec3 {
        self.orientation.up()
    }

    /// Snap the body frame (and the collision axis) to `up`: spawn, teleport, load.
    pub fn snap_up(&mut self, up: DVec3) {
        self.orientation.snap(up);
        self.up_axis = Face::from_dominant(up);
    }

    /// Stand in `gravity` (an acceleration). No-op in free fall. Identity at exact +Y.
    pub fn stand_in(&mut self, gravity: DVec3) {
        let (frame, up) = standing_pose(gravity);
        self.orientation.frame = frame;
        self.up_axis = up;
    }

    /// Full view direction, including pitch.
    pub fn forward(&self) -> DVec3 {
        self.orientation.direction()
    }

    /// The forward and right basis vectors in the body frame's horizontal plane, used for ground
    /// movement. Returned together because they share one `sin`/`cos` of the yaw, and both come
    /// out unit length already (no normalize needed).
    pub fn movement_basis(&self) -> (DVec3, DVec3) {
        let (sin_yaw, cos_yaw) = (self.orientation.yaw as f64).sin_cos();
        let frame = self.orientation.frame;
        (rotate(frame, DVec3::new(cos_yaw, 0.0, sin_yaw)), rotate(frame, DVec3::new(-sin_yaw, 0.0, cos_yaw)))
    }

}

/// The feet of an eye at `eye`: dropped by the stance's eye offset along `up`.
pub fn feet_of(eye: DVec3, stance: Stance, up: Face) -> DVec3 {
    let mut feet = eye;
    feet[up.axis()] -= up.sign() as f64 * stance.eye_offset();
    feet
}

/// The collision box for an eye at `eye` in the given `stance`, standing along the grid axis
/// `up`. Built from the feet up, not the eye: the feet sit [`eye_offset`](Stance::eye_offset)
/// below the eye, and the box rises the stance's full [`height`](Stance::height) from there, so a
/// shorter (sneaking) box lowers both its top and — via the proportional eye offset — the eye.
///
/// Free-standing (not a `Player` method) so the collision stepper, which advances a bare eye
/// position, can test candidate boxes without a whole `Player`.
pub fn collision_box(eye: DVec3, stance: Stance, up: Face) -> Aabb {
    let half_up = stance.height() / 2.0;
    let a = up.axis();
    let mut centre = eye;
    centre[a] = eye[a] - up.sign() as f64 * stance.eye_offset() + up.sign() as f64 * half_up;
    let mut half = DVec3::splat(PLAYER_HALF_WIDTH);
    half[a] = half_up;
    Aabb::new(centre, half)
}

impl Bounded for Player {
    fn aabb(&self) -> Aabb {
        collision_box(self.position, self.stance, self.up_axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;

    #[test]
    fn standing_pose_is_identity_on_plus_y_and_follows_a_side_face() {
        let (frame, face) = standing_pose(DVec3::new(0.0, -STANDARD_GRAVITY, 0.0));
        assert_eq!(frame, DQuat::IDENTITY);
        assert_eq!(face, Face::PosY);

        let (frame, face) = standing_pose(DVec3::new(-STANDARD_GRAVITY, 0.0, 0.0));
        assert_eq!(face, Face::PosX);
        let mut player = Player::new(DVec3::ZERO);
        player.stand_in(DVec3::new(-STANDARD_GRAVITY, 0.0, 0.0));
        assert_eq!(player.orientation.frame, frame);
        assert_eq!(player.up_axis, Face::PosX);
        assert!((player.up() - DVec3::X).length() < 1e-9);

        let mut loose = Player::new(DVec3::ZERO);
        loose.stand_in(DVec3::new(0.0, -0.01, 0.0));
        assert_eq!(loose.orientation.frame, DQuat::IDENTITY);
        assert_eq!(loose.up_axis, Face::PosY);
    }

    #[test]
    fn fly_toggles_walking_and_flying_and_keeps_level_speed() {
        let mut player = Player::new(DVec3::ZERO);
        player.motion = Motion::Walking { velocity: DVec3::new(3.0, -9.0, 4.0), on_ground: true };
        player.toggle_fly();
        assert!(player.flying());
        assert!(!player.noclip());
        assert_eq!(player.velocity(), DVec3::new(3.0, 0.0, 4.0));
        player.toggle_fly();
        assert!(!player.flying());
        assert_eq!(player.velocity(), DVec3::new(3.0, 0.0, 4.0));
    }

    #[test]
    fn noclip_toggles_from_walking_or_flying_back_to_walking() {
        let mut player = Player::new(DVec3::ZERO);
        player.toggle_noclip();
        assert!(player.noclip());
        player.toggle_noclip();
        assert!(!player.flying());

        player.set_flying(true);
        player.toggle_noclip();
        assert!(player.noclip());
        player.toggle_noclip();
        assert!(!player.flying());
    }
}
