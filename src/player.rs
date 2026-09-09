//! Player state: position, orientation, and the elements they carry. Positions
//! use `f64` for precision out to world borders; view angles use `f32` since
//! rotation doesn't accumulate magnitude (see [`math`](crate::math)).
use voxel_engine::DVec3;

use crate::camera::Orientation;
use crate::math::{Aabb, Bounded, PER_METER};
use crate::stash::{ElementStash, START_CAPACITY};

/// The player's collision half-width on the horizontal axes (x and z). Vertical
/// extent is not a constant — it derives from [`Stance::height`] — so there is no
/// `y` here to fall out of sync with the stance.
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
    /// Submerged in a liquid: buoyancy fights gravity and drag damps every axis,
    /// so there is neither ground contact nor a fall to accumulate — the reason
    /// this is its own variant rather than a flag on `Walking`.
    Swimming { velocity: DVec3 },
    /// Free flight: no gravity, no ground, velocity chases input on every axis.
    /// `noclip` additionally skips collision, letting the player pass through
    /// solid geometry — meaningful only in flight, so it rides on this variant
    /// rather than being a loose flag that could contradict walking/swimming.
    Flying { velocity: DVec3, noclip: bool },
}

impl Motion {
    /// The current velocity, whichever mode we're in.
    pub fn velocity(self) -> DVec3 {
        match self {
            Motion::Walking { velocity, .. }
            | Motion::Swimming { velocity }
            | Motion::Flying { velocity, .. } => velocity,
        }
    }
}

/// The player: where they are, where they're looking, and what they carry.
pub struct Player {
    /// Eye position in world space.
    pub position: DVec3,
    /// View angles — the one orientation; every camera mode is a function of it.
    pub orientation: Orientation,
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
            orientation: Orientation { yaw: 0.0, pitch: 0.0 },
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

    /// Whether the player is swimming in a liquid.
    pub fn swimming(&self) -> bool {
        matches!(self.motion, Motion::Swimming { .. })
    }

    /// Enter or leave flight. Horizontal momentum carries across the switch, but
    /// vertical velocity is cleared so the player neither keeps falling into the
    /// new mode nor launches when leaving it.
    pub fn set_flying(&mut self, flying: bool) {
        let v = self.velocity();
        let velocity = DVec3::new(v.x, 0.0, v.z);
        self.motion = if flying {
            Motion::Flying { velocity, noclip: false }
        } else {
            Motion::Walking { velocity, on_ground: false }
        };
    }

    /// Advance the flight state one step in the cycle
    /// walking → flying → flying+noclip → walking, carrying horizontal momentum
    /// across each switch (vertical is cleared, as in [`Player::set_flying`]).
    /// Landing back to `Walking` lets [`reconcile_liquid`] promote to swimming
    /// next frame if the feet are submerged, so no liquid special-case is needed.
    pub fn cycle_fly(&mut self) {
        let v = self.velocity();
        let velocity = DVec3::new(v.x, 0.0, v.z);
        self.motion = match self.motion {
            Motion::Flying { noclip: false, .. } => Motion::Flying { velocity, noclip: true },
            Motion::Flying { noclip: true, .. } => Motion::Walking { velocity, on_ground: false },
            _ => Motion::Flying { velocity, noclip: false },
        };
    }

    /// Drop any accumulated vertical velocity (e.g. after a teleport, so the
    /// player doesn't rocket down on arrival).
    pub fn cancel_fall(&mut self) {
        match &mut self.motion {
            Motion::Walking { velocity, .. }
            | Motion::Swimming { velocity }
            | Motion::Flying { velocity, .. } => velocity.y = 0.0,
        }
    }

    /// The world-space height of the player's feet: the eye dropped by the
    /// current stance's eye offset.
    pub fn feet_y(&self) -> f64 {
        self.position.y - self.stance.eye_offset()
    }

    /// Full view direction, including pitch.
    pub fn forward(&self) -> DVec3 {
        self.orientation.direction()
    }

    /// The forward and right basis vectors on the XZ plane, used for ground
    /// movement. Returned together because they share one `sin`/`cos` of the yaw,
    /// and both come out unit length already (no normalize needed).
    pub fn movement_basis(&self) -> (DVec3, DVec3) {
        let (sin_yaw, cos_yaw) = (self.orientation.yaw as f64).sin_cos();
        let forward = DVec3::new(cos_yaw, 0.0, sin_yaw);
        let right = DVec3::new(-sin_yaw, 0.0, cos_yaw);
        (forward, right)
    }

}

/// The collision box for an eye at `eye` in the given `stance`. Built from the
/// feet up, not the eye: the feet sit [`eye_offset`](Stance::eye_offset) below the
/// eye, and the box rises the stance's full [`height`](Stance::height) from there,
/// so a shorter (sneaking) box lowers both its top and — via the proportional eye
/// offset — the eye itself.
///
/// Free-standing (not a `Player` method) so the collision stepper, which advances a
/// bare eye position, can test candidate boxes without a whole `Player`.
pub fn collision_box(eye: DVec3, stance: Stance) -> Aabb {
    let half_y = stance.height() / 2.0;
    let feet = eye.y - stance.eye_offset();
    Aabb::new(
        DVec3::new(eye.x, feet + half_y, eye.z),
        DVec3::new(PLAYER_HALF_WIDTH, half_y, PLAYER_HALF_WIDTH),
    )
}

impl Bounded for Player {
    fn aabb(&self) -> Aabb {
        collision_box(self.position, self.stance)
    }
}
