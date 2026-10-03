//! movement.rs reads movement keys and advances the player each frame, resolving
//! collisions against the world. All speeds are expressed per second and scaled by
//! delta time so movement is frame-rate independent.
//!
//! Gravity is a vector from matter (any direction, any strength). Walking acts in the plane of
//! the collision axis — the grid axis nearest the body's up — and the collision box stands along
//! it, so a Y-up world under `(0, −g, 0)` reproduces the classic integrator bit for bit.
//!
//! Physics runs in `f64` because at large positions, `f32` steps become
//! too small to register, causing the player to stall.
//! `dt` crosses the `f32`→`f64` boundary here, once, at the physics entry.
//!
//! The movement intent comes from the input router: three axes in `[-1, 1]`
//! and the held/toggle states, resolved from bindings (see
//! [`MoveInput::from_view`]) rather than read directly here.
use voxel_engine::DVec3;

use crate::coord::Face;
use crate::input::intent::{GameplayAxis, GameplayEvent, GameplayState};
use crate::input::router::Gameplay;
use crate::math::{PER_METER, WORLD_BORDER};
use crate::player::{Motion, Player, STANDARD_GRAVITY, Stance, collision_box, feet_of};
use crate::world::World;

const SPRINT_MULT: f64 = 1.5; // horizontal speed multiplier while sprinting
#[cfg(test)]
const GRAVITY: f64 = STANDARD_GRAVITY; // the reference pull the tests stand in
const JUMP_SPEED: f64 = 8.5 * PER_METER; // initial upward velocity of a jump
/// Velocity-approach rates (units / second of exponential response). Acceleration,
/// braking, friction, and sprint transitions are all the *same* operation — velocity
/// chasing a target — so a single rate per context is the only knob. A high ground
/// rate keeps control snappy; a low air rate leaves a jump mostly ballistic with a
/// little steer; flying sits in between for responsive free movement.
const GROUND_ACCEL: f64 = 14.0;
const AIR_ACCEL: f64 = 2.0;
const FLY_ACCEL: f64 = 8.0;
/// Fastest fall, units / second, along the collision axis. Reached well past any normal jump
/// arc, so jump and short-fall feel are unchanged. Its real job is bounding the per-frame fall
/// distance so collision substepping has a small, fixed worst case (a declared numerical bound,
/// not drag).
const TERMINAL_SPEED: f64 = 60.0 * PER_METER;
#[cfg(test)]
const TERMINAL_VELOCITY: f64 = -TERMINAL_SPEED;
/// Steepest ground (between −gravity and the contact normal) that holds a standing player;
/// beyond it the player slides. Above the 54.7° of a cube corner so corners stay walkable.
const MAX_SLOPE_COS: f64 = 0.53; // cos 58°
/// The collision axis only switches when the new axis leads the old one by this much in cosine.
const AXIS_HYSTERESIS: f64 = 0.05;
/// Largest single collision step along one axis, in units. Axis deltas above
/// this are split into substeps so a fast fall stops at the first solid cell
/// instead of tunneling past thin terrain.
const MAX_COLLISION_STEP: f64 = 0.5;

/// The movement intent gathered for a single frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveInput {
    move_x: f32,
    move_y: f32,
    move_z: f32,
    jump: bool,
    toggle_fly: bool,
    /// Held: move horizontally faster on the ground.
    sprint: bool,
    /// Held: crouch to the shorter (sneaking) hitbox. Shares LeftShift with descent (safe: descent needs flying).
    sneak: bool,
}

impl MoveInput {
    /// Resolve this frame's movement intent from the gameplay view.
    pub fn from_view(gp: &Gameplay) -> Self {
        Self {
            move_x: gp.axis(GameplayAxis::MoveX),
            move_y: gp.axis(GameplayAxis::MoveY),
            move_z: gp.axis(GameplayAxis::MoveZ),
            jump: gp.state(GameplayState::Jump),
            toggle_fly: gp.event(GameplayEvent::ToggleFly),
            sprint: gp.state(GameplayState::Sprint),
            sneak: gp.state(GameplayState::Sneak),
        }
    }

    /// Edge-triggered flight toggle, exposed so a slower fixed physics clock can
    /// latch the event until it actually executes a tick.
    pub(crate) fn toggle_fly(&self) -> bool {
        self.toggle_fly
    }

    /// Held jump state, exposed so a fixed physics clock can also retain a
    /// short press that begins and ends between two physics ticks.
    pub(crate) fn jump(&self) -> bool {
        self.jump
    }

    /// Override only the edge-triggered field when replaying held input across
    /// fixed ticks; all held axes/states remain the current frame's values.
    pub(crate) fn set_toggle_fly(&mut self, toggle: bool) {
        self.toggle_fly = toggle;
    }

    pub(crate) fn set_jump(&mut self, jump: bool) {
        self.jump = jump;
    }

    /// Reuse the already-sampled axes for detached freecam instead of probing
    /// the same movement bindings a second time in the frame.
    pub(crate) fn freecam_axes(&self) -> (f64, f64, f64, bool) {
        (self.move_z as f64, self.move_x as f64, self.move_y as f64, self.sprint)
    }
}

/// Advance the player by one frame under `gravity` (units/s², any direction): build a movement
/// delta from input + physics, then apply it with per-axis collision resolution. Returns landing
/// trauma in `[0, 1]` (zero when the player did not land this tick).
pub fn update_player(player: &mut Player, world: &World, input: &MoveInput, dt: f32, gravity: DVec3) -> f32 {
    step(player, world, input, dt, gravity, WORLD_BORDER)
}

/// Advance a player standing in a curved patch of `atlas` (a round world): the step runs in the
/// patch's storage frame — axis-aligned cells, the ordinary collision code — and the result is
/// embedded back exactly. Velocity, gravity and the body frame cross through the patch's local
/// Jacobian; walking speeds are rescaled so physical speed stays the same where cells are narrower
/// than a block, while the hitbox stays in cells (a two-cell opening fits the player everywhere).
/// Outside every patch this is [`update_player`].
pub fn update_player_in(player: &mut Player, world: &World, atlas: &crate::space::atlas::Atlas, input: &MoveInput, dt: f32, gravity: DVec3) -> f32 {
    let Some(here) = atlas.local(player.position) else {
        return update_player(player, world, input, dt, gravity);
    };
    let (j, ji) = (here.jacobian, here.jacobian.inverse());
    let rot = glam::DQuat::from_mat3(&here.rotation());
    let width = 0.5 * (j.x_axis.length() + j.z_axis.length());
    let (speed, fly_speed) = (player.speed, player.fly_speed);
    player.speed = speed / width;
    player.fly_speed = fly_speed / width;
    player.position = here.storage;
    set_velocity(player, ji * player.velocity());
    player.orientation.frame = (rot.inverse() * player.orientation.frame).normalize();
    let trauma = step(player, world, input, dt, ji * gravity, crate::math::CELL_LIMIT);
    player.position = atlas.embed_storage(here.patch, player.position);
    set_velocity(player, j * player.velocity());
    player.orientation.frame = (rot * player.orientation.frame).normalize();
    // The camera aligns to the physical pull between steps.
    player.gravity = gravity;
    player.speed = speed;
    player.fly_speed = fly_speed;
    trauma
}

fn set_velocity(player: &mut Player, v: DVec3) {
    match &mut player.motion {
        Motion::Walking { velocity, .. } | Motion::Flying { velocity, .. } => *velocity = v,
    }
}

/// One physics step inside positions bounded by `±border`.
fn step(player: &mut Player, world: &World, input: &MoveInput, dt: f32, gravity: DVec3, border: f64) -> f32 {
    // The one f32 -> f64 physics boundary (see the module docs).
    let dt = dt as f64;
    player.gravity = gravity;

    if input.toggle_fly {
        player.cycle_fly();
    }

    resolve_axis(player, world);
    resolve_stance(player, world, input);

    // Unit heading from the movement keys in the body's horizontal plane (zero when none held);
    // each mode scales it by its own speed.
    let heading = horizontal_heading(player, input);
    let up = player.up();
    let axis = player.up_axis;

    // Read the intrinsics before borrowing `motion` mutably below.
    let walk_speed = player.speed;
    let fly_speed = player.fly_speed;

    let delta = match &mut player.motion {
        // Flying: velocity chases a directly-commanded target on all three body axes.
        Motion::Flying { velocity, .. } => {
            let target = heading * fly_speed + up * (input.move_y as f64 * fly_speed);
            *velocity = approach(*velocity, target, FLY_ACCEL, dt);
            *velocity * dt
        }
        // Walking: the velocity across the collision axis chases the target (snappier on the
        // ground than in the air); the component along it is the gravity/jump integrator.
        Motion::Walking { velocity, on_ground } => {
            let (a, s) = (axis.axis(), axis.sign() as f64);
            let ground_speed = if input.sprint { walk_speed * SPRINT_MULT } else { walk_speed };
            let target = level(heading, a) * ground_speed;
            let rate = if *on_ground { GROUND_ACCEL } else { AIR_ACCEL };
            let mut across = *velocity;
            across[a] = 0.0;
            let mut across = approach(across, target, rate, dt);
            let mut along = velocity[a] * s;

            // Apply the jump before integrating so it takes effect this frame.
            if input.jump && *on_ground {
                along = JUMP_SPEED;
            }
            along = (along + gravity[a] * s * dt).max(-TERMINAL_SPEED);
            // Standing on ground that is not too steep, static friction cancels the pull across
            // the contact; in the air or on a steep face it accelerates the player.
            if !(*on_ground && holds(gravity, axis)) {
                let mut pull = gravity;
                pull[a] = 0.0;
                across += pull * dt;
            }
            across[a] = along * s;
            *velocity = across;
            across * dt
        }
    };

    move_with_collision(player, world, delta, border)
}

/// `heading` with its component along axis `a` removed, rescaled to its old length: walking
/// follows the ground plane even when the body is tilted against the grid.
fn level(heading: DVec3, a: usize) -> DVec3 {
    if heading[a] == 0.0 {
        return heading;
    }
    let mut h = heading;
    h[a] = 0.0;
    let len = h.length();
    if len < 1e-9 { DVec3::ZERO } else { h * (heading.length() / len) }
}

/// Whether ground facing `axis` holds a standing player under `gravity` (not steeper than
/// [`MAX_SLOPE_COS`]).
fn holds(gravity: DVec3, axis: Face) -> bool {
    let g = gravity.length();
    g == 0.0 || -gravity[axis.axis()] * axis.sign() as f64 >= MAX_SLOPE_COS * g
}

/// Keep the collision axis on the grid axis nearest the body's up. It switches only when the new
/// axis clearly leads (hysteresis) and the re-stood box is free; the feet stay put, the eye moves.
fn resolve_axis(player: &mut Player, world: &World) {
    let up = player.up();
    let best = Face::from_dominant(up);
    let cur = player.up_axis;
    if best == cur {
        return;
    }
    let lead = |f: Face| up[f.axis()] * f.sign() as f64;
    if lead(best) < lead(cur) + AXIS_HYSTERESIS {
        return;
    }
    let feet = feet_of(player.position, player.stance, cur);
    let mut eye = feet;
    eye[best.axis()] += best.sign() as f64 * player.stance.eye_offset();
    if player.noclip() || !world.collides(&collision_box(eye, player.stance, best)) {
        player.position = eye;
        player.up_axis = best;
    }
}

/// The unit-length movement direction in the body's horizontal plane for this frame, or zero
/// when no movement key is held. Computed only when a key is active — otherwise we'd run the yaw
/// trig, a normalize, and a scale just to produce a zero vector.
fn horizontal_heading(player: &Player, input: &MoveInput) -> DVec3 {
    if input.move_z == 0.0 && input.move_x == 0.0 {
        return DVec3::ZERO;
    }
    let (forward, right) = player.movement_basis();
    let direction = forward * input.move_z as f64 + right * input.move_x as f64;

    // `forward` and `right` are already unit length, so a single axis needs no
    // normalize. Only a diagonal (both axes active) would otherwise move sqrt(2)
    // too fast, so that's the only case we pay for the sqrt.
    if input.move_z != 0.0 && input.move_x != 0.0 {
        direction.normalize()
    } else {
        direction
    }
}

/// Move `current` velocity toward `target` by an exponential, frame-rate-correct
/// step. One law covers acceleration, braking, and all speed changes — no separate
/// accel/decel clamps.
fn approach(current: DVec3, target: DVec3, rate: f64, dt: f64) -> DVec3 {
    let delta = target - current;
    // Exponential never reaches the target; snap so idle X/Z hits the
    // `delta == 0` collision fast path instead of three collides per tick.
    if delta.length_squared() < 1e-8 {
        return target;
    }
    let blend = 1.0 - (-rate * dt).exp();
    current + delta * blend
}

/// Update the player's [`Stance`] from the sneak key. Crouching down is always
/// possible (the box only shrinks); standing back up needs headroom, so it's
/// refused while a solid cell occupies the taller box — otherwise the player would
/// grow into the ceiling. Sneaking is a walking-only stance: flying uses `LeftShift`
/// to descend, so it never crouches.
fn resolve_stance(player: &mut Player, world: &World, input: &MoveInput) {
    // Sneaking is a walking-only stance: flying uses `LeftShift` to descend, so
    // it should not crouch the hitbox.
    let want_sneak = input.sneak && !player.flying();
    player.stance = match (player.stance, want_sneak) {
        (Stance::Standing, true) => Stance::Sneaking,
        (Stance::Sneaking, false)
            if !world.collides(&collision_box(player.position, Stance::Standing, player.up_axis)) =>
        {
            Stance::Standing
        }
        (current, _) => current,
    };
}

/// Apply `delta` one axis at a time so the player slides along walls instead of
/// sticking, and detects when they land on the ground. The two axes across the
/// collision axis go first, the collision axis last (Y-up: X, Z, then Y).
///
/// Every substep clamps to ±[`WORLD_BORDER`]. Positions stay inside the range
/// [`math::block_coord`](crate::math::block_coord) assumes. Axis deltas larger
/// than [`MAX_COLLISION_STEP`] go through [`step_axis`] substeps.
fn move_with_collision(player: &mut Player, world: &World, delta: DVec3, border: f64) -> f32 {
    let mut pos = player.position;
    let stance = player.stance;
    let up = player.up_axis;
    let (a, s) = (up.axis(), up.sign() as f64);
    // Noclip flight skips the solidity test (but not the world-border clamp) so
    // the player passes through geometry; every axis then reports unblocked.
    let noclip = player.noclip();

    let mut blocked = [false; 3];
    for axis in [0, 2, 1].into_iter().filter(|&i| i != a).chain([a]) {
        blocked[axis] = step_axis(&mut pos, axis, delta[axis], world, stance, up, noclip, border);
    }
    player.position = pos;

    // A velocity component that ran into geometry is spent — zero it so the player
    // doesn't accumulate speed into a wall. Landing (a block moving down the axis) grounds us.
    match &mut player.motion {
        Motion::Flying { velocity, .. } => {
            for axis in 0..3 {
                if blocked[axis] {
                    velocity[axis] = 0.0;
                }
            }
        }
        Motion::Walking { velocity, on_ground } => {
            for axis in (0..3).filter(|&i| i != a) {
                if blocked[axis] {
                    velocity[axis] = 0.0;
                }
            }
            let landed = blocked[a] && delta[a] * s < 0.0;
            let trauma = if landed { ((-(velocity[a] * s)) / TERMINAL_SPEED) as f32 } else { 0.0 };
            *on_ground = landed;
            if blocked[a] {
                velocity[a] = 0.0;
            }
            return trauma.min(1.0);
        }
    }
    0.0
}

/// Move `pos` along one `axis` (0 = x, 1 = y, 2 = z) by `delta`, clamping to
/// ±[`WORLD_BORDER`], and stop at the first colliding position. Returns `true`
/// if the move hit something; `pos` is then the last collision-free point.
///
/// `|delta| <= MAX_COLLISION_STEP`: one endpoint test. Larger deltas split into
/// `ceil(|delta| / 0.5)` substeps (~12 at terminal velocity under the 0.1 s dt
/// clamp). The last substep is the single-step endpoint, so an unobstructed
/// move matches the one-step path.
#[allow(clippy::too_many_arguments)]
fn step_axis(
    pos: &mut DVec3,
    axis: usize,
    delta: f64,
    world: &World,
    stance: Stance,
    up: Face,
    noclip: bool,
    border: f64,
) -> bool {
    // Standing still is overwhelmingly common. Avoid an AABB build plus a
    // world collision query for the two (often all three) idle axes.
    if delta == 0.0 {
        return false;
    }
    let start = pos[axis];

    // Fast path. Noclip clamps to the border and skips geometry.
    if noclip || delta.abs() <= MAX_COLLISION_STEP {
        pos[axis] = (start + delta).clamp(-border, border);
        if !noclip && world.collides(&collision_box(*pos, stance, up)) {
            pos[axis] = start;
            return true;
        }
        return false;
    }

    let target = (start + delta).clamp(-border, border);
    let steps = (delta.abs() / MAX_COLLISION_STEP).ceil() as u32;
    for i in 1..=steps {
        let next = if i == steps {
            target
        } else {
            (start + delta * (i as f64 / steps as f64)).clamp(-border, border)
        };
        let last_good = pos[axis];
        pos[axis] = next;
        if world.collides(&collision_box(*pos, stance, up)) {
            pos[axis] = last_good;
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::block_coord;
    use crate::player::Stance;

    /// The reference pull straight down −Y.
    fn down() -> DVec3 {
        DVec3::new(0.0, -GRAVITY, 0.0)
    }

    /// Standing eye height above the feet — the feet→eye conversion these tests use
    /// to place a player whose feet rest on a given block.
    fn stand_eye() -> f64 {
        Stance::Standing.eye_offset()
    }

    /// Hold forward, nothing else — the reported far-coordinate stall scenario.
    fn walk_forward() -> MoveInput {
        MoveInput { move_z: 1.0, ..idle() }
    }

    /// No keys held at all — freefall / settle frames.
    fn idle() -> MoveInput {
        MoveInput {
            move_x: 0.0,
            move_y: 0.0,
            move_z: 0.0,
            jump: false,
            toggle_fly: false,
            sprint: false,
            sneak: false,
        }
    }

    /// Build a flat stone runway at `y = floor_y` under the given start, long
    /// enough for the walk tests, and stand the player on it. The generator
    /// puts real terrain up here (heights reach ~52 at the far columns), so
    /// standing room is CARVED above the runway — the runway must be the only
    /// geometry the walker can touch.
    fn player_on_runway(world: &mut World, x: f64, z: f64) -> Player {
        let floor_y = 40;
        world.ensure_around(DVec3::new(x, floor_y as f64, z));
        let stone = world.registry().id_by_label("rock").unwrap();
        let (bx, bz) = (block_coord(x), block_coord(z));
        for dx in -2..=14 {
            for dz in -2..=2 {
                world.set_block(bx + dx, floor_y, bz + dz, stone);
                for y in (floor_y + 1)..=(floor_y + 3) {
                    world.set_block(bx + dx, y, bz + dz, crate::block::AIR);
                }
            }
        }
        // Feet on top of the runway: eye = feet + standing eye offset.
        let mut player = Player::new(DVec3::new(x, (floor_y + 1) as f64 + stand_eye(), z));
        player.orientation.yaw = 0.0; // forward = +X
        player
    }

    #[test]
    fn walking_at_1e8_advances_at_full_speed_at_4000_fps() {
        // The proven bug: at x = 1e8 an f32 position's ULP (8.0!) dwarfs the
        // per-frame step 6.0/4000 = 0.0015, so f32 movement added ZERO for a
        // whole second of frames. In f64 the step lands. We measure *steady-state*
        // speed — velocity now ramps in via the accel law, so a first-second
        // distance would undercount the ramp; warm up, then measure a clean second.
        let mut world = World::generate();
        let start_x = 1.0e8 + 0.5;
        let mut player = player_on_runway(&mut world, start_x, 0.5);

        let dt = 1.0 / 4000.0;
        for _ in 0..4000 {
            update_player(&mut player, &world, &walk_forward(), dt, down()); // ramp to full speed
        }
        let window_start = player.position.x;
        for _ in 0..4000 {
            update_player(&mut player, &world, &walk_forward(), dt, down()); // measured second
        }

        let moved = player.position.x - window_start;
        assert!(
            (moved - crate::player::DEFAULT_WALK_SPEED).abs() < 0.05,
            "one steady-state second at WALK_SPEED must cover ~{} blocks, moved {moved}",
            crate::player::DEFAULT_WALK_SPEED
        );
        assert!(player.on_ground(), "still standing on the runway");
        assert_eq!(player.position.z, 0.5, "no lateral drift");
    }

    #[test]
    fn movement_clamps_at_the_world_border_and_stays_finite() {
        // Leave the border region unloaded (unloaded chunks read as air), so fly
        // there — the clamp is the border, not a wall of blocks or a flying island.
        let world = World::generate();
        let start_x = WORLD_BORDER - 1000.0;
        let mut player = Player::new(DVec3::new(start_x, 300.0, 0.5));
        player.set_flying(true);
        player.orientation.yaw = 0.0; // forward = +X, straight at the border

        // 2000 steps x 0.1s x 14 units/s = 2800 blocks of intent: crosses the
        // remaining 1000 and keeps pushing.
        for _ in 0..2000 {
            let input = walk_forward();
            update_player(&mut player, &world, &input, 0.1, down());
        }

        assert!(player.position.x.is_finite());
        assert_eq!(
            player.position.x, WORLD_BORDER,
            "the border clamps forward progress exactly"
        );
        assert!(player.position.y.is_finite() && player.position.z.is_finite());
        // And block conversion of the clamped position is still safe i32.
        assert_eq!(block_coord(player.position.x), 1_000_000_000);
    }

    /// The tunneling regression: a terminal-velocity fall onto a one-block-thin
    /// floor must STOP on it. Before substepping, a 6-unit frame step (terminal
    /// 60 × the game's 0.1 s dt clamp) tested only its endpoint and could jump
    /// the floor's entire 1-block extent, dropping the player through.
    #[test]
    fn terminal_velocity_fall_stops_on_a_one_block_thin_floor() {
        let mut world = World::generate();
        let (x, z) = (0.5, 0.5);
        let floor_y = 40; // above the hills, below the island band: open air
        world.ensure_around(DVec3::new(x, floor_y as f64, z));
        let stone = world.registry().id_by_label("rock").unwrap();
        let (bx, bz) = (block_coord(x), block_coord(z));

        // Feet start 158 blocks up: freefall reaches terminal velocity after
        // ~78 blocks (2.5 s), leaving a long terminal-speed run whose 6-unit
        // steps hit the pre-fix tunneling window when they cross the floor.
        let start_feet = (floor_y + 1) as f64 + 158.0;

        // Build the exact scenario instead of hoping generation left it empty: a
        // one-block-thin platform under a carved-air fall column. Carving makes the
        // test independent of whatever terrain generation happens to place here.
        for dx in -2..=2 {
            for dz in -2..=2 {
                world.set_block(bx + dx, floor_y, bz + dz, stone);
                for y in (floor_y + 1)..=(start_feet as i32 + 2) {
                    world.set_block(bx + dx, y, bz + dz, crate::block::registry::AIR);
                }
            }
        }

        let mut player = Player::new(DVec3::new(x, start_feet + stand_eye(), z));

        let mut reached_terminal = false;
        for _ in 0..100 {
            update_player(&mut player, &world, &idle(), 0.1, down());
            assert!(
                player.velocity().y >= TERMINAL_VELOCITY,
                "velocity must never exceed terminal, got {}",
                player.velocity().y
            );
            if player.velocity().y == TERMINAL_VELOCITY {
                reached_terminal = true;
            }
        }

        assert!(reached_terminal, "158 blocks of freefall must reach terminal velocity");
        assert!(player.on_ground(), "the fall must end standing on the thin floor");
        let feet = player.position.y - stand_eye();
        let top = (floor_y + 1) as f64;
        assert!(
            feet >= top - 1e-9 && feet < top + 0.3,
            "feet must rest on the platform top ({top}), got {feet}"
        );
    }

    /// Neither the terminal-velocity clamp nor collision substepping may change
    /// how a jump feels: the apex of a normal jump must match the historical
    /// integrator exactly. (A jump peaks at |v| = 8.5, nowhere near terminal,
    /// and 60 fps deltas stay under the 0.5 substep threshold, so the fast path
    /// runs the same float ops as before the change.)
    #[test]
    fn jump_apex_is_unchanged() {
        let mut world = World::generate();
        let mut player = player_on_runway(&mut world, 0.5, 0.5);
        let dt = 1.0 / 60.0;

        // One idle frame to plant the player (Player::new starts !on_ground).
        update_player(&mut player, &world, &idle(), dt as f32, down());
        assert!(player.on_ground(), "must be standing before the jump");
        let start_y = player.position.y;

        let mut apex = start_y;
        for frame in 0..60 {
            let input = MoveInput { jump: frame == 0, ..idle() };
            update_player(&mut player, &world, &input, dt as f32, down());
            apex = apex.max(player.position.y);
        }

        // The jump reaches apex in 21 frames at 60 fps. Calculate the expected
        // height to verify the integrator hasn't changed.
        // (`dt` crosses the physics boundary as f32, so mirror that rounding.)
        let dt = (dt as f32) as f64;
        let n = 21.0_f64;
        let expected = JUMP_SPEED * n * dt - GRAVITY * dt * dt * (n * (n + 1.0) / 2.0);
        let jumped = apex - start_y;
        assert!(
            (jumped - expected).abs() < 1e-9,
            "jump apex changed: expected +{expected}, got +{jumped}"
        );
    }

    /// A 9x9 slab of rock whose top face points along `up`, with open air carved above it, and a
    /// player standing on it with the body frame snapped to `up`.
    fn player_on_slab(world: &mut World, up: Face) -> Player {
        let centre = DVec3::new(8.5, 300.5, 8.5);
        world.ensure_around(centre);
        let stone = world.registry().id_by_label("rock").unwrap();
        let (a, s) = (up.axis(), up.sign());
        let base = [8, 300, 8];
        for i in -4..=4 {
            for j in -4..=4 {
                for k in 0..=4 {
                    let mut c = base;
                    let (t1, t2) = match a { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
                    c[t1] += i;
                    c[t2] += j;
                    c[a] += s * k;
                    let id = if k == 0 { stone } else { crate::block::AIR };
                    world.set_block(c[0], c[1], c[2], id);
                }
            }
        }
        let mut eye = centre;
        eye[a] = base[a] as f64 + if s > 0 { 1.0 } else { 0.0 } + s as f64 * stand_eye();
        let mut player = Player::new(eye);
        player.snap_up(up.dvec());
        player
    }

    fn gravity_along(f: Face, g: f64) -> DVec3 {
        -f.dvec() * g
    }

    #[test]
    fn standing_and_walking_on_a_side_face_follow_its_axis() {
        let mut world = World::generate();
        let mut player = player_on_slab(&mut world, Face::PosX);
        let g = gravity_along(Face::PosX, GRAVITY);
        for _ in 0..30 {
            update_player(&mut player, &world, &idle(), 1.0 / 60.0, g);
        }
        assert_eq!(player.up_axis, Face::PosX);
        assert!(player.on_ground(), "standing on the +X face");
        let feet = player.feet();
        assert!((feet.x - 9.0).abs() < 1e-9, "feet rest on the slab top at x = 9, got {}", feet.x);
        let start = player.position;
        // Half a second of walking stays well inside the 9-block slab.
        for _ in 0..30 {
            update_player(&mut player, &world, &walk_forward(), 1.0 / 60.0, g);
        }
        let moved = player.position - start;
        assert!(moved.x.abs() < 1e-9, "walking stays on the face: {moved}");
        assert!(moved.length() > 1.5, "and actually walks: {moved}");
        assert!(player.on_ground());
    }

    #[test]
    fn a_gentle_tilt_holds_a_standing_player_and_a_steep_one_slides() {
        let mut world = World::generate();
        let mut player = player_on_slab(&mut world, Face::PosY);
        // 20 degrees off the slab normal: friction holds.
        let tilt = |deg: f64| {
            let r = deg.to_radians();
            DVec3::new(r.sin(), -r.cos(), 0.0) * GRAVITY
        };
        for _ in 0..120 {
            update_player(&mut player, &world, &idle(), 1.0 / 60.0, tilt(20.0));
        }
        let held = player.position;
        for _ in 0..120 {
            update_player(&mut player, &world, &idle(), 1.0 / 60.0, tilt(20.0));
        }
        assert!((player.position - held).length() < 1e-9, "a 20° slope holds still");
        // 70 degrees: past the steepest holding slope, the player slides along +X.
        for _ in 0..30 {
            update_player(&mut player, &world, &idle(), 1.0 / 60.0, tilt(70.0));
        }
        assert!(player.position.x > held.x + 0.5, "a 70° slope slides: {}", player.position - held);
    }

    #[test]
    fn zero_gravity_neither_falls_nor_grounds() {
        let mut world = World::generate();
        let mut player = player_on_slab(&mut world, Face::PosY);
        player.position.y += 3.0;
        let start = player.position;
        for _ in 0..60 {
            update_player(&mut player, &world, &idle(), 1.0 / 60.0, DVec3::ZERO);
        }
        assert_eq!(player.position, start, "nothing pulls a resting body in zero g");
        assert!(!player.on_ground());
    }

    #[test]
    fn the_collision_axis_switches_with_hysteresis_and_keeps_the_feet() {
        let mut world = World::generate();
        let mut player = player_on_slab(&mut world, Face::PosY);
        player.position.y += 4.0;
        let feet = player.feet();
        // Up just past the 45° diagonal toward +X but inside the hysteresis band: no switch.
        player.orientation.snap(DVec3::new(1.0, 0.98, 0.0).normalize());
        update_player(&mut player, &world, &idle(), 1.0 / 60.0, DVec3::ZERO);
        assert_eq!(player.up_axis, Face::PosY);
        // Clearly toward +X: the box re-stands along X around the same feet.
        player.orientation.snap(DVec3::new(1.0, 0.6, 0.0).normalize());
        update_player(&mut player, &world, &idle(), 1.0 / 60.0, DVec3::ZERO);
        assert_eq!(player.up_axis, Face::PosX);
        assert!((player.feet() - feet).length() < 1e-9, "{} vs {feet}", player.feet());
    }


    #[test]
    fn a_player_walks_around_a_curved_world_on_its_storage_cells() {
        use crate::space::atlas::{Atlas, Patch};
        let mut world = World::generate();
        let centre = DVec3::new(2.0e7, 3.0e7, -1.0e7);
        let r = 3_000i64;
        let atlas = Atlas::new(centre, r, r + 64, false, 0);
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let b = atlas.bands[0];
        let k = r - b.r_lo - 1; // the cell layer just below the datum radius
        let stone = world.registry().id_by_label("rock").unwrap();
        let mid = b.n / 2;
        let lift = atlas.storage(top, [mid, k, mid]);
        world.ensure_around(DVec3::new(lift[0] as f64, lift[1] as f64, lift[2] as f64));
        for di in -24..=24 {
            for dj in -24..=24 {
                let s = atlas.storage(top, [mid + di, k, mid + dj]);
                world.set_block(s[0] as i32, s[1] as i32, s[2] as i32, stone);
            }
        }
        // Stand above the floor, body up along the radius, gravity toward the centre.
        let start = atlas.embed(top, DVec3::new(mid as f64 + 0.5, (k + 1) as f64 + stand_eye() + 0.2, mid as f64 + 0.5));
        let mut player = Player::new(start);
        player.snap_up((start - centre).normalize());
        let pull = |p: DVec3| (centre - p).normalize() * GRAVITY;
        for _ in 0..60 {
            let g = pull(player.position);
            update_player_in(&mut player, &world, &atlas, &idle(), 1.0 / 60.0, g);
        }
        assert!(player.on_ground(), "landed on the curved floor");
        let rest = (player.position - centre).length();
        assert!((rest - (r as f64 + stand_eye())).abs() < 0.05, "eye at the datum + eye height: {rest}");
        let before = player.position;
        for _ in 0..90 {
            let g = pull(player.position);
            update_player_in(&mut player, &world, &atlas, &walk_forward(), 1.0 / 60.0, g);
        }
        let after = (player.position - centre).length();
        assert!((after - rest).abs() < 0.05, "walking follows the curve: {rest} -> {after}");
        assert!((player.gravity - pull(before)).length() < 0.2, "the stored pull is physical, not storage-frame");
        assert!((player.up() - (player.position - centre).normalize()).length() < 0.05, "the body stands along the radius");
        assert!((player.position - before).length() > 5.0, "and goes somewhere");
        assert!(player.on_ground());
    }

}
