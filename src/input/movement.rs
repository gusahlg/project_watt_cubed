//! movement.rs reads movement keys and advances the player each frame, resolving
//! collisions against the world. All speeds are expressed per second and scaled by
//! delta time so movement is frame-rate independent.
//!
//! Physics runs in `f64` because at large positions, `f32` steps become
//! too small to register, causing the player to stall.
//! `dt` crosses the `f32`→`f64` boundary here, once, at the physics entry.
//!
//! The movement intent comes from the input router: three axes in `[-1, 1]`
//! and the held/toggle states, resolved from bindings (see
//! [`MoveInput::from_view`]) rather than read directly here.
use voxel_engine::DVec3;

use crate::input::intent::{GameplayAxis, GameplayEvent, GameplayState};
use crate::input::router::Gameplay;
use crate::math::{PER_METER, WORLD_BORDER, block_coord};
use crate::player::{Motion, Player, Stance, collision_box};
use crate::world::World;

const SPRINT_MULT: f64 = 1.5; // horizontal speed multiplier while sprinting
const GRAVITY: f64 = 24.0 * PER_METER; // metres / second^2, in world units
const JUMP_SPEED: f64 = 8.5 * PER_METER; // initial upward velocity of a jump
/// Velocity-approach rates (units / second of exponential response). Acceleration,
/// braking, friction, and sprint transitions are all the *same* operation — velocity
/// chasing a target — so a single rate per context is the only knob. A high ground
/// rate keeps control snappy; a low air rate leaves a jump mostly ballistic with a
/// little steer; flying sits in between for responsive free movement.
const GROUND_ACCEL: f64 = 14.0;
const AIR_ACCEL: f64 = 2.0;
const FLY_ACCEL: f64 = 8.0;
/// Fastest fall, units / second. Reached well past any normal jump arc,
/// so jump and short-fall feel are unchanged. Its real job is bounding the
/// per-frame fall distance so collision substepping has a small, fixed worst case.
const TERMINAL_VELOCITY: f64 = -60.0 * PER_METER;
/// Largest single collision step along one axis, in units. Axis deltas above
/// this are split into substeps so a fast fall stops at the first solid cell
/// instead of tunneling past thin terrain.
const MAX_COLLISION_STEP: f64 = 0.5;

/// Swimming: horizontal reach is slower than a walk, and every axis chases its
/// target through the same [`approach`] law at a low rate — that single damping
/// *is* the water's drag, which is why swimming needs no separate friction or
/// terminal-velocity clamp. Vertical targets: a full-strength liquid buoys a
/// fully-submerged, idle player up at [`SWIM_FLOAT_SPEED`] until their head breaks
/// the surface, where they instead settle at [`SWIM_SETTLE_SPEED`] and bob; holding
/// ascend/descend overrides both at [`SWIM_VERT_SPEED`].
const SWIM_SPEED: f64 = 4.0 * PER_METER;
const SWIM_ACCEL: f64 = 6.0; // an approach RATE (1/s) — time-domain, never scaled
const SWIM_VERT_SPEED: f64 = 5.0 * PER_METER;
const SWIM_FLOAT_SPEED: f64 = 3.0 * PER_METER;
const SWIM_SETTLE_SPEED: f64 = 1.0 * PER_METER;

/// The movement intent gathered for a single frame.
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
}

/// Advance the player by one frame: build a movement delta from input + physics,
/// then apply it with per-axis collision resolution.
pub fn update_player(player: &mut Player, world: &World, input: &MoveInput, dt: f32) {
    // The one f32 -> f64 physics boundary (see the module docs).
    let dt = dt as f64;

    if input.toggle_fly {
        player.cycle_fly();
    }

    resolve_stance(player, world, input);

    // Reconcile the walking/swimming boundary before integrating, so this frame
    // runs under the right physics the instant the feet cross a water surface.
    let liquid = sample_liquid(player, world);
    reconcile_liquid(player, liquid);

    // Unit horizontal heading from the movement keys (zero when none held); each
    // mode scales it by its own speed.
    let heading = horizontal_heading(player, input);

    // Read the intrinsics before borrowing `motion` mutably below.
    let walk_speed = player.speed;
    let fly_speed = player.fly_speed;

    let delta = match &mut player.motion {
        // Flying: velocity chases a directly-commanded target on all three axes.
        Motion::Flying { velocity, .. } => {
            let target = heading * fly_speed + DVec3::Y * (input.move_y as f64 * fly_speed);
            *velocity = approach(*velocity, target, FLY_ACCEL, dt);
            *velocity * dt
        }
        // Walking: horizontal velocity chases the target (snappier on the ground
        // than in the air); vertical stays the gravity/jump integrator.
        Motion::Walking { velocity, on_ground } => {
            let ground_speed = if input.sprint { walk_speed * SPRINT_MULT } else { walk_speed };
            let target = heading * ground_speed;
            let rate = if *on_ground { GROUND_ACCEL } else { AIR_ACCEL };
            let horiz = approach(DVec3::new(velocity.x, 0.0, velocity.z), target, rate, dt);
            velocity.x = horiz.x;
            velocity.z = horiz.z;

            // Apply the jump before integrating so it takes effect this frame.
            if input.jump && *on_ground {
                velocity.y = JUMP_SPEED;
            }
            velocity.y = (velocity.y - GRAVITY * dt).max(TERMINAL_VELOCITY);
            *velocity * dt
        }
        // Swimming: every axis chases its target through one drag law. Horizontal
        // follows the movement keys; vertical is a held ascend/descend, or — idle —
        // buoyancy that floats the player to the surface and lets them bob there.
        Motion::Swimming { velocity } => {
            let vertical = if input.move_y != 0.0 {
                input.move_y as f64 * SWIM_VERT_SPEED
            } else if liquid.fully_submerged() {
                SWIM_FLOAT_SPEED * liquid.strength()
            } else {
                -SWIM_SETTLE_SPEED
            };
            let target = heading * SWIM_SPEED + DVec3::Y * vertical;
            *velocity = approach(*velocity, target, SWIM_ACCEL, dt);
            *velocity * dt
        }
    };

    move_with_collision(player, world, delta);
}

/// The buoyancy the player is immersed in this frame, sampled at the feet and the
/// eye. Two samples are enough to tell "wading / at the surface" (feet only) from
/// "fully under" (eye too), which is all the swim physics needs.
#[derive(Clone, Copy)]
struct Liquid {
    feet: u8,
    eye: u8,
}

impl Liquid {
    /// Feet in liquid — the player swims rather than walks.
    fn submerged(&self) -> bool {
        self.feet > 0
    }

    /// Head under the surface too — buoyancy floats the player upward.
    fn fully_submerged(&self) -> bool {
        self.eye > 0
    }

    /// Buoyancy strength on a `0.0..=1.0` scale (water ≈ 0.78), from whichever
    /// sample the player is most deeply immersed in.
    fn strength(&self) -> f64 {
        self.feet.max(self.eye) as f64 / 255.0
    }
}

/// Sample the liquid at the player's feet and eye voxels.
fn sample_liquid(player: &Player, world: &World) -> Liquid {
    let p = player.position;
    let (x, z) = (block_coord(p.x), block_coord(p.z));
    Liquid {
        feet: world.buoyancy_at(x, block_coord(player.feet_y()), z),
        eye: world.buoyancy_at(x, block_coord(p.y), z),
    }
}

/// Move the player across the walking/swimming boundary as they enter or leave a
/// liquid, carrying momentum across the switch. Flying is unaffected — it ignores
/// water entirely — so only the grounded/submerged pair converts here.
fn reconcile_liquid(player: &mut Player, liquid: Liquid) {
    let velocity = player.velocity();
    player.motion = match (&player.motion, liquid.submerged()) {
        (Motion::Walking { .. }, true) => Motion::Swimming { velocity },
        (Motion::Swimming { .. }, false) => Motion::Walking { velocity, on_ground: false },
        (motion, _) => *motion,
    };
}

/// The unit-length horizontal movement direction for this frame, or zero when no
/// movement key is held. Computed only when a key is active — otherwise we'd run
/// the yaw trig, a normalize, and a scale just to produce a zero vector.
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
    let blend = 1.0 - (-rate * dt).exp();
    current + (target - current) * blend
}

/// Update the player's [`Stance`] from the sneak key. Crouching down is always
/// possible (the box only shrinks); standing back up needs headroom, so it's
/// refused while a solid cell occupies the taller box — otherwise the player would
/// grow into the ceiling. Sneaking is a walking-only stance: flying uses `LeftShift`
/// to descend, so it never crouches.
fn resolve_stance(player: &mut Player, world: &World, input: &MoveInput) {
    // Sneaking is a walking-only stance: flying uses `LeftShift` to descend and
    // swimming uses it to dive, so neither should crouch the hitbox.
    let want_sneak = input.sneak && !player.flying() && !player.swimming();
    player.stance = match (player.stance, want_sneak) {
        (Stance::Standing, true) => Stance::Sneaking,
        (Stance::Sneaking, false)
            if !world.collides(&collision_box(player.position, Stance::Standing)) =>
        {
            Stance::Standing
        }
        (current, _) => current,
    };
}

/// Apply `delta` one axis at a time so the player slides along walls instead of
/// sticking, and detects when they land on the ground.
///
/// Each axis clamps to ±[`WORLD_BORDER`] as it moves (every substep clamps):
/// the world border IS the clamp. Movement (the only continuous position writer
/// besides `/tp`, which clamps the same way) can therefore never carry a
/// coordinate past ±1e9, the invariant
/// [`block_coord`](crate::math::block_coord)'s overflow-free i32 block math
/// rests on.
///
/// Axis deltas larger than [`MAX_COLLISION_STEP`] are applied in substeps (see
/// [`step_axis`]) so a fast fall stops at the first solid cell it crosses
/// instead of tunneling past thin terrain; a blocked axis leaves the position
/// at the last collision-free substep.
fn move_with_collision(player: &mut Player, world: &World, delta: DVec3) {
    let mut pos = player.position;
    let stance = player.stance;
    // Noclip flight skips the solidity test (but not the world-border clamp) so
    // the player passes through geometry; every axis then reports unblocked.
    let noclip = player.noclip();

    let blocked_x = step_axis(&mut pos, 0, delta.x, world, stance, noclip);
    let blocked_z = step_axis(&mut pos, 2, delta.z, world, stance, noclip);
    let blocked_y = step_axis(&mut pos, 1, delta.y, world, stance, noclip);
    player.position = pos;

    // A velocity component that ran into geometry is spent — zero it so the player
    // doesn't accumulate speed into a wall. Landing (a downward y block) grounds us.
    match &mut player.motion {
        Motion::Flying { velocity, .. } => {
            if blocked_x {
                velocity.x = 0.0;
            }
            if blocked_y {
                velocity.y = 0.0;
            }
            if blocked_z {
                velocity.z = 0.0;
            }
        }
        Motion::Walking { velocity, on_ground } => {
            if blocked_x {
                velocity.x = 0.0;
            }
            if blocked_z {
                velocity.z = 0.0;
            }
            *on_ground = blocked_y && delta.y < 0.0;
            if blocked_y {
                velocity.y = 0.0;
            }
        }
        // Swimming has no ground contact; a blocked axis just spends its velocity,
        // like flying into a wall.
        Motion::Swimming { velocity } => {
            if blocked_x {
                velocity.x = 0.0;
            }
            if blocked_y {
                velocity.y = 0.0;
            }
            if blocked_z {
                velocity.z = 0.0;
            }
        }
    }
}

/// Move `pos` along one `axis` (0 = x, 1 = y, 2 = z) by `delta`, clamping to
/// ±[`WORLD_BORDER`], and stop at the first colliding position. Returns `true`
/// if the move hit something; `pos` is then the last collision-free point
/// reached along the way.
///
/// Deltas of at most [`MAX_COLLISION_STEP`] take a fast path that is the exact
/// historical single-endpoint test (same float ops), so ordinary per-frame
/// movement is untouched. Larger deltas — a long fall, a dt spike — are split
/// into `ceil(|delta| / 0.5)` substeps (≈ 12 at terminal velocity under the
/// game's 0.1 s dt clamp) so no solid cell thicker than half a block can be
/// jumped over. The final substep lands exactly on the single-step endpoint, so
/// an unobstructed move is identical either way.
fn step_axis(
    pos: &mut DVec3,
    axis: usize,
    delta: f64,
    world: &World,
    stance: Stance,
    noclip: bool,
) -> bool {
    let start = pos[axis];

    // Fast path: the common per-frame case, identical to the pre-substepping
    // behavior. Noclip clamps to the border but never consults geometry, so it
    // always lands on the endpoint and reports unblocked.
    if noclip || delta.abs() <= MAX_COLLISION_STEP {
        pos[axis] = (start + delta).clamp(-WORLD_BORDER, WORLD_BORDER);
        if !noclip && world.collides(&collision_box(*pos, stance)) {
            pos[axis] = start;
            return true;
        }
        return false;
    }

    let target = (start + delta).clamp(-WORLD_BORDER, WORLD_BORDER);
    let steps = (delta.abs() / MAX_COLLISION_STEP).ceil() as u32;
    for i in 1..=steps {
        let next = if i == steps {
            target
        } else {
            (start + delta * (i as f64 / steps as f64)).clamp(-WORLD_BORDER, WORLD_BORDER)
        };
        let last_good = pos[axis];
        pos[axis] = next;
        if world.collides(&collision_box(*pos, stance)) {
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
        world.prepare_around(DVec3::new(x, floor_y as f64, z));
        let stone = world.registry().id_by_name("Stone").unwrap();
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
            update_player(&mut player, &world, &walk_forward(), dt); // ramp to full speed
        }
        let window_start = player.position.x;
        for _ in 0..4000 {
            update_player(&mut player, &world, &walk_forward(), dt); // measured second
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
            update_player(&mut player, &world, &input, 0.1);
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
        world.prepare_around(DVec3::new(x, floor_y as f64, z));
        let stone = world.registry().id_by_name("Stone").unwrap();
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
            update_player(&mut player, &world, &idle(), 0.1);
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
        update_player(&mut player, &world, &idle(), dt as f32);
        assert!(player.on_ground(), "must be standing before the jump");
        let start_y = player.position.y;

        let mut apex = start_y;
        for frame in 0..60 {
            let input = MoveInput { jump: frame == 0, ..idle() };
            update_player(&mut player, &world, &input, dt as f32);
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
}
