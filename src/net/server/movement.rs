//! Moves, teleports and cruise: the movement envelope, the swept body check and the visibility diff.
use super::*;

impl PlayerHandle {
    pub(super) fn correct_position(&self, id: u32, sends: &mut Vec<PendingSend>) {
        if self.ready {
            let frame = ServerMessage::Position { pos: self.pos, frame: self.frame, up: self.up }.encode().into();
            sends.push((id, frame));
        }
    }
}

/// Validation is a plausibility ENVELOPE, not full physics. Each player spends a
/// distance budget ([`move_allowance`]) that refills at the envelope speed and holds
/// at most one burst, so splitting a move into many messages gains nothing.
/// An implausible move is not committed — the server keeps its last accepted
/// position (which edit reach reads), and the client is snapped back with an
/// authoritative [`ServerMessage::Position`]. `/tp` discontinuities arrive as
/// [`ClientMessage::Teleport`] instead.
///
/// Locked: commit the move, keep the grid current, diff visibility, and queue the
/// enter/exit frames in order without waking anyone. Unlocked: wake the writers.
/// Poses inside range go out on the pose tick ([`broadcast_poses`]).
/// The orientation a [`ClientMessage::Move`] reports. A teleport passes `None`
/// and keeps whatever the handle already stored.
pub(super) struct ReportedPose {
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
}

pub(super) fn quat_finite(q: DQuat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}

pub(super) fn on_move(
    shared: &Arc<Mutex<State>>,
    ctx: &Ctx,
    id: u32,
    pos: DVec3,
    yaw: f32,
    pitch: f32,
    frame: DQuat,
    velocity: Vec3,
    up: Face,
    stance: Stance,
) {
    // A NaN position poisons distance checks/grid keys; a NaN angle, frame, or
    // velocity propagates into peer interpolation and render matrices.
    if !pos.x.is_finite()
        || !pos.y.is_finite()
        || !pos.z.is_finite()
        || !yaw.is_finite()
        || !pitch.is_finite()
        || !quat_finite(frame)
        || !velocity.x.is_finite()
        || !velocity.y.is_finite()
        || !velocity.z.is_finite()
    {
        return;
    }
    let mut sends = Vec::new();
    let wake = {
        let mut state = shared.lock_recover();
        let max_speed = state.max_speed;
        // Envelope: the speed the client reports (and the one we last accepted),
        // grown by gravity over the gap, capped by the server's speed limit.
        // Cruise raises that cap only after a `Cruise` message, and only up to
        // the game's cruise ceiling unless the server cap is tighter.
        // Outside the border is refused either way. Solid ground is tested along the
        // path unless this player's noclip policy allows the pass.
        let (free, too_far, left, burst, from, cruising, occupied, occupied_n) = {
            let Some(h) = state.players.get(&id) else { return };
            let free = noclip_allowed(ctx, h);
            let elapsed = h.last_move.elapsed().as_secs_f64().min(MOVE_WINDOW_CAP_SECS);
            let speed = envelope_speed(h, velocity, elapsed, max_speed);
            let burst = MOVE_FLOOR.max(speed * MOVE_SLACK_SECS);
            let available = move_allowance(h, velocity, elapsed, max_speed);
            let distance = h.pos.distance(pos);
            let too_far = outside_world(pos) || distance > available;
            let mut occupied = [(0i32, 0, 0); BODY_CELL_CAP];
            let occupied_n = h.occupied.len().min(BODY_CELL_CAP);
            occupied[..occupied_n].copy_from_slice(&h.occupied[..occupied_n]);
            (free, too_far, available - distance, burst, h.pos, h.cruising, occupied, occupied_n)
        };
        let blocked = !free
            && !too_far
            && move_blocked(&state, ctx, from, pos, cruising, stance, up, &occupied[..occupied_n]);
        if too_far || blocked {
            let Some(h) = state.players.get(&id) else { return };
            h.correct_position(id, &mut sends);
        } else {
            commit_pose(
                &mut state,
                id,
                pos,
                Some(ReportedPose { yaw, pitch, frame, velocity, up, stance }),
                &mut sends,
            );
            if let Some(h) = state.players.get_mut(&id) {
                h.budget = left;
                h.burst = burst;
                if !free {
                    remember_occupied(h, pos, stance, up);
                }
            }
        }
        queue(&state, sends)
    };
    drop(wake);
}

pub(super) fn noclip_allowed(ctx: &Ctx, h: &PlayerHandle) -> bool {
    match ctx.noclip {
        NoclipPolicy::All => true,
        NoclipPolicy::Ops => is_operator(ctx, h),
        NoclipPolicy::Off => false,
    }
}

pub(super) fn body_stance(stance: Stance) -> player::Stance {
    match stance {
        Stance::Standing => player::Stance::Standing,
        Stance::Sneaking => player::Stance::Sneaking,
    }
}

/// Physical cells the body strictly overlaps. `None` when the box is larger than
/// the stack, which the caller treats as blocked.
pub(super) fn fill_body_cells(pos: DVec3, stance: Stance, up: Face, out: &mut [(i32, i32, i32); BODY_CELL_CAP]) -> Option<usize> {
    let body = player::collision_box(pos, body_stance(stance), up);
    let mut n = 0;
    for cell in body.voxel_cells() {
        if n >= BODY_CELL_CAP {
            return None;
        }
        out[n] = cell;
        n += 1;
    }
    Some(n)
}

pub(super) fn remember_occupied(h: &mut PlayerHandle, pos: DVec3, stance: Stance, up: Face) {
    let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
    let Some(n) = fill_body_cells(pos, stance, up, &mut cells) else {
        return;
    };
    if h.occupied.len() == n && h.occupied.iter().zip(cells[..n].iter()).all(|(have, cell)| have == cell) {
        return;
    }
    h.occupied.clear();
    h.occupied.extend_from_slice(&cells[..n]);
}

/// The body's path from the last accepted pose meets solid ground. A path longer than
/// [`SWEEP_LIMIT`] fails closed, except under cruise, whose destination alone is tested.
#[allow(clippy::too_many_arguments)] // the move's pose fields, passed separately like on_move's
pub(super) fn move_blocked(
    state: &State,
    ctx: &Ctx,
    from: DVec3,
    to: DVec3,
    cruising: bool,
    stance: Stance,
    up: Face,
    occupied: &[(i32, i32, i32)],
) -> bool {
    let from = if from.distance(to) <= SWEEP_LIMIT {
        from
    } else if cruising {
        to
    } else {
        return true;
    };
    body_blocked(state, &ctx.generator, &ctx.seams, from, to, stance, up, occupied)
}

/// True when the body, swept from `from` to `to` in steps of at most [`SWEEP_STEP`], newly
/// enters a cell that is solid in the edit overlay or the generator. Each step queries only
/// the cells the step before did not hold; the first skips `occupied`, the cells of the last
/// accepted pose. A straight path never re-enters a cell it left, so one step back is enough.
/// Cells are read the way the client's collision reads them: in storage, with a cell just
/// past a chart's box glued to the neighbour chart's cell.
#[allow(clippy::too_many_arguments)] // the move's pose fields, passed separately like on_move's
pub(super) fn body_blocked(
    state: &State,
    generator: &crate::world::terrain::Generator,
    seams: &Seams,
    from: DVec3,
    to: DVec3,
    stance: Stance,
    up: Face,
    occupied: &[(i32, i32, i32)],
) -> bool {
    let steps = (from.distance(to) / SWEEP_STEP).ceil().max(1.0) as u32;
    let mut held = [(0i32, 0, 0); BODY_CELL_CAP];
    let mut held_n = occupied.len().min(BODY_CELL_CAP);
    held[..held_n].copy_from_slice(&occupied[..held_n]);
    let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
    for step in 1..=steps {
        let at = if step == steps { to } else { from.lerp(to, f64::from(step) / f64::from(steps)) };
        let Some(n) = fill_body_cells(at, stance, up, &mut cells) else {
            return true;
        };
        for &(x, y, z) in &cells[..n] {
            if held[..held_n].contains(&(x, y, z)) {
                continue;
            }
            let query = seams.glue_cell(BlockCoord::new(x, y, z)).map_or((x, y, z), |g| (g.x, g.y, g.z));
            if state.registry.is_solid(server_block(state, generator, query)) {
                return true;
            }
        }
        std::mem::swap(&mut held, &mut cells);
        held_n = n;
    }
    false
}

/// Snap to the last accepted pose without applying the request. Used when a
/// teleport is over its budget: the client still gets an answer.
pub(super) fn refuse_move(shared: &Arc<Mutex<State>>, id: u32) {
    let mut sends = Vec::new();
    let wake = {
        let state = shared.lock_recover();
        if let Some(h) = state.players.get(&id) {
            h.correct_position(id, &mut sends);
        }
        queue(&state, sends)
    };
    drop(wake);
}

pub(super) fn speed_of(v: Vec3) -> f64 {
    let s = (v.x as f64).hypot(v.y as f64).hypot(v.z as f64);
    if s.is_finite() { s } else { 0.0 }
}

/// Cruise ceiling. At or above [`crate::player::MAX_SPEED`] the declared cruise
/// may reach [`crate::player::CRUISE_MAX`]. A tighter server cap bounds cruise too.
pub(super) fn cruise_limit(max_speed: f64) -> f64 {
    if max_speed < crate::player::MAX_SPEED {
        max_speed
    } else {
        crate::player::CRUISE_MAX
    }
}

pub(super) fn move_cap(h: &PlayerHandle, max_speed: f64) -> f64 {
    if h.cruising {
        h.cruise_speed.min(cruise_limit(max_speed))
    } else {
        max_speed
    }
}

/// Distance this move may cover: the banked budget, at most one burst, plus what `elapsed`
/// refills at the envelope speed. That speed is the reported one (or the last accepted one)
/// grown by gravity over the gap, capped by [`move_cap`]. `reported` is a cap input, not a
/// grant: a cruise declaration does nothing until the velocity the client reports (bounded by
/// that declaration) justifies the hop. The burst is [`MOVE_FLOOR`], or [`MOVE_SLACK_SECS`] at
/// that speed when larger, so over any window the total stays within one burst plus the
/// speed times the window, however the client splits it.
pub(super) fn move_allowance(h: &PlayerHandle, reported: Vec3, elapsed: f64, max_speed: f64) -> f64 {
    let speed = envelope_speed(h, reported, elapsed, max_speed);
    let burst = MOVE_FLOOR.max(speed * MOVE_SLACK_SECS);
    (h.budget / h.burst).clamp(0.0, 1.0) * burst + speed * elapsed
}

pub(super) fn envelope_speed(h: &PlayerHandle, reported: Vec3, elapsed: f64, max_speed: f64) -> f64 {
    move_cap(h, max_speed).min(speed_of(reported).max(speed_of(h.velocity)) + GRAVITY_BOUND * elapsed)
}

pub(super) fn clamp_velocity(v: Vec3, cap: f64) -> Vec3 {
    let s = speed_of(v);
    if s > cap && cap > 0.0 {
        let k = (cap / s) as f32;
        Vec3::new(v.x * k, v.y * k, v.z * k)
    } else {
        v
    }
}

/// `speed` 0 ends cruise. Non-finite or negative is ignored. The stored cap
/// never exceeds [`cruise_limit`].
pub(super) fn on_cruise(shared: &Arc<Mutex<State>>, id: u32, speed: f64) {
    if !speed.is_finite() || speed < 0.0 {
        return;
    }
    let mut state = shared.lock_recover();
    let limit = cruise_limit(state.max_speed);
    let Some(h) = state.players.get_mut(&id) else { return };
    if speed == 0.0 {
        h.cruising = false;
        h.cruise_speed = 0.0;
    } else {
        h.cruising = true;
        h.cruise_speed = speed.min(limit);
    }
}

/// An explicit `/tp` discontinuity: exempt from the movement envelope, still
/// border-checked, and refused (Position, then a reason) when this player
/// may not teleport.
pub(super) fn on_teleport(shared: &Arc<Mutex<State>>, ctx: &Ctx, id: u32, pos: DVec3) {
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return;
    }
    let mut sends = Vec::new();
    let wake = {
        let mut state = shared.lock_recover();
        let Some(h) = state.players.get(&id) else { return };
        let allowed = match ctx.teleport {
            TeleportPolicy::All => true,
            TeleportPolicy::Ops => is_operator(ctx, h),
            TeleportPolicy::Off => false,
        };
        if outside_world(pos) || !allowed {
            h.correct_position(id, &mut sends);
            if !outside_world(pos) {
                let reason = if ctx.teleport == TeleportPolicy::Off {
                    "teleport is not permitted"
                } else {
                    "only an operator can teleport"
                };
                tell(h, id, reason, &mut sends);
            }
        } else {
            commit_pose(&mut state, id, pos, None, &mut sends);
            // Echo so a client with an in-flight `/tp` can tell accept from a
            // stale movement snap-back: the last Position is the committed pose.
            // The destination's cells are the held ones the next swept move starts from.
            if let Some(h) = state.players.get_mut(&id) {
                h.correct_position(id, &mut sends);
                let (stance, up) = (h.stance, h.up);
                remember_occupied(h, pos, stance, up);
            }
        }
        queue(&state, sends)
    };
    drop(wake);
}

/// Must run under the state lock; the queued sends go out after it drops.
/// Peers entering/leaving range get both sides' poses/[`PeerExited`], so
/// nobody keeps drawing a frozen ghost. Later poses go out on the pose tick
/// ([`broadcast_poses`]).
///
/// [`PeerExited`]: ServerMessage::PeerExited
pub(super) fn commit_pose(
    state: &mut State,
    id: u32,
    pos: DVec3,
    reported: Option<ReportedPose>,
    sends: &mut Vec<PendingSend>,
) {
    let max_speed = state.max_speed;
    let tick = state.tick;
    let Some(h) = state.players.get_mut(&id) else { return };
    let old = h.pos;
    let before = (h.pos, h.yaw, h.pitch, h.frame, h.velocity, h.up, h.stance);
    h.pos = pos;
    if let Some(r) = reported {
        let cap = move_cap(h, max_speed);
        h.yaw = r.yaw;
        h.pitch = r.pitch;
        h.frame = r.frame;
        h.velocity = clamp_velocity(r.velocity, cap);
        h.up = r.up;
        h.stance = r.stance;
    }
    h.last_move = Instant::now();
    if before != (h.pos, h.yaw, h.pitch, h.frame, h.velocity, h.up, h.stance) {
        h.moved = tick;
        h.body = PoseBody::new(h.yaw, h.pitch, h.frame, h.velocity, h.up, h.stance);
    }
    let body = h.body;
    let (from, to) = (bucket_of(old), bucket_of(pos));
    if from != to {
        state.grid_remove(id, old);
        state.grid_insert(id, pos);
    }
    // Set membership keeps the visibility diff linear in the nearby player count.
    let Scratch { mut near, mut gone, mut fresh } = std::mem::take(&mut state.scratch);
    state.visible_from(id, pos, &mut near);
    {
        let mover = &state.players[&id];
        gone.clear();
        gone.extend(mover.visible.iter().filter(|pid| !near.contains(*pid)));
        fresh.clear();
        fresh.extend(near.iter().filter(|pid| !mover.visible.contains(*pid)));
    }
    if !gone.is_empty() || !fresh.is_empty() {
        for &pid in &gone {
            let Some(other) = state.players.get_mut(&pid) else { continue };
            other.visible.remove(&id);
            sends.push((pid, ServerMessage::PeerExited { id }.encode().into()));
            sends.push((id, ServerMessage::PeerExited { id: pid }.encode().into()));
        }
        // An arriving peer needs the mover's pose AND the mover needs theirs, or
        // the mover keeps hiding them until they next move.
        for &pid in &fresh {
            let Some(other) = state.players.get_mut(&pid) else { continue };
            other.visible.insert(id);
            sends.push((pid, PosesWriter::single(other.pos, id, &body, pos)));
            sends.push((id, PosesWriter::single(pos, pid, &other.body, other.pos)));
        }
        if let Some(h) = state.players.get_mut(&id) {
            for pid in &gone {
                h.visible.remove(pid);
            }
            h.visible.extend(fresh.iter().copied());
        }
    }
    state.scratch = Scratch { near, gone, fresh };
}
