//! The 20 Hz server thread: reaction ticks, pose ticks and the shared clock.
use super::*;

pub(super) fn reactions_loop(shared: Arc<Mutex<State>>, ctx: Arc<Ctx>, shutdown: Arc<AtomicBool>) {
    let period = Duration::from_millis(50);
    let mut next_clock = Instant::now() + TIME_BROADCAST;
    while !shutdown.load(Ordering::Relaxed) {
        let start = Instant::now();
        // The tick restores the scheduler itself. This catch covers a panic after that
        // (the broadcast) so the thread, and the scheduler, both stay.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run_reactions(&shared, &ctx))) {
            eprintln!("reaction tick panicked: {}", panic_text(&payload));
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| broadcast_poses(&shared))) {
            eprintln!("pose tick panicked: {}", panic_text(&payload));
        }
        if Instant::now() >= next_clock {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| broadcast_clock(&shared, &ctx))) {
                eprintln!("time broadcast panicked: {}", panic_text(&payload));
            }
            next_clock = Instant::now() + TIME_BROADCAST;
        }
        if let Some(rest) = period.checked_sub(start.elapsed()) {
            thread::sleep(rest);
        }
    }
}

pub(super) fn panic_text(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// One sim tick of the scheduler. Committed mutations are [`ServerMessage::Snapshot`]
/// batches attributed to [`WORLD_PLAYER`], broadcast to every ready client. The
/// scheduler budget already bounds the count; every commit is sent. A panic inside
/// the tick puts the scheduler back before the error propagates.
pub(super) fn run_reactions(shared: &Arc<Mutex<State>>, ctx: &Ctx) {
    let mut state = shared.lock_recover();
    let force = take_panic_tick(&mut state);
    if state.reactions.pending() == 0 && !force {
        return;
    }
    let budget = reactions::Budget::DEFAULT;
    let mut sched = std::mem::take(&mut state.reactions);
    let mut mutations = std::mem::take(&mut state.mutations);
    mutations.clear();
    let tick = catch_unwind(AssertUnwindSafe(|| {
        if force {
            panic!("reaction tick");
        }
        let mut cells = ServerCells {
            state: &mut state,
            generator: &ctx.generator,
        };
        sched.tick_into(&mut cells, budget, &mut mutations);
    }));
    state.reactions = sched;
    let wake = match tick {
        Ok(()) => send_reaction_mutations(&mut state, &mutations),
        Err(payload) => {
            eprintln!("reaction tick panicked: {}", panic_text(&payload));
            Wake(Vec::new())
        }
    };
    state.mutations = mutations;
    drop(state);
    drop(wake);
}

pub(super) fn take_panic_tick(state: &mut State) -> bool {
    #[cfg(test)]
    {
        let force = state.panic_tick;
        state.panic_tick = false;
        force
    }
    #[cfg(not(test))]
    {
        let _ = state;
        false
    }
}

/// One [`ServerMessage::PeerPoses`] frame per ready player with the visible peers
/// that moved: near peers every tick, far ones every [`FAR_EVERY`]th tick (staggered
/// by id) when they moved within that window.
pub(super) fn broadcast_poses(shared: &Arc<Mutex<State>>) {
    let mut sends = Vec::new();
    let wake = {
        let mut guard = shared.lock_recover();
        let state = &mut *guard;
        let tick = state.tick;
        state.tick += 1;
        let writer = &mut state.poses;
        for (&rid, r) in &state.players {
            if !r.ready || r.visible.is_empty() {
                continue;
            }
            writer.begin(r.pos);
            for &pid in &r.visible {
                let Some(p) = state.players.get(&pid) else { continue };
                let due = if p.pos.distance_squared(r.pos) <= NEAR_SQ {
                    p.moved == tick
                } else {
                    p.moved + FAR_EVERY > tick && (tick + u64::from(pid)) % FAR_EVERY == 0
                };
                if due {
                    writer.push(pid, &p.body, p.pos);
                }
            }
            if !writer.is_empty() {
                sends.push((rid, writer.frame()));
            }
        }
        queue(state, &mut sends)
    };
    drop(wake);
}

/// The shared clock, sampled now and sent to every connected player.
pub(super) fn broadcast_clock(shared: &Arc<Mutex<State>>, ctx: &Ctx) {
    let mut state = shared.lock_recover();
    let day = state.day_now(ctx.day_secs);
    let wake = broadcast(&mut state, &ServerMessage::Time { day, day_secs: ctx.day_secs }, |_, _| true);
    drop(state);
    drop(wake);
}

/// Authoritative overlay edits from one scheduler tick, as snapshot batches
/// (the client applies [`ServerMessage::Snapshot`] after bootstrap), one entry
/// per distinct cell. One `S_Edit` per mutation would overflow [`OUT_CAPACITY`]
/// on two full ticks.
pub(super) fn send_reaction_mutations(state: &mut State, mutations: &[Mutation]) -> Wake {
    let mut wake = Wake(Vec::new());
    if mutations.is_empty() {
        return wake;
    }
    // A cell committed by several contacts in one turn is sent once, with its
    // final content, at the point of its last commit (order is preserved).
    let mut last = std::mem::take(&mut state.latest);
    last.clear();
    for (i, m) in mutations.iter().enumerate() {
        last.insert(m.pos, i);
    }
    let mut frames = Vec::new();
    {
        let mut writer = SnapshotWriter::new();
        let mut emit = |frame| frames.push(frame);
        for (i, m) in mutations.iter().enumerate() {
            if last.get(&m.pos) != Some(&i) {
                continue;
            }
            let Some(cell) = state.edits.get(&m.pos) else { continue };
            writer.push(m.pos, cell.rev, cell.block.0, &cell.spec, &mut emit);
        }
        writer.finish(&mut emit);
    }
    state.latest = last;
    for frame in frames {
        wake.join(broadcast_frame(state, frame, None, |pid, _| pid != WORLD_PLAYER));
    }
    wake
}
