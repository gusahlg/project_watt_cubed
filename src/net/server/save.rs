//! The world file: installing a loaded ledger, saving it, autosave, and the policy files beside it.
use super::*;

/// Skip a spec this build cannot parse or the pool cannot hold, and return those cells
/// so the next save writes them back. Revisions start at 1; the file has none.
pub(super) fn install_edits(state: &mut State, edits: &[(i32, i32, i32, String)]) -> Vec<(i32, i32, i32, String)> {
    let mut kept = Vec::new();
    let mut over = 0usize;
    for (x, y, z, spec) in edits {
        let Some(id) = state.registry.parse_spec(spec) else {
            eprintln!("skipping unknown block spec at {x},{y},{z}");
            kept.push((*x, *y, *z, spec.clone()));
            continue;
        };
        let Some(shared) = state.intern(id) else {
            over += 1;
            kept.push((*x, *y, *z, spec.clone()));
            continue;
        };
        state.edits.insert((*x, *y, *z), Cell { block: id, spec: shared, rev: 1, natural: false });
    }
    if over > 0 {
        eprintln!("warning: {over} edits name more than {MAX_SPEC_POOL} distinct blocks; they stay in the file but are not served");
    }
    kept
}

pub(super) fn contacts_of(pending: &[crate::save::format::PendingContact]) -> Vec<(u32, Contact)> {
    pending
        .iter()
        .map(|contact| (contact.age, Contact { lo: (contact.x, contact.y, contact.z), axis: contact.axis }))
        .collect()
}

pub(super) fn save_world(state: &Mutex<State>, ctx: &Ctx, gate: &Mutex<()>) {
    let Some(store) = &ctx.store else { return };
    let _gate = gate.lock_recover();
    let snap = {
        let state = state.lock_recover();
        let edits = state
            .edits
            .iter()
            .filter(|(_, cell)| !cell.natural)
            .map(|(&(x, y, z), cell)| (x, y, z, Arc::clone(&cell.spec)))
            .collect();
        let pending = state
            .reactions
            .snapshot()
            .into_iter()
            .map(|(age, contact)| crate::save::format::PendingContact {
                x: contact.lo.0,
                y: contact.lo.1,
                z: contact.lo.2,
                axis: contact.axis,
                age,
            })
            .collect();
        persist::Snapshot {
            seed: ctx.seed,
            worldgen: ctx.worldgen,
            terrain: ctx.terrain,
            day: state.day_now(ctx.day_secs),
            edits,
            pending,
        }
    };
    if let Err(e) = store.write(&snap) {
        eprintln!("could not save {}: {e}", store.path().display());
    }
}

pub(super) fn autosave_loop(
    state: Arc<Mutex<State>>,
    ctx: Arc<Ctx>,
    gate: Arc<Mutex<()>>,
    shutdown: Arc<AtomicBool>,
    every: Duration,
) {
    let mut next = Instant::now() + every;
    while !shutdown.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(200));
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        if Instant::now() < next {
            continue;
        }
        save_world(&state, &ctx, &gate);
        next = Instant::now() + every;
    }
}

/// Union `ops.txt` and `mods.toml` beside [`Config::world`] into the flag lists.
/// An `ops.txt` line with a secret goes to [`Config::op_secrets`]. A missing file
/// adds nothing. A `mods.toml` that is not `allow` / `deny` string arrays is an
/// error, so a dedicated server refuses to start open.
pub fn load_world_policy(config: &mut Config) -> Result<(), String> {
    let Some(world) = config.world.clone() else {
        return Ok(());
    };
    let side = persist::read_side_files(&world)?;
    for name in side.ops {
        if !config.ops.iter().any(|op| op.eq_ignore_ascii_case(&name)) {
            config.ops.push(name);
        }
    }
    for (name, secret) in side.op_secrets {
        if !config.op_secrets.iter().any(|(have, _)| have.eq_ignore_ascii_case(&name)) {
            config.op_secrets.push((name, secret));
        }
    }
    push_unique(&mut config.mods_allow, side.allow);
    push_unique(&mut config.mods_deny, side.deny);
    Ok(())
}

pub(super) fn push_unique(into: &mut Vec<String>, extra: Vec<String>) {
    for id in extra {
        if !into.iter().any(|have| have == &id) {
            into.push(id);
        }
    }
}
