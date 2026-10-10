//! Edits and tool uses against the ledger: reach, revisions, hooks and the novel-spec quota.
use super::*;

/// An edit that will not be applied still gets an ack, so the client's request
/// does not stay pending and poison the next expectation on that cell.
pub(super) fn reject_edit(shared: &Arc<Mutex<State>>, id: u32, req: u32, x: i32, y: i32, z: i32) {
    let state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let rev = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    let _ = h.out.try_send(ServerMessage::EditAck { req, accepted: false, rev }.encode().into());
}

/// Novel specs intern only while `block_count()` is below `limit`. A known spec resolves
/// even at the line. `None` is malformed, or novel at/above the line.
#[cfg(test)]
pub(super) fn resolve_client_spec_within(
    registry: &mut BlockRegistry,
    spec: &str,
    limit: usize,
) -> Option<BlockId> {
    if let Some(id) = registry.lookup_spec(spec) {
        return Some(id);
    }
    if registry.block_count() >= limit {
        return None;
    }
    registry.parse_spec(spec)
}

/// A known spec resolves with no quota spend. A novel one interns only while this
/// client is under [`NOVEL_SPEC_QUOTA`] and the table keeps [`CLIENT_INTERN_RESERVE`] free.
pub(super) fn take_novel_spec(state: &mut State, id: u32, spec: &str) -> Option<BlockId> {
    if let Some(found) = state.registry.lookup_spec(spec) {
        return Some(found);
    }
    let under_quota = state.players.get(&id).is_some_and(|h| h.novel < NOVEL_SPEC_QUOTA);
    let limit = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
    if !under_quota || state.registry.block_count() >= limit {
        return None;
    }
    let before = state.registry.block_count();
    let block = state.registry.parse_spec(spec)?;
    if state.registry.block_count() > before && let Some(h) = state.players.get_mut(&id) {
        h.novel = h.novel.saturating_add(1);
    }
    Some(block)
}


/// Gates, in order: reach (against the sender's last ACCEPTED position, per
/// [`on_move`]'s envelope), a canonical spec (no intern), the expected cell
/// revision, installed [`ServerMod::validate_edit`] hooks, the revision again,
/// then — only if it is still the winner — interning a novel spec under the
/// per-client quota and the world's reserve. A stale novel spec never enters
/// the registry. When two players race one cell, the loser rolls back. An edit that
/// puts back the generated block is a [`Cell::natural`] entry: it keeps the revision
/// but stays out of the world file, so no-op edits cannot grow it.
///
/// A hook [`Verdict::Deny`] uses this same reject path (no ledger write, one
/// `EditAck { accepted: false }`, no broadcast), so the client's
/// `EditRejected.restore` is true iff no newer confirmed revision has landed
/// on the cell — identical to a lost race.
///
/// Hook bodies run **outside** the [`State`] lock: facts are collected under
/// it, the lock is dropped, then the table is called. The generated block, when
/// the terrain cache does not hold it, is read the same way; the revision is
/// checked again after either.
#[allow(clippy::too_many_arguments)] // edit validation takes each protocol field separately
pub(super) fn on_edit(
    shared: &Arc<Mutex<State>>,
    hooks: Option<&Mutex<hooks::Table>>,
    generator: &crate::world::terrain::Generator,
    id: u32,
    req: u32,
    x: i32,
    y: i32,
    z: i32,
    expect: u32,
    spec: &str,
) {
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let ack_to = h.ready.then(|| h.out.clone());
    // Cloned before the registry mut-borrow; skipped when no hooks are installed.
    let name = hooks.is_some().then(|| h.name.clone());
    let reject = |state: &State, out: Option<&Outbox>| {
        let rev = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
        if let Some(out) = out {
            let _ = out.try_send(
                ServerMessage::EditAck { req, accepted: false, rev }.encode().into(),
            );
        }
    };
    // Y is unbounded (infinite world height/depth); reach is the real gate.
    // `as f64` so i32::MIN never hits signed-abs overflow; cells past the
    // playable border are still reach-checked (a player AT the border can
    // mine the slack column) but a forged i32::MAX coord is out of reach.
    // A round world's storage cell is judged where its chart embeds it.
    let target = crate::space::atlas::embed_cell(generator.atlases(), (x, y, z))
        .unwrap_or(DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5));
    if spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH {
        return reject(&state, ack_to.as_ref());
    }
    // Canonical form without touching the registry. Junk and `c:00` fail here,
    // before a revision miss could still have interned them.
    let Some(canonical) = state.registry.canonical_spec(spec) else {
        return reject(&state, ack_to.as_ref());
    };
    let current = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    if expect != current {
        return reject(&state, ack_to.as_ref());
    }
    // The generated block says whether the edit is natural. A generator read can take
    // a while, so a cache miss reads it with the lock released, as the hooks do below.
    let generated = match state.terrain.peek((x, y, z)) {
        Some(block) => block,
        None => {
            drop(state);
            let block = generator.voxel_at(x, y, z);
            state = shared.lock_recover();
            if !state.players.contains_key(&id) {
                return;
            }
            state.terrain.store((x, y, z), block);
            block
        }
    };
    if let (Some(hooks), Some(name)) = (hooks, name) {
        let intent = EditIntent {
            player: id,
            name,
            x,
            y,
            z,
            spec: Arc::from(canonical.as_str()),
            expect,
        };
        drop(state);
        let verdict = hooks.lock_recover().validate_edit(&intent);
        state = shared.lock_recover();
        if let Verdict::Deny { .. } = verdict {
            return reject(&state, ack_to.as_ref());
        }
        if !state.players.contains_key(&id) {
            return;
        }
    }
    let current = state.edits.get(&(x, y, z)).map_or(0, |c| c.rev);
    if expect != current {
        return reject(&state, ack_to.as_ref());
    }
    let Some(block) = take_novel_spec(&mut state, id, &canonical) else {
        return reject(&state, ack_to.as_ref());
    };
    if block == crate::block::AIR && canonical != "air" {
        return reject(&state, ack_to.as_ref());
    }
    let rev = current + 1;
    let Some(spec) = state.intern(&canonical) else {
        return reject(&state, ack_to.as_ref()); // pool at cap: refuse new content
    };
    let natural = block == generated;
    if let Some(old) = state.edits.insert((x, y, z), Cell { block, spec: spec.clone(), rev, natural }) {
        state.release(old.spec);
    }
    // Placed or removed: the cell's contacts wake.
    state.reactions.wake_cell((x, y, z));
    if let Some(out) = ack_to {
        // Reconcile even when a peer or tool result overwrote the prediction, or the request
        // expired locally. Send content before the verdict: if the verdict cannot be queued,
        // the confirmed revision still prevents timeout from undoing authoritative content.
        if out.try_send(ServerMessage::Snapshot { edits: vec![(x, y, z, rev, spec.clone())] }.encode().into()).is_ok() {
            let _ = out.try_send(ServerMessage::EditAck { req, accepted: true, rev }.encode().into());
        } else {
            // An essential confirmation cannot be silently lost: a fresh join will replay it.
            kick_slow(&state, &[id]);
        }
    }
    // The broadcast carries the SAME pooled Arc the ledger stores.
    let msg = ServerMessage::Edit { x, y, z, rev, spec };
    let wake = broadcast(&mut state, &msg, |pid, _| pid != id);
    drop(state);
    drop(wake);
}

/// Answer a tool use with "nothing happened": the cell's current content and the tool unchanged.
pub(super) fn refuse_tool(shared: &Arc<Mutex<State>>, generator: &crate::world::terrain::Generator, id: u32, req: u32, x: i32, y: i32, z: i32, tool_spec: &str) {
    let state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let out = h.out.clone();
    let found = state.edits.get(&(x, y, z)).map(|c| (c.rev, c.spec.clone()));
    let (rev, cell_spec) = match found {
        Some(found) => found,
        None => {
            drop(state);
            let block = generator.voxel_at(x, y, z);
            let state = shared.lock_recover();
            state.edits.get(&(x, y, z)).map_or_else(
                || (0, crate::save::block_spec(&state.registry, block).into()),
                |c| (c.rev, c.spec.clone()),
            )
        }
    };
    let msg = ServerMessage::ToolResult { req, reacted: false, rev, cell_spec, tool_spec: tool_spec.into() };
    let _ = out.try_send(msg.encode().into());
}

/// A player uses a held configuration as a tool on a cell. Gates: ready, reach, the tool spec
/// resolves (known, or novel below the reserve line), the cell revision is the one the client
/// expected. Then ONE operation of the law runs between the cell (A) and the tool (B); on a
/// change the cell is written, its contacts wake, the sender gets both results and everyone else
/// the cell edit. There is no holdings ledger, so the server trusts the client about what it holds
/// (as for placement).
#[allow(clippy::too_many_arguments)]
pub(super) fn on_tool_use(
    shared: &Arc<Mutex<State>>,
    generator: &crate::world::terrain::Generator,
    id: u32,
    req: u32,
    x: i32,
    y: i32,
    z: i32,
    expect: u32,
    tool_spec: &str,
) {
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    if !h.ready {
        return;
    }
    let out = h.out.clone();
    let target = crate::space::atlas::embed_cell(generator.atlases(), (x, y, z))
        .unwrap_or(DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5));
    let pos = (x, y, z);
    let current = state.edits.get(&pos).map_or(0, |c| c.rev);
    let reply = |state: &State, reacted: bool, rev: u32, tool: Arc<str>| {
        let cell_spec: Arc<str> = state.edits.get(&pos).map_or_else(
            || crate::save::block_spec(&state.registry, generator.voxel_at(x, y, z)).into(),
            |c| c.spec.clone(),
        );
        let msg = ServerMessage::ToolResult { req, reacted, rev, cell_spec, tool_spec: tool };
        let _ = out.try_send(msg.encode().into());
    };
    let unchanged: Arc<str> = tool_spec.into();
    if tool_spec.len() > MAX_SPEC || h.pos.distance(target) > EDIT_REACH || expect != current {
        return reply(&state, false, current, unchanged);
    }
    // Revision already matched, so a novel tool spends quota only for a live request.
    let Some(tool) = take_novel_spec(&mut state, id, tool_spec) else {
        return reply(&state, false, current, unchanged);
    };
    let cell = server_block(&state, generator, pos);
    if tool == AIR || cell == AIR {
        return reply(&state, false, current, unchanged);
    }
    let limit = crate::block::registry::MAX_BLOCK_TYPES - CLIENT_INTERN_RESERVE;
    if state.registry.block_count() + 2 > limit {
        return reply(&state, false, current, unchanged); // products would eat the world's reserve
    }
    let Some((_, new_cell, new_tool)) = state.registry.react(cell, tool) else {
        return reply(&state, false, current, unchanged);
    };
    let canonical = crate::save::block_spec(&state.registry, new_cell);
    let Some(spec) = state.intern(&canonical) else {
        return reply(&state, false, current, unchanged);
    };
    let rev = current + 1;
    if let Some(old) = state.edits.insert(pos, Cell { block: new_cell, spec: spec.clone(), rev, natural: false }) {
        state.release(old.spec);
    }
    state.reactions.wake_cell(pos);
    let tool_out: Arc<str> = state.registry.spec(new_tool).into();
    reply(&state, true, rev, tool_out);
    let wake = broadcast(&mut state, &ServerMessage::Edit { x, y, z, rev, spec }, |pid, _| pid != id);
    drop(state);
    drop(wake);
}
