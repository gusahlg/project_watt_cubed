//! Edits and tool uses against the ledger: reach, revisions, hooks and the novel-spec quota.
use super::*;

/// An edit that will not be applied still gets an ack, so the client's request
/// does not stay pending and poison the next expectation on that cell.
pub(super) fn reject_edit(shared: &Arc<Mutex<State>>, id: u32, req: u32, x: i32, y: i32, z: i32) {
    let state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    ack_reject(&state, &h.out, req, (x, y, z));
}

/// The one rejection an edit gets: `accepted: false` with the cell's current revision.
pub(super) fn ack_reject(state: &State, out: &Outbox, req: u32, at: Pos) {
    let _ = out.try_send(ServerMessage::EditAck { req, accepted: false, rev: state.rev(at) }.encode().into());
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
/// client is under [`NOVEL_SPEC_QUOTA`] and the table stays under [`CLIENT_INTERN_LIMIT`].
pub(super) fn take_novel_spec(state: &mut State, id: u32, spec: &str) -> Option<BlockId> {
    if let Some(found) = state.registry.lookup_spec(spec) {
        return Some(found);
    }
    let under_quota = state.players.get(&id).is_some_and(|h| h.novel < NOVEL_SPEC_QUOTA);
    if !under_quota || state.registry.block_count() >= CLIENT_INTERN_LIMIT {
        return None;
    }
    let before = state.registry.block_count();
    let block = state.registry.parse_spec(spec)?;
    if state.registry.block_count() > before && let Some(h) = state.players.get_mut(&id) {
        h.novel = h.novel.saturating_add(1);
    }
    Some(block)
}

/// What an edit asks for once its spec is known to be well formed.
enum Wanted {
    /// A block the registry already holds: its text is the registry's, with no copy.
    Known(BlockId),
    /// A configuration not interned yet, in canonical spelling. It enters the registry
    /// only when the edit wins.
    Novel(String),
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
    let at = (x, y, z);
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    let ack_to = h.ready.then(|| h.out.clone());
    // Cloned before the registry mut-borrow; skipped when no hooks are installed.
    let name = hooks.is_some().then(|| h.name.clone());
    let reject = |state: &State, out: Option<&Outbox>| {
        if let Some(out) = out {
            ack_reject(state, out, req, at);
        }
    };
    // Y is unbounded (infinite world height/depth); reach is the real gate.
    // `as f64` so i32::MIN never hits signed-abs overflow; cells past the
    // playable border are still reach-checked (a player AT the border can
    // mine the slack column) but a forged i32::MAX coord is out of reach.
    if spec.len() > MAX_SPEC || h.pos.distance(cell_centre(generator, at)) > EDIT_REACH {
        return reject(&state, ack_to.as_ref());
    }
    // Canonical form without touching the registry. Junk and `c:00` fail here,
    // before a revision miss could still have interned them. A known block is a
    // lookup that copies no text.
    let wanted = match state.registry.lookup_spec(spec) {
        Some(block) => Wanted::Known(block),
        None => match state.registry.canonical_spec(spec) {
            Some(canonical) => Wanted::Novel(canonical),
            None => return reject(&state, ack_to.as_ref()),
        },
    };
    if expect != state.rev(at) {
        return reject(&state, ack_to.as_ref());
    }
    // The generated block says whether the edit is natural. A generator read can take
    // a while, so a cache miss reads it with the lock released, as the hooks do below.
    let generated = match state.terrain.peek(at) {
        Some(block) => block,
        None => {
            drop(state);
            let block = generator.voxel_at(x, y, z);
            state = shared.lock_recover();
            if !state.players.contains_key(&id) {
                return;
            }
            state.terrain.store(at, block);
            block
        }
    };
    if let (Some(hooks), Some(name)) = (hooks, name) {
        let spec = match &wanted {
            Wanted::Known(block) => Arc::clone(state.registry.spec_ref(*block)),
            Wanted::Novel(canonical) => Arc::from(canonical.as_str()),
        };
        let intent = EditIntent { player: id, name, x, y, z, spec, expect };
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
    if expect != state.rev(at) {
        return reject(&state, ack_to.as_ref());
    }
    let block = match &wanted {
        Wanted::Known(block) => *block,
        Wanted::Novel(canonical) => match take_novel_spec(&mut state, id, canonical) {
            // Only `air` spells the void; a configuration that decodes to it is refused.
            Some(block) if block != AIR => block,
            _ => return reject(&state, ack_to.as_ref()),
        },
    };
    // Pool at its cap: refuse new content.
    let Some((rev, spec)) = state.write(at, block, block == generated) else {
        return reject(&state, ack_to.as_ref());
    };
    // Placed or removed: the cell's contacts wake.
    state.reactions.wake_cell(at);
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
    // The broadcast carries the SAME Arc the ledger stores.
    let msg = ServerMessage::Edit { x, y, z, rev, spec };
    let wake = broadcast(&mut state, &msg, |pid, _| pid != id);
    drop(state);
    drop(wake);
}

/// Answer a tool use with "nothing happened": the cell as it is now, at its current revision,
/// and the tool unchanged. A generated cell the terrain cache does not hold is read with the
/// lock released, so a refusal never runs the generator under it. The answer is queued under
/// the lock, in order with everything else the player is sent.
pub(super) fn refuse_tool<'a>(
    mut state: Guard<'a, State>,
    shared: &'a Mutex<State>,
    generator: &crate::world::terrain::Generator,
    id: u32,
    req: u32,
    at: Pos,
    tool: &Arc<str>,
) {
    if !state.edits.contains_key(&at) && state.terrain.peek(at).is_none() {
        drop(state);
        let block = generator.voxel_at(at.0, at.1, at.2);
        state = shared.lock_recover();
        state.terrain.store(at, block);
    }
    let Some(h) = state.players.get(&id) else { return };
    let cell_spec = match state.edits.get(&at) {
        Some(cell) => Arc::clone(&cell.spec),
        None => Arc::clone(state.registry.spec_ref(server_block(&state, generator, at))),
    };
    let msg = ServerMessage::ToolResult { req, reacted: false, rev: state.rev(at), cell_spec, tool_spec: Arc::clone(tool) };
    let _ = h.out.try_send(msg.encode().into());
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
    tool_spec: &Arc<str>,
) {
    let at = (x, y, z);
    let mut state = shared.lock_recover();
    let Some(h) = state.players.get(&id) else { return };
    if !h.ready {
        return;
    }
    let current = state.rev(at);
    if tool_spec.len() > MAX_SPEC || h.pos.distance(cell_centre(generator, at)) > EDIT_REACH || expect != current {
        return refuse_tool(state, shared, generator, id, req, at, tool_spec);
    }
    // Revision already matched, so a novel tool spends quota only for a live request.
    let Some(tool) = take_novel_spec(&mut state, id, tool_spec) else {
        return refuse_tool(state, shared, generator, id, req, at, tool_spec);
    };
    let cell = server_block(&state, generator, at);
    // A void on either side has nothing to react with, and the products must not eat the
    // world's reserve.
    if tool == AIR || cell == AIR || state.registry.block_count() + 2 > CLIENT_INTERN_LIMIT {
        return refuse_tool(state, shared, generator, id, req, at, tool_spec);
    }
    let Some((_, new_cell, new_tool)) = state.registry.react(cell, tool) else {
        return refuse_tool(state, shared, generator, id, req, at, tool_spec);
    };
    let Some((rev, spec)) = state.write(at, new_cell, false) else {
        return refuse_tool(state, shared, generator, id, req, at, tool_spec);
    };
    state.reactions.wake_cell(at);
    let tool_out = Arc::clone(state.registry.spec_ref(new_tool));
    if let Some(h) = state.players.get(&id) {
        let msg = ServerMessage::ToolResult { req, reacted: true, rev, cell_spec: Arc::clone(&spec), tool_spec: tool_out };
        let _ = h.out.try_send(msg.encode().into());
    }
    let wake = broadcast(&mut state, &ServerMessage::Edit { x, y, z, rev, spec }, |pid, _| pid != id);
    drop(state);
    drop(wake);
}
