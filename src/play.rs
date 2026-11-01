//! The world mirror, optimistic edits and one body step shared by the windowed game and a
//! headless client. Cues are the only part that differs: the game plays them, a headless client
//! records them.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use voxel_engine::DVec3;

use crate::block::{AIR, BlockId};
use crate::input::movement::{self, MoveInput};
use crate::math::{Aabb, Bounded};
use crate::modules::ToolUse;
use crate::net::client::{Connection, Incoming};
use crate::player::Player;
use crate::save;
use crate::world::World;

/// One optimistic edit awaiting the server's verdict.
pub(crate) struct PendingEdit {
    pub(crate) cell: (i32, i32, i32),
    /// What the cell held before the optimistic apply.
    pub(crate) prev: BlockId,
    pub(crate) kind: PendingKind,
}

/// What to give back if the server rejects the edit.
pub(crate) enum PendingKind {
    /// Breaking yielded this configuration. A rejection revokes it only if it fit (`gained`).
    Break { id: BlockId, gained: bool },
    /// Placing spent one block of this id. A rejection refunds it.
    Place(BlockId),
}

/// A fact the host turns into a sound, a module call, or a headless event.
pub(crate) enum Cue {
    RemoteEdit {
        cell: (i32, i32, i32),
        at: DVec3,
        id: BlockId,
        prev: BlockId,
    },
    Mutation {
        cell: (i32, i32, i32),
        id: BlockId,
    },
    Confirmed {
        cell: (i32, i32, i32),
    },
    Rejected {
        cell: (i32, i32, i32),
    },
    BreakRejected {
        id: BlockId,
    },
    PlaceRejected {
        id: BlockId,
    },
    LocalBreak {
        cell: (i32, i32, i32),
        at: DVec3,
        block: BlockId,
    },
    BreakNoted {
        id: BlockId,
        overflow: bool,
    },
    Trauma(f32),
    AnimSwing,
    LocalPlace {
        cell: (i32, i32, i32),
        at: DVec3,
        block: BlockId,
    },
    SwingAt {
        at: DVec3,
    },
    PeerSwing {
        at: DVec3,
        peer: u32,
    },
    ToolReacted {
        at: DVec3,
        block: BlockId,
    },
    ToolNoReaction,
    ToolChanged {
        tool: BlockId,
        new_tool: BlockId,
    },
    ToolUsed(ToolUse),
    ToolDone {
        cell: (i32, i32, i32),
        reacted: bool,
        cell_block: BlockId,
        tool_block: BlockId,
    },
    Chat {
        from: Arc<str>,
        channel: u8,
        text: Arc<str>,
    },
    Joined {
        name: Arc<str>,
    },
    Left {
        name: Arc<str>,
    },
    Time {
        day: f32,
        day_secs: f32,
    },
    Interrupted,
    Snapped {
        to: DVec3,
    },
}

/// Where a cue or a snap's chunk load goes. The game and a headless client each implement it.
pub(crate) trait Host {
    fn cue(&mut self, world: &World, cue: Cue);
    fn prepare(&mut self, world: &mut World, pos: DVec3);
}

/// Apply one poll's server events in order. Runs of snapshot cells install in bulk.
/// `Some(reason)` when the server dropped us.
pub(crate) fn apply_incoming(
    world: &mut World,
    player: &mut Player,
    pending_edits: &mut HashMap<u32, PendingEdit>,
    pending_tools: &mut HashMap<u32, BlockId>,
    net: Option<&Connection>,
    incoming: Vec<Incoming>,
    host: &mut impl Host,
) -> Option<String> {
    let mut disconnected = None;
    let mut ledger = crate::world::Ledger::default();
    for event in incoming {
        // Every other event lands after the snapshot cells before it.
        if !matches!(event, Incoming::Mutation { .. }) {
            world.install(&mut ledger);
        }
        match event {
            Incoming::Edit { x, y, z, spec } => {
                let id = save::parse_block(world.registry_mut(), &spec);
                let (prev, at) = write_cell(world, (x, y, z), id);
                host.cue(
                    world,
                    Cue::RemoteEdit {
                        cell: (x, y, z),
                        at,
                        id,
                        prev,
                    },
                );
            }
            Incoming::Mutation { x, y, z, spec } => {
                let id = save::parse_block(world.registry_mut(), &spec);
                ledger.push((x, y, z), id);
                host.cue(
                    world,
                    Cue::Mutation {
                        cell: (x, y, z),
                        id,
                    },
                );
            }
            Incoming::EditAccepted { req } => {
                if let Some(pending) = pending_edits.remove(&req) {
                    host.cue(world, Cue::Confirmed { cell: pending.cell });
                }
            }
            Incoming::EditRejected { req, restore } => {
                if let Some(pending) = pending_edits.remove(&req) {
                    rollback(world, player, pending, restore, host);
                }
            }
            Incoming::Position { pos, frame, up } => {
                // Authoritative snap-back. The host loads the collision slab; the pose itself
                // matches the windowed game.
                host.prepare(world, pos);
                player.position = pos;
                player.orientation.frame = frame;
                player.up_axis = up;
                player.cancel_fall();
                host.cue(world, Cue::Snapped { to: pos });
            }
            Incoming::Chat {
                from_name,
                channel,
                text,
            } => {
                host.cue(
                    world,
                    Cue::Chat {
                        from: from_name,
                        channel,
                        text,
                    },
                );
            }
            Incoming::Joined { name } => host.cue(world, Cue::Joined { name }),
            Incoming::Left { name } => host.cue(world, Cue::Left { name }),
            Incoming::Time { day, day_secs } => host.cue(world, Cue::Time { day, day_secs }),
            Incoming::Disconnected { reason } => disconnected = Some(reason),
            Incoming::Interrupted => host.cue(world, Cue::Interrupted),
            Incoming::ToolResult {
                req,
                reacted,
                cell,
                cell_spec,
                tool_spec,
            } => {
                let Some(tool) = pending_tools.remove(&req) else {
                    continue;
                };
                if !reacted {
                    // No intern: a miss does not add configurations the game would not have added.
                    let cell_block = world.registry().lookup_spec(&cell_spec).unwrap_or(AIR);
                    let tool_block = world.registry().lookup_spec(&tool_spec).unwrap_or(AIR);
                    host.cue(world, Cue::ToolNoReaction);
                    host.cue(
                        world,
                        Cue::ToolDone {
                            cell,
                            reacted: false,
                            cell_block,
                            tool_block,
                        },
                    );
                    continue;
                }
                let new_cell = save::parse_block(world.registry_mut(), &cell_spec);
                let new_tool = save::parse_block(world.registry_mut(), &tool_spec);
                let (target, at) = write_cell(world, cell, new_cell);
                host.cue(world, Cue::ToolReacted { at, block: target });
                finish_tool_change(world, player, tool, new_tool, target, new_cell, host);
                host.cue(
                    world,
                    Cue::ToolDone {
                        cell,
                        reacted: true,
                        cell_block: new_cell,
                        tool_block: new_tool,
                    },
                );
            }
            Incoming::PeerSwing { id } => {
                if let Some(peer) = net.and_then(|net| net.peer(id)) {
                    let at = peer.sample(Instant::now()).pos.0;
                    host.cue(world, Cue::PeerSwing { at, peer: id });
                }
            }
        }
    }
    world.install(&mut ledger);
    disconnected
}

/// Break `cell` into the inventory and tell the server. The apply is a prediction.
pub(crate) fn commit_break(
    world: &mut World,
    player: &mut Player,
    pending: &mut HashMap<u32, PendingEdit>,
    net: Option<&mut Connection>,
    cell: (i32, i32, i32),
    host: &mut impl Host,
) {
    let (id, at) = write_cell(world, cell, AIR);
    host.cue(
        world,
        Cue::LocalBreak {
            cell,
            at,
            block: id,
        },
    );
    let gained = player.inventory.add(id, 1);
    host.cue(
        world,
        Cue::BreakNoted {
            id,
            overflow: !gained,
        },
    );
    host.cue(world, Cue::Trauma(0.15));
    host.cue(world, Cue::AnimSwing);
    predict(
        world,
        player,
        pending,
        net,
        PendingEdit {
            cell,
            prev: id,
            kind: PendingKind::Break { id, gained },
        },
        host,
    );
}

/// Place `id` into `cell`. The caller already spent it and already checked [`blocks_placement`].
pub(crate) fn commit_place(
    world: &mut World,
    player: &mut Player,
    pending: &mut HashMap<u32, PendingEdit>,
    net: Option<&mut Connection>,
    cell: (i32, i32, i32),
    id: BlockId,
    host: &mut impl Host,
) {
    let (prev, at) = write_cell(world, cell, id);
    host.cue(
        world,
        Cue::LocalPlace {
            cell,
            at,
            block: id,
        },
    );
    host.cue(world, Cue::AnimSwing);
    predict(
        world,
        player,
        pending,
        net,
        PendingEdit {
            cell,
            prev,
            kind: PendingKind::Place(id),
        },
        host,
    );
}

/// True when `cell` is solid or overlaps the player's body. Same check as the game's placement guard.
pub(crate) fn blocks_placement(world: &World, player: &Player, (x, y, z): (i32, i32, i32)) -> bool {
    let cell = Aabb::new(
        DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
        DVec3::splat(0.5),
    );
    world.is_solid(x, y, z) || cell.intersects(&player.aabb())
}

/// Use `tool` on `cell`. On a server the law runs there; alone, it runs here.
pub(crate) fn use_tool(
    world: &mut World,
    player: &mut Player,
    pending_tools: &mut HashMap<u32, BlockId>,
    net: Option<&mut Connection>,
    tool: BlockId,
    cell: (i32, i32, i32),
    host: &mut impl Host,
) {
    let (x, y, z) = cell;
    let target = world.block_at(x, y, z);
    host.cue(world, Cue::AnimSwing);
    host.cue(
        world,
        Cue::SwingAt {
            at: sound_at(world, x, y, z),
        },
    );
    if let Some(net) = net {
        let spec = save::block_spec(world.registry(), tool);
        if let Some(req) = net.send_tool_use(x, y, z, spec.into()) {
            pending_tools.insert(req, tool);
        }
        net.send_swing();
        return;
    }
    match world.react_tool(target, tool) {
        Some((new_cell, new_tool)) => {
            let (prev, at) = write_cell(world, cell, new_cell);
            host.cue(world, Cue::ToolReacted { at, block: prev });
            finish_tool_change(world, player, tool, new_tool, target, new_cell, host);
            host.cue(world, Cue::Trauma(0.08));
        }
        None => host.cue(world, Cue::ToolNoReaction),
    }
}

/// Tell the server about a local edit. An edit that cannot be sent rolls back at once.
fn predict(
    world: &mut World,
    player: &mut Player,
    pending: &mut HashMap<u32, PendingEdit>,
    net: Option<&mut Connection>,
    edit: PendingEdit,
    host: &mut impl Host,
) {
    let Some(net) = net else { return };
    let spec: Arc<str> = match edit.kind {
        PendingKind::Break { .. } => "air".into(),
        PendingKind::Place(id) => save::block_spec(world.registry(), id).into(),
    };
    let (x, y, z) = edit.cell;
    match net.send_edit(x, y, z, spec) {
        Some(req) => {
            pending.insert(req, edit);
            net.send_swing();
        }
        None => rollback(world, player, edit, true, host),
    }
}

/// Undo a refused or unsent edit. `restore` is false when something newer already owns the cell.
pub(crate) fn rollback(
    world: &mut World,
    player: &mut Player,
    edit: PendingEdit,
    restore: bool,
    host: &mut impl Host,
) {
    if restore {
        let (x, y, z) = edit.cell;
        world.set_block(x, y, z, edit.prev);
        world.note_cell_changed(x, y, z);
    }
    let cell = edit.cell;
    match edit.kind {
        PendingKind::Break { id, gained } => {
            if gained {
                player.inventory.revoke(id, 1);
            }
            host.cue(world, Cue::BreakRejected { id });
        }
        PendingKind::Place(id) => {
            player.inventory.add(id, 1);
            host.cue(world, Cue::PlaceRejected { id });
        }
    }
    host.cue(world, Cue::Rejected { cell });
}

/// One unit of `tool` became `new_tool`. The inventory updates here; the host hears the outcome.
fn finish_tool_change(
    world: &mut World,
    player: &mut Player,
    tool: BlockId,
    new_tool: BlockId,
    target: BlockId,
    new_cell: BlockId,
    host: &mut impl Host,
) {
    if player.inventory.consume(tool, 1) && new_tool != AIR {
        player.inventory.add(new_tool, 1);
    }
    host.cue(world, Cue::ToolChanged { tool, new_tool });
    let reg = world.registry();
    let before = reg.configuration(tool).len();
    let after = reg.configuration(new_tool).len();
    let outcome = if new_cell == AIR {
        ToolUse::CellDissolved { cell: target }
    } else if new_tool == AIR {
        ToolUse::ToolDissolved { cell: target }
    } else if after > before {
        ToolUse::Drew { cell: target }
    } else if after < before {
        ToolUse::Gave { cell: target }
    } else {
        ToolUse::Exchanged { cell: target }
    };
    host.cue(world, Cue::ToolUsed(outcome));
}

/// Write `id` into `cell` and wake its contacts. Returns what it held and where it sounds.
fn write_cell(world: &mut World, (x, y, z): (i32, i32, i32), id: BlockId) -> (BlockId, DVec3) {
    let prev = world.block_at(x, y, z);
    world.set_block(x, y, z, id);
    world.note_cell_changed(x, y, z);
    (prev, sound_at(world, x, y, z))
}

fn cell_center(x: i32, y: i32, z: i32) -> DVec3 {
    DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5)
}

/// Where a sound at cell `(x, y, z)` plays. A round world's storage cell sits where its chart embeds it.
fn sound_at(world: &World, x: i32, y: i32, z: i32) -> DVec3 {
    crate::space::atlas::embed_cell(world.atlases(), (x, y, z))
        .unwrap_or_else(|| cell_center(x, y, z))
}

/// Turn the body frame toward −gravity. Weak gravity barely turns it; near-zero gravity leaves it.
pub(crate) fn align_body(player: &mut Player, dt: f32) {
    let g0 = crate::player::STANDARD_GRAVITY;
    let g = player.gravity.length();
    let weight = crate::math::smooth_between(0.02 * g0 as f32, 0.10 * g0 as f32, g as f32) as f64;
    if weight > 0.0 {
        let amount = 1.0 - (-crate::camera::ALIGN_RATE * weight * dt as f64).exp();
        player.orientation.align(-player.gravity / g, amount);
    }
}

/// One physics step in the patch under the player, or in world space when there is no patch.
/// A cruise feels no gravity. Returns landing trauma in `[0, 1]`.
pub(crate) fn integrate(
    player: &mut Player,
    world: &World,
    input: &MoveInput,
    dt: f32,
    gravity: DVec3,
) -> f32 {
    let cruising = player.cruising();
    let gravity = if cruising { DVec3::ZERO } else { gravity };
    match if cruising {
        None
    } else {
        world.atlas_at(player.position)
    } {
        Some(atlas) => movement::update_player_in(player, world, atlas, input, dt, gravity),
        None => movement::update_player(player, world, input, dt, gravity),
    }
}
