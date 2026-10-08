//! Multiplayer sync and world edits: the server's events, the player's breaks, placements and
//! tool uses, and the optimistic-edit bookkeeping that rolls a refused edit back.
use std::time::Instant;

use voxel_engine::DVec3;

use super::{Game, Signal};
use crate::audio::GameEvent;
use crate::block::{AIR, BlockId};
use crate::interact;
use crate::math::{Aabb, Bounded};
use crate::modding::{Mods, ToolUse};
use crate::net::chat;
use crate::net::client::Incoming;
use crate::presence::{Stance, WireAction};
use crate::save;
use crate::ui;
use crate::world::World;

/// One optimistic edit awaiting the server's verdict: everything needed to
/// undo it if the verdict is a rejection.
pub(super) struct PendingEdit {
    cell: (i32, i32, i32),
    /// What the cell held before the optimistic apply.
    prev: BlockId,
    kind: PendingKind,
}

/// The economy side of a pending edit — what to give back on rejection.
enum PendingKind {
    /// Breaking yielded this configuration; a rejection revokes it only if it fit (`gained`).
    Break { id: BlockId, gained: bool },
    /// Placing spent one crafted block of this id; a rejection refunds it.
    Place(BlockId),
}

impl Game {
    /// Drain server events and send our heartbeat. `Some(ExitToMenu)` when the
    /// server dropped us. Runs before input so edits and chat keep flowing even
    /// while the console is open or the player stands still — and the move
    /// report doubles as the keepalive, so it too runs unconditionally.
    pub(super) fn net_phase(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) -> Option<Signal> {
        // The overwhelmingly common singleplayer path should not even enter a
        // profiling scope or call through the event-poll seam.
        self.net.as_ref()?;
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events(mods, events)
        };
        if let Some(reason) = net_disconnected {
            let interrupted = reason == crate::net::client::INTERRUPTED;
            let line = if interrupted {
                crate::net::client::INTERRUPTED.to_string()
            } else if reason.to_ascii_lowercase().contains("shutting down") {
                "* server shutting down".to_string()
            } else if reason.is_empty() {
                "* disconnected from server".to_string()
            } else {
                format!("* disconnected: {reason}")
            };
            self.console.print(line.clone());
            self.leave_notice = Some(line);
            return Some(Signal::ExitToMenu);
        }
        if let Some(net) = &mut self.net {
            net.sync_cruise(self.player.cruise.map(|c| c.speed));
            net.send_move(
                self.player.position,
                self.player.orientation.yaw,
                self.player.orientation.pitch,
                self.player.orientation.frame,
                self.player.velocity().as_vec3(),
                self.player.up_axis,
                Stance::of_player(&self.player),
            );
        }
        None
    }

    /// Drain queued server messages: apply world edits, resolve our own edit
    /// verdicts (rolling back rejected predictions), surface chat, and report
    /// a lost connection. `Some(reason)` if the server dropped us.
    fn apply_net_events(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) -> Option<String> {
        let incoming = match &mut self.net {
            Some(net) => net.poll(),
            None => return None,
        };
        let mut disconnected = None;
        for event in incoming {
            match event {
                Incoming::Edit { x, y, z, spec } => {
                    // Resolve the portable spec against our own palette, then
                    // apply. The connection already dropped stale revisions,
                    // and our own edits come back as acks, not broadcasts.
                    // A remote break names the block that WAS there (its sound
                    // class), not a generic default.
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.write_cell((x, y, z), id, |at, prev| {
                        events.push(if id == AIR {
                            GameEvent::BlockBroken { at, block: prev, local: false }
                        } else {
                            GameEvent::BlockPlaced { at, block: id, local: false }
                        });
                    });
                }
                Incoming::Mutation { x, y, z, spec } => {
                    // Snapshot content: apply silently. A client is never the reaction
                    // authority, so there is nothing to note; a cascade of a thousand
                    // cells must not play a thousand block cues.
                    let id = save::parse_block(self.world.registry_mut(), &spec);
                    self.world.set_block(x, y, z, id);
                }
                Incoming::EditAccepted { req } => {
                    // Prediction confirmed: the optimistic apply IS the truth.
                    self.pending_edits.remove(&req);
                }
                Incoming::EditRejected { req, restore } => {
                    if let Some(pending) = self.pending_edits.remove(&req) {
                        self.rollback(pending, restore, mods);
                    }
                }
                Incoming::Position { pos, frame, up } => {
                    // Authoritative snap-back (refused teleport or implausible
                    // move): request the collision slab and freeze until it
                    // lands, exactly like a local teleport. The server's
                    // MOVE_WINDOW_CAP_SECS envelope tolerates a brief pause.
                    self.world.prepare_around(pos);
                    self.player.position = pos;
                    self.player.orientation.frame = frame;
                    self.player.up_axis = up;
                    self.player.cancel_fall();
                    self.force_stream = true;
                }
                Incoming::Chat {
                    from_name,
                    channel,
                    text,
                } => {
                    // Colour the scope tag and name so chat scans at a glance: a gold
                    // [global] tag, a blue <name>, and the message body white.
                    let name = ui::Line::of(ui::Role::Accent, format!("<{from_name}> "));
                    let line = if channel == chat::GLOBAL {
                        ui::Line::of(ui::Role::Warning, "[global] ")
                            .then(ui::Role::Accent, format!("<{from_name}> "))
                    } else {
                        name
                    };
                    self.console
                        .push(line.then(ui::Role::Muted, text.to_string()));
                }
                Incoming::Joined { name } => {
                    self.console
                        .push(ui::Line::of(ui::Role::Positive, format!("* {name} joined")));
                }
                Incoming::Left { name } => {
                    self.console
                        .push(ui::Line::of(ui::Role::Muted, format!("* {name} left")));
                }
                Incoming::Time { day, day_secs } => {
                    // The server owns the shared clock: phase AND cycle length.
                    self.sky.clock.set_day(day as f64);
                    self.sky.day_length = crate::sky::DayLength::clamped(day_secs as f64);
                }
                Incoming::Disconnected { reason } => disconnected = Some(reason),
                Incoming::Interrupted => {
                    self.console
                        .push(ui::Line::of(ui::Role::Warning, crate::net::client::INTERRUPTED));
                }
                Incoming::ToolResult { req, reacted, cell, cell_spec, tool_spec } => {
                    let Some(tool) = self.pending_tools.remove(&req) else { continue };
                    if !reacted {
                        mods.on_tool_used(ToolUse::NoReaction);
                        continue;
                    }
                    let new_cell = save::parse_block(self.world.registry_mut(), &cell_spec);
                    let new_tool = save::parse_block(self.world.registry_mut(), &tool_spec);
                    let target = self.write_cell(cell, new_cell, |at, block| {
                        events.push(GameEvent::ToolReacted { at, block });
                    });
                    self.finish_tool_change(tool, new_tool, target, new_cell, mods);
                }
                Incoming::PeerSwing { id } => {
                    // The swing edge → a whoosh at the peer's current position. The
                    // local animator update already happened in `Connection::apply`.
                    if let Some(peer) = self
                        .net
                        .as_ref()
                        .and_then(|net| net.peers().find(|p| p.id() == id))
                    {
                        events.push(GameEvent::PeerSwing {
                            at: peer.sample(Instant::now()).pos.0,
                            peer: id,
                        });
                    }
                }
            }
        }
        disconnected
    }

    /// Left click: with a tool, a reaction between the tool and the targeted block; with
    /// no tool, breaking the block into the inventory.
    pub(super) fn primary_action(&mut self, mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let Some(hit) = interact::raycast(&self.world, self.player.position, self.player.forward(), interact::REACH)
        else {
            return;
        };
        match mods.tool(&self.player) {
            Some(tool) => self.use_tool(tool, hit.block, mods, events),
            None => self.break_block(hit.block, mods, events),
        }
    }

    /// Use the held configuration `tool` on the block at `cell`: ONE operation of the law between
    /// the block (A, the world cell) and the tool (B). Elements move between them; the block may
    /// empty (the tool has absorbed it) and the tool may grow, shrink or change entirely. The
    /// changed cell wakes its contacts, so a disturbed block can start a cascade. On a server the
    /// law is the server's to run: the request goes out and [`Incoming::ToolResult`] applies it.
    fn use_tool(&mut self, tool: BlockId, cell: (i32, i32, i32), mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let (x, y, z) = cell;
        let target = self.world.block_at(x, y, z);
        self.local_anim.on_action(WireAction::Swing);
        let at = sound_at(&self.world, x, y, z);
        events.push(GameEvent::Swing { at });
        if let Some(net) = &mut self.net {
            let spec = save::block_spec(self.world.registry(), tool);
            if let Some(req) = net.send_tool_use(x, y, z, spec.into()) {
                self.pending_tools.insert(req, tool);
            }
            net.send_swing();
            return;
        }
        match self.world.registry_mut().react(target, tool) {
            Some((_, new_cell, new_tool)) => {
                self.write_cell(cell, new_cell, |at, block| events.push(GameEvent::ToolReacted { at, block }));
                self.finish_tool_change(tool, new_tool, target, new_cell, mods);
                self.camera.fx.add_trauma(0.08);
            }
            None => mods.on_tool_used(ToolUse::NoReaction),
        }
    }

    /// One unit of `tool` became `new_tool` (the cell went `target` → `new_cell`): update the
    /// inventory and tell the mods.
    fn finish_tool_change(&mut self, tool: BlockId, new_tool: BlockId, target: BlockId, new_cell: BlockId, mods: &mut Mods) {
        if self.player.inventory.consume(tool, 1) && new_tool != AIR {
            self.player.inventory.add(new_tool, 1);
        }
        mods.on_tool_changed(tool, new_tool);
        let reg = self.world.registry();
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
        mods.on_tool_used(outcome);
    }

    /// No tool: break the block at `cell` into the inventory.
    fn break_block(&mut self, cell: (i32, i32, i32), mods: &mut Mods, events: &mut Vec<GameEvent>) {
        let id = self.write_cell(cell, AIR, |at, block| {
            events.push(GameEvent::Swing { at });
            events.push(GameEvent::BlockBroken { at, block, local: true });
        });
        let gained = self.player.inventory.add(id, 1);
        mods.on_block_break(id, &self.world, !gained);
        self.camera.fx.add_trauma(0.15);
        self.local_anim.on_action(WireAction::Swing);
        self.predict(PendingEdit { cell, prev: id, kind: PendingKind::Break { id, gained } }, mods);
    }

    /// Write `id` into `cell`, wake its contacts, and report the write: `report` gets where the
    /// cell sounds and what it held, which is also returned.
    fn write_cell(&mut self, (x, y, z): (i32, i32, i32), id: BlockId, report: impl FnOnce(DVec3, BlockId)) -> BlockId {
        let prev = self.world.block_at(x, y, z);
        self.world.set_block(x, y, z, id);
        self.world.note_cell_changed(x, y, z);
        report(sound_at(&self.world, x, y, z), prev);
        prev
    }

    /// Tell the server about a local edit (it validates and relays to everyone else). The local
    /// apply is a PREDICTION for responsiveness: the verdict rolls it back, cell and economy
    /// both, if we lose the race for the cell. An edit that cannot be sent rolls back at once.
    fn predict(&mut self, edit: PendingEdit, mods: &mut Mods) {
        let Some(net) = &mut self.net else { return };
        let spec: std::sync::Arc<str> = match edit.kind {
            PendingKind::Break { .. } => "air".into(),
            PendingKind::Place(id) => save::block_spec(self.world.registry(), id).into(),
        };
        let (x, y, z) = edit.cell;
        match net.send_edit(x, y, z, spec) {
            Some(req) => {
                self.pending_edits.insert(req, edit);
                net.send_swing();
            }
            None => self.rollback(edit, true, mods),
        }
    }

    /// Undo a refused or unsent edit: give the cell back when `restore` (nothing newer landed on
    /// it), revoke loot only if it fit, and refund a spent block.
    fn rollback(&mut self, edit: PendingEdit, restore: bool, mods: &mut Mods) {
        if restore {
            let (x, y, z) = edit.cell;
            self.world.set_block(x, y, z, edit.prev);
            self.world.note_cell_changed(x, y, z);
        }
        match edit.kind {
            PendingKind::Break { id, gained } => {
                if gained {
                    self.player.inventory.revoke(id, 1);
                }
                mods.on_break_rejected(id);
            }
            PendingKind::Place(id) => {
                self.player.inventory.add(id, 1);
                mods.on_place_rejected(id, &self.world);
            }
        }
    }

    /// Apply the block placements mods queued this tick, draining the buffer in
    /// place so its capacity is retained. A placement lands only in a
    /// non-obstacle cell (air, or a liquid it replaces) that doesn't overlap
    /// the player. Well-behaved mods (the crafting mod) ran an equivalent check
    /// before queueing — and before spending a block on it — so within one tick
    /// the two always agree; re-checking here is a cheap guard against a mod that
    /// queues without validating.
    pub(super) fn apply_placements(
        &mut self,
        placements: &mut Vec<(i32, i32, i32, crate::block::BlockId)>,
        events: &mut Vec<GameEvent>,
        mods: &mut Mods,
    ) {
        for (x, y, z, id) in placements.drain(..) {
            // Overlap check in f64: at far coordinates an f32 cell centre
            // would land whole blocks away from the real cell.
            let cell = Aabb::new(
                DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
                DVec3::splat(0.5),
            );
            // Lands only in empty space clear of the player; a refused placement gives the
            // spent unit back.
            if self.world.is_solid(x, y, z) || cell.intersects(&self.player.aabb()) {
                self.player.inventory.add(id, 1);
                continue;
            }
            let prev = self.write_cell((x, y, z), id, |at, _| {
                events.push(GameEvent::Swing { at });
                events.push(GameEvent::BlockPlaced { at, block: id, local: true });
            });
            self.local_anim.on_action(WireAction::Swing);
            // The server gets the same portable spec form saves use.
            self.predict(PendingEdit { cell: (x, y, z), prev, kind: PendingKind::Place(id) }, mods);
        }
    }
}

/// The world-space centre of a voxel cell (occurrence position).
fn cell_center(x: i32, y: i32, z: i32) -> DVec3 {
    DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5)
}

/// Where a sound at cell `(x, y, z)` plays: the cell's physical centre. A round world's storage cell
/// (x past a billion) sits where its chart embeds it, next to the listener.
fn sound_at(world: &World, x: i32, y: i32, z: i32) -> DVec3 {
    crate::space::atlas::embed_cell(world.atlases(), (x, y, z)).unwrap_or_else(|| cell_center(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::{Game, PendingEdit, PendingKind};
    use crate::block::{AIR, BlockId};
    use crate::game::tests::game;
    use crate::modding::Mods;

    /// A break whose loot did not fit (full inventory) and that the server refuses gives the cell
    /// back and takes no unit the player already held.
    #[test]
    fn a_refused_break_with_a_full_inventory_keeps_the_held_unit() {
        use crate::net::client::Connection;
        use crate::net::server::{self, Config};
        use std::time::{Duration, Instant};

        let server = server::spawn(0, Config { seed: 1, ..Config::default() }).expect("loopback server");
        let conn = Connection::connect("127.0.0.1", server.addr().port(), "ada", "").expect("connect");
        let mut game = game().with_net(conn);
        let rock = game.world.registry().id_by_label("rock").unwrap();
        // A rock chunk far past the server's edit reach, so the break is refused.
        let coord = crate::coord::ChunkCoord::new(40, 2, 40);
        let chunk = crate::world::chunk::Chunk::from_uniform(40, 2, 40, rock);
        game.world.store_column_chunk(crate::coord::Face::PosY, coord, chunk);
        let (x, y, z) = (645, 40, 645);
        let held = game.player.inventory.capacity() as u32;
        assert!(game.player.inventory.add(rock, held));

        let (mut mods, mut events) = (Mods::empty(), Vec::new());
        game.break_block((x, y, z), &mut mods, &mut events);
        assert_eq!(game.world.block_at(x, y, z), AIR, "the break is predicted");
        assert_eq!(game.player.inventory.count(rock), held, "the loot did not fit");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !game.pending_edits.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            assert!(game.apply_net_events(&mut mods, &mut events).is_none(), "still connected");
        }
        assert!(game.pending_edits.is_empty(), "the server answered");
        assert_eq!(game.world.block_at(x, y, z), rock, "the refused break gives the cell back");
        assert_eq!(game.player.inventory.count(rock), held, "the refused break takes no held unit");
        server.stop();
    }

    /// A game with a loaded air chunk around the returned cell, and the rock id.
    fn edit_game() -> (Game, BlockId, (i32, i32, i32)) {
        let mut game = game();
        let rock = game.world.registry().id_by_label("rock").unwrap();
        let chunk = crate::world::chunk::Chunk::from_uniform(0, 2, 0, AIR);
        game.world.store_column_chunk(crate::coord::Face::PosY, crate::coord::ChunkCoord::new(0, 2, 0), chunk);
        (game, rock, (3, 40, 3))
    }

    #[test]
    fn a_refused_place_refunds_the_block_and_re_marks_the_restored_cell() {
        let (mut game, rock, (x, y, z)) = edit_game();
        game.world.set_block(x, y, z, rock);
        let woken = game.world.reactions().pending();
        let edit = PendingEdit { cell: (x, y, z), prev: AIR, kind: PendingKind::Place(rock) };
        game.rollback(edit, true, &mut Mods::empty());
        assert_eq!(game.world.block_at(x, y, z), AIR, "the cell is given back");
        assert_eq!(game.player.inventory.count(rock), 1, "the spent block is refunded");
        assert!(game.world.reactions().pending() > woken, "the restored cell wakes its contacts");
    }

    #[test]
    fn a_refused_edit_without_restore_keeps_the_cell_and_revokes_loot_that_fit() {
        let (mut game, rock, (x, y, z)) = edit_game();
        assert!(game.player.inventory.add(rock, 1));
        let edit = PendingEdit { cell: (x, y, z), prev: rock, kind: PendingKind::Break { id: rock, gained: true } };
        game.rollback(edit, false, &mut Mods::empty());
        assert_eq!(game.world.block_at(x, y, z), AIR, "a newer edit owns the cell");
        assert_eq!(game.player.inventory.count(rock), 0, "loot that fit is revoked");
        assert_eq!(game.world.reactions().pending(), 0, "an untouched cell wakes nothing");
    }
}
