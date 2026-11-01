//! Multiplayer sync and world edits: the server's events, the player's breaks, placements and
//! tool uses, and the optimistic-edit bookkeeping that rolls a refused edit back.
use super::{Game, Signal};
use crate::audio::GameEvent;
use crate::block::{AIR, BlockId};
use crate::camera::GameCamera;
use crate::interact;
use crate::modules::{Channel, Message, Modules, Notice, NoticeLevel, ToolUse};
use crate::net::client::Incoming;
use crate::play::{self, Cue, Host};
use crate::presence::{Animator, Stance, WireAction};
use crate::sky::Sky;
use crate::world::World;

pub(super) use crate::play::PendingEdit;
#[cfg(test)]
pub(super) use crate::play::PendingKind;

/// The game's half of a shared play cue: the same sounds, module calls and snap load as before.
struct GameHost<'a> {
    modules: &'a mut Modules,
    events: &'a mut Vec<GameEvent>,
    camera: &'a mut GameCamera,
    anim: &'a mut Animator,
    sky: &'a mut Sky,
    force_stream: &'a mut bool,
}

impl Host for GameHost<'_> {
    fn cue(&mut self, world: &World, cue: Cue) {
        match cue {
            Cue::RemoteEdit { at, id, prev, .. } => {
                self.events.push(if id == AIR {
                    GameEvent::BlockBroken { at, block: prev, local: false }
                } else {
                    GameEvent::BlockPlaced { at, block: id, local: false }
                });
            }
            Cue::LocalBreak { at, block, .. } => {
                self.events.push(GameEvent::Swing { at });
                self.events.push(GameEvent::BlockBroken { at, block, local: true });
            }
            Cue::LocalPlace { at, block, .. } => {
                self.events.push(GameEvent::Swing { at });
                self.events.push(GameEvent::BlockPlaced { at, block, local: true });
            }
            Cue::SwingAt { at } => self.events.push(GameEvent::Swing { at }),
            Cue::PeerSwing { at, peer } => self.events.push(GameEvent::PeerSwing { at, peer }),
            Cue::ToolReacted { at, block } => self.events.push(GameEvent::ToolReacted { at, block }),
            Cue::BreakNoted { id, overflow } => self.modules.on_block_break(id, world, overflow),
            Cue::BreakRejected { id } => self.modules.on_break_rejected(id),
            Cue::PlaceRejected { id } => self.modules.on_place_rejected(id, world),
            Cue::ToolNoReaction => self.modules.on_tool_used(ToolUse::NoReaction),
            Cue::ToolChanged { tool, new_tool } => self.modules.on_tool_changed(tool, new_tool),
            Cue::ToolUsed(outcome) => self.modules.on_tool_used(outcome),
            Cue::Chat { from, channel, text } => {
                self.modules.on_message(&Message::Chat {
                    from: from.as_ref(),
                    channel: Channel::from_wire(channel),
                    text: text.as_ref(),
                });
            }
            Cue::Joined { name } => {
                self.modules.on_message(&Message::Joined { name: name.as_ref() });
            }
            Cue::Left { name } => {
                self.modules.on_message(&Message::Left { name: name.as_ref() });
            }
            Cue::Time { day, day_secs } => {
                self.sky.clock.set_day(day as f64);
                self.sky.day_length = crate::sky::DayLength::clamped(day_secs as f64);
            }
            Cue::Interrupted => {
                let text = crate::net::client::INTERRUPTED.to_string();
                post(self.modules, Notice { level: NoticeLevel::Warning, text });
            }
            Cue::Trauma(amount) => self.camera.fx.add_trauma(amount),
            Cue::AnimSwing => self.anim.on_action(WireAction::Swing),
            Cue::Snapped { .. } => *self.force_stream = true,
            Cue::Mutation { .. } | Cue::Confirmed { .. } | Cue::Rejected { .. } | Cue::ToolDone { .. } => {}
        }
    }

    fn prepare(&mut self, world: &mut World, pos: voxel_engine::DVec3) {
        world.prepare_around(pos);
    }
}

impl Game {
    /// Drain server events and send our heartbeat. `Some(ExitToMenu)` when the
    /// server dropped us. Runs before input so edits and chat keep flowing even
    /// while a module is typing or the player stands still — and the move
    /// report doubles as the keepalive, so it too runs unconditionally.
    pub(super) fn net_phase(&mut self, modules: &mut Modules, events: &mut Vec<GameEvent>) -> Option<Signal> {
        // The overwhelmingly common singleplayer path should not even enter a
        // profiling scope or call through the event-poll seam.
        self.net.as_ref()?;
        let net_disconnected = {
            let _p = voxel_engine::profile::scope(voxel_engine::profile::Meter::NetEvents);
            self.apply_net_events(modules, events)
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
            post(modules, Notice { level: NoticeLevel::Warning, text: line.clone() });
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
    /// verdicts (rolling back rejected predictions), hand chat to the modules, and report
    /// a lost connection. `Some(reason)` if the server dropped us.
    fn apply_net_events(&mut self, modules: &mut Modules, events: &mut Vec<GameEvent>) -> Option<String> {
        let incoming = match &mut self.net {
            Some(net) => net.poll(),
            None => return None,
        };
        self.apply_incoming(incoming, modules, events)
    }

    /// The network phase alone, for probes. True once the join overlay is in.
    #[cfg(test)]
    pub(crate) fn pump_net(&mut self, modules: &mut Modules) -> bool {
        self.apply_net_events(modules, &mut Vec::new());
        self.net.as_ref().is_some_and(|net| net.snapshot_ready())
    }

    /// Apply one poll's server events in order; runs of snapshot cells install in bulk.
    pub(super) fn apply_incoming(
        &mut self,
        incoming: Vec<Incoming>,
        modules: &mut Modules,
        events: &mut Vec<GameEvent>,
    ) -> Option<String> {
        let mut host = GameHost {
            modules,
            events,
            camera: &mut self.camera,
            anim: &mut self.local_anim,
            sky: &mut self.sky,
            force_stream: &mut self.force_stream,
        };
        play::apply_incoming(
            &mut self.world,
            &mut self.player,
            &mut self.pending_edits,
            &mut self.pending_tools,
            self.net.as_ref(),
            incoming,
            &mut host,
        )
    }

    /// Left click: with a tool, a reaction between the tool and the targeted block; with
    /// no tool, breaking the block into the inventory.
    pub(super) fn primary_action(&mut self, modules: &mut Modules, events: &mut Vec<GameEvent>) {
        let Some(hit) = interact::raycast(&self.world, self.player.position, self.player.forward(), interact::REACH)
        else {
            return;
        };
        match modules.tool(&self.player) {
            Some(tool) => self.use_tool(tool, hit.block, modules, events),
            None => self.break_block(hit.block, modules, events),
        }
    }

    /// Use the held configuration `tool` on the block at `cell`. On a server the law runs there.
    fn use_tool(&mut self, tool: BlockId, cell: (i32, i32, i32), modules: &mut Modules, events: &mut Vec<GameEvent>) {
        let mut host = GameHost {
            modules,
            events,
            camera: &mut self.camera,
            anim: &mut self.local_anim,
            sky: &mut self.sky,
            force_stream: &mut self.force_stream,
        };
        play::use_tool(
            &mut self.world,
            &mut self.player,
            &mut self.pending_tools,
            self.net.as_mut(),
            tool,
            cell,
            &mut host,
        );
    }

    /// No tool: break the block at `cell` into the inventory.
    fn break_block(&mut self, cell: (i32, i32, i32), modules: &mut Modules, events: &mut Vec<GameEvent>) {
        let mut host = GameHost {
            modules,
            events,
            camera: &mut self.camera,
            anim: &mut self.local_anim,
            sky: &mut self.sky,
            force_stream: &mut self.force_stream,
        };
        play::commit_break(
            &mut self.world,
            &mut self.player,
            &mut self.pending_edits,
            self.net.as_mut(),
            cell,
            &mut host,
        );
    }

    /// Undo a refused or unsent edit: give the cell back when `restore` (nothing newer landed on
    /// it), revoke loot only if it fit, and refund a spent block. The live path rolls back inside
    /// [`crate::play`]; tests call this wrapper.
    #[cfg(test)]
    fn rollback(&mut self, edit: PendingEdit, restore: bool, modules: &mut Modules) {
        let mut events = Vec::new();
        let mut host = GameHost {
            modules,
            events: &mut events,
            camera: &mut self.camera,
            anim: &mut self.local_anim,
            sky: &mut self.sky,
            force_stream: &mut self.force_stream,
        };
        play::rollback(&mut self.world, &mut self.player, edit, restore, &mut host);
    }

    /// Apply the block placements modules queued this tick, draining the buffer in
    /// place so its capacity is retained. A placement lands only in a
    /// non-obstacle cell (air, or a liquid it replaces) that doesn't overlap
    /// the player. Well-behaved modules (the crafting module) ran an equivalent check
    /// before queueing — and before spending a block on it — so within one tick
    /// the two always agree; re-checking here is a cheap guard against a module that
    /// queues without validating.
    pub(super) fn apply_placements(
        &mut self,
        placements: &mut Vec<(i32, i32, i32, crate::block::BlockId)>,
        events: &mut Vec<GameEvent>,
        modules: &mut Modules,
    ) {
        let mut host = GameHost {
            modules,
            events,
            camera: &mut self.camera,
            anim: &mut self.local_anim,
            sky: &mut self.sky,
            force_stream: &mut self.force_stream,
        };
        for (x, y, z, id) in placements.drain(..) {
            // Lands only in empty space clear of the player; a refused placement gives the
            // spent unit back.
            if play::blocks_placement(&self.world, &self.player, (x, y, z)) {
                self.player.inventory.add(id, 1);
                continue;
            }
            play::commit_place(
                &mut self.world,
                &mut self.player,
                &mut self.pending_edits,
                self.net.as_mut(),
                (x, y, z),
                id,
                &mut host,
            );
        }
    }
}

/// Hand a session notice to the modules now; with no module showing it, stderr.
fn post(modules: &mut Modules, notice: Notice) {
    if !modules.on_message(&Message::Notice(&notice)) {
        eprintln!("{}", notice.text);
    }
}

#[cfg(test)]
mod tests {
    use super::{Game, PendingEdit, PendingKind};
    use crate::block::{AIR, BlockId};
    use crate::game::tests::game;
    use crate::modules::Modules;

    /// A break whose loot did not fit (full inventory) and that the server refuses gives the cell
    /// back and takes no unit the player already held.
    /// One poll's snapshot cells install in bulk, and the events between them keep their place:
    /// an edit lands after the cells before it and under the cells after it, as applying every
    /// event in order does.
    #[test]
    fn a_poll_of_snapshot_cells_installs_in_order_with_the_events_between() {
        use crate::net::client::Incoming;
        use crate::player::Player;
        use crate::render_config::RenderConfig;
        use crate::world::World;
        use std::sync::Arc;
        use voxel_engine::DVec3;
        let make = || {
            let flat = crate::world::generation::WorldgenKind::Flat;
            World::with_kind_cfg(3, RenderConfig::default(), flat, Default::default(), false)
        };
        let mut reference = make();
        let mut game = Game::new(make(), Player::new(DVec3::new(0.5, 80.0, 0.5)), "ledger".into());
        let soil = reference.registry().id_by_label("soil").unwrap();
        let rock = reference.registry().id_by_label("rock").unwrap();
        let ground = reference.surface_y(0, 0);
        let cells: Vec<(i32, i32, i32)> =
            (0..2000).map(|i| ((i * 37) % 400 - 200, ground + i % 3, (i * 53) % 400 - 200)).collect();
        let mut incoming = Vec::new();
        let cell = |incoming: &mut Vec<Incoming>, reference: &mut World, (x, y, z): (i32, i32, i32), id| {
            let spec: Arc<str> = reference.registry().spec(id).into();
            incoming.push(Incoming::Mutation { x, y, z, spec });
            reference.set_block(x, y, z, id);
        };
        for &c in &cells[..1500] {
            cell(&mut incoming, &mut reference, c, soil);
        }
        let (x, y, z) = cells[7];
        incoming.push(Incoming::Edit { x, y, z, spec: "air".into() });
        reference.set_block(x, y, z, crate::block::AIR);
        reference.note_cell_changed(x, y, z);
        for &c in cells[1500..].iter().chain(&cells[5..6]) {
            cell(&mut incoming, &mut reference, c, rock);
        }

        game.apply_incoming(incoming, &mut crate::modules::testing::standard(), &mut Vec::new());
        game.world.assert_same_edits(&reference);
    }

    /// A chat line a module queues goes to the server, which relays it back as a message for the
    /// modules; the core shows nothing itself.
    #[test]
    fn queued_chat_goes_to_the_server_and_returns_as_a_message() {
        use crate::game::tests::{probe_modules, type_word};
        use crate::net::client::Connection;
        use crate::net::server::{self, Config};
        use std::time::{Duration, Instant};

        let server = server::spawn(0, Config { seed: 1, ..Config::default() }).expect("loopback server");
        let conn = Connection::connect("127.0.0.1", server.addr().port(), "ada", "").expect("connect");
        let mut game = game().with_net(conn);
        let (mut modules, log) = probe_modules();
        let mut settings = crate::settings::Settings::default();
        type_word(&mut game, &mut modules, &mut settings, "chat");
        assert!(game.chat_out.is_empty(), "sent, not kept");
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut events = Vec::new();
        while !log.borrow().iter().any(|l| l.starts_with("chat")) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            assert!(game.apply_net_events(&mut modules, &mut events).is_none(), "still connected");
        }
        let chats: Vec<String> = log.borrow().iter().filter(|l| l.starts_with("chat")).cloned().collect();
        assert_eq!(chats, ["chat ada Local hello"]);
        server.stop();
    }

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

        let (mut modules, mut events) = (Modules::empty(), Vec::new());
        game.break_block((x, y, z), &mut modules, &mut events);
        assert_eq!(game.world.block_at(x, y, z), AIR, "the break is predicted");
        assert_eq!(game.player.inventory.count(rock), held, "the loot did not fit");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !game.pending_edits.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            assert!(game.apply_net_events(&mut modules, &mut events).is_none(), "still connected");
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
        game.rollback(edit, true, &mut Modules::empty());
        assert_eq!(game.world.block_at(x, y, z), AIR, "the cell is given back");
        assert_eq!(game.player.inventory.count(rock), 1, "the spent block is refunded");
        assert!(game.world.reactions().pending() > woken, "the restored cell wakes its contacts");
    }

    #[test]
    fn a_refused_edit_without_restore_keeps_the_cell_and_revokes_loot_that_fit() {
        let (mut game, rock, (x, y, z)) = edit_game();
        assert!(game.player.inventory.add(rock, 1));
        let edit = PendingEdit { cell: (x, y, z), prev: rock, kind: PendingKind::Break { id: rock, gained: true } };
        game.rollback(edit, false, &mut Modules::empty());
        assert_eq!(game.world.block_at(x, y, z), AIR, "a newer edit owns the cell");
        assert_eq!(game.player.inventory.count(rock), 0, "loot that fit is revoked");
        assert_eq!(game.world.reactions().pending(), 0, "an untouched cell wakes nothing");
    }
}
