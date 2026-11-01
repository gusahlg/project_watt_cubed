// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn audit_two_tool_commits_spend_one_held_unit() {
    let mut world = World::with_kind_cfg(1, crate::render_config::RenderConfig::default(), crate::world::generation::WorldgenKind::Flat, Default::default(), false);
    let (a, e) = crate::sim::reactions::destructive_pair(world.registry_mut());
    let (_, na, ne) = world.registry_mut().react(a, e).unwrap();
    let mut player = crate::player::Player::new(DVec3::new(0.5, 80.5, 0.5));
    player.inventory.add(e, 1);
    let mut game = Game::new(world, player, "audit".into());
    let mut mods = Mods::empty();
    game.pending_tools.insert(1, e);
    game.pending_tools.insert(2, e);
    let event = |req, cell| Incoming::ToolResult { req, reacted: true, cell, cell_spec: game.world.registry().spec(na).into(), tool_spec: game.world.registry().spec(ne).into() };
    let events = vec![event(1, (0, 80, 0)), event(2, (1, 80, 0))];
    game.apply_incoming(events, &mut mods, &mut Vec::new());
    assert_eq!(game.player.inventory.count(ne), 1);
    assert_eq!(game.player.inventory.count(e), 0);
    assert_eq!(crate::sim::reactions::CellStore::block_at(&game.world, (0, 80, 0)), Some(na));
    assert_eq!(crate::sim::reactions::CellStore::block_at(&game.world, (1, 80, 0)), Some(na));
    eprintln!("BUG: two tool commits transform two cells, but only one tool product survives");
}
