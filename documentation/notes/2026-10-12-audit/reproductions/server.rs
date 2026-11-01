// SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Included as a child of net::server in a disposable copy. These tests assert
// the observed bug, not the desired fixed behavior.
use super::*;

fn fixture(capacity: usize) -> (Arc<Mutex<State>>, Receiver<Arc<[u8]>>, crate::world::terrain::Generator, BlockId, BlockId) {
    let mut registry = BlockRegistry::with_builtins();
    let generator: crate::world::terrain::Generator = Arc::new(crate::world::generation::FlatTerrain::new(&mut registry, 1));
    let (a, e) = crate::sim::reactions::destructive_pair(&mut registry);
    let mut state = State::new(registry, 0.3, crate::player::MAX_SPEED);
    let (out, rx) = sync_channel(capacity);
    let out = Outbox { tx: Some(out), writer: Arc::default() };
    let mut h = PlayerHandle::new("auditor".into(), DVec3::new(0.5, 80.5, 0.5), DQuat::IDENTITY, Face::PosY, false, out, Arc::new(Notify::new()));
    h.ready = true;
    state.players.insert(1, h);
    state.write((0, 80, 0), a, false).unwrap();
    (Arc::new(Mutex::new(state)), rx, generator, a, e)
}

#[test]
fn audit_tool_bypasses_edit_policy() {
    struct Deny;
    impl ServerMod for Deny {
        fn id(&self) -> &'static str { "deny-all" }
        fn validate_edit(&mut self, _: &EditIntent) -> Verdict { Verdict::Deny { reason: "protected".into() } }
    }
    let (shared, _rx, generator, a, e) = fixture(8);
    let hooks = Mutex::new(hooks::Table::new(vec![Box::new(Deny)]));
    on_edit(&shared, Some(&hooks), &generator, 1, 1, 0, 80, 0, 1, "air");
    assert_eq!(shared.lock_recover().edits[&(0, 80, 0)].block, a);
    let tool = shared.lock_recover().registry.spec_ref(e).clone();
    on_tool_use(&shared, &generator, 1, 2, 0, 80, 0, 1, &tool);
    assert_ne!(shared.lock_recover().edits[&(0, 80, 0)].block, a);
    eprintln!("BUG: a denied edit became an accepted tool mutation");
}

#[test]
fn audit_full_outbox_loses_tool_commit_without_kicking_sender() {
    let (shared, rx, generator, a, e) = fixture(1);
    {
        let state = shared.lock_recover();
        state.players[&1].out.try_send(ServerMessage::Pong { nonce: 7 }.frame()).unwrap();
    }
    let tool = shared.lock_recover().registry.spec_ref(e).clone();
    on_tool_use(&shared, &generator, 1, 2, 0, 80, 0, 1, &tool);
    let state = shared.lock_recover();
    assert_ne!(state.edits[&(0, 80, 0)].block, a);
    assert!(!state.players[&1].kicked.load(Ordering::Relaxed));
    assert!(matches!(ServerMessage::decode(&rx.try_recv().unwrap()), Some(ServerMessage::Pong { .. })));
    assert!(rx.try_recv().is_err());
    eprintln!("BUG: the cell committed, but its sender receives no ToolResult or kick");
}

#[test]
fn audit_reaction_second_write_refusal_loses_matter() {
    let (shared, _rx, generator, a, e) = fixture(8);
    let mut state = shared.lock_recover();
    let (_, na, nb) = state.registry.react(a, e).unwrap();
    state.write((1, 80, 0), e, false).unwrap();
    // Keep the old A named elsewhere, so replacing it does not free a pool slot.
    state.write((0, 90, 0), a, false).unwrap();
    state.write((1, 90, 0), na, false).unwrap();
    let mut i = 0u32;
    while state.spec_pool.len() < MAX_SPEC_POOL {
        let cfg = material::Configuration::single(material::Element::new([i as u8, (i >> 8) as u8, 211, 17]));
        let id = state.registry.intern(&cfg).unwrap();
        if id != nb { state.write((i as i32 + 100, 90, 0), id, false).unwrap(); }
        i += 1;
    }
    let before = u32::from(state.registry.amount(a)) + u32::from(state.registry.amount(e));
    let mut scheduler = ReactionScheduler::new();
    scheduler.restore(&[(0, Contact { lo: (0, 80, 0), axis: 0 })]);
    let mutations = scheduler.tick(&mut ServerCells { state: &mut state, generator: &generator }, reactions::Budget { contacts_per_turn: 1 });
    let after = u32::from(state.registry.amount(state.edits[&(0, 80, 0)].block)) + u32::from(state.registry.amount(state.edits[&(1, 80, 0)].block));
    assert_eq!(mutations.len(), 1);
    assert_ne!(before, after);
    eprintln!("BUG: partial reaction changed contact amount from {before} to {after}");
}

#[test]
fn audit_physical_collision_misses_chart_ground() {
    let mut registry = BlockRegistry::with_builtins();
    let generator = crate::world::terrain::generator(&mut registry, 42, TerrainCfg::default());
    let seams = Seams::new(generator.atlases().to_vec());
    let state = State::new(registry, 0.3, crate::player::MAX_SPEED);
    let spawn = generator.chart_spawn().unwrap();
    let atlas = generator.atlases().iter().find(|a| a.local(spawn).is_some()).unwrap();
    let local = atlas.local(spawn).unwrap();
    let x = local.storage.x.floor() as i32;
    let z = local.storage.z.floor() as i32;
    let ground = generator.height(x, z);
    let storage_eye = DVec3::new(x as f64 + 0.5, ground as f64 - 2.0, z as f64 + 0.5);
    let physical_eye = atlas.embed_storage(local.patch, storage_eye);
    assert!(body_blocked(&state, &generator, &seams, storage_eye, storage_eye, Stance::Standing, Face::PosY, &[]));
    assert!(!body_blocked(&state, &generator, &seams, physical_eye, physical_eye, Stance::Standing, Face::PosY, &[]));
    eprintln!("BUG: the same embedded body is blocked in storage, clear in physical coordinates");
}

#[test]
fn audit_refused_tool_still_queries_arbitrary_terrain() {
    struct Counted { calls: Arc<AtomicUsize> }
    impl TerrainGenerator for Counted {
        fn height(&self, _: i32, _: i32) -> i32 { 0 }
        fn surface_at(&self, _: i32, _: i32) -> BlockId { AIR }
        fn deep(&self) -> BlockId { AIR }
        fn voxel_at(&self, _: i32, _: i32, _: i32) -> BlockId { self.calls.fetch_add(1, Ordering::Relaxed); AIR }
    }
    let (shared, _rx, _, _, e) = fixture(128);
    let calls = Arc::new(AtomicUsize::new(0));
    let generator: crate::world::terrain::Generator = Arc::new(Counted { calls: calls.clone() });
    let tool = shared.lock_recover().registry.spec_ref(e).clone();
    // This is exactly the Charge::Answer dispatch for over-budget tool messages.
    for i in 0..100 {
        refuse_tool(shared.lock_recover(), &shared, &generator, 1, i, (100000 + i as i32, 0, 0), &tool);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 100);
    eprintln!("BUG: 100 over-budget refusals still generated 100 arbitrary cells");
}
