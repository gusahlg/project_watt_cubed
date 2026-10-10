//! Edits, tool uses, the spec pool, hooks on edits, and reaction batches.
use super::super::*;
use super::support::*;

#[test]
fn client_specs_resolve_known_ids_without_growing_the_table() {
    let mut r = BlockRegistry::with_builtins();
    let known = r
        .intern(&material::Configuration::single(material::Element::new([40, 80, 120, 160])))
        .unwrap();
    let spec = r.spec(known);
    let count = r.block_count();
    // A known configuration resolves even with the table "full" (limit at the current count).
    assert_eq!(resolve_client_spec_within(&mut r, &spec, count), Some(known));
    assert_eq!(resolve_client_spec_within(&mut r, "air", 0), Some(AIR));
    assert_eq!(r.block_count(), count);
    // A novel one interns only below the line, and never more than once.
    let novel = "c:0105060708";
    assert_eq!(resolve_client_spec_within(&mut r, novel, count), None, "at the line: refused");
    assert_eq!(r.block_count(), count, "a refusal does not grow the table");
    let id = resolve_client_spec_within(&mut r, novel, count + 1).expect("below the line");
    assert_eq!(r.block_count(), count + 1);
    assert_eq!(resolve_client_spec_within(&mut r, novel, 0), Some(id), "now known");
    assert_eq!(resolve_client_spec_within(&mut r, "c:zz", usize::MAX), None, "malformed");
    assert_eq!(r.block_count(), count + 1);
}

#[test]
fn a_tool_use_from_an_unready_player_is_not_evaluated() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    let mut p = test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick());
    p.ready = false;
    players.insert(1u32, p);
    let shared = Arc::new(Mutex::new(test_state(players)));
    let (_, tool) = place_pair(&shared, (8, 20, 8));
    let count = shared.lock_recover().registry.block_count();
    on_tool_use(&shared, &test_generator(), 1, 7, 8, 20, 8, 1, &Arc::from(tool.as_str()));
    assert_eq!(shared.lock_recover().registry.block_count(), count, "nothing interned");
    assert!(rx.try_recv().is_err());
}

#[test]
fn a_tool_use_runs_one_operation_of_the_law_on_the_server() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (peer_out, peer_rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    players.insert(2u32, test_player(DVec3::new(9.5, 20.0, 8.5), peer_out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let (_, tool) = place_pair(&shared, (8, 20, 8));
    on_tool_use(&shared, &test_generator(), 1, 7, 8, 20, 8, 1, &Arc::from(tool.as_str()));
    let replies = drain_msgs(&rx);
    let [ServerMessage::ToolResult { req: 7, reacted: true, rev: 2, cell_spec, tool_spec }] = replies.as_slice() else {
        panic!("expected one ToolResult, got {replies:?}");
    };
    let state = shared.lock_recover();
    let cell = state.registry.lookup_spec(cell_spec).unwrap();
    let new_tool = state.registry.lookup_spec(tool_spec).unwrap();
    assert_eq!(state.registry.configuration(cell).len(), 3, "one element left the block");
    assert_eq!(state.registry.configuration(new_tool).len(), 5, "and joined the tool");
    assert_eq!(state.edits[&(8, 20, 8)].rev, 2);
    assert_eq!(state.reactions.pending(), 6, "the changed cell woke its contacts");
    drop(state);
    assert!(
        matches!(drain_msgs(&peer_rx).as_slice(), [ServerMessage::Edit { x: 8, y: 20, z: 8, rev: 2, .. }]),
        "peers see the cell change"
    );
    // A stale revision, a void tool and a tool out of reach are refused with `reacted: false`.
    on_tool_use(&shared, &test_generator(), 1, 8, 8, 20, 8, 1, &Arc::from(tool.as_str()));
    on_tool_use(&shared, &test_generator(), 1, 9, 8, 20, 8, 2, &Arc::from("air"));
    on_tool_use(&shared, &test_generator(), 1, 10, 80, 20, 8, 0, &Arc::from(tool.as_str()));
    let refused = drain_msgs(&rx);
    assert_eq!(refused.len(), 3);
    assert!(refused.iter().all(|m| matches!(m, ServerMessage::ToolResult { reacted: false, .. })));
}

#[test]
fn reaction_snapshots_carry_each_cell_once_with_its_final_content() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let mut state = test_state(players);
    let rock = state.registry.lookup_spec(&rock_spec()).or_else(|| state.registry.parse_spec(&rock_spec())).unwrap();
    let spec = state.intern(rock).unwrap();
    state.edits.insert((1, 2, 3), Cell { block: rock, spec: spec.clone(), rev: 2, natural: false });
    state.edits.insert((4, 5, 6), Cell { block: rock, spec, rev: 1, natural: false });
    let muts = [
        Mutation { pos: (1, 2, 3), from: AIR, to: rock },
        Mutation { pos: (4, 5, 6), from: AIR, to: rock },
        Mutation { pos: (1, 2, 3), from: rock, to: rock },
    ];
    send_reaction_mutations(&mut state, &muts);
    let frame = rx.try_recv().expect("one snapshot batch");
    let ServerMessage::Snapshot { edits } = ServerMessage::decode(&frame).unwrap() else {
        panic!("expected a Snapshot");
    };
    let cells: Vec<(i32, i32, i32, u32)> = edits.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();
    assert_eq!(cells, vec![(4, 5, 6, 1), (1, 2, 3, 2)], "each cell once, at its last commit");
    assert!(rx.try_recv().is_err(), "no second batch");
}

#[test]
fn on_edit_queues_place_and_break_events() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let rock = rock_spec();
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
    {
        let state = shared.lock_recover();
        assert_eq!(state.reactions.pending(), 6, "a placement wakes the cell's six contacts");
    }
    on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 1, "air");
    let state = shared.lock_recover();
    assert_eq!(state.reactions.pending(), 6, "a removal wakes the same six (deduplicated)");
}

#[test]
fn scripted_reactions_match_a_local_world() {
    use crate::render_config::RenderConfig;
    use crate::sim::reactions::{destructive_pair, Budget, ReactionScheduler};
    use crate::world::World;

    fn run<S: CellStore>(store: &mut S, a: BlockId, e: BlockId, y: i32) -> Vec<(Pos, Vec<u8>, Vec<u8>)> {
        store.set_block((0, y, 0), a);
        store.set_block((1, y, 0), e);
        let mut s = ReactionScheduler::new();
        s.wake_cell((1, y, 0));
        let mut out = Vec::new();
        for _ in 0..50 {
            for m in s.tick(store, Budget::DEFAULT) {
                let r = store.registry();
                out.push((m.pos, r.encoding(m.from).as_bytes().to_vec(), r.encoding(m.to).as_bytes().to_vec()));
            }
        }
        out
    }

    let mut world = World::with_kind(42, RenderConfig::default(), crate::world::generation::WorldgenKind::Diffusion, true);
    let (wa, we) = destructive_pair(world.registry_mut());
    let mut registry = BlockRegistry::with_builtins();
    let generator = crate::world::terrain::generator(&mut registry, 42, Default::default());
    let spawn = generator.chart_spawn().expect("a charted start world");
    let y = generator.atlases().iter().find_map(|a| a.storage_of(spawn)).expect("spawn storage")[1] as i32;
    let local = run(&mut world, wa, we, y);

    let (sa, se) = destructive_pair(&mut registry);
    let mut state = test_state(HashMap::new());
    state.registry = registry;
    let mut cells = ServerCells { state: &mut state, generator: &generator };
    let server = run(&mut cells, sa, se, y);
    assert_eq!(local, server);
    assert!(!local.is_empty(), "the pair must react");
}

/// An out-of-reach edit must be rejected; an in-reach one must be recorded.
/// Exercised directly against the shared state without a socket.
#[test]
fn edit_reach_is_enforced() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));

    on_edit(&shared, None, chartless(), 1, 1, 500, 20, 500, 0, "air"); // far away: rejected
    on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 0, "air"); // in reach: recorded

    let state = shared.lock_recover();
    assert!(state.edits.contains_key(&(8, 20, 8)), "in-reach edit recorded");
    assert!(!state.edits.contains_key(&(500, 20, 500)), "out-of-reach edit dropped");
}

/// The cell revision makes racing edits resolve to exactly one winner, and
/// the sender's ack — not a broadcast echo — carries the verdict prediction
/// rolls back on.
#[test]
fn a_full_outbox_disconnects_instead_of_losing_an_accepted_cells_content() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(1);
    out.try_send(ServerMessage::Pong { nonce: 0 }.encode().into()).unwrap();
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, "air");
    let state = shared.lock_recover();
    assert_eq!(state.edits[&(8, 20, 8)].rev, 1, "rejoining will replay committed content");
    assert!(state.players[&1].kicked.load(Ordering::Relaxed));
}

#[test]
fn edit_revisions_arbitrate_races_and_ack_the_sender() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let ack = |rx: &std::sync::mpsc::Receiver<Arc<[u8]>>| {
        let mut reply = ServerMessage::decode(&rx.try_recv().expect("an answer is owed"));
        if let Some(ServerMessage::Snapshot { edits }) = reply {
            assert_eq!(edits.len(), 1);
            assert_eq!((edits[0].0, edits[0].1, edits[0].2), (8, 20, 8));
            let rev = edits[0].3;
            reply = ServerMessage::decode(&rx.try_recv().expect("an ack follows content"));
            assert!(matches!(reply, Some(ServerMessage::EditAck { accepted: true, rev: got, .. }) if got == rev));
        }
        match reply {
            Some(ServerMessage::EditAck { req, accepted, rev }) => (req, accepted, rev),
            other => panic!("expected an EditAck, got {other:?}"),
        }
    };

    // First break wins at revision 1.
    on_edit(&shared, None, chartless(), 1, 10, 8, 20, 8, 0, "air");
    assert_eq!(ack(&rx), (10, true, 1));

    // The racing loser expected revision 0 and is rejected — exactly one
    // reward, and its ack is the rollback signal.
    on_edit(&shared, None, chartless(), 1, 11, 8, 20, 8, 0, "air");
    assert_eq!(ack(&rx), (11, false, 1));

    // Building on the current revision succeeds.
    let rock = rock_spec();
    on_edit(&shared, None, chartless(), 1, 12, 8, 20, 8, 1, &rock);
    assert_eq!(ack(&rx), (12, true, 2));

    // Junk specs are rejected before touching the overlay or the pool.
    on_edit(&shared, None, chartless(), 1, 13, 8, 20, 8, 2, "banana:zzz");
    assert_eq!(ack(&rx), (13, false, 2));
    assert_eq!(shared.lock_recover().edits[&(8, 20, 8)].spec.as_ref(), rock.as_str());
}

/// Equivalent spec spellings collapse to ONE canonical pool entry, and a
/// spec no live cell references leaves the pool instead of leaking.
#[test]
fn spec_pool_canonicalizes_and_releases_dead_entries() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));

    // The same spec interned twice: one canonical entry.
    let rock = rock_spec();
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
    on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, &rock);
    {
        let state = shared.lock_recover();
        assert_eq!(state.spec_pool.len(), 1, "equivalent spellings share one entry");
        assert_eq!(
            state.edits[&(8, 20, 8)].spec.as_ref(),
            state.edits[&(8, 21, 8)].spec.as_ref()
        );
    }

    // Overwriting both cells strands the old spec: it must leave the pool.
    on_edit(&shared, None, chartless(), 1, 3, 8, 20, 8, 1, "air");
    on_edit(&shared, None, chartless(), 1, 4, 8, 21, 8, 1, "air");
    {
        let state = shared.lock_recover();
        assert_eq!(state.spec_pool.len(), 1, "only \"air\" remains interned");
        assert_eq!(state.spec_pool.cells(AIR), 2, "both cells name air");
    }
}

/// The pool counts blocks by id, so any ids exercise its cap.
#[test]
fn spec_pool_is_bounded_under_unique_mints() {
    let mut pool = SpecPool::default();
    let id = |i: usize| BlockId((1000 + i) as u16);
    for i in 0..MAX_SPEC_POOL {
        assert!(pool.take(id(i)), "slot {i} must take");
    }
    assert!(!pool.take(id(MAX_SPEC_POOL)), "cap must refuse a new block");
    assert!(pool.take(id(0)), "a block already named still resolves");
    pool.release(id(1));
    assert!(pool.take(id(MAX_SPEC_POOL + 1)), "release must free a slot");
    assert_eq!(pool.len(), MAX_SPEC_POOL);
}

#[test]
fn edit_at_exact_reach_is_accepted_and_extreme_coords_do_not_panic() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let center = DVec3::new(8.5, 20.5, 8.5);
    // A hair inside the sphere so f64 rounding cannot push the construction past
    // `>`; a hair outside must still miss.
    let at_reach = DVec3::new(center.x + EDIT_REACH * 0.999, center.y, center.z);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(at_reach, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, "air");
    assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)), "exact REACH must land");

    let just_out = DVec3::new(center.x + EDIT_REACH * 1.001, center.y, center.z);
    shared.lock_recover().players.get_mut(&1).unwrap().pos = just_out;
    on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, "air");
    assert!(!shared.lock_recover().edits.contains_key(&(8, 21, 8)));

    on_edit(&shared, None, chartless(), 1, 3, i32::MIN, i32::MIN, i32::MIN, 0, "air");
    on_edit(&shared, None, chartless(), 1, 4, i32::MAX, i32::MAX, i32::MAX, 0, "air");
    let far = crate::math::WORLD_BORDER as i32 + 64;
    on_edit(&shared, None, chartless(), 1, 5, far, 20, far, 0, "air");
    assert!(!shared.lock_recover().edits.contains_key(&(far, 20, far)));
    let _ = rx;
}

/// A hook Deny is the same `EditAck { accepted: false }` a lost race sends:
/// exactly one reject, no ledger write, no broadcast to peers.
#[test]
fn hook_denied_edit_is_one_reject_and_no_broadcast() {
    let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out1, test_kick()));
    players.insert(2u32, test_player(DVec3::new(10.5, 20.0, 8.5), out2, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let (mut rec, log) = hooks::Recording::new("deny");
    rec.deny_edit = true;
    let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

    on_edit(&shared, Some(&table), chartless(), 1, 42, 8, 20, 8, 0, "air");

    let to_editor = drain_msgs(&rx1);
    assert_eq!(to_editor.len(), 1, "exactly one ack");
    match &to_editor[0] {
        ServerMessage::EditAck { req, accepted, rev } => {
            assert_eq!((*req, *accepted, *rev), (42, false, 0));
        }
        other => panic!("expected EditAck reject, got {other:?}"),
    }
    assert!(drain_msgs(&rx2).is_empty(), "denied edit must not broadcast");
    assert!(shared.lock_recover().edits.is_empty(), "ledger untouched");
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn panicking_edit_hook_is_neutralised_and_the_edit_commits() {
    let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out1, test_kick()));
    players.insert(2u32, test_player(DVec3::new(10.5, 20.0, 8.5), out2, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let (mut rec, _) = hooks::Recording::new("boom");
    rec.panic_edit = true;
    let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

    on_edit(&shared, Some(&table), chartless(), 1, 1, 8, 20, 8, 0, "air");

    match &drain_msgs(&rx1)[..] {
        [ServerMessage::Snapshot { edits }, ServerMessage::EditAck { req, accepted, rev }] => {
            assert_eq!(edits.len(), 1);
            assert_eq!((edits[0].0, edits[0].1, edits[0].2, edits[0].3, edits[0].4.as_ref()), (8, 20, 8, 1, "air"));
            assert_eq!((*req, *accepted, *rev), (1, true, 1));
        }
        other => panic!("expected one accepted ack, got {other:?}"),
    }
    match &drain_msgs(&rx2)[..] {
        [ServerMessage::Edit { x, y, z, rev, .. }] => {
            assert_eq!((*x, *y, *z, *rev), (8, 20, 8, 1));
        }
        other => panic!("expected one broadcast Edit, got {other:?}"),
    }
    assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)));
}

#[test]
fn six_hundred_reaction_mutations_reach_the_client_in_order() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let (block, spec) = {
        let mut state = shared.lock_recover();
        let block = state.registry.parse_spec(&rock_spec()).unwrap();
        (block, state.intern(block).expect("spec pool"))
    };
    let mutations: Vec<Mutation> = (0..600)
        .map(|i| Mutation {
            pos: (i, 20, 0),
            from: AIR,
            to: AIR,
        })
        .collect();
    {
        let mut state = shared.lock_recover();
        for m in &mutations {
            state.edits.insert(
                m.pos,
                Cell {
                    block,
                    spec: spec.clone(),
                    rev: (m.pos.0 as u32) + 1,
                    natural: false,
                },
            );
        }
        send_reaction_mutations(&mut state, &mutations);
    }
    let mut got = Vec::new();
    for msg in drain_msgs(&rx) {
        match msg {
            ServerMessage::Snapshot { edits } => got.extend(edits),
            other => panic!("expected Snapshot batches, got {other:?}"),
        }
    }
    assert_eq!(got.len(), 600, "every committed mutation must reach the client");
    for (i, (x, y, z, rev, s)) in got.into_iter().enumerate() {
        assert_eq!((x, y, z), (i as i32, 20, 0));
        assert_eq!(rev, i as u32 + 1);
        assert_eq!(&*s, &*spec);
    }
}

#[test]
fn cached_terrain_reads_match_the_generator() {
    let state = test_state(HashMap::new());
    let generator = &lax_ctx().generator;
    let cells: Vec<Pos> = (0..3 * TERRAIN_SLOTS as i32).map(|i| (i % 61 - 30, i / 61 % 50 - 10, i / 3050)).collect();
    for _ in 0..2 {
        for &(x, y, z) in &cells {
            assert_eq!(server_block(&state, generator, (x, y, z)), generator.voxel_at(x, y, z));
        }
    }
}

#[test]
fn server_block_reads_the_generator_like_the_client() {
    use crate::space::atlas::Patch;
    let mut registry = BlockRegistry::with_builtins();
    let g = crate::world::terrain::generator(&mut registry, 4242, Default::default());
    let home = g.cosmos().expect("cosmos").home();
    let atlas = g
        .atlases()
        .iter()
        .find(|a| (a.centre - home.centre_f()).length() < 1.0)
        .expect("the start world is charted");
    let n = atlas.bands[0].n;
    let s = atlas.storage(Patch::Shell { band: 0, face: Face::PosY }, [n / 2, 0, n / 2]);
    let (x0, z0) = (s[0] as i32, s[2] as i32);
    for x in x0..x0 + 40 {
        for z in z0..z0 + 40 {
            let h = g.height(x, z);
            assert_ne!(h, i32::MIN, "({x},{z}) has no chart surface");
            assert_eq!(g.block_at(x, h - 1, z, h), g.voxel_at(x, h - 1, z), "({x},{},{z})", h - 1);
        }
    }
}

#[test]
fn full_configuration_spec_fits_and_is_accepted() {
    let elems: Vec<_> = (0..material::CAPACITY).map(|i| material::Element::new([i as u8, 1, 2, 3])).collect();
    let cfg = material::Configuration::new(elems).unwrap();
    let spec = {
        let mut registry = BlockRegistry::with_builtins();
        let id = registry.intern(&cfg).unwrap();
        registry.spec(id)
    };
    assert_eq!(spec.len(), MAX_SPEC);
    assert!(spec.len() > 256, "the old cap rejected this block");
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &spec);
    match ServerMessage::decode(&rx.try_recv().unwrap()) {
        Some(ServerMessage::Snapshot { edits }) => assert_eq!(edits, vec![(8, 20, 8, 1, Arc::from(spec.as_str()))]),
        other => panic!("authoritative content precedes its ack, got {other:?}"),
    }
    match ServerMessage::decode(&rx.try_recv().unwrap()) {
        Some(ServerMessage::EditAck { accepted: true, .. }) => {}
        other => panic!("a full configuration must be accepted, got {other:?}"),
    }
    assert!(shared.lock_recover().edits.contains_key(&(8, 20, 8)));
}

#[test]
fn spec_pool_counts_cells_not_arc_clones() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let rock = rock_spec();
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &rock);
    on_edit(&shared, None, chartless(), 1, 2, 8, 21, 8, 0, &rock);
    let extra = {
        let state = shared.lock_recover();
        assert_eq!(state.spec_pool.cells(state.registry.lookup_spec(&rock).unwrap()), 2);
        state.edits[&(8, 21, 8)].spec.clone()
    };
    on_edit(&shared, None, chartless(), 1, 3, 8, 20, 8, 1, "air");
    let rock_id = shared.lock_recover().registry.lookup_spec(&rock).unwrap();
    assert_eq!(shared.lock_recover().spec_pool.cells(rock_id), 1);
    on_edit(&shared, None, chartless(), 1, 4, 8, 21, 8, 1, "air");
    assert_eq!(shared.lock_recover().spec_pool.cells(rock_id), 0, "zero cells frees the entry");
    assert_eq!(shared.lock_recover().spec_pool.len(), 1, "only air is named");
    drop(extra);
}

#[test]
fn stale_novel_spec_does_not_grow_the_registry_and_quota_holds() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(8);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let before = shared.lock_recover().registry.block_count();
    on_edit(&shared, None, chartless(), 1, 1, 8, 20, 8, 0, &numbered_spec(1));
    assert_eq!(shared.lock_recover().registry.block_count(), before + 1);
    on_edit(&shared, None, chartless(), 1, 2, 8, 20, 8, 0, &numbered_spec(2));
    assert_eq!(shared.lock_recover().registry.block_count(), before + 1, "stale expect must not intern");
    assert_eq!(shared.lock_recover().players[&1].novel, 1);
    let _ = rx;

    let (out, rx) = sync_channel::<Arc<[u8]>>(NOVEL_SPEC_QUOTA as usize + 8);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(DVec3::new(8.5, 20.0, 8.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let before = shared.lock_recover().registry.block_count();
    let mut expect = 0u32;
    for i in 0..NOVEL_SPEC_QUOTA {
        on_edit(&shared, None, chartless(), 1, i + 1, 8, 20, 8, expect, &numbered_spec(i as u8));
        expect += 1;
    }
    assert_eq!(shared.lock_recover().registry.block_count(), before + NOVEL_SPEC_QUOTA as usize);
    assert_eq!(shared.lock_recover().players[&1].novel, NOVEL_SPEC_QUOTA);
    let count = shared.lock_recover().registry.block_count();
    on_edit(&shared, None, chartless(), 1, 1000, 8, 20, 8, expect, &numbered_spec(250));
    assert_eq!(shared.lock_recover().registry.block_count(), count, "past the quota: no intern");
    on_edit(&shared, None, chartless(), 1, 1001, 8, 20, 8, expect, "air");
    assert!(shared.lock_recover().edits[&(8, 20, 8)].spec.as_ref() == "air");
    assert_eq!(shared.lock_recover().players[&1].novel, NOVEL_SPEC_QUOTA, "air is not novel");
    let _ = rx;
}

#[test]
fn edits_over_the_rate_budget_are_rejected() {
    use crate::net::client::Connection;
    let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
    let s = conn.spawn();
    let (x, y, z) = (
        crate::math::block_coord(s.x),
        crate::math::block_coord(s.y),
        crate::math::block_coord(s.z),
    );
    for _ in 0..(SWING_RATE + 5) {
        conn.send_swing();
    }
    // Swings have their own budget, so the first edit is still accepted.
    let kept = conn.send_edit(x, y, z, "air".into()).expect("air is sent");
    let mut over = kept;
    for i in 0..EDIT_RATE {
        over = conn.send_edit(x, y + 1 + i as i32, z, "air".into()).expect("air is sent");
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut accepted = false;
    let mut rejected = false;
    while !(accepted && rejected) && Instant::now() < deadline {
        for event in conn.poll() {
            match event {
                crate::net::client::Incoming::EditAccepted { req } if req == kept => accepted = true,
                crate::net::client::Incoming::EditRejected { req, restore: true } if req == over => rejected = true,
                _ => {}
            }
        }
        if !(accepted && rejected) {
            thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(accepted, "swings must not spend the edit budget");
    assert!(rejected, "an edit past the edit budget must be acked rejected");
    handle.stop();
}

#[test]
fn a_panicking_reaction_tick_keeps_the_server_running() {
    use crate::net::client::Connection;
    let handle = flat(Config::default());
    {
        let mut state = handle.state.lock_recover();
        state.panic_tick = true;
        state.reactions.wake_cell((1, 2, 3));
        assert_eq!(state.reactions.pending(), 6);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut restored = false;
    while Instant::now() < deadline {
        let state = handle.state.lock_recover();
        if !state.panic_tick && state.reactions.pending() == 6 {
            restored = true;
            break;
        }
        drop(state);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(restored, "a panicking tick must put the scheduler back with its contacts");
    thread::sleep(Duration::from_millis(200));
    assert_eq!(
        handle.state.lock_recover().reactions.pending(),
        0,
        "the following tick still runs"
    );
    let conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").expect("still accepting");
    assert!(conn.is_alive());
    drop(conn);
    handle.stop();
}

/// Every ledger cell names its block with the registry's own text, after edits, tool uses and
/// the reactions they wake, and the pool counts exactly the live cells.
#[test]
fn cells_name_their_block_after_edits_tool_uses_and_reactions() {
    let (players, _rx) = pose(DVec3::new(8.5, 20.0, 8.5));
    let (shared, ctx) = flat_shared(players, NoclipPolicy::All, &[]);
    let (_, tool) = place_pair(&shared, (8, 20, 8));
    on_edit(&shared, None, &ctx.generator, 1, 1, 8, 21, 8, 0, &tool);
    on_edit(&shared, None, &ctx.generator, 1, 2, 9, 20, 8, 0, &rock_spec());
    on_edit(&shared, None, &ctx.generator, 1, 3, 10, 20, 8, 0, "air");
    place_pair(&shared, (7, 20, 8));
    on_tool_use(&shared, &ctx.generator, 1, 4, 7, 20, 8, 1, &Arc::from(tool.as_str()));
    for _ in 0..40 {
        run_reactions(&shared, &ctx);
    }
    let state = shared.lock_recover();
    assert!(state.edits[&(7, 20, 8)].rev > 1, "the tool use wrote its cell");
    assert!(state.edits[&(8, 20, 8)].rev > 1 || state.edits[&(8, 21, 8)].rev > 1, "the pair reacted");
    for (at, cell) in &state.edits {
        assert_eq!(state.registry.lookup_spec(&cell.spec), Some(cell.block), "{at:?}");
        assert!(Arc::ptr_eq(&cell.spec, state.registry.spec_ref(cell.block)), "{at:?} holds the registry's text");
    }
    let named: HashSet<BlockId> = state.edits.values().map(|c| c.block).collect();
    assert_eq!(state.spec_pool.len(), named.len());
    for block in named {
        let cells = state.edits.values().filter(|c| c.block == block).count();
        assert_eq!(state.spec_pool.cells(block) as usize, cells, "{block:?}");
    }
}

/// A generator that counts its reads, and the reads made while the state lock was held.
struct Watched {
    inner: crate::world::terrain::Generator,
    state: OnceLock<std::sync::Weak<Mutex<State>>>,
    reads: AtomicUsize,
    locked: AtomicUsize,
}

impl TerrainGenerator for Watched {
    fn height(&self, x: i32, z: i32) -> i32 {
        self.inner.height(x, z)
    }

    fn surface_at(&self, x: i32, z: i32) -> BlockId {
        self.inner.surface_at(x, z)
    }

    fn deep(&self) -> BlockId {
        self.inner.deep()
    }

    fn voxel_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.state.get().and_then(std::sync::Weak::upgrade).is_some_and(|s| s.try_lock().is_err()) {
            self.locked.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.voxel_at(x, y, z)
    }
}

/// A refused tool use names the cell as it is, but an edited cell is answered from the ledger
/// and a generated one is read with the state lock released.
#[test]
fn a_refused_tool_use_reads_the_generator_only_outside_the_lock() {
    let (players, rx) = pose(DVec3::new(8.5, 20.0, 8.5));
    let (shared, ctx) = flat_shared(players, NoclipPolicy::All, &[]);
    let watched = Arc::new(Watched {
        inner: ctx.generator.clone(),
        state: OnceLock::new(),
        reads: AtomicUsize::new(0),
        locked: AtomicUsize::new(0),
    });
    let _ = watched.state.set(Arc::downgrade(&shared));
    let generator: crate::world::terrain::Generator = watched.clone();
    let tool: Arc<str> = rock_spec().into();
    on_edit(&shared, None, &generator, 1, 1, 8, 21, 8, 0, "air");
    let _ = drain(&rx);
    watched.reads.store(0, Ordering::Relaxed);

    // A stale revision, then a use over the tool budget, on the edited cell.
    on_tool_use(&shared, &generator, 1, 2, 8, 21, 8, 0, &tool);
    refuse_tool(shared.lock_recover(), &shared, &generator, 1, 3, (8, 21, 8), &tool);
    assert_eq!(watched.reads.load(Ordering::Relaxed), 0, "an edited cell is answered from the ledger");
    // The same on generated cells nobody has read yet.
    on_tool_use(&shared, &generator, 1, 4, 8, 19, 8, 5, &tool);
    refuse_tool(shared.lock_recover(), &shared, &generator, 1, 5, (9, 19, 8), &tool);
    assert_eq!(watched.reads.load(Ordering::Relaxed), 2, "each generated cell is read once");
    assert_eq!(watched.locked.load(Ordering::Relaxed), 0, "never under the state lock");

    let generated = {
        let state = shared.lock_recover();
        state.registry.spec(ctx.generator.voxel_at(8, 19, 8))
    };
    let answers: Vec<_> = drain(&rx)
        .into_iter()
        .map(|m| match m {
            ServerMessage::ToolResult { req, reacted, rev, cell_spec, tool_spec } => {
                assert!(!reacted && tool_spec == tool, "request {req} hands the tool back");
                (req, rev, cell_spec.to_string())
            }
            other => panic!("expected a ToolResult, got {other:?}"),
        })
        .collect();
    assert_eq!(
        answers,
        vec![(2, 1, "air".to_string()), (3, 1, "air".to_string()), (4, 0, generated.clone()), (5, 0, generated)]
    );
}
