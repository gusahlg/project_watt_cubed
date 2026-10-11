//! World files, shutdown and the policy files beside the world.
use super::super::*;
use super::support::*;
use crate::net::client::Incoming;

#[test]
fn restarted_server_serves_the_same_edits() {
    let path = crate::save::store::test_temp_path("restart");
    let _ = std::fs::remove_file(&path);
    let handle = spawn(0, Config {
        seed: 42,
        worldgen: WorldgenKind::Flat,
        world: Some(path.clone()),
        ops: vec!["ada".into()],
        ..Config::default()
    }).unwrap();
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
    let s = conn.spawn();
    // The grass under the spawn: breaking it changes the world, so the file keeps it.
    let (x, y, z) = (crate::math::block_coord(s.x), crate::world::generation::FLAT_HEIGHT - 1, crate::math::block_coord(s.z));
    let req = conn.send_edit(x, y, z, "air".into()).unwrap();
    let accepted = await_event(&mut conn, Duration::from_secs(3), |e| matches!(e, Incoming::EditAccepted { req: r } if r == req).then_some(()));
    assert!(accepted.is_some(), "the edit is committed before shutdown");
    conn.send_set_time(0.2);
    let day = await_event(&mut conn, Duration::from_secs(3), |e| match e {
        Incoming::Time { day, .. } => Some(day),
        _ => None,
    });
    assert!((day.expect("time reply") - 0.2).abs() < 0.02);
    drop(conn);
    handle.stop();

    let again = spawn(0, Config {
        seed: 99,
        worldgen: WorldgenKind::Diffusion,
        world: Some(path.clone()),
        warn_world_overrides: true,
        ..Config::default()
    }).unwrap();
    let mut bob = Connection::connect("127.0.0.1", again.addr().port(), "bob", "").unwrap();
    assert_eq!(bob.seed(), 42);
    assert_eq!(bob.worldgen(), WorldgenKind::Flat);
    let (mut saw_edit, mut saw_day) = (false, false);
    eventually(Duration::from_secs(3), || {
        for event in bob.poll() {
            match event {
                Incoming::Mutation { x: mx, y: my, z: mz, spec } if (mx, my, mz) == (x, y, z) && spec.as_ref() == "air" => {
                    saw_edit = true;
                }
                Incoming::Time { day, .. } if (day - 0.2).abs() < 0.05 => saw_day = true,
                _ => {}
            }
        }
        saw_edit && saw_day
    });
    assert!(saw_edit, "the restarted world still has the edit");
    assert!(saw_day, "the restarted world still has the clock");
    again.stop();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn stop_closes_connections_and_frees_the_port() {
    let handle = flat(Config::default());
    let port = handle.addr().port();
    let mut conn = Connection::connect("127.0.0.1", port, "ada", "").unwrap();
    handle.stop();
    let reason = await_event(&mut conn, Duration::from_secs(3), |e| match e {
        Incoming::Disconnected { reason } => Some(reason),
        _ => None,
    });
    let reason = reason.expect("the close arrives before the idle timeout");
    assert!(
        reason.to_ascii_lowercase().contains("shutting down"),
        "client saw {reason:?}"
    );
    drop(conn);
    let mut rebound = None;
    for _ in 0..50 {
        match spawn(port, Config { seed: 1, worldgen: WorldgenKind::Flat, ..Config::default() }) {
            Ok(handle) => {
                rebound = Some(handle);
                break;
            }
            Err(_) => thread::sleep(Duration::from_millis(20)),
        }
    }
    rebound.expect("the port accepts a new server").stop();
}

/// Sends SIGTERM to this test process: tokio's handler, installed by `run`, takes it.
#[cfg(unix)]
#[test]
fn sigterm_saves_the_world() {
    let path = crate::save::store::test_temp_path("sigterm");
    let _ = std::fs::remove_file(&path);
    let port = std::net::UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let server_path = path.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let server = thread::spawn(move || {
        let result = run(port, Config {
            seed: 42,
            worldgen: WorldgenKind::Flat,
            world: Some(server_path),
            ..Config::default()
        });
        let _ = tx.send(result);
    });
    let mut conn = None;
    eventually(Duration::from_secs(8), || {
        if let Ok(result) = rx.try_recv() {
            panic!("server exited before a client connected: {result:?}");
        }
        conn = Connection::connect("127.0.0.1", port, "ada", "").ok();
        conn.is_some()
    });
    let _conn = conn.expect("server accepted a connection");
    assert!(!path.exists(), "nothing is written until a save");
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status();
    match rx.recv_timeout(Duration::from_secs(8)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("server failed: {e}"),
        Err(_) => panic!("SIGTERM did not stop the server"),
    }
    server.join().unwrap();
    let bytes = std::fs::read(&path).expect("SIGTERM saved the world");
    let doc = match crate::save::format::decode(&bytes).unwrap() {
        crate::save::format::Decoded::Intact(doc) => doc,
        crate::save::format::Decoded::Salvaged { .. } => panic!("shutdown save must be intact"),
    };
    assert_eq!(doc.meta.seed, 42);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn world_policy_unions_flags_with_side_files() {
    let dir = crate::save::store::test_temp_path("policy");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("ops.txt"), "cara\n").unwrap();
    std::fs::write(
        dir.join("mods.toml"),
        "allow = [\"pwc.hotbar\"]\ndeny = [\"pwc.dev-toolkit\"]\n",
    ).unwrap();
    let mut config = Config {
        world: Some(dir.join("world.save")),
        ops: vec!["Ada".into()],
        mods_deny: vec!["pwc.other".into()],
        ..Config::default()
    };
    load_world_policy(&mut config).unwrap();
    assert!(config.ops.iter().any(|n| n.eq_ignore_ascii_case("ada")));
    assert!(config.ops.iter().any(|n| n.eq_ignore_ascii_case("cara")));
    assert_eq!(config.mods_allow, vec!["pwc.hotbar".to_string()]);
    assert!(config.mods_deny.iter().any(|id| id == "pwc.dev-toolkit"));
    assert!(config.mods_deny.iter().any(|id| id == "pwc.other"));

    let bad = crate::save::store::test_temp_path("policy-bad");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("mods.toml"), "allow = 1\n").unwrap();
    let mut broken = Config { world: Some(bad.join("world.save")), ..Config::default() };
    assert!(load_world_policy(&mut broken).is_err());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&bad);
}

#[test]
fn save_world_exports_the_live_scheduler() {
    let path = crate::save::store::test_temp_path("g25-pending");
    let _ = std::fs::remove_file(&path);
    let flags = persist::Flags {
        seed: 7,
        worldgen: WorldgenKind::Flat,
        terrain: TerrainCfg::default(),
        warn: false,
    };
    let loaded = persist::load(&path, &flags).unwrap();
    let (shared, mut ctx) = flat_shared(HashMap::new(), NoclipPolicy::All, &[]);
    ctx.store = Some(Arc::new(loaded.store.expect("a new file has a store")));
    ctx.seed = 7;
    shared.lock_recover().reactions.wake_cell((4, 5, 6));
    assert_eq!(shared.lock_recover().reactions.pending(), 6);
    save_world(&*shared, &ctx, &Mutex::new(()));
    let again = persist::load(&path, &flags).unwrap();
    assert_eq!(again.pending.len(), 6);
    let mut sched = ReactionScheduler::new();
    sched.restore(&contacts_of(&again.pending));
    assert_eq!(sched.pending(), 6);
    drop(ctx);
    let _ = std::fs::remove_file(&path);
    let mut bak = path.file_name().unwrap().to_os_string();
    bak.push(".bak");
    let _ = std::fs::remove_file(path.with_file_name(bak));
}

/// The final save waits until every connection has been told to go, so no edit can be
/// acknowledged after the save has read the ledger.
#[test]
fn stop_closes_connections_before_the_final_save() {
    let path = crate::save::store::test_temp_path("stop-order");
    let handle = flat(Config { world: Some(path.clone()), ..Config::default() });
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
    let gate = handle.save_gate.lock_recover();
    thread::scope(|scope| {
        let stopping = scope.spawn(|| handle.stop());
        let closed = await_event(&mut conn, Duration::from_secs(3), |e| matches!(e, Incoming::Disconnected { .. }).then_some(())).is_some();
        assert!(!path.exists(), "the save is still waiting");
        drop(gate);
        stopping.join().unwrap();
        assert!(closed, "connections close before the final save");
    });
    assert!(path.exists(), "stop still saves");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn edits_past_the_spec_pool_stay_in_the_file() {
    let mut state = test_state(HashMap::new());
    // Other blocks fill the pool: it counts ids, so ids past the registry stand in for them.
    for i in 0..MAX_SPEC_POOL {
        assert!(state.spec_pool.take(BlockId((40_000 + i) as u16)));
    }
    let rock = rock_spec();
    let kept = install_edits(&mut state, &[(1, 2, 3, rock.clone()), (4, 5, 6, "air".into())]);
    assert_eq!(kept, vec![(1, 2, 3, rock), (4, 5, 6, "air".to_string())]);
    assert!(state.edits.is_empty());
}

/// An edit that puts back the generated block keeps its revision in memory but is not saved.
#[test]
fn a_no_op_edit_stays_out_of_the_world_file() {
    let path = crate::save::store::test_temp_path("no-op");
    let flags = persist::Flags { seed: 1, worldgen: WorldgenKind::Flat, terrain: TerrainCfg::default(), warn: false };
    let ground = crate::world::generation::FLAT_HEIGHT;
    let (players, _rx) = pose(DVec3::new(0.5, f64::from(ground) + 2.0, 0.5));
    let (shared, mut ctx) = flat_shared(players, NoclipPolicy::All, &[]);
    ctx.store = Some(Arc::new(persist::load(&path, &flags).unwrap().store.unwrap()));
    on_edit(&shared, None, &ctx.generator, 1, 1, 0, ground + 1, 0, 0, "air");
    on_edit(&shared, None, &ctx.generator, 1, 2, 1, ground - 1, 0, 0, "air");
    assert_eq!(shared.lock_recover().edits.len(), 2, "both edits hold a revision");
    save_world(&shared, &ctx, &Mutex::new(()));
    let again = persist::load(&path, &flags).unwrap();
    assert_eq!(again.edits, vec![(1, ground - 1, 0, "air".to_string())]);
    let _ = std::fs::remove_file(&path);
}
