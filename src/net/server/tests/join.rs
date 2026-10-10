//! Handshakes, admission, bootstrap, names, mod policy and operators.
use super::super::*;
use super::support::*;
use crate::net::client::Incoming;

#[test]
fn names_are_capped_and_sanitised() {
    assert_eq!(&*clean_name("  guahlg\n "), "guahlg");
    assert_eq!(&*clean_name(""), "player");
    assert_eq!(clean_name(&"x".repeat(100)).len(), MAX_NAME);
}

#[test]
fn chat_is_sanitised() {
    assert_eq!(&*clean_chat("hi\tthere\n"), "hithere");
    assert_eq!(clean_chat(&"a".repeat(500)).len(), MAX_CHAT);
}

#[test]
fn spawn_points_sit_above_the_surface() {
    use crate::space::atlas::Patch;
    let terrain = test_generator();
    let centre = terrain.chart_spawn().expect("a charted start world");
    for id in 0..25 {
        let p = spawn_point(terrain.as_ref(), id);
        assert!((p.x - centre.x).abs() <= 2.0 + 1e-6, "id {id} x {}", p.x);
        assert!((p.z - centre.z).abs() <= 2.0 + 1e-6, "id {id} z {}", p.z);
        assert!((p.y - centre.y).abs() < 1e-6, "id {id} y {}", p.y);
    }
    let p = spawn_point(terrain.as_ref(), 12);
    assert!((p.x - centre.x).abs() < 1e-9 && (p.z - centre.z).abs() < 1e-9, "id 12 is the centre");
    let cell = terrain.atlases().iter().find_map(|a| a.storage_of(p)).expect("spawn storage");
    let (patch, local) = terrain.atlases().iter().find_map(|a| a.locate(cell)).expect("located");
    assert!(matches!(patch, Patch::Shell { band: 0, face: Face::PosY }), "{patch:?}");
    let open = i64::from(terrain.height(cell[0] as i32, cell[2] as i32));
    assert!((1..=2).contains(&(local[1] - open)), "local {} open {open}", local[1]);
}

#[test]
fn player_ids_never_use_the_reserved_world_id() {
    assert_eq!(WORLD_PLAYER, 0);
    let state = test_state(HashMap::new());
    assert!(state.next_id > WORLD_PLAYER);
}

/// A late joiner reads the CURRENT phase, not the last set value.
#[test]
fn shared_clock_advances_between_set_and_join() {
    let mut state = test_state(HashMap::new());
    state.day = 0.25;
    state.day_set = Instant::now() - Duration::from_secs(300);
    let now = state.day_now(600.0);
    assert!((now - 0.75).abs() < 0.01, "half a 600s cycle after 0.25, got {now}");
}

/// Same protocol, different generated content — the handshake must refuse
/// the join instead of letting two builds silently diverge on one seed.
#[test]
fn mismatched_content_fingerprint_is_rejected() {
    let handle = spawn(0, Config { password: String::new(), seed: 3, ..Config::default() }).unwrap();
    let id = server_content();
    let reason = reject_reason(
        handle.addr(),
        &hello("drifted", "", PROTOCOL_VERSION, crate::net::ContentId { palette: id.palette ^ 1, ..id }),
    );
    assert!(reason.contains("content"), "unexpected reason: {reason}");
    handle.stop();
}

/// Kind is not part of the content id. A client whose local mods would build Flat
/// still joins a Diffusion server and adopts the kind Welcome carries.
/// (Re-pinned: this used to reject `content_fingerprint_kind(Flat)`.)
#[test]
fn flat_client_joins_a_diffusion_server() {
    let handle = spawn(
        0,
        Config {
            password: String::new(),
            seed: 3,
            worldgen: WorldgenKind::Diffusion,
            ..Config::default()
        },
    )
    .unwrap();
    match raw_reply(handle.addr(), &hello("flat", "", PROTOCOL_VERSION, server_content())) {
        ServerMessage::Welcome { worldgen, .. } => assert_eq!(worldgen, WorldgenKind::Diffusion),
        other => panic!("a code-only fingerprint must be welcomed, got {other:?}"),
    }
    handle.stop();
}

#[test]
fn welcome_carries_the_servers_worldgen_kind_and_cfg() {
    let terrain = TerrainCfg { relief: 175, caves: 50, mines: 0, space: 125, ..Default::default() };
    let handle = spawn(
        0,
        Config {
            password: String::new(),
            seed: 11,
            worldgen: WorldgenKind::Diffusion,
            terrain,
            ..Config::default()
        },
    )
    .unwrap();
    let hello = hello("guest", "", PROTOCOL_VERSION, server_content());
    match raw_reply(handle.addr(), &hello) {
        ServerMessage::Welcome { worldgen, terrain: got, seed, .. } => {
            assert_eq!(seed, 11);
            assert_eq!(worldgen, WorldgenKind::Diffusion);
            assert_eq!(got, terrain);
        }
        other => panic!("expected Welcome with diffusion kind, got {other:?}"),
    }
    handle.stop();
}

/// Silent (never-authenticating) connections must be bounded by
/// [`HANDSHAKE_CAP`]: one past the cap is refused promptly instead of
/// squatting a thread until the handshake timeout, and releasing squatters
/// frees slots for a real join.
#[test]
fn silent_connections_beyond_the_handshake_cap_are_refused() {
    let handle = spawn(0, Config { password: String::new(), seed: 1, ..Config::default() }).unwrap();
    let addr = handle.addr();

    // Fill every pre-auth slot with connections that complete the QUIC
    // handshake but never open their stream — the server's `accept_bi` blocks,
    // holding the slot exactly as a silent TCP client did.
    let target = SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()));
    let (squat_rt, squat_ep) = client_endpoint();
    let squatters: Vec<quinn::Connection> = squat_rt.block_on(async {
        let mut v = Vec::new();
        for _ in 0..HANDSHAKE_CAP {
            v.push(squat_ep.connect(target, "watt").unwrap().await.unwrap());
        }
        v
    });
    // Let the accept loop take them all in before probing past the cap.
    thread::sleep(Duration::from_millis(300));

    // One more must be turned away promptly (a QUIC refusal), not left squatting.
    assert!(
        Connection::connect("127.0.0.1", addr.port(), "extra", "").is_err(),
        "a connection past the handshake cap must be refused"
    );

    // Freeing the squatters must free their slots for a real player. Reuse
    // the graceful-shutdown helper (these are raw quinn connections, not
    // our `Connection` wrapper) — merely dropping the handles would strand
    // the server-side handlers until HANDSHAKE_TIMEOUT, racing this
    // assertion's 5s against a 10s timeout.
    for conn in &squatters {
        crate::net::client::graceful_close(conn, &squat_ep, &squat_rt);
    }
    drop(squatters);
    drop(squat_ep);
    drop(squat_rt);
    let mut last = None;
    let joined = eventually(Duration::from_secs(5), || match Connection::connect("127.0.0.1", addr.port(), "late", "") {
        Ok(_) => true,
        Err(e) => {
            last = Some(e);
            false
        }
    });
    assert!(joined, "slots never freed after squatters left: {last:?}");

    handle.stop();
}

/// Join, wander across bucket borders, leave — repeatedly. The grid must
/// always hold exactly one entry per connected player and drain to zero
/// buckets when everyone is gone: no leaked ids, no leaked keys.
#[test]
fn grid_never_leaks_entries_under_churn() {
    let handle = spawn(0, Config { password: String::new(), seed: 7, ..Config::default() }).unwrap();
    let port = handle.addr().port();

    for round in 0..3 {
        let mut a = Connection::connect("127.0.0.1", port, "a", "").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "b", "").unwrap();
        assert!(
            eventually(Duration::from_secs(2), || handle.grid_entries() == 2),
            "round {round}: both joins should land in the grid"
        );

        // March both across several bucket borders. Teleports rather than
        // moves: a 200-unit hop is beyond the movement envelope, and the
        // grid must follow ACCEPTED discontinuities just as it follows walks.
        for step in 1..=3 {
            thread::sleep(Duration::from_millis(40));
            let d = (step * 200) as f64; // 200 > INTEREST_RADIUS: a new bucket each step
            a.send_teleport(DVec3::new(d, 30.0, 0.0));
            b.send_teleport(DVec3::new(-d, 30.0, -d));
        }
        thread::sleep(Duration::from_millis(150));
        assert_eq!(
            handle.grid_entries(),
            2,
            "round {round}: moving must never grow or shrink membership"
        );

        drop(a);
        drop(b);
        assert!(
            eventually(Duration::from_secs(2), || handle.grid_entries() == 0 && handle.grid_buckets() == 0),
            "round {round}: grid must drain to zero entries and zero buckets, got {} entries in {} buckets",
            handle.grid_entries(),
            handle.grid_buckets()
        );
    }

    handle.stop();
}

#[test]
fn refused_joins_release_the_pre_auth_slot() {
    let handle = spawn(0, Config { password: "pw".into(), seed: 1, ..Config::default() }).unwrap();
    let addr = handle.addr();
    let id = server_content();
    let reason = reject_reason(addr, &hello("eve", "nope", PROTOCOL_VERSION, id));
    assert!(reason.to_lowercase().contains("password"), "{reason}");
    let reason = reject_reason(
        addr,
        &hello("eve", "pw", PROTOCOL_VERSION.wrapping_add(1), id),
    );
    assert!(reason.to_lowercase().contains("protocol"), "{reason}");
    assert!(reason.contains(&format!("server v{PROTOCOL_VERSION}")), "{reason}");
    assert!(reason.contains(&format!("client v{}", PROTOCOL_VERSION.wrapping_add(1))), "{reason}");
    let drifted = crate::net::ContentId { law: id.law ^ 1, ..id };
    let reason = reject_reason(addr, &hello("eve", "pw", PROTOCOL_VERSION, drifted));
    assert!(reason.contains("content"), "{reason}");
    assert!(eventually(Duration::from_secs(5), || handle.handshake_slots() == 0), "refusals must release the pre-auth slot");
    Connection::connect("127.0.0.1", addr.port(), "late", "pw").expect("refusals must free the slot");
    handle.stop();
}

#[test]
fn day_secs_clamps_zero_negative_and_huge() {
    assert_eq!(clamp_day_secs(0.0), 10.0);
    assert_eq!(clamp_day_secs(-40.0), 10.0);
    assert_eq!(clamp_day_secs(f32::NAN), 600.0);
    assert_eq!(clamp_day_secs(f32::INFINITY), 86_400.0);
    assert_eq!(clamp_day_secs(1.0e20), 86_400.0);
    assert_eq!(clamp_day_secs(600.0), 600.0);

    let mut state = test_state(HashMap::new());
    state.day = 0.0;
    state.day_set = Instant::now() - Duration::from_secs(10);
    let zero = state.day_now(0.0);
    let neg = state.day_now(-5.0);
    assert!((zero - 1.0).abs() < 0.05 || (zero - 0.0).abs() < 0.05, "10s of a 10s day wraps, got {zero}");
    assert!((zero - neg).abs() < 1e-3, "zero and negative share the clamp");
    let huge = state.day_now(f32::MAX);
    assert!(huge.abs() < 0.01, "a huge cycle barely advances in 10s, got {huge}");
}

#[test]
fn join_leave_hooks_fire_in_order() {
    use crate::net::hooks::Recorded;

    let (rec, log) = hooks::Recording::new("rec");
    let handle = spawn(
        0,
        Config { seed: 1, hooks: vec![Box::new(rec)], ..Config::default() },
    )
    .unwrap();
    let port = handle.addr().port();
    let a = Connection::connect("127.0.0.1", port, "alice", "").unwrap();
    let b = Connection::connect("127.0.0.1", port, "bob", "").unwrap();
    let wait = |n: usize| {
        for _ in 0..80 {
            if log.lock().unwrap().len() >= n {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        panic!("timed out waiting for {n} hook events, have {:?}", log.lock().unwrap());
    };
    wait(2);
    drop(a);
    wait(3);
    drop(b);
    wait(4);
    handle.stop();

    let events = log.lock().unwrap().clone();
    let names: Vec<_> = events
        .iter()
        .map(|e| match e {
            Recorded::Join(f) => format!("join {} {}", f.player, f.name),
            Recorded::Leave(f) => format!("leave {} {}", f.player, f.name),
            other => format!("other {other:?}"),
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "join 1 alice".to_string(),
            "join 2 bob".to_string(),
            "leave 1 alice".to_string(),
            "leave 2 bob".to_string(),
        ]
    );
}

/// A join overlay larger than one poll's budget reaches the game over several polls,
/// whole, and the loading hold lifts only after the last cell.
#[test]
fn a_big_overlay_is_handed_to_the_game_over_several_polls() {
    use crate::net::client::APPLY_BUDGET;
    let handle = flat(Config::default());
    let cells: Vec<_> = (0..(2 * APPLY_BUDGET + 100) as i32).map(|i| (i % 50, -5 - i / 2500, i / 50 % 50, "air".to_string())).collect();
    assert!(install_edits(&mut handle.state.lock_recover(), &cells).is_empty());
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
    let (mut got, mut polls) = (0, 0);
    eventually(Duration::from_secs(10), || {
        let n = conn.poll().iter().filter(|e| matches!(e, Incoming::Mutation { .. })).count();
        assert!(n <= APPLY_BUDGET, "{n} cells in one poll");
        got += n;
        polls += usize::from(n > 0);
        conn.snapshot_ready()
    });
    assert_eq!(got, cells.len(), "every cell, and only then the end of the overlay");
    assert!(polls >= 3);
    handle.stop();
}

#[test]
fn hello_rejections_name_the_protocol_before_a_full_decode() {
    let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
    let addr = handle.addr();
    let reason = reject_reason(addr, &[1, 0, 0, 0, 0]);
    assert!(reason.contains("expected hello"), "{reason}");
    let mut old = vec![0u8];
    old.extend_from_slice(&12u32.to_le_bytes());
    old.extend_from_slice(&[0u8; 8]);
    let reason = reject_reason(addr, &old);
    assert!(reason.contains(&format!("server v{PROTOCOL_VERSION}")), "{reason}");
    assert!(reason.contains("client v12"), "{reason}");
    let mut trunc = vec![0u8];
    trunc.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    trunc.extend_from_slice(&[0xff, 0x00]);
    let reason = reject_reason(addr, &trunc);
    assert!(reason.contains("malformed"), "{reason}");
    handle.stop();
}

/// The join overlay reaches the joiner sorted chunk by chunk, every cell once, each
/// with its own revision and spec.
#[test]
fn a_join_overlay_arrives_whole_and_grouped_by_chunk() {
    let mut state = test_state(HashMap::new());
    let rock = rock_spec();
    let cells: Vec<_> = (0..5_000i32)
        .map(|i| (i % 37 - 18, i / 37 % 9, i / 333 - 7, if i % 3 == 0 { rock.clone() } else { "air".to_string() }))
        .collect();
    assert!(install_edits(&mut state, &cells).is_empty());
    let overlay = Overlay::of(&state);
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let out = Outbox { tx: Some(out), writer: Arc::default() };
    let shared = Arc::new(Mutex::new(state));
    let ctx = lax_ctx();
    assert!(send_join(&out, &AtomicBool::new(false), &shared, ctx, 1, DVec3::ZERO, overlay, &[]));
    let mut got = Vec::new();
    let mut ended = false;
    for msg in drain(&rx) {
        match msg {
            ServerMessage::Snapshot { edits } => {
                assert!(!ended, "SnapshotEnd comes after every batch");
                got.extend(edits);
            }
            ServerMessage::SnapshotEnd => ended = true,
            _ => {}
        }
    }
    assert!(ended);
    assert_eq!(got.len(), cells.len());
    let chunk = |c: &(i32, i32, i32, u32, Arc<str>)| (c.0 >> 4, c.2 >> 4, c.1 >> 4);
    assert!(got.windows(2).all(|w| chunk(&w[0]) <= chunk(&w[1])), "grouped by chunk");
    let state = shared.lock_recover();
    for (x, y, z, rev, spec) in got {
        let cell = &state.edits[&(x, y, z)];
        assert_eq!((rev, spec.as_ref()), (cell.rev, cell.spec.as_ref()));
    }
}

#[test]
fn joiner_time_matches_the_clock_at_send() {
    let handle = spawn(0, Config { seed: 1, day_secs: 600.0, ..Config::default() }).unwrap();
    {
        let mut state = handle.state.lock_recover();
        state.day = 0.0;
        state.day_set = Instant::now() - Duration::from_secs(300);
    }
    let mut conn = Connection::connect("127.0.0.1", handle.addr().port(), "ada", "").unwrap();
    let day = await_event(&mut conn, Duration::from_secs(2), |e| match e {
        Incoming::Time { day, .. } => Some(day),
        _ => None,
    });
    let day = day.expect("Welcome is consumed at connect; Time follows it");
    assert!((day - 0.5).abs() < 0.05, "live clock, got {day}");
    handle.stop();
}

#[test]
fn joins_over_ipv6_loopback_and_localhost() {
    let handle = spawn(0, Config { seed: 1, ..Config::default() }).unwrap();
    let port = handle.addr().port();
    let v6 = Connection::connect("::1", port, "v6", "").expect("::1");
    assert!(v6.is_alive());
    let local = Connection::connect("localhost", port, "local", "").expect("localhost");
    assert!(local.is_alive());
    assert_ne!(v6.player_id(), local.player_id());
    handle.stop();
}

#[test]
fn duplicate_and_reserved_names_are_rejected() {
    let handle = flat(Config::default());
    let port = handle.addr().port();
    let _ada = Connection::connect("127.0.0.1", port, "Ada", "").unwrap();
    let dup = match Connection::connect("127.0.0.1", port, "ada", "") {
        Ok(_) => panic!("duplicate name was admitted"),
        Err(err) => err,
    };
    assert!(dup.to_string().contains("already in use"), "{dup}");
    let reserved = match Connection::connect("127.0.0.1", port, "Server", "") {
        Ok(_) => panic!("reserved name was admitted"),
        Err(err) => err,
    };
    assert!(reserved.to_string().contains("reserved"), "{reserved}");
    handle.stop();
}

#[test]
fn allow_list_admits_a_fully_listed_client() {
    let handle = flat(Config {
        mods_allow: vec!["pwc.hotbar".into()],
        ..Config::default()
    });
    let port = handle.addr().port();
    let listed = [("pwc.hotbar".into(), "0.1.0".into())];
    let ok = Connection::connect_with("127.0.0.1", port, "ada", "", &listed).expect("listed mod");
    assert!(ok.is_alive());
    drop(ok);
    let extra = [
        ("pwc.hotbar".into(), "0.1.0".into()),
        ("pwc.dev-toolkit".into(), "1.0.0".into()),
    ];
    let err = match Connection::connect_with("127.0.0.1", port, "bea", "", &extra) {
        Ok(_) => panic!("an unlisted mod was admitted"),
        Err(err) => err,
    };
    assert_eq!(err.mods_denied, vec!["pwc.dev-toolkit".to_string()]);
    let none = Connection::connect_with("127.0.0.1", port, "cy", "", &[]).expect("no mods enabled");
    assert!(none.is_alive());
    handle.stop();
}

#[test]
fn denied_mod_is_refused_then_admitted_when_off() {
    let handle = flat(Config {
        mods_deny: vec!["pwc.dev-toolkit".into()],
        ..Config::default()
    });
    let port = handle.addr().port();
    let on = [("pwc.dev-toolkit".into(), "1.0.0".into())];
    let err = match Connection::connect_with("127.0.0.1", port, "ada", "", &on) {
        Ok(_) => panic!("a denied mod was admitted"),
        Err(err) => err,
    };
    assert_eq!(err.mods_denied, vec!["pwc.dev-toolkit".to_string()]);
    let retry = Connection::connect_with("127.0.0.1", port, "ada", "", &[]).expect("retry with the mod off");
    assert!(retry.is_alive());
    handle.stop();
}

#[test]
fn no_mod_restriction_admits_everyone() {
    let handle = flat(Config::default());
    let port = handle.addr().port();
    let mods = [("pwc.dev-toolkit".into(), "1.0.0".into()), ("pwc.hotbar".into(), "0.1.0".into())];
    let conn = Connection::connect_with("127.0.0.1", port, "ada", "", &mods).expect("default is open");
    assert!(conn.is_alive());
    handle.stop();
}

/// An `ops.txt` secret beats the bare name: the player is an operator only after `/op` with
/// that secret. The line is answered privately and reaches nobody else.
#[test]
fn an_operator_secret_is_proved_with_op_and_never_relayed() {
    let handle = flat(Config {
        teleport: TeleportPolicy::Ops,
        ops: vec!["ada".into()],
        op_secrets: vec![("Ada".into(), "s3cret".into())],
        ..Config::default()
    });
    let port = handle.addr().port();
    let mut ada = Connection::connect("127.0.0.1", port, "ada", "").unwrap();
    let mut bob = Connection::connect("127.0.0.1", port, "bob", "").unwrap();
    let far = DVec3::new(300.5, 30.0, 300.5);
    ada.send_teleport(far);
    assert!(chat_until(&mut ada, "only an operator").iter().any(|t| t == "only an operator can teleport"));
    ada.send_chat(chat::GLOBAL, "/op wrong");
    assert!(chat_until(&mut ada, "refused").iter().any(|t| t == "operator secret refused"));
    ada.send_chat(chat::GLOBAL, "/op s3cret");
    assert!(chat_until(&mut ada, "operator").iter().any(|t| t == "you are now an operator"));
    ada.send_teleport(far);
    let moved = eventually(Duration::from_secs(3), || {
        handle.state.lock_recover().players.values().any(|h| &*h.name == "ada" && h.pose.pos == far)
    });
    assert!(moved, "a proved operator may teleport");
    ada.send_chat(chat::GLOBAL, "hello");
    let heard = chat_until(&mut bob, "hello");
    assert!(heard.iter().any(|t| t == "hello"), "ordinary chat still flows: {heard:?}");
    assert!(!heard.iter().any(|t| t.contains("/op") || t.contains("s3cret") || t.contains("wrong")), "{heard:?}");
    assert_eq!(op_secret("/op  s3cret "), Some(Arc::from("s3cret")));
    assert_eq!(op_secret("/opera"), None);
    handle.stop();
}

/// Initials from a source that never hears the server (a spoofed address) get a stateless
/// retry and take no handshake slot; a real client still joins through the retry.
#[test]
fn unvalidated_initials_take_no_handshake_slot() {
    let handle = flat(Config::default());
    let server = SocketAddr::from((Ipv4Addr::LOCALHOST, handle.addr().port()));
    let relay = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    relay.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
    let relay_addr = relay.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let relay_stop = stop.clone();
    // Forward the client's packets to the server; drop every answer.
    let forward = thread::spawn(move || {
        let mut buf = [0u8; 2048];
        while !relay_stop.load(Ordering::Relaxed) {
            if let Ok((n, from)) = relay.recv_from(&mut buf)
                && from != server
            {
                let _ = relay.send_to(&buf[..n], server);
            }
        }
    });
    let (rt, ep) = client_endpoint();
    let attempts: Vec<_> = {
        let _g = rt.enter();
        (0..4).map(|_| ep.connect(relay_addr, "watt").unwrap()).collect()
    };
    rt.block_on(async { tokio::time::sleep(Duration::from_millis(400)).await });
    assert_eq!(handle.handshake_slots(), 0, "a source that cannot answer holds no slot");
    stop.store(true, Ordering::Relaxed);
    forward.join().unwrap();
    drop(attempts);
    Connection::connect("127.0.0.1", server.port(), "real", "").expect("a real client passes the retry");
    handle.stop();
}

#[test]
fn console_text_escapes_control_characters() {
    assert_eq!(console_text("\u{1b}[2Jpwc.hotbar\u{7}"), "\\u{1b}[2Jpwc.hotbar\\u{7}");
    assert_eq!(console_text("plain ünïcode"), "plain ünïcode");
}

#[test]
fn names_that_spell_server_are_reserved() {
    for name in ["server", "Server", "S.E.R.V.E.R", "<server>", "server>", " _server_ "] {
        assert!(reserved_name(name), "{name}");
    }
    for name in ["servers", "observer", "server2", "serve", "ada"] {
        assert!(!reserved_name(name), "{name}");
    }
}

/// The flags' spellings round-trip, and only `ops` asks whether the player is an operator.
#[test]
fn a_policy_reads_its_flag_and_decides_by_operator() {
    for policy in [Policy::Off, Policy::Ops, Policy::All] {
        assert_eq!(Policy::parse(policy.name()), Some(policy));
    }
    assert_eq!(Policy::parse("OPS"), None);
    assert_eq!([Policy::Off.allows(true), Policy::Ops.allows(true), Policy::Ops.allows(false), Policy::All.allows(false)], [false, true, false, true]);
}
