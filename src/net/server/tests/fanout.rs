//! Interest, visibility, pose ticks, relays, broadcasts and the join backlog.
use super::super::*;
use super::support::*;

/// A channel relays to the sender's interest set and nobody else, and the
/// server stamps the sender. A full queue simply drops the frame.
#[test]
fn mod_data_relays_only_to_the_visible_set() {
    let (out1, _rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out2, rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out3, rx3) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    // 1 speaks; 2 is in its interest set; 3 is not.
    let mut p1 = test_player(DVec3::new(0.0, 20.0, 0.0), out1, test_kick());
    p1.visible.insert(2);
    players.insert(1u32, p1);
    players.insert(2u32, test_player(DVec3::new(1.0, 20.0, 0.0), out2, test_kick()));
    players.insert(3u32, test_player(DVec3::new(9e3, 20.0, 0.0), out3, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));

    let channel = protocol::Channel::parse("voice").unwrap();
    on_mod_data(&shared, 1, channel, 42, vec![1, 2, 3].try_into().unwrap());

    match ServerMessage::decode(&rx2.try_recv().expect("the visible peer hears it")) {
        Some(ServerMessage::PeerModData { channel, sender, seq, bytes }) => {
            assert_eq!(channel.as_str(), "voice");
            assert_eq!((sender, seq, bytes.as_slice()), (1, 42, &[1, 2, 3][..]));
        }
        other => panic!("expected PeerModData, got {other:?}"),
    }
    assert!(rx3.try_recv().is_err(), "a peer outside interest hears nothing");
}

/// A hand-built state, no sockets: the grid entry must follow the player
/// across bucket borders, never duplicate within a bucket, and floor (not
/// truncate) on negative coordinates.
#[test]
fn grid_membership_follows_movement_across_bucket_borders() {
    let start = DVec3::new(10.0, 20.0, 10.0);
    let (mut state, [_rx]) = roster([(1, start)]);
    state.grid_insert(1, start);
    assert_eq!(state.grid.get(&(0, 0, 0)).map(Vec::len), Some(1));
    let shared = Arc::new(Mutex::new(state));

    // Crossing the x border: the entry moves buckets and the emptied bucket
    // is dropped, not left behind as a leaked key.
    walk(&shared, 1, DVec3::new(INTEREST_RADIUS + 5.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
    {
        let s = shared.lock_recover();
        assert_eq!(s.grid.get(&(1, 0, 0)).map(Vec::as_slice), Some(&[1u32][..]));
        assert!(!s.grid.contains_key(&(0, 0, 0)), "emptied bucket must be removed");
    }

    // Moving within the same bucket must not duplicate the entry.
    walk(&shared, 1, DVec3::new(INTEREST_RADIUS + 6.0, 20.0, 10.0), 0.0, 0.0, Stance::Standing);
    {
        let s = shared.lock_recover();
        assert_eq!(s.grid.get(&(1, 0, 0)).map(Vec::len), Some(1));
        assert_eq!(s.grid.len(), 1);
    }

    // Negative coordinates floor toward -infinity: -1.0 is bucket -1, not 0.
    // (Aged anchor: the hop back is real distance, and this test is about
    // grid bookkeeping, not the envelope.)
    age_move(&shared, 1);
    walk(&shared, 1, DVec3::new(-1.0, 20.0, -1.0), 0.0, 0.0, Stance::Standing);
    {
        let s = shared.lock_recover();
        assert_eq!(s.grid.get(&(-1, 0, -1)).map(Vec::len), Some(1));
        assert_eq!(s.grid.len(), 1);
    }
}

#[test]
fn visibility_changes_match_full_roster_distance_checks() {
    let radius = INTEREST_RADIUS;
    let positions = [
        DVec3::ZERO,
        DVec3::new(radius, 0.0, 0.0),
        DVec3::new(-radius, 0.0, 0.0),
        DVec3::new(0.0, radius + 1.0, 0.0),
        DVec3::new(4.0 * radius, 0.0, 0.0),
        DVec3::new(0.0, 0.0, radius / 2.0),
        DVec3::new(2.0 * radius, 0.0, 0.0),
    ];
    let (out, _rx) = sync_channel(OUT_CAPACITY);
    let mut state = test_state(HashMap::new());
    for (index, pos) in positions.into_iter().enumerate() {
        let id = index as u32 + 1;
        state.players.insert(id, test_player(pos, out.clone(), test_kick()));
        state.grid_insert(id, pos);
    }
    // A nearby joiner must finish its snapshot before it receives poses.
    state.players.get_mut(&6).unwrap().ready = false;

    let mut previous = HashSet::new();
    for pos in [
        DVec3::ZERO,
        DVec3::ZERO,
        positions[1],
        positions[4],
        DVec3::ZERO,
        DVec3::new(9.0 * radius, 0.0, 0.0),
    ] {
        let expected: HashSet<u32> = state.players
            .iter()
            .filter(|&(&id, player)| {
                id != 1 && player.ready && player.pose.pos.distance_squared(pos) <= radius * radius
            })
            .map(|(&id, _)| id)
            .collect();
        // (recipient, subject, entered): an exit both ways, and on entry both sides'
        // poses. Moves inside range wait for the pose tick.
        let mut expected_sends = Vec::new();
        for &id in previous.difference(&expected) {
            expected_sends.push((id, 1, false));
            expected_sends.push((1, id, false));
        }
        for &id in expected.difference(&previous) {
            expected_sends.push((id, 1, true));
            expected_sends.push((1, id, true));
        }

        let mut sends = Vec::new();
        commit_pose(&mut state, 1, pos, None, &mut sends);
        assert_eq!(state.players[&1].visible.iter().copied().collect::<HashSet<u32>>(), expected, "mover at {pos:?}");
        for (&id, player) in &state.players {
            assert_eq!(player.visible.contains(&1), expected.contains(&id), "peer {id}");
        }
        let actual: Vec<_> = sends
            .into_iter()
            .map(|(to, frame)| match ServerMessage::decode(&frame).unwrap() {
                ServerMessage::PeerExited { id } => (to, id, false),
                ServerMessage::PeerPoses { poses } => {
                    assert_eq!(poses.list.len(), 1);
                    let pose = poses.list[0];
                    let subject = &state.players[&pose.id];
                    assert!(pose.pos.distance(subject.pose.pos) < 0.01, "pose of {} at {pos:?}", pose.id);
                    (to, pose.id, true)
                }
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(actual.len(), expected_sends.len());
        for send in expected_sends {
            assert!(actual.contains(&send), "missing {send:?} at {pos:?}");
        }
        previous = expected;
    }
}

/// End-to-end over loopback: moves are only delivered inside the interest
/// radius, and delivery resumes when players end up adjacent again — i.e. the
/// grid entries genuinely follow the players around.
#[test]
fn far_players_hear_no_moves_until_adjacent() {
    let handle = spawn(0, Config { password: String::new(), seed: 4242, ..Config::default() }).unwrap();
    let port = handle.addr().port();
    let mut a = Connection::connect("127.0.0.1", port, "walnutty", "").unwrap();
    let mut b = Connection::connect("127.0.0.1", port, "guahlg", "").unwrap();

    let settle = Duration::from_millis(150);
    thread::sleep(settle);
    a.poll();
    b.poll();
    assert_eq!(b.peers().count(), 1, "guahlg should see walnutty");

    // Walnutty teleports many buckets away. guahlg (still at spawn) is far outside
    // her interest radius, so his view of her must not update.
    let far = DVec3::new(4000.0, 30.0, 4000.0);
    a.send_teleport(far);
    thread::sleep(settle);
    b.poll();
    let walnutty_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
    assert!(
        walnutty_as_seen.x < 100.0,
        "guahlg must not hear a move from {} units away (saw x={})",
        far.x,
        walnutty_as_seen.x
    );

    // guahlg moves right next to walnutty: she is within range of his new position,
    // so she hears it — which requires her grid entry to have followed her.
    b.send_teleport(DVec3::new(4004.0, 30.0, 4004.0));
    thread::sleep(settle);
    a.poll();
    let guahlg_as_seen = a.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
    assert!(
        guahlg_as_seen.x > 3900.0,
        "walnutty should hear guahlg once adjacent (saw x={})",
        guahlg_as_seen.x
    );

    // And the reverse direction: guahlg's entry followed him too.
    a.send_move(DVec3::new(4010.0, 30.0, 4010.0), 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
    thread::sleep(settle);
    b.poll();
    let walnutty_as_seen = b.peers().next().unwrap().sample(Instant::now() + Duration::from_secs(3600)).pos.0;
    assert!(
        walnutty_as_seen.x > 3900.0,
        "guahlg should hear walnutty once adjacent (saw x={})",
        walnutty_as_seen.x
    );

    handle.stop();
}

#[test]
fn interest_at_the_radius_bucket_edges_wrap_and_three_bucket_hops() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let origin = DVec3::new(0.0, 20.0, 0.0);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(origin, out.clone(), test_kick()));
    players.insert(2u32, test_player(DVec3::new(INTEREST_RADIUS, 20.0, 0.0), out.clone(), test_kick()));
    let mut state = test_state(players);
    state.grid_insert(1, origin);
    state.grid_insert(2, DVec3::new(INTEREST_RADIUS, 20.0, 0.0));
    let mut sends = Vec::new();
    commit_pose(&mut state, 1, origin, None, &mut sends);
    assert!(state.players[&1].visible.contains(&2), "exactly INTEREST_RADIUS is visible");
    assert!(state.players[&2].visible.contains(&1));

    // Bucket edge: INTEREST_RADIUS is the first point of bucket 1.
    let on_edge = DVec3::new(INTEREST_RADIUS, 20.0, 0.0);
    let just_inside = DVec3::new(INTEREST_RADIUS - 1.0, 20.0, 0.0);
    assert_eq!(bucket_of(on_edge), (1, 0, 0));
    assert_eq!(bucket_of(just_inside), (0, 0, 0));

    // i32-wrap-like coordinates clamp through block_coord; membership stays 1:1.
    age_state(&mut state, 1, Duration::from_secs(10));
    let wrap = DVec3::new(crate::math::WORLD_BORDER, 20.0, crate::math::WORLD_BORDER);
    sends.clear();
    commit_pose(&mut state, 1, wrap, None, &mut sends);
    let entries: usize = state.grid.values().map(Vec::len).sum();
    assert_eq!(entries, 2, "wrap-range move must not duplicate grid entries");
    assert!(!state.players[&1].visible.contains(&2), "world-border hop leaves interest");
    assert!(!state.players[&2].visible.contains(&1));
    let exited: Vec<_> = sends
        .iter()
        .filter_map(|(_, f)| match ServerMessage::decode(f) {
            Some(ServerMessage::PeerExited { id }) => Some(id),
            _ => None,
        })
        .collect();
    assert!(exited.contains(&1) && exited.contains(&2), "PeerExited reaches every peer");

    // Three buckets in one message (teleport-sized hop).
    let start = DVec3::new(10.0, 20.0, 10.0);
    state.players.get_mut(&1).unwrap().pose.pos = start;
    state.grid.clear();
    state.grid_insert(1, start);
    state.grid_insert(2, DVec3::new(INTEREST_RADIUS, 20.0, 0.0));
    let hop = DVec3::new(10.0 + 3.0 * INTEREST_RADIUS, 20.0, 10.0);
    assert_ne!(bucket_of(start), bucket_of(hop));
    sends.clear();
    commit_pose(&mut state, 1, hop, None, &mut sends);
    assert_eq!(state.grid.get(&bucket_of(hop)).map(Vec::as_slice), Some(&[1u32][..]));
    assert!(!state.grid.contains_key(&bucket_of(start)), "emptied start bucket is dropped");
    let _ = rx;
}

#[test]
fn visible_stays_symmetric_across_random_moves_of_twenty_players() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut state = test_state(HashMap::new());
    let mut rng = XorShift::new(0x0020_91A7);
    let span = 6.0 * INTEREST_RADIUS;
    for id in 1..=20u32 {
        let pos = DVec3::new(rng.f64(-span, span), 20.0, rng.f64(-span, span));
        state.players.insert(id, test_player(pos, out.clone(), test_kick()));
        state.grid_insert(id, pos);
    }
    for _ in 0..80 {
        let id = rng.u32(20) + 1;
        let pos = DVec3::new(rng.f64(-span, span), 20.0, rng.f64(-span, span));
        let mut sends = Vec::new();
        commit_pose(&mut state, id, pos, None, &mut sends);
        for (&a, ha) in &state.players {
            for (&b, hb) in &state.players {
                if a >= b {
                    continue;
                }
                assert_eq!(
                    ha.visible.contains(&b),
                    hb.visible.contains(&a),
                    "visibility {a}↔{b} broke after moving {id} to {pos:?}"
                );
            }
        }
        let entries: usize = state.grid.values().map(Vec::len).sum();
        assert_eq!(entries, 20);
    }
}

#[test]
fn bootstrap_backlog_overflow_kicks_without_poisoning_the_lock() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let kick = test_kick();
    let mut players = HashMap::new();
    let mut p = test_player(DVec3::new(0.0, 20.0, 0.0), out, kick.clone());
    p.ready = false;
    players.insert(1u32, p);
    let shared = Arc::new(Mutex::new(test_state(players)));
    // Swings are cosmetic: a flood past the byte cap must not kick.
    let swings = BACKLOG_BYTES / 4 + 8;
    for i in 0..swings {
        broadcast_all(&shared, &ServerMessage::PeerSwing { id: i as u32 }, None);
    }
    assert!(!shared.lock_recover().players[&1].kicked.load(Ordering::Relaxed), "swings are dropped, not a kick");
    assert!(shared.lock_recover().players[&1].backlog_bytes <= BACKLOG_BYTES);
    // Pong is essential. Once the swings are gone, a byte-cap overflow kicks.
    let pongs = BACKLOG_BYTES / 4 + 8;
    for i in 0..pongs {
        broadcast_all(&shared, &ServerMessage::Pong { nonce: i as u32 }, None);
    }
    assert!(shared.lock().is_ok(), "kick must not poison the state lock");
    assert!(shared.lock_recover().players.contains_key(&1), "overflow notifies, it does not drop the roster");
    let notified = kick.notified();
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        tokio::time::timeout(Duration::from_millis(50), notified).await.expect("kick must notify")
    });
}

#[test]
fn hook_denied_chat_reaches_only_the_sender() {
    let (shared, [rx1, rx2]) = lobby([(1, DVec3::new(8.5, 20.0, 8.5)), (2, DVec3::new(9.5, 20.0, 8.5))]);
    let (mut rec, _) = hooks::Recording::new("mute");
    rec.deny_chat = true;
    rec.reason = Arc::from("no talking");
    let table = Mutex::new(hooks::Table::new(vec![Box::new(rec)]));

    on_chat(&shared, Some(&table), 1, chat::GLOBAL, "hello");

    match &drain(&rx1)[..] {
        [ServerMessage::Chat { from_id, from_name, text, .. }] => {
            assert_eq!(*from_id, 0);
            assert_eq!(&**from_name, "server");
            assert_eq!(&**text, "no talking");
        }
        other => panic!("sender should hear the deny reason, got {other:?}"),
    }
    assert!(drain(&rx2).is_empty(), "denied chat must not reach peers");
}

#[test]
fn writer_error_kicks_and_a_clean_close_does_not() {
    let (tx, rx) = outbox(4);
    let kick = Arc::new(Notify::new());
    let kick2 = kick.clone();
    let writer = thread::spawn(move || {
        drain_writer(rx, &kick2, |_| Err(io::Error::other("closed")));
    });
    tx.try_send(Arc::<[u8]>::from([1u8, 2, 3].as_slice())).unwrap();
    writer.join().unwrap();
    let rt = Runtime::new().unwrap();
    let notified = kick.notified();
    rt.block_on(async {
        tokio::time::timeout(Duration::from_millis(50), notified)
            .await
            .expect("a write error must kick the reader");
    });

    let (tx, rx) = outbox(4);
    let kick = Arc::new(Notify::new());
    let kick2 = kick.clone();
    let writer = thread::spawn(move || {
        drain_writer(rx, &kick2, |_| Ok(()));
    });
    tx.try_send(Arc::<[u8]>::from([9u8].as_slice())).unwrap();
    drop(tx);
    writer.join().unwrap();
    let notified = kick.notified();
    rt.block_on(async {
        assert!(
            tokio::time::timeout(Duration::from_millis(50), notified).await.is_err(),
            "channel close is depart, not a kick"
        );
    });
}

#[test]
fn peer_joined_is_delivered_once() {
    let (out1, rx1) = sync_channel::<Arc<[u8]>>(8);
    let (out2, rx2) = sync_channel::<Arc<[u8]>>(8);
    let mut p1 = test_player(DVec3::new(0.0, 20.0, 0.0), out1, test_kick());
    p1.announced.insert(2);
    let p2 = test_player(DVec3::new(1.0, 20.0, 0.0), out2, test_kick());
    let mut players = HashMap::new();
    players.insert(1u32, p1);
    players.insert(2u32, p2);
    let shared = Arc::new(Mutex::new(test_state(players)));
    broadcast_all(&shared, &ServerMessage::PeerJoined { id: 2, name: "b".into() }, None);
    assert!(rx1.try_recv().is_err(), "player 1 was already told about 2");
    assert!(rx2.try_recv().is_ok(), "player 2 had not been told");
    assert!(shared.lock_recover().players[&1].announced.is_empty(), "the broadcast used up the roster entry");
    broadcast_all(&shared, &ServerMessage::PeerJoined { id: 3, name: "c".into() }, None);
    assert!(rx1.try_recv().is_ok(), "a later join is announced");
}

/// Joins and leaves through the real roster bookkeeping, around three players who stay: no
/// handle ever names more peers than the roster holds, and the sets drain as peers go.
#[test]
fn announced_never_outgrows_the_roster_over_a_thousand_joins() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut state = test_state(HashMap::new());
    let join = |state: &mut State, id: u32| {
        let pos = DVec3::new(f64::from(id % 7), 20.0, 0.0);
        let mut h = test_player(pos, out.clone(), test_kick());
        h.ready = false;
        state.admit(id, h);
    };
    for id in 1..=3 {
        join(&mut state, id);
    }
    let mut next = 4;
    let mut live: VecDeque<u32> = VecDeque::new();
    for cycle in 0..1000u32 {
        // Two join together, so each is in the other's roster before either is announced.
        for _ in 0..2 {
            join(&mut state, next);
            live.push_back(next);
            next += 1;
        }
        // Only some go live and are announced; the rest leave during their bootstrap.
        let id = live[live.len() - 2];
        if cycle % 3 != 0 {
            state.players.get_mut(&id).unwrap().ready = true;
            let name = state.players[&id].name.clone();
            drop(broadcast(&mut state, &ServerMessage::PeerJoined { id, name }, |pid, _| pid != id));
        }
        while live.len() > 4 {
            let gone = live.pop_front().unwrap();
            assert!(state.remove_player(gone).is_some());
        }
        let roster = state.players.len();
        for (pid, h) in &state.players {
            assert!(h.announced.len() < roster, "cycle {cycle}: #{pid} names {} peers, roster {roster}", h.announced.len());
            assert!(h.announced.iter().all(|a| state.players.contains_key(a)), "cycle {cycle}: #{pid} names a player who left");
        }
    }
    for gone in live {
        state.remove_player(gone);
    }
    let left: usize = state.players.values().map(|h| h.announced.len()).sum();
    assert!(left <= 6, "the three who stayed name at most each other: {left}");
}

#[test]
fn backlog_drops_aged_frames() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(4);
    let mut player = test_player(DVec3::ZERO, out, test_kick());
    player.ready = false;
    let now = Instant::now();
    let swing: Arc<[u8]> = ServerMessage::PeerSwing { id: 7 }.encode().into();
    assert!(enqueue_backlog(&mut player, swing, now));
    player.backlog[0].at = now - BACKLOG_AGE;
    let pong: Arc<[u8]> = ServerMessage::Pong { nonce: 1 }.encode().into();
    assert!(enqueue_backlog(&mut player, pong, now));
    assert_eq!(player.backlog.len(), 1, "a frame older than the age bound is dropped");
    assert!(!protocol::is_cosmetic(&player.backlog[0].frame));
    assert!(!player.kicked.load(Ordering::Relaxed));
}

/// The tick sends a moved near peer every tick and a far one on its slot, once per window.
#[test]
fn pose_tick_sends_near_peers_each_tick_and_far_peers_less_often() {
    let (out1, rx1) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out2, _rx2) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let (out3, _rx3) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let near = DVec3::new(4.0, 20.0, 0.0);
    let far = DVec3::new(NEAR + 20.0, 20.0, 0.0);
    let mut state = test_state(HashMap::new());
    for (id, pos, out) in [(1u32, DVec3::new(0.0, 20.0, 0.0), out1), (2, near, out2), (3, far, out3)] {
        state.players.insert(id, test_player(pos, out, test_kick()));
        state.grid_insert(id, pos);
    }
    let mut sends = Vec::new();
    for (id, pos) in [(1u32, DVec3::new(0.0, 20.0, 0.0)), (2, near), (3, far)] {
        commit_pose(&mut state, id, pos, None, &mut sends);
    }
    let shared = Arc::new(Mutex::new(state));
    let _ = drain(&rx1);
    let heard = |ticks: u64, step: f64| {
        let mut seen = Vec::new();
        for _ in 0..ticks {
            for (id, pos) in [(2u32, near), (3, far)] {
                let mut state = shared.lock_recover();
                let mut sends = Vec::new();
                let p = pos + DVec3::new(0.0, 0.0, step * state.tick as f64);
                commit_pose(&mut state, id, p, None, &mut sends);
                assert!(sends.is_empty(), "a move inside range waits for the tick");
            }
            broadcast_poses(&shared);
            for msg in drain(&rx1) {
                if let ServerMessage::PeerPoses { poses } = msg {
                    seen.extend(poses.list.iter().map(|p| p.id));
                }
            }
        }
        seen
    };
    let seen = heard(8, 0.5);
    assert_eq!(seen.iter().filter(|&&id| id == 2).count(), 8, "a near peer every tick");
    assert_eq!(seen.iter().filter(|&&id| id == 3).count(), 2, "a far peer every fourth tick");
    let still = heard(8, 0.0);
    assert!(still.iter().filter(|&&id| id == 2).count() <= 1, "a peer that stays put is not resent");
}

#[test]
fn send_blocking_honours_a_kick_and_a_deadline() {
    let kicked = AtomicBool::new(true);
    let (tx, _rx) = outbox(1);
    let started = Instant::now();
    assert!(!send_until(&tx, &kicked, Arc::from([0u8].as_slice()), Instant::now() + SEND_DEADLINE));
    assert!(started.elapsed() < Duration::from_millis(50), "a kick returns at once");

    let kicked = AtomicBool::new(false);
    let (tx, rx) = outbox(1);
    tx.try_send(Arc::from([0u8].as_slice())).unwrap();
    let started = Instant::now();
    assert!(!send_until(
        &tx,
        &kicked,
        Arc::from([1u8].as_slice()),
        Instant::now() + Duration::from_millis(30),
    ));
    assert!(kicked.load(Ordering::Relaxed), "a missed deadline is a kick");
    assert!(started.elapsed() < Duration::from_millis(500), "the wait is the deadline, not unbounded");
    let _ = rx;
}

#[test]
fn swing_reaches_only_visible_players() {
    let (out_a, rx_a) = sync_channel::<Arc<[u8]>>(4);
    let (out_b, rx_b) = sync_channel::<Arc<[u8]>>(4);
    let (out_c, rx_c) = sync_channel::<Arc<[u8]>>(4);
    let mut swinger = test_player(DVec3::new(0.5, 20.0, 0.5), out_a, test_kick());
    swinger.visible.insert(2);
    swinger.visible.insert(3);
    let seen = test_player(DVec3::new(1.5, 20.0, 0.5), out_b, test_kick());
    let mut hidden = test_player(DVec3::new(2.5, 20.0, 0.5), out_c, test_kick());
    hidden.ready = false;
    let mut players = HashMap::new();
    players.insert(1, swinger);
    players.insert(2, seen);
    players.insert(3, hidden);
    let shared = Arc::new(Mutex::new(test_state(players)));
    relay_swing(&shared, 1);
    assert!(rx_a.try_recv().is_err(), "the swinger does not hear their own swing");
    let frame = rx_b.try_recv().expect("a player who can see the swinger hears it");
    assert!(matches!(ServerMessage::decode(&frame), Some(ServerMessage::PeerSwing { id: 1 })));
    assert!(rx_c.try_recv().is_err(), "a peer who is not ready is not in the audience");
}

#[test]
fn minute_clock_broadcast_reaches_a_ready_player() {
    assert_eq!(TIME_BROADCAST, Duration::from_secs(60));
    let (out, rx) = sync_channel::<Arc<[u8]>>(4);
    let mut players = HashMap::new();
    players.insert(1, test_player(DVec3::new(0.5, 20.0, 0.5), out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    broadcast_clock(&shared, lax_ctx());
    assert!(drain(&rx).iter().any(|msg| matches!(msg, ServerMessage::Time { .. })));
}

/// One full frame plus another fits the join backlog, and an essential frame that waits
/// past the age bound kicks the joiner instead of vanishing.
#[test]
fn backlog_holds_a_full_frame_and_kicks_on_an_aged_essential_one() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(4);
    let mut player = test_player(DVec3::ZERO, out, test_kick());
    player.ready = false;
    let now = Instant::now();
    let air: Arc<str> = "air".into();
    let full: Arc<[u8]> = {
        let mut frames = Vec::new();
        let mut writer = SnapshotWriter::new();
        let mut emit = |frame| frames.push(frame);
        for i in 0..20_000 {
            writer.push((i * 1000, 0, 0), 1, 0, "air", &mut emit);
        }
        frames.swap_remove(0)
    };
    assert!(full.len() > MAX_FRAME - 64 && full.len() <= MAX_FRAME);
    let edit: Arc<[u8]> = ServerMessage::Edit { x: 1, y: 2, z: 3, rev: 1, spec: air.clone() }.encode().into();
    assert!(enqueue_backlog(&mut player, full, now));
    assert!(enqueue_backlog(&mut player, edit.clone(), now), "a full frame and one more fit");
    assert!(!enqueue_backlog(&mut player, edit, now + BACKLOG_AGE), "an aged edit kicks");
    assert_eq!(player.backlog.len(), 2, "no essential frame was dropped");
}

/// What [`move_cluster`] measured over its steady-state rounds.
struct Moves {
    moves: u64,
    busy: Duration,
    /// This thread's CPU time in the moves, so time the scheduler gives other processes is left out.
    cpu: Duration,
    /// Each counted round's CPU time per move, in nanoseconds.
    round_cpu: Vec<f64>,
    allocs: u64,
    corrections: usize,
}

/// `players` in one cluster, all inside each other's interest range, each moving `rounds` times
/// around a ring through the real [`on_move`] under a dedicated server's noclip policy (operators
/// only), so every move also runs the body check. Rounds before `warm` are not counted. `paced`
/// spaces the rounds at [`MOVE_RATE`]; unpaced rounds take steps small enough for the envelope.
fn move_cluster(players: u32, rounds: u32, warm: u32, paced: bool) -> Moves {
    use std::f64::consts::TAU;
    let air = f64::from(crate::world::generation::FLAT_HEIGHT) + 20.0;
    let period = Duration::from_secs(1) / MOVE_RATE;
    // Radians per round: about 6 blocks/s along the ring when paced, a sliver when not.
    let spin = if paced { 0.5 / f64::from(MOVE_RATE) } else { 0.0005 };
    let at = |id: u32, round: u32| {
        let a = f64::from(id) * TAU / f64::from(players) + spin * f64::from(round);
        DVec3::new(12.0 * a.cos() + 0.5, air + f64::from(id % 4) * 3.0, 12.0 * a.sin() + 0.5)
    };
    let mut roster = HashMap::new();
    let mut inboxes = Vec::new();
    for id in 1..=players {
        let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
        let mut p = test_player(at(id, 0), out, test_kick());
        p.name = format!("p{id}").into();
        roster.insert(id, p);
        inboxes.push(rx);
    }
    let (shared, ctx) = flat_shared(roster, NoclipPolicy::Ops, &["admin"]);
    {
        let mut state = shared.lock_recover();
        for id in 1..=players {
            let pos = state.players[&id].pose.pos;
            state.grid_insert(id, pos);
        }
    }
    let mut out = Moves { moves: 0, busy: Duration::ZERO, cpu: Duration::ZERO, round_cpu: Vec::new(), allocs: 0, corrections: 0 };
    // The connection's reader reuses one buffer for the frames a handler queues.
    let mut sends = Vec::new();
    let start = Instant::now();
    for round in 0..rounds {
        if round == warm {
            load::LOCK_HOLD.reset();
        }
        let mut round_cpu = Duration::ZERO;
        for id in 1..=players {
            let (pos, next) = (at(id, round + 1), at(id, round + 2));
            let velocity = ((next - pos) / period.as_secs_f64()).as_vec3();
            crate::alloc_count::reset();
            let (began, began_cpu) = (Instant::now(), thread_cpu());
            let pose = Pose { velocity, ..Pose::standing(pos, DQuat::IDENTITY, Face::PosY) };
            on_move(&shared, &ctx, id, pose, &mut sends);
            let (took, took_cpu) = (began.elapsed(), thread_cpu().saturating_sub(began_cpu));
            if round >= warm {
                out.cpu += took_cpu;
                round_cpu += took_cpu;
                out.allocs += crate::alloc_count::alloc_count();
                out.busy += took;
                out.moves += 1;
            }
        }
        if round >= warm {
            out.round_cpu.push(round_cpu.as_nanos() as f64 / f64::from(players));
        }
        for rx in &inboxes {
            while let Ok(frame) = rx.try_recv() {
                out.corrections += usize::from(matches!(ServerMessage::decode(&frame), Some(ServerMessage::Position { .. })));
            }
        }
        if paced && let Some(rest) = (start + period * (round + 1)).checked_duration_since(Instant::now()) {
            thread::sleep(rest);
        }
    }
    out
}

/// CPU time this thread has run.
fn thread_cpu() -> Duration {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `t` is a valid timespec for the call to fill.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

/// A move that leaves every interest set unchanged allocates nothing on the server.
#[test]
fn steady_state_moves_allocate_nothing() {
    let moves = move_cluster(16, 24, 8, false);
    assert_eq!(moves.corrections, 0, "honest moves are not corrected");
    assert_eq!(moves.allocs, 0, "{} allocations over {} steady-state moves", moves.allocs, moves.moves);
}

/// 64 players in one cluster at [`MOVE_RATE`] through the real [`on_move`]: µs per move, the
/// State lock hold, and allocations per steady-state move.
#[test]
#[ignore = "probe: cargo test --release --lib fanout_probe -- --ignored --nocapture"]
fn fanout_probe_64_players() {
    let moves = move_cluster(64, 240, 40, true);
    let lock = load::LOCK_HOLD.summary();
    let mut rounds = moves.round_cpu.clone();
    rounds.sort_by(f64::total_cmp);
    println!(
        "fan-out probe: 64 players, {} moves: {:.3} µs/move, cpu {:.3} µs/move, median round cpu {:.3} µs/move; lock p50 {:.3} / p99 {:.3} / max {:.1} µs over {} holds; {:.3} allocs/move; {} corrections",
        moves.moves,
        moves.busy.as_secs_f64() * 1e6 / moves.moves as f64,
        moves.cpu.as_secs_f64() * 1e6 / moves.moves as f64,
        rounds[rounds.len() / 2] / 1e3,
        lock.p50 as f64 / 1e3,
        lock.p99 as f64 / 1e3,
        lock.max as f64 / 1e3,
        lock.count,
        moves.allocs as f64 / moves.moves as f64,
        moves.corrections,
    );
    assert_eq!(moves.corrections, 0, "honest moves are not corrected");
}
