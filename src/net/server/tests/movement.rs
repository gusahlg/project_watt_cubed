//! The movement envelope, teleport permission, cruise and the noclip body check.
use super::super::*;
use super::support::*;

#[test]
fn nonfinite_angles_do_not_enter_authoritative_state() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let mut players = HashMap::new();
    let mut p = test_player(start, out, test_kick());
    p.yaw = 0.25;
    p.pitch = -0.5;
    players.insert(1, p);
    let shared = Arc::new(Mutex::new(test_state(players)));

    let attempted = DVec3::new(9.5, 20.0, 8.5);
    walk(&shared, 1, attempted, f32::NAN, 0.0, Stance::Sneaking);
    walk(&shared, 1, attempted, 0.0, f32::INFINITY, Stance::Sneaking);

    let state = shared.lock_recover();
    let player = &state.players[&1];
    assert_eq!(player.pos, start);
    assert_eq!(player.yaw, 0.25);
    assert_eq!(player.pitch, -0.5);
    assert_eq!(player.stance, Stance::Standing);
}

/// A jump no legitimate movement could make is NOT committed — the server
/// keeps the last accepted position (which edit reach reads) and snaps the
/// client back with an authoritative `Position`.
#[test]
fn implausible_moves_are_rejected_and_corrected() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));

    // A plausible walk step commits.
    let step = DVec3::new(10.5, 20.0, 8.5);
    walk(&shared, 1, step, 0.1, 0.0, Stance::Standing);
    assert_eq!(shared.lock_recover().players[&1].pos, step);
    assert!(rx.try_recv().is_err(), "an accepted move needs no correction");

    // Move(target)+Edit(target) forging: the cross-map hop is refused...
    let forged = DVec3::new(4000.0, 20.0, 4000.0);
    walk(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
    assert_eq!(shared.lock_recover().players[&1].pos, step, "position unchanged");
    match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
        Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, step),
        other => panic!("expected a Position snap-back, got {other:?}"),
    }
    // ...so the follow-up edit at the forged position stays out of reach.
    on_edit(&shared, None, chartless(), 1, 7, 4000, 20, 4000, 0, "air");
    assert!(!shared.lock_recover().edits.contains_key(&(4000, 20, 4000)));

    // Outside the world border: rejected no matter how slow.
    age_move(&shared, 1);
    walk(&shared, 1, DVec3::new(2.0e9, 20.0, 8.5), 0.0, 0.0, Stance::Standing);
    assert_eq!(shared.lock_recover().players[&1].pos, step);
}

/// `/tp` is an explicit, policy-gated discontinuity: allowed it commits
/// (envelope exempt), refused it snaps the client back.
#[test]
fn teleport_is_permissioned() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let far = DVec3::new(50_000.5, 30.0, -2_000.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));

    on_teleport(&shared, &test_ctx(true), 1, far);
    assert_eq!(shared.lock_recover().players[&1].pos, far, "allowed teleport commits");
    match ServerMessage::decode(&rx.try_recv().expect("accepted teleport echoes Position")) {
        Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, far),
        other => panic!("expected a Position echo, got {other:?}"),
    }

    on_teleport(&shared, &test_ctx(false), 1, start);
    assert_eq!(shared.lock_recover().players[&1].pos, far, "refused teleport is not committed");
    match ServerMessage::decode(&rx.try_recv().expect("a correction is sent")) {
        Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, far),
        other => panic!("expected a Position snap-back, got {other:?}"),
    }
}

#[test]
fn burst_faster_than_cap_then_a_legal_move_corrects_once_then_accepts() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let forged = DVec3::new(4000.0, 20.0, 4000.0);
    for _ in 0..8 {
        walk(&shared, 1, forged, 0.0, 0.0, Stance::Standing);
    }
    assert_eq!(shared.lock_recover().players[&1].pos, start);
    let mut corrections = 0;
    while let Ok(frame) = rx.try_recv() {
        match ServerMessage::decode(&frame) {
            Some(ServerMessage::Position { pos, .. }) => {
                assert_eq!(pos, start);
                corrections += 1;
            }
            other => panic!("expected Position, got {other:?}"),
        }
    }
    assert!(corrections >= 1, "the burst must snap back at least once");
    let legal = DVec3::new(10.5, 20.0, 8.5);
    walk(&shared, 1, legal, 0.0, 0.0, Stance::Standing);
    assert_eq!(shared.lock_recover().players[&1].pos, legal);
    assert!(rx.try_recv().is_err(), "a legal follow-up must not snap back");
}

#[test]
fn long_silence_then_a_legitimate_teleport_obeys_the_flag() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let dest = DVec3::new(50_000.5, 30.0, -2_000.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    age_move(&shared, 1);
    on_teleport(&shared, &test_ctx(true), 1, dest);
    assert_eq!(shared.lock_recover().players[&1].pos, dest);
    let _ = rx.try_recv();
    on_teleport(&shared, &test_ctx(false), 1, start);
    assert_eq!(shared.lock_recover().players[&1].pos, dest);
    match ServerMessage::decode(&rx.try_recv().expect("refused /tp snaps back")) {
        Some(ServerMessage::Position { pos, .. }) => assert_eq!(pos, dest),
        other => panic!("expected Position, got {other:?}"),
    }
}

#[test]
fn fall_flight_and_cruise_follow_the_reported_speed() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(8);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    stamp_gap(&shared, 1);

    let fall = DVec3::new(start.x, start.y - 2_000.0, start.z);
    on_move(
        &shared,
        lax_ctx(),
        1,
        fall,
        0.0,
        0.0,
        DQuat::IDENTITY,
        Vec3::new(0.0, -30_000.0, 0.0),
        Face::PosY,
        Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos, fall, "a long fall under MAX_SPEED is not snapped");
    assert!(rx.try_recv().is_err());

    // The fall left a stored speed. Clear it so the next drop is judged on a zero report.
    shared.lock_recover().players.get_mut(&1).unwrap().velocity = Vec3::ZERO;
    stamp_gap(&shared, 1);
    let forged = DVec3::new(fall.x, fall.y - 2_000.0, fall.z);
    on_move(&shared, lax_ctx(), 1, forged, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
    assert_eq!(shared.lock_recover().players[&1].pos, fall, "the same drop with no speed is a teleport");

    stamp_gap(&shared, 1);
    let flown = DVec3::new(start.x + 1_000.0, start.y, start.z);
    // Fly from `fall`, where the refused drop left the player.
    let speed = crate::player::MAX_SPEED as f32;
    on_move(
        &shared,
        lax_ctx(),
        1,
        DVec3::new(fall.x + 1_000.0, fall.y, fall.z),
        0.0,
        0.0,
        DQuat::IDENTITY,
        Vec3::new(speed, 0.0, 0.0),
        Face::PosY,
        Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos.x, fall.x + 1_000.0, "fast flight is not snapped");
    let _ = flown;

    stamp_gap(&shared, 1);
    let too_far = DVec3::new(fall.x + crate::player::MAX_SPEED * 2.0, fall.y, fall.z);
    on_move(
        &shared,
        lax_ctx(),
        1,
        too_far,
        0.0,
        0.0,
        DQuat::IDENTITY,
        Vec3::new(speed, 0.0, 0.0),
        Face::PosY,
        Stance::Standing,
    );
    assert!(
        (shared.lock_recover().players[&1].pos.x - (fall.x + 1_000.0)).abs() < 1.0,
        "past MAX_SPEED without cruise snaps back"
    );

    on_cruise(&shared, 1, crate::player::CRUISE_MAX);
    stamp_gap(&shared, 1);
    let cruise_to = DVec3::new(fall.x + crate::player::MAX_SPEED * 2.0, fall.y, fall.z);
    on_move(
        &shared,
        lax_ctx(),
        1,
        cruise_to,
        0.0,
        0.0,
        DQuat::IDENTITY,
        Vec3::new(crate::player::CRUISE_MAX as f32, 0.0, 0.0),
        Face::PosY,
        Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos, cruise_to, "a declared cruise is not snapped");
}

/// `/tp` and `/time` are operator-only under [`TeleportPolicy::Ops`]. The
/// reason is a private chat from "server". The speed cap is the same
/// whether or not the client claimed a mod.
#[test]
fn non_operator_time_and_teleport_are_refused_with_a_reason() {
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 20.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    let mut ops = test_ctx(true);
    ops.teleport = TeleportPolicy::Ops;
    ops.ops = vec!["p".into()];

    on_set_time(&shared, &ops, 1, 0.2);
    assert!((shared.lock_recover().day - 0.2).abs() < 1e-6);
    assert!(drain(&rx).iter().any(|m| matches!(m, ServerMessage::Time { day, .. } if (*day - 0.2).abs() < 1e-4)));

    let far = DVec3::new(80.5, 20.0, 8.5);
    on_teleport(&shared, &ops, 1, far);
    assert_eq!(shared.lock_recover().players[&1].pos, far);

    let mut guest = test_ctx(true);
    guest.teleport = TeleportPolicy::Ops;
    on_set_time(&shared, &guest, 1, 0.9);
    assert!((shared.lock_recover().day - 0.2).abs() < 1e-6, "a guest does not move the clock");
    assert!(
        drain(&rx).iter().any(|m| matches!(
            m,
            ServerMessage::Chat { from_name, text, .. } if from_name.as_ref() == "server" && text.as_ref() == "only an operator can set the time"
        ))
    );

    on_teleport(&shared, &guest, 1, start);
    assert_eq!(shared.lock_recover().players[&1].pos, far, "a guest teleport is not committed");
    let refused = drain(&rx);
    assert!(refused.iter().any(|m| matches!(m, ServerMessage::Position { pos, .. } if *pos == far)));
    assert!(refused.iter().any(|m| matches!(
        m,
        ServerMessage::Chat { text, .. } if text.as_ref() == "only an operator can teleport"
    )));

    let off = test_ctx(false);
    on_teleport(&shared, &off, 1, start);
    assert!(drain(&rx).iter().any(|m| matches!(
        m,
        ServerMessage::Chat { text, .. } if text.as_ref() == "teleport is not permitted"
    )));
}

/// A cap below [`crate::player::MAX_SPEED`] bounds flight and cruise. `/tp`
/// stays refused when the policy is off. Solid overlap is [`noclip_snaps_a_body_in_solid_ground`].
#[test]
fn flyspeed_above_the_server_cap_is_snapped() {
    let cap = 30.0 * crate::math::PER_METER;
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 40.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    shared.lock_recover().max_speed = cap;

    let near = DVec3::new(start.x + 10.0 * crate::math::PER_METER, start.y, start.z);
    on_move(
        &shared, lax_ctx(), 1, near, 0.0, 0.0, DQuat::IDENTITY,
        Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos, near, "10 m under the cap commits");

    age_move(&shared, 1);
    let leap = DVec3::new(near.x + 500.0 * crate::math::PER_METER, near.y, near.z);
    on_move(
        &shared, lax_ctx(), 1, leap, 0.0, 0.0, DQuat::IDENTITY,
        Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos, near, "500 m over the cap snaps");
    assert!(drain(&rx).iter().any(|m| matches!(m, ServerMessage::Position { .. })));

    on_cruise(&shared, 1, crate::player::CRUISE_MAX);
    assert!((shared.lock_recover().players[&1].cruise_speed - cap).abs() < 1e-6);
    age_move(&shared, 1);
    on_move(
        &shared, lax_ctx(), 1, leap, 0.0, 0.0, DQuat::IDENTITY,
        Vec3::new(crate::player::CRUISE_MAX as f32, 0.0, 0.0), Face::PosY, Stance::Standing,
    );
    assert_eq!(shared.lock_recover().players[&1].pos, near, "cruise cannot outrun a lower cap");

    on_teleport(&shared, &test_ctx(false), 1, leap);
    assert_eq!(shared.lock_recover().players[&1].pos, near);
}

#[test]
fn noclip_snaps_a_body_in_solid_ground() {
    let start = DVec3::new(0.5, 20.0, 0.5);
    let surface = DVec3::new(
        0.5,
        crate::world::generation::FLAT_HEIGHT as f64 + crate::player::Stance::Standing.eye_offset(),
        0.5,
    );
    let buried = DVec3::new(0.5, crate::world::generation::FLAT_HEIGHT as f64, 0.5);
    let step = |shared: &Arc<Mutex<State>>, ctx: &Ctx, pos: DVec3| {
        on_move(shared, ctx, 1, pos, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
    };

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    step(&shared, &ctx, surface);
    assert_eq!(shared.lock_recover().players[&1].pos, surface, "standing on the flat surface is clear");

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    step(&shared, &ctx, buried);
    assert_eq!(shared.lock_recover().players[&1].pos, start, "a body in solid ground snaps back");

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::All, &[]);
    step(&shared, &ctx, buried);
    assert_eq!(shared.lock_recover().players[&1].pos, buried, "noclip all accepts the buried pose");

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Ops, &["p"]);
    step(&shared, &ctx, buried);
    assert_eq!(shared.lock_recover().players[&1].pos, buried, "an operator may pass");

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Ops, &["p"]);
    shared.lock_recover().players.get_mut(&1).unwrap().name = "guest".into();
    step(&shared, &ctx, buried);
    assert_eq!(shared.lock_recover().players[&1].pos, start, "a guest under ops snaps back");

    let (players, _rx) = pose(start);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    step(&shared, &ctx, start);
    let into = DVec3::new(start.x + 1.0, start.y, start.z);
    let mut before = [(0i32, 0, 0); BODY_CELL_CAP];
    let mut after = [(0i32, 0, 0); BODY_CELL_CAP];
    let n0 = fill_body_cells(start, Stance::Standing, Face::PosY, &mut before).unwrap();
    let n1 = fill_body_cells(into, Stance::Standing, Face::PosY, &mut after).unwrap();
    let fresh = after[..n1]
        .iter()
        .copied()
        .find(|cell| !before[..n0].contains(cell))
        .expect("the step enters a new cell");
    {
        let mut state = shared.lock_recover();
        let spec = rock_spec();
        let id = state.registry.parse_spec(&spec).unwrap();
        assert!(state.registry.is_solid(id));
        let shared_spec = state.intern(id).unwrap();
        state.edits.insert(fresh, Cell { block: id, spec: shared_spec, rev: 1, natural: false });
    }
    step(&shared, &ctx, into);
    assert_eq!(shared.lock_recover().players[&1].pos, start, "a solid edit the body newly enters snaps back");
}

/// A body standing on the start world across a chart's edge reads the cells past the edge
/// through the seam, as the client's collision does: standing on the surface there is clear,
/// and the same body sunk into the ground is blocked.
#[test]
fn noclip_reads_a_chart_edge_like_the_client() {
    use crate::space::atlas::Patch;
    let mut registry = BlockRegistry::with_builtins();
    let generator = crate::world::terrain::generator(&mut registry, 4242, TerrainCfg::default());
    let seams = Seams::new(generator.atlases().to_vec());
    let state = {
        let mut state = State::new(registry, 0.3, crate::player::MAX_SPEED);
        state.next_id = 2;
        state
    };
    let home = generator.cosmos().expect("cosmos").home();
    let atlas = generator
        .atlases()
        .iter()
        .find(|a| (a.centre - home.centre_f()).length() < 1.0)
        .expect("the start world is charted");
    let patch = Patch::Shell { band: 0, face: Face::PosY };
    let (o, size) = atlas.storage_box(patch);
    let edge = (o[0] + size[0] - 1) as i32;
    let top = |x: i32, z: i32| {
        let g = seams.glue_cell(BlockCoord::new(x, 0, z)).map_or((x, z), |g| (g.x, g.z));
        generator.height(g.0, g.1)
    };
    let eye = crate::player::Stance::Standing.eye_offset();
    let mut checked = 0;
    for k in 0..64 {
        let z = (o[2] + size[2] / 2) as i32 + k * 7;
        let (inside, outside) = (generator.height(edge, z), top(edge + 1, z));
        if inside == i32::MIN || outside == i32::MIN || outside > inside {
            continue;
        }
        // Feet on the inside column's top, the body reaching past the box edge.
        let pos = DVec3::new(f64::from(edge) + 0.9, f64::from(inside) + eye, f64::from(z) + 0.5);
        assert!(
            !body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &[]),
            "standing across the edge at z {z} (inside top {inside}, glued top {outside}) was blocked"
        );
        let sunk = pos - DVec3::Y * 1.5;
        assert!(
            body_blocked(&state, &generator, &seams, sunk, sunk, Stance::Standing, Face::PosY, &[]),
            "a body sunk into the edge at z {z} was clear"
        );
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} edge columns were level enough to stand across");
}

#[test]
fn noclip_check_stays_cheap() {
    let mut registry = BlockRegistry::with_builtins();
    let generator = crate::world::terrain::generator(&mut registry, 1, TerrainCfg::default());
    let seams = Seams::new(generator.atlases().to_vec());
    let state = {
        let mut state = State::new(registry, 0.3, crate::player::MAX_SPEED);
        state.next_id = 2;
        state
    };
    let mut pos = DVec3::new(8.5, 80.0, 8.5);
    for _ in 0..40 {
        if !body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &[]) {
            break;
        }
        pos.y += 16.0;
    }
    let mut cells = [(0i32, 0, 0); BODY_CELL_CAP];
    let mut occupied = Vec::new();
    let n = fill_body_cells(pos, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
    occupied.extend_from_slice(&cells[..n]);
    for _ in 0..8 {
        pos.x += 0.5;
        let _ = body_blocked(&state, &generator, &seams, pos, pos, Stance::Standing, Face::PosY, &occupied);
        let n = fill_body_cells(pos, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
        occupied.clear();
        occupied.extend_from_slice(&cells[..n]);
    }
    let steps = 64u32;
    let started = Instant::now();
    for _ in 0..steps {
        let next = DVec3::new(pos.x + 0.5, pos.y, pos.z);
        let blocked = std::hint::black_box(body_blocked(
            &state,
            &generator,
            &seams,
            pos,
            next,
            Stance::Standing,
            Face::PosY,
            &occupied,
        ));
        let _ = blocked;
        let n = fill_body_cells(next, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
        occupied.clear();
        occupied.extend_from_slice(&cells[..n]);
        pos = next;
    }
    let us = started.elapsed().as_secs_f64() * 1.0e6 / f64::from(steps);
    let load_ms = 32.0 * 20.0 * us / 1000.0;
    eprintln!("noclip: {us:.2} µs/move; 32 players at 20 Hz = {load_ms:.2} ms/s");
    let stand = Instant::now();
    for _ in 0..steps {
        let blocked = std::hint::black_box(body_blocked(
            &state,
            &generator,
            &seams,
            pos,
            pos,
            Stance::Standing,
            Face::PosY,
            &occupied,
        ));
        assert!(!blocked, "a body that has not entered a new cell is not retested");
    }
    let stand_us = stand.elapsed().as_secs_f64() * 1.0e6 / f64::from(steps);
    eprintln!("noclip standing: {stand_us:.2} µs/move");
    assert!(us < 1000.0, "newly entered cells took {us:.2} µs/move");
    // Long sweeps, straight up from the spawn so the whole path is walked.
    let spawn = generator.chart_spawn().expect("a charted start world");
    let n = fill_body_cells(spawn, Stance::Standing, Face::PosY, &mut cells).expect("body fits");
    let mut worst = 0.0f64;
    for length in [8.0, 64.0, SWEEP_LIMIT] {
        let to = spawn + DVec3::Y * length;
        let started = Instant::now();
        let blocked = std::hint::black_box(body_blocked(
            &state,
            &generator,
            &seams,
            spawn,
            to,
            Stance::Standing,
            Face::PosY,
            &cells[..n],
        ));
        let sweep_us = started.elapsed().as_secs_f64() * 1.0e6;
        assert!(!blocked, "the sky above the spawn is clear");
        eprintln!("noclip sweep of {length} blocks: {sweep_us:.1} µs ({:.2} µs/block)", sweep_us / length);
        worst = worst.max(sweep_us);
    }
    assert!(worst < 200_000.0, "the longest sweep took {worst:.1} µs");
}

/// Under a speed cap, a move split into many messages covers no more than one burst plus
/// the cap over the time taken. A fast stream that stalls and lands at once still passes.
#[test]
fn movement_credit_keeps_its_fraction_when_speed_changes() {
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut h = test_player(DVec3::ZERO, out, test_kick());
    h.budget = MOVE_FLOOR / 4.0;
    let speed = (4000.0 * crate::math::PER_METER) as f32 as f64;
    let reported = Vec3::new(speed as f32, 0.0, 0.0);
    let burst = speed * MOVE_SLACK_SECS;
    assert!((move_allowance(&h, reported, 0.0, speed) - burst / 4.0).abs() < 1e-9);
    h.burst = burst;
    h.budget = burst / 4.0;
    assert!((move_allowance(&h, Vec3::ZERO, 0.0, speed) - MOVE_FLOOR / 4.0).abs() < 1e-9);
    h.budget = 0.0;
    for velocity in [reported, Vec3::ZERO, reported] {
        assert_eq!(move_allowance(&h, velocity, 0.0, speed), 0.0, "changing speed cannot refill spent credit");
    }
}

#[test]
fn split_moves_gain_nothing_over_the_speed_cap() {
    let cap = 30.0 * crate::math::PER_METER;
    let (out, _rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let start = DVec3::new(8.5, 40.0, 8.5);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    shared.lock_recover().max_speed = cap;
    let anchored = Instant::now();
    shared.lock_recover().players.get_mut(&1).unwrap().last_move = anchored;
    for _ in 0..40 {
        let at = shared.lock_recover().players[&1].pos;
        let to = DVec3::new(at.x + 5.0, at.y, at.z);
        on_move(&shared, lax_ctx(), 1, to, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(cap as f32, 0.0, 0.0), Face::PosY, Stance::Standing);
    }
    let covered = shared.lock_recover().players[&1].pos.x - start.x;
    let bound = MOVE_FLOOR + cap * anchored.elapsed().as_secs_f64();
    assert!(covered <= bound + 1e-9, "40 split moves covered {covered} blocks, the bound is {bound}");
    assert!(covered >= MOVE_FLOOR - 5.0, "the burst is spendable, covered {covered}");

    let fast = 1000.0 * crate::math::PER_METER;
    let (out, rx) = sync_channel::<Arc<[u8]>>(OUT_CAPACITY);
    let mut players = HashMap::new();
    players.insert(1u32, test_player(start, out, test_kick()));
    let shared = Arc::new(Mutex::new(test_state(players)));
    shared.lock_recover().players.get_mut(&1).unwrap().last_move = Instant::now() - Duration::from_millis(133);
    let step = fast * 0.033;
    for i in 1..=4 {
        let to = DVec3::new(start.x + step * f64::from(i), start.y, start.z);
        on_move(&shared, lax_ctx(), 1, to, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(fast as f32, 0.0, 0.0), Face::PosY, Stance::Standing);
        assert_eq!(shared.lock_recover().players[&1].pos, to, "stalled move {i} at 1 km/s");
    }
    assert!(rx.try_recv().is_err(), "no snap-back");
}

/// With noclip refused, a move whose destination is open but whose path crosses a wall
/// snaps back. A body teleported into ground can still walk out, a path longer than the
/// sweep fails closed, and a cruise is judged at its destination.
#[test]
fn a_move_cannot_pass_through_a_wall() {
    let eye = crate::player::Stance::Standing.eye_offset();
    let ground = crate::world::generation::FLAT_HEIGHT;
    let surface = DVec3::new(0.5, f64::from(ground) + eye, 0.5);
    let beyond = DVec3::new(6.5, surface.y, 0.5);
    let step = |shared: &Arc<Mutex<State>>, ctx: &Ctx, pos: DVec3, speed: f32| {
        on_move(shared, ctx, 1, pos, 0.0, 0.0, DQuat::IDENTITY, Vec3::new(speed, 0.0, 0.0), Face::PosY, Stance::Standing);
    };

    let (players, _rx) = pose(surface);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    step(&shared, &ctx, beyond, 0.0);
    assert_eq!(shared.lock_recover().players[&1].pos, beyond, "an open path is clear");

    let (players, _rx) = pose(surface);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    {
        let mut state = shared.lock_recover();
        let rock = state.registry.parse_spec(&rock_spec()).unwrap();
        for x in 2..=4 {
            for y in ground..ground + 3 {
                for z in -1..=1 {
                    let spec = state.intern(rock).unwrap();
                    state.edits.insert((x, y, z), Cell { block: rock, spec, rev: 1, natural: false });
                }
            }
        }
    }
    step(&shared, &ctx, beyond, 0.0);
    assert_eq!(shared.lock_recover().players[&1].pos, surface, "a three-block wall is not crossed");

    let buried = DVec3::new(0.5, f64::from(ground), 0.5);
    let (players, _rx) = pose(surface);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    on_teleport(&shared, &ctx, 1, buried);
    step(&shared, &ctx, surface, 0.0);
    assert_eq!(shared.lock_recover().players[&1].pos, surface, "a body teleported into ground walks out");

    let sky = DVec3::new(surface.x, surface.y + SWEEP_LIMIT + 10.0, surface.z);
    let fast = crate::player::MAX_SPEED as f32;
    let (players, _rx) = pose(surface);
    let (shared, ctx) = flat_shared(players, NoclipPolicy::Off, &[]);
    step(&shared, &ctx, sky, fast);
    assert_eq!(shared.lock_recover().players[&1].pos, surface, "a path past the sweep fails closed");
    on_cruise(&shared, 1, crate::player::CRUISE_MAX);
    step(&shared, &ctx, sky, fast);
    assert_eq!(shared.lock_recover().players[&1].pos, sky, "a cruise is judged at its destination");
}
