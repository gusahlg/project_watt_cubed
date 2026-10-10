//! A session against a server running in its own process, over real QUIC: start
//! `watt_server --port <p> --worldgen flat --seed 1 --teleport ops --ops ada`, then
//! `WATT_SERVER_PORT=<p> cargo test --release --lib external_server_session -- --ignored --nocapture`.
use super::super::*;
use super::support::*;
use crate::net::client::Incoming;
use crate::world::generation::FLAT_HEIGHT;

const ANSWER: Duration = Duration::from_secs(3);

fn cell_under(pos: DVec3) -> Pos {
    (block_coord(pos.x), FLAT_HEIGHT - 1, block_coord(pos.z))
}

/// Break, place, a rejected edit, a refused and an allowed teleport, chat and mod data, as two
/// players see them.
#[test]
#[ignore = "needs a watt_server process: see the module docs"]
fn external_server_session() {
    let port: u16 = std::env::var("WATT_SERVER_PORT").expect("WATT_SERVER_PORT").parse().expect("a port");
    let mut ada = Connection::connect("127.0.0.1", port, "ada", "").expect("ada joins");
    let mut bob = Connection::connect("127.0.0.1", port, "bob", "").expect("bob joins");
    let (ada_id, bob_id) = (ada.player_id(), bob.player_id());
    println!("ada #{ada_id} at {:?}, bob #{bob_id} at {:?}", ada.spawn(), bob.spawn());
    // A first move puts each in the other's interest set.
    for conn in [&mut ada, &mut bob] {
        let spawn = conn.spawn();
        conn.send_move(spawn, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
    }
    assert!(
        eventually(ANSWER, || {
            ada.poll();
            bob.poll();
            ada.peer(bob_id).is_some_and(|p| p.visible()) && bob.peer(ada_id).is_some_and(|p| p.visible())
        }),
        "the two see each other"
    );

    // Break: the grass under ada.
    let grass = cell_under(ada.spawn());
    let req = ada.send_edit(grass.0, grass.1, grass.2, "air".into()).unwrap();
    assert!(await_event(&mut ada, ANSWER, |e| matches!(e, Incoming::EditAccepted { req: r } if r == req).then_some(())).is_some(), "break accepted");
    let seen = await_event(&mut bob, ANSWER, |e| match e {
        Incoming::Edit { x, y, z, spec } if (x, y, z) == grass => Some(spec),
        _ => None,
    });
    assert_eq!(seen.as_deref(), Some("air"), "bob sees the break");
    println!("break: ok");

    // Place: a block back into the hole.
    let rock = rock_spec();
    let req = ada.send_edit(grass.0, grass.1, grass.2, rock.as_str().into()).unwrap();
    assert!(await_event(&mut ada, ANSWER, |e| matches!(e, Incoming::EditAccepted { req: r } if r == req).then_some(())).is_some(), "place accepted");
    let seen = await_event(&mut bob, ANSWER, |e| match e {
        Incoming::Edit { x, y, z, spec } if (x, y, z) == grass => Some(spec),
        _ => None,
    });
    assert_eq!(seen.as_deref(), Some(rock.as_str()), "bob sees the block");
    println!("place: ok");

    // A rejected edit: far out of reach.
    let req = ada.send_edit(grass.0 + 500, grass.1, grass.2, "air".into()).unwrap();
    let rejected = await_event(&mut ada, ANSWER, |e| match e {
        Incoming::EditRejected { req: r, restore } if r == req => Some(restore),
        _ => None,
    });
    assert_eq!(rejected, Some(true), "an out-of-reach edit is rejected and rolled back");
    println!("rejected edit: ok");

    // A refused teleport: bob is no operator.
    let far = DVec3::new(300.5, f64::from(FLAT_HEIGHT) + 3.0, 300.5);
    bob.send_teleport(far);
    let (mut snapped, mut told) = (None, false);
    eventually(ANSWER, || {
        for event in bob.poll() {
            match event {
                Incoming::Position { pos, .. } => snapped = Some(pos),
                Incoming::Chat { text, .. } if &*text == "only an operator can teleport" => told = true,
                _ => {}
            }
        }
        snapped.is_some() && told
    });
    assert!(told, "bob hears why");
    assert!(snapped.is_some_and(|p| p.distance(far) > 100.0), "bob is snapped back, not moved: {snapped:?}");
    println!("refused teleport: ok");
    // ada is one.
    ada.send_teleport(far);
    let landed = await_event(&mut ada, ANSWER, |e| match e {
        Incoming::Position { pos, .. } => Some(pos),
        _ => None,
    });
    assert_eq!(landed, Some(far), "an operator's teleport is echoed at its destination");
    ada.send_teleport(ada.spawn());
    assert!(await_event(&mut ada, ANSWER, |e| matches!(e, Incoming::Position { .. }).then_some(())).is_some());
    println!("operator teleport: ok");

    // Chat.
    ada.send_chat(chat::GLOBAL, "hello from ada");
    let line = await_event(&mut bob, ANSWER, |e| match e {
        Incoming::Chat { from_name, text, .. } if &*text == "hello from ada" => Some(from_name),
        _ => None,
    });
    assert_eq!(line.as_deref(), Some("ada"));
    println!("chat: ok");

    // Mod data, once both stand in each other's interest set again.
    let spawn = ada.spawn();
    ada.send_move(spawn, 0.0, 0.0, DQuat::IDENTITY, Vec3::ZERO, Face::PosY, Stance::Standing);
    assert!(eventually(ANSWER, || {
        ada.poll();
        bob.poll();
        bob.peer(ada_id).is_some_and(|p| p.visible())
    }));
    assert!(ada.send_channel("voice", 7, &[1, 2, 3, 4]));
    let mut got = Vec::new();
    eventually(ANSWER, || {
        bob.poll();
        got.extend(bob.drain_channel("voice"));
        !got.is_empty()
    });
    assert_eq!(got, vec![(ada_id, 7, vec![1, 2, 3, 4])], "bob hears ada's channel, stamped with her id");
    println!("mod data: ok");
}
