//! Multiplayer integration tests: an in-process server and several clients over loopback.
//! Each area has its own file; this module holds the shared harness.

mod chaos;
mod presence;
mod session;
mod world;

use std::time::{Duration, Instant};

use super::client::{Connection, Incoming};
use super::server::{self, Config, ServerHandle};
use crate::world::generation::WorldgenKind;

/// A running server and the port it bound.
pub(super) struct Lobby {
    pub server: ServerHandle,
    pub port: u16,
}

impl Lobby {
    /// A server on a free loopback port.
    pub fn start(config: Config) -> Self {
        let server = server::spawn(0, config).expect("server starts");
        let port = server.addr().port();
        Self { server, port }
    }

    /// A small flat world, cheap to serve: what most tests want.
    pub fn flat() -> Self {
        Self::start(Config { seed: 1, worldgen: WorldgenKind::Flat, ..Config::default() })
    }

    /// Join as `name` with no password, waiting until the join snapshot has landed.
    pub fn join(&self, name: &str) -> Connection {
        let mut conn = Connection::connect("127.0.0.1", self.port, name, "").expect("joins");
        settle(std::slice::from_mut(&mut conn), Duration::from_secs(10), |c, _| c.snapshot_ready());
        conn
    }
}

/// Poll every client until `done(client, its events so far)` holds for all of them, or `timeout`
/// passes. Returns each client's events in arrival order. Panics on timeout, naming the clients
/// that were not done.
pub(super) fn settle(
    clients: &mut [Connection],
    timeout: Duration,
    mut done: impl FnMut(&Connection, &[Incoming]) -> bool,
) -> Vec<Vec<Incoming>> {
    let deadline = Instant::now() + timeout;
    let mut events: Vec<Vec<Incoming>> = clients.iter().map(|_| Vec::new()).collect();
    loop {
        for (c, ev) in clients.iter_mut().zip(events.iter_mut()) {
            ev.extend(c.poll());
        }
        let pending: Vec<usize> =
            (0..clients.len()).filter(|&i| !done(&clients[i], &events[i])).collect();
        if pending.is_empty() {
            return events;
        }
        assert!(Instant::now() < deadline, "clients {pending:?} not done after {timeout:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Poll every client for `span`, collecting what arrives.
pub(super) fn listen(clients: &mut [Connection], span: Duration) -> Vec<Vec<Incoming>> {
    let deadline = Instant::now() + span;
    let mut events: Vec<Vec<Incoming>> = clients.iter().map(|_| Vec::new()).collect();
    while Instant::now() < deadline {
        for (c, ev) in clients.iter_mut().zip(events.iter_mut()) {
            ev.extend(c.poll());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    events
}

#[test]
fn two_clients_join_and_stay_connected() {
    let lobby = Lobby::flat();
    assert_eq!(lobby.server.addr().port(), lobby.port);
    let mut clients = vec![lobby.join("ada"), lobby.join("bob")];
    let events = listen(&mut clients, Duration::from_millis(300));
    assert_eq!(events.len(), 2);
    assert!(clients.iter().all(Connection::is_alive));
    assert_ne!(clients[0].player_id(), clients[1].player_id());
}
