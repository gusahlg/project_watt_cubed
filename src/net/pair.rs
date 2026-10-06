//! Two clients on one headless server, so a mod can test a channel without a window.
use std::thread;
use std::time::Duration;

use voxel_engine::DVec3;

use crate::audio::ModLink;
use crate::world::generation::WorldgenKind;

use super::client::Connection;
use super::server::{self, Config};

/// A pair of connected clients who can see each other. Dropping it stops the server.
pub struct ChannelPair {
    server: server::ServerHandle,
    a: Connection,
    b: Connection,
}

impl ChannelPair {
    /// Flat world, empty password, seed 1. Both players are moved next to each other
    /// so each is in the other's interest set.
    pub fn open() -> Self {
        let server = server::spawn(
            0,
            Config {
                password: String::new(),
                seed: 1,
                worldgen: WorldgenKind::Flat,
                ..Config::default()
            },
        )
        .expect("loopback server");
        let port = server.addr().port();
        let mut a = Connection::connect("127.0.0.1", port, "a", "").expect("client a");
        let mut b = Connection::connect("127.0.0.1", port, "b", "").expect("client b");
        a.send_teleport(DVec3::new(8.0, 40.0, 8.0));
        b.send_teleport(DVec3::new(10.0, 40.0, 8.0));
        for _ in 0..20 {
            thread::sleep(Duration::from_millis(50));
            a.poll();
            b.poll();
            if a.peers().any(|peer| peer.visible()) && b.peers().any(|peer| peer.visible()) {
                return Self { server, a, b };
            }
        }
        panic!("the pair never became visible to each other");
    }

    pub fn id_a(&self) -> u32 {
        self.a.player_id()
    }

    pub fn id_b(&self) -> u32 {
        self.b.player_id()
    }

    pub fn with_a<R>(&mut self, body: impl FnOnce(ModLink<'_>) -> R) -> R {
        body(ModLink::new(Some(&mut self.a)))
    }

    pub fn with_b<R>(&mut self, body: impl FnOnce(ModLink<'_>) -> R) -> R {
        body(ModLink::new(Some(&mut self.b)))
    }

    pub fn pump(&mut self) {
        self.a.poll();
        self.b.poll();
    }
}

impl Drop for ChannelPair {
    fn drop(&mut self) {
        self.server.stop();
    }
}
