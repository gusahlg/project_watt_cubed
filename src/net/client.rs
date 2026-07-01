//! The client side of multiplayer: a [`Connection`] the [`Game`](crate::game) owns
//! while playing on a server. It hides the socket behind a small, poll-based API —
//! the game hands it the local player each frame, drains the events it needs to act
//! on (world edits and chat), and reads the peer table to draw everyone else.
//!
//! A background thread does the blocking reads and feeds a channel, so the render
//! loop never stalls on the network. Sends happen inline from the game thread (they
//! are tiny and infrequent). Position sends are throttled and heartbeat so a
//! standing-still player still proves they are alive without spamming the wire.
use std::collections::HashMap;
use std::io;
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use raylib::prelude::*;

use crate::net::protocol::{self, ClientMessage, ServerMessage};
use crate::net::{MAX_CHAT, MAX_SPEC, PROTOCOL_VERSION};

/// How long to wait for the initial TCP connect and the server's `Welcome`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Fastest cadence position updates are sent at (~30 Hz), even while moving.
const MOVE_INTERVAL: Duration = Duration::from_millis(33);
/// A move is sent at least this often even when standing still, as a heartbeat so
/// the server's idle timeout never reaps an active-but-idle player.
const HEARTBEAT: Duration = Duration::from_secs(1);

/// Another player as this client last heard about them — enough to draw them.
pub struct RemotePlayer {
    pub name: String,
    pub pos: Vector3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Something from the server the game must act on. Peer presence and movement are
/// applied inside [`Connection::poll`]; these are what the game still has to handle.
pub enum Incoming {
    /// A block changed somewhere — apply it to the local world overlay.
    Edit { x: i32, y: i32, z: i32, spec: String },
    /// A chat line to show in the console.
    Chat { from_name: String, channel: u8, text: String },
    /// The server dropped us; the game should leave the world.
    Disconnected,
}

/// A live connection to a server. Dropping it closes the socket, which ends the
/// reader thread and signals the server that this player left.
pub struct Connection {
    stream: TcpStream,
    inbox: Receiver<ServerMessage>,
    player_id: u32,
    seed: i64,
    spawn: Vector3,
    peers: HashMap<u32, RemotePlayer>,
    alive: bool,
    // Throttling state for outbound moves.
    last_move: Instant,
    last_sent: Option<(Vector3, f32, f32)>,
}

impl Connection {
    /// Dial `host:port`, authenticate with `name`/`password`, and return the ready
    /// connection once the server's `Welcome` arrives. `Err` carries a human-readable
    /// reason (bad address, refused, wrong password, version mismatch).
    pub fn connect(host: &str, port: u16, name: &str, password: &str) -> Result<Self, String> {
        let addr = (host, port)
            .to_socket_addrs()
            .map_err(|e| format!("bad address: {e}"))?
            .next()
            .ok_or_else(|| "address resolved to nothing".to_string())?;

        let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
            .map_err(|e| format!("could not reach {addr}: {e}"))?;
        stream.set_nodelay(true).ok();

        // Send the handshake and wait, briefly, for the reply.
        let hello = ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            name: name.to_string(),
            password: password.to_string(),
        };
        write(&stream, &hello).map_err(|e| format!("send failed: {e}"))?;

        stream.set_read_timeout(Some(CONNECT_TIMEOUT)).ok();
        let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
        let frame = protocol::read_frame(&mut reader).map_err(|e| format!("no reply: {e}"))?;
        let (player_id, seed, spawn) = match ServerMessage::decode(&frame) {
            Some(ServerMessage::Welcome { player_id, seed, spawn }) => (player_id, seed, spawn),
            Some(ServerMessage::Reject { reason }) => return Err(reason),
            _ => return Err("unexpected reply from server".to_string()),
        };

        // Handshake done: reads now block indefinitely on the background thread.
        stream.set_read_timeout(None).ok();
        let (tx, inbox) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(frame) = protocol::read_frame(&mut reader) {
                match ServerMessage::decode(&frame) {
                    Some(msg) => {
                        if tx.send(msg).is_err() {
                            break; // The game side hung up.
                        }
                    }
                    None => continue, // Skip a junk frame rather than tear down.
                }
            }
        });

        Ok(Self {
            stream,
            inbox,
            player_id,
            seed,
            spawn,
            peers: HashMap::new(),
            alive: true,
            last_move: Instant::now(),
            last_sent: None,
        })
    }

    /// The world seed to generate terrain from.
    pub fn seed(&self) -> i64 {
        self.seed
    }
    /// Where the server placed this player.
    pub fn spawn(&self) -> Vector3 {
        self.spawn
    }
    /// This player's server-assigned id.
    pub fn player_id(&self) -> u32 {
        self.player_id
    }
    /// Whether the connection is still up.
    pub fn is_alive(&self) -> bool {
        self.alive
    }
    /// The other players currently known, for rendering.
    pub fn peers(&self) -> impl Iterator<Item = &RemotePlayer> {
        self.peers.values()
    }

    /// Drain everything the server has said since the last frame. Peer join/leave/
    /// move is applied to the local table here; edits and chat are returned for the
    /// game to handle.
    pub fn poll(&mut self) -> Vec<Incoming> {
        let mut out = Vec::new();
        loop {
            match self.inbox.try_recv() {
                Ok(msg) => self.apply(msg, &mut out),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.alive {
                        self.alive = false;
                        out.push(Incoming::Disconnected);
                    }
                    break;
                }
            }
        }
        out
    }

    /// Fold one server message into the peer table or the game's event list.
    fn apply(&mut self, msg: ServerMessage, out: &mut Vec<Incoming>) {
        match msg {
            ServerMessage::Snapshot { edits } => {
                for (x, y, z, spec) in edits {
                    out.push(Incoming::Edit { x, y, z, spec });
                }
            }
            ServerMessage::Edit { x, y, z, spec } => out.push(Incoming::Edit { x, y, z, spec }),
            ServerMessage::Chat { from_name, channel, text, .. } => {
                out.push(Incoming::Chat { from_name, channel, text })
            }
            ServerMessage::PeerJoined { id, name } => {
                self.peers.entry(id).or_insert(RemotePlayer {
                    name,
                    pos: Vector3::zero(),
                    yaw: 0.0,
                    pitch: 0.0,
                });
            }
            ServerMessage::PeerLeft { id } => {
                self.peers.remove(&id);
            }
            ServerMessage::PeerMove { id, pos, yaw, pitch } => {
                if let Some(p) = self.peers.get_mut(&id) {
                    p.pos = pos;
                    p.yaw = yaw;
                    p.pitch = pitch;
                }
            }
            ServerMessage::Reject { reason: _ } => {
                self.alive = false;
                out.push(Incoming::Disconnected);
            }
            // A second Welcome is meaningless mid-session.
            ServerMessage::Welcome { .. } => {}
        }
    }

    /// Report the local player's state, throttled and heartbeat. Cheap to call every
    /// frame; it only actually sends on the movement cadence or the heartbeat.
    pub fn send_move(&mut self, pos: Vector3, yaw: f32, pitch: f32) {
        if !self.alive {
            return;
        }
        let elapsed = self.last_move.elapsed();
        let changed = self.last_sent != Some((pos, yaw, pitch));
        let due = (changed && elapsed >= MOVE_INTERVAL) || elapsed >= HEARTBEAT;
        if !due {
            return;
        }
        self.last_move = Instant::now();
        self.last_sent = Some((pos, yaw, pitch));
        self.dispatch(&ClientMessage::Move { pos, yaw, pitch });
    }

    /// Tell the server about a block the player changed.
    pub fn send_edit(&mut self, x: i32, y: i32, z: i32, spec: String) {
        if spec.len() > MAX_SPEC {
            return;
        }
        self.dispatch(&ClientMessage::Edit { x, y, z, spec });
    }

    /// Send a chat line on the given channel.
    pub fn send_chat(&mut self, channel: u8, text: String) {
        let text: String = text.chars().take(MAX_CHAT).collect();
        self.dispatch(&ClientMessage::Chat { channel, text });
    }

    /// Write one message, marking the connection dead if the socket errors.
    fn dispatch(&mut self, msg: &ClientMessage) {
        if self.alive && write(&self.stream, msg).is_err() {
            self.alive = false;
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Closing the socket ends the reader thread and tells the server we left.
        let _ = self.stream.shutdown(Shutdown::Both);
    }
}

/// Frame and write one client message to the stream (usable from a shared borrow,
/// since `&TcpStream` implements `Write`).
fn write(stream: &TcpStream, msg: &ClientMessage) -> io::Result<()> {
    let mut w = stream;
    protocol::write_frame(&mut w, &msg.encode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::server::{self, Config};

    /// End-to-end over loopback: two clients on one server see each other join, sync
    /// an edit, and exchange chat. Exercises the real socket path, not just the codec.
    #[test]
    fn two_clients_sync_over_loopback() {
        let handle = server::spawn(
            0,
            Config { password: "pw".into(), seed: 4242 },
        )
        .unwrap();
        let port = handle.addr().port();

        let mut a = Connection::connect("127.0.0.1", port, "alice", "pw").unwrap();
        let mut b = Connection::connect("127.0.0.1", port, "bob", "pw").unwrap();
        assert_eq!(a.seed(), 4242);
        assert_eq!(b.seed(), 4242);
        assert_ne!(a.player_id(), b.player_id());

        // Give the join broadcasts time to land, then poll them in.
        thread::sleep(Duration::from_millis(150));
        a.poll();
        b.poll();
        assert_eq!(a.peers().count(), 1, "alice should see bob");
        assert_eq!(b.peers().count(), 1, "bob should see alice");

        // Alice edits a block right next to her spawn; bob should receive it.
        let s = a.spawn();
        let (bx, by, bz) = (s.x.floor() as i32, s.y.floor() as i32, s.z.floor() as i32);
        // Report position so the server's reach check passes, then edit.
        a.last_move = Instant::now() - HEARTBEAT; // force the throttle to send
        a.send_move(s, 0.0, 0.0);
        a.send_edit(bx, by, bz, "air".into());

        thread::sleep(Duration::from_millis(150));
        let events = b.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::Edit { x, y, z, .. } if (*x, *y, *z) == (bx, by, bz))),
            "bob should receive alice's edit"
        );

        // Global chat reaches everyone regardless of distance.
        a.send_chat(crate::net::chat::GLOBAL, "hello".into());
        thread::sleep(Duration::from_millis(150));
        let events = b.poll();
        assert!(
            events.iter().any(|e| matches!(e, Incoming::Chat { text, .. } if text == "hello")),
            "bob should receive alice's global chat"
        );

        handle.stop();
    }

    #[test]
    fn wrong_password_is_rejected() {
        let handle = server::spawn(0, Config { password: "secret".into(), seed: 1 }).unwrap();
        let port = handle.addr().port();
        let err = match Connection::connect("127.0.0.1", port, "eve", "guess") {
            Ok(_) => panic!("a wrong password must be refused"),
            Err(e) => e,
        };
        assert!(err.to_lowercase().contains("password"), "got: {err}");
        handle.stop();
    }
}
