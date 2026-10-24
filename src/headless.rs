//! One player on a server, with the game's world mirror and physics and none of its window.
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use voxel_engine::{DVec3, Vec2};

use crate::block::BlockId;
use crate::input::movement::MoveInput;
use crate::interact::{self, REACH};
use crate::net::client::Connection;
use crate::play::{self, Cue, Host, PendingEdit};
use crate::player::Player;
use crate::presence::Stance;
use crate::render_config::RenderConfig;
use crate::world::World;

/// Chunks farther than this from the eye are dropped. The collision slab reaches about 48 blocks
/// on the diagonal, so a walk does not reload the ground every step.
const KEEP: f64 = 96.0;
/// Break and tool use. The break key's auto-repeat delay in `input/intent.rs`.
const BREAK_GAP: f32 = 0.25;
/// Place. The break/place auto-repeat interval in `input/intent.rs`.
const PLACE_GAP: f32 = 0.20;
/// Same budget as the server's chat window (`net/server` `CHAT_RATE`).
const CHAT_PER_SEC: usize = 5;

/// Why [`HeadlessClient::connect`] failed.
#[derive(Debug)]
pub struct ConnectError {
    message: String,
    /// Package ids the server refuses. Empty unless it denied the mod list.
    pub mods_denied: Vec<String>,
}

impl ConnectError {
    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            mods_denied: Vec::new(),
        }
    }
}

impl From<crate::net::client::ConnectError> for ConnectError {
    fn from(err: crate::net::client::ConnectError) -> Self {
        Self {
            message: err.message().to_string(),
            mods_denied: err.mods_denied,
        }
    }
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConnectError {}

/// What [`HeadlessClient::connect`] needs. `mods` is the enabled package list the server may refuse.
#[derive(Clone, Debug)]
pub struct JoinOptions {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub password: String,
    pub mods: Vec<(String, String)>,
    pub timeout: Duration,
}

/// One frame of intent. Axes are walking input in `-1..=1`. `turn` and `tilt` are radians this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Input {
    pub forward: f32,
    pub right: f32,
    pub jump: bool,
    pub sprint: bool,
    pub sneak: bool,
    pub turn: f32,
    pub tilt: f32,
    pub act: Option<Act>,
}

/// One action a player could take this frame.
#[derive(Clone, Debug, PartialEq)]
pub enum Act {
    Break,
    Place(BlockId),
    UseTool(BlockId),
    Swing,
    Chat { global: bool, text: String },
}

/// The block the eye is looking at, and the cell a placement would occupy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub cell: (i32, i32, i32),
    pub block: BlockId,
    /// Where a placed block goes (`raycast`'s previous cell).
    pub before: (i32, i32, i32),
}

/// Another player in the session.
#[derive(Clone, Debug, PartialEq)]
pub struct Peer {
    pub id: u32,
    pub name: Option<Arc<str>>,
    pub eye: DVec3,
}

/// What one [`HeadlessClient::step`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    pub events: Vec<Event>,
}

/// A fact from one frame. Predicted edits are reported when they happen; the server's verdict
/// arrives on a later frame.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Broke {
        cell: (i32, i32, i32),
        block: BlockId,
    },
    Placed {
        cell: (i32, i32, i32),
        block: BlockId,
    },
    EditConfirmed {
        cell: (i32, i32, i32),
    },
    EditRejected {
        cell: (i32, i32, i32),
    },
    ToolResult {
        cell: (i32, i32, i32),
        reacted: bool,
        cell_block: BlockId,
        tool_block: BlockId,
    },
    Refused(Refusal),
    SnappedBack {
        to: DVec3,
    },
    WorldChanged {
        cell: (i32, i32, i32),
        block: BlockId,
    },
    Chat {
        from: Arc<str>,
        global: bool,
        text: Arc<str>,
    },
    Joined {
        name: Arc<str>,
    },
    Left {
        name: Arc<str>,
    },
    Disconnected {
        reason: String,
    },
}

/// Why an action was not sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoTarget,
    Cooldown,
    NotHeld,
    /// The cell is solid or overlaps the body.
    Occupied,
}

/// A player joined to a server: the connection, the world mirror, the body, and the edits
/// still waiting on a verdict.
pub struct HeadlessClient {
    conn: Connection,
    world: World,
    player: Player,
    pending_edits: HashMap<u32, PendingEdit>,
    pending_tools: HashMap<u32, BlockId>,
    day: f32,
    break_cd: f32,
    place_cd: f32,
    chat_at: VecDeque<Instant>,
    disconnect_seen: bool,
}

impl HeadlessClient {
    /// Connects, builds the world mirror, applies the whole join snapshot, loads the collision slab
    /// around spawn. Returns once `step` can run. Blocking; call it from the agent's own thread.
    pub fn connect(opts: &JoinOptions) -> Result<HeadlessClient, ConnectError> {
        let deadline = Instant::now() + opts.timeout;
        let conn = connect_within(
            &opts.host,
            opts.port,
            &opts.name,
            &opts.password,
            &opts.mods,
            deadline,
        )?;
        let world = World::mirror(
            conn.seed(),
            RenderConfig::default(),
            conn.worldgen(),
            conn.terrain(),
        );
        let spawn = conn.spawn();
        let player = Player::standing(spawn, world.gravity_at(spawn).accel);
        let mut client = HeadlessClient {
            conn,
            world,
            player,
            pending_edits: HashMap::new(),
            pending_tools: HashMap::new(),
            day: 0.3,
            break_cd: 0.0,
            place_cd: 0.0,
            chat_at: VecDeque::new(),
            disconnect_seen: false,
        };
        client.drain_snapshot(deadline)?;
        client.world.keep_around(client.player.position, KEEP);
        Ok(client)
    }

    /// One frame: poll the network and apply every incoming message to the mirror (edits, mutations,
    /// acks/rollbacks, tool results, snap-back positions, chat, joins/leaves, time); apply look and
    /// movement through the real physics (`input::movement::update_player_in` with the atlas exactly
    /// as `Game::motion_phase` does, `align_body`, stance resolution); perform `input.act`; send the
    /// move. `dt` is clamped to 0.1 s like the game. Returns what happened this frame.
    pub fn step(&mut self, input: &Input, dt: f32) -> Step {
        let dt = if dt.is_finite() {
            dt.min(0.1).max(0.0)
        } else {
            0.0
        };
        self.break_cd = (self.break_cd - dt).max(0.0);
        self.place_cd = (self.place_cd - dt).max(0.0);
        let mut events = Vec::new();
        let incoming = self.conn.poll();
        let reason = {
            let mut host = HeadlessHost {
                events: &mut events,
                day: &mut self.day,
                record: true,
            };
            play::apply_incoming(
                &mut self.world,
                &mut self.player,
                &mut self.pending_edits,
                &mut self.pending_tools,
                Some(&self.conn),
                incoming,
                &mut host,
            )
        };
        self.note_disconnect(reason, &mut events);
        if self.conn.is_alive() {
            let turn = if input.turn.is_finite() {
                input.turn
            } else {
                0.0
            };
            let tilt = if input.tilt.is_finite() {
                input.tilt
            } else {
                0.0
            };
            self.player.orientation.look(Vec2::new(turn, tilt), 1.0);
            play::align_body(&mut self.player, dt);
            let move_input = MoveInput::walk(
                input.forward,
                input.right,
                input.jump,
                input.sprint,
                input.sneak,
            );
            let gravity = self.world.gravity_at(self.player.position).accel;
            // Landing shake stays on the windowed camera. The step itself is the game's.
            let _trauma = play::integrate(&mut self.player, &self.world, &move_input, dt, gravity);
            self.world.keep_around(self.player.position, KEEP);
            self.act(input, &mut events);
            self.send_pose();
        }
        Step { events }
    }

    pub fn player(&self) -> &Player {
        &self.player
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// The solid block along the eye's forward ray, within [`interact::REACH`].
    pub fn target(&self) -> Option<Target> {
        self.ray(self.player.forward(), REACH)
    }

    /// The first solid block along `dir` from the eye, within `reach` blocks. The same march as
    /// [`target`](Self::target): on a round world it runs through the patch under the eye, so the
    /// cells it names are storage cells (see [`cell_center`](Self::cell_center)).
    pub fn ray(&self, dir: DVec3, reach: f64) -> Option<Target> {
        let hit = interact::raycast(&self.world, self.player.position, dir, reach)?;
        let (x, y, z) = hit.block;
        Some(Target {
            cell: hit.block,
            block: self.world.block_at(x, y, z),
            before: hit.previous,
        })
    }

    /// Where a cell's centre is in world space. A round world's storage cell sits where its chart
    /// embeds it; any other cell is its own centre.
    pub fn cell_center(&self, cell: (i32, i32, i32)) -> DVec3 {
        crate::space::atlas::embed_cell(self.world.atlases(), cell)
            .unwrap_or_else(|| DVec3::new(cell.0 as f64 + 0.5, cell.1 as f64 + 0.5, cell.2 as f64 + 0.5))
    }

    pub fn peers(&self) -> Vec<Peer> {
        let now = Instant::now();
        self.conn
            .peers()
            .map(|peer| Peer {
                id: peer.id(),
                name: Some(Arc::clone(&peer.name)),
                eye: peer.sample(now).pos.0,
            })
            .collect()
    }

    pub fn player_id(&self) -> u32 {
        self.conn.player_id()
    }

    /// Last server time-of-day fraction, in `0.0..1.0`. `0.3` until the first clock message.
    pub fn time_of_day(&self) -> f32 {
        self.day
    }

    pub fn ping_ms(&self) -> Option<u32> {
        self.conn.ping_ms()
    }

    /// False after the server drops the connection.
    pub fn alive(&self) -> bool {
        self.conn.is_alive()
    }

    pub fn eye(&self) -> DVec3 {
        self.player.position
    }

    pub fn forward(&self) -> DVec3 {
        self.player.forward()
    }

    pub fn up(&self) -> DVec3 {
        self.player.up()
    }

    pub fn yaw(&self) -> f32 {
        self.player.orientation.yaw
    }

    pub fn pitch(&self) -> f32 {
        self.player.orientation.pitch
    }

    pub fn on_ground(&self) -> bool {
        self.player.on_ground()
    }

    fn drain_snapshot(&mut self, deadline: Instant) -> Result<(), ConnectError> {
        loop {
            if !self.conn.is_alive() {
                return Err(ConnectError::plain("disconnected while joining"));
            }
            let incoming = self.conn.poll();
            let mut discarded = Vec::new();
            let reason = {
                let mut host = HeadlessHost {
                    events: &mut discarded,
                    day: &mut self.day,
                    record: false,
                };
                play::apply_incoming(
                    &mut self.world,
                    &mut self.player,
                    &mut self.pending_edits,
                    &mut self.pending_tools,
                    Some(&self.conn),
                    incoming,
                    &mut host,
                )
            };
            if reason.is_some() || !self.conn.is_alive() {
                return Err(ConnectError::plain("disconnected while joining"));
            }
            if self.conn.snapshot_ready() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(ConnectError::plain("timed out waiting for the world"));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn note_disconnect(&mut self, reason: Option<String>, events: &mut Vec<Event>) {
        let reason = match reason {
            Some(reason) => Some(reason),
            None if !self.conn.is_alive() && !self.disconnect_seen => Some(String::new()),
            None => None,
        };
        let Some(reason) = reason else { return };
        if self.disconnect_seen {
            return;
        }
        self.disconnect_seen = true;
        events.push(Event::Disconnected { reason });
    }

    fn send_pose(&mut self) {
        let cruise = self.player.cruise.map(|c| c.speed);
        self.conn.sync_cruise(cruise);
        self.conn.send_move(
            self.player.position,
            self.player.orientation.yaw,
            self.player.orientation.pitch,
            self.player.orientation.frame,
            self.player.velocity().as_vec3(),
            self.player.up_axis,
            Stance::of_player(&self.player),
        );
    }

    fn act(&mut self, input: &Input, events: &mut Vec<Event>) {
        let Some(act) = input.act.clone() else { return };
        match act {
            Act::Chat { global, text } => self.say(global, &text, events),
            Act::Swing => self.conn.send_swing(),
            Act::Break => self.break_block(events),
            Act::Place(id) => self.place_block(id, events),
            Act::UseTool(id) => self.use_tool(id, events),
        }
    }

    fn say(&mut self, global: bool, text: &str, events: &mut Vec<Event>) {
        let now = Instant::now();
        while self
            .chat_at
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= Duration::from_secs(1))
        {
            self.chat_at.pop_front();
        }
        if self.chat_at.len() >= CHAT_PER_SEC {
            events.push(Event::Refused(Refusal::Cooldown));
            return;
        }
        self.chat_at.push_back(now);
        let channel = if global {
            crate::net::chat::GLOBAL
        } else {
            crate::net::chat::LOCAL
        };
        self.conn.send_chat(channel, text);
    }

    fn break_block(&mut self, events: &mut Vec<Event>) {
        if self.break_cd > 0.0 {
            events.push(Event::Refused(Refusal::Cooldown));
            return;
        }
        let Some(hit) = self.target() else {
            events.push(Event::Refused(Refusal::NoTarget));
            return;
        };
        self.break_cd = BREAK_GAP;
        let mut host = HeadlessHost {
            events,
            day: &mut self.day,
            record: true,
        };
        play::commit_break(
            &mut self.world,
            &mut self.player,
            &mut self.pending_edits,
            Some(&mut self.conn),
            hit.cell,
            &mut host,
        );
    }

    fn place_block(&mut self, id: BlockId, events: &mut Vec<Event>) {
        if self.place_cd > 0.0 {
            events.push(Event::Refused(Refusal::Cooldown));
            return;
        }
        let Some(hit) = self.target() else {
            events.push(Event::Refused(Refusal::NoTarget));
            return;
        };
        if self.player.inventory.count(id) == 0 {
            events.push(Event::Refused(Refusal::NotHeld));
            return;
        }
        if play::blocks_placement(&self.world, &self.player, hit.before) {
            events.push(Event::Refused(Refusal::Occupied));
            return;
        }
        if !self.player.inventory.consume(id, 1) {
            events.push(Event::Refused(Refusal::NotHeld));
            return;
        }
        self.place_cd = PLACE_GAP;
        let mut host = HeadlessHost {
            events,
            day: &mut self.day,
            record: true,
        };
        play::commit_place(
            &mut self.world,
            &mut self.player,
            &mut self.pending_edits,
            Some(&mut self.conn),
            hit.before,
            id,
            &mut host,
        );
    }

    fn use_tool(&mut self, id: BlockId, events: &mut Vec<Event>) {
        if self.break_cd > 0.0 {
            events.push(Event::Refused(Refusal::Cooldown));
            return;
        }
        let Some(hit) = self.target() else {
            events.push(Event::Refused(Refusal::NoTarget));
            return;
        };
        if self.player.inventory.count(id) == 0 {
            events.push(Event::Refused(Refusal::NotHeld));
            return;
        }
        self.break_cd = BREAK_GAP;
        let mut host = HeadlessHost {
            events,
            day: &mut self.day,
            record: true,
        };
        play::use_tool(
            &mut self.world,
            &mut self.player,
            &mut self.pending_tools,
            Some(&mut self.conn),
            id,
            hit.cell,
            &mut host,
        );
    }
}

struct HeadlessHost<'a> {
    events: &'a mut Vec<Event>,
    day: &'a mut f32,
    /// False while the join snapshot is applied: the world updates, no events are returned.
    record: bool,
}

impl Host for HeadlessHost<'_> {
    fn cue(&mut self, _world: &World, cue: Cue) {
        if let Cue::Time { day, .. } = cue {
            *self.day = (day as f64).rem_euclid(1.0) as f32;
            return;
        }
        if !self.record {
            return;
        }
        let event = match cue {
            Cue::Time { .. } => return,
            Cue::RemoteEdit { cell, id, .. } | Cue::Mutation { cell, id } => {
                Event::WorldChanged { cell, block: id }
            }
            Cue::Confirmed { cell } => Event::EditConfirmed { cell },
            Cue::Rejected { cell } => Event::EditRejected { cell },
            Cue::LocalBreak { cell, block, .. } => Event::Broke { cell, block },
            Cue::LocalPlace { cell, block, .. } => Event::Placed { cell, block },
            Cue::ToolDone {
                cell,
                reacted,
                cell_block,
                tool_block,
            } => Event::ToolResult {
                cell,
                reacted,
                cell_block,
                tool_block,
            },
            Cue::Chat {
                from,
                channel,
                text,
            } => Event::Chat {
                from,
                global: channel == crate::net::chat::GLOBAL,
                text,
            },
            Cue::Joined { name } => Event::Joined { name },
            Cue::Left { name } => Event::Left { name },
            Cue::Snapped { to } => Event::SnappedBack { to },
            Cue::BreakRejected { .. }
            | Cue::PlaceRejected { .. }
            | Cue::BreakNoted { .. }
            | Cue::Trauma(_)
            | Cue::AnimSwing
            | Cue::SwingAt { .. }
            | Cue::PeerSwing { .. }
            | Cue::ToolReacted { .. }
            | Cue::ToolNoReaction
            | Cue::ToolChanged { .. }
            | Cue::ToolUsed(_)
            | Cue::Interrupted => return,
        };
        self.events.push(event);
    }

    fn prepare(&mut self, world: &mut World, pos: DVec3) {
        // Twin of `GameHost::prepare`, which calls `World::prepare_around` and waits on the workers.
        world.keep_around(pos, KEEP);
    }
}

fn connect_within(
    host: &str,
    port: u16,
    name: &str,
    password: &str,
    mods: &[(String, String)],
    deadline: Instant,
) -> Result<Connection, ConnectError> {
    let mut pending = Connection::begin_connect(host, port, name, password, mods);
    loop {
        if let Some(result) = pending.poll() {
            return result.map_err(ConnectError::from);
        }
        if Instant::now() >= deadline {
            pending.cancel();
            return match pending.wait() {
                Ok(conn) => Ok(conn),
                Err(_) => Err(ConnectError::plain("timed out joining")),
            };
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::block::AIR;
    use crate::math::PER_METER;
    use crate::net::server::{self, Config, EditIntent, ServerMod, Verdict};
    use crate::world::generation::WorldgenKind;

    const DT: f32 = 1.0 / 60.0;

    struct Deny;

    impl ServerMod for Deny {
        fn id(&self) -> &'static str {
            "deny-edits"
        }
        fn validate_edit(&mut self, _: &EditIntent) -> Verdict {
            Verdict::Deny {
                reason: Arc::from("no"),
            }
        }
    }

    fn join(port: u16, name: &str) -> HeadlessClient {
        HeadlessClient::connect(&JoinOptions {
            host: "127.0.0.1".into(),
            port,
            name: name.into(),
            password: String::new(),
            mods: Vec::new(),
            timeout: Duration::from_secs(15),
        })
        .unwrap_or_else(|e| panic!("join {name}: {e}"))
    }

    fn flat(hooks: Vec<Box<dyn ServerMod>>) -> server::ServerHandle {
        server::spawn(
            0,
            Config {
                seed: 1,
                worldgen: WorldgenKind::Flat,
                hooks,
                ..Config::default()
            },
        )
        .expect("flat server")
    }

    /// Two players. One walks, breaks a block in reach, puts it back, and leaves.
    #[test]
    fn two_clients_walk_break_place_and_leave() {
        let server = flat(Vec::new());
        let port = server.addr().port();
        let mut ada = join(port, "ada");
        let mut bea = join(port, "bea");
        let mut snapped = false;
        // Wall time, not just simulated dt: `send_move` throttles on the clock, and the server
        // judges an edit against the last pose it actually accepted.
        for _ in 0..45 {
            snapped |= stepped(&mut ada, &Input::default()).iter().any(is_snap);
            bea.step(&Input::default(), DT);
            std::thread::sleep(Duration::from_secs_f32(DT));
        }
        let start = ada.eye();
        for _ in 0..120 {
            snapped |= stepped(
                &mut ada,
                &Input {
                    forward: 1.0,
                    ..Input::default()
                },
            )
            .iter()
            .any(is_snap);
            bea.step(&Input::default(), DT);
            std::thread::sleep(Duration::from_secs_f32(DT));
        }
        let moved = (ada.eye() - start).length();
        assert!(
            moved > 3.0 * PER_METER,
            "moved {moved} units ({:.2} m), on_ground {}",
            moved / PER_METER,
            ada.on_ground()
        );
        assert!(!snapped, "the walk was not snapped back");

        aim(&mut ada, &mut bea);
        let target = ada.target().unwrap_or_else(|| {
            panic!(
                "no block in reach, pitch {} eye {:?}",
                ada.pitch(),
                ada.eye()
            )
        });
        let (cell, id) = (target.cell, target.block);
        assert_eq!(ada.ray(ada.forward(), REACH), Some(target), "target is the forward ray");
        let centre = (ada.cell_center(cell) - ada.eye()).length();
        assert!(centre <= REACH + 1.0, "the target's centre is {centre} units from the eye");
        let before = ada.player().inventory.count(id);
        let broke = ada.step(
            &Input {
                act: Some(Act::Break),
                ..Input::default()
            },
            DT,
        );
        bea.step(&Input::default(), DT);
        assert!(
            broke.events.iter().any(
                |e| matches!(e, Event::Broke { cell: c, block } if *c == cell && *block == id)
            ),
            "expected Broke, got {:?}",
            broke.events
        );
        assert_eq!(
            ada.player().inventory.count(id),
            before + 1,
            "the break entered the inventory"
        );

        let (confirmed, changed, seen) = wait_pair(
            &mut ada,
            &mut bea,
            |ev| matches!(ev, Event::EditConfirmed { cell: c } if *c == cell),
            |ev| matches!(ev, Event::WorldChanged { cell: c, block } if *c == cell && *block == AIR),
        );
        assert!(
            confirmed,
            "the break was not confirmed; eye {:?} cell {cell:?} seen {seen:?}",
            ada.eye()
        );
        assert!(
            changed,
            "the other client did not see the break; seen {seen:?}"
        );

        let placed = place_held(&mut ada, &mut bea, id);
        let Some(placed_cell) = placed else {
            panic!(
                "the block was not placed, pitch {} eye {:?}",
                ada.pitch(),
                ada.eye()
            );
        };
        assert_eq!(
            ada.player().inventory.count(id),
            before,
            "placing spent the block"
        );
        let confirmed = pump(
            &mut ada,
            &mut bea,
            |ev| matches!(ev, Event::EditConfirmed { cell } if *cell == placed_cell),
        );
        assert!(confirmed, "the place was not confirmed");

        for _ in 0..30 {
            ada.step(&Input::default(), DT);
            bea.step(&Input::default(), DT);
        }
        let refused = ada.step(
            &Input {
                act: Some(Act::Place(id)),
                ..Input::default()
            },
            DT,
        );
        assert!(
            refused
                .events
                .iter()
                .any(|e| matches!(e, Event::Refused(Refusal::NotHeld))),
            "expected NotHeld, got {:?}",
            refused.events
        );
        assert_eq!(ada.player().inventory.count(id), before);

        drop(ada);
        let left = wait_one(
            &mut bea,
            |ev| matches!(ev, Event::Left { name } if name.as_ref() == "ada"),
        );
        assert!(left, "the other client did not see the leave");
        server.stop();
    }

    /// A server that denies every edit rolls the predicted break back.
    #[test]
    fn a_denied_edit_rolls_back() {
        let server = flat(vec![Box::new(Deny)]);
        let mut ada = join(server.addr().port(), "ada");
        let mut sink = Vec::new();
        for _ in 0..30 {
            ada.step(&Input::default(), DT);
        }
        aim_one(&mut ada);
        let target = ada.target().expect("a block in reach");
        let (cell, id) = (target.cell, target.block);
        let broke = ada.step(
            &Input {
                act: Some(Act::Break),
                ..Input::default()
            },
            DT,
        );
        assert!(
            broke
                .events
                .iter()
                .any(|e| matches!(e, Event::Broke { cell: c, .. } if *c == cell))
        );
        assert_eq!(ada.player().inventory.count(id), 1);
        let rejected = wait_one(&mut ada, |ev| {
            sink.push(ev.clone());
            matches!(ev, Event::EditRejected { cell: c } if *c == cell)
        });
        assert!(rejected, "no EditRejected in {sink:?}");
        assert_eq!(
            ada.world().block_at(cell.0, cell.1, cell.2),
            id,
            "the cell was restored"
        );
        assert_eq!(ada.player().inventory.count(id), 0, "the loot was revoked");
        server.stop();
    }

    fn is_snap(ev: &Event) -> bool {
        matches!(ev, Event::SnappedBack { .. })
    }

    fn stepped(client: &mut HeadlessClient, input: &Input) -> Vec<Event> {
        client.step(input, DT).events
    }

    fn aim(ada: &mut HeadlessClient, bea: &mut HeadlessClient) {
        for _ in 0..8 {
            ada.step(
                &Input {
                    tilt: -0.12,
                    ..Input::default()
                },
                DT,
            );
            bea.step(&Input::default(), DT);
            std::thread::sleep(Duration::from_secs_f32(DT));
        }
    }

    fn aim_one(ada: &mut HeadlessClient) {
        for _ in 0..8 {
            ada.step(
                &Input {
                    tilt: -0.12,
                    ..Input::default()
                },
                DT,
            );
            std::thread::sleep(Duration::from_secs_f32(DT));
        }
    }

    /// Place `id`, adjusting aim until a cell accepts it. The cell is where it landed.
    fn place_held(
        ada: &mut HeadlessClient,
        bea: &mut HeadlessClient,
        id: BlockId,
    ) -> Option<(i32, i32, i32)> {
        for i in 0..180 {
            let tilt = if i % 15 == 0 { 0.08 } else { 0.0 };
            let forward = if i % 20 == 0 { 0.4 } else { 0.0 };
            let step = ada.step(
                &Input {
                    tilt,
                    forward,
                    act: Some(Act::Place(id)),
                    ..Input::default()
                },
                DT,
            );
            bea.step(&Input::default(), DT);
            if let Some(Event::Placed { cell, .. }) = step
                .events
                .iter()
                .find(|e| matches!(e, Event::Placed { block, .. } if *block == id))
            {
                return Some(*cell);
            }
        }
        None
    }

    fn pump(
        ada: &mut HeadlessClient,
        bea: &mut HeadlessClient,
        mut pred: impl FnMut(&Event) -> bool,
    ) -> bool {
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            for ev in ada.step(&Input::default(), DT).events {
                if pred(&ev) {
                    return true;
                }
            }
            bea.step(&Input::default(), DT);
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    fn wait_one(client: &mut HeadlessClient, mut pred: impl FnMut(&Event) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            for ev in client.step(&Input::default(), DT).events {
                if pred(&ev) {
                    return true;
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    fn wait_pair(
        ada: &mut HeadlessClient,
        bea: &mut HeadlessClient,
        mut ada_pred: impl FnMut(&Event) -> bool,
        mut bea_pred: impl FnMut(&Event) -> bool,
    ) -> (bool, bool, Vec<Event>) {
        let deadline = Instant::now() + Duration::from_secs(8);
        let (mut a, mut b) = (false, false);
        let mut seen = Vec::new();
        while Instant::now() < deadline && !(a && b) {
            for ev in ada.step(&Input::default(), DT).events {
                a |= ada_pred(&ev);
                seen.push(ev);
            }
            for ev in bea.step(&Input::default(), DT).events {
                b |= bea_pred(&ev);
                seen.push(ev);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        (a, b, seen)
    }

    #[test]
    #[ignore = "resident set and thread count of one connected client; not part of the suite"]
    fn measure_resident_after_a_walk() {
        measure(WorldgenKind::Flat);
        measure(WorldgenKind::Diffusion);
    }

    /// Id 12 stands on the unscattered chart column. The other ids of the 5×5 patch are
    /// shifted a block or two, which on this seed is solid rock, so a walker there cannot move.
    fn take_open_column(port: u16) {
        let mut seats = Vec::new();
        for i in 1..12 {
            seats.push(
                crate::net::client::Connection::connect("127.0.0.1", port, &format!("s{i}"), "")
                    .unwrap_or_else(|e| panic!("seat {i}: {e}")),
            );
        }
        drop(seats);
        std::thread::sleep(Duration::from_millis(300));
    }

    fn measure(kind: WorldgenKind) {
        let (rss0, threads0) = resident();
        let server = server::spawn(
            0,
            Config {
                seed: 1,
                worldgen: kind,
                ..Config::default()
            },
        )
        .expect("server");
        let (rss_server, threads_server) = resident();
        let port = server.addr().port();
        if kind == WorldgenKind::Diffusion {
            take_open_column(port);
        }
        let mut ada = join(port, "ada");
        let (rss_one, threads_one) = resident();
        let bea = join(port, "bea");
        let (_rss_two, threads_two) = resident();
        let chunks_at_spawn = ada.world().stream_gauges().chunks;
        let start = ada.eye();
        let began = Instant::now();
        let goal = 200.0 * PER_METER;
        // Flat ground takes a 0.1 s sprint step. A chart's slopes stick at that step, so diffusion
        // walks at the game's 60 Hz. Sleep matches dt: `send_move` throttles on the wall clock.
        let (dt, pause_ms, step_cap, time_cap) = if kind == WorldgenKind::Diffusion {
            (1.0 / 60.0, 16u64, 20_000u32, 150u64)
        } else {
            (0.1, 100, 4_000, 180)
        };
        let stall_steps = (2.0 / dt) as u32;
        let mut steps = 0u32;
        let mut snaps = 0u32;
        let mut turn = 0.0f32;
        let mut mark = start;
        let mut since = 0u32;
        while steps < step_cap && began.elapsed() < Duration::from_secs(time_cap) {
            let step = ada.step(
                &Input {
                    forward: 1.0,
                    sprint: true,
                    turn,
                    ..Input::default()
                },
                dt,
            );
            turn = 0.0;
            snaps += step
                .events
                .iter()
                .filter(|e| matches!(e, Event::SnappedBack { .. }))
                .count() as u32;
            steps += 1;
            since += 1;
            std::thread::sleep(Duration::from_millis(pause_ms));
            if since >= stall_steps {
                let here = ada.eye();
                let dx = here.x - mark.x;
                let dz = here.z - mark.z;
                if (dx * dx + dz * dz).sqrt() < PER_METER {
                    turn = 0.6;
                }
                mark = here;
                since = 0;
            }
            let dx = ada.eye().x - start.x;
            let dz = ada.eye().z - start.z;
            if (dx * dx + dz * dz).sqrt() >= goal {
                break;
            }
        }
        let dx = ada.eye().x - start.x;
        let dz = ada.eye().z - start.z;
        let metres = (dx * dx + dz * dz).sqrt() / PER_METER;
        let (rss_walk, threads_walk) = resident();
        let chunks = ada.world().stream_gauges().chunks;
        eprintln!(
            "HEADLESS_MEASURE kind={kind:?} id={} dt={dt} rss_before={rss0}KiB rss_server={rss_server}KiB rss_one={rss_one}KiB rss_walk={rss_walk}KiB threads_before={threads0} threads_server={threads_server} threads_one={threads_one} threads_two={threads_two} threads_walk={threads_walk} chunks_spawn={chunks_at_spawn} chunks_walk={chunks} metres={metres:.1} steps={steps} snaps={snaps} elapsed={:.1}s on_ground={}",
            ada.player_id(),
            began.elapsed().as_secs_f32(),
            ada.on_ground()
        );
        drop(bea);
        drop(ada);
        server.stop();
    }

    fn resident() -> (u64, u64) {
        let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut rss = 0u64;
        let mut threads = 0u64;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Threads:") {
                threads = rest.trim().parse().unwrap_or(0);
            }
        }
        (rss, threads)
    }
}
