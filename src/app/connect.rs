//! Joining a server off the render thread: one attempt behind the connecting screen, and the
//! single retry a mod refusal earns. The mod list a client reports is what an honest client
//! says; a modified client can lie. One refusal turns those packages off for this session and
//! tries again; a second failure turns them back on. Nothing here writes `mods.cfg`.
use voxel_engine::Engine;

use super::{App, Screen, fresh_seed, terrain_cfg_from_mods};
use super::entry::Loading;
use crate::menu::{HostInfo, JoinInfo};
use crate::modding::{ModDescriptor, Mods};
use crate::net::client::{Connection, PendingConnect};
use crate::net::server::{Config, NoclipPolicy, TeleportPolicy};

/// One join attempt and where it goes. A mod refusal restarts it once, with `retried` set.
pub(super) struct ConnectJob {
    pending: PendingConnect,
    /// The server is the integrated one this app started.
    hosted: bool,
    retried: bool,
    host: String,
    port: u16,
    name: String,
    password: String,
    /// Shown once the attempt joins: a skipped save, or mods held for the session.
    notice: Option<String>,
}

/// How a finished attempt ends.
pub(super) enum Landed {
    /// In: enter the world with `notice`. `hosted` says the integrated server is serving it.
    Joined { conn: Connection, notice: Option<String>, hosted: bool },
    /// Out: the menu line saying why.
    Failed(String),
}

/// The retry decision: a refusal that names mods earns exactly one more attempt, with those
/// mods held off. Any other failure, or a second refusal, ends the join.
pub(super) fn retry_after(mods_denied: &[String], retried: bool) -> bool {
    !mods_denied.is_empty() && !retried
}

impl ConnectJob {
    /// Start joining `host:port`, reporting the packages `mods` has enabled.
    pub(super) fn begin(
        target: (&str, u16, &str, &str),
        mods: &Mods,
        packages: &[ModDescriptor],
        hosted: bool,
        notice: Option<String>,
    ) -> Self {
        let (host, port, name, password) = target;
        let pending = Connection::begin_connect(host, port, name, password, &mods.enabled_package_reports(packages));
        Self {
            pending,
            hosted,
            retried: false,
            host: host.to_string(),
            port,
            name: name.to_string(),
            password: password.to_string(),
            notice,
        }
    }

    /// `None` while the attempt runs, and after a mod refusal has started the retry: the denied
    /// packages are held off in `mods` and the hold notice replaces any earlier one. A failure
    /// after the retry lifts the hold again.
    pub(super) fn poll(&mut self, mods: &mut Mods, packages: &[ModDescriptor]) -> Option<Landed> {
        match self.pending.poll()? {
            Ok(conn) => Some(Landed::Joined { conn, notice: self.notice.take(), hosted: self.hosted }),
            Err(err) if retry_after(&err.mods_denied, self.retried) => {
                self.notice = Some(mod_hold_notice(packages, &err.mods_denied));
                mods.hold_packages(&err.mods_denied);
                let reports = mods.enabled_package_reports(packages);
                self.pending = Connection::begin_connect(&self.host, self.port, &self.name, &self.password, &reports);
                self.retried = true;
                None
            }
            Err(err) => {
                if self.retried {
                    mods.release_server();
                }
                Some(Landed::Failed(if self.hosted {
                    format!("hosted, but could not connect: {err}")
                } else {
                    format!("could not join: {err}")
                }))
            }
        }
    }

    pub(super) fn cancel(&self) {
        self.pending.cancel();
    }

    pub(super) fn hosted(&self) -> bool {
        self.hosted
    }
}

/// "This server does not allow: Developer Toolkit; it is off while you are connected".
/// Several names use "they are". Display names come from the build; an unknown
/// id is shown as itself.
fn mod_hold_notice(packages: &[ModDescriptor], ids: &[String]) -> String {
    let names: Vec<&str> = ids
        .iter()
        .map(|id| packages.iter().find(|pkg| pkg.id == id).map(|pkg| pkg.name).unwrap_or(id.as_str()))
        .collect();
    let list = names.join("; ");
    if names.len() == 1 {
        format!("This server does not allow: {list}; it is off while you are connected")
    } else {
        format!("This server does not allow: {list}; they are off while you are connected")
    }
}

impl App {
    /// Spin up the integrated server on the newest save it can load and join it on
    /// loopback. Any previous host is stopped first so its port is free. The host is
    /// an operator and teleport stays open. A stored seed and generator win.
    pub(super) fn start_host(&mut self, info: HostInfo) {
        debug_assert!(self.active.is_none(), "hosting starts from the menu, never over an open world");
        let worldgen = self.mods.worldgen_kind();
        let terrain = terrain_cfg_from_mods(&self.mods);
        let started = self.host.start(&self.saves, info.port, |world| Config {
            password: info.password.clone(),
            seed: fresh_seed(),
            worldgen,
            terrain,
            teleport: TeleportPolicy::All,
            noclip: NoclipPolicy::All,
            world: Some(world),
            ops: vec![info.name.clone()],
            warn_world_overrides: false,
            ..Config::default()
        });
        match started {
            Ok((port, skipped)) => {
                let job = ConnectJob::begin(("127.0.0.1", port, &info.name, &info.password), &self.mods, &self.packages, true, skipped);
                self.screen = Screen::Connecting(job);
            }
            Err(e) => self.fail_to_menu(format!("could not host on port {}: {e}", info.port)),
        }
    }

    /// Connect to a remote server. The attempt runs behind the connecting screen.
    pub(super) fn start_join(&mut self, info: JoinInfo) {
        let job = ConnectJob::begin((&info.host, info.port, &info.name, &info.password), &self.mods, &self.packages, false, None);
        self.screen = Screen::Connecting(job);
    }

    /// Poll the attempt. Cancel returns to the menu and stops a host we started.
    /// A mod refusal retries once; any other failure leaves a spawned host running.
    pub(super) fn update_connecting(&mut self, eng: &mut Engine) {
        if self.cancel_pressed(eng) {
            self.cancel_connect();
            return;
        }
        let Screen::Connecting(job) = &mut self.screen else { return };
        let Some(landed) = job.poll(&mut self.mods, &self.packages) else { return };
        self.screen = Screen::Menus(self.standby_menu());
        match landed {
            Landed::Joined { conn, notice, hosted } => {
                let render = self.mods.effective_render(&self.settings);
                self.begin_loading(eng, Loading::join(conn, render, notice, hosted));
            }
            Landed::Failed(text) => self.fail_to_menu(text),
        }
    }

    fn cancel_connect(&mut self) {
        let hosted = match &self.screen {
            Screen::Connecting(job) => {
                job.cancel();
                job.hosted()
            }
            _ => false,
        };
        if hosted {
            self.host.stop();
        }
        self.return_to_menu(None);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::menu::ModRow;
    use crate::modding::GameBuild;
    use crate::net::server;
    use crate::session::Session;
    use crate::settings::Settings;
    use crate::world::generation::WorldgenKind;

    struct Named(&'static str, &'static str);

    impl crate::modding::Mod for Named {
        fn id(&self) -> &'static str {
            self.0
        }
        fn name(&self) -> &str {
            self.1
        }
    }

    fn register_toolkit(reg: &mut crate::modding::ModRegistrar) {
        reg.add(Named("dev-toolkit", "Developer Toolkit"));
    }

    fn register_hotbar(reg: &mut crate::modding::ModRegistrar) {
        reg.add(Named("hotbar", "Hotbar"));
    }

    fn sample_packages() -> [ModDescriptor; 2] {
        [
            ModDescriptor { id: "pwc.dev-toolkit", name: "Developer Toolkit", version: "1.0.0", register: register_toolkit },
            ModDescriptor { id: "pwc.hotbar", name: "Hotbar", version: "0.1.0", register: register_hotbar },
        ]
    }

    const TOOLKIT_HELD: &str = "This server does not allow: Developer Toolkit; it is off while you are connected";

    /// The mods of a client built with both sample packages, and the toolkit's index.
    fn sample_mods(packages: &[ModDescriptor; 2]) -> (Mods, usize) {
        let mods = GameBuild::new().with_mod(packages[0]).with_mod(packages[1]).mods();
        let toolkit = (0..mods.len()).find(|&i| mods.id(i) == "dev-toolkit").expect("toolkit");
        (mods, toolkit)
    }

    /// Poll `job` as the connecting screen does, every frame, until it lands.
    fn land(job: &mut ConnectJob, mods: &mut Mods, packages: &[ModDescriptor]) -> Landed {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(landed) = job.poll(mods, packages) {
                return landed;
            }
            assert!(Instant::now() < deadline, "the join never landed");
            std::thread::sleep(Duration::from_millis(8));
        }
    }

    #[test]
    fn mod_hold_notice_names_one_and_several() {
        let packages = sample_packages();
        assert_eq!(mod_hold_notice(&packages, &["pwc.dev-toolkit".into()]), TOOLKIT_HELD);
        assert_eq!(
            mod_hold_notice(&packages, &["pwc.dev-toolkit".into(), "pwc.hotbar".into()]),
            "This server does not allow: Developer Toolkit; Hotbar; they are off while you are connected"
        );
        assert_eq!(
            mod_hold_notice(&packages, &["pwc.unknown".into()]),
            "This server does not allow: pwc.unknown; it is off while you are connected"
        );
    }

    /// The whole decision table the connecting screen runs: only a first refusal that names
    /// mods earns a retry.
    #[test]
    fn only_a_first_mod_refusal_earns_a_retry() {
        let refused = ["pwc.dev-toolkit".to_string()];
        let table: [(&[String], bool, bool); 4] = [(&refused, false, true), (&refused, true, false), (&[], false, false), (&[], true, false)];
        for (denied, retried, retry) in table {
            assert_eq!(retry_after(denied, retried), retry, "denied {denied:?} retried {retried}");
        }
    }

    /// The server refuses the toolkit, the job turns it off and joins once, and the Mods menu
    /// will not turn it back on while that hold lasts.
    #[test]
    fn denied_mod_is_disabled_for_the_session_and_the_menu_cannot_reenable_it() {
        use crate::menu::menus::ModsMenu;
        use crate::menu::{Command, Menu, Msg, ValueView};
        let packages = sample_packages();
        let (mut mods, index) = sample_mods(&packages);
        let config = Config { seed: 1, worldgen: WorldgenKind::Flat, mods_deny: vec!["pwc.dev-toolkit".into()], ..Config::default() };
        let handle = server::spawn(0, config).unwrap();
        let mut job = ConnectJob::begin(("127.0.0.1", handle.addr().port(), "ada", ""), &mods, &packages, false, None);
        let Landed::Joined { conn, notice, hosted } = land(&mut job, &mut mods, &packages) else { panic!("the retry joins") };
        assert!(conn.is_alive() && !hosted);
        assert_eq!(notice.as_deref(), Some(TOOLKIT_HELD));
        assert!(!mods.is_enabled(index));
        assert!(mods.server_off(index));
        assert!(!mods.toggle(index), "the menu's toggle is refused while connected");
        let hotbar = (0..mods.len()).find(|&i| mods.id(i) == "hotbar").expect("hotbar");
        assert!(mods.is_enabled(hotbar));
        let snap = ModRow::snapshot(&mods);
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = crate::menu::Ctx { settings: &mut settings, saves: &[], mods: &snap, session: &session, mods_save_error: None };
        let view = ModsMenu.view(&ctx);
        let row = view.rows.iter().find(|row| row.label.contains("Developer Toolkit")).expect("row");
        match &row.kind {
            crate::menu::RowKind::Value(ValueView::Choice(value)) => assert_eq!(value, "off (server)"),
            _ => panic!("expected off (server)"),
        }
        assert!(matches!(ModsMenu.update(Msg::Pick(crate::menu::menus::ModsAction::ServerOff), &mut ctx), Command::Stay));
        drop(conn);
        mods.release_server();
        assert!(mods.is_enabled(index));
        assert!(!mods.server_off(index));
        handle.stop();
    }

    /// A release `watt_server` process, killed with SIGTERM on drop.
    struct ServerProcess(std::process::Child);

    impl ServerProcess {
        /// Start `bin` on loopback `port`: a flat world that denies the toolkit, with its data
        /// root in `data`.
        fn start(bin: &std::path::Path, port: u16, data: &std::path::Path) -> Self {
            let child = std::process::Command::new(bin)
                .args(["--port", &port.to_string(), "--seed", "5", "--worldgen", "flat", "--mods-deny", "pwc.dev-toolkit"])
                .arg("--data-dir")
                .arg(data)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("watt_server starts");
            Self(child)
        }

        /// SIGTERM, the way an operator stops it, then wait for the exit.
        fn stop(&mut self) {
            if let Ok(None) = self.0.try_wait() {
                unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
                let _ = self.0.wait();
            }
        }
    }

    impl Drop for ServerProcess {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// Join as the connecting screen does until the server answers (it may still be starting).
    fn join_when_up(port: u16, name: &str, mods: &mut Mods, packages: &[ModDescriptor]) -> (Connection, Option<String>) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let mut job = ConnectJob::begin(("127.0.0.1", port, name, ""), mods, packages, false, None);
            match land(&mut job, mods, packages) {
                Landed::Joined { conn, notice, .. } => return (conn, notice),
                Landed::Failed(why) => assert!(Instant::now() < deadline, "the server never answered: {why}"),
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Poll until the join overlay is in, or the link reports why it ended.
    fn until_ready(conn: &mut Connection) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !conn.snapshot_ready() {
            assert!(conn.is_alive() && Instant::now() < deadline, "the join overlay never landed");
            conn.poll();
            std::thread::sleep(Duration::from_millis(8));
        }
    }

    /// The release `watt_server` process on a loopback port, joined the way the connecting screen
    /// joins: the toolkit is refused and the retry gets in, an edit is answered, the player leaves
    /// and joins again under the same name, the server is stopped (the client hears why), and a
    /// new server process on the same port takes the reconnect. Build the binary first with
    /// `cargo build --release --bin watt_server` into the same target dir, or name it with
    /// `WATT_SERVER_BIN`.
    #[test]
    #[ignore]
    fn the_release_server_process_takes_a_mods_retry_a_rejoin_and_a_reconnect() {
        let bin = std::env::var_os("WATT_SERVER_BIN").map(std::path::PathBuf::from).unwrap_or_else(|| {
            let exe = std::env::current_exe().expect("test binary path");
            exe.parent().and_then(|deps| deps.parent()).expect("target profile dir").join("watt_server")
        });
        assert!(bin.exists(), "no {}: cargo build --release --bin watt_server", bin.display());
        let port = std::net::UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let data = std::env::temp_dir().join(format!("pwc-connect-process-{}", std::process::id()));
        let mut server = ServerProcess::start(&bin, port, &data);
        let packages = sample_packages();
        let (mut mods, toolkit) = sample_mods(&packages);

        // A refusal, then the retry with the toolkit held.
        let (mut conn, notice) = join_when_up(port, "ada", &mut mods, &packages);
        assert_eq!(notice.as_deref(), Some(TOOLKIT_HELD));
        assert!(mods.server_off(toolkit));
        until_ready(&mut conn);
        let s = conn.spawn();
        let cell = (crate::math::block_coord(s.x), crate::math::block_coord(s.y) - 3, crate::math::block_coord(s.z));
        let req = conn.send_edit(cell.0, cell.1, cell.2, "air".into()).expect("sent");
        let deadline = Instant::now() + Duration::from_secs(5);
        let answered = loop {
            let events = conn.poll();
            if let Some(accepted) = events.iter().find_map(|e| match e {
                crate::net::client::Incoming::EditAccepted { req: r } if *r == req => Some(true),
                crate::net::client::Incoming::EditRejected { req: r, .. } if *r == req => Some(false),
                _ => None,
            }) {
                break accepted;
            }
            assert!(Instant::now() < deadline, "the edit was never answered");
            std::thread::sleep(Duration::from_millis(8));
        };
        assert!(answered, "a block under the spawn is in reach");

        // Leave, and the name is free again for the same player.
        drop(conn);
        mods.release_server();
        assert!(!mods.server_off(toolkit));
        let (mut conn, notice) = join_when_up(port, "ada", &mut mods, &packages);
        assert_eq!(notice.as_deref(), Some(TOOLKIT_HELD));
        until_ready(&mut conn);

        // The operator stops the server: the client says why, once.
        server.stop();
        let deadline = Instant::now() + Duration::from_secs(10);
        let reason = loop {
            if let Some(reason) = conn.poll().into_iter().find_map(|e| match e {
                crate::net::client::Incoming::Disconnected { reason } => Some(reason),
                _ => None,
            }) {
                break reason;
            }
            assert!(Instant::now() < deadline, "the client never heard the server stop");
            std::thread::sleep(Duration::from_millis(8));
        };
        assert_eq!(reason, "server shutting down");
        assert!(!conn.is_alive());
        drop(conn);
        mods.release_server();

        // A new server process on the same port takes the reconnect.
        let _server = ServerProcess::start(&bin, port, &data);
        let (mut conn, notice) = join_when_up(port, "ada", &mut mods, &packages);
        assert_eq!(notice.as_deref(), Some(TOOLKIT_HELD));
        until_ready(&mut conn);
        assert!(conn.is_alive());
        drop(conn);
        let _ = std::fs::remove_dir_all(&data);
    }

    /// A refusal, then a retry that fails for another reason (the name is taken): the join
    /// ends with that reason and the held mods come back on.
    #[test]
    fn a_failed_retry_releases_the_held_mods() {
        let packages = sample_packages();
        let (mut mods, index) = sample_mods(&packages);
        let config = Config { seed: 1, worldgen: WorldgenKind::Flat, mods_deny: vec!["pwc.dev-toolkit".into()], ..Config::default() };
        let handle = server::spawn(0, config).unwrap();
        let port = handle.addr().port();
        let _ada = Connection::connect("127.0.0.1", port, "ada", "").expect("the first ada joins");
        let mut job = ConnectJob::begin(("127.0.0.1", port, "ada", ""), &mods, &packages, true, None);
        let Landed::Failed(text) = land(&mut job, &mut mods, &packages) else { panic!("the name is taken") };
        assert_eq!(text, "hosted, but could not connect: that name is already in use");
        assert!(mods.is_enabled(index) && !mods.server_off(index), "the hold is lifted");
        handle.stop();
    }
}
