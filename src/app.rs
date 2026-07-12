//! app.rs owns the top-level state machine: the start menu, an in-world
//! [`Game`], the mod menu, the host/join forms, and the graphics settings
//! screen. It routes each engine frame to the active screen, creates and
//! loads worlds, and autosaves when leaving one.
//!
//! The window itself belongs to the engine: [`App::run`] hands a per-frame
//! closure to [`voxel_engine::run`], which is the moral equivalent of the old
//! raylib `while !window_should_close()` loop.
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use voxel_engine::{Color, DVec3, Engine};

use crate::game::{Game, Signal};
use crate::input::router::{Context, Router, View};
use crate::menu::menus::MainMenu;
use crate::menu::theme::{DefaultTheme, MenuTheme};
use crate::menu::{AppEffect, Ctx, Framed, HostInfo, JoinInfo, MenuStack, ModRow};
use crate::mods::Mods;
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save::{self, Autosaver, SaveMeta, Slot, SlotId, Tick};
use crate::session::Session;
use crate::settings::Settings;
use crate::world::World;

const STARTING_WINDOW_WIDTH: u32 = 1280;
const STARTING_WINDOW_HEIGHT: u32 = 720;
/// Background for every non-world screen.
const MENU_CLEAR: Color = Color::new(18, 20, 28, 255);
/// How often a dirty world writes itself in the background.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

enum Screen {
    Menus(MenuStack),
    Playing(Box<Game>),
}

/// The whole program: the installed mods (persist across worlds), the graphics
/// settings, and either the menu stack or the current world.
pub struct App {
    /// Available save slots, refreshed on menu return.
    saves: Vec<Slot>,
    /// The slot behind the open singleplayer world; `None` on menus and in
    /// multiplayer (a networked world is a server mirror, never saved locally).
    active: Option<ActiveSlot>,
    /// Shared router for menus and in-game input.
    router: Router,
    /// Installed mods and their on/off state; shared with the game while playing.
    mods: Mods,
    screen: Screen,
    /// The integrated server when hosting, kept alive for the session so friends can
    /// stay connected; stopping it frees the port for a later host.
    host: Option<ServerHandle>,
    /// Graphics settings, persisted in `saves/settings.cfg`.
    settings: Settings,
    /// Last-used connection details, persisted in `saves/session.cfg`.
    session: Session,
    /// Headless-ish benchmark mode (`WATT_BENCH=<seconds>`): auto-enters a
    /// world, rotates the camera, prints one stats line, exits.
    bench: Option<Bench>,
}

/// The save slot behind the open singleplayer world: identity, header
/// metadata carried across writes, accumulated playtime, and the autosaver.
struct ActiveSlot {
    id: SlotId,
    /// Persists name/created; seed/playtime/edits restamped on writes.
    meta: SaveMeta,
    /// Total seconds played, fractional to avoid per-frame truncation.
    playtime: f64,
    autosaver: Autosaver,
}

impl ActiveSlot {
    fn new(id: SlotId, meta: SaveMeta) -> Self {
        let playtime = meta.playtime_secs as f64;
        Self { id, meta, playtime, autosaver: Autosaver::new(AUTOSAVE_INTERVAL) }
    }
}

/// State for the `WATT_BENCH` frame-rate benchmark.
struct Bench {
    /// Measurement length in seconds (after warmup).
    duration: f32,
    /// Seconds of warmup left before sampling starts (world streaming in).
    warmup: f32,
    /// Elapsed measured time.
    elapsed: f32,
    /// Per-frame durations, for avg and percentile stats.
    samples: Vec<f32>,
    started: bool,
    /// Where to park the bench player (`WATT_BENCH_POS="x,y,z"`), for
    /// far-coordinate fps parity checks. `None` benches at spawn.
    pos: Option<DVec3>,
}

impl App {
    pub fn new() -> Self {
        let mods = Mods::with_defaults();
        let saves = save::list();
        let settings = Settings::load();
        let session = Session::load();
        let bench = std::env::var("WATT_BENCH").ok().map(|v| Bench {
            duration: v.parse().unwrap_or(10.0),
            warmup: 3.0,
            elapsed: 0.0,
            samples: Vec::with_capacity(1 << 17),
            started: false,
            pos: std::env::var("WATT_BENCH_POS").ok().and_then(|s| parse_bench_pos(&s)),
        });
        // A benchmark run auto-enables the profiler (CPU subsystems + workers
        // via VOXEL_PROFILE) unless the caller set it explicitly. Safe here:
        // `new()` runs on the main thread at startup, before the renderer or
        // any worker thread — the only reader of this var — exists. Reads
        // happen later.
        if bench.is_some() && std::env::var_os("VOXEL_PROFILE").is_none() {
            unsafe { std::env::set_var("VOXEL_PROFILE", "1") };
        }
        Self {
            saves,
            active: None,
            router: Router::new(),
            mods,
            screen: Screen::Menus(MenuStack::new(Framed::boxed(MainMenu::new()))),
            host: None,
            settings,
            session,
            bench,
        }
    }

    /// Open the window and run until the player quits (menu or close button).
    pub fn run(self) {
        let mut app = self;
        let config = voxel_engine::Config {
            title: "Project Watt Cubed".into(),
            width: STARTING_WINDOW_WIDTH,
            height: STARTING_WINDOW_HEIGHT,
            target_fps: app.settings.max_fps,
            vsync: app.settings.vsync,
            msaa: app.settings.msaa,
            render_scale: app.settings.render_scale,
            resizable: true,
            fullscreen: app.settings.fullscreen,
            // Engine-side render lanes from the persisted settings (the single
            // source; the world's own occlusion/lod2 lanes come from the same
            // `Settings::render_config` at world entry).
            flags: app.settings.render_config().engine_flags(),
        };
        voxel_engine::run(config, move |eng| app.frame(eng));
    }

    /// One engine frame: update and draw the active screen.
    fn frame(&mut self, eng: &mut Engine) -> bool {
        // OS close button: save and go. Settings save too — the player may be
        // mid-edit on the Settings screen.
        if eng.should_close() {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.flush_save();
            return false;
        }

        if self.bench.is_some() && !self.bench_frame(eng) {
            return false;
        }

        let quit = match self.screen {
            Screen::Menus(_) => self.update_menus(eng),
            Screen::Playing(_) => {
                self.update_playing(eng);
                false
            }
        };
        if quit {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.flush_save();
            return false;
        }
        // Force vsync on whenever we're not in a live world (menus, loading):
        // there's nothing to gain from tearing/uncapped frames on a static
        // screen, and it keeps the GPU quiet. In-world we honour the setting.
        let in_world = matches!(self.screen, Screen::Playing(_));
        eng.set_vsync(!in_world || self.settings.vsync);
        self.draw(eng);
        true
    }

    /// Drive one benchmark frame: enter a world on the first frame, spin the
    /// camera, sample frame times, and print the stats line when done.
    /// Returns `false` when the benchmark is finished and the app should exit.
    fn bench_frame(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();
        let bench = self.bench.as_mut().expect("bench_frame without bench");

        if !bench.started {
            bench.started = true;
            let pos = bench.pos;
            // Uncapped and unsynced, or the bench measures the throttle.
            self.settings.vsync = false;
            self.settings.max_fps = 0;
            self.settings.apply(eng);
            self.start_new_world(eng);
            // Far-coordinate bench: park the player at the requested position
            // with the ground under them made real, and give streaming a
            // little extra warmup to catch up before sampling starts.
            if let (Some(pos), Screen::Playing(game)) = (pos, &mut self.screen) {
                game.player_mut().position = pos;
                game.world_mut().prepare_around(pos);
                if let Some(bench) = &mut self.bench {
                    bench.warmup += 2.0;
                }
            }
            return true;
        }
        let Screen::Playing(game) = &mut self.screen else {
            return true;
        };
        // A slow spin sweeps the frustum across the terrain like a player would.
        game.player_mut().yaw += 0.4 * dt;

        if bench.warmup > 0.0 {
            bench.warmup -= dt;
            return true;
        }
        bench.elapsed += dt;
        if dt > 0.0 {
            bench.samples.push(dt);
        }
        if bench.elapsed < bench.duration {
            return true;
        }

        let frames = bench.samples.len();
        let total: f32 = bench.samples.iter().sum();
        let avg_ms = total / frames.max(1) as f32 * 1000.0;
        let avg_fps = frames as f32 / total.max(f32::EPSILON);
        let mut sorted = bench.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // p1 fps = the fps of the 99th-percentile (slowest 1%) frame time. With no
        // samples there is no percentile to report, so emit it only when present.
        let p1_fps = sorted
            .get((frames.saturating_sub(1)) * 99 / 100)
            .map(|dt| format!("{:.0}", 1.0 / dt.max(f32::EPSILON)))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "BENCH frames={frames} avg_fps={avg_fps:.0} p1_fps={p1_fps} avg_ms={avg_ms:.3} rss_mb={}",
            resident_mb().unwrap_or(0),
        );
        false
    }

    /// Update the menu stack and apply settings live each frame.
    fn update_menus(&mut self, eng: &mut Engine) -> bool {
        let dt = eng.frame_time();
        // Menus are an exclusive router context.
        self.router.set_context(Context::Menu);
        let intents = match self.router.frame(eng, dt).view() {
            View::Menu(m) => crate::menu::gather(&m),
            _ => Vec::new(),
        };
        // A per-frame snapshot so a menu never holds a live `&Mods`.
        let mods = ModRow::snapshot(&self.mods);
        let before = self.settings.clone();
        let mut effect = None;
        if let Screen::Menus(stack) = &mut self.screen {
            let mut ctx = Ctx {
                settings: &mut self.settings,
                saves: &self.saves,
                mods: &mods,
                session: &self.session,
            };
            effect = stack.update(&intents, &mut ctx);
        }
        // Apply every frame for immediate feedback and to show hardware clamps.
        self.settings.apply(eng);
        // Persist whenever a step (or a hardware clamp) moved a value.
        if self.settings != before {
            self.settings.save();
        }
        match effect {
            Some(effect) => self.handle_effect(eng, effect),
            None => false,
        }
    }

    /// Interpret one menu effect. Returns `true` only for Quit.
    fn handle_effect(&mut self, eng: &mut Engine, effect: AppEffect) -> bool {
        match effect {
            AppEffect::NewWorld => self.start_new_world(eng),
            AppEffect::Load(name) => self.load_world(eng, &name),
            AppEffect::Host(info) => {
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_host(eng, info);
            }
            AppEffect::Join(info) => {
                self.session.address = info.host.clone();
                self.session.port = info.port.to_string();
                self.session.name = info.name.clone();
                self.session.save();
                self.start_join(eng, info);
            }
            AppEffect::ToggleMod(index) => self.mods.toggle(index),
            AppEffect::Quit => return true,
        }
        false
    }

    /// Return to the start menu with an optional notice (e.g. a failed connect).
    fn to_menu(&mut self, notice: Option<String>) {
        self.active = None;
        self.saves = save::list();
        self.screen = Screen::Menus(MenuStack::new(Framed::boxed(MainMenu::with_notice(notice))));
    }

    /// Spin up a fresh integrated server and join it on loopback. Any previous host
    /// is stopped first so its port is free to reuse.
    fn start_host(&mut self, eng: &mut Engine, info: HostInfo) {
        if let Some(previous) = self.host.take() {
            previous.stop();
            thread::sleep(Duration::from_millis(150));
        }
        let seed = fresh_seed();
        let config = Config { password: info.password.clone(), seed };
        match server::spawn(info.port, config) {
            Ok(handle) => {
                let port = handle.addr().port();
                self.host = Some(handle);
                // Connect our own client to the server we just started.
                match Connection::connect("127.0.0.1", port, &info.name, &info.password) {
                    Ok(conn) => self.enter_net_game(eng, conn),
                    Err(e) => self.fail_to_menu(format!("hosted, but could not connect: {e}")),
                }
            }
            Err(e) => self.fail_to_menu(format!("could not host on port {}: {e}", info.port)),
        }
    }

    /// Connect to a remote server and enter its world.
    fn start_join(&mut self, eng: &mut Engine, info: JoinInfo) {
        match Connection::connect(&info.host, info.port, &info.name, &info.password) {
            Ok(conn) => self.enter_net_game(eng, conn),
            Err(e) => self.fail_to_menu(format!("could not join: {e}")),
        }
    }

    /// Join a remote world via an existing connection.
    fn enter_net_game(&mut self, eng: &mut Engine, conn: Connection) {
        let world = World::new(conn.seed());
        let player = Player::new(conn.spawn());
        // A networked world is a live mirror, not a save — per-world mod state
        // starts clean, but the player's enable/disable choices persist.
        self.active = None;
        self.mods.reset_state();
        let game = Game::new(world, player, "multiplayer".to_string()).with_net(conn);
        self.enter_game(eng, game);
    }

    /// Report a connection/host failure and return to the menu.
    fn fail_to_menu(&mut self, message: String) {
        self.to_menu(Some(message));
    }

    /// Create a fresh world with a time-seeded generator and enter it.
    fn start_new_world(&mut self, eng: &mut Engine) {
        // Benchmarks pin the seed (`WATT_BENCH_SEED`, default when benching) so
        // fps/rss deltas measure the code, not terrain-lottery variance.
        let seed = match (&self.bench, std::env::var("WATT_BENCH_SEED")) {
            (_, Ok(s)) => s.parse().unwrap_or_else(|_| fresh_seed()),
            (Some(_), _) => 42,
            (None, _) => fresh_seed(),
        };
        let world = World::new(seed);
        let player = spawn_player(&world);
        let id = save::fresh_id();
        let now = save::unix_now();
        self.active = Some(ActiveSlot::new(
            id.clone(),
            SaveMeta {
                name: id.as_str().to_string(),
                seed,
                created: now,
                last_played: now,
                playtime_secs: 0,
                edit_count: 0,
            },
        ));

        // A new world starts from a clean default mod set (empty inventory, etc.);
        // the mod menu's enable/disable choices persist.
        self.mods.reset_state();
        self.enter_game(eng, Game::new(world, player, id.as_str().to_string()));
    }

    /// Load an existing save and enter it. Stays on the menu if loading fails.
    fn load_world(&mut self, eng: &mut Engine, name: &str) {
        let id = match SlotId::new(name) {
            Ok(id) => id,
            Err(e) => return self.fail_to_menu(format!("could not load {name}: {e}")),
        };
        self.mods.reset_state();
        match save::load(&id, &mut self.mods) {
            Ok((world, player, meta, report)) => {
                self.active = Some(ActiveSlot::new(id.clone(), meta));
                let mut game = Game::new(world, player, id.as_str().to_string());
                // Degraded loads still enter the world, but say so.
                if report.source == save::Source::Backup {
                    game.notify("* save was unreadable — restored from the backup");
                }
                if let Some((recovered, expected)) = report.salvage {
                    game.notify(format!(
                        "* save was damaged — recovered {recovered} of {expected} edits"
                    ));
                }
                self.enter_game(eng, game)
            }
            Err(e) => self.fail_to_menu(format!("could not load {name}: {e}")),
        }
    }

    /// Install a freshly built game as the active screen.
    fn enter_game(&mut self, eng: &mut Engine, mut game: Game) {
        // World-construction lanes apply on entry only, before streaming spins;
        // everything live-applicable goes through the same path `/gfx` uses.
        let render = self.settings.render_config();
        game.world_mut().set_render_lanes(render.occlusion, render.lod2);
        game.apply_settings(eng, &mut self.settings);
        // Saves and servers can place the player far from the pre-generated
        // origin; make the ground under them real before physics runs.
        let pos = game.player().position;
        game.world_mut().prepare_around(pos);
        game.on_enter(eng, &mut self.router);
        self.screen = Screen::Playing(Box::new(game));
    }

    /// In-world update: run the game and handle autosave.
    fn update_playing(&mut self, eng: &mut Engine) {
        let dt = eng.frame_time() as f64;
        let Screen::Playing(game) = &mut self.screen else {
            return;
        };
        let signal = game.update(eng, &mut self.router, &mut self.mods, &mut self.settings);
        if let Signal::ExitToMenu = signal {
            self.flush_save();
            if let Screen::Playing(game) = &mut self.screen {
                // Return the world's GPU meshes to the engine before dropping it.
                game.free_gpu(eng);
            }
            eng.enable_cursor();
            self.to_menu(None); // drops the Box<Game>
            return;
        }
        // Periodic autosave on edits; bench/multiplayer never save.
        if self.bench.is_some() {
            return;
        }
        let Some(active) = &mut self.active else {
            return;
        };
        active.playtime += dt;
        active.meta.playtime_secs = active.playtime as u64;
        let ActiveSlot { id, meta, autosaver, .. } = active;
        let tick = autosaver.tick(id, game.world().edit_generation(), || {
            save::encode_current(game.world(), game.player(), &self.mods, meta.clone())
        });
        if let Tick::Finished(Err(e)) = tick {
            game.notify(format!("* autosave failed: {e}"));
        }
    }

    /// Synchronously write the singleplayer world (networked and bench worlds
    /// are never saved).
    fn flush_save(&mut self) {
        let Screen::Playing(game) = &self.screen else {
            return;
        };
        if game.is_multiplayer() || self.bench.is_some() {
            return;
        }
        let Some(active) = &mut self.active else {
            return;
        };
        active.meta.playtime_secs = active.playtime as u64;
        let ActiveSlot { id, meta, autosaver, .. } = active;
        if let Err(e) = autosaver.flush_now(id, game.world().edit_generation(), || {
            save::encode_current(game.world(), game.player(), &self.mods, meta.clone())
        }) {
            eprintln!("could not save {id}: {e}");
        }
    }

    /// Draw the active screen (game or menu).
    fn draw(&mut self, eng: &mut Engine) {
        let (w, h) = (eng.screen_width(), eng.screen_height());
        if let Screen::Playing(game) = &mut self.screen {
            let fov = self.settings.fov;
            let shake = self.settings.shake;
            game.draw(eng, &mut self.mods, fov, shake);
            return;
        }
        // Mods snapshot avoids borrow conflict between theme and view.
        let mods = ModRow::snapshot(&self.mods);
        let fallback = DefaultTheme;
        let theme: &dyn MenuTheme = self.mods.menu_theme().unwrap_or(&fallback);
        let mut f = eng.begin_frame(MENU_CLEAR.to_linear());
        if let Screen::Menus(stack) = &self.screen {
            let ctx = Ctx {
                settings: &mut self.settings,
                saves: &self.saves,
                mods: &mods,
                session: &self.session,
            };
            stack.draw(&ctx, theme, &mut f, w, h);
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Resident set size in MB via one `ps` call (bench-end only): a memory
/// regression tripwire living next to the fps numbers, zero dependencies.
fn resident_mb() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb / 1024)
}

/// A world seed from the wall clock, so each new world differs.
fn fresh_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}

/// Spawn the player just above dry land near the world origin, so they drop and
/// land on solid ground instead of sinking into an ocean/lake column that
/// happens to sit at (0, 0). Spirals outward from the origin for the first
/// column above sea level, mirroring `net::server::spawn_point`.
fn spawn_player(world: &World) -> Player {
    let sea = world.sea_level();
    for r in 0..64 {
        for (dx, dz) in [(r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)] {
            let (x, z) = (dx * 8, dz * 8);
            let h = world.surface_y(x, z);
            if h > sea {
                return Player::new(DVec3::new(x as f64 + 0.5, h as f64 + 3.0, z as f64 + 0.5));
            }
        }
    }
    let h = world.surface_y(0, 0).max(sea);
    Player::new(DVec3::new(0.5, h as f64 + 3.0, 0.5))
}

/// Parse `WATT_BENCH_POS="x,y,z"` into a position (f64, comma-separated).
fn parse_bench_pos(raw: &str) -> Option<DVec3> {
    let mut parts = raw.split(',').map(|p| p.trim().parse::<f64>());
    let (x, y, z) = (parts.next()?.ok()?, parts.next()?.ok()?, parts.next()?.ok()?);
    parts.next().is_none().then(|| DVec3::new(x, y, z))
}
