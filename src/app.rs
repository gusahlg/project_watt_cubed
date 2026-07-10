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

use voxel_engine::{Color, DVec3, Engine, Frame};

use crate::game::{Game, Signal};
use crate::menu::{self, HostInfo, JoinInfo, MainChoice, MenuEvent, MenuModel, Notice};
use crate::mods::Mods;
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save;
use crate::session::Session;
use crate::settings::Settings;
use crate::world::World;

const STARTING_WINDOW_WIDTH: u32 = 1280;
const STARTING_WINDOW_HEIGHT: u32 = 720;
/// Background for every non-world screen.
const MENU_CLEAR: Color = Color::new(18, 20, 28, 255);

/// Which top-level screen is active.
enum Screen {
    Menu,
    Playing,
    Mods,
    Host,
    Join,
    Settings,
}

/// The whole program: the installed mods (persist across worlds), the menu
/// models, the graphics settings, and the current world if one is open.
///
/// Menus are plain data models here (see [`crate::menu`]): the App builds one
/// per screen, hands input to the first enabled menu-handling mod (or the
/// core fallback if none — disabling the "Menus" mod can never brick
/// navigation), and interprets the [`MenuEvent`]s that come back.
pub struct App {
    /// The live world, if the player is in one.
    game: Option<Game>,
    /// The saves list the main-menu model was built from — the index map that
    /// resolves a `Chosen(i)` on that screen back into a [`MainChoice`].
    saves: Vec<String>,
    main_model: MenuModel,
    mods_model: MenuModel,
    settings_model: MenuModel,
    host_model: MenuModel,
    join_model: MenuModel,
    /// Installed mods and their on/off state; shared with the game while playing.
    mods: Mods,
    screen: Screen,
    /// The integrated server when hosting, kept alive for the session so friends can
    /// stay connected; stopping it frees the port for a later host.
    host: Option<ServerHandle>,
    /// A one-line status/error shown under the start menu (e.g. a failed connect).
    status: Option<String>,
    /// Graphics settings, persisted in `saves/settings.cfg`.
    settings: Settings,
    /// Last-used connection details, persisted in `saves/session.cfg`.
    session: Session,
    /// Headless-ish benchmark mode (`WATT_BENCH=<seconds>`): auto-enters a
    /// world, rotates the camera, prints one stats line, exits.
    bench: Option<Bench>,
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
        let saves = save::list_saves();
        let settings = Settings::load();
        let session = Session::load();
        // Pre-fill the connection forms with what was used last time. Host has
        // Port(0)/Password(1)/Name(2); Join has Address(0)/Port(1)/Password(2)/Name(3).
        let mut host_model = menu::host_menu_model(None);
        host_model.set_text(0, &session.port);
        host_model.set_text(2, &session.name);
        let mut join_model = menu::join_menu_model(None);
        join_model.set_text(0, &session.address);
        join_model.set_text(1, &session.port);
        join_model.set_text(3, &session.name);
        Self {
            game: None,
            main_model: menu::main_menu_model(&saves),
            saves,
            mods_model: menu::mods_menu_model(&mods),
            settings_model: menu::settings_menu_model(&settings),
            host_model,
            join_model,
            mods,
            screen: Screen::Menu,
            host: None,
            status: None,
            settings,
            session,
            bench: std::env::var("WATT_BENCH").ok().map(|v| Bench {
                duration: v.parse().unwrap_or(10.0),
                warmup: 3.0,
                elapsed: 0.0,
                samples: Vec::with_capacity(1 << 17),
                started: false,
                pos: std::env::var("WATT_BENCH_POS").ok().and_then(|s| parse_bench_pos(&s)),
            }),
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
        };
        voxel_engine::run(config, move |eng| app.frame(eng));
    }

    /// One engine frame: update the active screen, then draw it.
    /// Returning `false` stops the engine (after autosaving any open world).
    fn frame(&mut self, eng: &mut Engine) -> bool {
        // OS close button: save and go. Settings save too — the player may be
        // mid-edit on the Settings screen.
        if eng.should_close() {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.autosave();
            return false;
        }

        if self.bench.is_some() && !self.bench_frame(eng) {
            return false;
        }

        let quit = match self.screen {
            Screen::Menu => self.update_menu(eng),
            Screen::Playing => {
                self.update_playing(eng);
                false
            }
            Screen::Mods => {
                self.update_mods(eng);
                false
            }
            Screen::Host => {
                self.update_host(eng);
                false
            }
            Screen::Join => {
                self.update_join(eng);
                false
            }
            Screen::Settings => {
                self.update_settings(eng);
                false
            }
        };
        if quit {
            if self.bench.is_none() {
                self.settings.save();
            }
            self.autosave();
            return false;
        }
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
            if let (Some(pos), Some(game)) = (pos, &mut self.game) {
                game.player_mut().position = pos;
                game.world_mut().prepare_around(pos);
                if let Some(bench) = &mut self.bench {
                    bench.warmup += 2.0;
                }
            }
            return true;
        }
        let Some(game) = &mut self.game else {
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

    /// Drive a menu model through the mod layer, or through the core fallback
    /// when no enabled mod handles menus (the no-brick guarantee).
    fn drive(mods: &mut Mods, eng: &Engine, model: &mut MenuModel) -> Option<MenuEvent> {
        if mods.menu_driver_available() {
            mods.drive_menu(eng, model)
        } else {
            menu::fallback_drive(eng, model)
        }
    }

    /// Draw a menu model through the mod layer, or through the core fallback.
    fn draw_model(mods: &mut Mods, f: &mut Frame, model: &MenuModel, w: i32, h: i32) {
        if mods.menu_driver_available() {
            mods.draw_menu(f, model, w, h);
        } else {
            menu::fallback_draw(f, model, w, h);
        }
    }

    /// Rebuild the main-menu model from the saves on disk (call when returning
    /// to the menu), keeping the cursor on a real row and re-surfacing any
    /// status line as the model's error text.
    fn refresh_main_menu(&mut self) {
        self.saves = save::list_saves();
        let cursor = self.main_model.entries.cursor;
        self.main_model = menu::main_menu_model(&self.saves);
        self.main_model.entries.cursor = cursor;
        self.main_model.clamp_cursor();
        self.main_model.notice = self.status.clone().map(Notice::info);
    }

    /// Rebuild the mod-list model from the mods' current on/off states.
    fn refresh_mods_menu(&mut self) {
        let cursor = self.mods_model.entries.cursor;
        self.mods_model = menu::mods_menu_model(&self.mods);
        self.mods_model.entries.cursor = cursor;
        self.mods_model.clamp_cursor();
    }

    /// Rebuild the settings model's value strings from the live settings.
    fn refresh_settings_menu(&mut self) {
        let cursor = self.settings_model.entries.cursor;
        self.settings_model = menu::settings_menu_model(&self.settings);
        self.settings_model.entries.cursor = cursor;
        self.settings_model.clamp_cursor();
    }

    /// Start-menu logic. Returns `true` to quit the program.
    fn update_menu(&mut self, eng: &mut Engine) -> bool {
        let event = Self::drive(&mut self.mods, eng, &mut self.main_model);
        if let Some(MenuEvent::Chosen(id)) = event {
            self.status = None;
            self.main_model.notice = None;
            match menu::main_choice_at(&self.saves, id) {
                MainChoice::NewWorld => self.start_new_world(eng),
                MainChoice::Load(name) => self.load_world(eng, &name),
                MainChoice::Host => self.screen = Screen::Host,
                MainChoice::Join => self.screen = Screen::Join,
                MainChoice::Mods => {
                    self.refresh_mods_menu();
                    self.screen = Screen::Mods;
                }
                MainChoice::Settings => {
                    // Values may have moved via /gfx in-game; show the truth.
                    self.refresh_settings_menu();
                    self.screen = Screen::Settings;
                }
                MainChoice::Quit => return true,
            }
        }
        // Back on the start menu means nothing — there is nowhere further out.
        false
    }

    /// Host screen: fill in the form, then start an integrated server and connect to
    /// it locally. Esc returns to the menu. A bad port refuses the submit and
    /// keeps the form up with an error in the hint area.
    fn update_host(&mut self, eng: &mut Engine) {
        match Self::drive(&mut self.mods, eng, &mut self.host_model) {
            Some(MenuEvent::Submit) => match menu::parse_port(self.host_model.text_value(0)) {
                Some(port) => {
                    let info = HostInfo {
                        port,
                        password: self.host_model.text_value(1).to_string(),
                        name: self.host_model.text_value(2).to_string(),
                    };
                    self.session.port = self.host_model.text_value(0).to_string();
                    self.session.name = info.name.clone();
                    self.session.save();
                    self.start_host(eng, info);
                }
                None => {
                    self.host_model.notice = Some(Notice::error(menu::PORT_ERROR.to_string()))
                }
            },
            Some(MenuEvent::Back) => self.screen = Screen::Menu,
            _ => {}
        }
    }

    /// Join screen: fill in the address/port/password, then connect. Esc returns.
    /// Same port contract as [`update_host`](Self::update_host).
    fn update_join(&mut self, eng: &mut Engine) {
        match Self::drive(&mut self.mods, eng, &mut self.join_model) {
            Some(MenuEvent::Submit) => match menu::parse_port(self.join_model.text_value(1)) {
                Some(port) => {
                    let info = JoinInfo {
                        host: self.join_model.text_value(0).trim().to_string(),
                        port,
                        password: self.join_model.text_value(2).to_string(),
                        name: self.join_model.text_value(3).to_string(),
                    };
                    self.session.address = info.host.clone();
                    self.session.port = self.join_model.text_value(1).to_string();
                    self.session.name = info.name.clone();
                    self.session.save();
                    self.start_join(eng, info);
                }
                None => {
                    self.join_model.notice = Some(Notice::error(menu::PORT_ERROR.to_string()))
                }
            },
            Some(MenuEvent::Back) => self.screen = Screen::Menu,
            _ => {}
        }
    }

    /// Settings screen: cycle values (what each row means stays here, not
    /// in any mod), apply them live, persist on the way out.
    fn update_settings(&mut self, eng: &mut Engine) {
        let event = Self::drive(&mut self.mods, eng, &mut self.settings_model);
        let mut back = false;
        let mut changed = false;
        match event {
            Some(MenuEvent::Cycled(row, step)) => {
                menu::apply_settings_cycle(&mut self.settings, row, step);
                changed = true;
            }
            // The settings screen's only Action row is Back.
            Some(MenuEvent::Chosen(_)) | Some(MenuEvent::Back) => back = true,
            _ => {}
        }
        // Apply every frame — the engine no-ops unchanged values, so toggles
        // take effect immediately while arrowing through the menu. Rebuild the
        // value strings AFTER applying, so hardware clamps (e.g. 8x MSAA on a
        // 4x device) show what actually took.
        let before = (self.settings.msaa, self.settings.render_scale);
        self.settings.apply(eng);
        // apply() can write back hardware-clamped values with no event this
        // frame (e.g. a hand-edited 8x MSAA config on a 4x device) — refresh
        // whenever the shown values went stale, not just on Cycled.
        if changed || (self.settings.msaa, self.settings.render_scale) != before {
            self.refresh_settings_menu();
        }
        if back {
            self.settings.save();
            self.screen = Screen::Menu;
        }
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

    /// Build the local world from the server's seed and spawn, then enter play with
    /// the connection attached.
    fn enter_net_game(&mut self, eng: &mut Engine, conn: Connection) {
        let world = World::new(conn.seed());
        let player = Player::new(conn.spawn());
        // A networked world is a live mirror, not a save — per-world mod state
        // starts clean, but the player's enable/disable choices persist.
        self.mods.reset_state();
        let game = Game::new(world, player, "multiplayer".to_string()).with_net(conn);
        self.enter_game(eng, game);
    }

    /// Report a connection/host failure and return to the menu.
    fn fail_to_menu(&mut self, message: String) {
        self.status = Some(message);
        self.refresh_main_menu();
        self.screen = Screen::Menu;
    }

    /// Create a fresh world with a time-seeded generator and enter it.
    fn start_new_world(&mut self, eng: &mut Engine) {
        let seed = fresh_seed();
        let world = World::new(seed);
        let player = spawn_player(&world);
        let name = save::next_new_name();

        // A new world starts from a clean default mod set (empty inventory, etc.);
        // the mod menu's enable/disable choices persist.
        self.mods.reset_state();
        self.enter_game(eng, Game::new(world, player, name));
    }

    /// Load an existing save and enter it. Stays on the menu if loading fails.
    fn load_world(&mut self, eng: &mut Engine, name: &str) {
        self.mods.reset_state();
        match save::load(name, &mut self.mods) {
            Ok((world, player)) => {
                self.enter_game(eng, Game::new(world, player, name.to_string()))
            }
            Err(e) => self.fail_to_menu(format!("could not load {name}: {e}")),
        }
    }

    /// Install a freshly built game as the active screen.
    fn enter_game(&mut self, eng: &mut Engine, mut game: Game) {
        game.world_mut().set_view_radius(self.settings.render_distance);
        // Saves and servers can place the player far from the pre-generated
        // origin; make the ground under them real before physics runs.
        let pos = game.player().position;
        game.world_mut().prepare_around(pos);
        game.on_enter(eng);
        self.game = Some(game);
        self.screen = Screen::Playing;
    }

    /// In-world logic; leaves to the menu (autosaving) when the game signals it.
    fn update_playing(&mut self, eng: &mut Engine) {
        let signal = match &mut self.game {
            Some(game) => game.update(eng, &mut self.mods, &mut self.settings),
            None => Signal::ExitToMenu,
        };
        if let Signal::ExitToMenu = signal {
            self.autosave();
            if let Some(game) = &mut self.game {
                // Return the world's GPU meshes to the engine before dropping it.
                game.free_gpu(eng);
            }
            eng.enable_cursor();
            self.game = None;
            self.refresh_main_menu();
            self.screen = Screen::Menu;
        }
    }

    /// Mod-menu logic; Esc (or h/Backspace) returns to the start menu. A
    /// toggle takes effect immediately — switching the "Menus" mod off here
    /// flips the very next frame's driving and drawing to the core fallback.
    fn update_mods(&mut self, eng: &mut Engine) {
        match Self::drive(&mut self.mods, eng, &mut self.mods_model) {
            Some(MenuEvent::Toggled(index)) => {
                self.mods.toggle(index);
                // Rebuild from the source of truth (the driver only flipped
                // the displayed state).
                self.refresh_mods_menu();
            }
            Some(MenuEvent::Back) => self.screen = Screen::Menu,
            _ => {}
        }
    }

    /// Save the open world, if any (best-effort — a failed save shouldn't crash).
    /// Networked worlds are server mirrors, not local saves, so they're never written.
    fn autosave(&mut self) {
        if let Some(game) = &self.game {
            if game.is_multiplayer() {
                return;
            }
            let _ = save::save(game.save_name(), game.world(), game.player(), &self.mods);
        }
    }

    /// Draw the active screen. Every menu screen goes through the mod layer
    /// (or the core fallback); the connect/host status line rides in the main
    /// model's `error`, so the renderer — whichever one — shows it.
    fn draw(&mut self, eng: &mut Engine) {
        let (w, h) = (eng.screen_width(), eng.screen_height());
        let model = match self.screen {
            Screen::Playing => {
                let fov = self.settings.fov;
                if let Some(game) = &mut self.game {
                    game.draw(eng, &mut self.mods, fov);
                }
                return;
            }
            Screen::Menu => &self.main_model,
            Screen::Mods => &self.mods_model,
            Screen::Host => &self.host_model,
            Screen::Join => &self.join_model,
            Screen::Settings => &self.settings_model,
        };
        let mut f = eng.begin_frame(MENU_CLEAR);
        Self::draw_model(&mut self.mods, &mut f, model, w, h);
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

/// Spawn the player just above the surface at the world origin, so they drop and
/// land on solid ground.
fn spawn_player(world: &World) -> Player {
    let surface = world.surface_y(0, 0);
    Player::new(DVec3::new(0.5, surface as f64 + 3.0, 0.5))
}

/// Parse `WATT_BENCH_POS="x,y,z"` into a position (f64, comma-separated).
fn parse_bench_pos(raw: &str) -> Option<DVec3> {
    let mut parts = raw.split(',').map(|p| p.trim().parse::<f64>());
    let (x, y, z) = (parts.next()?.ok()?, parts.next()?.ok()?, parts.next()?.ok()?);
    parts.next().is_none().then(|| DVec3::new(x, y, z))
}
