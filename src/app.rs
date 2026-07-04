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

use voxel_engine::{Color, Engine, Vec3};

use crate::console::shadowed;
use crate::game::{Game, Signal};
use crate::menu::{
    FormResult, HostInfo, HostMenu, JoinInfo, JoinMenu, MainChoice, MainMenu, ModMenu,
    SettingsMenu,
};
use crate::mods::Mods;
use crate::net::client::Connection;
use crate::net::server::{self, Config, ServerHandle};
use crate::player::Player;
use crate::save;
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

/// The whole program: the installed mods (persist across worlds), the menus,
/// the graphics settings, and the current world if one is open.
pub struct App {
    /// The live world, if the player is in one.
    game: Option<Game>,
    menu: MainMenu,
    mod_menu: ModMenu,
    host_menu: HostMenu,
    join_menu: JoinMenu,
    settings_menu: SettingsMenu,
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
}

impl App {
    pub fn new() -> Self {
        Self {
            game: None,
            menu: MainMenu::new(),
            mod_menu: ModMenu::new(),
            host_menu: HostMenu::new(),
            join_menu: JoinMenu::new(),
            settings_menu: SettingsMenu::new(),
            mods: Mods::with_defaults(),
            screen: Screen::Menu,
            host: None,
            status: None,
            settings: Settings::load(),
            bench: std::env::var("WATT_BENCH").ok().map(|v| Bench {
                duration: v.parse().unwrap_or(10.0),
                warmup: 3.0,
                elapsed: 0.0,
                samples: Vec::with_capacity(1 << 17),
                started: false,
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
            // Uncapped and unsynced, or the bench measures the throttle.
            self.settings.vsync = false;
            self.settings.max_fps = 0;
            self.settings.apply(eng);
            self.start_new_world(eng);
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
        // p1 fps = the fps of the 99th-percentile (slowest 1%) frame time.
        let p99_dt = sorted[(frames.saturating_sub(1)) * 99 / 100];
        println!(
            "BENCH frames={frames} avg_fps={avg_fps:.0} p1_fps={:.0} avg_ms={avg_ms:.3}",
            1.0 / p99_dt.max(f32::EPSILON)
        );
        false
    }

    /// Start-menu logic. Returns `true` to quit the program.
    fn update_menu(&mut self, eng: &mut Engine) -> bool {
        if let Some(choice) = self.menu.update(eng) {
            self.status = None;
            match choice {
                MainChoice::NewWorld => self.start_new_world(eng),
                MainChoice::Load(name) => self.load_world(eng, &name),
                MainChoice::Host => self.screen = Screen::Host,
                MainChoice::Join => self.screen = Screen::Join,
                MainChoice::Mods => self.screen = Screen::Mods,
                MainChoice::Settings => self.screen = Screen::Settings,
                MainChoice::Quit => return true,
            }
        }
        false
    }

    /// Host screen: fill in the form, then start an integrated server and connect to
    /// it locally. Esc returns to the menu.
    fn update_host(&mut self, eng: &mut Engine) {
        match self.host_menu.update(eng) {
            FormResult::Submit(info) => self.start_host(eng, info),
            FormResult::Cancel => self.screen = Screen::Menu,
            FormResult::Editing => {}
        }
    }

    /// Join screen: fill in the address/port/password, then connect. Esc returns.
    fn update_join(&mut self, eng: &mut Engine) {
        match self.join_menu.update(eng) {
            FormResult::Submit(info) => self.start_join(eng, info),
            FormResult::Cancel => self.screen = Screen::Menu,
            FormResult::Editing => {}
        }
    }

    /// Settings screen: edit values, apply them live, persist on the way out.
    fn update_settings(&mut self, eng: &mut Engine) {
        let back = self.settings_menu.update(eng, &mut self.settings);
        // Apply every frame — the engine no-ops unchanged values, so toggles
        // take effect immediately while arrowing through the menu.
        self.settings.apply(eng);
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
        self.menu.refresh();
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
            Err(_) => {}
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
            self.menu.refresh();
            self.screen = Screen::Menu;
        }
    }

    /// Mod-menu logic; Esc returns to the start menu.
    fn update_mods(&mut self, eng: &mut Engine) {
        if self.mod_menu.update(eng, &mut self.mods) {
            self.screen = Screen::Menu;
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

    /// Draw the active screen.
    fn draw(&mut self, eng: &mut Engine) {
        let (w, h) = (eng.screen_width(), eng.screen_height());
        match self.screen {
            Screen::Playing => {
                let fov = self.settings.fov;
                if let Some(game) = &mut self.game {
                    game.draw(eng, &mut self.mods, fov);
                }
            }
            Screen::Menu => {
                let status = self.status.clone();
                let mut f = eng.begin_frame(MENU_CLEAR);
                self.menu.draw(&mut f, w, h);
                // A connect/host error from the last attempt, in red under the list.
                if let Some(status) = &status {
                    let fs = 20;
                    let sx = (w - f.measure_text(status, fs)) / 2;
                    shadowed(&mut f, status, sx, h - 70, fs, Color::SALMON);
                }
            }
            Screen::Mods => {
                let mut f = eng.begin_frame(MENU_CLEAR);
                self.mod_menu.draw(&mut f, &self.mods, w, h);
            }
            Screen::Host => {
                let mut f = eng.begin_frame(MENU_CLEAR);
                self.host_menu.draw(&mut f, w, h);
            }
            Screen::Join => {
                let mut f = eng.begin_frame(MENU_CLEAR);
                self.join_menu.draw(&mut f, w, h);
            }
            Screen::Settings => {
                let mut f = eng.begin_frame(MENU_CLEAR);
                self.settings_menu.draw(&mut f, &self.settings, w, h);
            }
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
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
    Player::new(Vec3::new(0.5, surface as f32 + 3.0, 0.5))
}
