//! A standalone, headless dedicated server. It opens no window and needs no GPU —
//! it relays the seed, the edit overlay, player presence, and chat, and it
//! saves that world, so it runs on a plain box or VPS.
//!
//! Usage:
//! ```text
//! watt_server [--port <n>] [--password <pw>] [--seed <n>] [--day-secs <n>]
//!             [--world <path>] [--ops <name,name>] [--teleport off|ops|all]
//!             [--noclip off|ops|all] [--max-speed <m/s>] [--mods-allow <id,id>] [--mods-deny <id,id>]
//!             [--worldgen <flat|diffusion>] [--relief <n>] [--caves <n>] [--mines <n>]
//!             [--space <n>] [--variety <n>] [--features <n>] [--structures <n>] [--deep <n>]
//!             [--data-dir <dir>]
//! ```
//! `--world` loads and saves the seed, generator, edit ledger, and clock.
//! A stored seed and generator win over the flags. The world is written every
//! few minutes and on shutdown (SIGINT, SIGTERM). Next to that file, `ops.txt`
//! and `mods.toml` are read and united with the flags. An `ops.txt` line is
//! `name secret`: that player becomes an operator by sending the chat line
//! `/op secret`. Quote names containing spaces, e.g. `"Big Ada" secret` or
//! `"Big Ada"` for a trusted name with no secret. Unquoted lines keep their meaning.
//! A line with a name alone, like `--ops`, trusts the name, so
//! anyone who joins under it is an operator; the server warns about those.
//! A `mods.toml`:
//! ```toml
//! deny = ["pwc.dev-toolkit"]
//! # allow = [...] would admit only the listed packages: name every package players need.
//! ```
//! With no `--seed`, a fresh time-based seed is chosen and printed so it can be
//! reused. With no `--password`, the server is open to anyone who can reach the port.
//! `--worldgen` selects the generator (default diffusion) for a new world. The
//! eight knobs are percents of the designed terrain and are snapped onto the
//! game's stepper. `--day-secs` sets the shared day/night cycle length.
//! `--teleport` defaults to `ops` (only operators). `--noclip` defaults to `ops`:
//! a move whose body overlaps solid ground snaps back unless that player may pass.
//! `--max-speed` is metres per
//! second; omitted, it is the game's own cap, which leaves cruise alone.
//! `--mods-allow` and `--mods-deny` name package ids. With no allow list every
//! reported mod is admitted. The list is the client's own word.
//! `--data-dir` sets the data and config root (same as `WATT_DATA_DIR`).
use std::path::PathBuf;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use project_watt_cubed::math::PER_METER;
use project_watt_cubed::net::DEFAULT_PORT;
use project_watt_cubed::net::server::{self, Config, Policy};
use project_watt_cubed::paths::Paths;
use project_watt_cubed::world::generation::WorldgenKind;

const USAGE: &str = "\
usage: watt_server [--port <n>] [--password <pw>] [--seed <n>] [--day-secs <n>] \
[--world <path>] [--ops <name,name>] [--teleport off|ops|all] [--noclip off|ops|all] \
[--max-speed <m/s>] \
[--mods-allow <id,id>] [--mods-deny <id,id>] \
[--worldgen <flat|diffusion>] [--relief <n>] [--caves <n>] [--mines <n>] [--space <n>] \
[--variety <n>] [--features <n>] [--structures <n>] [--deep <n>] [--data-dir <dir>]";

fn main() {
    let mut port = DEFAULT_PORT;
    let mut config = Config {
        seed: fresh_seed(),
        teleport: Policy::Ops,
        noclip: Policy::Ops,
        warn_world_overrides: true,
        ..Config::default()
    };
    let mut data_dir: Option<PathBuf> = None;

    // Minimal `--flag value` parsing; anything unrecognised prints usage and exits.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                port = take(&args, &mut i, "--port").parse().unwrap_or_else(|_| die("port must be a number"));
            }
            "--password" => config.password = take(&args, &mut i, "--password"),
            "--seed" => {
                config.seed = take(&args, &mut i, "--seed").parse().unwrap_or_else(|_| die("seed must be a number"));
            }
            "--day-secs" => {
                config.day_secs = take(&args, &mut i, "--day-secs")
                    .parse()
                    .ok()
                    .filter(|s: &f32| s.is_finite() && *s >= 10.0)
                    .unwrap_or_else(|| die("day-secs must be a number >= 10"));
            }
            "--worldgen" => {
                let name = take(&args, &mut i, "--worldgen");
                config.worldgen = WorldgenKind::from_id(&name)
                    .unwrap_or_else(|| die("worldgen must be flat or diffusion"));
            }
            "--relief" => config.terrain.relief = knob(&args, &mut i, "--relief"),
            "--caves" => config.terrain.caves = knob(&args, &mut i, "--caves"),
            "--mines" => config.terrain.mines = knob(&args, &mut i, "--mines"),
            "--space" => config.terrain.space = knob(&args, &mut i, "--space"),
            "--variety" => config.terrain.variety = knob(&args, &mut i, "--variety"),
            "--features" => config.terrain.features = knob(&args, &mut i, "--features"),
            "--structures" => config.terrain.structures = knob(&args, &mut i, "--structures"),
            "--deep" => config.terrain.deep = knob(&args, &mut i, "--deep"),
            "--world" => config.world = Some(PathBuf::from(take(&args, &mut i, "--world"))),
            "--ops" => config.ops.extend(split_list(&take(&args, &mut i, "--ops"))),
            "--teleport" => {
                config.teleport = Policy::parse(&take(&args, &mut i, "--teleport"))
                    .unwrap_or_else(|| die("teleport must be off, ops, or all"));
            }
            "--noclip" => {
                config.noclip = Policy::parse(&take(&args, &mut i, "--noclip"))
                    .unwrap_or_else(|| die("noclip must be off, ops, or all"));
            }
            "--max-speed" => {
                let mps: f64 = take(&args, &mut i, "--max-speed")
                    .parse()
                    .unwrap_or_else(|_| die("max-speed must be a number of metres per second"));
                if !mps.is_finite() || mps < 0.0 {
                    die("max-speed must be zero or more metres per second");
                }
                config.max_speed = mps * PER_METER;
            }
            "--mods-allow" => config.mods_allow.extend(split_list(&take(&args, &mut i, "--mods-allow"))),
            "--mods-deny" => config.mods_deny.extend(split_list(&take(&args, &mut i, "--mods-deny"))),
            "--data-dir" => {
                data_dir = Some(PathBuf::from(take(&args, &mut i, "--data-dir")));
            }
            "--help" | "-h" => usage_and_exit(),
            other => die(&format!("unknown argument '{other}'")),
        }
        i += 1;
    }
    config.terrain = config.terrain.clamp();
    if let Err(e) = server::load_world_policy(&mut config) {
        die(&e);
    }

    Paths::init(data_dir.as_deref());
    println!(
        "starting watt-cubed server: seed {}, worldgen {}, port {port}",
        config.seed,
        config.worldgen.id()
    );
    println!(
        "teleport {}, noclip {}, max speed {:.0} m/s",
        config.teleport.name(),
        config.noclip.name(),
        config.max_speed / PER_METER
    );
    match (config.mods_allow.is_empty(), config.mods_deny.is_empty()) {
        (true, true) => println!("mods: unrestricted"),
        (true, false) => println!("mods: deny {}", config.mods_deny.join(",")),
        (false, true) => println!("mods: allow only {}", config.mods_allow.join(",")),
        (false, false) => println!(
            "mods: allow only {}; deny {}",
            config.mods_allow.join(","),
            config.mods_deny.join(",")
        ),
    }
    if !config.op_secrets.is_empty() {
        let names: Vec<&str> = config.op_secrets.iter().map(|(name, _)| name.as_str()).collect();
        println!("operators (after /op): {}", names.join(", "));
    }
    let by_name: Vec<&str> = config
        .ops
        .iter()
        .filter(|op| !config.op_secrets.iter().any(|(name, _)| name.eq_ignore_ascii_case(op)))
        .map(String::as_str)
        .collect();
    if !by_name.is_empty() {
        println!(
            "warning: operators with no secret, anyone who joins under these names is an operator: {}",
            by_name.join(", ")
        );
    }
    if config.password.is_empty() {
        println!("warning: no password set — anyone who can reach the port can join");
    }

    if let Err(e) = server::run(port, config) {
        eprintln!("server failed to start: {e}");
        process::exit(1);
    }
}

fn split_list(text: &str) -> Vec<String> {
    text.split(',').map(|part| part.trim().to_string()).filter(|part| !part.is_empty()).collect()
}

fn knob(args: &[String], i: &mut usize, flag: &str) -> u16 {
    take(args, i, flag).parse().unwrap_or_else(|_| die(&format!("{flag} must be a number")))
}

fn take(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| die(&format!("{flag} needs a value")))
}

fn fresh_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}

fn usage_and_exit() -> ! {
    println!("{USAGE}");
    process::exit(0);
}

fn die(message: &str) -> ! {
    eprintln!("error: {message}");
    eprintln!("{USAGE}");
    process::exit(1);
}
