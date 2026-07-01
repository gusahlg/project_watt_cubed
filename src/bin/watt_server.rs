//! A standalone, headless dedicated server. It opens no window and needs no GPU —
//! it only relays the seed, the edit overlay, player presence, and chat, so it runs
//! fine on a plain box or VPS.
//!
//! Usage:
//! ```text
//! watt_server [--port <n>] [--password <pw>] [--seed <n>]
//! ```
//! With no `--seed`, a fresh time-based seed is chosen and printed so it can be
//! reused. With no `--password`, the server is open to anyone who can reach the port.
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use project_watt_cubed::net::DEFAULT_PORT;
use project_watt_cubed::net::server::{self, Config};

fn main() {
    let mut port = DEFAULT_PORT;
    let mut password = String::new();
    let mut seed = fresh_seed();

    // Minimal `--flag value` parsing; anything unrecognised prints usage and exits.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                port = take(&args, &mut i, "--port").parse().unwrap_or_else(|_| die("port must be a number"));
            }
            "--password" => password = take(&args, &mut i, "--password"),
            "--seed" => {
                seed = take(&args, &mut i, "--seed").parse().unwrap_or_else(|_| die("seed must be a number"));
            }
            "--help" | "-h" => usage_and_exit(),
            other => die(&format!("unknown argument '{other}'")),
        }
        i += 1;
    }

    println!("starting watt-cubed server: seed {seed}, port {port}");
    if password.is_empty() {
        println!("warning: no password set — anyone who can reach the port can join");
    }

    if let Err(e) = server::run(port, Config { password, seed }) {
        eprintln!("server failed to start: {e}");
        process::exit(1);
    }
}

/// Read the value following a flag at `args[i]`, advancing the cursor past it.
fn take(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| die(&format!("{flag} needs a value")))
}

/// A seed from the wall clock, matching how the game seeds a fresh world.
fn fresh_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}

fn usage_and_exit() -> ! {
    println!("usage: watt_server [--port <n>] [--password <pw>] [--seed <n>]");
    process::exit(0);
}

fn die(message: &str) -> ! {
    eprintln!("error: {message}");
    eprintln!("usage: watt_server [--port <n>] [--password <pw>] [--seed <n>]");
    process::exit(1);
}
