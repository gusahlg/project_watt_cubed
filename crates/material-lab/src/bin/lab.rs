//! Command-line front of the material lab: scorecard, find-regions, sweep.

use material::Law;
use material_lab::{
    find_regions, law_from_hex, render, run_scorecard, sweep, Scale, LABELS,
};

fn usage() -> ! {
    eprintln!(
        "\
usage:
  lab scorecard <seed> [--law-stamp <hex>]
  lab find-regions <seed> <count>
  lab sweep <seed> <n>"
    );
    std::process::exit(2);
}

fn parse_u64(s: &str, what: &str) -> u64 {
    s.parse::<u64>().unwrap_or_else(|_| {
        eprintln!("bad {what}: {s}");
        std::process::exit(2);
    })
}

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "scorecard" => cmd_scorecard(&args),
        "find-regions" => cmd_find_regions(&args),
        "sweep" => cmd_sweep(&args),
        _ => usage(),
    }
}

fn cmd_scorecard(args: &[String]) {
    let mut seed: Option<u64> = None;
    let mut stamp: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--law-stamp" {
            i += 1;
            if i >= args.len() {
                eprintln!("--law-stamp needs a hex stamp");
                std::process::exit(2);
            }
            stamp = Some(args[i].clone());
        } else if seed.is_none() {
            seed = Some(parse_u64(&args[i], "seed"));
        } else {
            usage();
        }
        i += 1;
    }
    let seed = seed.unwrap_or_else(|| usage());
    let law = match stamp {
        Some(hex) => law_from_hex(&hex).unwrap_or_else(|e| {
            eprintln!("law stamp: {e}");
            std::process::exit(1);
        }),
        None => Law::v0(),
    };
    let card = run_scorecard(&law, seed, Scale::full());
    println!("{}", render(&card));
}

fn cmd_find_regions(args: &[String]) {
    if args.len() != 2 {
        usage();
    }
    let seed = parse_u64(&args[0], "seed");
    let count = parse_u64(&args[1], "count") as usize;
    let found = find_regions(&Law::v0(), seed, count);
    if found.is_empty() {
        eprintln!("no regions found");
        std::process::exit(1);
    }
    for r in &found {
        println!("{r}");
    }
    let missing: Vec<_> = LABELS
        .iter()
        .filter(|l| !found.iter().any(|r| r.label == **l))
        .copied()
        .collect();
    if !missing.is_empty() {
        eprintln!("missing labels: {}", missing.join(", "));
    }
}

fn cmd_sweep(args: &[String]) {
    if args.len() != 2 {
        usage();
    }
    let seed = parse_u64(&args[0], "seed");
    let n = parse_u64(&args[1], "n") as u32;
    let hits = sweep(seed, n);
    println!("sweep seed={seed} n={n}  (from Law::v0(), reduced scorecard)");
    for (i, h) in hits.iter().enumerate() {
        println!(
            "#{}  PASS {}/7  families={}  stamp={}",
            i + 1,
            h.passes,
            h.families,
            h.stamp_hex
        );
    }
}
