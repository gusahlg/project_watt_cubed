//! `golden` — run the golden-shot acceptance harness.
//!
//!   cargo run --bin golden            # check the current acceptance set
//!   cargo run --bin golden -- bless   # regenerate blessed goldens
//!
//! The four shots (`night_field`, `cave_interior`, `shadow_boundary`,
//! `horizon_fog_vs_sky`) ship WITHOUT blessed PNGs, so `bless` must run once
//! before a plain check can pass their ImageMatch — until then each fails LOUD
//! (a "load golden … No such file" `Failure`, never a panic).
//!
//! A thin runner over the live harness: builds the acceptance set, prints the
//! entry-time number on the golden seed, then runs the image-match, sky-hole,
//! entry-time, and frame-time criteria.

use project_watt_cubed::modding::GameBuild;

fn main() {
    project_watt_cubed::harness::golden_main(GameBuild::vanilla());
}
