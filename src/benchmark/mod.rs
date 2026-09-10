//! Self-describing runtime benchmark recorder.
//!
//! `WATT_BENCH=<seconds>` remains the entry switch. A run now waits for both a
//! minimum warmup and world readiness (with a bounded timeout), records frame
//! and streaming distributions, inventories the machine without optional
//! command-line tools, and emits one stable JSON record.
//!
//! Environment:
//! - `WATT_BENCH_OUTPUT=<path>` appends the same JSON as JSONL.
//! - `WATT_BENCH_SCREENSHOT=<path.png>` writes the final presented frame after
//!   the measured window (blocking engine capture; failure does not drop the report).
//! - `WATT_BENCH_YAW=<rad/s>` steady-rotate rate (default 0.4; `0` = static camera).
//! - `WATT_BENCH_MOVE=<m/s>` +X flight speed (default 0, static camera).
//! - `WATT_BENCH_WARMUP`, `WATT_BENCH_READY_TIMEOUT`, `WATT_BENCH_TAG`,
//!   `WATT_BENCH_POS`, `WATT_BENCH_PRESET`, `WATT_BENCH_SEED`,
//!   `WATT_BENCH_WORLDGEN`, `WATT_BENCH_VISUALS`, `WATT_BENCH_PROFILE`,
//!   `WATT_BENCH_GPU` — see `documentation/performance.md`.

mod json;
mod system;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use voxel_engine::{DVec3, Engine};

use crate::settings::Settings;
use crate::world::{MemoryCensus, StreamGauges, World};

use json::Json;
use system::{SystemInfo, display_json, resident_bytes, settings_json, software_json};

pub(crate) use system::graphics_caps;

const DEFAULT_DURATION_SECS: f64 = 10.0;
const DEFAULT_WARMUP_SECS: f64 = 3.0;
const DEFAULT_READY_TIMEOUT_SECS: f64 = 60.0;
const DEFAULT_YAW_RATE_RAD_S: f64 = 0.4;
const MAX_DURATION_SECS: f64 = 600.0;
const MAX_SAMPLE_RESERVE: usize = 2_000_000;
const SCHEMA_VERSION: u32 = 3;
/// Readiness, stream gauges, and RSS are sampled at this rate on the bench
/// wall clock. Peak gauges are therefore 4 Hz samples, not per-frame maxima.
const WORLD_SAMPLE_HZ: u32 = 4;

/// What the app should do after advancing the recorder by one callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Warming,
    /// Warmup ended without world readiness; measurement starts next frame.
    ReadyTimeout,
    Measuring,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    WaitingToStart,
    Warming,
    Measuring,
    Complete,
}

/// Pins parsed from `WATT_BENCH_WORLDGEN` / `WATT_BENCH_VISUALS`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BenchModPins {
    /// `Some(true)` enables InfiniteDiffusion; `Some(false)` pins classic.
    pub worldgen_diffusion: Option<bool>,
    /// `Some(true)` strips visual mods (core look); `Some(false)` leaves them on.
    pub visuals_core: Option<bool>,
}

/// Complete state for one `WATT_BENCH` run.
pub struct Benchmark {
    duration: Duration,
    min_warmup: Duration,
    ready_timeout: Duration,
    pos: Option<DVec3>,
    /// Flight speed along +X during the run (`WATT_BENCH_MOVE`, m/s); zero
    /// keeps the classic static steady-rotate scenario.
    move_mps: f64,
    /// Steady-rotate rate (`WATT_BENCH_YAW`, rad/s); zero holds the camera.
    yaw_rate: f64,
    /// Final presented-frame PNG (`WATT_BENCH_SCREENSHOT`); captured after
    /// the last sample, never during it.
    screenshot: Option<PathBuf>,
    output: Option<PathBuf>,
    tag: Option<String>,
    visuals_raw: Option<String>,
    phase: Phase,
    warmup_started: Option<Instant>,
    measure_started: Option<Instant>,
    ready_before_measure: bool,
    warmup_elapsed: Duration,
    samples: Vec<f32>,
    first_gauges: Option<StreamGauges>,
    last_gauges: StreamGauges,
    peaks: StreamPeaks,
    system: Option<SystemInfo>,
    started_unix_ms: u128,
    rss_start_bytes: Option<u64>,
    rss_peak_bytes: Option<u64>,
    last_rss_poll: Instant,
    ready_wait_logs: u32,
    world_sampled_at: Option<Instant>,
    cached_ready: bool,
    /// Wall seconds from [`Self::begin`] (first bench frame) to the first
    /// `entry_complete` sample. `None` if the world never settled.
    entry_seconds: Option<f64>,
    census_ready: Option<MemoryCensus>,
    /// Wall time of the measured window, frozen at `Step::Complete` so a
    /// later screenshot readback cannot inflate `wall_seconds`.
    measured_wall: Option<Duration>,
}

impl Benchmark {
    /// Parse the environment. Invalid optional values warn and fall back; a
    /// present `WATT_BENCH` always yields a finite, bounded run.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("WATT_BENCH").ok()?;
        let duration = parse_seconds(
            "WATT_BENCH",
            &raw,
            DEFAULT_DURATION_SECS,
            0.05,
            MAX_DURATION_SECS,
        );
        let min_warmup = env_seconds("WATT_BENCH_WARMUP", DEFAULT_WARMUP_SECS, 0.0, 300.0);
        let ready_timeout = env_seconds(
            "WATT_BENCH_READY_TIMEOUT",
            DEFAULT_READY_TIMEOUT_SECS,
            1.0,
            600.0,
        );
        let pos = std::env::var("WATT_BENCH_POS").ok().and_then(|raw| {
            parse_position(&raw).or_else(|| {
                eprintln!("WATT_BENCH_POS={raw:?} is invalid; using the spawn position");
                None
            })
        });
        let move_mps = env_seconds("WATT_BENCH_MOVE", 0.0, 0.0, 1000.0);
        let yaw_rate = env_seconds("WATT_BENCH_YAW", DEFAULT_YAW_RATE_RAD_S, 0.0, 1000.0);
        let screenshot = parse_screenshot(std::env::var_os("WATT_BENCH_SCREENSHOT"));
        let output = std::env::var_os("WATT_BENCH_OUTPUT")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let tag = std::env::var("WATT_BENCH_TAG")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let visuals_raw = std::env::var("WATT_BENCH_VISUALS").ok();
        let reserve = ((duration.ceil() as usize).saturating_mul(25_000)).min(MAX_SAMPLE_RESERVE);
        let now = Instant::now();
        Some(Self {
            duration: Duration::from_secs_f64(duration),
            min_warmup: Duration::from_secs_f64(min_warmup),
            ready_timeout: Duration::from_secs_f64(ready_timeout),
            pos,
            move_mps,
            yaw_rate,
            screenshot,
            output,
            tag,
            visuals_raw,
            phase: Phase::WaitingToStart,
            warmup_started: None,
            measure_started: None,
            ready_before_measure: false,
            warmup_elapsed: Duration::ZERO,
            samples: Vec::with_capacity(reserve),
            first_gauges: None,
            last_gauges: StreamGauges::default(),
            peaks: StreamPeaks::default(),
            system: None,
            started_unix_ms: unix_millis(),
            rss_start_bytes: None,
            rss_peak_bytes: None,
            last_rss_poll: now,
            ready_wait_logs: 0,
            world_sampled_at: None,
            cached_ready: false,
            entry_seconds: None,
            census_ready: None,
            measured_wall: None,
        })
    }

    /// The only parser for `WATT_BENCH_WORLDGEN` / `WATT_BENCH_VISUALS`.
    /// Applied even when `WATT_BENCH` itself is unset so a pin-and-play run
    /// uses the same accepted values as a timed harness run.
    pub fn mod_pins_from_env() -> BenchModPins {
        let worldgen_diffusion = match std::env::var("WATT_BENCH_WORLDGEN") {
            Ok(value) => match parse_bench_worldgen(&value) {
                Some(parsed) => Some(parsed),
                None => {
                    eprintln!(
                        "WATT_BENCH_WORLDGEN={value:?} not recognized; use classic|diffusion"
                    );
                    None
                }
            },
            Err(_) => None,
        };
        let visuals_core = match std::env::var("WATT_BENCH_VISUALS") {
            Ok(value) => match parse_bench_visuals(&value) {
                Some(parsed) => Some(parsed),
                None => {
                    eprintln!(
                        "WATT_BENCH_VISUALS={value:?} not recognized; use off|core|on|full"
                    );
                    None
                }
            },
            Err(_) => None,
        };
        BenchModPins {
            worldgen_diffusion,
            visuals_core,
        }
    }

    pub fn has_started(&self) -> bool {
        self.phase != Phase::WaitingToStart
    }

    pub fn position(&self) -> Option<DVec3> {
        self.pos
    }

    /// Flight speed along +X (m/s); zero for the static scenario.
    pub fn move_mps(&self) -> f64 {
        self.move_mps
    }

    /// Steady-rotate rate (rad/s); zero holds yaw.
    pub fn yaw_rate(&self) -> f64 {
        self.yaw_rate
    }

    /// PNG path for the post-measure capture, if requested.
    pub fn screenshot_path(&self) -> Option<&Path> {
        self.screenshot.as_deref()
    }

    /// True after the last sample; the next `bench_frame` may capture then finish.
    pub fn measurement_complete(&self) -> bool {
        self.phase == Phase::Complete
    }

    /// Translate the player along +X for a move-scenario run. Flight is forced
    /// so gravity cannot embed the player in terrain the stream has not
    /// prepared under the new x.
    pub fn apply_move(&self, player: &mut crate::player::Player, dt: f32) {
        if self.move_mps > 0.0 {
            player.set_flying(true);
            player.position.x += self.move_mps * dt as f64;
        }
    }

    /// Start metadata collection inside the already-created engine callback,
    /// safely outside the measured interval.
    pub fn begin(&mut self) {
        debug_assert_eq!(self.phase, Phase::WaitingToStart);
        self.phase = Phase::Warming;
        self.warmup_started = Some(Instant::now());
        self.system = Some(SystemInfo::collect());
        self.rss_start_bytes = resident_bytes();
        self.rss_peak_bytes = self.rss_start_bytes;
    }

    /// Far-coordinate setup does synchronous preparation, so retain the old
    /// extra grace while still using the readiness gate.
    pub fn add_warmup(&mut self, extra: Duration) {
        self.min_warmup = self.min_warmup.saturating_add(extra);
    }

    /// True at most once per 5 s of wall time while still warming.
    pub fn wait_log_due(&mut self) -> bool {
        if self.phase != Phase::Warming {
            return false;
        }
        let Some(started) = self.warmup_started else {
            return false;
        };
        let n = (started.elapsed().as_secs() / 5) as u32;
        if n > self.ready_wait_logs {
            self.ready_wait_logs = n;
            true
        } else {
            false
        }
    }

    /// Sample world readiness and stream gauges at [`WORLD_SAMPLE_HZ`]. Peaks
    /// recorded from these samples are 4 Hz, not per-frame maxima. Also
    /// refreshes RSS on the same cadence so `/proc` is not read every frame.
    pub fn poll_world(&mut self, world: &World) -> (bool, StreamGauges) {
        let period = Duration::from_secs_f64(1.0 / f64::from(WORLD_SAMPLE_HZ));
        let due = self
            .world_sampled_at
            .is_none_or(|t| t.elapsed() >= period);
        if due {
            self.world_sampled_at = Some(Instant::now());
            self.cached_ready = world.entry_complete();
            self.last_gauges = world.stream_gauges();
            if self.cached_ready {
                self.stamp_ready();
                if self.census_ready.is_none() {
                    self.census_ready = Some(world.memory_census());
                }
            }
            self.poll_rss();
        }
        (self.cached_ready, self.last_gauges)
    }

    /// Advance warmup/measurement using wall time for boundaries and the
    /// engine's previous-frame duration for the sample itself.
    pub fn step(&mut self, dt: f32, world_ready: bool, gauges: StreamGauges) -> Step {
        self.last_gauges = gauges;
        self.peaks.observe(gauges);
        self.poll_rss();
        match self.phase {
            Phase::WaitingToStart => Step::Warming,
            Phase::Warming => {
                let elapsed = self
                    .warmup_started
                    .expect("begin sets warmup clock")
                    .elapsed();
                let minimum_met = elapsed >= self.min_warmup;
                let timed_out = elapsed >= self.min_warmup.saturating_add(self.ready_timeout);
                if world_ready {
                    self.stamp_ready();
                }
                if !minimum_met || (!world_ready && !timed_out) {
                    return Step::Warming;
                }
                self.ready_before_measure = world_ready;
                self.warmup_elapsed = elapsed;
                self.phase = Phase::Measuring;
                self.measure_started = Some(Instant::now());
                self.first_gauges = Some(gauges);
                if timed_out && !world_ready {
                    eprintln!(
                        "benchmark: world did not become ready within {:.1}s; measuring with readiness=false",
                        self.ready_timeout.as_secs_f64()
                    );
                    return Step::ReadyTimeout;
                }
                // Do not count the final warmup frame as the first sample.
                Step::Warming
            }
            Phase::Measuring => {
                if world_ready {
                    self.stamp_ready();
                }
                if dt.is_finite() && dt > 0.0 {
                    self.samples.push(dt);
                }
                if self
                    .measure_started
                    .expect("measurement clock set")
                    .elapsed()
                    >= self.duration
                {
                    self.measured_wall = Some(
                        self.measure_started
                            .expect("measurement clock set")
                            .elapsed(),
                    );
                    self.phase = Phase::Complete;
                    Step::Complete
                } else {
                    Step::Measuring
                }
            }
            Phase::Complete => Step::Complete,
        }
    }

    /// Blocking capture of the last presented frame. No-op if unset. Prints one
    /// line on failure; the caller still emits the report.
    pub fn capture_screenshot(&self, eng: &mut Engine) {
        let Some(path) = &self.screenshot else {
            return;
        };
        let write = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| voxel_engine::skeleton::screenshot_to(eng, path));
        if let Err(err) = write {
            eprintln!("benchmark: screenshot failed: {err}");
        }
    }

    /// Build and emit the immutable report. This is called after sampling, so
    /// hardware/filesystem probes cannot contaminate headline frame times.
    pub fn finish(
        &mut self,
        settings: &Settings,
        eng: &Engine,
        world: &World,
        actual_position: DVec3,
    ) -> Report {
        let rss_end_bytes = resident_bytes();
        if let Some(rss) = rss_end_bytes {
            self.rss_peak_bytes = Some(self.rss_peak_bytes.unwrap_or(0).max(rss));
        }
        let wall = self.measured_wall.unwrap_or_else(|| {
            self.measure_started.map_or(Duration::ZERO, |t| t.elapsed())
        });
        let stats = FrameStats::from_samples(&self.samples, wall);
        let census_end = world.memory_census();
        let visuals = self.visuals_raw.clone();
        let report = Json::object(vec![
            ("schema_version", Json::from(SCHEMA_VERSION)),
            ("kind", Json::from("project_watt_cubed.runtime_benchmark")),
            ("started_unix_ms", Json::from(self.started_unix_ms)),
            ("tag", Json::optional_str(self.tag.as_deref())),
            ("software", software_json()),
            (
                "system",
                self.system.as_ref().map_or(Json::Null, SystemInfo::to_json),
            ),
            ("display", display_json(eng, settings, self.system.as_ref())),
            ("settings", settings_json(settings, eng)),
            (
                "scenario",
                Json::object(vec![
                    (
                        "name",
                        Json::from(if self.move_mps > 0.0 {
                            "rotate_and_fly"
                        } else {
                            "steady_rotate"
                        }),
                    ),
                    ("move_mps", Json::number(self.move_mps)),
                    ("seed", Json::from(world.seed())),
                    ("worldgen", Json::from(world.worldgen_kind())),
                    ("visuals", Json::optional_str(visuals.as_deref())),
                    ("requested_position", position_json(self.pos)),
                    ("actual_position", position_json(Some(actual_position))),
                    ("yaw_rate_rad_s", Json::number(self.yaw_rate)),
                    ("screenshot", path_json(self.screenshot.as_deref())),
                    (
                        "requested_duration_s",
                        Json::number(self.duration.as_secs_f64()),
                    ),
                    (
                        "minimum_warmup_s",
                        Json::number(self.min_warmup.as_secs_f64()),
                    ),
                    (
                        "actual_warmup_s",
                        Json::number(self.warmup_elapsed.as_secs_f64()),
                    ),
                    (
                        "ready_timeout_s",
                        Json::number(self.ready_timeout.as_secs_f64()),
                    ),
                    (
                        "ready_before_measure",
                        Json::from(self.ready_before_measure),
                    ),
                    ("ready_at_end", Json::from(world.entry_complete())),
                    (
                        "entry_seconds",
                        Json::optional_number(self.entry_seconds),
                    ),
                    (
                        "profiling_enabled",
                        Json::from(matches!(
                            std::env::var("WATT_BENCH_PROFILE").as_deref(),
                            Ok("1")
                        )),
                    ),
                ]),
            ),
            ("frames", stats.to_json()),
            (
                "memory",
                Json::object(vec![
                    ("rss_start_bytes", Json::optional_u64(self.rss_start_bytes)),
                    ("rss_peak_bytes", Json::optional_u64(self.rss_peak_bytes)),
                    ("rss_end_bytes", Json::optional_u64(rss_end_bytes)),
                    (
                        "census_ready",
                        self.census_ready.map_or(Json::Null, census_json),
                    ),
                    ("census_end", census_json(census_end)),
                ]),
            ),
            (
                "streaming",
                Json::object(vec![
                    (
                        "start",
                        self.first_gauges.map_or(Json::Null, stream_gauges_json),
                    ),
                    ("end", stream_gauges_json(self.last_gauges)),
                    ("peaks", self.peaks.to_json()),
                    ("peaks_sample_hz", Json::from(WORLD_SAMPLE_HZ)),
                ]),
            ),
        ]);
        let json = report.render();
        let summary = format!(
            "BENCH frames={} avg_fps={} p1_fps={} avg_ms={} p99_ms={} max_ms={} hitches_33ms={} rss_mb={} ready={} ready_s={} preset={} window={}x{} gpu={}",
            stats.frames,
            fmt_opt(stats.avg_fps, 0),
            fmt_opt(stats.p1_fps, 0),
            fmt_opt(stats.avg_ms, 3),
            fmt_opt(stats.p99_ms, 3),
            fmt_opt(stats.max_ms, 3),
            stats.over_33ms,
            rss_end_bytes.unwrap_or(0) / (1024 * 1024),
            self.ready_before_measure,
            fmt_opt(self.entry_seconds, 3),
            settings.preset.label().to_ascii_lowercase(),
            eng.screen_width(),
            eng.screen_height(),
            self.system.as_ref().map_or("unknown", SystemInfo::gpu_name),
        );
        let mem_line = format!(
            "BENCH_MEM ready_total={} end_total={} chunks=u{}/p{}/d{} light=u{}/c{} light_bytes=u{}/c{} mesh={} edits={} lod={} queues={}",
            self.census_ready.map_or_else(|| "n/a".into(), |c| c.total.to_string()),
            census_end.total,
            census_end.chunk_uniform_count,
            census_end.chunk_paletted_count,
            census_end.chunk_dense_count,
            census_end.light_uniform_count,
            census_end.light_cells_count,
            census_end.light_uniform_bytes,
            census_end.light_cells_bytes,
            census_end.mesh_cpu_bytes,
            census_end.edit_overlay_bytes,
            census_end.section_lod_bytes,
            census_end.worklist_bytes,
        );
        Report {
            summary,
            mem_line,
            json,
            output: self.output.clone(),
        }
    }

    fn stamp_ready(&mut self) {
        if self.entry_seconds.is_some() {
            return;
        }
        let Some(started) = self.warmup_started else {
            return;
        };
        self.entry_seconds = Some(started.elapsed().as_secs_f64());
    }

    fn poll_rss(&mut self) {
        if self.last_rss_poll.elapsed() < Duration::from_secs(1) && self.rss_peak_bytes.is_some() {
            return;
        }
        self.last_rss_poll = Instant::now();
        if let Some(rss) = resident_bytes() {
            self.rss_peak_bytes = Some(self.rss_peak_bytes.unwrap_or(0).max(rss));
        }
    }
}

pub struct Report {
    summary: String,
    mem_line: String,
    json: String,
    output: Option<PathBuf>,
}

impl Report {
    pub fn emit(self) {
        println!("{}", self.summary);
        println!("{}", self.mem_line);
        println!("BENCH_JSON {}", self.json);
        let Some(path) = self.output else { return };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
            && let Err(err) = fs::create_dir_all(parent)
        {
            eprintln!("benchmark: could not create {}: {err}", parent.display());
            return;
        }
        let mut line = self.json;
        line.push('\n');
        let write = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| {
                // One O_APPEND write keeps concurrent benchmark records from
                // interleaving their JSON and newline halves.
                file.write_all(line.as_bytes())?;
                file.flush()
            });
        if let Err(err) = write {
            eprintln!("benchmark: could not append {}: {err}", path.display());
        } else {
            eprintln!("benchmark: appended JSONL record to {}", path.display());
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameStats {
    frames: usize,
    sampled_secs: f64,
    wall_secs: f64,
    avg_fps: Option<f64>,
    throughput_fps: Option<f64>,
    p1_fps: Option<f64>,
    worst_1pct_avg_fps: Option<f64>,
    avg_ms: Option<f64>,
    min_ms: Option<f64>,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    p99_ms: Option<f64>,
    p999_ms: Option<f64>,
    max_ms: Option<f64>,
    stddev_ms: Option<f64>,
    over_16ms: usize,
    over_33ms: usize,
    over_50ms: usize,
}

impl FrameStats {
    fn from_samples(samples: &[f32], wall: Duration) -> Self {
        if samples.is_empty() {
            return Self {
                wall_secs: wall.as_secs_f64(),
                ..Self::default()
            };
        }
        let mut sorted: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        sorted.sort_by(f64::total_cmp);
        let frames = sorted.len();
        let total: f64 = sorted.iter().sum();
        let mean = total / frames as f64;
        let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / frames as f64;
        let p99 = percentile(&sorted, 0.99);
        let worst_count = frames.div_ceil(100).max(1);
        let worst_total: f64 = sorted[frames - worst_count..].iter().sum();
        Self {
            frames,
            sampled_secs: total,
            wall_secs: wall.as_secs_f64(),
            avg_fps: Some(frames as f64 / total),
            throughput_fps: (wall.as_secs_f64() > 0.0)
                .then_some(frames as f64 / wall.as_secs_f64()),
            p1_fps: Some(1.0 / p99),
            worst_1pct_avg_fps: Some(worst_count as f64 / worst_total),
            avg_ms: Some(mean * 1000.0),
            min_ms: Some(sorted[0] * 1000.0),
            p50_ms: Some(percentile(&sorted, 0.50) * 1000.0),
            p95_ms: Some(percentile(&sorted, 0.95) * 1000.0),
            p99_ms: Some(p99 * 1000.0),
            p999_ms: Some(percentile(&sorted, 0.999) * 1000.0),
            max_ms: Some(sorted[frames - 1] * 1000.0),
            stddev_ms: Some(variance.sqrt() * 1000.0),
            over_16ms: sorted.iter().filter(|&&s| s > 1.0 / 60.0).count(),
            over_33ms: sorted.iter().filter(|&&s| s > 1.0 / 30.0).count(),
            over_50ms: sorted.iter().filter(|&&s| s > 0.050).count(),
        }
    }

    fn to_json(self) -> Json {
        Json::object(vec![
            ("count", Json::from(self.frames)),
            ("sampled_seconds", Json::number(self.sampled_secs)),
            ("wall_seconds", Json::number(self.wall_secs)),
            ("average_fps", Json::optional_number(self.avg_fps)),
            (
                "wall_throughput_fps",
                Json::optional_number(self.throughput_fps),
            ),
            ("p1_fps", Json::optional_number(self.p1_fps)),
            (
                "worst_1pct_average_fps",
                Json::optional_number(self.worst_1pct_avg_fps),
            ),
            ("average_ms", Json::optional_number(self.avg_ms)),
            ("minimum_ms", Json::optional_number(self.min_ms)),
            ("p50_ms", Json::optional_number(self.p50_ms)),
            ("p95_ms", Json::optional_number(self.p95_ms)),
            ("p99_ms", Json::optional_number(self.p99_ms)),
            ("p99_9_ms", Json::optional_number(self.p999_ms)),
            ("maximum_ms", Json::optional_number(self.max_ms)),
            ("stddev_ms", Json::optional_number(self.stddev_ms)),
            ("frames_over_16_67ms", Json::from(self.over_16ms)),
            ("frames_over_33_33ms", Json::from(self.over_33ms)),
            ("frames_over_50ms", Json::from(self.over_50ms)),
            // Engine::frames_rendered / frames_coalesced are not on this engine
            // revision; filled when those accessors land.
            ("rendered", Json::Null),
            ("coalesced", Json::Null),
            ("rendered_fps", Json::Null),
        ])
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = ((sorted.len() as f64 * q).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

/// Peak stream gauges observed during the run. Values are 4 Hz wall-clock
/// samples (see [`WORLD_SAMPLE_HZ`]), not per-frame maxima.
#[derive(Clone, Copy, Debug)]
struct StreamPeaks {
    max_chunks: usize,
    max_generating: usize,
    max_mesh_worklist: usize,
    max_upload_queue: usize,
    max_light_worklist: usize,
    max_light_inflight: usize,
    max_light_apply_queue: usize,
    max_worker_near_queue: usize,
    max_worker_far_queue: usize,
    min_active_workers: usize,
    max_worker_capacity: usize,
    max_speed_mps: f64,
    min_effort: f32,
}

impl Default for StreamPeaks {
    fn default() -> Self {
        Self {
            max_chunks: 0,
            max_generating: 0,
            max_mesh_worklist: 0,
            max_upload_queue: 0,
            max_light_worklist: 0,
            max_light_inflight: 0,
            max_light_apply_queue: 0,
            max_worker_near_queue: 0,
            max_worker_far_queue: 0,
            min_active_workers: usize::MAX,
            max_worker_capacity: 0,
            max_speed_mps: 0.0,
            min_effort: 1.0,
        }
    }
}

impl StreamPeaks {
    fn observe(&mut self, g: StreamGauges) {
        self.max_chunks = self.max_chunks.max(g.chunks);
        self.max_generating = self.max_generating.max(g.generating);
        self.max_mesh_worklist = self.max_mesh_worklist.max(g.mesh_worklist);
        self.max_upload_queue = self.max_upload_queue.max(g.upload_queue);
        self.max_light_worklist = self.max_light_worklist.max(g.light_worklist);
        self.max_light_inflight = self.max_light_inflight.max(g.light_inflight);
        self.max_light_apply_queue = self.max_light_apply_queue.max(g.light_apply_queue);
        self.max_worker_near_queue = self.max_worker_near_queue.max(g.worker_near_queue);
        self.max_worker_far_queue = self.max_worker_far_queue.max(g.worker_far_queue);
        if g.worker_capacity != 0 {
            self.min_active_workers = self.min_active_workers.min(g.active_workers);
        }
        self.max_worker_capacity = self.max_worker_capacity.max(g.worker_capacity);
        self.max_speed_mps = self.max_speed_mps.max(g.travel_speed_mps);
        self.min_effort = self.min_effort.min(g.effort);
    }

    fn to_json(self) -> Json {
        Json::object(vec![
            ("chunks", Json::from(self.max_chunks)),
            ("generating", Json::from(self.max_generating)),
            ("mesh_worklist", Json::from(self.max_mesh_worklist)),
            ("upload_queue", Json::from(self.max_upload_queue)),
            ("light_worklist", Json::from(self.max_light_worklist)),
            ("light_inflight", Json::from(self.max_light_inflight)),
            ("light_apply_queue", Json::from(self.max_light_apply_queue)),
            ("worker_near_queue", Json::from(self.max_worker_near_queue)),
            ("worker_far_queue", Json::from(self.max_worker_far_queue)),
            (
                "minimum_active_workers",
                if self.min_active_workers == usize::MAX {
                    Json::Null
                } else {
                    Json::from(self.min_active_workers)
                },
            ),
            ("worker_capacity", Json::from(self.max_worker_capacity)),
            ("travel_speed_mps", Json::number(self.max_speed_mps)),
            ("minimum_effort", Json::number(f64::from(self.min_effort))),
        ])
    }
}

fn census_json(c: MemoryCensus) -> Json {
    Json::object(vec![
        ("chunk_uniform_bytes", Json::from(c.chunk_uniform_bytes)),
        ("chunk_uniform_count", Json::from(c.chunk_uniform_count)),
        ("chunk_paletted_bytes", Json::from(c.chunk_paletted_bytes)),
        ("chunk_paletted_count", Json::from(c.chunk_paletted_count)),
        ("chunk_dense_bytes", Json::from(c.chunk_dense_bytes)),
        ("chunk_dense_count", Json::from(c.chunk_dense_count)),
        ("light_uniform_bytes", Json::from(c.light_uniform_bytes)),
        ("light_uniform_count", Json::from(c.light_uniform_count)),
        ("light_cells_bytes", Json::from(c.light_cells_bytes)),
        ("light_cells_count", Json::from(c.light_cells_count)),
        ("mesh_cpu_bytes", Json::from(c.mesh_cpu_bytes)),
        ("edit_overlay_bytes", Json::from(c.edit_overlay_bytes)),
        ("section_lod_bytes", Json::from(c.section_lod_bytes)),
        ("worklist_bytes", Json::from(c.worklist_bytes)),
        ("total", Json::from(c.total)),
    ])
}

fn stream_gauges_json(g: StreamGauges) -> Json {
    Json::object(vec![
        ("chunks", Json::from(g.chunks)),
        ("generating", Json::from(g.generating)),
        ("mesh_worklist", Json::from(g.mesh_worklist)),
        ("upload_queue", Json::from(g.upload_queue)),
        ("light_worklist", Json::from(g.light_worklist)),
        ("light_inflight", Json::from(g.light_inflight)),
        ("light_apply_queue", Json::from(g.light_apply_queue)),
        ("worker_near_queue", Json::from(g.worker_near_queue)),
        ("worker_far_queue", Json::from(g.worker_far_queue)),
        ("active_workers", Json::from(g.active_workers)),
        ("worker_capacity", Json::from(g.worker_capacity)),
        ("travel_speed_mps", Json::number(g.travel_speed_mps)),
        ("effort", Json::number(f64::from(g.effort))),
        ("light_admitted", Json::from(g.light_admitted as u64)),
        ("light_admitted_last", Json::from(g.light_admitted_last)),
        ("light_seed_inserts", Json::from(g.light_seed_inserts as u64)),
        ("light_seed_store", Json::from(g.light_seed_split.store)),
        ("light_seed_border", Json::from(g.light_seed_split.border)),
        ("light_seed_edit", Json::from(g.light_seed_split.edit)),
        ("light_seed_degrade", Json::from(g.light_seed_split.degrade)),
        ("light_seed_terminal", Json::from(g.light_seed_split.terminal)),
        ("light_seed_remesh", Json::from(g.light_seed_split.remesh)),
        ("remesh_async_calls", Json::from(g.remesh_async_calls)),
        ("drop_stale_uploads", Json::from(g.drop_stale_uploads)),
        ("drop_stale_this_frame", Json::from(g.drop_stale_this_frame as u64)),
        (
            "remesh_between_upload_mean",
            Json::number(f64::from(g.remesh_between_upload_mean)),
        ),
        (
            "remesh_between_upload_p95",
            Json::number(f64::from(g.remesh_between_upload_p95)),
        ),
        ("remesh_between_upload_n", Json::from(g.remesh_between_upload_n)),
        (
            "mesh_jobs_before_fixpoint_mean",
            Json::number(f64::from(g.mesh_jobs_before_fixpoint_mean)),
        ),
        (
            "mesh_jobs_before_fixpoint_p95",
            Json::number(f64::from(g.mesh_jobs_before_fixpoint_p95)),
        ),
        (
            "mesh_jobs_before_fixpoint_n",
            Json::from(g.mesh_jobs_before_fixpoint_n),
        ),
    ])
}

fn position_json(pos: Option<DVec3>) -> Json {
    pos.map_or(Json::Null, |p| {
        Json::array(vec![
            Json::number(p.x),
            Json::number(p.y),
            Json::number(p.z),
        ])
    })
}

fn path_json(path: Option<&Path>) -> Json {
    path.map_or(Json::Null, |p| Json::from(p.to_string_lossy().as_ref()))
}

fn parse_screenshot(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    value.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `Some(true)` enables InfiniteDiffusion; `Some(false)` pins classic.
fn parse_bench_worldgen(value: &str) -> Option<bool> {
    match value {
        "diffusion" => Some(true),
        "classic" => Some(false),
        _ => None,
    }
}

/// `Some(true)` strips the visual mods (core look); `Some(false)` leaves them on.
fn parse_bench_visuals(value: &str) -> Option<bool> {
    match value {
        "off" | "core" => Some(true),
        "on" | "full" => Some(false),
        _ => None,
    }
}

fn parse_position(raw: &str) -> Option<DVec3> {
    let mut parts = raw.split(',').map(|p| p.trim().parse::<f64>());
    let position = DVec3::new(
        parts.next()?.ok()?,
        parts.next()?.ok()?,
        parts.next()?.ok()?,
    );
    (parts.next().is_none() && position.is_finite()).then_some(position)
}

fn parse_seconds(name: &str, raw: &str, default: f64, min: f64, max: f64) -> f64 {
    match raw.trim().parse::<f64>() {
        Ok(value) if value.is_finite() => {
            let clamped = value.clamp(min, max);
            if clamped != value {
                eprintln!("{name}={value} is outside {min}..={max}; clamped to {clamped}");
            }
            clamped
        }
        _ => {
            eprintln!("{name}={raw:?} is invalid; using {default}");
            default
        }
    }
}

fn env_seconds(name: &str, default: f64, min: f64, max: f64) -> f64 {
    std::env::var(name)
        .map(|raw| parse_seconds(name, &raw, default, min, max))
        .unwrap_or(default)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn fmt_opt(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(|| "n/a".into(), |v| format!("{v:.decimals$}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_statistics_are_defined_and_percentiles_are_nearest_rank() {
        let samples = [0.001, 0.002, 0.003, 0.004, 0.100];
        let stats = FrameStats::from_samples(&samples, Duration::from_millis(110));
        assert_eq!(stats.frames, 5);
        assert!(stats.p50_ms.is_some_and(|ms| (ms - 3.0).abs() < 0.001));
        assert!(stats.p99_ms.is_some_and(|ms| (ms - 100.0).abs() < 0.001));
        assert_eq!(stats.over_50ms, 1);
        assert!(stats.avg_fps.is_some_and(|fps| fps > 45.0 && fps < 46.0));
    }

    #[test]
    fn empty_statistics_emit_nulls_instead_of_nan_or_infinity() {
        let stats = FrameStats::from_samples(&[], Duration::ZERO);
        let json = stats.to_json().render();
        assert!(json.contains("\"average_fps\":null"));
        assert!(!json.contains("NaN"));
        assert!(!json.contains("inf"));
    }

    #[test]
    fn apply_move_sets_flying_so_gravity_cannot_embed() {
        let mut bench = test_bench(Duration::from_millis(1), Duration::from_millis(1));
        bench.move_mps = 40.0;
        let mut player = crate::player::Player::new(DVec3::new(0.0, 40.0, 0.0));
        assert!(!player.flying());
        bench.apply_move(&mut player, 0.25);
        assert!(player.flying(), "move scenario must fly rather than walk");
        assert!((player.position.x - 10.0).abs() < 1e-9);
        bench.apply_move(&mut player, 0.0);
        assert!(player.flying());
    }

    #[test]
    fn json_escaping_and_position_validation_are_strict() {
        assert_eq!(Json::from("a\n\"b").render(), "\"a\\n\\\"b\"");
        assert!(parse_position("1,2,3").is_some());
        assert!(parse_position("1,2,3,4").is_none());
        assert!(parse_position("NaN,2,3").is_none());
    }

    fn test_bench(min_warmup: Duration, ready_timeout: Duration) -> Benchmark {
        Benchmark {
            duration: Duration::from_secs(1),
            min_warmup,
            ready_timeout,
            pos: None,
            move_mps: 0.0,
            yaw_rate: DEFAULT_YAW_RATE_RAD_S,
            screenshot: None,
            output: None,
            tag: None,
            visuals_raw: None,
            phase: Phase::WaitingToStart,
            warmup_started: None,
            measure_started: None,
            ready_before_measure: false,
            warmup_elapsed: Duration::ZERO,
            samples: Vec::new(),
            first_gauges: None,
            last_gauges: StreamGauges::default(),
            peaks: StreamPeaks::default(),
            system: None,
            started_unix_ms: 0,
            rss_start_bytes: None,
            rss_peak_bytes: None,
            last_rss_poll: Instant::now(),
            ready_wait_logs: 0,
            world_sampled_at: None,
            cached_ready: false,
            entry_seconds: None,
            census_ready: None,
            measured_wall: None,
        }
    }

    #[test]
    fn entry_seconds_stamps_on_the_first_ready_frame() {
        let mut bench = test_bench(Duration::from_millis(1), Duration::from_secs(60));
        bench.begin();
        bench.warmup_started = Some(Instant::now() - Duration::from_millis(250));
        let gauges = StreamGauges::default();
        assert_eq!(bench.step(0.016, false, gauges), Step::Warming);
        assert!(bench.entry_seconds.is_none());
        assert_eq!(bench.step(0.016, true, gauges), Step::Warming);
        let secs = bench.entry_seconds.expect("ready frame stamps entry_seconds");
        assert!(secs >= 0.25, "got {secs}");
        assert!(secs < 2.0, "got {secs}");
        assert_eq!(bench.step(0.016, true, gauges), Step::Measuring);
        let again = bench.entry_seconds.expect("stays set");
        assert_eq!(format!("{secs:.6}"), format!("{again:.6}"));
    }

    #[test]
    fn census_json_contains_the_new_fields() {
        let json = census_json(MemoryCensus {
            chunk_uniform_bytes: 4,
            chunk_uniform_count: 1,
            total: 4,
            ..MemoryCensus::default()
        })
        .render();
        assert!(json.contains("\"chunk_uniform_bytes\":4"));
        assert!(json.contains("\"chunk_uniform_count\":1"));
        assert!(json.contains("\"light_uniform_count\":0"));
        assert!(json.contains("\"light_uniform_bytes\":0"));
        assert!(json.contains("\"light_cells_count\":0"));
        assert!(json.contains("\"light_cells_bytes\":0"));
        assert!(json.contains("\"mesh_cpu_bytes\":0"));
        assert!(json.contains("\"total\":4"));
        let frames = FrameStats::from_samples(&[], Duration::ZERO).to_json().render();
        assert!(frames.contains("\"rendered\":null"));
        assert!(frames.contains("\"coalesced\":null"));
        assert!(frames.contains("\"rendered_fps\":null"));
        let scenario = Json::object(vec![
            ("entry_seconds", Json::optional_number(Some(1.5))),
        ])
        .render();
        assert!(scenario.contains("\"entry_seconds\":1.5"));
        let missing = Json::object(vec![("entry_seconds", Json::optional_number(None))]).render();
        assert!(missing.contains("\"entry_seconds\":null"));
    }

    #[test]
    fn ready_timeout_is_signaled_once_then_measurement_starts() {
        let mut bench = test_bench(Duration::from_millis(1), Duration::from_millis(1));
        bench.begin();
        bench.warmup_started = Some(Instant::now() - Duration::from_secs(1));
        let gauges = StreamGauges::default();
        assert_eq!(bench.step(0.016, false, gauges), Step::ReadyTimeout);
        assert!(!bench.ready_before_measure);
        assert_eq!(bench.step(0.016, false, gauges), Step::Measuring);
    }

    #[test]
    fn wait_log_due_fires_once_per_five_seconds_while_warming() {
        let mut bench = test_bench(Duration::from_secs(60), Duration::from_secs(60));
        bench.begin();
        assert!(!bench.wait_log_due());
        bench.warmup_started = Some(Instant::now() - Duration::from_secs(5));
        assert!(bench.wait_log_due());
        assert!(!bench.wait_log_due());
        bench.warmup_started = Some(Instant::now() - Duration::from_secs(10));
        assert!(bench.wait_log_due());
        assert!(!bench.wait_log_due());
    }

    #[test]
    fn bench_env_accepted_values() {
        assert_eq!(parse_bench_worldgen("diffusion"), Some(true));
        assert_eq!(parse_bench_worldgen("classic"), Some(false));
        assert_eq!(parse_bench_worldgen("Diffusion"), None);
        assert_eq!(parse_bench_visuals("off"), Some(true));
        assert_eq!(parse_bench_visuals("core"), Some(true));
        assert_eq!(parse_bench_visuals("on"), Some(false));
        assert_eq!(parse_bench_visuals("full"), Some(false));
        assert_eq!(parse_bench_visuals("pretty"), None);
    }

    #[test]
    fn move_and_yaw_parse_like_seconds() {
        assert_eq!(parse_seconds("WATT_BENCH_MOVE", "0", 0.0, 0.0, 1000.0), 0.0);
        assert_eq!(parse_seconds("WATT_BENCH_MOVE", "40", 0.0, 0.0, 1000.0), 40.0);
        assert_eq!(parse_seconds("WATT_BENCH_MOVE", "-5", 0.0, 0.0, 1000.0), 0.0);
        assert_eq!(
            parse_seconds("WATT_BENCH_MOVE", "nope", 0.0, 0.0, 1000.0),
            0.0
        );
        assert_eq!(
            parse_seconds("WATT_BENCH_YAW", "0", DEFAULT_YAW_RATE_RAD_S, 0.0, 1000.0),
            0.0
        );
        assert_eq!(
            parse_seconds(
                "WATT_BENCH_YAW",
                "0.4",
                DEFAULT_YAW_RATE_RAD_S,
                0.0,
                1000.0
            ),
            0.4
        );
        assert_eq!(
            parse_seconds(
                "WATT_BENCH_YAW",
                "not-a-number",
                DEFAULT_YAW_RATE_RAD_S,
                0.0,
                1000.0
            ),
            DEFAULT_YAW_RATE_RAD_S
        );
        assert_eq!(
            parse_seconds("WATT_BENCH_YAW", "1e9", DEFAULT_YAW_RATE_RAD_S, 0.0, 1000.0),
            1000.0
        );
    }

    #[test]
    fn screenshot_env_is_a_png_path_or_absent() {
        assert_eq!(parse_screenshot(None), None);
        assert_eq!(parse_screenshot(Some(std::ffi::OsString::from(""))), None);
        assert_eq!(
            parse_screenshot(Some(std::ffi::OsString::from("captures/final.png"))),
            Some(PathBuf::from("captures/final.png"))
        );
    }

    #[test]
    fn scenario_report_includes_screenshot_and_live_yaw_rate() {
        assert_eq!(path_json(None).render(), "null");
        assert_eq!(
            path_json(Some(Path::new("out.png"))).render(),
            "\"out.png\""
        );
        let mut bench = test_bench(Duration::from_millis(1), Duration::from_millis(1));
        bench.yaw_rate = 0.0;
        bench.screenshot = Some(PathBuf::from("shot.png"));
        assert_eq!(
            Json::object(vec![
                ("yaw_rate_rad_s", Json::number(bench.yaw_rate)),
                ("screenshot", path_json(bench.screenshot.as_deref())),
            ])
            .render(),
            "{\"yaw_rate_rad_s\":0,\"screenshot\":\"shot.png\"}"
        );
        bench.screenshot = None;
        bench.yaw_rate = DEFAULT_YAW_RATE_RAD_S;
        assert_eq!(
            Json::object(vec![
                ("yaw_rate_rad_s", Json::number(bench.yaw_rate)),
                ("screenshot", path_json(bench.screenshot.as_deref())),
            ])
            .render(),
            "{\"yaw_rate_rad_s\":0.4,\"screenshot\":null}"
        );
    }

    #[test]
    fn complete_is_sticky_and_does_not_sample_further() {
        let mut bench = test_bench(Duration::from_millis(1), Duration::from_millis(1));
        bench.begin();
        bench.warmup_started = Some(Instant::now() - Duration::from_secs(1));
        let gauges = StreamGauges::default();
        assert_eq!(bench.step(0.016, true, gauges), Step::Warming);
        bench.measure_started = Some(Instant::now() - Duration::from_secs(1));
        assert_eq!(bench.step(0.016, true, gauges), Step::Complete);
        assert!(bench.measurement_complete());
        assert_eq!(bench.samples.len(), 1);
        assert!(bench.measured_wall.is_some());
        assert_eq!(bench.step(0.016, true, gauges), Step::Complete);
        assert_eq!(bench.samples.len(), 1);
    }

    #[test]
    fn from_env_parses_screenshot_yaw_and_move() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let keys = [
            "WATT_BENCH",
            "WATT_BENCH_SCREENSHOT",
            "WATT_BENCH_YAW",
            "WATT_BENCH_MOVE",
        ];
        let previous: Vec<_> = keys
            .iter()
            .map(|k| (*k, std::env::var_os(k)))
            .collect();
        unsafe {
            std::env::set_var("WATT_BENCH", "1");
            std::env::remove_var("WATT_BENCH_SCREENSHOT");
            std::env::remove_var("WATT_BENCH_YAW");
            std::env::remove_var("WATT_BENCH_MOVE");
        }
        let bench = Benchmark::from_env().expect("WATT_BENCH set");
        assert!(bench.screenshot_path().is_none());
        assert!((bench.yaw_rate() - DEFAULT_YAW_RATE_RAD_S).abs() < 1e-12);
        assert_eq!(bench.move_mps(), 0.0);

        unsafe {
            std::env::set_var("WATT_BENCH_SCREENSHOT", "captures/final.png");
            std::env::set_var("WATT_BENCH_YAW", "0");
            std::env::set_var("WATT_BENCH_MOVE", "40");
        }
        let bench = Benchmark::from_env().expect("WATT_BENCH set");
        assert_eq!(
            bench.screenshot_path(),
            Some(Path::new("captures/final.png"))
        );
        assert_eq!(bench.yaw_rate(), 0.0);
        assert_eq!(bench.move_mps(), 40.0);

        unsafe {
            std::env::set_var("WATT_BENCH_SCREENSHOT", "");
            std::env::set_var("WATT_BENCH_YAW", "not-a-number");
            std::env::set_var("WATT_BENCH_MOVE", "-5");
        }
        let bench = Benchmark::from_env().expect("WATT_BENCH set");
        assert!(bench.screenshot_path().is_none());
        assert!((bench.yaw_rate() - DEFAULT_YAW_RATE_RAD_S).abs() < 1e-12);
        assert_eq!(bench.move_mps(), 0.0);

        for (k, v) in previous {
            unsafe {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
