//! Periodic autosave with dirty tracking.
//!
//! A cheap main-thread snapshot (overlay-map clone + player/mod records) is
//! handed to the writer thread, which encodes the document and writes it.
//! `in_flight` ensures at most one pending write to prevent races. The *how
//! often* — the interval throttle — is not here: it rides the scheduler's
//! frame clock as an interval gate (`sched::Scheduler::register_interval`),
//! so no wall-clock `Instant` timer is hand-rolled in this lane.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use super::bridge::SaveSnapshot;
use super::slot::{SaveError, SlotId};
use super::store;

/// How often the world is autosaved when dirty. Consumed by the scheduler's
/// interval gate (registered per world by `Game::new`), not by any timer here.
pub const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

pub struct Autosaver {
    /// Last generation successfully written.
    saved_gen: u64,
    /// Generation handed to the writer thread, promoted on success.
    pending_gen: u64,
    in_flight: bool,
    /// Spawned only for an actually-due periodic write. Clean worlds, disabled
    /// autosave, and synchronous exit-only saves create no background thread.
    writer: Option<Writer>,
}

struct Writer {
    tx: mpsc::Sender<(SlotId, SaveSnapshot)>,
    rx: mpsc::Receiver<Result<(), SaveError>>,
}

/// What one `tick` did, so the caller can surface "Saving…" / failures.
#[derive(Debug)]
pub enum Tick {
    Idle,
    Started,
    Finished(Result<(), SaveError>),
}

fn writer_died() -> SaveError {
    SaveError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "autosave thread died"))
}

impl Autosaver {
    pub fn new() -> Self {
        Self { saved_gen: 0, pending_gen: 0, in_flight: false, writer: None }
    }

    fn spawn_writer() -> Writer {
        let (tx, job_rx) = mpsc::channel::<(SlotId, SaveSnapshot)>();
        let (done_tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("autosave".to_string())
            .spawn(move || {
                // Exits when the Autosaver (and thus `tx`) is dropped.
                while let Ok((id, snapshot)) = job_rx.recv() {
                    let result = snapshot
                        .encode()
                        .and_then(|bytes| store::write(&id, &bytes).map_err(SaveError::from));
                    let _ = done_tx.send(result);
                }
            })
            .expect("spawn autosave thread");
        Writer { tx, rx }
    }

    /// The live writer for an in-flight write. Both `poll` and `flush_now`
    /// only reach here after checking `in_flight`, which is never set true
    /// without a writer.
    fn writer(&self) -> &Writer {
        self.writer.as_ref().expect("in-flight autosave must have a writer")
    }

    /// Note a freshly loaded/created world so its current state doesn't count
    /// as dirty.
    pub fn reset(&mut self, generation: u64) {
        self.saved_gen = generation;
    }

    /// Poll a completing background write. `Finished` once it lands; `Idle`
    /// while a write is still in flight or none is. Call once per frame; with
    /// no write in flight (the common case) this is a branch, no channel read.
    pub fn poll(&mut self) -> Tick {
        if !self.in_flight {
            return Tick::Idle;
        }
        match self.writer().rx.try_recv() {
            Ok(result) => {
                self.in_flight = false;
                if result.is_ok() {
                    self.saved_gen = self.pending_gen;
                }
                Tick::Finished(result)
            }
            Err(mpsc::TryRecvError::Empty) => Tick::Idle,
            Err(mpsc::TryRecvError::Disconnected) => {
                // Drop the dead channel so a later due write spawns a fresh worker.
                self.in_flight = false;
                self.writer = None;
                Tick::Finished(Err(writer_died()))
            }
        }
    }

    /// A new write is warranted iff the world advanced past what's saved and no
    /// write is already in flight (the artifact mutex — at most one pending
    /// write). The scheduler's interval gate throttles how often the caller
    /// acts on this.
    pub fn wants_write(&self, generation: u64) -> bool {
        !self.in_flight && generation != self.saved_gen
    }

    /// Snapshot on the caller's thread and hand it to the writer, which encodes
    /// and writes. The writer thread is created on the first actually-due write.
    /// The caller must have checked [`Self::wants_write`] and the scheduler's
    /// interval gate first.
    ///
    /// 100k-edit overlay (this box, 2026-09-09, release
    /// `autosave_snapshot_and_encode_at_100k_edits`, median of 5): snapshot
    /// 1.12 ms on the caller; encode 7.07 ms + write 2.54 ms (9.61 ms) on the
    /// worker.
    pub fn start(&mut self, id: &SlotId, generation: u64, snapshot: SaveSnapshot) -> Tick {
        if self.in_flight {
            return Tick::Idle;
        }
        self.pending_gen = generation;
        let sent = self.writer.get_or_insert_with(Self::spawn_writer).tx.send((id.clone(), snapshot));
        match sent {
            Ok(()) => {
                self.in_flight = true;
                Tick::Started
            }
            Err(_) => {
                // Drop the dead sender so a later due tick creates a fresh worker.
                self.writer = None;
                Tick::Finished(Err(writer_died()))
            }
        }
    }

    /// Exit flush: drain pending writes then save unconditionally, on this
    /// thread — a clean exit never spawns the writer just to say goodbye.
    /// Edit generation doesn't track position/mods, so exit captures complete state.
    pub fn flush_now(
        &mut self,
        id: &SlotId,
        generation: u64,
        encode: impl FnOnce() -> Result<Vec<u8>, SaveError>,
    ) -> Result<(), SaveError> {
        if self.in_flight {
            // The drained result doesn't matter — we overwrite right below.
            // A dead worker isn't fatal either: the synchronous write is the
            // one that has to land.
            if self.writer().rx.recv().is_err() {
                self.writer = None;
            }
            self.in_flight = false;
        }
        store::write(id, &encode()?)?;
        self.saved_gen = generation;
        Ok(())
    }
}

impl Default for Autosaver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::format::PlayerState;
    use super::super::slot::SaveMeta;
    use std::fs;

    fn empty_snap() -> SaveSnapshot {
        SaveSnapshot::empty(
            SaveMeta {
                name: "auto".to_string(),
                seed: 1,
                created: 0,
                last_played: 0,
                playtime_secs: 0,
                edit_count: 0,
            },
            PlayerState {
                pos: [0.0, 40.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                flying: false,
                noclip: false,
            },
        )
    }

    fn wait_finished(auto: &mut Autosaver) {
        loop {
            match auto.poll() {
                Tick::Finished(result) => {
                    result.unwrap();
                    break;
                }
                _ => thread::yield_now(),
            }
        }
    }

    #[test]
    fn clean_worlds_never_spawn_a_writer_thread() {
        let id = SlotId::new("__autosave_lazy__").unwrap();
        let mut auto = Autosaver::new();
        assert!(auto.writer.is_none(), "construction spawns nothing");
        auto.reset(7);
        assert!(!auto.wants_write(7));
        assert!(matches!(auto.poll(), Tick::Idle));
        assert!(auto.writer.is_none(), "polling a clean world spawns nothing");
        // A synchronous exit save also needs no background worker.
        auto.flush_now(&id, 7, || empty_snap().encode()).unwrap();
        assert!(auto.writer.is_none());
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn dirty_state_writes_in_the_background() {
        let id = SlotId::new("__autosave_bg__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new();
        assert!(!auto.wants_write(0), "gen 0 is clean");
        assert!(auto.wants_write(1), "gen 1 is dirty");
        assert!(matches!(auto.start(&id, 1, empty_snap()), Tick::Started));
        assert!(!auto.wants_write(1), "no second write while one is in flight");
        wait_finished(&mut auto);
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(!auto.wants_write(1), "gen 1 now saved");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn flush_drains_in_flight_then_writes_synchronously() {
        let id = SlotId::new("__autosave_flush__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new();
        assert!(matches!(auto.start(&id, 1, empty_snap()), Tick::Started));
        auto.flush_now(&id, 2, || empty_snap().encode()).unwrap();
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(!auto.wants_write(2), "gen 2 saved by flush");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn start_refuses_to_queue_a_second_write_while_one_is_in_flight() {
        let id = SlotId::new("__autosave_no_interleave__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
        let mut auto = Autosaver::new();
        assert!(matches!(auto.start(&id, 1, empty_snap()), Tick::Started));
        assert!(matches!(auto.start(&id, 2, empty_snap()), Tick::Idle));
        wait_finished(&mut auto);
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    fn surface(world: &crate::world::World, x: i32, z: i32) -> i32 {
        (0..64).rev().find(|&y| world.is_solid(x, y, z)).expect("origin column has a surface")
    }

    #[test]
    fn in_flight_snapshot_does_not_see_edits_that_land_after_start() {
        use crate::block::AIR;
        use crate::mods::Mods;
        use crate::player::Player;
        use crate::save::{self, Source};
        use voxel_engine::DVec3;

        let id = SlotId::new("__autosave_race__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut world = crate::world::World::new(11);
        let y0 = surface(&world, 8, 8);
        let y1 = surface(&world, 9, 8);
        world.set_block(8, y0, 8, AIR);
        let gen1 = world.edit_generation();
        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mut mods = Mods::with_defaults();
        let meta = SaveMeta {
            name: "race".to_string(),
            seed: 0,
            created: 1,
            last_played: 0,
            playtime_secs: 0,
            edit_count: 0,
        };

        let mut auto = Autosaver::new();
        let snap = save::snapshot(&world, &player, &mods, meta.clone());
        assert!(matches!(auto.start(&id, gen1, snap), Tick::Started));

        // Edit while the first snapshot is in flight — must not land in it.
        world.set_block(9, y1, 8, AIR);
        let gen2 = world.edit_generation();
        assert_ne!(gen2, gen1);
        assert!(!auto.wants_write(gen2), "in-flight write is the artifact mutex");
        assert!(matches!(
            auto.start(&id, gen2, save::snapshot(&world, &player, &mods, meta.clone())),
            Tick::Idle
        ));

        wait_finished(&mut auto);
        let (loaded, _, _, report) = save::load(&id, &mut mods, crate::world::World::new).unwrap();
        assert_eq!(report.source, Source::Live);
        assert_eq!(loaded.block_at(8, y0, 8), AIR, "first edit is in this snapshot");
        assert_ne!(
            loaded.block_at(9, y1, 8),
            AIR,
            "edit after start must wait for the next snapshot"
        );

        assert!(auto.wants_write(gen2), "new edit dirties the next snapshot");
        assert!(matches!(
            auto.start(&id, gen2, save::snapshot(&world, &player, &mods, meta)),
            Tick::Started
        ));
        wait_finished(&mut auto);
        let mut mods = Mods::with_defaults();
        let (loaded, _, _, _) = save::load(&id, &mut mods, crate::world::World::new).unwrap();
        assert_eq!(loaded.block_at(9, y1, 8), AIR, "second edit lands in the next snapshot");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    /// Main-thread snapshot vs writer-thread encode+write on a 100k-edit overlay.
    /// Ignored: a timing probe, not a correctness gate. Run with
    /// `cargo test -j 4 --release --lib autosave_snapshot_and_encode_at_100k_edits -- --ignored --nocapture`.
    /// 2026-09-09 (this box, median of 5): snapshot 1.12 ms, encode 7.07 ms,
    /// write 2.54 ms, encode+write 9.61 ms.
    #[test]
    #[ignore]
    fn autosave_snapshot_and_encode_at_100k_edits() {
        use crate::block::BlockId;
        use crate::mods::Mods;
        use crate::player::Player;
        use crate::render_config::RenderConfig;
        use std::time::Instant;
        use voxel_engine::DVec3;

        const N: usize = 100_000;
        const LOOPS: usize = 5;
        let mut world = crate::world::World::with_config_lazy(7, RenderConfig::default());
        assert!(world.registry().block_count() > 1);
        world.test_fill_overlay(N, BlockId(1));
        assert_eq!(world.edits().count(), N);
        let player = Player::new(DVec3::new(0.0, 40.0, 0.0));
        let mods = Mods::with_defaults();
        let meta = SaveMeta {
            name: "bench".to_string(),
            seed: 0,
            created: 1,
            last_played: 0,
            playtime_secs: 0,
            edit_count: 0,
        };
        let id = SlotId::new("__autosave_100k__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut snap_ns = Vec::with_capacity(LOOPS);
        let mut encode_ns = Vec::with_capacity(LOOPS);
        let mut write_ns = Vec::with_capacity(LOOPS);
        for _ in 0..LOOPS {
            let t0 = Instant::now();
            let snap = crate::save::snapshot(&world, &player, &mods, meta.clone());
            snap_ns.push(t0.elapsed().as_nanos());
            let t1 = Instant::now();
            let bytes = snap.encode().unwrap();
            encode_ns.push(t1.elapsed().as_nanos());
            let t2 = Instant::now();
            crate::save::store::write(&id, &bytes).unwrap();
            write_ns.push(t2.elapsed().as_nanos());
        }
        snap_ns.sort_unstable();
        encode_ns.sort_unstable();
        write_ns.sort_unstable();
        let med = |v: &[u128]| v[v.len() / 2];
        let snap_us = med(&snap_ns) as f64 / 1_000.0;
        let encode_us = med(&encode_ns) as f64 / 1_000.0;
        let write_us = med(&write_ns) as f64 / 1_000.0;
        eprintln!(
            "autosave_100k median of {LOOPS}: snapshot={snap_us:.1}µs encode={encode_us:.1}µs write={write_us:.1}µs encode+write={:.1}µs",
            encode_us + write_us
        );

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }
}
