//! Periodic autosave with dirty tracking.
//!
//! Serialization on caller's thread keeps I/O off render; `in_flight` ensures
//! at most one pending write to prevent races.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use super::slot::{SaveError, SlotId};
use super::store;

pub struct Autosaver {
    interval: Duration,
    /// Last generation successfully written.
    saved_gen: u64,
    /// Generation handed to the writer thread, promoted on success.
    pending_gen: u64,
    in_flight: bool,
    last_attempt: Instant,
    /// Spawned only for an actually-due periodic write. Clean worlds, disabled
    /// autosave, and synchronous exit-only saves create no background thread.
    writer: Option<Writer>,
}

struct Writer {
    tx: mpsc::Sender<(SlotId, Vec<u8>)>,
    rx: mpsc::Receiver<std::io::Result<()>>,
}

/// What one `tick` did, so the caller can surface "Saving…" / failures.
#[derive(Debug)]
pub enum Tick {
    Idle,
    Started,
    Finished(Result<(), SaveError>),
}

impl Autosaver {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            saved_gen: 0,
            pending_gen: 0,
            in_flight: false,
            last_attempt: Instant::now(),
            writer: None,
        }
    }

    fn spawn_writer() -> Writer {
        let (tx, job_rx) = mpsc::channel::<(SlotId, Vec<u8>)>();
        let (done_tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("autosave".to_string())
            .spawn(move || {
                // Exits when the Autosaver (and thus `tx`) is dropped.
                while let Ok((id, bytes)) = job_rx.recv() {
                    let _ = done_tx.send(store::write(&id, &bytes));
                }
            })
            .expect("spawn autosave thread");
        Writer { tx, rx }
    }

    /// Note a freshly loaded/created world so its current state doesn't count
    /// as dirty.
    pub fn reset(&mut self, generation: u64) {
        self.saved_gen = generation;
        self.last_attempt = Instant::now();
    }

    /// Call once per frame. `encode` runs only when a write is actually due.
    pub fn tick(
        &mut self,
        id: &SlotId,
        generation: u64,
        encode: impl FnOnce() -> Result<Vec<u8>, SaveError>,
    ) -> Tick {
        if self.in_flight {
            let writer = self
                .writer
                .as_ref()
                .expect("in-flight autosave must have a writer");
            match writer.rx.try_recv() {
                Ok(result) => {
                    self.in_flight = false;
                    if result.is_ok() {
                        self.saved_gen = self.pending_gen;
                    }
                    return Tick::Finished(result.map_err(SaveError::from));
                }
                Err(mpsc::TryRecvError::Empty) => return Tick::Idle,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.in_flight = false;
                    self.writer = None;
                    return Tick::Finished(Err(SaveError::Io(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "autosave thread died",
                    ))));
                }
            }
        }
        if generation == self.saved_gen || self.last_attempt.elapsed() < self.interval {
            return Tick::Idle;
        }
        self.last_attempt = Instant::now();
        let bytes = match encode() {
            Ok(bytes) => bytes,
            Err(e) => return Tick::Finished(Err(e)),
        };
        self.pending_gen = generation;
        let sent = self
            .writer
            .get_or_insert_with(Self::spawn_writer)
            .tx
            .send((id.clone(), bytes));
        match sent {
            Ok(()) => {
                self.in_flight = true;
                Tick::Started
            }
            Err(_) => {
                // Drop the dead sender so a later due tick can create a fresh worker.
                self.writer = None;
                Tick::Finished(Err(SaveError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "autosave thread died",
                ))))
            }
        }
    }

    /// Exit flush: drain pending writes then save unconditionally.
    /// Edit generation doesn't track position/mods, so exit captures complete state.
    pub fn flush_now(
        &mut self,
        id: &SlotId,
        generation: u64,
        encode: impl FnOnce() -> Result<Vec<u8>, SaveError>,
    ) -> Result<(), SaveError> {
        if self.in_flight {
            // The drained result doesn't matter — we overwrite right below.
            let writer = self
                .writer
                .as_ref()
                .expect("in-flight autosave must have a writer");
            if writer.rx.recv().is_err() {
                // Exit's synchronous snapshot is authoritative and does not
                // depend on the dead worker. Retire it, then continue instead
                // of turning a background failure into lost exit state.
                self.writer = None;
            }
            self.in_flight = false;
        }
        store::write(id, &encode()?)?;
        self.saved_gen = generation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn bytes() -> Result<Vec<u8>, SaveError> {
        use super::super::format::{PlayerState, SaveDoc, encode};
        use super::super::slot::SaveMeta;
        encode(&SaveDoc {
            worldgen_version: 2,
            meta: SaveMeta {
                name: "auto".to_string(),
                seed: 1,
                created: 0,
                last_played: 0,
                playtime_secs: 0,
                edit_count: 0,
            },
            player: PlayerState {
                pos: [0.0, 40.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                flying: false,
                noclip: false,
            },
            specs: vec![],
            edits: vec![],
            mods: vec![],
        })
    }

    #[test]
    fn dirty_state_writes_in_the_background() {
        let id = SlotId::new("__autosave_bg__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new(Duration::ZERO);
        assert!(matches!(auto.tick(&id, 0, bytes), Tick::Idle), "gen 0 is clean");
        assert!(matches!(auto.tick(&id, 1, bytes), Tick::Started));
        loop {
            match auto.tick(&id, 1, bytes) {
                Tick::Finished(result) => {
                    result.unwrap();
                    break;
                }
                _ => thread::yield_now(),
            }
        }
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(matches!(auto.tick(&id, 1, bytes), Tick::Idle), "gen 1 now saved");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn flush_drains_in_flight_then_writes_synchronously() {
        let id = SlotId::new("__autosave_flush__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new(Duration::ZERO);
        assert!(matches!(auto.tick(&id, 1, bytes), Tick::Started));
        auto.flush_now(&id, 2, bytes).unwrap();
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(matches!(auto.tick(&id, 2, bytes), Tick::Idle), "gen 2 saved by flush");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn clean_reset_and_fresh_flush_do_not_spawn_a_worker() {
        let id = SlotId::new("__autosave_lazy__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new(Duration::ZERO);
        auto.reset(7);
        assert!(matches!(auto.tick(&id, 7, || panic!("clean state encoded")), Tick::Idle));
        assert!(auto.writer.is_none());

        auto.flush_now(&id, 7, bytes).unwrap();
        assert!(auto.writer.is_none(), "synchronous exit save needs no worker");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }

    #[test]
    fn disconnected_completion_channel_releases_and_retires_writer() {
        let id = SlotId::new("__autosave_dead__").unwrap();
        let mut auto = Autosaver::new(Duration::ZERO);
        let (tx, _jobs) = mpsc::channel();
        let (done_tx, rx) = mpsc::channel();
        drop(done_tx);
        auto.writer = Some(Writer { tx, rx });
        auto.in_flight = true;

        let Tick::Finished(Err(SaveError::Io(error))) = auto.tick(&id, 1, bytes) else {
            panic!("disconnected writer must surface an I/O failure");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
        assert!(!auto.in_flight);
        assert!(auto.writer.is_none());
    }

    #[test]
    fn disconnected_writer_cannot_skip_the_exit_flush() {
        let id = SlotId::new("__autosave_dead_flush__").unwrap();
        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));

        let mut auto = Autosaver::new(Duration::ZERO);
        let (tx, _jobs) = mpsc::channel();
        let (done_tx, rx) = mpsc::channel();
        drop(done_tx);
        auto.writer = Some(Writer { tx, rx });
        auto.in_flight = true;

        auto.flush_now(&id, 9, bytes).unwrap();
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(!auto.in_flight);
        assert!(auto.writer.is_none());
        assert_eq!(auto.saved_gen, 9);

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }
}
