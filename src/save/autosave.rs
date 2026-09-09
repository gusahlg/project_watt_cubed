//! Periodic autosave with dirty tracking.
//!
//! Serialization on caller's thread keeps I/O off render; `in_flight` ensures
//! at most one pending write to prevent races. The *how often* — the interval
//! throttle — is not here: it rides the scheduler's frame clock as an interval
//! gate (`sched::Scheduler::register_interval`), so no wall-clock `Instant`
//! timer is hand-rolled in this lane.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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

fn writer_died() -> SaveError {
    SaveError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "autosave thread died"))
}

impl Autosaver {
    pub fn new() -> Self {
        Self { saved_gen: 0, pending_gen: 0, in_flight: false, writer: None }
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
                Tick::Finished(result.map_err(SaveError::from))
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

    /// Encode on the caller's thread and hand the bytes to the writer,
    /// creating the writer thread on the first actually-due write. The caller
    /// must have checked [`Self::wants_write`] and the scheduler's interval
    /// gate first.
    pub fn start(
        &mut self,
        id: &SlotId,
        generation: u64,
        encode: impl FnOnce() -> Result<Vec<u8>, SaveError>,
    ) -> Tick {
        let bytes = match encode() {
            Ok(bytes) => bytes,
            Err(e) => return Tick::Finished(Err(e)),
        };
        self.pending_gen = generation;
        let sent = self.writer.get_or_insert_with(Self::spawn_writer).tx.send((id.clone(), bytes));
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
    use std::fs;

    fn bytes() -> Result<Vec<u8>, SaveError> {
        use super::super::format::{PlayerState, SaveDoc, WorldgenStamp, encode};
        use super::super::slot::SaveMeta;
        encode(&SaveDoc {
            worldgen_version: 2,
            worldgen: WorldgenStamp::default(),
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
                stash: Some(vec![]),
            },
            specs: vec![],
            edits: vec![],
            mods: vec![],
        })
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
        auto.flush_now(&id, 7, bytes).unwrap();
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
        assert!(matches!(auto.start(&id, 1, bytes), Tick::Started));
        assert!(!auto.wants_write(1), "no second write while one is in flight");
        loop {
            match auto.poll() {
                Tick::Finished(result) => {
                    result.unwrap();
                    break;
                }
                _ => thread::yield_now(),
            }
        }
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
        assert!(matches!(auto.start(&id, 1, bytes), Tick::Started));
        auto.flush_now(&id, 2, bytes).unwrap();
        assert!(fs::metadata(format!("saves/{id}.save")).is_ok());
        assert!(!auto.wants_write(2), "gen 2 saved by flush");

        let _ = fs::remove_file(format!("saves/{id}.save"));
        let _ = fs::remove_file(format!("saves/{id}.save.bak"));
    }
}
