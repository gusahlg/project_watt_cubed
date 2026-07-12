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
        Self {
            interval,
            saved_gen: 0,
            pending_gen: 0,
            in_flight: false,
            last_attempt: Instant::now(),
            tx,
            rx,
        }
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
            match self.rx.try_recv() {
                Ok(result) => {
                    self.in_flight = false;
                    if result.is_ok() {
                        self.saved_gen = self.pending_gen;
                    }
                    return Tick::Finished(result.map_err(SaveError::from));
                }
                Err(_) => return Tick::Idle,
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
        self.in_flight = true;
        let _ = self.tx.send((id.clone(), bytes));
        Tick::Started
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
            let _ = self
                .rx
                .recv()
                .map_err(|_| SaveError::Io(std::io::Error::other("autosave thread died")))?;
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
}
