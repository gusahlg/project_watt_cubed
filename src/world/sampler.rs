//! A thread that answers batches of a pure function of keys, for state the main thread would
//! otherwise compute in the frame it first needs it. Its answers equal the main thread's own, so
//! a caller keeps whichever comes first and never waits: an answer that has not landed is computed
//! where it is needed, as before.

use std::hash::Hash;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use super::FastSet;

/// The thread's two ends and the keys asked for whose answers have not landed.
pub(in crate::world) struct Sampler<K, V> {
    ask: Sender<Vec<K>>,
    got: Receiver<Vec<(K, V)>>,
    asked: FastSet<K>,
}

impl<K: Copy + Eq + Hash + Send + 'static, V: Send + 'static> Sampler<K, V> {
    /// A thread called `name` answering `f`; it ends when this is dropped. `None` if no thread
    /// starts.
    pub(in crate::world) fn start(name: &str, f: impl Fn(K) -> V + Send + 'static) -> Option<Self> {
        let (ask, todo) = channel::<Vec<K>>();
        let (done, got) = channel::<Vec<(K, V)>>();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                while let Ok(batch) = todo.recv() {
                    if done.send(batch.into_iter().map(|k| (k, f(k))).collect()).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Sampler { ask, got, asked: FastSet::default() })
    }

    /// Keys asked for that have not landed.
    pub(in crate::world) fn waiting(&self) -> usize {
        self.asked.len()
    }

    /// Whether `k` was asked for and has not landed.
    #[cfg(test)]
    pub(in crate::world) fn asked(&self, k: &K) -> bool {
        self.asked.contains(k)
    }

    /// Mark `k` asked for; `false` if it already was. Asked keys go out with [`send`](Self::send).
    pub(in crate::world) fn mark(&mut self, k: K) -> bool {
        self.asked.insert(k)
    }

    /// Send `batch` (keys already [marked](Self::mark)), nearest need first.
    pub(in crate::world) fn send(&mut self, batch: Vec<K>) {
        if !batch.is_empty() && self.ask.send(batch).is_err() {
            self.asked.clear();
        }
    }

    /// Hand every answer that has landed to `keep`.
    pub(in crate::world) fn land(&mut self, mut keep: impl FnMut(K, V)) {
        loop {
            match self.got.try_recv() {
                Ok(batch) => {
                    for (k, v) in batch {
                        self.asked.remove(&k);
                        keep(k, v);
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.asked.clear();
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key asked for lands once with the function's answer, and nothing is left waiting.
    #[test]
    fn answers_land_once_each() {
        let mut s = Sampler::start("test-sampler", |k: u32| k * 3).expect("a thread");
        let batch: Vec<u32> = (0..100).filter(|&k| s.mark(k)).collect();
        assert!(!s.mark(5), "a key was asked for twice");
        s.send(batch);
        let mut seen = Vec::new();
        let start = std::time::Instant::now();
        while s.waiting() > 0 {
            assert!(start.elapsed().as_secs() < 30, "the sampler never answered");
            std::thread::yield_now();
            s.land(|k, v| seen.push((k, v)));
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..100).map(|k| (k, k * 3)).collect::<Vec<_>>());
        assert!(!s.asked(&5));
    }
}
