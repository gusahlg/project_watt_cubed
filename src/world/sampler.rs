//! A thread that answers batches of a pure function of keys, for state the main thread would
//! otherwise compute in the frame it first needs it. Its answers equal the main thread's own, so
//! a caller keeps whichever comes first and never waits: an answer that has not landed is computed
//! where it is needed, as before.

use std::hash::Hash;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use super::{FastMap, FastSet};

/// Entries a [`Recent`] map holds before it lets go of those unused since it last did.
pub(in crate::world) const MEMO_CAP: usize = 1 << 16;

/// A memo that keeps what was used lately. Past [`MEMO_CAP`] entries it starts afresh, keeping the
/// previous entries aside until it is full again: one read brings an entry back, and only what
/// went unread for that long is dropped, so a long flight never measures its whole view again.
pub(in crate::world) struct Recent<K, V> {
    now: FastMap<K, V>,
    old: FastMap<K, V>,
}

impl<K, V> Default for Recent<K, V> {
    fn default() -> Self {
        Recent { now: FastMap::default(), old: FastMap::default() }
    }
}

impl<K: Copy + Eq + Hash, V: Copy> Recent<K, V> {
    pub(in crate::world) fn get(&mut self, k: &K) -> Option<V> {
        if let Some(&v) = self.now.get(k) {
            return Some(v);
        }
        let v = self.old.remove(k)?;
        self.insert(*k, v);
        Some(v)
    }

    pub(in crate::world) fn contains(&self, k: &K) -> bool {
        self.now.contains_key(k) || self.old.contains_key(k)
    }

    pub(in crate::world) fn insert(&mut self, k: K, v: V) {
        if self.now.len() >= MEMO_CAP {
            self.old = std::mem::take(&mut self.now);
        }
        self.now.insert(k, v);
    }

    pub(in crate::world) fn remove(&mut self, k: &K) {
        self.now.remove(k);
        self.old.remove(k);
    }

    pub(in crate::world) fn clear(&mut self) {
        self.now.clear();
        self.old.clear();
    }

    #[cfg(test)]
    pub(in crate::world) fn len(&self) -> usize {
        self.now.len() + self.old.len()
    }
}

/// The thread's two ends, the keys asked for whose answers have not landed, and answers received
/// but not yet handed over.
pub(in crate::world) struct Sampler<K, V> {
    ask: Sender<Vec<K>>,
    got: Receiver<Vec<(K, V)>>,
    asked: FastSet<K>,
    held: Vec<(K, V)>,
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
        Some(Sampler { ask, got, asked: FastSet::default(), held: Vec::new() })
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

    /// Hand up to `most` of the answers that have landed to `keep`, oldest first; the rest wait for
    /// the next call.
    pub(in crate::world) fn land(&mut self, most: usize, mut keep: impl FnMut(K, V)) {
        let mut left = most;
        loop {
            while left > 0
                && let Some((k, v)) = self.held.pop()
            {
                self.asked.remove(&k);
                keep(k, v);
                left -= 1;
            }
            if left == 0 {
                return;
            }
            match self.got.try_recv() {
                Ok(mut batch) => {
                    batch.reverse();
                    self.held = batch;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.asked.clear();
                    return;
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
            s.land(7, |k, v| seen.push((k, v)));
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..100).map(|k| (k, k * 3)).collect::<Vec<_>>());
        assert!(!s.asked(&5));
    }

    /// Past its cap the memo keeps what was read since it last let go, and drops what was not.
    #[test]
    fn recent_keeps_what_was_read_lately() {
        let cap = MEMO_CAP as i32;
        let mut m = Recent::<i32, i32>::default();
        for x in 0..=cap {
            m.insert(x, x);
        }
        assert_eq!(m.get(&5), Some(5));
        for x in cap + 1..2 * cap {
            m.insert(x, x);
        }
        assert!(m.contains(&5), "an entry read since the memo last let go was dropped");
        assert!(!m.contains(&6), "an entry nobody read survived two fills");
        for x in 2 * cap..3 * cap {
            m.insert(x, x);
        }
        assert!(!m.contains(&5));
        assert!(m.len() <= 2 * MEMO_CAP);
        m.remove(&(3 * cap - 1));
        assert!(!m.contains(&(3 * cap - 1)));
    }
}
