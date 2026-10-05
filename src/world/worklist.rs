//! Ring-bucketed seed set: nearest streaming-order first, far keys unvisited.
//!
//! Admission only needs the nearest few dozen READY seeds, but a flat set
//! forced every pass to `ready()`-probe the whole worklist. Keys live in
//! buckets of [`World::order`] around the streaming centre (the centre's up
//! face; +Y weights that axis ×2). A pass walks nearest-first and stops once
//! it has enough ready work. Far buckets stay put — their re-seed events
//! still fire. Recentre is O(n), once per boundary cross or up-face change.

use crate::coord::Face;

use super::seam::Unfold;
use super::{Coord, FastMap, FastSet, World};

/// Nearest-first worklist. Bucket `i` holds keys with `order(key, center) == i`;
/// anything past the last bucket clamps there (outside the data box).
pub struct RingWorklist {
    center: Coord,
    /// Up face the buckets were built with. `None` is isotropic chess.
    up: Option<Face>,
    /// The chart net keys are measured in (identity off round worlds).
    fold: Unfold,
    buckets: Vec<FastSet<Coord>>,
    /// Every key and its bucket: membership is one probe, with no ring order to measure.
    ring: FastMap<Coord, u32>,
}

impl RingWorklist {
    pub fn new(center: Coord, rings: usize) -> Self {
        let rings = rings.max(1);
        Self {
            center,
            up: Some(Face::PosY),
            fold: Unfold::IDENTITY,
            buckets: (0..rings).map(|_| FastSet::default()).collect(),
            ring: FastMap::default(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(FastSet::capacity).sum::<usize>() + self.ring.capacity()
    }

    #[inline]
    pub fn contains(&self, key: &Coord) -> bool {
        self.ring.contains_key(key)
    }

    /// Returns whether the key was newly inserted.
    pub fn insert(&mut self, key: Coord) -> bool {
        let i = self.index(key);
        if self.ring.insert(key, i as u32).is_some() {
            return false;
        }
        self.buckets[i].insert(key);
        true
    }

    /// Returns whether the key was present.
    pub fn remove(&mut self, key: &Coord) -> bool {
        match self.ring.remove(key) {
            Some(i) => {
                self.buckets[i as usize].remove(key);
                true
            }
            None => false,
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.ring.clear();
    }

    /// Nearest bucket first.
    pub fn iter(&self) -> impl Iterator<Item = &Coord> + '_ {
        self.buckets.iter().flat_map(|b| b.iter())
    }

    #[cfg(test)]
    pub fn retain(&mut self, mut f: impl FnMut(&Coord) -> bool) {
        for b in &mut self.buckets {
            b.retain(|c| f(c));
        }
        let buckets = &self.buckets;
        self.ring.retain(|c, i| buckets[*i as usize].contains(c));
    }

    /// Re-bucket every key around `center`. Ring count and up face are unchanged.
    #[cfg(test)]
    pub fn recenter(&mut self, center: Coord) {
        self.fit(center, self.buckets.len(), self.up);
    }

    /// Measure keys through `fold` (the chart net around a storage centre), re-bucketing if it moved.
    pub fn set_fold(&mut self, fold: Unfold) {
        if fold != self.fold {
            self.fold = fold;
            self.rebucket(self.center, self.buckets.len(), self.up);
        }
    }

    /// Grow/shrink the ring count, re-bucketing if it moved. Up face unchanged.
    pub fn resize(&mut self, rings: usize) {
        self.fit(self.center, rings, self.up);
    }

    /// Re-bucket around `center` into `rings` buckets (clamped to at least 1)
    /// for streaming up `up`. No-op when centre, count, and up already match,
    /// so a per-pass call is free at rest.
    pub fn fit(&mut self, center: Coord, rings: usize, up: Option<Face>) {
        let rings = rings.max(1);
        if center == self.center && rings == self.buckets.len() && up == self.up {
            return;
        }
        self.rebucket(center, rings, up);
    }

    /// Buckets keep their allocations, so a warm re-bucket allocates nothing. Once the keys fall
    /// below a quarter of the map's capacity, a past flood's peak is released.
    fn rebucket(&mut self, center: Coord, rings: usize, up: Option<Face>) {
        let shrink = self.ring.len() < self.ring.capacity() / 4;
        for b in &mut self.buckets {
            b.clear();
        }
        self.buckets.resize_with(rings.max(1), FastSet::default);
        self.center = center;
        self.up = up;
        let last = self.buckets.len() - 1;
        for (&k, i) in &mut self.ring {
            *i = ring_of(&self.fold, k, center, up, last) as u32;
            self.buckets[*i as usize].insert(k);
        }
        // After the refill, so each bucket shrinks to its new keys in one move.
        if shrink {
            self.ring.shrink_to_fit();
            for b in &mut self.buckets {
                b.shrink_to_fit();
            }
        }
    }

    /// Walk buckets nearest-first. `visit` returns `false` to stop after the
    /// current bucket (farther rings are not touched). Returns whether every
    /// bucket was visited.
    pub fn walk_nearest(&self, mut visit: impl FnMut(&FastSet<Coord>) -> bool) -> bool {
        for b in &self.buckets {
            if !visit(b) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn index(&self, key: Coord) -> usize {
        ring_of(&self.fold, key, self.center, self.up, self.buckets.len() - 1)
    }
}

/// The bucket of `key`: its streaming order around `center`, clamped to `last`.
#[inline]
fn ring_of(fold: &Unfold, key: Coord, center: Coord, up: Option<Face>, last: usize) -> usize {
    (World::order(fold.fold(key), center, up).max(0) as usize).min(last)
}

impl Default for RingWorklist {
    fn default() -> Self {
        Self::new(Coord::new(0, 0, 0), 1)
    }
}

impl Extend<Coord> for RingWorklist {
    fn extend<T: IntoIterator<Item = Coord>>(&mut self, iter: T) {
        for k in iter {
            self.insert(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: i32, y: i32, z: i32) -> Coord {
        Coord::new(x, y, z)
    }

    #[test]
    fn insert_remove_contains_and_len() {
        let mut w = RingWorklist::new(c(0, 0, 0), 4);
        assert!(w.is_empty());
        assert!(w.insert(c(1, 0, 0)));
        assert!(!w.insert(c(1, 0, 0)));
        assert_eq!(w.len(), 1);
        assert!(w.contains(&c(1, 0, 0)));
        assert!(w.remove(&c(1, 0, 0)));
        assert!(!w.remove(&c(1, 0, 0)));
        assert!(w.is_empty());
    }

    #[test]
    fn iter_is_nearest_bucket_first() {
        let mut w = RingWorklist::new(c(0, 0, 0), 5);
        w.insert(c(3, 0, 0));
        w.insert(c(0, 0, 0));
        w.insert(c(1, 0, 0));
        let v: Vec<_> = w.iter().copied().collect();
        assert_eq!(v[0], c(0, 0, 0));
        assert_eq!(v[1], c(1, 0, 0));
        assert_eq!(v[2], c(3, 0, 0));
    }

    #[test]
    fn far_keys_clamp_to_the_last_bucket() {
        let mut w = RingWorklist::new(c(0, 0, 0), 3);
        w.insert(c(20, 0, 0));
        assert!(w.contains(&c(20, 0, 0)));
        assert_eq!(w.buckets[2].len(), 1);
        assert!(w.buckets[0].is_empty());
    }

    #[test]
    fn recenter_rebuckets() {
        let mut w = RingWorklist::new(c(0, 0, 0), 8);
        w.insert(c(3, 0, 0));
        w.recenter(c(3, 0, 0));
        assert!(w.contains(&c(3, 0, 0)));
        assert!(w.buckets[0].contains(&c(3, 0, 0)));
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn retain_drops_and_keeps_len() {
        let mut w = RingWorklist::new(c(0, 0, 0), 4);
        w.extend([c(0, 0, 0), c(1, 0, 0), c(2, 0, 0)]);
        w.retain(|k| k.x != 1);
        assert_eq!(w.len(), 2);
        assert!(!w.contains(&c(1, 0, 0)));
        assert!(w.contains(&c(0, 0, 0)));
    }

    #[test]
    fn up_axis_reorders_buckets() {
        let mut w = RingWorklist::new(c(0, 0, 0), 8);
        w.fit(c(0, 0, 0), 8, Some(Face::PosX));
        w.insert(c(2, 0, 0));
        w.insert(c(0, 2, 0));
        assert!(w.buckets[4].contains(&c(2, 0, 0)), "along +X is weighted ×2");
        assert!(w.buckets[2].contains(&c(0, 2, 0)), "across +X is plain chess");
    }

    /// Membership and bucket placement agree after every re-bucket: each key sits once, in the
    /// bucket of its order, and removes cleanly.
    #[test]
    fn membership_follows_every_rebucket() {
        let mut w = RingWorklist::new(c(0, 0, 0), 6);
        let keys: Vec<Coord> = (-4..=4).flat_map(|x| (-2..=2).map(move |y| c(x, y, 3 - x))).collect();
        w.extend(keys.iter().copied());
        let fits = [(c(1, 0, 0), 6, Some(Face::PosY)), (c(1, 2, -1), 9, Some(Face::NegX)), (c(0, 0, 0), 3, None)];
        for (center, rings, up) in fits {
            w.fit(center, rings, up);
            assert_eq!(w.len(), keys.len());
            assert_eq!(w.iter().count(), keys.len());
            for (i, b) in w.buckets.iter().enumerate() {
                assert!(b.iter().all(|k| w.index(*k) == i && w.contains(k)));
            }
        }
        for k in &keys {
            assert!(w.remove(k));
            assert!(!w.contains(k));
        }
        assert!(w.is_empty() && w.buckets.iter().all(FastSet::is_empty));
    }

    /// A drained flood's peak allocation goes at the next re-bucket; the keys left stay put.
    #[test]
    fn rebucket_releases_a_drained_flood() {
        let mut w = RingWorklist::new(c(0, 0, 0), 4);
        let key = |i: i32| c(i % 64, 0, i / 64);
        w.extend((0..4096).map(key));
        for i in 8..4096 {
            w.remove(&key(i));
        }
        let drained = w.capacity();
        w.recenter(c(1, 0, 0));
        assert!(w.capacity() < drained / 8, "{} of {drained} kept", w.capacity());
        assert_eq!(w.len(), 8);
        assert!((0..8).all(|i| w.contains(&key(i)) && w.buckets[w.index(key(i))].contains(&key(i))));
    }

    #[test]
    fn walk_nearest_stops_without_visiting_far_buckets() {
        let mut w = RingWorklist::new(c(0, 0, 0), 5);
        w.insert(c(0, 0, 0));
        w.insert(c(4, 0, 0));
        let mut seen = 0;
        let visited_all = w.walk_nearest(|b| {
            if b.is_empty() {
                return true;
            }
            seen += b.len();
            false
        });
        assert!(!visited_all);
        assert_eq!(seen, 1);
    }
}
