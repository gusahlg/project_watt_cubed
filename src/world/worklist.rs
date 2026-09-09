//! Ring-bucketed seed set: nearest chess-ring first, far keys unvisited.
//!
//! Admission only needs the nearest few dozen READY seeds, but a flat set
//! forced every pass to `ready()`-probe the whole worklist. Keys live in
//! buckets of [`World::order`] (chess distance, vertical ×2) around the
//! streaming centre; a pass walks nearest-first and stops once it has enough
//! ready work. Far buckets stay put — their re-seed events still fire.
//! Recentre is O(n), once per boundary cross.

use super::{Coord, FastSet, World};

/// Nearest-first worklist. Bucket `i` holds keys with `order(key, center) == i`;
/// anything past the last bucket clamps there (outside the data box).
pub struct RingWorklist {
    center: Coord,
    buckets: Vec<FastSet<Coord>>,
    len: usize,
}

impl RingWorklist {
    pub fn new(center: Coord, rings: usize) -> Self {
        let rings = rings.max(1);
        Self {
            center,
            buckets: (0..rings).map(|_| FastSet::default()).collect(),
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(FastSet::capacity).sum()
    }

    #[inline]
    pub fn contains(&self, key: &Coord) -> bool {
        self.buckets[self.index(*key)].contains(key)
    }

    /// Returns whether the key was newly inserted.
    pub fn insert(&mut self, key: Coord) -> bool {
        let i = self.index(key);
        if self.buckets[i].insert(key) {
            self.len += 1;
            true
        } else {
            false
        }
    }

    /// Returns whether the key was present.
    pub fn remove(&mut self, key: &Coord) -> bool {
        let i = self.index(*key);
        if self.buckets[i].remove(key) {
            self.len -= 1;
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.len = 0;
    }

    /// Nearest bucket first.
    pub fn iter(&self) -> impl Iterator<Item = &Coord> + '_ {
        self.buckets.iter().flat_map(|b| b.iter())
    }

    pub fn retain(&mut self, mut f: impl FnMut(&Coord) -> bool) {
        let mut n = 0;
        for b in &mut self.buckets {
            b.retain(|c| f(c));
            n += b.len();
        }
        self.len = n;
    }

    /// Re-bucket every key around `center`. Ring count is unchanged.
    pub fn recenter(&mut self, center: Coord) {
        self.fit(center, self.buckets.len());
    }

    /// Grow/shrink the ring count, re-bucketing if it moved.
    pub fn resize(&mut self, rings: usize) {
        self.fit(self.center, rings);
    }

    /// Re-bucket around `center` into `rings` buckets (clamped to at least 1).
    /// No-op when both already match, so a per-pass call is free at rest.
    pub fn fit(&mut self, center: Coord, rings: usize) {
        let rings = rings.max(1);
        if center == self.center && rings == self.buckets.len() {
            return;
        }
        let old = std::mem::replace(
            &mut self.buckets,
            (0..rings).map(|_| FastSet::default()).collect(),
        );
        self.center = center;
        self.len = 0;
        for set in old {
            for k in set {
                self.insert(k);
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
        let o = World::order(key, self.center).max(0) as usize;
        o.min(self.buckets.len() - 1)
    }
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
