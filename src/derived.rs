//! The derived-view spine: a small foundation for values that are *derived*
//! from some mutable source and cached against that source's revision.
//!
//! Two types live here:
//!
//! - [`Revision`] — an opaque version stamp. Two views agree exactly when their
//!   revisions are equal; there's no ordering or increment operation, just
//!   equality checks.
//! - [`Derived<T>`] — a `T` cached against the [`Revision`] of its source and
//!   rebuilt *only* at an explicit sync point (never lazily on read). It hands
//!   out cheap `Arc<T>` snapshots so consumers — including worker threads — can
//!   hold an immutable view while a later rebuild makes a *new* `Arc` and leaves
//!   the old one valid.
//!
//! `World.tables` is a [`Derived<HotTables>`] stamped with the block registry's
//! revision: it is read while meshing/streaming (`src/world/mod.rs`) and rebuilt
//! at the `refresh_tables` sync point (`src/world/streaming.rs`) whenever the
//! registry grows.

use std::sync::Arc;

/// An opaque version stamp for a derived source.
///
/// A [`Derived`] view is up to date when its stamp equals the source
/// stamp. The implementation is free to use any stable identifier—the API
/// only cares about equality.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Revision(pub u64);

impl Revision {
    /// A stamp from an append-only count (e.g. registry `block_count()`).
    pub fn from_count(n: usize) -> Self {
        Revision(n as u64)
    }
}

/// A value cached against the [`Revision`] of its source.
///
/// The cached value is rebuilt only when the caller calls [`sync`](Self::sync)
/// and the source revision has changed — i.e. on the owning thread's `&mut`
/// boundary. There's no interior mutability and no lazy rebuild on read, so a
/// reader (including a worker thread holding a snapshot) can never trigger one.
///
/// Every accessor yields an `Arc<T>` via a cheap `Arc::clone`. A rebuild
/// installs a *new* `Arc`, so any snapshot taken earlier keeps observing the
/// value it was built from.
pub struct Derived<T> {
    rev: Revision,
    value: Arc<T>,
}

impl<T> Derived<T> {
    /// Create a derived view holding `value`, stamped with `rev`.
    pub fn new(rev: Revision, value: T) -> Self {
        Self {
            rev,
            value: Arc::new(value),
        }
    }

    /// Rebuild the cached value if the source revision has changed, then return
    /// a cheap `Arc` snapshot.
    ///
    /// `build` is invoked only on a revision mismatch; if the revision is
    /// unchanged the current value is handed back untouched.
    pub fn sync(&mut self, src: Revision, build: impl FnOnce() -> T) -> Arc<T> {
        if self.rev != src {
            self.value = Arc::new(build());
            self.rev = src;
        }
        Arc::clone(&self.value)
    }

    /// A cheap `Arc` snapshot of the current cached value, without any rebuild.
    pub fn get(&self) -> Arc<T> {
        Arc::clone(&self.value)
    }

}

impl<T: Default> Default for Derived<T> {
    fn default() -> Self {
        Self::new(Revision::default(), T::default())
    }
}

/// A value cached against an arbitrary equality key — the single-threaded,
/// per-frame sibling of [`Derived`]: no revision protocol, no `Arc`, one slot.
/// The caller keys it by a cheap projection of the inputs (bit patterns for
/// floats) and rebuilds in place only when the key changes; [`invalidate`]
/// empties the slot when an input outside the key changes (a settings edit).
///
/// [`invalidate`]: Memo::invalidate
#[derive(Default)]
pub struct Memo<K, V> {
    slot: Option<(K, V)>,
}

impl<K: PartialEq, V> Memo<K, V> {
    pub const fn new() -> Self {
        Self { slot: None }
    }

    /// The cached value for `key`, building it only on a key change.
    pub fn get_or(&mut self, key: K, build: impl FnOnce() -> V) -> &V {
        let stale = !matches!(&self.slot, Some((k, _)) if *k == key);
        if stale {
            self.slot = Some((key, build()));
        }
        &self.slot.as_ref().expect("slot filled above").1
    }

    /// The cached value, whatever key it was built for; `None` when empty.
    pub fn get(&self) -> Option<&V> {
        self.slot.as_ref().map(|(_, v)| v)
    }

    /// Empty the slot: the next [`get_or`](Self::get_or) rebuilds.
    pub fn invalidate(&mut self) {
        self.slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_rebuilds_only_on_key_change_or_invalidate() {
        let mut memo: Memo<u32, u32> = Memo::new();
        let mut builds = 0;
        assert_eq!(*memo.get_or(1, || { builds += 1; 10 }), 10);
        assert_eq!(*memo.get_or(1, || { builds += 1; 99 }), 10);
        assert_eq!(builds, 1);
        assert_eq!(*memo.get_or(2, || { builds += 1; 20 }), 20);
        assert_eq!(builds, 2);
        memo.invalidate();
        assert_eq!(*memo.get_or(2, || { builds += 1; 21 }), 21);
        assert_eq!(builds, 3);
    }

    #[test]
    fn sync_does_not_rebuild_when_revision_unchanged() {
        let mut d = Derived::new(Revision::from_count(1), 10u32);
        // A build closure that would panic proves it is never called.
        let snap = d.sync(Revision::from_count(1), || panic!("must not rebuild"));
        assert_eq!(*snap, 10);
    }

    #[test]
    fn sync_rebuilds_when_revision_changes() {
        let mut calls = 0u32;
        let mut d = Derived::new(Revision::from_count(1), 10u32);

        let snap = d.sync(Revision::from_count(2), || {
            calls += 1;
            42
        });
        assert_eq!(*snap, 42);
        assert_eq!(calls, 1);

        // Syncing again at the same revision must not rebuild.
        let snap2 = d.sync(Revision::from_count(2), || {
            calls += 1;
            99
        });
        assert_eq!(*snap2, 42);
        assert_eq!(calls, 1);
    }

    #[test]
    fn old_arc_snapshot_stays_valid_after_rebuild() {
        let mut d = Derived::new(Revision::from_count(1), 10u32);

        // Snapshot taken before the rebuild.
        let old = d.get();
        assert_eq!(*old, 10);

        // Rebuild installs a brand-new Arc with a new value.
        let new = d.sync(Revision::from_count(2), || 99);
        assert_eq!(*new, 99);

        // The old snapshot still observes the old value.
        assert_eq!(*old, 10);
        assert!(!Arc::ptr_eq(&old, &new));
    }
}
