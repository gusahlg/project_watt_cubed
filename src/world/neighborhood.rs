//! The one padded halo (18³): a chunk's 16³ cells plus a one-cell shell
//! copied from its 26 neighbours, indexed by signed coords `x, y, z ∈
//! -1..=16`. Mesh ([`Padded`](super::mesh::Padded), over `BlockId`) and light
//! ([`PaddedLight`](super::light::PaddedLight), over `Lumel`) are
//! instantiations of this one capture/index/pool machinery, not separate
//! implementations — they differ only in cell type and how a local
//! cell is read out of their respective source grid.
use std::sync::Mutex;

use super::chunk::CHUNK_SIZE;

/// Chunk size as a signed coordinate, for the `-1..=16` padded range.
const CS: i32 = CHUNK_SIZE as i32;
/// Padded neighbourhood edge: the 16 chunk cells plus one shell cell each side.
const PAD: usize = CHUNK_SIZE + 2;
/// Cells in one [`Neighborhood`] buffer.
const PAD_VOL: usize = PAD * PAD * PAD;
/// Max buffers held per pool. Snapshots are captured on the main thread and
/// dropped by workers, so a thread-local pool would strand every returned
/// buffer on the wrong thread (the capturer's list stays empty and allocates
/// forever). One bounded SHARED pool completes the ownership round trip;
/// enough slack for a traversal burst across a 12-worker pool
/// (64 × 18³ × 2 B ≈ 0.7 MiB per type).
const POOL_CAP: usize = 64;

/// A bounded cross-thread free list. `take` hands back a retired value or
/// `None` (the caller allocates fresh — a miss never blocks); `put` keeps a
/// value only while under `cap`, so the cap bounds RETAINED capacity, nothing
/// else. Poison is swallowed: a sibling panic leaves the list usable. One
/// invariant shared by the halo pools here and the mesh-output pool
/// ([`pipeline::MeshOutput`](super::pipeline)).
pub struct BoundedPool<T> {
    free: Mutex<Vec<T>>,
    cap: usize,
}

impl<T> BoundedPool<T> {
    pub(in crate::world) const fn new(cap: usize) -> Self {
        Self { free: Mutex::new(Vec::new()), cap }
    }
    pub(in crate::world) fn take(&self) -> Option<T> {
        self.free.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop()
    }
    pub(in crate::world) fn put(&self, item: T) {
        let mut free = self.free.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if free.len() < self.cap {
            free.push(item);
        }
    }
}

/// A cell type that owns a shared free list of halo buffers. A `static`
/// cannot name a generic `T`, so the per-type pool lives behind this trait —
/// one concrete monomorphic [`BoundedPool`] per implementor, keeping BlockId
/// and Lumel free lists isolated. Written by `pooled_cell!`.
pub trait Pooled: Copy + Send + 'static {
    fn pool() -> &'static BoundedPool<Box<[Self]>>;
}

/// One `impl Pooled` = one cross-thread buffer free list for `$t`.
macro_rules! pooled_cell {
    ($t:ty) => {
        impl $crate::world::neighborhood::Pooled for $t {
            fn pool() -> &'static $crate::world::neighborhood::BoundedPool<Box<[$t]>> {
                static POOL: $crate::world::neighborhood::BoundedPool<Box<[$t]>> =
                    $crate::world::neighborhood::BoundedPool::new(
                        $crate::world::neighborhood::POOL_CAP,
                    );
                &POOL
            }
        }
    };
}

pooled_cell!(crate::block::registry::BlockId);
pooled_cell!(super::light::Lumel);

/// A padded halo buffer over cell type `T`. Backing storage is pooled
/// per-thread, per `T` ([`Pooled`] gives each type its own free list), so
/// repeated captures of the same shape reuse an allocation instead of
/// churning one per remesh/relight job.
pub struct Neighborhood<T: Pooled> {
    buf: Box<[T]>,
}

/// Linear index of signed coord `(x, y, z)` (each `∈ -1..=16`) in the 18³
/// x-fastest halo. The one layout law: any buffer of this shape — a
/// [`Neighborhood`] or the light page it lowers to — must address cells this
/// way, so both share this formula.
#[inline]
pub(in crate::world) fn padded_index(x: i32, y: i32, z: i32) -> usize {
    (x + 1) as usize + (z + 1) as usize * PAD + (y + 1) as usize * PAD * PAD
}

impl<T: Pooled> Neighborhood<T> {
    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        padded_index(x, y, z)
    }

    /// Direct flat read at a [`padded_index`] — the mesher's stride walk,
    /// which carries indices as `base + n·s_n + u·s_u + v·s_v` adds instead
    /// of rebuilding coordinates through dynamic axis indexing per probe.
    #[inline]
    pub(in crate::world) fn at_flat(&self, i: usize) -> T {
        self.buf[i]
    }

    /// A `PAD_VOL`-cell buffer, recycled from the pool if one is available
    /// (else freshly allocated, filled with `seed`). Recycled contents are
    /// UNSPECIFIED — a recycled buffer holds a previous job's cells — so every
    /// caller must fully overwrite it before reading back.
    fn take_buf(seed: T) -> Box<[T]> {
        T::pool()
            .take()
            .filter(|b| b.len() == PAD_VOL)
            .unwrap_or_else(|| vec![seed; PAD_VOL].into_boxed_slice())
    }

    /// The cell at signed coord `(x, y, z)`, each `∈ -1..=16`.
    #[inline]
    pub fn at(&self, x: i32, y: i32, z: i32) -> T {
        self.buf[Self::index(x, y, z)]
    }

    /// A halo filled uniformly with `value` — the neutral / full-bright /
    /// all-dark constructors of the mesh and light instantiations.
    pub fn filled(value: T) -> Self {
        let mut buf = Self::take_buf(value);
        buf.fill(value);
        Self { buf }
    }

    /// A halo filled from a per-cell closure over signed coords `-1..=16` —
    /// for exercising a known field (tests).
    #[cfg(test)]
    pub fn from_fn(f: impl Fn(i32, i32, i32) -> T, fill: T) -> Self {
        let mut buf = Self::take_buf(fill);
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    buf[Self::index(x, y, z)] = f(x, y, z);
                }
            }
        }
        Self { buf }
    }

    /// Copy a chunk-shaped neighbourhood out of a source grid. `src_at(dx,
    /// dy, dz)` yields the source at chunk-offset `(dx, dy, dz)` (each `∈
    /// -1..=1`, `(0,0,0)` the chunk itself) or `None`; `extract(src, lx, ly,
    /// lz)` reads one local cell (`∈ 0..CHUNK_SIZE`) out of a present source.
    /// Resolves the 27 offsets once, then fills 18³ cells with plain array
    /// reads. A missing neighbour's cells are left at `fill`.
    pub fn capture<S: Copy>(
        fill: T,
        src_at: impl Fn(i32, i32, i32) -> Option<S>,
        extract: impl Fn(S, usize, usize, usize) -> T,
    ) -> Self {
        let neigh: [Option<S>; 27] =
            std::array::from_fn(|k| src_at(k as i32 % 3 - 1, k as i32 / 9 - 1, k as i32 / 3 % 3 - 1));
        let get = |dx: i32, dy: i32, dz: i32| neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize];
        // Split a padded coord into (chunk offset, local 0..=15).
        let split = |c: i32| -> (i32, usize) {
            if c < 0 {
                (-1, CHUNK_SIZE - 1)
            } else if c >= CS {
                (1, 0)
            } else {
                (0, c as usize)
            }
        };
        let mut buf = Self::take_buf(fill);
        // Missing neighbours must read `fill`, and only present cells are
        // written below, so a recycled buffer MUST be cleared first (else a
        // prior job's cells would leak into the unwritten shell — a silent
        // visual bug).
        buf.fill(fill);
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    let (dx, lx) = split(x);
                    let (dy, ly) = split(y);
                    let (dz, lz) = split(z);
                    if let Some(s) = get(dx, dy, dz) {
                        buf[Self::index(x, y, z)] = extract(s, lx, ly, lz);
                    }
                }
            }
        }
        Self { buf }
    }

    /// [`capture`](Self::capture), row-wise: every padded x-row's 16 interior
    /// cells come from ONE source (only `dx` varies along x), so they fill
    /// through `extract_row` — one bulk read per row — leaving just the two
    /// x-end shell cells as per-cell `extract` calls. 18² row reads + 2·18²
    /// cell reads replace 18³ dependent closure calls with three coordinate
    /// splits each; this runs on the MAIN thread inside every mesh admit, so
    /// its cost gates admission throughput directly. Cell-identical to
    /// `capture` (pinned by the law test).
    pub fn capture_rows<S: Copy>(
        fill: T,
        src_at: impl Fn(i32, i32, i32) -> Option<S>,
        extract: impl Fn(S, usize, usize, usize) -> T,
        extract_row: impl Fn(S, usize, usize, &mut [T]),
    ) -> Self {
        let neigh: [Option<S>; 27] =
            std::array::from_fn(|k| src_at(k as i32 % 3 - 1, k as i32 / 9 - 1, k as i32 / 3 % 3 - 1));
        let get = |dx: i32, dy: i32, dz: i32| neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize];
        let split = |c: i32| -> (i32, usize) {
            if c < 0 {
                (-1, CHUNK_SIZE - 1)
            } else if c >= CS {
                (1, 0)
            } else {
                (0, c as usize)
            }
        };
        let mut buf = Self::take_buf(fill);
        buf.fill(fill); // same recycled-buffer rule as `capture`
        for y in -1..=CS {
            let (dy, ly) = split(y);
            for z in -1..=CS {
                let (dz, lz) = split(z);
                let row = Self::index(-1, y, z);
                if let Some(s) = get(-1, dy, dz) {
                    buf[row] = extract(s, CHUNK_SIZE - 1, ly, lz);
                }
                if let Some(s) = get(0, dy, dz) {
                    extract_row(s, ly, lz, &mut buf[row + 1..row + 1 + CHUNK_SIZE]);
                }
                if let Some(s) = get(1, dy, dz) {
                    buf[row + 1 + CHUNK_SIZE] = extract(s, 0, ly, lz);
                }
            }
        }
        Self { buf }
    }
}

impl<T: Pooled> Drop for Neighborhood<T> {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        if buf.len() == PAD_VOL {
            T::pool().put(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pooled_cell!(i32);

    /// The pre-unification `capture` loop (identical split/index math, no
    /// pooling), inlined as the law-test oracle: [`Neighborhood::capture`]
    /// must reproduce it cell-for-cell, including missing-neighbour cells.
    fn oracle_capture<S: Copy>(
        fill: i32,
        src_at: impl Fn(i32, i32, i32) -> Option<S>,
        extract: impl Fn(S, usize, usize, usize) -> i32,
    ) -> Vec<i32> {
        let split = |c: i32| -> (i32, usize) {
            if c < 0 {
                (-1, CHUNK_SIZE - 1)
            } else if c >= CS {
                (1, 0)
            } else {
                (0, c as usize)
            }
        };
        let mut buf = vec![fill; PAD_VOL];
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    let (dx, lx) = split(x);
                    let (dy, ly) = split(y);
                    let (dz, lz) = split(z);
                    if let Some(s) = src_at(dx, dy, dz) {
                        let idx = (x + 1) as usize + (z + 1) as usize * PAD + (y + 1) as usize * PAD * PAD;
                        buf[idx] = extract(s, lx, ly, lz);
                    }
                }
            }
        }
        buf
    }

    /// `capture_rows` must be cell-identical to `capture` under the same
    /// sources — including the missing-neighbour holes — when its row
    /// extractor is the per-cell extractor unrolled. This is what lets the
    /// mesh/light captures switch to the row path with no behavioral wiggle.
    #[test]
    fn capture_rows_matches_capture_cell_for_cell() {
        let present = |dx: i32, dy: i32, dz: i32| {
            !(dx == 1 && dy == 1 && dz == -1) && !(dx == -1 && dy == 0 && dz == 0)
        };
        let src_at = |dx: i32, dy: i32, dz: i32| present(dx, dy, dz).then_some((dx, dy, dz));
        let extract = |(dx, dy, dz): (i32, i32, i32), lx: usize, ly: usize, lz: usize| {
            dx * 10_000 + dy * 1_000 + dz * 100 + lx as i32 * 256 + ly as i32 * 16 + lz as i32
        };

        let want: Neighborhood<i32> = Neighborhood::capture(-1, src_at, extract);
        let got: Neighborhood<i32> = Neighborhood::capture_rows(
            -1,
            src_at,
            extract,
            |s, ly, lz, out: &mut [i32]| {
                for (lx, o) in out.iter_mut().enumerate() {
                    *o = extract(s, lx, ly, lz);
                }
            },
        );
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    assert_eq!(got.at(x, y, z), want.at(x, y, z), "cell ({x}, {y}, {z})");
                }
            }
        }
    }

    #[test]
    fn capture_matches_the_reference_split_index_loop_including_missing_neighbours() {
        // A 3x3x3 chunk-offset grid with two holes (missing neighbours at a
        // face and at a corner/diagonal), so the halo exercises interior,
        // border, and diagonal cells alike. Each present cell's value encodes
        // its own chunk offset and local coord, so a mismatch anywhere in the
        // 18³ halo is distinguishable.
        let present =
            |dx: i32, dy: i32, dz: i32| !(dx == 1 && dy == 1 && dz == -1) && !(dx == -1 && dy == 0 && dz == 0);
        let src_at = |dx: i32, dy: i32, dz: i32| present(dx, dy, dz).then_some((dx, dy, dz));
        let extract = |(dx, dy, dz): (i32, i32, i32), lx: usize, ly: usize, lz: usize| {
            dx * 10_000 + dy * 1_000 + dz * 100 + lx as i32 * 256 + ly as i32 * 16 + lz as i32
        };

        let got = Neighborhood::capture(-1, src_at, extract);
        let want = oracle_capture(-1, src_at, extract);

        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    let idx = (x + 1) as usize + (z + 1) as usize * PAD + (y + 1) as usize * PAD * PAD;
                    assert_eq!(got.at(x, y, z), want[idx], "mismatch at ({x},{y},{z})");
                }
            }
        }
    }
}
