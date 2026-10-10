//! Edit ledgers installed in bulk: a save's edits on load, a server's overlay on join. Edits are
//! grouped by chunk; each cell's generated block comes from batch generation on every core; and a
//! chunk nothing streamed depends on takes its whole share in one pass of overlay, gravity, column
//! and section bookkeeping. The world ends exactly as `set_block` on each edit in order leaves it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::block::BlockId;
use crate::coord::{BlockCoord, Face};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::TerrainGenerator;
use super::seam::Seams;
use super::{ColumnKey, Coord, FastMap, Sky, World};

/// A chunk with this many edits generates whole: about what one chunk costs in single-cell
/// queries. Fewer query their cells.
const DENSE: usize = 24;
/// Fewer edits than this prepare on the calling thread.
const PARALLEL_MIN: usize = 64;

/// One chunk's edits in arrival order: in-chunk index and block.
type Group = (Coord, Vec<(u16, BlockId)>);

/// Each edit's generated block and the physical cell its matter sits in.
type Prepared = Vec<(BlockId, (i32, i32, i32))>;

/// Edits waiting to be installed, grouped by chunk in order of each chunk's first edit.
#[derive(Default)]
pub struct Ledger {
    groups: Vec<Group>,
    /// Each chunk's group.
    slot: FastMap<Coord, usize>,
}

impl Ledger {
    /// Queue an edit behind every edit queued before it.
    pub fn push(&mut self, (x, y, z): (i32, i32, i32), id: BlockId) {
        let (coord, local) = BlockCoord::new(x, y, z).split();
        let index = Chunk::index(local.lx(), local.ly(), local.lz()) as u16;
        let groups = &mut self.groups;
        let at = *self.slot.entry(coord).or_insert_with(|| {
            groups.push((coord, Vec::new()));
            groups.len() - 1
        });
        self.groups[at].1.push((index, id));
    }
}

impl World {
    /// Install a whole ledger now.
    pub fn install_edits(&mut self, edits: impl IntoIterator<Item = ((i32, i32, i32), BlockId)>) {
        let mut ledger = Ledger::default();
        for (cell, id) in edits {
            ledger.push(cell, id);
        }
        self.install(&mut ledger);
    }

    /// Install every edit of `ledger`, leaving it empty.
    pub fn install(&mut self, ledger: &mut Ledger) {
        if ledger.groups.is_empty() {
            return;
        }
        ledger.slot.clear();
        let groups = std::mem::take(&mut ledger.groups);
        static THREADS: OnceLock<usize> = OnceLock::new();
        let threads = *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let prepared = self.prepare(&groups, threads);
        for ((coord, cells), prepared) in groups.iter().zip(&prepared) {
            self.install_chunk(*coord, cells, prepared);
        }
    }

    /// Every edit's generated block and matter cell. Dense chunks of one column at consecutive
    /// altitudes generate in one call and share the column's profile (an open-sky chunk generates
    /// alone, as streaming does); a sparse chunk queries its cells. The work spreads over up to
    /// `threads` threads, the dense runs first.
    fn prepare(&self, groups: &[Group], threads: usize) -> Vec<Prepared> {
        let mut dense: Vec<(ColumnKey, bool, i32, usize)> = Vec::new();
        let mut sparse: Vec<usize> = Vec::new();
        for (i, (coord, cells)) in groups.iter().enumerate() {
            if cells.len() < DENSE {
                sparse.push(i);
                continue;
            }
            let (face, open) = match self.generator.sky(*coord) {
                Sky::Axis(face) => (face, false),
                Sky::Open => (Face::PosY, true),
            };
            let (key, alt) = ColumnKey::of(face, *coord);
            dense.push((key, open, alt, i));
        }
        dense.sort_unstable_by_key(|&(key, open, alt, _)| (key.face.index(), key.a, key.b, open, alt));
        // A run: its column and altitudes, whether it is open sky, and its groups by altitude.
        let mut items: Vec<(Option<(ColumnKey, i32, i32, bool)>, Vec<usize>)> = Vec::new();
        for (key, open, alt, i) in dense {
            match items.last_mut() {
                Some((Some((k, _, hi, false)), members)) if !open && *k == key && *hi + 1 == alt => {
                    *hi = alt;
                    members.push(i);
                }
                _ => items.push((Some((key, alt, alt, open)), vec![i])),
            }
        }
        items.extend(sparse.into_iter().map(|i| (None, vec![i])));

        let edits: usize = groups.iter().map(|(_, cells)| cells.len()).sum();
        let threads = if edits < PARALLEL_MIN { 1 } else { threads.min(items.len()) };
        let (generator, seams, next) = (&*self.generator, &self.seams, AtomicUsize::new(0));
        let work = || {
            let mut out = Vec::new();
            while let Some((run, members)) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
                prepare_item(generator, seams, groups, run.map(|(key, lo, hi, _)| (key, lo, hi)), members, &mut out);
            }
            out
        };
        let parts: Vec<Vec<(usize, Prepared)>> = std::thread::scope(|s| {
            let helpers: Vec<_> = (1..threads).map(|_| s.spawn(&work)).collect();
            let mut parts = vec![work()];
            let joined = helpers.into_iter().map(|h| h.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)));
            parts.extend(joined);
            parts
        });
        let mut prepared = vec![Vec::new(); groups.len()];
        for (i, cells) in parts.into_iter().flatten() {
            prepared[i] = cells;
        }
        prepared
    }

    /// Nothing streamed depends on `coord`: neither it nor a face neighbour is loaded, and its
    /// column caches no ceiling. An edit there changes only the overlay, gravity and the far field.
    fn untouched(&self, coord: Coord) -> bool {
        let ceiling = match self.generator.sky(coord) {
            Sky::Axis(face) => self.ceilings.contains_key(&ColumnKey::of(face, coord).0),
            Sky::Open => false,
        };
        !ceiling
            && !self.chunks.contains_key(&coord)
            && Face::ALL.iter().all(|&face| !self.chunks.contains_key(&self.neighbour(coord, face)))
    }

    /// One chunk's edits: in one pass when [`untouched`](Self::untouched), else edit by edit.
    fn install_chunk(&mut self, coord: Coord, cells: &[(u16, BlockId)], prepared: &Prepared) {
        if !self.untouched(coord) {
            for (&(index, id), &(generated, _)) in cells.iter().zip(prepared) {
                let (x, y, z) = cell_of(coord, index);
                self.place(x, y, z, id, Some(generated));
            }
            return;
        }
        // The same overlay operations, in the same order, as `place`: the map ends identical.
        let mut overlay = self.edits.get_mut(&coord);
        let mut stored = false;
        for (&(index, id), &(generated, at)) in cells.iter().zip(prepared) {
            let index = usize::from(index);
            let old = overlay.as_ref().and_then(|c| c.get(&index)).copied().unwrap_or(generated);
            self.gravity.record(at, self.registry.amount(id) as i32 - self.registry.amount(old) as i32);
            if id != generated {
                if overlay.is_none() {
                    overlay = Some(self.edits.entry(coord).or_default());
                }
                if let Some(c) = overlay.as_mut() {
                    c.insert(index, id);
                }
                stored = true;
            } else if let Some(c) = overlay.as_mut() {
                c.remove(&index);
                if c.is_empty() {
                    overlay = None;
                    self.edits.remove(&coord);
                }
            }
        }
        if stored {
            let span = self.edit_columns.entry((coord.x, coord.z)).or_insert([coord.y, coord.y]);
            *span = [span[0].min(coord.y), span[1].max(coord.y)];
            let sky = self.generator.sky(coord);
            self.index_edited_chunk(coord, sky);
        }
        self.edit_generation += cells.len() as u64;
        self.window.stale = true;
        if self.lod2 {
            self.mark_dirty_sections(coord, cells.iter().map(|&(index, _)| cell_of(coord, index)));
        }
    }
}

/// Prepare one run of dense chunks (`groups[members]` at altitudes `lo..=hi` of a column), or one
/// sparse chunk cell by cell.
fn prepare_item(
    generator: &dyn TerrainGenerator,
    seams: &Seams,
    groups: &[Group],
    run: Option<(ColumnKey, i32, i32)>,
    members: &[usize],
    out: &mut Vec<(usize, Prepared)>,
) {
    let column = run.map(|(key, lo, hi)| (lo, generator.generate_column(key, lo..=hi).0));
    for (k, &i) in members.iter().enumerate() {
        let (coord, cells) = &groups[i];
        let data = column.as_ref().map(|(lo, chunks)| {
            let alt = lo + k as i32;
            &chunks.iter().find(|(a, _)| *a == alt).expect("generate_column emits every requested layer").1
        });
        let prepared = cells
            .iter()
            .map(|&(index, _)| {
                let (x, y, z) = cell_of(*coord, index);
                let generated = match data {
                    Some(data) => data.get(usize::from(index)),
                    None => generator.voxel_at(x, y, z),
                };
                (generated, seams.physical_cell(BlockCoord::new(x, y, z)).unwrap_or((x, y, z)))
            })
            .collect();
        out.push((i, prepared));
    }
}

/// World cell of in-chunk `index` of `coord`.
fn cell_of(coord: Coord, index: u16) -> (i32, i32, i32) {
    let (lx, ly, lz) = Chunk::local_of(usize::from(index));
    let s = CHUNK_SIZE as i32;
    (coord.x * s + lx as i32, coord.y * s + ly as i32, coord.z * s + lz as i32)
}
