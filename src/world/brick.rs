//! Bricks and light: pure types + tests, nothing wired. `Brick` is the ONE
//! structure a future GPU traversal walks; a chunk (k=0), an LOD section
//! (k>0), and a half-block (k<0) are all values of this one type, not three
//! parallel representations.

use crate::ident::{BlockState, Detail};

/// A brick: 16³ cells at `level`. World extent = 16·2^level. The payload
/// packing is a type parameter: sections use the full [`BrickPayload`]
/// (default), chunks the Rle-free [`ChunkPayload`], so "an Rle chunk" is
/// unrepresentable rather than a guarded-against runtime state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Brick<P = BrickPayload> {
    pub level: Detail,
    pub rev: voxel_engine::Rev,
    pub payload: P,
}

/// Brick edge length: frozen — every payload variant and cross-chunk seam
/// logic assumes 16³.
pub const BRICK_DIM: usize = 16;
/// Cells per brick (16³).
pub const BRICK_VOLUME: usize = BRICK_DIM * BRICK_DIM * BRICK_DIM;
/// Per-brick palette cap: index fits `u8`; above this, [`BrickPayload::Dense`].
pub const PALETTE_MAX: usize = 256;

/// Cell index within a brick's flat `[BlockState; BRICK_VOLUME]` arrays: x
/// fastest, then z, then y. All payload variants and `cells()`/`from_cells()`
/// must agree on this ordering.
#[inline]
pub const fn cell_index(x: usize, y: usize, z: usize) -> usize {
    x + z * BRICK_DIM + y * BRICK_DIM * BRICK_DIM
}

/// One palette-indexed run: `count` consecutive cells (along y, within one
/// (x, z) column) at `palette_index`. Fields are private — [`Run::new`] is
/// the only constructor, so `count == 0` or `count > 16` (both outside a
/// brick column's height) are unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Run {
    palette_index: u8,
    count: u8,
}

impl Run {
    /// `count` must be in `1..=16` — a run is never empty and never taller
    /// than one brick column. `pub(crate)`: section extraction/downsample
    /// builds bricks straight from already-RLE column data, bypassing
    /// [`BrickPayload::from_cells`]'s 4096-cell detour.
    pub(crate) fn new(palette_index: u8, count: u8) -> Self {
        assert!((1..=16).contains(&count), "Run count out of a brick column's 1..=16 range: {count}");
        Run { palette_index, count }
    }

    pub fn palette_index(&self) -> u8 {
        self.palette_index
    }

    pub fn count(&self) -> u8 {
        self.count
    }
}

/// Vertical run-length columns: a flat run buffer plus one cumulative
/// end-offset per (x, z) column, columns x-major (`x + z*16`) to match the
/// section quadrant order. Private fields + [`RleColumns::from_column_runs`]
/// as the only constructor keep the canonical-form invariants (each column's
/// counts sum to exactly 16, adjacent runs differ in `palette_index`)
/// unrepresentable to violate from outside this module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RleColumns {
    runs: Box<[Run]>,
    /// `column_ends[x + z*16]` = exclusive end offset into `runs` for that
    /// column; offsets are cumulative, so the column's start is the
    /// previous column's end (0 for column 0).
    column_ends: Box<[u16; BRICK_DIM * BRICK_DIM]>,
}

impl RleColumns {
    /// Builds from one run list per column, `column_runs[x + z*16]`,
    /// x-major. Panics if a column's counts don't sum to 16 or adjacent
    /// runs share a palette index — the canonical-form invariant this type
    /// exists to enforce. `pub(crate)`: see [`Run::new`].
    pub(crate) fn from_column_runs(column_runs: [Vec<Run>; BRICK_DIM * BRICK_DIM]) -> Self {
        let mut runs = Vec::with_capacity(BRICK_VOLUME / 2);
        let mut column_ends = [0u16; BRICK_DIM * BRICK_DIM];
        for (col, col_runs) in column_runs.into_iter().enumerate() {
            let sum: u32 = col_runs.iter().map(|r| r.count as u32).sum();
            assert_eq!(sum, BRICK_DIM as u32, "column {col} runs must sum to 16 cells");
            for pair in col_runs.windows(2) {
                assert_ne!(pair[0].palette_index, pair[1].palette_index, "adjacent runs in column {col} must differ");
            }
            runs.extend(col_runs);
            column_ends[col] = runs.len() as u16;
        }
        RleColumns { runs: runs.into_boxed_slice(), column_ends: Box::new(column_ends) }
    }

    /// Runs for column `(x, z)`, in ascending-y order.
    pub fn runs_for_column(&self, x: usize, z: usize) -> &[Run] {
        let col = x + z * BRICK_DIM;
        let start = if col == 0 { 0 } else { self.column_ends[col - 1] as usize };
        let end = self.column_ends[col] as usize;
        &self.runs[start..end]
    }
}

/// Strategy hint for [`BrickPayload::from_cells`]'s Paletted/Rle choice.
/// A hint rather than a heuristic keeps the choice deterministic by
/// construction: the caller's hint is part of the input, so equal
/// `(cells, strategy)` pairs always produce equal payloads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PackStrategy {
    Paletted,
    Rle,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BrickPayload {
    Uniform(BlockState),
    /// Dense small-palette: one palette index per cell.
    Paletted {
        palette: Box<[BlockState]>,
        cells: Box<[u8]>,
    },
    /// Vertical runs, one column list per (x, z).
    Rle {
        palette: Box<[BlockState]>,
        columns: RleColumns,
    },
    /// >[`PALETTE_MAX`]-distinct-value escape hatch: one full [`BlockState`]
    /// > per cell, no palette indirection.
    Dense(Box<[BlockState]>),
    // Per-face payloads deliberately excluded: this is the one traversal
    // structure, not one of several parallel representations.
}

/// A chunk's payload — the k=0 [`Brick`]'s packing: `Uniform`/`Paletted`/
/// `Dense` only. The vertical-run [`BrickPayload::Rle`] variant is a section
/// concern, absent here so chunk code needs no dead match arm for a state it
/// can never hold — field-identical to the three shared [`BrickPayload`]
/// variants, so storage stays byte-for-byte what it always was; only the
/// unreachable fourth case is gone.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ChunkPayload {
    Uniform(BlockState),
    Paletted { palette: Box<[BlockState]>, cells: Box<[u8]> },
    Dense(Box<[BlockState]>),
}

impl BrickPayload {
    /// Canonical constructor: `cells` is exactly [`BRICK_VOLUME`] values
    /// indexed by [`cell_index`] (x-fastest, then z, then y). Picks the
    /// variant deterministically — all-equal ⇒ `Uniform`; else palette
    /// ≤[`PALETTE_MAX`] ⇒ `Paletted`/`Rle` per `strategy`; else `Dense` — so
    /// equal `(cells, strategy)` always yields a structurally equal payload.
    pub fn from_cells(cells: &[BlockState; BRICK_VOLUME], strategy: PackStrategy) -> BrickPayload {
        let first = cells[0];
        if cells.iter().all(|c| *c == first) {
            return BrickPayload::Uniform(first);
        }

        // First-seen-order dedup: deterministic given `cells`'s order, and
        // BlockState has no Hash impl, so linear probing over a ≤256-cap
        // palette is the straightforward choice (bounded to 4096×256 probes).
        let mut palette: Vec<BlockState> = Vec::new();
        let mut indices = vec![0u8; BRICK_VOLUME];
        for (i, cell) in cells.iter().enumerate() {
            let idx = match palette.iter().position(|p| p == cell) {
                Some(idx) => idx,
                None => {
                    palette.push(*cell);
                    palette.len() - 1
                }
            };
            if palette.len() > PALETTE_MAX {
                return BrickPayload::Dense(cells.to_vec().into_boxed_slice());
            }
            indices[i] = idx as u8;
        }

        match strategy {
            PackStrategy::Paletted => BrickPayload::Paletted { palette: palette.into_boxed_slice(), cells: indices.into_boxed_slice() },
            PackStrategy::Rle => {
                let mut column_runs: [Vec<Run>; BRICK_DIM * BRICK_DIM] = std::array::from_fn(|_| Vec::new());
                for z in 0..BRICK_DIM {
                    for x in 0..BRICK_DIM {
                        let col = &mut column_runs[x + z * BRICK_DIM];
                        for y in 0..BRICK_DIM {
                            let idx = indices[cell_index(x, y, z)];
                            match col.last_mut() {
                                Some(run) if run.palette_index() == idx && run.count() < 16 => {
                                    *run = Run::new(idx, run.count() + 1);
                                }
                                _ => col.push(Run::new(idx, 1)),
                            }
                        }
                    }
                }
                BrickPayload::Rle { palette: palette.into_boxed_slice(), columns: RleColumns::from_column_runs(column_runs) }
            }
        }
    }

    /// Inverse of [`Self::from_cells`]: reconstructs the [`BRICK_VOLUME`]
    /// cells in [`cell_index`] order.
    pub fn cells(&self) -> Box<[BlockState; BRICK_VOLUME]> {
        let mut out = [BlockState { id: crate::block::BlockId(0), state: 0 }; BRICK_VOLUME];
        match self {
            BrickPayload::Uniform(v) => out.fill(*v),
            BrickPayload::Paletted { palette, cells } => {
                for (i, &idx) in cells.iter().enumerate() {
                    out[i] = palette[idx as usize];
                }
            }
            BrickPayload::Rle { palette, columns } => {
                for z in 0..BRICK_DIM {
                    for x in 0..BRICK_DIM {
                        let mut y = 0usize;
                        for run in columns.runs_for_column(x, z) {
                            let v = palette[run.palette_index() as usize];
                            for _ in 0..run.count() {
                                out[cell_index(x, y, z)] = v;
                                y += 1;
                            }
                        }
                    }
                }
            }
            BrickPayload::Dense(cells) => out.copy_from_slice(cells),
        }
        Box::new(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(id: u16) -> BlockState {
        BlockState { id: crate::block::BlockId(id), state: 0 }
    }

    /// splitmix64: deterministic PRNG for seeded state-space grids, no `rand` dep.
    fn splitmix64(state: &mut u64) -> u64 {
        crate::hash::splitmix_next(state)
    }

    fn roundtrips(cells: &[BlockState; BRICK_VOLUME], strategy: PackStrategy) {
        let payload = BrickPayload::from_cells(cells, strategy);
        assert_eq!(*payload.cells(), *cells, "cells -> payload -> cells must be identity under {strategy:?}");
    }

    // Reachable failure: fails if from_cells's all-equal check is wrong (e.g.
    // compares by reference/index instead of value) and skips the Uniform path.
    #[test]
    fn roundtrip_air_uniform() {
        let cells = [bs(0); BRICK_VOLUME];
        roundtrips(&cells, PackStrategy::Paletted);
        roundtrips(&cells, PackStrategy::Rle);
        assert_eq!(BrickPayload::from_cells(&cells, PackStrategy::Rle), BrickPayload::Uniform(bs(0)));
    }

    // Reachable failure: fails if the uniform fast path only special-cases
    // id 0 (air) instead of "all cells equal, whatever the value".
    #[test]
    fn roundtrip_full_uniform_nonair() {
        let cells = [bs(42); BRICK_VOLUME];
        roundtrips(&cells, PackStrategy::Paletted);
        roundtrips(&cells, PackStrategy::Rle);
        assert_eq!(BrickPayload::from_cells(&cells, PackStrategy::Rle), BrickPayload::Uniform(bs(42)));
    }

    // Reachable failure: fails if column-run construction splits a column
    // that has no value change (off-by-one iteration bug) into >1 run.
    #[test]
    fn roundtrip_single_run_columns() {
        let mut cells = [bs(0); BRICK_VOLUME];
        for z in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let v = bs(((x + z) % 3) as u16);
                for y in 0..BRICK_DIM {
                    cells[cell_index(x, y, z)] = v;
                }
            }
        }
        roundtrips(&cells, PackStrategy::Rle);
        let BrickPayload::Rle { columns, .. } = BrickPayload::from_cells(&cells, PackStrategy::Rle) else {
            panic!("expected Rle");
        };
        for z in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                assert_eq!(columns.runs_for_column(x, z).len(), 1, "column ({x},{z}) has a constant value, must be one run");
            }
        }
    }

    // Reachable failure: fails if adjacent-run merging over-merges (misses a
    // value change) or `Run::count` can't represent sixteen length-1 runs.
    #[test]
    fn roundtrip_worst_case_alternating_16_run_columns() {
        let mut cells = [bs(0); BRICK_VOLUME];
        for z in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                for y in 0..BRICK_DIM {
                    cells[cell_index(x, y, z)] = bs((y % 2) as u16);
                }
            }
        }
        roundtrips(&cells, PackStrategy::Rle);
        let BrickPayload::Rle { columns, .. } = BrickPayload::from_cells(&cells, PackStrategy::Rle) else {
            panic!("expected Rle");
        };
        for z in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let runs = columns.runs_for_column(x, z);
                assert_eq!(runs.len(), 16, "alternating column ({x},{z}) must produce the maximal 16 runs");
                assert!(runs.iter().all(|r| r.count() == 1));
            }
        }
    }

    // Reachable failure: fails if the palette cap check uses `>=` instead of
    // `>` (would wrongly promote an exactly-256-distinct grid to Dense).
    #[test]
    fn roundtrip_exactly_256_distinct_palette_stays_paletted_or_rle() {
        let mut cells = [bs(0); BRICK_VOLUME];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = bs((i % PALETTE_MAX) as u16);
        }
        roundtrips(&cells, PackStrategy::Paletted);
        roundtrips(&cells, PackStrategy::Rle);
        assert!(!matches!(BrickPayload::from_cells(&cells, PackStrategy::Paletted), BrickPayload::Dense(_)));
        assert!(!matches!(BrickPayload::from_cells(&cells, PackStrategy::Rle), BrickPayload::Dense(_)));
    }

    // Reachable failure: fails if the constructor forgets the Dense
    // promotion at 257 distinct values (palette index would overflow u8).
    #[test]
    fn roundtrip_257_distinct_palette_promotes_to_dense() {
        let mut cells = [bs(0); BRICK_VOLUME];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = bs((i % (PALETTE_MAX + 1)) as u16);
        }
        roundtrips(&cells, PackStrategy::Paletted);
        roundtrips(&cells, PackStrategy::Rle);
        assert!(matches!(BrickPayload::from_cells(&cells, PackStrategy::Rle), BrickPayload::Dense(_)));
    }

    // Reachable failure: fails if `from_cells` is non-deterministic (e.g.
    // palette order depends on a HashMap) so equal inputs diverge.
    #[test]
    fn canonical_uniqueness_equal_cells_produce_structurally_equal_payloads() {
        let mut state = 12345u64;
        let cells: [BlockState; BRICK_VOLUME] = std::array::from_fn(|_| bs((splitmix64(&mut state) % 6) as u16));
        let a = BrickPayload::from_cells(&cells, PackStrategy::Rle);
        let b = BrickPayload::from_cells(&cells, PackStrategy::Rle);
        assert_eq!(a, b);
        let a = BrickPayload::from_cells(&cells, PackStrategy::Paletted);
        let b = BrickPayload::from_cells(&cells, PackStrategy::Paletted);
        assert_eq!(a, b);
    }

    // Reachable failure: fails if the all-equal check is fuzzy/approximate
    // and treats "one cell differs" as still uniform, losing that cell.
    #[test]
    fn canonical_uniqueness_uniform_paletted_boundary() {
        let uniform = [bs(7); BRICK_VOLUME];
        assert_eq!(BrickPayload::from_cells(&uniform, PackStrategy::Rle), BrickPayload::Uniform(bs(7)));

        let mut almost = uniform;
        almost[BRICK_VOLUME - 1] = bs(8);
        assert_ne!(BrickPayload::from_cells(&almost, PackStrategy::Rle), BrickPayload::Uniform(bs(7)));
        roundtrips(&almost, PackStrategy::Rle);
    }

    // Reachable failure: fails if the 256/257 promotion boundary is
    // off-by-one in either direction (see the two dedicated tests above);
    // this test pins both sides of the boundary in one assertion pair.
    #[test]
    fn canonical_uniqueness_paletted_dense_boundary() {
        let mut at_cap = [bs(0); BRICK_VOLUME];
        let mut over_cap = [bs(0); BRICK_VOLUME];
        for i in 0..BRICK_VOLUME {
            at_cap[i] = bs((i % PALETTE_MAX) as u16);
            over_cap[i] = bs((i % (PALETTE_MAX + 1)) as u16);
        }
        assert!(!matches!(BrickPayload::from_cells(&at_cap, PackStrategy::Rle), BrickPayload::Dense(_)));
        assert!(matches!(BrickPayload::from_cells(&over_cap, PackStrategy::Rle), BrickPayload::Dense(_)));
    }

    // Reachable failure: fails on any grid where run-splitting drops or
    // duplicates a cell, or palette lookup returns the wrong index — seeded
    // so failures reproduce deterministically across a handful of "plausible
    // world" shapes (small distinct-id counts, mixed within a brick).
    #[test]
    fn roundtrip_mixed_random_seeded_grids() {
        for seed in [1u64, 2, 3, 42, 999] {
            let mut state = seed;
            let cells: [BlockState; BRICK_VOLUME] = std::array::from_fn(|_| bs((splitmix64(&mut state) % 6) as u16));
            roundtrips(&cells, PackStrategy::Paletted);
            roundtrips(&cells, PackStrategy::Rle);
        }
    }

}
