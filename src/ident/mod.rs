//! World identity: `W = fold(edits)(generator@version)`. The single source of
//! truth the rest of the engine memoizes — pure types only, no I/O, nothing
//! wired to it yet.

use crate::block::BlockId;

pub mod codec;

/// Canonical definition lives in the engine (`producer::Detail`, alongside the
/// GPU-boundary biased encoding it shares with the app) — one type, not two
/// structurally identical ones.
pub use voxel_engine::producer::Detail;

/// Finest addressable level for edit coordinates — reserve below half blocks.
pub const EDIT_LEVEL: Detail = Detail(-2);

/// Cell position at [`EDIT_LEVEL`]. Explicit widths are the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellPos {
    pub x: i64,
    pub y: i32,
    pub z: i64,
}

/// Block value = id + variant/orientation state. State 0 = today's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockState {
    pub id: BlockId,
    pub state: u16,
}

/// Opaque per-connection player identifier (net roster key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);

/// Key into a block's typed side-channel data (an [`Edit::Data`] payload's slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataKey(pub u32);

/// An edit's position within a [`WorldIdentity`]'s log. Monotone within one log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EditSeq(pub u64);

/// Pinned terrain-generator algorithm version. Old versions remain constructible
/// so a save keeps replaying against the generator it was created with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratorVersion(pub u32);

/// Multi-dimension reserve; single-world play uses [`DimensionKey::MAIN`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimensionKey(pub u16);

impl DimensionKey {
    pub const MAIN: DimensionKey = DimensionKey(0);
}

/// Which sim solver produced a synthetic edit. Solvers mutate the world ONLY
/// through `Edit`s tagged `EditSource::Sim` — never directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Thermal,
    Electrical,
}

// Wire discriminants for `Edit`'s tagged union. Fixed here so a variant's
// serialized identity is never accidentally tied to enum declaration order.
// 0x10..=0x7F reserved for future verbs; changing an existing tag is forbidden.
pub const EDIT_TAG_CELL: u8 = 0x01;
pub const EDIT_TAG_FILL: u8 = 0x02;
pub const EDIT_TAG_SPHERE: u8 = 0x03;
pub const EDIT_TAG_DATA: u8 = 0x04;

/// The edit vocabulary. Serialized as a tagged union with RESERVED tags; adding a
/// variant is a codec-compatible extension, changing one is forbidden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    Cell {
        pos: CellPos,
        block: BlockState,
    },
    Fill {
        min: CellPos,
        max: CellPos,
        block: BlockState,
    },
    Sphere {
        center: CellPos,
        radius_cells: u32,
        block: BlockState,
    },
    Data {
        pos: CellPos,
        key: DataKey,
        value: Box<[u8]>,
    },
}

/// Who committed the edit. Sim solvers mutate the world ONLY through this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditSource {
    Player(PlayerId),
    Sim(FieldKind),
    System,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamped {
    pub seq: EditSeq,
    pub source: EditSource,
    pub edit: Edit,
}

/// Append-only edit log. Compaction folds a prefix into an external snapshot and
/// advances the marker; it never truncates `entries`, so "replay from snapshot +
/// suffix" and "replay the whole log" stay the same operation over different
/// starting points (compaction soundness, tested below).
#[derive(Debug, Clone, Default)]
pub struct EditLog {
    entries: Vec<Stamped>,
    compacted_through: usize,
}

impl EditLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, stamped: Stamped) {
        self.entries.push(stamped);
    }

    pub fn entries(&self) -> &[Stamped] {
        &self.entries
    }

    /// Index of the first entry not yet folded into some external snapshot.
    pub fn compacted_through(&self) -> usize {
        self.compacted_through
    }

    /// Record that `[0, through)` has been folded into a snapshot elsewhere.
    pub fn mark_compacted(&mut self, through: usize) {
        debug_assert!(through <= self.entries.len());
        self.compacted_through = self.compacted_through.max(through);
    }
}

/// The world's name. Save persists it; net replicates it; everything else is cache.
pub struct WorldIdentity {
    pub generator: GeneratorVersion,
    pub dimension: DimensionKey,
    pub log: EditLog,
}

#[cfg(test)]
mod laws {
    use super::*;
    use std::collections::HashMap;

    /// Toy world state for the fold laws: nothing here is the real world
    /// representation (that's `src/world/brick.rs`, landing separately) — it's
    /// the smallest structure that can host both block content and cell-data
    /// side channels so `Edit`'s fold laws are exercised, not assumed.
    #[derive(Clone, PartialEq, Eq, Debug, Default)]
    struct Grid {
        blocks: HashMap<CellPos, BlockState>,
        data: HashMap<(CellPos, DataKey), Vec<u8>>,
    }

    fn apply(grid: &mut Grid, edit: &Edit) {
        for pos in footprint(edit) {
            match edit {
                Edit::Cell { block, .. } | Edit::Fill { block, .. } | Edit::Sphere { block, .. } => {
                    grid.blocks.insert(pos, *block);
                }
                Edit::Data { key, value, .. } => {
                    grid.data.insert((pos, *key), value.to_vec());
                }
            }
        }
    }

    fn fold(edits: &[Edit]) -> Grid {
        let mut grid = Grid::default();
        for e in edits {
            apply(&mut grid, e);
        }
        grid
    }

    /// Every cell an edit's footprint covers — the same enumeration `apply` uses,
    /// so the locality test can name "everything else" precisely.
    fn footprint(edit: &Edit) -> Vec<CellPos> {
        match edit {
            Edit::Cell { pos, .. } => vec![*pos],
            Edit::Data { pos, .. } => vec![*pos],
            Edit::Fill { min, max, .. } => {
                let mut cells = Vec::new();
                for x in min.x..=max.x {
                    for y in min.y..=max.y {
                        for z in min.z..=max.z {
                            cells.push(CellPos { x, y, z });
                        }
                    }
                }
                cells
            }
            Edit::Sphere {
                center,
                radius_cells,
                ..
            } => {
                let r = *radius_cells as i64;
                let mut cells = Vec::new();
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if dx * dx + dy * dy + dz * dz <= r * r {
                                cells.push(CellPos {
                                    x: center.x + dx,
                                    y: center.y + dy as i32,
                                    z: center.z + dz,
                                });
                            }
                        }
                    }
                }
                cells
            }
        }
    }

    fn block(id: u16) -> BlockState {
        BlockState {
            id: BlockId(id),
            state: 0,
        }
    }

    /// Small but mixed corpus: one Cell per grid point plus one of each region verb,
    /// touching overlapping cells so later edits can shadow earlier ones. State-space
    /// coverage over cleverness — every variant, every overlap kind, once.
    fn small_edits() -> Vec<Edit> {
        let mut edits = Vec::new();
        for x in 0..3i64 {
            for y in 0..2i32 {
                for z in 0..3i64 {
                    edits.push(Edit::Cell {
                        pos: CellPos { x, y, z },
                        block: block(((x + y as i64 + z) % 3) as u16),
                    });
                }
            }
        }
        edits.push(Edit::Fill {
            min: CellPos { x: 0, y: 0, z: 0 },
            max: CellPos { x: 1, y: 1, z: 1 },
            block: block(7),
        });
        edits.push(Edit::Sphere {
            center: CellPos { x: 1, y: 0, z: 1 },
            radius_cells: 1,
            block: block(9),
        });
        edits.push(Edit::Data {
            pos: CellPos { x: 2, y: 1, z: 2 },
            key: DataKey(5),
            value: Box::new([1, 2, 3]),
        });
        edits
    }

    /// Law 1 (compaction soundness): folding a prefix into a snapshot, then folding
    /// the suffix from there, must equal folding the whole log — for every split.
    #[test]
    fn edit_action_associativity() {
        let edits = small_edits();
        let whole = fold(&edits);
        for split in 0..=edits.len() {
            let mut snapshot = fold(&edits[..split]);
            for e in &edits[split..] {
                apply(&mut snapshot, e);
            }
            assert_eq!(snapshot, whole, "split at {split} diverged from whole-log fold");
        }
    }

    /// Law 2: `Fill` over a region ≡ the fold of that region's constituent `Cell`s.
    #[test]
    fn fill_equals_fold_of_cells() {
        let min = CellPos { x: 0, y: 0, z: 0 };
        let max = CellPos { x: 1, y: 1, z: 1 };
        let value = block(4);

        let via_fill = fold(&[Edit::Fill { min, max, block: value }]);

        let mut via_cells = Vec::new();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    via_cells.push(Edit::Cell {
                        pos: CellPos { x, y, z },
                        block: value,
                    });
                }
            }
        }
        assert_eq!(via_fill, fold(&via_cells));
    }

    /// Law 3 (locality): applying an edit changes only cells in its footprint.
    #[test]
    fn locality_outside_footprint() {
        let edits = small_edits();
        for i in 0..edits.len() {
            let before = fold(&edits[..i]);
            let touched = footprint(&edits[i]);
            let mut after = before.clone();
            apply(&mut after, &edits[i]);

            for x in 0..3i64 {
                for y in 0..2i32 {
                    for z in 0..3i64 {
                        let p = CellPos { x, y, z };
                        if touched.contains(&p) {
                            continue;
                        }
                        assert_eq!(
                            before.blocks.get(&p),
                            after.blocks.get(&p),
                            "edit {i} touched {p:?} outside its declared footprint"
                        );
                    }
                }
            }
        }
    }

    /// Law 4: replaying the same log twice yields identical grids (integer-only fold,
    /// no floats/hash-order-dependent reads in `apply`, so this holds by construction —
    /// pinned as a regression test on the fold itself, not on `HashMap` iteration order).
    #[test]
    fn replay_determinism() {
        let edits = small_edits();
        assert_eq!(fold(&edits), fold(&edits));
    }
}
