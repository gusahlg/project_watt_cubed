//! The cosmos's round bodies painted on curved charts in storage (SPACE-ARCHITECTURE §7).
//!
//! Every ball-shaped body (Verdance, the Ember, the moons) gets an atlas of cube-sphere charts down
//! to a Cartesian core, the Hollow two shell atlases (its outer crust, and its inner surface facing
//! the Ember); each has a painter ([`Round`]). The generator answers storage chunk coordinates from
//! here. Physical space holds none of their cells; their gravity is the cosmos's analytic matter,
//! unchanged. Which body gets a chart is initial world state chosen by the generator — physics never
//! asks (a body a player builds behaves by its matter alone).

use std::sync::Arc;

use glam::DVec3;

use super::Materials;
use super::cosmos::{Body, Cosmos, Kind, RELIEF, Shape};
use super::round::{Round, Style};
use crate::block::registry::{AIR, BlockId};
use crate::coord::ChunkCoord;
use crate::space::atlas::{Atlas, Patch, STORAGE_X0};
use crate::world::chunk::{CHUNK_SIZE, ChunkData};

/// Unit direction from a moon's parent (the nearest body that is not a moon and not the Ember,
/// which shares the Hollow's centre) out to the moon. Ice caps face this way.
fn parent_pole(bodies: &[Body], moon: &Body) -> DVec3 {
    let c = moon.centre_f();
    let parent = bodies.iter().filter(|b| b.id != moon.id && b.kind != Kind::Moon && b.kind != Kind::Ember).min_by(|a, b| {
        (a.centre_f() - c).length_squared().total_cmp(&(b.centre_f() - c).length_squared())
    });
    match parent {
        Some(p) => {
            let d = c - p.centre_f();
            let len = d.length();
            if len > 1.0 { d / len } else { DVec3::Y }
        }
        None => DVec3::Y,
    }
}

const CS: i64 = CHUNK_SIZE as i64;
/// Storage y no relief reaches below its datum by (with the deepest caves under it): band-0 chunks
/// entirely deeper than this are the body's uniform deep fill.
const DEEP: i64 = 1_200 + 400;
/// A storage column covered by a box but not by a surface chart: underground, never open sky.
pub const BURIED: i32 = i32::MAX / 2;

/// One charted body: its painter and its boxes in chunk coordinates (`hi` exclusive).
struct Charted {
    round: Round,
    boxes: Vec<(Patch, [i64; 3], [i64; 3])>,
}

/// The round bodies of one cosmos.
pub struct StorageWorlds {
    worlds: Vec<Charted>,
    atlases: Vec<Arc<Atlas>>,
}

impl StorageWorlds {
    /// Chart every round body of `cosmos` (slots in catalog order).
    pub fn new(cosmos: &Cosmos, m: &Arc<Materials>) -> Self {
        let mut worlds = Vec::new();
        let mut slot = 0u32;
        let add = |atlas: Atlas, seed: u32, style: Style, pole: DVec3, worlds: &mut Vec<Charted>| {
            let boxes = atlas
                .patches()
                .map(|p| {
                    let (o, size) = atlas.storage_box(p);
                    (p, o.map(|v| v / CS), std::array::from_fn(|a| (o[a] + size[a]) / CS))
                })
                .collect();
            worlds.push(Charted { round: Round::new(atlas, seed, style, m.clone(), pole), boxes });
        };
        // Copied so a moon can look up its parent while the loop still holds a body.
        let bodies = cosmos.bodies().to_vec();
        for b in &bodies {
            let c = b.centre_f();
            match (b.kind, b.shape) {
                (_, Shape::Cube { .. }) => {}
                (Kind::Hollow, Shape::Shell { outer, inner }) => {
                    let mid = (outer + inner) / 2;
                    add(Atlas::shell(c, outer, mid, outer + RELIEF, false, slot), b.seed, Style::HollowOuter, DVec3::ZERO, &mut worlds);
                    add(Atlas::shell(c, inner, inner - RELIEF, mid, true, slot + 1), b.seed ^ 0x1A2B, Style::HollowInner, DVec3::ZERO, &mut worlds);
                    slot += 2;
                }
                (_, Shape::Shell { .. }) => {}
                (kind, Shape::Ball { r }) => {
                    let style = match kind {
                        Kind::Verdant => Style::Verdant,
                        Kind::Ember => Style::Ember,
                        _ => Style::Moon { tone: (b.seed % 3) as u8 },
                    };
                    let pole = if kind == Kind::Moon { parent_pole(&bodies, b) } else { DVec3::ZERO };
                    add(Atlas::new(c, r, r + RELIEF, false, slot), b.seed, style, pole, &mut worlds);
                    slot += 1;
                }
            }
        }
        let atlases = worlds.iter().map(|w| Arc::new(w.round.atlas.clone())).collect();
        Self { worlds, atlases }
    }

    /// The atlases, one per charted surface.
    pub fn atlases(&self) -> &[Arc<Atlas>] {
        &self.atlases
    }

    /// Whether chunk `c` lies in the storage region (beyond the physical universe).
    #[inline]
    pub fn owns(c: ChunkCoord) -> bool {
        c.x as i64 * CS >= STORAGE_X0
    }

    /// The body and box holding storage chunk `c`.
    fn find(&self, c: ChunkCoord) -> Option<(&Charted, Patch, [i64; 3])> {
        let k = [c.x as i64, c.y as i64, c.z as i64];
        self.worlds.iter().find_map(|w| {
            w.boxes
                .iter()
                .find(|(_, lo, hi)| (0..3).all(|a| k[a] >= lo[a] && k[a] < hi[a]))
                .map(|&(p, lo, _)| (w, p, lo))
        })
    }

    /// The uniform block of a storage chunk known without painting it (outside every box: air; the
    /// deep body of a band, the deeper bands, the transition and the core).
    pub fn uniform(&self, c: ChunkCoord) -> Option<BlockId> {
        let Some((w, patch, lo)) = self.find(c) else { return Some(AIR) };
        match patch {
            Patch::Shell { band: 0, .. } => {
                let atlas = &w.round.atlas;
                let b = atlas.bands[0];
                // Storage y of the datum in this chart, and how far below it this chunk's top lies.
                let datum = if atlas.inward { b.r_hi - atlas.radius } else { atlas.radius - b.r_lo };
                let top = (c.y as i64 - lo[1]) * CS + CS - 1;
                (top < datum - DEEP).then(|| w.round.deep())
            }
            Patch::Shell { .. } => Some(w.round.deep()),
            Patch::Transition { .. } => Some(w.round.heart().0),
            Patch::Core => Some(w.round.heart().1),
        }
    }

    /// The storage chunk at `c`.
    pub fn generate(&self, c: ChunkCoord) -> ChunkData {
        if let Some(id) = self.uniform(c) {
            return ChunkData::Uniform(id);
        }
        let (w, _, _) = self.find(c).expect("a non-uniform storage chunk lies in a box");
        w.round.fill_chunk([c.x as i64, c.y as i64, c.z as i64])
    }

    /// The block in storage cell `(x, y, z)`.
    pub fn voxel(&self, x: i32, y: i32, z: i32) -> BlockId {
        let c = ChunkCoord::new(x.div_euclid(CS as i32), y.div_euclid(CS as i32), z.div_euclid(CS as i32));
        if let Some(id) = self.uniform(c) {
            return id;
        }
        let (w, _, _) = self.find(c).expect("a non-uniform storage chunk lies in a box");
        w.round.voxel([x as i64, y as i64, z as i64])
    }

    /// [`surface`](Self::surface) of the 16×16 columns of storage chunk column `(cx, cz)`, indexed
    /// `lx + lz·16`, sampling each chart's relief lattice once.
    pub fn heights_16(&self, cx: i32, cz: i32) -> [i32; CHUNK_SIZE * CHUNK_SIZE] {
        let (kx, kz) = (cx as i64, cz as i64);
        for w in &self.worlds {
            for &(patch, lo, hi) in &w.boxes {
                if kx < lo[0] || kx >= hi[0] || kz < lo[2] || kz >= hi[2] {
                    continue;
                }
                return match patch {
                    Patch::Shell { band: 0, .. } => {
                        let s = w.round.chunk_surfaces(patch, (kx - lo[0]) * CS, (kz - lo[2]) * CS);
                        std::array::from_fn(|k| (s[k] + lo[1] * CS).clamp(i32::MIN as i64 + 1, BURIED as i64 - 1) as i32)
                    }
                    _ => [BURIED; CHUNK_SIZE * CHUNK_SIZE],
                };
            }
        }
        [i32::MIN; CHUNK_SIZE * CHUNK_SIZE]
    }

    /// The charted surface `body - CHART_BODY_BASE` names, if it is one of these atlases.
    pub fn chart(&self, index: usize) -> Option<&Atlas> {
        self.atlases.get(index).map(|a| a.as_ref())
    }

    /// Storage y of relief 0 on atlas `index`'s band-0 charts (every face shares the radial window).
    pub fn datum(&self, index: usize) -> Option<i32> {
        let atlas = self.chart(index)?;
        let b = atlas.bands.first()?;
        let y = if atlas.inward { b.r_hi - atlas.radius } else { atlas.radius - b.r_lo };
        i32::try_from(y).ok()
    }

    /// Min and max storage y of the surface over the square `[x0, x0+span) × [z0, z0+span)`,
    /// plus plant reach on the top. `None` when the square is not one band-0 chart.
    pub fn bounds(&self, x0: i32, z0: i32, span: i32) -> Option<(i32, i32)> {
        if span <= 0 {
            return None;
        }
        let (cx, cz) = (x0.div_euclid(CS as i32) as i64, z0.div_euclid(CS as i32) as i64);
        let (x1, z1) = (x0 as i64 + span as i64 - 1, z0 as i64 + span as i64 - 1);
        for w in &self.worlds {
            for &(patch, lo, hi) in &w.boxes {
                if cx < lo[0] || cx >= hi[0] || cz < lo[2] || cz >= hi[2] {
                    continue;
                }
                let (bx0, bx1) = (lo[0] * CS, hi[0] * CS);
                let (bz0, bz1) = (lo[2] * CS, hi[2] * CS);
                if (x0 as i64) < bx0 || x1 >= bx1 || (z0 as i64) < bz0 || z1 >= bz1 {
                    return None;
                }
                let Patch::Shell { band: 0, .. } = patch else { return None };
                let (lo_h, hi_h) = w.round.relief_bounds(patch, x0 as i64 - bx0, z0 as i64 - bz0, span as i64);
                let y_lo = w.round.surface_of_relief(patch, lo_h) + lo[1] * CS;
                let y_hi = w.round.surface_of_relief(patch, hi_h) + lo[1] * CS + super::round::PLANT_REACH;
                let lo_i = y_lo.clamp(i32::MIN as i64 + 1, i32::MAX as i64 - 1) as i32;
                let hi_i = y_hi.clamp(lo_i as i64 + 1, i32::MAX as i64) as i32;
                return Some((lo_i, hi_i));
            }
        }
        None
    }

    /// Far-LOD blocks of storage column `(x, z)` at altitudes `ys` (solid below the surface, plants
    /// above, no caves). Air outside every box.
    pub fn lod_column(&self, x: i32, z: i32, ys: &[i32], out: &mut [BlockId]) {
        let n = out.len().min(ys.len());
        let (cx, cz) = (x.div_euclid(CS as i32) as i64, z.div_euclid(CS as i32) as i64);
        for w in &self.worlds {
            for &(patch, lo, hi) in &w.boxes {
                if cx < lo[0] || cx >= hi[0] || cz < lo[2] || cz >= hi[2] {
                    continue;
                }
                match patch {
                    Patch::Shell { band: 0, .. } => {
                        let (i, j) = (x as i64 - lo[0] * CS, z as i64 - lo[2] * CS);
                        w.round.lod_column(patch, i, j, lo[1] * CS, &ys[..n], &mut out[..n]);
                    }
                    _ => out[..n].fill(w.round.deep()),
                }
                return;
            }
        }
        out[..n].fill(AIR);
    }

    /// Storage y of the first open cell of storage column `(x, z)` (+Y is every chart's up): the
    /// painter's surface on a surface chart, [`BURIED`] under the deeper bands and the core, and
    /// `i32::MIN` (open) outside every box. Boxes never share a column.
    pub fn surface(&self, x: i32, z: i32) -> i32 {
        let (cx, cz) = (x.div_euclid(CS as i32) as i64, z.div_euclid(CS as i32) as i64);
        for w in &self.worlds {
            for &(patch, lo, hi) in &w.boxes {
                if cx < lo[0] || cx >= hi[0] || cz < lo[2] || cz >= hi[2] {
                    continue;
                }
                return match patch {
                    Patch::Shell { band: 0, .. } => {
                        let (i, j) = (x as i64 - lo[0] * CS, z as i64 - lo[2] * CS);
                        let s = w.round.column_surface(patch, i, j) + lo[1] * CS;
                        s.clamp(i32::MIN as i64 + 1, BURIED as i64 - 1) as i32
                    }
                    _ => BURIED,
                };
            }
        }
        i32::MIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use crate::world::chunk::Chunk;

    fn worlds() -> (StorageWorlds, Cosmos) {
        let mut reg = BlockRegistry::with_builtins();
        let m = Arc::new(Materials::intern(&mut reg));
        let cosmos = Cosmos::new(42, 1.0);
        (StorageWorlds::new(&cosmos, &m), cosmos)
    }

    #[test]
    fn every_round_body_is_charted_once_in_its_own_slot() {
        let (w, cosmos) = worlds();
        let balls = cosmos.bodies().iter().filter(|b| matches!(b.shape, Shape::Ball { .. })).count();
        let shells = cosmos.bodies().iter().filter(|b| matches!(b.shape, Shape::Shell { .. })).count();
        assert_eq!(w.atlases().len(), balls + 2 * shells);
        assert!(w.atlases().len() >= 4, "Verdance, the Hollow's two surfaces, the Ember, moons");
        // Boxes of different bodies never share a storage column (or anything else).
        let all: Vec<_> = w.worlds.iter().flat_map(|c| c.boxes.iter().map(|b| (b.1, b.2))).collect();
        for (i, (lo, hi)) in all.iter().enumerate() {
            for (lo2, hi2) in &all[i + 1..] {
                let overlap = [0, 2].iter().all(|&k| lo[k] < hi2[k] && lo2[k] < hi[k]);
                assert!(!overlap, "{lo:?}..{hi:?} and {lo2:?}..{hi2:?}");
            }
        }
        assert!(all.iter().all(|(_, hi)| hi[0] * CS < crate::math::CELL_LIMIT as i64), "inside i32 chunk math");
    }

    #[test]
    fn storage_chunks_are_the_painter_and_deep_ones_are_uniform() {
        let (w, _) = worlds();
        for c in &w.worlds {
            let atlas = &c.round.atlas;
            let b = atlas.bands[0];
            let patch = Patch::Shell { band: 0, face: crate::coord::Face::PosZ };
            let (i, j) = (b.n / 2 + 3, b.n / 3);
            let surf = c.round.column_surface(patch, i, j);
            let s = atlas.storage(patch, [i, surf, j]);
            let k = ChunkCoord::new((s[0] / CS) as i32, (s[1] / CS) as i32, (s[2] / CS) as i32);
            let data = w.generate(k);
            for (lx, ly, lz) in [(0, 0, 0), (7, 3, 9), (15, 15, 15), (2, 12, 6)] {
                let cell = (k.x * 16 + lx as i32, k.y * 16 + ly as i32, k.z * 16 + lz as i32);
                assert_eq!(data.get(Chunk::index(lx, ly, lz)), w.voxel(cell.0, cell.1, cell.2));
            }
            assert_eq!(w.surface(s[0] as i32, s[2] as i32), surf as i32 + (atlas.storage_box(patch).0[1]) as i32);
            // Far below the surface: uniform, and equal to what the painter paints there.
            let deep = atlas.storage(patch, [i, surf - 3 * DEEP, j]);
            let kd = ChunkCoord::new((deep[0] / CS) as i32, (deep[1] / CS) as i32, (deep[2] / CS) as i32);
            let id = w.uniform(kd).expect("deep chunks are uniform");
            assert_eq!(id, c.round.voxel(deep));
        }
        assert_eq!(w.uniform(ChunkCoord::new((STORAGE_X0 / CS) as i32 - 5, 0, 0)), Some(AIR));
    }

    /// Storage chunk cost at the surface of each charted body (release:
    /// `cargo test --release --lib storage_chunk_cost -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn storage_chunk_cost() {
        let (w, _) = worlds();
        for c in &w.worlds {
            let atlas = &c.round.atlas;
            let b = atlas.bands[0];
            let patch = Patch::Shell { band: 0, face: crate::coord::Face::PosX };
            let mut chunks = Vec::new();
            for k in 0..64i64 {
                let (i, j) = (b.n / 2 + k * 16, b.n / 3 + (k % 8) * 16);
                let surf = c.round.column_surface(patch, i, j);
                for dy in [-24i64, -8, 8] {
                    let s = atlas.storage(patch, [i, surf + dy, j]);
                    chunks.push(ChunkCoord::new((s[0] / CS) as i32, (s[1] / CS) as i32, (s[2] / CS) as i32));
                }
            }
            let t = std::time::Instant::now();
            let mut solid = 0usize;
            for &k in &chunks {
                if !matches!(w.generate(k), ChunkData::Uniform(AIR)) {
                    solid += 1;
                }
            }
            let per = t.elapsed().as_secs_f64() * 1e6 / chunks.len() as f64;
            eprintln!("{:?}: {per:.0} µs per surface chunk ({solid}/{} not air)", c.round.style(), chunks.len());
        }
    }
}
