//! Far-terrain LOD tiles: a coarse *volumetric* downsample of the generator,
//! meshed with the very same greedy chunk mesher the full-res world uses. A tile
//! is a 16³ cell block covering a cube of `2^k`-metre cells beyond the full-res
//! region; each cell samples the generator (islands + overhangs included), and
//! the result meshes through [`build_chunk_mesh`](super::mesh::build_chunk_mesh)
//! exactly like a chunk — so a buried tile meshes to nothing, a sky tile to
//! nothing, and only the silhouette tiles carry geometry.
//!
//! Tiles are a pure function of the generator seed and coordinates — never edited,
//! never invalidated — so [`TileState`] has no `Dirty` variant (strictly fewer
//! states than a chunk's [`MeshState`](super::MeshState)); it keeps an `Air`
//! variant for the born-empty (sky/buried) tiles, mirroring the chunk path.
use voxel_engine::{Engine, Frame3D, MeshHandle, Vec3};

use super::ChunkMeshes;
use super::chunk::{CHUNK_SIZE, Chunk};
use super::generation::TerrainGenerator;
use super::mesh::{ChunkMeshData, Padded, build_chunk_mesh, new_chunk_mesh_data};
use crate::block::registry::{AIR, BlockId, HotTables};
use crate::coord::{ByPass, ChunkCoord};
use crate::world::light::PaddedLight;

/// LOD level carrier: everything about a tile's size derives from `k`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Lod(pub u8);

impl Lod {
    /// Metres per cell (`2^k`); also the per-tile `draw_mesh` scale.
    pub const fn cell(self) -> i32 {
        1 << self.0
    }
    /// Metres per tile side (`16·2^k`).
    pub const fn span(self) -> i32 {
        (CHUNK_SIZE as i32) << self.0
    }
    /// Chunk columns spanned per tile side (`2^k`).
    pub const fn chunks_per_side(self) -> i32 {
        1 << self.0
    }
}

/// The pyramid's finest tile level: 4 m cells, 64 m (4-chunk) tiles. Ring
/// selection lives in [`pyramid`](crate::world::pyramid) (`PyramidCfg.finest`
/// = this, plus coarser rings; band-edge seams are carried by calibrated droop
/// + ring overlap, not by grid alignment). This const remains only as the
/// compile-time floor for the skin coarseness assert (`skin.rs`:
/// `SKIN_LOD > TILE_LOD`).
pub(in crate::world) const TILE_LOD: Lod = Lod(2);

/// A far tile: 3D cell with level and grid coords. World min corner is (x, y, z) * span.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tile {
    pub lod: Lod,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Tile {
    pub fn origin_x(self) -> i32 {
        self.x * self.lod.span()
    }
    pub fn origin_y(self) -> i32 {
        self.y * self.lod.span()
    }
    pub fn origin_z(self) -> i32 {
        self.z * self.lod.span()
    }
}

/// Tile mesh lifecycle: Air (born-empty), Meshing (in flight), Ready (drawable).
/// No Dirty state; tiles are never edited.
pub(in crate::world) enum TileState {
    /// All air or fully buried (shell hides all faces); nothing drawn.
    Air,
    Meshing,
    /// Owns GPU mesh(es), one per present pass; draw offset = tile.origin
    /// (recomputed at draw time).
    Ready { meshes: ChunkMeshes },
}

impl TileState {
    /// The state a fresh multi-pass upload produces: `Ready` if any pass
    /// yielded a handle, else `Air` (an all-empty tile uploads to nothing).
    pub(in crate::world) fn from_upload(handles: ByPass<Option<MeshHandle>>) -> Self {
        match ChunkMeshes::from_upload_handles(handles) {
            Some(meshes) => TileState::Ready { meshes },
            None => TileState::Air,
        }
    }
    /// Free the tile's GPU mesh(es), if any. Consumes `self` (no double-free).
    pub(in crate::world) fn free(self, eng: &mut Engine) {
        if let TileState::Ready { meshes } = self {
            meshes.free(eng);
        }
    }
    /// Record a draw for each present pass, depth-biased so full-res chunks
    /// win on overlap.
    pub(in crate::world) fn draw(&self, f: &mut Frame3D, offset: Vec3, scale: f32) {
        if let TileState::Ready { meshes } = self {
            meshes.draw_biased(f, offset, scale);
        }
    }
}

/// Chebyshev distance from center's tile cell (streaming priority).
pub(in crate::world) fn tile_order(tile: Tile, center: super::Coord) -> i32 {
    let cps = tile.lod.chunks_per_side();
    let (tx, ty, tz) =
        (center.x.div_euclid(cps), center.y.div_euclid(cps), center.z.div_euclid(cps));
    (tile.x - tx).abs().max((tile.y - ty).abs()).max((tile.z - tz).abs())
}

/// Chunk size as a signed coordinate.
const CS: i32 = CHUNK_SIZE as i32;

/// Sample generator at tile's 2^k-metre stride into a padded 16³ block. Returns
/// padded neighbourhood and uniform interior id (if all agree). Shell is sampled
/// directly; tiles whose shell matches interior mesh to nothing.
fn sample_coarse<T: TerrainGenerator>(
    tile: Tile,
    terrain: &T,
    edits: &[(i32, i32, i32, BlockId)],
) -> (Padded, Option<BlockId>) {
    let cell = tile.lod.cell();
    let half = cell / 2;
    let (ox, oy, oz) = (tile.origin_x(), tile.origin_y(), tile.origin_z());
    // The padded y-range is -1..=16 (PAD = CHUNK_SIZE + 2). Each level's world-y is
    // the same across every column, so build the run once and share it — letting the
    // generator sample one column profile per run instead of one per cell.
    let ys: Vec<i32> = (0..CHUNK_SIZE + 2).map(|i| oy + (i as i32 - 1) * cell + half).collect();
    // Skip per-cell generation if tile is uniformly one block (sky/water) — but
    // only when no edit could break that uniformity within this tile's footprint.
    if edits.is_empty() {
        if let Some(id) = terrain.lod_tile_uniform(ox, oz, cell, ys[0], ys[CHUNK_SIZE + 1]) {
            return (Padded::uniform(id), Some(id));
        }
    }
    let padded = Padded::from_columns(|x, z, out| {
        let wx = ox + x * cell + half;
        let wz = oz + z * cell + half;
        terrain.lod_column(wx, wz, &ys, out);
        reduce_edits_into_column(out, edits, ox, oy, oz, x, z, cell, half);
    });
    // Interior-only uniform detection (the shell is excluded, matching how a
    // dense chunk's `uniform()` keys off its own 16³).
    let first = padded.at(0, 0, 0);
    let mut uniform = Some(first);
    'scan: for y in 0..CS {
        for z in 0..CS {
            for x in 0..CS {
                if padded.at(x, y, z) != first {
                    uniform = None;
                    break 'scan;
                }
            }
        }
    }
    (padded, uniform)
}

/// Flatten a `GenerateColumn`-shaped overlay (per-chunk flat-index cells) to
/// absolute world voxels `(wx, wy, wz, block)`, the shape the coarse-cell reducer
/// scans.
fn flatten_edits(edits: &[(ChunkCoord, Vec<(usize, BlockId)>)]) -> Vec<(i32, i32, i32, BlockId)> {
    let mut out = Vec::new();
    for (coord, cells) in edits {
        for &(index, id) in cells {
            let (lx, ly, lz) = Chunk::local_of(index);
            out.push((
                coord.x * CS + lx as i32,
                coord.y * CS + ly as i32,
                coord.z * CS + lz as i32,
                id,
            ));
        }
    }
    out
}

/// Coarse-cell reducer for ONE padded column `(x, z)` of a tile. For each
/// edit whose world XZ lands in this column's `cell`-wide footprint, resolve the
/// vertical cell it hits: a SOLID edit overwrites the cell (latest wins, since
/// edits replay in order); an AIR edit clears the cell ONLY when it covers the
/// cell's exact sample point — air never wins a vote it didn't earn.
#[allow(clippy::too_many_arguments)]
fn reduce_edits_into_column(
    out: &mut [BlockId],
    edits: &[(i32, i32, i32, BlockId)],
    ox: i32,
    oy: i32,
    oz: i32,
    x: i32,
    z: i32,
    cell: i32,
    half: i32,
) {
    if edits.is_empty() {
        return;
    }
    let (fx, fz) = (ox + x * cell, oz + z * cell); // this column's footprint min corner
    for &(ewx, ewy, ewz, id) in edits {
        if ewx < fx || ewx >= fx + cell || ewz < fz || ewz >= fz + cell {
            continue;
        }
        // `out[i]` is padded-y `i - 1`, world-y-centred at `oy + (i-1)*cell + half`.
        let i = (ewy - oy).div_euclid(cell) + 1;
        let Some(slot) = usize::try_from(i).ok().and_then(|i| out.get_mut(i)) else {
            continue;
        };
        if id != AIR {
            *slot = id; // latest solid wins the cell
        } else {
            let (sx, sy, sz) = (fx + half, oy + (i - 1) * cell + half, fz + half);
            if ewx == sx && ewy == sy && ewz == sz {
                *slot = AIR; // air only when it covers the sample point
            }
        }
    }
}

/// Build tile's mesh (every pass) from downsampled generator using full-res
/// mesher. Vertices are cell-local 0..=16, drawn at scale=lod.cell() + tile
/// origin. Lit as open-sky (full skylight, no blocklight) so tile shading
/// tracks day/night like real surface chunks. Empty tiles become Air state.
pub fn build_tile_mesh<T: TerrainGenerator>(
    tile: Tile,
    terrain: &T,
    edits: &[(ChunkCoord, Vec<(usize, BlockId)>)],
    tables: &HotTables,
) -> ChunkMeshData {
    use voxel_engine::profile::{Meter, add};
    // Split the tile job into its two halves — coarse generator sampling vs.
    // greedy meshing — so the unified report shows which one the ~46ms/job cost
    // lives in. Worker-thread code, but `profile` is an atomic global sink.
    let t0 = std::time::Instant::now();
    // Flatten the `GenerateColumn`-shaped overlay to world-space voxels once, so
    // the per-column reducer is a flat scan (edit counts per tile are small).
    let flat = flatten_edits(edits);
    let (padded, uniform) = sample_coarse(tile, terrain, &flat);
    add(Meter::TileSample, t0.elapsed());

    let t1 = std::time::Instant::now();
    let mut data = new_chunk_mesh_data();
    build_chunk_mesh(&padded, uniform, tables, &PaddedLight::open_sky(), &mut data);
    add(Meter::TileMesh, t1.elapsed());

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use voxel_engine::{MeshData, MeshVertex, Normal, Pass};

    /// These fixtures use only fully-opaque blocks (Grass/Stone), so their
    /// geometry always lands in the opaque pass.
    fn opaque(data: &ChunkMeshData) -> &MeshData {
        &data[Pass::Opaque]
    }

    fn tables() -> HotTables {
        BlockRegistry::with_builtins().hot_tables()
    }

    /// Flat generator: deep below h, surface on top, air above. Lets tile invariants
    /// be exactly reasoned (vs opaque SineHills with caves/islands). Uses real ids.
    struct FlatGen {
        h: i32,
        surface: BlockId,
        deep: BlockId,
    }
    impl TerrainGenerator for FlatGen {
        fn height(&self, _: i32, _: i32) -> i32 {
            self.h
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            self.surface
        }
        fn deep(&self) -> BlockId {
            self.deep
        }
        /// Uniform air for spans wholly above the surface.
        fn lod_tile_uniform(&self, _ox: i32, _oz: i32, _cell: i32, y0: i32, _y1: i32) -> Option<BlockId> {
            (y0 >= self.h).then_some(crate::block::registry::AIR)
        }
    }
    fn flat(h: i32) -> FlatGen {
        let reg = BlockRegistry::with_builtins();
        FlatGen { h, surface: reg.id_by_name("Grass").unwrap(), deep: reg.id_by_name("Stone").unwrap() }
    }

    /// Every quad must wind CCW from outside (engine back-face-culls otherwise).
    fn assert_winds_outward(data: &MeshData) {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        for q in data.vertices().chunks_exact(4) {
            let p: Vec<[f32; 3]> = q.iter().map(|v| v.local_pos()).collect();
            let c = cross(sub(p[1], p[0]), sub(p[3], p[0]));
            let d = q[0].normal().direction();
            let dot = c[0] * d[0] as f32 + c[1] * d[1] as f32 + c[2] * d[2] as f32;
            assert!(dot > 0.0, "quad faces inward: normal {:?}", q[0].normal());
        }
    }

    #[test]
    fn lod_units_derive_from_the_level() {
        let k = Lod(2);
        assert_eq!(k.cell(), 4);
        assert_eq!(k.chunks_per_side(), 4);
        assert_eq!(k.span(), 64);
    }

    #[test]
    fn tile_origin_is_indexed_on_every_axis() {
        let tile = Tile { lod: Lod(2), x: 1, y: -2, z: 3 };
        assert_eq!(tile.origin_x(), 64);
        assert_eq!(tile.origin_y(), -128);
        assert_eq!(tile.origin_z(), 192);
    }

    /// Surface tile meshes outward-winding surface with top face.
    #[test]
    fn surface_tile_meshes_a_visible_surface() {
        // h = 100 → the tile y=1 (world Y 64..128) brackets the surface.
        let tile = Tile { lod: Lod(2), x: 0, y: 1, z: 0 };
        let data = build_tile_mesh(tile, &flat(100), &[], &tables());
        let mesh = opaque(&data);
        assert!(!mesh.vertices().is_empty(), "the surface tile has geometry");
        assert!(mesh.vertices().iter().any(|v: &MeshVertex| v.normal() == Normal::PosY), "a top");
        assert_winds_outward(mesh);
    }

    /// Sky tile (above surface) meshes empty (born Air).
    #[test]
    fn sky_tile_meshes_empty() {
        // h = 100; tile y=5 spans world Y 320..384, all above the surface.
        let tile = Tile { lod: Lod(2), x: 0, y: 5, z: 0 };
        assert!(opaque(&build_tile_mesh(tile, &flat(100), &[], &tables())).vertices().is_empty(), "sky is empty");
    }

    /// Early-out produces same result as full per-cell sample.
    #[test]
    fn sky_tile_early_out_matches_full_sample() {
        // h = 100; tile y=5 spans world Y 320..384, wholly above the surface.
        let tile = Tile { lod: Lod(2), x: 0, y: 5, z: 0 };
        let (padded, uniform) = sample_coarse(tile, &flat(100), &[]);
        assert_eq!(uniform, Some(crate::block::registry::AIR), "sky tile detected uniform air");
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    assert_eq!(padded.at(x, y, z), crate::block::registry::AIR);
                }
            }
        }
        assert!(opaque(&build_tile_mesh(tile, &flat(100), &[], &tables())).vertices().is_empty());
    }

    /// Buried tile (below surface) meshes empty; uniform shell hides interior faces.
    #[test]
    fn buried_tile_meshes_empty() {
        // h = 100; tile y=0 spans world Y 0..64, all deep (surface is at 99).
        let tile = Tile { lod: Lod(2), x: 0, y: 0, z: 0 };
        assert!(opaque(&build_tile_mesh(tile, &flat(100), &[], &tables())).vertices().is_empty(), "buried is empty");
    }
}
