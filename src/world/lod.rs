//! Far-terrain LOD tiles: a coarse *volumetric* downsample of the generator,
//! meshed with the very same greedy chunk mesher the full-res world uses. A tile
//! is a 16³ cell block covering a cube of `2^k`-metre cells beyond the full-res
//! region; each cell samples the generator (islands + overhangs included), and
//! the result meshes through [`build_chunk_mesh`](super::mesh::build_chunk_mesh)
//! exactly like a chunk — so a buried tile meshes to nothing, a sky tile to
//! nothing, and only the silhouette tiles carry geometry (WORLD-DESIGN §7.7, §12).
//!
//! Tiles are a pure function of the generator seed and coordinates — never edited,
//! never invalidated — so [`TileState`] has no `Dirty` variant (strictly fewer
//! states than a chunk's [`MeshState`](super::MeshState)); it keeps an `Air`
//! variant for the born-empty (sky/buried) tiles, mirroring the chunk path.
use voxel_engine::{Engine, MeshData, MeshHandle};

use super::OwnedMesh;
use super::chunk::CHUNK_SIZE;
use super::generation::TerrainGenerator;
use super::mesh::{Padded, build_chunk_mesh, new_chunk_mesh_data};
use crate::block::registry::{BlockId, HotTables};
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

/// The single far-terrain LOD level: 4 m cells, 64 m (4-chunk) tiles. A pyramid
/// of levels was tried but its differently-sized grids left cracks at the
/// level-to-level band edges; one uniform level tiles the far shell seamlessly.
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
    /// Owns GPU mesh; draw offset = tile.origin (recomputed at draw time).
    Ready { mesh: OwnedMesh },
}

impl TileState {
    /// Wrap a freshly uploaded tile handle.
    pub(in crate::world) fn ready(handle: MeshHandle) -> Self {
        TileState::Ready { mesh: OwnedMesh::new(handle) }
    }
    /// Free the tile's GPU mesh, if any. Consumes `self` (no double-free).
    pub(in crate::world) fn free(self, eng: &mut Engine) {
        if let TileState::Ready { mesh } = self {
            mesh.free(eng);
        }
    }
    /// Drawable handle if Ready.
    pub(in crate::world) fn drawable(&self) -> Option<MeshHandle> {
        match self {
            TileState::Ready { mesh } => Some(mesh.id()),
            TileState::Air | TileState::Meshing => None,
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
fn sample_coarse<T: TerrainGenerator>(tile: Tile, terrain: &T) -> (Padded, Option<BlockId>) {
    let cell = tile.lod.cell();
    let half = cell / 2;
    let (ox, oy, oz) = (tile.origin_x(), tile.origin_y(), tile.origin_z());
    let sample = |x: i32, y: i32, z: i32| -> BlockId {
        let wx = ox + x * cell + half;
        let wy = oy + y * cell + half;
        let wz = oz + z * cell + half;
        terrain.block_at(wx, wy, wz, terrain.height(wx, wz))
    };
    let padded = Padded::from_cells(sample);
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

/// Build tile's opaque mesh from downsampled generator using full-res mesher.
/// Vertices are cell-local 0..=16, drawn at scale=lod.cell() + tile origin.
/// Lit flat; only opaque pass kept. Empty tiles become Air state.
pub fn build_tile_mesh<T: TerrainGenerator>(tile: Tile, terrain: &T, tables: &HotTables) -> MeshData {
    let (padded, uniform) = sample_coarse(tile, terrain);
    let mut data = new_chunk_mesh_data();
    build_chunk_mesh(&padded, uniform, tables, &PaddedLight::full(), &mut data);
    let [opaque, _transparent] = data.into_slots();
    opaque
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::BlockRegistry;
    use voxel_engine::{MeshVertex, Normal};

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
        let data = build_tile_mesh(tile, &flat(100), &tables());
        assert!(!data.vertices().is_empty(), "the surface tile has geometry");
        assert!(data.vertices().iter().any(|v: &MeshVertex| v.normal() == Normal::PosY), "a top");
        assert_winds_outward(&data);
    }

    /// Sky tile (above surface) meshes empty (born Air).
    #[test]
    fn sky_tile_meshes_empty() {
        // h = 100; tile y=5 spans world Y 320..384, all above the surface.
        let tile = Tile { lod: Lod(2), x: 0, y: 5, z: 0 };
        assert!(build_tile_mesh(tile, &flat(100), &tables()).vertices().is_empty(), "sky is empty");
    }

    /// Buried tile (below surface) meshes empty; uniform shell hides interior faces.
    #[test]
    fn buried_tile_meshes_empty() {
        // h = 100; tile y=0 spans world Y 0..64, all deep (surface is at 99).
        let tile = Tile { lod: Lod(2), x: 0, y: 0, z: 0 };
        assert!(build_tile_mesh(tile, &flat(100), &tables()).vertices().is_empty(), "buried is empty");
    }
}
