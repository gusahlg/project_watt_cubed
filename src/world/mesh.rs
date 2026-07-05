//! Turns a chunk's voxels into triangle-mesh data ready for GPU upload.
//!
//! The original renderer issued one immediate-mode `draw_cube` per visible voxel
//! *every frame*. This module replaces all of that with a one-time **greedy
//! mesh** per chunk:
//!
//! - Only faces that border non-solid space are emitted; interior faces and
//!   faces shared between two solid voxels never exist, so there is no overdraw.
//! - Adjacent exposed faces of the same block are merged into maximal
//!   rectangles (per face direction), cutting vertex counts by an order of
//!   magnitude on rolling terrain. Merging is lossless because two faces merge
//!   only on identical [`BlockId`] (== texture layer, and shade is constant per
//!   direction), and UVs are the face plane's CHUNK-LOCAL coordinates: a quad
//!   spanning k blocks gets a uv extent of k, so REPEAT sampling tiles the
//!   16x16 block texture once per block across the merged span. Local UVs tile
//!   seamlessly *across* chunks too: the texture period is 1 uv unit and a
//!   chunk spans exactly 16 — a whole number of periods — so the pattern phase
//!   at a chunk's 16-edge equals the neighbour's 0-edge.
//! - Vertices are CHUNK-LOCAL (positions in 0..=16, exact in f32); the world
//!   places each chunk with a per-draw camera-relative offset
//!   (`Frame3D::draw_mesh(handle, offset)`), so far-from-origin chunks carry
//!   no giant world coordinates that would round in f32 and jitter. Mesh AABBs
//!   (computed by the engine from the vertices) are local for the same reason;
//!   culling tests them against the same offset.
//! - Per-face directional shading is baked into the vertex colour multiplier,
//!   which fakes cheap lighting without needing lit shaders (the engine is
//!   unlit); `color.a` carries the block-texture-array layer, i.e. the block id.
//! - Neighbour culling never touches the world's chunk map: the six bordering
//!   chunks are resolved once per build and everything else is flat-array reads.
//! - [`ChunkData::Uniform`] fast paths: a uniform non-solid chunk is empty
//!   without any scanning, and a uniform solid chunk only sweeps its six
//!   border slices (its interior can never expose a face) — so the deep-rock
//!   and sky bulk of an infinite-Y world meshes in effectively zero time.
//!
//! Building is pure CPU (`&Chunk` in, [`MeshData`] out) so it runs headless in
//! tests; the caller uploads the result via `Engine::upload_mesh`.
use voxel_engine::{MeshData, Vertex};

use super::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};
use crate::block::registry::{AIR, BlockId};

/// The six orthogonal neighbours of a chunk, prefetched by the caller so the
/// mesher can cull border faces without hashmap lookups. A missing neighbour
/// reads as air (in practice the world only meshes once all six have data).
pub struct Neighbours<'a> {
    pub neg_x: Option<&'a Chunk>,
    pub pos_x: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
    pub neg_y: Option<&'a Chunk>,
    pub pos_y: Option<&'a Chunk>,
}

/// A chunk border the sweep can read across. The discriminant doubles as the
/// plane index in [`BorderPlanes`].
#[derive(Clone, Copy)]
pub enum Side {
    NegX = 0,
    PosX = 1,
    NegZ = 2,
    PosZ = 3,
    NegY = 4,
    PosY = 5,
}

/// How the mesher reads the voxel just across a chunk border. Implemented for
/// [`Neighbours`] (sync path: borrow the six loaded chunks) and
/// [`BorderPlanes`] (worker path: owned copies of just the facing planes), so
/// both share the sweep and emit code in [`build_chunk_mesh_with`].
pub trait NeighbourRead {
    /// The block across `side` at in-plane coordinates `(u, v)`, given in the
    /// face direction's (U, V) axis order: (z, y) for the X sides, (x, y) for
    /// the Z sides, (x, z) for the Y sides. A missing neighbour reads as air.
    fn across(&self, side: Side, u: usize, v: usize) -> BlockId;
}

impl NeighbourRead for Neighbours<'_> {
    fn across(&self, side: Side, u: usize, v: usize) -> BlockId {
        const EDGE: usize = CHUNK_SIZE - 1;
        match side {
            Side::NegX => self.neg_x.map_or(AIR, |c| c.get_local(EDGE, v, u)),
            Side::PosX => self.pos_x.map_or(AIR, |c| c.get_local(0, v, u)),
            Side::NegZ => self.neg_z.map_or(AIR, |c| c.get_local(u, v, EDGE)),
            Side::PosZ => self.pos_z.map_or(AIR, |c| c.get_local(u, v, 0)),
            Side::NegY => self.neg_y.map_or(AIR, |c| c.get_local(u, EDGE, v)),
            Side::PosY => self.pos_y.map_or(AIR, |c| c.get_local(u, 0, v)),
        }
    }
}

/// Cells in one border plane: 16 x 16, indexed `[u + v * 16]`.
const PLANE_CELLS: usize = CHUNK_SIZE * CHUNK_SIZE;

/// The six neighbour facing planes copied out for a worker-thread mesh build:
/// each is [`PLANE_CELLS`] voxels in [`NeighbourRead::across`]'s `(u, v)`
/// order, `None` when the neighbour has no data (reads as air, like a missing
/// [`Neighbours`] entry). ~0.5 KiB per plane, owned, so a mesh job borrows
/// nothing from the live chunk map.
pub struct BorderPlanes {
    planes: [Option<Box<[BlockId]>>; 6],
}

impl BorderPlanes {
    /// Copy the facing plane out of each present neighbour: the neg-X
    /// neighbour's `x == 15` plane, the pos-X neighbour's `x == 0` plane, and
    /// likewise for Z and Y.
    pub fn capture(n: &Neighbours) -> Self {
        fn plane(
            chunk: Option<&Chunk>,
            read: impl Fn(&Chunk, usize, usize) -> BlockId,
        ) -> Option<Box<[BlockId]>> {
            let chunk = chunk?;
            let mut out = Vec::with_capacity(PLANE_CELLS);
            for v in 0..CHUNK_SIZE {
                for u in 0..CHUNK_SIZE {
                    out.push(read(chunk, u, v));
                }
            }
            Some(out.into_boxed_slice())
        }
        const EDGE: usize = CHUNK_SIZE - 1;
        Self {
            planes: [
                plane(n.neg_x, |c, u, v| c.get_local(EDGE, v, u)),
                plane(n.pos_x, |c, u, v| c.get_local(0, v, u)),
                plane(n.neg_z, |c, u, v| c.get_local(u, v, EDGE)),
                plane(n.pos_z, |c, u, v| c.get_local(u, v, 0)),
                plane(n.neg_y, |c, u, v| c.get_local(u, EDGE, v)),
                plane(n.pos_y, |c, u, v| c.get_local(u, 0, v)),
            ],
        }
    }
}

impl NeighbourRead for BorderPlanes {
    fn across(&self, side: Side, u: usize, v: usize) -> BlockId {
        self.planes[side as usize]
            .as_deref()
            .map_or(AIR, |plane| plane[u + v * CHUNK_SIZE])
    }
}

/// One face direction of the greedy sweep. Each direction slices the chunk
/// perpendicular to its normal and merges exposed faces in the slice plane,
/// whose axes are called U and V below.
struct Dir {
    /// +1 or -1: the step along the normal axis to the voxel a face borders.
    step: i32,
    /// World axis indices (0 = X, 1 = Y, 2 = Z) of the normal and the slice
    /// plane's U and V axes.
    n_axis: usize,
    u_axis: usize,
    v_axis: usize,
    /// The chunk border this direction's edge slice reads across.
    side: Side,
    /// Quad corners as (normal, u, v) components, each 0 or 1: the normal
    /// component picks the face plane, the U/V components are scaled by the
    /// merged rectangle's extents. Wound counter-clockwise seen from *outside*
    /// the block (matching the engine's backface culling), emitted as
    /// (0,1,2)+(0,2,3). These reproduce the pre-greedy mesher's corner order
    /// exactly, so winding and vertex layout are unchanged.
    corners: [[f32; 3]; 4],
    /// Brightness multiplier baked into the vertex colour: top brightest,
    /// bottom darkest, sides in between.
    shade: f32,
}

const DIRS: [Dir; 6] = [
    // +X
    Dir {
        step: 1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        side: Side::PosX,
        corners: [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        shade: 0.80,
    },
    // -X
    Dir {
        step: -1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        side: Side::NegX,
        corners: [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        shade: 0.70,
    },
    // +Y (top) — full brightness.
    Dir {
        step: 1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        side: Side::PosY,
        corners: [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        shade: 1.00,
    },
    // -Y (bottom) — darkest.
    Dir {
        step: -1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        side: Side::NegY,
        corners: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        shade: 0.50,
    },
    // +Z
    Dir {
        step: 1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        side: Side::PosZ,
        corners: [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        shade: 0.85,
    },
    // -Z
    Dir {
        step: -1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        side: Side::NegZ,
        corners: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        shade: 0.65,
    },
];

/// Flat-index delta for a one-voxel step along each world axis
/// (invariant: [`Chunk::index`] is `x + z*16 + y*256`).
const AXIS_STRIDE: [isize; 3] = [1, (CHUNK_SIZE * CHUNK_SIZE) as isize, CHUNK_SIZE as isize];

/// One slice of the sweep: 16 x 16 cells.
const MASK_CAP: usize = CHUNK_SIZE * CHUNK_SIZE;

/// How the sweep reads this chunk's own cells — monomorphized so the dense
/// path keeps its direct flat-array reads and the uniform path is a constant.
trait CellRead {
    fn get(&self, index: usize) -> BlockId;
}

struct DenseCells<'a>(&'a [u8; CHUNK_VOLUME]);
impl CellRead for DenseCells<'_> {
    #[inline(always)]
    fn get(&self, index: usize) -> BlockId {
        BlockId(self.0[index] as u16)
    }
}

struct UniformCells(BlockId);
impl CellRead for UniformCells {
    #[inline(always)]
    fn get(&self, _index: usize) -> BlockId {
        self.0
    }
}

/// Build one chunk's greedy mesh into `out` (cleared first — pass the world's
/// reusable scratch to avoid per-chunk allocations; `upload_mesh` copies out of
/// it). `solid` is the registry's hot solidity table snapshotted as a plain
/// slice indexed by [`BlockId`], so the per-voxel loops never leave L1. Colour
/// comes from the block texture array: `color.a` = block id = texture layer.
///
/// Positions are CHUNK-LOCAL (0..=16); the caller draws the mesh with a
/// camera-relative offset (see the module docs).
pub fn build_chunk_mesh(
    chunk: &Chunk,
    neighbours: &Neighbours,
    solid: &[bool],
    out: &mut MeshData,
) {
    build_chunk_mesh_with(chunk, neighbours, solid, out);
}

/// The generic core behind [`build_chunk_mesh`]: an identical sweep and emit
/// for the synchronous path (`&Neighbours`, borrowing live chunks) and the
/// worker path ([`BorderPlanes`], owning copies of just the facing planes) —
/// only the read across a chunk border differs, via [`NeighbourRead`].
pub fn build_chunk_mesh_with<N: NeighbourRead>(
    chunk: &Chunk,
    neighbours: &N,
    solid: &[bool],
    out: &mut MeshData,
) {
    out.clear();
    match chunk.data() {
        ChunkData::Uniform(id) => {
            if !solid[id.0 as usize] {
                return; // uniform air (or other non-solid): empty, no scan
            }
            // Uniform solid: interior faces are impossible, so only the six
            // border slices are swept. With six fully-solid neighbour planes
            // every mask comes up empty and the mesh stays empty.
            sweep(&UniformCells(*id), true, neighbours, solid, out);
        }
        ChunkData::Dense(cells) => {
            sweep(&DenseCells(cells), false, neighbours, solid, out);
        }
    }
}

/// The greedy sweep over all six directions. `edge_only` restricts each
/// direction to its border slice (the uniform-solid fast path); the emitted
/// geometry is identical to a full sweep because a uniform chunk's interior
/// slices can never contain an exposed face.
fn sweep<C: CellRead, N: NeighbourRead>(
    cells: &C,
    edge_only: bool,
    neighbours: &N,
    solid: &[bool],
    out: &mut MeshData,
) {
    let mut mask: [BlockId; MASK_CAP] = [AIR; MASK_CAP];

    for dir in &DIRS {
        let stride = AXIS_STRIDE[dir.n_axis] * dir.step as isize;
        // The slice whose neighbour test would step outside this chunk's
        // array; there the neighbour is read across the border.
        let edge_n = if dir.step > 0 { CHUNK_SIZE - 1 } else { 0 };

        for n in 0..CHUNK_SIZE {
            let at_edge = n == edge_n;
            if edge_only && !at_edge {
                continue;
            }

            // Phase 1: mask of exposed faces in this slice, keyed by BlockId.
            let mut any = false;
            for v in 0..CHUNK_SIZE {
                for u in 0..CHUNK_SIZE {
                    let mut c = [0usize; 3];
                    c[dir.n_axis] = n;
                    c[dir.u_axis] = u;
                    c[dir.v_axis] = v;
                    let idx = Chunk::index(c[0], c[1], c[2]);
                    let id = cells.get(idx);
                    let mut cell = AIR;
                    if solid[id.0 as usize] {
                        let covered = if at_edge {
                            let across = neighbours.across(dir.side, u, v);
                            solid[across.0 as usize]
                        } else {
                            solid[cells.get((idx as isize + stride) as usize).0 as usize]
                        };
                        if !covered {
                            cell = id;
                        }
                    }
                    mask[u + v * CHUNK_SIZE] = cell;
                    any |= cell != AIR;
                }
            }
            if !any {
                continue;
            }

            // Phase 2: greedy rectangles — grow along U while the run matches,
            // then along V while the whole row matches, clear, emit.
            for v0 in 0..CHUNK_SIZE {
                for u0 in 0..CHUNK_SIZE {
                    let id = mask[u0 + v0 * CHUNK_SIZE];
                    if id == AIR {
                        continue;
                    }
                    let mut w = 1;
                    while u0 + w < CHUNK_SIZE && mask[u0 + w + v0 * CHUNK_SIZE] == id {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while v0 + h < CHUNK_SIZE {
                        let row = (v0 + h) * CHUNK_SIZE;
                        for k in 0..w {
                            if mask[u0 + k + row] != id {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        let row = (v0 + dv) * CHUNK_SIZE;
                        mask[u0 + row..u0 + w + row].fill(AIR);
                    }
                    emit_rect(out, dir, n, u0, v0, w, h, id);
                }
            }
        }
    }
}

/// Append one merged rectangle: 4 vertices and 6 indices, corners scaled from
/// the direction's unit-quad table by the rectangle's U/V extents.
///
/// UV = the two varying CHUNK-LOCAL coordinates of the face's plane (+Y/-Y:
/// (x,z); +X/-X: (z,y); +Z/-Z: (x,y)) — exactly `(pos[u_axis], pos[v_axis])`
/// — so a rect spanning k blocks spans k uv units and REPEAT shows one
/// texture repetition per block. Local coords keep tiling seamless across
/// chunk borders because 16 is a whole number of texture periods (period =
/// 1 uv unit). Colour rgb = the direction's shade as a gray multiplier;
/// colour a = the block id, i.e. the texture-array layer.
#[allow(clippy::too_many_arguments)]
fn emit_rect(
    out: &mut MeshData,
    dir: &Dir,
    n: usize,
    u0: usize,
    v0: usize,
    w: usize,
    h: usize,
    id: BlockId,
) {
    debug_assert!(id.0 < 256, "block texture layers are u8 for now");
    let shade = (255.0 * dir.shade) as u8;

    let mut origin = [0.0f32; 3]; // chunk-local position of the rect's minimum block corner
    origin[dir.n_axis] = n as f32;
    origin[dir.u_axis] = u0 as f32;
    origin[dir.v_axis] = v0 as f32;

    let start = out.vertices.len() as u32;
    for corner in &dir.corners {
        let mut pos = [0.0f32; 3];
        pos[dir.n_axis] = origin[dir.n_axis] + corner[0];
        pos[dir.u_axis] = origin[dir.u_axis] + corner[1] * w as f32;
        pos[dir.v_axis] = origin[dir.v_axis] + corner[2] * h as f32;
        out.vertices.push(Vertex::textured(
            pos,
            [pos[dir.u_axis], pos[dir.v_axis]],
            [shade, shade, shade],
            id.0 as u8,
        ));
    }
    out.indices
        .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generation::TerrainGenerator;

    /// All-air generator so tests place voxels by hand.
    struct EmptyGen;
    impl TerrainGenerator for EmptyGen {
        fn height(&self, _wx: i32, _wz: i32) -> i32 {
            0
        }
        fn surface(&self) -> BlockId {
            AIR
        }
        fn subsoil(&self) -> BlockId {
            AIR
        }
        fn deep(&self) -> BlockId {
            AIR
        }
    }

    /// Everything-solid generator: chunks anywhere come out uniform STONE.
    struct SolidGen;
    impl TerrainGenerator for SolidGen {
        fn height(&self, _wx: i32, _wz: i32) -> i32 {
            i32::MAX
        }
        fn surface(&self) -> BlockId {
            STONE
        }
        fn subsoil(&self) -> BlockId {
            STONE
        }
        fn deep(&self) -> BlockId {
            STONE
        }
    }

    /// Two distinct solid test blocks (distinct ids = distinct texture layers).
    const STONE: BlockId = BlockId(1);
    const DIRT: BlockId = BlockId(2);

    fn solid_table() -> Vec<bool> {
        vec![false, true, true]
    }

    fn empty_chunk() -> Chunk {
        Chunk::new(0, 0, 0, &EmptyGen)
    }

    const NO_NEIGHBOURS: Neighbours = Neighbours {
        neg_x: None,
        pos_x: None,
        neg_z: None,
        pos_z: None,
        neg_y: None,
        pos_y: None,
    };

    fn build(chunk: &Chunk) -> MeshData {
        let solid = solid_table();
        let mut out = MeshData::default();
        build_chunk_mesh(chunk, &NO_NEIGHBOURS, &solid, &mut out);
        out
    }

    /// Reference: exposed-face count from a plain per-voxel culled sweep over
    /// the whole cube, using the same solidity rules as the mesher.
    /// Greedy merging must preserve total face area exactly.
    fn culled_face_area(chunk: &Chunk, solid: &[bool]) -> usize {
        let solid_at = |x: i32, y: i32, z: i32| -> bool {
            let range = 0..CHUNK_SIZE as i32;
            if !range.contains(&x) || !range.contains(&y) || !range.contains(&z) {
                return false; // no neighbours in these tests: outside is air
            }
            solid[chunk.get_local(x as usize, y as usize, z as usize).0 as usize]
        };
        let mut area = 0;
        for y in 0..CHUNK_SIZE as i32 {
            for z in 0..CHUNK_SIZE as i32 {
                for x in 0..CHUNK_SIZE as i32 {
                    if !solid_at(x, y, z) {
                        continue;
                    }
                    for (dx, dy, dz) in
                        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                    {
                        if !solid_at(x + dx, y + dy, z + dz) {
                            area += 1;
                        }
                    }
                }
            }
        }
        area
    }

    /// Area of each emitted quad. Quads are axis-aligned rectangles whose
    /// corners run around the perimeter, so the two edges from corner 0 span it.
    fn quad_areas(data: &MeshData) -> Vec<f32> {
        assert_eq!(data.vertices.len() % 4, 0, "quads are 4 vertices each");
        assert_eq!(data.indices.len(), data.vertices.len() / 4 * 6);
        data.vertices
            .chunks_exact(4)
            .map(|q| {
                let e = |a: &Vertex, b: &Vertex| {
                    let d = [b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2]];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                };
                e(&q[0], &q[1]) * e(&q[0], &q[3])
            })
            .collect()
    }

    fn total_area(data: &MeshData) -> f32 {
        quad_areas(data).iter().sum()
    }

    /// Quads facing +Y sit entirely in one horizontal plane; side faces always
    /// span two Y levels, so "all four corners at the same Y" identifies tops
    /// (and bottoms, which the given plane's Y filters out).
    fn quads_in_y_plane(data: &MeshData, y: f32) -> usize {
        data.vertices
            .chunks_exact(4)
            .filter(|q| q.iter().all(|v| v.pos[1] == y))
            .count()
    }

    #[test]
    fn flat_slab_top_merges_to_one_quad() {
        let mut chunk = empty_chunk();
        for x in 0..3 {
            for z in 0..3 {
                chunk.set_local(x, 0, z, STONE);
            }
        }
        let data = build(&chunk);
        let solid = solid_table();

        // One merged 3x3 top face at y=1 — 4 vertices / 6 indices for that direction.
        assert_eq!(quads_in_y_plane(&data, 1.0), 1, "top of the slab is one quad");

        // Merging must not create or drop area: 9 top + 9 bottom + 12 side faces.
        let reference = culled_face_area(&chunk, &solid);
        assert_eq!(reference, 30);
        assert_eq!(total_area(&data), reference as f32);

        // The merged top quad carries full-brightness shade, STONE's layer,
        // and local-coordinate uvs spanning the full 3-block extent so REPEAT
        // tiles the texture once per block.
        let top = data
            .vertices
            .chunks_exact(4)
            .find(|q| q.iter().all(|v| v.pos[1] == 1.0))
            .expect("top quad exists");
        for v in top {
            assert_eq!(v.color, [255, 255, 255, STONE.0 as u8], "(shade, layer)");
            assert_eq!(v.uv, [v.pos[0], v.pos[2]], "+Y uv = world (x, z)");
        }
        let span = |axis: usize| {
            let lo = top.iter().map(|v| v.uv[axis]).fold(f32::INFINITY, f32::min);
            let hi = top.iter().map(|v| v.uv[axis]).fold(f32::NEG_INFINITY, f32::max);
            hi - lo
        };
        assert_eq!((span(0), span(1)), (3.0, 3.0), "uv extent == blocks spanned");
    }

    #[test]
    fn checkerboard_never_merges() {
        let mut chunk = empty_chunk();
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if (x + z) % 2 == 0 {
                    chunk.set_local(x, 0, z, STONE);
                }
            }
        }
        let data = build(&chunk);
        let solid = solid_table();

        // Every face is isolated, so quad count equals face area exactly.
        let areas = quad_areas(&data);
        let reference = culled_face_area(&chunk, &solid);
        assert_eq!(areas.len(), reference, "no two faces merged");
        assert!(areas.iter().all(|&a| a == 1.0), "every quad is 1x1");
    }

    #[test]
    fn different_blocks_do_not_merge() {
        let mut chunk = empty_chunk();
        chunk.set_local(0, 0, 0, STONE);
        chunk.set_local(1, 0, 0, DIRT);
        let data = build(&chunk);
        let solid = solid_table();

        // Adjacent tops of different blocks stay two quads.
        assert_eq!(quads_in_y_plane(&data, 1.0), 2, "different blocks never merge");
        // The shared vertical face is culled on both sides: 5 exposed faces each.
        let reference = culled_face_area(&chunk, &solid);
        assert_eq!(reference, 10);
        assert_eq!(total_area(&data), reference as f32);

        // Same shade on both tops; only the texture layer distinguishes them.
        let tops: Vec<_> = data
            .vertices
            .chunks_exact(4)
            .filter(|q| q.iter().all(|v| v.pos[1] == 1.0))
            .collect();
        let mut layers: Vec<u8> = tops.iter().map(|q| q[0].color[3]).collect();
        layers.sort_unstable();
        assert_eq!(layers, vec![STONE.0 as u8, DIRT.0 as u8]);
        assert!(
            tops.iter().all(|q| q.iter().all(|v| v.color[..3] == [255, 255, 255])),
            "top shade is identical across blocks"
        );
    }

    #[test]
    fn uniform_air_meshes_empty() {
        let chunk = empty_chunk();
        assert_eq!(chunk.uniform(), Some(AIR), "generated sky chunk is uniform");
        let data = build(&chunk);
        assert!(data.vertices.is_empty() && data.indices.is_empty());
    }

    #[test]
    fn uniform_solid_fast_path_matches_a_dense_fill() {
        // The edge-slice-only sweep is purely a shortcut: a uniform stone cube
        // and a dense chunk holding identical cells must mesh byte-identically.
        let uniform = Chunk::new(0, 0, 0, &SolidGen);
        assert_eq!(uniform.uniform(), Some(STONE));

        let mut dense = Chunk::new(0, 0, 0, &SolidGen);
        dense.set_local(0, 0, 0, DIRT); // promote...
        dense.set_local(0, 0, 0, STONE); // ...and restore the same cells
        assert!(dense.uniform().is_none(), "promotion kept dense storage");

        let (a, b) = (build(&uniform), build(&dense));
        assert_eq!(total_area(&a), (6 * CHUNK_SIZE * CHUNK_SIZE) as f32, "6 full faces");
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.vertices.len(), b.vertices.len());
        for (va, vb) in a.vertices.iter().zip(b.vertices.iter()) {
            assert_eq!((va.pos, va.uv, va.color), (vb.pos, vb.uv, vb.color));
        }
    }

    #[test]
    fn uniform_solid_boxed_in_by_solid_neighbours_meshes_empty() {
        let chunk = Chunk::new(0, 0, 0, &SolidGen);
        let nx = Chunk::new(-1, 0, 0, &SolidGen);
        let px = Chunk::new(1, 0, 0, &SolidGen);
        let nz = Chunk::new(0, 0, -1, &SolidGen);
        let pz = Chunk::new(0, 0, 1, &SolidGen);
        let ny = Chunk::new(0, -1, 0, &SolidGen);
        let py = Chunk::new(0, 1, 0, &SolidGen);
        let neighbours = Neighbours {
            neg_x: Some(&nx),
            pos_x: Some(&px),
            neg_z: Some(&nz),
            pos_z: Some(&pz),
            neg_y: Some(&ny),
            pos_y: Some(&py),
        };
        let solid = solid_table();
        let mut out = MeshData::default();
        build_chunk_mesh(&chunk, &neighbours, &solid, &mut out);
        assert!(out.vertices.is_empty(), "deep rock boxed in by rock draws nothing");
    }

    #[test]
    fn face_uvs_are_the_planes_local_coords_per_direction() {
        // A single cube in a chunk FAR from the origin: vertices must be
        // chunk-local (0..=16 exactly — the far world coordinate never touches
        // the f32 mesh), and every face's uv must equal the two varying local
        // coordinates of its plane. Faces are identified by their baked shade
        // byte, which is unique per direction.
        let mut chunk = Chunk::new(6_250_000, 40, -6_250_000, &EmptyGen);
        chunk.set_local(2, 3, 4, STONE);
        let data = build(&chunk);
        assert_eq!(data.vertices.len(), 24, "six 1x1 faces");

        for q in data.vertices.chunks_exact(4) {
            let shade = q[0].color[0];
            for v in q {
                assert!(
                    v.pos.iter().all(|&c| (0.0..=CHUNK_SIZE as f32).contains(&c)),
                    "vertices are chunk-local, got {:?}",
                    v.pos
                );
                assert_eq!(v.color[3], STONE.0 as u8, "layer = block id");
                assert_eq!([v.color[0], v.color[1], v.color[2]], [shade; 3], "gray shade");
                match shade {
                    255 | 127 => assert_eq!(v.uv, [v.pos[0], v.pos[2]], "+Y/-Y: (x, z)"),
                    204 | 178 => assert_eq!(v.uv, [v.pos[2], v.pos[1]], "+X/-X: (z, y)"),
                    216 | 165 => assert_eq!(v.uv, [v.pos[0], v.pos[1]], "+Z/-Z: (x, y)"),
                    other => panic!("unexpected shade byte {other}"),
                }
            }
        }
        // The cube sits at local (2, 3, 4) regardless of the chunk coordinate.
        let min = |axis: usize| {
            data.vertices.iter().map(|v| v.pos[axis]).fold(f32::INFINITY, f32::min)
        };
        assert_eq!((min(0), min(1), min(2)), (2.0, 3.0, 4.0));
    }

    #[test]
    fn far_chunk_meshes_byte_identically_to_the_origin_chunk() {
        // Chunk-local emission means the mesh is a pure function of contents —
        // the chunk coordinate must not leak into a single float. This is what
        // makes far terrain render exactly (the offset is applied per draw).
        let mut near = Chunk::new(0, 0, 0, &EmptyGen);
        let mut far = Chunk::new(62_500_000, -3_000, -62_500_000, &EmptyGen);
        for (x, y, z, id) in [(0, 0, 0, STONE), (1, 0, 0, STONE), (5, 9, 15, DIRT)] {
            near.set_local(x, y, z, id);
            far.set_local(x, y, z, id);
        }
        let (a, b) = (build(&near), build(&far));
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.vertices.len(), b.vertices.len());
        for (va, vb) in a.vertices.iter().zip(b.vertices.iter()) {
            assert_eq!((va.pos, va.uv, va.color), (vb.pos, vb.uv, vb.color));
        }
    }

    #[test]
    fn border_faces_cull_against_neighbour_chunks() {
        // A block on this chunk's +X border, hidden by a block on the
        // neighbour's -X border: the shared face must vanish only when the
        // neighbour is supplied.
        let mut chunk = empty_chunk();
        chunk.set_local(CHUNK_SIZE - 1, 0, 0, STONE);
        let mut other = Chunk::new(1, 0, 0, &EmptyGen);
        other.set_local(0, 0, 0, STONE);

        let solid = solid_table();
        let mut alone = MeshData::default();
        build_chunk_mesh(&chunk, &NO_NEIGHBOURS, &solid, &mut alone);
        let with_neighbour = Neighbours { pos_x: Some(&other), ..NO_NEIGHBOURS };
        let mut culled = MeshData::default();
        build_chunk_mesh(&chunk, &with_neighbour, &solid, &mut culled);

        assert_eq!(total_area(&alone), 6.0, "isolated cube shows all six faces");
        assert_eq!(total_area(&culled), 5.0, "the face against the neighbour is culled");
    }

    #[test]
    fn vertical_border_faces_cull_against_the_chunk_above() {
        // Cube chunks join in Y too: a block on the top border, hidden by a
        // block at the bottom of the chunk above.
        let mut chunk = empty_chunk();
        chunk.set_local(4, CHUNK_SIZE - 1, 4, STONE);
        let mut above = Chunk::new(0, 1, 0, &EmptyGen);
        above.set_local(4, 0, 4, STONE);

        let solid = solid_table();
        let mut alone = MeshData::default();
        build_chunk_mesh(&chunk, &NO_NEIGHBOURS, &solid, &mut alone);
        let with_above = Neighbours { pos_y: Some(&above), ..NO_NEIGHBOURS };
        let mut culled = MeshData::default();
        build_chunk_mesh(&chunk, &with_above, &solid, &mut culled);

        assert_eq!(total_area(&alone), 6.0);
        assert_eq!(total_area(&culled), 5.0, "the top face is culled by the chunk above");
    }
}
