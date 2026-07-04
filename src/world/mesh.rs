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
//!   direction), and UVs are the face plane's world coordinates: a quad
//!   spanning k blocks gets a uv extent of k, so REPEAT sampling tiles the
//!   16x16 block texture once per block across the merged span.
//! - Per-face directional shading is baked into the vertex colour multiplier,
//!   which fakes cheap lighting without needing lit shaders (the engine is
//!   unlit); `color.a` carries the block-texture-array layer, i.e. the block id.
//! - Neighbour culling never touches the world's chunk map: the four bordering
//!   chunks are resolved once per build and everything else is flat-array reads.
//!
//! Building is pure CPU (`&Chunk` in, [`MeshData`] out) so it runs headless in
//! tests; the caller uploads the result via `Engine::upload_mesh`.
use voxel_engine::{MeshData, Vertex};

use super::chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use crate::block::registry::{AIR, BlockId};

/// The four orthogonal neighbours of a chunk, prefetched by the caller so the
/// mesher can cull border faces without hashmap lookups. A missing neighbour
/// reads as air (in practice the world only meshes once all four have data).
pub struct Neighbours<'a> {
    pub neg_x: Option<&'a Chunk>,
    pub pos_x: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
}

/// A chunk border the sweep can read across. The discriminant doubles as the
/// plane index in [`BorderPlanes`].
#[derive(Clone, Copy)]
pub enum Side {
    NegX = 0,
    PosX = 1,
    NegZ = 2,
    PosZ = 3,
}

/// How the mesher reads the voxel just across a chunk border. Implemented for
/// [`Neighbours`] (sync path: borrow the four loaded chunks) and
/// [`BorderPlanes`] (worker path: owned copies of just the facing planes), so
/// both share the sweep and emit code in [`build_chunk_mesh_with`].
pub trait NeighbourRead {
    /// The block across `side` at height `y` and in-plane coordinate `u`
    /// (Z for the X sides, X for the Z sides). A missing neighbour reads as air.
    fn across(&self, side: Side, y: usize, u: usize) -> BlockId;
}

impl NeighbourRead for Neighbours<'_> {
    fn across(&self, side: Side, y: usize, u: usize) -> BlockId {
        match side {
            Side::NegX => self.neg_x.map_or(AIR, |c| c.get_local(CHUNK_WIDTH - 1, y, u)),
            Side::PosX => self.pos_x.map_or(AIR, |c| c.get_local(0, y, u)),
            Side::NegZ => self.neg_z.map_or(AIR, |c| c.get_local(u, y, CHUNK_DEPTH - 1)),
            Side::PosZ => self.pos_z.map_or(AIR, |c| c.get_local(u, y, 0)),
        }
    }
}

/// Width of a border plane's in-plane horizontal axis. Both horizontal axes
/// are 16, so one constant serves the X and Z sides alike.
const PLANE_W: usize = CHUNK_WIDTH;
const _: () = assert!(CHUNK_WIDTH == CHUNK_DEPTH, "border planes assume square chunks");

/// The four neighbour facing planes copied out for a worker-thread mesh build:
/// each is [`PLANE_W`] x [`CHUNK_HEIGHT`] voxels indexed `[u + y * PLANE_W]`,
/// `None` when the neighbour has no data (reads as air, like a missing
/// [`Neighbours`] entry). ~1 KiB per plane, owned, so a mesh job borrows
/// nothing from the live chunk map.
pub struct BorderPlanes {
    planes: [Option<Box<[BlockId]>>; 4],
}

impl BorderPlanes {
    /// Copy the facing plane out of each present neighbour: the neg-X
    /// neighbour's `x == 15` plane, the pos-X neighbour's `x == 0` plane, and
    /// likewise for Z.
    pub fn capture(n: &Neighbours) -> Self {
        fn plane(
            chunk: Option<&Chunk>,
            read: impl Fn(&Chunk, usize, usize) -> BlockId,
        ) -> Option<Box<[BlockId]>> {
            let chunk = chunk?;
            let mut out = Vec::with_capacity(PLANE_W * CHUNK_HEIGHT);
            for y in 0..CHUNK_HEIGHT {
                for u in 0..PLANE_W {
                    out.push(read(chunk, y, u));
                }
            }
            Some(out.into_boxed_slice())
        }
        Self {
            planes: [
                plane(n.neg_x, |c, y, z| c.get_local(CHUNK_WIDTH - 1, y, z)),
                plane(n.pos_x, |c, y, z| c.get_local(0, y, z)),
                plane(n.neg_z, |c, y, x| c.get_local(x, y, CHUNK_DEPTH - 1)),
                plane(n.pos_z, |c, y, x| c.get_local(x, y, 0)),
            ],
        }
    }
}

impl NeighbourRead for BorderPlanes {
    fn across(&self, side: Side, y: usize, u: usize) -> BlockId {
        self.planes[side as usize]
            .as_deref()
            .map_or(AIR, |plane| plane[u + y * PLANE_W])
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
        corners: [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        shade: 0.80,
    },
    // -X
    Dir {
        step: -1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        shade: 0.70,
    },
    // +Y (top) — full brightness.
    Dir {
        step: 1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        shade: 1.00,
    },
    // -Y (bottom) — darkest.
    Dir {
        step: -1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        shade: 0.50,
    },
    // +Z
    Dir {
        step: 1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        shade: 0.85,
    },
    // -Z
    Dir {
        step: -1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        shade: 0.65,
    },
];

/// Physical voxel-array extent along each world axis, for bounds checks.
const AXIS_MAX: [usize; 3] = [CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH];

/// Flat-index delta for a one-voxel step along each world axis
/// (invariant: `Chunk::index` is `x + z*WIDTH + y*WIDTH*DEPTH`).
const AXIS_STRIDE: [isize; 3] = [
    1,
    (CHUNK_WIDTH * CHUNK_DEPTH) as isize,
    CHUNK_WIDTH as isize,
];

/// Largest slice the sweep ever scans: 16 wide by 64 tall (X/Z directions).
const MASK_CAP: usize =
    (if CHUNK_WIDTH > CHUNK_DEPTH { CHUNK_WIDTH } else { CHUNK_DEPTH }) * CHUNK_HEIGHT;

/// Build one chunk's greedy mesh into `out` (cleared first — pass the world's
/// reusable scratch to avoid per-chunk allocations; `upload_mesh` copies out of
/// it). `solid` is the registry's hot solidity table snapshotted as a plain
/// slice indexed by [`BlockId`], so the per-voxel loops never leave L1. Colour
/// comes from the block texture array: `color.a` = block id = texture layer.
///
/// World-space positions are baked straight into the vertices, so every
/// chunk's mesh is drawn at the origin.
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
    let max_y = chunk.max_solid_y();
    if max_y < 0 {
        return; // all air
    }
    // Invariant: every voxel above `max_y` is air, so no direction can expose a
    // face there — Y-spanning loops stop at `y_count` instead of CHUNK_HEIGHT.
    let y_count = max_y as usize + 1;
    let voxels = chunk.voxels();
    let base = [chunk.cx * CHUNK_WIDTH as i32, 0, chunk.cz * CHUNK_DEPTH as i32];

    let mut mask: [BlockId; MASK_CAP] = [AIR; MASK_CAP];

    for dir in &DIRS {
        // Slice geometry: N is the normal axis; the mask covers the U x V plane.
        let (n_count, u_count, v_count) = match dir.n_axis {
            0 => (CHUNK_WIDTH, CHUNK_DEPTH, y_count),
            1 => (y_count, CHUNK_WIDTH, CHUNK_DEPTH),
            _ => (CHUNK_DEPTH, CHUNK_WIDTH, y_count),
        };
        let stride = AXIS_STRIDE[dir.n_axis] * dir.step as isize;
        // The slice whose neighbour test would step outside this chunk's array;
        // there the neighbour is read across the border via `NeighbourRead`
        // (X/Z) or is the world floor/ceiling (Y), which always reads as air.
        let edge_n = if dir.step > 0 { AXIS_MAX[dir.n_axis] - 1 } else { 0 };
        let edge_side = match (dir.n_axis, dir.step > 0) {
            (0, true) => Some(Side::PosX),
            (0, false) => Some(Side::NegX),
            (2, true) => Some(Side::PosZ),
            (2, false) => Some(Side::NegZ),
            _ => None,
        };

        for n in 0..n_count {
            let at_edge = n == edge_n;

            // Phase 1: mask of exposed faces in this slice, keyed by BlockId.
            let mut any = false;
            for v in 0..v_count {
                for u in 0..u_count {
                    let mut c = [0usize; 3];
                    c[dir.n_axis] = n;
                    c[dir.u_axis] = u;
                    c[dir.v_axis] = v;
                    let idx = Chunk::index(c[0], c[1], c[2]);
                    let id = voxels[idx];
                    let mut cell = AIR;
                    if solid[id.0 as usize] {
                        let covered = if at_edge {
                            match edge_side {
                                // For X/Z sides the plane coords are V (always
                                // Y here) and U (the other horizontal axis),
                                // matching `NeighbourRead::across`.
                                Some(side) => {
                                    let id =
                                        neighbours.across(side, c[dir.v_axis], c[dir.u_axis]);
                                    solid[id.0 as usize]
                                }
                                None => false, // beyond the world: air
                            }
                        } else {
                            solid[voxels[(idx as isize + stride) as usize].0 as usize]
                        };
                        if !covered {
                            cell = id;
                        }
                    }
                    mask[u + v * u_count] = cell;
                    any |= cell != AIR;
                }
            }
            if !any {
                continue;
            }

            // Phase 2: greedy rectangles — grow along U while the run matches,
            // then along V while the whole row matches, clear, emit.
            for v0 in 0..v_count {
                for u0 in 0..u_count {
                    let id = mask[u0 + v0 * u_count];
                    if id == AIR {
                        continue;
                    }
                    let mut w = 1;
                    while u0 + w < u_count && mask[u0 + w + v0 * u_count] == id {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while v0 + h < v_count {
                        let row = (v0 + h) * u_count;
                        for k in 0..w {
                            if mask[u0 + k + row] != id {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        let row = (v0 + dv) * u_count;
                        mask[u0 + row..u0 + w + row].fill(AIR);
                    }
                    emit_rect(out, dir, base, n, u0, v0, w, h, id);
                }
            }
        }
    }
}

/// Append one merged rectangle: 4 vertices and 6 indices, corners scaled from
/// the direction's unit-quad table by the rectangle's U/V extents.
///
/// UV = the two varying world coordinates of the face's plane (+Y/-Y: (x,z);
/// +X/-X: (z,y); +Z/-Z: (x,y)) — exactly `(pos[u_axis], pos[v_axis])` — so a
/// rect spanning k blocks spans k uv units and REPEAT shows one texture
/// repetition per block. Colour rgb = the direction's shade as a gray
/// multiplier; colour a = the block id, i.e. the texture-array layer.
#[allow(clippy::too_many_arguments)]
fn emit_rect(
    out: &mut MeshData,
    dir: &Dir,
    base: [i32; 3],
    n: usize,
    u0: usize,
    v0: usize,
    w: usize,
    h: usize,
    id: BlockId,
) {
    debug_assert!(id.0 < 256, "block texture layers are u8 for now");
    let shade = (255.0 * dir.shade) as u8;

    let mut origin = [0.0f32; 3]; // world position of the rect's minimum block corner
    origin[dir.n_axis] = (base[dir.n_axis] + n as i32) as f32;
    origin[dir.u_axis] = (base[dir.u_axis] + u0 as i32) as f32;
    origin[dir.v_axis] = (base[dir.v_axis] + v0 as i32) as f32;

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

    /// Two distinct solid test blocks (distinct ids = distinct texture layers).
    const STONE: BlockId = BlockId(1);
    const DIRT: BlockId = BlockId(2);

    fn solid_table() -> Vec<bool> {
        vec![false, true, true]
    }

    fn empty_chunk() -> Chunk {
        Chunk::new(0, 0, &EmptyGen)
    }

    const NO_NEIGHBOURS: Neighbours = Neighbours {
        neg_x: None,
        pos_x: None,
        neg_z: None,
        pos_z: None,
    };

    fn build(chunk: &Chunk) -> MeshData {
        let solid = solid_table();
        let mut out = MeshData::default();
        build_chunk_mesh(chunk, &NO_NEIGHBOURS, &solid, &mut out);
        out
    }

    /// Reference: exposed-face count from a plain per-voxel culled sweep over
    /// the full chunk height, using the same solidity rules as the mesher.
    /// Greedy merging must preserve total face area exactly.
    fn culled_face_area(chunk: &Chunk, solid: &[bool]) -> usize {
        let solid_at = |x: i32, y: i32, z: i32| -> bool {
            if x < 0 || x >= CHUNK_WIDTH as i32 || y < 0 || y >= CHUNK_HEIGHT as i32 || z < 0
                || z >= CHUNK_DEPTH as i32
            {
                return false; // no neighbours in these tests: outside is air
            }
            solid[chunk.get_local(x as usize, y as usize, z as usize).0 as usize]
        };
        let mut area = 0;
        for y in 0..CHUNK_HEIGHT as i32 {
            for z in 0..CHUNK_DEPTH as i32 {
                for x in 0..CHUNK_WIDTH as i32 {
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
        // and world-coordinate uvs spanning the full 3-block extent so REPEAT
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
        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_DEPTH {
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
    fn height_clamp_stops_at_max_solid_y_and_changes_nothing() {
        let build_terraced = |chunk: &mut Chunk| {
            // Terraced terrain topping out at y = 16 + 3 = 19.
            for x in 0..4 {
                for z in 0..4 {
                    for y in 0..=(16 + x) {
                        chunk.set_local(x, y, z, STONE);
                    }
                }
            }
        };
        let mut clamped = empty_chunk();
        build_terraced(&mut clamped);
        assert_eq!(clamped.max_solid_y(), 19);
        let data = build(&clamped);

        // Nothing above the top face plane of the highest block.
        let top = clamped.max_solid_y() as f32 + 1.0;
        assert!(
            data.vertices.iter().all(|v| v.pos[1] <= top),
            "no geometry above max_solid_y + 1"
        );

        // A stale-high max_solid_y (place a block at the ceiling, remove it)
        // must produce byte-identical output — the clamp is purely a shortcut.
        let mut stale = empty_chunk();
        build_terraced(&mut stale);
        stale.set_local(0, CHUNK_HEIGHT - 1, 0, STONE);
        stale.set_local(0, CHUNK_HEIGHT - 1, 0, AIR);
        assert_eq!(stale.max_solid_y(), CHUNK_HEIGHT as i32 - 1, "stale-high kept");
        let unclamped = build(&stale);

        assert_eq!(data.indices, unclamped.indices);
        assert_eq!(data.vertices.len(), unclamped.vertices.len());
        for (a, b) in data.vertices.iter().zip(unclamped.vertices.iter()) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.uv, b.uv);
            assert_eq!(a.color, b.color);
        }
    }

    #[test]
    fn face_uvs_are_the_planes_world_coords_per_direction() {
        // A single cube away from the origin: every face's uv must equal the
        // two varying world coordinates of its plane. Faces are identified by
        // their baked shade byte, which is unique per direction.
        let mut chunk = empty_chunk();
        chunk.set_local(2, 3, 4, STONE);
        let data = build(&chunk);
        assert_eq!(data.vertices.len(), 24, "six 1x1 faces");

        for q in data.vertices.chunks_exact(4) {
            let shade = q[0].color[0];
            for v in q {
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
    }

    #[test]
    fn border_faces_cull_against_neighbour_chunks() {
        // A block on this chunk's +X border, hidden by a block on the
        // neighbour's -X border: the shared face must vanish only when the
        // neighbour is supplied.
        let mut chunk = empty_chunk();
        chunk.set_local(CHUNK_WIDTH - 1, 0, 0, STONE);
        let mut other = Chunk::new(1, 0, &EmptyGen);
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
}
