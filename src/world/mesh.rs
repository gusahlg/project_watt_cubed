//! Turns a chunk's voxels into triangle-mesh data ready for GPU upload.
//!
//! A one-time **greedy mesh** per chunk:
//!
//! - The sweep reads a [`Padded`] neighbourhood — the chunk's 16³ voxels plus a
//!   one-voxel shell copied from its 26 neighbours — so face culling, the AO
//!   stencil, and the own-cell reads are ALL a single uniform `Padded::at`, with
//!   no interior/border branch and no per-direction axis remapping. The shell is
//!   what makes ambient occlusion seamless across chunk borders.
//! - Only faces bordering see-through space are emitted. The cull key is
//!   **opacity**, not solidity ([`covered`]): an opaque neighbour hides a face; a
//!   translucent one (glass) does not, and two of the same translucent block hide
//!   their shared internal face.
//! - Each exposed face routes to the **opaque** or **transparent** [`MeshData`]
//!   by its block's opacity, so a chunk yields up to two meshes ([`ChunkMeshData`]).
//! - Adjacent faces merge into maximal rectangles keyed on the whole
//!   [`FaceSample`] (id + per-corner AO + per-corner light), so a gradient never
//!   merges into a flat quad.
//! - Vertices are CHUNK-LOCAL (0..=16, exact in f32); the world draws each with a
//!   camera-relative offset, so far terrain never jitters.
//! - Uniform fast paths: a uniform non-solid chunk is empty; a uniform solid one
//!   only sweeps its six border slices.
use voxel_engine::{Ao, Light, MeshData, MeshVertex, Normal, Pass};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::light::PaddedLight;
use crate::block::registry::{AIR, BlockId, HotTables};
use crate::coord::ByPass;

/// A chunk's greedy mesh, split by draw pass: the CPU dual of the resident
/// `ChunkMeshes`. Either [`MeshData`] may be empty; the caller uploads only
/// non-empty passes.
pub type ChunkMeshData = ByPass<MeshData>;

/// A fresh, empty [`ChunkMeshData`] with each slot tagged with its pass; also the
/// shape of the world's reusable mesh scratch.
pub fn new_chunk_mesh_data() -> ChunkMeshData {
    ByPass::from_fn(MeshData::new)
}

/// Chunk size as a signed coordinate, for the `-1..=16` padded range.
const CS: i32 = CHUNK_SIZE as i32;
/// Padded neighbourhood edge: the 16 chunk cells plus one shell voxel each side.
const PAD: usize = CHUNK_SIZE + 2;

/// The chunk's 16³ voxels plus a one-voxel shell pulled from its 26 neighbours,
/// indexed by signed coords `x, y, z ∈ -1..=16`. Owned, so a mesh job shares
/// nothing with the live chunk map, and captured on the main thread where the
/// neighbourhood is resolvable. A missing neighbour reads as [`AIR`].
///
/// This one structure serves every neighbour read the sweep makes — cull (the
/// immediate outward cell), AO (the 3 occluders around each face corner, which
/// at a chunk edge fall into a neighbour's *interior* layer), and the own-cell
/// scan — so there is no interior/border special-casing anywhere.
pub struct Padded {
    ids: Box<[u8]>, // PAD*PAD*PAD, raw BlockId bytes
}

impl Padded {
    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        (x + 1) as usize + (z + 1) as usize * PAD + (y + 1) as usize * PAD * PAD
    }

    /// The block at signed coord `(x, y, z)`, each `∈ -1..=16`. Shared with the
    /// light pass, which reads the same padded interior.
    #[inline]
    pub(in crate::world) fn at(&self, x: i32, y: i32, z: i32) -> BlockId {
        BlockId(self.ids[Self::index(x, y, z)])
    }

    /// Copy the chunk and its shell out of the map. `chunk_at(dx, dy, dz)` yields
    /// the chunk at chunk-offset `(dx, dy, dz)` with each component in `-1..=1`
    /// (`(0,0,0)` is the chunk itself), or `None` (→ air). Resolves the 27 chunks
    /// once, then fills 18³ cells with plain array reads.
    pub fn capture<'a>(chunk_at: impl Fn(i32, i32, i32) -> Option<&'a Chunk>) -> Self {
        let neigh: [Option<&Chunk>; 27] =
            std::array::from_fn(|k| chunk_at(k as i32 % 3 - 1, k as i32 / 9 - 1, k as i32 / 3 % 3 - 1));
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
        let mut ids = vec![AIR.0; PAD * PAD * PAD];
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    let (dx, lx) = split(x);
                    let (dy, ly) = split(y);
                    let (dz, lz) = split(z);
                    if let Some(c) = get(dx, dy, dz) {
                        ids[Self::index(x, y, z)] = c.get_local(lx, ly, lz).0;
                    }
                }
            }
        }
        Self { ids: ids.into_boxed_slice() }
    }

    /// Build a padded neighbourhood by sampling a per-cell function over the whole
    /// `-1..=16` range, including the shell. Used by the far LOD tile mesher, whose
    /// "neighbours" are more coarse cells of the same pure generator — so the shell
    /// is sampled directly (no neighbour-tile handshake), which makes a fully-buried
    /// coarse tile mesh to *nothing* (its solid shell hides every interior face)
    /// exactly as a buried chunk does.
    pub fn from_cells(cell: impl Fn(i32, i32, i32) -> BlockId) -> Self {
        let mut ids = vec![AIR.0; PAD * PAD * PAD];
        for y in -1..=CS {
            for z in -1..=CS {
                for x in -1..=CS {
                    ids[Self::index(x, y, z)] = cell(x, y, z).0;
                }
            }
        }
        Self { ids: ids.into_boxed_slice() }
    }
}

/// Opaque neighbour or same block hides a face (two glass blocks share a hidden internal face).
#[inline]
fn covered(my: BlockId, nbr: BlockId, tables: &HotTables) -> bool {
    tables.opaque[nbr.0 as usize] || nbr == my
}

/// Fully dark if both edge occluders, else `3 - (count)`. Shared with LOD tile mesher.
#[inline]
pub(in crate::world) fn corner_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - (side1 as u8 + side2 as u8 + corner as u8)
    }
}

/// One face direction of the greedy sweep.
struct Dir {
    /// +1 / -1 step along the normal axis to the cell a face borders.
    step: i32,
    /// World axis indices (0=X,1=Y,2=Z) of the normal and the slice U/V axes.
    n_axis: usize,
    u_axis: usize,
    v_axis: usize,
    /// Quad corners as (normal, u, v) components (0/1); u/v scaled by the merged
    /// rectangle's extents. CCW seen from outside, matching engine backface cull.
    corners: [[f32; 3]; 4],
    normal: Normal,
}

const DIRS: [Dir; 6] = [
    Dir {
        step: 1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        normal: Normal::PosX,
    },
    Dir {
        step: -1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        normal: Normal::NegX,
    },
    Dir {
        step: 1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        normal: Normal::PosY,
    },
    Dir {
        step: -1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        normal: Normal::NegY,
    },
    Dir {
        step: 1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        normal: Normal::PosZ,
    },
    Dir {
        step: -1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        normal: Normal::NegZ,
    },
];

/// One slice of the sweep: 16 x 16 cells.
const MASK_CAP: usize = CHUNK_SIZE * CHUNK_SIZE;

/// The unified greedy-merge key (C-1). Two faces merge only when the whole sample
/// matches — id (⇒ texture layer and pass), per-corner AO, and per-corner light —
/// so an AO or light gradient never merges into a flat quad. `PartialEq` *is* the
/// merge rule.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceSample {
    id: BlockId,
    ao: [u8; 4],
    sky: [u8; 4],
    block: [u8; 4],
}

/// A merged rectangle in a slice: normal-layer `n`, min corner `(u0, v0)`, size `w×h`.
#[derive(Clone, Copy)]
struct Rect {
    n: usize,
    u0: usize,
    v0: usize,
    w: usize,
    h: usize,
}

/// Build one chunk's greedy mesh into `out` (cleared first — pass the world's
/// reusable [`ChunkMeshData`] scratch). `padded` carries the chunk + its shell,
/// `uniform` is the chunk's uniform block id (if any, for the fast paths),
/// `tables` the hot registry snapshot, and `light` the settled light shell the
/// per-vertex smooth light samples (interior, border, and diagonal alike).
pub fn build_chunk_mesh(
    padded: &Padded,
    uniform: Option<BlockId>,
    tables: &HotTables,
    light: &PaddedLight,
    out: &mut ChunkMeshData,
) {
    for (_, m) in out.iter_mut() {
        m.clear();
    }
    match uniform {
        Some(id) if !tables.solid[id.0 as usize] => {} // uniform non-solid: empty
        Some(_) => sweep(padded, true, tables, light, out), // uniform solid: borders only
        None => sweep(padded, false, tables, light, out),   // dense
    }
}

/// The greedy sweep over all six directions. `edge_only` restricts each direction
/// to its border slice (the uniform-solid fast path — a uniform chunk's interior
/// slices can never expose a face).
fn sweep(
    padded: &Padded,
    edge_only: bool,
    tables: &HotTables,
    light: &PaddedLight,
    out: &mut ChunkMeshData,
) {
    let mut mask: [Option<FaceSample>; MASK_CAP] = [None; MASK_CAP];

    for dir in &DIRS {
        let edge_n = if dir.step > 0 { CHUNK_SIZE - 1 } else { 0 };

        for n in 0..CHUNK_SIZE {
            if edge_only && n != edge_n {
                continue;
            }

            // Phase 1: mask of exposed faces in this slice, as FaceSamples.
            let mut any = false;
            for v in 0..CHUNK_SIZE {
                for u in 0..CHUNK_SIZE {
                    mask[u + v * CHUNK_SIZE] = face_sample(padded, tables, light, dir, n, u, v);
                    any |= mask[u + v * CHUNK_SIZE].is_some();
                }
            }
            if !any {
                continue;
            }

            // Phase 2: greedy rectangles — grow along U while the sample matches,
            // then along V while the whole row matches, clear, emit.
            for v0 in 0..CHUNK_SIZE {
                for u0 in 0..CHUNK_SIZE {
                    let key = mask[u0 + v0 * CHUNK_SIZE];
                    let Some(sample) = key else { continue };
                    let mut w = 1;
                    while u0 + w < CHUNK_SIZE && mask[u0 + w + v0 * CHUNK_SIZE] == key {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while v0 + h < CHUNK_SIZE {
                        let row = (v0 + h) * CHUNK_SIZE;
                        for k in 0..w {
                            if mask[u0 + k + row] != key {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for dv in 0..h {
                        let row = (v0 + dv) * CHUNK_SIZE;
                        mask[u0 + row..u0 + w + row].fill(None);
                    }
                    emit_rect(out, dir, Rect { n, u0, v0, w, h }, sample, tables);
                }
            }
        }
    }
}

/// The [`FaceSample`] for one cell's face in `dir`, or `None` if the cell is
/// non-solid or the face is culled. Reads the padded neighbourhood for the cull
/// neighbour, the AO occluders, and (via the chunk-local light grid) the light of
/// the empty cell the face opens into.
fn face_sample(
    padded: &Padded,
    tables: &HotTables,
    light: &PaddedLight,
    dir: &Dir,
    n: usize,
    u: usize,
    v: usize,
) -> Option<FaceSample> {
    let mut c = [0i32; 3];
    c[dir.n_axis] = n as i32;
    c[dir.u_axis] = u as i32;
    c[dir.v_axis] = v as i32;
    let id = padded.at(c[0], c[1], c[2]);
    if !tables.solid[id.0 as usize] {
        return None;
    }
    // The cell the face opens into: one step along the normal.
    let mut o = c;
    o[dir.n_axis] += dir.step;
    let nbr = padded.at(o[0], o[1], o[2]);
    if covered(id, nbr, tables) {
        return None;
    }

    let opaque_at = |p: [i32; 3]| tables.opaque[padded.at(p[0], p[1], p[2]).0 as usize];
    // Per corner: AO from three outward occluders; smooth light as average of
    // up to 4 touching cells (opaque cells skipped). Always has one light term.
    let mut ao = [0u8; 4];
    let mut sky = [0u8; 4];
    let mut block = [0u8; 4];
    for i in 0..4 {
        let du = if dir.corners[i][1] > 0.0 { 1 } else { -1 };
        let dv = if dir.corners[i][2] > 0.0 { 1 } else { -1 };
        let mut s1 = o;
        s1[dir.u_axis] += du;
        let mut s2 = o;
        s2[dir.v_axis] += dv;
        let mut cor = o;
        cor[dir.u_axis] += du;
        cor[dir.v_axis] += dv;
        ao[i] = corner_ao(opaque_at(s1), opaque_at(s2), opaque_at(cor));
        let (mut ssum, mut bsum, mut count) = (0u32, 0u32, 0u32);
        for p in [o, s1, s2, cor] {
            if opaque_at(p) {
                continue;
            }
            let lum = light.at(p[0], p[1], p[2]);
            ssum += lum.sky.get() as u32;
            bsum += lum.block.get() as u32;
            count += 1;
        }
        sky[i] = (ssum / count) as u8;
        block[i] = (bsum / count) as u8;
    }

    Some(FaceSample { id, ao, sky, block })
}

/// Append one merged rectangle as a single [`MeshData::quad`], routed to its
/// block's pass. Four chunk-local corners scaled from the direction's unit-quad
/// table by the rectangle's extents; each vertex carries the sample's per-corner
/// AO and light.
fn emit_rect(out: &mut ChunkMeshData, dir: &Dir, rect: Rect, sample: FaceSample, tables: &HotTables) {
    let mut origin = [0u32; 3];
    origin[dir.n_axis] = rect.n as u32;
    origin[dir.u_axis] = rect.u0 as u32;
    origin[dir.v_axis] = rect.v0 as u32;

    let corners = std::array::from_fn(|i| {
        let cr = &dir.corners[i];
        let mut pos = [0u8; 3];
        pos[dir.n_axis] = (origin[dir.n_axis] + cr[0] as u32) as u8;
        pos[dir.u_axis] = (origin[dir.u_axis] + cr[1] as u32 * rect.w as u32) as u8;
        pos[dir.v_axis] = (origin[dir.v_axis] + cr[2] as u32 * rect.h as u32) as u8;
        MeshVertex::new(
            pos,
            dir.normal,
            sample.id.0,
            Ao::new(sample.ao[i]),
            Light::new(sample.sky[i], sample.block[i]),
        )
    });

    let pass = if tables.opaque[sample.id.0 as usize] { Pass::Opaque } else { Pass::Transparent };
    out[pass].quad(corners);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::chunk::CHUNK_VOLUME;
    use crate::world::generation::TerrainGenerator;

    struct EmptyGen;
    impl TerrainGenerator for EmptyGen {
        fn height(&self, _wx: i32, _wz: i32) -> i32 {
            0
        }
        fn surface_at(&self, _wx: i32, _wz: i32) -> BlockId {
            AIR
        }
        fn deep(&self) -> BlockId {
            AIR
        }
    }

    struct SolidGen;
    impl TerrainGenerator for SolidGen {
        fn height(&self, _wx: i32, _wz: i32) -> i32 {
            i32::MAX
        }
        fn surface_at(&self, _wx: i32, _wz: i32) -> BlockId {
            STONE
        }
        fn deep(&self) -> BlockId {
            STONE
        }
    }

    const STONE: BlockId = BlockId(1);
    const DIRT: BlockId = BlockId(2);

    fn tables() -> HotTables {
        HotTables {
            solid: vec![false, true, true].into(),
            opaque: vec![false, true, true].into(),
            emission: vec![0, 0, 0].into(),
        }
    }

    fn empty_chunk() -> Chunk {
        Chunk::new(0, 0, 0, &EmptyGen)
    }

    /// A padded neighbourhood holding just `chunk` (air shell).
    fn solo(chunk: &Chunk) -> Padded {
        Padded::capture(|dx, dy, dz| (dx == 0 && dy == 0 && dz == 0).then_some(chunk))
    }

    /// Build the opaque mesh for a chunk in isolation, full-bright light (so
    /// merge/cull assertions don't see light-driven splits).
    fn build(chunk: &Chunk) -> MeshData {
        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(chunk), chunk.uniform(), &tables(), &PaddedLight::full(), &mut out);
        assert!(out[Pass::Transparent].is_empty(), "opaque blocks make no transparent geometry");
        let [opaque, _] = out.into_slots();
        opaque
    }

    fn culled_face_area(chunk: &Chunk, t: &HotTables) -> usize {
        let solid_at = |x: i32, y: i32, z: i32| -> bool {
            let range = 0..CS;
            if !range.contains(&x) || !range.contains(&y) || !range.contains(&z) {
                return false;
            }
            t.solid[chunk.get_local(x as usize, y as usize, z as usize).0 as usize]
        };
        let mut area = 0;
        for y in 0..CS {
            for z in 0..CS {
                for x in 0..CS {
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

    fn index_count(data: &MeshData) -> usize {
        data.buckets().iter().map(|b| b.len()).sum()
    }

    fn quad_areas(data: &MeshData) -> Vec<f32> {
        assert_eq!(data.vertices().len() % 4, 0, "quads are 4 vertices each");
        assert_eq!(index_count(data), data.vertices().len() / 4 * 6);
        data.vertices()
            .chunks_exact(4)
            .map(|q| {
                let e = |a: &MeshVertex, b: &MeshVertex| {
                    let (a, b) = (a.local_pos(), b.local_pos());
                    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                };
                e(&q[0], &q[1]) * e(&q[0], &q[3])
            })
            .collect()
    }

    fn total_area(data: &MeshData) -> f32 {
        quad_areas(data).iter().sum()
    }

    fn quads_in_y_plane(data: &MeshData, y: f32) -> usize {
        data.vertices()
            .chunks_exact(4)
            .filter(|q| q.iter().all(|v| v.local_pos()[1] == y))
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
        assert_eq!(quads_in_y_plane(&data, 1.0), 1, "flat top with uniform AO is one quad");
        let reference = culled_face_area(&chunk, &tables());
        assert_eq!(reference, 30);
        assert_eq!(total_area(&data), reference as f32);
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
        let areas = quad_areas(&data);
        let reference = culled_face_area(&chunk, &tables());
        assert_eq!(areas.len(), reference, "no two faces merged");
        assert!(areas.iter().all(|&a| a == 1.0), "every quad is 1x1");
    }

    #[test]
    fn different_blocks_do_not_merge() {
        let mut chunk = empty_chunk();
        chunk.set_local(0, 0, 0, STONE);
        chunk.set_local(1, 0, 0, DIRT);
        let data = build(&chunk);
        assert_eq!(quads_in_y_plane(&data, 1.0), 2, "different blocks never merge");
        let reference = culled_face_area(&chunk, &tables());
        assert_eq!(reference, 10);
        assert_eq!(total_area(&data), reference as f32);
    }

    #[test]
    fn ambient_occlusion_splits_a_run_the_flat_case_would_merge() {
        // A 3-long floor run; a wall block rises above the first cell. The wall
        // occludes the near end's top corners but not the far end's, so the two
        // remaining tops carry different AO and must NOT merge — where a flat
        // (unoccluded) run of the same tops would be a single quad.
        let flat = {
            let mut c = empty_chunk();
            c.set_local(1, 0, 0, STONE);
            c.set_local(2, 0, 0, STONE);
            build(&c)
        };
        assert_eq!(quads_in_y_plane(&flat, 1.0), 1, "flat run of two tops merges to one");

        let mut chunk = empty_chunk();
        chunk.set_local(0, 0, 0, STONE);
        chunk.set_local(1, 0, 0, STONE);
        chunk.set_local(2, 0, 0, STONE);
        chunk.set_local(0, 1, 0, STONE); // wall above the first cell (its own top is culled)
        let data = build(&chunk);
        assert_eq!(quads_in_y_plane(&data, 1.0), 2, "AO from the wall splits the two exposed tops");
    }

    #[test]
    fn smooth_light_gives_a_face_a_gradient_across_a_lit_shadow_edge() {
        use super::super::light::{LightLevel, Lumel};
        // A flat stone floor whose tops open into a y=1 light layer that steps
        // from full sky (x<8) to dark (x>=8). Per-corner averaging straddles the
        // step, so boundary tops carry differing corner light — a gradient a flat
        // per-face value can't express, and which the merge key won't collapse.
        let mut chunk = empty_chunk();
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                chunk.set_local(x, 0, z, STONE);
            }
        }
        let light = PaddedLight::from_fn(|x, _, _| Lumel {
            sky: LightLevel::new(if x < 8 { 15 } else { 0 }),
            block: LightLevel::DARK,
        });
        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(&chunk), None, &tables(), &light, &mut out);

        let tops = &out[Pass::Opaque];
        assert!(quads_in_y_plane(tops, 1.0) > 1, "the light step splits the flat floor's tops");
        let gradient_quad = tops.vertices().chunks_exact(4).any(|q| {
            q.iter().all(|v| v.local_pos()[1] == 1.0)
                && q.iter().any(|v| v.light() != q[0].light())
        });
        assert!(gradient_quad, "a boundary top face carries differing corner light");
    }

    #[test]
    fn uniform_air_meshes_empty() {
        let chunk = empty_chunk();
        assert_eq!(chunk.uniform(), Some(AIR));
        assert!(build(&chunk).is_empty());
    }

    #[test]
    fn uniform_solid_fast_path_matches_a_dense_fill() {
        let uniform = Chunk::new(0, 0, 0, &SolidGen);
        assert_eq!(uniform.uniform(), Some(STONE));
        let dense = Chunk::from_dense(0, 0, 0, Box::new([STONE.0; CHUNK_VOLUME]));
        assert!(dense.uniform().is_none());

        let (a, b) = (build(&uniform), build(&dense));
        assert_eq!(total_area(&a), (6 * CHUNK_SIZE * CHUNK_SIZE) as f32, "6 full faces");
        assert_eq!(a.buckets(), b.buckets());
        assert_eq!(a.vertices(), b.vertices(), "uniform and dense paths mesh identically");
    }

    #[test]
    fn translucent_faces_route_to_the_transparent_pass() {
        const GLASS: BlockId = BlockId(3);
        let t = HotTables {
            solid: vec![false, true, true, true].into(),
            opaque: vec![false, true, true, false].into(),
            emission: vec![0, 0, 0, 0].into(),
        };
        let mut chunk = empty_chunk();
        chunk.set_local(5, 5, 5, GLASS);
        chunk.set_local(6, 5, 5, GLASS); // adjacent glass: shared face culled
        chunk.set_local(5, 5, 6, STONE); // opaque behind glass: that glass face culled

        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(&chunk), None, &t, &PaddedLight::full(), &mut out);
        assert!(!out[Pass::Transparent].is_empty(), "glass emits transparent geometry");
        assert_eq!(total_area(&out[Pass::Transparent]), 9.0, "internal + occluded glass faces culled");
        assert_eq!(total_area(&out[Pass::Opaque]), 6.0, "stone keeps all six faces (glass doesn't cull)");
    }

    #[test]
    fn border_faces_cull_against_neighbour_chunks() {
        let mut chunk = empty_chunk();
        chunk.set_local(CHUNK_SIZE - 1, 0, 0, STONE);
        let mut other = Chunk::new(1, 0, 0, &EmptyGen);
        other.set_local(0, 0, 0, STONE);

        let alone = build(&chunk);
        // A padded neighbourhood with the +X neighbour present.
        let padded =
            Padded::capture(|dx, dy, dz| match (dx, dy, dz) {
                (0, 0, 0) => Some(&chunk),
                (1, 0, 0) => Some(&other),
                _ => None,
            });
        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&padded, None, &tables(), &PaddedLight::full(), &mut out);

        assert_eq!(total_area(&alone), 6.0, "isolated cube shows all six faces");
        assert_eq!(total_area(&out[Pass::Opaque]), 5.0, "the face against the neighbour is culled");
    }

    #[test]
    fn far_chunk_meshes_byte_identically_to_the_origin_chunk() {
        let mut near = Chunk::new(0, 0, 0, &EmptyGen);
        let mut far = Chunk::new(62_500_000, -3_000, -62_500_000, &EmptyGen);
        for (x, y, z, id) in [(0, 0, 0, STONE), (1, 0, 0, STONE), (5, 9, 15, DIRT)] {
            near.set_local(x, y, z, id);
            far.set_local(x, y, z, id);
        }
        let (a, b) = (build(&near), build(&far));
        assert_eq!(a.buckets(), b.buckets());
        assert_eq!(a.vertices(), b.vertices(), "far chunk meshes byte-identically");
    }

    #[test]
    fn single_cube_vertices_are_chunk_local() {
        let mut chunk = Chunk::new(6_250_000, 40, -6_250_000, &EmptyGen);
        chunk.set_local(2, 3, 4, STONE);
        let data = build(&chunk);
        assert_eq!(data.vertices().len(), 24, "six 1x1 faces");
        for v in data.vertices() {
            let pos = v.local_pos();
            assert!(pos.iter().all(|&c| (0.0..=CHUNK_SIZE as f32).contains(&c)), "chunk-local {pos:?}");
        }
    }
}
