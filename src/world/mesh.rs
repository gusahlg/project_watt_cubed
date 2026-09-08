//! Turns a chunk's voxels into triangle-mesh data ready for GPU upload.
//!
//! A one-time **greedy mesh** per chunk:
//!
//! - The sweep reads a [`Padded`] neighbourhood — the chunk's 16³ voxels plus a
//!   one-voxel shell copied from its 26 neighbours — so face culling, the AO
//!   stencil, and own-cell reads are uniform flat-index stride walks, with no
//!   interior/border branch or per-direction axis remapping. The shell is
//!   what makes ambient occlusion seamless across chunk borders.
//! - Only faces bordering see-through space are emitted. The cull key is
//!   **opacity**, not solidity ([`covered`]): an opaque neighbour hides a face; a
//!   translucent one (glass) does not, and two of the same translucent block hide
//!   their shared internal face.
//! - Each exposed face routes to the **opaque** or **transparent** [`MeshData`]
//!   by its block's opacity, so a chunk yields up to two meshes ([`ChunkMeshData`]).
//! - Adjacent faces merge into maximal rectangles keyed on the whole
//!   [`FaceSample`] (id + per-corner AO + per-corner sky/block light), so an AO
//!   or smooth-light gradient never merges into a flat quad.
//! - Vertices are CHUNK-LOCAL (0..=16, exact in f32); the world draws each with a
//!   camera-relative offset, so far terrain never jitters.
//! - Uniform fast paths: a uniform non-solid chunk is empty; a uniform solid one
//!   only sweeps its six border slices.
use voxel_engine::{Ao, Light, MeshData, MeshVertex, Normal};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::light::{Lumel, MAX_LIGHT, PaddedLight};
use super::neighborhood::{Neighborhood, padded_index};
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

/// Chunk size as a signed coordinate, for the `-1..=16` padded range (tests).
#[cfg(test)]
const CS: i32 = CHUNK_SIZE as i32;

/// The chunk's 16³ voxels plus a one-voxel shell pulled from its 26 neighbours,
/// indexed by signed coords `x, y, z ∈ -1..=16`. Owned, so a mesh job shares
/// nothing with the live chunk map, and captured on the main thread where the
/// neighbourhood is resolvable. A missing neighbour reads as [`AIR`]. The mesh
/// instantiation of [`Neighborhood`]: capture/index/pooling live there.
///
/// This one structure serves every neighbour read the sweep makes — cull (the
/// immediate outward cell), AO (the 3 occluders around each face corner, which
/// at a chunk edge fall into a neighbour's *interior* layer), and the own-cell
/// scan — so there is no interior/border special-casing anywhere.
pub struct Padded {
    inner: Neighborhood<BlockId>,
}

impl Padded {
    /// Flat-index read — the sweep's stride walk.
    #[inline]
    fn at_flat(&self, i: usize) -> BlockId {
        self.inner.at_flat(i)
    }

    /// Copy the chunk and its shell out of the map. `chunk_at(dx, dy, dz)` yields
    /// the chunk at chunk-offset `(dx, dy, dz)` with each component in `-1..=1`
    /// (`(0,0,0)` is the chunk itself), or `None` (→ air). Row-wise: the bulk
    /// of the halo fills through [`Chunk::copy_row`]'s one-dispatch-per-row
    /// reads (this runs on the main thread inside every mesh admit).
    pub fn capture<'a>(chunk_at: impl Fn(i32, i32, i32) -> Option<&'a Chunk>) -> Self {
        Self {
            inner: Neighborhood::capture_rows(
                AIR,
                chunk_at,
                |c: &Chunk, lx, ly, lz| c.get_local(lx, ly, lz),
                |c: &Chunk, ly, lz, out| c.copy_row(ly, lz, out),
            ),
        }
    }
}

/// Opaque neighbour or same block hides a face (two glass blocks share a hidden internal face).
#[inline]
fn covered(my: BlockId, nbr: BlockId, tables: &HotTables) -> bool {
    tables.opaque(nbr) || nbr == my
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
    corners: [[u8; 3]; 4],
    normal: Normal,
}

const DIRS: [Dir; 6] = [
    Dir {
        step: 1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        normal: Normal::PosX,
    },
    Dir {
        step: -1,
        n_axis: 0,
        u_axis: 2,
        v_axis: 1,
        corners: [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        normal: Normal::NegX,
    },
    Dir {
        step: 1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
        normal: Normal::PosY,
    },
    Dir {
        step: -1,
        n_axis: 1,
        u_axis: 0,
        v_axis: 2,
        corners: [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        normal: Normal::NegY,
    },
    Dir {
        step: 1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
        normal: Normal::PosZ,
    },
    Dir {
        step: -1,
        n_axis: 2,
        u_axis: 0,
        v_axis: 1,
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        normal: Normal::NegZ,
    },
];

/// One slice of the sweep: 16 x 16 cells.
const MASK_CAP: usize = CHUNK_SIZE * CHUNK_SIZE;

/// Empty mask slot. Packed samples never use all 64 bits (56 used), so this
/// sentinel cannot collide with a real face.
const NO_FACE: u64 = u64::MAX;

/// The greedy-merge key: two faces merge only when the whole sample matches —
/// block id, per-corner AO, and per-corner sky/block light — so an AO or smooth-
/// light gradient never merges into a flat quad. Packed into one `u64` in the
/// slice mask (`NO_FACE` = empty). `ao[i]`/`sky[i]`/`block[i]` correspond to
/// `Dir::corners[i]`.
#[derive(Clone, Copy)]
struct FaceSample {
    id: BlockId,
    ao: [u8; 4],
    sky: [u8; 4],
    block: [u8; 4],
}

/// `id` 16 + 4×2 AO + 4×4 sky + 4×4 block = 56 bits.
#[inline]
fn pack_sample(s: FaceSample) -> u64 {
    let mut w = s.id.0 as u64;
    for i in 0..4 {
        w |= (s.ao[i] as u64) << (16 + 2 * i);
        w |= (s.sky[i] as u64) << (24 + 4 * i);
        w |= (s.block[i] as u64) << (40 + 4 * i);
    }
    w
}

#[inline]
fn unpack_sample(w: u64) -> FaceSample {
    FaceSample {
        id: BlockId(w as u16),
        ao: std::array::from_fn(|i| ((w >> (16 + 2 * i)) & 3) as u8),
        sky: std::array::from_fn(|i| ((w >> (24 + 4 * i)) & 15) as u8),
        block: std::array::from_fn(|i| ((w >> (40 + 4 * i)) & 15) as u8),
    }
}

/// Integer mean of up to four 0..=15 light samples. `count ∈ 1..=4`; the
/// power-of-two arms are bit-identical to `/ count`.
#[inline]
fn avg_light(sum: u32, count: u32) -> u8 {
    (match count {
        1 => sum,
        2 => sum >> 1,
        4 => sum >> 2,
        _ => sum / 3,
    }) as u8
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
/// `uniform` is the chunk's uniform block id (if any, for the fast paths), and
/// `tables` the hot registry snapshot, and `light` the settled light shell the
/// per-vertex smooth light samples (interior, border, and diagonal alike).
pub fn build_chunk_mesh(
    padded: &Padded,
    uniform: Option<BlockId>,
    tables: &HotTables,
    light: &PaddedLight,
    out: &mut ChunkMeshData,
) {
    build_chunk_mesh_inner(padded, uniform, tables, Some(light), out);
}

/// Build with constant full light and no light-shell input at all. This is the
/// stripped-profile path (voxel lighting disabled); when AO is also disabled,
/// face sampling returns immediately after culling and skips the entire
/// four-corner stencil.
pub fn build_chunk_mesh_unlit(
    padded: &Padded,
    uniform: Option<BlockId>,
    tables: &HotTables,
    out: &mut ChunkMeshData,
) {
    build_chunk_mesh_inner(padded, uniform, tables, None, out);
}

fn build_chunk_mesh_inner(
    padded: &Padded,
    uniform: Option<BlockId>,
    tables: &HotTables,
    light: Option<&PaddedLight>,
    out: &mut ChunkMeshData,
) {
    for (_, m) in out.iter_mut() {
        m.clear();
    }
    match uniform {
        Some(id) if !tables.solid(id) => {} // uniform non-solid: empty
        Some(_) => sweep(padded, true, tables, light, out), // uniform solid: borders only
        None => sweep(padded, false, tables, light, out),   // dense
    }
}

/// The index delta of one step along world axis `axis` (0=X, 1=Y, 2=Z) in the
/// padded 18³ layout — derived from the ONE layout law ([`padded_index`])
/// rather than restating it. The sweep walks flat indices with these strides:
/// every probe becomes `base ± s` adds where the coordinate form paid three
/// scattered `[i32; 3]` writes through dynamic axis indices plus two
/// multiplies, ~30 times per face candidate.
fn axis_stride(axis: usize) -> i32 {
    let mut p = [0i32; 3];
    p[axis] = 1;
    padded_index(p[0], p[1], p[2]) as i32 - padded_index(0, 0, 0) as i32
}

/// The greedy sweep over all six directions. `edge_only` restricts each direction
/// to its border slice (the uniform-solid fast path — a uniform chunk's interior
/// slices can never expose a face).
fn sweep(
    padded: &Padded,
    edge_only: bool,
    tables: &HotTables,
    light: Option<&PaddedLight>,
    out: &mut ChunkMeshData,
) {
    let mut mask = [NO_FACE; MASK_CAP];
    let flat_origin = padded_index(0, 0, 0) as i32;

    for dir in &DIRS {
        let edge_n = if dir.step > 0 { CHUNK_SIZE - 1 } else { 0 };
        let (s_n, s_u, s_v) =
            (axis_stride(dir.n_axis), axis_stride(dir.u_axis), axis_stride(dir.v_axis));
        let corner_uv: [[i32; 2]; 4] = std::array::from_fn(|i| {
            [
                if dir.corners[i][1] > 0 { 1 } else { -1 },
                if dir.corners[i][2] > 0 { 1 } else { -1 },
            ]
        });

        let ns = if edge_only { edge_n..edge_n + 1 } else { 0..CHUNK_SIZE };
        for n in ns {
            let base_n = flat_origin + n as i32 * s_n;

            // Phase 1: mask of exposed faces in this slice, packed.
            let mut any = false;
            for v in 0..CHUNK_SIZE {
                let base_v = base_n + v as i32 * s_v;
                for u in 0..CHUNK_SIZE {
                    let packed = face_sample(
                        padded,
                        tables,
                        light,
                        dir,
                        s_n,
                        s_u,
                        s_v,
                        base_v + u as i32 * s_u,
                        &corner_uv,
                    );
                    mask[u + v * CHUNK_SIZE] = packed;
                    any |= packed != NO_FACE;
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
                    if key == NO_FACE {
                        continue;
                    }
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
                        mask[u0 + row..u0 + w + row].fill(NO_FACE);
                    }
                    emit_rect(out, dir, Rect { n, u0, v0, w, h }, unpack_sample(key), tables);
                }
            }
        }
    }
}

/// Packed [`FaceSample`] for one cell's face in `dir`, or [`NO_FACE`] if the
/// cell is non-solid or the face is culled. Reads the padded neighbourhood for
/// the cull neighbour and a 3×3 in-plane stencil (AO occluders + smooth light),
/// and the settled light shell for per-corner sky/block. `idx` is the cell's
/// flat padded index; every probe is a stride add off it.
#[allow(clippy::too_many_arguments)]
fn face_sample(
    padded: &Padded,
    tables: &HotTables,
    light: Option<&PaddedLight>,
    dir: &Dir,
    s_n: i32,
    s_u: i32,
    s_v: i32,
    idx: i32,
    corner_uv: &[[i32; 2]; 4],
) -> u64 {
    let id = padded.at_flat(idx as usize);
    if !tables.solid(id) {
        return NO_FACE;
    }
    // The cell the face opens into: one step along the normal.
    let open = idx + dir.step * s_n;
    let nbr = padded.at_flat(open as usize);
    if covered(id, nbr, tables) {
        return NO_FACE;
    }

    // Minimum/Fast disable both lighting and AO. Their merge key is constant,
    // so none of the neighbour probes or light reads can affect the result —
    // culling alone decides the mesh.
    if !tables.ao && light.is_none() {
        return pack_sample(FaceSample { id, ao: [3; 4], sky: [MAX_LIGHT; 4], block: [MAX_LIGHT; 4] });
    }

    // One 3×3 stencil in the OPEN layer: AO and smooth light share the nine
    // cells. Light lumels load only on the lit path.
    let mut opaque = [[false; 3]; 3];
    let mut sky = [MAX_LIGHT; 4];
    let mut block = [MAX_LIGHT; 4];
    if let Some(light) = light {
        let mut lumel = [[Lumel::DARK; 3]; 3];
        for dv in 0..3 {
            for du in 0..3 {
                let p = (open + (du as i32 - 1) * s_u + (dv as i32 - 1) * s_v) as usize;
                opaque[du][dv] = tables.opaque(padded.at_flat(p));
                lumel[du][dv] = light.at_flat(p);
            }
        }
        for i in 0..4 {
            let ou = (corner_uv[i][0] + 1) as usize;
            let ov = (corner_uv[i][1] + 1) as usize;
            let (mut ssum, mut bsum, mut count) = (0u32, 0u32, 0u32);
            for (u, v) in [(1, 1), (ou, 1), (1, ov), (ou, ov)] {
                if opaque[u][v] {
                    continue;
                }
                let l = lumel[u][v];
                ssum += l.sky.get() as u32;
                bsum += l.block.get() as u32;
                count += 1;
            }
            sky[i] = avg_light(ssum, count);
            block[i] = avg_light(bsum, count);
        }
    } else {
        for dv in 0..3 {
            for du in 0..3 {
                let p = (open + (du as i32 - 1) * s_u + (dv as i32 - 1) * s_v) as usize;
                opaque[du][dv] = tables.opaque(padded.at_flat(p));
            }
        }
    }

    // AO off: every corner reads unoccluded (uniform 3) — a perf lever, and it
    // also merges quads a gradient would split (matches the pre-rewrite toggle).
    let ao = std::array::from_fn(|i| {
        if !tables.ao {
            return 3;
        }
        let ou = (corner_uv[i][0] + 1) as usize;
        let ov = (corner_uv[i][1] + 1) as usize;
        vertex_ao(opaque[ou][1], opaque[1][ov], opaque[ou][ov])
    });

    pack_sample(FaceSample { id, ao, sky, block })
}

/// Per-vertex ambient-occlusion level `0..=3` (`3` = unoccluded) from its three
/// occluders. Two touching sides fully occlude the corner (the classic clamp).
fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    3 - (side1 as u8 + side2 as u8 + corner as u8)
}

/// Append one merged rectangle as a single [`MeshData::quad`], routed to its
/// block's pass. Four chunk-local corners scaled from the direction's unit-quad
/// table by the rectangle's extents; each vertex carries the sample's per-corner
/// AO and sky/block light.
fn emit_rect(out: &mut ChunkMeshData, dir: &Dir, rect: Rect, sample: FaceSample, tables: &HotTables) {
    let mut origin = [0u32; 3];
    origin[dir.n_axis] = rect.n as u32;
    origin[dir.u_axis] = rect.u0 as u32;
    origin[dir.v_axis] = rect.v0 as u32;

    let mut corners: [MeshVertex; 4] = std::array::from_fn(|i| {
        let cr = dir.corners[i];
        let mut pos = [0u8; 3];
        pos[dir.n_axis] = (origin[dir.n_axis] + cr[0] as u32) as u8;
        pos[dir.u_axis] = (origin[dir.u_axis] + cr[1] as u32 * rect.w as u32) as u8;
        pos[dir.v_axis] = (origin[dir.v_axis] + cr[2] as u32 * rect.h as u32) as u8;
        MeshVertex::new(
            pos,
            dir.normal,
            // Vertex layer only — table lookups stay on the true id. Wraps
            // once the palette outgrows the device's texture-layer cap
            // (identity below it; the growth path logs the crossing once).
            sample.id.0 % tables.layer_cap,
            Ao::new(sample.ao[i]),
            Light::new(sample.sky[i], sample.block[i]),
            tables.fluid_surface(sample.id),
        )
    });

    // Anisotropy flip: the fixed 0-1-2/0-2-3 fan diagonal (corner 0↔2) smears AO
    // when that diagonal spans the brighter pair. Rotating the quad by one vertex
    // moves the diagonal to 1↔3, keeping the seam on the darker pair (0fps).
    if sample.ao[0] + sample.ao[2] < sample.ao[1] + sample.ao[3] {
        corners.rotate_left(1);
    }

    out[tables.layer[sample.id.0 as usize]].quad(corners);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::chunk::CHUNK_VOLUME;
    use super::super::light::{LightLevel, Lumel};
    use crate::world::generation::TerrainGenerator;
    use voxel_engine::Pass;

    /// FNV-1a over every pass's decoded vertex fields and index buckets.
    /// MeshVertex's packed words are private to the engine, so this hashes the
    /// public fields (pos/normal/layer/AO/light/water/micro) via `to_le_bytes`.
    fn hash_mesh(data: &ChunkMeshData) -> u32 {
        let mut bytes = Vec::new();
        for (pass, mesh) in data.iter() {
            bytes.push(pass as u8);
            for v in mesh.vertices() {
                for c in v.local_pos() {
                    bytes.extend_from_slice(&c.to_le_bytes());
                }
                bytes.push(v.normal() as u8);
                bytes.extend_from_slice(&v.layer().to_le_bytes());
                bytes.push((0..=3).find(|&a| v.ao() == Ao::new(a)).expect("ao 0..=3"));
                let l = v.light();
                let (sky, block) = (0u8..=15)
                    .flat_map(|s| (0u8..=15).map(move |b| (s, b)))
                    .find(|&(s, b)| l == Light::new(s, b))
                    .expect("light 0..=15");
                bytes.push(sky);
                bytes.push(block);
                bytes.push(v.is_water() as u8);
                for m in v.micro() {
                    bytes.extend_from_slice(&m.to_le_bytes());
                }
            }
            for bucket in mesh.buckets() {
                bytes.extend_from_slice(&(bucket.len() as u32).to_le_bytes());
                for i in bucket {
                    bytes.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
        crate::hash::fnv1a_32(&bytes)
    }

    /// A non-constant light shell: every padded cell differs from its
    /// neighbours, so per-corner averaging and the merge key actually vary.
    fn gradient_light() -> PaddedLight {
        PaddedLight::from_fn(|x, y, z| Lumel {
            sky: LightLevel::new(((x + y + 17) as u8) & 15),
            block: LightLevel::new(((z * 3 + y + 17) as u8) & 15),
        })
    }

    /// Vertex-byte pin: four seed-42 neighbourhoods, dense (non-uniform), meshed
    /// unlit / full-bright / gradient-lit. Hashes must stay bit-identical across
    /// mesher edits. Print with `--nocapture` to refresh the table.
    #[test]
    fn vertex_byte_pin() {
        use crate::block::registry::BlockRegistry;
        use crate::world::generation::SineHills;

        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, 42);
        let tables = registry.hot_tables();
        // (coord, unlit, full, gradient) — filled from the first `--nocapture` run.
        // (2,2,2) is uniform sky at seed 42; (2,1,2) is the dense surface stand-in.
        let want: [((i32, i32, i32), u32, u32, u32); 4] = [
            ((0, 1, 0), 0xb0e2c9fb, 0xb0e2c9fb, 0x897aa7e2),
            ((3, 1, -2), 0x0e6322e1, 0x0e6322e1, 0x5302192d),
            ((-5, 0, 4), 0x844d5350, 0x844d5350, 0xd8f141d4),
            ((2, 1, 2), 0xc9d80078, 0xc9d80078, 0xf2649ff0),
        ];

        let mut got = [(0u32, 0u32, 0u32); 4];
        for (i, ((cx, cy, cz), _, _, _)) in want.iter().copied().enumerate() {
            let neigh: Vec<Chunk> = (0..27)
                .map(|k| Chunk::new(cx + k % 3 - 1, cy + k / 9 - 1, cz + k / 3 % 3 - 1, &generator))
                .collect();
            let chunk = &neigh[1 + 3 + 9]; // (dx,dy,dz) = (0,0,0)
            assert!(
                chunk.uniform().is_none(),
                "pin coord ({cx},{cy},{cz}) must be non-uniform"
            );
            let at = |dx: i32, dy: i32, dz: i32| -> Option<&Chunk> {
                Some(&neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize])
            };
            let padded = Padded::capture(at);

            let mut unlit = new_chunk_mesh_data();
            build_chunk_mesh_unlit(&padded, chunk.uniform(), &tables, &mut unlit);
            let mut full = new_chunk_mesh_data();
            build_chunk_mesh(&padded, chunk.uniform(), &tables, &PaddedLight::full(), &mut full);
            let mut grad = new_chunk_mesh_data();
            build_chunk_mesh(&padded, chunk.uniform(), &tables, &gradient_light(), &mut grad);

            got[i] = (hash_mesh(&unlit), hash_mesh(&full), hash_mesh(&grad));
            println!(
                "vertex_byte_pin ({cx},{cy},{cz}) unlit=0x{:08x} full=0x{:08x} grad=0x{:08x}",
                got[i].0, got[i].1, got[i].2
            );
        }
        for (i, ((cx, cy, cz), unlit_h, full_h, grad_h)) in want.iter().copied().enumerate() {
            assert_eq!(got[i].0, unlit_h, "({cx},{cy},{cz}) unlit");
            assert_eq!(got[i].1, full_h, "({cx},{cy},{cz}) full");
            assert_eq!(got[i].2, grad_h, "({cx},{cy},{cz}) gradient");
        }
    }

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
        HotTables::from_parts(
            &[false, true, true],
            &[false, true, true],
            &[false, false, false],
            vec![Pass::Opaque, Pass::Opaque, Pass::Opaque].into(),
            vec![0, 0, 0].into(),
            vec![0, 0, 0].into(),
        )
    }

    fn empty_chunk() -> Chunk {
        Chunk::new(0, 0, 0, &EmptyGen)
    }

    /// Snapshot-capture cost (the main-thread half of every mesh admit) — the
    /// gauge for the row-wise capture redesign. Ignored: a timing benchmark,
    /// not a correctness gate. Run with
    /// `cargo test --release padded_capture_throughput -- --ignored --nocapture`.
    /// 2026-07-19 (12-core box), per-cell closure capture: ~26.7k captures/s;
    /// row-wise (`capture_rows` + `copy_row`): ~337k captures/s (12.7×).
    /// 2026-09-08: before 456k captures/s; after gated fill 468k captures/s (median of 3).
    #[test]
    #[ignore]
    fn padded_capture_throughput() {
        use crate::block::registry::BlockRegistry;
        use crate::world::generation::SineHills;
        use crate::world::light::LightGrid;

        let mut registry = BlockRegistry::with_builtins();
        let generator = SineHills::new(&mut registry, 20.0, 5);
        // The SURFACE band (world y 48..96): mixed paletted chunks — the case
        // that actually reaches the pool (uniform chunks capture cheap).
        let neigh: Vec<Chunk> = (0..27)
            .map(|k| Chunk::new(k % 3 - 1, 4 + k / 9 - 1, k / 3 % 3 - 1, &generator))
            .collect();
        let at = |dx: i32, dy: i32, dz: i32| -> Option<&Chunk> {
            Some(&neigh[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize])
        };
        let grids: Vec<LightGrid> = (0..27).map(|_| LightGrid::open_sky()).collect();
        let light_at = |dx: i32, dy: i32, dz: i32| -> Option<&LightGrid> {
            Some(&grids[((dx + 1) + (dz + 1) * 3 + (dy + 1) * 9) as usize])
        };

        const N: usize = 4000;
        let start = std::time::Instant::now();
        for _ in 0..N {
            let p = Padded::capture(at);
            let l = PaddedLight::capture(light_at);
            std::hint::black_box((&p, &l));
        }
        let dt = start.elapsed();
        println!(
            "{N} padded+light captures in {:.3}s = {:.0} captures/s",
            dt.as_secs_f64(),
            N as f64 / dt.as_secs_f64()
        );
    }

    /// The unlit path must be byte-identical to meshing against a full-bright
    /// shell — it is the same computation minus the reads. Checked with AO on
    /// AND off (off additionally takes the constant-sample early return).
    #[test]
    fn unlit_mesh_matches_full_bright_shell_exactly() {
        let solid = Chunk::new(0, 0, 0, &SolidGen);
        let mut dense = empty_chunk();
        for (x, y, z) in [(0, 0, 0), (1, 0, 0), (5, 9, 3), (15, 15, 15), (8, 8, 8)] {
            dense.set_local(x, y, z, if (x + y + z) % 2 == 0 { STONE } else { DIRT });
        }
        for chunk in [&solid, &dense] {
            let padded = Padded::capture(|dx, dy, dz| ((dx, dy, dz) == (0, 0, 0)).then_some(chunk));
            for ao in [false, true] {
                let mut tables = tables();
                tables.ao = ao;
                let mut lit = new_chunk_mesh_data();
                build_chunk_mesh(&padded, chunk.uniform(), &tables, &PaddedLight::full(), &mut lit);
                let mut unlit = new_chunk_mesh_data();
                build_chunk_mesh_unlit(&padded, chunk.uniform(), &tables, &mut unlit);
                for ((_, a), (_, b)) in lit.iter().zip(unlit.iter()) {
                    assert_eq!(a.vertices(), b.vertices(), "ao={ao}");
                    assert_eq!(a.buckets(), b.buckets(), "ao={ao}");
                }
            }
        }
    }

    #[test]
    fn blocks_past_the_old_u8_cap_mesh_with_their_own_layer() {
        // A block id above 255 must reach the vertex intact — the whole point
        // of the 14-bit layer field. Real registry, grown past the old cap.
        let mut reg = crate::block::registry::BlockRegistry::with_builtins();
        let els = 0..reg.elements().len() as u16;
        let mut high = AIR;
        'grow: for i in els.clone() {
            for j in els.clone().filter(|&j| j > i) {
                for p in 1..=99u8 {
                    use crate::block::element::ElementId;
                    high = reg
                        .mixture(&[(ElementId(i), p), (ElementId(j), 100 - p)])
                        .expect("mixture registers below the cap");
                    if reg.block_count() > 300 {
                        break 'grow;
                    }
                }
            }
        }
        assert!(high.0 > 255, "registry grew past the old u8 cap");

        let mut chunk = Chunk::from_uniform(0, 0, 0, AIR);
        chunk.set_local(8, 8, 8, high);
        let real = reg.hot_tables(); // layer_cap = u16::MAX → identity
        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(&chunk), None, &real, &PaddedLight::full(), &mut out);
        let pass = real.layer[high.0 as usize];
        let layers: Vec<u16> = out[pass].vertices().iter().map(|v| v.layer()).collect();
        assert!(!layers.is_empty(), "the lone block meshed");
        assert!(layers.iter().all(|&l| l == high.0), "vertex carries the full 14-bit id");
    }

    #[test]
    fn vertex_layers_wrap_at_the_device_texture_cap() {
        // Past the device's texture-layer ceiling the mesher wraps the VERTEX
        // layer only (tables still index the true id) — crafting keeps working
        // on min-spec GPUs, textures just repeat.
        let high = BlockId(300);
        let mut chunk = Chunk::from_uniform(0, 0, 0, AIR);
        chunk.set_local(8, 8, 8, high);
        // Air (id 0) stays non-solid/clear or the lone block's faces get culled.
        let mut bools = vec![true; 301];
        bools[0] = false;
        let mut t = HotTables::from_parts(
            &bools,
            &bools,
            &vec![false; 301],
            vec![Pass::Opaque; 301].into(),
            vec![0; 301].into(),
            vec![0; 301].into(),
        );
        t.layer_cap = 256; // a min-spec-ish ceiling
        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(&chunk), None, &t, &PaddedLight::full(), &mut out);
        let layers: Vec<u16> = out[Pass::Opaque].vertices().iter().map(|v| v.layer()).collect();
        assert!(!layers.is_empty());
        assert!(layers.iter().all(|&l| l == 300 % 256), "vertex layer wraps, id stays true");
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
        assert!(out[Pass::Blend].is_empty(), "opaque blocks make no transparent geometry");
        let [opaque, _cutout, _blend] = out.into_slots();
        opaque
    }

    fn culled_face_area(chunk: &Chunk, t: &HotTables) -> usize {
        let solid_at = |x: i32, y: i32, z: i32| -> bool {
            let range = 0..CS;
            if !range.contains(&x) || !range.contains(&y) || !range.contains(&z) {
                return false;
            }
            t.solid(chunk.get_local(x as usize, y as usize, z as usize))
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
        // Chunk construction canonicalizes single-valued palettes to Uniform
        // (chunk_data_to_brick), so this Paletted{palette:[STONE], cells:[0;N]}
        // input also becomes Uniform — not a bug, the canonical form. What the
        // test proves: two independently constructed, content-equal chunks
        // mesh bit-identically.
        let dense = Chunk::from_data(
            0,
            0,
            0,
            super::super::chunk::ChunkData::Paletted {
                palette: vec![STONE],
                cells: Box::new([0u8; CHUNK_VOLUME]),
            },
        );
        assert_eq!(dense.uniform(), Some(STONE), "canonical construction collapses this to Uniform too");

        let (a, b) = (build(&uniform), build(&dense));
        assert_eq!(total_area(&a), (6 * CHUNK_SIZE * CHUNK_SIZE) as f32, "6 full faces");
        assert_eq!(a.buckets(), b.buckets());
        assert_eq!(a.vertices(), b.vertices(), "uniform and dense paths mesh identically");
    }

    #[test]
    fn translucent_faces_route_to_the_blend_pass() {
        const GLASS: BlockId = BlockId(3);
        let t = HotTables::from_parts(
            &[false, true, true, true],
            &[false, true, true, false],
            &[false, false, false, false],
            vec![Pass::Opaque, Pass::Opaque, Pass::Opaque, Pass::Blend].into(),
            vec![0, 0, 0, 0].into(),
            vec![0, 0, 0, 0].into(),
        );
        let mut chunk = empty_chunk();
        chunk.set_local(5, 5, 5, GLASS);
        chunk.set_local(6, 5, 5, GLASS); // adjacent glass: shared face culled
        chunk.set_local(5, 5, 6, STONE); // opaque behind glass: that glass face culled

        let mut out = new_chunk_mesh_data();
        build_chunk_mesh(&solo(&chunk), None, &t, &PaddedLight::full(), &mut out);
        assert!(!out[Pass::Blend].is_empty(), "glass emits blend geometry");
        assert_eq!(total_area(&out[Pass::Blend]), 9.0, "internal + occluded glass faces culled");
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
