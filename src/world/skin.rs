//! Zone 3 — the far grey height-skin ring. Beyond the full-res chunks (Zone 1)
//! and the volumetric LOD tiles (Zone 2), the skin fills the horizon with a
//! wide, cheap, low-fidelity backdrop: the terrain *surface only*, tinted by the
//! surface block's palette colour and fogged toward the sky at the far edge.
//! Because world-gen is deterministic, a skin column is a
//! pure function of `(seed, x, z)` — never stored beyond its cached surface
//! mesh, never edited, never invalidated (WORLD-DESIGN Zone 3).
//!
//! The fidelity ladder drops state zone to zone so impossible states are
//! unconstructable: `MeshState` (5) → [`TileState`](super::lod::TileState) (3,
//! no `Dirty`) → [`SkinState`] (2, no `Air` — every column has a surface — and
//! no `Dirty`). The skin needs a NEW retained mesh primitive: the packed voxel
//! vertex cannot encode a continuous arbitrary-Y grey surface, so this lane
//! rides [`SurfaceData`]/[`SurfaceHandle`] instead of `MeshData`/`MeshHandle`.
use voxel_engine::{Color, Engine, SurfaceData, SurfaceHandle, SurfaceVertex};

use super::generation::TerrainGenerator;
use super::lod::Lod;
use crate::block::registry::BlockId;

/// The single far-skin LOD level: 64 m cells, 1024 m columns — strictly coarser
/// than the Zone-2 tile ring (asserted below), so the skin always sits outside
/// it.
pub(in crate::world) const SKIN_LOD: Lod = Lod(6);
const _: () = assert!(SKIN_LOD.0 > super::lod::TILE_LOD.0);

/// A far-skin column at grid `(x, z)` on the [`SKIN_LOD`] grid. No `y`: the
/// surface is a function of `(x, z)` alone, so a buried/sky column is
/// unrepresentable. World min corner is `(x, z) * SKIN_LOD.span()`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SkinColumn {
    pub x: i32,
    pub z: i32,
}

impl SkinColumn {
    pub fn origin_x(self) -> i32 {
        self.x * SKIN_LOD.span()
    }
    pub fn origin_z(self) -> i32 {
        self.z * SKIN_LOD.span()
    }
}

/// Sole owner of one GPU surface allocation — the skin's dual of
/// [`OwnedMesh`](super::OwnedMesh). Deliberately NOT `Copy`/`Clone` and with no
/// `Drop` (freeing needs `&mut Engine`): the only disposals are
/// [`free`](Self::free) (consumes `self`) or moving it into another
/// [`SkinState`].
pub(in crate::world) struct OwnedSurface(SurfaceHandle);

impl OwnedSurface {
    fn new(handle: SurfaceHandle) -> Self {
        Self(handle)
    }
    fn id(&self) -> SurfaceHandle {
        self.0
    }
    fn free(self, eng: &mut Engine) {
        eng.free_surface(self.0);
    }
}

/// Skin-column lifecycle: `Meshing` (in flight) or `Ready` (drawable). Fewer
/// states than a [`TileState`](super::lod::TileState): no `Air` (a column always
/// has a surface, so it is always born `Ready`) and no `Dirty` (never edited).
pub(in crate::world) enum SkinState {
    Meshing,
    /// Owns the GPU surface; draw offset = `column origin − camera`.
    Ready {
        mesh: OwnedSurface,
    },
}

impl SkinState {
    /// Wrap a freshly uploaded surface handle.
    pub(in crate::world) fn ready(handle: SurfaceHandle) -> Self {
        SkinState::Ready {
            mesh: OwnedSurface::new(handle),
        }
    }
    /// Free the column's GPU surface, if any. Consumes `self` (no double-free).
    pub(in crate::world) fn free(self, eng: &mut Engine) {
        if let SkinState::Ready { mesh } = self {
            mesh.free(eng);
        }
    }
    /// Drawable handle if `Ready`.
    pub(in crate::world) fn drawable(&self) -> Option<SurfaceHandle> {
        match self {
            SkinState::Ready { mesh } => Some(mesh.id()),
            SkinState::Meshing => None,
        }
    }
}

/// Cells along a column side — the skin's horizontal resolution, decoupled from
/// `CHUNK_SIZE`. A column is [`SKIN_LOD`]`.span()` (1024 m) wide regardless, so a
/// larger value samples the terrain height on a finer grid ([`SKIN_CELL`] m per
/// cell) for a sharper horizon silhouette, at the cost of more `height()` calls
/// and verts per column. Must divide the column span evenly. Tunable.
const SKIN_SUBDIV: i32 = 32;
const _: () = assert!(SKIN_LOD.span() % SKIN_SUBDIV == 0);

/// Number of cells / corners along a column side (`SKIN_SUBDIV` cells, `+1`
/// corners).
const N: i32 = SKIN_SUBDIV;

/// Metres per skin cell — the column span divided into `N` cells.
const SKIN_CELL: i32 = SKIN_LOD.span() / N;

/// The `(N+1)²` corner heights (world Y) sampled across a column. The `+1`
/// shares the far edge/corner with the next column so a shared boundary is
/// sampled identically on both sides — the determinism that makes interior
/// seams crack-free with no inter-column skirt.
struct Corners([i32; ((N + 1) * (N + 1)) as usize]);

/// Metres the whole skin is sunk *below* the true surface. Both the Zone-2 tiles
/// and the skin are depth-biased into the ground in their overlap band, so which
/// one shows is decided purely by geometry: pushing every skin corner down by
/// `SKIN_DROOP` guarantees the (higher) tile ring wins wherever they coexist,
/// while the skin still fills the horizon past the tiles. Tunable.
const SKIN_DROOP: i32 = 8;

/// Metres the perimeter apron skirt hangs below its edge. It exists only to give
/// the outer shell a solid silhouette under fog (no see-through gap at the column
/// boundary), so it need only out-reach the worst inter-column height step at
/// this LOD. Tunable.
const APRON: i32 = 24;

/// Row-major corner index on the `(N+1)²` grid.
const fn cidx(ix: i32, iz: i32) -> usize {
    (ix + iz * (N + 1)) as usize
}

/// Sample the `(N+1)²` corner heights on the exact integer world grid shared with
/// the four neighbouring columns (`ix=N` of column *k* is `ix=0` of column *k+1*),
/// then sink each by [`SKIN_DROOP`]. Because [`TerrainGenerator::height`] is
/// deterministic and sampled on identical integer coords, adjacent columns agree
/// bit-for-bit on their shared edge — interior seams are crack-free with no
/// inter-column skirt.
fn sample_corners<T: TerrainGenerator>(col: SkinColumn, terrain: &T) -> Corners {
    let (ox, oz) = (col.origin_x(), col.origin_z());
    let mut c = [0i32; ((N + 1) * (N + 1)) as usize];
    for iz in 0..=N {
        for ix in 0..=N {
            c[cidx(ix, iz)] = terrain.height(ox + ix * SKIN_CELL, oz + iz * SKIN_CELL) - SKIN_DROOP;
        }
    }
    Corners(c)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Fallback tint when the palette has no colour for a surface block (an empty
/// colour table in tests, or an out-of-range id) — a neutral mid-grey.
const SKIN_FALLBACK: [f32; 3] = [130.0, 130.0, 130.0];

/// The skin's base tint for a surface block: the palette's render colour for
/// that block. This is where the flat `SKIN_GREY` died — the horizon now carries
/// the real terrain palette (grass green, sand, snow), and the *fog-is-sky*
/// handoff does the rest: the fragment shader fogs distant skin toward the sky
/// colour, so the coloured backdrop dissolves into the sky at the horizon with
/// no grey seam and no separate sky-fill pass.
fn block_average_color(colors: &[Color], id: BlockId) -> [f32; 3] {
    match colors.get(id.0 as usize) {
        Some(c) => [c.r as f32, c.g as f32, c.b as f32],
        None => SKIN_FALLBACK,
    }
}

/// Vertical period (metres) of the topographic height band, and its brightness
/// amplitude. A gentle sinusoid of absolute world Y lightens/darkens the grey to
/// suggest contour relief under fog — pure detail cue, no geometry cost. Because
/// it keys off absolute (shared) corner Y, adjacent columns agree on their
/// shared edge, so the tint is seam-consistent. Tunable.
const SKIN_BAND_M: f32 = 96.0;
const SKIN_BAND_AMT: f32 = 0.12;

/// Per-quad Lambert factor from a cell's geometric normal, lit straight down. The
/// term is `abs`'d (the `surface3d` pipeline is `cull: NONE`, so a back-viewed
/// facet must still be lit, not black) and floored with a fixed ambient so
/// vertical apron walls (normal `.y ≈ 0`) stay a dim grey rather than pure black.
fn lambert(normal: [f32; 3]) -> f32 {
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let l = if len > 0.0 { (normal[1] / len).abs() } else { 0.0 };
    0.4 + 0.6 * l
}

/// A shaded skin vertex: the surface block's `base` palette colour scaled by the
/// per-quad `lambert` factor and a per-vertex height band keyed off absolute
/// world Y `y`.
fn skin_vertex(pos: [f32; 3], lambert: f32, base: [f32; 3]) -> SurfaceVertex {
    let band = 0.5 + 0.5 * (pos[1] * (std::f32::consts::TAU / SKIN_BAND_M)).sin();
    let f = lambert * (1.0 - SKIN_BAND_AMT + SKIN_BAND_AMT * band);
    let ch = |c: f32| (c * f).round().clamp(0.0, 255.0) as u8;
    SurfaceVertex { pos, color: [ch(base[0]), ch(base[1]), ch(base[2]), 255] }
}

/// Build a far-skin column's grey surface mesh off-thread from the generator
/// alone (height only — no block tables). A pure function of `(seed, x, z)`.
///
/// Column-LOCAL horizontal coords (`0..=span`) and ABSOLUTE (drooped) world-Y:
/// the caller draws the column at camera-relative offset `(ox−cam.x, 0, oz−cam.z)`
/// with scale 1.0. Always non-empty (`N²` top cells) → the column is born `Ready`.
pub fn build_skin_mesh<T: TerrainGenerator>(col: SkinColumn, terrain: &T, colors: &[Color]) -> SurfaceData {
    let (ox, oz) = (col.origin_x(), col.origin_z());
    let corners = sample_corners(col, terrain);
    let y = |ix: i32, iz: i32| corners.0[cidx(ix, iz)];
    // Column-local horizontal (0..=span), absolute drooped Y.
    let pos =
        |ix: i32, iz: i32| [(ix * SKIN_CELL) as f32, y(ix, iz) as f32, (iz * SKIN_CELL) as f32];
    // Per-corner base tint from the terrain surface block's palette colour,
    // sampled on the same shared integer grid as the heights (so adjacent columns
    // agree on their shared edge — seam-consistent colour, like the height band).
    let base = |ix: i32, iz: i32| {
        block_average_color(colors, terrain.surface_at(ox + ix * SKIN_CELL, oz + iz * SKIN_CELL))
    };

    let mut data = SurfaceData::new();

    // Top surface: 2 tris/cell, wound so the geometric normal points up (+Y).
    for iz in 0..N {
        for ix in 0..N {
            let p = [pos(ix, iz), pos(ix, iz + 1), pos(ix + 1, iz + 1), pos(ix + 1, iz)];
            let c = [base(ix, iz), base(ix, iz + 1), base(ix + 1, iz + 1), base(ix + 1, iz)];
            let n = cross(sub(p[1], p[0]), sub(p[3], p[0]));
            let s = lambert(n);
            data.quad(std::array::from_fn(|i| skin_vertex(p[i], s, c[i])));
        }
    }

    // Perimeter apron: a short vertical skirt down the 4 OUTER edges only. Along
    // an interior boundary the neighbour column drops the identical skirt to the
    // same floor (shared corners), so those back-to-back quads are never seen;
    // only the true outer shell boundary is visible.
    let mut skirt = |a: [f32; 3], b: [f32; 3], ca: [f32; 3], cb: [f32; 3]| {
        let floor = a[1].min(b[1]) - APRON as f32;
        let (ba, bb) = ([a[0], floor, a[2]], [b[0], floor, b[2]]);
        let n = cross(sub(b, a), sub(ba, a));
        let s = lambert(n);
        let quad = [a, b, bb, ba];
        let col = [ca, cb, cb, ca];
        data.quad(std::array::from_fn(|i| skin_vertex(quad[i], s, col[i])));
    };
    for i in 0..N {
        skirt(pos(i, 0), pos(i + 1, 0), base(i, 0), base(i + 1, 0));
        skirt(pos(i, N), pos(i + 1, N), base(i, N), base(i + 1, N));
        skirt(pos(0, i), pos(0, i + 1), base(0, i), base(0, i + 1));
        skirt(pos(N, i), pos(N, i + 1), base(N, i), base(N, i + 1));
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::registry::{AIR, BlockId};

    /// Flat generator — constant height. Makes the skin's geometry exactly
    /// predictable (every corner at `h − SKIN_DROOP`, every top normal +Y).
    struct FlatGen {
        h: i32,
    }
    /// Planar-slope generator — `height = a·wx + b·wz + c`. Exercises slanted
    /// cells (non-trivial per-cell normals) while staying perfectly deterministic
    /// on the shared integer grid (crack-free by construction).
    struct SlopedGen {
        a: i32,
        b: i32,
        c: i32,
    }
    impl TerrainGenerator for FlatGen {
        fn height(&self, _: i32, _: i32) -> i32 {
            self.h
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            AIR
        }
        fn deep(&self) -> BlockId {
            AIR
        }
    }
    impl TerrainGenerator for SlopedGen {
        fn height(&self, wx: i32, wz: i32) -> i32 {
            self.a * wx + self.b * wz + self.c
        }
        fn surface_at(&self, _: i32, _: i32) -> BlockId {
            AIR
        }
        fn deep(&self) -> BlockId {
            AIR
        }
    }

    /// Representative columns: origin, negative grid (origin sign), and a far cell
    /// (large absolute coords exercise the f64→i32 integer sampling).
    const COLS: [SkinColumn; 3] = [
        SkinColumn { x: 0, z: 0 },
        SkinColumn { x: -1, z: 2 },
        SkinColumn { x: 37, z: -19 },
    ];

    fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        super::cross(a, b)
    }

    /// (a) CRACK-FREE: the shared edge of two x-adjacent (and z-adjacent) columns
    /// is sampled bit-identically from both sides — no seam, no inter-column skirt.
    #[test]
    fn shared_edges_are_bit_identical() {
        let gens = [SlopedGen { a: 3, b: -7, c: 11 }, SlopedGen { a: 0, b: 0, c: 64 }];
        for g in &gens {
            for &col in &COLS {
                let east = SkinColumn { x: col.x + 1, z: col.z };
                let (l, r) = (sample_corners(col, g), sample_corners(east, g));
                for iz in 0..=N {
                    assert_eq!(l.0[cidx(N, iz)], r.0[cidx(0, iz)], "x-seam mismatch");
                }
                let north = SkinColumn { x: col.x, z: col.z + 1 };
                let (l, u) = (sample_corners(col, g), sample_corners(north, g));
                for ix in 0..=N {
                    assert_eq!(l.0[cidx(ix, N)], u.0[cidx(ix, 0)], "z-seam mismatch");
                }
            }
        }
    }

    /// (b) OUTWARD WINDING: every top surface quad's geometric normal points up
    /// (+Y). The tops are emitted before the aprons, so they are the first
    /// `N²` quads (aprons are vertical walls, `.y ≈ 0`, excluded from this check).
    #[test]
    fn top_quads_wind_upward() {
        let g = SlopedGen { a: 2, b: 5, c: 30 };
        for &col in &COLS {
            let data = build_skin_mesh(col, &g, &[]);
            let tops = (N * N) as usize;
            for q in data.verts().chunks_exact(4).take(tops) {
                let p: Vec<[f32; 3]> = q.iter().map(|v| v.pos).collect();
                let n = cross3(sub(p[1], p[0]), sub(p[3], p[0]));
                assert!(n[1] > 0.0, "top quad faces down: normal {n:?}");
            }
        }
    }

    /// (c) NON-EMPTY + AABB: a column always carries geometry, and its vertex Y
    /// AABB brackets the drooped surface height (top verts at `h−SKIN_DROOP`, apron
    /// floor below it).
    #[test]
    fn non_empty_and_aabb_brackets_height() {
        for h in [-50, 0, 200] {
            let g = FlatGen { h };
            for &col in &COLS {
                let data = build_skin_mesh(col, &g, &[]);
                assert!(!data.is_empty(), "a column always has a surface");
                let ys: Vec<f32> = data.verts().iter().map(|v| v.pos[1]).collect();
                let (lo, hi) = ys.iter().fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
                let surf = (h - SKIN_DROOP) as f32;
                assert_eq!(hi, surf, "top verts sit at the drooped surface");
                assert_eq!(lo, surf - APRON as f32, "apron floor is APRON below");
            }
        }
    }
}
