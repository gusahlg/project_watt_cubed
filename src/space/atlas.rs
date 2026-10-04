//! The atlas of a round body (guide §§5.3, 10.4–10.6): six equiangular cube-sphere charts per depth
//! band (angular resolution halving as the radius halves), a transition shell, and a Cartesian core.
//! Every chart cell has an ordinary `i32` **storage** address in a box of the reserved storage region
//! beyond the physical universe (`|x| ≥ 1.1e9`), with storage `+Y` along the chart's up (outward, or
//! inward for an inner surface). So streaming, light, meshing, edits, saves and the wire handle chart
//! cells as plain cells; only the embedding (storage → physical) and the glue at chart edges know
//! about curvature.
//!
//! Glue: a storage cell just outside a chart's box (within [`GLUE`] cells) is the cell of the
//! neighbouring chart that holds the same physical point; cells conform face to face across chart
//! edges, so the first ring is exact. Beyond two edges at once (the eight valence-3 corners) and
//! across band interfaces (1 : 4) the glue is approximate: those are the declared exceptional
//! regions of the stage-0 report.

use glam::DVec3;

use super::chart::{basis, tan_quarter, Map};
use crate::coord::Face;

/// First storage x of the reserved region (beyond the physical border, inside i32 chunk math).
pub const STORAGE_X0: i64 = 1_100_000_000;
/// Empty storage cells kept between neighbouring boxes (so glue reads never hit another box).
const GAP: i64 = 64;

fn snap16(v: i64) -> i64 {
    v.div_euclid(16) * 16
}

/// How many depth bands a body of datum radius `radius` grows before the core.
fn band_count(radius: i64) -> u32 {
    let mut count = 0u32;
    let mut r = radius / 2;
    while r > 512 {
        count += 1;
        r /= 2;
    }
    count
}

/// Band-0 chart edge, in cells, for a full atlas of datum radius `radius`.
///
/// A multiple of `2^(bands + 1)` chunks, so every deeper band and the transition divide evenly.
pub fn surface_n(radius: i64) -> i64 {
    let unit = 1i64 << (band_count(radius) + 5);
    ((std::f64::consts::FRAC_PI_2 * radius as f64) / unit as f64).round().max(1.0) as i64 * unit
}
/// How far outside a box the glue answers.
pub const GLUE: i64 = 2;

/// The six faces in storage order.
pub const FACES: [Face; 6] = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];

/// A patch of an atlas.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Patch {
    /// A shell chart of depth band `band` (0 = outermost).
    Shell { band: u8, face: Face },
    /// A chart of the transition shell between the innermost band and the core.
    Transition { face: Face },
    /// The Cartesian core cube.
    Core,
}

/// How the cells of a virtual neighbour chunk (outside every box) map onto the real chunk across a
/// seam: local cell `l` of the virtual chunk is cell `base + Σ cols[a]·(l[a] − near[a])` of the
/// real one (columns are signed unit steps across a chart seam, possibly halved across a band
/// interface — then the map is approximate, an exceptional region).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Remap {
    pub cols: [[i64; 3]; 3],
    pub near: [i64; 3],
    pub base: [i64; 3],
}

impl Remap {
    /// The real chunk's local cell for virtual local cell `l` (may leave the chunk for cells far
    /// from the seam; callers read only the first layer or two).
    pub fn apply(&self, l: [i64; 3]) -> [i64; 3] {
        std::array::from_fn(|k| self.base[k] + (0..3).map(|a| self.cols[a][k] * (l[a] - self.near[a])).sum::<i64>())
    }
}

/// A point's place in an atlas: its patch, continuous storage coordinates and the local Jacobian.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Local {
    pub patch: Patch,
    pub storage: DVec3,
    /// `∂physical/∂storage` (columns: storage x, y, z in physical space).
    pub jacobian: glam::DMat3,
}

impl Local {
    /// The rotation part of the Jacobian (storage axes → physical), orthonormalised with the chart's
    /// up (storage y) kept exact.
    pub fn rotation(&self) -> glam::DMat3 {
        let y = self.jacobian.y_axis.normalize();
        let x = (self.jacobian.x_axis - y * self.jacobian.x_axis.dot(y)).normalize();
        glam::DMat3::from_cols(x, y, x.cross(y).normalize())
    }
}

/// One depth band: `n × n` cells per face, radial cells `[r_lo, r_hi)` (one block each).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Band {
    pub n: i64,
    pub r_lo: i64,
    pub r_hi: i64,
    /// Storage origin of each face's box, in [`FACES`] order.
    pub origin: [[i64; 3]; 6],
}

/// A round body's atlas.
#[derive(Clone, PartialEq, Debug)]
pub struct Atlas {
    pub centre: DVec3,
    /// Datum radius.
    pub radius: i64,
    /// Storage `+Y` points toward the centre (an inner surface, like the Hollow's).
    pub inward: bool,
    pub bands: Vec<Band>,
    /// The transition shell and the core (absent for a hollow shell's atlas).
    pub inner: Option<Inner>,
    /// The relaxed shape's departure from the sphere of `radius` (physics decides the shape, the
    /// layout is fitted to it); `None` for a spherical datum.
    pub datum: Option<std::sync::Arc<crate::space::datum::DatumField>>,
}

/// The transition shell and the Cartesian core below an atlas's innermost band.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Inner {
    /// Transition shell: chart resolution, outer radius, radial layers, per-face storage origins.
    pub t_n: i64,
    pub t_r: i64,
    pub t_layers: i64,
    pub t_origin: [[i64; 3]; 6],
    /// Core cube half-size (cells of one block) and its storage origin (its minimum corner).
    pub core_half: i64,
    pub core_origin: [i64; 3],
}

/// The physical centre of storage cell `c` in whichever atlas holds it (`None` for a physical cell
/// or storage outside every box): where an edit or a reach check really acts.
pub fn embed_cell(atlases: &[std::sync::Arc<Atlas>], c: (i32, i32, i32)) -> Option<DVec3> {
    if (c.0 as i64) < STORAGE_X0 {
        return None;
    }
    atlases.iter().find_map(|a| {
        let (patch, l) = a.locate([c.0 as i64, c.1 as i64, c.2 as i64])?;
        Some(a.embed(patch, DVec3::new(l[0] as f64 + 0.5, l[1] as f64 + 0.5, l[2] as f64 + 0.5)))
    })
}

fn face_index(f: Face) -> usize {
    FACES.iter().position(|&g| g == f).expect("a face")
}

impl Atlas {
    /// The atlas of a body of datum radius `radius` whose cells reach `top` (≥ radius, the relief)
    /// and go down to the centre. Boxes begin at storage x `x0` (chunk-aligned, ≥ [`STORAGE_X0`]).
    pub fn new(centre: DVec3, radius: i64, top: i64, inward: bool, x0: i64) -> Self {
        // Bands halve the radius and the angular resolution together until the inner radius would
        // drop under 512; the surface resolution is a multiple of 2^(bands + 1) so every band (and
        // the transition at half the last band) divides evenly.
        // Everything is chunk aligned (radii, resolutions and storage origins are multiples of 16),
        // so chart seams fall on chunk boundaries and a chunk's neighbour across a seam is a whole
        // chunk of the neighbouring chart.
        let count = band_count(radius);
        let mut n = surface_n(radius);
        // Boxes sit side by side along storage x (faces along z), never stacked: every storage
        // column belongs to at most one box, so a column's skylight ceiling is that box's surface.
        let mut bands = Vec::new();
        let (mut r_hi, mut r_lo) = (snap16(top + 15), snap16(radius / 2));
        let mut x = x0;
        for _ in 0..count {
            let origin = std::array::from_fn(|f| [x, 0, f as i64 * (n + GAP)]);
            bands.push(Band { n, r_lo, r_hi, origin });
            x = snap16(x + n + GAP + 15);
            r_hi = r_lo;
            r_lo = snap16(r_lo / 2);
            n /= 2;
        }
        // Transition shell from the core cube (half-size a) out to r_hi, at half the last band's
        // resolution so the core cube's corners stay inside the sphere.
        let t_n = (n / 2).max(32);
        let core_half = t_n / 2;
        let t_r = r_hi;
        let t_layers = snap16((t_r - core_half).max(16) + 15);
        let t_origin = std::array::from_fn(|f| [x, 0, f as i64 * (t_n + GAP)]);
        x = snap16(x + t_n + GAP + 15);
        let core_origin = [x, 0, 0];
        let inner = Some(Inner { t_n, t_r, t_layers, t_origin, core_half, core_origin });
        let atlas = Self { centre, radius, inward, bands, inner, datum: None };
        atlas.check_room(x0);
        atlas
    }

    /// The atlas of a hollow shell's surface: one band of cells between radii `r_lo` and `r_hi`
    /// (sized for one block of arc at the datum radius `radius`), no core. `inward` turns it into an
    /// inner surface (storage up toward the centre). Boxes begin at storage x `x0`.
    pub fn shell(centre: DVec3, radius: i64, r_lo: i64, r_hi: i64, inward: bool, x0: i64) -> Self {
        let n = ((std::f64::consts::FRAC_PI_2 * radius as f64) / 16.0).round().max(1.0) as i64 * 16;
        let (r_lo, r_hi) = (snap16(r_lo), snap16(r_hi + 15));
        let origin = std::array::from_fn(|f| [x0, 0, f as i64 * (n + GAP)]);
        let atlas = Self { centre, radius, inward, bands: vec![Band { n, r_lo, r_hi, origin }], inner: None, datum: None };
        atlas.check_room(x0);
        atlas
    }

    /// Storage x where this atlas's boxes begin.
    pub fn x0(&self) -> i64 {
        self.bands[0].origin[0][0]
    }

    /// Storage x occupied by the boxes, from [`x0`](Self::x0) through the end of the last one
    /// (no trailing gap).
    pub fn x_span(&self) -> i64 {
        let x0 = self.x0();
        self.patches().map(|p| {
            let (o, s) = self.storage_box(p);
            o[0] + s[0]
        }).max().unwrap_or(x0) - x0
    }

    /// Storage x where the next atlas may start: this one's boxes, then the gap, chunk-aligned.
    pub fn next_x(&self) -> i64 {
        snap16(self.x0() + self.x_span() + GAP + 15)
    }

    fn check_room(&self, x0: i64) {
        assert!(x0 >= STORAGE_X0 && x0 % 16 == 0, "storage origin {x0} is outside the reserved region");
        let end = x0 + self.x_span();
        assert!(end < i32::MAX as i64, "a round body of radius {} overflows i32 storage ({end})", self.radius);
    }

    /// The storage box of a patch: `(min, size)`.
    pub fn storage_box(&self, patch: Patch) -> ([i64; 3], [i64; 3]) {
        match patch {
            Patch::Shell { band, face } => {
                let b = &self.bands[band as usize];
                (b.origin[face_index(face)], [b.n, b.r_hi - b.r_lo, b.n])
            }
            Patch::Transition { face } => {
                let i = self.inner.expect("a transition needs an inner part");
                (i.t_origin[face_index(face)], [i.t_n, i.t_layers, i.t_n])
            }
            Patch::Core => {
                let i = self.inner.expect("a core needs an inner part");
                (i.core_origin, [2 * i.core_half; 3])
            }
        }
    }

    /// Datum offset (blocks) above the sphere of `radius` at continuous column `(i, j)` of a shell
    /// patch (0 elsewhere, or without a datum field).
    pub fn datum_offset(&self, patch: Patch, i: f64, j: f64) -> f64 {
        let (Some(field), Patch::Shell { band, face }) = (&self.datum, patch) else { return 0.0 };
        let n = self.bands[band as usize].n as f64;
        // The same column parameters `embed` uses (an inward chart flips x).
        let i = if self.inward { n - i } else { i };
        field.offset(face_index(face), -1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n)
    }

    /// Every patch.
    pub fn patches(&self) -> impl Iterator<Item = Patch> + '_ {
        let shells = (0..self.bands.len()).flat_map(|b| FACES.map(move |face| Patch::Shell { band: b as u8, face }));
        let inner = self.inner.is_some();
        shells
            .chain(FACES.map(|face| Patch::Transition { face }).into_iter().filter(move |_| inner))
            .chain(std::iter::once(Patch::Core).filter(move |_| inner))
    }

    /// The patch holding storage cell `s`, with the cell's local coordinates in that patch.
    pub fn locate(&self, s: [i64; 3]) -> Option<(Patch, [i64; 3])> {
        self.patches().find_map(|p| {
            let (o, size) = self.storage_box(p);
            let l = [s[0] - o[0], s[1] - o[1], s[2] - o[2]];
            (0..3).all(|a| l[a] >= 0 && l[a] < size[a]).then_some((p, l))
        })
    }

    /// The physical point of continuous local coordinates in a patch (cell corners at integers;
    /// local x along the face's `t_u`, y up the chart, z along `t_v`).
    pub fn embed(&self, patch: Patch, l: DVec3) -> DVec3 {
        let map = Map::Equiangular;
        let radial = |face: Face, n: i64, r: f64, i: f64, j: f64| {
            let step = 2.0 / n as f64;
            let d = map.dir(-1.0 + i * step, -1.0 + j * step);
            let (tu, nn, tv) = basis(face);
            (tu * d.x + nn * d.y + tv * d.z) * r
        };
        match patch {
            Patch::Shell { band, face } => {
                let b = &self.bands[band as usize];
                // Inward charts flip x with y so storage stays right-handed (det J > 0).
                let (r, i) = if self.inward { (b.r_hi as f64 - l.y, b.n as f64 - l.x) } else { (b.r_lo as f64 + l.y, l.x) };
                self.centre + radial(face, b.n, r, i, l.z)
            }
            Patch::Transition { face } => {
                let i = self.inner.expect("a transition needs an inner part");
                let step = 2.0 / i.t_n as f64;
                let lx = if self.inward { i.t_n as f64 - l.x } else { l.x };
                let (xi, eta) = (-1.0 + lx * step, -1.0 + l.z * step);
                let (tu, nn, tv) = basis(face);
                let cube = (tu * tan_quarter(xi) + nn + tv * tan_quarter(eta)) * i.core_half as f64;
                let sphere = radial(face, i.t_n, i.t_r as f64, lx, l.z);
                let t = l.y / i.t_layers as f64;
                let t = if self.inward { 1.0 - t } else { t };
                self.centre + cube + (sphere - cube) * t
            }
            Patch::Core => self.centre + l - DVec3::splat(self.inner.expect("a core needs an inner part").core_half as f64),
        }
    }

    /// The patch and continuous local coordinates of a physical point, if the atlas covers it.
    pub fn find(&self, p: DVec3) -> Option<(Patch, DVec3)> {
        let rel = p - self.centre;
        let r = rel.length();
        let face = Face::from_dominant(rel);
        let (tu, nn, tv) = basis(face);
        let local = DVec3::new(rel.dot(tu), rel.dot(nn), rel.dot(tv));
        if self.bands.first().is_some_and(|b| r >= b.r_hi as f64) {
            return None;
        }
        for (bi, b) in self.bands.iter().enumerate() {
            if r >= b.r_lo as f64 && r < b.r_hi as f64 {
                let (xi, eta) = Map::Equiangular.inverse(local);
                let step = 2.0 / b.n as f64;
                let (y, i) = if self.inward { (b.r_hi as f64 - r, b.n as f64 - (xi + 1.0) / step) } else { (r - b.r_lo as f64, (xi + 1.0) / step) };
                return Some((Patch::Shell { band: bi as u8, face }, DVec3::new(i, y, (eta + 1.0) / step)));
            }
        }
        // Inside the transition or the core: the core cube first (L∞ test), else Newton on the
        // transition's lerp map. A shell atlas covers nothing below its band.
        let inner = self.inner?;
        if self.bands.last().is_some_and(|b| r < b.r_lo as f64) && r >= inner.t_r as f64 {
            return None;
        }
        let a = inner.core_half as f64;
        if rel.abs().max_element() < a {
            return Some((Patch::Core, rel + DVec3::splat(a)));
        }
        let patch = Patch::Transition { face };
        let (xi, eta) = Map::Equiangular.inverse(local);
        let step = 2.0 / inner.t_n as f64;
        let i = if self.inward { inner.t_n as f64 - (xi + 1.0) / step } else { (xi + 1.0) / step };
        let mut l = DVec3::new(i, inner.t_layers as f64 * 0.5, (eta + 1.0) / step);
        for _ in 0..32 {
            let f = self.embed(patch, l) - p;
            if f.length() < 1e-9 {
                break;
            }
            let h = 1e-4;
            let col = |d: DVec3| (self.embed(patch, l + d * h) - self.embed(patch, l - d * h)) / (2.0 * h);
            let j = glam::DMat3::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z));
            if j.determinant().abs() < 1e-18 {
                break;
            }
            l -= j.inverse() * f;
        }
        Some((patch, l))
    }

    /// The physical point of continuous storage coordinates inside `patch`'s box.
    pub fn embed_storage(&self, patch: Patch, s: DVec3) -> DVec3 {
        let (o, _) = self.storage_box(patch);
        self.embed(patch, s - DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64))
    }

    /// The patch around physical point `p`: where `p` sits in storage and the Jacobian
    /// `∂physical/∂storage` there (the local affine frame motion and picking run in).
    pub fn local(&self, p: DVec3) -> Option<Local> {
        let (patch, l) = self.find(p)?;
        let (o, _) = self.storage_box(patch);
        let storage = l + DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64);
        let h = 1e-3;
        let col = |d: DVec3| (self.embed(patch, l + d * h) - self.embed(patch, l - d * h)) / (2.0 * h);
        Some(Local { patch, storage, jacobian: glam::DMat3::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z)) })
    }

    /// The storage cell of a patch-local cell.
    pub fn storage(&self, patch: Patch, l: [i64; 3]) -> [i64; 3] {
        let (o, _) = self.storage_box(patch);
        [o[0] + l[0], o[1] + l[1], o[2] + l[2]]
    }

    /// The storage cell holding physical point `p`.
    pub fn storage_of(&self, p: DVec3) -> Option<[i64; 3]> {
        let (patch, l) = self.find(p)?;
        let (_, size) = self.storage_box(patch);
        let cell: [i64; 3] = std::array::from_fn(|a| (l[a].floor() as i64).clamp(0, size[a] - 1));
        Some(self.storage(patch, cell))
    }

    /// What storage cell `s` (inside a box, or within [`GLUE`] cells outside one) holds: itself if it
    /// is inside a box, else the neighbouring patch's cell at the same physical point.
    pub fn glue(&self, s: [i64; 3]) -> Option<[i64; 3]> {
        if self.locate(s).is_some() {
            return Some(s);
        }
        // Find the box it hangs off, extend that patch's map to the cell centre, and look it up.
        for p in self.patches() {
            let (o, size) = self.storage_box(p);
            let l: [i64; 3] = std::array::from_fn(|a| s[a] - o[a]);
            let outside: i64 = (0..3).map(|a| (-l[a]).max(l[a] - size[a] + 1).max(0)).max().unwrap_or(0);
            if outside == 0 || outside > GLUE {
                continue;
            }
            let centre = self.embed(p, DVec3::new(l[0] as f64 + 0.5, l[1] as f64 + 0.5, l[2] as f64 + 0.5));
            return self.storage_of(centre);
        }
        None
    }

    /// The chunk across face `(axis, dir)` of storage chunk `c` (chunk coordinates) when that
    /// neighbour lies outside every box (a chart seam or a band interface): the neighbouring
    /// patch's chunk and the map from this side's (virtual) neighbour cells to its cells. `None`
    /// when the plain neighbour is itself inside a box, or nothing lies there (beyond the top, the
    /// eight corners).
    pub fn chunk_across(&self, c: [i64; 3], axis: usize, dir: i64) -> Option<([i64; 3], Remap)> {
        let mut next = c;
        next[axis] += dir;
        let probe = |l: [i64; 3]| [next[0] * 16 + l[0], next[1] * 16 + l[1], next[2] * 16 + l[2]];
        if self.locate(probe([8, 8, 8])).is_some() {
            return None;
        }
        // Where do three cells of the virtual neighbour chunk land? The first gives the chunk, the
        // other two the axis permutation (cells conform across seams, so the map is a signed
        // permutation within the chunk, up to a 1 : 2 scale across band interfaces).
        let near: [i64; 3] = std::array::from_fn(|a| if a == axis { if dir > 0 { 0 } else { 15 } } else { 7 });
        let g0 = self.glue(probe(near))?;
        let step = |a: usize| {
            let mut l = near;
            l[a] += 1;
            self.glue(probe(l))
        };
        let delta = |g: [i64; 3]| [g[0] - g0[0], g[1] - g0[1], g[2] - g0[2]];
        let cols: [[i64; 3]; 3] = [delta(step(0)?), delta(step(1)?), delta(step(2)?)];
        let chunk = [g0[0].div_euclid(16), g0[1].div_euclid(16), g0[2].div_euclid(16)];
        let base = [g0[0] - chunk[0] * 16, g0[1] - chunk[1] * 16, g0[2] - chunk[2] * 16];
        Some((chunk, Remap { cols, near, base }))
    }

    /// The eight physical corners of the storage chunk whose minimum cell is `s` (a chunk of 16³
    /// cells, all inside one patch box): the cage the renderer interpolates.
    pub fn chunk_cage(&self, s: [i64; 3]) -> Option<[DVec3; 8]> {
        let (patch, l) = self.locate(s)?;
        Some(std::array::from_fn(|c| {
            let d = [(c & 1) as i64 * 16, (c >> 1 & 1) as i64 * 16, (c >> 2 & 1) as i64 * 16];
            self.embed(patch, DVec3::new((l[0] + d[0]) as f64, (l[1] + d[1]) as f64, (l[2] + d[2]) as f64))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas() -> Atlas {
        Atlas::new(DVec3::new(4.0e8, -1.2e8, 3.0e8), 200_000, 202_048, false, STORAGE_X0)
    }

    #[test]
    fn bands_halve_down_to_a_core_inside_the_sphere() {
        let a = atlas();
        assert!(a.bands.len() >= 4, "{} bands", a.bands.len());
        for w in a.bands.windows(2) {
            assert_eq!(w[0].r_lo, w[1].r_hi, "bands meet");
            assert_eq!(w[0].n, w[1].n * 2, "resolution halves");
        }
        let last = a.bands.last().unwrap();
        let inner = a.inner.unwrap();
        assert_eq!(inner.t_r, last.r_lo);
        assert!((inner.core_half as f64) * 3f64.sqrt() < inner.t_r as f64, "core corners inside the transition sphere");
        // Storage boxes are disjoint and beyond the physical border.
        let boxes: Vec<_> = a.patches().map(|p| a.storage_box(p)).collect();
        for (i, (o, s)) in boxes.iter().enumerate() {
            assert!(o[0] >= STORAGE_X0 && o[0] + s[0] < i32::MAX as i64);
            for (oo, ss) in &boxes[i + 1..] {
                let overlap = (0..3).all(|k| o[k] < oo[k] + ss[k] + GAP && oo[k] < o[k] + s[k] + GAP);
                assert!(!overlap, "boxes {o:?}/{s:?} and {oo:?}/{ss:?} too close");
            }
        }
    }

    #[test]
    fn no_two_boxes_share_a_storage_column() {
        let a = atlas();
        let boxes: Vec<_> = a.patches().map(|p| a.storage_box(p)).collect();
        for (i, (o, s)) in boxes.iter().enumerate() {
            for (oo, ss) in &boxes[i + 1..] {
                let overlap = [0, 2].iter().all(|&k| o[k] < oo[k] + ss[k] && oo[k] < o[k] + s[k]);
                assert!(!overlap, "boxes {o:?}/{s:?} and {oo:?}/{ss:?} share columns");
            }
        }
    }

    #[test]
    fn find_inverts_embed_in_every_kind_of_patch() {
        let a = atlas();
        for p in a.patches() {
            let (_, size) = a.storage_box(p);
            for f in [0.13, 0.5, 0.87] {
                let l = DVec3::new(size[0] as f64 * f, size[1] as f64 * (1.0 - f) * 0.9 + 0.3, size[2] as f64 * (f * 0.7 + 0.1));
                let x = a.embed(p, l);
                let (q, m) = a.find(x).expect("covered");
                assert_eq!(q, p, "{l} in {p:?} found in {q:?}");
                assert!((m - l).length() < 1e-5, "{p:?}: {l} -> {m}");
            }
        }
    }

    #[test]
    fn glue_across_a_chart_edge_is_the_neighbour_cell_sharing_the_face() {
        let a = atlas();
        let b = a.bands[0];
        let top = Patch::Shell { band: 0, face: Face::PosY };
        // A cell just beyond the +u edge of the +Y chart, mid-edge, at the datum radius (where cells
        // are one block of arc; at the band's inner radius R/2 they are half that).
        let k = a.radius - b.r_lo;
        let s = a.storage(top, [b.n, k, b.n / 2]);
        let g = a.glue(s).expect("glued");
        let (patch, l) = a.locate(g).expect("into a box");
        assert!(matches!(patch, Patch::Shell { band: 0, face } if face != Face::PosY), "{patch:?}");
        // The glued cell's centre and the edge cell's centre on +Y are one block apart: neighbours.
        let inside = a.embed(top, DVec3::new(b.n as f64 - 0.5, k as f64 + 0.5, b.n as f64 / 2.0 + 0.5));
        let other = a.embed(patch, DVec3::new(l[0] as f64 + 0.5, l[1] as f64 + 0.5, l[2] as f64 + 0.5));
        assert!(((inside - other).length() - 1.0).abs() < 0.05, "{} via {patch:?} {l:?} from {s:?} -> {g:?}", (inside - other).length());
    }

    #[test]
    fn storage_is_right_handed_outward_and_inward() {
        for inward in [false, true] {
            let a = Atlas::new(DVec3::new(1.0e8, 0.0, 0.0), 50_000, 52_048, inward, STORAGE_X0);
            for p in a.patches() {
                let (_, size) = a.storage_box(p);
                let l = DVec3::new(size[0] as f64 * 0.4, size[1] as f64 * 0.6, size[2] as f64 * 0.3);
                let x = a.embed(p, l);
                let loc = a.local(x).expect("covered");
                assert!(loc.jacobian.determinant() > 0.0, "{p:?} inward={inward} mirrored");
                let rot = loc.rotation();
                assert!((rot.determinant() - 1.0).abs() < 1e-9 && (rot * rot.transpose()).abs_diff_eq(glam::DMat3::IDENTITY, 1e-9));
            }
            // Outward charts point storage +y away from the centre, inward ones toward it.
            let s = a.local(a.centre + DVec3::new(0.0, 51_000.0, 0.0)).unwrap();
            assert_eq!(s.jacobian.y_axis.y > 0.0, !inward);
        }
    }

    #[test]
    fn a_shell_atlas_covers_only_its_shell_both_ways() {
        let c = DVec3::new(-3.0e8, 1.0e8, 2.0e8);
        for inward in [false, true] {
            let a = Atlas::shell(c, 60_000, 59_000, 61_000, inward, STORAGE_X0);
            assert!(a.inner.is_none() && a.bands.len() == 1);
            assert!(a.find(c + DVec3::new(0.0, 60_000.0, 0.0)).is_some());
            assert!(a.find(c + DVec3::new(0.0, 30_000.0, 0.0)).is_none(), "the cavity is not covered");
            let loc = a.local(c + DVec3::new(60_000.0, 1.0, 2.0)).unwrap();
            assert!(loc.jacobian.determinant() > 0.0);
            assert_eq!(loc.jacobian.y_axis.x > 0.0, !inward, "inner surfaces face the centre");
        }
    }

    #[test]
    fn everything_is_chunk_aligned() {
        let a = atlas();
        for p in a.patches() {
            let (o, size) = a.storage_box(p);
            assert!(o.iter().all(|v| v % 16 == 0), "{p:?} origin {o:?}");
            assert!(size[0] % 16 == 0 && size[2] % 16 == 0 && size[1] % 16 == 0, "{p:?} size {size:?}");
        }
        assert_eq!(a.x0(), STORAGE_X0);
        assert!(a.x_span() > 0 && a.x_span() % 16 == 0);
        assert_eq!(a.next_x(), a.x0() + a.x_span() + GAP);
        assert!(a.next_x() < i32::MAX as i64);
    }

    #[test]
    fn the_chunk_across_a_seam_is_a_whole_chunk_with_a_signed_permutation() {
        let a = atlas();
        let b = a.bands[0];
        let top = Patch::Shell { band: 0, face: Face::PosY };
        let s = a.storage(top, [b.n - 16, a.radius - b.r_lo, b.n / 2]);
        let c = [s[0].div_euclid(16), s[1].div_euclid(16), s[2].div_euclid(16)];
        let (other, remap) = a.chunk_across(c, 0, 1).expect("a seam on +x");
        assert!(a.locate([other[0] * 16, other[1] * 16, other[2] * 16]).is_some(), "lands in a box");
        for col in remap.cols {
            assert_eq!(col.iter().map(|v| v.abs()).sum::<i64>(), 1, "unit step {:?}", remap.cols);
        }
        // Every first-layer cell of the virtual neighbour maps to the glued cell.
        for (y, z) in [(0, 0), (5, 9), (15, 15)] {
            let l = [0, y, z];
            let real = remap.apply(l);
            let glued = a.glue([(c[0] + 1) * 16 + l[0], c[1] * 16 + l[1], c[2] * 16 + l[2]]).unwrap();
            assert_eq!([other[0] * 16 + real[0], other[1] * 16 + real[1], other[2] * 16 + real[2]], glued);
        }
        // Inside a box there is no seam.
        assert!(a.chunk_across(c, 2, 1).is_none());
    }

    #[test]
    fn a_surface_chunk_cage_is_nearly_a_unit_cube_scaled_by_16() {
        let a = atlas();
        let b = a.bands[0];
        // A chunk at the datum radius (cells there are one block of arc).
        let s = a.storage(Patch::Shell { band: 0, face: Face::PosZ }, [b.n / 2, a.radius - b.r_lo, b.n / 2]);
        let c = a.chunk_cage(s).unwrap();
        // Tangential edges: 16 cells of one quarter-circle arc / n each (n is rounded for alignment).
        let arc = std::f64::consts::FRAC_PI_2 * a.radius as f64 / b.n as f64 * 16.0;
        for (i, j, want) in [(0, 1, arc), (0, 2, 16.0), (0, 4, arc)] {
            assert!(((c[i] - c[j]).length() - want).abs() < 0.01, "{} vs {want}", (c[i] - c[j]).length());
        }
    }
}
