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

use super::chart::{basis, Map};
use crate::coord::Face;

/// First storage x of the reserved region (beyond the physical border, inside i32 chunk math).
pub const STORAGE_X0: i64 = 1_100_000_000;
/// Storage x span reserved per round body.
pub const SLOT: i64 = 1 << 26;
/// Empty storage cells kept between neighbouring boxes (so glue reads never hit another box).
const GAP: i64 = 64;
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
    /// Transition shell: chart resolution, outer radius, radial layers, per-face storage origins.
    pub t_n: i64,
    pub t_r: i64,
    pub t_layers: i64,
    pub t_origin: [[i64; 3]; 6],
    /// Core cube half-size (cells of one block) and its storage origin (its minimum corner).
    pub core_half: i64,
    pub core_origin: [i64; 3],
}

fn face_index(f: Face) -> usize {
    FACES.iter().position(|&g| g == f).expect("a face")
}

impl Atlas {
    /// The atlas of a body of datum radius `radius` whose cells reach `top` (≥ radius, the relief)
    /// and go down to the centre, in storage slot `slot`.
    pub fn new(centre: DVec3, radius: i64, top: i64, inward: bool, slot: u32) -> Self {
        let x0 = STORAGE_X0 + slot as i64 * SLOT;
        // Bands halve the radius and the angular resolution together until the inner radius would
        // drop under 512; the surface resolution is a multiple of 2^(bands + 1) so every band (and
        // the transition at half the last band) divides evenly.
        let mut count = 0u32;
        let mut r = radius / 2;
        while r > 512 {
            count += 1;
            r /= 2;
        }
        let unit = 1i64 << (count + 1);
        let mut n = ((std::f64::consts::FRAC_PI_2 * radius as f64) / unit as f64).round().max(1.0) as i64 * unit;
        let mut bands = Vec::new();
        let (mut r_hi, mut r_lo) = (top, radius / 2);
        let mut y = 0i64;
        for _ in 0..count {
            let layers = r_hi - r_lo;
            let origin = std::array::from_fn(|f| [x0, y, f as i64 * (n + GAP)]);
            bands.push(Band { n, r_lo, r_hi, origin });
            y += layers + GAP;
            r_hi = r_lo;
            r_lo /= 2;
            n /= 2;
        }
        // Transition shell from the core cube (half-size a) out to r_hi, at half the last band's
        // resolution so the core cube's corners stay inside the sphere.
        let t_n = (n / 2).max(2);
        let core_half = t_n / 2;
        let t_r = r_hi;
        let t_layers = (t_r - core_half).max(1);
        let t_origin = std::array::from_fn(|f| [x0, y, f as i64 * (t_n + GAP)]);
        y += t_layers + GAP;
        let core_origin = [x0, y, 0];
        Self { centre, radius, inward, bands, t_n, t_r, t_layers, t_origin, core_half, core_origin }
    }

    /// The storage box of a patch: `(min, size)`.
    pub fn storage_box(&self, patch: Patch) -> ([i64; 3], [i64; 3]) {
        match patch {
            Patch::Shell { band, face } => {
                let b = &self.bands[band as usize];
                (b.origin[face_index(face)], [b.n, b.r_hi - b.r_lo, b.n])
            }
            Patch::Transition { face } => (self.t_origin[face_index(face)], [self.t_n, self.t_layers, self.t_n]),
            Patch::Core => (self.core_origin, [2 * self.core_half; 3]),
        }
    }

    /// Every patch.
    pub fn patches(&self) -> impl Iterator<Item = Patch> + '_ {
        let shells = (0..self.bands.len()).flat_map(|b| FACES.map(move |face| Patch::Shell { band: b as u8, face }));
        shells.chain(FACES.map(|face| Patch::Transition { face })).chain(std::iter::once(Patch::Core))
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
                let r = if self.inward { (b.r_hi as f64) - l.y } else { b.r_lo as f64 + l.y };
                self.centre + radial(face, b.n, r, l.x, l.z)
            }
            Patch::Transition { face } => {
                let step = 2.0 / self.t_n as f64;
                let (xi, eta) = (-1.0 + l.x * step, -1.0 + l.z * step);
                let q = std::f64::consts::FRAC_PI_4;
                let (tu, nn, tv) = basis(face);
                let cube = (tu * (xi * q).tan() + nn + tv * (eta * q).tan()) * self.core_half as f64;
                let sphere = radial(face, self.t_n, self.t_r as f64, l.x, l.z);
                let t = l.y / self.t_layers as f64;
                let t = if self.inward { 1.0 - t } else { t };
                self.centre + cube + (sphere - cube) * t
            }
            Patch::Core => self.centre + l - DVec3::splat(self.core_half as f64),
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
                let y = if self.inward { b.r_hi as f64 - r } else { r - b.r_lo as f64 };
                return Some((Patch::Shell { band: bi as u8, face }, DVec3::new((xi + 1.0) / step, y, (eta + 1.0) / step)));
            }
        }
        // Inside the transition or the core: the core cube first (L∞ test), else Newton on the
        // transition's lerp map.
        let a = self.core_half as f64;
        if rel.abs().max_element() < a {
            return Some((Patch::Core, rel + DVec3::splat(a)));
        }
        let patch = Patch::Transition { face };
        let (xi, eta) = Map::Equiangular.inverse(local);
        let step = 2.0 / self.t_n as f64;
        let mut l = DVec3::new((xi + 1.0) / step, self.t_layers as f64 * 0.5, (eta + 1.0) / step);
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
        Atlas::new(DVec3::new(4.0e8, -1.2e8, 3.0e8), 200_000, 202_048, false, 1)
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
        assert_eq!(a.t_r, last.r_lo);
        assert!((a.core_half as f64) * 3f64.sqrt() < a.t_r as f64, "core corners inside the transition sphere");
        // Storage boxes are disjoint and beyond the physical border.
        let boxes: Vec<_> = a.patches().map(|p| a.storage_box(p)).collect();
        for (i, (o, s)) in boxes.iter().enumerate() {
            assert!(o[0] >= STORAGE_X0 && o[0] + s[0] < STORAGE_X0 + 2 * SLOT);
            for (oo, ss) in &boxes[i + 1..] {
                let overlap = (0..3).all(|k| o[k] < oo[k] + ss[k] + GAP && oo[k] < o[k] + s[k] + GAP);
                assert!(!overlap, "boxes {o:?}/{s:?} and {oo:?}/{ss:?} too close");
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
    fn a_surface_chunk_cage_is_nearly_a_unit_cube_scaled_by_16() {
        let a = atlas();
        let b = a.bands[0];
        // A chunk at the datum radius (cells there are one block of arc).
        let s = a.storage(Patch::Shell { band: 0, face: Face::PosZ }, [b.n / 2, a.radius - b.r_lo, b.n / 2]);
        let c = a.chunk_cage(s).unwrap();
        for (i, j) in [(0, 1), (0, 2), (0, 4)] {
            assert!(((c[i] - c[j]).length() - 16.0).abs() < 0.05, "{}", (c[i] - c[j]).length());
        }
    }
}
