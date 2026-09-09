//! Quadtree selection and progressive covering for the far field.
//!
//! `desired_sections` picks coarse or fine sections by distance band.
//! `resolve_covering` maps each desired section to the nearest Ready ancestor-or-self,
//! so unready finer sections are transparently covered by their ready parents.
//!
//! Covering draws only the quadrant(s) needed from each parent stand-in, avoiding
//! overlap with ready siblings. This yields an exact partition when stand-ins are
//! at most one level deep (the common case). Stand-ins deeper than one level
//! (during initial load) are drawn whole; overlap is arbitrated by depth bias.

use crate::ident::Detail;

use super::metric::EyeMetric;
use super::pyramid::PyramidCfg;
use super::section::{Quadrant, SectionPos};
use super::summary::{CellSummary, SseBudget};
use super::{FastMap, FastSet};

/// Which of a section's four quadrants to draw, as a 4-bit set. Bit `q` selects
/// [`SectionPos::child`]'s [`Quadrant`], matching [`SectionPos::quadrant`] exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::world) struct QuadrantMask(u8);

impl QuadrantMask {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(0b1111);

    pub fn contains(self, q: Quadrant) -> bool {
        self.0 & (1 << q.get()) != 0
    }
    pub fn insert(&mut self, q: Quadrant) {
        self.0 |= 1 << q.get();
    }
    pub fn remove(&mut self, q: Quadrant) {
        self.0 &= !(1 << q.get());
    }
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn iter(self) -> impl Iterator<Item = Quadrant> {
        Quadrant::ALL.into_iter().filter(move |&q| self.contains(q))
    }
}

/// Section-to-quadrant-mask pairs drawn this frame. Private ctor ensures no stale
/// data: the ctor debug-asserts no empty masks and that pruning reached fixpoint
/// (no drawn quadrant is already fully tiled by finer cells). This exact partition
/// holds only for one-level stand-ins; deeper stand-ins (during load) are drawn
/// whole with overlap arbitrated by depth bias.
#[derive(Debug)]
pub(in crate::world) struct CoverCut(Vec<(SectionPos, QuadrantMask)>);

impl CoverCut {
    fn new(cut: Vec<(SectionPos, QuadrantMask)>) -> CoverCut {
        if cfg!(debug_assertions) {
            let map: FastMap<SectionPos, QuadrantMask> = cut.iter().copied().collect();
            debug_assert_eq!(map.len(), cut.len(), "a section appears twice in the cut");
            for &(p, mask) in &cut {
                debug_assert!(!mask.is_empty(), "empty-mask entry {p:?} survived the prune");
                for q in mask.iter() {
                    debug_assert!(
                        p.detail.0 == 0 || !area_covered(p.child(q), &map),
                        "drawn quadrant {q:?} of {p:?} is already tiled by finer cells"
                    );
                }
            }
        }
        CoverCut(cut)
    }
    pub fn iter(&self) -> impl Iterator<Item = &(SectionPos, QuadrantMask)> {
        self.0.iter()
    }
}

/// Inner and outer distance bounds for a detail level's annulus. Innermost band
/// starts at 0; coarsest band ends at infinity.
fn band_radii(ring: usize, cfg: &PyramidCfg) -> (f32, f32) {
    let lo = if ring == 0 { 0.0 } else { cfg.unit * cfg.base.powi(ring as i32) };
    let coarsest = ring + 1 == cfg.levels.get() as usize;
    let hi = if coarsest { f32::INFINITY } else { cfg.unit * cfg.base.powi(ring as i32 + 1) };
    (lo, hi)
}

/// Desired sections around the player's XZ metre centre. For each detail level,
/// a section is desired iff its distance range intersects that level's annulus.
/// Result may over-select at band seams; call [`resolve_covering`] to prune.
pub(in crate::world) fn desired_sections(eye: &EyeMetric, cfg: &PyramidCfg) -> Vec<SectionPos> {
    let mut out = Vec::new();
    let (ax, az) = eye.anchor();
    for (ring, lod) in cfg.active_lods().enumerate() {
        let detail = lod;
        let span = SectionPos { detail, x: 0, z: 0 }.span();
        // Test membership against the full 3D range (including eye altitude).
        // XZ projection skips bands wholly overhead and bounds the grid sweep.
        let (lo, hi) = band_radii(ring, cfg);
        let Some(annulus) = eye.xz_annulus(lo, hi) else {
            continue;
        };
        // Reach to the band's outer radius; +1 covers partial edge sections.
        let outer = if annulus.hi().is_finite() { annulus.hi() } else { cfg.outer_m() };
        let reach = (outer / span as f32).ceil() as i32 + 1;
        let (psx, psz) = (ax.div_euclid(span), az.div_euclid(span));
        for sx in (psx - reach)..=(psx + reach) {
            for sz in (psz - reach)..=(psz + reach) {
                let sec = SectionPos { detail, x: sx, z: sz };
                let r = eye.range(sec);
                if r.near().get() < hi && r.far().get() >= lo {
                    out.push(sec);
                }
            }
        }
    }
    out
}

/// Coarsen `desired_sections` output by merging sibling quads to their parent when
/// the parent fits the relief budget. Repeat until fixpoint (no more merges).
/// Worst-case relief blocks coarsening (frontier stays radial). Flat relief (0 error)
/// coalesces sharply. Result is a valid quadtree cut; overlaps are pruned by
/// [`resolve_covering`].
pub(in crate::world) fn coarsen_by_error(
    frontier: Vec<SectionPos>,
    eye: &EyeMetric,
    cfg: &PyramidCfg,
    summary: &impl Fn(SectionPos) -> CellSummary,
    budget: &SseBudget,
) -> Vec<SectionPos> {
    let mut set: FastSet<SectionPos> = frontier.into_iter().collect();
    // One detail level per pass to keep merges order-independent. Adjacent bands
    // can overlap (a cell and its parent both present), so level-by-level ensures
    // each pass is deterministic, with newly-formed parents reconsidered at the next level.
    for target in (cfg.finest.0 + 1)..=cfg.coarsest().0 {
        let mut kids: FastMap<SectionPos, u8> = FastMap::default();
        for &c in &set {
            if c.detail.0 == target - 1 {
                *kids.entry(c.parent()).or_insert(0) += 1;
            }
        }
        let merges: Vec<SectionPos> = kids
            .into_iter()
            .filter(|&(p, n)| {
                // Use the parent's envelope (range_in) for per-cell altitude, not global
                // range. This lets a high eye coarsen the ground beneath it.
                let s = summary(p);
                n == 4 && budget.coarse_ok(s.err, eye.range_in(p, s.env).near())
            })
            .map(|(p, _)| p)
            .collect();
        for p in merges {
            for q in Quadrant::ALL {
                set.remove(&p.child(q));
            }
            set.insert(p);
        }
    }
    // Deterministic order: sort to avoid arbitrary set iteration.
    let mut out: Vec<SectionPos> = set.into_iter().collect();
    out.sort_unstable_by_key(|s| (s.detail, s.x, s.z));
    out
}

/// Merge two frontiers for velocity prediction (real eye + predicted eye).
/// Result is a superset of each input, never dropping cells. Overlaps are pruned
/// by [`resolve_covering`].
pub(in crate::world) fn union_frontiers(a: Vec<SectionPos>, b: Vec<SectionPos>) -> Vec<SectionPos> {
    let mut set: FastSet<SectionPos> = a.into_iter().collect();
    set.extend(b);
    let mut out: Vec<SectionPos> = set.into_iter().collect();
    out.sort_unstable_by_key(|s| (s.detail, s.x, s.z));
    out
}

/// Nearest Ready ancestor-or-self of `cell`, up to `max_detail`. Returns `None`
/// if no ancestor up the tree is loaded yet (transient during progressive load).
pub(in crate::world) fn drawable_cover(
    cell: SectionPos,
    max_detail: Detail,
    ready: &impl Fn(SectionPos) -> bool,
) -> Option<SectionPos> {
    let mut c = cell;
    loop {
        if ready(c) {
            return Some(c);
        }
        if c.detail >= max_detail {
            return None;
        }
        c = c.parent();
    }
}

/// Map each desired cell to a drawable ancestor-or-self and return the masked sections
/// to draw this frame. Each cell uses [`drawable_cover`] to find its ancestor.
/// If the ancestor is the cell's parent, draw only that quadrant (avoiding overlap
/// with ready siblings). Otherwise draw whole. Prune any coarse bit that's already
/// tiled by finer cells (recursive check via [`area_covered`]).
pub(in crate::world) fn resolve_covering(
    desired: &[SectionPos],
    max_detail: Detail,
    ready: &impl Fn(SectionPos) -> bool,
) -> CoverCut {
    let mut draw: FastMap<SectionPos, QuadrantMask> = FastMap::default();
    for &cell in desired {
        if let Some(cover) = drawable_cover(cell, max_detail, ready) {
            let mask = if cover == cell.parent() {
                let mut m = QuadrantMask::EMPTY;
                m.insert(cell.quadrant());
                m
            } else {
                // cover == cell, or a two-plus-level stand-in: the whole tile.
                QuadrantMask::ALL
            };
            let e = draw.entry(cover).or_insert(QuadrantMask::EMPTY);
            *e = e.union(mask);
        }
    }
    let mut out = Vec::with_capacity(draw.len());
    for (&p, &mask) in &draw {
        let mut m = mask;
        if p.detail.0 > 0 {
            for q in Quadrant::ALL {
                if m.contains(q) && area_covered(p.child(q), &draw) {
                    m.remove(q);
                }
            }
        }
        if !m.is_empty() {
            out.push((p, m));
        }
    }
    CoverCut::new(out)
}

/// Whether `cell`'s area is fully tiled by finer drawn cells. Prunes against
/// the current draw set; recursion depth bounded by pyramid level count.
fn area_covered(cell: SectionPos, draw: &FastMap<SectionPos, QuadrantMask>) -> bool {
    let m = draw.get(&cell).copied().unwrap_or(QuadrantMask::EMPTY);
    if m == QuadrantMask::ALL {
        return true;
    }
    if cell.detail.0 == 0 {
        return false;
    }
    Quadrant::ALL.into_iter().all(|q| m.contains(q) || area_covered(cell.child(q), draw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::metric::{DyCap, EyeDist, EyeMetric, HeightEnvelope};
    use crate::world::pyramid::{self, LodChoice};
    use crate::world::section::FINEST_DETAIL;
    use crate::world::summary::{CellError, CellSummary};
    use std::collections::HashSet;
    use voxel_engine::DVec3;

    fn cfg() -> PyramidCfg {
        PyramidCfg::sections(96.0)
    }

    /// Eye at ground level (dy = 0) for LOD2 parity testing.
    fn eye(x: i32, z: i32, cfg: &PyramidCfg) -> EyeMetric {
        EyeMetric::new(
            DVec3::new(x as f64, 0.0, z as f64),
            HeightEnvelope::new(0.0, 0.0),
            DyCap::new(cfg.outer_m(), cfg.base),
        )
    }

    /// Domain envelope for altitude-relative LOD tests.
    const ENV: (f32, f32) = (0.0, 512.0);

    /// Eye at world altitude `y` for altitude-metric tests.
    fn eye_at(x: i32, y: f64, z: i32, cfg: &PyramidCfg) -> EyeMetric {
        EyeMetric::new(
            DVec3::new(x as f64, y, z as f64),
            HeightEnvelope::new(ENV.0, ENV.1),
            DyCap::new(cfg.outer_m(), cfg.base),
        )
    }

    fn q(i: u8) -> Quadrant {
        Quadrant::new(i)
    }

    /// Reference implementation for parity testing at dy = 0.
    fn reference_desired(pcx: i32, pcz: i32, cfg: &PyramidCfg) -> Vec<SectionPos> {
        fn dist_range(sec: SectionPos, px: i32, pz: i32) -> (f32, f32) {
            let span = sec.span();
            let axis = |lo: i32, p: i32| -> (i64, i64) {
                let hi = lo + span;
                let near = if p < lo {
                    (lo - p) as i64
                } else if p >= hi {
                    (p - hi + 1) as i64
                } else {
                    0
                };
                let far = ((p - lo).abs().max((p - (hi - 1)).abs())) as i64;
                (near, far)
            };
            let (nx, fx) = axis(sec.min_x(), px);
            let (nz, fz) = axis(sec.min_z(), pz);
            (((nx * nx + nz * nz) as f32).sqrt(), ((fx * fx + fz * fz) as f32).sqrt())
        }
        let mut out = Vec::new();
        for (ring, lod) in cfg.active_lods().enumerate() {
            let detail = lod;
            let span = SectionPos { detail, x: 0, z: 0 }.span();
            let (lo, hi) = band_radii(ring, cfg);
            let outer = if hi.is_finite() { hi } else { cfg.outer_m() };
            let reach = (outer / span as f32).ceil() as i32 + 1;
            let (psx, psz) = (pcx.div_euclid(span), pcz.div_euclid(span));
            for sx in (psx - reach)..=(psx + reach) {
                for sz in (psz - reach)..=(psz + reach) {
                    let sec = SectionPos { detail, x: sx, z: sz };
                    let (near, far) = dist_range(sec, pcx, pcz);
                    if near < hi && far >= lo {
                        out.push(sec);
                    }
                }
            }
        }
        out
    }

    /// At dy = 0, desired_sections matches the old LOD2 integer path exactly.
    /// This parity holds because the metric anchors XZ to chunk centres.
    /// Shifting to sub-chunk coordinates would break this guarantee.
    #[test]
    fn dy_zero_selection_is_bit_identical_to_lod2() {
        let cfg = cfg();
        let unit = cfg.unit as i32;
        let grid = [
            (0, 0),
            (unit, 0),
            (0, -unit),
            (unit, unit),
            (2_000, -1_500),
            (-500, 300),
            (96, 96),
            (unit * 3 - 1, unit * 3 + 1), // straddling a band edge
            (-1, -1),
        ];
        for &(x, z) in &grid {
            let got = desired_sections(&eye(x, z, &cfg), &cfg);
            assert_eq!(got, reference_desired(x, z, &cfg), "parity at eye ({x},{z})");
            // Determinism: same state, same cut.
            assert_eq!(got, desired_sections(&eye(x, z, &cfg), &cfg), "deterministic at ({x},{z})");
        }
    }

    /// Every desired section's detail matches its band, and detail coarsens outward.
    #[test]
    fn selection_is_banded_and_monotone() {
        let cfg = cfg();
        let desired = desired_sections(&eye(0, 0, &cfg), &cfg);
        assert!(!desired.is_empty(), "some far field is desired near the origin");
        for c in &desired {
            assert!(c.detail >= FINEST_DETAIL, "never finer than the finest section");
            assert!(c.detail <= cfg.coarsest(), "never coarser than the horizon (root excluded)");
            let ring = (c.detail.0 - cfg.finest.0) as usize;
            let (lo, hi) = band_radii(ring, &cfg);
            let r = eye(0, 0, &cfg).range(*c);
            assert!(r.near().get() < hi && r.far().get() >= lo, "desired cell {c:?} intersects its band");
        }
        for r in (cfg.unit as i32)..(cfg.outer_m() as i32) {
            let want_detail = match pyramid::level_for(EyeDist::new(r as f32), &cfg) {
                LodChoice::Level(l) => l,
                _ => continue,
            };
            let covered = desired.iter().any(|c| {
                c.detail == want_detail
                    && (c.min_x()..c.min_x() + c.span()).contains(&r)
                    && (c.min_z()..c.min_z() + c.span()).contains(&0)
            });
            assert!(covered, "distance {r} (detail {want_detail:?}) has a desired section on the +X ray");
        }
        for &(dx, dz) in &[(1i32, 0i32), (0, 1), (1, 1), (-1, 2)] {
            let mut coarsest_seen = 0i8;
            for step in 0..400 {
                let (pcx, pcz) = (dx * step * 8, dz * step * 8);
                if let LodChoice::Level(l) =
                    pyramid::level_for(EyeDist::new((pcx as f32).hypot(pcz as f32)), &cfg)
                {
                    assert!(l.0 >= coarsest_seen, "detail never refines with distance");
                    coarsest_seen = l.0;
                }
            }
        }
    }

    /// The covering invariant on a synthetic loaded set: every desired cell gets
    /// a drawable ancestor-or-self, and no parent bit is drawn for a quadrant whose
    /// child is already drawn `ALL` (the pruned redundancy).
    #[test]
    fn covering_covers_all_and_never_parent_plus_four_children() {
        let cfg = cfg();
        let desired = desired_sections(&eye(2_000, -1_500, &cfg), &cfg);
        let max = cfg.coarsest();

        let all_ready = |_p: SectionPos| true;
        let draw = resolve_covering(&desired, max, &all_ready);
        let drawn: FastMap<SectionPos, QuadrantMask> = draw.iter().copied().collect();
        for &cell in &desired {
            assert!(
                drawable_cover(cell, max, &all_ready).is_some(),
                "every desired cell has a drawable ancestor-or-self"
            );
        }
        for &(p, mask) in draw.iter() {
            assert!(!mask.is_empty(), "an empty-mask entry survived the prune");
            for q in mask.iter() {
                assert!(
                    p.detail.0 == 0 || drawn.get(&p.child(q)) != Some(&QuadrantMask::ALL),
                    "parent {p:?} draws quadrant {q:?} whose child is already drawn ALL"
                );
            }
        }

        let coarse = cfg.coarsest();
        let coarse_only = move |p: SectionPos| p.detail.0 >= coarse.0 - 1;
        for &cell in &desired {
            assert!(
                drawable_cover(cell, max, &coarse_only).is_some(),
                "a finer cell {cell:?} is covered by its coarse ancestor"
            );
        }
    }

    /// One-level stand-in with three ready siblings yields exact partition.
    #[test]
    fn one_missing_child_yields_an_exact_partition() {
        let parent = SectionPos { detail: Detail(FINEST_DETAIL.0 + 2), x: 3, z: -2 };
        let missing_q = q(2);
        let children: [SectionPos; 4] = std::array::from_fn(|i| parent.child(q(i as u8)));
        let ready = move |p: SectionPos| p == parent || (children.contains(&p) && p != parent.child(missing_q));

        let draw = resolve_covering(&children, Detail(FINEST_DETAIL.0 + 2), &ready);
        let map: FastMap<SectionPos, QuadrantMask> = draw.iter().copied().collect();

        let mut want_parent = QuadrantMask::EMPTY;
        want_parent.insert(missing_q);
        assert_eq!(map.get(&parent), Some(&want_parent), "parent stands in for the one missing quadrant");
        for qi in Quadrant::ALL {
            if qi == missing_q {
                assert!(!map.contains_key(&parent.child(qi)) || map[&parent.child(qi)].is_empty());
            } else {
                assert_eq!(map.get(&parent.child(qi)), Some(&QuadrantMask::ALL), "ready child {qi:?} draws whole");
            }
        }

        let mut covers: std::collections::HashMap<(i32, i32), u32> = Default::default();
        for &(p, mask) in draw.iter() {
            for qi in mask.iter() {
                let c = p.child(qi);
                let span = 1i32 << (c.detail.0 - FINEST_DETAIL.0);
                for dx in 0..span {
                    for dz in 0..span {
                        *covers.entry((c.x * span + dx, c.z * span + dz)).or_insert(0) += 1;
                    }
                }
            }
        }
        assert!(covers.values().all(|&n| n == 1), "every finest cell is drawn exactly once (no overlap, no hole)");
        // The covered footprint is exactly the parent's area: (2^(parent-finest))² finest cells.
        let parent_span = 1i32 << (parent.detail.0 - FINEST_DETAIL.0);
        assert_eq!(covers.len() as i32, parent_span * parent_span);
    }

    /// Coarse cells pruned even against partially drawn children, avoiding overlay.
    #[test]
    fn coarse_overlay_prunes_against_partially_drawn_children() {
        let g = SectionPos { detail: Detail(FINEST_DETAIL.0 + 2), x: 1, z: 1 };
        let c = g.child(q(1)); // partially drawn: one of its own children is missing
        let missing = c.child(q(3));
        let mut desired: Vec<SectionPos> = vec![g];
        desired.extend(Quadrant::ALL.into_iter().map(|qi| c.child(qi))); // finest band inside c
        desired.extend(Quadrant::ALL.into_iter().filter(|&qi| g.child(qi) != c).map(|qi| g.child(qi)));
        let ready = move |p: SectionPos| p != missing;

        let draw = resolve_covering(&desired, Detail(FINEST_DETAIL.0 + 2), &ready);
        let map: FastMap<SectionPos, QuadrantMask> = draw.iter().copied().collect();
        assert!(!map.contains_key(&g), "coarse cell fully tiled by finer draws is pruned");
        let mut want_c = QuadrantMask::EMPTY;
        want_c.insert(missing.quadrant());
        assert_eq!(map.get(&c), Some(&want_c), "partial child stands in for its missing quadrant only");

        // Exact partition over g's whole footprint at finest granularity.
        let mut covers: std::collections::HashMap<(i32, i32), u32> = Default::default();
        // Use FINEST_DETAIL - 1 base to avoid exponent underflow.
        const BASE: i8 = FINEST_DETAIL.0 - 1;
        for &(p, mask) in draw.iter() {
            for qi in mask.iter() {
                let cell = p.child(qi);
                let span = 1i32 << (cell.detail.0 - BASE);
                for dx in 0..span {
                    for dz in 0..span {
                        *covers.entry((cell.x * span + dx, cell.z * span + dz)).or_insert(0) += 1;
                    }
                }
            }
        }
        assert!(covers.values().all(|&n| n == 1), "no overlap, no double-draw");
        let g_span = 1i32 << (g.detail.0 - BASE);
        assert_eq!(covers.len() as i32, g_span * g_span, "no hole across g's footprint");
    }

    /// At a band edge, one finer detail stays acceptable to avoid thrashing.
    #[test]
    fn acceptable_keeps_one_finer_at_a_band_edge() {
        let cfg = cfg();
        for m in 100..30_000u32 {
            let d = EyeDist::new(m as f32);
            if let LodChoice::Level(l) = pyramid::level_for(d, &cfg) {
                assert!(pyramid::acceptable(d, l, &cfg), "the band's own level is kept");
                if l.0 > cfg.finest.0 {
                    assert!(
                        pyramid::acceptable(d, Detail(l.0 - 1), &cfg),
                        "one detail finer is kept (expected-1 hysteresis)"
                    );
                }
                assert!(
                    !pyramid::acceptable(d, Detail(l.0 + 1), &cfg),
                    "one detail coarser is NOT acceptable"
                );
            }
        }
    }

    // Altitude-driven selection

    /// Climbing only coarsens. Finest-band count decreases with altitude;
    /// far field stays nonempty.
    #[test]
    fn selection_coarsens_monotonically_with_altitude() {
        let cfg = cfg();
        let mut last_finest = usize::MAX;
        for &y in &[200.0, 700.0, 1_200.0, 2_000.0, 5_000.0] {
            let d = desired_sections(&eye_at(0, y, 0, &cfg), &cfg);
            assert!(!d.is_empty(), "the far field is never empty at altitude {y}");
            let finest = d.iter().filter(|c| c.detail == FINEST_DETAIL).count();
            assert!(finest <= last_finest, "finest-band count grew climbing to {y}");
            last_finest = finest;
        }
        let high = desired_sections(&eye_at(0, 5_000.0, 0, &cfg), &cfg);
        assert_eq!(high.iter().filter(|c| c.detail == FINEST_DETAIL).count(), 0, "no finest sections at extreme altitude");
    }

    /// Covering stays an exact partition at every altitude as dy varies.
    #[test]
    fn covering_partition_holds_across_altitudes() {
        let cfg = cfg();
        let max = cfg.coarsest();
        let all_ready = |_p: SectionPos| true;
        for &y in &[0.0, 400.0, 900.0, 2_000.0, 10_000.0] {
            let desired = desired_sections(&eye_at(500, y, -300, &cfg), &cfg);
            let cut = resolve_covering(&desired, max, &all_ready);
            for &cell in &desired {
                assert!(
                    drawable_cover(cell, max, &all_ready).is_some(),
                    "cell {cell:?} uncovered at altitude {y}"
                );
            }
            assert_eq!(desired.is_empty(), cut.iter().count() == 0, "cut and desired match at {y}");
        }
    }

    /// Eye oscillating across an altitude band-edge doesn't thrash: desired cells
    /// below the edge stay in the kept set above it.
    #[test]
    fn vertical_hysteresis_absorbs_a_band_edge_oscillation() {
        let cfg = cfg();
        // Altitude at a band boundary (dy = 384).
        let edge_y = ENV.1 as f64 + 384.0;
        let below = eye_at(0, edge_y - 4.0, 0, &cfg);
        let above = eye_at(0, edge_y + 4.0, 0, &cfg);
        let kept_above: std::collections::HashSet<SectionPos> =
            desired_sections(&above, &cfg).into_iter().collect();
        for s in desired_sections(&below, &cfg) {
            let span = s.span();
            let (cx, cz) = (s.x * span + span / 2, s.z * span + span / 2);
            let kept = kept_above.contains(&s)
                || pyramid::acceptable(above.point(cx as f64, cz as f64), s.detail, &cfg);
            assert!(kept, "a small climb unloaded {s:?} at the band edge (thrash)");
        }
    }

    /// At extreme altitude, dy clamp keeps coarsest-only ring nonempty.
    #[test]
    fn dy_cap_keeps_the_coarsest_ring_nonempty_at_extreme_altitude() {
        let cfg = cfg();
        let cap = cfg.outer_m() * (1.0 - 1.0 / cfg.base);
        let cap_y = ENV.1 as f64 + cap as f64;
        let at_cap = desired_sections(&eye_at(0, cap_y, 0, &cfg), &cfg);
        let way_up = desired_sections(&eye_at(0, 100_000.0, 0, &cfg), &cfg);
        assert_eq!(at_cap, way_up, "dy is clamped: extreme altitude == cap altitude");
        assert!(!way_up.is_empty(), "cap keeps the coarsest ring nonempty");
        assert!(way_up.iter().all(|c| c.detail == cfg.coarsest()), "only the coarsest ring survives");
    }

    /// NaN altitude maps to dy = 0 (ground field), never panic or empty.
    #[test]
    fn nan_altitude_falls_to_the_ground_field() {
        let cfg = cfg();
        let nan = desired_sections(&eye_at(300, f64::NAN, -200, &cfg), &cfg);
        let ground = desired_sections(&eye_at(300, 100.0, -200, &cfg), &cfg); // inside envelope: dy = 0
        assert_eq!(nan, ground);
    }

    /// An eye below the envelope is symmetric with one equally far above it: dy depends
    /// only on the distance to the envelope, not the sign.
    #[test]
    fn eye_below_envelope_is_symmetric_with_above() {
        let cfg = cfg();
        let below = desired_sections(&eye_at(0, ENV.0 as f64 - 300.0, 0, &cfg), &cfg);
        let above = desired_sections(&eye_at(0, ENV.1 as f64 + 300.0, 0, &cfg), &cfg);
        assert_eq!(below, above);
    }

    /// Column under the eye is always desired at ground, within-domain, and above.
    /// Near field clamps to finest ring; no coverage gap.
    #[test]
    fn the_nadir_column_is_always_covered() {
        let cfg = cfg();
        let covers_nadir = |m: &EyeMetric| {
            desired_sections(m, &cfg).iter().any(|c| {
                (c.min_x()..c.min_x() + c.span()).contains(&0)
                    && (c.min_z()..c.min_z() + c.span()).contains(&0)
            })
        };
        for &y in &[0.0, 100.0, 400.0, 5_000.0] {
            assert!(covers_nadir(&eye_at(0, y, 0, &cfg)), "no section under the eye at y={y}");
        }
        // At ground the nadir is the finest ring (chunks draw over it via the clip).
        let ground = desired_sections(&eye(0, 0, &cfg), &cfg);
        assert!(
            ground.iter().any(|c| c.detail == FINEST_DETAIL
                && (c.min_x()..c.min_x() + c.span()).contains(&0)
                && (c.min_z()..c.min_z() + c.span()).contains(&0)),
            "the near field clamps to the finest ring"
        );
    }

    // Error-driven coarsening (relief-based)

    /// Budget pinned to the distance ladder.
    fn ladder(cfg: &PyramidCfg) -> SseBudget {
        SseBudget::ladder(1.0, cfg.unit, cfg.finest.0)
    }
    /// Worst-case relief per cell (2^detail fallback).
    fn worst(c: SectionPos) -> CellSummary {
        CellSummary { env: HeightEnvelope::new(0.0, 512.0), err: CellError::worst_case(c.detail) }
    }
    /// Dead-flat relief (0 height error everywhere).
    fn flat(_c: SectionPos) -> CellSummary {
        CellSummary { env: HeightEnvelope::new(0.0, 0.0), err: CellError::from_metres(0.0) }
    }
    /// True if `coarser` tiles the same footprint with equal-or-coarser cells.
    fn is_coarsening_of(finer: &[SectionPos], coarser: &HashSet<SectionPos>, max: Detail) -> bool {
        finer.iter().all(|&c| {
            let mut a = c;
            loop {
                if coarser.contains(&a) {
                    return true;
                }
                if a.detail >= max {
                    return false;
                }
                a = a.parent();
            }
        })
    }

    /// At dy = 0 with worst-case relief, coarsening is a no-op: selection stays
    /// radial. This is the ground-player guarantee.
    #[test]
    fn worst_case_relief_is_the_radial_ladder_at_every_eye() {
        let cfg = cfg();
        for &(x, y, z) in &[(0, 0.0, 0), (2_000, 0.0, -1_500), (300, 900.0, -200), (0, 2_048.0, 0)] {
            let m = eye_at(x, y, z, &cfg);
            let radial = desired_sections(&m, &cfg);
            let coarsened = coarsen_by_error(radial.clone(), &m, &cfg, &worst, &ladder(&cfg));
            let a: HashSet<_> = radial.into_iter().collect();
            let b: HashSet<_> = coarsened.into_iter().collect();
            assert_eq!(a, b, "worst-case relief coarsened the ladder at eye ({x},{y},{z})");
        }
    }

    /// Flat relief (0 error) coalesces far field to coarsest ring.
    #[test]
    fn ocean_collapse_drops_the_draw_count() {
        let cfg = cfg();
        let m = eye(0, 0, &cfg);
        let radial = desired_sections(&m, &cfg);
        let coarsened = coarsen_by_error(radial.clone(), &m, &cfg, &flat, &ladder(&cfg));
        assert!(
            coarsened.len() * 3 <= radial.len() * 2,
            "ocean collapse: {} to {} cells (expected 1/3+ drop)",
            radial.len(),
            coarsened.len()
        );
        let coarsest = coarsened.iter().filter(|c| c.detail == cfg.coarsest()).count();
        assert!(coarsest > 0, "flat terrain reaches the coarsest ring");
    }

    /// Mixed-detail cut (mountain near, ocean far) still yields an exact partition
    /// via recursive pruning.
    #[test]
    fn mountain_near_ocean_far_covers_an_exact_partition() {
        let cfg = cfg();
        let m = eye(0, 0, &cfg);
        // 400 m of relief within 1200 m of the eye; flat beyond.
        let summary = |c: SectionPos| {
            let span = c.span();
            let (cx, cz) = (c.x * span + span / 2, c.z * span + span / 2);
            let d = ((cx * cx + cz * cz) as f32).sqrt();
            let err = if d < 1_200.0 { CellError::from_metres(400.0) } else { CellError::from_metres(0.0) };
            CellSummary { env: HeightEnvelope::new(0.0, 512.0), err }
        };
        let desired = coarsen_by_error(desired_sections(&m, &cfg), &m, &cfg, &summary, &ladder(&cfg));
        let details: HashSet<Detail> = desired.iter().map(|c| c.detail).collect();
        assert!(details.len() >= 2, "expected a mixed-detail cut, got {details:?}");
        let max = cfg.coarsest();
        let all_ready = |_p: SectionPos| true;
        let cut = resolve_covering(&desired, max, &all_ready);
        for &cell in &desired {
            assert!(drawable_cover(cell, max, &all_ready).is_some(), "{cell:?} uncovered");
        }
        assert_eq!(desired.is_empty(), cut.iter().count() == 0);
    }

    /// Zero budget forbids coarsening any cell with relief; frontier stays radial.
    /// Flat cells still coalesce at any budget.
    #[test]
    fn tau_zero_clamps_to_the_ladder_but_flat_still_collapses() {
        let cfg = cfg();
        let m = eye(0, 0, &cfg);
        let tau0 = SseBudget::new(0.0, 1.0);
        let radial = desired_sections(&m, &cfg);
        let clamped = coarsen_by_error(radial.clone(), &m, &cfg, &worst, &tau0);
        assert_eq!(
            radial.iter().copied().collect::<HashSet<_>>(),
            clamped.iter().copied().collect::<HashSet<_>>(),
            "zero budget must not coarsen relief-bearing cells"
        );
        assert!(clamped.iter().all(|c| c.detail >= FINEST_DETAIL), "never finer than finest");
        let ocean = coarsen_by_error(radial.clone(), &m, &cfg, &flat, &tau0);
        assert!(ocean.len() < radial.len(), "flat terrain still collapses at zero budget");
    }

    /// As relief tightens (worst to half to flat), selection only coarsens.
    /// No re-splitting, so no oscillation.
    #[test]
    fn tightening_relief_only_coarsens() {
        let cfg = cfg();
        let m = eye(500, 0, &cfg);
        let max = cfg.coarsest();
        let radial = desired_sections(&m, &cfg);
        let half = |c: SectionPos| CellSummary {
            env: HeightEnvelope::new(0.0, 512.0),
            err: CellError::from_metres(CellError::worst_case(c.detail).get() / 2.0),
        };
        let a = coarsen_by_error(radial.clone(), &m, &cfg, &worst, &ladder(&cfg));
        let b = coarsen_by_error(radial.clone(), &m, &cfg, &half, &ladder(&cfg));
        let c = coarsen_by_error(radial.clone(), &m, &cfg, &flat, &ladder(&cfg));
        assert!(a.len() >= b.len() && b.len() >= c.len(), "draw count monotonically decreases with relief");
        let (bs, cs): (HashSet<_>, HashSet<_>) = (b.iter().copied().collect(), c.iter().copied().collect());
        assert!(is_coarsening_of(&a, &bs, max), "half-relief must coarsen worst-case");
        assert!(is_coarsening_of(&b, &cs, max), "flat must coarsen half-relief");
        // Deterministic: equal inputs yield equal output.
        assert_eq!(c, coarsen_by_error(radial, &m, &cfg, &flat, &ladder(&cfg)));
    }

    // Velocity prediction (union of static and predicted eyes)

    /// At zero velocity, prediction is identity: no inflation cost at rest.
    #[test]
    fn prediction_is_identity_at_zero_velocity() {
        let cfg = cfg();
        let f = desired_sections(&eye_at(500, 200.0, -300, &cfg), &cfg);
        let union = union_frontiers(f.clone(), f.clone());
        assert_eq!(
            f.iter().copied().collect::<HashSet<_>>(),
            union.iter().copied().collect::<HashSet<_>>()
        );
        assert_eq!(union.len(), f.len(), "union of identical frontiers must not duplicate");
    }

    /// Horizontal motion inflates forward without dropping static cells.
    /// Union is a superset of both inputs.
    #[test]
    fn horizontal_motion_inflates_forward_and_drops_nothing() {
        let cfg = cfg();
        let stat = desired_sections(&eye_at(0, 100.0, 0, &cfg), &cfg);
        let pred = desired_sections(&eye_at(6_000, 100.0, 0, &cfg), &cfg); // advanced +X
        let union = union_frontiers(stat.clone(), pred.clone());
        let uset: HashSet<_> = union.iter().copied().collect();
        assert!(stat.iter().all(|c| uset.contains(c)), "prediction dropped a static cell");
        assert!(union.len() > stat.len(), "prediction added no cells under motion");
        assert!(pred.iter().any(|c| !stat.contains(c)), "predicted footprint adds nothing forward");
    }

    /// Falling eye desires finer cells before arriving. High altitude yields coarse
    /// static field; descended prediction refines it and those cells enter desire.
    #[test]
    fn descent_desires_finer_cells_before_arrival() {
        let cfg = cfg();
        let stat = desired_sections(&eye_at(0, 4_000.0, 0, &cfg), &cfg);
        let pred = desired_sections(&eye_at(0, 400.0, 0, &cfg), &cfg);
        let hi_finest = stat.iter().map(|c| c.detail).min().unwrap();
        let lo_finest = pred.iter().map(|c| c.detail).min().unwrap();
        assert!(lo_finest < hi_finest, "descent didn't refine (hi {hi_finest:?}, lo {lo_finest:?})");
        let union = union_frontiers(stat, pred);
        assert!(union.iter().any(|c| c.detail == lo_finest), "finer descent cells missing from desire");
    }

    /// NaN altitude maps to dy = 0 (ground field), so union stays total.
    #[test]
    fn a_nan_predicted_eye_stays_total() {
        let cfg = cfg();
        let stat = desired_sections(&eye_at(0, 100.0, 0, &cfg), &cfg);
        let pred = desired_sections(&eye_at(0, f64::NAN, 0, &cfg), &cfg);
        let union = union_frontiers(stat.clone(), pred);
        assert!(!union.is_empty());
        assert!(stat.iter().all(|c| union.contains(c)));
    }

    // Terrain-relative altitude (per-cell dy, envelope-based)

    /// Flat low-terrain relief field to isolate altitude-driven coarsening.
    fn low_ground(c: SectionPos) -> CellSummary {
        CellSummary { env: HeightEnvelope::new(60.0, 70.0), err: CellError::worst_case(c.detail) }
    }

    /// High within-domain eye coarsens ground via per-cell altitude, but a ground
    /// player inside the envelope stays radial. Worst-case relief everywhere, so
    /// only altitude gap can coarsen.
    #[test]
    fn terrain_relative_dy_coarsens_ground_under_a_within_domain_high_eye() {
        let cfg = cfg();
        // Ground player inside envelope: per-cell dy = 0, coarsening is a no-op.
        let ground = eye_at(0, 65.0, 0, &cfg);
        let gr = desired_sections(&ground, &cfg);
        let gc = coarsen_by_error(gr.clone(), &ground, &cfg, &low_ground, &ladder(&cfg));
        assert_eq!(
            gr.iter().copied().collect::<HashSet<_>>(),
            gc.iter().copied().collect::<HashSet<_>>(),
            "ground player inside envelope stays radial"
        );
        // In-domain eyes share the base frontier.
        let high = eye_at(0, 300.0, 0, &cfg);
        let hr = desired_sections(&high, &cfg);
        assert_eq!(hr, gr, "in-domain eyes share base frontier");
        // High eye coarsens via per-cell altitude.
        let hc = coarsen_by_error(hr.clone(), &high, &cfg, &low_ground, &ladder(&cfg));
        assert!(hc.len() < hr.len(), "altitude coarsened the field ({} vs {})", hc.len(), hr.len());
        let finest = |v: &[SectionPos]| v.iter().filter(|c| c.detail == FINEST_DETAIL).count();
        assert!(finest(&hc) < finest(&gc), "the nadir did not coarsen with altitude");
    }

    /// Base frontier is same for all in-domain eyes, so as eye climbs the per-cell
    /// altitude grows and field only coarsens. No oscillation.
    #[test]
    fn coarsening_is_monotone_in_altitude_under_fixed_relief() {
        let cfg = cfg();
        let mut last = usize::MAX;
        for &y in &[65.0, 150.0, 300.0, 480.0] {
            let m = eye_at(0, y, 0, &cfg);
            let coarsened = coarsen_by_error(desired_sections(&m, &cfg), &m, &cfg, &low_ground, &ladder(&cfg));
            assert!(coarsened.len() <= last, "climbing to y={y} refined the field ({} > {last})", coarsened.len());
            last = coarsened.len();
        }
    }

    /// Mixed-detail altitude-coarsened cut yields an exact partition via
    /// recursive pruning.
    #[test]
    fn altitude_coarsened_cut_is_an_exact_partition() {
        let cfg = cfg();
        let m = eye_at(700, 300.0, -400, &cfg);
        let desired = coarsen_by_error(desired_sections(&m, &cfg), &m, &cfg, &low_ground, &ladder(&cfg));
        let details: HashSet<Detail> = desired.iter().map(|c| c.detail).collect();
        assert!(details.len() >= 2, "expected a mixed-detail cut from altitude, got {details:?}");
        let max = cfg.coarsest();
        let all_ready = |_p: SectionPos| true;
        let cut = resolve_covering(&desired, max, &all_ready);
        for &cell in &desired {
            assert!(drawable_cover(cell, max, &all_ready).is_some(), "{cell:?} uncovered");
        }
        assert_eq!(desired.is_empty(), cut.iter().count() == 0);
    }
}
