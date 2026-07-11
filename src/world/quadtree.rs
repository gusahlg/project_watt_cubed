//! Quadtree selection and progressive covering for
//! the column-section far field — the pure, GPU-free spatial logic the
//! [`SectionLane`](super::SectionLane) and the render path drive.
//!
//! Two pure functions: [`desired_sections`] picks one detail per distance band via
//! [`pyramid::level_for`], yielding a partition of the annulus (fine details near the
//! player, coarse far out, never the root). [`resolve_covering`] does progressive
//! covering: for each desired cell it emits the nearest Ready ancestor-or-self, so a
//! not-yet-Ready finer section is transparently covered by its Ready parent (the parent
//! stays loaded until all four children are Ready, enforced in `streaming.rs`). The
//! sole render invariant is never draw a parent with all four of its children; this can
//! only happen transiently at a band seam and is resolved by preferring the finer ones.

use super::pyramid::PyramidCfg;
use super::section::SectionPos;
use super::FastSet;

/// The band radii — inner/outer distance bounds for a detail level's annulus.
/// The coarsest band has no outer edge (infinity).
fn band_radii(ring: usize, cfg: &PyramidCfg) -> (f32, f32) {
    let lo = cfg.unit * cfg.base.powi(ring as i32);
    let coarsest = ring + 1 == cfg.levels.get() as usize;
    let hi = if coarsest { f32::INFINITY } else { cfg.unit * cfg.base.powi(ring as i32 + 1) };
    (lo, hi)
}

/// Nearest and farthest XZ distances from player to a section's footprint.
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

/// The desired quadtree frontier around the player's XZ metre centre
/// `(pcx, pcz)`. For each active detail, a section is desired iff its distance
/// RANGE (nearest..farthest corner) intersects that detail's annulus — the
/// standard DH quadtree cut, which tiles the far field with no gap at either band
/// edge (a section straddling the chunk box is selected at the finest detail
/// because its far corner reaches past `unit`) and never the root (levels stop at
/// `cfg.coarsest()`). Small over-selection at band seams is pruned by
/// [`resolve_covering`].
pub(in crate::world) fn desired_sections(pcx: i32, pcz: i32, cfg: &PyramidCfg) -> Vec<SectionPos> {
    let mut out = Vec::new();
    for (ring, lod) in cfg.active_lods().enumerate() {
        let detail = lod.0;
        let span = SectionPos { detail, x: 0, z: 0 }.span();
        let (lo, hi) = band_radii(ring, cfg);
        // Reach out to the band's outer edge (the horizon for the coarsest); `+1`
        // section covers the partial section the edge sits inside.
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

/// The nearest Ready ancestor-or-self of `cell`, climbing no further than
/// `max_detail` (never the root). `None` when nothing up to the horizon is loaded
/// yet — a transient hole during progressive load, tolerated by design.
pub(in crate::world) fn drawable_cover(
    cell: SectionPos,
    max_detail: u8,
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

/// Resolve the desired frontier to the flat set of sections to DRAW this frame.
///
/// Every desired cell contributes its [`drawable_cover`] (nearest Ready
/// ancestor-or-self). The result is then pruned of any node whose four children
/// are ALL drawn — the sole render invariant (never a parent plus all four
/// children), which can only appear transiently at a band seam and always
/// resolves in favour of the finer children.
pub(in crate::world) fn resolve_covering(
    desired: &[SectionPos],
    max_detail: u8,
    ready: &impl Fn(SectionPos) -> bool,
) -> Vec<SectionPos> {
    let mut draw: FastSet<SectionPos> = FastSet::default();
    for &cell in desired {
        if let Some(cover) = drawable_cover(cell, max_detail, ready) {
            draw.insert(cover);
        }
    }
    let redundant: Vec<SectionPos> = draw
        .iter()
        .copied()
        .filter(|p| p.detail > 0 && (0..4).all(|q| draw.contains(&p.child(q))))
        .collect();
    for p in redundant {
        draw.remove(&p);
    }
    draw.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::lod::Lod;
    use crate::world::pyramid::{self, LodChoice};
    use crate::world::section::FINEST_DETAIL;

    fn cfg() -> PyramidCfg {
        PyramidCfg::sections(96.0)
    }

    /// Every desired section's detail matches its band, and detail coarsens outward.
    #[test]
    fn selection_is_banded_and_monotone() {
        let cfg = cfg();
        let desired = desired_sections(0, 0, &cfg);
        assert!(!desired.is_empty(), "some far field is desired near the origin");
        for c in &desired {
            assert!(c.detail >= FINEST_DETAIL, "never finer than the finest section");
            assert!(c.detail <= cfg.coarsest(), "never coarser than the horizon (root excluded)");
            let ring = (c.detail - cfg.finest.0) as usize;
            let (lo, hi) = band_radii(ring, &cfg);
            let (near, far) = dist_range(*c, 0, 0);
            assert!(near < hi && far >= lo, "desired cell {c:?} intersects its band");
        }
        for r in (cfg.unit as i32)..(cfg.outer_m() as i32) {
            let want_detail = match pyramid::level_for(r as f32, &cfg) {
                LodChoice::Level(l) => l.0,
                _ => continue,
            };
            let covered = desired.iter().any(|c| {
                c.detail == want_detail
                    && (c.min_x()..c.min_x() + c.span()).contains(&r)
                    && (c.min_z()..c.min_z() + c.span()).contains(&0)
            });
            assert!(covered, "distance {r} (detail {want_detail}) has a desired section on the +X ray");
        }
        for &(dx, dz) in &[(1i32, 0i32), (0, 1), (1, 1), (-1, 2)] {
            let mut coarsest_seen = 0u8;
            for step in 0..400 {
                let (pcx, pcz) = (dx * step * 8, dz * step * 8);
                if let LodChoice::Level(l) = pyramid::level_for((pcx as f32).hypot(pcz as f32), &cfg)
                {
                    assert!(l.0 >= coarsest_seen, "detail never refines with distance");
                    coarsest_seen = l.0;
                }
            }
        }
    }

    /// The covering invariant on a synthetic loaded set: every desired cell gets
    /// a drawable ancestor-or-self, and no parent is drawn together with all four
    /// of its children.
    #[test]
    fn covering_covers_all_and_never_parent_plus_four_children() {
        let cfg = cfg();
        let desired = desired_sections(2_000, -1_500, &cfg);
        let max = cfg.coarsest();

        let all_ready = |_p: SectionPos| true;
        let draw = resolve_covering(&desired, max, &all_ready);
        let drawn: FastSet<SectionPos> = draw.iter().copied().collect();
        for &cell in &desired {
            assert!(
                drawable_cover(cell, max, &all_ready).is_some(),
                "every desired cell has a drawable ancestor-or-self"
            );
        }
        for &p in &draw {
            assert!(
                p.detail == 0 || !(0..4).all(|q| drawn.contains(&p.child(q))),
                "no parent {p:?} drawn with all four children"
            );
        }

        let coarse = cfg.coarsest();
        let coarse_only = move |p: SectionPos| p.detail >= coarse - 1;
        for &cell in &desired {
            assert!(
                drawable_cover(cell, max, &coarse_only).is_some(),
                "a finer cell {cell:?} is covered by its coarse ancestor"
            );
        }
    }

    /// Hysteresis test: at a band edge the one-detail-finer section is still
    /// acceptable, so an outward step doesn't immediately unload the prior section.
    #[test]
    fn acceptable_keeps_one_finer_at_a_band_edge() {
        let cfg = cfg();
        for m in 100..30_000u32 {
            let d = m as f32;
            if let LodChoice::Level(l) = pyramid::level_for(d, &cfg) {
                assert!(pyramid::acceptable(d, l, &cfg), "the band's own level is kept");
                if l.0 > cfg.finest.0 {
                    assert!(
                        pyramid::acceptable(d, Lod(l.0 - 1), &cfg),
                        "one detail finer is kept (expected-1 hysteresis)"
                    );
                }
                assert!(
                    !pyramid::acceptable(d, Lod(l.0 + 1), &cfg),
                    "one detail coarser is NOT acceptable"
                );
            }
        }
    }
}
