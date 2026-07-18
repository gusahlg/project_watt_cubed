//! Coverage for the section far field — at most one mask drawn per region. LOD cuts
//! hard-pop: adopting a new cut replaces the prior assignment in the same frame, no
//! cross-fade dissolve. Selection (the desired cut) is unchanged; this is a view of it
//! that also emits the per-region drawn-quadrant diff callers need to patch GPU
//! visibility slots.
//!
//! Terrain and hierarchy levels are untouched: this is a view of the selection state,
//! not part of it.

use super::FastMap;
use super::quadtree::QuadrantMask;
use super::section::SectionPos;

/// A region whose drawn quadrants changed: the new mask, or `None` once it draws nothing.
pub(in crate::world) type VisibilityChange = (SectionPos, Option<QuadrantMask>);

/// The far-field's coverage: one mask per drawn region, replaced wholesale each cut.
#[derive(Default)]
pub(in crate::world) struct Coverage {
    settled: FastMap<SectionPos, QuadrantMask>,
}

impl Coverage {
    pub fn update_now(&mut self, cut: &[(SectionPos, QuadrantMask)]) -> Vec<VisibilityChange> {
        self.update(cut)
    }

    /// Adopt a new desired `cut`, replacing the prior assignment. Returns every region
    /// whose drawn quadrants changed — a DIFF across the update, so an in-place mask
    /// change (same region, different quadrants) is caught even though it is neither an
    /// enter nor a leave. Diffing the projection cannot omit a case it did not think of.
    pub fn update(&mut self, cut: &[(SectionPos, QuadrantMask)]) -> Vec<VisibilityChange> {
        let desired: FastMap<SectionPos, QuadrantMask> = cut.iter().copied().collect();
        let changes: Vec<VisibilityChange> = self
            .settled
            .keys()
            .chain(desired.keys())
            .copied()
            .collect::<super::FastSet<_>>()
            .into_iter()
            .filter(|pos| self.settled.get(pos) != desired.get(pos))
            .map(|pos| (pos, desired.get(&pos).copied()))
            .collect();
        self.settled = desired;
        changes
    }

    /// The regions to draw this frame with their masks.
    #[cfg(test)]
    pub fn draws(&self) -> impl Iterator<Item = (SectionPos, QuadrantMask)> + '_ {
        self.settled.iter().map(|(&p, &m)| (p, m))
    }

    /// The quadrants `pos` currently draws, if any — the projection a section consults
    /// when it lands Ready, since its slots did not exist at the transition that decided it.
    pub fn drawn_mask(&self, pos: SectionPos) -> Option<QuadrantMask> {
        self.settled.get(&pos).copied()
    }

    /// Every tracked region — the set selection must keep loaded.
    pub fn tracked(&self) -> impl Iterator<Item = SectionPos> + '_ {
        self.settled.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ident::Detail;
    use crate::world::section::FINEST_DETAIL;

    fn cell(x: i32, detail: i8) -> (SectionPos, QuadrantMask) {
        (SectionPos { detail: Detail(detail), x, z: 0 }, QuadrantMask::ALL)
    }

    /// A cut replaces the prior one wholesale in one frame (hard pop): exactly the new
    /// cut draws, the old regions are gone.
    #[test]
    fn cut_hard_pops_to_exactly_the_new_cut() {
        let mut c = Coverage::default();
        let old = [cell(0, FINEST_DETAIL.0 + 1), cell(1, FINEST_DETAIL.0 + 1)];
        c.update(&old);
        let new = [cell(5, FINEST_DETAIL.0 + 2), cell(6, FINEST_DETAIL.0 + 2)];
        c.update(&new);
        let drawn: std::collections::HashMap<_, _> = c.draws().collect();
        assert_eq!(drawn.len(), new.len(), "exactly the new cut draws");
        for (pos, _) in new {
            assert!(drawn.contains_key(&pos), "new region present");
        }
        for (pos, _) in old {
            assert!(!drawn.contains_key(&pos), "old region gone");
        }
        assert_eq!(c.tracked().count(), new.len());
    }

    /// Build the mask with bits `b` (`0b1101` = quadrants 0, 2, 3).
    fn mask(b: u8) -> QuadrantMask {
        let mut m = QuadrantMask::EMPTY;
        for q in crate::world::section::Quadrant::ALL {
            if b & (1 << q.get()) != 0 {
                m.insert(q);
            }
        }
        m
    }

    /// Invariant: a mask maintained ONLY by applying the patches `update` returns is
    /// identical to one a reference draw walk rebuilds from scratch — over enter / leave /
    /// in-place-mask-change crossed with every quadrant mask `1..=15`.
    ///
    /// The mask the GPU holds is this projection resolved to slots, so an omission here
    /// is a stranded slot there. Covers the in-place quadrant-mask change specifically:
    /// step 2 mutates a region's mask without an enter/leave transition at all.
    #[test]
    fn patch_stream_reproduces_the_reference_draw_walk() {
        let pos = |x: i32| SectionPos { detail: Detail(FINEST_DETAIL.0 + 1), x, z: 0 };
        let script: Vec<Vec<(SectionPos, QuadrantMask)>> = (1..=15u8)
            .flat_map(|b| {
                [
                    // two regions at mask `b`
                    vec![(pos(0), mask(b)), (pos(1), mask(b))],
                    // in-place mask change on region 0, no enter/leave: THE TRAP
                    vec![(pos(0), mask(15 - b + 1)), (pos(1), mask(b))],
                    // region 1 leaves, region 2 enters
                    vec![(pos(0), mask(b)), (pos(2), mask(b))],
                    // everything leaves
                    vec![],
                ]
            })
            .collect();

        let mut c = Coverage::default();
        let mut patched: std::collections::HashMap<SectionPos, QuadrantMask> = Default::default();
        for (step, cut) in script.iter().enumerate() {
            for (pos, m) in c.update(cut) {
                match m {
                    Some(m) => patched.insert(pos, m),
                    None => patched.remove(&pos),
                };
            }
            let reference: std::collections::HashMap<SectionPos, QuadrantMask> = c.draws().collect();
            assert_eq!(patched, reference, "patch stream diverged from the draw walk at step {step}");
        }
    }

    /// Equal inputs produce equal draw sets (order-independent).
    #[test]
    fn update_is_deterministic() {
        let seq = [
            vec![cell(0, 3), cell(1, 3)],
            vec![cell(1, 3), cell(2, 4)],
            vec![cell(2, 4)],
        ];
        let run = || {
            let mut c = Coverage::default();
            for cut in &seq {
                c.update(cut);
            }
            let mut v: Vec<_> = c.draws().collect();
            v.sort_by_key(|(p, _)| (p.detail, p.x, p.z));
            v
        };
        assert_eq!(run(), run());
    }
}
