//! The far field's drawn tiles. A new cut replaces the previous one in the same frame (hard pop,
//! no cross-fade), and every tile is drawn whole. This is a view of the selection that also emits
//! the diff callers need to patch GPU visibility.

use super::FastSet;
use super::section::SectionPos;

/// A tile whose visibility changed, and whether it now draws.
pub(in crate::world) type VisibilityChange = (SectionPos, bool);

/// The tiles the far field draws, replaced wholesale each cut.
#[derive(Default)]
pub(in crate::world) struct Coverage {
    drawn: FastSet<SectionPos>,
    /// The previous cut's set, kept for its allocation.
    spare: FastSet<SectionPos>,
}

impl Coverage {
    /// Adopt a new `cut`, replacing the prior one. Returns every tile whose visibility changed.
    pub fn update_now(&mut self, cut: &[SectionPos]) -> Vec<VisibilityChange> {
        let mut next = std::mem::take(&mut self.spare);
        next.clear();
        next.extend(cut.iter().copied());
        let mut changes: Vec<VisibilityChange> = self.drawn.iter().filter(|p| !next.contains(p)).map(|&p| (p, false)).collect();
        changes.extend(next.iter().filter(|p| !self.drawn.contains(p)).map(|&p| (p, true)));
        self.spare = std::mem::replace(&mut self.drawn, next);
        changes
    }

    /// Whether `pos` draws now: what a section consults when it lands Ready, since its slots did
    /// not exist at the transition that decided it.
    pub fn draws(&self, pos: SectionPos) -> bool {
        self.drawn.contains(&pos)
    }

    /// Every drawn tile: the set selection must keep loaded.
    pub fn tracked(&self) -> impl Iterator<Item = SectionPos> + '_ {
        self.drawn.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Face;
    use crate::ident::Detail;

    fn cell(x: i32, detail: i8) -> SectionPos {
        SectionPos { body: 0, face: Face::PosY, detail: Detail(detail), x, z: 0 }
    }

    /// The visibility a tile holds, maintained only by applying the diffs `update_now` returns,
    /// is exactly the cut adopted, through enters, leaves and a full clear.
    #[test]
    fn diffs_reproduce_the_cut() {
        let script = [vec![cell(0, 3), cell(1, 3)], vec![cell(1, 3), cell(2, 4)], vec![], vec![cell(5, 2)]];
        let mut c = Coverage::default();
        let mut patched: std::collections::HashSet<SectionPos> = Default::default();
        for cut in &script {
            for (pos, on) in c.update_now(cut) {
                if on {
                    assert!(patched.insert(pos), "{pos:?} shown twice");
                } else {
                    assert!(patched.remove(&pos), "{pos:?} hidden while hidden");
                }
            }
            let want: std::collections::HashSet<SectionPos> = cut.iter().copied().collect();
            assert_eq!(patched, want);
            assert!(cut.iter().all(|&p| c.draws(p)) && c.tracked().count() == cut.len());
        }
    }
}
