//! Quadtree selection and the far field's drawn cut.
//!
//! [`whole_cut`] draws every tile whole: a desired tile once it is Ready, until then the Ready
//! finer tiles under it (a merge waits for its parent) or a Ready ancestor (a split waits for all
//! four children). No tile is drawn in part and no two drawn tiles overlap.

use crate::ident::Detail;

use super::section::{Quadrant, SectionPos};
use super::{FastMap, FastSet};

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

/// Levels below an unready desired tile searched for Ready finer tiles to draw it with, when no
/// Ready ancestor can (the first load).
const STAND_IN_DEPTH: i8 = 2;

/// The tiles to draw for the cut `desired` (the leaves of complete splits, coarsest at `max_detail`)
/// given which tiles are Ready. A desired tile draws once it is Ready. Until every desired tile
/// under a node is, the node's nearest Ready ancestor draws whole in their place: a split waits
/// for all four children, and with ancestors resident a merge is at once. Only where no ancestor is
/// Ready do the Ready finer tiles under an unready desired tile draw it. Every tile is drawn whole
/// and no two overlap; a square nothing Ready covers stays empty. Sorted by
/// `(body, face, detail, x, z)`.
pub(in crate::world) fn whole_cut(desired: &[SectionPos], max_detail: Detail, ready: &impl Fn(SectionPos) -> bool) -> Vec<SectionPos> {
    let leaves: FastSet<SectionPos> = desired.iter().copied().collect();
    let mut inner: FastSet<SectionPos> = FastSet::default();
    let mut tops: Vec<SectionPos> = Vec::new();
    for &s in desired {
        let mut c = s;
        loop {
            if c.detail >= max_detail {
                tops.push(c);
                break;
            }
            let p = c.parent();
            if !inner.insert(p) {
                break;
            }
            c = p;
        }
    }
    tops.sort_unstable_by_key(|s| (s.body, s.face as u8, s.detail, s.x, s.z));
    tops.dedup();
    let mut cut = Cut { leaves: &leaves, inner: &inner, ready, done: FastMap::default() };
    let mut out = Vec::with_capacity(desired.len());
    for t in tops {
        cut.draw(t, &mut out);
    }
    out.sort_unstable_by_key(|s| (s.body, s.face as u8, s.detail, s.x, s.z));
    out
}

/// [`whole_cut`]'s walk: which nodes are complete (every desired tile under them Ready), memoised.
struct Cut<'a, F: Fn(SectionPos) -> bool> {
    leaves: &'a FastSet<SectionPos>,
    inner: &'a FastSet<SectionPos>,
    ready: &'a F,
    done: FastMap<SectionPos, bool>,
}

impl<F: Fn(SectionPos) -> bool> Cut<'_, F> {
    fn complete(&mut self, s: SectionPos) -> bool {
        if let Some(&c) = self.done.get(&s) {
            return c;
        }
        let c = if self.leaves.contains(&s) {
            (self.ready)(s)
        } else if self.inner.contains(&s) {
            Quadrant::ALL.into_iter().all(|q| self.complete(s.child(q)))
        } else {
            true
        };
        self.done.insert(s, c);
        c
    }

    fn draw(&mut self, s: SectionPos, out: &mut Vec<SectionPos>) {
        if self.leaves.contains(&s) {
            if (self.ready)(s) {
                out.push(s);
            } else {
                self.draw_below(s, STAND_IN_DEPTH, out);
            }
            return;
        }
        if !self.inner.contains(&s) {
            return;
        }
        if !self.complete(s) && (self.ready)(s) {
            out.push(s);
            return;
        }
        for q in Quadrant::ALL {
            self.draw(s.child(q), out);
        }
    }

    /// The Ready tiles at most `depth` levels down `s`'s square, coarsest first.
    fn draw_below(&self, s: SectionPos, depth: i8, out: &mut Vec<SectionPos>) {
        if depth <= 0 || s.detail.0 == 0 {
            return;
        }
        for q in Quadrant::ALL {
            let c = s.child(q);
            if (self.ready)(c) {
                out.push(c);
            } else {
                self.draw_below(c, depth - 1, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Face;

    fn xs(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A random cut of complete splits under one d6 root: each node splits with a chance falling
    /// with its detail.
    fn random_cut(state: &mut u64) -> Vec<SectionPos> {
        let root = SectionPos { detail: Detail(6), body: 0, face: Face::PosY, x: 3, z: -2 };
        let (mut stack, mut out) = (vec![root], Vec::new());
        while let Some(s) = stack.pop() {
            if s.detail.0 > 0 && xs(state) % 7 < u64::from(s.detail.0.unsigned_abs()) {
                stack.extend(Quadrant::ALL.map(|q| s.child(q)));
            } else {
                out.push(s);
            }
        }
        out
    }

    fn contains(a: SectionPos, b: SectionPos) -> bool {
        let mut c = b;
        while c.detail < a.detail {
            c = c.parent();
        }
        c == a
    }

    /// Over random cuts and random Ready sets: no two drawn tiles overlap; every drawn tile is a
    /// desired tile, a Ready ancestor of one drawn whole, or (only where no ancestor is Ready) a
    /// Ready finer tile; a desired tile with every desired tile of its parent Ready draws itself;
    /// and with every desired tile Ready the drawn tiles are exactly the cut.
    #[test]
    fn whole_cut_never_overlaps_and_draws_the_target_or_an_ancestor() {
        let mut state = 0x5EED_0F_C0DE_u64;
        for round in 0..400 {
            let cut = random_cut(&mut state);
            let all: FastSet<SectionPos> = cut.iter().flat_map(|&s| std::iter::successors(Some(s), |p| (p.detail.0 < 6).then(|| p.parent()))).collect();
            let mut ready: FastSet<SectionPos> = all.iter().copied().filter(|_| xs(&mut state) % 3 != 0).collect();
            // Some finer tiles under desired ones (leftovers of an earlier cut).
            for &s in &cut {
                if s.detail.0 > 0 && xs(&mut state) % 4 == 0 {
                    ready.extend(Quadrant::ALL.map(|q| s.child(q)));
                }
            }
            let drawn = whole_cut(&cut, Detail(6), &|p| ready.contains(&p));
            for (i, &a) in drawn.iter().enumerate() {
                assert!(ready.contains(&a), "round {round}: {a:?} drawn unready");
                for &b in &drawn[i + 1..] {
                    assert!(!contains(a, b) && !contains(b, a), "round {round}: {a:?} and {b:?} overlap");
                }
                let target = cut.contains(&a);
                let ancestor = cut.iter().any(|&c| c != a && contains(a, c));
                let finer = cut.iter().any(|&c| c != a && contains(c, a));
                assert!(target || ancestor || finer, "round {round}: {a:?} is not of the cut");
                if finer {
                    let c = *cut.iter().find(|&&c| contains(c, a)).expect("its desired tile");
                    let mut p = c;
                    while p.detail.0 < 6 {
                        p = p.parent();
                        assert!(!ready.contains(&p), "round {round}: {a:?} drawn finer while {p:?} is Ready");
                    }
                }
            }
            let ready_all: FastSet<SectionPos> = cut.iter().copied().collect();
            let mut full = whole_cut(&cut, Detail(6), &|p| ready_all.contains(&p));
            let mut want = cut.clone();
            full.sort_unstable_by_key(|s| (s.detail, s.x, s.z));
            want.sort_unstable_by_key(|s| (s.detail, s.x, s.z));
            assert_eq!(full, want, "round {round}: a ready cut draws exactly itself");
        }
    }

    /// A split waits for all four children: with three Ready the Ready parent draws whole, and
    /// once the fourth lands the children draw. A merge with the parent Ready draws it at once.
    #[test]
    fn a_split_waits_for_all_four_children() {
        let parent = SectionPos { detail: Detail(3), body: 0, face: Face::PosY, x: 0, z: 0 };
        let kids = Quadrant::ALL.map(|q| parent.child(q));
        let mut ready: FastSet<SectionPos> = [parent, kids[0], kids[1], kids[2]].into_iter().collect();
        assert_eq!(whole_cut(&kids, Detail(5), &|p| ready.contains(&p)), vec![parent]);
        ready.insert(kids[3]);
        assert_eq!(whole_cut(&kids, Detail(5), &|p| ready.contains(&p)).len(), 4);
        assert_eq!(whole_cut(&[parent], Detail(5), &|p| ready.contains(&p)), vec![parent], "a merge draws the parent at once");
    }
}
