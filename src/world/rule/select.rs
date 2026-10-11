//! The rule's cut: a top-down traversal of every root near the eye that splits a node while the
//! eye is within its level's range. Its leaves partition the roots, each is the detail its distance
//! gives, and nothing past the horizon is drawn. The eye is a box (the cell holding it); a point is
//! the box holding only itself.

use voxel_engine::DVec3;

use super::{PhysBox, Rule};
use crate::world::section::{Quadrant, SectionPos};

/// Where the rule finds its trees: the roots near an eye and the physical box of any node.
pub(in crate::world) trait Surface {
    /// Every root (a tile of `rule.coarsest`) whose box may come within the rule's horizon of `eye`.
    fn roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>);
    /// The node's physical box; `None` where it holds no surface (the subtree is skipped).
    fn node_box(&mut self, s: SectionPos) -> Option<PhysBox>;
    /// Whether the node's box is final. One that is not (its terrain still being read) is drawn as
    /// it is rather than split on a box that is only a bound.
    fn settled(&mut self, _s: SectionPos) -> bool {
        true
    }
}

/// One tile of the cut and its distance from the eye.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::world) struct Leaf {
    pub pos: SectionPos,
    pub dist: f64,
}

/// Most leaves a cut may hold. The rule bounds the cut by construction (about 640 tiles); a cut
/// that runs past this is a broken box, and the traversal stops rather than exhaust memory.
const LEAF_GUARD: usize = 1 << 14;

impl Rule {
    /// The cut around `eye` on `surface`, into `out` (cleared first). `stack` is reused scratch.
    pub fn select(&self, eye: &PhysBox, surface: &mut impl Surface, stack: &mut Vec<SectionPos>, out: &mut Vec<Leaf>) {
        out.clear();
        stack.clear();
        surface.roots(self, eye.centre(), stack);
        let horizon = self.horizon();
        while let Some(s) = stack.pop() {
            if out.len() + stack.len() > LEAF_GUARD {
                debug_assert!(false, "the rule's cut ran past {LEAF_GUARD} nodes at {s:?}");
                break;
            }
            let Some(b) = surface.node_box(s) else { continue };
            let dist = b.dist_box(eye);
            if dist >= horizon {
                continue;
            }
            if self.splits(s.detail.0, dist, b.span) && surface.settled(s) {
                stack.extend(Quadrant::ALL.map(|q| s.child(q)));
            } else {
                out.push(Leaf { pos: s, dist });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Face;
    use crate::ident::Detail;
    use crate::space::chart::{Map, basis};
    use std::collections::HashMap;

    /// Synthetic terrain in blocks over a plane coordinate, and a bound on its slope.
    fn hills(x: f64, z: f64) -> f64 {
        120.0 * (x / 1300.0).sin() * (z / 1700.0).cos() + 40.0 * ((x + 2.0 * z) / 233.0).sin()
    }
    const SLOPE: f64 = 0.5;

    /// Min and max of `h` over the square `[x0, x0+span)²`, sampled and padded by the slope.
    fn bounds(h: &dyn Fn(f64, f64) -> f64, x0: f64, z0: f64, span: f64) -> (f64, f64) {
        let n = 16;
        let step = span / f64::from(n);
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for i in 0..=n {
            for j in 0..=n {
                let y = h(x0 + step * f64::from(i), z0 + step * f64::from(j));
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
        let pad = SLOPE * step;
        (lo - pad, hi + pad)
    }

    /// An endless plane of terrain `h`, in physical space as it is.
    struct Plane {
        h: fn(f64, f64) -> f64,
    }

    impl Surface for Plane {
        fn roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
            let span = f64::from(SectionPos { detail: Detail(rule.coarsest), body: 0, face: Face::PosY, x: 0, z: 0 }.span());
            let reach = (rule.horizon() / span).ceil() as i32 + 1;
            let (cx, cz) = ((eye.x / span).floor() as i32, (eye.z / span).floor() as i32);
            for z in cz - reach..=cz + reach {
                for x in cx - reach..=cx + reach {
                    out.push(SectionPos { detail: Detail(rule.coarsest), body: 0, face: Face::PosY, x, z });
                }
            }
        }

        fn node_box(&mut self, s: SectionPos) -> Option<PhysBox> {
            let (x0, z0, span) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
            let (lo, hi) = bounds(&self.h, x0, z0, span);
            Some(PhysBox { lo: DVec3::new(x0, lo, z0), hi: DVec3::new(x0 + span, hi, z0 + span), span })
        }
    }

    /// A cube-sphere of radius `radius` with terrain `h` (blocks over local face coordinates), each
    /// face a tree of `n`-block local coordinates `[0, n)²` (a power-of-two multiple of the root).
    struct Ball {
        radius: f64,
        n: i32,
        h: fn(f64, f64) -> f64,
    }

    impl Ball {
        /// The physical point at local `(x, z)` of `face`, `up` blocks over the radius.
        fn point(&self, face: Face, x: f64, z: f64, up: f64) -> DVec3 {
            let (tu, nn, tv) = basis(face);
            let n = f64::from(self.n);
            let d = Map::Equiangular.dir(2.0 * x / n - 1.0, 2.0 * z / n - 1.0);
            (tu * d.x + nn * d.y + tv * d.z) * (self.radius + up)
        }

        /// The face and local coordinates of physical direction `p`.
        fn locate(&self, p: DVec3) -> (Face, f64, f64) {
            let face = Face::from_dominant(p);
            let (tu, nn, tv) = basis(face);
            let (xi, eta) = Map::Equiangular.inverse(DVec3::new(p.dot(tu), p.dot(nn), p.dot(tv)));
            let n = f64::from(self.n);
            (face, (xi + 1.0) * 0.5 * n, (eta + 1.0) * 0.5 * n)
        }
    }

    impl Surface for Ball {
        fn roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
            let span = SectionPos { detail: Detail(rule.coarsest), body: 0, face: Face::PosY, x: 0, z: 0 }.span();
            for face in Face::ALL {
                for z in 0..self.n / span {
                    for x in 0..self.n / span {
                        let s = SectionPos { detail: Detail(rule.coarsest), body: 0, face, x, z };
                        if self.box_of(s).is_some_and(|b| b.dist(eye) < rule.horizon()) {
                            out.push(s);
                        }
                    }
                }
            }
        }

        fn node_box(&mut self, s: SectionPos) -> Option<PhysBox> {
            self.box_of(s)
        }
    }

    impl Ball {
        fn box_of(&self, s: SectionPos) -> Option<PhysBox> {
            let (x0, z0, span) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
            let (lo, hi) = bounds(&self.h, x0, z0, span);
            let mut pts = Vec::with_capacity(18);
            for (i, j) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 1.0), (0.5, 0.5)] {
                for up in [lo, hi] {
                    pts.push(self.point(s.face, x0 + span * i, z0 + span * j, up));
                }
            }
            // The face stretches its cells by up to √2 at the corners, and the sphere bulges
            // between sampled points by the sagitta of half the stretched diagonal.
            let half_diag = span * std::f64::consts::SQRT_2;
            let side = (pts[2] - pts[0]).length().max((pts[4] - pts[0]).length());
            PhysBox::around(pts, half_diag * half_diag / (8.0 * self.radius) + 1.0, side)
        }
    }

    fn leaf_map(leaves: &[Leaf]) -> HashMap<SectionPos, f64> {
        leaves.iter().map(|l| (l.pos, l.dist)).collect()
    }

    /// The leaf of `cut` holding local point `(x, z)` of `face`.
    fn leaf_at(cut: &HashMap<SectionPos, f64>, rule: &Rule, face: Face, x: f64, z: f64) -> Option<SectionPos> {
        (rule.finest..=rule.coarsest).find_map(|d| {
            let span = f64::from(SectionPos { detail: Detail(d), body: 0, face, x: 0, z: 0 }.span());
            let s = SectionPos { detail: Detail(d), body: 0, face, x: (x / span).floor() as i32, z: (z / span).floor() as i32 };
            cut.contains_key(&s).then_some(s)
        })
    }

    /// P1: each leaf is the detail its distance gives (no split it should take, and its parent
    /// would split). P4: no leaf is two levels coarser than a farther one.
    fn assert_rule(name: &str, rule: &Rule, surface: &mut impl Surface, eye: DVec3, leaves: &[Leaf]) {
        assert!(!leaves.is_empty(), "{name}: empty cut");
        for l in leaves {
            let d = l.pos.detail.0;
            let span = surface.node_box(l.pos).expect("a box").span;
            assert!(!rule.splits(d, l.dist, span), "{name}: {:?} at {} should split", l.pos, l.dist);
            if d < rule.coarsest {
                let p = surface.node_box(l.pos.parent()).expect("parent box");
                assert!(rule.splits(d + 1, p.dist(eye), p.span), "{name}: {:?}'s parent at {} would not split", l.pos, p.dist(eye));
            }
        }
        let mut by_dist: Vec<&Leaf> = leaves.iter().collect();
        by_dist.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let mut farther: Option<&Leaf> = None;
        for l in by_dist.iter().rev() {
            if let Some(f) = farther {
                assert!(l.pos.detail.0 <= f.pos.detail.0 + 1, "{name}: {:?} at {} is two levels coarser than {:?} at {}", l.pos, l.dist, f.pos, f.dist);
            }
            if farther.is_none_or(|f| l.pos.detail.0 < f.pos.detail.0) {
                farther = Some(l);
            }
        }
    }

    /// P3 on one plane: each leaf's edge neighbours are within one level.
    fn assert_balance(name: &str, rule: &Rule, cut: &HashMap<SectionPos, f64>, face: Face, inside: impl Fn(f64, f64) -> bool) {
        for (&s, _) in cut.iter().filter(|(s, _)| s.face == face) {
            let (x0, z0, w) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
            for (x, z) in [(x0 - 0.5, z0 + w / 2.0), (x0 + w + 0.5, z0 + w / 2.0), (x0 + w / 2.0, z0 - 0.5), (x0 + w / 2.0, z0 + w + 0.5)] {
                if !inside(x, z) {
                    continue;
                }
                if let Some(n) = leaf_at(cut, rule, face, x, z) {
                    assert!((n.detail.0 - s.detail.0).abs() <= 1, "{name}: {s:?} beside {n:?}");
                }
            }
        }
    }

    fn cut_of(rule: &Rule, surface: &mut impl Surface, eye: DVec3) -> Vec<Leaf> {
        let (mut stack, mut out) = (Vec::new(), Vec::new());
        rule.select(&PhysBox::point(eye), surface, &mut stack, &mut out);
        out
    }

    /// On hilly terrain at render distances 6 to 32, from the ground and from 300 and 2000 blocks
    /// up, every leaf is the detail its distance gives, detail never falls inward, neighbours stay
    /// within one level, and the leaves tile the ground under the horizon exactly once.
    #[test]
    fn plane_cut_follows_the_rule() {
        let mut plane = Plane { h: hills };
        for rd in [6, 8, 12, 16, 24, 32] {
            let rule = Rule::new(f64::from(rd * 16), 9);
            for (i, up) in [2.0, 300.0, 2_000.0].into_iter().enumerate() {
                let (x, z) = (37.0 + 1_011.0 * i as f64, -12.0 - 517.0 * i as f64);
                let eye = DVec3::new(x, hills(x, z) + up, z);
                let name = format!("render distance {rd}, {up} up");
                let leaves = cut_of(&rule, &mut plane, eye);
                assert_rule(&name, &rule, &mut plane, eye, &leaves);
                let cut = leaf_map(&leaves);
                let disc = rule.horizon() * 0.5;
                assert_balance(&name, &rule, &cut, Face::PosY, |px, pz| (px - x).hypot(pz - z) < disc);
                // Exactly one leaf over each sampled column inside half the horizon.
                for k in 0..2_000 {
                    let (a, r) = (f64::from(k) * 2.399_963, disc * (f64::from(k) / 2_000.0).sqrt());
                    let (px, pz) = (x + r * a.cos(), z + r * a.sin());
                    let over = cut.keys().filter(|s| {
                        let (x0, z0, w) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
                        px >= x0 && px < x0 + w && pz >= z0 && pz < z0 + w
                    });
                    assert_eq!(over.count(), 1, "{name}: column ({px}, {pz})");
                }
            }
        }
    }

    /// On a cube-sphere, across its face seams: every leaf is the detail its distance gives, detail
    /// never falls inward, and neighbours (across a seam too) stay within one level.
    #[test]
    fn ball_cut_follows_the_rule_across_seams() {
        let mut ball = Ball { radius: 150_000.0, n: 1 << 18, h: hills };
        let rule = Rule::new(256.0, 9);
        let n = f64::from(ball.n);
        for (face, x, z, up) in [(Face::PosY, n / 2.0, n / 2.0, 2.0), (Face::PosY, n - 40.0, n / 3.0, 2.0), (Face::PosX, n - 3.0, n - 5.0, 250.0)] {
            let eye = ball.point(face, x, z, hills(x, z) + up);
            let name = format!("{face:?} ({x}, {z}) {up} up");
            let leaves = cut_of(&rule, &mut ball, eye);
            assert_rule(&name, &rule, &mut ball, eye, &leaves);
            let cut = leaf_map(&leaves);
            for f in Face::ALL {
                assert_balance(&name, &rule, &cut, f, |px, pz| px >= 0.0 && pz >= 0.0 && px < n && pz < n);
            }
            // Across each seam: the leaves either side of a point on a face edge.
            for (&s, _) in &cut {
                let (x0, z0, w) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
                for (px, pz) in [(x0, z0 + w / 2.0), (x0 + w, z0 + w / 2.0), (x0 + w / 2.0, z0), (x0 + w / 2.0, z0 + w)] {
                    let on_edge = px <= 0.0 || pz <= 0.0 || px >= n || pz >= n;
                    if !on_edge {
                        continue;
                    }
                    let inward = DVec3::new(x0 + w / 2.0 - px, 0.0, z0 + w / 2.0 - pz).normalize() * 0.5;
                    let across = ball.point(s.face, px - inward.x, pz - inward.z, 0.0);
                    let (f2, x2, z2) = ball.locate(across);
                    if f2 == s.face {
                        continue;
                    }
                    if let Some(o) = leaf_at(&cut, &rule, f2, x2, z2) {
                        assert!((o.detail.0 - s.detail.0).abs() <= 1, "{name}: {s:?} across the seam from {o:?}");
                    }
                }
            }
        }
    }

    /// Over flat ground the cut is a function of distance alone: turning the eye's surroundings a
    /// quarter turn about a grid corner under it turns the cut with them.
    #[test]
    fn flat_cut_is_rotation_invariant() {
        let mut flat = Plane { h: |_, _| 0.0 };
        let rule = Rule::new(256.0, 9);
        let eye = DVec3::new(0.0, 40.0, 0.0);
        let cut = leaf_map(&cut_of(&rule, &mut flat, eye));
        for (&s, _) in &cut {
            let turned = SectionPos { x: -s.z - 1, z: s.x, ..s };
            assert!(cut.contains_key(&turned), "{s:?} has no quarter-turned twin");
        }
    }

    /// Every shell holds the same count of tiles, so the cut stays near five hundred at every
    /// render distance.
    #[test]
    fn cut_size_is_bounded_at_every_render_distance() {
        let mut plane = Plane { h: hills };
        for rd in [2, 4, 6, 8, 12, 16, 20, 24, 31, 32] {
            let rule = Rule::new(f64::from(rd * 16), 9);
            let eye = DVec3::new(5.0, hills(5.0, 7.0) + 2.0, 7.0);
            let leaves = cut_of(&rule, &mut plane, eye);
            let mut per = [0usize; 10];
            for l in &leaves {
                per[l.pos.detail.0 as usize] += 1;
            }
            println!("render distance {rd}: finest d{} {} leaves {per:?}", rule.finest, leaves.len());
            assert!(leaves.len() <= CUT_BOUND, "render distance {rd}: {} leaves", leaves.len());
        }
    }

    /// Most tiles a cut holds on open ground, at any render distance.
    const CUT_BOUND: usize = 640;

    /// Where 2:1 balance breaks. It holds while a finest far tile's 3D extent stays under its
    /// parent's range, so ordinary relief keeps neighbours within one level (the tests above). A
    /// tower that rises most of the way to an eye 1000 blocks up pulls the tiles holding it two
    /// levels finer than the open ground beside them; the skirt walls seal that step. A tower of
    /// ordinary height does not.
    #[test]
    fn balance_breaks_only_beside_towering_relief() {
        fn step(tower: f64) -> Option<(SectionPos, SectionPos)> {
            let h = move |x: f64, z: f64| if (400.0..416.0).contains(&x) && (100.0..116.0).contains(&z) { tower } else { 0.0 };
            struct Tower<F: Fn(f64, f64) -> f64>(F);
            impl<F: Fn(f64, f64) -> f64> Surface for Tower<F> {
                fn roots(&self, rule: &Rule, eye: DVec3, out: &mut Vec<SectionPos>) {
                    Plane { h: |_, _| 0.0 }.roots(rule, eye, out);
                }
                fn node_box(&mut self, s: SectionPos) -> Option<PhysBox> {
                    let (x0, z0, span) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
                    let n = 32;
                    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                    for i in 0..=n {
                        for j in 0..=n {
                            let (x, z) = (x0 + span * f64::from(i) / f64::from(n), z0 + span * f64::from(j) / f64::from(n));
                            let y = (self.0)(x.min(x0 + span - 0.5), z.min(z0 + span - 0.5));
                            lo = lo.min(y);
                            hi = hi.max(y);
                        }
                    }
                    Some(PhysBox { lo: DVec3::new(x0, lo, z0), hi: DVec3::new(x0 + span, hi, z0 + span), span })
                }
            }
            let rule = Rule::new(256.0, 9);
            let cut = leaf_map(&cut_of(&rule, &mut Tower(h), DVec3::new(0.0, 1_000.0, 0.0)));
            let mut found = None;
            for &s in cut.keys() {
                let (x0, z0, w) = (f64::from(s.min_x()), f64::from(s.min_z()), f64::from(s.span()));
                for (x, z) in [(x0 - 0.5, z0 + w / 2.0), (x0 + w + 0.5, z0 + w / 2.0), (x0 + w / 2.0, z0 - 0.5), (x0 + w / 2.0, z0 + w + 0.5)] {
                    if let Some(n) = leaf_at(&cut, &rule, Face::PosY, x, z)
                        && n.detail.0 >= s.detail.0 + 2
                    {
                        found = Some((s, n));
                    }
                }
            }
            found
        }
        let (fine, coarse) = step(950.0).expect("a 950-block tower 400 blocks out gives a two-level step");
        let p = fine.parent();
        let holds = |a: i32, at: i32| a <= at && at < a + p.span();
        assert!(holds(p.min_x(), 400) && holds(p.min_z(), 100), "the fine side {fine:?} is a child of the tile holding the tower");
        assert_eq!(coarse.detail.0 - fine.detail.0, 2, "{fine:?} beside {coarse:?}");
        assert_eq!(step(150.0), None, "a 150-block tower keeps neighbours within one level");
    }
}
