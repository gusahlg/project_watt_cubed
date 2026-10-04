//! Self-gravity of a deforming body: a Barnes–Hut octree over mass points (element centroids),
//! evaluated at the nodes. Pairs closer than the softening length use a Plummer kernel so a node
//! never feels a singular pull from the element it belongs to; the softened sum of a uniform ball
//! reproduces its interior field to the element scale. Bodies lie well inside the law's inner
//! range, where the kernel is exactly Newtonian.
//!
//! Deterministic: the tree is built from Morton-sorted points and evaluated in a fixed order.

use glam::DVec3;

use crate::gravity::G;

/// Opening criterion: a cell of size `s` at distance `d` is accepted as one mass when `s < θ d`.
pub const THETA: f64 = 0.6;

/// One mass point.
#[derive(Clone, Copy, Debug)]
pub struct Mass {
    pub at: DVec3,
    pub mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    com: DVec3,
    mass: f64,
    /// Edge of the cube this node covers.
    size: f64,
    /// Children index range in `nodes` (`first..first + count`), or a leaf range into `points`.
    first: u32,
    count: u32,
    leaf: bool,
}

/// The octree.
pub struct Tree {
    nodes: Vec<Node>,
    points: Vec<Mass>,
    soft2: f64,
}

const LEAF: usize = 8;

impl Tree {
    /// Build over `points` with Plummer softening length `soft`.
    pub fn build(points: &[Mass], soft: f64) -> Self {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in points {
            lo = lo.min(p.at);
            hi = hi.max(p.at);
        }
        let size = (hi - lo).max_element().max(1.0) * (1.0 + 1e-9);
        let key = |p: &Mass| morton(((p.at - lo) / size).clamp(DVec3::ZERO, DVec3::splat(1.0 - 1e-12)));
        let mut sorted: Vec<(u64, Mass)> = points.iter().map(|p| (key(p), *p)).collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.at.x.total_cmp(&b.1.at.x)));
        let keys: Vec<u64> = sorted.iter().map(|s| s.0).collect();
        let points: Vec<Mass> = sorted.into_iter().map(|s| s.1).collect();
        let mut tree = Tree { nodes: Vec::new(), points, soft2: soft * soft };
        if !tree.points.is_empty() {
            tree.nodes.push(Node { com: DVec3::ZERO, mass: 0.0, size, first: 0, count: 0, leaf: true });
            tree.split(0, 0, tree.points.len(), &keys, 0, size);
        }
        tree
    }

    /// Fill node `n` covering points `from..to` (sharing the top `level` Morton digits).
    fn split(&mut self, n: usize, from: usize, to: usize, keys: &[u64], level: u32, size: f64) {
        let (com, mass) = summary(&self.points[from..to]);
        if to - from <= LEAF || level >= 20 {
            self.nodes[n] = Node { com, mass, size, first: from as u32, count: (to - from) as u32, leaf: true };
            return;
        }
        let shift = 3 * (20 - 1 - level);
        let mut bounds = [from; 9];
        let mut i = from;
        for octant in 0..8u64 {
            bounds[octant as usize] = i;
            while i < to && (keys[i] >> shift) & 7 == octant {
                i += 1;
            }
        }
        bounds[8] = to;
        let first = self.nodes.len();
        let kids: Vec<(usize, usize)> = (0..8).map(|o| (bounds[o], bounds[o + 1])).filter(|(a, b)| b > a).collect();
        for _ in &kids {
            self.nodes.push(Node { com: DVec3::ZERO, mass: 0.0, size: size * 0.5, first: 0, count: 0, leaf: true });
        }
        self.nodes[n] = Node { com, mass, size, first: first as u32, count: kids.len() as u32, leaf: false };
        for (k, (a, b)) in kids.into_iter().enumerate() {
            self.split(first + k, a, b, keys, level + 1, size * 0.5);
        }
    }

    /// Gravitational acceleration at `p` (blocks/s²).
    pub fn accel(&self, p: DVec3) -> DVec3 {
        if self.nodes.is_empty() {
            return DVec3::ZERO;
        }
        let mut a = DVec3::ZERO;
        let mut stack = [0u32; 512];
        let mut top = 1usize;
        stack[0] = 0;
        while top > 0 {
            top -= 1;
            let node = self.nodes[stack[top] as usize];
            let d = node.com - p;
            let r2 = d.length_squared();
            if node.leaf {
                for m in &self.points[node.first as usize..(node.first + node.count) as usize] {
                    a += self.pull(m.at - p, m.mass);
                }
            } else if node.size * node.size < THETA * THETA * r2 {
                a += self.pull(d, node.mass);
            } else {
                for c in (node.first..node.first + node.count).rev() {
                    stack[top] = c;
                    top += 1;
                }
            }
        }
        a
    }

    #[inline]
    fn pull(&self, d: DVec3, mass: f64) -> DVec3 {
        let r2 = d.length_squared() + self.soft2;
        d * (G * mass / (r2 * r2.sqrt()))
    }
}

fn summary(points: &[Mass]) -> (DVec3, f64) {
    let mass: f64 = points.iter().map(|p| p.mass).sum();
    if mass <= 0.0 {
        let c = points.iter().map(|p| p.at).sum::<DVec3>() / points.len().max(1) as f64;
        return (c, 0.0);
    }
    (points.iter().map(|p| p.at * p.mass).sum::<DVec3>() / mass, mass)
}

/// 60-bit Morton code of a point in `[0, 1)³` (20 bits per axis).
fn morton(u: DVec3) -> u64 {
    let q = |v: f64| ((v * (1u64 << 20) as f64) as u64).min((1 << 20) - 1);
    let spread = |mut x: u64| {
        x &= 0xF_FFFF;
        x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
        x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
        x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
        x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
        x = (x | (x << 2)) & 0x1249_2492_4924_9249;
        x
    };
    spread(q(u.x)) | (spread(q(u.y)) << 1) | (spread(q(u.z)) << 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform ball sampled on a grid: the tree's field matches the analytic interior (linear in r)
    /// and exterior (inverse square) profiles to a few percent away from the surface.
    #[test]
    fn a_sampled_ball_pulls_like_a_ball() {
        let (r, rho, h) = (1000.0f64, 5.0, 50.0);
        let mut pts = Vec::new();
        let n = (r / h) as i32 + 1;
        for k in -n..=n {
            for j in -n..=n {
                for i in -n..=n {
                    let at = DVec3::new(i as f64, j as f64, k as f64) * h;
                    if at.length() <= r {
                        pts.push(Mass { at, mass: rho * h * h * h });
                    }
                }
            }
        }
        let total: f64 = pts.iter().map(|p| p.mass).sum();
        let tree = Tree::build(&pts, 0.5 * h);
        let m_ball = total;
        for (dist, inside) in [(400.0, true), (700.0, true), (2000.0, false), (5000.0, false)] {
            let p = DVec3::new(dist, 0.3 * h, 0.1 * h);
            let g = tree.accel(p).length();
            let want = if inside { G * m_ball * dist / (r * r * r) } else { G * m_ball / (dist * dist) };
            assert!((g - want).abs() / want < 0.05, "r={dist}: {g} vs {want}");
            assert!(tree.accel(p).x < 0.0, "points inward");
        }
        // Near the centre the pull vanishes.
        assert!(tree.accel(DVec3::new(0.2 * h, 0.1 * h, 0.0)).length() < 0.05 * G * m_ball / (r * r));
    }

    #[test]
    fn the_tree_is_deterministic() {
        let pts: Vec<Mass> = (0..500)
            .map(|i| {
                let f = i as f64;
                Mass { at: DVec3::new((f * 7.3).sin() * 100.0, (f * 3.1).cos() * 80.0, f * 0.37), mass: 1.0 + (i % 7) as f64 }
            })
            .collect();
        let a = Tree::build(&pts, 2.0).accel(DVec3::new(5.0, 6.0, 7.0));
        let b = Tree::build(&pts, 2.0).accel(DVec3::new(5.0, 6.0, 7.0));
        assert_eq!(a.to_array().map(f64::to_bits), b.to_array().map(f64::to_bits));
    }
}
