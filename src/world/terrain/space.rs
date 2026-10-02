//! Space: above [`SPACE_FLOOR`] the sky is a field of planets, moons, rings, asteroids and stars.
//!
//! Planets live one per 256 m cell (if any): a sphere with a noisy surface, a crust, a mantle and
//! a glowing core, of one of six kinds. Some wear a flat ring of rubble, some a moon. Between them
//! drift small asteroids; single stars hang in the dark.

use std::sync::Arc;

use super::noise::{hash3, perlin3, unit};
use super::{Materials, SPACE_FLOOR};
use crate::block::registry::{AIR, BlockId};

/// Planet cell edge.
const CELL: i32 = 256;
/// Planet cells start this far above the floor (the lowest planets float well clear of it).
const FIRST: i32 = SPACE_FLOOR + 32;
/// Asteroid cell edge.
const ROCK_CELL: i32 = 48;
/// Star cell edge.
const STAR_CELL: i32 = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Rocky,
    Icy,
    Verdant,
    Desert,
    Crystal,
    Molten,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Planet {
    pub(super) c: [i32; 3],
    pub(super) r: f32,
    kind: Kind,
    seed: u32,
    ring: Option<(f32, f32)>,
    moon: Option<([i32; 3], f32)>,
}

impl Planet {
    /// Half the edge of a box around everything this planet paints.
    fn reach(&self) -> i32 {
        let ring = self.ring.map_or(0.0, |(_, out)| out);
        let moon = self.moon.map_or(0.0, |(m, r)| {
            let d = ((m[0] - self.c[0]).abs().max((m[1] - self.c[1]).abs()).max((m[2] - self.c[2]).abs())) as f32;
            d + r
        });
        (self.r * 1.15).max(ring).max(moon) as i32 + 2
    }
}

pub struct Space {
    s: u32,
    density: f32,
    m: Arc<Materials>,
}

impl Space {
    pub fn new(s: u32, density: f32, m: Arc<Materials>) -> Self {
        Self { s, density, m }
    }

    pub(super) fn planet(&self, ix: i32, iy: i32, iz: i32) -> Option<Planet> {
        if iy < 0 || self.density <= 0.0 {
            return None;
        }
        let h = hash3(self.s, ix, iy, iz);
        if unit(h) >= 0.5 * self.density.min(2.0) {
            return None;
        }
        let g = |k: u32| hash3(self.s ^ k.wrapping_mul(0x9E37_79B9), ix, iy, iz);
        let r = 14.0 + (g(1) % 38) as f32;
        let kind = [Kind::Rocky, Kind::Icy, Kind::Verdant, Kind::Desert, Kind::Crystal, Kind::Molten][(g(2) % 6) as usize];
        let ring = (unit(g(3)) < 0.3).then(|| (r * 1.45, r * (1.9 + unit(g(4)) * 0.4)));
        let moon_r = (r * 0.22).max(4.0);
        let moon = (unit(g(5)) < 0.45).then(|| {
            let d = ((r * 2.3 + moon_r) as i32).min(100);
            let a = g(6) % 8;
            let (dx, dz) = [(d, 0), (0, d), (-d, 0), (0, -d), (d * 7 / 10, d * 7 / 10), (-d * 7 / 10, d * 7 / 10), (d * 7 / 10, -d * 7 / 10), (-d * 7 / 10, -d * 7 / 10)][a as usize];
            ([dx, (g(7) % 21) as i32 - 10, dz], moon_r)
        });
        let mut p = Planet { c: [0; 3], r, kind, seed: g(8), ring, moon };
        // Centre the planet so everything it paints stays inside its cell.
        let reach = p.reach().min(CELL / 2 - 2);
        let free = (CELL - 2 * reach).max(1) as u32;
        let base = [ix * CELL, FIRST + iy * CELL, iz * CELL];
        p.c = std::array::from_fn(|a| base[a] + reach + (g(9 + a as u32) % free) as i32);
        if let Some((m, r)) = p.moon {
            p.moon = Some(([p.c[0] + m[0], p.c[1] + m[1], p.c[2] + m[2]], r));
        }
        Some(p)
    }

    fn planet_cell(&self, x: i32, y: i32, z: i32) -> Option<Planet> {
        if y < FIRST {
            return None;
        }
        self.planet(x.div_euclid(CELL), (y - FIRST).div_euclid(CELL), z.div_euclid(CELL))
    }

    /// A noisy sphere: signed depth below its surface (positive inside).
    fn depth(seed: u32, c: [i32; 3], r: f32, x: i32, y: i32, z: i32) -> f32 {
        let (dx, dy, dz) = ((x - c[0]) as f32, (y - c[1]) as f32, (z - c[2]) as f32);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d > r * 1.15 + 1.0 {
            return -1.0;
        }
        let k = 3.0 / r as f64;
        let bump = 0.09 * perlin3(seed, dx as f64 * k, dy as f64 * k, dz as f64 * k)
            + 0.03 * perlin3(seed ^ 0x55, dx as f64 * k * 3.0, dy as f64 * k * 3.0, dz as f64 * k * 3.0);
        r * (1.0 + bump) - d
    }

    fn planet_block(&self, p: &Planet, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let m = &self.m;
        let depth = Self::depth(p.seed, p.c, p.r, x, y, z);
        if depth >= 0.0 {
            let core = depth > p.r * 0.72;
            let (crust, mantle, heart) = match p.kind {
                Kind::Rocky => (m.regolith, m.basalt, m.magma),
                Kind::Icy => (m.frost, m.ice, m.core),
                Kind::Verdant => (m.moss, m.soil, m.core),
                Kind::Desert => (m.ochre, m.sandstone[0], m.magma),
                Kind::Crystal => (m.violet, m.crystal, m.core),
                Kind::Molten => (m.basalt, m.basalt, m.magma),
            };
            if core {
                return Some(heart);
            }
            if p.kind == Kind::Molten {
                // Rivers of magma crack the black crust: thin ridges of the noise (|n| small),
                // wider with depth so the seams read as lit fissures from orbit.
                let (dx, dy, dz) = ((x - p.c[0]) as f64, (y - p.c[1]) as f64, (z - p.c[2]) as f64);
                let n = perlin3(p.seed ^ 0x77, dx / 9.0, dy / 9.0, dz / 9.0).abs();
                if n < 0.06 + 0.01 * depth.min(8.0) {
                    return Some(m.magma);
                }
            }
            return Some(if depth < 2.0 { crust } else { mantle });
        }
        if let Some((inner, outer)) = p.ring {
            let (dx, dz) = ((x - p.c[0]) as f32, (z - p.c[2]) as f32);
            let d = (dx * dx + dz * dz).sqrt();
            let dy = y - p.c[1];
            if (0..=1).contains(&dy) && d >= inner && d <= outer {
                let n = perlin3(p.seed ^ 0x313, x as f64 / 5.0, dy as f64, z as f64 / 5.0);
                if n > 0.05 {
                    return Some(if p.kind == Kind::Icy { m.frost } else { m.regolith });
                }
            }
        }
        if let Some((c, r)) = p.moon
            && Self::depth(p.seed ^ 0xA0A0, c, r, x, y, z) >= 0.0
        {
            return Some(m.regolith);
        }
        None
    }

    fn asteroid_block(&self, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let (ix, iy, iz) = (x.div_euclid(ROCK_CELL), y.div_euclid(ROCK_CELL), z.div_euclid(ROCK_CELL));
        let h = hash3(self.s ^ 0xA57E, ix, iy, iz);
        if unit(h) >= 0.16 * self.density.min(2.0) {
            return None;
        }
        let r = 2.0 + (h % 5) as f32;
        let margin = 10;
        let free = (ROCK_CELL - 2 * margin) as u32;
        let c = [
            ix * ROCK_CELL + margin + (hash3(h, 1, 0, 0) % free) as i32,
            iy * ROCK_CELL + margin + (hash3(h, 2, 0, 0) % free) as i32,
            iz * ROCK_CELL + margin + (hash3(h, 3, 0, 0) % free) as i32,
        ];
        if c[1] < SPACE_FLOOR + 8 {
            return None;
        }
        (Self::depth(h, c, r, x, y, z) >= 0.0).then(|| {
            let m = &self.m;
            match h % 9 {
                0 => m.gold,
                1 => m.basalt,
                2 => m.frost,
                _ => m.regolith,
            }
        })
    }

    fn star_block(&self, x: i32, y: i32, z: i32) -> Option<BlockId> {
        let (ix, iy, iz) = (x.div_euclid(STAR_CELL), y.div_euclid(STAR_CELL), z.div_euclid(STAR_CELL));
        let h = hash3(self.s ^ 0x57A2, ix, iy, iz);
        if unit(h) >= 0.03 {
            return None;
        }
        let at = [
            ix * STAR_CELL + (h % 16) as i32,
            iy * STAR_CELL + ((h >> 4) % 16) as i32,
            iz * STAR_CELL + ((h >> 8) % 16) as i32,
        ];
        (at == [x, y, z] && y >= SPACE_FLOOR + 4).then_some(self.m.star)
    }

    /// The block at a cell at or above [`SPACE_FLOOR`].
    pub fn block(&self, x: i32, y: i32, z: i32) -> BlockId {
        if let Some(p) = self.planet_cell(x, y, z)
            && let Some(id) = self.planet_block(&p, x, y, z)
        {
            return id;
        }
        if self.density > 0.0
            && let Some(id) = self.asteroid_block(x, y, z)
        {
            return id;
        }
        self.star_block(x, y, z).unwrap_or(AIR)
    }

    /// Conservative: false only when the `n`-cube at `(x0, y0, z0)` is certainly empty.
    pub fn may_touch(&self, x0: i32, y0: i32, z0: i32, n: i32) -> bool {
        // Stars: the cube is a whole number of star cells (16 | n), test each.
        let mut sy = y0.div_euclid(STAR_CELL);
        while sy * STAR_CELL < y0 + n {
            let mut sz = z0.div_euclid(STAR_CELL);
            while sz * STAR_CELL < z0 + n {
                let mut sx = x0.div_euclid(STAR_CELL);
                while sx * STAR_CELL < x0 + n {
                    if unit(hash3(self.s ^ 0x57A2, sx, sy, sz)) < 0.03 {
                        return true;
                    }
                    sx += 1;
                }
                sz += 1;
            }
            sy += 1;
        }
        if self.density <= 0.0 {
            return false;
        }
        // Asteroids: their whole cell.
        let rc = |v: i32| v.div_euclid(ROCK_CELL);
        for iy in rc(y0)..=rc(y0 + n - 1) {
            for iz in rc(z0)..=rc(z0 + n - 1) {
                for ix in rc(x0)..=rc(x0 + n - 1) {
                    if unit(hash3(self.s ^ 0xA57E, ix, iy, iz)) < 0.16 * self.density.min(2.0) {
                        return true;
                    }
                }
            }
        }
        // Planets: the box around everything a planet paints.
        if y0 + n <= FIRST {
            return false;
        }
        let pc = |v: i32| v.div_euclid(CELL);
        let py = |v: i32| (v - FIRST).div_euclid(CELL);
        for iy in py(y0.max(FIRST))..=py(y0 + n - 1) {
            for iz in pc(z0)..=pc(z0 + n - 1) {
                for ix in pc(x0)..=pc(x0 + n - 1) {
                    if let Some(p) = self.planet(ix, iy, iz) {
                        let r = p.reach();
                        let hit = (0..3).all(|a| {
                            let (lo, hi) = ([x0, y0, z0][a], [x0, y0, z0][a] + n - 1);
                            p.c[a] + r >= lo && p.c[a] - r <= hi
                        });
                        if hit {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}
