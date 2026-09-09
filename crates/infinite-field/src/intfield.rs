//! Integer InfiniteDiffusion field: Q16.16 tiles, a Jacobi stencil, and an LRU cache.
//!
//! Values are `i32` Q16.16. A phase never reads its own writes; stencil samples
//! are the previous phase's blended field (phase 0 reads the prior). Tile order
//! and query order cannot change values.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::inoise::{div_floor, hash32, mul_q16, uniform_q16, value_noise_q16, ONE};

/// How tiles are laid out and how many denoising phases run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntSpec {
    /// World seed mixed into the noise lattice.
    pub seed: u32,
    /// Tile edge in samples (e.g. 32 or 64).
    pub tile: u32,
    /// Tile origin stride; `tile/2` is 50% overlap.
    pub stride: u32,
    /// Denoising phases.
    pub phases: u32,
    /// Planar channels stored per sample.
    pub channels: u32,
}

impl IntSpec {
    /// A compact default: 32² tiles, 50% overlap, 2 phases, 4 channels.
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            tile: 32,
            stride: 16,
            phases: 2,
            channels: 4,
        }
    }

    fn valid(self) -> bool {
        self.tile >= 4
            && self.stride >= 1
            && self.stride <= self.tile
            && self.phases >= 1
            && self.phases <= 12
            && self.channels >= 1
            && self.channels <= 16
    }
}

/// Previous-phase (or prior) samples around one cell. `|dx|`, `|dz|` ≤ 2.
pub struct Stencil<'a> {
    halo: &'a [i32],
    halo_w: u32,
    channels: u32,
    lx: u32,
    lz: u32,
}

impl<'a> Stencil<'a> {
    /// Blended previous-phase value at `(x+dx, z+dz)`. Out of range → 0.
    pub fn prev(&self, ch: u32, dx: i32, dz: i32) -> i32 {
        if dx < -2 || dx > 2 || dz < -2 || dz > 2 || ch >= self.channels {
            return 0;
        }
        let hx = (self.lx as i32).wrapping_add(dx).wrapping_add(2);
        let hz = (self.lz as i32).wrapping_add(dz).wrapping_add(2);
        if hx < 0 || hz < 0 {
            return 0;
        }
        let hx = hx as u32;
        let hz = hz as u32;
        if hx >= self.halo_w || hz >= self.halo_w {
            return 0;
        }
        let i = hz
            .wrapping_mul(self.halo_w)
            .wrapping_add(hx)
            .wrapping_mul(self.channels)
            .wrapping_add(ch) as usize;
        match self.halo.get(i) {
            Some(&v) => v,
            None => 0,
        }
    }
}

/// Predicts a clean Q16 value from the current noisy sample at one cell.
pub trait IntScore: Send + Sync {
    fn predict(
        &self,
        st: &Stencil,
        ch: u32,
        x: i32,
        z: i32,
        phase: u32,
        phases: u32,
        current: i32,
    ) -> i32;
}

/// Integer twin of [`crate::HashScore`]: pull toward value-noise whose cell
/// size halves as the phase index rises.
#[derive(Clone, Copy, Debug)]
pub struct IntHashScore {
    pub seed: u32,
}

/// 0.28 in Q16: `(28 * ONE) / 100`.
const ALPHA0: i32 = (28 * ONE) / 100;
/// 0.12 in Q16: `(12 * ONE) / 100`.
const ALPHA_D: i32 = (12 * ONE) / 100;

impl IntScore for IntHashScore {
    fn predict(
        &self,
        _st: &Stencil,
        ch: u32,
        x: i32,
        z: i32,
        phase: u32,
        phases: u32,
        current: i32,
    ) -> i32 {
        let remaining = phases.saturating_sub(1).saturating_sub(phase);
        let shift = if remaining > 8 { 8 } else { remaining };
        let cell = 8i32.wrapping_shl(shift);
        let n = value_noise_q16(
            self.seed ^ ch.wrapping_mul(0x9E37_79B9),
            x,
            z,
            cell,
            0,
        );
        let p = if phases == 0 { 1 } else { phases };
        let frac = (((phase as i64) << 16) / (p as i64)) as i32;
        let alpha = ALPHA0.wrapping_add(mul_q16(ALPHA_D, frac));
        current.wrapping_add(mul_q16(n.wrapping_sub(current), alpha))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TileKey {
    phase: u32,
    tx: i32,
    tz: i32,
}

/// Finished tiles kept around; values are pure, so eviction is only a recompute.
pub const INT_TILE_CACHE_CAP: usize = 4096;

/// Hit / miss / eviction counters for tests and gauges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

const NIL: u32 = u32::MAX;

struct Node {
    key: TileKey,
    val: Arc<[i32]>,
    prev: u32,
    next: u32,
}

struct Lru {
    map: HashMap<TileKey, u32>,
    nodes: Vec<Node>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
    cap: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl Lru {
    fn new(cap: usize) -> Self {
        let cap = if cap == 0 { 1 } else { cap };
        Self {
            map: HashMap::new(),
            nodes: Vec::with_capacity(cap),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            cap,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<[i32]>> {
        let idx = match self.map.get(key) {
            Some(&i) => i,
            None => return None,
        };
        self.touch(idx);
        self.hits = self.hits.wrapping_add(1);
        Some(Arc::clone(&self.nodes[idx as usize].val))
    }

    fn insert(&mut self, key: TileKey, val: Arc<[i32]>) {
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx as usize].val = val;
            self.touch(idx);
            return;
        }
        while self.map.len() >= self.cap {
            self.evict();
        }
        let idx = match self.free.pop() {
            Some(free) => {
                self.nodes[free as usize] = Node {
                    key,
                    val,
                    prev: NIL,
                    next: NIL,
                };
                free
            }
            None => {
                let idx = self.nodes.len() as u32;
                self.nodes.push(Node {
                    key,
                    val,
                    prev: NIL,
                    next: NIL,
                });
                idx
            }
        };
        self.map.insert(key, idx);
        self.link_front(idx);
        self.misses = self.misses.wrapping_add(1);
    }

    fn evict(&mut self) {
        let idx = self.tail;
        if idx == NIL {
            return;
        }
        let key = self.nodes[idx as usize].key;
        self.unlink(idx);
        self.map.remove(&key);
        self.free.push(idx);
        self.evictions = self.evictions.wrapping_add(1);
    }

    fn touch(&mut self, idx: u32) {
        self.unlink(idx);
        self.link_front(idx);
    }

    fn unlink(&mut self, idx: u32) {
        let prev = self.nodes[idx as usize].prev;
        let next = self.nodes[idx as usize].next;
        if prev != NIL {
            self.nodes[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[idx as usize].prev = NIL;
        self.nodes[idx as usize].next = NIL;
    }

    fn link_front(&mut self, idx: u32) {
        self.nodes[idx as usize].prev = NIL;
        self.nodes[idx as usize].next = self.head;
        if self.head != NIL {
            self.nodes[self.head as usize].prev = idx;
        } else {
            self.tail = idx;
        }
        self.head = idx;
    }
}

/// Dense covering of loaded tiles, indexed by `(tx - tx0, tz - tz0)`.
struct TileGrid {
    tx0: i32,
    tz0: i32,
    nx: usize,
    nz: usize,
    tiles: Vec<Option<Arc<[i32]>>>,
}

impl TileGrid {
    #[inline]
    fn get(&self, tx: i32, tz: i32) -> Option<&[i32]> {
        let dx = tx.wrapping_sub(self.tx0);
        let dz = tz.wrapping_sub(self.tz0);
        if dx < 0 || dz < 0 {
            return None;
        }
        let ux = dx as usize;
        let uz = dz as usize;
        if ux >= self.nx || uz >= self.nz {
            return None;
        }
        match self.tiles.get(uz.wrapping_mul(self.nx).wrapping_add(ux)) {
            Some(Some(t)) => Some(t.as_ref()),
            _ => None,
        }
    }
}

/// Stack slots for a point's covering grid. Game defaults fit in 3×3.
const COVER_STACK: usize = 81;

/// Lazy, thread-safe integer field.
pub struct IntField<S: IntScore> {
    spec: IntSpec,
    score: S,
    cache: Mutex<Lru>,
    kernel: Vec<u32>,
}

impl<S: IntScore> IntField<S> {
    pub fn new(spec: IntSpec, score: S) -> Self {
        Self::with_cache_cap(spec, score, INT_TILE_CACHE_CAP)
    }

    pub fn with_cache_cap(spec: IntSpec, score: S, cap: usize) -> Self {
        assert!(spec.valid(), "IntField spec out of range");
        Self {
            spec,
            score,
            cache: Mutex::new(Lru::new(cap)),
            kernel: kernel_table(spec.tile),
        }
    }

    pub fn spec(&self) -> IntSpec {
        self.spec
    }

    pub fn cached_tiles(&self) -> usize {
        self.cache.lock().expect("int field cache").len()
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.lock().expect("int field cache").stats()
    }

    /// One channel at one lattice point.
    pub fn sample(&self, channel: u32, x: i32, z: i32) -> i32 {
        if channel >= self.spec.channels {
            return 0;
        }
        let mut out = [0i32; 16];
        self.sample_all(x, z, &mut out);
        match out.get(channel as usize) {
            Some(&v) => v,
            None => 0,
        }
    }

    /// All channels at one point, written into `out`.
    pub fn sample_all(&self, x: i32, z: i32, out: &mut [i32]) {
        let n = self.spec.channels as usize;
        assert!(out.len() >= n);
        let phase = self.spec.phases.saturating_sub(1);
        let (tx0, tx1, tz0, tz1) = Self::covering_range(self.spec, x, z);
        if let Some((nx, nz)) = covering_dims(tx0, tx1, tz0, tz1) {
            if nx.saturating_mul(nz) <= COVER_STACK {
                let mut cache = self.cache.lock().expect("int field cache");
                let mut slots: [Option<Arc<[i32]>>; COVER_STACK] =
                    std::array::from_fn(|_| None);
                if covering_hit(&mut cache, phase, tx0, tx1, tz0, tz1, nx, &mut slots) {
                    blend_lookup(
                        &self.spec,
                        &self.kernel,
                        |tx, tz| slot_at(&slots, nx, nz, tx0, tz0, tx, tz),
                        x,
                        z,
                        out,
                    );
                    return;
                }
            }
        }
        let loaded = self.load_covering(phase, x, z);
        blend_loaded(&self.spec, &self.kernel, &loaded, x, z, out);
    }

    /// All channels over a rectangle. `out` is `(z * w + x) * channels + c`.
    /// Loads each covering tile once.
    pub fn fill_all(&self, x0: i32, z0: i32, w: u32, h: u32, out: &mut [i32]) {
        let ch = self.spec.channels;
        let expect = w as u64 * h as u64 * ch as u64;
        assert_eq!(out.len() as u64, expect);
        if w == 0 || h == 0 {
            return;
        }
        let phase = self.spec.phases.saturating_sub(1);
        let x1 = x0.wrapping_add(w as i32).wrapping_sub(1);
        let z1 = z0.wrapping_add(h as i32).wrapping_sub(1);
        let n = ch as usize;
        let tile = self.spec.tile as i32;
        let stride = self.spec.stride as i32;
        let tx0 = div_floor(x0.wrapping_sub(tile).wrapping_add(1), stride);
        let tx1 = div_floor(x1, stride);
        let tz0 = div_floor(z0.wrapping_sub(tile).wrapping_add(1), stride);
        let tz1 = div_floor(z1, stride);
        if let Some((nx, nz)) = covering_dims(tx0, tx1, tz0, tz1) {
            if nx.saturating_mul(nz) <= COVER_STACK {
                let mut cache = self.cache.lock().expect("int field cache");
                let mut slots: [Option<Arc<[i32]>>; COVER_STACK] =
                    std::array::from_fn(|_| None);
                if covering_hit(&mut cache, phase, tx0, tx1, tz0, tz1, nx, &mut slots) {
                    let mut dz = 0i32;
                    while dz < h as i32 {
                        let mut dx = 0i32;
                        while dx < w as i32 {
                            let x = x0.wrapping_add(dx);
                            let z = z0.wrapping_add(dz);
                            let base = ((dz as u32)
                                .wrapping_mul(w)
                                .wrapping_add(dx as u32)
                                .wrapping_mul(ch)) as usize;
                            blend_lookup(
                                &self.spec,
                                &self.kernel,
                                |tx, tz| slot_at(&slots, nx, nz, tx0, tz0, tx, tz),
                                x,
                                z,
                                &mut out[base..base + n],
                            );
                            dx = dx.wrapping_add(1);
                        }
                        dz = dz.wrapping_add(1);
                    }
                    return;
                }
            }
        }
        let map = self.load_rect(phase, x0, z0, x1, z1);
        let mut dz = 0i32;
        while dz < h as i32 {
            let mut dx = 0i32;
            while dx < w as i32 {
                let x = x0.wrapping_add(dx);
                let z = z0.wrapping_add(dz);
                let base = ((dz as u32)
                    .wrapping_mul(w)
                    .wrapping_add(dx as u32)
                    .wrapping_mul(ch)) as usize;
                blend_loaded(
                    &self.spec,
                    &self.kernel,
                    &map,
                    x,
                    z,
                    &mut out[base..base + n],
                );
                dx = dx.wrapping_add(1);
            }
            dz = dz.wrapping_add(1);
        }
    }

    /// Final-phase tile array: `tile*tile*channels`, row-major z, then x, then c.
    pub fn tile(&self, tx: i32, tz: i32) -> Arc<[i32]> {
        let phase = self.spec.phases.saturating_sub(1);
        self.raw_tile(phase, tx, tz)
    }

    fn covering_range(spec: IntSpec, x: i32, z: i32) -> (i32, i32, i32, i32) {
        let tile = spec.tile as i32;
        let stride = spec.stride as i32;
        (
            div_floor(x.wrapping_sub(tile).wrapping_add(1), stride),
            div_floor(x, stride),
            div_floor(z.wrapping_sub(tile).wrapping_add(1), stride),
            div_floor(z, stride),
        )
    }

    fn load_covering(&self, phase: u32, x: i32, z: i32) -> TileGrid {
        let (tx0, tx1, tz0, tz1) = Self::covering_range(self.spec, x, z);
        self.load_range(phase, tx0, tx1, tz0, tz1)
    }

    fn load_rect(&self, phase: u32, x0: i32, z0: i32, x1: i32, z1: i32) -> TileGrid {
        let tile = self.spec.tile as i32;
        let stride = self.spec.stride as i32;
        let tx0 = div_floor(x0.wrapping_sub(tile).wrapping_add(1), stride);
        let tx1 = div_floor(x1, stride);
        let tz0 = div_floor(z0.wrapping_sub(tile).wrapping_add(1), stride);
        let tz1 = div_floor(z1, stride);
        self.load_range(phase, tx0, tx1, tz0, tz1)
    }

    fn load_range(&self, phase: u32, tx0: i32, tx1: i32, tz0: i32, tz1: i32) -> TileGrid {
        if tx1 < tx0 || tz1 < tz0 {
            return TileGrid {
                tx0,
                tz0,
                nx: 0,
                nz: 0,
                tiles: Vec::new(),
            };
        }
        let nx = tx1.wrapping_sub(tx0).wrapping_add(1) as usize;
        let nz = tz1.wrapping_sub(tz0).wrapping_add(1) as usize;
        let mut tiles = vec![None; nx.saturating_mul(nz)];
        let mut missing = Vec::new();
        {
            let mut cache = self.cache.lock().expect("int field cache");
            let mut tz = tz0;
            loop {
                let mut tx = tx0;
                loop {
                    let key = TileKey { phase, tx, tz };
                    let i = tz.wrapping_sub(tz0) as usize * nx + tx.wrapping_sub(tx0) as usize;
                    if let Some(hit) = cache.get(&key) {
                        if i < tiles.len() {
                            tiles[i] = Some(hit);
                        }
                    } else {
                        missing.push((i, key));
                    }
                    if tx == tx1 {
                        break;
                    }
                    tx = tx.wrapping_add(1);
                }
                if tz == tz1 {
                    break;
                }
                tz = tz.wrapping_add(1);
            }
        }
        for (i, key) in missing {
            if i < tiles.len() {
                tiles[i] = Some(self.raw_tile(key.phase, key.tx, key.tz));
            }
        }
        TileGrid {
            tx0,
            tz0,
            nx,
            nz,
            tiles,
        }
    }

    fn raw_tile(&self, phase: u32, tx: i32, tz: i32) -> Arc<[i32]> {
        let key = TileKey { phase, tx, tz };
        {
            let mut cache = self.cache.lock().expect("int field cache");
            if let Some(hit) = cache.get(&key) {
                return hit;
            }
        }
        let built = self.build_tile(phase, tx, tz);
        let mut cache = self.cache.lock().expect("int field cache");
        if let Some(hit) = cache.get(&key) {
            return hit;
        }
        cache.insert(key, Arc::clone(&built));
        built
    }

    fn build_tile(&self, phase: u32, tx: i32, tz: i32) -> Arc<[i32]> {
        let spec = self.spec;
        let tile = spec.tile;
        let stride = spec.stride as i32;
        let ox = tx.wrapping_mul(stride);
        let oz = tz.wrapping_mul(stride);
        let halo_w = tile.wrapping_add(4);
        let ch = spec.channels;
        let halo_len = (halo_w as usize)
            .saturating_mul(halo_w as usize)
            .saturating_mul(ch as usize);
        let mut halo = vec![0i32; halo_len];
        let x0 = ox.wrapping_sub(2);
        let z0 = oz.wrapping_sub(2);
        if phase == 0 {
            let mut hz = 0u32;
            while hz < halo_w {
                let mut hx = 0u32;
                while hx < halo_w {
                    let x = x0.wrapping_add(hx as i32);
                    let z = z0.wrapping_add(hz as i32);
                    let mut c = 0u32;
                    while c < ch {
                        let i = index(ch, halo_w, c, hx, hz);
                        if i < halo.len() {
                            halo[i] = prior_q16(spec.seed, c, x, z);
                        }
                        c = c.wrapping_add(1);
                    }
                    hx = hx.wrapping_add(1);
                }
                hz = hz.wrapping_add(1);
            }
        } else {
            let x1 = ox.wrapping_add(tile as i32).wrapping_add(1);
            let z1 = oz.wrapping_add(tile as i32).wrapping_add(1);
            let prev = self.load_rect(phase.wrapping_sub(1), x0, z0, x1, z1);
            let n = ch as usize;
            let mut hz = 0u32;
            while hz < halo_w {
                let mut hx = 0u32;
                while hx < halo_w {
                    let x = x0.wrapping_add(hx as i32);
                    let z = z0.wrapping_add(hz as i32);
                    let i = index(ch, halo_w, 0, hx, hz);
                    if i + n <= halo.len() {
                        blend_loaded(&spec, &self.kernel, &prev, x, z, &mut halo[i..i + n]);
                    }
                    hx = hx.wrapping_add(1);
                }
                hz = hz.wrapping_add(1);
            }
        }
        let n = (ch.wrapping_mul(tile).wrapping_mul(tile)) as usize;
        let mut buf = vec![0i32; n];
        let mut lz = 0u32;
        while lz < tile {
            let mut lx = 0u32;
            while lx < tile {
                let x = ox.wrapping_add(lx as i32);
                let z = oz.wrapping_add(lz as i32);
                let st = Stencil {
                    halo: &halo,
                    halo_w,
                    channels: ch,
                    lx,
                    lz,
                };
                let mut c = 0u32;
                while c < ch {
                    let hi = index(ch, halo_w, c, lx.wrapping_add(2), lz.wrapping_add(2));
                    let current = match halo.get(hi) {
                        Some(&v) => v,
                        None => 0,
                    };
                    let i = index(ch, tile, c, lx, lz);
                    if i < buf.len() {
                        buf[i] = self.score.predict(&st, c, x, z, phase, spec.phases, current);
                    }
                    c = c.wrapping_add(1);
                }
                lx = lx.wrapping_add(1);
            }
            lz = lz.wrapping_add(1);
        }
        Arc::from(buf)
    }
}

fn covering_dims(tx0: i32, tx1: i32, tz0: i32, tz1: i32) -> Option<(usize, usize)> {
    if tx1 < tx0 || tz1 < tz0 {
        return None;
    }
    let nx = tx1.wrapping_sub(tx0).wrapping_add(1);
    let nz = tz1.wrapping_sub(tz0).wrapping_add(1);
    if nx <= 0 || nz <= 0 {
        return None;
    }
    Some((nx as usize, nz as usize))
}

fn covering_hit(
    cache: &mut Lru,
    phase: u32,
    tx0: i32,
    tx1: i32,
    tz0: i32,
    tz1: i32,
    nx: usize,
    slots: &mut [Option<Arc<[i32]>>],
) -> bool {
    let mut tz = tz0;
    loop {
        let mut tx = tx0;
        loop {
            match cache.get(&TileKey { phase, tx, tz }) {
                Some(raw) => {
                    let i = tz.wrapping_sub(tz0) as usize * nx + tx.wrapping_sub(tx0) as usize;
                    if i >= slots.len() {
                        return false;
                    }
                    slots[i] = Some(raw);
                }
                None => return false,
            }
            if tx == tx1 {
                break;
            }
            tx = tx.wrapping_add(1);
        }
        if tz == tz1 {
            break;
        }
        tz = tz.wrapping_add(1);
    }
    true
}

fn slot_at<'a>(
    slots: &'a [Option<Arc<[i32]>>],
    nx: usize,
    nz: usize,
    tx0: i32,
    tz0: i32,
    tx: i32,
    tz: i32,
) -> Option<&'a [i32]> {
    let dx = tx.wrapping_sub(tx0);
    let dz = tz.wrapping_sub(tz0);
    if dx < 0 || dz < 0 {
        return None;
    }
    let ux = dx as usize;
    let uz = dz as usize;
    if ux >= nx || uz >= nz {
        return None;
    }
    let i = uz.wrapping_mul(nx).wrapping_add(ux);
    match slots.get(i) {
        Some(Some(t)) => Some(t.as_ref()),
        _ => None,
    }
}

#[inline]
fn blend_loaded(spec: &IntSpec, kernel: &[u32], tiles: &TileGrid, x: i32, z: i32, out: &mut [i32]) {
    blend_lookup(spec, kernel, |tx, tz| tiles.get(tx, tz), x, z, out);
}

/// Tent-kernel blend. Unnormalised 1-D weights multiply; Q16 weights are
/// `(w << 16) / total` for every contributor but the last, which takes
/// `ONE - sum` so the weights sum to exactly `1 << 16`.
fn blend_lookup<'a>(
    spec: &IntSpec,
    kernel: &[u32],
    lookup: impl Fn(i32, i32) -> Option<&'a [i32]>,
    x: i32,
    z: i32,
    out: &mut [i32],
) {
    let tile = spec.tile as i32;
    let stride = spec.stride as i32;
    let tx0 = div_floor(x.wrapping_sub(tile).wrapping_add(1), stride);
    let tx1 = div_floor(x, stride);
    let tz0 = div_floor(z.wrapping_sub(tile).wrapping_add(1), stride);
    let tz1 = div_floor(z, stride);
    let mut raws: [&[i32]; 16] = [&[]; 16];
    let mut ws = [0u64; 16];
    let mut bases = [0u32; 16];
    let mut n = 0usize;
    let mut total = 0u64;
    let mut extra: Vec<(&[i32], u64, u32)> = Vec::new();
    if tx1 >= tx0 && tz1 >= tz0 {
        let mut tz = tz0;
        loop {
            let mut tx = tx0;
            loop {
                if let Some(raw) = lookup(tx, tz) {
                    let lx = x.wrapping_sub(tx.wrapping_mul(stride));
                    let lz = z.wrapping_sub(tz.wrapping_mul(stride));
                    if lx >= 0 && lz >= 0 && lx < tile && lz < tile {
                        let kx = match kernel.get(lx as usize) {
                            Some(&v) => v,
                            None => 0,
                        };
                        let kz = match kernel.get(lz as usize) {
                            Some(&v) => v,
                            None => 0,
                        };
                        let w = (kx as u64).wrapping_mul(kz as u64);
                        if w != 0 {
                            let base = (lz as u32)
                                .wrapping_mul(spec.tile)
                                .wrapping_add(lx as u32)
                                .wrapping_mul(spec.channels);
                            total = total.wrapping_add(w);
                            if extra.is_empty() && n < 16 {
                                raws[n] = raw;
                                ws[n] = w;
                                bases[n] = base;
                                n += 1;
                            } else {
                                if extra.is_empty() {
                                    extra.reserve(n + 1);
                                    for i in 0..n {
                                        extra.push((raws[i], ws[i], bases[i]));
                                    }
                                }
                                extra.push((raw, w, base));
                            }
                        }
                    }
                }
                if tx == tx1 {
                    break;
                }
                tx = tx.wrapping_add(1);
            }
            if tz == tz1 {
                break;
            }
            tz = tz.wrapping_add(1);
        }
    }
    if extra.is_empty() {
        apply_blend(spec, x, z, &raws[..n], &ws[..n], &bases[..n], total, out);
    } else {
        let mut er = Vec::with_capacity(extra.len());
        let mut ew = Vec::with_capacity(extra.len());
        let mut eb = Vec::with_capacity(extra.len());
        for &(raw, w, base) in &extra {
            er.push(raw);
            ew.push(w);
            eb.push(base);
        }
        apply_blend(spec, x, z, &er, &ew, &eb, total, out);
    }
}

fn apply_blend(
    spec: &IntSpec,
    x: i32,
    z: i32,
    raws: &[&[i32]],
    ws: &[u64],
    bases: &[u32],
    total: u64,
    out: &mut [i32],
) {
    let ch = spec.channels as usize;
    let n = raws.len();
    if n == 0 || total == 0 {
        let mut c = 0usize;
        while c < ch {
            if c < out.len() {
                out[c] = prior_q16(spec.seed, c as u32, x, z);
            }
            c += 1;
        }
        return;
    }
    let mut wts_stack = [0i32; 16];
    let mut wts_extra: Vec<i32> = Vec::new();
    let wts: &mut [i32] = if n <= 16 {
        &mut wts_stack[..n]
    } else {
        wts_extra.resize(n, 0);
        &mut wts_extra
    };
    let mut acc = 0i32;
    let mut i = 0usize;
    while i + 1 < n {
        let w = ((ws[i] << 16) / total) as i32;
        wts[i] = w;
        acc = acc.wrapping_add(w);
        i += 1;
    }
    wts[n - 1] = ONE.wrapping_sub(acc);
    let mut c = 0usize;
    while c < ch {
        let mut sum = 0i32;
        let mut k = 0usize;
        while k < n {
            let idx = bases[k] as usize + c;
            let v = match raws[k].get(idx) {
                Some(&v) => v,
                None => 0,
            };
            sum = sum.wrapping_add(mul_q16(v, wts[k]));
            k += 1;
        }
        if c < out.len() {
            out[c] = sum;
        }
        c += 1;
    }
}

fn kernel_table(tile: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(tile as usize);
    let mut i = 0u32;
    while i < tile {
        v.push(kernel_unorm(i, tile));
        i = i.wrapping_add(1);
    }
    v
}

/// Unnormalised 1-D tent: `tile - |2*local - (tile-1)|`, always ≥ 1 for `tile ≥ 2`.
fn kernel_unorm(local: u32, tile: u32) -> u32 {
    if tile <= 1 {
        return 1;
    }
    let denom = tile.wrapping_sub(1);
    let two_l = local.wrapping_mul(2);
    let dist = if two_l >= denom {
        two_l.wrapping_sub(denom)
    } else {
        denom.wrapping_sub(two_l)
    };
    tile.wrapping_sub(dist)
}

#[inline]
fn index(channels: u32, tile: u32, c: u32, x: u32, z: u32) -> usize {
    z.wrapping_mul(tile)
        .wrapping_add(x)
        .wrapping_mul(channels)
        .wrapping_add(c) as usize
}

fn prior_q16(seed: u32, channel: u32, x: i32, z: i32) -> i32 {
    uniform_q16(hash32(seed, x, z, channel))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(seed: u32) -> IntField<IntHashScore> {
        IntField::new(IntSpec::new(seed), IntHashScore { seed })
    }

    struct Identity;

    impl IntScore for Identity {
        fn predict(
            &self,
            _st: &Stencil,
            _ch: u32,
            _x: i32,
            _z: i32,
            _phase: u32,
            _phases: u32,
            current: i32,
        ) -> i32 {
            current
        }
    }

    struct Rec {
        x: i32,
        z: i32,
        got: Mutex<[[i32; 5]; 5]>,
        cur: Mutex<i32>,
        saw: Mutex<bool>,
    }

    impl IntScore for Rec {
        fn predict(
            &self,
            st: &Stencil,
            ch: u32,
            x: i32,
            z: i32,
            phase: u32,
            phases: u32,
            current: i32,
        ) -> i32 {
            if ch == 0 && x == self.x && z == self.z && phase.wrapping_add(1) == phases {
                let mut g = [[0i32; 5]; 5];
                let mut dz = -2i32;
                while dz <= 2 {
                    let mut dx = -2i32;
                    while dx <= 2 {
                        g[(dz + 2) as usize][(dx + 2) as usize] = st.prev(0, dx, dz);
                        dx = dx.wrapping_add(1);
                    }
                    dz = dz.wrapping_add(1);
                }
                *self.got.lock().expect("rec") = g;
                *self.cur.lock().expect("rec") = current;
                *self.saw.lock().expect("rec") = true;
                assert_eq!(st.prev(0, 0, 0), current);
                assert_eq!(st.prev(0, 3, 0), 0);
                assert_eq!(st.prev(0, 0, -3), 0);
            }
            current
        }
    }

    #[test]
    fn seed_consistency() {
        let a = field(7);
        let b = field(7);
        let c = field(8);
        for z in -3..5 {
            for x in 10..18 {
                assert_eq!(a.sample(0, x, z), b.sample(0, x, z));
                assert_ne!(a.sample(0, x, z), c.sample(0, x, z));
            }
        }
    }

    #[test]
    fn query_order_sample_then_fill() {
        let a = field(11);
        let b = field(11);
        let mut seq = Vec::new();
        for z in 0..40 {
            for x in 0..40 {
                seq.push(a.sample(1, x, z));
            }
        }
        let ch = a.spec().channels;
        let mut buf = vec![0i32; (48 * 40 * ch) as usize];
        a.fill_all(-24, 8, 48, 40, &mut buf);
        let mut i = 0usize;
        for z in 0..40 {
            for x in 0..40 {
                assert_eq!(a.sample(1, x, z), seq[i], "after fill at {x},{z}");
                i += 1;
            }
        }
        b.fill_all(-24, 8, 48, 40, &mut buf);
        i = 0;
        for z in 0..40 {
            for x in 0..40 {
                assert_eq!(b.sample(1, x, z), seq[i], "fill-first at {x},{z}");
                i += 1;
            }
        }
        let mut rev = Vec::new();
        for z in (0..40).rev() {
            for x in (0..40).rev() {
                rev.push(b.sample(1, x, z));
            }
        }
        rev.reverse();
        assert_eq!(seq, rev);
    }

    #[test]
    fn overlapping_queries_agree() {
        let f = field(3);
        let ch = f.spec().channels;
        let mut left = vec![0i32; (16 * 8 * ch) as usize];
        let mut right = vec![0i32; (16 * 8 * ch) as usize];
        f.fill_all(0, 0, 16, 8, &mut left);
        f.fill_all(8, 0, 16, 8, &mut right);
        for z in 0..8u32 {
            for x in 0..8u32 {
                let a = ((z * 16 + (x + 8)) * ch) as usize;
                let b = ((z * 16 + x) * ch) as usize;
                for c in 0..ch as usize {
                    assert_eq!(left[a + c], right[b + c], "at {x},{z} ch {c}");
                }
            }
        }
    }

    fn assert_fill_matches_sample(f: &IntField<IntHashScore>, x0: i32, z0: i32, w: u32, h: u32) {
        let ch = f.spec().channels;
        let mut buf = vec![0i32; (w * h * ch) as usize];
        f.fill_all(x0, z0, w, h, &mut buf);
        let mut point = vec![0i32; ch as usize];
        let mut dz = 0i32;
        while dz < h as i32 {
            let mut dx = 0i32;
            while dx < w as i32 {
                f.sample_all(x0.wrapping_add(dx), z0.wrapping_add(dz), &mut point);
                let base = ((dz as u32).wrapping_mul(w).wrapping_add(dx as u32).wrapping_mul(ch))
                    as usize;
                for c in 0..ch as usize {
                    assert_eq!(
                        buf[base + c],
                        point[c],
                        "at {},{} ch {c}",
                        x0.wrapping_add(dx),
                        z0.wrapping_add(dz)
                    );
                }
                dx = dx.wrapping_add(1);
            }
            dz = dz.wrapping_add(1);
        }
    }

    #[test]
    fn fill_all_matches_sample_all() {
        assert_fill_matches_sample(&field(19), -4, 7, 16, 8);
    }

    #[test]
    fn fill_all_matches_sample_all_straddling_tiles() {
        let f = IntField::new(
            IntSpec {
                seed: 19,
                tile: 32,
                stride: 16,
                phases: 4,
                channels: 4,
            },
            IntHashScore { seed: 19 },
        );
        assert_fill_matches_sample(&f, -24, 8, 48, 40);
    }

    #[test]
    fn stencil_prev_matches_truncated_field() {
        let spec2 = IntSpec {
            seed: 21,
            tile: 32,
            stride: 16,
            phases: 2,
            channels: 2,
        };
        let spec1 = IntSpec {
            phases: 1,
            ..spec2
        };
        let points = [(0i32, 0i32), (16, 16), (31, 0), (32, 32), (-1, 8)];
        for &(px, pz) in &points {
            let rec = Rec {
                x: px,
                z: pz,
                got: Mutex::new([[0; 5]; 5]),
                cur: Mutex::new(0),
                saw: Mutex::new(false),
            };
            let two = IntField::new(spec2, rec);
            let one = IntField::new(spec1, Identity);
            let _ = two.sample(0, px, pz);
            assert!(*two.score.saw.lock().expect("rec"), "predict at {px},{pz}");
            let got = *two.score.got.lock().expect("rec");
            let cur = *two.score.cur.lock().expect("rec");
            assert_eq!(got[2][2], cur);
            let mut dz = -2i32;
            while dz <= 2 {
                let mut dx = -2i32;
                while dx <= 2 {
                    let want = one.sample(0, px.wrapping_add(dx), pz.wrapping_add(dz));
                    assert_eq!(
                        got[(dz + 2) as usize][(dx + 2) as usize],
                        want,
                        "stencil at {px},{pz} d=({dx},{dz})"
                    );
                    dx = dx.wrapping_add(1);
                }
                dz = dz.wrapping_add(1);
            }
        }
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let spec = IntSpec {
            seed: 1,
            tile: 8,
            stride: 8,
            phases: 1,
            channels: 1,
        };
        let f = IntField::with_cache_cap(spec, IntHashScore { seed: 1 }, 2);
        let a0 = f.tile(0, 0);
        let b0 = f.tile(1, 0);
        assert_eq!(f.cached_tiles(), 2);
        let stats_ab = f.cache_stats();
        assert_eq!(stats_ab.misses, 2);
        assert_eq!(stats_ab.evictions, 0);
        let c0 = f.tile(2, 0);
        assert_eq!(f.cached_tiles(), 2);
        let stats_c = f.cache_stats();
        assert_eq!(stats_c.evictions, 1);
        assert_eq!(stats_c.misses, 3);
        let hits_before = f.cache_stats().hits;
        let b1 = f.tile(1, 0);
        assert_eq!(f.cache_stats().hits, hits_before + 1, "tile (1,0) still cached");
        assert_eq!(&b0[..], &b1[..]);
        let misses_before = f.cache_stats().misses;
        let a1 = f.tile(0, 0);
        assert_eq!(f.cache_stats().misses, misses_before + 1, "tile (0,0) was LRU");
        assert_eq!(&a0[..], &a1[..]);
        let g = IntField::new(spec, IntHashScore { seed: 1 });
        assert_eq!(&a0[..], &g.tile(0, 0)[..]);
        assert_eq!(&c0[..], &g.tile(2, 0)[..]);
        assert_eq!(f.sample(0, 0, 0), g.sample(0, 0, 0));
    }

    #[test]
    fn far_coords_do_not_panic() {
        let f = field(1);
        let g = field(1);
        for &(x, z) in &[
            (1_000_000_000i32, -1_000_000_000i32),
            (-1_000_000_000, 1_000_000_000),
            (i32::MAX - 40, 0),
            (i32::MAX - 40, i32::MAX - 40),
            (-1_000_000_000, -40),
        ] {
            let va = f.sample(0, x, z);
            let vb = g.sample(0, x, z);
            assert_eq!(va, vb, "at {x},{z}");
        }
        let mut buf = vec![0i32; 8 * 8 * 4];
        f.fill_all(i32::MAX - 40, 0, 8, 8, &mut buf);
        let mut point = [0i32; 16];
        f.sample_all(i32::MAX - 40, 0, &mut point);
        assert_eq!(&buf[..4], &point[..4]);
        let _ = f.tile(i32::MAX / 16, 0);
    }

    #[test]
    fn tile_layout_matches_sample_unblended_when_stride_equals_tile() {
        let spec = IntSpec {
            seed: 4,
            tile: 8,
            stride: 8,
            phases: 1,
            channels: 2,
        };
        let f = IntField::new(spec, IntHashScore { seed: 4 });
        let t = f.tile(1, 2);
        assert_eq!(t.len(), 8 * 8 * 2);
        let ox = 8i32;
        let oz = 16i32;
        let mut point = [0i32; 16];
        for lz in 0u32..8 {
            for lx in 0u32..8 {
                f.sample_all(ox + lx as i32, oz + lz as i32, &mut point);
                let i = (lz * 8 + lx) * 2;
                assert_eq!(t[i as usize], point[0]);
                assert_eq!(t[i as usize + 1], point[1]);
            }
        }
    }

    #[test]
    fn blend_weights_sum_to_one() {
        // With overlap, a blended sample is a convex combination; reconstruct
        // by checking fill == sample (already) and that values stay in the
        // range of the prior (identity score, 1 phase): prior is 0..=65535.
        let spec = IntSpec {
            seed: 5,
            tile: 32,
            stride: 16,
            phases: 1,
            channels: 1,
        };
        let f = IntField::new(spec, Identity);
        for z in -8..40 {
            for x in -8..40 {
                let v = f.sample(0, x, z);
                assert!(v >= 0 && v <= 65535, "blend {v} at {x},{z}");
            }
        }
    }

    #[test]
    fn kernel_table_positive() {
        for tile in [4u32, 8, 16, 32, 64] {
            let t = kernel_table(tile);
            assert_eq!(t.len(), tile as usize);
            for (i, &v) in t.iter().enumerate() {
                assert!(v >= 1, "tile={tile} local={i} w={v}");
                assert_eq!(v, kernel_unorm(i as u32, tile));
            }
        }
    }

    #[test]
    #[ignore]
    fn gauge_int_hash_tiles() {
        let spec = IntSpec {
            seed: 1,
            tile: 32,
            stride: 16,
            phases: 6,
            channels: 8,
        };
        let f = IntField::new(spec, IntHashScore { seed: 1 });
        let n = 16i32;
        let t0 = std::time::Instant::now();
        let mut k = 0u32;
        let mut tz = 0i32;
        while tz < n {
            let mut tx = 0i32;
            while tx < n {
                let _ = f.tile(tx, tz);
                k = k.wrapping_add(1);
                tx = tx.wrapping_add(1);
            }
            tz = tz.wrapping_add(1);
        }
        let ms = t0.elapsed().as_millis().max(1);
        let tps = (k as u128 * 1000) / ms as u128;
        eprintln!(
            "intfield gauge: 32² tile, 8 ch, 6 phases, IntHashScore: {k} tiles in {ms} ms ({tps} tiles/s)"
        );
    }
}
