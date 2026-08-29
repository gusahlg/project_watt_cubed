//! placement.rs — element-first world generation: where each element wants to
//! exist, as data.
//!
//! Terrain stops picking named blocks. Every element carries placement rules
//! over the shared context fields (depth below surface, surface kind, flood,
//! overhang, island shell, cave walls), and the block at a cell IS the union of
//! the elements placed there: several → their natural mixture, one → a pure
//! block, none → air.
//!
//! The vocabulary is deliberately CLOSED — no closures, no arbitrary
//! predicates. That is what makes the reachable combination set mechanically
//! enumerable at startup ([`PlacementTable::compile`]): every block terrain can
//! ever emit is registered before any worker thread exists, in canonical order,
//! so ids are a pure function of (builtins, table) and workers/multiplayer
//! clients reproduce identical chunk bytes. The generator receives only
//! pre-resolved ids ([`Resolved`]) — it cannot register, by construction.
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

use crate::block::composition::Composition;
use crate::block::element::{El, ElementId};
use crate::block::registry::{BlockId, BlockRegistry};

/// Loud tripwire for a table whose reachable set explodes (a mod authoring
/// pathological rules fails at startup, not at chunk 40,000). The builtin
/// table lands well under 100 combos; the cap leaves room for mods while
/// staying far inside `MAX_BLOCK_TYPES`.
pub const ENUM_CAP: usize = 1024;

/// Bumped whenever the same (seed, coord) can yield different chunk MATERIALS
/// than before. Saves stamp it (loader warns on mismatch — edits replay over
/// terrain whose materials moved) and the join handshake folds it into the
/// protocol version (mixed peers get an error instead of silent divergence).
/// v1: the legacy hand-written picker. v2: element-first placement (this
/// module). v3: the alien pass — trees/Wood/Leaves retired, elements recolored,
/// per-biome crust, luminous surface scatter, bolder landforms.
pub const WORLDGEN_VERSION: u16 = 3;

/// Scattered stream B rarities are stream A's scaled down by this — pairs stay
/// genuine finds (P(pair) ~ p²/8 per stone cell), singles move by ~+12%.
pub const STREAM_B_SCALE: u32 = 8;

/// The ground column's surface dressing, classified once per column from the
/// shared context (water table, temperature, humidity, altitude — see
/// `Terrain::surface_kind`). One axis, so the case space stays finite.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceKind {
    /// Temperate ground — organic growth over soil.
    Grassy = 0,
    /// At or under the water table: beaches, riverbeds, lakebeds.
    Shore = 1,
    /// Cold or high ground.
    Snowy = 2,
    /// Hot and dry ground.
    Desert = 3,
    /// The hash-dithered rim one block above the water table — the sand/soil
    /// transition band between beach and grass.
    BeachEdge = 4,
}

impl SurfaceKind {
    pub const COUNT: usize = 5;
    pub const ALL: [SurfaceKind; Self::COUNT] =
        [Self::Grassy, Self::Shore, Self::Snowy, Self::Desert, Self::BeachEdge];
}

/// Which surface kinds a ground rule places under — a closed bitmask, so rules
/// stay serializable and enumerable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SurfaceMask(pub u8);

impl SurfaceMask {
    pub const GRASSY: Self = Self(1 << SurfaceKind::Grassy as u8);
    pub const SHORE: Self = Self(1 << SurfaceKind::Shore as u8);
    pub const SNOWY: Self = Self(1 << SurfaceKind::Snowy as u8);
    pub const DESERT: Self = Self(1 << SurfaceKind::Desert as u8);
    pub const BEACH_EDGE: Self = Self(1 << SurfaceKind::BeachEdge as u8);
    pub const ANY: Self = Self(0x1F);

    pub const fn or(self, o: Self) -> Self {
        Self(self.0 | o.0)
    }
    pub const fn contains(self, k: SurfaceKind) -> bool {
        self.0 & (1 << k as u8) != 0
    }
}

/// Island surface eligibility (islands frost over above `ICE_SURFACE_Y`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IslandSurface {
    Any,
    OnlyIcy,
    NotIcy,
}

/// Where in the shared world a rule applies. Each variant reads only context
/// the generator already computes — no rule ever grows its own noise field.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Context {
    /// Below the column surface, uncarved. `depth` is surface-relative
    /// (`height - wy`), 1 = the top ground cell.
    Ground { depth: RangeInclusive<i32>, surface: SurfaceMask },
    /// Between the surface and the water table (oceans, rivers, lakes).
    Flood,
    /// Overhang-shelf cells above the surface.
    Overhang,
    /// Island-solid cells. `below` counts solid cells to the island's upper
    /// surface (0 = the surface cell itself).
    Island { below: RangeInclusive<i32>, surface: IslandSurface },
    /// Uncarved stone bordering a carved cell (cave/ravine wall), by depth.
    /// Carve-gated, so the deep Uniform(stone) proof only needs the carve
    /// dormancy it already checks (inflated by one cell for the adjacency).
    CaveWall { depth: RangeInclusive<i32> },
}

/// How an element occupies its context.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Present wherever the context holds. The union of banded elements is the
    /// base material — layering emerges from band overlap.
    Banded,
    /// Present where the context holds AND a per-cell hash roll lands in a
    /// `u32::MAX / rarity` slice, into cells whose banded union is exactly
    /// `{host}` — which structurally bounds combinations to {host} ∪ payloads.
    /// Compilation derives BOTH hash streams from one row (B at rarity ×
    /// [`STREAM_B_SCALE`]), so overlap arity is capped at two by construction.
    Scattered { rarity: u32, host: ElementId },
}

/// Where one element wants to exist. An element may own several rules.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PlacementRule {
    pub element: ElementId,
    pub context: Context,
    pub kind: Kind,
}

/// One resolved scattered slice: eligible from `min_depth` down (`break` gate
/// in the cumulative walk), hit when the roll lands inside `width`.
#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub min_depth: i32,
    pub width: u32,
    pub id: BlockId,
}

/// The whole placement table. [`builtin`](Self::builtin) is today's terrain as
/// data; mods will append rules here before compilation, later.
pub struct PlacementTable {
    pub rules: Vec<PlacementRule>,
}

/// Everything the generator reads, fully pre-resolved — its only view of the
/// table. Holds ids, never the registry.
#[derive(Clone)]
pub struct Resolved {
    /// Ground surface (depth 1) by [`SurfaceKind`].
    pub dress: [BlockId; SurfaceKind::COUNT],
    /// Ground depths 2..=3, by [`SurfaceKind`] — the crust composition follows
    /// the biome (frozen ground under snow, sandy clay under deserts).
    pub crust: [BlockId; SurfaceKind::COUNT],
    /// Surface (depth 1) scatter by [`SurfaceKind`]: cumulative slices whose id
    /// is the surface union PLUS the payload (glowing tufts on the plains,
    /// phosphor sparks in the desert). One stream, one payload — surface finds
    /// stay singles by construction.
    pub surface_scatter: Vec<Vec<Slice>>,
    /// Ground depth ≥ 4, overhang shelves, island interiors.
    pub stone: BlockId,
    /// Flooded cells.
    pub water: BlockId,
    /// Ground ore slices, stream A — byte-identical to the legacy seams walk.
    pub seams: Vec<Slice>,
    /// Ground ore slices, stream B (rarity ÷ [`STREAM_B_SCALE`], same order).
    pub seams_b: Vec<Slice>,
    /// Pair block for ground seams i and j (i > j): `{host, eᵢ, eⱼ}`.
    /// Lower-triangular; indexed `pairs[i][j]`.
    pub pairs: Vec<Vec<BlockId>>,
    /// Island surface, not icy / icy.
    pub island_grass: BlockId,
    pub island_ice: BlockId,
    /// Island cells 1..=3 below the surface.
    pub island_crust: BlockId,
    /// Island interior scatter (aerium, quartz), streams A and B, plus the
    /// single reachable island pair.
    pub island_seams: Vec<Slice>,
    pub island_seams_b: Vec<Slice>,
    pub island_pairs: Vec<Vec<BlockId>>,
    /// Depth below which no seam places — re-derives the deep Uniform(stone)
    /// proof bound from the table instead of a hardcoded constant.
    pub max_scattered_depth: i32,
    /// The cave-wall rule (Lumin clustering on deep cavern walls), if any.
    pub cave_wall: Option<Slice>,
}

/// Today's terrain, expressed as the builtin table. Ordering of scattered
/// ground rows IS the legacy `seams` array order — the cumulative slice walk
/// depends on it, and [`compile`](PlacementTable::compile) asserts the
/// min-depth monotonicity that walk assumes.
pub fn builtin() -> PlacementTable {
    let ground = |depth: RangeInclusive<i32>, surface: SurfaceMask| Context::Ground { depth, surface };
    let island = |below: RangeInclusive<i32>, surface: IslandSurface| Context::Island { below, surface };
    let rule = |element: El, context: Context, kind: Kind| PlacementRule {
        element: element.id(),
        context,
        kind,
    };
    let banded = |element: El, context: Context| rule(element, context, Kind::Banded);
    let seam = |element: El, depth: RangeInclusive<i32>, rarity: u32| {
        rule(element, ground(depth, SurfaceMask::ANY), Kind::Scattered { rarity, host: El::Stone.id() })
    };
    let island_seam = |element: El, rarity: u32| {
        rule(
            element,
            island(4..=i32::MAX, IslandSurface::Any),
            Kind::Scattered { rarity, host: El::Stone.id() },
        )
    };

    PlacementTable {
        rules: vec![
            // --- Ground crust: the surface dress and the soil band under it.
            banded(El::Organic, ground(1..=1, SurfaceMask::GRASSY)),
            banded(El::Soil, ground(1..=1, SurfaceMask::GRASSY)),
            banded(El::Snow, ground(1..=1, SurfaceMask::SNOWY)),
            banded(El::Sand, ground(1..=1, SurfaceMask::SHORE.or(SurfaceMask::DESERT))),
            // The dithered rim one block above the water line: sand still
            // holding soil — the beach fading into the grass.
            banded(El::Sand, ground(1..=1, SurfaceMask::BEACH_EDGE)),
            banded(El::Soil, ground(1..=1, SurfaceMask::BEACH_EDGE)),
            // Crust follows the biome: temperate/shore/beach ground binds soil
            // with clay; deserts run sandy clay; snowy ground freezes through.
            banded(El::Soil, ground(2..=3, SurfaceMask::GRASSY.or(SurfaceMask::SHORE).or(SurfaceMask::BEACH_EDGE))),
            banded(El::Clay, ground(2..=3, SurfaceMask::GRASSY.or(SurfaceMask::SHORE).or(SurfaceMask::BEACH_EDGE).or(SurfaceMask::DESERT))),
            banded(El::Sand, ground(2..=3, SurfaceMask::DESERT)),
            banded(El::Soil, ground(2..=3, SurfaceMask::SNOWY)),
            banded(El::Ice, ground(2..=3, SurfaceMask::SNOWY)),
            // --- The world's rock.
            banded(El::Stone, ground(4..=i32::MAX, SurfaceMask::ANY)),
            banded(El::Stone, Context::Overhang),
            // --- Water fills to the table.
            banded(El::Water, Context::Flood),
            // --- Ground ores: the legacy seams array, verbatim (order = slice order).
            seam(El::Coal, 3..=64, 90),
            seam(El::Iron, 8..=64, 110),
            seam(El::Copper, 8..=64, 130),
            seam(El::Sulfur, 20..=64, 240),
            seam(El::Quartz, 20..=64, 200),
            seam(El::Lead, 20..=64, 220),
            seam(El::Gold, 32..=64, 300),
            seam(El::Lumin, 32..=64, 380),
            seam(El::Titan, 48..=64, 460),
            seam(El::Obsidian, 48..=64, 240),
            // --- Flying islands: grass-or-ice skin, soil crust, stone heart.
            banded(El::Organic, island(0..=0, IslandSurface::NotIcy)),
            banded(El::Soil, island(0..=0, IslandSurface::NotIcy)),
            banded(El::Ice, island(0..=0, IslandSurface::OnlyIcy)),
            banded(El::Soil, island(1..=3, IslandSurface::Any)),
            banded(El::Clay, island(1..=3, IslandSurface::Any)),
            banded(El::Stone, island(4..=i32::MAX, IslandSurface::Any)),
            island_seam(El::Aerium, 45),
            island_seam(El::Quartz, 160),
            // --- Surface glow: scattered luminous growth. Lumin tufts across
            // the teal plains; phosphor sparks in the ash deserts — the night
            // face of the planet. Depth-1 scattered rules join the SURFACE
            // union (host names the union's anchor, asserted at compile).
            rule(
                El::Lumin,
                ground(1..=1, SurfaceMask::GRASSY),
                Kind::Scattered { rarity: 700, host: El::Soil.id() },
            ),
            rule(
                El::Phosphor,
                ground(1..=1, SurfaceMask::DESERT),
                Kind::Scattered { rarity: 900, host: El::Sand.id() },
            ),
            // --- Lumin clusters on deep cavern walls: the deep caves glow.
            rule(
                El::Lumin,
                Context::CaveWall { depth: 32..=i32::MAX },
                Kind::Scattered { rarity: 40, host: El::Stone.id() },
            ),
        ],
    }
}

impl PlacementTable {
    /// Enumerate every composition terrain can emit, register the set in
    /// canonical order (sorted element lists — table row order can never
    /// reshuffle ids), and resolve the generator's LUTs. Runs at startup on
    /// the main thread, before any worker thread exists. Idempotent: identical
    /// compositions dedup inside the registry.
    pub fn compile(&self, registry: &mut BlockRegistry) -> Resolved {
        // --- 1. The reachable set, canonically ordered by BTreeSet.
        let mut reachable: BTreeSet<Vec<u16>> = BTreeSet::new();
        let mut add = |els: &[u16]| {
            let mut v = els.to_vec();
            v.sort_unstable();
            v.dedup();
            reachable.insert(v);
        };

        // Banded ground unions: every depth band × every surface kind, since
        // the crust now varies by biome (frozen ground, sandy desert).
        for band in self.ground_depth_bands() {
            for kind in SurfaceKind::ALL {
                add(&self.ground_banded_union(band, kind));
            }
        }
        add(&self.banded_union(|c| matches!(c, Context::Flood)));
        add(&self.banded_union(|c| matches!(c, Context::Overhang)));
        for below in [0, 1, 4] {
            for surf in [IslandSurface::OnlyIcy, IslandSurface::NotIcy] {
                add(&self.island_banded_union(below, surf));
            }
        }

        // Sub-surface ore seams (depth ≥ 2, below the crust) and island interior
        // scatter: singles and same-region pairs (arity ≤ 2 by the two-stream
        // construction — one payload per stream, deduped). Depth-1 SURFACE
        // scatter is handled separately below (its payload joins the full
        // surface union, not just a host).
        let ground_seams = self.ore_seams();
        let island_seams = self.scattered(|c| matches!(c, Context::Island { .. }));
        for seams in [&ground_seams, &island_seams] {
            for (i, a) in seams.iter().enumerate() {
                let (host, ea, ra) = a.scatter_parts();
                add(&[host, ea]);
                for b in &seams[i + 1..] {
                    let (hb, eb, rb) = b.scatter_parts();
                    debug_assert_eq!(host, hb, "scattered rules in one region share a host");
                    if ranges_overlap(&ra, &rb) {
                        add(&[host, ea, eb]);
                    }
                }
            }
        }
        // Surface scatter: the payload joins the FULL depth-1 banded union of
        // its kind (a glowing tuft is Organic+Soil+Lumin, not Soil+Lumin).
        for kind in SurfaceKind::ALL {
            let base = self.ground_banded_union(1, kind);
            for r in self.surface_scatter_rules(kind) {
                let mut union = base.clone();
                union.push(r.element.0);
                add(&union);
            }
        }
        for r in self.scattered(|c| matches!(c, Context::CaveWall { .. })) {
            let (host, e, _) = r.scatter_parts();
            add(&[host, e]);
        }

        assert!(
            reachable.len() <= ENUM_CAP,
            "placement table enumerates {} reachable combinations (cap {ENUM_CAP}) — \
             a rule set this permissive is a bug, not a feature",
            reachable.len()
        );

        // --- 2. Canonical registration. BTreeSet iteration IS the canonical
        // order; identical compositions dedup to existing (builtin) ids.
        // Reactive terrain is a feature (a Sulfur+Coal seam SHOULD be exciting
        // to mine) — but never a mystery: audit every reactive combo here so a
        // surprising world behaviour is a read of startup output.
        for els in &reachable {
            let ids: Vec<ElementId> = els.iter().map(|&e| ElementId(e)).collect();
            let id = registry
                .natural(&ids)
                .expect("block palette cannot hold the worldgen enumeration");
            let block = registry.block(id);
            if !block.reactions.is_empty() {
                eprintln!(
                    "placement: terrain can emit reactive {} ({} reaction{})",
                    block.name,
                    block.reactions.len(),
                    if block.reactions.len() == 1 { "" } else { "s" }
                );
            }
        }

        // --- 3. Resolve the LUTs (pure lookups — everything registered above).
        let id = |els: &[ElementId]| -> BlockId {
            registry
                .lookup(&Composition::natural(els))
                .expect("enumerated composition must be registered")
        };
        let union_id = |els: Vec<u16>| -> BlockId {
            let ids: Vec<ElementId> = els.iter().map(|&e| ElementId(e)).collect();
            id(&ids)
        };

        let dress = SurfaceKind::ALL.map(|k| union_id(self.ground_banded_union(1, k)));
        let crust = SurfaceKind::ALL.map(|k| union_id(self.ground_banded_union(2, k)));
        let stone = union_id(self.ground_banded_union(4, SurfaceKind::Grassy));

        // Surface scatter per kind: cumulative slices, payload joined onto the
        // full depth-1 union. One stream (surface finds stay singles).
        let surface_scatter: Vec<Vec<Slice>> = SurfaceKind::ALL
            .iter()
            .map(|&kind| {
                let base = self.ground_banded_union(1, kind);
                self.surface_scatter_rules(kind)
                    .iter()
                    .map(|r| {
                        let mut union = base.clone();
                        union.push(r.element.0);
                        Slice { min_depth: 1, width: u32::MAX / r.rarity(), id: union_id(union) }
                    })
                    .collect()
            })
            .collect();

        let slices = |seams: &[&PlacementRule], scale: u32| -> Vec<Slice> {
            let mut min_depth = i32::MIN;
            seams
                .iter()
                .map(|r| {
                    let (host, e, range) = r.scatter_parts();
                    assert!(
                        *range.start() >= min_depth,
                        "scattered rules must be authored shallow-to-deep: the cumulative \
                         slice walk breaks at the first ineligible depth"
                    );
                    min_depth = *range.start();
                    Slice {
                        min_depth: *range.start(),
                        width: u32::MAX / (r.rarity() * scale),
                        id: id(&[ElementId(host), ElementId(e)]),
                    }
                })
                .collect()
        };
        let pair_matrix = |seams: &[&PlacementRule]| -> Vec<Vec<BlockId>> {
            (0..seams.len())
                .map(|i| {
                    let (host, ei, ri) = seams[i].scatter_parts();
                    (0..i)
                        .map(|j| {
                            let (_, ej, rj) = seams[j].scatter_parts();
                            if ranges_overlap(&ri, &rj) {
                                union_id(vec![host, ei, ej])
                            } else {
                                // Unreachable overlap: fall back to the single —
                                // the roll can't produce it anyway.
                                id(&[ElementId(host), ElementId(ei)])
                            }
                        })
                        .collect()
                })
                .collect()
        };

        let max_scattered_depth = ground_seams
            .iter()
            .map(|r| r.scatter_parts().2.end().min(&i32::MAX).to_owned())
            .max()
            .unwrap_or(0);

        let cave_wall = self
            .scattered(|c| matches!(c, Context::CaveWall { .. }))
            .first()
            .map(|r| {
                let (host, e, range) = r.scatter_parts();
                Slice {
                    min_depth: *range.start(),
                    width: u32::MAX / r.rarity(),
                    id: id(&[ElementId(host), ElementId(e)]),
                }
            });

        Resolved {
            dress,
            crust,
            surface_scatter,
            stone,
            water: union_id(self.banded_union(|c| matches!(c, Context::Flood))),
            seams: slices(&ground_seams, 1),
            seams_b: slices(&ground_seams, STREAM_B_SCALE),
            pairs: pair_matrix(&ground_seams),
            island_grass: union_id(self.island_banded_union(0, IslandSurface::NotIcy)),
            island_ice: union_id(self.island_banded_union(0, IslandSurface::OnlyIcy)),
            island_crust: union_id(self.island_banded_union(1, IslandSurface::Any)),
            island_seams: slices(&island_seams, 1),
            island_seams_b: slices(&island_seams, STREAM_B_SCALE),
            island_pairs: pair_matrix(&island_seams),
            max_scattered_depth,
            cave_wall,
        }
    }

    /// Banded ground union at one depth for one surface kind (element ids,
    /// sorted, deduped).
    fn ground_banded_union(&self, depth: i32, kind: SurfaceKind) -> Vec<u16> {
        let mut v: Vec<u16> = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Banded))
            .filter(|r| match &r.context {
                Context::Ground { depth: d, surface } => {
                    d.contains(&depth) && surface.contains(kind)
                }
                _ => false,
            })
            .map(|r| r.element.0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Banded island union at one below-surface distance for one surface state.
    fn island_banded_union(&self, below: i32, surf: IslandSurface) -> Vec<u16> {
        let mut v: Vec<u16> = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Banded))
            .filter(|r| match &r.context {
                Context::Island { below: b, surface } => {
                    b.contains(&below)
                        && match (surface, surf) {
                            (IslandSurface::Any, _) | (_, IslandSurface::Any) => true,
                            (a, b) => *a == b,
                        }
                }
                _ => false,
            })
            .map(|r| r.element.0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Banded union over an arbitrary context predicate (Flood/Overhang).
    fn banded_union(&self, pred: impl Fn(&Context) -> bool) -> Vec<u16> {
        let mut v: Vec<u16> = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Banded) && pred(&r.context))
            .map(|r| r.element.0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Distinct ground depth-band representatives from banded rule endpoints,
    /// so enumeration visits every distinct union at every kind.
    fn ground_depth_bands(&self) -> Vec<i32> {
        let mut starts: Vec<i32> = self
            .rules
            .iter()
            .filter_map(|r| match (&r.kind, &r.context) {
                (Kind::Banded, Context::Ground { depth, .. }) => Some(*depth.start()),
                _ => None,
            })
            .collect();
        starts.sort_unstable();
        starts.dedup();
        starts
    }

    /// Scattered rules matching a context predicate, in authored order.
    fn scattered(&self, pred: impl Fn(&Context) -> bool) -> Vec<&PlacementRule> {
        self.rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Scattered { .. }) && pred(&r.context))
            .collect()
    }

    /// Sub-surface ore seams: scattered ground rules below the crust (depth
    /// start ≥ 2), in authored order — the cumulative slice walk's rows.
    fn ore_seams(&self) -> Vec<&PlacementRule> {
        self.scattered(|c| matches!(c, Context::Ground { depth, .. } if *depth.start() >= 2))
    }

    /// Depth-1 surface scatter rules for one kind, in authored order — the
    /// luminous surface growth whose payload joins the full surface union.
    fn surface_scatter_rules(&self, kind: SurfaceKind) -> Vec<&PlacementRule> {
        self.scattered(|c| {
            matches!(c, Context::Ground { depth, surface }
                if *depth.start() == 1 && *depth.end() == 1 && surface.contains(kind))
        })
    }
}

impl PlacementRule {
    /// (host element, payload element, eligible depth range) of a scattered rule.
    fn scatter_parts(&self) -> (u16, u16, RangeInclusive<i32>) {
        let Kind::Scattered { host, .. } = &self.kind else {
            panic!("scatter_parts on a banded rule")
        };
        let range = match &self.context {
            Context::Ground { depth, .. } | Context::CaveWall { depth } => depth.clone(),
            Context::Island { below, .. } => below.clone(),
            Context::Flood | Context::Overhang => 0..=0,
        };
        (host.0, self.element.0, range)
    }

    fn rarity(&self) -> u32 {
        match self.kind {
            Kind::Scattered { rarity, .. } => rarity,
            Kind::Banded => unreachable!("rarity of a banded rule"),
        }
    }
}

fn ranges_overlap(a: &RangeInclusive<i32>, b: &RangeInclusive<i32>) -> bool {
    a.start() <= b.end() && b.start() <= a.end()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiled() -> (BlockRegistry, Resolved) {
        let mut reg = BlockRegistry::with_builtins();
        let resolved = builtin().compile(&mut reg);
        (reg, resolved)
    }

    #[test]
    fn enumeration_matches_the_doc_arithmetic() {
        let baseline = BlockRegistry::with_builtins().block_count();
        let (reg, _) = compiled();
        let added = reg.block_count() - baseline;
        // 45 ground ore pairs C(10,2) + 10 single seams + 1 island pair +
        // Stone+Aerium + five biome-crust unions + two surface-scatter unions.
        // The island's Stone+Quartz and cave-wall Stone+Lumin dedup into ground
        // seams; pure banded elements were registered with the element palette.
        assert_eq!(added, 45 + 10 + 1 + 1 + 5 + 2, "reachable set drifted — re-derive before accepting");
        assert!(reg.block_count() <= ENUM_CAP);
    }

    #[test]
    fn compile_is_idempotent_and_canonical() {
        let (mut reg, first) = compiled();
        let count = reg.block_count();
        let again = builtin().compile(&mut reg);
        assert_eq!(reg.block_count(), count, "recompile registers nothing new");
        assert_eq!(first.stone, again.stone);
        assert_eq!(first.dress, again.dress);
        assert_eq!(first.pairs, again.pairs);
    }

    #[test]
    fn banded_row_order_cannot_reshuffle_ids() {
        let (_, a) = compiled();
        // Reverse ONLY the banded rows (scattered order is the slice-walk
        // semantics and is asserted monotonic); id assignment must not move.
        let mut table = builtin();
        let (mut banded, scattered): (Vec<_>, Vec<_>) =
            table.rules.into_iter().partition(|r| matches!(r.kind, Kind::Banded));
        banded.reverse();
        banded.extend(scattered);
        table.rules = banded;
        let mut reg = BlockRegistry::with_builtins();
        let b = table.compile(&mut reg);
        assert_eq!(a.dress, b.dress);
        assert_eq!(a.crust, b.crust);
        assert_eq!(a.pairs, b.pairs);
        assert_eq!(a.island_pairs, b.island_pairs);
    }

    #[test]
    fn seams_match_the_legacy_table() {
        let (reg, r) = compiled();
        let legacy: [(&str, i32, u32); 10] = [
            ("Stone+Coal", 3, 90),
            ("Stone+Iron", 8, 110),
            ("Stone+Copper", 8, 130),
            ("Stone+Sulfur", 20, 240),
            ("Stone+Quartz", 20, 200),
            ("Stone+Lead", 20, 220),
            ("Stone+Gold", 32, 300),
            ("Stone+Lumin", 32, 380),
            ("Stone+Titan", 48, 460),
            // Obsidian semantics changed deliberately: {Stone, Obsidian}, no
            // longer the pure block (the one seam that was the odd one out).
            ("Obsidian", 48, 240),
        ];
        assert_eq!(r.seams.len(), legacy.len());
        for (slice, (name, min_depth, rarity)) in r.seams.iter().zip(legacy) {
            assert_eq!(slice.min_depth, min_depth, "{name}");
            assert_eq!(slice.width, u32::MAX / rarity, "{name}");
            if name != "Obsidian" {
                assert_eq!(slice.id, reg.id_by_name(name).unwrap(), "{name} dedups into the builtin");
            }
        }
        // Stream B: same order, an eighth the width.
        for (a, b) in r.seams.iter().zip(&r.seams_b) {
            assert_eq!(a.id, b.id);
            assert_eq!(b.width, u32::MAX / ((u32::MAX / a.width) * STREAM_B_SCALE));
        }
        assert_eq!(r.max_scattered_depth, 64, "the deep-proof bound, now derived");
    }

    #[test]
    fn crust_and_dress_are_the_expected_unions() {
        let (reg, r) = compiled();
        let comp = |id: BlockId| &reg.block(id).composition;
        let natural = |els: &[El]| {
            let mut ids = els.iter().map(|e| e.id().0).collect::<Vec<_>>();
            ids.sort_unstable(); // registration order is canonical (sorted ids)
            Composition::natural(&ids.into_iter().map(ElementId).collect::<Vec<_>>())
        };
        assert_eq!(*comp(r.dress[SurfaceKind::Grassy as usize]), natural(&[El::Organic, El::Soil]));
        assert_eq!(*comp(r.dress[SurfaceKind::Shore as usize]), natural(&[El::Sand]));
        assert_eq!(*comp(r.dress[SurfaceKind::Snowy as usize]), natural(&[El::Snow]));
        assert_eq!(*comp(r.dress[SurfaceKind::Desert as usize]), natural(&[El::Sand]));
        assert_eq!(
            *comp(r.dress[SurfaceKind::BeachEdge as usize]),
            natural(&[El::Soil, El::Sand])
        );
        // Crust follows the biome now.
        let cr = |k: SurfaceKind| comp(r.crust[k as usize]);
        assert_eq!(*cr(SurfaceKind::Grassy), natural(&[El::Soil, El::Clay]));
        assert_eq!(*cr(SurfaceKind::Desert), natural(&[El::Sand, El::Clay]));
        assert_eq!(*cr(SurfaceKind::Snowy), natural(&[El::Soil, El::Ice]));
        assert_eq!(r.stone, reg.id_by_name("Stone").unwrap());
        assert_eq!(r.water, reg.id_by_name("Water").unwrap());
        assert_eq!(r.island_ice, reg.id_by_name("Ice").unwrap());
        assert_eq!(r.island_grass, r.dress[SurfaceKind::Grassy as usize]);
        assert_eq!(r.island_crust, r.crust[SurfaceKind::Grassy as usize]);
        // Cave-wall Lumin dedups into the ordinary Stone+Lumin composition.
        assert_eq!(r.cave_wall.unwrap().id, reg.id_by_name("Stone+Lumin").unwrap());
        // Surface scatter: grassy plains grow Organic+Soil+Lumin tufts; the
        // ash desert grows Sand+Phosphor sparks; other kinds have none.
        assert_eq!(
            *comp(r.surface_scatter[SurfaceKind::Grassy as usize][0].id),
            natural(&[El::Organic, El::Soil, El::Lumin])
        );
        assert_eq!(
            *comp(r.surface_scatter[SurfaceKind::Desert as usize][0].id),
            natural(&[El::Sand, El::Phosphor])
        );
        assert!(r.surface_scatter[SurfaceKind::Shore as usize].is_empty());
    }

    #[test]
    fn every_pair_is_a_three_element_natural_hosting_stone() {
        let (reg, r) = compiled();
        for (i, row) in r.pairs.iter().enumerate() {
            assert_eq!(row.len(), i);
            for &pid in row {
                match &reg.block(pid).composition {
                    Composition::Natural(els) => {
                        assert_eq!(els.len(), 3, "pair blocks are host + two payloads");
                        assert!(els.contains(&El::Stone.id()));
                    }
                    other => panic!("pair registered as {other:?}"),
                }
            }
        }
        assert_eq!(r.island_pairs[1].len(), 1, "one island pair: stone+aerium+quartz");
    }
}
