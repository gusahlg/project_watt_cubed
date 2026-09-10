//! placement.rs — region-first world generation: where each family wants to
//! exist, as data.
//!
//! Terrain picks REGION LABELS, not authored element names. At compile the
//! builtin regions contribute a centre, six variants and three strata;
//! columns pick a stratum by geology. Ores are `[rock, guest]` with the guest
//! drawn from a nearby region at shallow depth and a far one when deep.
//! Every configuration the generator can emit is interned here, in canonical
//! order, before any worker thread exists.
use std::ops::RangeInclusive;

use material::{Configuration, Element};

use crate::block::regions::{self, Region};
use crate::block::registry::{BlockId, BlockRegistry};

/// Loud tripwire for a table whose reachable set explodes (a mod authoring
/// pathological rules fails at startup, not at chunk 40,000).
pub const ENUM_CAP: usize = 1024;

/// Bumped whenever the same (seed, coord) can yield different chunk MATERIALS
/// than before. Saves stamp it (loader warns on mismatch — edits replay over
/// terrain whose materials moved) and the join handshake folds it into the
/// protocol version (mixed peers get an error instead of silent divergence).
/// v1: the legacy hand-written picker. v2: element-first placement.
/// v3: the alien pass. v4: emergent material table (regions, not named elements).
/// v5: Collision-rest families, recorded jitter, geological strata, neighbourhood ores.
pub const WORLDGEN_VERSION: u16 = 5;

/// Geology cells are `1 << GEOLOGY_SHIFT` blocks on a side. A 16³ chunk sits
/// inside one cell except on the 64-block seams, so the deep Uniform(stone)
/// shortcut survives for most columns.
pub const GEOLOGY_SHIFT: i32 = 6;

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
    CaveWall { depth: RangeInclusive<i32> },
}

/// How a region occupies its context.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Present wherever the context holds. The union of banded regions is the
    /// base material — layering emerges from band overlap.
    Banded,
    /// Present where the context holds AND a per-cell hash roll lands in a
    /// `u32::MAX / rarity` slice, into cells whose banded union is the host
    /// region. Compilation derives BOTH hash streams from one row (B at rarity ×
    /// [`STREAM_B_SCALE`]), so overlap arity is capped at two by construction.
    Scattered { rarity: u32, host: &'static str },
}

/// Where one region (centre or named variant) wants to exist.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PlacementRule {
    pub material: &'static str,
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
    /// Ground depths 2..=3, by [`SurfaceKind`].
    pub crust: [BlockId; SurfaceKind::COUNT],
    /// Surface (depth 1) scatter by [`SurfaceKind`].
    pub surface_scatter: Vec<Vec<Slice>>,
    /// Ground depth ≥ 4, overhang shelves, island interiors (region centre).
    pub stone: BlockId,
    /// Three rock strata the generator picks by geology index 0..=2.
    pub stone_strata: [BlockId; 3],
    /// Flooded cells.
    pub water: BlockId,
    /// Ground ore slices, stream A — byte-identical walk to the legacy seams.
    pub seams: Vec<Slice>,
    /// Ground ore slices, stream B (rarity ÷ [`STREAM_B_SCALE`], same order).
    pub seams_b: Vec<Slice>,
    /// Pair block for ground seams i and j (i > j). Lower-triangular; indexed
    /// `pairs[i][j]`.
    pub pairs: Vec<Vec<BlockId>>,
    /// Island surface, not icy / icy.
    pub island_grass: BlockId,
    pub island_ice: BlockId,
    /// Island cells 1..=3 below the surface.
    pub island_crust: BlockId,
    /// Island interior scatter, streams A and B, plus the reachable island pair.
    pub island_seams: Vec<Slice>,
    pub island_seams_b: Vec<Slice>,
    pub island_pairs: Vec<Vec<BlockId>>,
    /// Depth below which no seam places — re-derives the deep Uniform(stone)
    /// proof bound from the table instead of a hardcoded constant.
    pub max_scattered_depth: i32,
    /// The cave-wall rule, if any.
    pub cave_wall: Option<Slice>,
}

impl Resolved {
    /// Rock stratum the column at `(wx, wz)` wears. `index` is 0..=2.
    pub fn stone_at(&self, index: usize) -> BlockId {
        self.stone_strata[index % 3]
    }
}

/// Guest mapping for the ore table. Depth bands and rarities match the
/// pre-material seams; the guest is chosen at compile from a region near
/// rock (shallow) or far from rock (deep) in lattice distance. Lamp-like
/// guests glow; glass-like guests are see-through veins.
///
/// | legacy   | depth   | rarity |
/// |----------|---------|--------|
/// | Coal     | 3..=64  | 90     |
/// | Iron     | 8..=64  | 110    |
/// | Copper   | 8..=64  | 130    |
/// | Sulfur   | 20..=64 | 240    |
/// | Quartz   | 20..=64 | 200    |
/// | Lead     | 20..=64 | 220    |
/// | Gold     | 32..=64 | 300    |
/// | Lumin    | 32..=64 | 380    |
/// | Titan    | 48..=64 | 460    |
/// | Obsidian | 48..=64 | 240    |
///
/// Island: two far guests, r45 and r160. Surface: grassy lamp on [organic,
/// soil] r700; desert glass on [sand] r900. Cave-wall: lamp on rock, r40,
/// depth 32.. The `"ore"` material on a ground/island seam is a slot the
/// compiler rewrites to the neighbourhood guest.
pub fn builtin() -> PlacementTable {
    let ground = |depth: RangeInclusive<i32>, surface: SurfaceMask| Context::Ground { depth, surface };
    let island = |below: RangeInclusive<i32>, surface: IslandSurface| Context::Island { below, surface };
    let rule = |material: &'static str, context: Context, kind: Kind| PlacementRule {
        material,
        context,
        kind,
    };
    let banded = |material: &'static str, context: Context| rule(material, context, Kind::Banded);
    let seam = |depth: RangeInclusive<i32>, rarity: u32| {
        rule("ore", ground(depth, SurfaceMask::ANY), Kind::Scattered { rarity, host: "rock" })
    };
    let island_seam = |rarity: u32| {
        rule(
            "ore",
            island(4..=i32::MAX, IslandSurface::Any),
            Kind::Scattered { rarity, host: "rock" },
        )
    };

    PlacementTable {
        rules: vec![
            // --- Ground crust: the surface dress and the soil band under it.
            banded("organic", ground(1..=1, SurfaceMask::GRASSY)),
            banded("soil", ground(1..=1, SurfaceMask::GRASSY)),
            banded("snow", ground(1..=1, SurfaceMask::SNOWY)),
            banded("sand", ground(1..=1, SurfaceMask::SHORE.or(SurfaceMask::DESERT))),
            banded("soil", ground(1..=1, SurfaceMask::BEACH_EDGE)),
            banded("sand", ground(1..=1, SurfaceMask::BEACH_EDGE)),
            banded("soil", ground(2..=3, SurfaceMask::GRASSY.or(SurfaceMask::SHORE).or(SurfaceMask::BEACH_EDGE))),
            banded("clay", ground(2..=3, SurfaceMask::GRASSY.or(SurfaceMask::SHORE).or(SurfaceMask::BEACH_EDGE).or(SurfaceMask::DESERT))),
            banded("sand", ground(2..=3, SurfaceMask::DESERT)),
            banded("soil", ground(2..=3, SurfaceMask::SNOWY)),
            banded("ice", ground(2..=3, SurfaceMask::SNOWY)),
            // --- The world's rock.
            banded("rock", ground(4..=i32::MAX, SurfaceMask::ANY)),
            banded("rock", Context::Overhang),
            // --- Water fills to the table.
            banded("water", Context::Flood),
            // --- Ground ores: depth bands and rarities of the legacy seams table.
            seam(3..=64, 90),
            seam(8..=64, 110),
            seam(8..=64, 130),
            seam(20..=64, 240),
            seam(20..=64, 200),
            seam(20..=64, 220),
            seam(32..=64, 300),
            seam(32..=64, 380),
            seam(48..=64, 460),
            seam(48..=64, 240),
            // --- Flying islands.
            banded("organic", island(0..=0, IslandSurface::NotIcy)),
            banded("soil", island(0..=0, IslandSurface::NotIcy)),
            banded("ice", island(0..=0, IslandSurface::OnlyIcy)),
            banded("soil", island(1..=3, IslandSurface::Any)),
            banded("clay", island(1..=3, IslandSurface::Any)),
            banded("rock", island(4..=i32::MAX, IslandSurface::Any)),
            island_seam(45),
            island_seam(160),
            // --- Surface glow.
            rule(
                "lamp",
                ground(1..=1, SurfaceMask::GRASSY),
                Kind::Scattered { rarity: 700, host: "soil" },
            ),
            rule(
                "glass",
                ground(1..=1, SurfaceMask::DESERT),
                Kind::Scattered { rarity: 900, host: "sand" },
            ),
            // --- Lamp clusters on deep cavern walls.
            rule(
                "lamp",
                Context::CaveWall { depth: 32..=i32::MAX },
                Kind::Scattered { rarity: 40, host: "rock" },
            ),
        ],
    }
}

impl PlacementTable {
    /// Intern every configuration terrain can emit, in canonical order, and
    /// resolve the generator's LUTs. Runs at startup on the main thread.
    pub fn compile(&self, registry: &mut BlockRegistry) -> Resolved {
        let before = registry.block_count();
        let law = *registry.law();
        let regions = regions::builtin(&law);
        intern_families(registry, &regions);

        let intern = |specs: &[&str], registry: &mut BlockRegistry| -> BlockId {
            intern_union(registry, &regions, specs)
        };

        let dress = SurfaceKind::ALL.map(|k| {
            intern(&self.ground_banded_specs(1, k), registry)
        });
        let crust = SurfaceKind::ALL.map(|k| {
            intern(&self.ground_banded_specs(2, k), registry)
        });
        let stone = intern(&self.ground_banded_specs(4, SurfaceKind::Grassy), registry);
        let rock = regions
            .iter()
            .find(|r| r.label == "rock")
            .expect("builtin regions include rock");
        let stone_strata = [0, 1, 2].map(|i| {
            intern_elements(
                registry,
                &regions,
                &[rock.stratum(i)],
                &format!("rock:{}", i),
            )
        });

        let surface_scatter: Vec<Vec<Slice>> = SurfaceKind::ALL
            .iter()
            .map(|&kind| {
                let base = self.ground_banded_specs(1, kind);
                self.surface_scatter_rules(kind)
                    .iter()
                    .map(|r| {
                        let mut union = base.clone();
                        union.push(r.material);
                        Slice {
                            min_depth: 1,
                            width: u32::MAX / r.rarity(),
                            id: intern(&union, registry),
                        }
                    })
                    .collect()
            })
            .collect();

        let ground_seams = self.ore_seams();
        let island_seams = self.scattered(|c| matches!(c, Context::Island { .. }));
        let ground_depths: Vec<i32> = ground_seams
            .iter()
            .map(|r| *r.scatter_parts().2.start())
            .collect();
        let island_depths: Vec<i32> = island_seams.iter().map(|_| 48).collect();
        let mut used: Vec<Element> = Vec::new();
        let ground_guests = pick_guests(&regions, rock.centre, &ground_depths, &mut used);
        let island_guests = pick_guests(&regions, rock.centre, &island_depths, &mut used);

        let slices = |seams: &[&PlacementRule],
                      guests: &[(String, Element)],
                      scale: u32,
                      registry: &mut BlockRegistry|
         -> Vec<Slice> {
            let mut min_depth = i32::MIN;
            seams
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let (host, range) = {
                        let (h, _, range) = r.scatter_parts();
                        (h, range)
                    };
                    assert!(
                        *range.start() >= min_depth,
                        "scattered rules must be authored shallow-to-deep: the cumulative \
                         slice walk breaks at the first ineligible depth"
                    );
                    min_depth = *range.start();
                    let guest = guests[i].0.as_str();
                    Slice {
                        min_depth: *range.start(),
                        width: u32::MAX / (r.rarity() * scale),
                        id: intern(&[host, guest], registry),
                    }
                })
                .collect()
        };
        let pair_matrix = |seams: &[&PlacementRule],
                           guests: &[(String, Element)],
                           registry: &mut BlockRegistry|
         -> Vec<Vec<BlockId>> {
            (0..seams.len())
                .map(|i| {
                    let (host, ri) = {
                        let (h, _, ri) = seams[i].scatter_parts();
                        (h, ri)
                    };
                    let ei = guests[i].0.as_str();
                    (0..i)
                        .map(|j| {
                            let (_, rj) = {
                                let (_, _, rj) = seams[j].scatter_parts();
                                ((), rj)
                            };
                            let ej = guests[j].0.as_str();
                            if ranges_overlap(&ri, &rj) {
                                intern(&[host, ei, ej], registry)
                            } else {
                                intern(&[host, ei], registry)
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
                    id: intern(&[host, e], registry),
                }
            });

        let resolved = Resolved {
            dress,
            crust,
            surface_scatter,
            stone,
            stone_strata,
            water: intern(&self.banded_specs(|c| matches!(c, Context::Flood)), registry),
            seams: slices(&ground_seams, &ground_guests, 1, registry),
            seams_b: slices(&ground_seams, &ground_guests, STREAM_B_SCALE, registry),
            pairs: pair_matrix(&ground_seams, &ground_guests, registry),
            island_grass: intern(&self.island_banded_specs(0, IslandSurface::NotIcy), registry),
            island_ice: intern(&self.island_banded_specs(0, IslandSurface::OnlyIcy), registry),
            island_crust: intern(&self.island_banded_specs(1, IslandSurface::Any), registry),
            island_seams: slices(&island_seams, &island_guests, 1, registry),
            island_seams_b: slices(&island_seams, &island_guests, STREAM_B_SCALE, registry),
            island_pairs: pair_matrix(&island_seams, &island_guests, registry),
            max_scattered_depth,
            cave_wall,
        };

        assert!(
            registry.block_count() - before <= ENUM_CAP,
            "placement table interned {} configurations (cap {ENUM_CAP})",
            registry.block_count() - before
        );
        debug_assert!(
            regions::families_at_rest(&law, &regions),
            "worldgen families are not at rest under Collision"
        );
        resolved
    }

    fn ground_banded_specs(&self, depth: i32, kind: SurfaceKind) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Banded))
            .filter(|r| match &r.context {
                Context::Ground { depth: d, surface } => d.contains(&depth) && surface.contains(kind),
                _ => false,
            })
            .map(|r| r.material)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn island_banded_specs(&self, below: i32, surf: IslandSurface) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self
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
            .map(|r| r.material)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn banded_specs(&self, pred: impl Fn(&Context) -> bool) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Banded) && pred(&r.context))
            .map(|r| r.material)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn scattered(&self, pred: impl Fn(&Context) -> bool) -> Vec<&PlacementRule> {
        self.rules
            .iter()
            .filter(|r| matches!(r.kind, Kind::Scattered { .. }) && pred(&r.context))
            .collect()
    }

    fn ore_seams(&self) -> Vec<&PlacementRule> {
        self.scattered(|c| matches!(c, Context::Ground { depth, .. } if *depth.start() >= 2))
    }

    fn surface_scatter_rules(&self, kind: SurfaceKind) -> Vec<&PlacementRule> {
        self.scattered(|c| {
            matches!(c, Context::Ground { depth, surface }
                if *depth.start() == 1 && *depth.end() == 1 && surface.contains(kind))
        })
    }
}

impl PlacementRule {
    fn scatter_parts(&self) -> (&'static str, &'static str, RangeInclusive<i32>) {
        let Kind::Scattered { host, .. } = &self.kind else {
            panic!("scatter_parts on a banded rule")
        };
        let range = match &self.context {
            Context::Ground { depth, .. } | Context::CaveWall { depth } => depth.clone(),
            Context::Island { below, .. } => below.clone(),
            Context::Flood | Context::Overhang => 0..=0,
        };
        (*host, self.material, range)
    }

    fn rarity(&self) -> u32 {
        match self.kind {
            Kind::Scattered { rarity, .. } => rarity,
            Kind::Banded => unreachable!("rarity of a banded rule"),
        }
    }
}

fn intern_families(registry: &mut BlockRegistry, regions: &[Region]) {
    for r in regions {
        for i in 0..7 {
            let id = registry
                .intern(&Configuration::single(r.member(i)))
                .expect("block palette cannot hold a region family");
            let label = if i == 0 {
                r.label.to_string()
            } else {
                format!("{}#{i}", r.label)
            };
            registry.set_label(id, &label);
        }
        for i in 0..3 {
            let id = registry
                .intern(&Configuration::single(r.stratum(i)))
                .expect("block palette cannot hold a region stratum");
            registry.set_label(id, &format!("{}:{i}", r.label));
        }
    }
}

fn intern_union(registry: &mut BlockRegistry, regions: &[Region], specs: &[&str]) -> BlockId {
    assert!(!specs.is_empty(), "placement union is empty");
    let elems: Vec<Element> = specs.iter().map(|s| resolve_element(regions, s)).collect();
    intern_elements(registry, regions, &elems, &specs_label(specs))
}

fn intern_elements(
    registry: &mut BlockRegistry,
    regions: &[Region],
    elems: &[Element],
    label: &str,
) -> BlockId {
    assert!(!elems.is_empty(), "placement union is empty");
    let mut elems = elems.to_vec();
    // Canonical intern order: region table index, then variant/stratum index, so
    // rule authoring order cannot reshuffle ids. Multiplicity is kept.
    elems.sort_by_key(|e| element_rank(regions, *e));
    let cfg = Configuration::new(elems).expect("union fits CONFIG_MAX");
    let id = registry.intern(&cfg).expect("block palette cannot hold a placement union");
    registry.set_label(id, label);
    id
}

fn element_rank(regions: &[Region], e: Element) -> (usize, usize) {
    regions
        .iter()
        .enumerate()
        .find_map(|(ri, r)| {
            (0..7)
                .find(|&vi| r.member(vi) == e)
                .map(|vi| (ri, vi))
                .or_else(|| (0..3).find(|&si| r.stratum(si) == e).map(|si| (ri, 7 + si)))
        })
        .unwrap_or((usize::MAX, 0))
}

/// Guest elements for `depths`, near rock when shallow and far when deep.
/// Already-used elements are skipped so neighbouring bands stay distinct.
fn pick_guests(
    regions: &[Region],
    rock: Element,
    depths: &[i32],
    used: &mut Vec<Element>,
) -> Vec<(String, Element)> {
    let mut cands: Vec<(String, Element, u32)> = Vec::new();
    for r in regions {
        if r.label == "rock" || r.label == "water" {
            continue;
        }
        for i in 0..7 {
            let e = r.member(i);
            if e == rock || cands.iter().any(|(_, x, _)| *x == e) {
                continue;
            }
            let spec = if i == 0 {
                r.label.to_string()
            } else {
                format!("{}#{i}", r.label)
            };
            cands.push((spec, e, e.distance(rock)));
        }
    }
    assert!(!cands.is_empty(), "no ore-guest candidates in the region table");
    let d_min = cands.iter().map(|c| c.2).min().unwrap();
    let d_max = cands.iter().map(|c| c.2).max().unwrap().max(d_min + 1);
    const LO: i32 = 3;
    const HI: i32 = 48;
    let span = (HI - LO) as u32;
    let mut out = Vec::with_capacity(depths.len());
    for &depth in depths {
        let t = (depth - LO).clamp(0, HI - LO) as u32;
        let want = d_min + (d_max - d_min) * t / span;
        let unused = cands.iter().enumerate().filter(|(_, (_, e, _))| !used.contains(e));
        let mut best: Option<(u32, u32, usize)> = None;
        let pool: Vec<(usize, u32)> = if unused.clone().next().is_some() {
            unused.map(|(i, (_, _, d))| (i, *d)).collect()
        } else {
            cands.iter().enumerate().map(|(i, (_, _, d))| (i, *d)).collect()
        };
        for (i, dist) in pool {
            let key = (dist.abs_diff(want), dist);
            match best {
                Some((bd, br, _)) if (key.0, key.1) >= (bd, br) => {}
                _ => best = Some((key.0, key.1, i)),
            }
        }
        let i = best.expect("ore-guest pool is non-empty").2;
        let (spec, e, _) = cands[i].clone();
        used.push(e);
        out.push((spec, e));
    }
    ensure_guest(&mut out, regions, rock, "lamp");
    ensure_guest(&mut out, regions, rock, "glass");
    out
}

fn ensure_guest(out: &mut [(String, Element)], regions: &[Region], rock: Element, label: &str) {
    if out.iter().any(|(s, _)| s == label || s.starts_with(&format!("{label}#"))) {
        return;
    }
    let Some(r) = regions.iter().find(|r| r.label == label) else { return };
    if out.is_empty() {
        return;
    }
    let want = r.centre.distance(rock);
    let slot = out
        .iter()
        .enumerate()
        .min_by_key(|(_, (_, e))| e.distance(rock).abs_diff(want))
        .map(|(i, _)| i)
        .unwrap_or(0);
    out[slot] = (label.to_string(), r.centre);
}

fn specs_label(specs: &[&str]) -> String {
    let mut v = specs.to_vec();
    v.sort_unstable();
    v.dedup();
    v.join("+")
}

fn resolve_element(regions: &[Region], spec: &str) -> Element {
    if let Some((label, rest)) = spec.split_once('#') {
        let idx: usize = rest.parse().unwrap_or_else(|_| panic!("bad variant spec {spec}"));
        assert!((1..=6).contains(&idx), "variant index of {spec} is not 1..=6");
        let r = regions
            .iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("unknown region {label}"));
        r.member(idx)
    } else {
        regions
            .iter()
            .find(|r| r.label == spec)
            .unwrap_or_else(|| panic!("unknown region {spec}"))
            .centre
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
    fn compile_is_idempotent_and_canonical() {
        let (mut reg, first) = compiled();
        let count = reg.block_count();
        let again = builtin().compile(&mut reg);
        assert_eq!(reg.block_count(), count, "recompile registers nothing new");
        assert_eq!(first.stone, again.stone);
        assert_eq!(first.stone_strata, again.stone_strata);
        assert_eq!(first.dress, again.dress);
        assert_eq!(first.pairs, again.pairs);
    }

    #[test]
    fn compile_interns_within_enum_cap() {
        let mut reg = BlockRegistry::with_builtins();
        let before = reg.block_count();
        builtin().compile(&mut reg);
        assert!(reg.block_count() - before <= ENUM_CAP);
    }

    #[test]
    #[should_panic(expected = "placement union is empty")]
    fn intern_union_rejects_empty() {
        let mut reg = BlockRegistry::with_builtins();
        let regions = regions::builtin(reg.law());
        intern_union(&mut reg, &regions, &[]);
    }

    #[test]
    fn two_compiles_give_identical_ids() {
        let (a, ra) = compiled();
        let (b, rb) = compiled();
        assert_eq!(a.block_count(), b.block_count());
        for i in 0..a.block_count() {
            let id = BlockId(i as u16);
            assert_eq!(a.configuration(id), b.configuration(id));
            assert_eq!(a.label(id), b.label(id));
        }
        assert_eq!(ra.stone, rb.stone);
        assert_eq!(ra.dress, rb.dress);
        assert_eq!(ra.seams.iter().map(|s| s.id).collect::<Vec<_>>(), rb.seams.iter().map(|s| s.id).collect::<Vec<_>>());
    }

    #[test]
    fn banded_row_order_cannot_reshuffle_ids() {
        let (_, a) = compiled();
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
    fn dress_crust_and_stone_are_the_region_unions() {
        let (reg, r) = compiled();
        let id = |n: &str| reg.id_by_label(n).unwrap_or_else(|| panic!("no label {n}"));
        assert_eq!(r.dress[SurfaceKind::Grassy as usize], id("organic+soil"));
        assert_eq!(r.dress[SurfaceKind::Shore as usize], id("sand"));
        assert_eq!(r.dress[SurfaceKind::Snowy as usize], id("snow"));
        assert_eq!(r.dress[SurfaceKind::Desert as usize], id("sand"));
        assert_eq!(r.dress[SurfaceKind::BeachEdge as usize], id("sand+soil"));
        assert_eq!(r.crust[SurfaceKind::Grassy as usize], id("clay+soil"));
        assert_eq!(r.crust[SurfaceKind::Desert as usize], id("clay+sand"));
        assert_eq!(r.crust[SurfaceKind::Snowy as usize], id("ice+soil"));
        assert_eq!(r.stone, id("rock"));
        assert_eq!(r.stone_strata.len(), 3);
        for s in r.stone_strata {
            let els = reg.configuration(s).elements();
            assert_eq!(els.len(), 1, "a stone stratum is a single rock-family element");
        }
        assert_eq!(r.water, id("water"));
        assert_eq!(r.island_ice, id("ice"));
        assert_eq!(r.island_grass, r.dress[SurfaceKind::Grassy as usize]);
        assert_eq!(r.island_crust, r.crust[SurfaceKind::Grassy as usize]);
        assert_eq!(r.cave_wall.unwrap().id, id("lamp+rock"));
        assert_eq!(
            r.surface_scatter[SurfaceKind::Grassy as usize][0].id,
            id("lamp+organic+soil")
        );
        assert_eq!(
            r.surface_scatter[SurfaceKind::Desert as usize][0].id,
            id("glass+sand")
        );
        assert!(r.surface_scatter[SurfaceKind::Shore as usize].is_empty());
    }

    #[test]
    fn seams_keep_the_legacy_depth_bands() {
        let (_reg, r) = compiled();
        let legacy: [(i32, u32); 10] = [
            (3, 90),
            (8, 110),
            (8, 130),
            (20, 240),
            (20, 200),
            (20, 220),
            (32, 300),
            (32, 380),
            (48, 460),
            (48, 240),
        ];
        assert_eq!(r.seams.len(), legacy.len());
        for (slice, (min_depth, rarity)) in r.seams.iter().zip(legacy) {
            assert_eq!(slice.min_depth, min_depth);
            assert_eq!(slice.width, u32::MAX / rarity);
        }
        for (a, b) in r.seams.iter().zip(&r.seams_b) {
            assert_eq!(a.id, b.id);
            assert_eq!(b.width, u32::MAX / ((u32::MAX / a.width) * STREAM_B_SCALE));
        }
        assert_eq!(r.max_scattered_depth, 64);
    }

    #[test]
    fn ore_guests_recede_from_rock_with_depth() {
        let (reg, r) = compiled();
        let rock = *reg.configuration(r.stone).elements().first().unwrap();
        let mut last = 0u32;
        for slice in &r.seams {
            let els = reg.configuration(slice.id).elements();
            assert!(els.contains(&rock), "ore {} is not [rock, guest]", slice.min_depth);
            let guest = els.iter().copied().find(|e| *e != rock).expect("ore has a guest");
            let d = guest.distance(rock);
            assert!(
                d + 32 >= last,
                "ore depth {} is closer to rock ({d}) than a shallower seam ({last})",
                slice.min_depth
            );
            last = last.max(d);
        }
        let mut saw_lamp = false;
        let mut saw_glass = false;
        let mut check = |id: BlockId| {
            for &e in reg.configuration(id).elements() {
                let o = material::observe(reg.law(), &Configuration::single(e));
                if o.emission >= 8 {
                    saw_lamp = true;
                }
                if o.transparency >= 160 {
                    saw_glass = true;
                }
            }
        };
        for slice in r.seams.iter().chain(r.island_seams.iter()).chain(r.cave_wall.iter()) {
            check(slice.id);
        }
        for row in r.surface_scatter.iter() {
            for s in row {
                check(s.id);
            }
        }
        assert!(saw_lamp, "a lamp-like guest must glow (crystal caves)");
        assert!(saw_glass, "a glass-like guest must be see-through");
    }

    #[test]
    fn every_pair_hosts_rock() {
        let (reg, r) = compiled();
        let rock = reg.id_by_label("rock").unwrap();
        let rock_e = *reg.configuration(rock).elements().first().unwrap();
        for (i, row) in r.pairs.iter().enumerate() {
            assert_eq!(row.len(), i);
            for &pid in row {
                let els = reg.configuration(pid).elements();
                assert!(els.len() >= 2, "pair blocks are host + payloads");
                assert!(els.contains(&rock_e));
            }
        }
        assert_eq!(r.island_pairs[1].len(), 1, "one island pair");
    }

    #[test]
    fn families_are_labelled() {
        let (reg, _) = compiled();
        for label in ["rock", "soil", "sand", "clay", "organic", "water", "ice", "snow", "glass", "lamp"] {
            assert!(reg.id_by_label(label).is_some(), "missing {label}");
            assert!(reg.id_by_label(&format!("{label}#1")).is_some(), "missing {label}#1");
            assert!(reg.id_by_label(&format!("{label}:0")).is_some(), "missing {label}:0");
        }
    }

    #[test]
    fn families_and_ores_rest_under_contact_and_collision() {
        use crate::block::regions::pair_rest;
        use material::EventKind;
        let (reg, r) = compiled();
        let law = *reg.law();
        let regions = regions::builtin(&law);
        let mut members: Vec<Configuration> = regions.iter().flat_map(|g| g.matter().map(Configuration::single)).collect();
        for slice in r
            .seams
            .iter()
            .chain(r.seams_b.iter())
            .chain(r.island_seams.iter())
            .chain(r.cave_wall.iter())
            .chain(r.surface_scatter.iter().flatten())
        {
            members.push(reg.configuration(slice.id).clone());
        }
        for row in r.pairs.iter().chain(r.island_pairs.iter()) {
            for &id in row {
                members.push(reg.configuration(id).clone());
            }
        }
        for s in r.stone_strata {
            members.push(reg.configuration(s).clone());
        }
        assert!(pair_rest(&law, &members, EventKind::NewContact), "a pair reacted under NewContact");
        assert!(pair_rest(&law, &members, EventKind::Collision), "a pair reacted under Collision");
    }

    #[test]
    fn generated_chunk_stays_still_under_external_change() {
        use crate::sim::reactions::{Budget, CellStore, MaterialEvent, Pos, ReactionScheduler};
        use crate::world::chunk::{Chunk, CHUNK_SIZE};
        use crate::world::generation::Terrain;
        use material::EventKind;
        use std::collections::HashMap;

        struct Map {
            cells: HashMap<Pos, BlockId>,
            reg: BlockRegistry,
        }
        impl CellStore for Map {
            fn block_at(&self, pos: Pos) -> Option<BlockId> {
                Some(*self.cells.get(&pos).unwrap_or(&crate::block::registry::AIR))
            }
            fn set_block(&mut self, pos: Pos, id: BlockId) -> BlockId {
                self.cells.insert(pos, id).unwrap_or(crate::block::registry::AIR)
            }
            fn registry(&self) -> &BlockRegistry {
                &self.reg
            }
            fn registry_mut(&mut self) -> &mut BlockRegistry {
                &mut self.reg
            }
        }

        let mut reg = BlockRegistry::with_builtins();
        let g = Terrain::new(&mut reg, 20.0, 42);
        let chunk = Chunk::new(0, 1, 0, &g);
        let mut m = Map {
            cells: HashMap::new(),
            reg,
        };
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    m.set_block((x as i32, y as i32, z as i32), chunk.get_local(x, y, z));
                }
            }
        }
        let before = m.cells.clone();
        let mut sched = ReactionScheduler::new();
        for z in 0..CHUNK_SIZE as i32 {
            for y in 0..CHUNK_SIZE as i32 {
                for x in 0..CHUNK_SIZE as i32 {
                    sched.push(MaterialEvent {
                        at: (x, y, z),
                        kind: EventKind::ExternallyChanged,
                    });
                }
            }
        }
        let law = *m.registry().law();
        let out = sched.tick(
            &mut m,
            &law,
            Budget {
                events_per_generation: CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE,
                generations_per_tick: 20,
                max_followups: CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 6,
            },
        );
        assert!(out.is_empty(), "generated matter mutated under ExternallyChanged: {} edits", out.len());
        assert_eq!(m.cells, before, "16³ placement fill drifted after 20 generations");
    }
}
