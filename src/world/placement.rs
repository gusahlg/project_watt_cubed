//! placement.rs — region-first world generation: where each family wants to
//! exist, as data.
//!
//! Terrain picks REGION LABELS, not authored element names. At compile the
//! builtin regions contribute a centre plus six variants; columns pick a
//! family member through the existing hash stream. Every configuration the
//! generator can emit is interned here, in canonical order, before any worker
//! thread exists.
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
pub const WORLDGEN_VERSION: u16 = 4;

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
    /// Ground depth ≥ 4, overhang shelves, island interiors.
    pub stone: BlockId,
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

/// Guest mapping for the ore table. Depth bands and rarities match the
/// pre-material seams; names are region variants, not elements:
///
/// | legacy   | depth   | rarity | guest   |
/// |----------|---------|--------|---------|
/// | Coal     | 3..=64  | 90     | lamp#1  |
/// | Iron     | 8..=64  | 110    | clay#1  |
/// | Copper   | 8..=64  | 130    | glass#1 |
/// | Sulfur   | 20..=64 | 240    | lamp#2  |
/// | Quartz   | 20..=64 | 200    | glass#2 |
/// | Lead     | 20..=64 | 220    | clay#2  |
/// | Gold     | 32..=64 | 300    | lamp#3  |
/// | Lumin    | 32..=64 | 380    | lamp#4  |
/// | Titan    | 48..=64 | 460    | glass#3 |
/// | Obsidian | 48..=64 | 240    | clay#3  |
///
/// Island: Aerium r45 → lamp#5, Quartz r160 → glass#4.
/// Surface: grassy Lumin r700 → lamp#6 on [organic, soil]; desert Phosphor
/// r900 → glass#5 on [sand]. Cave-wall: lamp (centre) on rock, r40, depth 32..
pub fn builtin() -> PlacementTable {
    let ground = |depth: RangeInclusive<i32>, surface: SurfaceMask| Context::Ground { depth, surface };
    let island = |below: RangeInclusive<i32>, surface: IslandSurface| Context::Island { below, surface };
    let rule = |material: &'static str, context: Context, kind: Kind| PlacementRule {
        material,
        context,
        kind,
    };
    let banded = |material: &'static str, context: Context| rule(material, context, Kind::Banded);
    let seam = |guest: &'static str, depth: RangeInclusive<i32>, rarity: u32| {
        rule(guest, ground(depth, SurfaceMask::ANY), Kind::Scattered { rarity, host: "rock" })
    };
    let island_seam = |guest: &'static str, rarity: u32| {
        rule(
            guest,
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
            seam("lamp#1", 3..=64, 90),
            seam("clay#1", 8..=64, 110),
            seam("glass#1", 8..=64, 130),
            seam("lamp#2", 20..=64, 240),
            seam("glass#2", 20..=64, 200),
            seam("clay#2", 20..=64, 220),
            seam("lamp#3", 32..=64, 300),
            seam("lamp#4", 32..=64, 380),
            seam("glass#3", 48..=64, 460),
            seam("clay#3", 48..=64, 240),
            // --- Flying islands.
            banded("organic", island(0..=0, IslandSurface::NotIcy)),
            banded("soil", island(0..=0, IslandSurface::NotIcy)),
            banded("ice", island(0..=0, IslandSurface::OnlyIcy)),
            banded("soil", island(1..=3, IslandSurface::Any)),
            banded("clay", island(1..=3, IslandSurface::Any)),
            banded("rock", island(4..=i32::MAX, IslandSurface::Any)),
            island_seam("lamp#5", 45),
            island_seam("glass#4", 160),
            // --- Surface glow.
            rule(
                "lamp#6",
                ground(1..=1, SurfaceMask::GRASSY),
                Kind::Scattered { rarity: 700, host: "soil" },
            ),
            rule(
                "glass#5",
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

        let slices = |seams: &[&PlacementRule], scale: u32, registry: &mut BlockRegistry| -> Vec<Slice> {
            let mut min_depth = i32::MIN;
            seams
                .iter()
                .map(|r| {
                    let (host, guest, range) = r.scatter_parts();
                    assert!(
                        *range.start() >= min_depth,
                        "scattered rules must be authored shallow-to-deep: the cumulative \
                         slice walk breaks at the first ineligible depth"
                    );
                    min_depth = *range.start();
                    Slice {
                        min_depth: *range.start(),
                        width: u32::MAX / (r.rarity() * scale),
                        id: intern(&[host, guest], registry),
                    }
                })
                .collect()
        };
        let pair_matrix = |seams: &[&PlacementRule], registry: &mut BlockRegistry| -> Vec<Vec<BlockId>> {
            (0..seams.len())
                .map(|i| {
                    let (host, ei, ri) = seams[i].scatter_parts();
                    (0..i)
                        .map(|j| {
                            let (_, ej, rj) = seams[j].scatter_parts();
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
            water: intern(&self.banded_specs(|c| matches!(c, Context::Flood)), registry),
            seams: slices(&ground_seams, 1, registry),
            seams_b: slices(&ground_seams, STREAM_B_SCALE, registry),
            pairs: pair_matrix(&ground_seams, registry),
            island_grass: intern(&self.island_banded_specs(0, IslandSurface::NotIcy), registry),
            island_ice: intern(&self.island_banded_specs(0, IslandSurface::OnlyIcy), registry),
            island_crust: intern(&self.island_banded_specs(1, IslandSurface::Any), registry),
            island_seams: slices(&island_seams, 1, registry),
            island_seams_b: slices(&island_seams, STREAM_B_SCALE, registry),
            island_pairs: pair_matrix(&island_seams, registry),
            max_scattered_depth,
            cave_wall,
        };

        assert!(
            registry.block_count() <= ENUM_CAP,
            "placement table interned {} configurations (cap {ENUM_CAP})",
            registry.block_count()
        );
        debug_assert!(
            regions::families_at_rest(&law, &regions),
            "worldgen families are not at rest under NewContact"
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
    }
}

fn intern_union(registry: &mut BlockRegistry, regions: &[Region], specs: &[&str]) -> BlockId {
    let mut elems: Vec<Element> = specs.iter().map(|s| resolve_element(regions, s)).collect();
    // Canonical intern order: region table index, then variant index, so rule
    // authoring order cannot reshuffle ids. Multiplicity is kept.
    elems.sort_by_key(|e| {
        regions
            .iter()
            .enumerate()
            .find_map(|(ri, r)| (0..7).find(|&vi| r.member(vi) == *e).map(|vi| (ri, vi)))
            .unwrap_or((usize::MAX, 0))
    });
    let cfg = Configuration::new(elems).expect("union fits CONFIG_MAX");
    let id = registry.intern(&cfg).expect("block palette cannot hold a placement union");
    let label = specs_label(specs);
    registry.set_label(id, &label);
    id
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
        assert_eq!(first.dress, again.dress);
        assert_eq!(first.pairs, again.pairs);
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
        assert_eq!(r.water, id("water"));
        assert_eq!(r.island_ice, id("ice"));
        assert_eq!(r.island_grass, r.dress[SurfaceKind::Grassy as usize]);
        assert_eq!(r.island_crust, r.crust[SurfaceKind::Grassy as usize]);
        assert_eq!(r.cave_wall.unwrap().id, id("lamp+rock"));
        assert_eq!(
            r.surface_scatter[SurfaceKind::Grassy as usize][0].id,
            id("lamp#6+organic+soil")
        );
        assert_eq!(
            r.surface_scatter[SurfaceKind::Desert as usize][0].id,
            id("glass#5+sand")
        );
        assert!(r.surface_scatter[SurfaceKind::Shore as usize].is_empty());
    }

    #[test]
    fn seams_keep_the_legacy_depth_bands() {
        let (reg, r) = compiled();
        let legacy: [(&str, i32, u32); 10] = [
            ("lamp#1+rock", 3, 90),
            ("clay#1+rock", 8, 110),
            ("glass#1+rock", 8, 130),
            ("lamp#2+rock", 20, 240),
            ("glass#2+rock", 20, 200),
            ("clay#2+rock", 20, 220),
            ("lamp#3+rock", 32, 300),
            ("lamp#4+rock", 32, 380),
            ("glass#3+rock", 48, 460),
            ("clay#3+rock", 48, 240),
        ];
        assert_eq!(r.seams.len(), legacy.len());
        for (slice, (name, min_depth, rarity)) in r.seams.iter().zip(legacy) {
            assert_eq!(slice.min_depth, min_depth, "{name}");
            assert_eq!(slice.width, u32::MAX / rarity, "{name}");
            assert_eq!(slice.id, reg.id_by_label(name).unwrap(), "{name}");
        }
        for (a, b) in r.seams.iter().zip(&r.seams_b) {
            assert_eq!(a.id, b.id);
            assert_eq!(b.width, u32::MAX / ((u32::MAX / a.width) * STREAM_B_SCALE));
        }
        assert_eq!(r.max_scattered_depth, 64);
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
        }
    }
}
