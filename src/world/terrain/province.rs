//! Realms, regions and provinces.
//!
//! Each face is a realm: a weighted table of themes, biased by climate. Regions (~60 km) shift
//! climate and relief; provinces (~5 km) pick a theme. Both fields are cellular noise on the
//! body-space surface point, so the two faces of an edge read one province. A border a few
//! hundred blocks wide blends the numbers and dithers the discrete choices (surface, trees).

use super::noise::{cellular3, hash3, perlin3, unit};
use crate::coord::Face;
use crate::space::FaceFrame;

/// Province spacing, in blocks (~5 km).
pub const PROVINCE: f64 = 4_500.0;
/// Region spacing, in blocks (~60 km).
pub const REGION: f64 = 60_000.0;
/// Province border, in blocks. Numeric parameters finish blending across it.
pub const BORDER: f32 = 600.0;
/// How many feature densities a theme carries for later passes.
pub const FEAT: usize = 10;

const REGION_BORDER: f32 = 4_000.0;
/// Keeps a face plane (a multiple of the body's half-size) off a cell wall.
const SHIFT: [f64; 3] = [173.0, 419.0, 281.0];
const PROVINCE_SALT: u32 = 0x5A11_CE11;
const REGION_SALT: u32 = 0x6E61_0E11;

/// Feature slots, in order: spires, cones, crystals, mushrooms, islands, craters, ruins,
/// giants, bones, hoodoos.
#[allow(dead_code)]
pub const FEATS: [&str; FEAT] =
    ["spires", "cones", "crystals", "mushrooms", "islands", "craters", "ruins", "giants", "bones", "hoodoos"];

/// One face's character.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Realm {
    Green = 0,
    Ashen,
    Dune,
    Shattered,
    Glass,
    Fungal,
    /// A lush twin: forests and flowers on every face.
    Lush,
    /// A crystalline twin.
    Crystal,
}

impl Realm {
    pub fn of_home(face: Face) -> Self {
        match face {
            Face::PosY => Self::Green,
            Face::NegY => Self::Ashen,
            Face::PosX => Self::Dune,
            Face::NegX => Self::Shattered,
            Face::PosZ => Self::Glass,
            Face::NegZ => Self::Fungal,
        }
    }

    fn bias(self) -> (f32, f32) {
        match self {
            Self::Green => (0.0, 0.06),
            Self::Ashen => (0.16, -0.34),
            Self::Dune => (0.10, -0.40),
            Self::Shattered => (-0.06, -0.16),
            Self::Glass => (-0.42, -0.04),
            Self::Fungal => (0.02, 0.34),
            Self::Lush => (0.04, 0.20),
            Self::Crystal => (-0.30, -0.10),
        }
    }
}

/// A province theme. The discriminant is the index into [`THEMES`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ThemeId {
    Meadow = 0,
    Flower,
    Broadleaf,
    Giant,
    Autumn,
    Blossom,
    Taiga,
    Alpine,
    Glacier,
    Canyon,
    Mesa,
    Dune,
    Badlands,
    Salt,
    Karst,
    Volcanic,
    Ash,
    Crystal,
    Fungal,
    GlowMoss,
    Bone,
    Islands,
    Crater,
    Petrified,
    Terraced,
    Tundra,
}

pub const THEME_COUNT: usize = 26;

/// What the top of a column is made of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surf {
    Grass,
    Meadow,
    Sand,
    Redsand,
    Snow,
    Gravel,
    Moss,
    Tundra,
    Lichen,
    Ash,
    Salt,
    Clay,
    Limestone,
    Basalt,
    Bone,
    Marble,
    Petrified,
    Regolith,
    Mud,
    Ice,
    Obsidian,
    Soil,
}

/// Cliff bands under the soil.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strata {
    Rock,
    Sandstone,
    Basalt,
    Limestone,
    Ice,
    Ash,
    Bone,
    Crystal,
}

/// Which of today's tree shapes a theme grows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Species {
    None,
    Broadleaf,
    Conifer,
    Autumn,
    Blossom,
}

/// Flower colours a theme scatters. Empty means none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Petals {
    None,
    Mixed,
    Warm,
    Cool,
    White,
}

/// One theme's parameters. Numbers blend across a border; the materials and the species dither.
struct Theme {
    /// Map export.
    #[allow(dead_code)]
    name: &'static str,
    /// Province-colour map.
    #[allow(dead_code)]
    rgb: [u8; 3],
    relief: f32,
    hills: f32,
    base: f32,
    terrace: f32,
    dune: f32,
    flat: f32,
    surf: Surf,
    sub: Surf,
    depth: i32,
    strata: Strata,
    trees: f32,
    species: Species,
    flowers: f32,
    petals: Petals,
    temp: f32,
    moist: f32,
    feats: [f32; FEAT],
}

const fn row(
    name: &'static str,
    rgb: [u8; 3],
    relief: f32,
    hills: f32,
    base: f32,
    terrace: f32,
    dune: f32,
    flat: f32,
    surf: Surf,
    sub: Surf,
    depth: i32,
    strata: Strata,
    trees: f32,
    species: Species,
    flowers: f32,
    petals: Petals,
    temp: f32,
    moist: f32,
    feats: [f32; FEAT],
) -> Theme {
    Theme {
        name,
        rgb,
        relief,
        hills,
        base,
        terrace,
        dune,
        flat,
        surf,
        sub,
        depth,
        strata,
        trees,
        species,
        flowers,
        petals,
        temp,
        moist,
        feats,
    }
}

const fn feat(
    spires: f32,
    cones: f32,
    crystals: f32,
    mushrooms: f32,
    islands: f32,
    craters: f32,
    ruins: f32,
    giants: f32,
    bones: f32,
    hoodoos: f32,
) -> [f32; FEAT] {
    [spires, cones, crystals, mushrooms, islands, craters, ruins, giants, bones, hoodoos]
}

const Z: [f32; FEAT] = [0.0; FEAT];

const THEMES: [Theme; THEME_COUNT] = [
    row("meadow plains", [96, 168, 64], 1.08, 0.75, 1.0, 0.0, 0.0, 0.05, Surf::Grass, Surf::Soil, 3, Strata::Rock, 0.10, Species::Broadleaf, 0.055, Petals::Mixed, 0.66, 0.52, Z),
    row("flower fields", [214, 126, 168], 1.00, 0.5, 0.0, 0.0, 0.0, 0.10, Surf::Meadow, Surf::Soil, 3, Strata::Rock, 0.04, Species::Broadleaf, 0.18, Petals::Mixed, 0.70, 0.64, Z),
    row("broadleaf forest", [46, 118, 48], 1.02, 0.9, 0.0, 0.0, 0.0, 0.02, Surf::Grass, Surf::Soil, 4, Strata::Rock, 0.78, Species::Broadleaf, 0.02, Petals::White, 0.60, 0.74, Z),
    row("giant-tree forest", [28, 96, 42], 1.05, 1.05, 4.0, 0.0, 0.0, 0.0, Surf::Grass, Surf::Soil, 5, Strata::Rock, 0.6, Species::Broadleaf, 0.01, Petals::Cool, 0.64, 0.82, feat(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    row("autumn wood", [176, 96, 32], 1.0, 0.85, 0.0, 0.0, 0.0, 0.04, Surf::Grass, Surf::Soil, 4, Strata::Rock, 0.7, Species::Autumn, 0.02, Petals::Warm, 0.52, 0.56, Z),
    row("blossom grove", [206, 140, 170], 0.98, 0.6, 0.0, 0.0, 0.0, 0.08, Surf::Meadow, Surf::Soil, 4, Strata::Rock, 0.48, Species::Blossom, 0.09, Petals::Warm, 0.72, 0.60, Z),
    row("taiga", [58, 96, 78], 1.15, 0.7, 6.0, 0.05, 0.0, 0.08, Surf::Lichen, Surf::Soil, 3, Strata::Rock, 0.55, Species::Conifer, 0.008, Petals::White, 0.36, 0.58, Z),
    row("alpine range", [128, 132, 140], 1.85, 1.3, 26.0, 0.12, 0.0, 0.0, Surf::Gravel, Surf::Gravel, 2, Strata::Rock, 0.2, Species::Conifer, 0.0, Petals::None, 0.28, 0.40, feat(0.55, 0.0, 0.15, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.25)),
    row("glacier", [214, 228, 236], 1.05, 0.32, 14.0, 0.0, 0.0, 0.62, Surf::Snow, Surf::Ice, 3, Strata::Ice, 0.0, Species::None, 0.0, Petals::None, 0.10, 0.42, feat(0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    row("canyonlands", [168, 84, 52], 1.55, 0.42, -14.0, 0.28, 0.0, 0.0, Surf::Redsand, Surf::Redsand, 2, Strata::Sandstone, 0.02, Species::Broadleaf, 0.0, Petals::None, 0.58, 0.18, feat(0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9)),
    row("mesa steppe", [186, 102, 64], 0.72, 0.28, 8.0, 0.92, 2.0, 0.05, Surf::Redsand, Surf::Redsand, 2, Strata::Sandstone, 0.0, Species::None, 0.0, Petals::None, 0.66, 0.20, feat(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45)),
    row("dune sea", [214, 186, 120], 0.42, 0.22, -4.0, 0.0, 28.0, 0.18, Surf::Sand, Surf::Sand, 6, Strata::Sandstone, 0.0, Species::None, 0.0, Petals::None, 0.76, 0.10, Z),
    row("badlands", [150, 72, 48], 1.2, 0.85, 2.0, 0.58, 3.0, 0.0, Surf::Clay, Surf::Redsand, 2, Strata::Sandstone, 0.0, Species::None, 0.0, Petals::None, 0.68, 0.16, feat(0.25, 0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.0, 0.0, 0.95)),
    row("salt flat", [226, 224, 214], 0.32, 0.12, -8.0, 0.0, 1.2, 0.9, Surf::Salt, Surf::Salt, 3, Strata::Limestone, 0.0, Species::None, 0.0, Petals::None, 0.64, 0.08, Z),
    row("karst stone forest", [168, 176, 160], 1.35, 1.15, 10.0, 0.08, 0.0, 0.0, Surf::Limestone, Surf::Limestone, 2, Strata::Limestone, 0.04, Species::Broadleaf, 0.0, Petals::None, 0.50, 0.30, feat(1.0, 0.0, 0.1, 0.0, 0.0, 0.15, 0.25, 0.0, 0.0, 0.35)),
    row("volcanic field", [72, 48, 46], 1.4, 1.0, 6.0, 0.18, 0.0, 0.0, Surf::Basalt, Surf::Obsidian, 3, Strata::Basalt, 0.0, Species::None, 0.0, Petals::None, 0.80, 0.16, feat(0.15, 1.0, 0.05, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0)),
    row("ash waste", [96, 92, 88], 0.5, 0.28, -6.0, 0.0, 5.0, 0.74, Surf::Ash, Surf::Ash, 4, Strata::Ash, 0.0, Species::None, 0.0, Petals::None, 0.72, 0.12, feat(0.0, 0.4, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0)),
    row("crystal field", [150, 130, 210], 1.15, 0.75, 8.0, 0.0, 0.0, 0.1, Surf::Marble, Surf::Marble, 2, Strata::Crystal, 0.0, Species::None, 0.0, Petals::None, 0.24, 0.28, feat(0.6, 0.0, 1.0, 0.0, 0.15, 0.0, 0.1, 0.0, 0.0, 0.0)),
    row("fungal forest", [120, 78, 150], 0.85, 0.8, 0.0, 0.0, 0.0, 0.06, Surf::Moss, Surf::Mud, 4, Strata::Rock, 0.0, Species::None, 0.0, Petals::None, 0.58, 0.84, feat(0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.0)),
    row("glow-moss hollow", [70, 150, 120], 0.55, 0.4, -10.0, 0.0, 0.0, 0.4, Surf::Moss, Surf::Mud, 4, Strata::Rock, 0.0, Species::None, 0.04, Petals::Cool, 0.50, 0.76, feat(0.0, 0.0, 0.4, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    row("bone lands", [210, 204, 184], 0.9, 0.65, 2.0, 0.1, 0.0, 0.22, Surf::Bone, Surf::Limestone, 2, Strata::Bone, 0.0, Species::None, 0.0, Petals::None, 0.44, 0.34, feat(0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0, 1.0, 0.15)),
    row("sky-island archipelago", [120, 168, 196], 1.65, 1.2, 34.0, 0.0, 0.0, 0.0, Surf::Grass, Surf::Soil, 3, Strata::Rock, 0.28, Species::Broadleaf, 0.03, Petals::Cool, 0.48, 0.55, feat(0.25, 0.0, 0.1, 0.0, 1.0, 0.0, 0.15, 0.3, 0.0, 0.0)),
    row("crater field", [120, 110, 100], 0.8, 0.5, 0.0, 0.0, 0.0, 0.3, Surf::Regolith, Surf::Gravel, 3, Strata::Rock, 0.015, Species::Broadleaf, 0.0, Petals::None, 0.48, 0.22, feat(0.0, 0.25, 0.1, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0)),
    row("petrified forest", [140, 122, 100], 0.85, 0.55, 2.0, 0.16, 0.0, 0.12, Surf::Petrified, Surf::Clay, 3, Strata::Sandstone, 0.0, Species::None, 0.0, Petals::None, 0.46, 0.28, feat(0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.75, 0.15, 0.0)),
    row("terraced hills", [112, 150, 72], 1.0, 0.55, 4.0, 0.8, 0.0, 0.06, Surf::Grass, Surf::Soil, 3, Strata::Rock, 0.14, Species::Broadleaf, 0.03, Petals::Warm, 0.58, 0.42, feat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.2)),
    row("tundra", [150, 158, 140], 0.7, 0.42, 2.0, 0.0, 0.0, 0.42, Surf::Tundra, Surf::Gravel, 2, Strata::Rock, 0.06, Species::Conifer, 0.01, Petals::White, 0.22, 0.40, Z),
];

struct Listed {
    theme: ThemeId,
    weight: f32,
}

const fn listed(theme: ThemeId, weight: f32) -> Listed {
    Listed { theme, weight }
}

const GREEN: &[Listed] = &[
    listed(ThemeId::Meadow, 1.5),
    listed(ThemeId::Flower, 1.05),
    listed(ThemeId::Broadleaf, 1.3),
    listed(ThemeId::Giant, 0.65),
    listed(ThemeId::Autumn, 0.9),
    listed(ThemeId::Blossom, 0.8),
    listed(ThemeId::Taiga, 0.85),
    listed(ThemeId::Alpine, 0.75),
    listed(ThemeId::Terraced, 0.7),
    listed(ThemeId::Tundra, 0.55),
    listed(ThemeId::Canyon, 0.3),
    listed(ThemeId::Mesa, 0.28),
    listed(ThemeId::Glacier, 0.32),
];

const ASHEN: &[Listed] = &[
    listed(ThemeId::Volcanic, 1.6),
    listed(ThemeId::Ash, 1.35),
    listed(ThemeId::Badlands, 0.75),
    listed(ThemeId::Mesa, 0.45),
    listed(ThemeId::Crater, 0.7),
    listed(ThemeId::Canyon, 0.5),
    listed(ThemeId::Petrified, 0.4),
    listed(ThemeId::Bone, 0.3),
    listed(ThemeId::Tundra, 0.28),
    listed(ThemeId::Glacier, 0.35),
];

const DUNE_SEA: &[Listed] = &[
    listed(ThemeId::Dune, 1.7),
    listed(ThemeId::Salt, 1.05),
    listed(ThemeId::Mesa, 1.15),
    listed(ThemeId::Canyon, 1.05),
    listed(ThemeId::Badlands, 0.95),
    listed(ThemeId::Terraced, 0.4),
    listed(ThemeId::Crater, 0.35),
    listed(ThemeId::Ash, 0.3),
    listed(ThemeId::Tundra, 0.22),
    listed(ThemeId::Glacier, 0.28),
];

const SHATTERED: &[Listed] = &[
    listed(ThemeId::Karst, 1.5),
    listed(ThemeId::Canyon, 0.95),
    listed(ThemeId::Crater, 1.05),
    listed(ThemeId::Petrified, 0.75),
    listed(ThemeId::Badlands, 0.6),
    listed(ThemeId::Mesa, 0.5),
    listed(ThemeId::Alpine, 0.55),
    listed(ThemeId::Bone, 0.5),
    listed(ThemeId::Islands, 0.4),
    listed(ThemeId::Terraced, 0.35),
    listed(ThemeId::Tundra, 0.35),
    listed(ThemeId::Glacier, 0.32),
];

const GLASS: &[Listed] = &[
    listed(ThemeId::Glacier, 1.55),
    listed(ThemeId::Crystal, 1.4),
    listed(ThemeId::Tundra, 1.0),
    listed(ThemeId::Alpine, 0.85),
    listed(ThemeId::Taiga, 0.45),
    listed(ThemeId::Crater, 0.4),
    listed(ThemeId::Karst, 0.4),
    listed(ThemeId::Salt, 0.25),
    listed(ThemeId::Islands, 0.28),
];

const FUNGAL: &[Listed] = &[
    listed(ThemeId::Fungal, 1.6),
    listed(ThemeId::GlowMoss, 1.25),
    listed(ThemeId::Bone, 0.85),
    listed(ThemeId::Giant, 0.6),
    listed(ThemeId::Flower, 0.4),
    listed(ThemeId::Broadleaf, 0.35),
    listed(ThemeId::Petrified, 0.4),
    listed(ThemeId::Islands, 0.32),
    listed(ThemeId::Ash, 0.25),
    listed(ThemeId::Tundra, 0.3),
    listed(ThemeId::Glacier, 0.28),
];

const LUSH: &[Listed] = &[
    listed(ThemeId::Broadleaf, 1.35),
    listed(ThemeId::Giant, 1.15),
    listed(ThemeId::Flower, 1.05),
    listed(ThemeId::Meadow, 1.0),
    listed(ThemeId::Blossom, 0.9),
    listed(ThemeId::Autumn, 0.6),
    listed(ThemeId::Fungal, 0.45),
    listed(ThemeId::Terraced, 0.45),
    listed(ThemeId::Taiga, 0.4),
    listed(ThemeId::Alpine, 0.4),
    listed(ThemeId::Islands, 0.35),
];

const CRYSTAL: &[Listed] = &[
    listed(ThemeId::Crystal, 1.7),
    listed(ThemeId::Glacier, 1.05),
    listed(ThemeId::Karst, 0.85),
    listed(ThemeId::Tundra, 0.7),
    listed(ThemeId::Alpine, 0.65),
    listed(ThemeId::Crater, 0.55),
    listed(ThemeId::Islands, 0.4),
    listed(ThemeId::Petrified, 0.35),
    listed(ThemeId::Salt, 0.35),
    listed(ThemeId::Mesa, 0.3),
    listed(ThemeId::Bone, 0.28),
];

fn table(realm: Realm) -> &'static [Listed] {
    match realm {
        Realm::Green => GREEN,
        Realm::Ashen => ASHEN,
        Realm::Dune => DUNE_SEA,
        Realm::Shattered => SHATTERED,
        Realm::Glass => GLASS,
        Realm::Fungal => FUNGAL,
        Realm::Lush => LUSH,
        Realm::Crystal => CRYSTAL,
    }
}

fn theme(id: ThemeId) -> &'static Theme {
    &THEMES[id as usize]
}

/// Map colour of a theme.
#[cfg(test)]
pub fn theme_rgb(id: ThemeId) -> [u8; 3] {
    THEMES[id as usize].rgb
}

/// Short name of a theme.
#[cfg(test)]
pub fn theme_name(id: ThemeId) -> &'static str {
    THEMES[id as usize].name
}

/// Geopotential altitude on a face: 0 at the centre, 1 at a corner.
/// Fit of the cube's surface potential, `0.5·(max(|u|,|v|)/H)² + 0.5·|u|·|v|/H²`.
pub fn geopotential(u: f64, v: f64, half: f64) -> f32 {
    let h = half.max(1.0);
    let au = (u.abs() / h) as f32;
    let av = (v.abs() / h) as f32;
    let m = au.max(av);
    0.5 * m * m + 0.5 * au * av
}

/// One column's climate and blended theme.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    /// Nearest province's theme (the map colour).
    pub theme: ThemeId,
    /// Discrete choice after the border dither.
    pub pick: ThemeId,
    pub temp: f32,
    /// Moisture. Later passes read it; the surface uses temperature.
    #[allow(dead_code)]
    pub moist: f32,
    /// Theme relief times the region's relief.
    pub relief: f32,
    pub hills: f32,
    pub base: f32,
    pub terrace: f32,
    pub dune: f32,
    pub flat: f32,
    pub trees: f32,
    pub flowers: f32,
    pub feats: [f32; FEAT],
    /// Nearest and second-nearest province ids.
    #[allow(dead_code)]
    pub province: [u32; 2],
    /// Weight of the nearest province: 0.5 on a border, 1 deep inside.
    #[allow(dead_code)]
    pub weight: f32,
}

/// The theme field of one face.
pub struct Provinces {
    seed: u32,
    variety: f32,
    realm: Realm,
    face: Face,
    half: i64,
    spawn: [(u32, ThemeId); 6],
    spawn_n: u8,
    /// The two regions under the spawn point. Both are kept off a low relief.
    spawn_region: u32,
    spawn_region_b: u32,
    lift: bool,
}

const SPAWN_THEMES: [ThemeId; 6] = [
    ThemeId::Meadow,
    ThemeId::Broadleaf,
    ThemeId::Flower,
    ThemeId::Autumn,
    ThemeId::Terraced,
    ThemeId::Blossom,
];

impl Provinces {
    /// `garden` forces the home +Y centre into a meadow with several themes inside 3 km.
    pub fn new(seed: u32, variety: f32, realm: Realm, face: Face, half: i64, garden: bool) -> Self {
        let (spawn, spawn_n, spawn_region, spawn_region_b) = if garden {
            let s = surface(face, half, 0, 0).0;
            let (spawn, n) = garden_ids(seed, face, s);
            let (a, b, _, _) = region_pair(seed, s);
            (spawn, n, a, b)
        } else {
            ([(0, ThemeId::Meadow); 6], 0, 0, 0)
        };
        Self {
            seed,
            variety,
            realm,
            face,
            half,
            spawn,
            spawn_n,
            spawn_region,
            spawn_region_b,
            lift: garden,
        }
    }

    #[cfg(test)]
    pub(super) fn realm(&self) -> Realm {
        self.realm
    }

    /// Climate and blended theme at face-local `(u, v)`.
    pub fn at(&self, u: i32, v: i32) -> Place {
        let (s, uc, vc) = surface(self.face, self.half, u, v);
        let g = geopotential(f64::from(uc), f64::from(vc), self.half as f64);
        let (rt, rm, region_relief) = self.region_at(s);
        let (bias_t, bias_m) = self.realm.bias();
        let n = fbm3(self.seed ^ 0x7E30_0001, scale(s, 9_000.0));
        // Warm basin, cold rim, frozen corner: g is 0 at the face centre and 1 at a corner.
        let temp = (0.66 - 0.92 * g + 0.16 * n + rt + bias_t).clamp(0.0, 1.0);
        let m = fbm3(self.seed ^ 0x3015_7002, scale(s, 7_400.0));
        let moist = (0.50 + 0.40 * m + rm + bias_m).clamp(0.0, 1.0);

        let (f1, f2, id1, id2) = province_pair(self.seed, s);
        let w = border_weight((f2 - f1).max(0.0) * PROVINCE as f32, BORDER);
        let t1 = self.choose(id1, temp, moist);
        let t2 = self.choose(id2, temp, moist);
        let a = theme(t1);
        let b = theme(t2);
        let mix = |p: f32, q: f32| p * w + q * (1.0 - w);
        let mut feats = [0.0; FEAT];
        for i in 0..FEAT {
            feats[i] = mix(a.feats[i], b.feats[i]);
        }
        let roll = unit(hash3(self.seed ^ 0xD174_E200, floor_i(s[0]), floor_i(s[1]), floor_i(s[2])));
        let pick = if roll < w { t1 } else { t2 };
        let species = theme(pick).species;
        Place {
            theme: t1,
            pick,
            temp,
            moist,
            relief: mix(a.relief, b.relief) * region_relief,
            hills: mix(a.hills, b.hills),
            base: mix(a.base, b.base),
            terrace: mix(a.terrace, b.terrace),
            dune: mix(a.dune, b.dune),
            flat: mix(a.flat, b.flat),
            trees: if species == Species::None { 0.0 } else { mix(a.trees, b.trees) },
            flowers: mix(a.flowers, b.flowers),
            feats,
            province: [id1, id2],
            weight: w,
        }
    }

    fn choose(&self, id: u32, temp: f32, moist: f32) -> ThemeId {
        for i in 0..self.spawn_n as usize {
            if self.spawn[i].0 == id {
                return self.spawn[i].1;
            }
        }
        let rows = table(self.realm);
        let variety = self.variety.clamp(0.0, 2.0);
        let floor = 0.25 + 0.35 * (variety * 0.5);
        let mut total = 0.0f32;
        let mut acc = [0.0f32; 16];
        for (i, row) in rows.iter().enumerate() {
            let th = theme(row.theme);
            let fit = (1.0 - (th.temp - temp).abs()).max(0.05) * (1.0 - (th.moist - moist).abs()).max(0.05);
            total += row.weight * (floor + (1.0 - floor) * fit);
            acc[i] = total;
        }
        if total <= 0.0 {
            return rows[0].theme;
        }
        let roll = unit(hash3(self.seed ^ 0x71E3_E000, id as i32, self.realm as i32, 0x7E3E)) * total;
        let idx = acc.iter().position(|&c| roll < c).unwrap_or(rows.len() - 1);
        rows[idx].theme
    }

    fn region_at(&self, s: [f64; 3]) -> (f32, f32, f32) {
        let (id1, id2, f1, f2) = region_pair(self.seed, s);
        let w = border_weight((f2 - f1).max(0.0) * REGION as f32, REGION_BORDER);
        let lift = |id: u32| self.lift && (id == self.spawn_region || id == self.spawn_region_b);
        let a = region_params(self.seed, id1, lift(id1));
        let b = region_params(self.seed, id2, lift(id2));
        (a.0 * w + b.0 * (1.0 - w), a.1 * w + b.1 * (1.0 - w), a.2 * w + b.2 * (1.0 - w))
    }
}

/// `(temp offset, moisture offset, relief multiplier)`.
fn region_params(seed: u32, id: u32, lift: bool) -> (f32, f32, f32) {
    let temp = (unit(hash3(seed ^ 0x51E0_0001, id as i32, 1, 0)) - 0.5) * 0.22;
    let moist = (unit(hash3(seed ^ 0x51E0_0002, id as i32, 2, 0)) - 0.5) * 0.30;
    let u = unit(hash3(seed ^ 0x51E0_0003, id as i32, 3, 0));
    let mut relief = if u < 0.5 { 0.4 + u * 1.2 } else { 1.0 + (u - 0.5) * 2.4 };
    if lift {
        relief = relief.max(1.0);
    }
    (temp, moist, relief)
}

fn province_pair(seed: u32, s: [f64; 3]) -> (f32, f32, u32, u32) {
    cellular3(seed ^ PROVINCE_SALT, lattice(s, PROVINCE))
}

fn region_pair(seed: u32, s: [f64; 3]) -> (u32, u32, f32, f32) {
    let (f1, f2, a, b) = cellular3(seed ^ REGION_SALT, lattice(s, REGION));
    (a, b, f1, f2)
}

/// Provinces whose cells meet the spawn disc, nearest first, each pinned to a temperate theme.
fn garden_ids(seed: u32, face: Face, s: [f64; 3]) -> ([(u32, ThemeId); 6], u8) {
    let mut found: Vec<(i64, u32)> = Vec::new();
    let mut note = |d2: i64, id: u32| {
        if let Some(slot) = found.iter_mut().find(|e| e.1 == id) {
            if d2 < slot.0 {
                slot.0 = d2;
            }
        } else {
            found.push((d2, id));
        }
    };
    note(0, province_pair(seed, s).2);
    for z in -20..=20 {
        for x in -20..=20 {
            let (dx, dz) = (x * 200, z * 200);
            let d2 = i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz);
            if d2 > 3_000 * 3_000 {
                continue;
            }
            let (ox, oy, oz) = FaceFrame::new(face).cell_to_world((dx, 0, dz));
            let p = [s[0] + f64::from(ox), s[1] + f64::from(oy), s[2] + f64::from(oz)];
            note(d2, province_pair(seed, p).2);
        }
    }
    found.sort_by_key(|e| (e.0, e.1));
    let mut out = [(0, ThemeId::Meadow); 6];
    let n = found.len().min(SPAWN_THEMES.len());
    for (i, e) in found.iter().take(n).enumerate() {
        out[i] = (e.1, SPAWN_THEMES[i]);
    }
    (out, n as u8)
}

/// `0.5` where the two distances are equal, `1` once `gap` reaches `border`.
fn border_weight(gap: f32, border: f32) -> f32 {
    let t = (gap / border).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    0.5 + 0.5 * s
}

fn lattice(p: [f64; 3], scale: f64) -> [f64; 3] {
    [(p[0] + SHIFT[0]) / scale, (p[1] + SHIFT[1]) / scale, (p[2] + SHIFT[2]) / scale]
}

fn scale(p: [f64; 3], k: f64) -> [f64; 3] {
    [p[0] / k, p[1] / k, p[2] / k]
}

fn floor_i(p: f64) -> i32 {
    p.floor() as i64 as i32
}

fn fbm3(seed: u32, p: [f64; 3]) -> f32 {
    let mut q = p;
    let (mut sum, mut amp, mut norm) = (0.0f32, 1.0f32, 0.0f32);
    for o in 0u32..3 {
        sum += amp * perlin3(seed.wrapping_add(o.wrapping_mul(0x632B_E5AB)), q[0], q[1], q[2]);
        norm += amp;
        amp *= 0.5;
        q = [q[0] * 2.0 + 19.0, q[1] * 2.0 - 7.0, q[2] * 2.0 + 11.0];
    }
    sum / norm
}

/// Body-space surface point, and the clamped face coordinates climate reads.
fn surface(face: Face, half: i64, u: i32, v: i32) -> ([f64; 3], i32, i32) {
    let uc = i64::from(u).clamp(-half, half) as i32;
    let vc = i64::from(v).clamp(-half, half) as i32;
    let (x, y, z) = FaceFrame::new(face).cell_to_world((uc, half as i32, vc));
    ([f64::from(x), f64::from(y), f64::from(z)], uc, vc)
}

/// Whether a surface can hold flowers.
pub fn grassy(surf: Surf) -> bool {
    matches!(surf, Surf::Grass | Surf::Meadow | Surf::Moss | Surf::Tundra | Surf::Lichen)
}

/// The resolved materials of a dithered theme.
pub struct Skin {
    pub surf: Surf,
    pub sub: Surf,
    pub depth: i32,
    pub strata: Strata,
    pub species: Species,
    pub petals: Petals,
}

pub fn skin(id: ThemeId) -> Skin {
    let t = theme(id);
    Skin { surf: t.surf, sub: t.sub, depth: t.depth, strata: t.strata, species: t.species, petals: t.petals }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geopotential_is_zero_at_centre_and_one_at_a_corner() {
        assert!(geopotential(0.0, 0.0, 100.0).abs() < 1e-6);
        assert!((geopotential(100.0, 0.0, 100.0) - 0.5).abs() < 1e-5);
        assert!((geopotential(-100.0, 100.0, 100.0) - 1.0).abs() < 1e-5);
        assert!((geopotential(40.0, -70.0, 100.0) - geopotential(-40.0, 70.0, 100.0)).abs() < 1e-6);
    }

    #[test]
    fn every_theme_is_offered_by_a_realm_and_named() {
        assert_eq!(THEMES.len(), THEME_COUNT);
        assert_eq!(FEATS.len(), FEAT);
        assert_eq!(ThemeId::Tundra as usize, THEME_COUNT - 1);
        let mut seen = [false; THEME_COUNT];
        for realm in [Realm::Green, Realm::Ashen, Realm::Dune, Realm::Shattered, Realm::Glass, Realm::Fungal, Realm::Lush, Realm::Crystal] {
            let rows = table(realm);
            assert!(!rows.is_empty());
            assert!(rows.len() <= 16, "{realm:?} table does not fit the picker");
            for row in rows {
                assert!(row.weight > 0.0);
                seen[row.theme as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s), "a theme is in no realm");
        for (i, t) in THEMES.iter().enumerate() {
            assert!(!t.name.is_empty());
            assert_eq!(theme_name(ThemeId::Meadow), "meadow plains");
            let _ = theme_rgb(unsafe_id(i));
        }
    }

    fn unsafe_id(i: usize) -> ThemeId {
        match i {
            0 => ThemeId::Meadow,
            1 => ThemeId::Flower,
            2 => ThemeId::Broadleaf,
            3 => ThemeId::Giant,
            4 => ThemeId::Autumn,
            5 => ThemeId::Blossom,
            6 => ThemeId::Taiga,
            7 => ThemeId::Alpine,
            8 => ThemeId::Glacier,
            9 => ThemeId::Canyon,
            10 => ThemeId::Mesa,
            11 => ThemeId::Dune,
            12 => ThemeId::Badlands,
            13 => ThemeId::Salt,
            14 => ThemeId::Karst,
            15 => ThemeId::Volcanic,
            16 => ThemeId::Ash,
            17 => ThemeId::Crystal,
            18 => ThemeId::Fungal,
            19 => ThemeId::GlowMoss,
            20 => ThemeId::Bone,
            21 => ThemeId::Islands,
            22 => ThemeId::Crater,
            23 => ThemeId::Petrified,
            24 => ThemeId::Terraced,
            _ => ThemeId::Tundra,
        }
    }

    #[test]
    fn the_same_point_is_the_same_province_twice() {
        let p = Provinces::new(42, 1.0, Realm::Green, Face::PosY, 25_000_000, true);
        let a = p.at(123, -456);
        let b = p.at(123, -456);
        assert_eq!(a.theme, b.theme);
        assert_eq!(a.province, b.province);
        assert_eq!(a.relief.to_bits(), b.relief.to_bits());
        assert_eq!(a.temp.to_bits(), b.temp.to_bits());
        assert!((0.5..=1.0).contains(&a.weight));
    }

    #[test]
    fn provinces_match_across_an_edge_and_a_corner() {
        let half = 25_000_000i64;
        let h = half as i32;
        let y = Provinces::new(7, 1.0, Realm::Green, Face::PosY, half, false);
        let x = Provinces::new(7, 1.0, Realm::Dune, Face::PosX, half, false);
        let z = Provinces::new(7, 1.0, Realm::Glass, Face::PosZ, half, false);
        for v in [0, 10_000, -80_000, 1_000_000, h + 50] {
            let a = y.at(h, v);
            let b = x.at(-h, v);
            assert_eq!(a.province, b.province, "edge v={v}");
            assert!((a.weight - b.weight).abs() < 1e-3, "weight v={v}");
        }
        let a = y.at(h, h);
        let b = x.at(-h, h);
        let c = z.at(h, -h);
        assert_eq!(a.province, b.province, "corner +X");
        assert_eq!(a.province, c.province, "corner +Z");
    }

    #[test]
    fn a_province_border_does_not_step() {
        let p = Provinces::new(5, 1.0, Realm::Ashen, Face::NegY, 25_000_000, false);
        let mut crossed = 0;
        for v in [0, 2_400, -6_000, 18_000] {
            let mut prev = p.at(0, v);
            for u in (200..48_000).step_by(200) {
                let here = p.at(u, v);
                if here.province[0] != prev.province[0] {
                    // Refine to the block where the nearest id changes.
                    let mut lo = u - 200;
                    let mut hi = u;
                    while hi - lo > 1 {
                        let mid = lo + (hi - lo) / 2;
                        if p.at(mid, v).province[0] == prev.province[0] {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let left = p.at(lo, v);
                    let right = p.at(hi, v);
                    assert!((left.relief - right.relief).abs() < 0.08, "relief jump at ({hi},{v})");
                    assert!((left.hills - right.hills).abs() < 0.08);
                    assert!((left.base - right.base).abs() < 1.0);
                    assert!((left.dune - right.dune).abs() < 1.5);
                    assert!((left.flat - right.flat).abs() < 0.08);
                    crossed += 1;
                    break;
                }
                prev = here;
            }
        }
        assert!(crossed >= 2, "only {crossed} borders in the walk");
    }

    #[test]
    fn spawn_is_a_meadow_with_three_themes_nearby() {
        for seed in [1u32, 2, 7, 42, 99, 256, 1_000, 99_991] {
            let p = Provinces::new(seed, 1.0, Realm::Green, Face::PosY, 25_000_000, true);
            let c = p.at(0, 0);
            assert_eq!(c.theme, ThemeId::Meadow, "seed {seed} theme {:?}", c.theme);
            assert!(c.temp > 0.35 && c.temp < 0.98, "seed {seed} temp {}", c.temp);
            let mut themes = Vec::new();
            for z in -15..=15 {
                for x in -15..=15 {
                    let (dx, dz) = (x * 200, z * 200);
                    if i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz) > 3_000 * 3_000 {
                        continue;
                    }
                    let t = p.at(dx, dz).theme;
                    if !themes.contains(&t) {
                        themes.push(t);
                    }
                }
            }
            assert!(themes.len() >= 3, "seed {seed}: {themes:?}");
        }
    }
}
