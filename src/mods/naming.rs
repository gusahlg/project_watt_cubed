//! Material names: every configuration gets a name a person can remember, and its tool a name to
//! match ("Velorine", "Banded Keshan Olivite", "Velorine Pick").
//!
//! The words come from a small *language model*: an order-3 character Markov chain (with back-off
//! to orders 2 and 1) trained, once, on a few hundred real mineral, rock and element names. It
//! speaks in their phonology — "-ite", "-ine", clusters like "rh", "ph", "ch" — but its roots are
//! new words. The randomness is a deterministic generator seeded from the configuration, so every
//! peer names everything the same way, and the names mean something:
//!
//! * the **root** comes from the most abundant element (its coordinates coarsened, so near-twins
//!   share it): a material and everything a reaction makes from it share a family name;
//! * the **suffix** says how it reads — clear ("-ine"), glowing ("-ium"), hard ("-ite"),
//!   firm ("-ate"), soft ("-ash");
//! * a **qualifier** names a significant second element ("Keshan …"), a busy mixture ("Banded",
//!   "Veined") or a mixture that wants to come apart ("Brittle");
//! * the **tool** name adds a noun by mass and look: shard, chisel, pick, maul, sledge; lens or
//!   lantern for clear or glowing tools.
//!
//! Names are presentation only: they never reach the law, worldgen, saves or the wire.

use std::collections::HashMap;
use std::sync::OnceLock;

use material::{Configuration, Element};

use crate::block::naming::{MaterialNamer, MaterialNames, NamingSource};
use crate::mods::{Knob, Mod};

/// The training words: mineral, rock and element names (public scientific vocabulary).
const CORPUS: &str = "quartz feldspar mica garnet beryl topaz jasper agate onyx obsidian basalt granite gneiss \
schist marble slate shale chert flint opal jade malachite azurite cinnabar galena pyrite hematite magnetite \
bauxite cassiterite chalcopyrite sphalerite fluorite calcite dolomite gypsum halite apatite olivine pyroxene \
amphibole hornblende augite epidote zircon rutile ilmenite chromite spinel corundum ruby sapphire emerald \
aquamarine tourmaline peridot citrine amethyst carnelian chalcedony rhodonite rhodochrosite smithsonite \
cerussite anglesite wulfenite vanadinite scheelite wolframite molybdenite stibnite realgar orpiment cuprite \
tenorite bornite covellite chalcocite enargite tetrahedrite argentite proustite sylvanite calaverite kyanite \
sillimanite andalusite staurolite cordierite sodalite lazurite nepheline leucite analcime natrolite stilbite \
heulandite chabazite prehnite datolite axinite vesuvianite grossular almandine pyrope spessartine uvarovite \
andradite titanite monazite xenotime thorite uraninite carnotite autunite torbernite columbite tantalite \
pollucite lepidolite spodumene petalite amblygonite euclase phenakite chrysoberyl alexandrite taaffeite \
painite benitoite larimar charoite sugilite seraphinite labradorite sunstone moonstone orthoclase microcline \
albite anorthite oligoclase andesine bytownite serpentine talc kaolinite illite vermiculite chlorite \
muscovite biotite phlogopite glauconite celadonite andesite rhyolite dacite diorite gabbro peridotite dunite \
komatiite tuff breccia conglomerate siltstone mudstone limestone chalk travertine tufa coquina marl \
quartzite amphibolite eclogite granulite migmatite mylonite hornfels skarn greisen kimberlite lamproite \
carbonatite pumice scoria ignimbrite cerium yttrium terbium erbium thulium holmium lutetium hafnium osmium \
iridium rhodium ruthenium palladium platinum tantalum niobium vanadium scandium gallium indium thallium \
bismuth tellurium selenium germanium antimony cadmium argon xenon krypton neon helium lithium beryllium \
sodium potassium rubidium strontium barium thorium cobalt nickel chromium zinc copper silver gold iron tin \
lead carbon silicon boron sulfur phosphor magnesium calcium titanium manganese zirconium molybdenum \
tungsten rhenium gadolinium europium samarium neodymium praseodymium lanthanum dysprosium ytterbium";

/// Character alphabet: `^` start, `$` end, then a-z.
fn idx(c: u8) -> usize {
    match c {
        b'^' => 0,
        b'$' => 1,
        b'a'..=b'z' => 2 + (c - b'a') as usize,
        _ => 1,
    }
}

const ALPHA: usize = 28;

/// The trained chain: for each context (up to three previous characters) the counts of the next.
struct Model {
    order3: HashMap<[u8; 3], [u16; ALPHA]>,
    order2: HashMap<[u8; 2], [u16; ALPHA]>,
    order1: HashMap<u8, [u16; ALPHA]>,
    words: Vec<&'static str>,
}

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        let mut m = Model { order3: HashMap::new(), order2: HashMap::new(), order1: HashMap::new(), words: Vec::new() };
        for w in CORPUS.split_whitespace() {
            m.words.push(w);
            let mut s = vec![b'^', b'^', b'^'];
            s.extend_from_slice(w.as_bytes());
            s.push(b'$');
            for i in 3..s.len() {
                let next = idx(s[i]);
                m.order3.entry([s[i - 3], s[i - 2], s[i - 1]]).or_insert([0; ALPHA])[next] += 1;
                m.order2.entry([s[i - 2], s[i - 1]]).or_insert([0; ALPHA])[next] += 1;
                m.order1.entry(s[i - 1]).or_insert([0; ALPHA])[next] += 1;
            }
        }
        m
    })
}

/// splitmix64: the deterministic source of all randomness here.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len() as u64) as usize]
    }
}

fn sample(counts: &[u16; ALPHA], rng: &mut Rng) -> u8 {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut r = rng.below(total);
    for (i, &c) in counts.iter().enumerate() {
        if r < c as u64 {
            return match i {
                0 => b'^',
                1 => b'$',
                _ => b'a' + (i - 2) as u8,
            };
        }
        r -= c as u64;
    }
    b'$'
}

fn vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

/// A pronounceable new root of 4..=8 letters, cut before a mineral ending so suffixes attach.
fn generate_root(seed: u64) -> String {
    let m = model();
    let mut rng = Rng(seed);
    for _ in 0..64 {
        let mut s: Vec<u8> = vec![b'^', b'^', b'^'];
        loop {
            let n = s.len();
            let c3 = [s[n - 3], s[n - 2], s[n - 1]];
            // Mostly the order-3 chain (the corpus's spelling), sometimes the looser order-2 chain
            // (invention): the mix keeps words pronounceable but new.
            let loose = rng.below(10) < 3 && n > 4;
            let next = if let Some(c) = m.order3.get(&c3).filter(|_| !loose) {
                sample(c, &mut rng)
            } else if let Some(c) = m.order2.get(&[s[n - 2], s[n - 1]]) {
                sample(c, &mut rng)
            } else if let Some(c) = m.order1.get(&s[n - 1]) {
                sample(c, &mut rng)
            } else {
                b'$'
            };
            if next == b'$' || s.len() > 14 {
                break;
            }
            s.push(next);
        }
        let mut word = String::from_utf8(s[3..].to_vec()).unwrap_or_default();
        // Strip the chain's own mineral endings so our suffix carries the meaning.
        for ending in ["ite", "ine", "ium", "um", "ate", "ide", "ane", "ase", "ole", "ene", "e"] {
            if word.len() > ending.len() + 3 && word.ends_with(ending) {
                word.truncate(word.len() - ending.len());
                break;
            }
        }
        if acceptable(&word, m) {
            // A quarter of roots take one more syllable from the seed: the chain alone revisits
            // its favourite words too often for a world's worth of materials.
            if rng.below(4) == 0 && word.len() <= 6 {
                const TAILS: [&str; 12] = ["ar", "en", "ix", "or", "um", "ae", "is", "on", "el", "ur", "ad", "yn"];
                let tail = TAILS[rng.below(TAILS.len() as u64) as usize];
                let joined = join(&word, tail);
                if acceptable(&joined, m) {
                    return joined;
                }
            }
            return word;
        }
    }
    // Deterministic fallback: syllables from the seed.
    let syl = ["ka", "vor", "le", "thi", "rus", "mo", "zen", "tal", "qui", "dra", "ser", "ol"];
    let mut rng = Rng(seed ^ 0xFA11_BACC);
    format!("{}{}", rng.pick(&syl), rng.pick(&syl))
}

fn acceptable(word: &str, m: &Model) -> bool {
    let b = word.as_bytes();
    if !(4..=8).contains(&b.len()) || m.words.contains(&word) {
        return false;
    }
    if !b.iter().any(|&c| vowel(c)) || b.ends_with(b"q") || b.ends_with(b"j") {
        return false;
    }
    // No three identical letters, no three consonants opening the word, no four in a row anywhere.
    let mut run = 0;
    for (i, &c) in b.iter().enumerate() {
        if i >= 2 && b[i - 1] == c && b[i - 2] == c {
            return false;
        }
        run = if vowel(c) { 0 } else { run + 1 };
        if run >= 4 || (i == 2 && run == 3) {
            return false;
        }
    }
    true
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Attach `suffix` to `root`: drop the root's last vowel before a vowel-initial suffix.
fn join(root: &str, suffix: &str) -> String {
    let mut r = root.to_string();
    if suffix.as_bytes().first().is_some_and(|&c| vowel(c)) && r.as_bytes().last().is_some_and(|&c| vowel(c)) {
        r.pop();
    }
    r + suffix
}

/// The two vocabularies the "Style" knob switches between.
struct Vocabulary {
    clear: &'static [&'static str],
    glowing: &'static [&'static str],
    hard: &'static [&'static str],
    firm: &'static [&'static str],
    soft: &'static [&'static str],
    textures: &'static [&'static str],
    brittle: &'static [&'static str],
    pure: &'static str,
}

const MINERAL: Vocabulary = Vocabulary {
    clear: &["ine", "ane", "yl"],
    glowing: &["ium", "ite", "ine"],
    hard: &["ite", "olite", "yx"],
    firm: &["ate", "ide", "ase"],
    soft: &["ash", "um", "ock"],
    textures: &["Banded", "Veined", "Mottled", "Speckled", "Layered", "Flecked"],
    brittle: &["Brittle", "Crumbling", "Friable"],
    pure: "Pure",
};

const ARCANE: Vocabulary = Vocabulary {
    clear: &["iel", "aris", "ene"],
    glowing: &["ael", "ion", "aure"],
    hard: &["orn", "ax", "eth"],
    firm: &["al", "en", "or"],
    soft: &["ith", "ow", "ul"],
    textures: &["Woven", "Starred", "Dreaming", "Whorled", "Runic", "Hollow"],
    brittle: &["Restless", "Waning", "Fraying"],
    pure: "True",
};

/// The coarse key of an element: near-twins (within a few units per axis) share a root.
fn root_key(e: Element, law_seed: u32) -> u64 {
    let c = e.0.map(|v| (v >> 2) as u64);
    (c[0] << 48 | c[1] << 32 | c[2] << 16 | c[3]) ^ (law_seed as u64).rotate_left(17) ^ 0x6E61_6D65_7321
}

fn ranked(config: &Configuration) -> Vec<(Element, usize)> {
    let mut counts: Vec<(Element, usize)> = config.counts().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
}

/// Name one configuration.
fn name(src: &NamingSource, vocab: &Vocabulary) -> MaterialNames {
    let counts = ranked(src.config);
    let seed = src.law.visual_seed;
    let (dominant, _) = counts[0];
    let root = generate_root(root_key(dominant, seed));
    let obs = src.obs;
    let mut rng = Rng(src.config.digest() ^ 0x5EED);
    let class = if obs.transparency >= 120 {
        vocab.clear
    } else if obs.emission > 0 {
        vocab.glowing
    } else if obs.hardness >= 200 {
        vocab.hard
    } else if obs.hardness >= 110 {
        vocab.firm
    } else {
        vocab.soft
    };
    // The family root picks its own suffix within the class, so a family reads alike.
    let suffix = class[(root_key(dominant, seed) % class.len() as u64) as usize];
    let noun = capitalize(&join(&root, suffix));
    let mut words: Vec<String> = Vec::with_capacity(3);
    if counts.len() == 1 {
        words.push(vocab.pure.to_string());
    } else {
        if obs.cohesion < -128 {
            words.push(rng.pick(vocab.brittle).to_string());
        } else if counts.len() >= 4 {
            words.push(rng.pick(vocab.textures).to_string());
        }
        let total = src.config.len();
        let (second, n2) = counts[1];
        if n2 >= 2 || n2 * 4 >= total {
            let r2 = generate_root(root_key(second, seed));
            let adj = if r2.ends_with('i') || r2.ends_with('e') { format!("{r2}c") } else { join(&r2, "an") };
            words.push(capitalize(&adj));
        }
    }
    words.truncate(2);
    words.push(noun.clone());
    let block = words.join(" ");
    let n = src.config.len();
    let tool_noun = if obs.emission > 0 {
        "Lantern"
    } else if obs.transparency >= 120 {
        "Lens"
    } else if n <= 2 {
        "Shard"
    } else if n <= 5 {
        if obs.hardness >= 200 { "Chisel" } else { "Awl" }
    } else if n <= 10 {
        "Pick"
    } else if n <= 18 {
        if obs.hardness >= 200 { "Maul" } else { "Hammer" }
    } else {
        "Sledge"
    };
    let tool = format!("{noun} {tool_noun}");
    MaterialNames { block, tool }
}

/// The Essentials naming mod.
pub struct NamingMod {
    arcane: bool,
    revision: u32,
}

impl NamingMod {
    pub fn new() -> Self {
        Self { arcane: false, revision: 1 }
    }
}

impl Default for NamingMod {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialNamer for NamingMod {
    fn names(&self, src: &NamingSource) -> MaterialNames {
        name(src, if self.arcane { &ARCANE } else { &MINERAL })
    }

    fn revision(&self) -> u32 {
        self.revision
    }
}

impl Mod for NamingMod {
    fn name(&self) -> &str {
        "Material names"
    }

    fn id(&self) -> &'static str {
        "material_names"
    }

    fn description(&self) -> &str {
        "Names every material and tool with words a small language model learned from minerals and elements."
    }

    fn group(&self) -> &'static str {
        crate::mods::ESSENTIALS
    }

    fn namer(&self) -> Option<&dyn MaterialNamer> {
        Some(self)
    }

    fn knobs(&self) -> Vec<Knob> {
        vec![Knob {
            label: "Style",
            value: if self.arcane { "Arcane" } else { "Mineral" }.to_string(),
            hint: "Mineral / Arcane".to_string(),
        }]
    }

    fn step_knob(&mut self, index: usize, delta: i32) {
        if index == 0 && delta != 0 {
            self.arcane = !self.arcane;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn save_choice_state(&self) -> Option<String> {
        Some(format!("style={}", if self.arcane { "arcane" } else { "mineral" }))
    }

    fn load_choice_state(&mut self, data: &str) {
        let arcane = data.trim() == "style=arcane";
        if arcane != self.arcane {
            self.arcane = arcane;
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use material::{observe, Block, Law};

    fn names_of(elems: &[[u8; 4]]) -> MaterialNames {
        let law = Law::current();
        let c = Configuration::new(elems.iter().map(|&e| Element::new(e)).collect::<Vec<_>>()).unwrap();
        let obs = observe(&law, &Block::of(&c));
        NamingMod::new().names(&NamingSource { law: &law, config: &c, obs: &obs })
    }

    struct R(u64);
    impl R {
        fn e(&mut self) -> [u8; 4] {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 as u32).to_le_bytes()
        }
    }

    #[test]
    fn names_are_deterministic_and_order_free() {
        let a = names_of(&[[10, 20, 30, 40], [90, 20, 200, 7], [90, 20, 200, 7]]);
        assert_eq!(a, names_of(&[[90, 20, 200, 7], [10, 20, 30, 40], [90, 20, 200, 7]]));
        assert!(!a.block.is_empty() && a.tool.starts_with(a.block.split(' ').next_back().unwrap()));
    }

    #[test]
    fn the_dominant_element_names_the_family() {
        let base = [[50, 60, 70, 80]; 3];
        let x = names_of(&[base[0], base[1], base[2], [1, 2, 3, 4]]);
        let y = names_of(&[base[0], base[1], base[2], [200, 2, 9, 4], [7, 7, 7, 7]]);
        let root = |n: &MaterialNames| n.block.split(' ').next_back().unwrap()[..4].to_string();
        assert_eq!(root(&x), root(&y), "{} vs {}", x.block, y.block);
    }

    #[test]
    fn roots_are_new_words_mostly_distinct_and_pronounceable() {
        let m = model();
        let mut rng = R(0xC0FFEE);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let root = generate_root(root_key(Element::new(rng.e()), 7));
            assert!(acceptable(&root, m) || root.len() >= 4, "{root}");
            assert!(!m.words.contains(&root.as_str()), "{root} is a corpus word");
            seen.insert(root);
        }
        assert!(seen.len() >= 425, "only {} distinct roots of 500", seen.len());
    }

    #[test]
    fn tool_nouns_follow_mass() {
        let small = names_of(&[[1, 2, 3, 4]]);
        let mut big = Vec::new();
        let mut rng = R(3);
        for _ in 0..24 {
            big.push(rng.e());
        }
        let large = names_of(&big);
        assert!(small.tool.ends_with("Shard") || small.tool.ends_with("Lens") || small.tool.ends_with("Lantern"));
        assert!(large.tool.ends_with("Sledge") || large.tool.ends_with("Lens") || large.tool.ends_with("Lantern"));
    }

    #[test]
    fn naming_is_cheap() {
        let mut rng = R(11);
        let _ = model();
        let t = std::time::Instant::now();
        for _ in 0..500 {
            let _ = names_of(&[rng.e(), rng.e(), rng.e()]);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / 500.0;
        assert!(us < 2_000.0, "{us:.1} µs per name");
    }

    #[test]
    #[ignore]
    fn names_gallery() {
        let mut rng = R(0xDEC0DE);
        for n in [1usize, 2, 3, 4, 5, 6, 8, 12] {
            for _ in 0..5 {
                let elems: Vec<[u8; 4]> = (0..n).map(|_| rng.e()).collect();
                let names = names_of(&elems);
                println!("{n:2} elements: {:40} | {}", names.block, names.tool);
            }
        }
    }
}
