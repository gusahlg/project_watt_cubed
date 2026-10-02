//! Material naming seam: a configuration becomes words a person can hold on to.
//!
//! The simulation has no names. A naming mod ([`MaterialNamer`]) maps each interned configuration to a
//! block name and a tool name once, when the configuration first appears; the registry keeps them
//! beside the visuals as presentation. Names never flow back into the law, worldgen, saves or the
//! wire. With no naming mod enabled the core falls back to [`describe`]: words read off the
//! observation ("glowing clear hard solid").

use material::{Configuration, Law, Observation};

/// Everything a namer may read about one configuration.
pub struct NamingSource<'a> {
    /// The world's law (for element colours, probes).
    pub law: &'a Law,
    /// The canonical configuration (sorted multiset).
    pub config: &'a Configuration,
    /// Its readings.
    pub obs: &'a Observation,
}

/// The two names of one configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialNames {
    /// What the block is called ("Vorthite", "Banded Keshan Olivine").
    pub block: String,
    /// What the same configuration is called when held as a tool ("Vorthite Pick").
    pub tool: String,
}

/// Turns a configuration into names. Must be a pure function of the source (and the namer's own
/// settings): two clients naming the same configuration must agree.
pub trait MaterialNamer {
    /// Name one configuration. Never called for the void.
    fn names(&self, src: &NamingSource) -> MaterialNames;
    /// Bump when settings change so every name is recomputed.
    fn revision(&self) -> u32;
}

/// Words for an observation: glow, clarity, hardness band, grip — each a threshold on a reading, so
/// two configurations that read alike are described alike. The void is "air".
pub fn describe(obs: &Observation) -> String {
    if !obs.solid {
        return "air".to_string();
    }
    let mut words: Vec<&str> = Vec::with_capacity(5);
    if obs.emission > 0 {
        words.push("glowing");
    }
    match obs.transparency {
        200..=255 => words.push("clear"),
        100..=199 => words.push("hazy"),
        _ => {}
    }
    words.push(match obs.hardness {
        200..=255 => "hard",
        100..=199 => "firm",
        _ => "soft",
    });
    match obs.friction {
        192..=255 => words.push("rough"),
        0..=63 => words.push("slick"),
        _ => {}
    }
    words.push("solid");
    words.join(" ")
}

/// Core fallback names: the description, and "<description> tool".
pub fn fallback_names(obs: &Observation) -> MaterialNames {
    let block = describe(obs);
    let tool = format!("{block} tool");
    MaterialNames { block, tool }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_reads_the_observation() {
        assert_eq!(describe(&Observation::AIR), "air");
        let mut obs = Observation::AIR;
        obs.solid = true;
        obs.transparency = 0;
        obs.hardness = 230;
        obs.friction = 128;
        assert_eq!(describe(&obs), "hard solid");
        obs.emission = 9;
        obs.transparency = 210;
        obs.friction = 20;
        assert_eq!(describe(&obs), "glowing clear hard slick solid");
        assert_eq!(fallback_names(&obs).tool, "glowing clear hard slick solid tool");
    }
}
