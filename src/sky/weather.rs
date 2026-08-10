//! `Weather` — the shared parameter block that couples into the other sky
//! systems. It owns no pixels: it only nudges fog and (later) cloud density.
/// The kinds of precipitation that can fall. Closed on purpose — these are the
/// only variants the rest of the sky knows how to react to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precip {
    Clear,
    Rain,
    Snow,
}

/// Current weather. `coverage` (0 clear … 1 overcast) and `wetness` are plain
/// lerp targets; they feed fog and, in a later phase, the cloud layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weather {
    pub coverage: f32,
    pub precip: Precip,
    pub wetness: f32,
}

impl Default for Weather {
    fn default() -> Self {
        Self { coverage: 0.15, precip: Precip::Clear, wetness: 0.0 }
    }
}

impl Weather {
    /// Extra fog density contributed by the current weather: overcast and
    /// precipitation both shorten visibility.
    pub fn fog_bonus(&self) -> f32 {
        let precip = match self.precip {
            Precip::Clear => 0.0,
            Precip::Rain => 0.004,
            Precip::Snow => 0.006,
        };
        self.coverage * 0.003 + precip
    }

    /// Rain strength [0,1] driving the sky palette overrides: the precip
    /// `wetness`, zero unless it is actually raining. Snow leaves the (cool, wet)
    /// rain sky tint untouched.
    pub fn rain_strength(&self) -> f32 {
        match self.precip {
            Precip::Rain => self.wetness,
            _ => 0.0,
        }
    }
}
