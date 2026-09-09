//! Default-enabled visual mods. The core renderer is sunlight-only; these
//! restore the shipped look when enabled and strip their lanes when not.

use crate::mods::Mod;
use crate::render_config::VisualGroup;

macro_rules! visual_mod {
    ($ty:ident, $name:literal, $id:literal, $desc:literal, $group:ident) => {
        pub struct $ty;

        impl Mod for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn id(&self) -> &'static str {
                $id
            }
            fn description(&self) -> &str {
                $desc
            }
            fn visual_group(&self) -> Option<VisualGroup> {
                Some(VisualGroup::$group)
            }
        }
    };
}

visual_mod!(
    AtmosphereMod,
    "Atmosphere",
    "atmosphere",
    "Sky, clouds, weather, stars, day/night, fog, and animated water.",
    Atmosphere
);
visual_mod!(
    PostMod,
    "Post",
    "post",
    "Bloom, god rays, TAA, exposure, vignette, and variable-rate shading.",
    Post
);
visual_mod!(
    LightingMod,
    "Lighting",
    "lighting",
    "Shadows, ambient fill, and block light. Sunlight stays in the core.",
    Lighting
);
