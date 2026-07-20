//! Game-typed glue between `World`/`Player`/`Mods` and `SaveDoc`.
//! Only file in `save/` that knows about game types.

use voxel_engine::DVec3;

use crate::mods::Mods;
use crate::player::Player;
use crate::world::World;

use super::format::{self, Edit, PlayerState, SaveDoc};
use super::slot::{SaveError, SaveMeta, SlotId};
use super::store::{self, Source};
use super::{block_spec, parse_block};

/// How a load actually went; the menu formats whichever fields are set
/// ("restored from backup", "recovered 48,112 of 48,300 edits").
#[derive(Clone, Copy, Debug)]
pub struct LoadReport {
    pub source: Source,
    /// (recovered, expected) edits; set if file was truncated.
    pub salvage: Option<(u32, u32)>,
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Snapshot the game into a plain `SaveDoc`. Seed, edit count, and
/// last-played are stamped here; name/created/playtime come from the caller.
pub fn to_doc(
    world: &World,
    player: &Player,
    mods: &Mods,
    mut meta: SaveMeta,
) -> Result<SaveDoc, SaveError> {
    meta.seed = world.seed();
    meta.last_played = unix_now();

    let mut specs: Vec<String> = Vec::new();
    let mut index_of: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
    let mut edits: Vec<Edit> = Vec::new();
    for ((x, y, z), id) in world.edits() {
        let spec = block_spec(world, id);
        let index = match index_of.get(&spec) {
            Some(&index) => index,
            None => {
                let index = u16::try_from(specs.len())
                    .map_err(|_| SaveError::Corrupt("too many distinct block specs to save"))?;
                index_of.insert(spec.clone(), index);
                specs.push(spec);
                index
            }
        };
        edits.push(Edit { x, y, z, spec: index });
    }
    meta.edit_count = u32::try_from(edits.len())
        .map_err(|_| SaveError::Corrupt("too many edits to save"))?;

    Ok(SaveDoc {
        meta,
        worldgen_version: crate::world::placement::WORLDGEN_VERSION,
        player: PlayerState {
            pos: [player.position.x, player.position.y, player.position.z],
            yaw: player.orientation.yaw,
            pitch: player.orientation.pitch,
            flying: player.flying(),
            noclip: player.noclip(),
        },
        specs,
        edits,
        mods: mods.save_states(world),
    })
}

/// Rebuild a ready-to-play world and player from a doc, restoring mod state
/// into `mods`. Unknown specs degrade to air, exactly like the network path.
/// `make_world` is the caller's choice of `World` constructor — `World::new`
/// for tests/headless callers, `|seed| World::with_config_lazy(seed, render)`
/// for interactive sessions that want their render config installed before
/// any terrain generates. One function instead of a config/no-config pair:
/// the constructor closure already expresses the choice `World` itself offers.
pub fn from_doc(
    doc: SaveDoc,
    mods: &mut Mods,
    make_world: impl FnOnce(i64) -> World,
) -> (World, Player, SaveMeta) {
    // Warn, never reject: the seed regenerates terrain fine, but a save from
    // another worldgen replays its edits over terrain whose MATERIALS may have
    // moved (a mined-out iron vein may now sit in coal). Geometry never moves
    // across worldgen versions — that invariant is what keeps old saves sane.
    if doc.worldgen_version != crate::world::placement::WORLDGEN_VERSION {
        eprintln!(
            "save '{}' was written by worldgen v{} (current v{}): terrain materials \
             may differ under old edits",
            doc.meta.name,
            doc.worldgen_version,
            crate::world::placement::WORLDGEN_VERSION,
        );
    }
    let mut world = make_world(doc.meta.seed);

    let mut player = Player::new(DVec3::new(
        doc.player.pos[0],
        doc.player.pos[1],
        doc.player.pos[2],
    ));
    player.orientation.yaw = doc.player.yaw;
    player.orientation.pitch = doc.player.pitch;
    if doc.player.flying {
        player.set_flying(true);
        // Noclip is only reachable through the fly cycle (fly → fly+noclip).
        if doc.player.noclip {
            player.cycle_fly();
        }
    }

    let block_ids: Vec<_> = doc
        .specs
        .iter()
        .map(|spec| parse_block(&mut world, spec))
        .collect();
    for edit in &doc.edits {
        // Index valid: decode drops out-of-range edits.
        world.set_block(edit.x, edit.y, edit.z, block_ids[usize::from(edit.spec)]);
    }

    for (name, data) in &doc.mods {
        // Mutable world: restoring crafted blocks re-registers them by name.
        mods.load_state(name, data, &mut world);
    }

    (world, player, doc.meta)
}

/// Snapshot straight to bytes — the `encode` closure the [`Autosaver`] wants.
///
/// [`Autosaver`]: super::autosave::Autosaver
pub fn encode_current(
    world: &World,
    player: &Player,
    mods: &Mods,
    meta: SaveMeta,
) -> Result<Vec<u8>, SaveError> {
    format::encode(&to_doc(world, player, mods, meta)?)
}

/// Save synchronously (exit paths; the in-game path goes through the
/// Autosaver instead).
pub fn save(
    id: &SlotId,
    world: &World,
    player: &Player,
    mods: &Mods,
    meta: SaveMeta,
) -> Result<(), SaveError> {
    store::write(id, &encode_current(world, player, mods, meta)?)?;
    Ok(())
}

/// Load a slot, laddering to the backup and salvaging a truncated tail if it
/// comes to that. The report says how far down the ladder we went.
/// `make_world` is forwarded straight to [`from_doc`].
pub fn load(
    id: &SlotId,
    mods: &mut Mods,
    make_world: impl FnOnce(i64) -> World,
) -> Result<(World, Player, SaveMeta, LoadReport), SaveError> {
    let (decoded, source) = store::read(id)?;
    let (doc, salvage) = match decoded {
        format::Decoded::Intact(doc) => (doc, None),
        format::Decoded::Salvaged { doc, recovered, expected } => {
            (doc, Some((recovered, expected)))
        }
    };
    let (world, player, meta) = from_doc(doc, mods, make_world);
    Ok((world, player, meta, LoadReport { source, salvage }))
}
