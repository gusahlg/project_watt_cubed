//! Game-typed glue between `World`/`Player`/`Mods` and `SaveDoc`.
//! Only file in `save/` that knows about game types.

use voxel_engine::DVec3;

use crate::block::element::ElementId;
use crate::block::{AIR, BlockId, Composition};
use crate::coord::{BlockCoord, ChunkCoord, Local};
use crate::mods::{self, Mods};
use crate::player::Player;
use crate::world::chunk::Chunk;
use crate::world::diffusion::DiffusionCfg;
use crate::world::generation::WorldgenKind;
use crate::world::{FastMap, World};

use super::format::{self, Edit, PlayerState, SaveDoc, WorldgenStamp};
use super::slot::{SaveError, SaveMeta, SlotId};
use super::store::{self, Source};
use super::{composition_spec, parse_block};

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

/// Cheap main-thread snapshot of everything a save needs to encode.
///
/// Clones the compact edit overlay (`FastMap` memcpy, no spec strings) plus
/// player pose, mod-state records, and the small palette/element tables the
/// writer uses to build spec strings. Encode and disk write happen on the
/// autosave worker; see [`SaveSnapshot::encode`] and [`Autosaver::start`].
///
/// 100k-edit overlay (this box, 2026-09-09, release
/// `autosave_snapshot_and_encode_at_100k_edits`, median of 5): snapshot
/// 1.12 ms; encode 7.07 ms; write 2.54 ms; encode+write 9.61 ms.
///
/// [`Autosaver::start`]: super::autosave::Autosaver::start
pub struct SaveSnapshot {
    overlay: FastMap<ChunkCoord, FastMap<usize, BlockId>>,
    player: PlayerState,
    mods: Vec<(String, String)>,
    meta: SaveMeta,
    worldgen_version: u16,
    worldgen: WorldgenStamp,
    compositions: Vec<Composition>,
    element_names: Vec<Box<str>>,
}

impl SaveSnapshot {
    /// Capture live game state. O(edits) overlay clone plus O(palette) composition
    /// copies; spec strings are not built here.
    pub fn capture(world: &World, player: &Player, mods: &Mods, mut meta: SaveMeta) -> Self {
        meta.seed = world.seed();
        meta.last_played = unix_now();
        let registry = world.registry();
        let compositions = (0..registry.block_count())
            .map(|i| registry.block(BlockId(i as u16)).composition.clone())
            .collect();
        let elements = registry.elements();
        let element_names = (0..elements.len())
            .map(|i| elements.get(ElementId(i as u16)).name.clone())
            .collect();
        Self {
            overlay: world.clone_edit_overlay(),
            player: PlayerState {
                pos: [player.position.x, player.position.y, player.position.z],
                yaw: player.orientation.yaw,
                pitch: player.orientation.pitch,
                flying: player.flying(),
                noclip: player.noclip(),
                stash: Some(player.stash.to_portable(|id| {
                    world.registry().elements().get(id).name.as_ref()
                })),
            },
            mods: mods.save_states(world),
            meta,
            worldgen_version: crate::world::placement::WORLDGEN_VERSION,
            worldgen: stamp_from_world(world),
            compositions,
            element_names,
        }
    }

    /// Empty overlay, air-only palette — writer-lifecycle tests that must not
    /// construct a `World`.
    #[cfg(test)]
    pub(crate) fn empty(meta: SaveMeta, player: PlayerState) -> Self {
        Self {
            overlay: FastMap::default(),
            player,
            mods: Vec::new(),
            meta,
            worldgen_version: crate::world::placement::WORLDGEN_VERSION,
            worldgen: WorldgenStamp::default(),
            compositions: vec![Composition::Natural(Box::new([]))],
            element_names: Vec::new(),
        }
    }

    fn block_spec(&self, id: BlockId) -> String {
        if id == AIR {
            return "air".to_string();
        }
        composition_spec(&self.compositions[id.0 as usize], |e| {
            self.element_names[e.0 as usize].as_ref()
        })
    }

    fn edits(&self) -> impl Iterator<Item = ((i32, i32, i32), BlockId)> + '_ {
        self.overlay.iter().flat_map(|(&coord, cells)| {
            cells.iter().map(move |(&index, &id)| {
                let (lx, ly, lz) = Chunk::local_of(index);
                let local = Local::new(lx as u8, ly as u8, lz as u8)
                    .expect("chunk-local index is < CHUNK_SIZE");
                (BlockCoord::join(coord, local).to_tuple(), id)
            })
        })
    }

    /// Build the document the codec writes. Same spec-table order as walking
    /// `World::edits` on the cloned overlay.
    pub(crate) fn to_doc(&self) -> Result<SaveDoc, SaveError> {
        let mut specs: Vec<String> = Vec::new();
        let mut index_of: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
        let mut edits: Vec<Edit> = Vec::new();
        for ((x, y, z), id) in self.edits() {
            let spec = self.block_spec(id);
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
        let mut meta = self.meta.clone();
        meta.edit_count = u32::try_from(edits.len())
            .map_err(|_| SaveError::Corrupt("too many edits to save"))?;

        Ok(SaveDoc {
            meta,
            worldgen_version: self.worldgen_version,
            worldgen: self.worldgen,
            player: self.player.clone(),
            specs,
            edits,
            mods: self.mods.clone(),
        })
    }

    /// Spec table + document bytes. Runs on the writer thread for periodic
    /// autosave; the exit path still calls this on the caller's thread.
    pub fn encode(&self) -> Result<Vec<u8>, SaveError> {
        format::encode(&self.to_doc()?)
    }
}

/// Snapshot the game into a plain `SaveDoc`. Seed, edit count, and
/// last-played are stamped here; name/created/playtime come from the caller.
pub fn to_doc(
    world: &World,
    player: &Player,
    mods: &Mods,
    meta: SaveMeta,
) -> Result<SaveDoc, SaveError> {
    snapshot(world, player, mods, meta).to_doc()
}

/// Cheap main-thread capture; encode happens later via [`SaveSnapshot::encode`].
pub fn snapshot(world: &World, player: &Player, mods: &Mods, meta: SaveMeta) -> SaveSnapshot {
    SaveSnapshot::capture(world, player, mods, meta)
}

fn stamp_from_world(world: &World) -> WorldgenStamp {
    let cfg = world.diffusion_cfg();
    WorldgenStamp {
        kind: world.worldgen().wire(),
        tile: cfg.tile,
        stride: cfg.stride,
        phases: cfg.phases,
        relief: cfg.relief,
    }
}

fn restore_stash(player: &mut Player, doc: &SaveDoc, world: &World) {
    let elements = world.registry().elements();
    match &doc.player.stash {
        Some(items) => player.stash.load_portable(items, |n| elements.id_by_name(n)),
        None => {
            // Pre-v7: the inventory mod owned the counts.
            if let Some((_, data)) = doc
                .mods
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("inventory"))
            {
                let (_, payload) = mods::split_mod_version(data);
                player.stash.load_names(payload, |n| elements.id_by_name(n));
            }
        }
    }
}

fn kind_cfg_from_stamp(stamp: WorldgenStamp) -> (WorldgenKind, DiffusionCfg) {
    let kind = WorldgenKind::from_wire(stamp.kind).unwrap_or(WorldgenKind::Classic);
    let cfg = DiffusionCfg {
        tile: stamp.tile,
        stride: stamp.stride,
        phases: stamp.phases,
        relief: stamp.relief,
    }
    .clamp();
    (kind, cfg)
}

/// Rebuild a ready-to-play world and player from a doc, restoring mod state
/// into `mods`. Unknown specs degrade to air, exactly like the network path.
/// `make_world` is the caller's choice of `World` constructor — the header's
/// seed, kind, and diffusion knobs are passed in so a loaded world rebuilds
/// with the generator that wrote it, not the caller's current mod flags.
pub fn from_doc(
    doc: SaveDoc,
    mods: &mut Mods,
    make_world: impl FnOnce(i64, WorldgenKind, DiffusionCfg) -> World,
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
    let (kind, cfg) = kind_cfg_from_stamp(doc.worldgen);
    let mut world = make_world(doc.meta.seed, kind, cfg);

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

    restore_stash(&mut player, &doc, &world);

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

/// Snapshot straight to bytes — the synchronous encode the exit-flush path
/// still runs on the caller's thread.
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
    make_world: impl FnOnce(i64, WorldgenKind, DiffusionCfg) -> World,
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
