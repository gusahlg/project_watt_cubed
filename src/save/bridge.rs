//! Game-typed glue between `World`/`Player`/`Mods` and `SaveDoc`.
//! Only file in `save/` that knows about game types.

use voxel_engine::DVec3;

use crate::block::registry::SpecKind;
use crate::block::{AIR, BlockId};
use crate::coord::{BlockCoord, ChunkCoord, Local};
use crate::modding::Mods;
use crate::coord::Face;
use crate::player::{Motion, Player};
use crate::world::chunk::Chunk;
use crate::world::terrain::TerrainCfg;
use crate::world::generation::WorldgenKind;
use crate::world::{FastMap, World};

use super::format::{self, Edit, PendingContact, PlayerState, SaveDoc, SpecTable, WorldgenStamp};
use super::slot::{SaveError, SaveMeta, SlotId};
use super::store::{self, Source};

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
pub struct SaveSnapshot {
    overlay: FastMap<ChunkCoord, FastMap<usize, BlockId>>,
    player: PlayerState,
    mods: Vec<(String, String)>,
    meta: SaveMeta,
    worldgen_version: u16,
    worldgen: WorldgenStamp,
    law_stamp: Vec<u8>,
    specs: Vec<String>,
    pending: Vec<PendingContact>,
}

impl SaveSnapshot {
    /// Capture live game state. O(edits) overlay clone plus O(palette) spec
    /// strings.
    pub fn capture(world: &World, player: &Player, mods: &Mods, mut meta: SaveMeta) -> Self {
        meta.seed = world.seed();
        meta.last_played = unix_now();
        let registry = world.registry();
        let specs = (0..registry.block_count())
            .map(|i| registry.spec(BlockId(i as u16)))
            .collect();
        // A cruise is not saved: the save holds where ending it now would land (at rest, in flight,
        // clear of the ground), so a rejoin never resumes a light-speed flight.
        let (pos, velocity, flying, noclip) = match player.cruise {
            Some(cruise) => (world.clear_of_ground(player.position), DVec3::ZERO, true, cruise.noclip),
            None => (player.position, player.velocity(), player.flying(), player.noclip()),
        };
        Self {
            overlay: world.clone_edit_overlay(),
            player: PlayerState {
                pos: pos.to_array(),
                yaw: player.orientation.yaw,
                pitch: player.orientation.pitch,
                frame: player.orientation.frame,
                velocity: velocity.to_array(),
                up: player.up_axis as u8,
                legacy_pose: false,
                flying,
                noclip,
                inventory: Some(player.inventory.to_portable(|id| world.registry().spec(id))),
            },
            mods: mods.save_states(world),
            meta,
            worldgen_version: crate::world::terrain::WORLDGEN_VERSION,
            worldgen: stamp_from_world(world),
            law_stamp: world.registry().law().stamp(),
            specs,
            pending: world
                .reactions()
                .snapshot()
                .into_iter()
                .map(|(age, c)| PendingContact { x: c.lo.0, y: c.lo.1, z: c.lo.2, axis: c.axis, age })
                .collect(),
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
            worldgen_version: crate::world::terrain::WORLDGEN_VERSION,
            worldgen: WorldgenStamp::default(),
            law_stamp: material::Law::current().stamp(),
            specs: vec!["air".into()],
            pending: Vec::new(),
        }
    }

    fn block_spec(&self, id: BlockId) -> &str {
        if id == AIR {
            return "air";
        }
        self.specs.get(usize::from(id.0)).map_or("air", String::as_str)
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
        let mut table = SpecTable::default();
        // Each block's table index, so the spec text is looked up once per block, not per edit.
        // `u16::MAX` is unset; a block that really sits at that index just takes the lookup again.
        let mut index_of = vec![u16::MAX; self.specs.len()];
        let mut edits = Vec::with_capacity(self.overlay.values().map(|cells| cells.len()).sum());
        for ((x, y, z), id) in self.edits() {
            let spec = match index_of.get_mut(usize::from(id.0)) {
                Some(&mut at) if at != u16::MAX => at,
                Some(at) => {
                    *at = table.index(self.block_spec(id))?;
                    *at
                }
                None => table.index(self.block_spec(id))?,
            };
            edits.push(Edit { x, y, z, spec });
        }
        let mut meta = self.meta.clone();
        meta.edit_count = u32::try_from(edits.len())
            .map_err(|_| SaveError::Corrupt("too many edits to save"))?;

        Ok(SaveDoc {
            meta,
            worldgen_version: self.worldgen_version,
            worldgen: self.worldgen,
            law_stamp: self.law_stamp.clone(),
            player: self.player.clone(),
            specs: table.specs,
            edits,
            mods: self.mods.clone(),
            pending: self.pending.clone(),
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
    WorldgenStamp { kind: world.worldgen().wire(), knobs: world.terrain_cfg().to_wire() }
}

fn restore_inventory(player: &mut Player, doc: &SaveDoc, world: &mut World) -> UnknownMaterials {
    let Some(items) = &doc.player.inventory else {
        return UnknownMaterials::default();
    };
    let mut u = UnknownMaterials::default();
    let mut pairs = Vec::new();
    for (spec, count) in items {
        match world.registry_mut().read_spec(spec) {
            SpecKind::Ok(id) => pairs.push((id, *count)),
            SpecKind::Legacy => u.legacy_holdings += 1,
            SpecKind::Full => u.full_holdings += 1,
            SpecKind::Bad => {}
        }
    }
    player.inventory.clear();
    for (id, count) in pairs {
        player.inventory.add(id, count);
    }
    u
}

/// Counts of specs the loader could not intern, split by cause so the notice
/// can say "legacy names" versus "the material table is full".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnknownMaterials {
    pub legacy_edits: u32,
    pub full_edits: u32,
    pub legacy_holdings: u32,
    pub full_holdings: u32,
}

/// One load notice covering unknown edits and unknown inventory/pouch holdings.
pub(crate) fn unknown_material_notice(u: UnknownMaterials) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if u.legacy_edits > 0 {
        parts.push(format!(
            "save predates the material model; {} edits of unknown materials became air",
            u.legacy_edits
        ));
    }
    if u.full_edits > 0 {
        parts.push(format!(
            "the material table is full; {} edits of unknown materials became air",
            u.full_edits
        ));
    }
    if u.legacy_holdings > 0 {
        parts.push(format!(
            "{} holdings of unknown materials were dropped",
            u.legacy_holdings
        ));
    }
    if u.full_holdings > 0 {
        parts.push(format!(
            "{} holdings could not be interned (table full)",
            u.full_holdings
        ));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("; "))
    }
}

/// Pre-v8 documents carry no law stamp; the load assumes law v0.
fn kind_cfg_from_stamp(stamp: WorldgenStamp) -> (WorldgenKind, TerrainCfg) {
    let kind = WorldgenKind::from_wire(stamp.kind).unwrap_or_default();
    (kind, TerrainCfg::from_wire(stamp.knobs))
}

/// A save rebuilt up to its mod state and pending reactions, which wait for the live mods:
/// [`finish`](Self::finish) adds them on the thread that owns them.
pub struct Restored {
    world: World,
    player: Player,
    meta: SaveMeta,
    mods: Vec<(String, String)>,
    pending: Vec<(u32, crate::sim::reactions::Contact)>,
    unknown: UnknownMaterials,
}

impl Restored {
    /// Restore mod state into `mods`, then the pending reaction work, in the order a save
    /// always loaded them.
    pub fn finish(mut self, mods: &mut Mods) -> (World, Player, SaveMeta) {
        for (name, data) in &self.mods {
            self.unknown.legacy_holdings += mods.load_state(name, data, &mut self.world);
        }
        self.world.restore_reactions(&self.pending);
        if let Some(msg) = unknown_material_notice(self.unknown) {
            eprintln!("{msg}");
        }
        (self.world, self.player, self.meta)
    }
}

/// Rebuild the world and player from a doc, everything but what [`Restored::finish`] adds.
/// Unknown specs degrade to air. A law stamp that does not match this game's law is a
/// different universe and is refused, never a panic.
fn rebuild(
    doc: SaveDoc,
    make_world: impl FnOnce(i64, WorldgenKind, TerrainCfg) -> World,
) -> Result<Restored, SaveError> {
    if doc.law_stamp != material::Law::current().stamp() {
        return Err(SaveError::LawMismatch);
    }
    if doc.worldgen_version != crate::world::terrain::WORLDGEN_VERSION {
        eprintln!(
            "save '{}' was written by worldgen v{} (current v{}): terrain materials \
             may differ under old edits",
            doc.meta.name,
            doc.worldgen_version,
            crate::world::terrain::WORLDGEN_VERSION,
        );
    }
    let (kind, cfg) = kind_cfg_from_stamp(doc.worldgen);
    let mut world = make_world(doc.meta.seed, kind, cfg);

    // A saved position outside the world (a corrupt or runaway save) restarts at the spawn rather than
    // streaming around nonsense.
    let saved_pos = DVec3::from_array(doc.player.pos);
    let border = crate::math::WORLD_BORDER;
    let pos = if saved_pos.is_finite() {
        saved_pos.clamp(DVec3::splat(-border), DVec3::splat(border))
    } else {
        world.chart_spawn().unwrap_or_else(|| DVec3::new(0.5, world.surface_y(0, 0) as f64 + 3.0, 0.5))
    };
    let mut player = Player::new(pos);
    player.orientation.yaw = doc.player.yaw;
    player.orientation.pitch = doc.player.pitch;
    if doc.player.legacy_pose {
        // v9 fabricated an identity pose. Stand in the gravity at the saved position.
        player.stand_in(world.gravity_at(player.position).accel);
    } else {
        // Assign the saved frame. `snap_up` would rebuild it and drop any twist around up.
        player.orientation.frame = doc.player.frame;
        player.up_axis = Face::from_index(doc.player.up).unwrap_or(Face::PosY);
    }
    if doc.player.noclip {
        player.toggle_noclip();
    } else if doc.player.flying {
        player.set_flying(true);
    }
    // `set_flying` / `toggle_noclip` zero the component along up. Write the saved velocity after.
    // A speed past the limit only ever came from a runaway `/flyspeed` (fly speed itself is not
    // saved): restart at rest instead of resuming the runaway.
    let saved = DVec3::from_array(doc.player.velocity);
    let saved = if saved.is_finite() && saved.length() <= crate::player::MAX_SPEED { saved } else { DVec3::ZERO };
    match &mut player.motion {
        Motion::Walking { velocity, .. } | Motion::Flying { velocity, .. } => *velocity = saved,
    }

    let mut unknown = restore_inventory(&mut player, &doc, &mut world);

    let kinds: Vec<SpecKind> = doc
        .specs
        .iter()
        .map(|spec| world.registry_mut().read_spec(spec))
        .collect();
    let block_ids: Vec<BlockId> = kinds
        .iter()
        .map(|k| match k {
            SpecKind::Ok(id) => *id,
            _ => AIR,
        })
        .collect();
    for edit in &doc.edits {
        match kinds[usize::from(edit.spec)] {
            SpecKind::Ok(_) => {}
            SpecKind::Legacy => unknown.legacy_edits += 1,
            SpecKind::Full => unknown.full_edits += 1,
            SpecKind::Bad => {}
        }
        world.set_block(edit.x, edit.y, edit.z, block_ids[usize::from(edit.spec)]);
    }

    let pending = doc
        .pending
        .iter()
        .map(|c| (c.age, crate::sim::reactions::Contact { lo: (c.x, c.y, c.z), axis: c.axis }))
        .collect();
    Ok(Restored { world, player, meta: doc.meta, mods: doc.mods, pending, unknown })
}

/// Rebuild a ready-to-play world and player from a doc, restoring mod state into `mods` and
/// the pending reaction work into the world's scheduler.
#[cfg(test)]
pub fn from_doc(
    doc: SaveDoc,
    mods: &mut Mods,
    make_world: impl FnOnce(i64, WorldgenKind, TerrainCfg) -> World,
) -> Result<(World, Player, SaveMeta), SaveError> {
    Ok(rebuild(doc, make_world)?.finish(mods))
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
/// Autosaver instead). Tests pin the encode+write pairing; production
/// writes via [`encode_current`] + the autosaver.
#[cfg(test)]
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

/// Read a slot, laddering to the backup and salvaging a truncated tail if it comes to that, and
/// rebuild it up to [`Restored::finish`]. Needs no mods, so it runs on any thread. The report says
/// how far down the ladder we went. `make_world` builds the world from the save's seed and generator.
pub fn restore(
    id: &SlotId,
    make_world: impl FnOnce(i64, WorldgenKind, TerrainCfg) -> World,
) -> Result<(Restored, LoadReport), SaveError> {
    let (decoded, source) = store::read(id)?;
    let (doc, salvage) = match decoded {
        format::Decoded::Intact(doc) => (doc, None),
        format::Decoded::Salvaged { doc, recovered, expected } => {
            (doc, Some((recovered, expected)))
        }
    };
    Ok((rebuild(doc, make_world)?, LoadReport { source, salvage }))
}

/// [`restore`] and [`Restored::finish`] in one go.
#[cfg(test)]
pub fn load(
    id: &SlotId,
    mods: &mut Mods,
    make_world: impl FnOnce(i64, WorldgenKind, TerrainCfg) -> World,
) -> Result<(World, Player, SaveMeta, LoadReport), SaveError> {
    let (restored, report) = restore(id, make_world)?;
    let (world, player, meta) = restored.finish(mods);
    Ok((world, player, meta, report))
}
