//! The integrated server a player hosts from the menu, and the save slot it serves.
//! It outlives a return to the menu so friends stay connected. Singleplayer never
//! has that slot open at the same time: the app stops the host before it opens or
//! deletes a world, and stopping saves.

use std::io;
use std::path::PathBuf;

use crate::net::persist::LoadError;
use crate::net::server::{self, Config, ServerHandle};
use crate::save::{self, Slot, SlotId};

/// Shown when opening or deleting a world stopped the integrated server.
const STOPPED: &str = "hosting stopped; the hosted world was saved";

/// The running integrated server, if any, and the slot it serves.
#[derive(Default)]
pub(super) struct Host(Option<(ServerHandle, SlotId)>);

impl Host {
    pub(super) fn running(&self) -> bool {
        self.0.is_some()
    }

    pub(super) fn serves(&self, id: &SlotId) -> bool {
        self.0.as_ref().is_some_and(|(_, slot)| slot == id)
    }

    /// Save and stop. The notice that says so, or `None` when nothing ran.
    pub(super) fn stop(&mut self) -> Option<&'static str> {
        let (server, _) = self.0.take()?;
        server.stop();
        Some(STOPPED)
    }

    /// Stop any running server, then serve the newest readable save the server can
    /// load, else a fresh slot. `config` builds the settings for one world file.
    /// Returns the bound port and, when a save was skipped, a notice naming it and why.
    pub(super) fn start(
        &mut self,
        saves: &[Slot],
        port: u16,
        config: impl Fn(PathBuf) -> Config,
    ) -> io::Result<(u16, Option<String>)> {
        self.stop();
        let mut skipped = Vec::new();
        for id in host_slots(saves) {
            let server = match server::spawn(port, config(save::file_path(&id))) {
                Ok(server) => server,
                Err(e) => match e.get_ref().and_then(|inner| inner.downcast_ref::<LoadError>()) {
                    Some(why) => {
                        skipped.push(format!("{id}: {}", why.reason()));
                        continue;
                    }
                    None => return Err(e),
                },
            };
            let port = server.addr().port();
            let notice = (!skipped.is_empty()).then(|| {
                let fresh = !saves.iter().any(|slot| slot.id == id);
                format!("skipped {}; hosting {}", skipped.join("; "), if fresh { "a new world" } else { id.as_str() })
            });
            self.0 = Some((server, id));
            return Ok((port, notice));
        }
        Err(io::Error::other(skipped.join("; ")))
    }
}

/// Slots to try, in order: each readable save, most recently played first, then a fresh one.
fn host_slots(saves: &[Slot]) -> impl Iterator<Item = SlotId> + '_ {
    saves
        .iter()
        .filter(|slot| slot.meta.is_ok())
        .map(|slot| slot.id.clone())
        .chain(std::iter::once_with(save::fresh_id))
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::{Duration, Instant};

    use voxel_engine::DVec3;

    use super::*;
    use crate::block::AIR;
    use crate::math::block_coord;
    use crate::modding::Mods;
    use crate::net::client::{Connection, Incoming};
    use crate::player::Player;
    use crate::render_config::RenderConfig;
    use crate::save::SaveMeta;
    use crate::save::slot::SaveError;
    use crate::world::World;
    use crate::world::generation::WorldgenKind;
    use crate::world::terrain::TerrainCfg;

    fn world(seed: i64, kind: WorldgenKind, cfg: TerrainCfg) -> World {
        World::with_kind_cfg(seed, RenderConfig::default(), kind, cfg, true)
    }

    fn config(world: PathBuf) -> Config {
        Config { worldgen: WorldgenKind::Flat, world: Some(world), ..Config::default() }
    }

    fn remove(id: &SlotId) {
        let live = save::file_path(id);
        let _ = std::fs::remove_file(live.with_extension("save.bak"));
        let _ = std::fs::remove_file(live);
    }

    /// A real singleplayer slot in the test data dir. `last_played` orders the list.
    fn write_slot(name: &str, last_played: u64) -> Slot {
        let id = SlotId::new(name).unwrap();
        remove(&id);
        let meta = SaveMeta { name: name.into(), seed: 5, created: 1, last_played, playtime_secs: 0, edit_count: 0 };
        let player = Player::new(DVec3::new(0.5, 40.0, 0.5));
        save::save(&id, &world(5, WorldgenKind::Flat, TerrainCfg::default()), &player, &Mods::empty(), meta.clone()).unwrap();
        Slot { id, meta: Ok(meta) }
    }

    #[test]
    fn host_slots_are_readable_saves_newest_first_then_a_fresh_one() {
        let slots = vec![
            Slot::for_test("newer", 3, 1),
            Slot { id: SlotId::new("broken").unwrap(), meta: Err(SaveError::Corrupt("not a save")) },
            Slot::for_test("older", 1, 0),
        ];
        let order: Vec<SlotId> = host_slots(&slots).take(2).collect();
        assert_eq!(order, [slots[0].id.clone(), slots[2].id.clone()]);
        let fresh = host_slots(&[]).next().expect("a fresh slot");
        assert_eq!(save::file_path(&fresh).extension().and_then(|ext| ext.to_str()), Some("save"));
    }

    /// The app stops the host before singleplayer opens a world. The stop saves the
    /// friends' edits into the slot, and nothing the host held can overwrite the
    /// singleplayer save afterwards.
    #[test]
    fn singleplayer_opens_the_hosted_slot_only_after_the_host_saved_and_stopped() {
        let slot = write_slot("__r2_hosted_slot__", 10);
        let id = slot.id.clone();
        let mut host = Host::default();
        let (port, notice) = host.start(std::slice::from_ref(&slot), 0, config).unwrap();
        assert!(notice.is_none());
        assert!(host.serves(&id));

        let mut friend = Connection::connect("127.0.0.1", port, "bob", "").unwrap();
        // Break the flat world's top block under the spawn: a real edit (air where air already is
        // would be a no-op the server leaves out of the file).
        let s = friend.spawn();
        let ground = crate::world::generation::FLAT_HEIGHT - 1;
        let req = friend.send_edit(block_coord(s.x), ground, block_coord(s.z), "air".into()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut accepted = false;
        while !accepted && Instant::now() < deadline {
            accepted = friend.poll().into_iter().any(|e| matches!(e, Incoming::EditAccepted { req: r } if r == req));
            thread::sleep(Duration::from_millis(10));
        }
        assert!(accepted, "the friend's edit is committed on the host");
        drop(friend);

        assert_eq!(host.stop(), Some(STOPPED));
        assert!(!host.running() && !host.serves(&id));
        let mut mods = Mods::empty();
        let (mut sp, player, meta, _) = save::load(&id, &mut mods, world).unwrap();
        assert!(meta.edit_count >= 1, "the stop saved the friend's edit before singleplayer read the slot");
        let (x, z) = (8, 8);
        let y = sp.surface_y(x, z) - 1;
        assert_ne!(sp.block_at(x, y, z), AIR);
        sp.set_block(x, y, z, AIR);
        save::save(&id, &sp, &player, &mods, meta).unwrap();

        drop(host);
        let (again, _, _, _) = save::load(&id, &mut Mods::empty(), world).unwrap();
        assert_eq!(again.block_at(x, y, z), AIR, "the singleplayer edit survives the app's exit");
        remove(&id);
    }

    /// The newest save is from another universe: hosting skips it, serves the next one,
    /// and says which save it skipped and why. With nothing loadable it hosts a new world.
    #[test]
    fn hosting_skips_a_save_the_server_cannot_load() {
        let foreign = write_slot("__r2_host_foreign__", 20);
        let good = write_slot("__r2_host_good__", 10);
        let path = save::file_path(&foreign.id);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[save::format::HEADER_LEN - material::STAMP_LEN] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        let saves = [foreign, good];
        let why = "__r2_host_foreign__: save belongs to a different universe (law stamp mismatch)";

        let mut host = Host::default();
        let (_, notice) = host.start(&saves, 0, config).unwrap();
        assert!(host.serves(&saves[1].id));
        assert_eq!(notice, Some(format!("skipped {why}; hosting __r2_host_good__")));

        let fresh = save::fresh_id();
        let (_, notice) = host.start(&saves[..1], 0, config).unwrap();
        assert!(host.serves(&fresh), "starting again stopped the first host");
        assert_eq!(notice, Some(format!("skipped {why}; hosting a new world")));
        host.stop();
        for id in [&fresh, &saves[0].id, &saves[1].id] {
            remove(id);
        }
    }
}
