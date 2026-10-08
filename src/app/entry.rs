//! World entry off the render thread. The render thread captures every input that decides the
//! world, a job builds it (and decodes the save, for a load), and the render thread only swaps
//! the finished game in.
use std::thread::{self, JoinHandle};

use crate::net::client::Connection;
use crate::player::Player;
use crate::render_config::RenderConfig;
use crate::save::slot::SaveError;
use crate::save::{self, LoadReport, Restored, SlotId};
use crate::world::World;
use crate::world::generation::WorldgenKind;
use crate::world::terrain::TerrainCfg;

/// One value computed on its own thread. Dropping the job detaches it: the value is dropped on
/// that thread when it lands.
pub(super) struct Job<T>(JoinHandle<T>);

impl<T: Send + 'static> Job<T> {
    fn spawn(work: impl FnOnce() -> T + Send + 'static) -> Self {
        let handle = thread::Builder::new()
            .name("world-entry".into())
            .spawn(work)
            .expect("spawn the world-entry thread");
        Self(handle)
    }
}

impl<T> Job<T> {
    fn done(&self) -> bool {
        self.0.is_finished()
    }

    /// Block until the value lands. A panic on the job resumes here, as if it ran inline.
    pub(super) fn wait(self) -> T {
        self.0.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }
}

/// What decides a generated world, read before its job starts.
#[derive(Clone, Copy)]
pub(super) struct Recipe {
    pub seed: i64,
    pub render: RenderConfig,
    pub kind: WorldgenKind,
    pub cfg: TerrainCfg,
}

impl Recipe {
    fn world(self) -> World {
        World::with_kind_cfg(self.seed, self.render, self.kind, self.cfg, false)
    }
}

/// A world being built, with what the render thread adds when it lands.
pub(super) enum Loading {
    /// A new world, its spawn, and a fresh slot.
    New(Job<(World, Player, SlotId)>),
    Load(SlotId, Job<Result<(Restored, LoadReport), SaveError>>),
    /// The server's world; the connection waits on the render thread.
    Join { job: Job<(World, Player)>, conn: Connection, notice: Option<String>, hosted: bool },
}

impl Loading {
    pub(super) fn new_world(recipe: Recipe) -> Self {
        Self::New(Job::spawn(move || {
            let world = recipe.world();
            let player = super::spawn_player(&world);
            (world, player, save::fresh_id())
        }))
    }

    /// The save names its own seed and generator; only the render lanes come from here.
    pub(super) fn load(id: SlotId, render: RenderConfig) -> Self {
        let slot = id.clone();
        Self::Load(
            id,
            Job::spawn(move || save::restore(&slot, |seed, kind, cfg| Recipe { seed, render, kind, cfg }.world())),
        )
    }

    pub(super) fn join(conn: Connection, render: RenderConfig, notice: Option<String>, hosted: bool) -> Self {
        let recipe = Recipe { seed: conn.seed(), render, kind: conn.worldgen(), cfg: conn.terrain() };
        let spawn = conn.spawn();
        let job = Job::spawn(move || {
            let world = recipe.world();
            let mut player = Player::new(spawn);
            player.stand_in(world.gravity_at(player.position).accel);
            (world, player)
        });
        Self::Join { job, conn, notice, hosted }
    }

    pub(super) fn done(&self) -> bool {
        match self {
            Self::New(job) => job.done(),
            Self::Load(_, job) => job.done(),
            Self::Join { job, .. } => job.done(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{AIR, BlockId};
    use crate::save::{SaveMeta, Source};
    use voxel_engine::DVec3;

    fn recipe(seed: i64, kind: WorldgenKind, cfg: TerrainCfg) -> Recipe {
        Recipe { seed, render: RenderConfig::default(), kind, cfg }
    }

    fn diffusion(seed: i64) -> Recipe {
        recipe(seed, WorldgenKind::Diffusion, TerrainCfg::default())
    }

    fn slot(name: &str) -> SlotId {
        let id = SlotId::new(name).unwrap();
        clear(&id);
        id
    }

    fn clear(id: &SlotId) {
        let live = save::file_path(id);
        let _ = std::fs::remove_file(live.with_extension("save.bak"));
        let _ = std::fs::remove_file(live);
    }

    fn sorted_edits(world: &World) -> Vec<((i32, i32, i32), u16)> {
        let mut edits: Vec<_> = world.edits().map(|(cell, id)| (cell, id.0)).collect();
        edits.sort_unstable();
        edits
    }

    /// Bit-exact player state a save or a spawn decides.
    fn pose(player: &Player) -> Vec<u64> {
        let (p, q, v) = (player.position, player.orientation.frame, player.velocity());
        let flags = player.up_axis as u64 | (player.flying() as u64) << 8 | (player.noclip() as u64) << 9;
        let mut bits: Vec<u64> = [p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, player.orientation.yaw as f64]
            .iter()
            .map(|f| f.to_bits())
            .collect();
        bits.extend([flags, player.inventory.total() as u64]);
        bits
    }

    /// The storage cell under a physical position: the position itself off the charts.
    fn cell(world: &World, p: DVec3) -> [i32; 3] {
        let charted = world.terrain().atlases().iter().find_map(|a| a.storage_of(p));
        charted.map_or([p.x, p.y, p.z].map(crate::math::block_coord), |c| c.map(|v| v as i32))
    }

    /// The generated 16-cube around the player's cell, voxel by voxel.
    fn ground(world: &World, player: &Player) -> Vec<BlockId> {
        let [x0, y0, z0] = cell(world, player.position).map(|v| v - 8);
        let mut out = Vec::with_capacity(16 * 16 * 16);
        for y in y0..y0 + 16 {
            for z in z0..z0 + 16 {
                for x in x0..x0 + 16 {
                    out.push(world.terrain().voxel_at(x, y, z));
                }
            }
        }
        out
    }

    /// A Diffusion world with `edits` edits around the spawn, a moved player, an inventory and
    /// mod state.
    fn edited_save(id: &SlotId, edits: i32) {
        let mut world = diffusion(31).world();
        let mut player = super::super::spawn_player(&world);
        let soil = world.registry().id_by_label("soil").unwrap();
        let rock = world.registry().id_by_label("rock").unwrap();
        let [x0, y0, z0] = cell(&world, player.position).map(|v| v - 4);
        for i in 0..edits {
            let (x, y, z) = (x0 + i % 8, y0 + i / 64, z0 + (i / 8) % 8);
            let id = if world.terrain().voxel_at(x, y, z) == AIR { soil } else { AIR };
            world.set_block(x, y, z, id);
        }
        player.position += DVec3::new(0.25, 1.5, -0.75);
        player.orientation.yaw = 1.25;
        player.set_flying(true);
        player.inventory.add(rock, 3);
        let mut mods = crate::modding::testing::standard();
        let rock_spec = world.registry().spec(rock);
        mods.load_state("hotbar", &format!("v1;sel=1;1={rock_spec}"), &mut world);
        let meta = SaveMeta { name: id.as_str().into(), seed: 0, created: 1, last_played: 0, playtime_secs: 7, edit_count: 0 };
        save::save(id, &world, &player, &mods, meta).unwrap();
    }

    type Loaded = (World, Player, SaveMeta, LoadReport, Vec<(String, String)>);

    /// The load as the render thread ran it before: `save::load` inline.
    fn load_inline(id: &SlotId) -> Loaded {
        let mut mods = crate::modding::testing::standard();
        mods.reset_state();
        let (world, player, meta, report) =
            save::load(id, &mut mods, |seed, kind, cfg| recipe(seed, kind, cfg).world()).unwrap();
        let states = mods.save_states(&world);
        (world, player, meta, report, states)
    }

    /// The load through the job, finished as the render thread finishes it.
    fn load_by_job(id: &SlotId) -> Loaded {
        let Loading::Load(_, job) = Loading::load(id.clone(), RenderConfig::default()) else { unreachable!() };
        let (restored, report) = job.wait().unwrap();
        let mut mods = crate::modding::testing::standard();
        mods.reset_state();
        let (world, player, meta) = restored.finish(&mut mods);
        let states = mods.save_states(&world);
        (world, player, meta, report, states)
    }

    fn assert_same_load(id: &SlotId) -> LoadReport {
        let (w0, p0, m0, r0, s0) = load_inline(id);
        let (w1, p1, m1, r1, s1) = load_by_job(id);
        assert_eq!((w1.seed(), w1.worldgen(), w1.terrain_cfg()), (w0.seed(), w0.worldgen(), w0.terrain_cfg()));
        assert_eq!(sorted_edits(&w1), sorted_edits(&w0));
        assert_eq!(w1.reactions().snapshot(), w0.reactions().snapshot());
        assert_eq!(pose(&p1), pose(&p0));
        assert_eq!(m1, m0);
        assert_eq!((r1.source, r1.salvage), (r0.source, r0.salvage));
        assert_eq!(s1, s0, "mod state lands the same");
        r1
    }

    /// The benchmark's seed (42) and any other enter through the job with the world and spawn an
    /// inline build gives.
    #[test]
    fn a_new_world_through_the_job_is_the_inline_world() {
        for seed in [42, 1234] {
            let inline = diffusion(seed).world();
            let inline_player = super::super::spawn_player(&inline);
            let Loading::New(job) = Loading::new_world(diffusion(seed)) else { unreachable!() };
            let (world, player, _) = job.wait();
            assert_eq!((world.seed(), world.worldgen()), (seed, WorldgenKind::Diffusion));
            assert_eq!(world.edits().count(), 0);
            assert_eq!(pose(&player), pose(&inline_player));
            assert_eq!(ground(&world, &player), ground(&inline, &inline_player));
        }
    }

    #[test]
    fn a_saved_world_through_the_job_is_the_inline_load() {
        let id = slot("__entry_job_load__");
        edited_save(&id, 300);
        let report = assert_same_load(&id);
        assert_eq!((report.source, report.salvage), (Source::Live, None));
        let (world, player, meta, _, states) = load_by_job(&id);
        assert_eq!(world.edits().count(), 300);
        assert!(player.flying());
        assert_eq!((player.orientation.yaw, player.inventory.total(), meta.playtime_secs), (1.25, 3, 7));
        assert!(states.iter().any(|(name, _)| name == "hotbar"));
        clear(&id);
    }

    #[test]
    fn a_degraded_save_through_the_job_reports_as_inline() {
        let id = slot("__entry_job_backup__");
        edited_save(&id, 40);
        edited_save(&id, 80); // rotates the first to .bak
        std::fs::write(save::file_path(&id), b"not a save").unwrap();
        assert_eq!(assert_same_load(&id).source, Source::Backup);
        clear(&id);

        let id = slot("__entry_job_salvage__");
        edited_save(&id, 120);
        let bytes = std::fs::read(save::file_path(&id)).unwrap();
        std::fs::write(save::file_path(&id), &bytes[..bytes.len() - 3]).unwrap();
        assert!(assert_same_load(&id).salvage.is_some(), "a truncated save salvages");
        clear(&id);
    }

    /// Esc drops the job, early or on the very frame it lands: nothing is written and the slot
    /// loads as before.
    #[test]
    fn a_cancelled_load_leaves_the_save_untouched() {
        let id = slot("__entry_job_cancel__");
        edited_save(&id, 60);
        let path = save::file_path(&id);
        let bytes = std::fs::read(&path).unwrap();

        drop(Loading::load(id.clone(), RenderConfig::default()));
        let late = Loading::load(id.clone(), RenderConfig::default());
        while !late.done() {
            thread::sleep(std::time::Duration::from_millis(1));
        }
        drop(late);

        assert_eq!(std::fs::read(&path).unwrap(), bytes, "the save is untouched");
        assert!(!path.with_extension("save.bak").exists(), "no backup rotation");
        assert_same_load(&id);
        clear(&id);
    }

    #[test]
    fn a_joined_world_through_the_job_is_the_server_world() {
        use crate::net::server::{self, Config};
        let terrain = TerrainCfg { relief: 150, ..TerrainCfg::default() }.clamp();
        let config = Config { seed: 77, worldgen: WorldgenKind::Diffusion, terrain, ..Config::default() };
        let handle = server::spawn(0, config).unwrap();
        let conn = crate::app::join_server("127.0.0.1", handle.addr().port(), "ada", "").expect("join");
        let spawn = conn.spawn();
        let loading = Loading::join(conn, RenderConfig::default(), Some("hi".into()), true);
        let Loading::Join { job, conn, notice, hosted } = loading else { unreachable!() };
        let (world, player) = job.wait();
        assert_eq!((world.seed(), world.worldgen(), world.terrain_cfg()), (77, WorldgenKind::Diffusion, terrain));
        let mut inline = Player::new(spawn);
        inline.stand_in(recipe(77, WorldgenKind::Diffusion, terrain).world().gravity_at(spawn).accel);
        assert_eq!(pose(&player), pose(&inline));
        assert!(conn.is_alive());
        assert_eq!((notice.as_deref(), hosted), (Some("hi"), true));
        drop(conn);
        handle.stop();
    }
}
