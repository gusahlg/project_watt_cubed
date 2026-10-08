//! The console's submitted lines: chat in multiplayer, commands the mods handle, and the core's
//! follow-up on what a command changed.
use voxel_engine::Engine;

use super::Game;
use crate::audio::{GameEvent, SoundSystem};
use crate::console;
use crate::modding::{Command, CommandContext, Mods};
use crate::net::chat;
use crate::settings::Settings;

impl Game {
    /// Handle one submitted console line (see [`run_line`](Self::run_line)). A command that edits
    /// settings (`/gfx`, the audio rows) goes through the one application path, is marked for
    /// saving and re-mixes the audio, only when something actually changed.
    pub(super) fn submit_line(
        &mut self,
        line: String,
        eng: &mut Engine,
        settings: &mut Settings,
        sound: &mut SoundSystem,
        events: &mut Vec<GameEvent>,
        mods: &mut Mods,
    ) {
        if self.run_line(line, settings, events, mods) {
            self.apply_settings(eng, settings);
            self.settings_dirty = true;
            sound.set_mix(settings.mix_change());
        }
    }

    /// A submitted console line, short of the engine. A leading `/` is a command, except that in
    /// multiplayer `/op <secret>` is the server's operator login, sent as chat and not echoed; in
    /// multiplayer any other line is chat (a leading `!` sends it to global chat), while in
    /// singleplayer it stays a command. The first enabled mod that knows the command runs it, then
    /// the core follows up on the state it changed. Returns whether it changed the settings.
    fn run_line(&mut self, line: String, settings: &mut Settings, events: &mut Vec<GameEvent>, mods: &mut Mods) -> bool {
        if (line == "/op" || line.starts_with("/op "))
            && let Some(net) = &mut self.net
        {
            net.send_chat(chat::GLOBAL, &line);
            return false;
        }
        if !line.starts_with('/')
            && let Some(net) = &mut self.net
        {
            let (channel, text) = match line.strip_prefix('!') {
                Some(rest) => (chat::GLOBAL, rest.trim().to_string()),
                None => (chat::LOCAL, line),
            };
            if !text.is_empty() {
                // The server echoes chat back to us, so we don't print it here.
                net.send_chat(channel, &text);
            }
            return false;
        }
        self.console.echo(&line);
        let mut parts = line.strip_prefix('/').unwrap_or(line.as_str()).split_whitespace();
        let Some(cmd) = parts.next() else {
            return false;
        };
        let args: Vec<&str> = parts.collect();
        let commands: Vec<Command> = mods.commands().copied().collect();
        let before = settings.clone();
        let day_before = self.sky.clock.day();
        let day_len_before = self.sky.day_length;
        let pos_before = self.player.position;
        let mut ctx = CommandContext::new(&mut self.player, &mut self.world, settings, &mut self.sky);
        ctx.visuals = self.visual_mask;
        ctx.networked = self.net.is_some();
        ctx.commands = &commands;
        let out = mods.run_command(&mut ctx, cmd, &args);
        // The test cue is a fact for the sounds mod (it runs this frame even though the console
        // owns input).
        if ctx.voice_test {
            events.push(GameEvent::VoiceTest);
        }
        // Each output line already carries its role (output vs rejection): just show them.
        for line in out.unwrap_or_else(|| vec![console::unknown_command(cmd, &commands)]) {
            self.console.push(line);
        }
        // A `/time` change is shared: tell the server so every client's clock
        // follows (the server relays it and hands it to future joiners).
        if self.sky.clock.day() != day_before
            && let Some(net) = &mut self.net
        {
            net.send_set_time(self.sky.clock.day() as f32);
        }
        // The cycle LENGTH is server-owned in multiplayer: a local change
        // would silently desync every clock's advance rate.
        if self.sky.day_length != day_len_before && self.net.is_some() {
            self.sky.day_length = day_len_before;
            self.console
                .print("* day length is set by the server".to_string());
        }
        // A moved player is a position discontinuity: ordinary moves are envelope-
        // checked server-side, so report it as an explicit teleport (the
        // server may still snap us back if teleports are disabled) — and
        // stream out of band so the destination doesn't wait on `stream_hz`.
        if self.player.position != pos_before {
            self.force_stream = true;
            if let Some(net) = &mut self.net {
                net.send_teleport(self.player.position);
            }
        }
        *settings != before
    }
}

#[cfg(test)]
mod tests {
    use super::Game;
    use crate::audio::GameEvent;
    use crate::game::tests::{game, probe_mods};
    use crate::modding::{Command, Mod, Mods};
    use crate::settings::Settings;
    use crate::ui::Role;
    use voxel_engine::DVec3;

    /// Run `line` as the console would; the scrollback's newest line and whether settings changed.
    fn run(game: &mut Game, mods: &mut Mods, settings: &mut Settings, line: &str) -> (String, Role, bool, Vec<GameEvent>) {
        let mut events = Vec::new();
        let changed = game.run_line(line.to_string(), settings, &mut events, mods);
        let last = game.console.last().expect("the console printed a line");
        let role = last.spans().next().expect("a line has a span").role;
        (last.text().to_string(), role, changed, events)
    }

    #[test]
    fn a_command_no_mod_handles_prints_the_hint() {
        let mut game = game();
        let mut settings = Settings::default();
        let (text, role, changed, _) = run(&mut game, &mut Mods::empty(), &mut settings, "/tp 1 2 3");
        assert_eq!(text, "unknown command 'tp' - commands come from mods such as the Developer Toolkit");
        assert_eq!(role, Role::Danger);
        assert!(!changed);
        assert_eq!(game.player.position, DVec3::new(0.5, 80.0, 0.5), "the base game has no /tp");
        let (text, ..) = run(&mut game, &mut probe_mods(), &mut settings, "/nope");
        assert!(text.starts_with("unknown command 'nope'"), "{text}");
        // With a mod that offers /help, the hint points there.
        struct Helper;
        impl Mod for Helper {
            fn name(&self) -> &str {
                "helper"
            }
            fn id(&self) -> &'static str {
                "helper"
            }
            fn commands(&self) -> &[Command] {
                &[Command { name: "help", args: "", help: "list commands" }]
            }
        }
        let mut mods = Mods::empty();
        mods.install(Box::new(Helper), true);
        let (text, ..) = run(&mut game, &mut mods, &mut settings, "/nope");
        assert_eq!(text, "unknown command 'nope' - type '/help'");
    }

    #[test]
    fn a_mod_command_edits_the_player_world_and_settings() {
        let (mut game, mut mods, mut settings) = (game(), probe_mods(), Settings::default());
        let blocks = game.world.registry().block_count();
        let (text, role, changed, _) = run(&mut game, &mut mods, &mut settings, "/probe intern");
        assert_eq!((text.as_str(), role), ("1 command(s)", Role::Dim), "the context lists every enabled command");
        assert!(!changed);
        assert_eq!(game.world.registry().block_count(), blocks + 1);
        let fov = settings.fov;
        let (.., changed, _) = run(&mut game, &mut mods, &mut settings, "probe fov");
        assert_eq!(settings.fov, fov + 5.0);
        assert!(changed, "changed settings are applied, saved and re-mixed by the caller");
        game.force_stream = false;
        let (.., changed, _) = run(&mut game, &mut mods, &mut settings, "/probe move");
        assert_eq!(game.player.position.x, 10.5);
        assert!(!changed);
        assert!(game.force_stream, "a moved player streams its destination at once (and is reported as a teleport)");
    }

    #[test]
    fn a_command_asking_for_the_voice_test_plays_the_cue() {
        let (mut game, mut mods, mut settings) = (game(), probe_mods(), Settings::default());
        let (.., events) = run(&mut game, &mut mods, &mut settings, "/probe");
        assert!(events.is_empty());
        let (.., events) = run(&mut game, &mut mods, &mut settings, "/probe voice");
        assert!(matches!(events.as_slice(), [GameEvent::VoiceTest]));
    }
}
