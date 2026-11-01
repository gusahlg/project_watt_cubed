//! Command-line entry points a module registers. [`dispatch`] runs before a window or an audio
//! device opens.
use std::path::PathBuf;

use super::Modules;
use super::build::{BuildInfo, GameBuild};
use crate::settings::{Options, Settings};

/// A command a package offers on the executable. The first registration of `name` wins.
pub struct Cli {
    pub name: &'static str,
    pub about: &'static str,
    pub run: fn(&CliContext) -> i32,
}

/// What a registered command receives. `args` is everything after the command name.
pub struct CliContext {
    pub args: Vec<String>,
    pub data_dir: PathBuf,
    pub build: BuildInfo,
}

/// `Some(code)` when the first argument is a registered command, `help`, or `--commands`.
/// `None` starts the game. No arguments does not register modules.
pub(crate) fn dispatch(build: &GameBuild, args: &[String]) -> Option<i32> {
    let name = args.get(1).map(String::as_str)?;
    let modules = load(build);
    if name == "help" || name == "--commands" {
        print!("{}", command_text(&modules));
        let _ = std::io::Write::flush(&mut std::io::stdout());
        return Some(0);
    }
    let cli = modules.clis().iter().find(|cli| cli.name == name)?;
    let ctx = CliContext {
        args: args[2..].to_vec(),
        data_dir: crate::paths::Paths::get().data.clone(),
        build: build.info().clone(),
    };
    Some((cli.run)(&ctx))
}

fn load(build: &GameBuild) -> Modules {
    crate::paths::Paths::init(None);
    let mut options = Options::new();
    let mut modules = Modules::from_build(build, &mut options);
    let _settings = Settings::load(&mut options);
    modules.options_changed(&options);
    modules
}

fn command_text(modules: &Modules) -> String {
    let mut out = String::from("commands:\n");
    for cli in modules.clis() {
        out.push_str("  ");
        out.push_str(cli.name);
        out.push_str("  ");
        out.push_str(cli.about);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::modules::{GameBuild, Module, ModuleDescriptor, ModuleRegistrar};

    thread_local! {
        static OPTIONS_HITS: Cell<u32> = const { Cell::new(0) };
        static HELP_RAN: Cell<bool> = const { Cell::new(false) };
    }

    struct Probe;

    impl Module for Probe {
        fn id(&self) -> &'static str {
            "cli.probe"
        }

        fn on_options(&mut self, _options: &Options) {
            OPTIONS_HITS.with(|hits| hits.set(hits.get() + 1));
        }
    }

    fn register(registrar: &mut ModuleRegistrar) {
        registrar.add(Probe);
        registrar.cli(Cli {
            name: "probe",
            about: "a probe command",
            run: probe,
        });
        registrar.cli(Cli {
            name: "help",
            about: "not this one",
            run: |_| {
                HELP_RAN.with(|ran| ran.set(true));
                9
            },
        });
    }

    fn probe(ctx: &CliContext) -> i32 {
        let dir_ok = ctx.data_dir == crate::paths::Paths::get().data;
        let pkg_ok = ctx.build.package("cli.probe").is_some();
        if ctx.args == ["one"] && dir_ok && pkg_ok {
            7
        } else {
            1
        }
    }

    fn build() -> GameBuild {
        GameBuild::new().with_module(ModuleDescriptor {
            id: "cli.probe",
            name: "Probe",
            version: "1.0.0",
            register,
        })
    }

    fn args(rest: &[&str]) -> Vec<String> {
        std::iter::once("pwc-game")
            .chain(rest.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn dispatch_picks_a_cli_and_reserves_help() {
        OPTIONS_HITS.with(|hits| hits.set(0));
        HELP_RAN.with(|ran| ran.set(false));
        let build = build();

        assert_eq!(dispatch(&build, &args(&[])), None);
        assert_eq!(
            OPTIONS_HITS.with(|hits| hits.get()),
            0,
            "no arguments does not register"
        );

        OPTIONS_HITS.with(|hits| hits.set(0));
        assert_eq!(dispatch(&build, &args(&["nope"])), None);
        assert!(
            OPTIONS_HITS.with(|hits| hits.get()) >= 2,
            "an unknown argument still loads options"
        );

        assert_eq!(dispatch(&build, &args(&["--commands"])), Some(0));
        assert_eq!(dispatch(&build, &args(&["help"])), Some(0));
        assert!(!HELP_RAN.with(|ran| ran.get()), "help is reserved");
        let text = command_text(&load(&build));
        assert!(text.contains("  probe  a probe command\n"), "{text}");
        assert!(text.starts_with("commands:\n"), "{text}");

        OPTIONS_HITS.with(|hits| hits.set(0));
        assert_eq!(dispatch(&build, &args(&["probe", "one"])), Some(7));
        assert!(OPTIONS_HITS.with(|hits| hits.get()) >= 2);
    }
}
