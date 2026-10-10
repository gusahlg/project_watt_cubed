//! Core screens that exist even without a menu mod: Mods and Settings.
//! The start screen (main/load/host/join) lives in the Start mod.
use crate::menu::{Command, Ctx, Framed, Menu, Msg, Notice, Row, Style, ValueView, View};
use crate::modding::PackageKind;
use crate::settings::{Category, MenuKind, SETTINGS};

// Mods menu.

/// The packages compiled into this build, read-only: what each is and whether the server
/// suspended it for this session. Nothing on it switches anything.
pub struct ModsMenu;

/// Persistent mods-screen notice: the build decides what is in.
pub const MODS_NOTICE: &str = "Mods are compiled in: change them with `pwc mod add` / `pwc mod remove` and `pwc build`.";

/// Marker shown on a lane whose visual group no installed, unsuspended package provides.
pub const UNAVAILABLE: &str = "(unavailable in this build)";

impl Menu for ModsMenu {
    type Action = usize;

    fn view(&self, ctx: &Ctx) -> View<usize> {
        let mut rows = Vec::new();
        for (i, package) in ctx.build.packages().iter().enumerate() {
            if package.kind == PackageKind::Bundle {
                continue;
            }
            let suspended = ctx.suspended.iter().any(|id| id == package.id);
            let label = if suspended {
                format!("{} {} (off on this server)", package.name, package.version)
            } else {
                format!("{} {}", package.name, package.version)
            };
            let row = Row::action(label, i);
            rows.push(if package.description.is_empty() { row } else { row.detail(package.description) });
        }
        View {
            title: "MODS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Esc back".to_string(),
            notice: Some(Notice::info(MODS_NOTICE.to_string())),
        }
    }

    fn update(&mut self, msg: Msg<usize>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

// Settings: a hub that pushes one page per category.

/// The settings hub: one row per [`Category`], each pushing its page.
pub struct SettingsHub;

impl Menu for SettingsHub {
    type Action = Category;

    fn view(&self, ctx: &Ctx) -> View<Category> {
        let rows = Category::ALL.iter().map(|(c, name)| Row::action(*name, *c)).collect();
        View {
            title: "SETTINGS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter open   Esc back".to_string(),
            notice: ctx.settings.vram_notice.as_ref().map(|t| Notice::info(t.clone())),
        }
    }

    fn update(&mut self, msg: Msg<Category>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Pick(cat) => Command::Push(Framed::boxed(SettingsPage::new(cat))),
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

/// Settings page for a category; field semantics stay table-side.
pub struct SettingsPage {
    category: Category,
}

impl SettingsPage {
    pub fn new(category: Category) -> Self {
        Self { category }
    }
}

impl Menu for SettingsPage {
    type Action = usize;

    fn view(&self, ctx: &Ctx) -> View<usize> {
        let title = Category::ALL
            .iter()
            .find(|(c, _)| *c == self.category)
            .map_or("SETTINGS", |(_, n)| n)
            .to_uppercase();
        let rows = SETTINGS
            .iter()
            .enumerate()
            .filter(|(_, s)| s.category() == self.category)
            .map(|(i, s)| {
                let stored = s.show(ctx.settings);
                let stripped = ctx.visuals.strips(s.key());
                let shown = if stripped { format!("{stored} {UNAVAILABLE}") } else { stored.clone() };
                let value = match s.menu_kind() {
                    MenuKind::Toggle if !stripped => ValueView::Toggle(stored == "On"),
                    MenuKind::Toggle | MenuKind::Choice => ValueView::Choice(shown),
                    MenuKind::Bar => ValueView::Bar { t: s.fraction(ctx.settings), label: shown },
                };
                Row::value(s.label(), value, i)
            })
            .collect();
        View {
            title,
            style: Style::Panel,
            rows,
            default: None,
            hint: "Left/Right or h/l change   Esc back".to_string(),
            notice: ctx.settings.vram_notice.as_ref().map(|t| Notice::info(t.clone())),
        }
    }

    fn update(&mut self, msg: Msg<usize>, ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(i, dir) => {
                SETTINGS[i].step(ctx.settings, dir.delta());
                Command::Stay
            }
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modding::{BuildInfo, GameBuild, PackageInfo, VisualMask};
    use crate::render_config::VisualGroup;
    use crate::session::Session;
    use crate::settings::Settings;

    const fn package(id: &'static str, name: &'static str, kind: PackageKind) -> PackageInfo {
        PackageInfo { id, name, version: "1.0.0", description: "", kind, dependencies: &[], register: None }
    }

    static PACKAGES: &[PackageInfo] = &[
        PackageInfo { description: "Developer tools.", ..package("pwc.dev-toolkit", "Developer Toolkit", PackageKind::Mod) },
        package("pwc.ui-kit", "UI kit", PackageKind::Library),
        package("pwc.essentials", "Essentials", PackageKind::Bundle),
    ];

    #[test]
    fn mods_menu_lists_the_build_read_only_with_suspended_packages_marked() {
        let build: BuildInfo = GameBuild::from_static("sha256:00", PACKAGES).info().clone();
        let mut settings = Settings::default();
        let session = Session::default();
        let suspended = ["pwc.dev-toolkit".to_string()];
        let mut ctx = Ctx { build: &build, suspended: &suspended, ..Ctx::bare(&mut settings, &session) };
        let view = ModsMenu.view(&ctx);
        let labels: Vec<&str> = view.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["Developer Toolkit 1.0.0 (off on this server)", "UI kit 1.0.0"], "bundles are not rows");
        assert_eq!(view.rows[0].detail.as_deref(), Some("Developer tools."));
        assert_eq!(view.notice.map(|n| n.text).as_deref(), Some(MODS_NOTICE));
        assert!(matches!(ModsMenu.update(Msg::Pick(0), &mut ctx), Command::Stay), "picking a row changes nothing");
        assert!(matches!(ModsMenu.update(Msg::Back, &mut ctx), Command::Pop));
    }

    /// The lanes a settings page marks are exactly the ones the renderer strips, and the marker
    /// names no mod.
    #[test]
    fn settings_rows_mark_exactly_the_lanes_the_renderer_strips() {
        let mask = VisualMask::of([VisualGroup::Atmosphere, VisualGroup::Lighting]);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx { visuals: mask, ..Ctx::bare(&mut settings, &session) };
        let mut lanes = 0;
        for (category, _) in Category::ALL {
            for row in SettingsPage::new(category).view(&ctx).rows {
                let key = SETTINGS[row.tag.expect("settings rows are selectable")].key();
                let shown = match &row.kind {
                    crate::menu::RowKind::Value(ValueView::Choice(s) | ValueView::Bar { label: s, .. }) => s.as_str(),
                    _ => "",
                };
                assert_eq!(shown.ends_with(UNAVAILABLE), mask.strips(key), "{key}: {shown:?}");
                lanes += crate::render_config::lane_group(key).is_some() as usize;
            }
        }
        assert!(lanes > 0, "the settings pages list the visual lanes");
        let bloom = SettingsPage::new(Category::Video).view(&ctx).rows.into_iter().find(|r| r.label == "Bloom").expect("bloom row");
        assert!(matches!(bloom.kind, crate::menu::RowKind::Value(ValueView::Choice(ref s)) if s == "On (unavailable in this build)"));
    }
}
