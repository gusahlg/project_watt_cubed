//! Core screens that must exist even if every mod is off: Mods and Settings.
//! The start screen (main/load/host/join) lives in the Start mod.
use crate::menu::{
    AppEffect, Command, Ctx, Dir, Framed, Menu, Msg, Notice, Row, Style, ValueView, View,
};
use crate::modding::{annotate_setting, VisualMask};
use crate::settings::{Category, MenuKind, SETTINGS};

/// The renderer's mask ([`Mods::visual_mask`](crate::modding::Mods::visual_mask)) from the snapshot.
fn visual_mask(ctx: &Ctx) -> VisualMask {
    VisualMask::of(ctx.mods.iter().filter(|m| m.enabled).filter_map(|m| m.visual_group))
}

// Mods menu.

/// Grouped toggle rows per installed mod, plus knobs for mods that have them.
pub struct ModsMenu;

/// Persistent mods-screen notice: choices hit disk now; they apply only after
/// a rebuild because mods are compiled into the binary.
const MODS_NOTICE: &str = "Choices are saved at once and apply from the next world (worldgen: the next new world).\nMods are compiled in: add or remove them with `pwc mod add` / `pwc mod remove`.";

#[derive(Clone, Copy)]
pub enum ModsAction {
    Toggle(usize),
    Knob { mod_index: usize, knob: usize },
    SetGroup { id: &'static str, on: bool },
    /// Shown as "off (server)". Picking it does nothing.
    ServerOff,
}

impl Menu for ModsMenu {
    type Action = ModsAction;

    fn view(&self, ctx: &Ctx) -> View<ModsAction> {
        let mut rows = Vec::new();
        let mut placed = vec![false; ctx.mods.len()];
        // Groups in the order their first member appears (package registration order).
        let mut groups: Vec<crate::modding::Group> = Vec::new();
        for g in ctx.mods.iter().filter_map(|m| m.group) {
            if !groups.iter().any(|seen| seen.id == g.id) {
                groups.push(g);
            }
        }
        for g in groups {
            let members: Vec<usize> = ctx
                .mods
                .iter()
                .enumerate()
                .filter(|(_, m)| m.group.is_some_and(|mg| mg.id == g.id))
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }
            for &i in &members {
                placed[i] = true;
            }
            rows.push(Row::heading(g.name).detail(g.description));
            let all_on = members.iter().all(|&i| ctx.mods[i].enabled);
            rows.push(Row::value(
                "  Enable all / Disable all",
                ValueView::Toggle(all_on),
                ModsAction::SetGroup {
                    id: g.id,
                    on: !all_on,
                },
            ));
            for i in members {
                push_mod_rows(&mut rows, i, &ctx.mods[i]);
            }
        }
        let other: Vec<usize> = placed
            .iter()
            .enumerate()
            .filter(|(_, seen)| !**seen)
            .map(|(i, _)| i)
            .collect();
        if !other.is_empty() {
            rows.push(Row::heading("Other"));
            for i in other {
                push_mod_rows(&mut rows, i, &ctx.mods[i]);
            }
        }
        View {
            title: "MODS".to_string(),
            style: Style::Panel,
            rows,
            default: None,
            hint: "Enter toggle/cycle   Left/Right tune   Esc back".to_string(),
            notice: Some(match ctx.mods_save_error {
                Some(err) => Notice::error(format!("Could not save mod choices: {err}")),
                None => Notice::info(MODS_NOTICE.to_string()),
            }),
        }
    }

    fn update(&mut self, msg: Msg<ModsAction>, _ctx: &mut Ctx) -> Command {
        match msg {
            Msg::Step(ModsAction::Toggle(i), _) | Msg::Pick(ModsAction::Toggle(i)) => {
                Command::Effect(AppEffect::ToggleMod(i))
            }
            Msg::Step(ModsAction::SetGroup { id, on }, _)
            | Msg::Pick(ModsAction::SetGroup { id, on }) => {
                Command::Effect(AppEffect::SetGroup { id, on })
            }
            Msg::Step(ModsAction::Knob { mod_index, knob }, dir) => {
                Command::Effect(AppEffect::StepModKnob {
                    mod_index,
                    knob,
                    delta: dir.delta(),
                })
            }
            Msg::Pick(ModsAction::Knob { mod_index, knob }) => {
                Command::Effect(AppEffect::StepModKnob {
                    mod_index,
                    knob,
                    delta: Dir::Next.delta(),
                })
            }
            Msg::Step(ModsAction::ServerOff, _) | Msg::Pick(ModsAction::ServerOff) => Command::Stay,
            Msg::Back => Command::Pop,
            _ => Command::Stay,
        }
    }
}

fn push_mod_rows(rows: &mut Vec<Row<ModsAction>>, i: usize, m: &crate::menu::ModRow) {
    if m.server_off {
        rows.push(
            Row::value(
                format!("  {}", m.name),
                ValueView::Choice("off (server)".to_string()),
                ModsAction::ServerOff,
            )
            .detail(m.description.clone()),
        );
        return;
    }
    rows.push(
        Row::value(
            format!("  {}", m.name),
            ValueView::Toggle(m.enabled),
            ModsAction::Toggle(i),
        )
        .detail(m.description.clone()),
    );
    if !m.enabled {
        return;
    }
    for (k, (label, value, hint)) in m.knobs.iter().enumerate() {
        let mut row = Row::value(
            format!("    {label}"),
            ValueView::Choice(value.clone()),
            ModsAction::Knob {
                mod_index: i,
                knob: k,
            },
        );
        let mut detail = hint.clone();
        if m.worldgen {
            detail = if detail.is_empty() {
                "next new world".to_string()
            } else {
                format!("{detail} · next new world")
            };
        }
        if !detail.is_empty() {
            row = row.detail(detail);
        }
        rows.push(row);
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
        let mask = visual_mask(ctx);
        let rows = SETTINGS
            .iter()
            .enumerate()
            .filter(|(_, s)| s.category() == self.category)
            .map(|(i, s)| {
                let stored = s.show(ctx.settings);
                let shown = annotate_setting(stored.clone(), s.key(), mask);
                let value = match s.menu_kind() {
                    MenuKind::Toggle if shown == stored => ValueView::Toggle(stored == "On"),
                    MenuKind::Toggle | MenuKind::Choice => ValueView::Choice(shown),
                    MenuKind::Bar => {
                        ValueView::Bar { t: s.fraction(ctx.settings), label: shown }
                    }
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
    use crate::render_config::VisualGroup;
    use crate::session::Session;
    use crate::settings::Settings;

    fn ctx<'a>(settings: &'a mut Settings, session: &'a Session) -> Ctx<'a> {
        Ctx {
            settings,
            saves: &[],
            mods: &[],
            session,
            mods_save_error: None,
        }
    }

    #[test]
    fn mods_menu_notice_says_when_changes_apply() {
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = ctx(&mut settings, &session);
        let view = ModsMenu.view(&ctx);
        let notice = view.notice.expect("mods screen has a persistent notice");
        assert_eq!(notice.level, crate::menu::Level::Info);
        assert_eq!(notice.text, MODS_NOTICE);
        let lines: Vec<_> = notice.text.lines().collect();
        assert_eq!(
            lines,
            [
                "Choices are saved at once and apply from the next world (worldgen: the next new world).",
                "Mods are compiled in: add or remove them with `pwc mod add` / `pwc mod remove`.",
            ]
        );
    }

    #[test]
    fn enter_on_a_knob_cycles_forward_like_right() {
        let mut menu = ModsMenu;
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = ctx(&mut settings, &session);
        let pick = menu.update(
            Msg::Pick(ModsAction::Knob {
                mod_index: 3,
                knob: 1,
            }),
            &mut ctx,
        );
        let step = menu.update(
            Msg::Step(
                ModsAction::Knob {
                    mod_index: 3,
                    knob: 1,
                },
                Dir::Next,
            ),
            &mut ctx,
        );
        match (pick, step) {
            (
                Command::Effect(AppEffect::StepModKnob {
                    mod_index: p_m,
                    knob: p_k,
                    delta: p_d,
                }),
                Command::Effect(AppEffect::StepModKnob {
                    mod_index: s_m,
                    knob: s_k,
                    delta: s_d,
                }),
            ) => {
                assert_eq!((p_m, p_k, p_d), (s_m, s_k, s_d));
                assert_eq!(p_d, 1);
            }
            _ => panic!("expected StepModKnob from both Enter and Right"),
        }
    }

    #[test]
    fn settings_row_uses_the_same_forced_off_marker_as_gfx() {
        let mut settings = Settings::default();
        let session = Session::default();
        let mods = vec![crate::menu::ModRow {
            name: "Post".into(),
            description: String::new(),
            enabled: false,
            knobs: vec![],
            visual_group: Some(VisualGroup::Post),
            worldgen: false,
            group: None,
            server_off: false,
        }];
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &mods,
            session: &session,
            mods_save_error: None,
        };
        let view = SettingsPage::new(Category::Video).view(&ctx);
        let bloom = view
            .rows
            .iter()
            .find(|r| r.label == "Bloom")
            .expect("bloom row");
        let marker = crate::modding::forced_off_marker("Post");
        match &bloom.kind {
            crate::menu::RowKind::Value(ValueView::Choice(s)) => {
                assert!(s.contains(&marker), "settings value {s:?} must include {marker}");
            }
            _ => panic!("expected annotated choice for a stripped bloom row"),
        }
    }

    /// Two mods own Post and the later one is off. The renderer keeps Post (one owner is on), so
    /// no settings row may say a mod forced its lane off; every other row matches the renderer too.
    #[test]
    fn settings_rows_mark_exactly_the_lanes_the_renderer_strips() {
        use crate::modding::testing::Stub;
        let mut mods = crate::modding::Mods::empty();
        mods.install(Box::new(Stub::new("post").visual(VisualGroup::Post)), true);
        mods.install(Box::new(Stub::new("post_extra").visual(VisualGroup::Post)), false);
        let mask = mods.visual_mask();
        assert!(mask.post && !mask.atmosphere && !mask.lighting, "{mask:?}");
        let snap = crate::menu::ModRow::snapshot(&mods);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
            mods_save_error: None,
        };
        let mut lanes = 0;
        for (category, _) in Category::ALL {
            for row in SettingsPage::new(category).view(&ctx).rows {
                let key = SETTINGS[row.tag.expect("settings rows are selectable")].key();
                let shown = match &row.kind {
                    crate::menu::RowKind::Value(ValueView::Choice(s) | ValueView::Bar { label: s, .. }) => s.as_str(),
                    _ => "",
                };
                let marked = shown.contains("(off: ");
                assert_eq!(marked, mask.forced_off(key).is_some(), "{key}: {shown:?}");
                lanes += crate::render_config::lane_group(key).is_some() as usize;
            }
        }
        assert!(lanes > 0, "the settings pages list the visual lanes");
    }

    #[test]
    fn mods_menu_nests_essentials_under_group_header() {
        let installed = crate::modding::testing::standard();
        let snap = crate::menu::ModRow::snapshot(&installed);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
            mods_save_error: None,
        };
        let view = ModsMenu.view(&ctx);
        assert!(matches!(view.rows[0].kind, crate::menu::RowKind::Heading));
        assert_eq!(view.rows[0].label, "Essentials");
        assert_eq!(
            view.rows[0].detail.as_deref(),
            Some("Menus, inventory, looks, names and worldgen.")
        );
        assert!(view.rows[0].tag.is_none(), "group header is not selectable");
        assert_eq!(view.rows[1].label.trim(), "Enable all / Disable all");
        assert!(
            view.rows[2].label.contains("Menus"),
            "first member is indented under the group: {:?}",
            view.rows[2].label
        );
        assert!(
            !view.rows.iter().any(|r| r.label == "Other"),
            "every built-in is an Essential: no Other section"
        );
        let names: Vec<&str> = view
            .rows
            .iter()
            .filter(|r| matches!(r.kind, crate::menu::RowKind::Value(ValueView::Toggle(_))))
            .map(|r| r.label.trim())
            .collect();
        assert_eq!(
            names,
            [
                "Enable all / Disable all",
                "Menus",
                "Start",
                "Inventory",
                "Hotbar",
                "Atmosphere",
                "Post",
                "Lighting",
                "Neural textures",
                "Material names",
                "InfiniteDiffusion"
            ]
        );
    }

    #[test]
    fn mods_menu_lists_ungrouped_under_other() {
        let extra = crate::menu::ModRow {
            name: "Extra".into(),
            description: "future external".into(),
            enabled: true,
            knobs: vec![],
            visual_group: None,
            worldgen: false,
            group: None,
            server_off: false,
        };
        let installed = crate::modding::testing::standard();
        let mut snap = crate::menu::ModRow::snapshot(&installed);
        snap.push(extra);
        let mut settings = Settings::default();
        let session = Session::default();
        let ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &snap,
            session: &session,
            mods_save_error: None,
        };
        let view = ModsMenu.view(&ctx);
        let other = view
            .rows
            .iter()
            .position(|r| r.label == "Other" && matches!(r.kind, crate::menu::RowKind::Heading))
            .expect("Other section");
        assert!(
            view.rows[other + 1].label.contains("Extra"),
            "an ungrouped mod is listed under Other"
        );
        assert!(
            view.rows.iter().any(|r| r.label.contains("Extra")),
            "appended ungrouped mod is listed under Other"
        );
    }

    #[test]
    fn group_toggle_row_emits_set_group() {
        let mut menu = ModsMenu;
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = ctx(&mut settings, &session);
        match menu.update(
            Msg::Pick(ModsAction::SetGroup {
                id: crate::modding::ESSENTIALS,
                on: false,
            }),
            &mut ctx,
        ) {
            Command::Effect(AppEffect::SetGroup { id, on }) => {
                assert_eq!(id, crate::modding::ESSENTIALS);
                assert!(!on);
            }
            _ => panic!("expected SetGroup"),
        }
    }

    #[test]
    fn server_off_mod_shows_off_server_and_ignores_the_pick() {
        let held = crate::menu::ModRow {
            name: "Developer Toolkit".into(),
            description: "commands".into(),
            enabled: false,
            knobs: vec![],
            visual_group: None,
            worldgen: false,
            group: None,
            server_off: true,
        };
        let mut settings = Settings::default();
        let session = Session::default();
        let mods = vec![held];
        let mut ctx = Ctx {
            settings: &mut settings,
            saves: &[],
            mods: &mods,
            session: &session,
            mods_save_error: None,
        };
        let view = ModsMenu.view(&ctx);
        let row = view.rows.iter().find(|r| r.label.contains("Developer Toolkit")).expect("row");
        match &row.kind {
            crate::menu::RowKind::Value(ValueView::Choice(value)) => {
                assert_eq!(value, "off (server)");
            }
            _ => panic!("expected off (server)"),
        }
        let mut menu = ModsMenu;
        assert!(matches!(
            menu.update(Msg::Pick(ModsAction::ServerOff), &mut ctx),
            Command::Stay
        ));
    }

    #[test]
    fn mods_menu_shows_save_error_instead_of_info_notice() {
        let mut settings = Settings::default();
        let session = Session::default();
        let mut ctx = ctx(&mut settings, &session);
        ctx.mods_save_error = Some("permission denied");
        let view = ModsMenu.view(&ctx);
        let notice = view.notice.expect("error notice");
        assert_eq!(notice.level, crate::menu::Level::Error);
        assert_eq!(
            notice.text,
            "Could not save mod choices: permission denied"
        );
    }
}
