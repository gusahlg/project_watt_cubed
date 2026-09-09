//! Minimal core start interface: the actions the core can perform and the
//! read-only facts a start screen may show.
//!
//! This is the public start contract — serialisable plain data, no engine or
//! frame types. A start screen (the default-enabled Start mod, or the core
//! fallback when that mod is off) builds [`MenuModel`]s from [`StartFacts`]
//! and returns [`StartAction`]s. App routes those through the same
//! [`AppEffect`](super::AppEffect) `handle_effect` path as before.
//!
//! First enabled mod that returns `Some` from [`Mod::start_screen`](crate::mods::Mod::start_screen)
//! wins; the core fallback is a plain list so the game is always startable.

use crate::save::{Slot, SlotId};
use crate::session::Session;

use super::theme::{MenuTheme, PresentedRow, PresentedView};
use super::{
    drive, AppEffect, Command, Cursor, Intent, Msg, Notice, Row, RowKind, Screen, Style, View,
};
use voxel_engine::Frame;

/// Package version a start screen may show.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Actions the core can perform from a start screen, with the data each needs.
/// Plain serialisable data: no engine or frame types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartAction {
    NewWorld,
    Load(SlotId),
    Host(HostInfo),
    Join(JoinInfo),
    Settings,
    Mods,
    Quit,
}

impl From<StartAction> for AppEffect {
    fn from(action: StartAction) -> Self {
        match action {
            StartAction::NewWorld => AppEffect::NewWorld,
            StartAction::Load(id) => AppEffect::Load(id),
            StartAction::Host(info) => AppEffect::Host(info),
            StartAction::Join(info) => AppEffect::Join(info),
            StartAction::Settings => AppEffect::Settings,
            StartAction::Mods => AppEffect::Mods,
            StartAction::Quit => AppEffect::Quit,
        }
    }
}

/// Host form result: port, optional password, player name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostInfo {
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// Join form result: address, port, password, player name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinInfo {
    pub host: String,
    pub port: u16,
    pub password: String,
    pub name: String,
}

/// Read-only facts a start screen may show. Plain data; no engine or frame types.
pub struct StartFacts<'a> {
    pub saves: &'a [Slot],
    pub session: &'a Session,
    pub version: &'a str,
    /// Whether this process currently has an integrated server running.
    pub hosting: bool,
    pub notice: Option<&'a str>,
}

/// One start-screen page as data. Core presents it through the theme; the
/// start screen owns meaning and interaction.
pub struct MenuModel {
    pub title: String,
    pub style: Style,
    pub rows: Vec<ModelRow>,
    pub hint: String,
    pub notice: Option<Notice>,
    pub selected: usize,
}

/// One row of a [`MenuModel`].
pub struct ModelRow {
    pub label: String,
    pub detail: Option<String>,
    pub kind: RowKind,
    pub selectable: bool,
    /// Core action this row commits on pick, if picking it is a core action.
    /// Host/Join on the default main menu are `None` — they open forms first.
    pub action: Option<StartAction>,
}

impl MenuModel {
    pub fn from_view<A: Copy>(
        view: &View<A>,
        selected: usize,
        mut to_action: impl FnMut(&A) -> Option<StartAction>,
    ) -> Self {
        let rows = view
            .rows
            .iter()
            .map(|r| ModelRow {
                label: r.label.clone(),
                detail: r.detail.clone(),
                kind: r.kind.clone(),
                selectable: r.tag.is_some(),
                action: r.tag.as_ref().and_then(&mut to_action),
            })
            .collect();
        Self {
            title: view.title.clone(),
            style: view.style.clone(),
            rows,
            hint: view.hint.clone(),
            notice: view.notice.clone(),
            selected,
        }
    }

    pub fn to_presented(&self, scale: f32) -> PresentedView {
        PresentedView {
            title: self.title.clone(),
            style: self.style.clone(),
            rows: self
                .rows
                .iter()
                .map(|r| PresentedRow {
                    label: r.label.clone(),
                    detail: r.detail.clone(),
                    kind: r.kind.clone(),
                    selectable: r.selectable,
                })
                .collect(),
            scale,
            hint: self.hint.clone(),
            notice: self.notice.clone(),
        }
    }

    /// Core actions on selectable rows, in display order.
    pub fn actions(&self) -> Vec<StartAction> {
        self.rows.iter().filter_map(|r| r.action.clone()).collect()
    }

    pub fn labels(&self) -> Vec<&str> {
        self.rows.iter().map(|r| r.label.as_str()).collect()
    }
}

/// A start screen: builds [`MenuModel`]s from [`StartFacts`] and returns
/// [`StartAction`]s. Plain-data signatures only — the core owns presentation
/// (theme) and routing (`handle_effect`).
pub trait StartScreen {
    fn view(&self, facts: &StartFacts) -> MenuModel;
    fn update(&mut self, intents: &[Intent], facts: &StartFacts) -> Option<StartAction>;
}

/// Core fallback when no enabled mod returns a start screen. Plain list:
/// New world / Load (most recent) / Settings / Mods / Quit.
pub fn fallback(facts: &StartFacts) -> Box<dyn StartScreen> {
    Box::new(FallbackStart {
        cursor: Cursor::default(),
        notice: facts.notice.map(str::to_string),
    })
}

#[derive(Clone, Copy)]
enum FallbackAction {
    NewWorld,
    LoadRecent,
    Settings,
    Mods,
    Quit,
}

struct FallbackStart {
    cursor: Cursor,
    notice: Option<String>,
}

impl FallbackStart {
    fn page(&self, _facts: &StartFacts) -> View<FallbackAction> {
        View {
            title: "START".to_string(),
            style: Style::Panel,
            rows: vec![
                Row::action("New world", FallbackAction::NewWorld),
                Row::action("Load (most recent)", FallbackAction::LoadRecent),
                Row::action("Settings", FallbackAction::Settings),
                Row::action("Mods", FallbackAction::Mods),
                Row::action("Quit", FallbackAction::Quit),
            ],
            default: None,
            hint: "Up/Down select   Enter choose".to_string(),
            notice: self.notice.clone().map(Notice::info),
        }
    }

    fn commit(&self, action: FallbackAction, facts: &StartFacts) -> Option<StartAction> {
        match action {
            FallbackAction::NewWorld => Some(StartAction::NewWorld),
            FallbackAction::LoadRecent => facts
                .saves
                .first()
                .map(|slot| StartAction::Load(slot.id.clone())),
            FallbackAction::Settings => Some(StartAction::Settings),
            FallbackAction::Mods => Some(StartAction::Mods),
            FallbackAction::Quit => Some(StartAction::Quit),
        }
    }
}

impl StartScreen for FallbackStart {
    fn view(&self, facts: &StartFacts) -> MenuModel {
        let view = self.page(facts);
        let selected = self.cursor.resolved(&view);
        MenuModel::from_view(&view, selected, |a| self.commit(*a, facts))
    }

    fn update(&mut self, intents: &[Intent], facts: &StartFacts) -> Option<StartAction> {
        let view = self.page(facts);
        self.cursor.normalize(&view);
        match drive(intents, &view, &mut self.cursor) {
            Some(Msg::Pick(action)) => {
                self.notice = None;
                self.commit(action, facts)
            }
            Some(Msg::Back) => {
                self.notice = None;
                None
            }
            _ => None,
        }
    }
}

/// Root screen wrapping a start screen so it can live on the menu stack.
/// Settings/Mods come back as [`StartAction`]s; App pushes the core screens.
pub struct StartRoot {
    inner: Box<dyn StartScreen>,
    hosting: bool,
}

impl StartRoot {
    pub fn wrap(inner: Box<dyn StartScreen>, hosting: bool) -> Box<dyn Screen> {
        Box::new(Self { inner, hosting })
    }
}

impl Screen for StartRoot {
    fn update(&mut self, intents: &[Intent], ctx: &mut super::Ctx) -> Command {
        let facts = StartFacts {
            saves: ctx.saves,
            session: ctx.session,
            version: VERSION,
            hosting: self.hosting,
            notice: None,
        };
        match self.inner.update(intents, &facts) {
            Some(action) => Command::Effect(action.into()),
            None => Command::Stay,
        }
    }

    fn draw(&self, ctx: &super::Ctx, theme: &dyn MenuTheme, f: &mut Frame, w: i32, h: i32) {
        let facts = StartFacts {
            saves: ctx.saves,
            session: ctx.session,
            version: VERSION,
            hosting: self.hosting,
            notice: None,
        };
        let model = self.inner.view(&facts);
        let pv = model.to_presented(ctx.settings.menu_scale);
        theme.draw(f, &pv, model.selected, w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menu::Dir;
    use crate::save::{SaveMeta, Slot, SlotId};
    use crate::session::Session;

    fn slot(name: &str, playtime_secs: u64, edit_count: u32) -> Slot {
        Slot {
            id: SlotId::new(name).expect("legal slot id"),
            meta: Ok(SaveMeta {
                name: name.to_string(),
                seed: 1,
                created: 0,
                last_played: 10,
                playtime_secs,
                edit_count,
            }),
        }
    }

    fn facts<'a>(saves: &'a [Slot], session: &'a Session, notice: Option<&'a str>) -> StartFacts<'a> {
        StartFacts {
            saves,
            session,
            version: VERSION,
            hosting: false,
            notice,
        }
    }

    #[test]
    fn fallback_model_contains_the_core_actions() {
        let session = Session::default();
        let saves = [slot("alpha", 90, 3)];
        let f = facts(&saves, &session, None);
        let screen = fallback(&f);
        let model = screen.view(&f);
        assert_eq!(
            model.labels(),
            ["New world", "Load (most recent)", "Settings", "Mods", "Quit"]
        );
        assert_eq!(
            model.actions(),
            [
                StartAction::NewWorld,
                StartAction::Load(saves[0].id.clone()),
                StartAction::Settings,
                StartAction::Mods,
                StartAction::Quit,
            ]
        );
        assert!(matches!(model.style, Style::Panel));
    }

    #[test]
    fn fallback_load_without_saves_is_present_but_inert() {
        let session = Session::default();
        let f = facts(&[], &session, None);
        let mut screen = fallback(&f);
        let model = screen.view(&f);
        assert_eq!(
            model.labels(),
            ["New world", "Load (most recent)", "Settings", "Mods", "Quit"]
        );
        assert_eq!(
            model.actions(),
            [
                StartAction::NewWorld,
                StartAction::Settings,
                StartAction::Mods,
                StartAction::Quit,
            ]
        );
        assert!(model.rows[1].selectable);
        assert_eq!(model.rows[1].action, None);
        // Cursor starts on New world; step to Load and confirm.
        assert_eq!(
            screen.update(&[Intent::Nav(Dir::Next)], &f),
            None
        );
        assert_eq!(screen.update(&[Intent::Confirm], &f), None);
    }

    #[test]
    fn fallback_picks_map_to_start_actions() {
        let session = Session::default();
        let saves = [slot("alpha", 0, 0)];
        let f = facts(&saves, &session, Some("could not join: x"));
        let mut screen = fallback(&f);
        let model = screen.view(&f);
        assert_eq!(
            model.notice.as_ref().map(|n| n.text.as_str()),
            Some("could not join: x")
        );
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::NewWorld)
        );
        assert_eq!(screen.view(&f).notice.is_some(), false, "any pick clears notice");
        assert_eq!(
            screen.update(&[Intent::Nav(Dir::Next)], &f),
            None
        );
        assert_eq!(
            screen.update(&[Intent::Confirm], &f),
            Some(StartAction::Load(saves[0].id.clone()))
        );
    }

    #[test]
    fn start_action_converts_to_app_effect() {
        let id = SlotId::new("w").unwrap();
        assert!(matches!(
            AppEffect::from(StartAction::NewWorld),
            AppEffect::NewWorld
        ));
        assert!(matches!(
            AppEffect::from(StartAction::Load(id.clone())),
            AppEffect::Load(_)
        ));
        assert!(matches!(
            AppEffect::from(StartAction::Settings),
            AppEffect::Settings
        ));
        assert!(matches!(AppEffect::from(StartAction::Mods), AppEffect::Mods));
        assert!(matches!(AppEffect::from(StartAction::Quit), AppEffect::Quit));
    }
}
