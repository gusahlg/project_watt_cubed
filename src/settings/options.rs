//! The options registry: every value a player may tune, in one place.
//!
//! The core's own settings are one owner, `"core"`: the [`SETTINGS`] table over [`Settings`].
//! Every package can declare more with [`ModRegistrar::option`](crate::modding::ModRegistrar::option);
//! the core keeps their values here, persists them in `settings.cfg` as `<package-id>.<key>=`, and
//! any UI renders both through one [`OptionsView`]. Neither side knows the other: a package does
//! not depend on a settings menu, and a settings menu names no package.
//!
//! Reads are by [`OptionId`], an index the declaration returned, so no frame path compares
//! strings. A mod copies what it needs in [`Mod::on_options`](crate::modding::Mod::on_options),
//! which the core calls once the file is loaded and again whenever [`Options::revision`] moves.

use std::fmt::Write as _;

use super::{Category, MenuKind, Settings, SETTINGS};

/// The owner name of the core's own entries in an [`OptionsView`].
pub const CORE: &str = "core";

/// What values an option takes, and how a menu steps it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptionKind {
    /// On or off.
    Toggle,
    /// One of these labels (stored as its index; persisted as the label, lowercased).
    Choice(&'static [&'static str]),
    /// A whole percent from `min` to `max`, stepped by `step` from `min`.
    Percent { min: i32, max: i32, step: i32 },
    /// A number from `min` to `max`, stepped and rounded to `step`.
    Float { min: f32, max: f32, step: f32 },
}

/// One option's value. Its variant follows the [`OptionKind`]: `Bool` for a toggle, `Choice` for
/// a choice, `Int` for a percent, `Float` for a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptionValue {
    Bool(bool),
    Choice(usize),
    Int(i32),
    Float(f32),
}

/// When a changed value takes effect. Metadata for menus: the core stores the value at once
/// either way, and the owning mod reads it when it needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applies {
    /// At once.
    Live,
    /// When the next new world is made (worldgen).
    NextWorld,
}

/// One tunable a package declares. Build it with [`toggle`](Self::toggle),
/// [`choice`](Self::choice), [`percent`](Self::percent) or [`float`](Self::float), then
/// [`next_world`](Self::next_world) or [`legacy_key`](Self::legacy_key) as needed.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct OptionSpec {
    /// Persisted as `<package-id>.<key>` in `settings.cfg`. Unique within its package.
    pub key: &'static str,
    /// The menu row label.
    pub label: &'static str,
    /// The settings page it is listed on.
    pub page: Category,
    pub kind: OptionKind,
    pub default: OptionValue,
    pub applies: Applies,
    /// A `settings.cfg` key the value was saved under before it was this package's option. It is
    /// read when the file has no `<package-id>.<key>` line, and never written.
    pub legacy_key: Option<&'static str>,
}

impl OptionSpec {
    const fn new(key: &'static str, label: &'static str, page: Category, kind: OptionKind, default: OptionValue) -> Self {
        Self { key, label, page, kind, default, applies: Applies::Live, legacy_key: None }
    }

    /// An on/off option.
    pub const fn toggle(key: &'static str, label: &'static str, page: Category, default: bool) -> Self {
        Self::new(key, label, page, OptionKind::Toggle, OptionValue::Bool(default))
    }

    /// One of `choices`; `default` is an index into them.
    pub const fn choice(
        key: &'static str,
        label: &'static str,
        page: Category,
        choices: &'static [&'static str],
        default: usize,
    ) -> Self {
        Self::new(key, label, page, OptionKind::Choice(choices), OptionValue::Choice(default))
    }

    /// A whole percent in `min..=max`, stepped by `step` from `min`.
    pub const fn percent(
        key: &'static str,
        label: &'static str,
        page: Category,
        (min, max, step): (i32, i32, i32),
        default: i32,
    ) -> Self {
        Self::new(key, label, page, OptionKind::Percent { min, max, step }, OptionValue::Int(default))
    }

    /// A number in `min..=max`, rounded to `step`.
    pub const fn float(
        key: &'static str,
        label: &'static str,
        page: Category,
        (min, max, step): (f32, f32, f32),
        default: f32,
    ) -> Self {
        Self::new(key, label, page, OptionKind::Float { min, max, step }, OptionValue::Float(default))
    }

    /// The value takes effect in the next new world.
    pub const fn next_world(self) -> Self {
        Self { applies: Applies::NextWorld, ..self }
    }

    /// Also read `key` from `settings.cfg` when the file has no line of this option's own.
    pub const fn legacy_key(self, key: &'static str) -> Self {
        Self { legacy_key: Some(key), ..self }
    }

    /// `value` in this option's range and shape: clamped and snapped, or the default when its
    /// variant does not fit the kind.
    fn normalize(&self, value: OptionValue) -> OptionValue {
        match (self.kind, value) {
            (OptionKind::Toggle, OptionValue::Bool(on)) => OptionValue::Bool(on),
            (OptionKind::Choice(choices), OptionValue::Choice(i)) if i < choices.len() => OptionValue::Choice(i),
            (OptionKind::Percent { min, max, step }, OptionValue::Int(v)) => {
                let step = step.max(1);
                let snapped = min + (v.clamp(min, max) - min + step / 2) / step * step;
                OptionValue::Int(snapped.min(max))
            }
            (OptionKind::Float { min, max, step }, OptionValue::Float(v)) if v.is_finite() => {
                let rounded = if step > 0.0 { (v / step).round() * step } else { v };
                OptionValue::Float(round_to(rounded.clamp(min, max), decimals(step)))
            }
            _ if value_fits(self.kind, self.default) => self.default,
            _ => fallback(self.kind),
        }
    }
}

fn value_fits(kind: OptionKind, value: OptionValue) -> bool {
    matches!(
        (kind, value),
        (OptionKind::Toggle, OptionValue::Bool(_))
            | (OptionKind::Choice(_), OptionValue::Choice(_))
            | (OptionKind::Percent { .. }, OptionValue::Int(_))
            | (OptionKind::Float { .. }, OptionValue::Float(_))
    )
}

/// A value of the right variant for a spec whose own default does not fit its kind.
fn fallback(kind: OptionKind) -> OptionValue {
    match kind {
        OptionKind::Toggle => OptionValue::Bool(false),
        OptionKind::Choice(_) => OptionValue::Choice(0),
        OptionKind::Percent { min, .. } => OptionValue::Int(min),
        OptionKind::Float { min, .. } => OptionValue::Float(min),
    }
}

/// Decimal places that show a multiple of `step` exactly (0 to 6).
fn decimals(step: f32) -> usize {
    (0..6).find(|&d| {
        let scaled = step as f64 * 10f64.powi(d as i32);
        (scaled - scaled.round()).abs() < 1e-4
    })
    .unwrap_or(6)
}

fn round_to(v: f32, decimals: usize) -> f32 {
    let scale = 10f64.powi(decimals as i32);
    ((v as f64 * scale).round() / scale) as f32
}

/// An option's typed index: what [`ModRegistrar::option`](crate::modding::ModRegistrar::option)
/// returns, and what every read takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OptionId(u32);

impl OptionId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

struct Declared {
    owner: &'static str,
    spec: OptionSpec,
}

/// The values of every option the packages declared, and the persisted lines no package claims.
pub struct Options {
    declared: Vec<Declared>,
    values: Vec<OptionValue>,
    /// `<package>.<key>=` lines from `settings.cfg` that no declared option claims, in file order.
    /// They are written back, so a package removed and added again keeps its values.
    unknown: Vec<(String, String)>,
    revision: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

impl Options {
    /// No options declared.
    pub fn new() -> Self {
        Self { declared: Vec::new(), values: Vec::new(), unknown: Vec::new(), revision: 0 }
    }

    /// Declare `spec` for the package `owner` and return its id. Declaring the same key for the
    /// same owner again returns the first id. A persisted line already read for it applies.
    pub fn declare(&mut self, owner: &'static str, spec: OptionSpec) -> OptionId {
        if let Some(i) = self.declared.iter().position(|d| d.owner == owner && d.spec.key == spec.key) {
            return OptionId(i as u32);
        }
        let id = OptionId(self.declared.len() as u32);
        self.values.push(spec.normalize(spec.default));
        self.declared.push(Declared { owner, spec });
        if let Some(at) = self.unknown.iter().position(|(key, _)| key_is(key, owner, spec.key)) {
            let (_, text) = self.unknown.remove(at);
            self.parse_stored(id, &text);
        }
        self.revision = self.revision.wrapping_add(1);
        id
    }

    /// How many options are declared.
    pub fn len(&self) -> usize {
        self.declared.len()
    }

    pub fn is_empty(&self) -> bool {
        self.declared.is_empty()
    }

    /// Every declared option, in declaration (package registration) order.
    pub fn ids(&self) -> impl Iterator<Item = OptionId> + '_ {
        (0..self.declared.len() as u32).map(OptionId)
    }

    /// The package that declared `id`.
    pub fn owner(&self, id: OptionId) -> &'static str {
        self.declared[id.index()].owner
    }

    /// What `id` is.
    pub fn spec(&self, id: OptionId) -> &OptionSpec {
        &self.declared[id.index()].spec
    }

    /// The option `<package>.<key>` names, if declared.
    pub fn find(&self, full_key: &str) -> Option<OptionId> {
        self.declared
            .iter()
            .position(|d| key_is(full_key, d.owner, d.spec.key))
            .map(|i| OptionId(i as u32))
    }

    /// The value of `id`.
    pub fn value(&self, id: OptionId) -> OptionValue {
        self.values[id.index()]
    }

    /// A toggle's value (false for any other kind).
    pub fn bool(&self, id: OptionId) -> bool {
        matches!(self.value(id), OptionValue::Bool(true))
    }

    /// A choice's index (0 for any other kind).
    pub fn choice(&self, id: OptionId) -> usize {
        match self.value(id) {
            OptionValue::Choice(i) => i,
            _ => 0,
        }
    }

    /// A percent's value (0 for any other kind).
    pub fn int(&self, id: OptionId) -> i32 {
        match self.value(id) {
            OptionValue::Int(v) => v,
            _ => 0,
        }
    }

    /// A float's value (0 for any other kind).
    pub fn float(&self, id: OptionId) -> f32 {
        match self.value(id) {
            OptionValue::Float(v) => v,
            _ => 0.0,
        }
    }

    /// Changes whenever a value changes (through an [`OptionsView`] that includes the core's
    /// settings, too), so a mod can cache what it derives from its options.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Set `id` to `value`, normalised to its range. True when the stored value changed.
    pub fn set(&mut self, id: OptionId, value: OptionValue) -> bool {
        let value = self.spec(id).normalize(value);
        if self.values[id.index()] == value {
            return false;
        }
        self.values[id.index()] = value;
        self.touch();
        true
    }

    /// One menu step (`dir` −1 or +1): a toggle flips, a choice wraps, a number moves by its step
    /// and stops at its ends. True when the value changed.
    pub fn step(&mut self, id: OptionId, dir: i32) -> bool {
        let next = match (self.spec(id).kind, self.value(id)) {
            (OptionKind::Toggle, OptionValue::Bool(on)) => OptionValue::Bool(!on),
            (OptionKind::Choice(choices), OptionValue::Choice(i)) if !choices.is_empty() => {
                OptionValue::Choice((i as i32 + dir).rem_euclid(choices.len() as i32) as usize)
            }
            (OptionKind::Percent { step, .. }, OptionValue::Int(v)) => OptionValue::Int(v + dir.signum() * step),
            (OptionKind::Float { step, .. }, OptionValue::Float(v)) => OptionValue::Float(v + dir.signum() as f32 * step),
            (_, value) => value,
        };
        self.set(id, next)
    }

    /// The value as a menu shows it: `On`/`Off`, the choice's label, `125%`, `1.3`.
    pub fn show(&self, id: OptionId) -> String {
        let mut out = String::new();
        self.write_value(id, &mut out, true);
        out
    }

    /// Where the value sits in its range, `0..=1` (a menu bar); 0 for toggles and choices.
    pub fn fraction(&self, id: OptionId) -> f32 {
        match (self.spec(id).kind, self.value(id)) {
            (OptionKind::Percent { min, max, .. }, OptionValue::Int(v)) if max > min => {
                (v - min) as f32 / (max - min) as f32
            }
            (OptionKind::Float { min, max, .. }, OptionValue::Float(v)) if max > min => (v - min) / (max - min),
            _ => 0.0,
        }
    }

    /// How a menu shows `id`: a toggle, a choice, or a bar for numbers.
    pub fn menu_kind(&self, id: OptionId) -> MenuKind {
        match self.spec(id).kind {
            OptionKind::Toggle => MenuKind::Toggle,
            OptionKind::Choice(_) => MenuKind::Choice,
            OptionKind::Percent { .. } | OptionKind::Float { .. } => MenuKind::Bar,
        }
    }

    /// Parse a typed value (`on`/`off`, a choice's label, a number; a percent may end in `%`).
    /// True when it parsed; the value is then normalised and stored.
    pub fn parse(&mut self, id: OptionId, text: &str) -> bool {
        match parse_value(self.spec(id).kind, text.trim()) {
            Some(value) => {
                self.set(id, value);
                true
            }
            None => false,
        }
    }

    /// The persisted form of `id`'s value: `true`/`false`, the lowercased label, a number.
    fn write_value(&self, id: OptionId, out: &mut String, human: bool) {
        let spec = self.spec(id);
        match (spec.kind, self.value(id)) {
            (_, OptionValue::Bool(on)) => out.push_str(match (on, human) {
                (true, true) => "On",
                (false, true) => "Off",
                (true, false) => "true",
                (false, false) => "false",
            }),
            (OptionKind::Choice(choices), OptionValue::Choice(i)) => {
                let label = choices.get(i).copied().unwrap_or("");
                if human {
                    out.push_str(label);
                } else {
                    out.extend(label.chars().map(|c| c.to_ascii_lowercase()));
                }
            }
            (_, OptionValue::Choice(i)) => {
                let _ = write!(out, "{i}");
            }
            (_, OptionValue::Int(v)) => {
                let _ = write!(out, "{v}{}", if human { "%" } else { "" });
            }
            (kind, OptionValue::Float(v)) => {
                let places = match kind {
                    OptionKind::Float { step, .. } => decimals(step),
                    _ => 3,
                };
                let _ = write!(out, "{v:.places$}");
            }
        }
    }

    /// Apply a persisted value; a line that does not parse leaves the default.
    fn parse_stored(&mut self, id: OptionId, text: &str) {
        if let Some(value) = parse_value(self.spec(id).kind, text.trim()) {
            let value = self.spec(id).normalize(value);
            self.values[id.index()] = value;
        }
    }

    /// Read the option lines of a `settings.cfg` text: `<package>.<key>=` lines (kept when no
    /// declared option claims them), then the legacy keys of options the text has no line for,
    /// from `text` first and then from `legacy` (old files flattened by [`flatten_mods_cfg`]).
    /// `core` claims the keys of the core's own table first.
    pub(crate) fn read_text(&mut self, text: &str, legacy: &str, core: impl Fn(&str) -> bool) {
        let mut seen = vec![false; self.declared.len()];
        super::each_kv_line(text, |key, value| {
            if core(key) || !key.contains('.') {
                return;
            }
            match self.find(key) {
                Some(id) => {
                    self.parse_stored(id, value);
                    seen[id.index()] = true;
                }
                None => match self.unknown.iter_mut().find(|(k, _)| k == key) {
                    Some(line) => line.1 = value.to_string(),
                    None => self.unknown.push((key.to_string(), value.to_string())),
                },
            }
        });
        for source in [text, legacy] {
            super::each_kv_line(source, |key, value| {
                for i in 0..self.declared.len() {
                    if !seen[i] && self.declared[i].spec.legacy_key == Some(key) {
                        self.parse_stored(OptionId(i as u32), value);
                        seen[i] = true;
                    }
                }
            });
        }
        self.touch();
    }

    /// Append every option's line (`<package>.<key>=<value>`), in declaration order, then the
    /// kept unknown lines.
    pub(crate) fn write_text(&self, text: &mut String) {
        for id in self.ids() {
            let d = &self.declared[id.index()];
            let _ = write!(text, "{}.{}=", d.owner, d.spec.key);
            self.write_value(id, text, false);
            text.push('\n');
        }
        for (key, value) in &self.unknown {
            let _ = writeln!(text, "{key}={value}");
        }
    }
}

/// The knob payloads of an old `mods.cfg` as legacy keys: each `<mod-id>.state=<k>=<v>,<k>=<v>`
/// line becomes one `<mod-id>.state.<k>=<v>` line per pair; every other line is dropped. A
/// package option names such a key with [`OptionSpec::legacy_key`] to carry a player's old knob
/// value over once. The core reads `mods.cfg` only until `settings.cfg` records
/// [`FORMAT_KEY`].
pub(crate) fn flatten_mods_cfg(text: &str) -> String {
    let mut out = String::new();
    super::each_kv_line(text, |key, value| {
        let Some(id) = key.strip_suffix(".state") else { return };
        for (k, v) in value.split(',').filter_map(|pair| pair.split_once('=')) {
            let _ = writeln!(out, "{id}.state.{}={}", k.trim(), v.trim());
        }
    });
    out
}

/// The `settings.cfg` line that says the options format is in use, so an old `mods.cfg` has been
/// read once and never needs reading again.
pub(crate) const FORMAT_KEY: &str = "options_format";

/// Whether `full` is `<owner>.<key>`.
fn key_is(full: &str, owner: &str, key: &str) -> bool {
    full.len() == owner.len() + 1 + key.len()
        && full.starts_with(owner)
        && full.as_bytes()[owner.len()] == b'.'
        && full.ends_with(key)
}

fn parse_value(kind: OptionKind, text: &str) -> Option<OptionValue> {
    match kind {
        OptionKind::Toggle => super::parse_toggle(&text.to_ascii_lowercase()).map(OptionValue::Bool),
        OptionKind::Choice(choices) => choices
            .iter()
            .position(|c| c.eq_ignore_ascii_case(text))
            .map(OptionValue::Choice),
        OptionKind::Percent { .. } => text.trim_end_matches('%').trim().parse().ok().map(OptionValue::Int),
        OptionKind::Float { .. } => text.parse::<f32>().ok().filter(|v| v.is_finite()).map(OptionValue::Float),
    }
}

/// Which entry of an [`OptionsView`] an index names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    Core(usize),
    Mod(OptionId),
}

/// One row of an [`OptionsView`], as a menu or a command lists it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionInfo {
    /// `"core"` for the core's settings, else the declaring package's id.
    pub owner: &'static str,
    /// The core setting's key (`bloom`), or the option's key within its package (`relief`).
    pub key: &'static str,
    pub label: &'static str,
    pub page: Category,
    pub menu_kind: MenuKind,
    pub applies: Applies,
}

/// Every tunable, read-only: the core's settings first (owner `"core"`, in [`SETTINGS`] order),
/// then each package's options in declaration order. Entries are read by index. `Copy`, so a
/// screen's draw can hold one.
#[derive(Clone, Copy)]
pub struct OptionsRef<'a> {
    settings: &'a Settings,
    options: &'a Options,
}

impl<'a> OptionsRef<'a> {
    pub fn new(settings: &'a Settings, options: &'a Options) -> Self {
        Self { settings, options }
    }

    fn entry(&self, i: usize) -> Entry {
        if i < SETTINGS.len() {
            Entry::Core(i)
        } else {
            Entry::Mod(OptionId((i - SETTINGS.len()) as u32))
        }
    }

    /// The core's settings.
    pub fn settings(&self) -> &'a Settings {
        self.settings
    }

    /// The packages' options.
    pub fn options(&self) -> &'a Options {
        self.options
    }

    /// How many entries: the core's settings plus every declared option.
    pub fn len(&self) -> usize {
        SETTINGS.len() + self.options.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What entry `i` is.
    pub fn info(&self, i: usize) -> OptionInfo {
        match self.entry(i) {
            Entry::Core(i) => {
                let s = &SETTINGS[i];
                OptionInfo {
                    owner: CORE,
                    key: s.key(),
                    label: s.label(),
                    page: s.category(),
                    menu_kind: s.menu_kind(),
                    applies: Applies::Live,
                }
            }
            Entry::Mod(id) => {
                let spec = self.options.spec(id);
                OptionInfo {
                    owner: self.options.owner(id),
                    key: spec.key,
                    label: spec.label,
                    page: spec.page,
                    menu_kind: self.options.menu_kind(id),
                    applies: spec.applies,
                }
            }
        }
    }

    /// The value as a menu shows it.
    pub fn show(&self, i: usize) -> String {
        match self.entry(i) {
            Entry::Core(i) => SETTINGS[i].show(self.settings),
            Entry::Mod(id) => self.options.show(id),
        }
    }

    /// A toggle's state, `None` for other kinds.
    pub fn toggled(&self, i: usize) -> Option<bool> {
        match self.entry(i) {
            Entry::Core(i) if SETTINGS[i].menu_kind() == MenuKind::Toggle => Some(SETTINGS[i].show(self.settings) == "On"),
            Entry::Mod(id) => match self.options.value(id) {
                OptionValue::Bool(on) => Some(on),
                _ => None,
            },
            Entry::Core(_) => None,
        }
    }

    /// Where a bar sits, `0..=1`.
    pub fn fraction(&self, i: usize) -> f32 {
        match self.entry(i) {
            Entry::Core(i) => SETTINGS[i].fraction(self.settings),
            Entry::Mod(id) => self.options.fraction(id),
        }
    }

    /// The entry a name picks: a core key or alias (`bloom`, `fps`), or `<package>.<key>`.
    pub fn find(&self, name: &str) -> Option<usize> {
        SETTINGS
            .iter()
            .position(|s| s.matches(name))
            .or_else(|| self.options.find(name).map(|id| SETTINGS.len() + id.index()))
    }

    /// The name [`find`](Self::find) takes for entry `i`: a core key (`bloom`), or
    /// `<package>.<key>`.
    pub fn full_key(&self, i: usize) -> String {
        match self.entry(i) {
            Entry::Core(i) => SETTINGS[i].key().to_string(),
            Entry::Mod(id) => format!("{}.{}", self.options.owner(id), self.options.spec(id).key),
        }
    }

    /// The values entry `i` takes, for a command's usage line (`on|off`, `Mineral|Arcane`,
    /// `25-200%`, `0.1-2.0`).
    pub fn hint(&self, i: usize) -> String {
        match self.entry(i) {
            Entry::Core(i) => SETTINGS[i].usage().split_once(' ').map_or(String::new(), |(_, v)| v.to_string()),
            Entry::Mod(id) => match self.options.spec(id).kind {
                OptionKind::Toggle => "on|off".to_string(),
                OptionKind::Choice(choices) => choices.join("|"),
                OptionKind::Percent { min, max, step } => format!("{min}-{max}% (steps of {step})"),
                OptionKind::Float { min, max, step } => {
                    let d = decimals(step);
                    format!("{min:.d$}-{max:.d$} (steps of {step:.d$})")
                }
            },
        }
    }

    /// Moves whenever a value changes through any view, or a package's option changes.
    pub fn revision(&self) -> u64 {
        self.options.revision()
    }
}

/// Every tunable, to read and change: what [`OptionsRef`] reads, plus [`step`](Self::step) and
/// [`parse`](Self::parse). Writes go through here, so [`Options::revision`] tells the host when
/// to save.
pub struct OptionsView<'a> {
    settings: &'a mut Settings,
    options: &'a mut Options,
}

impl<'a> OptionsView<'a> {
    pub fn new(settings: &'a mut Settings, options: &'a mut Options) -> Self {
        Self { settings, options }
    }

    /// The read-only view.
    pub fn read(&self) -> OptionsRef<'_> {
        OptionsRef::new(self.settings, self.options)
    }

    /// The core's settings, read-only.
    pub fn settings(&self) -> &Settings {
        self.settings
    }

    /// The packages' options, read-only.
    pub fn options(&self) -> &Options {
        self.options
    }

    /// See [`OptionsRef::len`].
    pub fn len(&self) -> usize {
        self.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// See [`OptionsRef::info`].
    pub fn info(&self, i: usize) -> OptionInfo {
        self.read().info(i)
    }

    /// See [`OptionsRef::show`].
    pub fn show(&self, i: usize) -> String {
        self.read().show(i)
    }

    /// See [`OptionsRef::toggled`].
    pub fn toggled(&self, i: usize) -> Option<bool> {
        self.read().toggled(i)
    }

    /// See [`OptionsRef::fraction`].
    pub fn fraction(&self, i: usize) -> f32 {
        self.read().fraction(i)
    }

    /// See [`OptionsRef::find`].
    pub fn find(&self, name: &str) -> Option<usize> {
        self.read().find(name)
    }

    /// See [`OptionsRef::full_key`].
    pub fn full_key(&self, i: usize) -> String {
        self.read().full_key(i)
    }

    /// See [`OptionsRef::hint`].
    pub fn hint(&self, i: usize) -> String {
        self.read().hint(i)
    }

    /// See [`OptionsRef::revision`].
    pub fn revision(&self) -> u64 {
        self.options.revision()
    }

    /// One menu step (`dir` −1 or +1).
    pub fn step(&mut self, i: usize, dir: i32) {
        match self.read().entry(i) {
            Entry::Core(i) => {
                let before = self.settings.clone();
                SETTINGS[i].step(self.settings, dir);
                if *self.settings != before {
                    self.options.touch();
                }
            }
            Entry::Mod(id) => {
                self.options.step(id, dir);
            }
        }
    }

    /// Parse a typed value. True when it parsed.
    pub fn parse(&mut self, i: usize, text: &str) -> bool {
        match self.read().entry(i) {
            Entry::Core(i) => {
                let parsed = SETTINGS[i].parse_human(self.settings, text);
                if parsed {
                    self.options.touch();
                }
                parsed
            }
            Entry::Mod(id) => self.options.parse(id, text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STYLES: &[&str] = &["Mineral", "Arcane"];

    fn sample() -> (Options, [OptionId; 4]) {
        let mut o = Options::new();
        let relief = o.declare("pwc.worldgen", OptionSpec::percent("relief", "Relief", Category::World, (25, 300, 25), 100).next_world());
        let detail = o.declare("pwc.textures", OptionSpec::float("detail", "Detail", Category::Video, (0.1, 2.0, 0.1), 1.0));
        let style = o.declare("pwc.names", OptionSpec::choice("style", "Style", Category::Interface, STYLES, 0));
        let voice = o.declare("pwc.voice", OptionSpec::toggle("voice_enabled", "Voice Chat", Category::Audio, true).legacy_key("voice_enabled"));
        (o, [relief, detail, style, voice])
    }

    #[test]
    fn values_round_trip_through_the_settings_text_and_unknown_lines_are_kept() {
        let (mut o, [relief, detail, style, voice]) = sample();
        assert!(o.step(relief, 1) && o.step(detail, 1) && o.step(style, 1) && o.step(voice, 1));
        let mut text = String::new();
        o.write_text(&mut text);
        assert_eq!(
            text,
            "pwc.worldgen.relief=125\npwc.textures.detail=1.1\npwc.names.style=arcane\npwc.voice.voice_enabled=false\n"
        );
        let (mut fresh, ids) = sample();
        fresh.read_text(&format!("{text}gone.pkg.knob=7\nbloom=false\nnot a line\n"), "", |key| key == "bloom");
        assert_eq!(ids.map(|id| fresh.value(id)), [relief, detail, style, voice].map(|id| o.value(id)));
        let mut again = String::new();
        fresh.write_text(&mut again);
        assert_eq!(again, format!("{text}gone.pkg.knob=7\n"), "a removed package's line is written back");
        // Added again, the package finds its value.
        let late = fresh.declare("gone.pkg", OptionSpec::percent("knob", "Knob", Category::World, (0, 10, 1), 1));
        assert_eq!(fresh.int(late), 7);
        let mut last = String::new();
        fresh.write_text(&mut last);
        assert_eq!(last.matches("gone.pkg.knob=7").count(), 1);
    }

    #[test]
    fn stored_values_clamp_and_snap_and_junk_keeps_the_default() {
        let (mut o, [relief, detail, style, voice]) = sample();
        o.read_text(
            "pwc.worldgen.relief=137\npwc.textures.detail=9\npwc.names.style=sparkly\npwc.voice.voice_enabled=maybe\n",
            "",
            |_| false,
        );
        assert_eq!(o.int(relief), 125, "snapped onto the stepper");
        assert_eq!(o.float(detail), 2.0, "clamped");
        assert_eq!(o.choice(style), 0, "an unknown label keeps the default");
        assert!(o.bool(voice), "an unparseable toggle keeps the default");
        o.read_text("pwc.worldgen.relief=9000\npwc.textures.detail=-3\npwc.names.style=ARCANE\n", "", |_| false);
        assert_eq!((o.int(relief), o.float(detail), o.choice(style)), (300, 0.1, 1));
        for _ in 0..30 {
            o.step(relief, 1);
        }
        assert_eq!(o.int(relief), 300, "numbers stop at their ends");
        assert!(o.step(style, 1) && o.choice(style) == 0, "choices wrap");
    }

    #[test]
    fn a_legacy_key_is_read_only_when_the_option_has_no_line_of_its_own() {
        let (mut o, [.., voice]) = sample();
        o.read_text("voice_enabled=false\n", "", |_| false);
        assert!(!o.bool(voice), "the old core key carries the player's choice over");
        let (mut o, [.., voice]) = sample();
        o.read_text("voice_enabled=false\npwc.voice.voice_enabled=true\n", "", |_| false);
        assert!(o.bool(voice), "the option's own line wins");
        let mut text = String::new();
        o.write_text(&mut text);
        assert!(!text.contains("\nvoice_enabled="), "a legacy key is never written: {text}");
    }

    #[test]
    fn the_revision_moves_with_every_change_and_not_with_reads() {
        let (mut o, [relief, ..]) = sample();
        let r = o.revision();
        let _ = (o.int(relief), o.show(relief), o.fraction(relief));
        assert_eq!(o.revision(), r);
        assert!(!o.set(relief, OptionValue::Int(100)), "the same value changes nothing");
        assert_eq!(o.revision(), r);
        assert!(o.set(relief, OptionValue::Int(150)));
        assert_ne!(o.revision(), r);
        let r = o.revision();
        let mut settings = Settings::default();
        let mut view = OptionsView::new(&mut settings, &mut o);
        let bloom = view.find("bloom").expect("a core key");
        view.step(bloom, 1);
        assert_ne!(view.revision(), r, "a core setting stepped through the view moves it too");
        assert_ne!(settings.bloom, Settings::default().bloom);
    }

    #[test]
    fn reading_by_index_allocates_nothing() {
        let (o, [relief, detail, style, voice]) = sample();
        crate::alloc_count::reset();
        let mut sum = 0.0;
        for _ in 0..1000 {
            sum += o.int(relief) as f32 + o.float(detail) + o.choice(style) as f32 + o.bool(voice) as u8 as f32;
        }
        assert_eq!(crate::alloc_count::alloc_count(), 0);
        assert!(sum > 0.0);
    }

    #[test]
    fn the_view_lists_core_settings_then_package_options() {
        let (mut o, [relief, ..]) = sample();
        let mut settings = Settings::default();
        let mut view = OptionsView::new(&mut settings, &mut o);
        assert_eq!(view.len(), SETTINGS.len() + 4);
        let first = view.info(0);
        assert_eq!((first.owner, first.key), (CORE, SETTINGS[0].key()));
        let at = view.find("pwc.worldgen.relief").expect("a package option by its full key");
        assert_eq!(at, SETTINGS.len());
        let info = view.info(at);
        assert_eq!((info.owner, info.key, info.label, info.page), ("pwc.worldgen", "relief", "Relief", Category::World));
        assert_eq!((info.menu_kind, info.applies), (MenuKind::Bar, Applies::NextWorld));
        assert_eq!(view.show(at), "100%");
        assert!((view.fraction(at) - 75.0 / 275.0).abs() < 1e-6);
        view.step(at, 1);
        assert_eq!(view.show(at), "125%");
        assert!(view.parse(at, "200%"));
        assert_eq!(view.options().int(relief), 200);
        assert!(!view.parse(at, "lots"));
        let voice = view.find("pwc.voice.voice_enabled").unwrap();
        assert_eq!(view.toggled(voice), Some(true));
        assert_eq!((view.full_key(voice), view.hint(voice)), ("pwc.voice.voice_enabled".to_string(), "on|off".to_string()));
        assert_eq!(view.hint(at), "25-300% (steps of 25)");
        assert_eq!(view.hint(view.find("pwc.textures.detail").unwrap()), "0.1-2.0 (steps of 0.1)");
        assert_eq!(view.hint(view.find("pwc.names.style").unwrap()), "Mineral|Arcane");
        let fps = view.find("fps").expect("an alias");
        assert_eq!((view.full_key(fps), view.hint(fps)), ("max_fps".to_string(), "<10-1000>|off".to_string()));
        assert_eq!(view.show(view.find("pwc.textures.detail").unwrap()), "1.0");
        assert_eq!(view.show(view.find("pwc.names.style").unwrap()), "Mineral");
        assert_eq!(view.find("pwc.nothing.here"), None);
    }

    #[test]
    fn declaring_twice_returns_the_first_id_and_full_keys_do_not_collide() {
        let mut o = Options::new();
        let a = o.declare("p.a", OptionSpec::toggle("x", "X", Category::World, false));
        let b = o.declare("p.a", OptionSpec::toggle("x", "X again", Category::World, true));
        let c = o.declare("p", OptionSpec::toggle("a.x", "Other", Category::World, true));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(o.len(), 2);
        assert_eq!(o.find("p.a.x"), Some(a), "the first declaration of a full key wins");
    }
}
