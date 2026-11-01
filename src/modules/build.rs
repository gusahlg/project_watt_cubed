//! A build's package list: which packages are compiled into this executable, what each one is,
//! and the registrar each module package's entry point receives.
//!
//! A PWC with modules is an exact build: one game version plus one exact set of packages. The PWC
//! builder (`pwc build`) generates a tiny crate whose `main` hands [`crate::run`] a [`GameBuild`]
//! made by [`GameBuild::from_static`]: one [`PackageInfo`] per locked package of every kind (modules,
//! libraries and bundles), dependencies before dependents. The vanilla executable passes
//! [`GameBuild::vanilla`]. Nothing here scans directories or loads code at run time.
//!
//! The list is data the builder generated. The core iterates it and names no package; packages
//! read it through [`ModuleRegistrar::build`].

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;

use super::{Module, Modules};
use crate::settings::{OptionId, OptionSpec, Options};

/// What a package is: `kind` in its `module.toml`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackageKind {
    /// Code with an entry point. Its `register` installs modules.
    Module,
    /// Code other packages call. It has no entry point.
    Library,
    /// No code: a named set of packages, its dependencies.
    Bundle,
}

impl PackageKind {
    /// The pre-3.4 name of [`PackageKind::Module`], so packages written before the rename compile.
    #[doc(hidden)]
    #[allow(non_upper_case_globals)]
    pub const Mod: PackageKind = PackageKind::Module;

    /// The `module.toml` spelling: `"module"`, `"library"` or `"bundle"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Library => "library",
            Self::Bundle => "bundle",
        }
    }
}

/// One package of a build, as the builder generated it from `module.toml` and `pwc.lock`.
/// Everything is `'static` and the value is `Copy`, so a package can keep what it needs.
#[derive(Clone, Copy, Debug)]
pub struct PackageInfo {
    /// The permanent, namespaced package id (`pwc.example`).
    pub id: &'static str,
    /// Display name from `module.toml`.
    pub name: &'static str,
    /// Exact package version.
    pub version: &'static str,
    /// Description from `module.toml`.
    pub description: &'static str,
    /// Module, library or bundle.
    pub kind: PackageKind,
    /// Ids of the packages this one depends on directly, sorted. A bundle's are its members.
    pub dependencies: &'static [&'static str],
    /// The entry point. The builder sets it for `kind = "module"` packages only; packages without
    /// one are listed but install nothing.
    pub register: Option<fn(&mut ModuleRegistrar)>,
}

/// One compiled-in module package by its identity and entry point: the 2.x shorthand that
/// [`GameBuild::with_module`] turns into a [`PackageInfo`] of kind [`PackageKind::Module`] with no
/// description and no dependencies. Kept so 2.x tests build unchanged; prefer [`PackageInfo`].
#[derive(Clone, Copy, Debug)]
pub struct ModuleDescriptor {
    /// The permanent, namespaced package id from `module.toml` (`pwc.example`).
    pub id: &'static str,
    /// Display name from `module.toml`.
    pub name: &'static str,
    /// Exact package version.
    pub version: &'static str,
    /// The package's `register` function.
    pub register: fn(&mut ModuleRegistrar),
}

impl From<ModuleDescriptor> for PackageInfo {
    fn from(package: ModuleDescriptor) -> Self {
        Self {
            id: package.id,
            name: package.name,
            version: package.version,
            description: "",
            kind: PackageKind::Module,
            dependencies: &[],
            register: Some(package.register),
        }
    }
}

/// What one build is made of: every package in registration (dependency) order, and the
/// environment that produced them. Read-only; packages see it through [`ModuleRegistrar::build`].
#[derive(Clone, Debug, Default)]
pub struct BuildInfo {
    packages: Cow<'static, [PackageInfo]>,
    environment: Option<&'static str>,
}

impl BuildInfo {
    /// A build with no package and no environment.
    pub const EMPTY: BuildInfo = BuildInfo { packages: Cow::Borrowed(&[]), environment: None };

    /// Every package of every kind, dependencies before dependents (ties by id in generated
    /// builds).
    pub fn packages(&self) -> &[PackageInfo] {
        &self.packages
    }

    /// The environment hash of the `pwc.lock` this build was made from, or `None` for a vanilla
    /// or hand-assembled build.
    pub fn environment(&self) -> Option<&'static str> {
        self.environment
    }

    /// The package with this id, if the build has it.
    pub fn package(&self, id: &str) -> Option<&PackageInfo> {
        self.packages.iter().find(|p| p.id == id)
    }
}

/// The packages of one build and the identity of the environment that produced them: what
/// [`crate::run`] starts the game with.
#[derive(Clone, Debug, Default)]
pub struct GameBuild {
    info: BuildInfo,
}

impl GameBuild {
    /// The bare game: no package at all.
    pub fn vanilla() -> Self {
        Self::default()
    }

    /// An empty build to add packages to (the same as [`vanilla`](Self::vanilla)).
    pub fn new() -> Self {
        Self::default()
    }

    /// A generated build: the lock's environment hash and its packages in registration order.
    /// Nothing is copied.
    pub const fn from_static(environment: &'static str, packages: &'static [PackageInfo]) -> Self {
        Self { info: BuildInfo { packages: Cow::Borrowed(packages), environment: Some(environment) } }
    }

    /// Add a package after the ones already listed. Its dependencies must come first.
    pub fn with_package(mut self, package: PackageInfo) -> Self {
        self.info.packages.to_mut().push(package);
        self
    }

    /// Add a module package from its 2.x descriptor (see [`ModuleDescriptor`]).
    pub fn with_module(self, package: ModuleDescriptor) -> Self {
        self.with_package(package.into())
    }

    /// The pre-3.4 name of [`with_module`](Self::with_module).
    #[doc(hidden)]
    pub fn with_mod(self, package: ModuleDescriptor) -> Self {
        self.with_module(package)
    }

    /// Record the environment hash of the `pwc.lock` this build was made from.
    pub fn with_environment(mut self, environment: &'static str) -> Self {
        self.info.environment = Some(environment);
        self
    }

    /// The read-only description packages see.
    pub fn info(&self) -> &BuildInfo {
        &self.info
    }

    /// Every package, in registration order.
    pub fn packages(&self) -> &[PackageInfo] {
        self.info.packages()
    }

    /// The environment hash, or `None` for a vanilla or hand-assembled build.
    pub fn environment(&self) -> Option<&'static str> {
        self.info.environment()
    }

    /// Instantiate this build's modules, dropping the options they declare (tests that read options
    /// call [`Modules::from_build`] with their own).
    #[cfg(test)]
    pub(crate) fn modules(&self) -> Modules {
        Modules::from_build(self, &mut Options::new())
    }
}

/// Values packages share during registration (a type-keyed map). A package provides a handle;
/// a package that depends on it reads the handle back. Dropped once every package registered.
#[derive(Default)]
pub(super) struct Resources {
    values: HashMap<TypeId, Box<dyn Any>>,
}

/// What a package's `register` function gets: the place to install its modules, share handles with
/// the packages that depend on it, and read what the build contains.
pub struct ModuleRegistrar<'a> {
    package: &'a PackageInfo,
    build: &'a BuildInfo,
    modules: &'a mut Modules,
    options: &'a mut Options,
    resources: &'a mut Resources,
}

impl<'a> ModuleRegistrar<'a> {
    pub(super) fn new(
        package: &'a PackageInfo,
        build: &'a BuildInfo,
        modules: &'a mut Modules,
        options: &'a mut Options,
        resources: &'a mut Resources,
    ) -> Self {
        Self { package, build, modules, options, resources }
    }

    /// The package being registered.
    pub fn package(&self) -> &'a PackageInfo {
        self.package
    }

    /// Every package compiled into this build, of every kind, in registration order. Packages
    /// registered later are listed too.
    pub fn build(&self) -> &'a BuildInfo {
        self.build
    }

    /// Install a module. It runs for as long as the build has it, unless the core suspends this
    /// package for a session.
    pub fn add(&mut self, module: impl Module + 'static) {
        self.modules.install_from(Some(self.package.id), Box::new(module));
    }

    /// Declare one tunable of this package in the core's options registry, and get the index
    /// its value is read by ([`Options::bool`](crate::settings::Options::bool), `int`, `float`,
    /// `choice`). It persists in `settings.cfg` as `<package-id>.<key>=`, and any settings screen
    /// lists it on `spec.page` without knowing this package. Read it in
    /// [`Module::on_options`](super::Module::on_options).
    pub fn option(&mut self, spec: OptionSpec) -> OptionId {
        self.options.declare(self.package.id, spec)
    }

    /// Offer a screen (a settings menu, a package list) on the screen out of a world and/or the
    /// pause screen: a root or pause screen lists the entries for its place without knowing this
    /// package. See [`crate::screen`].
    pub fn screen_entry(&mut self, entry: crate::screen::ScreenEntry) {
        self.modules.add_entry_from(Some(self.package.id), entry);
    }

    /// Register a command-line entry point. The first registration of a name wins. `help` and
    /// `--commands` stay reserved and list every registered command.
    pub fn cli(&mut self, cli: super::cli::Cli) {
        self.modules.add_cli(cli);
    }

    /// Share a value with the packages registered after this one (handles are usually `Rc`s).
    /// A second value of the same type replaces the first.
    pub fn provide<T: Any>(&mut self, value: T) {
        self.resources.values.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// A clone of a value an earlier package provided.
    pub fn get<T: Any + Clone>(&self) -> Option<T> {
        self.resources.values.get(&TypeId::of::<T>()).and_then(|v| v.downcast_ref::<T>()).cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::modules::testing::Stub;

    #[derive(Clone)]
    struct Shared(u32);

    fn provider(r: &mut ModuleRegistrar) {
        r.provide(Shared(7));
        r.add(Stub::new("first"));
    }

    fn consumer(r: &mut ModuleRegistrar) {
        let shared = r.get::<Shared>().expect("the dependency provided it");
        assert_eq!(shared.0, 7);
        r.add(Stub::new("second"));
    }

    #[test]
    fn packages_register_in_order_and_share_resources() {
        let build = GameBuild::new()
            .with_module(ModuleDescriptor { id: "test.provider", name: "Provider", version: "1.0.0", register: provider })
            .with_module(ModuleDescriptor { id: "test.consumer", name: "Consumer", version: "1.0.0", register: consumer })
            .with_environment("sha256:00");
        let modules = build.modules();
        assert_eq!(modules.len(), 2);
        assert_eq!((modules.id(0), modules.package(0)), ("first", Some("test.provider")));
        assert_eq!((modules.id(1), modules.package(1), modules.is_active(1)), ("second", Some("test.consumer"), true));
        assert_eq!(build.environment(), Some("sha256:00"));
        let kinds: Vec<PackageKind> = build.packages().iter().map(|p| p.kind).collect();
        assert_eq!(kinds, [PackageKind::Module, PackageKind::Module], "a 2.x descriptor is a module package");
    }

    #[test]
    fn the_vanilla_build_is_empty() {
        for build in [GameBuild::vanilla(), GameBuild::new()] {
            assert!(build.packages().is_empty());
            assert_eq!(build.environment(), None);
            assert!(build.modules().is_empty());
        }
    }

    type Seen = Vec<(&'static str, Vec<(&'static str, PackageKind)>)>;

    thread_local! {
        /// What `probe` saw: the package it registered as, and the build's ids and kinds.
        static SEEN: RefCell<Seen> = const { RefCell::new(Vec::new()) };
    }

    fn probe(r: &mut ModuleRegistrar) {
        let listed = r.build().packages().iter().map(|p| (p.id, p.kind)).collect();
        SEEN.with(|s| s.borrow_mut().push((r.package().id, listed)));
        r.add(Stub::new(r.package().name));
    }

    /// A build as the builder generates it: a library, two modules and a bundle, in registration
    /// order, in a `static`.
    static GENERATED: &[PackageInfo] = &[
        PackageInfo {
            id: "test.names",
            name: "names",
            version: "1.0.0",
            description: "A library.",
            kind: PackageKind::Library,
            dependencies: &[],
            register: None,
        },
        PackageInfo {
            id: "test.alpha",
            name: "alpha",
            version: "1.2.0",
            description: "Needs the names.",
            kind: PackageKind::Module,
            dependencies: &["test.names"],
            register: Some(probe),
        },
        PackageInfo {
            id: "test.beta",
            name: "beta",
            version: "0.1.0",
            description: "",
            kind: PackageKind::Module,
            dependencies: &[],
            register: Some(probe),
        },
        PackageInfo {
            id: "test.bundle",
            name: "Bundle",
            version: "2.0.0",
            description: "Both modules.",
            kind: PackageKind::Bundle,
            dependencies: &["test.alpha", "test.beta"],
            register: None,
        },
    ];

    #[test]
    fn a_generated_build_registers_only_entry_points_and_every_package_sees_the_whole_list() {
        let build = GameBuild::from_static("sha256:01", GENERATED);
        SEEN.with(|s| s.borrow_mut().clear());
        let modules = build.modules();
        let all = vec![
            ("test.names", PackageKind::Library),
            ("test.alpha", PackageKind::Module),
            ("test.beta", PackageKind::Module),
            ("test.bundle", PackageKind::Bundle),
        ];
        let seen = SEEN.with(|s| s.take());
        assert_eq!(seen, [("test.alpha", all.clone()), ("test.beta", all)], "the library and bundle register nothing");
        assert_eq!((modules.len(), modules.id(0), modules.package(1)), (2, "alpha", Some("test.beta")));
        assert_eq!(build.environment(), Some("sha256:01"));
        assert!(std::ptr::eq(build.packages(), GENERATED), "a generated list is borrowed, not copied");

        let info = build.info();
        let bundle = info.package("test.bundle").expect("listed");
        assert_eq!((bundle.kind.as_str(), bundle.dependencies), ("bundle", &["test.alpha", "test.beta"][..]));
        assert_eq!(info.package("test.alpha").map(|p| (p.version, p.description)), Some(("1.2.0", "Needs the names.")));
        assert!(info.package("test.gone").is_none());

        // Adding to a generated build copies it once and keeps the order.
        let more = build.clone().with_package(PackageInfo { id: "test.extra", register: None, ..GENERATED[0] });
        let ids: Vec<&str> = more.packages().iter().map(|p| p.id).collect();
        assert_eq!(ids, ["test.names", "test.alpha", "test.beta", "test.bundle", "test.extra"]);
        assert_eq!(build.packages().len(), 4, "the generated list itself is untouched");
    }

    #[test]
    fn kinds_spell_like_the_manifest() {
        let spelled = [PackageKind::Module, PackageKind::Library, PackageKind::Bundle].map(PackageKind::as_str);
        assert_eq!(spelled, ["module", "library", "bundle"]);
    }
}
