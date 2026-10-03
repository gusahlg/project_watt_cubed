//! A build's mod list: which packages are compiled into this executable, and the registrar each
//! package's entry point receives.
//!
//! A modded PWC is an exact build: one game version plus one exact set of mod packages. The PWC
//! builder (`pwc build`) generates a tiny crate whose `main` hands [`crate::run`] a [`GameBuild`]
//! listing every package in dependency order; the vanilla executable passes
//! [`GameBuild::vanilla`]. Nothing here scans directories or loads code at run time.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use super::{Group, Mod, Mods};

/// One compiled-in mod package: its manifest identity and its entry point.
#[derive(Clone, Copy, Debug)]
pub struct ModDescriptor {
    /// The permanent, namespaced package id from `mod.toml` (`pwc.hotbar`).
    pub id: &'static str,
    /// Display name from `mod.toml`.
    pub name: &'static str,
    /// Exact package version.
    pub version: &'static str,
    /// The package's `register` function.
    pub register: fn(&mut ModRegistrar),
}

/// The packages of one build, in registration (dependency) order, and the identity of the
/// environment that produced them.
#[derive(Clone, Debug, Default)]
pub struct GameBuild {
    packages: Vec<ModDescriptor>,
    environment: Option<&'static str>,
}

impl GameBuild {
    /// The bare game: no mod package at all.
    pub fn vanilla() -> Self {
        Self::default()
    }

    /// An empty build to add packages to (the same as [`vanilla`](Self::vanilla)).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a package. Generated builds add them in dependency order (dependencies first), which
    /// is also the order their mods appear on the mods screen.
    pub fn with_mod(mut self, package: ModDescriptor) -> Self {
        self.packages.push(package);
        self
    }

    /// Record the environment hash of the `pwc.lock` this build was made from.
    pub fn with_environment(mut self, environment: &'static str) -> Self {
        self.environment = Some(environment);
        self
    }

    /// Every package, in registration order.
    pub fn packages(&self) -> &[ModDescriptor] {
        &self.packages
    }

    /// The environment hash, or `None` for a vanilla or hand-assembled build.
    pub fn environment(&self) -> Option<&'static str> {
        self.environment
    }

    /// Instantiate this build's mods.
    pub fn mods(&self) -> Mods {
        Mods::from_build(self)
    }
}

/// Values packages share during registration (a type-keyed map). A package provides a handle;
/// a package that depends on it reads the handle back. Dropped once every package registered.
#[derive(Default)]
pub(super) struct Resources {
    values: HashMap<TypeId, Box<dyn Any>>,
}

/// What a package's `register` function gets: the place to install its mods, declare groups and
/// share handles with the packages that depend on it.
pub struct ModRegistrar<'a> {
    package: &'a ModDescriptor,
    mods: &'a mut Mods,
    resources: &'a mut Resources,
}

impl<'a> ModRegistrar<'a> {
    pub(super) fn new(package: &'a ModDescriptor, mods: &'a mut Mods, resources: &'a mut Resources) -> Self {
        Self { package, mods, resources }
    }

    /// The package being registered.
    pub fn package(&self) -> &ModDescriptor {
        self.package
    }

    /// Install a mod, enabled by default (the player's `mods.cfg` choice still wins).
    pub fn add(&mut self, module: impl Mod + 'static) {
        self.mods.install_from(Some(self.package.id), Box::new(module), true);
    }

    /// Install a mod that starts disabled.
    pub fn add_disabled(&mut self, module: impl Mod + 'static) {
        self.mods.install_from(Some(self.package.id), Box::new(module), false);
    }

    /// Declare a group for the mods screen (its members return `group.id` from [`Mod::group`]).
    /// Declaring the same id twice keeps the first.
    pub fn declare_group(&mut self, group: Group) {
        self.mods.declare_group(group);
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
    use super::*;

    #[derive(Clone)]
    struct Shared(u32);

    struct Named(&'static str);
    impl Mod for Named {
        fn name(&self) -> &str {
            self.0
        }
        fn id(&self) -> &'static str {
            self.0
        }
    }

    fn provider(r: &mut ModRegistrar) {
        r.provide(Shared(7));
        r.add(Named("first"));
    }

    fn consumer(r: &mut ModRegistrar) {
        let shared = r.get::<Shared>().expect("the dependency provided it");
        assert_eq!(shared.0, 7);
        r.add_disabled(Named("second"));
        r.declare_group(Group { id: "tools", name: "Tools", description: "" });
    }

    #[test]
    fn packages_register_in_order_and_share_resources() {
        let build = GameBuild::new()
            .with_mod(ModDescriptor { id: "test.provider", name: "Provider", version: "1.0.0", register: provider })
            .with_mod(ModDescriptor { id: "test.consumer", name: "Consumer", version: "1.0.0", register: consumer })
            .with_environment("sha256:00");
        let mods = build.mods();
        assert_eq!(mods.len(), 2);
        assert_eq!((mods.id(0), mods.package(0)), ("first", Some("test.provider")));
        assert_eq!((mods.id(1), mods.is_enabled(1)), ("second", false));
        assert_eq!(mods.groups().map(|g| g.id).collect::<Vec<_>>(), ["essentials", "tools"]);
        assert_eq!(build.environment(), Some("sha256:00"));
        assert!(GameBuild::vanilla().mods().is_empty());
    }
}
