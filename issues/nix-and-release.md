# Nix and release reproducibility follow-up

This report is the second audit pass triggered by a real `nix run` failure on
2026-07-13. It separates what was repaired immediately from release-engineering
work that should remain visible on the roadmap.

## Incident: Cargo and Nix selected different engines

The game had already merged the new LOD/render API, but the locked Nix input
still contained a pre-merge voxel engine. The pure build therefore compiled the
current game against an engine without `Detail`, `RenderFlags::vignette`, or
`Frame3D::draw_mesh_faded`.

This was not a Rust source regression. It was dependency-source skew:

- `Cargo.toml` uses the live sibling path `../voxel-engine`;
- the flake used a content-addressed snapshot recorded in `flake.lock`;
- the engine merge and GPU follow-ups changed the live checkout, but the lock
  was not refreshed with the game merge;
- ordinary `cargo build` consequently passed while pure `nix run` failed.

The old lock entry had NAR hash
`sha256-FseCETwIsIisxny0jS+liOaKYpCULs9/C6BWOPdG4fE=`. The corrected entry pins
engine revision `6edce7470eeafcf95426ca24445b02634462a173`, which contains the
LOD API plus both audit GPU-safety commits.

## Repairs made in this pass

### 1. Pin the compatible engine revision

`flake.lock` now resolves the same committed engine API that the game was
tested against. This directly fixes the reported missing-import, missing-field,
and missing-method errors.

### 2. Stop copying build products into the Nix store

The engine input was a raw `path:` source. Because that source is not Git
filtered, its ignored `target/` directory was included. Measured store sizes:

- old raw engine/source snapshot: about **5.0 GiB**;
- Git-filtered engine revision: about **944 KiB** by `nix path-info` (about
  1.2 MiB apparent size).

The input is now `git+file:` at the local `experimental` branch. This records an
exact commit, excludes ignored/untracked build products, and cut the observed
unpack phase from roughly 47 seconds to effectively immediate. A new
`checks.engine-source-budget` flake check fails if the input grows beyond a
generous 16 MiB, making a future source-filter regression explicit.

### 3. Make the package build a release check

`checks.package` points at the pure package derivation. The important contract
is now executable through `nix flake check`: compile the game against the locked
engine, run the release test suite, install the binaries, and complete fixup.
This catches cross-repository API skew in the environment where it matters.

### 4. Remove package-version drift

The Nix derivation still advertised `0.1.0` while `Cargo.toml` declares `0.2.0`.
The flake now reads the Cargo package version, leaving one source of truth.

### 5. Treat every installed binary as a package contract

The package installed `project_watt_cubed`, `watt_server`, `golden`, and
`golden_compare`, but only the first two received the Vulkan/window-library
runpath. Running the packaged `golden` reproduced an immediate
`LibraryOpenError` for `libX11.so.6`. The fixup now patches all four binaries and
no longer hides missing binaries or `patchelf` failures behind `|| true`.

## Verification record

The repaired pure package was checked independently of the live Cargo target:

```sh
nix flake check --print-build-logs

env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
  WATT_BENCH=1 \
  VOXEL_ENGINE_VALIDATION=1 \
  RUST_LOG=info \
  nix run .
```

The flake build compiled `project_watt_cubed 0.2.0` against the locked engine
and ran 342 library tests: 340 passed, zero failed, and two intentional tests
were ignored. The packaged runtime then enabled Vulkan validation, recreated
FIFO/immediate swapchains at 1280×720 and 1920×1080, rendered 55 benchmark
frames at 54 average FPS / 52 FPS 1% low, and exited 0 without a validation
warning or error.

## Remaining release opportunities

### R-01 — Replace the machine-local engine URL

**Priority: P1.** The new Git input is small and revisioned, but its URL still
contains `/home/gusahlg/repos/voxel-engine`. A clone on another machine cannot
resolve it. Once commits `00d7803`, `5efc734`, and `6edce74` are published, use a
remote Git URL and let `flake.lock` carry the exact revision. Do not return to a
raw absolute `path:` input.

### R-02 — Give breaking engine APIs an explicit compatibility identity

**Priority: P1.** The voxel engine crate remains version `0.1.0` across breaking
LOD/render API changes. Add a deliberate engine version/API revision and require
it from the game (Cargo supports a version requirement alongside a path during
local development). A stale package should fail with “engine API revision does
not match,” not three incidental missing-symbol errors.

For changes spanning both repositories, the release checklist should be atomic:

1. engine tests and commit;
2. game update and tests against that exact commit;
3. engine-input lock refresh;
4. pure flake check;
5. packaged finite renderer smoke;
6. publish both revisions together.

### R-03 — Unify dependency truth where practical

**Priority: P1.** Live-path Cargo and locked-input Nix intentionally answer
different questions, but the difference is easy to forget. Longer-term options
are a small super-workspace/release flake that owns both repositories, or a
remote Git Cargo dependency used by release builds with a local patch override
for development. Whichever model is chosen, one manifest should identify the
release engine revision.

### R-04 — Split product binaries from repository tools

**Priority: P2.** The default Nix package installs the game, dedicated server,
golden harness, and PNG comparison utility as one output. The golden tools
assume repository-relative fixtures/capture directories, while the headless
server needlessly inherits the GUI/Vulkan runpath and closure.

Prefer separate outputs/apps:

- `packages.default` / `apps.default`: the game;
- `packages.server` / `apps.server`: a genuinely headless closure;
- `packages.golden`: harness plus explicitly installed fixtures;
- a development-only compare tool if it is not useful outside the checkout.

Add one startup/`--help` smoke per shipped executable. Installation should not
silently turn internal test utilities into unsupported user-facing tools.

### R-05 — Stop claiming unsupported default systems

**Priority: P2.** `eachDefaultSystem` advertises Linux and Darwin outputs, but
the package fixup is ELF/`patchelf` specific and the runtime library set is
Linux X11/Wayland specific. Either implement a real Darwin/MoltenVK package path
or expose package/check outputs only on supported Linux systems while retaining
a clearly documented Cargo workflow for macOS.

### R-06 — Run the locked engine's own suite in CI

**Priority: P2.** The pure game package compiles the engine as a dependency but
does not run the engine's 71 unit tests. A coordinated CI job should run both
repositories' suites, the pure package check, and a finite validation smoke.
Compilation proves API compatibility; it does not prove renderer-internal
invariants.

### R-07 — Make lock freshness visible without surprising mutation

**Priority: P3.** `play.sh` intentionally refreshes the engine input before
running, which is convenient but mutates a tracked release artifact as a side
effect of launching the game. Consider a `check-engine-pin` command that compares
the local committed engine HEAD to the locked revision and prints one exact
remediation command. Keep `play.sh` as the explicit “refresh then run” path, and
use `nix run`/`nix build --no-update-lock-file` for reproducibility checks.

Uncommitted engine edits should continue to be tested through `nix develop -c
cargo ...`; they cannot honestly be part of a revision-pinned release package.

### R-08 — Keep shader provenance inside the release contract

**Priority: P2.** The broader audit already recommends separating shader
generation from ordinary builds. Release packaging should additionally record
the Slang version/options and validate every checked-in SPIR-V module before
building the game. Otherwise a pure Rust build can be reproducible while the
GPU program inventory remains compiler- or worktree-dependent.

### R-09 — Move persistent state out of the launch directory

**Priority: P1.** Saves, graphics settings, and remembered session data all use
relative `saves/...` paths. That works when launching from the repository, but a
desktop entry, service, read-only directory, or arbitrary shell directory can
silently create a second save location or make persistence fail.

Centralize a configurable application-path service: XDG data for worlds,
XDG config for settings/session metadata, and an explicit `--data-dir` or
`--world-dir` for the dedicated server. Tests should inject a temporary root.
Launch from an unwritable current directory with temporary XDG variables and
verify that every write lands under the configured root.

### R-10 — Decide how visual references travel

**Priority: P2.** `.gitignore` excludes all of `tests/golden`, while the golden
harness and comparison tool expect those repository-relative references. That
explains why the tool is useful locally but means a clean clone or packaged
binary cannot reproduce image comparisons.

Choose explicitly between tracked platform/renderer-keyed references, CI
artifacts downloaded by hash, or device-independent structural captures that
can be versioned normally. Package `golden` only with its declared fixtures, or
classify it as a checkout-only development tool rather than a standalone binary.

### R-11 — Establish a clean-checkout CI baseline

**Priority: P1.** No CI workflow is currently present. A clean-checkout job
would have caught the stale engine input, raw-source inflation, package-version
drift, and missing golden runpath even though local Cargo and renderer tests
were green.

Start with the gates that are already green: both test suites, pure Nix package,
source budget, server help smoke, and finite validation run. Formatting and
Clippy currently have documented pre-existing failures; burn those baselines
down or adopt an explicit changed-lines policy before making them hard gates.

## Suggested release gate

After replacing the local input with a publishable remote, a useful minimal gate
is:

```sh
cargo test --manifest-path ../voxel-engine/Cargo.toml --all-targets
cargo test --all-targets
nix flake check --no-update-lock-file
nix build --no-link --no-update-lock-file
```

Follow it on a validation-capable runner with the finite packaged benchmark used
above. Keep the 16 MiB source budget check: it is cheap, deterministic, and
guards against a measured multi-gigabyte regression rather than a hypothetical
one.
