<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Third-party software and content

Project Watt Cubed uses free-software dependencies and retains their original
licences. Those components are not relicensed under the project's AGPL or
Creative Commons licences. This document is an inventory aid, not a substitute
for the dependencies' licence texts.

## Dependency notice process

`Cargo.lock` and `flake.lock` are the authoritative version and source pins for
a build. For each official release, the project should:

1. resolve the complete dependency graph for every supported target;
2. run `cargo deny check licenses`, `cargo deny check bans`, and
   `cargo deny check sources`;
3. generate an SPDX SBOM containing package name, exact version, source,
   checksum, and declared licence expression;
4. preserve required notices and full licence texts in the release;
5. make the exact dependency source available where a licence requires it; and
6. inspect the Nix runtime closure and build tools as well as Rust crates.

Unknown licences, mutable Git dependencies, proprietary terms, and
source-available-but-non-free terms fail the release gate. A project-owned
component may not adopt a permissive licence merely because compatible
third-party dependencies use one.

At the 2026-07-24 audit snapshot, `Cargo.lock` contained 336 packages: 334 from
the crates.io registry, no Git-sourced packages, and two local project crates.
The full locked graph passed `cargo deny check licenses bans sources`: all
packages satisfied the free-licence allowlist, registry/Git-source rules, and
version-bound dependency rule. Existing duplicate dependency versions remain
warnings. CI must repeat the all-target check for every release.

## Direct Rust dependencies

The current direct third-party dependencies are:

| Package | Locked version | Declared licence expression |
| --- | --- | --- |
| `ash` | `0.38.0+1.3.281` | `MIT OR Apache-2.0` |
| `cpal` | `0.18.1` | `Apache-2.0` |
| `glam` | `0.32.1` | `MIT OR Apache-2.0` |
| `kira` | `0.12.2` | `MIT OR Apache-2.0` |
| `opus-rs` | `0.1.23` | `BSD-3-Clause` |
| `quinn` | `0.11.11` | `MIT OR Apache-2.0` |
| `rcgen` | `0.14.8` | `MIT OR Apache-2.0` |
| `rtrb` | `0.3.4` | `MIT OR Apache-2.0` |
| `rustls` | `0.23.42` | `Apache-2.0 OR ISC OR MIT` |
| `tokio` | `1.53.0` | `MIT` |
| `toml` | `0.9.12+spec-1.1.0` | `MIT OR Apache-2.0` |

`voxel_engine` is a project-owned sibling crate, not a third-party dependency.
It is intended to carry the same `AGPL-3.0-or-later` software licence.

The Linux transitive graph also contains MPL-2.0 components, notably
`audio_thread_priority`, `triple_buffer`, and the `symphonia` family. Official
binary notices and source distribution must preserve their file-level copyleft
terms. Other observed expressions include BSD, ISC, Zlib, 0BSD, Unlicense,
Unicode-3.0, and Apache-2.0 with LLVM exception. The generated SBOM and notice
bundle, rather than this summary, must enumerate the complete release graph.

## Nix and native inputs

The Nix build pins `nixpkgs`, `flake-utils`, and `nix-systems/default` in
`flake.lock`; it also pins the sibling engine source. The shader compiler is
provided by the pinned nixpkgs revision when available. Release provenance
should record the actual compiler version and command used for every generated
SPIR-V file.

Linux runtime or build inputs include the Vulkan loader, Wayland, X11,
libxkbcommon, ALSA, D-Bus, pkg-config, and the Rust toolchain. Release SBOM and
notice generation must inspect their exact Nix closure rather than assigning
licences from this general list.

## Embedded `font8x8` data in the engine

The sibling engine embeds the ASCII table from `font8x8_basic` in
`../voxel-engine/src/font.rs`. The source comment credits Daniel Hepper and
Marcel Sondaar and identifies the upstream project as
<https://github.com/dhepper/font8x8>, file `font8x8_basic.h`.

The engine records the table as public-domain material; it does **not** identify
it as Creative Commons content, and this project does not relicense it as such.
The engine now pins upstream commit
`8e279d2d864e79128e96188a6b9526cfa3fbfef9`, records the source SHA-256, retains
the upstream notice under `LicenseRef-font8x8-Public-Domain`, marks the mixed
Rust file with an SPDX snippet, and provides `tools/gen_font.py` to reproduce
and verify the extracted table.

## Project-generated sound placeholders

The WAV files under `assets/sounds/generated/` are deterministic project output,
not third-party recordings. Their provenance and hashes are recorded in
`assets/attribution/generated-sounds.toml`; the sound files are
`CC-BY-SA-4.0`, while their generator is `AGPL-3.0-or-later`.

No third-party art, music, or recorded sound was found in the game repository at
the audit snapshot. New binary assets require an author-specific licence
sidecar and source/provenance record before they may ship.

## Historical supplied-patch provenance

The game commit `3b2b112` (“Apply supplied Watt Cubed patch”) and sibling-engine
commit `6c247bf` (“Apply supplied voxel-engine patch”) do not identify the
supplier in their commit metadata. During this audit, the maintainer confirmed
that they personally supplied both patches. They are therefore treated as
project contributions rather than unidentified third-party material.

Retain that provenance statement with the licensing-transition records. As with
the rest of the historical work, do not publish the licensing transition until
the affected human contributors' direct written confirmations have been
retained.
