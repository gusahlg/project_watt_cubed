<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Official Project Watt Cubed mod policy

The official mod ecosystem exists to distribute software that every recipient
can study, modify, rebuild, and share. Commercial free-software mods are
welcome. Binary-only or proprietary mods are not accepted by the official
registry.

This policy governs official registries, signing infrastructure, clients, and
mod listings. It does not add restrictions to private activity or to
independent forks beyond the licences of the relevant code and content.

## Licensing

All original software code in a submitted mod must be licensed
`AGPL-3.0-or-later`. This includes client, server, shared, and tool components,
build scripts, generated software source, and project-specific SDK glue.

Source dependencies must be free software under licences accepted by the
registry's published compatibility allowlist. Their notices and source must be
preserved. A dependency may retain a compatible free-software licence; that
does not permit the mod's own code to use a permissive or proprietary licence.

Mod assets must have complete provenance and use an approved free-content
licence. The initial approved list is:

- `CC-BY-SA-4.0`;
- `CC-BY-4.0`;
- `CC0-1.0`;
- `OFL-1.1` for fonts; and
- `FAL-1.3`.

A work claimed to be in the public domain requires evidence of a valid
dedication or expired copyright; a bare assertion is insufficient. Additional
free-content licences may be added to the registry allowlist after compatibility
and notice review. Noncommercial, no-derivatives, source-available-only, and
custom field-of-use restrictions are not accepted.

Receiving payment for a mod, support, or development is allowed. Recipients
must retain all freedoms and source rights required by the applicable licences.

## Complete source

Every submission must identify a public, immutable source revision and include
everything needed to rebuild and modify the package, including:

- preferred source forms for code and assets;
- exact dependency versions and integrity hashes;
- build and test instructions;
- generators, schemas, migrations, and configuration;
- licence texts, copyright notices, and attribution;
- an SPDX software bill of materials; and
- the source revision, build recipe, and hashes represented by the package.

Obfuscated source, missing editable asset originals, proprietary toolchain
requirements, and prebuilt dependencies without corresponding source are
grounds for rejection.

## Package and execution model

Official user-installable mods must target the versioned, sandboxed WebAssembly
mod ABI once that ABI and registry are available. Native dynamic libraries and
arbitrary native executables are not accepted as official mods.

A package declares whether each component is `server`, `client`, `shared`, or
`tool`, along with its ABI and game-version requirements. It must declare every
requested capability. The host grants only declared capabilities; mods receive
snapshots and capability handles and submit typed intents rather than taking
unrestricted mutable access to authoritative game state.

A package is expected to contain or reference its manifest, WebAssembly
modules, assets, licence information, source metadata, SBOM, build recipe, and
registry signature. Gameplay changes require an authoritative server component;
client-only code may provide presentation and accessibility features but may
not determine authoritative outcomes.

## Registry build and publication

The official registry builds from source rather than accepting an author's
opaque executable as the release artifact. For every accepted revision it:

1. verifies declared licences, provenance, dependencies, and capabilities;
2. builds in an isolated environment from pinned inputs;
3. runs tests, ABI checks, and determinism checks where applicable;
4. compares clean rebuilds and rejects unexplained output drift;
5. generates an SPDX SBOM, build log, and content hashes;
6. signs the resulting package; and
7. publishes the package, exact corresponding source, notices, and build
   metadata together.

The official client installs registry-built, signed packages by default and
shows each mod's source and licence. Because the client is free software, users
and forks remain able to use other distribution channels; registry signing is a
statement of provenance, not DRM.

Packages that add proprietary analytics, advertising SDKs, launchers, account
requirements, or undisclosed network behaviour are not accepted.

## Multiplayer and updates

Required mods participate in the canonical multiplayer content fingerprint,
including mod ID and version, source revision, WebAssembly hash, asset hash,
content-registration hash, ABI version, save-schema version, and deterministic
configuration. Incompatible clients must be rejected before world entry with a
specific explanation.

An update is a new reviewed source revision and registry build. A signature does
not carry over to changed bytes.

## Enforcement and appeals

The registry may reject, quarantine, or remove a package that lacks source,
misstates its licence or provenance, violates declared capabilities, contains
malware, or cannot be reproduced. Security-sensitive details may be temporarily
withheld during coordinated remediation, but corrected releases use the normal
free licences.

Authors should receive a concrete reason and may submit a corrected revision or
request maintainer review. Trademark permission is separate from registry
acceptance; listing a mod does not make it an official project release.
