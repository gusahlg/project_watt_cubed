<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Project Watt Cubed licensing

Project Watt Cubed is a free-software and free-content project. Commercial use
is allowed; the applicable copyleft and attribution requirements still apply.
Copyright remains with the individual contributors.

This file is a guide to the repository's licensing structure. A file's own
SPDX identifier or licence sidecar is authoritative when it records a different
compatible licence.

## Software

Project-owned software is licensed under the GNU Affero General Public License,
version 3 or any later version (`AGPL-3.0-or-later`). This includes:

- Rust source for the client, dedicated server, engine, protocol, save system,
  mod APIs, default mods, registry software, launchers, and development tools;
- build scripts, Nix expressions, CI configuration, schemas, software examples,
  and generated software metadata;
- shader source and generated shader binaries, including SPIR-V; and
- project-owned software distributed as part of an official mod.

The full licence text is in
[`LICENSES/AGPL-3.0-or-later.txt`](LICENSES/AGPL-3.0-or-later.txt).
When a modified covered program is offered for use over a network, the AGPL's
network-source requirements apply. Distributing object code also requires
providing the corresponding source in the manner required by the licence.

## Creative material and documentation

Project-owned artistic and written material is licensed under Creative Commons
Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`). This includes:

- textures, models, sound effects, music, UI artwork, and editable asset source;
- lore and other written game content; and
- documentation, design documents, and the prose of project policy files.

The full licence text is in
[`LICENSES/CC-BY-SA-4.0.txt`](LICENSES/CC-BY-SA-4.0.txt).

Substantial source-code examples inside documentation are
`AGPL-3.0-or-later`, even when the surrounding prose is
`CC-BY-SA-4.0`. Short command invocations and purely illustrative fragments do
not change the licence of the surrounding document. Larger examples should
normally live in an AGPL-licensed source file and be linked from the
documentation.

Generated creative assets carry the creative-content licence recorded in their
sidecar or attribution manifest. The generator remains software under
`AGPL-3.0-or-later`.

## Third-party material

Third-party code and content remain under their original licences. Their
inclusion does not relicense them under either project licence. Notices and
provenance are recorded in [`THIRD_PARTY.md`](THIRD_PARTY.md), the relevant
file headers or sidecars, and, where required, `LICENSES/third-party/`.

Do not add material with unknown provenance or redistribution rights. A
dependency's appearance in a lockfile is not itself a legal compatibility
determination.

## Names, logos, and identity

Copyright licences do not grant permission to misrepresent an unofficial fork,
server, registry, or service as official. Use of the Project Watt Cubed name,
logos, and other project identifiers is governed separately by
[`TRADEMARKS.md`](TRADEMARKS.md). That policy permits truthful references,
community discussion, and clearly distinguished forks.

## Contributions

Contributions use inbound-equals-outbound licensing:

- software contributions are `AGPL-3.0-or-later`;
- creative-content contributions are `CC-BY-SA-4.0`; and
- an explicitly identified third-party file may retain another approved,
  compatible licence.

Contributors retain copyright. The project does not require copyright
assignment, a proprietary-relicensing CLA, or a special closed-source
exception. Contributions must carry a Developer Certificate of Origin 1.1
sign-off as described in `CONTRIBUTING.md`.

This repository is provided without warranty, as described by the applicable
licences.
