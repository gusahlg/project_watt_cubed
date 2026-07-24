<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Contributing to Project Watt Cubed

Project Watt Cubed is developed as a free-software game and ecosystem. Thank
you for helping improve it.

## Inbound equals outbound

By contributing, you agree that:

- software contributions are licensed under `AGPL-3.0-or-later`;
- artistic, audio, and written-content contributions are licensed under
  `CC-BY-SA-4.0`; and
- a file that clearly records another approved compatible licence remains
  under that licence.

Substantial code examples belong under `AGPL-3.0-or-later`, even when they
accompany documentation.

You retain your copyright. You do not assign it to the project, grant a
separate proprietary relicensing right, or agree to a proprietary
dual-licensing CLA. Project maintainers do not issue closed-source exceptions
for project-owned components.

## Developer Certificate of Origin

All commits must carry a Developer Certificate of Origin 1.1 sign-off. Add it
with:

```sh
git commit --signoff
```

The resulting trailer has this form:

```text
Signed-off-by: Your Legal Name <your-email@example.org>
```

The sign-off certifies that you have the right to submit the contribution
under the applicable project licence. It is not a copyright assignment. Do
not sign off for another person unless you are legally authorised to do so.
Web-based contributions must use the hosting platform's verified commit
sign-off feature.

The DCO requirement applies prospectively. Historical relicensing consent is
recorded separately in [LICENSING-AUDIT.md](LICENSING-AUDIT.md); a new DCO
sign-off does not retroactively replace that consent.

## Contribution requirements

Before submitting a change:

1. Confirm that you wrote it or have the right to submit it.
2. Identify copied, adapted, generated, or third-party material in the pull
   request, including its source and licence.
3. Do not add proprietary, source-available-only, or licence-unknown
   dependencies.
4. Include preferred editable sources for new art, audio, models, and similar
   content whenever they exist.
5. Add attribution and SPDX metadata for every new asset. Binary assets need
   a matching `.license` sidecar.
6. Explain material AI assistance. The human submitter remains responsible
   for reviewing the output and establishing that it contains no
   incompatible third-party material.
7. Keep builds reproducible: do not add unexplained prebuilt binaries or
   generation outputs without the corresponding generator and build recipe.
8. Run the relevant formatting, lint, test, and licence checks.

Third-party components keep their original compatible licences and notices.
Do not replace those notices with the Project Watt Cubed licence.

## Changes to licensing or governance

Ordinary contributions cannot weaken the project's copyleft or grant
proprietary exceptions. Proposed licensing, governance, trademark, mod-policy,
or contributor-policy changes require explicit review under
[GOVERNANCE.md](GOVERNANCE.md).
