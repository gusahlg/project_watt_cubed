<!-- SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors -->
<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through a
[GitHub security advisory](https://github.com/gusahlg/project_watt_cubed/security/advisories/new).
Do not open a public issue for an unpatched vulnerability.

Include, when available:

- the affected commit, release, platform, and configuration;
- a concise impact description and realistic attack scenario;
- reproduction steps or a minimal proof of concept;
- relevant logs or crash output with credentials and personal data removed; and
- any suggested mitigation.

Reports concerning the sibling `voxel-engine` repository may use the same
private advisory channel when the issue affects Project Watt Cubed.

## What to expect

This is a volunteer project. Reports are reviewed and fixes are developed as
maintainer capacity permits. The project makes no promise of an acknowledgement
time, remediation deadline, supported-version lifetime, bounty, reward,
confidentiality period, or advance notice.

Maintainers will try to coordinate publication when practical. A temporary
security embargo may be requested, but it is not a contract or a promise of a
particular disclosure date. Security fixes and their source are published under
the project's normal licences.

Only the current development branch and releases explicitly identified by the
maintainers should be assumed to receive fixes. Older builds may remain
vulnerable.

## Responsible testing

This policy does not grant permission to test systems, accounts, or data you do
not own or have explicit authority to test. Please avoid:

- privacy violations or access to another person's data;
- destructive testing, denial of service, or resource exhaustion;
- social engineering, credential theft, or physical attacks; and
- retaining or publishing secrets encountered accidentally.

Use the smallest demonstration necessary to establish the issue and stop if
testing risks harm.

## Known transport limitation

Multiplayer uses QUIC with TLS 1.3 encryption, but the server currently creates
a fresh self-signed certificate and clients accept any server certificate.
Server identity is therefore not authenticated. Passive observers cannot read
the encrypted traffic, but an active man-in-the-middle can impersonate a server
and capture the application password. Use a trusted network or VPN until
certificate pinning or trust-on-first-use is implemented.

Passwords, private keys, access tokens, and unredacted user data must never be
included in public bug reports, logs, save files, or test fixtures.

## No warranty

This policy describes a reporting process, not a security guarantee or support
agreement. The software is provided without warranty under its applicable
licence.
