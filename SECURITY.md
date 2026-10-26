# Security Policy

We take the security of FailureLens seriously. Thank you for helping keep the
project and its users safe.

## Supported versions

FailureLens is in early development. Security fixes target the `main` branch and
the most recent release (once releases exist).

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, report privately using GitHub's **"Report a vulnerability"** flow:

1. Go to the repository's **Security** tab.
2. Choose **Report a vulnerability** (GitHub Private Vulnerability Reporting).

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected versions/commit, and any suggested mitigation.

We will acknowledge your report, investigate, and keep you informed of progress.
Please give us a reasonable opportunity to fix the issue before any public
disclosure, and avoid accessing or modifying other users' data during testing.

## Scope notes

- The **baseline** performs no network access and does not execute the analyzed
  repository's code. Reports that require disabling these guarantees should say so
  explicitly.
- Optional model backends load **external** models; see
  [docs/security.md](docs/security.md) for safe model-loading guidance (pinned
  revisions, no `trust_remote_code` without review).

## Hardening guidance for users and contributors

Technical security guidance — model loading, dependency floors, input handling,
secrets, and licensing — lives in [docs/security.md](docs/security.md). Please
review it before adding dependencies or model backends.
