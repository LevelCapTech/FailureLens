# Tooling

Repository-level developer and CI tooling that is not part of the published
Python package.

| Path | Purpose | Status |
|------|---------|--------|
| `scripts/` | Helper scripts for maintainers (housekeeping, checks). | Planned |
| `containers/` | Dockerfiles / container definitions for reproducible environments. | Planned |

Prefer the root `Makefile` targets for everyday tasks (`make check`, `make test`,
`make build`). Add tooling here only when a task does not belong in the package
or the Makefile.

> Status: **scaffolding.** Add scripts/containers as concrete needs arise.
