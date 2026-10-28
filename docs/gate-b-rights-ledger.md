# Gate B source-rights ledger (Issue #11)

Rights were checked against license text at each pinned upstream commit. This confirms code/test use for local synthetic mutation and evaluation; it is not authorization to publish a derived dataset. Do not ingest user logs, third-party issue text, or vendored dependencies under these entries. License notices must accompany any redistribution.

| Source | Pin | License evidence | Approximate test-file count | Training use | Redistribution | Log rights | Status |
|---|---|---|---:|---|---|---|---|
| `psf/requests` | `3c7baaa05ab4aa49cc819ce3a819d260de19c157` | [`LICENSE`](https://github.com/psf/requests/blob/3c7baaa05ab4aa49cc819ce3a819d260de19c157/LICENSE), Apache-2.0 text | 9 | Yes | Yes; retain license/NOTICE | Own generated logs only | Rights-cleared source |
| `pallets/werkzeug` | `b73dce5a2ce14f7c7fd645699d1d878de48d2948` | [`LICENSE.txt`](https://github.com/pallets/werkzeug/blob/b73dce5a2ce14f7c7fd645699d1d878de48d2948/LICENSE.txt), BSD-3-Clause text | 27 | Yes | Yes; retain notice/disclaimer, no endorsement | Own generated logs only | Rights-cleared source |
| `pypa/pip` | `1000636317b856a7a8d0ae6662c3eb30c2a9bfed` | [`LICENSE.txt`](https://github.com/pypa/pip/blob/1000636317b856a7a8d0ae6662c3eb30c2a9bfed/LICENSE.txt), MIT text | 144 | Yes | Yes; retain notice | Own generated logs only | Rights-cleared source |

The license files were read at the pinned commits. Counts use `find tests -name 'test*.py' | wc -l`. These are source candidates, not accepted cases. Each record still requires pass-before/fail-after execution, secret/PII screening, and a root-cause rationale.
