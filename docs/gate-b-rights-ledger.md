# Gate B source-rights ledger (Issue #11)

Rights were checked against license text at each pinned upstream commit. This confirms code/test use for local synthetic mutation and evaluation; it is not authorization to publish a derived dataset. Do not ingest user logs, third-party issue text, or vendored dependencies under these entries. License notices must accompany any redistribution.

| Source | Pin | License evidence | Approximate test-file count | Training use | Redistribution | Log rights | Status |
|---|---|---|---:|---|---|---|---|
| `psf/requests` | `3c7baaa05ab4aa49cc819ce3a819d260de19c157` | [`LICENSE`](https://github.com/psf/requests/blob/3c7baaa05ab4aa49cc819ce3a819d260de19c157/LICENSE), Apache-2.0 text | 9 | Yes | Yes; retain license/NOTICE | Own generated logs only | Rights-cleared source |
| `pallets/werkzeug` | `b73dce5a2ce14f7c7fd645699d1d878de48d2948` | [`LICENSE.txt`](https://github.com/pallets/werkzeug/blob/b73dce5a2ce14f7c7fd645699d1d878de48d2948/LICENSE.txt), BSD-3-Clause text | 27 | Yes | Yes; retain notice/disclaimer, no endorsement | Own generated logs only | Rights-cleared source |
| `pypa/pip` | `1000636317b856a7a8d0ae6662c3eb30c2a9bfed` | [`LICENSE.txt`](https://github.com/pypa/pip/blob/1000636317b856a7a8d0ae6662c3eb30c2a9bfed/LICENSE.txt), MIT text | 144 | Yes | Yes; retain notice | Own generated logs only | Rights-cleared source |

The license files were read at the pinned commits. Counts use `find tests -name 'test*.py' | wc -l`. These are source candidates, not accepted cases. Each record still requires pass-before/fail-after execution, secret/PII screening, and a root-cause rationale.


## v0.2 corpus sources (train and dev; added 2026-11-07)

Additional permissively licensed, pure-Python sources for the v0.2 private synthetic corpus (`docs/corpus-v02.md`). Each license file was read at the pinned commit and its text classified (MIT or BSD-3-Clause) by `training/corpus/build.py prepare`, which also checks that the pin is the checked-out HEAD and that the clone is clean. The same conditions as above apply: local synthetic mutation and own-generated logs only, no authorization to redistribute derived records, retain notices on any redistribution, no endorsement. `python-attrs/attrs` (MIT, pin in `docs/issue17-private-training-handoff.md`) stays train-only and is not regenerated here.

The splits are by repository. **Train and dev share no repository, and neither contains `psf/requests`, `pallets/werkzeug` or `pypa/pip`** (the Gate C hold-out above); `training/corpus/repos.py` fails closed if that is violated.

| Source | Pin | License evidence | Split | Status |
|---|---|---|---|---|
| `pallets/click` | `2247b35ea1c47c727d7a06e51fa280e12a863ff6` | [`LICENSE.txt`](https://github.com/pallets/click/blob/2247b35ea1c47c727d7a06e51fa280e12a863ff6/LICENSE.txt), BSD-3-Clause text (blob SHA-256 `9a8ad106a394e853...`) | train | Rights-cleared source; probe validation pending |
| `more-itertools/more-itertools` | `f42daddf205f335fea6e200dad46049f33e71221` | [`LICENSE`](https://github.com/more-itertools/more-itertools/blob/f42daddf205f335fea6e200dad46049f33e71221/LICENSE), MIT text (blob SHA-256 `09f1c8c9e941af3e...`) | train | Rights-cleared source; probe validation pending |
| `pytoolz/toolz` | `451af60dec590a6010e2babdbf391ea8f815122f` | [`LICENSE.txt`](https://github.com/pytoolz/toolz/blob/451af60dec590a6010e2babdbf391ea8f815122f/LICENSE.txt), BSD-3-Clause text (blob SHA-256 `053664057b295b2f...`) | train | Rights-cleared source; probe validation pending |
| `astanin/python-tabulate` | `268615a5c27dc40e5c22454c07b44d5c50410da0` | [`LICENSE`](https://github.com/astanin/python-tabulate/blob/268615a5c27dc40e5c22454c07b44d5c50410da0/LICENSE), MIT text (blob SHA-256 `cdfab50d37d8ead0...`) | train | Rights-cleared source; probe validation pending |
| `kjd/idna` | `d357725082245ff90f5cf8800a3fc297b1f2319a` | [`LICENSE.md`](https://github.com/kjd/idna/blob/d357725082245ff90f5cf8800a3fc297b1f2319a/LICENSE.md), BSD-3-Clause text (blob SHA-256 `1a9a4f0e3d479a27...`) | train | Rights-cleared source; probe validation pending |
| `jpvanhal/inflection` | `88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1` | [`LICENSE`](https://github.com/jpvanhal/inflection/blob/88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1/LICENSE), MIT text (blob SHA-256 `9e9f9b4b2e7543ad...`) | train | Rights-cleared source; probe validation pending |
| `pallets/itsdangerous` | `672971d66a2ef9f85151e53283113f33d642dabd` | [`LICENSE.txt`](https://github.com/pallets/itsdangerous/blob/672971d66a2ef9f85151e53283113f33d642dabd/LICENSE.txt), BSD-3-Clause text (blob SHA-256 `63af09891b6be8ad...`) | train | Rights-cleared source; probe validation pending |
| `andialbrecht/sqlparse` | `09dbe1eec31ed1b3cd05366748865f11ba427d32` | [`LICENSE`](https://github.com/andialbrecht/sqlparse/blob/09dbe1eec31ed1b3cd05366748865f11ba427d32/LICENSE), BSD-3-Clause text (blob SHA-256 `c1938235b80d39e9...`) | dev | Rights-cleared source; probe validation pending |
| `python-humanize/humanize` | `d16e6352c9785c5ab86de0133222bd22e89b0db2` | [`LICENCE`](https://github.com/python-humanize/humanize/blob/d16e6352c9785c5ab86de0133222bd22e89b0db2/LICENCE), MIT text (blob SHA-256 `8ba6c18112a43140...`) | dev | Rights-cleared source; probe validation pending |
| `mahmoud/boltons` | `d31d821d56c8f7756af9c30a864077f2e5855321` | [`LICENSE`](https://github.com/mahmoud/boltons/blob/d31d821d56c8f7756af9c30a864077f2e5855321/LICENSE), BSD-3-Clause text (blob SHA-256 `617273daf4b7b4fb...`) | dev | Rights-cleared source; probe validation pending |
| `jmespath/jmespath.py` | `2812594e69d43098ef60f81f4efc404c071b0418` | [`LICENSE`](https://github.com/jmespath/jmespath.py/blob/2812594e69d43098ef60f81f4efc404c071b0418/LICENSE), MIT text (blob SHA-256 `6eefacfa4d71b82d...`) | dev | Rights-cleared source; probe validation pending |

These are source candidates, not accepted cases. Records exist only after the target project's own tests pass on the clean tree and fail after one exact single-line mutation, and pass the schema and semantic gates.


## Gate C v2 evaluation sources (final-evaluation set; added 2026-11-07)

Sources for a fresh, private final-evaluation set (`docs/gate-c-v2.md`). Each license file was read at the pinned commit and its text classified as MIT by `training/corpus/build.py prepare`. The same conditions as above apply: local synthetic mutation and own-generated logs only, no authorization to redistribute derived records, retain notices on any redistribution, no endorsement.

**These repositories are disjoint from every repository in the first hold-out (`psf/requests`, `pallets/werkzeug`, `pypa/pip`), from all v0.2 train and dev repositories above, from the legacy training repositories (`pallets/click`, `python-attrs/attrs`), and from any repository added to a training manifest.** `training/corpus/gatec_v2.py` fails closed on any overlap and rewrites only the training manifest when the training list changes.

| Source | Pin | License evidence | Split | Status |
|---|---|---|---|---|
| `tkem/cachetools` | `1ce79efa552d1bdda70d144d401ef9105d999184` | [`LICENSE`](https://github.com/tkem/cachetools/blob/1ce79efa552d1bdda70d144d401ef9105d999184/LICENSE), MIT text (blob SHA-256 `28c000b52b0ee27138a68ef778227e4057046e86b65f62f1cacb99b0cc49e0d2`) | evaluation | Rights-cleared source |
| `PyCQA/pycodestyle` | `10a4427c75740717b43448339fcf71f11bc33d1a` | [`LICENSE`](https://github.com/PyCQA/pycodestyle/blob/10a4427c75740717b43448339fcf71f11bc33d1a/LICENSE), MIT (Expat) text (blob SHA-256 `f772295e81af3478d34e88e52d076200230ec7dd6a39e123bc5cb35aa66abdae`) | evaluation | Rights-cleared source |
| `python-validators/validators` | `e2e49fb3bb4b9f1ce955638386aa167356d24055` | [`LICENSE.txt`](https://github.com/python-validators/validators/blob/e2e49fb3bb4b9f1ce955638386aa167356d24055/LICENSE.txt), MIT text (blob SHA-256 `9567be0573324442d69eaf0b3e1e91c3f05fb4736b3202ab83eadbb4aff2f0b5`) | evaluation | Rights-cleared source |
| `bottlepy/bottle` | `b0a0a1014f7a7e6f6b7c193886a0913f46ee3ee6` | [`LICENSE`](https://github.com/bottlepy/bottle/blob/b0a0a1014f7a7e6f6b7c193886a0913f46ee3ee6/LICENSE), MIT text (blob SHA-256 `af150257b1f7cce92b6207854f2e8f90ea9ae52d7cac758d13a47bd12e2a2443`) | evaluation | Rights-cleared source |

These are source candidates, not accepted cases. Each record needs the project's own tests to pass on the clean tree and fail after one exact edit, plus the schema and semantic gates.

## v0.3 corpus sources (train only; added 2026-11-07)

Four more permissively licensed, pure-Python sources for the v0.3 private synthetic corpus (`docs/corpus-v03.md`). They were chosen from repositories whose names start with a-m so that they do not overlap a separately selected evaluation set. As above, each license file was read at the pinned commit and classified by `training/corpus/build.py prepare`, which also checks that the pin is the checked-out HEAD and that the clone is clean. The same conditions apply: local synthetic mutation and own-generated logs only, no authorization to redistribute derived records, retain notices on any redistribution, no endorsement. They join the v0.2 train repositories; the four v0.2 dev repositories are reused for v0.3 dev. **No v0.3 train or dev repository is `psf/requests`, `pallets/werkzeug` or `pypa/pip`**, and `training/corpus/repos.py` fails closed if that is violated.

| Source | Pin | License evidence | Split | Status |
|---|---|---|---|---|
| `jawah/charset_normalizer` | `7d74fe4a8341ae27a8f42c759793301dbd304e10` | [`LICENSE`](https://github.com/jawah/charset_normalizer/blob/7d74fe4a8341ae27a8f42c759793301dbd304e10/LICENSE), MIT text (blob SHA-256 `6d0d41bfe170ac6c...`) | train | Rights-cleared source |
| `micheles/decorator` | `956b6396c7369aab4acc7a44851572fcb92fb34a` | [`LICENSE.txt`](https://github.com/micheles/decorator/blob/956b6396c7369aab4acc7a44851572fcb92fb34a/LICENSE.txt), BSD-2-Clause text (blob SHA-256 `bd12cc51405c1087...`) | train | Rights-cleared source |
| `executablebooks/mdurl` | `524d2edbbcb8bb48301ba716c7482827bcabb281` | [`LICENSE`](https://github.com/executablebooks/mdurl/blob/524d2edbbcb8bb48301ba716c7482827bcabb281/LICENSE), MIT text (blob SHA-256 `7c605df6e28667a9...`) | train | Rights-cleared source |
| `lepture/mistune` | `f7ec702edfc1ad5579a76f881238841af266d93c` | [`LICENSE`](https://github.com/lepture/mistune/blob/f7ec702edfc1ad5579a76f881238841af266d93c/LICENSE), BSD-3-Clause text (blob SHA-256 `539013fd8e19f744...`) | train | Rights-cleared source |

These are source candidates, not accepted cases. Records exist only after the target project's own tests pass on the clean tree and fail after one exact mutation, and pass the schema and semantic gates.
