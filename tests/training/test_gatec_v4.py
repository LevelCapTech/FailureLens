"""Tests for the Gate C v4 table checks (no network, no third-party code)."""

from __future__ import annotations

import pytest
from training.corpus import gatec_v2
from training.corpus.gatec_v3 import V3_REPOSITORIES
from training.corpus.gatec_v4 import (
    EARLIER_V3_SLUGS,
    V4_REPOSITORIES,
    validate_v4_table,
)
from training.corpus.repos import HOLDOUT_REPOSITORIES, REPOSITORIES, SourceRepo


def test_v4_table_is_valid_and_disjoint_from_everything_used() -> None:
    validate_v4_table()
    slugs = {r.slug.lower() for r in V4_REPOSITORIES}
    keys = {r.key for r in V4_REPOSITORIES}
    assert len(V4_REPOSITORIES) >= 5
    assert not slugs & {r.slug.lower() for r in REPOSITORIES}
    assert not slugs & {s.lower() for s in HOLDOUT_REPOSITORIES}
    assert not slugs & {s.lower() for s in gatec_v2.LEGACY_TRAIN_REPOSITORIES}
    assert not slugs & {r.slug.lower() for r in gatec_v2.EVAL_REPOSITORIES}
    assert not slugs & {r.slug.lower() for r in V3_REPOSITORIES}
    assert not slugs & {s.lower() for s in EARLIER_V3_SLUGS}
    assert not keys & {r.key for r in V3_REPOSITORIES}
    assert all(r.role == "eval" and len(r.commit) == 40 for r in V4_REPOSITORIES)


def test_v4_table_rejects_a_v3_source_and_a_v3_dropped_repository() -> None:
    for slug in (V3_REPOSITORIES[0].slug, "tartley/colorama"):
        clash = SourceRepo(**{**V4_REPOSITORIES[0].__dict__, "slug": slug, "key": "zzz"})
        with pytest.raises(ValueError, match="overlaps"):
            validate_v4_table((clash,))
