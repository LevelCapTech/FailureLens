"""Baseline ranking of retrieved candidates.

Scores are a transparent weighted sum of a few interpretable features:

* **depth** - how deep in the call stack the candidate appears (frames closer to
  where the error surfaced score higher);
* **exception site** - a small boost for the deepest in-repository frame;
* **lexical overlap** - identifiers / quoted literals from the error message that
  also appear in the candidate's source span.

The result is a *heuristic* score (:class:`~failurelens.schemas.ScoreType`),
explicitly **not** a probability. Ties are broken deterministically.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from failurelens.retrieval.baseline import RawCandidate
from failurelens.schemas import Candidate, Evidence, EvidenceKind, ScoreType

SourceReader = Callable[[str], str | None]

_WEIGHT_DEPTH = 0.6
_WEIGHT_EXCEPTION_SITE = 0.2
_WEIGHT_LEXICAL = 0.4
_LEXICAL_CAP = 3

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")

# Common words that carry no localizing signal if matched.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "was",
        "not",
        "with",
        "from",
        "that",
        "this",
        "object",
        "attribute",
        "argument",
        "error",
        "exception",
        "none",
        "type",
        "value",
        "list",
        "dict",
        "str",
        "int",
    }
)


def extract_error_tokens(error_message: str | None) -> set[str]:
    """Extract candidate identifier/literal tokens from an error message."""
    if not error_message:
        return set()
    tokens: set[str] = set()
    for match in _QUOTED_RE.finditer(error_message):
        literal = match.group(1) or match.group(2) or ""
        literal = literal.strip()
        if literal:
            tokens.add(literal)
    for match in _IDENTIFIER_RE.finditer(error_message):
        token = match.group(0)
        if token.lower() not in _STOPWORDS:
            tokens.add(token)
    return tokens


def _lexical_matches(
    candidate: RawCandidate, tokens: set[str], source_reader: SourceReader | None
) -> list[str]:
    if not tokens or source_reader is None or candidate.start_line is None:
        return []
    source = source_reader(candidate.file)
    if source is None:
        return []
    lines = source.splitlines()
    start = max(candidate.start_line - 1, 0)
    end = candidate.end_line if candidate.end_line is not None else candidate.start_line
    span = "\n".join(lines[start:end])
    return sorted({token for token in tokens if token in span})


def rank_candidates(
    raw_candidates: list[RawCandidate],
    *,
    frame_count: int,
    error_message: str | None,
    source_reader: SourceReader | None = None,
    top_k: int = 5,
) -> list[Candidate]:
    """Score, sort and truncate candidates into ranked :class:`Candidate` objects."""
    tokens = extract_error_tokens(error_message)
    denominator = max(frame_count, 1)
    scored: list[tuple[float, int, RawCandidate, list[Evidence]]] = []

    for candidate in raw_candidates:
        depth_score = (candidate.max_frame_order + 1) / denominator
        exception_bonus = 1.0 if candidate.is_exception_site else 0.0

        matched = _lexical_matches(candidate, tokens, source_reader)
        lexical_score = min(len(matched), _LEXICAL_CAP) / _LEXICAL_CAP

        extra_evidence = [
            Evidence(
                kind=EvidenceKind.LEXICAL_MATCH,
                description=f"Error token {token!r} occurs in the candidate's source.",
            )
            for token in matched[:_LEXICAL_CAP]
        ]

        score = (
            _WEIGHT_DEPTH * depth_score
            + _WEIGHT_EXCEPTION_SITE * exception_bonus
            + _WEIGHT_LEXICAL * lexical_score
        )
        scored.append((score, candidate.max_frame_order, candidate, extra_evidence))

    # Sort by score, then by frame depth, then by path for stable, deterministic output.
    scored.sort(key=lambda item: (item[0], item[1], item[2].file), reverse=True)

    results: list[Candidate] = []
    for rank, (score, _order, candidate, extra_evidence) in enumerate(scored[:top_k], start=1):
        results.append(
            Candidate(
                rank=rank,
                file=candidate.file,
                function=candidate.function,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                score=round(score, 6),
                score_type=ScoreType.HEURISTIC,
                evidence=[*candidate.evidence, *extra_evidence],
            )
        )
    return results
