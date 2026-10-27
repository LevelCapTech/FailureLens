"""Baseline candidate retrieval.

Resolves stack-trace frames to files inside the repository and turns the frames
that land in indexed code into candidate locations. This is the lexical/
structural baseline: no model, no network, fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from failurelens.indexing.base import RepositoryIndex
from failurelens.schemas import Evidence, EvidenceKind, StackFrame


def resolve_frame_path(frame_file: str | None, index: RepositoryIndex) -> str | None:
    """Resolve a traceback file path to a repo-relative indexed path.

    Stack traces frequently contain absolute paths from another machine, so we
    match by the longest path suffix that corresponds to an indexed file, then
    fall back to a unique basename match.
    """
    if not frame_file:
        return None
    normalized = frame_file.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if not parts:
        return None
    paths = set(index.paths())
    for start in range(len(parts)):
        suffix = "/".join(parts[start:])
        if suffix in paths:
            return suffix
    basename = parts[-1]
    matches = [p for p in paths if p.rsplit("/", 1)[-1] == basename]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_frames(frames: list[StackFrame], index: RepositoryIndex) -> list[StackFrame]:
    """Return copies of ``frames`` annotated with repository resolution info."""
    resolved: list[StackFrame] = []
    for frame in frames:
        repo_path = resolve_frame_path(frame.file, index)
        resolved.append(
            frame.model_copy(
                update={
                    "in_repository": repo_path is not None,
                    "repository_path": repo_path,
                }
            )
        )
    return resolved


@dataclass
class RawCandidate:
    """An un-scored candidate location accumulated from one or more frames."""

    file: str
    function: str | None
    qualified_name: str | None
    start_line: int | None
    end_line: int | None
    max_frame_order: int
    is_exception_site: bool = False
    evidence: list[Evidence] = field(default_factory=list)


def retrieve_candidates(frames: list[StackFrame], index: RepositoryIndex) -> list[RawCandidate]:
    """Build de-duplicated candidates from the in-repository frames."""
    in_repo = [f for f in frames if f.in_repository and f.repository_path]
    if not in_repo:
        return []

    exception_site_order = max(f.order for f in in_repo)
    candidates: dict[tuple[str, str | None], RawCandidate] = {}

    for frame in in_repo:
        assert frame.repository_path is not None
        file_index = index.file(frame.repository_path)
        unit = file_index.unit_at(frame.line) if (file_index and frame.line) else None
        qualified = unit.qualified_name if unit else None
        key = (frame.repository_path, qualified)

        candidate = candidates.get(key)
        if candidate is None:
            candidate = RawCandidate(
                file=frame.repository_path,
                function=unit.name if unit else None,
                qualified_name=qualified,
                start_line=unit.start_line if unit else None,
                end_line=unit.end_line if unit else None,
                max_frame_order=frame.order,
            )
            candidates[key] = candidate

        candidate.max_frame_order = max(candidate.max_frame_order, frame.order)
        location = f"{frame.repository_path}:{frame.line}"
        if frame.function:
            location += f" in {frame.function}"
        candidate.evidence.append(
            Evidence(
                kind=EvidenceKind.STACK_FRAME,
                description=f"Appears in stack frame {frame.order} ({location}).",
                frame_order=frame.order,
            )
        )
        if frame.order == exception_site_order:
            candidate.is_exception_site = True
            candidate.evidence.append(
                Evidence(
                    kind=EvidenceKind.EXCEPTION_SITE,
                    description="Deepest in-repository frame (closest to where the error surfaced).",
                    frame_order=frame.order,
                )
            )

    return list(candidates.values())
