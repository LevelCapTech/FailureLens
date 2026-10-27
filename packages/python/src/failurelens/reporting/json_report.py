"""Rendering of :class:`~failurelens.schemas.DiagnosisResult` for output.

Two renderers are provided:

* :func:`to_json` - stable, versioned JSON (identical to the Python API data).
* :func:`to_text` - a compact, human-readable summary for terminals.

Both are pure functions so they are trivial to unit-test and reuse.
"""

from __future__ import annotations

from failurelens.schemas import DiagnosisResult


def to_json(result: DiagnosisResult, *, indent: int | None = 2) -> str:
    """Serialize a result to JSON."""
    return result.to_json(indent=indent)


def _format_location(file: str | None, start: int | None, end: int | None) -> str:
    if file is None:
        return "<unknown location>"
    if start is None:
        return file
    if end is None or end == start:
        return f"{file}:{start}"
    return f"{file}:{start}-{end}"


def to_text(result: DiagnosisResult, *, show_evidence: bool = True) -> str:
    """Render a result as a human-readable report."""
    lines: list[str] = []
    header = f"FailureLens diagnosis [{result.status.value}]"
    lines.append(header)
    lines.append("=" * len(header))
    if result.error_type or result.error_message:
        error = result.error_type or "Error"
        if result.error_message:
            error += f": {result.error_message}"
        lines.append(f"Error    : {error}")
    lines.append(f"Method   : {result.analysis_method}")
    if result.model_id:
        revision = result.model_revision or "unpinned"
        lines.append(f"Model    : {result.model_id} ({revision})")
    lines.append(f"Time     : {result.execution_time_ms:.1f} ms")
    lines.append("")

    if not result.candidates:
        lines.append("No candidates were identified.")
    else:
        lines.append(f"Top {len(result.candidates)} candidate(s):")
        for candidate in result.candidates:
            location = _format_location(candidate.file, candidate.start_line, candidate.end_line)
            function = f" {candidate.function}()" if candidate.function else ""
            lines.append(
                f"  {candidate.rank}. {location}{function}  "
                f"[score={candidate.score:.3f} ({candidate.score_type.value})]"
            )
            if show_evidence:
                for item in candidate.evidence:
                    lines.append(f"       - {item.description}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  ! {warning}")

    return "\n".join(lines)


__all__ = ["to_json", "to_text"]
