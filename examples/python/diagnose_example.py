#!/usr/bin/env python
"""Runnable example: diagnose the bundled sample failure with the baseline.

Run it from anywhere:

    python examples/python/diagnose_example.py

It uses only the core install (no ML dependencies, no network).
"""

from __future__ import annotations

from pathlib import Path

from failurelens import AnalysisStatus, FailureLens

# Resolve the bundled fixture relative to this file so the script is runnable
# from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPO = REPO_ROOT / "examples" / "fixtures" / "sample_project"
SAMPLE_LOG = SAMPLE_REPO / "failure.log"


def main() -> None:
    lens = FailureLens.from_baseline()
    result = lens.diagnose(repository=str(SAMPLE_REPO), error_log=str(SAMPLE_LOG), top_k=5)

    print(f"status        : {result.status}")
    print(f"error         : {result.error_type}: {result.error_message}")
    print(f"analysis_method: {result.analysis_method}")
    print(f"time          : {result.execution_time_ms:.2f} ms")

    if result.status is not AnalysisStatus.OK:
        print("\nNo ranked candidates. Warnings:", result.warnings)
        return

    print(f"\nTop {len(result.candidates)} candidate(s):")
    for c in result.candidates:
        location = f"{c.file}:{c.start_line}-{c.end_line}" if c.file else "(location unknown)"
        print(f"  #{c.rank} {location} {c.function or ''} [score={c.score:.3f} ({c.score_type})]")
        for ev in c.evidence:
            print(f"       - {ev.description}")

    print(
        "\nNote: the baseline ranks the exception site (divide) first, but the "
        "root cause is average() - see the evidence above."
    )


if __name__ == "__main__":
    main()
