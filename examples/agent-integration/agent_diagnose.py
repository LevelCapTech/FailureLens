#!/usr/bin/env python
"""Minimal agent-style driver: shell out to the FailureLens CLI and parse JSON.

This mimics how an AI coding agent or CI step would invoke FailureLens: run the
CLI, read JSON from stdout, and act on the top candidates. It uses only the
standard library.

    python examples/agent-integration/agent_diagnose.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REPO = REPO_ROOT / "examples" / "fixtures" / "sample_project"
SAMPLE_LOG = SAMPLE_REPO / "failure.log"


def diagnose(repo: Path, log: Path, top_k: int = 3) -> dict:
    """Run `failurelens diagnose ... --format json --quiet` and return parsed JSON."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "failurelens.cli",
            "diagnose",
            "--repo",
            str(repo),
            "--log",
            str(log),
            "--top",
            str(top_k),
            "--format",
            "json",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit code 0 => ok; 3 => inconclusive (still valid JSON). Anything else is an error.
    if proc.returncode not in (0, 3):
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return json.loads(proc.stdout)


def main() -> None:
    result = diagnose(SAMPLE_REPO, SAMPLE_LOG)
    print(f"status={result['status']} method={result['analysis_method']}")
    for c in result["candidates"]:
        print(
            f"  #{c['rank']} {c['file']}::{c['function']} "
            f"(lines {c['start_line']}-{c['end_line']}, score={c['score']})"
        )
    # An agent would now open these files, read the evidence, and attempt a fix.


if __name__ == "__main__":
    main()
