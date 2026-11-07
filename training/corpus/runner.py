"""Run a target project's pytest and sanitize the resulting failure log.

Executes third-party test code. It is intended only for the pinned,
license-reviewed repositories in :mod:`training.corpus.repos`, in a private
working copy and a dedicated virtualenv, never in CI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).resolve().parent
_FAILURES_HEADER = re.compile(r"^=+ FAILURES =+$", re.MULTILINE)
_SECTION_END = re.compile(r"^=+ [^=\n]*(short test summary|warnings summary|ERRORS)", re.MULTILINE)
_SUMMARY_LINE = re.compile(r"^\d+ (failed|passed|error)[^\n]*$", re.MULTILINE)


@dataclass
class PytestRun:
    returncode: int
    stdout: str
    duration_ms: float
    outcomes: dict[str, str] = field(default_factory=dict)
    calls: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    timed_out: bool = False


def run_pytest(
    python: str | Path,
    workdir: Path,
    args: list[str],
    *,
    pythonpath: str = ".",
    skip_prefixes: tuple[str, ...] = (),
    trace_calls: bool = True,
    timeout: float = 600.0,
) -> PytestRun:
    """Run ``python -m pytest`` in ``workdir`` with the corpus trace plugin."""
    with tempfile.TemporaryDirectory(prefix="fl_corpus_trace_") as tmp:
        out = Path(tmp) / "trace.json"
        paths = [str((workdir / pythonpath).resolve()), str(_PLUGIN_DIR)]
        env = {
            **{k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "LC_ALL"}},
            "PYTHONPATH": os.pathsep.join(paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "FL_TRACE_ROOT": str(workdir.resolve()),
            "FL_TRACE_SKIP": ",".join(skip_prefixes),
            "FL_TRACE_OUT": str(out),
            "FL_TRACE_CALLS": "1" if trace_calls else "0",
            "NO_COLOR": "1",
        }
        command = [
            str(python),
            "-m",
            "pytest",
            "-p",
            "trace_plugin",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:warnings",
            "-o",
            "addopts=",
            "--tb=native",
            "-q",
            "--no-header",
            *args,
        ]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            code, stdout, timed_out = completed.returncode, completed.stdout, False
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")
            code, stdout, timed_out = -9, text, True
        duration = round((time.monotonic() - start) * 1000, 2)
        payload: dict[str, Any] = {}
        if out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
    calls = {k: [(a, b, c) for a, b, c in v] for k, v in payload.get("calls", {}).items()}
    return PytestRun(code, stdout, duration, dict(payload.get("outcomes", {})), calls, timed_out)


def extract_failure_log(stdout: str, workdir: Path) -> tuple[str, str] | None:
    """Return ``(error_log, stack_trace)`` for a single failing test, paths made relative.

    ``error_log`` is the FAILURES section (test header and native traceback);
    ``stack_trace`` starts at the first ``Traceback`` line. Absolute private
    paths are removed. Returns ``None`` if the output has no
    Python traceback (for example a collection error or a timeout).
    """
    header = _FAILURES_HEADER.search(stdout)
    if header is None:
        return None
    body = stdout[header.end() :]
    end = _SECTION_END.search(body)
    if end is not None:
        body = body[: end.start()]
    else:
        summary = _SUMMARY_LINE.search(body)
        if summary is not None:
            body = body[: summary.start()]
    body = body.strip("\n")
    for prefix in {str(workdir.resolve()) + "/", str(workdir) + "/"}:
        body = body.replace(prefix, "")
    marker = "Traceback (most recent call last):"
    index = body.find(marker)
    if index < 0:
        return None
    return body + "\n", body[index:] + "\n"


def terminal_exception(stack_trace: str) -> str:
    """Last non-empty line of a traceback (the exception type and message)."""
    lines = [line for line in stack_trace.splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""
