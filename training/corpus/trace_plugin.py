"""Pytest plugin (loaded with ``-p trace_plugin``) recording outcomes and calls.

Standalone on purpose: it runs inside the *target* project's pytest process and
therefore imports nothing from FailureLens. Output is a JSON file named by the
``FL_TRACE_OUT`` environment variable:

    {"outcomes": {nodeid: "passed"|"failed"|...},
     "calls": {nodeid: [[relative_file, function_name, first_line], ...]}}

Only functions defined under ``FL_TRACE_ROOT`` and outside ``FL_TRACE_SKIP``
(comma-separated relative path prefixes, normally the tests) are recorded.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_ROOT = os.path.realpath(os.environ.get("FL_TRACE_ROOT", os.getcwd())) + os.sep
_SKIP = tuple(p for p in os.environ.get("FL_TRACE_SKIP", "").split(",") if p)
_OUT = os.environ.get("FL_TRACE_OUT", "")
_TRACE = os.environ.get("FL_TRACE_CALLS", "1") == "1"

_outcomes: dict[str, str] = {}
_calls: dict[str, set[tuple[str, str, int]]] = {}
_state: dict[str, set[tuple[str, str, int]] | None] = {"current": None}
_cache: dict[str, str | None] = {}


def _relative(filename: str) -> str | None:
    if filename in _cache:
        return _cache[filename]
    rel: str | None = None
    real = os.path.realpath(filename) if filename.endswith(".py") else ""
    if real.startswith(_ROOT):
        candidate = real[len(_ROOT) :]
        if not (_SKIP and candidate.startswith(_SKIP)) and "site-packages" not in candidate:
            rel = candidate
    _cache[filename] = rel
    return rel


def _profile(frame: Any, event: str, arg: Any) -> None:
    current = _state["current"]
    if event != "call" or current is None:
        return
    rel = _relative(frame.f_code.co_filename)
    if rel is not None:
        current.add((rel, frame.f_code.co_name, frame.f_code.co_firstlineno))


def pytest_runtest_call(item: Any) -> None:
    if not _TRACE:
        return
    _state["current"] = _calls.setdefault(item.nodeid, set())
    sys.setprofile(_profile)


def pytest_runtest_teardown(item: Any) -> None:
    sys.setprofile(None)
    _state["current"] = None


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcomes[report.nodeid] = report.outcome


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    sys.setprofile(None)
    if not _OUT:
        return
    payload = {"outcomes": _outcomes, "calls": {k: sorted(v) for k, v in _calls.items()}}
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
