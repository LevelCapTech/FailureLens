"""Shared pytest fixtures for the FailureLens unit tests.

The fixtures build a tiny, self-contained repository on disk together with a
matching traceback. The example is deliberately constructed so that the frame
where the exception *surfaces* (``divide``) is different from the frame that is
the *root cause* (``average`` divides by a zero-length count), which is exactly
the situation FailureLens is designed to illuminate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Line numbers below are relied upon by the traceback, so keep them in sync.
CORE_PY = '''\
"""Core math helpers."""


def divide(numerator, denominator):
    result = numerator / denominator
    return result


def average(values):
    total = sum(values)
    count = len(values)
    return divide(total, count)
'''

APP_PY = """\
from pkg.core import average


def main():
    data = []
    return average(data)


if __name__ == "__main__":
    main()
"""

SAMPLE_TRACEBACK = """\
Traceback (most recent call last):
  File "app.py", line 10, in <module>
    main()
  File "app.py", line 6, in main
    return average(data)
  File "pkg/core.py", line 12, in average
    return divide(total, count)
  File "pkg/core.py", line 5, in divide
    result = numerator / denominator
ZeroDivisionError: division by zero
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal repository whose structure matches ``SAMPLE_TRACEBACK``."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text(CORE_PY, encoding="utf-8")
    (tmp_path / "app.py").write_text(APP_PY, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_traceback() -> str:
    return SAMPLE_TRACEBACK
