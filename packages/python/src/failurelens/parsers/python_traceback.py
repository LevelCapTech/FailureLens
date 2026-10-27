"""Parser for CPython stack traces (tracebacks).

This parser recognizes the standard ``Traceback (most recent call last):``
format, including chained exceptions ("During handling of the above
exception..." / "The above exception was the direct cause...") and the special
``SyntaxError`` layout. For chained traces, frames from every block are kept in
document order and the *final* propagated exception is reported as the error.

Only facts present in the text are extracted; nothing is inferred. When a field
is absent from the log it is left as ``None``.
"""

from __future__ import annotations

import re

from failurelens.parsers.base import LogParser
from failurelens.schemas import ParsedLog, StackFrame

_TRACEBACK_HEADER = "Traceback (most recent call last):"

_CHAINING_NOTES = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)

# e.g.  File "/path/to/file.py", line 42, in function_name
_FRAME_RE = re.compile(r'^\s*File "(?P<file>.+?)", line (?P<line>\d+)(?:, in (?P<func>.*))?\s*$')

# A non-indented exception line such as `pkg.mod.ValueError: message` or `KeyError`.
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::[ \t]?(?P<msg>.*))?$")


class PythonTracebackParser:
    """Extracts structured information from CPython tracebacks."""

    name = "python-traceback"
    language = "python"

    def matches(self, text: str) -> bool:
        if _TRACEBACK_HEADER in text:
            return True
        return any(_FRAME_RE.match(line) for line in text.splitlines())

    def parse(self, text: str) -> ParsedLog | None:
        lines = text.splitlines()
        frames: list[StackFrame] = []
        last_frame_idx = -1

        for idx, line in enumerate(lines):
            match = _FRAME_RE.match(line)
            if match is None:
                continue
            func = match.group("func")
            frames.append(
                StackFrame(
                    order=len(frames),
                    file=match.group("file"),
                    line=int(match.group("line")),
                    function=func.strip() if func else None,
                )
            )
            last_frame_idx = idx

        error_type, error_message = self._extract_exception(lines, last_frame_idx)

        if not frames and error_type is None:
            return None

        return ParsedLog(
            parser=self.name,
            language=self.language,
            error_type=error_type,
            error_message=error_message,
            frames=frames,
        )

    def _extract_exception(
        self, lines: list[str], last_frame_idx: int
    ) -> tuple[str | None, str | None]:
        """Return the final exception ``(type, message)`` following the last frame."""
        start = last_frame_idx + 1 if last_frame_idx >= 0 else 0
        for raw in lines[start:]:
            if not raw.strip():
                continue
            if raw[:1].isspace():
                # Indented lines are source context or the SyntaxError caret.
                continue
            stripped = raw.strip()
            if stripped == _TRACEBACK_HEADER or stripped in _CHAINING_NOTES:
                continue
            match = _EXC_RE.match(stripped)
            if match is None:
                continue
            message = match.group("msg")
            return match.group("type"), (message if message else None)
        return None, None


# Static check: the concrete class satisfies the LogParser protocol.
_parser: LogParser = PythonTracebackParser()
