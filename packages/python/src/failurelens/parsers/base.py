"""Pluggable log parsers.

A :class:`LogParser` turns raw failure text into a structured
:class:`~failurelens.schemas.ParsedLog`. Parsers are intentionally small and
format-specific; new languages/formats are added by implementing this protocol
and registering an instance. The baseline ships with a Python traceback parser.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from failurelens.schemas import ParsedLog


@runtime_checkable
class LogParser(Protocol):
    """Interface implemented by every log parser."""

    #: Stable, unique parser identifier (recorded in :attr:`ParsedLog.parser`).
    name: str
    #: Language this parser targets (e.g. ``"python"``).
    language: str

    def matches(self, text: str) -> bool:
        """Return ``True`` if this parser recognizes ``text``."""
        ...

    def parse(self, text: str) -> ParsedLog | None:
        """Parse ``text`` into a :class:`ParsedLog`, or ``None`` if unrecognized."""
        ...


class ParserRegistry:
    """An ordered collection of parsers tried in registration order."""

    def __init__(self) -> None:
        self._parsers: list[LogParser] = []

    def register(self, parser: LogParser) -> None:
        self._parsers.append(parser)

    def parsers(self) -> list[LogParser]:
        return list(self._parsers)

    def parse(self, text: str) -> ParsedLog | None:
        """Return the first successful parse among registered parsers."""
        for parser in self._parsers:
            if parser.matches(text):
                parsed = parser.parse(text)
                if parsed is not None:
                    return parsed
        return None


def default_registry() -> ParserRegistry:
    """Build a registry populated with the built-in parsers."""
    # Imported here to avoid a circular import at module load time.
    from failurelens.parsers.python_traceback import PythonTracebackParser

    registry = ParserRegistry()
    registry.register(PythonTracebackParser())
    return registry


__all__ = ["LogParser", "ParserRegistry", "default_registry"]
