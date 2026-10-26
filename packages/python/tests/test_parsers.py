"""Tests for the Python traceback parser and parser registry."""

from __future__ import annotations

from failurelens.parsers.base import ParserRegistry, default_registry
from failurelens.parsers.python_traceback import PythonTracebackParser


def test_parses_basic_traceback(sample_traceback: str) -> None:
    parsed = PythonTracebackParser().parse(sample_traceback)
    assert parsed is not None
    assert parsed.parser == "python-traceback"
    assert parsed.language == "python"
    assert parsed.error_type == "ZeroDivisionError"
    assert parsed.error_message == "division by zero"
    assert [f.order for f in parsed.frames] == [0, 1, 2, 3]
    # Frames are kept in document order; the last/deepest is the exception site.
    deepest = parsed.frames[-1]
    assert deepest.file == "pkg/core.py"
    assert deepest.line == 5
    assert deepest.function == "divide"


def test_matches_requires_traceback_shape(sample_traceback: str) -> None:
    parser = PythonTracebackParser()
    assert parser.matches(sample_traceback) is True
    assert parser.matches("just a normal log line\nnothing to see here") is False


def test_parses_chained_exception() -> None:
    text = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 2, in <module>\n'
        '    raise ValueError("inner")\n'
        "ValueError: inner\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "b.py", line 5, in <module>\n'
        '    raise RuntimeError("outer")\n'
        "RuntimeError: outer\n"
    )
    parsed = PythonTracebackParser().parse(text)
    assert parsed is not None
    # The final propagated exception is reported.
    assert parsed.error_type == "RuntimeError"
    assert parsed.error_message == "outer"
    # Frames from every block are preserved in order.
    assert [f.file for f in parsed.frames] == ["a.py", "b.py"]


def test_parses_syntax_error_layout() -> None:
    text = (
        "Traceback (most recent call last):\n"
        '  File "bad.py", line 1\n'
        "    def f(:\n"
        "          ^\n"
        "SyntaxError: invalid syntax\n"
    )
    parsed = PythonTracebackParser().parse(text)
    assert parsed is not None
    assert parsed.error_type == "SyntaxError"
    assert parsed.error_message == "invalid syntax"
    assert len(parsed.frames) == 1
    assert parsed.frames[0].file == "bad.py"
    assert parsed.frames[0].line == 1
    assert parsed.frames[0].function is None


def test_registry_returns_none_for_unrecognized_text() -> None:
    registry = default_registry()
    assert registry.parse("this is not a traceback at all") is None


def test_registry_parse_uses_first_matching_parser(sample_traceback: str) -> None:
    registry = ParserRegistry()
    registry.register(PythonTracebackParser())
    parsed = registry.parse(sample_traceback)
    assert parsed is not None
    assert parsed.error_type == "ZeroDivisionError"
