"""Single-line source mutations for the synthetic fault corpus (v0.2).

Pure and offline: given the text of a Python module, enumerate candidate
single-line mutations of module-level functions. Nothing here executes code.

Every mutation is expressed as an exact textual recipe ``(old, new)`` that is
compatible with ``evaluation.product_pool.verify_failing_tree``: ``old`` is a
complete source line that occurs exactly once inside the function text, and
``new`` is its replacement (one line for in-place edits, two lines for
inserted statements). Either shape moves a function's recorded line span by at
most one line, which ``evaluation.private_adapter`` accepts.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "FunctionInfo",
    "Mutation",
    "apply_mutation",
    "enumerate_functions",
    "enumerate_mutations",
]

#: Mutation category names; the dataset records the category as ``injection_method``.
KIND_COMPARE = "comparison-operator flip"
KIND_BOOLEAN = "boolean-operator flip"
KIND_ARITHMETIC = "arithmetic-operator flip"
KIND_CONSTANT = "constant off-by-one"
KIND_BOOL_CONST = "boolean constant flip"
KIND_NEGATION = "negation removal"
KIND_RETURN_VALUE = "wrong-value return"
KIND_EARLY_RETURN = "early-return None"
KIND_RAISE = "raised exception"

ALL_KINDS: tuple[str, ...] = (
    KIND_COMPARE,
    KIND_BOOLEAN,
    KIND_ARITHMETIC,
    KIND_CONSTANT,
    KIND_BOOL_CONST,
    KIND_NEGATION,
    KIND_RETURN_VALUE,
    KIND_EARLY_RETURN,
    KIND_RAISE,
)

_COMPARE_FLIPS: dict[type[ast.cmpop], tuple[str, str]] = {
    ast.Eq: ("==", "!="),
    ast.NotEq: ("!=", "=="),
    ast.Lt: ("<", "<="),
    ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="),
    ast.GtE: (">=", ">"),
}
_ARITH_FLIPS: dict[type[ast.operator], tuple[str, str]] = {
    ast.Add: ("+", "-"),
    ast.Sub: ("-", "+"),
    ast.Mult: ("*", "+"),
}


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    start_line: int  # 1-based, the ``def`` line (decorators excluded)
    end_line: int
    first_stmt_line: int  # first non-docstring statement


@dataclass(frozen=True)
class Mutation:
    """One exact textual edit inside a function."""

    function: str
    kind: str
    line_no: int  # 1-based line of ``old`` in the clean file
    old: str  # complete clean line, no trailing newline
    new: str  # replacement text; may contain one newline (inserted statement)
    in_place: bool  # True: one line replaced; False: one statement inserted

    @property
    def inserted_lines(self) -> int:
        return 0 if self.in_place else 1


def _all_function_names(tree: ast.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts


def _docstring_end(node: ast.FunctionDef) -> int:
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno or first.lineno
    return 0


def enumerate_functions(source: str, *, min_body_lines: int = 3) -> list[FunctionInfo]:
    """Module-level, synchronous, uniquely named functions suitable as fault sites.

    Uniqueness (no other function or method of the same name in the file) keeps
    function-level labels unambiguous for the product indexer.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    counts = _all_function_names(tree)
    result: list[FunctionInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or counts.get(node.name) != 1:
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if node.end_lineno is None or node.end_lineno - node.lineno + 1 < min_body_lines:
            continue
        doc_end = _docstring_end(node)
        body = [stmt for stmt in node.body if stmt.lineno > doc_end]
        if not body:
            continue
        result.append(FunctionInfo(node.name, node.lineno, node.end_lineno, body[0].lineno))
    return result


def _line_replace(
    lines: list[str], node: ast.AST, start_col: int, end_col: int, text: str
) -> str | None:
    lineno = getattr(node, "lineno", 0)
    if not lineno or getattr(node, "end_lineno", lineno) != lineno:
        return None
    line = lines[lineno - 1]
    # ast col offsets count UTF-8 bytes
    raw = line.encode("utf-8")
    return (raw[:start_col] + text.encode("utf-8") + raw[end_col:]).decode("utf-8")


def _operator_span(
    lines: list[str], lineno: int, left_end: int, right_start: int, token: str
) -> tuple[int, int] | None:
    raw = lines[lineno - 1].encode("utf-8")
    segment = raw[left_end:right_start]
    index = segment.find(token.encode())
    if index < 0 or segment.count(token.encode()) != 1:
        return None
    return left_end + index, left_end + index + len(token)


def _single_line(node: ast.AST) -> bool:
    return getattr(node, "end_lineno", None) == getattr(node, "lineno", -1)


def _candidates_in_function(
    lines: list[str], func: ast.FunctionDef, info: FunctionInfo
) -> Iterator[tuple[int, str, str]]:
    """Yield ``(line_no, kind, new_line)`` for in-place edits (may repeat per line)."""
    doc_end = _docstring_end(func)
    for node in ast.walk(func):
        lineno = getattr(node, "lineno", 0)
        if lineno <= doc_end or lineno < info.start_line + 1:
            continue
        if isinstance(node, ast.Compare) and _single_line(node) and len(node.ops) == 1:
            op = node.ops[0]
            pair = _COMPARE_FLIPS.get(type(op))
            left_end, right = node.left.end_col_offset, node.comparators[0]
            if pair and left_end is not None and _single_line(node.left) and _single_line(right):
                span = _operator_span(lines, lineno, left_end, right.col_offset, pair[0])
                if span:
                    new = _line_replace(lines, node, span[0], span[1], pair[1])
                    if new:
                        yield lineno, KIND_COMPARE, new
            elif isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) and left_end is not None:
                token, repl = {
                    ast.Is: ("is", "is not"),
                    ast.IsNot: ("is not", "is"),
                    ast.In: ("in", "not in"),
                    ast.NotIn: ("not in", "in"),
                }[type(op)]
                span = _operator_span(lines, lineno, left_end, right.col_offset, token)
                if span:
                    new = _line_replace(lines, node, span[0], span[1], repl)
                    if new:
                        yield lineno, KIND_COMPARE, new
        elif isinstance(node, ast.BoolOp) and _single_line(node) and len(node.values) == 2:
            token, repl = ("and", "or") if isinstance(node.op, ast.And) else ("or", "and")
            first, second = node.values
            if first.end_col_offset is not None:
                span = _operator_span(lines, lineno, first.end_col_offset, second.col_offset, token)
                if span:
                    new = _line_replace(lines, node, span[0], span[1], repl)
                    if new:
                        yield lineno, KIND_BOOLEAN, new
        elif isinstance(node, ast.BinOp) and _single_line(node):
            pair = _ARITH_FLIPS.get(type(node.op))
            left_end = node.left.end_col_offset
            if pair and left_end is not None and _single_line(node.left):
                span = _operator_span(lines, lineno, left_end, node.right.col_offset, pair[0])
                if span:
                    new = _line_replace(lines, node, span[0], span[1], pair[1])
                    if new:
                        yield lineno, KIND_ARITHMETIC, new
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and _single_line(node):
            end = node.operand.col_offset
            new = _line_replace(lines, node, node.col_offset, end, "")
            if new:
                yield lineno, KIND_NEGATION, new
        elif isinstance(node, ast.Constant) and _single_line(node):
            value = node.value
            if isinstance(value, bool):
                new = _line_replace(
                    lines, node, node.col_offset, node.end_col_offset or 0, str(not value)
                )
                if new:
                    yield lineno, KIND_BOOL_CONST, new
            elif isinstance(value, int) and 0 <= value < 1000:
                new = _line_replace(
                    lines, node, node.col_offset, node.end_col_offset or 0, str(value + 1)
                )
                if new:
                    yield lineno, KIND_CONSTANT, new
        elif isinstance(node, ast.Return) and node.value is not None and _single_line(node):
            end = node.end_col_offset or 0
            new = _line_replace(lines, node.value, node.value.col_offset, end, "None")
            if new and ast.unparse(node.value) != "None":
                yield lineno, KIND_RETURN_VALUE, new


def _inserted_statements(indent: str) -> Iterator[tuple[str, str]]:
    yield KIND_EARLY_RETURN, f"{indent}return None"
    yield KIND_RAISE, f"{indent}raise RuntimeError('unexpected state')"


def enumerate_mutations(source: str, info: FunctionInfo) -> list[Mutation]:
    """All valid, deduplicated, exactly-recorded single edits of ``info``.

    A mutation is kept only if the result still parses, differs from the
    original, and ``old`` appears exactly once in the function's text (the
    exactness condition required by the failing-tree verifier).
    """
    tree = ast.parse(source)
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == info.name and n.lineno == info.start_line
    )
    lines = source.splitlines()
    function_text = "\n".join(lines[info.start_line - 1 : info.end_line])
    seen: set[tuple[int, str]] = set()
    result: list[Mutation] = []

    def accept(mutation: Mutation) -> None:
        if function_text.count(mutation.old) != 1:
            return
        key = (mutation.line_no, mutation.new)
        if key in seen:
            return
        edited = list(lines)
        replacement = mutation.new.split("\n")
        edited[mutation.line_no - 1 : mutation.line_no] = replacement
        try:
            ast.parse("\n".join(edited) + "\n")
        except SyntaxError:
            return
        seen.add(key)
        result.append(mutation)

    for line_no, kind, new_line in _candidates_in_function(lines, func, info):
        old = lines[line_no - 1]
        if new_line != old:
            accept(Mutation(info.name, kind, line_no, old, new_line, True))

    first = lines[info.first_stmt_line - 1]
    indent = first[: len(first) - len(first.lstrip())]
    # Only statement-starting lines are safe insertion points.
    for kind, statement in _inserted_statements(indent):
        accept(
            Mutation(info.name, kind, info.first_stmt_line, first, f"{statement}\n{first}", False)
        )
    return result


def apply_mutation(source: str, mutation: Mutation) -> str:
    """Return ``source`` with ``mutation`` applied (single-edit, trailing newline kept)."""
    lines = source.split("\n")
    if lines[mutation.line_no - 1] != mutation.old:
        raise ValueError("mutation does not match the source line")
    lines[mutation.line_no - 1 : mutation.line_no] = mutation.new.split("\n")
    return "\n".join(lines)
