"""Source mutations for the synthetic fault corpus (v0.2 base, v0.3 hard kinds).

Pure and offline: given the text of a Python module, enumerate candidate
mutations of module-level functions. Nothing here executes code.

Every mutation is expressed as an exact textual recipe ``(old, new)`` that is
compatible with ``evaluation.product_pool.verify_failing_tree``: ``old`` is a
run of complete source lines (one line for the v0.2 kinds, several consecutive
lines for some v0.3 hard kinds) that occurs exactly once inside the function
text, and ``new`` is its replacement. The net line change is always 0 (in
place) or +1 (inserted statement), which moves a function's recorded line span
by at most one line, as ``evaluation.private_adapter`` requires.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = [
    "ALL_KINDS",
    "BASE_KINDS",
    "HARD_KINDS",
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

#: v0.3 hard kinds. ``KIND_CALLEE`` is not a text shape: the builder assigns it to
#: a case whose root-cause function does not appear in the failure's stack trace.
KIND_MULTI = "multi-line edit"
KIND_SWAP_ARGS = "swapped call arguments"
KIND_DROP_ARG = "dropped call argument"
KIND_BOUND = "loop or slice bound off-by-one"
KIND_DEFAULT = "changed default value"
KIND_GUARD = "neutralised guard clause"
KIND_CALLEE = "callee change surfacing elsewhere"

BASE_KINDS: tuple[str, ...] = (
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
HARD_KINDS: frozenset[str] = frozenset(
    {
        KIND_MULTI,
        KIND_SWAP_ARGS,
        KIND_DROP_ARG,
        KIND_BOUND,
        KIND_DEFAULT,
        KIND_GUARD,
        KIND_CALLEE,
    }
)
ALL_KINDS: tuple[str, ...] = (*BASE_KINDS, *sorted(HARD_KINDS))

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
    line_no: int  # 1-based first line of ``old`` in the clean file
    old: str  # complete clean line(s) joined by newlines, no trailing newline
    new: str  # replacement text
    in_place: bool  # True: net line count unchanged; False: one statement inserted

    @property
    def old_line_count(self) -> int:
        return self.old.count("\n") + 1

    @property
    def inserted_lines(self) -> int:
        return self.new.count("\n") - self.old.count("\n")


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


def _splice(
    node: ast.AST, lines: list[str], replacements: list[tuple[int, int, str]]
) -> str | None:
    """Replace byte ranges on the (single) line of ``node``; ranges must not overlap."""
    lineno = getattr(node, "lineno", 0)
    if not lineno:
        return None
    raw = lines[lineno - 1].encode("utf-8")
    for begin, end, text in sorted(replacements, reverse=True):
        raw = raw[:begin] + text.encode("utf-8") + raw[end:]
    return raw.decode("utf-8")


def _parenthesized(node: ast.expr, text: str) -> str:
    simple = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Constant)
    return text if isinstance(node, simple) else f"({text})"


def _expr_text(lines: list[str], node: ast.expr) -> str:
    raw = lines[node.lineno - 1].encode("utf-8")
    return raw[node.col_offset : node.end_col_offset].decode("utf-8")


def _call_edits(lines: list[str], node: ast.Call, lineno: int) -> Iterator[tuple[int, str, str]]:
    args = node.args
    plain = not any(isinstance(a, ast.Starred) for a in args) and all(
        _single_line(a) and a.end_col_offset is not None for a in args
    )
    if not plain:
        return
    if len(args) >= 2:
        first, second = args[0], args[1]
        text_a, text_b = _expr_text(lines, first), _expr_text(lines, second)
        if text_a != text_b and first.end_col_offset and second.end_col_offset:
            new = _splice(
                node,
                lines,
                [
                    (first.col_offset, first.end_col_offset, text_b),
                    (second.col_offset, second.end_col_offset, text_a),
                ],
            )
            if new:
                yield lineno, KIND_SWAP_ARGS, new
        if not node.keywords:
            prev, last = args[-2], args[-1]
            if prev.end_col_offset is not None and last.end_col_offset is not None:
                new = _splice(node, lines, [(prev.end_col_offset, last.end_col_offset, "")])
                if new:
                    yield lineno, KIND_DROP_ARG, new
    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and args
        and not isinstance(args[-1], ast.Constant)
        and args[-1].end_col_offset is not None
    ):
        last = args[-1]
        text = _parenthesized(last, _expr_text(lines, last))
        new = _splice(node, lines, [(last.col_offset, last.end_col_offset or 0, f"{text} - 1")])
        if new:
            yield lineno, KIND_BOUND, new


def _hard_single_line_edits(
    lines: list[str], func: ast.FunctionDef, info: FunctionInfo
) -> Iterator[tuple[int, str, str]]:
    """Yield ``(line_no, kind, new_line)`` for v0.3 in-place, single-line hard edits."""
    doc_end = _docstring_end(func)
    for node in ast.walk(func):
        lineno = getattr(node, "lineno", 0)
        if lineno <= max(doc_end, info.start_line):
            continue
        if isinstance(node, ast.Call) and _single_line(node):
            yield from _call_edits(lines, node, lineno)
        elif isinstance(node, ast.Slice) and _single_line(node):
            for bound, suffix in ((node.upper, " - 1"), (node.lower, " + 1")):
                if (
                    bound is not None
                    and not isinstance(bound, ast.Constant)
                    and _single_line(bound)
                    and bound.end_col_offset is not None
                ):
                    text = _parenthesized(bound, _expr_text(lines, bound))
                    new = _splice(
                        bound, lines, [(bound.col_offset, bound.end_col_offset, f"{text}{suffix}")]
                    )
                    if new:
                        yield bound.lineno, KIND_BOUND, new
    # Default values live in the ``def`` header, which the walk above does not cover.
    arguments = func.args
    defaults = [*arguments.defaults, *[d for d in arguments.kw_defaults if d is not None]]
    for default in defaults:
        if not (
            isinstance(default, ast.Constant)
            and _single_line(default)
            and default.end_col_offset is not None
            and info.start_line <= default.lineno < info.first_stmt_line
        ):
            continue
        value = default.value
        if isinstance(value, bool):
            replacement = str(not value)
        elif isinstance(value, int) and 0 <= value < 1000:
            replacement = str(value + 1)
        elif isinstance(value, str) and value:
            replacement = '""'
        else:
            continue
        new = _splice(default, lines, [(default.col_offset, default.end_col_offset, replacement)])
        if new:
            yield default.lineno, KIND_DEFAULT, new


def _guard_edits(lines: list[str], func: ast.FunctionDef, info: FunctionInfo) -> Iterator[Mutation]:
    """``if cond: <raise/return/continue/break>`` with the action replaced by ``pass``."""
    for node in ast.walk(func):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        action = node.body[0]
        if not isinstance(action, (ast.Raise, ast.Return, ast.Continue, ast.Break)):
            continue
        if (
            action.lineno != node.lineno + 1
            or not _single_line(action)
            or node.test.end_lineno != node.lineno
            or node.lineno <= info.start_line
        ):
            continue
        header, body = lines[node.lineno - 1], lines[action.lineno - 1]
        if not header.rstrip().endswith(":"):
            continue
        indent = body[: len(body) - len(body.lstrip())]
        yield Mutation(
            info.name,
            KIND_GUARD,
            node.lineno,
            f"{header}\n{body}",
            f"{header}\n{indent}pass",
            True,
        )


def _combined_edits(
    lines: list[str],
    edits: list[tuple[int, str, str]],
    info: FunctionInfo,
    *,
    max_gap: int = 6,
    limit: int = 3,
) -> Iterator[Mutation]:
    """Two independent in-place edits on nearby lines, recorded as one multi-line recipe."""
    by_line: dict[int, str] = {}
    for line_no, _kind, new_line in sorted(edits):
        by_line.setdefault(line_no, new_line)
    ordered = sorted(by_line)
    pairs = [
        (a, b)
        for index, a in enumerate(ordered)
        for b in ordered[index + 1 :]
        if 0 < b - a <= max_gap
    ]
    pairs.sort(key=lambda pair: (pair[1] - pair[0], pair[0]))
    for a, b in pairs[:limit]:
        old_lines = lines[a - 1 : b]
        new_lines = list(old_lines)
        new_lines[0] = by_line[a]
        new_lines[-1] = by_line[b]
        yield Mutation(info.name, KIND_MULTI, a, "\n".join(old_lines), "\n".join(new_lines), True)


def enumerate_mutations(source: str, info: FunctionInfo, *, hard: bool = False) -> list[Mutation]:
    """All valid, deduplicated, exactly-recorded edits of ``info``.

    A mutation is kept only if the result still parses, differs from the
    original, and ``old`` appears exactly once in the function's text (the
    exactness condition required by the failing-tree verifier). With
    ``hard=True`` the v0.3 hard text shapes (see :data:`HARD_KINDS`) are
    enumerated as well.
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
        if mutation.new == mutation.old or function_text.count(mutation.old) != 1:
            return
        if mutation.inserted_lines not in {0, 1}:
            return
        key = (mutation.line_no, mutation.new)
        if key in seen:
            return
        edited = list(lines)
        edited[mutation.line_no - 1 : mutation.line_no - 1 + mutation.old_line_count] = (
            mutation.new.split("\n")
        )
        try:
            ast.parse("\n".join(edited) + "\n")
        except SyntaxError:
            return
        seen.add(key)
        result.append(mutation)

    base_edits = list(_candidates_in_function(lines, func, info))
    for line_no, kind, new_line in base_edits:
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
    if hard:
        for line_no, kind, new_line in _hard_single_line_edits(lines, func, info):
            accept(Mutation(info.name, kind, line_no, lines[line_no - 1], new_line, True))
        for mutation in _guard_edits(lines, func, info):
            accept(mutation)
        changed = [(n, k, t) for n, k, t in base_edits if t != lines[n - 1]]
        for mutation in _combined_edits(lines, changed, info):
            accept(mutation)
    return result


def apply_mutation(source: str, mutation: Mutation) -> str:
    """Return ``source`` with ``mutation`` applied (single-edit, trailing newline kept)."""
    lines = source.split("\n")
    begin = mutation.line_no - 1
    if "\n".join(lines[begin : begin + mutation.old_line_count]) != mutation.old:
        raise ValueError("mutation does not match the source line")
    lines[begin : begin + mutation.old_line_count] = mutation.new.split("\n")
    return "\n".join(lines)
