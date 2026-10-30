"""Stage 2 dataset loader: extract actual source code from mutated failing snapshots.

Reads the private train-issue17-v2 dataset and reconstructs training examples
whose positive/negative texts are actual function source code extracted from
ISOLATED COPIES of the source files, with the injection mutation applied for
the positive candidate.

Reconstruction method (aggregate, no individual labels here):
- Positive text: the injection_location function extracted AFTER applying the
  injection mutation to an isolated per-record copy of the source file.
  The mutation statement is reconstructed deterministically from the record's
  ``injection_method`` field and, for raised-exception mutations, from the
  ``error_log`` field (the exact raise statement is present in the traceback).
- Negative text: the negative candidate function extracted from the CLEAN
  (pre-mutation) source file.
- Reconstruction uses ``ast.parse`` to locate function boundaries, so line
  numbers are authoritative and the full function body is included.
- The isolated copy is a tmpdir; the original source file is never modified.

Generator signals stripped from all three features (query, positive, negative):
- Marker comment ``# issue17 injected fault``.
- Private absolute paths inside traceback ``File "..."`` lines (replaced with
  the source-relative path only).
- Generator-specific injected exception messages such as ``"injected X fault"``
  (replaced with ``""`` to preserve the exception type/category).

Constraints honoured:
- ``label_rationale``, ground-truth markers, and invented identifiers are NOT
  used as features.  Only ``injection_method``, ``error_log``, and the source
  code at the candidate locations are used.
- ``psf/requests``, ``pallets/werkzeug``, ``pypa/pip`` are excluded.
- Data digest is verified before loading.

Split policy: repo-level 6 train / 6 val --- all ``pallets/click`` records are
assigned to training, all ``python-attrs/attrs`` records to validation.
This avoids within-repository leakage between the train and val sets.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIVATE_DATA_ROOT: Path = Path.home() / ".local/share/failurelens/train-issue17-v2"
EXPECTED_DATASET_SHA256: str = "74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1"

EXCLUDED_REPOS: frozenset[str] = frozenset({"psf/requests", "pallets/werkzeug", "pypa/pip"})

_REPO_SOURCE_KEYS: dict[str, str] = {
    "pallets/click": "click",
    "python-attrs/attrs": "attrs",
}

_TRAIN_REPO: str = "pallets/click"
_VAL_REPO: str = "python-attrs/attrs"

# Marker appended by the private dataset generator to injected lines and
# preserved verbatim in error_log tracebacks.  It must NOT appear in any
# training feature passed to the model.
_GENERATOR_MARKER: str = "# issue17 injected fault"

# Private absolute path prefix inside traceback File "..." entries.
# Captures the source-relative portion after /sources/{repo_key}/.
_PRIVATE_SOURCE_PATH_RE: re.Pattern[str] = re.compile(
    r'(?<=")/[^"]+/sources/(?:click|attrs)/([^"]+)(?=")'
)

# Generator-specific injected exception message in a quoted-string context,
# e.g. RuntimeError("injected short-help fault").  Replaced with "" to
# preserve the exception type/category without the generator label.
_INJECTED_QUOTED_MSG_RE: re.Pattern[str] = re.compile(r'"injected [^"]*"')

# Generator-specific injected exception message in the final traceback summary
# line, e.g. "RuntimeError: injected short-help fault".  Replaced with the
# exception type alone to preserve the fault category.
_INJECTED_TRAILING_MSG_RE: re.Pattern[str] = re.compile(
    r"^(\s*\w[\w.]*(?:Error|Exception)\w*): injected .+$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Stage2Example:
    record_id: str
    query: str
    positive: str
    negative: str
    source_repository: str
    dataset_version: str
    split: str


def _verify_data_digest(records_path: Path) -> None:
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    if digest != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset digest mismatch --- expected {EXPECTED_DATASET_SHA256!r} "
            f"but got {digest!r}. Use the authorised private dataset only."
        )


def _verify_source_commits(records: list[dict[str, Any]], root: Path) -> None:
    """Verify each record's source_commit against the pinned source repo HEAD.

    Ensures the private source files match the exact commits recorded in the
    dataset, so reconstructed function texts are reproducible and tied to a
    known clean snapshot.
    """
    expected: dict[str, str] = {}
    for rec in records:
        repo: str = rec["source_repository"]
        commit: str = rec["source_commit"]
        if (
            repo in expected
            and not expected[repo].startswith(commit)
            and not commit.startswith(expected[repo])
        ):
            raise ValueError(
                f"Inconsistent source_commit for {repo!r}: {expected[repo]!r} vs {commit!r}"
            )
        expected[repo] = commit

    for repo, commit in expected.items():
        repo_key = _REPO_SOURCE_KEYS[repo]
        source_dir = root / "sources" / repo_key
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValueError(
                f"Cannot determine HEAD of source repo {repo!r} at {source_dir}: {exc}"
            ) from exc
        actual = result.stdout.strip()
        if not actual.startswith(commit) and not commit.startswith(actual[: len(commit)]):
            raise ValueError(
                f"Source repo {repo!r} HEAD is {actual!r} "
                f"but dataset records expect commit {commit!r}. "
                "Ensure the private source snapshot matches the dataset version."
            )


def _strip_generator_marker(text: str) -> str:
    """Remove ALL generator-specific signals from feature text.

    Strips:
    - The ``# issue17 injected fault`` inline comment from every line.
    - Private absolute paths inside traceback ``File "..."`` entries,
      keeping only the source-relative portion.
    - Generator-specific injected exception messages (``"injected X fault"``),
      replacing them with ``""`` to preserve exception type/category.
    - Generator-specific injected exception messages in traceback summary lines
      (e.g. ``RuntimeError: injected X fault``), keeping only the type name.
    """
    # Step 1: strip the inline marker comment
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line.rstrip("\n")
        if _GENERATOR_MARKER in content:
            idx = content.index(_GENERATOR_MARKER)
            content = content[:idx].rstrip()
        result.append(content + newline)
    text = "".join(result)

    # Step 2: strip private absolute paths, keep source-relative path only
    text = _PRIVATE_SOURCE_PATH_RE.sub(r"\1", text)

    # Step 3: strip generator-specific injected exception message strings
    text = _INJECTED_QUOTED_MSG_RE.sub('""', text)

    # Step 4: strip injected message from traceback summary lines
    text = _INJECTED_TRAILING_MSG_RE.sub(r"\1", text)

    return text


def _reconstruct_injection_statement(record: dict[str, Any]) -> str:
    """Reconstruct the injected statement from record fields (no MUT dict hardcoded).

    For ``wrong-value return`` and ``early-return None`` the statement is fully
    determined by ``injection_method``.  For ``raised exception`` the exact
    raise statement is recoverable from the ``error_log`` traceback (the
    injected line appears verbatim in the stack trace with the
    ``# issue17 injected fault`` comment).
    """
    method: str = record["injection_method"]
    if method == "wrong-value return":
        return "return False"
    if method == "early-return None":
        return "return None"
    if method == "raised exception":
        for raw_line in record["error_log"].splitlines():
            stripped = raw_line.strip()
            if _GENERATOR_MARKER in stripped and stripped.startswith("raise"):
                return _strip_generator_marker(stripped).strip()
        raise ValueError(
            f"Cannot reconstruct raise statement from error_log "
            f"for record {record.get('record_id')!r}"
        )
    raise ValueError(f"Unknown injection_method: {method!r}")


def _find_function_node(
    tree: ast.Module,
    func_name: str,
    start_line: int,
    end_line: int,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Find function by name AND line range; fail on ambiguity or location mismatch.

    Collects all top-level and nested function definitions named ``func_name``.
    If multiple definitions exist (e.g. duplicate names in different scopes),
    ``start_line`` is used to select the unique match.  Raises ``ValueError``
    if the match is ambiguous, absent, or at the wrong location.
    """
    all_named: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
    ]
    if not all_named:
        raise ValueError(f"Function {func_name!r} not found in parsed AST")

    matching = [n for n in all_named if n.lineno == start_line]
    if not matching:
        candidates = [(n.lineno, getattr(n, "end_lineno", "?")) for n in all_named]
        raise ValueError(
            f"Function {func_name!r} exists at {candidates} "
            f"but not at expected start_line={start_line}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"Ambiguous: {len(matching)} definitions of {func_name!r} "
            f"all start at line {start_line}"
        )
    node = matching[0]
    actual_end = getattr(node, "end_lineno", None)
    if actual_end is not None and actual_end != end_line:
        raise ValueError(
            f"Function {func_name!r} at line {start_line} ends at line {actual_end} "
            f"but expected end_line={end_line}"
        )
    return node


def _extract_mutated_function_text(
    source_file: Path,
    func_name: str,
    injection_statement: str,
    start_line: int,
    end_line: int,
) -> str:
    """Extract function source code with the injection mutation applied.

    Works on an isolated copy in a tmpdir so the original is never modified.
    After insertion the function spans [start_line, end_line + 1] (1-indexed
    inclusive), because one line is prepended inside the body.

    The generator marker is NOT appended to the injected line --- callers
    must pass the clean injection statement without any label comment.
    """
    with tempfile.TemporaryDirectory(prefix="fl_stage2_iso_") as tmp:
        isolated = Path(tmp) / source_file.name
        shutil.copy2(source_file, isolated)

        text = isolated.read_text(encoding="utf-8")
        tree = ast.parse(text)
        node = _find_function_node(tree, func_name, start_line, end_line)

        insert_at = node.body[0].lineno - 1  # 0-indexed, before first body stmt
        indent = " " * node.body[0].col_offset
        lines = text.splitlines(keepends=True)
        mutated = [
            *lines[:insert_at],
            indent + injection_statement + "\n",
            *lines[insert_at:],
        ]
        # def line stays at start_line; end shifts +1 after insertion
        return "".join(mutated[start_line - 1 : end_line + 1])


def _extract_clean_function_text(
    source_file: Path,
    start_line: int,
    end_line: int,
) -> str:
    lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(lines[start_line - 1 : end_line])


def load_stage2_examples(
    data_root: Path | None = None,
    *,
    verify_digest: bool = True,
) -> tuple[list[Stage2Example], list[Stage2Example]]:
    """Load Stage 2 training examples from the private dataset.

    Returns ``(train_examples, val_examples)`` using a repo-level 6/6 split:
    ``pallets/click`` -> train, ``python-attrs/attrs`` -> val.

    Generator markers and private signals are stripped from all features before
    the examples are returned; no generator-specific information reaches the model.
    """
    root = data_root if data_root is not None else PRIVATE_DATA_ROOT
    records_path = root / "records.json"

    if not records_path.exists():
        raise FileNotFoundError(
            f"Private dataset not found at {records_path}. "
            "Run on the authorised node with the private data in place."
        )

    if verify_digest:
        _verify_data_digest(records_path)

    records: list[dict[str, Any]] = json.loads(records_path.read_text(encoding="utf-8"))

    _verify_source_commits(records, root)

    train: list[Stage2Example] = []
    val: list[Stage2Example] = []

    for rec in records:
        repo: str = rec["source_repository"]
        if repo in EXCLUDED_REPOS:
            raise ValueError(f"Excluded repository found in dataset: {repo!r}")
        if repo not in _REPO_SOURCE_KEYS:
            raise ValueError(f"Unknown source repository: {repo!r}")

        split = "train" if repo == _TRAIN_REPO else "val"

        il: dict[str, Any] = rec["injection_location"]
        positive_cand = next(c for c in rec["candidates"] if c["label"] == "positive")
        negative_cand = next(c for c in rec["candidates"] if c["label"] == "negative")

        assert positive_cand["location"] == il, (
            f"Positive candidate location mismatch for {rec['record_id']!r}"
        )

        repo_key = _REPO_SOURCE_KEYS[repo]
        source_dir = root / "sources" / repo_key

        injection_stmt = _reconstruct_injection_statement(rec)
        pos_file = source_dir / il["file"]
        positive_text = _strip_generator_marker(
            _extract_mutated_function_text(
                pos_file,
                il["function"],
                injection_stmt,
                il["start_line"],
                il["end_line"],
            )
        )

        neg_loc: dict[str, Any] = negative_cand["location"]
        neg_file = source_dir / neg_loc["file"]
        negative_text = _strip_generator_marker(
            _extract_clean_function_text(
                neg_file,
                neg_loc["start_line"],
                neg_loc["end_line"],
            )
        )

        example = Stage2Example(
            record_id=rec["record_id"],
            query=_strip_generator_marker(rec["error_log"].strip()),
            positive=positive_text,
            negative=negative_text,
            source_repository=repo,
            dataset_version=rec["dataset_id"],
            split=split,
        )
        if split == "train":
            train.append(example)
        else:
            val.append(example)

    if len(train) != 6 or len(val) != 6:
        raise ValueError(
            f"Repo-level 6/6 split failed: got {len(train)} train + {len(val)} val. "
            "Expected 6 pallets/click (train) + 6 python-attrs/attrs (val)."
        )

    return train, val
