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

Constraints honoured:
- ``label_rationale``, ground-truth markers, and invented identifiers are NOT
  used as features.  Only ``injection_method``, ``error_log``, and the source
  code at the candidate locations are used.
- ``psf/requests``, ``pallets/werkzeug``, ``pypa/pip`` are excluded.
- Data digest is verified before loading.

Split policy: repo-level 6 train / 6 val — all ``pallets/click`` records are
assigned to training, all ``python-attrs/attrs`` records to validation.
This avoids within-repository leakage between the train and val sets.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
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
            f"Dataset digest mismatch — expected {EXPECTED_DATASET_SHA256!r} "
            f"but got {digest!r}. Use the authorised private dataset only."
        )


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
            if "# issue17 injected fault" in stripped and stripped.startswith("raise"):
                return stripped.replace("  # issue17 injected fault", "").strip()
        raise ValueError(
            f"Cannot reconstruct raise statement from error_log "
            f"for record {record.get('record_id')!r}"
        )
    raise ValueError(f"Unknown injection_method: {method!r}")


def _find_function_node(tree: ast.Module, func_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
    raise ValueError(f"Function {func_name!r} not found in parsed AST")


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
    """
    with tempfile.TemporaryDirectory(prefix="fl_stage2_iso_") as tmp:
        isolated = Path(tmp) / source_file.name
        shutil.copy2(source_file, isolated)

        text = isolated.read_text(encoding="utf-8")
        tree = ast.parse(text)
        node = _find_function_node(tree, func_name)

        insert_at = node.body[0].lineno - 1  # 0-indexed, before first body stmt
        indent = " " * node.body[0].col_offset
        lines = text.splitlines(keepends=True)
        mutated = [
            *lines[:insert_at],
            indent + injection_statement + "  # issue17 injected fault\n",
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
        positive_text = _extract_mutated_function_text(
            pos_file,
            il["function"],
            injection_stmt,
            il["start_line"],
            il["end_line"],
        )

        neg_loc: dict[str, Any] = negative_cand["location"]
        neg_file = source_dir / neg_loc["file"]
        negative_text = _extract_clean_function_text(
            neg_file,
            neg_loc["start_line"],
            neg_loc["end_line"],
        )

        example = Stage2Example(
            record_id=rec["record_id"],
            query=rec["error_log"].strip(),
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
