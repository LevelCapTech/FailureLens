"""Check dataset splits for leakage and run the semantic quality gates.

Usage::

    python datasets/validation/check_splits.py \
        --split train=datasets/generated/train_records \
        --split eval=datasets/generated/eval_records [--redistributed]

Each ``NAME=PATH`` value is a JSON record file or a directory whose ``*.json``
files are records; a file may hold one record object or a list of them.

The report is printed to stdout as JSON. Exit status is ``0`` when there is no
cross-split leakage and every record passes :func:`failurelens.quality.validate_record_set`,
``1`` when either check fails, and ``2`` for usage or input errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from failurelens.quality import validate_record_set
from failurelens.split_leakage import find_split_leaks


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from one JSON file or from a directory of JSON files."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records: list[dict[str, Any]] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{file}: every record must be a JSON object")
            records.append(item)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A named split and its record file or directory. Repeat for each split.",
    )
    parser.add_argument(
        "--redistributed",
        action="store_true",
        help="Require each license to be on the redistribution allowlist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    splits: dict[str, list[dict[str, Any]]] = {}
    for item in args.split:
        name, sep, raw = item.partition("=")
        if not sep or not name or not raw:
            parser.error(f"--split expects NAME=PATH, got {item!r}")
        if name in splits:
            parser.error(f"split {name!r} given more than once")
        splits[name] = []
        try:
            splits[name] = load_records(Path(raw))
        except (OSError, ValueError) as exc:
            print(f"error: split {name!r}: {exc}", file=sys.stderr)
            return 2

    try:
        leaks = find_split_leaks(splits)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {"splits": {}, "leaks": [leak.to_dict() for leak in leaks]}
    clean = not leaks
    for name, records in splits.items():
        results = validate_record_set(records, redistributed=args.redistributed)
        rejected = [
            {
                "index": index,
                "record_id": records[index].get("record_id"),
                "rejection_reasons": result.to_dict()["rejection_reasons"],
            }
            for index, result in enumerate(results)
            if not result.passed
        ]
        clean = clean and not rejected
        report["splits"][name] = {
            "records": len(records),
            "passed": len(records) - len(rejected),
            "rejected": rejected,
        }
    report["ok"] = clean

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
