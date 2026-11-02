"""Read-only Hugging Face Job and model metadata inspection.

Never starts Jobs, changes repositories, prints Job commands/secrets, fetches
logs, or downloads weights. Output is not an inference/completion verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TERMINAL_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"})


def job_target(value: str, default_owner: str | None) -> tuple[str, str]:
    """Resolve a Job URL or bare ID without guessing the namespace."""
    if "://" in value:
        parsed = urlparse(value)
        parts = parsed.path.strip("/").split("/")
        if (
            parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or parsed.query
            or parsed.fragment
            or len(parts) != 3
            or parts[0] != "jobs"
            or not all(parts)
        ):
            raise ValueError("Use an https://huggingface.co/jobs/OWNER/ID URL")
        return parts[1], parts[2]
    if not value or "/" in value or not default_owner or "/" in default_owner:
        raise ValueError("A bare Job ID requires --job-owner OWNER")
    return default_owner, value


def collect(
    args: argparse.Namespace,
    api_factory: Callable[..., Any],
    card_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Collect explicit fields; never serialize SDK objects wholesale."""
    report: dict[str, Any] = {
        "read_only": True,
        "scope": "Job status and Hub metadata; inference is not verified",
    }
    auth_api = api_factory() if args.job or args.hardware else None
    if args.job:
        jobs = []
        for value in args.job:
            owner, job_id = job_target(value, args.job_owner)
            job = auth_api.inspect_job(job_id=job_id, namespace=owner)
            stage = str(job.status.stage)
            jobs.append(
                {
                    "id": job.id,
                    "owner": owner,
                    "stage": stage,
                    "terminal": stage in TERMINAL_STAGES,
                    "completed": stage == "COMPLETED",
                    "url": f"https://huggingface.co/jobs/{owner}/{job.id}",
                    "flavor": str(job.flavor),
                }
            )
        report["jobs"] = jobs
    if args.hardware:
        hardware = next((x for x in auth_api.list_jobs_hardware() if x.name == args.hardware), None)
        if hardware is None:
            raise ValueError("Requested hardware was not returned by the live API")
        report["hardware"] = {
            "name": hardware.name,
            "unit_cost_usd": hardware.unit_cost_usd,
            "unit_label": hardware.unit_label,
        }
    if args.repo:
        model_api = api_factory(token=False) if args.public else api_factory()
        info = model_api.model_info(args.repo, revision=args.revision, files_metadata=True)
        files = []
        for item in info.siblings or []:
            lfs = getattr(item, "lfs", None)
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            files.append({"path": item.rfilename, "bytes": item.size, "lfs_sha256": sha})
        report["model"] = {
            "id": info.id,
            "revision": info.sha,
            "private": info.private,
            "gated": info.gated,
            "metadata_access": "anonymous" if args.public else "configured_authentication",
            "files": files,
        }
    if args.card:
        if card_factory is None:
            raise ValueError("Card validation support is unavailable")
        card_factory(args.card.read_text(encoding="utf-8")).validate(repo_type="model")
        report["card"] = {"hub_metadata_valid": True}
    return report


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", help="Model repo ID, for example owner/model")
    cli.add_argument("--revision", help="Immutable commit; omit to inspect current main")
    cli.add_argument("--public", action="store_true", help="Inspect model metadata anonymously")
    cli.add_argument("--job", action="append", default=[], help="Job URL or bare ID; repeatable")
    cli.add_argument("--job-owner", help="Namespace for bare Job IDs")
    cli.add_argument("--hardware", help="Look up this hardware's live price and unit")
    cli.add_argument("--card", type=Path, help="Validate this non-secret card via the Hub API")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if not any((args.repo, args.job, args.card, args.hardware)):
        cli.error("Specify --repo, --job, --card or --hardware")
    if (args.public or args.revision) and not args.repo:
        cli.error("--public and --revision require --repo")
    try:
        for value in args.job:
            job_target(value, args.job_owner)
        from huggingface_hub import HfApi, ModelCard

        report = collect(args, HfApi, ModelCard)
    except ImportError:
        print("huggingface_hub is required in this Python environment.", file=sys.stderr)
        return 1
    except Exception as error:
        # Do not echo SDK exceptions, request bodies or credentials on failure.
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(
            json.dumps({"error_type": type(error).__name__, "http_status": status}), file=sys.stderr
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
