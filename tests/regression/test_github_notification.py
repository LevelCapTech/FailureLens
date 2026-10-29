"""Regression checks for GitHub events forwarded to the coordinator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/notify_discord.py"
SPEC = importlib.util.spec_from_file_location("notify_discord", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
notify_discord = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notify_discord)


def test_pr_merge_is_notified_but_closed_without_merge_is_not() -> None:
    event = {
        "repository": {"full_name": "LevelCapTech/FailureLens"},
        "action": "closed",
        "pull_request": {
            "number": 42,
            "html_url": "https://github.com/LevelCapTech/FailureLens/pull/42",
            "merged": True,
        },
    }

    message = notify_discord.notification("pull_request_target", event)
    assert message is not None and "PRマージ: #42" in message

    event["pull_request"]["merged"] = False
    assert notify_discord.notification("pull_request_target", event) is None


def test_pr_opened_notification_is_preserved() -> None:
    event = {
        "repository": {"full_name": "LevelCapTech/FailureLens"},
        "action": "opened",
        "pull_request": {
            "number": 42,
            "html_url": "https://github.com/LevelCapTech/FailureLens/pull/42",
        },
    }

    message = notify_discord.notification("pull_request_target", event)
    assert message is not None and "PR作成: #42" in message
