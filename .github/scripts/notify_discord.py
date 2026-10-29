"""Send narrowly selected FailureLens GitHub events to Discord."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "LevelCapTech/FailureLens"
COPILOT_AUTHORS = {
    (198982749, "Copilot"),  # Copilot coding agent
    (175728472, "Copilot"),  # Copilot line comments
    (175728472, "copilot-pull-request-reviewer[bot]"),  # Review summary
}


def notification(event_name: str, event: dict) -> str | None:
    if event.get("repository", {}).get("full_name") != REPOSITORY:
        return None

    action = event.get("action")
    pr = event.get("pull_request")
    if (
        event_name == "pull_request_target"
        and pr
        and (action == "opened" or (action == "closed" and pr.get("merged") is True))
    ):
        kind = "PR作成" if action == "opened" else "PRマージ"
        url = pr.get("html_url")
        number = pr.get("number")
    else:
        sender = event.get("sender", {})
        if (sender.get("id"), sender.get("login")) not in COPILOT_AUTHORS:
            return None
        if (
            event_name == "issue_comment"
            and action == "created"
            and event.get("issue", {}).get("pull_request")
        ):
            kind = "CopilotのPRコメント"
            url = event.get("comment", {}).get("html_url")
            number = event["issue"].get("number")
        elif event_name == "pull_request_review_comment" and action == "created" and pr:
            kind = "CopilotのPRレビューコメント"
            url = event.get("comment", {}).get("html_url")
            number = pr.get("number")
        elif (
            event_name == "pull_request_review"
            and action == "submitted"
            and pr
            and (event.get("review", {}).get("body") or "").strip()
        ):
            kind = "CopilotのPRレビュー本文"
            url = event.get("review", {}).get("html_url")
            number = pr.get("number")
        else:
            return None

    if (
        not isinstance(number, int)
        or not isinstance(url, str)
        or not url.startswith(f"https://github.com/{REPOSITORY}/pull/")
    ):
        return None
    return f"FailureLens GitHub通知 (自動)\n{kind}: #{number}\n{url}\n案件統括はGitHub上の原文を確認して対応を判断してください。"


def main() -> int:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    content = notification(event_name, event)
    if content is None:
        print("Event did not match notification filter")
        return 0

    webhook = os.environ.get("FAILURELENS_DISCORD_WEBHOOK_URL")
    if not webhook:
        print("FAILURELENS_DISCORD_WEBHOOK_URL is not configured", file=sys.stderr)
        return 1
    payload = {"content": content, "allowed_mentions": {"parse": []}}
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "FailureLens-GitHub-Notifier"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status not in (200, 204):
                print(f"Discord returned HTTP {response.status}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as error:
        print(f"Discord returned HTTP {error.code}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"Discord request failed: {error.reason.__class__.__name__}", file=sys.stderr)
        return 1
    print("Discord notification sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
