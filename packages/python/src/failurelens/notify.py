"""Best-effort run notifications for training and evaluation entry points.

Standard library only. Channels are configured once (a user config file and
environment overrides), never per run. Notification problems are swallowed:
they must neither fail nor mask the experiment. Messages carry only the run
kind and name, status, elapsed time and aggregate numeric metrics -- never
paths, records, exception text, or secrets. Webhook URLs are never logged.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_CONFIG = Path("~/.config/failurelens/notify.json")
DEFAULT_LOG = Path("~/.local/state/failurelens/notifications.log")
TIMEOUT_SECONDS = 5.0
_NAME_LIMIT = 80
_METRIC_LIMIT = 12
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

STATUS_STARTED = "STARTED"
STATUS_FINISHED = "FINISHED"
STATUS_FAILED = "FAILED"
STATUS_INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class NotifyConfig:
    """Resolved channels. ``webhook_url`` is a secret: never print or log it."""

    enabled: bool = True
    webhook_url: str | None = field(default=None, repr=False)
    command: tuple[str, ...] = ()
    desktop: bool = True
    log_path: Path = field(default_factory=lambda: DEFAULT_LOG.expanduser())
    notes: tuple[str, ...] = ()


def _clean_text(value: object, limit: int = _NAME_LIMIT) -> str:
    text = re.sub(r"[^\x20-\x7e]", "?", str(value))
    return text[:limit]


def config_path() -> Path:
    override = os.environ.get("FAILURELENS_NOTIFY_CONFIG")
    return Path(override).expanduser() if override else DEFAULT_CONFIG.expanduser()


def _valid_webhook(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme in ("https", "http") and bool(parts.netloc)


def load_config(environ: Mapping[str, str] | None = None) -> NotifyConfig:
    """Merge the user config file with environment overrides (env wins)."""
    env = os.environ if environ is None else environ
    notes: list[str] = []
    data: dict[str, Any] = {}
    path = (
        Path(env["FAILURELENS_NOTIFY_CONFIG"]).expanduser()
        if env.get("FAILURELENS_NOTIFY_CONFIG")
        else DEFAULT_CONFIG.expanduser()
    )
    try:
        raw = path.read_text(encoding="utf-8")
        mode = stat.S_IMODE(path.stat().st_mode)
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            data = loaded
        if mode & 0o077 and data.get("webhook_url"):
            data = {k: v for k, v in data.items() if k != "webhook_url"}
            notes.append("config file is group/other accessible; webhook ignored (chmod 600)")
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        notes.append("config file unreadable or invalid JSON; ignored")

    enabled = env.get("FAILURELENS_NOTIFY", "1").strip().lower() not in ("0", "false", "off", "no")
    if "enabled" in data and data["enabled"] is False:
        enabled = False
    webhook = env.get("FAILURELENS_NOTIFY_WEBHOOK_URL") or data.get("webhook_url")
    if webhook and not _valid_webhook(webhook):
        notes.append("webhook URL is not a valid http(s) URL; ignored")
        webhook = None
    command_text = env.get("FAILURELENS_NOTIFY_COMMAND")
    command: tuple[str, ...] = ()
    if command_text:
        command = tuple(shlex.split(command_text))
    elif isinstance(data.get("command"), list) and all(isinstance(c, str) for c in data["command"]):
        command = tuple(data["command"])
    desktop_env = env.get("FAILURELENS_NOTIFY_DESKTOP")
    desktop = (
        desktop_env.strip().lower() not in ("0", "false", "off", "no")
        if desktop_env is not None
        else data.get("desktop", True) is not False
    )
    log_text = env.get("FAILURELENS_NOTIFY_LOG") or data.get("log_path")
    log_path = Path(str(log_text)).expanduser() if log_text else DEFAULT_LOG.expanduser()
    return NotifyConfig(
        enabled=enabled,
        webhook_url=str(webhook) if webhook else None,
        command=command,
        desktop=desktop,
        log_path=log_path,
        notes=tuple(notes),
    )


def format_message(
    kind: str,
    name: str | None,
    status: str,
    *,
    elapsed_seconds: float | None = None,
    metrics: Mapping[str, float] | None = None,
    error_type: str | None = None,
) -> str:
    """Build the single-line message from the allowed, sanitized fields only."""
    parts = [f"[FailureLens] {_clean_text(kind)} {status}"]
    if name:
        parts.append(f"run={_clean_text(name)}")
    if elapsed_seconds is not None and math.isfinite(elapsed_seconds):
        parts.append(f"elapsed={elapsed_seconds:.1f}s")
    if error_type:
        parts.append(f"error={_clean_text(error_type, 40)}")
    shown = 0
    for key, value in (metrics or {}).items():
        if shown >= _METRIC_LIMIT:
            break
        if (
            _SAFE_KEY.match(key)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            parts.append(f"{key}={value:.4g}")
            shown += 1
    return " ".join(parts)


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")


def _post_webhook(url: str, message: str) -> None:
    body = json.dumps({"content": message[:1900]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "failurelens-notify/1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
        pass


def _run_command(command: tuple[str, ...], message: str, status: str) -> None:
    env = {**os.environ, "FAILURELENS_NOTIFY_STATUS": status}
    subprocess.run(
        [*command, message],
        env=env,
        timeout=10,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _desktop(message: str) -> None:
    binary = shutil.which("notify-send")
    if binary is None:
        raise FileNotFoundError("notify-send")
    subprocess.run(
        [binary, "FailureLens", message],
        timeout=5,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def send(message: str, status: str, config: NotifyConfig | None = None) -> list[str]:
    """Deliver ``message``; return the names of channels that succeeded. Never raises."""
    delivered: list[str] = []
    try:
        cfg = config or load_config()
        if not cfg.enabled:
            return delivered
        try:
            _append_log(cfg.log_path, f"{status} {message}")
            delivered.append("log")
            for note in cfg.notes:
                _append_log(cfg.log_path, f"NOTE {note}")
        except OSError:
            pass
        remote_ok = False
        if cfg.webhook_url:
            try:
                _post_webhook(cfg.webhook_url, message)
                delivered.append("webhook")
                remote_ok = True
            except Exception as exc:  # error text may contain the URL: log the type only
                _safe_log(cfg, f"NOTE webhook delivery failed ({type(exc).__name__})")
        if cfg.command:
            try:
                _run_command(cfg.command, message, status)
                delivered.append("command")
                remote_ok = True
            except Exception as exc:
                _safe_log(cfg, f"NOTE command delivery failed ({type(exc).__name__})")
        if cfg.desktop and not remote_ok:
            try:
                _desktop(message)
                delivered.append("desktop")
            except Exception:
                pass
    except Exception:
        pass
    return delivered


def _safe_log(cfg: NotifyConfig, line: str) -> None:
    with suppress(OSError):
        _append_log(cfg.log_path, line)


class RunNotifier:
    """Handle yielded by :func:`notify_run`; set aggregate metrics before exit."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def set_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics.update({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})


@contextmanager
def notify_run(kind: str, name: str | None = None) -> Iterator[RunNotifier]:
    """Announce start, then finish/fail/interrupt with the run's elapsed time."""
    started = time.perf_counter()
    notifier = RunNotifier()
    send(format_message(kind, name, STATUS_STARTED), STATUS_STARTED)
    try:
        yield notifier
    except SystemExit as exc:
        ok = exc.code in (None, 0)
        status = STATUS_FINISHED if ok else STATUS_FAILED
        send(
            format_message(
                kind,
                name,
                status,
                elapsed_seconds=time.perf_counter() - started,
                metrics=notifier.metrics,
            ),
            status,
        )
        raise
    except KeyboardInterrupt:
        send(
            format_message(
                kind,
                name,
                STATUS_INTERRUPTED,
                elapsed_seconds=time.perf_counter() - started,
                metrics=notifier.metrics,
            ),
            STATUS_INTERRUPTED,
        )
        raise
    except BaseException as exc:
        send(
            format_message(
                kind,
                name,
                STATUS_FAILED,
                elapsed_seconds=time.perf_counter() - started,
                metrics=notifier.metrics,
                error_type=type(exc).__name__,
            ),
            STATUS_FAILED,
        )
        raise
    else:
        send(
            format_message(
                kind,
                name,
                STATUS_FINISHED,
                elapsed_seconds=time.perf_counter() - started,
                metrics=notifier.metrics,
            ),
            STATUS_FINISHED,
        )


def write_config(webhook_url: str | None = None, *, command: list[str] | None = None) -> Path:
    """One-time setup helper: write the user config with mode 0600."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    if webhook_url is not None:
        if not _valid_webhook(webhook_url):
            raise ValueError("webhook URL must be an http(s) URL")
        data["webhook_url"] = webhook_url
    if command is not None:
        data["command"] = command
    data.setdefault("desktop", True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)
    return path


def main(argv: list[str] | None = None) -> int:
    """``python -m failurelens.notify {status|test|init}``."""
    import argparse
    import getpass

    parser = argparse.ArgumentParser(prog="failurelens.notify", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status", help="Show which channels are configured (no secrets).")
    sub.add_parser("test", help="Send one test notification.")
    init = sub.add_parser("init", help="Write ~/.config/failurelens/notify.json (mode 600).")
    init.add_argument(
        "--webhook-from-env",
        action="store_true",
        help="Read FAILURELENS_NOTIFY_WEBHOOK_URL instead of prompting.",
    )
    init.add_argument("--command", nargs="+", help="Local command; the message is appended.")
    args = parser.parse_args(argv)

    if args.action == "init":
        url: str | None = None
        if args.webhook_from_env:
            url = os.environ.get("FAILURELENS_NOTIFY_WEBHOOK_URL") or None
        elif args.command is None and sys.stdin.isatty():
            url = getpass.getpass("Webhook URL (input hidden, blank to skip): ").strip() or None
        path = write_config(url, command=args.command)
        print(f"wrote {path} (mode 600)")
        return 0
    cfg = load_config()
    if args.action == "status":
        print(f"enabled={cfg.enabled}")
        print(f"webhook={'configured' if cfg.webhook_url else 'not configured'}")
        print(f"command={'configured' if cfg.command else 'not configured'}")
        print(f"desktop={cfg.desktop and shutil.which('notify-send') is not None}")
        print(f"log={cfg.log_path}")
        for note in cfg.notes:
            print(f"note: {note}")
        return 0
    delivered = send(
        format_message("notify-test", "setup-check", STATUS_FINISHED), STATUS_FINISHED, cfg
    )
    print("delivered via: " + (", ".join(delivered) or "nothing"))
    return 0 if delivered else 1


if __name__ == "__main__":
    raise SystemExit(main())
