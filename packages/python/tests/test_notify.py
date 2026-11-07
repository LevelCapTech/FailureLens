"""Notification module: sanitising, channels, and never masking experiments."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from failurelens import notify

SECRET = "https://discord.example/api/webhooks/123/SECRET-TOKEN"


@pytest.fixture
def cfg(tmp_path: Path) -> notify.NotifyConfig:
    return notify.NotifyConfig(desktop=False, log_path=tmp_path / "n.log")


def test_message_contains_only_allowed_fields() -> None:
    text = notify.format_message(
        "stage2-training",
        "run\n/home/user/secret",
        notify.STATUS_FAILED,
        elapsed_seconds=7.64,
        metrics={"train.loss": 0.4123456, "bad key/with path": 1.0, "nan": float("nan")},
        error_type="ValueError",
    )
    assert "\n" not in text
    assert "elapsed=7.6s" in text
    assert "train.loss=0.4123" in text
    assert "bad key" not in text and "nan=" not in text
    assert "error=ValueError" in text


def test_log_always_written_and_webhook_payload(
    cfg: notify.NotifyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(notify, "_post_webhook", lambda url, msg: sent.append((url, msg)))
    full = notify.NotifyConfig(webhook_url=SECRET, desktop=False, log_path=cfg.log_path)
    delivered = notify.send("hello", notify.STATUS_FINISHED, full)
    assert delivered == ["log", "webhook"]
    assert sent == [(SECRET, "hello")]
    assert "FINISHED hello" in full.log_path.read_text()
    assert stat.S_IMODE(full.log_path.stat().st_mode) == 0o600


def test_webhook_failure_never_raises_or_logs_url(
    cfg: notify.NotifyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(url: str, msg: str) -> None:
        raise OSError(f"cannot reach {url}")

    monkeypatch.setattr(notify, "_post_webhook", boom)
    full = notify.NotifyConfig(webhook_url=SECRET, desktop=False, log_path=cfg.log_path)
    assert notify.send("m", notify.STATUS_FINISHED, full) == ["log"]
    logged = full.log_path.read_text()
    assert "SECRET-TOKEN" not in logged
    assert "webhook delivery failed (OSError)" in logged
    assert "SECRET-TOKEN" not in repr(full)


def test_local_command_receives_message(cfg: notify.NotifyConfig, tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    script = tmp_path / "hook.sh"
    script.write_text(f'#!/bin/sh\necho "$FAILURELENS_NOTIFY_STATUS $1" > {out}\n')
    script.chmod(0o700)
    full = notify.NotifyConfig(command=(str(script),), desktop=False, log_path=cfg.log_path)
    assert "command" in notify.send("done", notify.STATUS_FINISHED, full)
    assert out.read_text().strip() == "FINISHED done"


def test_disabled_sends_nothing(tmp_path: Path) -> None:
    disabled = notify.NotifyConfig(enabled=False, log_path=tmp_path / "n.log")
    assert notify.send("x", "FINISHED", disabled) == []
    assert not (tmp_path / "n.log").exists()


def test_config_file_permissions_and_env_override(tmp_path: Path) -> None:
    path = tmp_path / "notify.json"
    path.write_text(json.dumps({"webhook_url": SECRET, "desktop": False}))
    env = {"FAILURELENS_NOTIFY_CONFIG": str(path), "FAILURELENS_NOTIFY_LOG": str(tmp_path / "l")}
    os.chmod(path, 0o644)
    loose = notify.load_config(env)
    assert loose.webhook_url is None and loose.notes
    os.chmod(path, 0o600)
    assert notify.load_config(env).webhook_url == SECRET
    override = notify.load_config({**env, "FAILURELENS_NOTIFY_WEBHOOK_URL": "https://x.example/h"})
    assert override.webhook_url == "https://x.example/h"
    assert not notify.load_config({**env, "FAILURELENS_NOTIFY": "0"}).enabled
    assert notify.load_config({**env, "FAILURELENS_NOTIFY_WEBHOOK_URL": "file:///etc"}).notes


def test_write_config_is_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "cfg" / "notify.json"
    monkeypatch.setenv("FAILURELENS_NOTIFY_CONFIG", str(target))
    notify.write_config(SECRET)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text())["webhook_url"] == SECRET
    with pytest.raises(ValueError):
        notify.write_config("ftp://nope")


def _enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    log = tmp_path / "run.log"
    monkeypatch.setenv("FAILURELENS_NOTIFY", "1")
    monkeypatch.setenv("FAILURELENS_NOTIFY_LOG", str(log))
    monkeypatch.setenv("FAILURELENS_NOTIFY_DESKTOP", "0")
    return log


def test_notify_run_success_failure_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _enable(monkeypatch, tmp_path)
    with notify.notify_run("toy-training", "ok") as n:
        n.set_metrics({"train.final_loss": 0.05})
    with pytest.raises(RuntimeError), notify.notify_run("toy-training", "bad"):
        raise RuntimeError("/private/path leaked?")
    with pytest.raises(KeyboardInterrupt), notify.notify_run("toy-training", "stop"):
        raise KeyboardInterrupt
    with pytest.raises(SystemExit), notify.notify_run("toy-training", "exit1"):
        raise SystemExit(1)
    text = log.read_text()
    assert "STARTED" in text and "FINISHED [FailureLens] toy-training FINISHED run=ok" in text
    assert "train.final_loss=0.05" in text
    assert (
        "FAILED [FailureLens] toy-training FAILED run=bad" in text and "error=RuntimeError" in text
    )
    assert "/private/path" not in text
    assert "INTERRUPTED" in text
    assert "run=exit1" in text and "FAILED [FailureLens] toy-training FAILED run=exit1" in text


def test_notification_failure_does_not_mask_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(monkeypatch, tmp_path)
    monkeypatch.setenv("FAILURELENS_NOTIFY_LOG", str(tmp_path))  # a directory: log write fails
    monkeypatch.setattr(notify, "_post_webhook", lambda *_: 1 / 0)
    monkeypatch.setenv("FAILURELENS_NOTIFY_WEBHOOK_URL", SECRET)
    with notify.notify_run("toy-training", "x"):
        value = 42
    assert value == 42
    with pytest.raises(ValueError, match="real"), notify.notify_run("toy-training", "y"):
        raise ValueError("real")
