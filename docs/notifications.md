# Run notifications

**Working.** Training, evaluation and history-import entry points announce when
a run **starts**, **finishes**, **fails** or is **interrupted**, so a long run does
not need watching. You set the channels up **once**; nothing has to be
re-registered for each run.

Covered entry points: `training.scripts.smoke_train`,
`training.scripts.finetune_train`, `evaluation/run_baseline_eval.py` and
`training.scripts.import_history`. Notification is independent of MLflow: it works
with or without `--mlflow`.

## What a message contains

One line: run kind, status, run name (if you gave `--mlflow-run-name`), elapsed
time, and a few aggregate numbers (for example final train/validation loss).
Failures show only the error *type* (such as `ValueError`). Messages never contain
file paths, dataset records, exception text, tokens or webhook URLs.

```text
[FailureLens] stage2-training FINISHED run=v0.2-candidate-1 elapsed=7.6s train.loss=0.4125 validation.loss=0.7208
```

## Channels

Every notification goes to all channels that apply:

| Channel | Always on? | Notes |
| --- | --- | --- |
| Local log `~/.local/state/failurelens/notifications.log` | Yes | Append-only, mode 600. The record of truth if a network channel is down. |
| Webhook (Discord-compatible JSON `{"content": ...}`) | If configured | HTTPS (or HTTP) URL. A 5 s timeout; errors are logged by type only. |
| Local command | If configured | Run without a shell; the message is the last argument and `FAILURELENS_NOTIFY_STATUS` is set. |
| Desktop `notify-send` | Fallback | Used only when no webhook/command delivered and the tool exists. |

A notification problem never fails or hides the experiment's own result or error.

## One-time setup

```bash
# Option A: interactive (the URL is typed hidden, not stored in shell history)
python -m failurelens.notify init

# Option B: from an environment variable you already hold
FAILURELENS_NOTIFY_WEBHOOK_URL='https://discord.com/api/webhooks/...' \
  python -m failurelens.notify init --webhook-from-env

# Option C: a local command, e.g. a script that forwards to your own tool
python -m failurelens.notify init --command /path/to/my-notify.sh

python -m failurelens.notify status   # shows which channels are set; never prints secrets
make notify-test                      # sends one test message
```

This writes `~/.config/failurelens/notify.json` with mode 600:

```json
{"webhook_url": "https://...", "command": ["/path/to/hook"], "desktop": true}
```

If the file is readable by other users the webhook in it is **ignored** (the log
records a note); run `chmod 600 ~/.config/failurelens/notify.json`.

## Environment overrides

Environment variables win over the file: `FAILURELENS_NOTIFY=0` (turn off),
`FAILURELENS_NOTIFY_WEBHOOK_URL`, `FAILURELENS_NOTIFY_COMMAND` (shell-style words),
`FAILURELENS_NOTIFY_DESKTOP=0`, `FAILURELENS_NOTIFY_LOG`, `FAILURELENS_NOTIFY_CONFIG`.
The test suite sets `FAILURELENS_NOTIFY=0`, so tests never contact a channel or write
to your log.

## Using it from new runners

```python
from failurelens.notify import notify_run

with notify_run("my-kind", run_name) as notifier:
    result = do_the_work()
    notifier.set_metrics({"validation.loss": result.val_loss})  # aggregates only
```

The context manager announces the start, then the outcome. Only numbers with
simple names are included in messages.
