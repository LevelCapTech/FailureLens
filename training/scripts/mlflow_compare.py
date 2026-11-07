"""Render version-to-version comparison graphs from the local MLflow store.

Reads registered ``failurelens`` model versions (and evaluation runs tagged with
``model.version``), then writes PNG graphs plus ``index.html`` to
``.mlflow/reports/``. Read-only with respect to MLflow; no weights are touched.

    python -m training.scripts.mlflow_compare            # or: make compare
"""

from __future__ import annotations

import argparse
import html
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib.ticker import MaxNLocator

from failurelens.tracking_registry import REGISTERED_MODEL_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_B_TARGETS = {"function_level.top_1_accuracy": 0.4, "function_level.mrr": 0.6}
SCORE_KEYS = (
    "function_level.top_1_accuracy",
    "function_level.mrr",
    "file_level.top_1_accuracy",
    "file_level.mrr",
)
PALETTE = ("#2a6fdb", "#e07b00", "#2e9e5b", "#b03aa8", "#c9372c", "#5a6b7b")


@dataclass
class VersionRecord:
    label: str
    number: str
    status: str
    evaluation_status: str
    run_id: str
    historical: bool
    train: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    scores_source: str = ""


def _history(client: Any, run_id: str, key: str) -> list[float]:
    points = {m.step: m.value for m in client.get_metric_history(run_id, key)}
    return [points[step] for step in sorted(points) if math.isfinite(points[step])]


def collect(client: Any, *, include_tainted: bool = False) -> list[VersionRecord]:
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    latest: dict[str, Any] = {}
    for version in versions:
        label = version.tags.get("failurelens.version", f"registry-{version.version}")
        if label not in latest or int(version.version) > int(latest[label].version):
            latest[label] = version
    records: list[VersionRecord] = []
    for label, version in sorted(latest.items()):
        status = version.tags.get("status", "experimental")
        if status == "tainted-do-not-use" and not include_tainted:
            continue
        run = client.get_run(version.run_id)
        record = VersionRecord(
            label=label,
            number=str(version.version),
            status=status,
            evaluation_status=version.tags.get("evaluation_status", "not evaluated"),
            run_id=version.run_id,
            historical=version.tags.get("historical") == "true",
            train=_history(client, version.run_id, "train.loss"),
            validation=_history(client, version.run_id, "validation.loss"),
            gap=_history(client, version.run_id, "train.val_gap"),
            learning_rate=_history(client, version.run_id, "train.learning_rate"),
            diagnostics={k: v for k, v in run.data.metrics.items() if k.startswith("diag.")},
        )
        for key, value in version.tags.items():
            if key.startswith("eval."):
                try:
                    record.scores[key.removeprefix("eval.")] = float(value)
                except ValueError:
                    continue
        if record.scores:
            record.scores_source = "registry version tags"
        records.append(record)
    _attach_evaluation_runs(client, records)
    return records


def _attach_evaluation_runs(client: Any, records: list[VersionRecord]) -> None:
    by_label = {r.label: r for r in records}
    for experiment in client.search_experiments():
        for run in client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.`failurelens.kind` = 'baseline-evaluation'",
            order_by=["attributes.start_time ASC"],
        ):
            label = run.data.tags.get("model.version")
            record = by_label.get(label or "")
            if record is None or record.scores:
                continue
            record.scores = {k: v for k, v in run.data.metrics.items() if k in SCORE_KEYS}
            if record.scores:
                record.scores_source = f"evaluation run {run.info.run_name or run.info.run_id[:8]}"


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _save(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=140)


def _line_figure(
    records: list[VersionRecord], attr: str, title: str, ylabel: str, *, zero_line: bool = False
) -> Any:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.2), layout="constrained")
    axes = figure.subplots()
    plotted = False
    for index, record in enumerate(records):
        values = getattr(record, attr)
        if not values:
            continue
        plotted = True
        label = record.label + (" (historical)" if record.historical else "")
        axes.plot(range(1, len(values) + 1), values, "o-", color=_color(index), label=label)
    if zero_line:
        axes.axhline(0, color="#888888", linewidth=0.8)
    axes.set(xlabel="Epoch", ylabel=ylabel, title=title)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    if plotted:
        axes.legend()
    else:
        axes.text(0.5, 0.5, "No data recorded", ha="center", transform=axes.transAxes)
    axes.grid(alpha=0.2)
    return figure


def _loss_figure(records: list[VersionRecord]) -> Any:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.2), layout="constrained")
    axes = figure.subplots()
    for index, record in enumerate(records):
        epochs = range(1, len(record.train) + 1)
        axes.plot(epochs, record.train, "--", color=_color(index), label=f"{record.label} train")
        axes.plot(
            range(1, len(record.validation) + 1),
            record.validation,
            "o-",
            color=_color(index),
            label=f"{record.label} validation",
        )
    axes.set(xlabel="Epoch", ylabel="Loss (lower is better)", title="Training vs validation loss")
    axes.set_ylim(bottom=0)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    if records:
        axes.legend(fontsize=8)
    axes.grid(alpha=0.2)
    return figure


def _score_figure(records: list[VersionRecord]) -> Any:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.2), layout="constrained")
    axes = figure.subplots()
    scored = [r for r in records if r.scores]
    if not scored:
        axes.text(
            0.5,
            0.5,
            "No evaluation scores recorded yet.\nA loss value is not an accuracy score.",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        axes.set_title("Evaluation scores per version")
        return figure
    keys = [k for k in SCORE_KEYS if any(k in r.scores for r in scored)]
    width = 0.8 / len(scored)
    for index, record in enumerate(scored):
        positions = [i + index * width for i in range(len(keys))]
        axes.bar(
            positions,
            [record.scores.get(k, 0.0) for k in keys],
            width,
            color=_color(index),
            label=record.label,
        )
    axes.set_xticks([i + 0.4 - width / 2 for i in range(len(keys))])
    axes.set_xticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    axes.set(
        ylabel="Score (higher is better)", title="Evaluation scores per version", ylim=(0, 1.05)
    )
    for key, target in GATE_B_TARGETS.items():
        if key in keys:
            axes.hlines(
                target,
                keys.index(key) - 0.1,
                keys.index(key) + 0.9,
                colors="#c9372c",
                linestyles=":",
            )
    axes.legend()
    axes.grid(axis="y", alpha=0.2)
    return figure


def _signal(record: VersionRecord) -> str:
    flag = record.diagnostics.get("diag.overfit_signal")
    if flag is None:
        return "n/a"
    return "warning: validation got worse while training improved" if flag else "none seen"


def _fmt(values: list[float]) -> str:
    return f"{values[-1]:.3f}" if values else "-"


def _score_text(record: VersionRecord) -> str:
    if not record.scores:
        return "not evaluated"
    shown = [f"{k.replace('_level', '')}={v:.3f}" for k, v in sorted(record.scores.items())]
    return html.escape(", ".join(shown))


def render_index(records: list[VersionRecord]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(r.label)}</td>"
        f"<td>{html.escape(r.status)}{' / historical record' if r.historical else ''}</td>"
        f"<td>{_fmt(r.train)}</td><td>{_fmt(r.validation)}</td>"
        f"<td>{int(r.diagnostics['diag.best_epoch']) if 'diag.best_epoch' in r.diagnostics else '-'}"
        f" of {len(r.train) or '-'}</td>"
        f"<td>{html.escape(_signal(r))}</td>"
        f"<td>{_score_text(r)}</td>"
        f"<td>{html.escape(r.evaluation_status)}</td>"
        "</tr>"
        for r in records
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FailureLens version comparison</title>
<style>body{{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;
line-height:1.5}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}
td,th{{border:1px solid #ccc;padding:.35rem .5rem;text-align:left}}img{{max-width:100%}}</style>
</head><body>
<h1>FailureLens: comparing model versions</h1>
<p><b>How to read this page.</b> Each model version is trained with a loss that measures how
wrong its rankings were while learning (lower is better). <i>Training loss</i> uses the
examples it learned from; <i>validation loss</i> uses examples held back from learning.
If validation gets worse while training keeps improving, the model may be memorising
(overfitting). Loss is <b>not</b> an accuracy score: only the evaluation scores answer
"does it find the faulty code?" and they are shown as "not evaluated" until measured.
(日本語: loss は学習中の間違いの大きさで、精度ではありません。小さいほど良い値です。)</p>
<table>
<tr><th>Version</th><th>Status</th><th>Train loss (final)</th><th>Validation loss (final)</th>
<th>Best epoch</th><th>Overfitting signal</th><th>Evaluation scores</th>
<th>Evaluation status</th></tr>
{rows}
</table>
<h2>Loss curves</h2><img src="loss_curves.png" alt="Training and validation loss per version">
<h2>Generalisation gap (validation minus training)</h2>
<p>Near zero is healthy; a rising gap suggests the model is fitting the training set only.</p>
<img src="generalization_gap.png" alt="Gap per epoch">
<h2>Learning rate</h2><img src="learning_rate.png" alt="Learning rate per epoch">
<p>Historical versions show the rate derived from the known schedule, not a measurement.</p>
<h2>Evaluation scores</h2><img src="evaluation_scores.png" alt="Evaluation scores per version">
<p>Dotted red lines are the frozen acceptance targets (Top-1 0.4, MRR 0.6). Scores from the
bundled 1-case demo do not show generalisation.</p>
</body></html>
"""


def build_report(client: Any, output_dir: Path, *, include_tainted: bool = False) -> list[Path]:
    records = collect(client, include_tainted=include_tainted)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "loss_curves.png": _loss_figure(records),
        "generalization_gap.png": _line_figure(
            records, "gap", "Validation minus training loss", "Gap", zero_line=True
        ),
        "learning_rate.png": _line_figure(
            records, "learning_rate", "Learning rate after each epoch", "Learning rate"
        ),
        "evaluation_scores.png": _score_figure(records),
    }
    written: list[Path] = []
    for name, figure in figures.items():
        _save(figure, output_dir / name)
        written.append(output_dir / name)
    index = output_dir / "index.html"
    index.write_text(render_index(records), encoding="utf-8")
    written.append(index)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-tainted", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    from mlflow.tracking import MlflowClient

    uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{REPO_ROOT / '.mlflow' / 'mlflow.db'}"
    )
    output = args.output_dir or REPO_ROOT / ".mlflow" / "reports"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    for path in build_report(client, output, include_tainted=args.include_tainted):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
