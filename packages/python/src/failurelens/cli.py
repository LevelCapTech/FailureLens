"""``failurelens`` command-line interface.

Design goals (see docs/cli.md):

* ``--help`` works for every command; results go to **stdout**, logs/progress to
  **stderr**, so ``--format json`` output is always clean and machine-readable.
* Distinct, documented **exit codes** let CI and agents branch on the outcome.
* **No network access** is ever performed by the baseline.
"""

import json
import sys
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Annotated

import typer

from failurelens import __version__
from failurelens.api import FailureLens
from failurelens.errors import InputError, MissingDependencyError, ModelError
from failurelens.models.registry import BASELINE_MODEL_ID, get_model_info, list_models
from failurelens.reporting import to_json, to_text
from failurelens.schemas import AnalysisStatus


class ExitCode(IntEnum):
    """Process exit codes returned by the CLI."""

    OK = 0
    ERROR = 1
    INCONCLUSIVE = 3
    INPUT_ERROR = 4
    DEPENDENCY = 5


class OutputFormat(StrEnum):
    text = "text"
    json = "json"


app = typer.Typer(
    name="failurelens",
    help="AI-powered fault localization: find the code likely responsible for a failure.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Inspect available analysis backends / models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


def _echo_err(message: str) -> None:
    typer.echo(message, err=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(code=ExitCode.OK)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """FailureLens command-line interface."""


@app.command()
def version() -> None:
    """Print the installed FailureLens version."""
    typer.echo(__version__)


def _build_lens(model: str, revision: str | None) -> FailureLens:
    if model == BASELINE_MODEL_ID:
        return FailureLens.from_baseline()
    try:
        return FailureLens.from_pretrained(model, revision=revision)
    except MissingDependencyError as exc:
        _echo_err(f"error: {exc}")
        raise typer.Exit(code=ExitCode.DEPENDENCY) from exc
    except ModelError as exc:
        _echo_err(f"error: {exc}")
        raise typer.Exit(code=ExitCode.ERROR) from exc


def _status_exit_code(status: AnalysisStatus) -> ExitCode:
    if status is AnalysisStatus.OK:
        return ExitCode.OK
    if status is AnalysisStatus.ERROR:
        return ExitCode.ERROR
    return ExitCode.INCONCLUSIVE


RepoOption = Annotated[Path, typer.Option("--repo", "-r", help="Path to the repository.")]
FormatOption = Annotated[OutputFormat, typer.Option("--format", "-f", help="Output format.")]
QuietOption = Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress messages.")]


@app.command()
def index(
    repo: RepoOption,
    output_format: FormatOption = OutputFormat.text,
    quiet: QuietOption = False,
) -> None:
    """Build and summarize a repository index."""
    lens = FailureLens.from_baseline()
    if not quiet:
        _echo_err(f"Indexing {repo} ...")
    try:
        repo_index = lens.index(repo)
    except InputError as exc:
        _echo_err(f"error: {exc}")
        raise typer.Exit(code=ExitCode.INPUT_ERROR) from exc

    if output_format is OutputFormat.json:
        typer.echo(repo_index.model_dump_json(indent=2))
    else:
        typer.echo(
            f"Indexed {repo_index.file_count()} file(s), {repo_index.unit_count()} code unit(s)."
        )
        for warning in repo_index.warnings:
            _echo_err(f"! {warning}")


@app.command()
def diagnose(
    repo: RepoOption,
    log: Annotated[
        Path, typer.Option("--log", "-l", help="Path to the failure log / stack trace.")
    ],
    top: Annotated[
        int, typer.Option("--top", "-k", min=1, help="Number of candidates to return.")
    ] = 5,
    output_format: FormatOption = OutputFormat.text,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Analysis backend id (default: baseline).")
    ] = BASELINE_MODEL_ID,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Pinned model revision (model backends only).")
    ] = None,
    quiet: QuietOption = False,
) -> None:
    """Diagnose a failure and rank candidate fault locations."""
    lens = _build_lens(model, revision)
    if not quiet:
        _echo_err(f"Analyzing {repo} with log {log} ...")
    try:
        result = lens.diagnose(repo, error_log=log, top_k=top)
    except InputError as exc:
        _echo_err(f"error: {exc}")
        raise typer.Exit(code=ExitCode.INPUT_ERROR) from exc
    except ModelError as exc:
        _echo_err(f"error: {exc}")
        raise typer.Exit(code=ExitCode.ERROR) from exc

    if output_format is OutputFormat.json:
        typer.echo(to_json(result))
    else:
        typer.echo(to_text(result))

    raise typer.Exit(code=_status_exit_code(result.status))


@models_app.command("list")
def models_list(output_format: FormatOption = OutputFormat.text) -> None:
    """List available analysis backends and candidate base models."""
    infos = list_models()
    if output_format is OutputFormat.json:
        typer.echo(json.dumps([info.model_dump() for info in infos], indent=2))
        return
    for info in infos:
        extras = ", ".join(info.requires_extras) or "n/a"
        availability = "available" if info.available else f"needs extras: {extras}"
        released = "released" if info.released else "not released"
        typer.echo(f"{info.id}  [{info.kind}]  ({availability}; {released})")
        typer.echo(f"    {info.description}")


@models_app.command("info")
def models_info(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model/backend id to describe.")
    ] = BASELINE_MODEL_ID,
    output_format: FormatOption = OutputFormat.text,
) -> None:
    """Show details for a specific analysis backend / model."""
    info = get_model_info(model)
    if info is None:
        _echo_err(f"error: unknown model id: {model}")
        raise typer.Exit(code=ExitCode.INPUT_ERROR)
    if output_format is OutputFormat.json:
        typer.echo(info.model_dump_json(indent=2))
        return
    typer.echo(f"id         : {info.id}")
    typer.echo(f"kind       : {info.kind}")
    typer.echo(f"description: {info.description}")
    typer.echo(f"released   : {info.released}")
    typer.echo(f"available  : {info.available}")
    typer.echo(f"license    : {info.license or 'n/a'}")
    typer.echo(f"revision   : {info.revision or 'n/a'}")
    typer.echo(f"extras     : {', '.join(info.requires_extras) or 'none'}")
    if info.source_url:
        typer.echo(f"source     : {info.source_url}")


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
    sys.exit(ExitCode.OK)
