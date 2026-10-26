"""Public API for FailureLens.

``FailureLens`` is the single entry point used by both the Python API and the
CLI. Construct it with one of the factory methods and call :meth:`diagnose`:

    >>> from failurelens import FailureLens
    >>> lens = FailureLens.from_baseline()
    >>> result = lens.diagnose(repository="./project", error_log="./failure.log", top_k=5)

The pretrained path is also exposed, but note that FailureLens has **not** yet
published its own model; :meth:`from_pretrained` loads an *external* embedding
model (via the optional ``ml`` extra) and never silently falls back to the
baseline, so results are never misrepresented.
"""

from __future__ import annotations

import time
from os import PathLike
from pathlib import Path

from failurelens.config import DiagnoseConfig, IndexConfig
from failurelens.errors import InputError
from failurelens.indexing.base import RepositoryIndex
from failurelens.indexing.repository import build_index
from failurelens.inference.baseline import BaselineAnalyzer
from failurelens.inference.engine import Analyzer
from failurelens.parsers.base import ParserRegistry, default_registry
from failurelens.schemas import AnalysisStatus, DiagnosisResult


class FailureLens:
    """A reusable fault-localization engine bound to a single analysis backend."""

    def __init__(
        self,
        analyzer: Analyzer,
        *,
        parser_registry: ParserRegistry | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._parsers = parser_registry or default_registry()

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_baseline(cls, *, parser_registry: ParserRegistry | None = None) -> FailureLens:
        """Create an engine using the dependency-free baseline analyzer."""
        return cls(BaselineAnalyzer(), parser_registry=parser_registry)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        parser_registry: ParserRegistry | None = None,
    ) -> FailureLens:
        """Create an engine that reranks candidates with an embedding model.

        Requires the optional ``ml`` extra (``pip install 'failurelens[ml]'``).
        ``model_id`` refers to an **externally hosted** model; nothing is bundled.
        Pinning ``revision`` is strongly recommended for reproducibility.
        """
        # Imported lazily so the baseline never pays for optional dependencies.
        from failurelens.inference.embedding import EmbeddingAnalyzer, load_sentence_transformer

        encoder = load_sentence_transformer(model_id, revision=revision, device=device)
        analyzer = EmbeddingAnalyzer(encoder=encoder, model_id=model_id, model_revision=revision)
        return cls(analyzer, parser_registry=parser_registry)

    # ------------------------------------------------------------------ properties
    @property
    def analyzer(self) -> Analyzer:
        return self._analyzer

    @property
    def parsers(self) -> ParserRegistry:
        return self._parsers

    # ------------------------------------------------------------------ operations
    def index(
        self, repository: str | PathLike[str], *, config: IndexConfig | None = None
    ) -> RepositoryIndex:
        """Build a reusable index for ``repository`` (pass it back to :meth:`diagnose`)."""
        return build_index(repository, config)

    def diagnose(
        self,
        repository: str | PathLike[str],
        *,
        error_log: str | PathLike[str] | None = None,
        error_text: str | None = None,
        top_k: int | None = None,
        index: RepositoryIndex | None = None,
        config: DiagnoseConfig | None = None,
    ) -> DiagnosisResult:
        """Diagnose a failure and return ranked candidate locations.

        Provide exactly one of ``error_log`` (a path to a log file) or
        ``error_text`` (the raw log contents). Pass a prebuilt ``index`` to reuse
        a previous :meth:`index` call and avoid re-scanning the repository.
        """
        started = time.perf_counter()
        config = config or DiagnoseConfig()
        if top_k is not None:
            config = config.model_copy(update={"top_k": top_k})

        text = self._resolve_log_text(error_log, error_text)
        repository_str = str(repository)

        parsed_log = self._parsers.parse(text)
        if parsed_log is None:
            elapsed = (time.perf_counter() - started) * 1000.0
            return DiagnosisResult(
                status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
                repository=repository_str,
                analysis_method=self._analyzer.analysis_method,
                model_id=self._analyzer.model_id,
                model_revision=self._analyzer.model_revision,
                execution_time_ms=round(elapsed, 3),
                warnings=["Could not parse the provided log with any registered parser."],
            )

        repo_index = index if index is not None else build_index(repository, config.index)
        result = self._analyzer.analyze(
            repository=repository_str,
            parsed_log=parsed_log,
            index=repo_index,
            config=config,
        )
        result.execution_time_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _resolve_log_text(error_log: str | PathLike[str] | None, error_text: str | None) -> str:
        if (error_log is None) == (error_text is None):
            raise InputError(
                "Provide exactly one of 'error_log' (path) or 'error_text' (raw text)."
            )
        if error_text is not None:
            return error_text
        assert error_log is not None
        path = Path(error_log)
        if not path.exists():
            raise InputError(f"Log file does not exist: {path}")
        if not path.is_file():
            raise InputError(f"Log path is not a file: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InputError(f"Could not read log file {path}: {exc}") from exc
