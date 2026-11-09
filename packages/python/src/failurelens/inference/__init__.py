"""Inference / analysis subpackage."""

from __future__ import annotations

from failurelens.inference.baseline import BaselineAnalyzer
from failurelens.inference.callgraph import CallGraphAnalyzer
from failurelens.inference.engine import Analyzer

#: Selectable dependency-free baseline strategies (keys are the public option values).
BASELINE_STRATEGIES: dict[str, type[BaselineAnalyzer] | type[CallGraphAnalyzer]] = {
    "stacktrace-lexical": BaselineAnalyzer,
    "stacktrace-callgraph": CallGraphAnalyzer,
}
#: Strategy used when none is requested.
DEFAULT_BASELINE_STRATEGY = "stacktrace-lexical"

__all__ = [
    "BASELINE_STRATEGIES",
    "DEFAULT_BASELINE_STRATEGY",
    "Analyzer",
    "BaselineAnalyzer",
    "CallGraphAnalyzer",
]
