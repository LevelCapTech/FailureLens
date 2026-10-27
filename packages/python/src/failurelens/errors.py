"""Exception hierarchy for FailureLens.

All public errors derive from :class:`FailureLensError` so callers can catch a
single base type. Errors are deliberately granular to let the CLI map them onto
distinct, documented exit codes (see :mod:`failurelens.cli`).
"""

from __future__ import annotations


class FailureLensError(Exception):
    """Base class for every error raised by FailureLens."""


class InputError(FailureLensError):
    """A user-supplied input is missing or invalid (e.g. a path that does not exist)."""


class RepositoryError(InputError):
    """The target repository could not be read or indexed."""


class LogParseError(FailureLensError):
    """A failure log could not be parsed by any registered parser."""


class ModelError(FailureLensError):
    """A model (or analysis backend) could not be resolved, loaded, or executed."""


class MissingDependencyError(ModelError):
    """An optional dependency required for a backend is not installed.

    The message always includes an actionable ``pip install`` hint so users and
    agents know exactly how to enable the requested feature.
    """

    def __init__(self, feature: str, extra: str, packages: str) -> None:
        self.feature = feature
        self.extra = extra
        self.packages = packages
        super().__init__(
            f"{feature} requires optional dependencies that are not installed "
            f"({packages}). Install them with: pip install 'failurelens[{extra}]'"
        )
