"""Single source of truth for the package version.

The build backend (hatchling) reads ``__version__`` from this module, and the
public package re-exports it as :data:`failurelens.__version__`.
"""

from __future__ import annotations

__version__ = "0.1.0"
