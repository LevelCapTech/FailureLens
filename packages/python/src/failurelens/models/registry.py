"""Registry of analysis backends and candidate base models.

This registry is intentionally honest about what exists today:

* The ``baseline`` backend is **implemented and available** with no extra deps.
* The embedding backend is **optional** and requires the ``ml`` extra plus an
  externally hosted base model (none is bundled or auto-downloaded).
* FailureLens has **not** published its own fine-tuned model yet; such entries
  are marked ``released=False`` so results are never misrepresented.

Nothing here downloads anything. It only describes options.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Identifier of the always-available, dependency-free baseline backend.
BASELINE_MODEL_ID = "baseline"


class ModelInfo(BaseModel):
    """Metadata describing an analysis backend or a candidate base model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Backend id or Hugging Face model id.")
    kind: str = Field(description="One of: baseline, embedding.")
    description: str
    released: bool = Field(
        description="Whether usable weights are actually published for this entry."
    )
    available: bool = Field(
        description="Whether this backend can run in the current environment without downloads."
    )
    revision: str | None = Field(
        default=None, description="Recommended pinned revision for reproducibility."
    )
    license: str | None = Field(default=None, description="License of the referenced model.")
    requires_extras: tuple[str, ...] = Field(
        default=(), description="pip extras needed to use this backend (e.g. 'ml')."
    )
    source_url: str | None = Field(default=None)


_REGISTRY: tuple[ModelInfo, ...] = (
    ModelInfo(
        id=BASELINE_MODEL_ID,
        kind="baseline",
        description=(
            "Dependency-free stack-trace + lexical fault localization. Always available; "
            "runs on CPU with no network access."
        ),
        released=True,
        available=True,
        license="Apache-2.0",
    ),
    ModelInfo(
        id="nomic-ai/CodeRankEmbed",
        kind="embedding",
        description=(
            "Candidate base model for the optional embedding reranker: a ~137M-parameter "
            "code-search bi-encoder. External model; not bundled or auto-downloaded."
        ),
        released=True,
        available=False,
        revision=None,
        license="MIT (verify on the model card before use)",
        requires_extras=("ml",),
        source_url="https://huggingface.co/nomic-ai/CodeRankEmbed",
    ),
    ModelInfo(
        id="Salesforce/SweRankEmbed-Small",
        kind="embedding",
        description=(
            "Candidate base model for evaluation/comparison. Check its (non-commercial) "
            "license terms before use; do not redistribute. External model."
        ),
        released=True,
        available=False,
        revision=None,
        license="Non-commercial (verify on the model card before use)",
        requires_extras=("ml",),
        source_url="https://huggingface.co/Salesforce/SweRankEmbed-Small",
    ),
)


def list_models() -> list[ModelInfo]:
    """Return all known backends / candidate base models."""
    return list(_REGISTRY)


def get_model_info(model_id: str) -> ModelInfo | None:
    """Return registry metadata for ``model_id``, or ``None`` if unknown."""
    for info in _REGISTRY:
        if info.id == model_id:
            return info
    return None


__all__ = ["BASELINE_MODEL_ID", "ModelInfo", "get_model_info", "list_models"]
