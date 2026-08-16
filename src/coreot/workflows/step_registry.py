from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepDefinition:
    name: str
    config: Path
    depends_on: tuple[str, ...]
    description: str


DEFAULT_STEP_REGISTRY: dict[str, StepDefinition] = {
    "raw-import": StepDefinition(
        name="raw-import",
        config=Path("experiments/missing_celltype/configs/raw_import.yaml"),
        depends_on=(),
        description="Standardize raw AnnData input.",
    ),
    "benchmark-build": StepDefinition(
        name="benchmark-build",
        config=Path("experiments/missing_celltype/configs/benchmark.yaml"),
        depends_on=("raw-import",),
        description="Create incomplete-reference and full-reference-control conditions.",
    ),
    "model-visible-derivation": StepDefinition(
        name="model-visible-derivation",
        config=Path("experiments/missing_celltype/configs/derivation.yaml"),
        depends_on=("benchmark-build",),
        description="Preprocess expression and construct model-visible priors.",
    ),
    "embedding": StepDefinition(
        name="embedding",
        config=Path("experiments/missing_celltype/configs/embedding.yaml"),
        depends_on=("model-visible-derivation",),
        description="Compute fixed provider embeddings.",
    ),
    "candidate-cost": StepDefinition(
        name="candidate-cost",
        config=Path("experiments/missing_celltype/configs/candidates.yaml"),
        depends_on=("embedding",),
        description="Build query-to-reference candidate graph and scaled costs.",
    ),
    "transport": StepDefinition(
        name="transport",
        config=Path("experiments/missing_celltype/configs/transport.yaml"),
        depends_on=("candidate-cost",),
        description="Run transport and baseline methods.",
    ),
    "score-abstain": StepDefinition(
        name="score-abstain",
        config=Path("experiments/missing_celltype/configs/scoring.yaml"),
        depends_on=("transport",),
        description="Compute prior-adjusted scores and abstention calls without hidden truth.",
    ),
    "evaluation": StepDefinition(
        name="evaluation",
        config=Path("experiments/missing_celltype/configs/evaluation.yaml"),
        depends_on=("score-abstain",),
        description="Evaluate against hidden query labels.",
    ),
    "report": StepDefinition(
        name="report",
        config=Path("experiments/missing_celltype/configs/report.yaml"),
        depends_on=("evaluation",),
        description="Assemble benchmark report and figures.",
    ),
}

OPTIONAL_STEP_REGISTRY: dict[str, StepDefinition] = {}
