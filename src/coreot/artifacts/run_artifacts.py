from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coreot.data.hidden import EVALUATION_TRUTH_DIRNAME
from coreot.data.schemas import (
    CELL_SCORES_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    FORCED_LABEL_SUMMARY_COLUMNS,
    QUERY_TRUTH_COLUMNS,
)
from coreot.data.validation import HiddenTruthAccessError, validate_required_columns, validate_stage_can_read


@dataclass(frozen=True)
class ArtifactFailure:
    run_id: str
    artifact_kind: str
    state: str
    path: Path
    condition: str | None = None
    candidate_set: str | None = None
    method: str | None = None
    detail: str | None = None


class ArtifactContractError(Exception):
    state = "error"

    def __init__(
        self,
        message: str,
        *,
        artifact_kind: str,
        path: Path,
        run_id: str,
        condition: str | None = None,
        candidate_set: str | None = None,
        method: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.artifact_kind = artifact_kind
        self.path = path
        self.run_id = run_id
        self.condition = condition
        self.candidate_set = candidate_set
        self.method = method
        self.detail = detail

    def to_failure(self) -> ArtifactFailure:
        return ArtifactFailure(
            run_id=self.run_id,
            artifact_kind=self.artifact_kind,
            state=self.state,
            path=self.path,
            condition=self.condition,
            candidate_set=self.candidate_set,
            method=self.method,
            detail=self.detail or str(self),
        )


class ArtifactMissing(ArtifactContractError, FileNotFoundError):
    state = "missing"


class ArtifactInvalid(ArtifactContractError, ValueError):
    state = "invalid"


class ArtifactAccessDenied(ArtifactContractError, HiddenTruthAccessError):
    state = "access_denied"


class ArtifactBatchError(ValueError):
    def __init__(self, failures: list[ArtifactFailure]) -> None:
        self.failures = failures
        super().__init__(self._message())

    def _message(self) -> str:
        lines = ["Artifact batch validation failed:"]
        for failure in self.failures:
            location = [f"run_id={failure.run_id}", f"artifact={failure.artifact_kind}"]
            if failure.condition is not None:
                location.append(f"condition={failure.condition}")
            if failure.candidate_set is not None:
                location.append(f"candidate_set={failure.candidate_set}")
            if failure.method is not None:
                location.append(f"method={failure.method}")
            lines.append(
                f"{failure.state}: " + ", ".join(location) + f", path={failure.path}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _ArtifactContext:
    artifact_kind: str
    path: Path
    run_id: str
    stage: str
    condition: str | None = None
    candidate_set: str | None = None
    method: str | None = None
    allow_empty: bool = False


class CsvArtifact:
    def __init__(self, context: _ArtifactContext, required_columns: tuple[str, ...]) -> None:
        self._context = context
        self._required_columns = required_columns

    @property
    def path(self) -> Path:
        return self._context.path

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> pd.DataFrame:
        self._check_access()
        if not self.path.is_file():
            raise ArtifactMissing(
                f"Artifact does not exist: {self.path}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            )
        try:
            frame = pd.read_csv(self.path)
            self._validate_frame(frame)
            return frame
        except ArtifactContractError:
            raise
        except Exception as exc:
            raise ArtifactInvalid(
                f"Failed to read CSV artifact {self.path}: {exc}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc

    def write(self, frame: pd.DataFrame) -> None:
        self._validate_frame(frame)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False)

    def write_empty(self) -> None:
        self.write(pd.DataFrame(columns=self._required_columns))

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        try:
            validate_required_columns(frame, self._required_columns, str(self.path))
        except ValueError as exc:
            raise ArtifactInvalid(
                str(exc),
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc
        if frame.empty and not self._context.allow_empty:
            raise ArtifactInvalid(
                f"Artifact must not be empty: {self.path}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            )

    def _check_access(self) -> None:
        try:
            validate_stage_can_read(self.path, self._context.stage)
        except HiddenTruthAccessError as exc:
            raise ArtifactAccessDenied(
                str(exc),
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc


class ParquetArtifact:
    def __init__(self, context: _ArtifactContext, required_columns: tuple[str, ...]) -> None:
        self._context = context
        self._required_columns = required_columns

    @property
    def path(self) -> Path:
        return self._context.path

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> pd.DataFrame:
        self._check_access()
        if not self.path.is_file():
            raise ArtifactMissing(
                f"Artifact does not exist: {self.path}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            )
        try:
            frame = pd.read_parquet(self.path)
            self._validate_frame(frame)
            return frame
        except ArtifactContractError:
            raise
        except Exception as exc:
            raise ArtifactInvalid(
                f"Failed to read Parquet artifact {self.path}: {exc}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc

    def write(self, frame: pd.DataFrame) -> None:
        self._validate_frame(frame)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.path, index=False)

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        try:
            validate_required_columns(frame, self._required_columns, str(self.path))
        except ValueError as exc:
            raise ArtifactInvalid(
                str(exc),
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc
        if frame.empty and not self._context.allow_empty:
            raise ArtifactInvalid(
                f"Artifact must not be empty: {self.path}",
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            )

    def _check_access(self) -> None:
        try:
            validate_stage_can_read(self.path, self._context.stage)
        except HiddenTruthAccessError as exc:
            raise ArtifactAccessDenied(
                str(exc),
                artifact_kind=self._context.artifact_kind,
                path=self.path,
                run_id=self._context.run_id,
                condition=self._context.condition,
                candidate_set=self._context.candidate_set,
                method=self._context.method,
            ) from exc


class EvaluationTruthArtifacts:
    def __init__(self, run_root: Path, run_id: str, stage: str, condition: str) -> None:
        self._run_root = run_root
        self._run_id = run_id
        self._stage = stage
        self._condition = condition

    def query_truth(self) -> CsvArtifact:
        return CsvArtifact(
            _ArtifactContext(
                artifact_kind="query_truth",
                path=self._run_root
                / "benchmark"
                / self._condition
                / EVALUATION_TRUTH_DIRNAME
                / "query_truth.csv",
                run_id=self._run_id,
                stage=self._stage,
                condition=self._condition,
            ),
            QUERY_TRUTH_COLUMNS,
        )


class ScoringArtifacts:
    def __init__(
        self, run_root: Path, run_id: str, stage: str, condition: str, candidate_set: str
    ) -> None:
        self._run_root = run_root
        self._run_id = run_id
        self._stage = stage
        self._condition = condition
        self._candidate_set = candidate_set

    def cell_scores(self) -> ParquetArtifact:
        return ParquetArtifact(
            _ArtifactContext(
                artifact_kind="cell_scores",
                path=self._run_root
                / "scoring"
                / self._condition
                / self._candidate_set
                / "cell_scores.parquet",
                run_id=self._run_id,
                stage=self._stage,
                condition=self._condition,
                candidate_set=self._candidate_set,
            ),
            CELL_SCORES_COLUMNS,
        )


class EvaluationArtifacts:
    def __init__(self, run_root: Path, run_id: str, stage: str) -> None:
        self._run_root = run_root
        self._run_id = run_id
        self._stage = stage

    def metrics(self) -> CsvArtifact:
        return CsvArtifact(
            _ArtifactContext(
                artifact_kind="metrics",
                path=self._run_root / "evaluation" / "metrics.csv",
                run_id=self._run_id,
                stage=self._stage,
            ),
            EVALUATION_METRICS_COLUMNS,
        )

    def forced_label_summary(self) -> CsvArtifact:
        return CsvArtifact(
            _ArtifactContext(
                artifact_kind="forced_label_summary",
                path=self._run_root / "evaluation" / "forced_label_summary.csv",
                run_id=self._run_id,
                stage=self._stage,
                allow_empty=True,
            ),
            FORCED_LABEL_SUMMARY_COLUMNS,
        )


class RunArtifacts:
    def __init__(self, run_root: Path, stage: str) -> None:
        self.run_root = Path(run_root)
        self.stage = stage

    @property
    def run_id(self) -> str:
        return self.run_root.name

    def evaluation_truth(self, condition: str) -> EvaluationTruthArtifacts:
        return EvaluationTruthArtifacts(self.run_root, self.run_id, self.stage, condition)

    def scoring(self, condition: str, candidate_set: str) -> ScoringArtifacts:
        return ScoringArtifacts(
            self.run_root,
            self.run_id,
            self.stage,
            condition,
            candidate_set,
        )

    def evaluation(self) -> EvaluationArtifacts:
        return EvaluationArtifacts(self.run_root, self.run_id, self.stage)
