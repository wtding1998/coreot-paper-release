from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


PACKAGE_SCHEMA = "uot_portable_package_v1"
CONTROLLED_FAMILIES = {"hiha", "pbmc"}
FIGURE_METHODS = {
    "hiha": {"scdot", "tacco_ot"},
    "pbmc": {"scdot", "tacco_ot"},
    "mouse_spleen": {"pamona", "scotv2"},
}


class SealedFigureInputError(ValueError):
    """Raised when sealed evidence cannot satisfy a figure input contract."""


@dataclass(frozen=True)
class FigureInputBinding:
    family: Literal["hiha", "pbmc", "mouse_spleen"]
    endpoint: str
    seed: int
    method: str
    condition_id: str


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_unique_ids(frame: pd.DataFrame, *, side: str) -> None:
    if "cell_id" not in frame:
        raise SealedFigureInputError(f"{side} frame is missing cell_id")
    ids = frame["cell_id"].astype(str)
    if ids.eq("").any():
        raise SealedFigureInputError(f"{side} frame contains empty cell_id values")
    if ids.duplicated().any():
        duplicates = sorted(ids.loc[ids.duplicated(keep=False)].unique())[:5]
        raise SealedFigureInputError(
            f"{side} frame contains duplicate cell_id values: {duplicates}"
        )


def exact_cell_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    suffixes: tuple[str, str] = ("", "_sealed"),
) -> pd.DataFrame:
    """Join two cell tables after proving exact, unique cell-id identity."""

    _require_unique_ids(left, side="left")
    _require_unique_ids(right, side="right")
    left_ids = set(left["cell_id"].astype(str))
    right_ids = set(right["cell_id"].astype(str))
    if left_ids != right_ids:
        missing = sorted(left_ids - right_ids)[:5]
        extra = sorted(right_ids - left_ids)[:5]
        raise SealedFigureInputError(
            "cell_id sets differ; "
            f"missing_from_right={missing}, extra_in_right={extra}"
        )
    left_copy = left.copy()
    right_copy = right.copy()
    left_copy["cell_id"] = left_copy["cell_id"].astype(str)
    right_copy["cell_id"] = right_copy["cell_id"].astype(str)
    return left_copy.merge(
        right_copy,
        on="cell_id",
        how="inner",
        validate="one_to_one",
        sort=False,
        suffixes=suffixes,
    )


class UOTBaselineFigureInputs:
    """Read-only, manifest-checked access to sealed figure evidence."""

    def __init__(self, package_root: Path | str) -> None:
        self.package_root = Path(package_root).resolve()
        self._manifest_path = self.package_root / "manifest.json"
        self._package_path = self.package_root / "package.json"
        self._report_path = self.package_root / "verification/report.json"
        for path in (self._manifest_path, self._package_path, self._report_path):
            if not path.is_file():
                raise SealedFigureInputError(f"Missing sealed package member: {path}")

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        package = json.loads(self._package_path.read_text(encoding="utf-8"))
        report = json.loads(self._report_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PACKAGE_SCHEMA:
            raise SealedFigureInputError("Unexpected sealed manifest schema")
        if package.get("schema_version") != PACKAGE_SCHEMA:
            raise SealedFigureInputError("Unexpected sealed package schema")
        if package.get("model_fitting_permitted") is not False:
            raise SealedFigureInputError("Sealed package must prohibit model fitting")
        if report.get("schema_version") != PACKAGE_SCHEMA:
            raise SealedFigureInputError("Unexpected verification-report schema")
        if report.get("verification_status") != "passed":
            raise SealedFigureInputError("Sealed package verification did not pass")
        if report.get("model_fitting_executed") is not False:
            raise SealedFigureInputError("Verification report records model fitting")

        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise SealedFigureInputError("Sealed manifest has no file inventory")
        self._manifest = {
            str(entry["path"]): str(entry["sha256"])
            for entry in files
            if isinstance(entry, dict) and "path" in entry and "sha256" in entry
        }
        if len(self._manifest) != len(files):
            raise SealedFigureInputError("Sealed manifest contains invalid or duplicate entries")
        self._verify_member("verification/report.json")

    def _member(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SealedFigureInputError(f"Unsafe sealed member path: {relative_path}")
        path = (self.package_root / relative).resolve()
        if self.package_root not in path.parents:
            raise SealedFigureInputError(f"Sealed member escapes package root: {relative_path}")
        return path

    def _verify_member(self, relative_path: str, *, expected_sha256: str | None = None) -> Path:
        if relative_path not in self._manifest:
            raise SealedFigureInputError(
                f"Sealed member is absent from manifest: {relative_path}"
            )
        path = self._member(relative_path)
        if not path.is_file() or path.is_symlink():
            raise SealedFigureInputError(f"Sealed member is missing or not regular: {path}")
        observed = _sha256_file(path)
        declared = self._manifest[relative_path]
        if observed != declared:
            raise SealedFigureInputError(f"Manifest checksum mismatch: {relative_path}")
        if expected_sha256 is not None and observed != expected_sha256:
            raise SealedFigureInputError(f"Metric checksum mismatch: {relative_path}")
        return path

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        return pd.read_csv(self._verify_member(relative_path))

    @staticmethod
    def _validate_binding(binding: FigureInputBinding) -> None:
        allowed = FIGURE_METHODS.get(binding.family)
        if allowed is None or binding.method not in allowed:
            raise SealedFigureInputError(f"Unsupported figure binding: {binding}")
        if not binding.endpoint or not binding.condition_id:
            raise SealedFigureInputError("Figure binding fields must be nonempty")
        if binding.seed < 0:
            raise SealedFigureInputError("Figure binding seed must be nonnegative")

    def controlled_metrics(
        self,
        binding: FigureInputBinding,
        *,
        evaluation_scope: str,
    ) -> pd.DataFrame:
        self._validate_binding(binding)
        if binding.family not in CONTROLLED_FAMILIES:
            raise SealedFigureInputError("Controlled metrics require HIHA or PBMC")
        frame = self._read_csv("metrics/all_by_run_metrics.csv")
        selected = frame.loc[
            frame["family"].eq(binding.family)
            & frame["endpoint"].eq(binding.endpoint)
            & frame["seed"].eq(binding.seed)
            & frame["method"].eq(binding.method)
            & frame["condition_id"].eq(binding.condition_id)
            & frame["evaluation_scope"].eq(evaluation_scope)
        ].copy()
        if selected.empty:
            raise SealedFigureInputError(
                f"No controlled metrics for binding={binding}, scope={evaluation_scope}"
            )
        if selected["row_id"].duplicated().any() or selected["metric"].duplicated().any():
            raise SealedFigureInputError("Controlled metric binding is not unique")
        if not selected["execution_status"].eq("success").all():
            raise SealedFigureInputError("Controlled metric binding includes failed execution")
        return selected.sort_values("metric", ignore_index=True)

    def controlled_summary(
        self,
        *,
        family: str,
        endpoint: str,
        method: str,
        condition_id: str,
        evaluation_scope: str,
    ) -> pd.DataFrame:
        if family not in CONTROLLED_FAMILIES or method not in FIGURE_METHODS[family]:
            raise SealedFigureInputError("Unsupported controlled-summary binding")
        frame = self._read_csv("metrics/controlled_summary_metrics.csv")
        selected = frame.loc[
            frame["family"].eq(family)
            & frame["endpoint"].eq(endpoint)
            & frame["method"].eq(method)
            & frame["condition_id"].eq(condition_id)
            & frame["evaluation_scope"].eq(evaluation_scope)
        ].copy()
        if selected.empty or selected["metric"].duplicated().any():
            raise SealedFigureInputError("Controlled summary binding is missing or duplicated")
        return selected.sort_values("metric", ignore_index=True)

    def predictions(self, binding: FigureInputBinding) -> pd.DataFrame:
        self._validate_binding(binding)
        if binding.family in CONTROLLED_FAMILIES:
            metrics = self.controlled_metrics(binding, evaluation_scope="represented_state")
        else:
            metrics = self.mouse_summary(binding.method, evaluation_scope="represented_state")
            metrics = metrics.loc[
                metrics["endpoint"].eq(binding.endpoint)
                & metrics["seed"].eq(binding.seed)
                & metrics["condition_id"].eq(binding.condition_id)
            ]
        artifacts = metrics[["prediction_artifact", "prediction_sha256"]].drop_duplicates()
        if len(artifacts) != 1:
            raise SealedFigureInputError("Prediction binding does not resolve uniquely")
        artifact = str(artifacts.iloc[0]["prediction_artifact"])
        artifact_sha256 = str(artifacts.iloc[0]["prediction_sha256"])
        path = self._verify_member(artifact, expected_sha256=artifact_sha256)
        frame = pd.read_csv(path)
        required = {
            "cell_id",
            "method",
            "condition_id",
            "pred_label",
            "z_absent_score",
            "detection_eligible",
            "implementation_provenance_id",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SealedFigureInputError(f"Prediction schema is missing columns: {missing}")
        _require_unique_ids(frame, side="prediction")
        if not frame["method"].eq(binding.method).all():
            raise SealedFigureInputError("Prediction method differs from binding")
        if not frame["condition_id"].eq(binding.condition_id).all():
            raise SealedFigureInputError("Prediction condition differs from binding")
        scores = pd.to_numeric(frame["z_absent_score"], errors="coerce")
        if not np.isfinite(scores).all():
            raise SealedFigureInputError("Prediction scores contain missing or nonfinite values")
        if not frame["detection_eligible"].astype(bool).all():
            raise SealedFigureInputError("Figure binding includes detection-ineligible records")
        forced = frame["pred_label"].fillna("").astype(str)
        if forced.eq("").any():
            raise SealedFigureInputError("Prediction forced labels are incomplete")
        result = frame.copy()
        result["cell_id"] = result["cell_id"].astype(str)
        result["weak_support_score"] = scores
        result["forced_label"] = forced
        result["prediction_artifact"] = artifact
        result["prediction_sha256"] = artifact_sha256
        return result

    def mouse_summary(self, method: str, *, evaluation_scope: str) -> pd.DataFrame:
        if method not in FIGURE_METHODS["mouse_spleen"]:
            raise SealedFigureInputError(f"Unsupported mouse figure method: {method}")
        frame = self._read_csv("metrics/mouse_summary_metrics.csv")
        selected = frame.loc[
            frame["method"].eq(method)
            & frame["evaluation_scope"].eq(evaluation_scope)
        ].copy()
        if selected.empty or selected["metric"].duplicated().any():
            raise SealedFigureInputError("Mouse summary binding is missing or duplicated")
        values = pd.to_numeric(selected["value"], errors="coerce")
        bounds = selected[["ci_low", "ci_high"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values).all() or not np.isfinite(bounds.to_numpy()).all():
            raise SealedFigureInputError("Mouse summary contains nonfinite estimates")
        return selected.sort_values("metric", ignore_index=True)

    def mouse_bootstrap(self, method: str, *, evaluation_scope: str) -> pd.DataFrame:
        if method not in FIGURE_METHODS["mouse_spleen"]:
            raise SealedFigureInputError(f"Unsupported mouse figure method: {method}")
        frame = self._read_csv("metrics/mouse_bootstrap_metrics.csv")
        selected = frame.loc[
            frame["method"].eq(method)
            & frame["evaluation_scope"].eq(evaluation_scope)
        ].copy()
        if selected.empty or selected.duplicated(["replicate", "metric"]).any():
            raise SealedFigureInputError("Mouse bootstrap binding is missing or duplicated")
        values = pd.to_numeric(selected["value"], errors="coerce")
        if not np.isfinite(values).all():
            raise SealedFigureInputError("Mouse bootstrap contains nonfinite values")
        return selected.sort_values(["replicate", "metric"], ignore_index=True)
