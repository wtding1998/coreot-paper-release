from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from coreot.results.grid_common import (
    ResultsGridError,
    RunDescriptor,
    _read_yaml,
    build_detection_by_run,
    build_shared_label_transfer_by_run,
)
from coreot.results.sweep_common import (
    _failed_run_ids,
    _partition_completed_sweep_runs,
)


def parse_tau_from_transport_config(run_config_dir: Path) -> float:
    transport_path = run_config_dir / "transport.yaml"
    config = _read_yaml(transport_path)
    methods = config.get("methods", ())
    for method in methods:
        if not isinstance(method, dict):
            continue
        name = method.get("name")
        if name not in ("uniform_uot", "coreot_constant_tau"):
            continue
        tau_source = float(method["tau_source"])
        tau_target = float(method.get("tau_target", tau_source))
        if tau_target != tau_source:
            raise ResultsGridError(
                f"tau_target ({tau_target}) != tau_source ({tau_source}) "
                f"in {transport_path} for method {name}"
            )
        return tau_source
    raise ResultsGridError(
        f"No swept method with tau_source found in {transport_path}"
    )


def _build_tau_map(
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
) -> dict[str, float]:
    return {
        descriptor.run_id: parse_tau_from_transport_config(
            grid_dir / descriptor.run_id
        )
        for descriptor in descriptors
    }


def build_tau_detection_by_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    completed, failures = _partition_completed_sweep_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
    )
    failed = _failed_run_ids(failures)
    if failed:
        print(
            f"Skipping {len(failed)} failed run(s): {', '.join(failed)}",
            file=sys.stderr,
        )
    if not completed:
        return pd.DataFrame()
    frame = build_detection_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=methods,
    )
    if frame.empty:
        return frame
    frame["tau"] = frame["run_id"].map(_build_tau_map(grid_dir, completed))
    frame["sensitivity_role"] = "swept"
    return frame


def build_tau_shared_label_transfer_by_run(
    *,
    runs_root: Path,
    grid_dir: Path,
    descriptors: tuple[RunDescriptor, ...],
    candidate_set: str,
    condition: str,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    completed, failures = _partition_completed_sweep_runs(
        runs_root=runs_root,
        descriptors=descriptors,
        candidate_set=candidate_set,
        condition=condition,
    )
    failed = _failed_run_ids(failures)
    if failed:
        print(
            f"Skipping {len(failed)} failed run(s): {', '.join(failed)}",
            file=sys.stderr,
        )
    if not completed:
        return pd.DataFrame()
    frame = build_shared_label_transfer_by_run(
        runs_root=runs_root,
        descriptors=completed,
        candidate_set=candidate_set,
        condition=condition,
        methods=methods,
    )
    if frame.empty:
        return frame
    frame["tau"] = frame["run_id"].map(_build_tau_map(grid_dir, completed))
    frame["sensitivity_role"] = "swept"
    return frame
