from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from PIL import Image
import pytest
import yaml

from coreot.results.hiha_figure2 import (
    ENDPOINTS,
    PANEL_A_SIZE_INCHES,
    PANEL_A_REFERENCE_OMITTED_LABEL,
    PANEL_A_RESTORED_LABEL,
    PANEL_B_METHODS,
    PANEL_B_POSITIONS,
    METHOD_COLORS,
    PANEL_B_SIZE_INCHES,
    PANEL_B_XLABEL,
    PANEL_C_COMPARATORS,
    PANEL_C_XLABEL,
    PANEL_C_REFERENCE,
    PANEL_C_SIZE_INCHES,
    PANEL_C_GROUPS,
    PANEL_D_LIM,
    PANEL_D_METRICS,
    PANEL_D_METHODS,
    PANEL_D_SIZE_INCHES,
    PANEL_D_XLABEL,
    PANEL_E_MAPS,
    PANEL_E_LEGEND_LABELS,
    PANEL_E_METHODS,
    PANEL_E_SIZE_INCHES,
    PANEL_F_ENDPOINT_LABEL_Y,
    PANEL_F_HELD_OUT_COLOR,
    PANEL_F_LEGEND_NCOLS,
    PANEL_F_LABEL_COLORS,
    PANEL_F_MAPS,
    PANEL_F_HELD_OUT_POINT_SIZE,
    PANEL_F_REPRESENTED_POINT_SIZE,
    PANEL_F_SIZE_INCHES,
    PANEL_RASTER_DPI,
    REPRESENTATIVE_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
    HIHAFigure2Error,
    _draw_panel_a,
    _draw_panel_c_rescue,
    _draw_panel_e_umap,
    generate_panel_a,
    generate_panel_b,
    generate_panel_c,
    generate_panel_d,
)
from coreot.results.hiha_figure2_composite import (
    MAIN_FIGURE_MIN_FONT_SIZE,
    MAIN_FIGURE_PANEL_SET,
    MAIN_FIGURE_SIZE_INCHES,
    PANEL_A_CONDITION_FONT_SIZE,
    PANEL_A_MIN_FONT_SIZE,
    PANEL_A_SMALL_TEXT_GIDS,
    PANEL_B_ENDPOINT_FONT_SIZE,
    PANEL_B_ENDPOINT_GIDS,
    _draw_main_figure,
    _draw_panel_figure,
    _draw_panel_a_narrow,
    _draw_panel_b_restoration,
    _draw_panel_c_metric_bars,
    _draw_panel_d_metric_bars,
    _panel_crop_bbox_inches,
    _read_composite_sources,
    _rescue_source,
    generate_main_figure,
    generate_panel_images,
)
from coreot.results.hiha import (
    SCORES,
    build_hiha_detection_by_run,
    build_hiha_rescue,
)
from coreot.results.main_figure_external_baselines import (
    prepare_hiha_candidate_sources,
)

HIHA_AUTHORITATIVE_FIXTURE_RUNS = Path("tests/fixtures/hiha_review_v3_runs")


def test_figure2_renderer_uses_noninteractive_backend() -> None:
    environment = {**os.environ, "MPLBACKEND": "macosx"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import coreot.results.hiha_figure2; "
                "import matplotlib; print(matplotlib.get_backend())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip().casefold() == "agg"


def _write_overview(path: Path) -> None:
    pd.DataFrame(
        [
            {"characteristic": "Number of subjects", "value": "108"},
            {
                "characteristic": "Donor split",
                "value": (
                    "80% reference / 20% query, donor-aware (subject-level); "
                    "at least 30 query cells"
                ),
            },
            {
                "characteristic": "Main held-out labels",
                "value": "HLA-DRhi cDC2, ISG+ cDC2",
            },
        ]
    ).to_csv(path, index=False)


def test_detection_aggregation_preserves_replacement_score_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = SimpleNamespace(
        run_id="evaluation-run",
        held_out_label="HLA-DRhi cDC2",
        seed=1,
    )
    replacement = SimpleNamespace(
        run_id="uniform-score-run",
        held_out_label="HLA-DRhi cDC2",
        seed=1,
    )
    cell_ids = ["q1", "q2", "q3", "q4"]
    truth = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "true_label": ["held", "held", "shared", "shared"],
            "is_absent_state": [True, True, False, False],
            "is_shared_state": [False, False, True, True],
        }
    )
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "AIFI_L2": ["cDC2"] * 4,
            "AIFI_L3": ["held", "held", "shared", "shared"],
        }
    )
    score_rows = []
    for method, score_names in SCORES.items():
        for index, cell_id in enumerate(cell_ids):
            row = {
                "cell_id": cell_id,
                "method": method,
                "candidate_set": (
                    "external_reference_mapping"
                    if method
                    in {
                        "seurat_anchor",
                        "singleR",
                        "celltypist_l3",
                        "scmap_cell",
                        "scmap_cluster",
                        "chetah",
                    }
                    else "hiha_harmony30_k100"
                ),
            }
            for score_name in score_names:
                row[score_name] = float(4 - index)
            score_rows.append(row)
    score_frame = pd.DataFrame(score_rows)
    replacement_scores = score_frame.loc[score_frame["method"].eq("uniform_uot")].drop(
        columns="candidate_set"
    )

    monkeypatch.setattr(
        "coreot.results.hiha.discover_run_descriptors",
        lambda grid_dir: [replacement] if Path(grid_dir).name == "replacement" else [evaluation],
    )
    monkeypatch.setattr("coreot.results.hiha._input_path", lambda *_: Path("input.h5ad"))
    monkeypatch.setattr("coreot.results.hiha._metadata", lambda *_: metadata.copy())
    monkeypatch.setattr("coreot.results.hiha._truth", lambda *_: truth.copy())
    monkeypatch.setattr("coreot.results.hiha._method_scores", lambda *_: score_frame.copy())
    monkeypatch.setattr(
        "coreot.results.hiha._scores", lambda *_args, **_kwargs: replacement_scores.copy()
    )

    result = build_hiha_detection_by_run(
        runs_root=Path("runs"),
        grid_dir=Path("selected"),
        method_grid_dirs={"uniform_uot": Path("replacement")},
    )

    uniform = result.loc[result["method"].eq("uniform_uot")]
    assert uniform["run_id"].eq("evaluation-run").all()
    assert uniform["evaluation_run_id"].eq("evaluation-run").all()
    assert uniform["score_run_id"].eq("uniform-score-run").all()
    assert uniform["candidate_set"].eq("hiha_harmony30_k100").all()
    coreot = result.loc[result["method"].eq("coreot_full")]
    assert coreot["score_run_id"].eq("evaluation-run").all()


def test_rescue_aggregation_uses_replacement_scores_and_preserves_run_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = SimpleNamespace(
        run_id="evaluation-run",
        held_out_label="HLA-DRhi cDC2",
        seed=1,
    )
    replacement = SimpleNamespace(
        run_id="uniform-score-run",
        held_out_label="HLA-DRhi cDC2",
        seed=1,
    )
    cell_ids = ["q1", "q2", "q3", "q4"]
    truth = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "is_absent_state": [True, True, False, False],
            "is_shared_state": [False, False, True, True],
        }
    )
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "AIFI_L2": ["cDC2"] * 4,
        }
    )

    def scores(run_root: Path, condition: str, _candidate_set: str) -> pd.DataFrame:
        replacement_run = run_root.name == "uniform-score-run"
        if replacement_run:
            methods = ("uniform_uot",)
            base = 0.60 if condition == "incomplete_reference" else 0.20
        else:
            methods = ("coreot_full", "uniform_uot")
            base = 0.40 if condition == "incomplete_reference" else 0.10
        return pd.DataFrame(
            [
                {
                    "cell_id": cell_id,
                    "method": method,
                    "u": base + 0.01 * index,
                }
                for method in methods
                for index, cell_id in enumerate(cell_ids)
            ]
        )

    monkeypatch.setattr(
        "coreot.results.hiha.discover_run_descriptors",
        lambda grid_dir: [replacement] if Path(grid_dir).name == "replacement" else [evaluation],
    )
    monkeypatch.setattr("coreot.results.hiha._input_path", lambda *_: Path("input.h5ad"))
    monkeypatch.setattr("coreot.results.hiha._metadata", lambda *_: metadata.copy())
    monkeypatch.setattr("coreot.results.hiha._truth", lambda *_: truth.copy())
    monkeypatch.setattr("coreot.results.hiha._scores", scores)

    cells, summary = build_hiha_rescue(
        runs_root=Path("runs"),
        grid_dir=Path("selected"),
        method_grid_dirs={"uniform_uot": Path("replacement")},
    )

    uniform_cells = cells.loc[cells["method"].eq("uniform_uot")]
    assert uniform_cells["run_id"].eq("evaluation-run").all()
    assert uniform_cells["evaluation_run_id"].eq("evaluation-run").all()
    assert uniform_cells["score_run_id"].eq("uniform-score-run").all()
    assert uniform_cells["u_ablated"].to_list() == pytest.approx([0.60, 0.61, 0.62, 0.63])
    assert uniform_cells["u_full"].to_list() == pytest.approx([0.20, 0.21, 0.22, 0.23])
    coreot_cells = cells.loc[cells["method"].eq("coreot_full")]
    assert coreot_cells["score_run_id"].eq("evaluation-run").all()
    uniform_summary = summary.loc[summary["method"].eq("uniform_uot")]
    assert uniform_summary["evaluation_run_id"].eq("evaluation-run").all()
    assert uniform_summary["score_run_id"].eq("uniform-score-run").all()


def _write_detection(path: Path, *, omit_last: bool = False) -> None:
    rows = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            prevalence = 0.24 if endpoint_index == 0 else 0.12
            for method_index, (method, score, _) in enumerate(PANEL_B_METHODS):
                evaluation_run_id = f"{endpoint_index}-{seed}"
                rows.append(
                    {
                        "run_id": evaluation_run_id,
                        "evaluation_run_id": evaluation_run_id,
                        "score_run_id": (
                            f"{evaluation_run_id}-tau05-uniform"
                            if method == "uniform_uot"
                            else evaluation_run_id
                        ),
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "score": score,
                        "auprc": 0.75 - 0.12 * method_index + 0.005 * seed,
                        "auprc_baseline": prevalence,
                    }
                )
    if omit_last:
        rows.pop()
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_panel_c_detection(path: Path, *, omit_last: bool = False) -> None:
    rows = []
    selected_methods = (PANEL_C_REFERENCE, *PANEL_C_COMPARATORS)
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            full_ap = 0.72 + 0.01 * seed
            deltas = {
                "coreot_full": 0.0,
                "coreot_match_only": 0.08 if endpoint_index == 0 else -0.02,
                "uniform_uot": 0.18 if endpoint_index == 0 else 0.08,
                "prior_only": 0.51 if endpoint_index == 0 else 0.58,
            }
            for method, score, _ in selected_methods:
                rows.append(
                    {
                        "run_id": f"{endpoint_index}-{seed}",
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "score": score,
                        "auprc": full_ap - deltas[method],
                    }
                )
    if omit_last:
        rows.pop()
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_panel_f_label_transfer(path: Path, *, omit_last: bool = False) -> None:
    rows = []
    method_means = {
        "coreot_full": 0.955,
        "uniform_uot": 0.948,
        "scdot": 0.938,
        "tacco_ot": 0.928,
        "seurat_anchor": 0.978,
        "scmap_cluster": 0.932,
        "chetah": 0.942,
    }
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            for method, score, _ in PANEL_D_METHODS:
                rows.append(
                    {
                        "run_id": f"panel-f-{endpoint_index}-{seed}",
                        "held_out_label": endpoint,
                        "seed": seed,
                        "condition_id": "incomplete_reference",
                        "candidate_set": (
                            "hiha_harmony30_k100"
                            if method in {"coreot_full", "uniform_uot"}
                            else "external_reference_mapping"
                        ),
                        "method": method,
                        "score": score,
                        "n_shared": 100 + seed,
                        "n_forced_labeled": 100 + seed,
                        "forced_macro_f1": (
                            method_means[method] - 0.008 * endpoint_index + 0.001 * (seed - 3)
                        ),
                        "forced_accuracy": (
                            method_means[method] - 0.012 * endpoint_index + 0.0015 * (seed - 3)
                        ),
                    }
                )
    if omit_last:
        rows.pop()
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_main_figure_sources(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "field": "number_of_subjects",
                "value": "108",
                "source_path": "overview.csv",
            },
            {
                "field": "donor_split",
                "value": "80% reference / 20% query, subject-level",
                "source_path": "overview.csv",
            },
            {
                "field": "displayed_endpoints",
                "value": "HLA-DRhi cDC2 | ISG+ cDC2",
                "source_path": "overview.csv",
            },
            {
                "field": "primary_evaluation",
                "value": "within-cDC2 held-out-state ranking",
                "source_path": "adr.md",
            },
        ]
    ).to_csv(source_root / "panel_a_design.csv", index=False)

    panel_b_rows = []
    panel_c_rows = []
    panel_c_deficit_rows = []
    panel_c_restoration_rows = []
    panel_e_umap_rows = []
    panel_d_label_rows = []
    panel_f_rows = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            for method_index, (method, score, display) in enumerate(PANEL_B_METHODS):
                panel_b_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "method_display": display,
                        "score": score,
                        "average_precision": (
                            0.76 - 0.11 * method_index - 0.03 * endpoint_index + 0.002 * seed
                        ),
                        "auroc": (
                            0.91 - 0.05 * method_index - 0.02 * endpoint_index + 0.001 * seed
                        ),
                        "positive_prevalence": 0.24 - 0.11 * endpoint_index,
                    }
                )
            for comparator_index, (method, score, display) in enumerate(PANEL_C_COMPARATORS):
                panel_c_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": seed,
                        "comparator_method": method,
                        "comparator_display": display,
                        "comparator_score": score,
                        "delta_average_precision": (
                            -0.02 + 0.22 * comparator_index + 0.03 * endpoint_index + 0.001 * seed
                        ),
                    }
                )
            for group_index, (group, _) in enumerate(PANEL_C_GROUPS):
                ablated = 0.19 - 0.05 * group_index - 0.02 * endpoint_index
                full = 0.06 - 0.005 * endpoint_index
                panel_c_deficit_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": seed,
                        "truth_group": group,
                        "median_u_ablated": ablated + 0.002 * seed,
                        "median_u_full": full + 0.001 * seed,
                    }
                )
            panel_c_restoration_rows.append(
                {
                    "held_out_label": endpoint,
                    "seed": seed,
                    "restoration_specificity": (0.07 + 0.02 * endpoint_index + 0.001 * seed),
                    "median_restored_state_probability": (
                        0.75 + 0.18 * endpoint_index + 0.002 * seed
                    ),
                }
            )
            for method, score, display in PANEL_D_METHODS:
                method_index = [item[0] for item in PANEL_D_METHODS].index(method)
                panel_d_label_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "method_display": display,
                        "score": score,
                        "forced_macro_f1": (
                            0.98 - 0.01 * method_index - 0.015 * endpoint_index + 0.001 * seed
                        ),
                        "forced_accuracy": (
                            0.975 - 0.008 * method_index - 0.012 * endpoint_index + 0.001 * seed
                        ),
                    }
                )
        for cell_index in range(24):
            for map_id, score, display in (
                ("truth", "evaluation_truth", "Held-out truth"),
                *PANEL_B_METHODS,
            ):
                panel_e_umap_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": REPRESENTATIVE_SEED,
                        "map_id": map_id,
                        "map_display": display,
                        "score_name": score,
                        "cell_id": f"{endpoint_index}-{cell_index}",
                        "is_held_out_truth": cell_index < 6,
                        "is_within_cdc2": cell_index < 18,
                        "is_selected": (cell_index < 6 if map_id == "truth" else cell_index < 6),
                        "umap_1": float(cell_index % 6 + 8 * (cell_index >= 12)),
                        "umap_2": float(cell_index // 6 + endpoint_index),
                    }
                )
            true_label = (
                endpoint if cell_index < 6 else ("CD14+ cDC2" if cell_index < 18 else "pDC")
            )
            for map_id, display in PANEL_F_MAPS:
                represented = cell_index >= 6
                panel_f_rows.append(
                    {
                        "held_out_label": endpoint,
                        "seed": REPRESENTATIVE_SEED,
                        "cell_id": f"{endpoint_index}-{cell_index}",
                        "umap_1": float(cell_index % 6 + 8 * (cell_index >= 12)),
                        "umap_2": float(cell_index // 6 + endpoint_index),
                        "true_label": true_label,
                        "is_represented_state": represented,
                        "is_held_out_state": not represented,
                        "map_id": map_id,
                        "map_display": display,
                        "displayed_assignment": (
                            true_label if represented else "Held-out state (not evaluated)"
                        ),
                    }
                )
    pd.DataFrame(panel_b_rows).to_csv(
        source_root / "panel_b_detection.csv",
        index=False,
    )
    pd.DataFrame(panel_c_rows).to_csv(
        source_root / "panel_c_controls.csv",
        index=False,
    )
    pd.DataFrame(panel_c_deficit_rows).to_csv(
        source_root / "panel_c_deficit_by_seed.csv",
        index=False,
    )
    pd.DataFrame(panel_c_restoration_rows).to_csv(
        source_root / "panel_c_restoration_by_seed.csv",
        index=False,
    )
    pd.DataFrame(panel_e_umap_rows).to_csv(
        source_root / "panel_e_umap_cells.csv",
        index=False,
    )
    pd.DataFrame(panel_d_label_rows).to_csv(
        source_root / "panel_d_label_transfer.csv",
        index=False,
    )
    pd.DataFrame(panel_f_rows).to_csv(
        source_root / "panel_f_label_assignment_cells.csv",
        index=False,
    )


def _write_panel_d_artifacts(runs_root: Path, detection_path: Path) -> None:
    detection_rows = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            run_id = f"panel_d_{endpoint_index}_{seed}"
            run_root = runs_root / run_id
            cell_ids = [f"{run_id}_q{index}" for index in range(6)]
            true_labels = [endpoint, endpoint, "CD14+ cDC2", "CD14+ cDC2", "pDC", "pDC"]
            for condition in ("incomplete_reference", "full_reference_control"):
                incomplete = condition == "incomplete_reference"
                truth = pd.DataFrame(
                    {
                        "cell_id": cell_ids,
                        "true_label": true_labels,
                        "removed_state": endpoint,
                        "is_absent_state": (
                            [True, True, False, False, False, False] if incomplete else [False] * 6
                        ),
                        "is_shared_state": (
                            [False, False, True, True, True, True] if incomplete else [True] * 6
                        ),
                    }
                )
                truth_root = run_root / "benchmark" / condition / "evaluation_truth"
                truth_root.mkdir(parents=True, exist_ok=True)
                truth.to_csv(truth_root / "query_truth.csv", index=False)

                label_root = run_root / "benchmark" / condition / "model_visible"
                label_root.mkdir(parents=True, exist_ok=True)
                target_labels = (
                    [endpoint, "CD14+ cDC2", "pDC"]
                    if condition == "full_reference_control"
                    else ["CD14+ cDC2", "pDC"]
                )
                pd.DataFrame(
                    {
                        "cell_id": [
                            f"{run_id}_{condition}_r{index}" for index in range(len(target_labels))
                        ],
                        "target_label": target_labels,
                        "broad_label": [
                            "cDC2" if "cDC2" in label else "pDC" for label in target_labels
                        ],
                    }
                ).to_csv(label_root / "target_labels.csv", index=False)

                if condition == "incomplete_reference":
                    u = np.array([0.20, 0.21, 0.10, 0.11, 0.08, 0.09])
                else:
                    u = np.array([0.05, 0.06, 0.06, 0.07, 0.08, 0.09])
                u = u + seed * 0.001
                a = np.full(6, 1 / 6)
                score_root = run_root / "transport" / condition / "hiha_harmony30_k100/coreot_full"
                score_root.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "cell_id": cell_ids,
                        "a": a,
                        "a_hat": a * (1 - u),
                        "u": u,
                    }
                ).to_parquet(score_root / "cell_transport_scores.parquet", index=False)

            probabilities = np.array(
                [
                    [0.80, 0.15, 0.05],
                    [0.80, 0.15, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.01, 0.09, 0.90],
                    [0.01, 0.09, 0.90],
                ]
            )
            np.savez_compressed(
                run_root / "transport/full_reference_control/hiha_harmony30_k100/"
                "coreot_full/label_probabilities.npz",
                cell_ids=np.asarray(cell_ids, dtype=object),
                labels=np.asarray([endpoint, "CD14+ cDC2", "pDC"], dtype=object),
                probabilities=probabilities,
            )
            detection_rows.append(
                {
                    "run_id": run_id,
                    "held_out_label": endpoint,
                    "seed": seed,
                    "method": "coreot_full",
                    "score": "u",
                }
            )
    pd.DataFrame(detection_rows).to_csv(detection_path, index=False)


def _write_panel_e_artifacts(runs_root: Path, detection_path: Path) -> None:
    detection_rows = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            run_id = f"panel_e_{endpoint_index}_{seed}"
            uniform_run_id = f"{run_id}_tau05_uniform"
            for method, score, _ in PANEL_E_METHODS:
                detection_rows.append(
                    {
                        "run_id": (uniform_run_id if method == "uniform_uot" else run_id),
                        "held_out_label": endpoint,
                        "seed": seed,
                        "method": method,
                        "score": score,
                    }
                )
            if seed != REPRESENTATIVE_SEED:
                continue
            run_root = runs_root / run_id
            cell_ids = [f"{run_id}_q{index:02d}" for index in range(20)]
            true_labels = [endpoint] * 5 + ["CD14+ cDC2"] * 10 + ["pDC"] * 5
            truth_root = run_root / "benchmark/incomplete_reference/evaluation_truth"
            truth_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "true_label": true_labels,
                    "removed_state": endpoint,
                    "is_absent_state": [True] * 5 + [False] * 15,
                    "is_shared_state": [False] * 5 + [True] * 15,
                }
            ).to_csv(truth_root / "query_truth.csv", index=False)

            incomplete_label_root = run_root / "benchmark/incomplete_reference/model_visible"
            incomplete_label_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "cell_id": [f"{run_id}_incomplete_reference_r{index}" for index in range(2)],
                    "target_label": ["CD14+ cDC2", "pDC"],
                    "broad_label": ["cDC2", "pDC"],
                }
            ).to_csv(
                incomplete_label_root / "target_labels.csv",
                index=False,
            )

            label_root = run_root / "benchmark/full_reference_control/model_visible"
            label_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "cell_id": [f"{run_id}_full_reference_control_r{index}" for index in range(3)],
                    "target_label": [endpoint, "CD14+ cDC2", "pDC"],
                    "broad_label": ["cDC2", "cDC2", "pDC"],
                }
            ).to_csv(label_root / "target_labels.csv", index=False)

            embedding_root = run_root / "embeddings/incomplete_reference/hiha_harmony30"
            embedding_root.mkdir(parents=True, exist_ok=True)
            reference_ids = [f"{run_id}_r{index}" for index in range(5)]
            pd.DataFrame(
                {
                    "cell_id": [*cell_ids, *reference_ids],
                    "domain": ["query"] * len(cell_ids) + ["reference"] * 5,
                }
            ).to_csv(embedding_root / "embedding_cells.csv", index=False)
            random = np.random.default_rng(10 + endpoint_index)
            np.save(embedding_root / "embedding.npy", random.normal(size=(25, 30)))

            base_score = np.concatenate(
                [
                    np.linspace(0.95, 0.75, 5),
                    np.linspace(0.70, 0.10, 10),
                    np.linspace(0.99, 0.80, 5),
                ]
            )
            internal_rows = []
            for method_index, method in enumerate(("coreot_full", "prior_only")):
                for cell_id, value in zip(cell_ids, base_score, strict=True):
                    internal_rows.append(
                        {
                            "cell_id": cell_id,
                            "condition_id": "incomplete_reference",
                            "method": method,
                            "u": value - 0.02 * method_index,
                            "prior_risk": value - 0.02 * method_index,
                            "forced_label": (
                                "CD14+ cDC2" if cell_id not in cell_ids[-5:] else "pDC"
                            ),
                        }
                    )
            internal_root = run_root / "scoring/incomplete_reference/hiha_harmony30_k100"
            internal_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(internal_rows).to_parquet(
                internal_root / "cell_scores.parquet", index=False
            )

            uniform_root = (
                runs_root / uniform_run_id / "scoring/incomplete_reference/hiha_harmony30_k100"
            )
            uniform_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "condition_id": "incomplete_reference",
                    "method": "uniform_uot",
                    "u": np.roll(base_score, 2),
                    "forced_label": [
                        *["CD14+ cDC2"] * 15,
                        *["pDC"] * 5,
                    ],
                }
            ).to_parquet(uniform_root / "cell_scores.parquet", index=False)

            external_rows = []
            for method_index, method in enumerate(("seurat_anchor", "scmap_cluster", "chetah")):
                method_scores = np.roll(base_score, method_index + 3)
                for cell_id, value in zip(cell_ids, method_scores, strict=True):
                    external_rows.append(
                        {
                            "cell_id": cell_id,
                            "condition_id": "incomplete_reference",
                            "method": method,
                            "u": value,
                            "forced_label": (
                                "CD14+ cDC2" if cell_id not in cell_ids[-5:] else "pDC"
                            ),
                        }
                    )
            external_root = run_root / "scoring/incomplete_reference/external_reference_mapping"
            external_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(external_rows).to_parquet(
                external_root / "cell_scores.parquet", index=False
            )
    pd.DataFrame(detection_rows).to_csv(detection_path, index=False)


def _write_panel_f_artifacts(
    runs_root: Path,
    label_transfer_path: Path,
) -> None:
    detection_path = label_transfer_path.with_name("detection.csv")
    _write_panel_e_artifacts(runs_root, detection_path)
    rows = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for seed in range(1, 6):
            selected_run_id = f"panel_e_{endpoint_index}_{seed}"
            for method, score, _ in PANEL_D_METHODS:
                rows.append(
                    {
                        "run_id": (
                            f"{selected_run_id}_tau05_uniform"
                            if method == "uniform_uot"
                            else selected_run_id
                        ),
                        "held_out_label": endpoint,
                        "seed": seed,
                        "condition_id": "incomplete_reference",
                        "candidate_set": (
                            "hiha_harmony30_k100"
                            if method in {"coreot_full", "uniform_uot"}
                            else "external_reference_mapping"
                        ),
                        "method": method,
                        "score": score,
                        "n_shared": 15,
                        "n_forced_labeled": 15,
                        "forced_macro_f1": 1.0,
                        "forced_accuracy": 1.0,
                    }
                )
    pd.DataFrame(rows).to_csv(label_transfer_path, index=False)


def test_panel_a_is_a_horizontal_publication_strip() -> None:
    assert 6.9 <= PANEL_A_SIZE_INCHES[0] <= 7.1
    assert 1.6 <= PANEL_A_SIZE_INCHES[1] <= 1.7
    assert PANEL_A_SIZE_INCHES[0] / PANEL_A_SIZE_INCHES[1] >= 4.0
    assert PANEL_RASTER_DPI >= 350


def test_panel_b_uses_ap_terminology_and_double_column_width() -> None:
    assert PANEL_B_XLABEL == "AP for omitted-state ranking"
    assert 6.9 <= PANEL_B_SIZE_INCHES[0] <= 7.1
    assert 2.7 <= PANEL_B_SIZE_INCHES[1] <= 2.9
    assert [(method, score) for method, score, _ in PANEL_B_METHODS] == [
        ("coreot_full", "u"),
        ("scdot", "z_absent_score"),
        ("tacco_ot", "z_absent_score"),
        ("seurat_anchor", "u"),
        ("scmap_cluster", "u"),
        ("chetah", "u"),
    ]
    assert PANEL_B_METHODS[0] == ("coreot_full", "u", "CoRe-OT")
    assert PANEL_E_METHODS[0] == ("coreot_full", "u", "CoRe-OT")
    assert len(PANEL_B_POSITIONS) == len(PANEL_B_METHODS)
    assert PANEL_B_POSITIONS[3] - PANEL_B_POSITIONS[2] > 1
    assert {method: METHOD_COLORS[method] for method, _, _ in PANEL_B_METHODS} == {
        "coreot_full": "#0072B2",
        "scdot": "#D55E00",
        "tacco_ot": "#009E73",
        "seurat_anchor": "#CC79A7",
        "scmap_cluster": "#E69F00",
        "chetah": "#6A3D9A",
    }


def test_panel_c_uses_rescue_contrast_and_double_column_width() -> None:
    assert PANEL_C_XLABEL == "Decrease in median query-marginal deficit"
    assert 6.9 <= PANEL_C_SIZE_INCHES[0] <= 7.1
    assert 3.1 <= PANEL_C_SIZE_INCHES[1] <= 3.3


def test_panel_e_uses_prespecified_fixed_query_umap_contract() -> None:
    assert REPRESENTATIVE_SEED == 1
    assert UMAP_N_NEIGHBORS == 15
    assert UMAP_MIN_DIST == 0.3
    assert UMAP_RANDOM_STATE == 1
    assert 6.9 <= PANEL_E_SIZE_INCHES[0] <= 7.1
    assert 4.2 <= PANEL_E_SIZE_INCHES[1] <= 4.3


def test_panel_d_uses_two_metric_represented_state_contract() -> None:
    assert PANEL_D_XLABEL == "Metric value"
    assert PANEL_D_LIM == (0.80, 1.0)
    assert PANEL_D_METHODS[0] == ("coreot_full", "u", "CoRe-OT")
    assert PANEL_D_METRICS == (
        ("forced_macro_f1", "Forced macro-F1", "#4C78A8"),
        ("forced_accuracy", "Forced accuracy", "#F2A65A"),
    )
    assert 6.9 <= PANEL_D_SIZE_INCHES[0] <= 7.1
    assert 2.7 <= PANEL_D_SIZE_INCHES[1] <= 2.9


def test_panel_f_uses_figure4_label_assignment_contract() -> None:
    assert 6.9 <= PANEL_F_SIZE_INCHES[0] <= 7.1
    assert 3.1 <= PANEL_F_SIZE_INCHES[1] <= 3.3
    assert [map_id for map_id, _ in PANEL_F_MAPS] == [
        "truth",
        "coreot_full",
        "scdot",
        "tacco_ot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    ]
    assert dict(PANEL_F_MAPS)["coreot_full"] == "CoRe-OT"
    assert dict(PANEL_F_MAPS)["seurat_anchor"] == "Seurat"
    assert PANEL_F_LABEL_COLORS == {
        "ASDC": "#B279A2",
        "CD14+ cDC2": "#F58518",
        "HLA-DRhi cDC2": "#4C78A8",
        "ISG+ cDC2": "#54A24B",
        "cDC1": "#E45756",
        "pDC": "#72B7B2",
    }
    assert PANEL_F_HELD_OUT_COLOR == "#D9D9D9"
    assert PANEL_F_REPRESENTED_POINT_SIZE == 1.5
    assert PANEL_F_HELD_OUT_POINT_SIZE == 2.5
    assert PANEL_F_ENDPOINT_LABEL_Y == (0.73, 0.33)
    assert PANEL_F_LEGEND_NCOLS == 7


def test_figure2_uses_concise_seurat_display_label() -> None:
    assert (
        dict((method, display) for method, _, display in PANEL_B_METHODS)["seurat_anchor"]
        == "Seurat"
    )
    assert (
        dict((method, display) for method, _, display in PANEL_D_METHODS)["seurat_anchor"]
        == "Seurat"
    )


def test_figure2_method_colors_match_mouse_spleen_panel_a() -> None:
    expected = {
        "coreot_full": "#0072B2",
        "scdot": "#D55E00",
        "tacco_ot": "#009E73",
        "seurat_anchor": "#CC79A7",
        "scmap_cluster": "#E69F00",
        "chetah": "#6A3D9A",
    }
    assert {method: METHOD_COLORS[method] for method in expected} == expected


def test_main_figure_uses_a_to_f_on_a_double_column_page() -> None:
    assert MAIN_FIGURE_PANEL_SET == ("A", "B", "C", "D", "E", "F")
    assert MAIN_FIGURE_MIN_FONT_SIZE >= 6.5
    assert 6.9 <= MAIN_FIGURE_SIZE_INCHES[0] <= 7.1
    assert 9.0 <= MAIN_FIGURE_SIZE_INCHES[1] <= 9.1


def _assert_condition_labels_have_padding(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = {PANEL_A_REFERENCE_OMITTED_LABEL, PANEL_A_RESTORED_LABEL}
    patches = [patch.get_window_extent(renderer) for patch in fig.findobj(match=FancyBboxPatch)]
    for label in labels:
        text = next(text for text in fig.findobj(match=Text) if text.get_text() == label)
        text_box = text.get_window_extent(renderer)
        enclosing = [
            patch
            for patch in patches
            if (
                text_box.x0 >= patch.x0
                and text_box.x1 <= patch.x1
                and text_box.y0 >= patch.y0
                and text_box.y1 <= patch.y1
            )
        ]
        assert enclosing, f"{label!r} is not contained by a reference box"
        padding = min(text_box.x0 - patch.x0 for patch in enclosing)
        padding = min(
            padding,
            min(text_box.x1 - patch.x1 for patch in enclosing) * -1,
            min(text_box.y0 - patch.y0 for patch in enclosing),
            min(text_box.y1 - patch.y1 for patch in enclosing) * -1,
        )
        assert padding >= 1.0


def test_panel_a_condition_labels_fit_standalone_and_composite_boxes() -> None:
    standalone = plt.figure(figsize=PANEL_A_SIZE_INCHES)
    standalone_axis = standalone.add_axes((0, 0, 1, 1))
    _draw_panel_a(standalone_axis, n_subjects="108")
    _assert_condition_labels_have_padding(standalone)

    composite = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES)
    _draw_panel_a_narrow(composite, {"number_of_subjects": "108"})
    _assert_condition_labels_have_padding(composite)
    composite_axis = composite.axes[0]
    boxes = [patch for patch in composite_axis.patches if isinstance(patch, FancyBboxPatch)]
    scenario_box, split_box, query_box, omitted_box, restored_box = boxes
    assert scenario_box.get_width() == pytest.approx(split_box.get_width())
    assert scenario_box.get_x() == pytest.approx(split_box.get_x())
    assert scenario_box.get_y() - (split_box.get_y() + split_box.get_height()) >= 0.06
    assert omitted_box.get_width() == pytest.approx(restored_box.get_width())
    assert omitted_box.get_height() == pytest.approx(restored_box.get_height())
    assert query_box.get_width() < omitted_box.get_width()
    assert query_box.get_height() > omitted_box.get_height()
    condition_text = {text.get_text(): text for text in composite.findobj(match=Text)}
    assert condition_text["Five donor splits"].get_fontsize() == pytest.approx(
        6.0
    )
    assert condition_text["Same query cells"].get_fontsize() == pytest.approx(
        5.6
    )
    assert condition_text[PANEL_A_REFERENCE_OMITTED_LABEL].get_fontsize() == pytest.approx(
        PANEL_A_CONDITION_FONT_SIZE
    )
    assert condition_text[PANEL_A_RESTORED_LABEL].get_fontsize() == pytest.approx(
        PANEL_A_CONDITION_FONT_SIZE
    )

    plt.close(standalone)
    plt.close(composite)


def test_standalone_panel_a_summary_border_and_text_are_contained() -> None:
    figure = plt.figure(figsize=PANEL_A_SIZE_INCHES)
    axis = figure.add_axes((0, 0, 1, 1))
    _draw_panel_a(axis, n_subjects="108")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_box = axis.get_window_extent(renderer)

    summary_text = next(
        text
        for text in figure.findobj(match=Text)
        if text.get_text().startswith("HIHA dendritic-cell benchmark")
    )
    summary_box = next(
        patch
        for patch in axis.patches
        if isinstance(patch, FancyBboxPatch) and patch.get_x() == pytest.approx(0.74)
    )
    summary_patch_box = summary_box.get_window_extent(renderer)
    assert axis_box.contains(summary_patch_box.x0, summary_patch_box.y0)
    assert axis_box.contains(summary_patch_box.x1, summary_patch_box.y1)

    text_box = summary_text.get_window_extent(renderer)
    assert summary_patch_box.contains(text_box.x0, text_box.y0)
    assert summary_patch_box.contains(text_box.x1, text_box.y1)
    assert all(
        not arrow.get_window_extent(renderer).overlaps(text_box)
        for arrow in figure.findobj(match=FancyArrowPatch)
    )
    plt.close(figure)


def test_live_figure2_text_uses_canonical_terminology(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    sources = _read_composite_sources(source_root)
    figure = _draw_main_figure(*sources)
    figure.canvas.draw()
    visible = "\n".join(
        text.get_text() for text in figure.findobj(match=Text) if text.get_text().strip()
    )

    assert PANEL_A_REFERENCE_OMITTED_LABEL in visible
    assert PANEL_A_RESTORED_LABEL in visible
    assert "AP" in visible
    assert "AUROC" in visible
    assert "Mean prevalence (AP)" in visible
    assert "Reference-omitted" in visible
    assert "Restored-state destination fraction" in visible
    assert "Method-specific top-ranked $N_+$ cells" in visible
    assert "Reference-omitted cells (not evaluated)" in visible
    assert "Label transfer among represented states" not in visible
    for retired in (
        "Incomplete-reference",
        "incomplete-reference",
        "Held-out state",
        "held-out-state",
        "Average precision (AP)",
        "conditional probability",
        "Restored-state weight",
    ):
        assert retired not in visible
    standard_text = [
        text
        for text in figure.findobj(match=Text)
        if text.get_gid() not in PANEL_A_SMALL_TEXT_GIDS
        and text.get_gid() not in PANEL_B_ENDPOINT_GIDS
    ]
    assert min(text.get_fontsize() for text in standard_text) >= MAIN_FIGURE_MIN_FONT_SIZE
    plt.close(figure)


def test_composite_review_layout_contains_panel_a_and_e_text(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    figure = _draw_main_figure(*_read_composite_sources(source_root))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox

    panel_a_axis = next(
        axis for axis in figure.axes if not axis.axison and axis.get_position().x0 < 0.1
    )
    panel_a_boxes = [
        patch for patch in panel_a_axis.patches if isinstance(patch, FancyBboxPatch)
    ]
    assert len(panel_a_boxes) == 5
    visible = {
        text.get_text()
        for text in panel_a_axis.findobj(match=Text)
        if text.get_text().strip()
    }
    for label in (
        "Separate scenarios",
        "HLA-DRhi cDC2 | ISG+ cDC2",
        "Five donor splits",
        "Same query cells",
        "Evaluated state\nretained",
        "Same reference donors",
        PANEL_A_REFERENCE_OMITTED_LABEL,
        "Evaluated-state cells\nomitted",
        PANEL_A_RESTORED_LABEL,
        "Omitted-state cells\nrestored",
    ):
        assert label in visible
    assert not any("108" in label or "benchmark" in label for label in visible)

    panel_a_box = panel_a_axis.get_window_extent(renderer)
    for text in panel_a_axis.findobj(match=Text):
        if not text.get_text().strip():
            continue
        text_box = text.get_window_extent(renderer)
        assert panel_a_box.contains(text_box.x0, text_box.y0)
        assert panel_a_box.contains(text_box.x1, text_box.y1)

    arrows = [
        patch for patch in panel_a_axis.patches if isinstance(patch, FancyArrowPatch)
    ]
    assert len(arrows) == 6

    e_legend = next(
        legend
        for legend in figure.legends
        if [text.get_text() for text in legend.get_texts()] == list(PANEL_E_LEGEND_LABELS)
    )
    legend_box = e_legend.get_window_extent(renderer)
    assert figure_box.contains(legend_box.x0, legend_box.y0)
    assert figure_box.contains(legend_box.x1, legend_box.y1)
    plt.close(figure)


def test_composite_spatial_panels_share_aligned_enlarged_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    figure = _draw_main_figure(*_read_composite_sources(source_root))

    expected_rows = (0.360, 0.275, 0.145, 0.060)
    row_axes = {
        y_position: sorted(
            (
                axis
                for axis in figure.axes
                if np.isclose(
                    axis.get_position(original=True).y0,
                    y_position,
                )
            ),
            key=lambda axis: axis.get_position(original=True).x0,
        )
        for y_position in expected_rows
    }
    assert all(len(axes) == 7 for axes in row_axes.values())

    expected_x = np.linspace(0.070, 0.860, 7)
    for axes in row_axes.values():
        positions = [axis.get_position(original=True) for axis in axes]
        assert [position.x0 for position in positions] == pytest.approx(expected_x)
        assert [position.width for position in positions] == pytest.approx([0.128] * 7)
        assert [position.height for position in positions] == pytest.approx([0.075] * 7)

    labels = {
        text.get_gid(): text
        for text in figure.findobj(match=Text)
        if text.get_gid()
        in {
            "panel-e-endpoint-0",
            "panel-e-endpoint-1",
            "panel-f-endpoint-0",
            "panel-f-endpoint-1",
        }
    }
    assert set(labels) == {
        "panel-e-endpoint-0",
        "panel-e-endpoint-1",
        "panel-f-endpoint-0",
        "panel-f-endpoint-1",
    }
    expected_labels = {
        "panel-e-endpoint-0": ("HLA-DRhi cDC2", 0.3975),
        "panel-e-endpoint-1": ("ISG+ cDC2", 0.3125),
        "panel-f-endpoint-0": ("HLA-DRhi cDC2", 0.1825),
        "panel-f-endpoint-1": ("ISG+ cDC2", 0.0975),
    }
    for gid, (expected_text, expected_y) in expected_labels.items():
        label = labels[gid]
        assert label.get_text() == expected_text
        assert label.get_position() == pytest.approx((0.032, expected_y))
        assert label.get_rotation() == pytest.approx(270.0)
        assert label.get_fontweight() == "normal"
        assert label.get_horizontalalignment() == "center"
        assert label.get_verticalalignment() == "center"

    plt.close(figure)


def test_standalone_panel_c_export_contains_all_live_artists(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    outputs = generate_panel_images(
        letters=("C",),
        panel_root=tmp_path / "panels",
        source_root=source_root,
    )
    assert outputs == {"panel_c": tmp_path / "panels/panel_c.png"}
    for suffix in ("png", "pdf", "tiff"):
        assert (tmp_path / f"panels/panel_c.{suffix}").is_file()
    assert not (tmp_path / "panels/panel_c.svg").exists()

    figure = _draw_panel_figure(_read_composite_sources(source_root), "C")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    crop = _panel_crop_bbox_inches(figure, padding_inches=0.04)
    for text in figure.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer).transformed(figure.dpi_scale_trans.inverted())
        assert box.x0 > crop.x0
        assert box.y0 > crop.y0
        assert box.x1 < crop.x1
        assert box.y1 < crop.y1
    plt.close(figure)


def test_panel_b_active_renderers_preserve_control_adjusted_notation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    summary = pd.read_csv(source_root / "panel_c_deficit_by_seed.csv")
    restoration = pd.read_csv(source_root / "panel_c_restoration_by_seed.csv")
    pivot = summary.pivot(
        index=["held_out_label", "seed"],
        columns="truth_group",
        values=["median_u_ablated", "median_u_full"],
    )
    pivot.columns = ["_".join(column) for column in pivot.columns]
    standalone_specificity = pivot.reset_index().merge(
        restoration,
        on=["held_out_label", "seed"],
        validate="one_to_one",
    )
    standalone_specificity["held_out_delta_median_u"] = (
        standalone_specificity["median_u_ablated_held_out"]
        - standalone_specificity["median_u_full_held_out"]
    )
    standalone_specificity["control_delta_median_u"] = (
        standalone_specificity["median_u_ablated_represented_cdc2"]
        - standalone_specificity["median_u_full_represented_cdc2"]
    )

    standalone = _draw_panel_c_rescue(summary, standalone_specificity)
    composite = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES)
    _draw_panel_b_restoration(composite, summary, restoration)
    for figure in (standalone, composite):
        figure.canvas.draw()
        figure_box = figure.bbox
        renderer = figure.canvas.get_renderer()
        for item in figure.findobj(match=Text):
            if item.get_visible() and item.get_text():
                text_box = item.get_window_extent(renderer)
                assert figure_box.contains(text_box.x0, text_box.y0)
                assert figure_box.contains(text_box.x1, text_box.y1)
        text = "\n".join(
            item.get_text() for item in figure.findobj(match=Text) if item.get_text().strip()
        )
        assert r"\Delta^{\mathrm{CA}}" in text
        assert r"\Delta_s^{\mathrm{CA}}" not in text
        plt.close(figure)


def test_panel_e_legend_preserves_four_category_visual_key(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    cells = pd.read_csv(source_root / "panel_e_umap_cells.csv")
    figure = _draw_panel_e_umap(cells)
    figure.canvas.draw()

    assert {
        (endpoint, map_id)
        for endpoint, map_id in cells[["held_out_label", "map_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    } == {(endpoint, map_id) for endpoint in ENDPOINTS for map_id, _, _ in PANEL_E_MAPS}
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == list(PANEL_E_LEGEND_LABELS)
    assert not legend.get_frame().get_visible()
    assert all(
        handle.get_markeredgewidth() > 0
        for handle in legend.legend_handles
        if hasattr(handle, "get_markeredgewidth")
    )
    assert min(text.get_fontsize() for text in legend.get_texts()) >= 6.5

    composite = _draw_main_figure(*_read_composite_sources(source_root))
    composite.canvas.draw()
    matching_legends = [
        item
        for item in composite.legends
        if [text.get_text() for text in item.get_texts()] == list(PANEL_E_LEGEND_LABELS)
    ]
    assert len(matching_legends) == 1
    assert not matching_legends[0].get_frame().get_visible()
    assert matching_legends[0]._ncols == 4
    renderer = composite.canvas.get_renderer()
    legend_box = matching_legends[0].get_window_extent(renderer)
    assert composite.bbox.contains(legend_box.x0, legend_box.y0)
    assert composite.bbox.contains(legend_box.x1, legend_box.y1)

    plt.close(figure)
    plt.close(composite)


def test_panel_b_uses_merged_mean_bars_and_sample_sd_whiskers(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    sources = _read_composite_sources(source_root)
    figure = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES)
    _draw_panel_b_restoration(figure, sources[2], sources[3])

    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_xlim() == (0.0, 1.02)
    assert np.allclose(axis.get_xticks(), (0.0, 0.5, 1.0))
    assert axis.get_xlabel() == "Metric value"
    assert len(axis.patches) == len(ENDPOINTS) * 4
    assert not any(
        isinstance(collection, PathCollection) for collection in axis.collections
    )
    expected_rows = (
        "held_out_delta_median_u",
        "control_delta_median_u",
        "restoration_specificity",
        "median_restored_state_probability",
    )
    data = _rescue_source(sources[2], sources[3])
    patches = {patch.get_gid(): patch for patch in axis.patches}
    for endpoint in ENDPOINTS:
        endpoint_data = data.loc[data["held_out_label"].eq(endpoint)]
        for column in expected_rows:
            gid = f"panel-b-bar-{endpoint}-{column}".replace(" ", "_")
            assert patches[gid].get_width() == pytest.approx(
                endpoint_data[column].mean()
            )
    plt.close(figure)


def test_panel_b_endpoint_labels_are_right_rotated_and_smaller(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    sources = _read_composite_sources(source_root)
    figure = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES)
    _draw_panel_b_restoration(figure, sources[2], sources[3])
    figure.canvas.draw()

    endpoint_labels = [
        text
        for text in figure.findobj(match=Text)
        if text.get_gid()
        in {
            "panel-b-endpoint-label-hla-drhi-cdc2",
            "panel-b-endpoint-label-isg-cdc2",
        }
    ]
    assert {text.get_text() for text in endpoint_labels} == set(ENDPOINTS)
    assert all(text.get_horizontalalignment() == "center" for text in endpoint_labels)
    assert all(text.get_position()[0] == pytest.approx(0.975) for text in endpoint_labels)
    assert all(text.get_rotation() == pytest.approx(270.0) for text in endpoint_labels)
    assert all(
        text.get_fontsize() == pytest.approx(PANEL_B_ENDPOINT_FONT_SIZE)
        for text in endpoint_labels
    )
    renderer = figure.canvas.get_renderer()
    assert all(
        figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1)
        for box in (text.get_window_extent(renderer) for text in endpoint_labels)
    )
    plt.close(figure)


def test_panel_b_row_labels_do_not_overlap_panel_a_schematic(tmp_path: Path) -> None:
    source_root = tmp_path / "source_data"
    _write_main_figure_sources(source_root)
    figure = _draw_main_figure(*_read_composite_sources(source_root))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    panel_a_axis = next(
        axis for axis in figure.axes if not axis.axison and axis.get_position().x0 < 0.1
    )
    panel_a_rightmost_box = max(
        (patch for patch in panel_a_axis.patches if isinstance(patch, FancyBboxPatch)),
        key=lambda patch: patch.get_x() + patch.get_width(),
    )
    panel_a_rightmost_box = panel_a_rightmost_box.get_window_extent(renderer)
    panel_b_axis = next(
        axis
        for axis in figure.axes
        if len(axis.patches) == len(ENDPOINTS) * 4
        and axis.get_xlabel() == "Metric value"
    )
    row_labels = panel_b_axis.get_yticklabels()

    assert any(label.get_text() == "Control-adjusted $\\Delta^{\\mathrm{CA}}$" for label in row_labels)
    assert all(
        label.get_window_extent(renderer).x0 > panel_a_rightmost_box.x1
        for label in row_labels
    )
    plt.close(figure)


def test_generate_panel_a_writes_render_and_provenance(tmp_path: Path) -> None:
    overview = tmp_path / "overview.csv"
    _write_overview(overview)

    paths = generate_panel_a(
        panel_root=tmp_path / "panels",
        result_root=tmp_path / "results",
        overview_path=overview,
    )

    assert set(paths) == {"png", "pdf", "tiff", "source_data", "manifest"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert paths["png"] == tmp_path / "panels/panel_a.png"
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["artifacts"]["png"] == str(paths["png"])
    source = pd.read_csv(paths["source_data"])
    assert set(source["field"]) == {
        "number_of_subjects",
        "donor_split",
        "displayed_endpoints",
        "primary_evaluation",
    }
    with Image.open(paths["png"]) as image:
        assert image.width / image.height >= 4.0
        assert image.width >= round(PANEL_A_SIZE_INCHES[0] * PANEL_RASTER_DPI) - 4
        assert image.height >= round(PANEL_A_SIZE_INCHES[1] * PANEL_RASTER_DPI) - 4


def test_generate_panel_b_writes_exact_method_seed_source_data(tmp_path: Path) -> None:
    detection = tmp_path / "detection.csv"
    _write_detection(detection)

    paths = generate_panel_b(
        panel_root=tmp_path / "panels",
        result_root=tmp_path / "results",
        detection_path=detection,
    )

    source = pd.read_csv(paths["source_data"])
    assert len(source) == 2 * 5 * len(PANEL_B_METHODS)
    assert set(source["held_out_label"]) == set(ENDPOINTS)
    assert set(source["seed"]) == set(range(1, 6))
    assert set(source["method"]) == {method for method, _, _ in PANEL_B_METHODS}
    assert source["average_precision"].between(0, 1).all()
    uniform = source.loc[source["method"].eq("uniform_uot")]
    assert uniform["run_id"].eq(uniform["evaluation_run_id"]).all()
    assert uniform["score_run_id"].str.endswith("-tau05-uniform").all()
    assert uniform["score_run_id"].ne(uniform["evaluation_run_id"]).all()
    manifest = yaml.safe_load(paths["manifest"].read_text())
    assert manifest["parameters"]["method_colors"] == {
        method: METHOD_COLORS[method] for method, _, _ in PANEL_B_METHODS
    }
    assert (
        manifest["parameters"]["method_color_source"]
        == "experiments/mouse_spleen/generate_figure4_panels.py"
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_panel_b_filters_to_primary_within_cdc2_scope(tmp_path: Path) -> None:
    detection = tmp_path / "detection.csv"
    _write_detection(detection)
    within = pd.read_csv(detection)
    within["evaluation_scope"] = "local_within_broad_state"
    global_rows = within.copy()
    global_rows["evaluation_scope"] = "global_all_query"
    global_rows["auprc"] = 0.01
    pd.concat([within, global_rows], ignore_index=True).to_csv(detection, index=False)

    paths = generate_panel_b(
        panel_root=tmp_path / "panels",
        result_root=tmp_path / "results",
        detection_path=detection,
    )

    source = pd.read_csv(paths["source_data"])
    assert len(source) == 2 * 5 * len(PANEL_B_METHODS)
    assert not source["average_precision"].eq(0.01).any()


def test_panel_b_rejects_incomplete_endpoint_seed_keys(tmp_path: Path) -> None:
    detection = tmp_path / "detection.csv"
    _write_detection(detection, omit_last=True)

    with pytest.raises(HIHAFigure2Error, match="Panel B keys differ"):
        generate_panel_b(
            panel_root=tmp_path / "panels",
            result_root=tmp_path / "results",
            detection_path=detection,
        )


def test_panel_b_rejects_missing_score_run_lineage(tmp_path: Path) -> None:
    detection = tmp_path / "detection.csv"
    _write_detection(detection)
    frame = pd.read_csv(detection).drop(columns="score_run_id")
    frame.to_csv(detection, index=False)

    with pytest.raises(HIHAFigure2Error, match="score_run_id"):
        generate_panel_b(
            panel_root=tmp_path / "panels",
            result_root=tmp_path / "results",
            detection_path=detection,
        )


def test_generate_panel_c_reconstructs_paired_rescue_and_destination(
    tmp_path: Path,
) -> None:
    detection = tmp_path / "detection.csv"
    runs_root = tmp_path / "runs"
    _write_panel_d_artifacts(runs_root, detection)

    paths = generate_panel_c(
        panel_root=tmp_path / "panels",
        result_root=tmp_path / "results",
        detection_path=detection,
        runs_root=runs_root,
    )

    cells = pd.read_parquet(paths["paired_cells"])
    deficit = pd.read_csv(paths["deficit_by_seed"])
    restoration = pd.read_csv(paths["restoration_by_seed"])
    assert len(cells) == 2 * 5 * 4
    assert len(deficit) == 2 * 5 * 2
    assert len(restoration) == 2 * 5
    assert (restoration["restoration_specificity"] > 0).all()
    assert np.allclose(restoration["median_restored_state_probability"], 0.8)
    manifest = yaml.safe_load(paths["manifest"].read_text())
    assert manifest["panel"] == "C"
    assert manifest["parameters"]["display_rows"] == [
        "held_out_delta_median_u",
        "control_delta_median_u",
        "restoration_specificity",
    ]
    assert manifest["parameters"]["destination_axis_limits"] == [0.0, 1.0]
    assert not manifest["parameters"]["connect_split_points_across_rows"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_panel_c_rejects_non_normalized_conditional_probabilities(
    tmp_path: Path,
) -> None:
    detection = tmp_path / "detection.csv"
    runs_root = tmp_path / "runs"
    _write_panel_d_artifacts(runs_root, detection)
    bad_path = (
        runs_root / "panel_d_0_1/transport/full_reference_control/hiha_harmony30_k100/"
        "coreot_full/label_probabilities.npz"
    )
    with np.load(bad_path, allow_pickle=True) as archive:
        cell_ids = archive["cell_ids"]
        labels = archive["labels"]
        probabilities = archive["probabilities"]
    probabilities[0, 0] -= 0.1
    np.savez_compressed(
        bad_path,
        cell_ids=cell_ids,
        labels=labels,
        probabilities=probabilities,
    )

    with pytest.raises(HIHAFigure2Error, match="do not sum to one"):
        generate_panel_c(
            panel_root=tmp_path / "panels",
            result_root=tmp_path / "results",
            detection_path=detection,
            runs_root=runs_root,
        )


def test_generate_panel_e_writes_truth_and_six_top_n_maps(
    tmp_path: Path,
) -> None:
    paths = prepare_hiha_candidate_sources(
        package_root=Path("results/uot_baseline_pilot/portable_decision_package_v1"),
        canonical_source_root=Path("results/HIHA_DC/manuscript/figure2/source_data"),
        candidate_figure_root=tmp_path / "figure2",
        authoritative_runs_root=HIHA_AUTHORITATIVE_FIXTURE_RUNS,
    )
    cells = pd.read_csv(paths["source_root"] / "panel_d_umap_cells.csv")
    assert set(cells["seed"]) == {REPRESENTATIVE_SEED}
    assert set(cells["map_id"]) == {map_id for map_id, _, _ in PANEL_E_MAPS}
    assert not cells.duplicated(["held_out_label", "map_id", "cell_id"]).any()
    method_cells = cells.loc[cells["map_id"].ne("truth")]
    assert method_cells.loc[method_cells["is_selected"], "is_within_cdc2"].all()
    selected = method_cells.groupby(["held_out_label", "map_id"])["is_selected"].sum()
    held_out = (
        cells.loc[cells["map_id"].eq("truth")].groupby("held_out_label")["is_held_out_truth"].sum()
    )
    assert all(count == held_out.loc[endpoint] for (endpoint, _), count in selected.items())


def test_generate_panel_d_writes_exact_represented_state_rows(
    tmp_path: Path,
) -> None:
    label_transfer = tmp_path / "label_transfer.csv"
    _write_panel_f_label_transfer(label_transfer)

    paths = generate_panel_d(
        panel_root=tmp_path / "panels",
        result_root=tmp_path / "results",
        label_transfer_path=label_transfer,
    )

    source = pd.read_csv(paths["source_data"])
    assert len(source) == 2 * 5 * len(PANEL_D_METHODS)
    assert set(source["held_out_label"]) == set(ENDPOINTS)
    assert set(source["seed"]) == set(range(1, 6))
    assert set(source["method"]) == {method for method, _, _ in PANEL_D_METHODS}
    assert source["n_shared"].eq(source["n_forced_labeled"]).all()
    assert source["forced_macro_f1"].between(*PANEL_D_LIM).all()
    assert source["forced_accuracy"].between(*PANEL_D_LIM).all()
    manifest = yaml.safe_load(paths["manifest"].read_text())
    assert manifest["panel"] == "D"
    assert manifest["parameters"]["encoding"] == (
        "five_donor_split_points_plus_arithmetic_mean_and_sample_sd_interval"
    )
    assert manifest["parameters"]["mean_marker"] is True
    assert manifest["parameters"]["metric_legend"] == "point_and_interval"
    assert manifest["parameters"]["split_points"] == "five_donor_split_values"
    assert "bar_baseline" not in manifest["parameters"]
    assert "bar_baseline_is_truncated" not in manifest["parameters"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_composite_panels_c_and_d_use_approved_metric_bars(tmp_path: Path) -> None:
    source_root = tmp_path / "composite_source"
    _write_main_figure_sources(source_root)
    sources = _read_composite_sources(source_root)
    ranking = sources[1]
    transfer = sources[4]

    for drawer, frame in (
        (_draw_panel_c_metric_bars, ranking),
        (_draw_panel_d_metric_bars, transfer),
    ):
        figure = plt.figure(figsize=MAIN_FIGURE_SIZE_INCHES)
        drawer(figure, frame)
        assert len(figure.axes) == 2
        for axis in figure.axes:
            assert axis.get_xlim() == (0.0, 1.02)
            assert len(axis.patches) == len(PANEL_D_METHODS) * 2
            assert not any(
                isinstance(collection, PathCollection)
                for collection in axis.collections
            )
            assert axis.title.get_fontweight() == "normal"
        scmap_label = figure.axes[0].get_yticklabels()[-2]
        assert scmap_label.get_text() == "scmap-\ncluster"
        figure.canvas.draw()
        if drawer is _draw_panel_d_metric_bars:
            assert scmap_label.get_window_extent().x0 > figure.bbox.width * 0.485
        plt.close(figure)


def test_panel_e_rejects_incomplete_endpoint_seed_keys(tmp_path: Path) -> None:
    label_transfer = tmp_path / "label_transfer.csv"
    _write_panel_f_label_transfer(label_transfer, omit_last=True)

    with pytest.raises(HIHAFigure2Error, match="Panel D keys differ"):
        generate_panel_d(
            panel_root=tmp_path / "panels",
            result_root=tmp_path / "results",
            label_transfer_path=label_transfer,
        )


def test_generate_panel_f_writes_label_assignment_maps(
    tmp_path: Path,
) -> None:
    paths = prepare_hiha_candidate_sources(
        package_root=Path("results/uot_baseline_pilot/portable_decision_package_v1"),
        canonical_source_root=Path("results/HIHA_DC/manuscript/figure2/source_data"),
        candidate_figure_root=tmp_path / "figure2",
        authoritative_runs_root=HIHA_AUTHORITATIVE_FIXTURE_RUNS,
    )
    cells = pd.read_csv(paths["source_root"] / "panel_f_label_assignment_cells.csv")
    summary = pd.read_csv(paths["source_root"] / "panel_f_label_assignment_summary.csv")
    assert len(summary) == 2 * len(PANEL_D_METHODS)
    assert set(cells["map_id"]) == {map_id for map_id, _ in PANEL_F_MAPS}
    assert not cells.duplicated(["held_out_label", "map_id", "cell_id"]).any()
    coordinate_counts = cells.groupby(["held_out_label", "cell_id"])[["umap_1", "umap_2"]].nunique()
    assert coordinate_counts.eq(1).all().all()
    held_out = cells.loc[cells["is_held_out_state"]]
    assert held_out["displayed_assignment"].eq("Held-out state (not evaluated)").all()
    represented = cells.loc[cells["is_represented_state"]]
    assert set(represented["displayed_assignment"]) <= set(PANEL_F_LABEL_COLORS)
    assert np.allclose(summary["forced_accuracy"], summary["panel_e_forced_accuracy"])
    assert np.allclose(summary["forced_macro_f1"], summary["panel_e_forced_macro_f1"])
    assert paths["manifest"].is_file()


def test_panel_f_assignments_reproduce_panel_d_sealed_metrics(tmp_path: Path) -> None:
    paths = prepare_hiha_candidate_sources(
        package_root=Path("results/uot_baseline_pilot/portable_decision_package_v1"),
        canonical_source_root=Path("results/HIHA_DC/manuscript/figure2/source_data"),
        candidate_figure_root=tmp_path / "figure2",
        authoritative_runs_root=HIHA_AUTHORITATIVE_FIXTURE_RUNS,
    )
    summary = pd.read_csv(paths["source_root"] / "panel_f_label_assignment_summary.csv")
    assert np.allclose(summary["forced_accuracy"], summary["panel_e_forced_accuracy"])
    assert np.allclose(summary["forced_macro_f1"], summary["panel_e_forced_macro_f1"])


def test_generate_main_figure_writes_a_to_f_composite(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    source_root = result_root / "source_data"
    _write_main_figure_sources(source_root)
    docs_svg = tmp_path / "docs/figs/manuscript_fig_hiha_main.svg"
    result_svg = result_root / "main_figure.svg"
    docs_svg.parent.mkdir(parents=True)
    result_root.mkdir(parents=True, exist_ok=True)
    docs_svg.write_text("legacy", encoding="utf-8")
    result_svg.write_text("legacy", encoding="utf-8")

    paths = generate_main_figure(
        docs_root=tmp_path / "docs",
        result_root=result_root,
        source_root=source_root,
        project_root=tmp_path,
    )

    assert set(paths) == {"png", "pdf", "tiff", "manifest"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert not docs_svg.exists()
    assert not result_svg.exists()
    assert paths["png"] == tmp_path / "docs/figs/manuscript_fig_hiha_main.png"
    with Image.open(paths["png"]) as image:
        assert image.width >= round(MAIN_FIGURE_SIZE_INCHES[0] * PANEL_RASTER_DPI) - 4
        assert image.height >= round(MAIN_FIGURE_SIZE_INCHES[1] * PANEL_RASTER_DPI) - 4
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["panel_set"] == list(MAIN_FIGURE_PANEL_SET)
    assert manifest["panel_contract"] == {
        "B": "paired_reference_restoration_response",
        "C": "within_cdc2_omitted_state_ranking",
        "D": "represented_state_label_transfer",
    }
    assert (
        manifest["parameters"]["panel_b_endpoint_label_alignment"]
        == "right_edge_rotated_270"
    )
    assert manifest["parameters"]["panel_b_endpoint_font_size_points"] == pytest.approx(
        PANEL_B_ENDPOINT_FONT_SIZE
    )
    assert manifest["parameters"]["panel_b_axis_bounds_fraction"] == pytest.approx(
        [0.54, 0.82, 0.37, 0.135]
    )
    assert manifest["parameters"]["panel_b_encoding"] == {
        "rows": [
            "reference_omitted_cell_median_deficit_decrease",
            "represented_control_median_deficit_decrease",
            "control_adjusted_median_deficit_decrease",
            "median_restored_state_destination_fraction",
        ],
        "bar": "arithmetic_mean_across_five_donor_splits",
        "whisker": "sample_sd_ddof1",
        "points": "not_rendered",
        "axis": [0.0, 1.0],
        "axis_label": "Metric value",
    }
    assert manifest["parameters"]["panel_e_legend_columns"] == 4
    assert manifest["parameters"]["spatial_panel_layout"] == {
        "panels": ["E", "F"],
        "method_columns": "aligned",
        "axis_width_fraction": 0.128,
        "axis_height_fraction": 0.075,
        "endpoint_labels": "left_border_rotated_270",
        "endpoint_label_weight": "normal",
    }
    assert manifest["parameters"]["method_label_layout"] == {
        "panels": ["C", "D"],
        "scmap_cluster_lines": 2,
    }
    assert manifest["panel_d_disposition"] == "main_figure_panel"
    assert manifest["panel_e_disposition"] == "main_figure_panel"
    assert manifest["panel_f_disposition"] == "main_figure_panel"
    assert manifest["parameters"]["minimum_font_size_points"] == pytest.approx(
        PANEL_A_MIN_FONT_SIZE
    )
    assert manifest["parameters"]["standard_minimum_font_size_points"] == pytest.approx(
        MAIN_FIGURE_MIN_FONT_SIZE
    )
    assert manifest["parameters"]["panel_c_encoding"] == {
        "solid": "average_precision",
        "hollow": "auroc",
        "reference": "mean_positive_prevalence",
        "bar": "arithmetic_mean_across_five_donor_splits",
        "whisker": "sample_sd_ddof1",
        "points": "not_rendered",
        "axis": [0.0, 1.0],
    }
    assert manifest["parameters"]["panel_d_encoding"] == {
        "solid": "forced_accuracy",
        "hollow": "forced_macro_f1",
        "bar": "arithmetic_mean_across_five_donor_splits",
        "whisker": "sample_sd_ddof1",
        "points": "not_rendered",
        "axis": [0.0, 1.0],
    }
    assert manifest["result_artifacts"] == {
        suffix: str(Path("results") / f"main_figure.{suffix}") for suffix in ("png", "pdf", "tiff")
    }
    assert all(
        (tmp_path / path).is_file() and (tmp_path / path).stat().st_size > 0
        for path in manifest["result_artifacts"].values()
    )
    assert all(not Path(path).is_absolute() for path in manifest["sources"])
    assert all(not path.endswith(".pdf") for path in manifest["output_sha256"])
    assert manifest["container_metadata_policy"]["pdf_hashes_recorded"] is False
