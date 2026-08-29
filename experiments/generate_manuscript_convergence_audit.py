from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


EVIDENCE_VERIFIED_CONVERGED = "verified_converged"
EVIDENCE_VERIFIED_NONCONVERGED = "verified_nonconverged"
EVIDENCE_RECORD_UNAVAILABLE = "per_fit_convergence_record_unavailable"
SETTINGS_FIT_METHOD_PARAMS = "fit_method_params"
SETTINGS_RETAINED_PER_FIT_TABLE = "retained_per_fit_table"
SETTINGS_CONFIGURED_ONLY = "configured_only_fit_records_unavailable"

HIHA_HLA_PARAMETER_INDEX = Path(
    "results/HIHA_DC/sensitivity/full_tau_target2_alpha2_hla/tables/"
    "detection_within_cdc2_by_run.csv"
)
HIHA_PARAMETER_FIGURE_MANIFEST = Path(
    "results/HIHA_DC/figures/hiha_parameter_sensitivity_manifest.yaml"
)
AUXILIARY_INDEX_INPUTS = (
    HIHA_HLA_PARAMETER_INDEX,
    HIHA_PARAMETER_FIGURE_MANIFEST,
)
RETIRED_FIT_FAMILIES = (
    {
        "experiment": "HIHA DC",
        "analysis_family": "constant-tau compatibility sensitivity",
        "historical_source": (
            "results/HIHA_DC/sensitivity/constant_tau_alpha/tables/"
            "detection_by_run.csv"
        ),
        "reason": (
            "The historical broad grid is not an input to an active manuscript "
            "claim, display, Supplementary Data 2 member, or release unit."
        ),
    },
    {
        "experiment": "Mouse spleen",
        "analysis_family": "prior-risk diagnostic",
        "historical_source": (
            "results/mouse_spleen_core_ot/manuscript/prior_dependence/"
        ),
        "reason": (
            "The mouse prior-dependence unit is outside the current supplement "
            "hierarchy and is not an active manuscript claim or display."
        ),
    },
)

RETAINED_FIT_FAMILIES = (
    {
        "experiment": "HIHA DC",
        "analysis_family": "selected comparison",
        "manuscript_scope": (
            "S2.3 selected ranking and represented-state forced-transfer "
            "comparisons; S2.4 paired reference-restoration response; S2.5 "
            "operational abstention and calibration sensitivity"
        ),
        "convergence_evidence_source": (
            "runs/hiha_dc_*/transport/*/hiha_harmony30_k100/*/"
            "transport_manifest.yaml"
        ),
    },
    {
        "experiment": "HIHA DC",
        "analysis_family": "component-ablation surface",
        "manuscript_scope": (
            "S2.6.1 manuscript-facing compatibility-sensitivity surface; "
            "supporting component-sensitivity fits retained in Supplementary "
            "Data 2"
        ),
        "convergence_evidence_source": (
            "results/HIHA_DC/sensitivity/component_ablation/tables/"
            "component_ablation_by_seed.csv"
        ),
    },
    {
        "experiment": "HIHA DC",
        "analysis_family": "compatibility-weight mean-matched attribution",
        "manuscript_scope": (
            "Supporting across-alpha mean-matched attribution associated with "
            "S2.6.2 and retained in Supplementary Data 2; not the "
            "manuscript-facing fixed-alpha-zero surface"
        ),
        "convergence_evidence_source": (
            "results/HIHA_DC/sensitivity/rho_attribution_alpha_search/"
            "tables/rho_attribution_by_replicate.csv"
        ),
    },
    {
        "experiment": "HIHA DC",
        "analysis_family": "fixed-alpha-zero matchability surface",
        "manuscript_scope": (
            "S2.6.2 fixed-alpha-zero mean-matched matchability-penalty "
            "attribution"
        ),
        "convergence_evidence_source": (
            "results/HIHA_DC/sensitivity/"
            "rho_attribution_discovery_confirmation/tables/"
            "discovery_alpha0_focused025_125_by_split.csv"
        ),
    },
    {
        "experiment": "HIHA DC",
        "analysis_family": "query-penalty sensitivity",
        "manuscript_scope": "S2.6.3 query-penalty parameter sensitivity",
        "convergence_evidence_source": (
            "results/HIHA_DC/figures/"
            "hiha_parameter_sensitivity_manifest.yaml; "
            "results/HIHA_DC/sensitivity/full_tau_target2_alpha2_hla/tables/"
            "detection_within_cdc2_by_run.csv; "
            "results/HIHA_DC/sensitivity/"
            "isg_tau_range_target1_alpha025_discovery/tables/"
            "discovery_by_split.csv"
        ),
    },
    {
        "experiment": "PBMC",
        "analysis_family": "parameter sensitivity",
        "manuscript_scope": (
            "S3.2 selected CoRe-OT operating points; S3.3 selected CoRe-OT "
            "results; S3.4 paired reference-restoration response; S3.5 "
            "operational abstention and calibration sensitivity; S3.6.3 "
            "query-penalty sensitivity; S3.6.4 selected-fit "
            "prior-dependence diagnostic"
        ),
        "convergence_evidence_source": (
            "results/PBMC/manuscript/supplement/"
            "pbmc_sensitivity_by_seed.csv; selected run transport manifests "
            "under runs/pbmc_ifnb_*"
        ),
    },
    {
        "experiment": "PBMC",
        "analysis_family": "selected comparison",
        "manuscript_scope": (
            "S3.3 selected match-only and Uniform UOT comparisons; S3.5 "
            "Uniform UOT calibration sensitivity"
        ),
        "convergence_evidence_source": (
            "runs/pbmc_ifnb_*/transport/*/pca30_k100/*/"
            "transport_manifest.yaml"
        ),
    },
    {
        "experiment": "PBMC",
        "analysis_family": "component-ablation surface",
        "manuscript_scope": (
            "S3.6.1 manuscript-facing constant-tau compatibility-sensitivity "
            "surface; supporting alpha-zero penalty-sensitivity fits retained "
            "in Supplementary Data 3"
        ),
        "convergence_evidence_source": (
            "results/PBMC/sensitivity/component_ablation/tables/"
            "component_ablation_by_seed.csv"
        ),
    },
    {
        "experiment": "PBMC",
        "analysis_family": "fixed-alpha-zero matchability surface",
        "manuscript_scope": (
            "S3.6.2 fixed-alpha-zero mean-matched matchability-penalty "
            "attribution"
        ),
        "convergence_evidence_source": (
            "results/PBMC/sensitivity/"
            "rho_attribution_tau_surface_alpha0_range075_175/tables/"
            "rho_tau_surface_range075_175_by_seed.csv"
        ),
    },
    {
        "experiment": "PBMC",
        "analysis_family": "compatibility-weight mean-matched attribution",
        "manuscript_scope": (
            "Supporting across-alpha mean-matched attribution associated with "
            "S3.6.2 and retained in Supplementary Data 3; not the "
            "manuscript-facing fixed-alpha-zero surface"
        ),
        "convergence_evidence_source": (
            "results/PBMC/sensitivity/rho_attribution_alpha_search/"
            "tables/rho_attribution_by_replicate.csv"
        ),
    },
    {
        "experiment": "Mouse spleen",
        "analysis_family": "selected comparison",
        "manuscript_scope": (
            "S4.2 selected operating point; S4.3 selected transport "
            "comparisons; S4.4 selected-fit destination and "
            "transported-support summaries"
        ),
        "convergence_evidence_source": (
            "results/mouse_spleen_core_ot/runs/"
            "mouse_spleen_natural_proliferating/transport/natural_mismatch/"
            "mouse_spleen_provider_k100/*/transport_manifest.yaml"
        ),
    },
    {
        "experiment": "Mouse spleen",
        "analysis_family": "constant-tau compatibility-sensitivity surface",
        "manuscript_scope": "S4.5.1 constant-tau compatibility-sensitivity surface",
        "convergence_evidence_source": (
            "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
            "component_ablation_proliferating/tables/"
            "compatibility_only_metrics.csv"
        ),
    },
    {
        "experiment": "Mouse spleen",
        "analysis_family": "fixed-alpha-five matchability surface",
        "manuscript_scope": (
            "S4.5.2 fixed-alpha-five mean-matched matchability-penalty "
            "attribution"
        ),
        "convergence_evidence_source": (
            "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
            "rho_attribution_tau_alpha_search/tables/"
            "coarse_by_dataset.csv"
        ),
    },
    {
        "experiment": "Mouse spleen",
        "analysis_family": "query-penalty sensitivity",
        "manuscript_scope": "S4.5.3 query-penalty parameter sensitivity",
        "convergence_evidence_source": (
            "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
            "figure4_full_tau_alpha_40_tau_target_8/tables/"
            "metrics_by_grid.csv"
        ),
    },
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _manifest_row(
    *,
    project_root: Path,
    experiment: str,
    analysis_family: str,
    endpoint: str,
    seed: int | None,
    condition: str,
    run_id: str,
    candidate_set: str,
    method: str,
    max_iter: int,
    tol: float,
    runs_root: Path = Path("runs"),
) -> dict[str, object]:
    fit_directory = (
        project_root
        / runs_root
        / run_id
        / "transport"
        / condition
        / candidate_set
        / method
    )
    manifest_path = fit_directory / "transport_manifest.yaml"
    method_params_path = fit_directory / "method_params.yaml"
    manifest_available = manifest_path.is_file()
    method_params_available = method_params_path.is_file()
    if not manifest_available and not method_params_available:
        return {
            "experiment": experiment,
            "analysis_family": analysis_family,
            "endpoint": endpoint,
            "seed": seed,
            "condition": condition,
            "run_id": run_id,
            "method": method,
            "convergence_evidence": EVIDENCE_RECORD_UNAVAILABLE,
            "converged": pd.NA,
            "n_iter": pd.NA,
            "max_iter": max_iter,
            "tol": tol,
            "source_artifact": "",
            "solver_settings_evidence": SETTINGS_CONFIGURED_ONLY,
            "method_params_artifact": "",
        }
    if manifest_available != method_params_available:
        raise ValueError(
            "Transport fit has an incomplete manifest/method-parameter pair: "
            f"{fit_directory}"
        )
    manifest = _read_yaml(manifest_path)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Transport manifest omits metadata: {manifest_path}")
    manifest_method = str(metadata.get("method", ""))
    if manifest_method != method:
        raise ValueError(
            f"Transport manifest method mismatch at {manifest_path}: "
            f"expected {method}, found {manifest_method or '<missing>'}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("method_params"):
        raise ValueError(
            f"Transport manifest omits its method-parameter artifact: {manifest_path}"
        )
    recorded_method_params = Path(str(artifacts["method_params"]))
    if not recorded_method_params.is_absolute():
        recorded_method_params = project_root / recorded_method_params
    if recorded_method_params.resolve() != method_params_path.resolve():
        raise ValueError(
            "Transport manifest and method parameters do not resolve to the same "
            f"fit directory: {manifest_path} records {recorded_method_params}, "
            f"expected {method_params_path}"
        )
    method_params = _read_yaml(method_params_path)
    params_method = str(method_params.get("name", ""))
    if params_method != method:
        raise ValueError(
            f"Method-parameter identity mismatch at {method_params_path}: "
            f"expected {method}, found {params_method or '<missing>'}"
        )
    if "max_iter" not in method_params or "tol" not in method_params:
        raise ValueError(
            f"Method parameters omit max_iter or tol: {method_params_path}"
        )
    converged = bool(metadata["converged"])
    return {
        "experiment": experiment,
        "analysis_family": analysis_family,
        "endpoint": endpoint,
        "seed": seed,
        "condition": condition,
        "run_id": run_id,
        "method": method,
        "convergence_evidence": (
            EVIDENCE_VERIFIED_CONVERGED
            if converged
            else EVIDENCE_VERIFIED_NONCONVERGED
        ),
        "converged": converged,
        "n_iter": int(metadata["n_iter"]),
        "max_iter": int(method_params["max_iter"]),
        "tol": float(method_params["tol"]),
        "source_artifact": str(manifest_path.relative_to(project_root)),
        "solver_settings_evidence": SETTINGS_FIT_METHOD_PARAMS,
        "method_params_artifact": str(
            method_params_path.relative_to(project_root)
        ),
    }


def _recorded_row(
    *,
    experiment: str,
    analysis_family: str,
    endpoint: str,
    seed: int | None,
    condition: str,
    run_id: str,
    method: str,
    converged: bool,
    n_iter: int | None,
    max_iter: int,
    tol: float,
    source_artifact: Path,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "analysis_family": analysis_family,
        "endpoint": endpoint,
        "seed": seed,
        "condition": condition,
        "run_id": run_id,
        "method": method,
        "convergence_evidence": (
            EVIDENCE_VERIFIED_CONVERGED
            if converged
            else EVIDENCE_VERIFIED_NONCONVERGED
        ),
        "converged": converged,
        "n_iter": n_iter if n_iter is not None else pd.NA,
        "max_iter": max_iter,
        "tol": tol,
        "source_artifact": str(source_artifact),
        "solver_settings_evidence": SETTINGS_RETAINED_PER_FIT_TABLE,
        "method_params_artifact": "",
    }


def _parameterized_run_id(
    base_run_id: str,
    item: pd.Series,
    parameter_columns: tuple[str, ...],
) -> str:
    parameters = ",".join(
        f"{column}={float(item[column]):g}" for column in parameter_columns
    )
    return f"{base_run_id}[{parameters}]"


def _single_fit_table_rows(
    *,
    project_root: Path,
    relative_path: Path,
    experiment: str,
    analysis_family: str,
    condition: str,
    parameter_columns: tuple[str, ...],
    endpoint_default: str | None = None,
    seed_default: int | None = None,
    run_id_default: str | None = None,
) -> list[dict[str, object]]:
    table = pd.read_csv(project_root / relative_path)
    rows: list[dict[str, object]] = []
    for _, item in table.iterrows():
        endpoint = (
            str(item["endpoint"])
            if "endpoint" in table.columns
            else str(endpoint_default)
        )
        seed = (
            int(item["seed"])
            if "seed" in table.columns
            else seed_default
        )
        base_run_id = (
            str(item["run_id"])
            if "run_id" in table.columns
            else str(run_id_default)
        )
        n_iter = (
            int(item["n_iterations"])
            if "n_iterations" in table.columns
            else None
        )
        rows.append(
            _recorded_row(
                experiment=experiment,
                analysis_family=analysis_family,
                endpoint=endpoint,
                seed=seed,
                condition=condition,
                run_id=_parameterized_run_id(
                    base_run_id,
                    item,
                    parameter_columns,
                ),
                method=str(item["method"]),
                converged=bool(item["converged"]),
                n_iter=n_iter,
                max_iter=int(item["max_iterations"]),
                tol=float(item["tolerance"]),
                source_artifact=relative_path,
            )
        )
    return rows


def _paired_fit_table_rows(
    *,
    project_root: Path,
    relative_path: Path,
    experiment: str,
    analysis_family: str,
    condition: str,
    parameter_columns: tuple[str, ...],
    max_iter_default: int,
    tol_default: float,
    endpoint_default: str | None = None,
    seed_default: int | None = None,
    run_id_default: str | None = None,
    pair_specs: tuple[tuple[str, str, str | None], ...] = (
        (
            "heterogeneous_query_penalty",
            "heterogeneous_converged",
            "heterogeneous_n_iterations",
        ),
        (
            "mean_matched_uniform_query_penalty",
            "uniform_converged",
            "uniform_n_iterations",
        ),
    ),
) -> list[dict[str, object]]:
    table = pd.read_csv(project_root / relative_path)
    rows: list[dict[str, object]] = []
    for _, item in table.iterrows():
        endpoint = (
            str(item["endpoint"])
            if "endpoint" in table.columns
            else str(endpoint_default)
        )
        seed = (
            int(item["seed"])
            if "seed" in table.columns
            else seed_default
        )
        base_run_id = (
            str(item["run_id"])
            if "run_id" in table.columns
            else str(run_id_default)
        )
        run_id = _parameterized_run_id(
            base_run_id,
            item,
            parameter_columns,
        )
        max_iter = (
            int(item["max_iterations"])
            if "max_iterations" in table.columns
            else max_iter_default
        )
        tol = (
            float(item["tolerance"])
            if "tolerance" in table.columns
            else tol_default
        )
        for method, converged_column, iteration_column in pair_specs:
            n_iter = (
                int(item[iteration_column])
                if iteration_column is not None
                and iteration_column in table.columns
                else None
            )
            rows.append(
                _recorded_row(
                    experiment=experiment,
                    analysis_family=analysis_family,
                    endpoint=endpoint,
                    seed=seed,
                    condition=condition,
                    run_id=run_id,
                    method=method,
                    converged=bool(item[converged_column]),
                    n_iter=n_iter,
                    max_iter=max_iter,
                    tol=tol,
                    source_artifact=relative_path,
                )
            )
    return rows


def _mouse_rows(project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_id = "mouse_spleen_natural_proliferating"
    candidate_set = "mouse_spleen_provider_k100"
    for method in (
        "coreot_full",
        "coreot_constant_tau",
        "coreot_match_only",
        "uniform_uot",
    ):
        rows.append(
            _manifest_row(
                project_root=project_root,
                experiment="Mouse spleen",
                analysis_family="selected comparison",
                endpoint="Proliferating",
                seed=None,
                condition="natural_mismatch",
                run_id=run_id,
                candidate_set=candidate_set,
                method=method,
                max_iter=5000,
                tol=1.0e-6,
                runs_root=Path("results/mouse_spleen_core_ot/runs"),
            )
        )

    table_specs = (
        (
            "constant-tau compatibility-sensitivity surface",
            Path(
                "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
                "component_ablation_proliferating/tables/"
                "compatibility_only_metrics.csv"
            ),
            ("tau_source", "alpha", "tau_target"),
        ),
    )
    for family, relative_path, parameter_columns in table_specs:
        table = pd.read_csv(project_root / relative_path)
        for _, item in table.iterrows():
            converged = bool(item["converged"])
            fit_id = ",".join(
                f"{name}={item[name]:g}" for name in parameter_columns
            )
            rows.append(
                {
                    "experiment": "Mouse spleen",
                    "analysis_family": family,
                    "endpoint": "Proliferating",
                    "seed": None,
                    "condition": "natural_mismatch",
                    "run_id": f"{item['run_id']}[{fit_id}]",
                    "method": str(item["method"]),
                    "convergence_evidence": (
                        EVIDENCE_VERIFIED_CONVERGED
                        if converged
                        else EVIDENCE_VERIFIED_NONCONVERGED
                    ),
                    "converged": converged,
                    "n_iter": (
                        int(item["n_iterations"])
                        if "n_iterations" in table.columns
                        else pd.NA
                    ),
                    "max_iter": int(item["max_iterations"]),
                    "tol": (
                        float(item["tolerance"])
                        if "tolerance" in table.columns
                        else 1.0e-6
                    ),
                    "source_artifact": str(relative_path),
                    "solver_settings_evidence": SETTINGS_RETAINED_PER_FIT_TABLE,
                    "method_params_artifact": "",
                }
            )

    surface_path = Path(
        "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_tau_alpha_search/tables/"
        "rho_tau_surface_alpha5_range025_4.csv"
    )
    fit_table_path = Path(
        "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "rho_attribution_tau_alpha_search/tables/coarse_by_dataset.csv"
    )
    surface = pd.read_csv(project_root / surface_path)
    fit_table = pd.read_csv(project_root / fit_table_path)
    parameter_columns = ("alpha", "tau_min", "tau_max")
    surface_keys = set(
        surface.loc[:, list(parameter_columns)].itertuples(index=False, name=None)
    )
    retained_fits = fit_table.loc[
        fit_table.loc[:, list(parameter_columns)].apply(tuple, axis=1).isin(surface_keys)
    ]
    retained_keys = set(
        retained_fits.loc[:, list(parameter_columns)].itertuples(
            index=False, name=None
        )
    )
    if (
        len(surface_keys) != 15
        or retained_keys != surface_keys
        or len(retained_fits) != 15
        or retained_fits.duplicated(list(parameter_columns)).any()
    ):
        raise ValueError(
            "The retained mouse-spleen alpha-five surface does not map one-to-one "
            "to detailed convergence rows."
        )
    for _, item in retained_fits.iterrows():
        parameterized_run = _parameterized_run_id(
            str(item["run_id"]), item, parameter_columns
        )
        for method, converged_column, iteration_column in (
            (
                "heterogeneous_query_penalty",
                "heterogeneous_converged",
                "heterogeneous_n_iterations",
            ),
            (
                "mean_matched_uniform_query_penalty",
                "uniform_converged",
                "uniform_n_iterations",
            ),
        ):
            rows.append(
                _recorded_row(
                    experiment="Mouse spleen",
                    analysis_family="fixed-alpha-five matchability surface",
                    endpoint=str(item["endpoint"]),
                    seed=int(item["seed"]),
                    condition="natural_mismatch",
                    run_id=parameterized_run,
                    method=method,
                    converged=bool(item[converged_column]),
                    n_iter=int(item[iteration_column]),
                    max_iter=int(item["max_iterations"]),
                    tol=float(item["tolerance"]),
                    source_artifact=fit_table_path,
                )
            )

    parameter_path = Path(
        "results/mouse_spleen_core_ot/natural_mismatch/sensitivity/"
        "figure4_full_tau_alpha_40_tau_target_8/tables/metrics_by_grid.csv"
    )
    parameter_table = pd.read_csv(project_root / parameter_path)
    for _, item in parameter_table.iterrows():
        rows.append(
            _recorded_row(
                experiment="Mouse spleen",
                analysis_family="query-penalty sensitivity",
                endpoint=str(item["natural_endpoint"]),
                seed=None,
                condition=str(item["condition"]),
                run_id=str(item["run_id"]),
                method=str(item["method"]),
                converged=bool(item["converged"]),
                n_iter=int(item["n_iterations"]),
                max_iter=int(item["max_iterations"]),
                tol=1.0e-6,
                source_artifact=parameter_path,
            )
        )
    return rows


def _hiha_rows(project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    endpoints = {
        "hladrhi_cdc2": "HLA-DRhi cDC2",
        "isg_cdc2": "ISG+ cDC2",
    }
    for slug, endpoint in endpoints.items():
        for seed in range(1, 6):
            selected_run = (
                f"hiha_dc_{slug}_seed{seed}_report_leave_one_HIHA_DC"
            )
            for condition in ("incomplete_reference", "full_reference_control"):
                for method in ("coreot_full", "coreot_match_only"):
                    rows.append(
                        _manifest_row(
                            project_root=project_root,
                            experiment="HIHA DC",
                            analysis_family="selected comparison",
                            endpoint=endpoint,
                            seed=seed,
                            condition=condition,
                            run_id=selected_run,
                            candidate_set="hiha_harmony30_k100",
                            method=method,
                            max_iter=2000,
                            tol=1.0e-6,
                        )
                    )
                rows.append(
                    _manifest_row(
                        project_root=project_root,
                        experiment="HIHA DC",
                        analysis_family="selected comparison",
                        endpoint=endpoint,
                        seed=seed,
                        condition=condition,
                        run_id=f"hiha_dc_{slug}_seed{seed}_tau05_uniform",
                        candidate_set="hiha_harmony30_k100",
                        method="uniform_uot",
                        max_iter=2000,
                        tol=1.0e-6,
                    )
                )
    rows.extend(
        _single_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/HIHA_DC/sensitivity/component_ablation/tables/"
                "component_ablation_by_seed.csv"
            ),
            experiment="HIHA DC",
            analysis_family="component-ablation surface",
            condition="incomplete_reference",
            parameter_columns=(
                "tau_min",
                "tau_max",
                "tau_source",
                "alpha",
            ),
        )
    )
    rows.extend(
        _paired_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/HIHA_DC/sensitivity/"
                "rho_attribution_alpha_search/tables/"
                "rho_attribution_by_replicate.csv"
            ),
            experiment="HIHA DC",
            analysis_family="compatibility-weight mean-matched attribution",
            condition="incomplete_reference",
            parameter_columns=("alpha", "tau_min", "tau_max"),
            max_iter_default=5000,
            tol_default=1.0e-6,
        )
    )
    rows.extend(
        _paired_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/HIHA_DC/sensitivity/"
                "rho_attribution_discovery_confirmation/tables/"
                "discovery_alpha0_focused025_125_by_split.csv"
            ),
            experiment="HIHA DC",
            analysis_family="fixed-alpha-zero matchability surface",
            condition="incomplete_reference",
            parameter_columns=("alpha", "tau_min", "tau_max"),
            max_iter_default=5000,
            tol_default=1.0e-6,
        )
    )

    hla_parameter_path = HIHA_HLA_PARAMETER_INDEX
    hla_parameter_table = pd.read_csv(project_root / hla_parameter_path)
    for _, item in hla_parameter_table.iterrows():
        rows.append(
            _manifest_row(
                project_root=project_root,
                experiment="HIHA DC",
                analysis_family="query-penalty sensitivity",
                endpoint=str(item["held_out_label"]),
                seed=int(item["seed"]),
                condition="incomplete_reference",
                run_id=str(item["run_id"]),
                candidate_set="hiha_harmony30_k100",
                method=str(item["method"]),
                max_iter=2000,
                tol=1.0e-6,
            )
        )

    parameter_manifest = _read_yaml(
        project_root / HIHA_PARAMETER_FIGURE_MANIFEST
    )
    parameter_grids = parameter_manifest.get("parameter_grids")
    if not isinstance(parameter_grids, dict):
        raise ValueError(
            "The HIHA parameter-sensitivity manifest lacks parameter_grids"
        )
    isg_grid = parameter_grids.get("ISG+ cDC2")
    if not isinstance(isg_grid, dict):
        raise ValueError(
            "The HIHA parameter-sensitivity manifest lacks the ISG+ cDC2 grid"
        )
    isg_parameter_path = Path(
        "results/HIHA_DC/sensitivity/"
        "isg_tau_range_target1_alpha025_discovery/tables/"
        "discovery_by_split.csv"
    )
    isg_parameter_table = pd.read_csv(project_root / isg_parameter_path)
    retained_isg = isg_parameter_table.loc[
        isg_parameter_table["tau_min"].isin(isg_grid["tau_min"])
        & isg_parameter_table["tau_max"].isin(isg_grid["tau_max"])
        & isg_parameter_table["alpha"].eq(float(isg_grid["alpha"]))
    ]
    grid_counts = retained_isg.groupby(["tau_min", "tau_max"]).size()
    if len(grid_counts) != 9 or not grid_counts.eq(5).all():
        raise ValueError(
            "The retained HIHA ISG+ cDC2 parameter grid is not a complete "
            "3-by-3 grid with five donor splits per configuration"
        )
    for _, item in retained_isg.iterrows():
        rows.append(
            _recorded_row(
                experiment="HIHA DC",
                analysis_family="query-penalty sensitivity",
                endpoint="ISG+ cDC2",
                seed=int(item["seed"]),
                condition="incomplete_reference",
                run_id=_parameterized_run_id(
                    str(item["run_id"]),
                    item,
                    ("tau_min", "tau_max", "alpha"),
                ),
                method="coreot_full",
                converged=bool(item["coreot_converged"]),
                n_iter=int(item["coreot_iterations"]),
                max_iter=5000,
                tol=1.0e-6,
                source_artifact=isg_parameter_path,
            )
        )

    return rows


def _pbmc_rows(project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sensitivity_path = Path(
        "results/PBMC/manuscript/supplement/pbmc_sensitivity_by_seed.csv"
    )
    sensitivity = pd.read_csv(project_root / sensitivity_path)
    retained = {"B cells", "NK cells", "Dendritic cells"}
    sensitivity = sensitivity.loc[sensitivity["held_out_label"].isin(retained)]
    for _, item in sensitivity.iterrows():
        rows.append(
            _manifest_row(
                project_root=project_root,
                experiment="PBMC",
                analysis_family="parameter sensitivity",
                endpoint=str(item["held_out_label"]),
                seed=int(item["seed"]),
                condition="incomplete_reference",
                run_id=str(item["run_id"]),
                candidate_set="pca30_k100",
                method="coreot_full",
                max_iter=2000,
                tol=1.0e-6,
            )
        )
        if not rows[-1]["source_artifact"]:
            rows[-1]["source_artifact"] = str(sensitivity_path)

    selected = sensitivity.loc[
        sensitivity.groupby(["held_out_label", "seed"])["within_celltype_auprc"]
        .transform("size")
        .gt(0)
    ]
    selected_run_ids = {
        "B cells": "taumin0p5_taumax1_alpha4",
        "NK cells": "taumin0p5_taumax1_alpha2",
        "Dendritic cells": "taumin0p75_taumax1_alpha3",
    }
    for endpoint, token in selected_run_ids.items():
        endpoint_rows = selected.loc[selected["held_out_label"].eq(endpoint)]
        for seed in range(1, 6):
            run_id = next(
                str(value)
                for value in endpoint_rows.loc[
                    endpoint_rows["seed"].eq(seed), "run_id"
                ]
                if token in str(value)
            )
            rows.append(
                _manifest_row(
                    project_root=project_root,
                    experiment="PBMC",
                    analysis_family="parameter sensitivity",
                    endpoint=endpoint,
                    seed=seed,
                    condition="full_reference_control",
                    run_id=run_id,
                    candidate_set="pca30_k100",
                    method="coreot_full",
                    max_iter=2000,
                    tol=1.0e-6,
                )
            )
            for condition in ("incomplete_reference", "full_reference_control"):
                for method in ("coreot_match_only", "uniform_uot"):
                    rows.append(
                        _manifest_row(
                            project_root=project_root,
                            experiment="PBMC",
                            analysis_family="selected comparison",
                            endpoint=endpoint,
                            seed=seed,
                            condition=condition,
                            run_id=run_id,
                            candidate_set="pca30_k100",
                            method=method,
                            max_iter=2000,
                            tol=1.0e-6,
                        )
                    )

    rows.extend(
        _single_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/PBMC/sensitivity/component_ablation/tables/"
                "component_ablation_by_seed.csv"
            ),
            experiment="PBMC",
            analysis_family="component-ablation surface",
            condition="incomplete_reference",
            parameter_columns=(
                "tau_min",
                "tau_max",
                "tau_source",
                "alpha",
            ),
        )
    )
    rows.extend(
        _paired_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/PBMC/sensitivity/rho_attribution_alpha_search/"
                "tables/rho_attribution_by_replicate.csv"
            ),
            experiment="PBMC",
            analysis_family="compatibility-weight mean-matched attribution",
            condition="incomplete_reference",
            parameter_columns=("alpha", "tau_min", "tau_max"),
            max_iter_default=5000,
            tol_default=1.0e-6,
        )
    )
    rows.extend(
        _paired_fit_table_rows(
            project_root=project_root,
            relative_path=Path(
                "results/PBMC/sensitivity/"
                "rho_attribution_tau_surface_alpha0_range075_175/tables/"
                "rho_tau_surface_range075_175_by_seed.csv"
            ),
            experiment="PBMC",
            analysis_family="fixed-alpha-zero matchability surface",
            condition="incomplete_reference",
            parameter_columns=("alpha", "tau_min", "tau_max"),
            max_iter_default=5000,
            tol_default=1.0e-6,
        )
    )
    return rows


def _reported_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def generate_audit(
    project_root: Path,
    *,
    output_root: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    by_fit = pd.DataFrame(
        _mouse_rows(project_root)
        + _hiha_rows(project_root)
        + _pbmc_rows(project_root)
    )
    by_fit["seed"] = by_fit["seed"].astype("Int64")
    by_fit["converged"] = by_fit["converged"].astype("boolean")
    by_fit["n_iter"] = by_fit["n_iter"].astype("Int64")
    by_fit = by_fit.sort_values(
        [
            "experiment",
            "analysis_family",
            "endpoint",
            "seed",
            "run_id",
            "condition",
            "method",
        ],
        na_position="first",
    )

    summary = (
        by_fit.groupby(
            ["experiment", "analysis_family", "convergence_evidence"],
            sort=True,
        )
        .size()
        .rename("n_fits")
        .reset_index()
    )

    registry = pd.DataFrame(RETAINED_FIT_FAMILIES)
    family_columns = ["experiment", "analysis_family"]
    expected_families = set(
        registry.loc[:, family_columns].itertuples(index=False, name=None)
    )
    observed_families = set(
        by_fit.loc[:, family_columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_families != expected_families:
        missing = sorted(expected_families - observed_families)
        unexpected = sorted(observed_families - expected_families)
        raise ValueError(
            "Retained convergence-audit family mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    fit_key = [
        "experiment",
        "analysis_family",
        "endpoint",
        "seed",
        "condition",
        "run_id",
        "method",
    ]
    duplicated = by_fit.duplicated(fit_key, keep=False)
    if duplicated.any():
        duplicate_rows = by_fit.loc[duplicated, fit_key].head(10)
        raise ValueError(
            "Convergence audit contains duplicate fit records:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    family_counts = (
        by_fit.assign(
            record_available=by_fit["convergence_evidence"].ne(
                EVIDENCE_RECORD_UNAVAILABLE
            ),
            record_unavailable=by_fit["convergence_evidence"].eq(
                EVIDENCE_RECORD_UNAVAILABLE
            ),
        )
        .groupby(family_columns, sort=True)
        .agg(
            n_indexed_fits=("run_id", "size"),
            n_with_per_fit_record=("record_available", "sum"),
            n_without_per_fit_record=("record_unavailable", "sum"),
        )
        .reset_index()
    )
    coverage = registry.merge(
        family_counts,
        on=family_columns,
        how="left",
        validate="one_to_one",
    )
    coverage["coverage_status"] = coverage[
        "n_without_per_fit_record"
    ].map(
        lambda value: (
            "indexed_with_complete_per_fit_records"
            if int(value) == 0
            else "indexed_with_record_gaps"
        )
    )

    if output_root is None:
        output_root = project_root / "results/manuscript/convergence_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    by_fit_path = output_root / "transport_convergence_by_fit.csv"
    summary_path = output_root / "transport_convergence_summary.csv"
    coverage_path = output_root / "retained_analysis_coverage.csv"
    manifest_path = output_root / "manifest.yaml"
    by_fit.to_csv(by_fit_path, index=False)
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "generate_manuscript_convergence_audit",
                "generated_on": date.today().isoformat(),
                "scope": (
                    "retained transport fits contributing to manuscript "
                    "comparisons, component analyses, or sensitivity summaries"
                ),
                "coverage_policy": (
                    "Every retained supplementary transport-fit family is "
                    "registered. Derived analyses that reuse fitted transports "
                    "are mapped to the corresponding family in the manuscript "
                    "scope column."
                ),
                "evidence_policy": {
                    EVIDENCE_VERIFIED_CONVERGED: (
                        "per-fit manifest or retained per-fit table records "
                        "convergence"
                    ),
                    EVIDENCE_VERIFIED_NONCONVERGED: (
                        "per-fit manifest or retained per-fit table records an "
                        "iteration-cap terminal iterate"
                    ),
                    EVIDENCE_RECORD_UNAVAILABLE: (
                        "aggregate numerical result is retained but the per-fit "
                        "convergence record is unavailable"
                    ),
                },
                "solver_settings_policy": {
                    SETTINGS_FIT_METHOD_PARAMS: (
                        "max_iter and tol are read from the method_params.yaml "
                        "declared by the same fit's transport manifest"
                    ),
                    SETTINGS_RETAINED_PER_FIT_TABLE: (
                        "max_iter and tol are recorded in the retained per-fit "
                        "source table"
                    ),
                    SETTINGS_CONFIGURED_ONLY: (
                        "fit-level manifest and method_params.yaml are both "
                        "unavailable; max_iter and tol are configured index "
                        "fallbacks, not observed fit evidence"
                    ),
                },
                "retired_analysis_families": list(RETIRED_FIT_FAMILIES),
                "artifacts": {
                    "by_fit": _reported_path(by_fit_path, project_root),
                    "summary": _reported_path(summary_path, project_root),
                    "coverage": _reported_path(coverage_path, project_root),
                },
                "auxiliary_index_inputs": [
                    str(path) for path in AUXILIARY_INDEX_INPUTS
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return by_fit_path, summary_path, coverage_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    paths = generate_audit(
        args.project_root.resolve(),
        output_root=(
            args.output_root.resolve()
            if args.output_root is not None
            else None
        ),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
