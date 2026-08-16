# ruff: noqa: F401
from __future__ import annotations

from coreot.results.broad_grid import (
    BroadGridArtifactPaths,
    DEFAULT_EXCLUSION_FILTERS,
    DEFAULT_BROAD_ALPHA_VALUES,
    DEFAULT_BROAD_TAU_VALUES,
    HELD_OUT_LABELS,
    build_pareto_table,
    render_broad_grid_report,
    render_broad_grid_table_1,
    render_broad_grid_table_2,
    write_broad_grid_results,
)
from coreot.results.coreot_full_sensitivity import (
    CoreotFullTauRangeArtifactPaths,
    parse_coreot_full_tau_range_params,
    render_coreot_full_tau_range_report,
    render_coreot_full_tau_range_table_1,
    render_coreot_full_tau_range_table_2,
    write_coreot_full_tau_range_results,
)
from coreot.results.compare_baselines import (
    CompareBaselinesArtifactPaths,
    write_compare_baselines_results,
)
from coreot.results.grid_common import (
    DISPLAY_NAME_BY_METHOD,
    EXTERNAL_PRIMARY_SCORE_BY_METHOD,
    EXTERNAL_TABLE_ROW_ORDER,
    LABEL_TRANSFER_METHODS,
    MAIN_TABLE_1_COLUMN_HEADERS,
    MAIN_TABLE_1_QUANTITIES,
    MAIN_TABLE_2_COLUMN_HEADERS,
    MAIN_TABLE_2_QUANTITIES,
    METHOD_GROUP_BY_METHOD,
    PRIMARY_SCORE_BY_METHOD,
    STAGE,
    TABLE_ROW_ORDER,
    ResultsGridError,
    RunDescriptor,
    _format_table_value,
    build_detection_by_run,
    build_forced_label_summary_by_run,
    build_full_reference_false_abstention_by_run,
    build_shared_label_transfer_by_run,
    discover_run_descriptors,
    primary_score_for_method,
    summarize_run_metrics,
)
from coreot.results.main_grid import (
    ResultsArtifactPaths,
    render_full_reference_table,
    render_main_table_1,
    render_main_table_2,
    render_main_tables,
    render_markdown_summary,
    write_grid_results,
)
from coreot.results.sweep_common import (
    FIXED_REFERENCE_METHODS,
    LABELWISE_DISPLAY_NAME,
    LABELWISE_HELD_OUT_LABELS,
    LABELWISE_LABEL_GRID,
    _filter_completed_runs,
    _labelwise_sort_key,
    _labelwise_swept_display_name,
    _validate_labelwise_grid,
    parse_labelwise_params,
    read_fixed_reference_rows,
)
from coreot.results.pbmc import (
    PbmcArtifactPaths,
    build_pbmc_rescue_by_run,
    build_pbmc_shared_label_transfer_by_run,
    build_pbmc_split_summary_by_run,
    build_pbmc_within_celltype_detection_by_run,
    write_pbmc_results,
)
