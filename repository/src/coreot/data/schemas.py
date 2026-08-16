BENCHMARK_OBS_REQUIRED_COLUMNS = (
    "cell_id",
    "cell_type",
    "sample_id",
    "donor_id",
    "batch_id",
)

SPLIT_MANIFEST_COLUMNS = ("cell_id", "split_domain", "split_seed")

CONDITION_MANIFEST_COLUMNS = (
    "condition_id",
    "removed_state",
    "removes_state_from_reference",
)

MODEL_VISIBLE_CELLS_COLUMNS = (
    "cell_id",
    "domain",
    "sample_id",
    "donor_id",
    "batch_id",
    "condition_id",
    "split_seed",
)

TARGET_LABELS_COLUMNS = ("cell_id", "target_label", "broad_label")

QUERY_TRUTH_COLUMNS = (
    "cell_id",
    "true_label",
    "removed_state",
    "is_absent_state",
    "is_shared_state",
)

STATE_PRESENCE_COLUMNS = ("cell_type", "present_in_query", "present_in_reference")

BROAD_ANCHOR_PRIORS_COLUMNS = ("cell_id", "broad_anchor_class", "anchor_confidence")

INITIAL_PRIORS_COLUMNS = ("cell_id", "rho", "prior_risk", "prior_source")

SOURCE_PRIORS_COLUMNS = (
    "cell_id",
    "rho",
    "rho_source",
    "rho_recipe",
    "prior_risk",
    "pmax_reference_classifier",
    "anchor_class_pred",
    "anchor_confidence",
)

TARGET_PRIORS_COLUMNS = ("cell_id", "target_label_visible", "broad_anchor_class", "rho_target")

CELL_TRANSPORT_SCORES_COLUMNS = (
    "cell_id",
    "condition_id",
    "method",
    "domain",
    "a",
    "a_hat",
    "u",
    "e",
    "rho",
    "tau_source",
    "max_label_probability",
    "label_entropy",
    "forced_label",
    "hub_exposure",
    "nn_distance",
)

SPARSE_COUPLING_COLUMNS = (
    "source_cell_id",
    "target_cell_id",
    "source_index",
    "target_index",
    "coupling",
)

PRIOR_ONLY_SCORES_COLUMNS = (
    "cell_id",
    "condition_id",
    "method",
    "domain",
    "rho",
    "prior_risk",
)

CELL_SCORES_COLUMNS = (
    "cell_id",
    "condition_id",
    "method",
    "u",
    "u_tilde",
    "prior_risk",
    "e",
    "hub_exposure",
    "max_label_probability",
    "label_uncertainty",
    "label_entropy",
    "forced_label",
    "nn_distance",
    "abstain_u",
    "abstain_u_or_entropy",
    "final_label_abstention_aware",
)

ABSTENTION_CALLS_COLUMNS = (
    "cell_id",
    "condition_id",
    "method",
    "abstain_u",
    "abstain_u_or_entropy",
    "final_label_abstention_aware",
)

EVALUATION_METRICS_COLUMNS = (
    "condition_id",
    "candidate_set",
    "method",
    "score",
    "metric",
    "value",
)

FORCED_LABEL_SUMMARY_COLUMNS = (
    "condition_id",
    "candidate_set",
    "method",
    "subset",
    "forced_label",
    "n_cells",
    "mean_max_label_probability",
    "abstention_rate",
)

MODEL_VISIBLE_FORBIDDEN_COLUMNS = (
    "cell_type",
    "true_label",
    "removed_state",
    "is_absent_state",
    "is_shared_state",
)

EMBEDDING_CELLS_COLUMNS = ("row_index", "cell_id", "domain", "condition_id")
