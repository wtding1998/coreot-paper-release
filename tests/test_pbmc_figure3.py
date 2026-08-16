from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.text import Text

from coreot.results.hiha_figure2 import (
    METHOD_COLORS as FIGURE2_METHOD_COLORS,
)
from coreot.results.hiha_figure2 import (
    PANEL_E_CONTEXT_GRAY as FIGURE2_PANEL_E_CONTEXT_GRAY,
)
from coreot.results.hiha_figure2 import (
    PANEL_E_ELIGIBLE_GRAY as FIGURE2_PANEL_E_ELIGIBLE_GRAY,
)
from coreot.results.hiha_figure2 import (
    PANEL_E_TRUTH_COLOR as FIGURE2_PANEL_E_TRUTH_COLOR,
)
from coreot.results.hiha_figure2 import PANEL_D_METRICS as FIGURE2_PANEL_D_METRICS
from coreot.results.pbmc_figure3 import (
    ENDPOINTS,
    METHOD_COLORS,
    PANEL_B_AP_DISPLAY_LIMITS,
    PANEL_B_AP_LIMITS,
    PANEL_C_PROBABILITY_DISPLAY_LIMITS,
    PANEL_C_CONTRASTS,
    PANEL_E_CONTEXT_GRAY,
    PANEL_E_ELIGIBLE_GRAY,
    PANEL_E_TRUTH_COLOR,
    PANEL_D_METRICS,
    PANEL_F_METHODS,
    SEEDS,
    _conditional_destination_table,
    _forced_destinations,
    _panel_crop_bbox_inches,
    compute_within_celltype_metrics,
)
from coreot.results.grid_common import _represented_truth_labels
from coreot.results.pbmc_figure3_composite import (
    MAIN_FIGURE_MIN_FONT_SIZE,
    PANEL_SET,
    draw_main_figure,
    generate_panel_images,
    read_composite_sources,
)


def test_shared_figure2_and_figure3_colors_are_identical() -> None:
    assert METHOD_COLORS == FIGURE2_METHOD_COLORS
    assert PANEL_E_TRUTH_COLOR == FIGURE2_PANEL_E_TRUTH_COLOR
    assert PANEL_E_ELIGIBLE_GRAY == FIGURE2_PANEL_E_ELIGIBLE_GRAY
    assert PANEL_E_CONTEXT_GRAY == FIGURE2_PANEL_E_CONTEXT_GRAY
    assert PANEL_D_METRICS == FIGURE2_PANEL_D_METRICS


def test_composite_uses_letter_aligned_panel_contract() -> None:
    sources = read_composite_sources()

    assert PANEL_SET == ("A", "B", "C", "D", "E", "F")
    assert set(sources["D"]["method"]) == {
        "coreot_full",
        "uniform_uot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    }
    assert set(sources["E"]["map_id"]) == {
        "truth",
        "coreot_full",
        "uniform_uot",
        "prior_only",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    }


def test_composite_uses_figure2_font_and_display_label_contract() -> None:
    fig = draw_main_figure(read_composite_sources())
    texts = [text for text in fig.findobj(match=Text) if text.get_text()]

    assert min(float(text.get_fontsize()) for text in texts) >= (
        MAIN_FIGURE_MIN_FONT_SIZE
    )
    assert any(text.get_text() == "Seurat" for text in texts)
    assert not any(text.get_text() == "Seurat anchor" for text in texts)
    assert any(
        text.get_text() == "Control-adjusted, $\\Delta^{\\mathrm{CA}}$"
        for text in texts
    )
    assert not any(text.get_text() == "Specificity, $R$" for text in texts)
    plt.close(fig)


def test_panel_b_uses_requested_truncated_ap_axis() -> None:
    fig = draw_main_figure(read_composite_sources())
    panel_b_axes = [
        axis
        for axis in fig.axes
        if axis.get_title() in {"B held out", "NK held out", "DC held out"}
        and axis.get_position().y0 > 0.8
    ]

    assert len(panel_b_axes) == 3
    assert PANEL_B_AP_LIMITS == (0.4, 1.0)
    assert all(
        axis.get_xlim() == PANEL_B_AP_DISPLAY_LIMITS for axis in panel_b_axes
    )
    probability_axes = [
        axis
        for axis in fig.axes
        if axis.get_xlabel() == "Restored-state\nprobability"
    ]
    assert len(probability_axes) == 1
    assert probability_axes[0].get_xlim() == PANEL_C_PROBABILITY_DISPLAY_LIMITS
    plt.close(fig)


def test_standalone_panels_are_result_side_review_artifacts(tmp_path) -> None:
    outputs = generate_panel_images(letters=("b",), panel_root=tmp_path)

    assert outputs["panel_b"] == tmp_path / "figure_3_pbmc_panel_b.png"
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"figure_3_pbmc_panel_b.{suffix}").is_file()


def test_represented_truth_labels_include_held_out_broad_celltype_controls() -> None:
    shared = pd.DataFrame(
        {
            "true_label": [
                "B cells",
                "B cells",
                "NK cells",
                "Dendritic cells",
            ]
        }
    )

    assert _represented_truth_labels(shared) == [
        "B cells",
        "Dendritic cells",
        "NK cells",
    ]


def test_within_celltype_metrics_exclude_unrelated_cell_types() -> None:
    truth = pd.DataFrame(
        {
            "cell_id": ["b_stim", "b_ctrl", "nk_stim", "nk_ctrl"],
            "true_label": ["B cells", "B cells", "NK cells", "NK cells"],
            "is_absent_state": [True, False, False, False],
        }
    )
    scores = pd.DataFrame(
        {
            "cell_id": ["nk_ctrl", "b_ctrl", "b_stim", "nk_stim"],
            "u": [0.99, 0.1, 0.9, 0.98],
        }
    )
    metadata = pd.DataFrame(
        {
            "cell_type": ["B cells", "B cells", "NK cells", "NK cells"],
            "condition": ["stim", "ctrl", "stim", "ctrl"],
        },
        index=["b_stim", "b_ctrl", "nk_stim", "nk_ctrl"],
    )

    actual = compute_within_celltype_metrics(
        truth,
        scores,
        metadata,
        endpoint="B cells",
        score_column="u",
    )

    assert actual["n_positive"] == 1
    assert actual["n_negative"] == 1
    assert actual["prevalence"] == 0.5
    assert actual["auprc"] == 1.0
    assert actual["auroc"] == 1.0


def test_conditional_destination_is_computed_per_cell() -> None:
    coupling = pd.DataFrame(
        {
            "source_cell_id": ["s1", "s1", "s2", "s2"],
            "target_cell_id": ["b1", "nk1", "b1", "nk1"],
            "coupling": [9.0, 1.0, 0.1, 0.9],
        }
    )
    metadata = pd.DataFrame(
        {
            "cell_type": ["B cells", "NK cells"],
            "condition": ["ctrl", "ctrl"],
        },
        index=["b1", "nk1"],
    )
    transported = pd.DataFrame(
        {
            "cell_id": ["s1", "s2"],
            "a": [10.0, 1.0],
            "a_hat": [10.0, 1.0],
            "u": [0.0, 0.0],
        }
    )

    per_cell, selected = _conditional_destination_table(
        coupling=coupling,
        source_ids={"s1", "s2"},
        metadata=metadata,
        transported=transported,
        numerator_mask=lambda frame: frame["target_cell_type"].eq("B cells"),
        probability_name="same_celltype_conditional_mass",
    )

    values = per_cell.set_index("cell_id")["same_celltype_conditional_mass"]
    np.testing.assert_allclose(values.loc["s1"], 0.9)
    np.testing.assert_allclose(values.loc["s2"], 0.1)
    np.testing.assert_allclose(values.median(), 0.5)
    assert len(selected) == 4


def test_forced_destination_names_are_descriptive_and_deterministic() -> None:
    selected = pd.DataFrame(
        {
            "source_cell_id": ["s1", "s1", "s2", "s2"],
            "target_cell_type": ["B cells", "B cells", "NK cells", "B cells"],
            "target_condition": ["ctrl", "stim", "ctrl", "ctrl"],
            "coupling": [0.6, 0.4, 0.5, 0.5],
        }
    )

    actual = _forced_destinations(selected).set_index("source_cell_id")

    assert actual.loc["s1", "condition_specific_destination"] == "B cells::ctrl"
    assert actual.loc["s1", "coupling_forced_cell_type"] == "B cells"
    assert actual.loc["s2", "condition_specific_destination"] == "B cells::ctrl"
    assert actual.loc["s2", "coupling_forced_cell_type"] == "B cells"
    assert ENDPOINTS == ("B cells", "NK cells", "Dendritic cells")
    assert SEEDS == (1, 2, 3, 4, 5)
    assert [column for column, _ in PANEL_C_CONTRASTS] == [
        "delta_full_match_only",
        "delta_match_only_uniform",
        "delta_full_prior",
    ]
    assert [method for method, _, _ in PANEL_F_METHODS] == [
        "coreot_full",
        "uniform_uot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    ]


def test_panel_crop_bbox_excludes_neighboring_axes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 2), gridspec_kw={"wspace": 0.8})
    axes[0].set(xlabel="Selected x", ylabel="Selected y", title="Selected")
    axes[1].set(xlabel="Neighbor x", ylabel="Neighbor y", title="Neighbor")
    fig.canvas.draw()

    selected_tight = axes[0].get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    )
    neighbor_tight = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    )
    crop = _panel_crop_bbox_inches(fig, [axes[0]], padding_inches=0.01)

    assert crop.x0 <= selected_tight.x0
    assert crop.x1 >= selected_tight.x1
    assert crop.x1 < neighbor_tight.x0
    plt.close(fig)
