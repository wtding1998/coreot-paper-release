from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from pathlib import Path
import pytest
import shutil
import yaml

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
from coreot.results.main_figure_external_baselines import (
    prepare_pbmc_candidate_sources,
    sha256_file,
)
from coreot.results.pbmc_figure3_composite import (
    MAIN_FIGURE_MIN_FONT_SIZE,
    PANEL_A_CONDITION_FONT_SIZE,
    PANEL_A_MIN_FONT_SIZE,
    PANEL_A_SMALL_TEXT_GIDS,
    PANEL_B_ENDPOINT_FONT_SIZE,
    PANEL_B_ENDPOINT_GIDS,
    PANEL_SPATIAL_ENDPOINT_FONT_SIZE,
    PANEL_SPATIAL_ENDPOINT_GIDS,
    PANEL_SET,
    _draw_panel_b_restoration,
    draw_main_figure,
    draw_panel_figure,
    generate_main_figure,
    generate_panel_images,
    _panel_bbox,
    read_composite_sources,
)


@pytest.fixture(scope="module")
def candidate_source_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pbmc_figure3_candidate")
    prepared = prepare_pbmc_candidate_sources(
        package_root=Path("results/uot_baseline_pilot/portable_decision_package_v1"),
        canonical_source_root=Path("results/PBMC/manuscript/figure3/figure_3_pbmc_source_data"),
        candidate_figure_root=root,
    )
    return prepared["source_root"]


def test_shared_figure2_and_figure3_colors_are_identical() -> None:
    assert METHOD_COLORS == FIGURE2_METHOD_COLORS
    assert PANEL_E_TRUTH_COLOR == FIGURE2_PANEL_E_TRUTH_COLOR
    assert PANEL_E_ELIGIBLE_GRAY == FIGURE2_PANEL_E_ELIGIBLE_GRAY
    assert PANEL_E_CONTEXT_GRAY == FIGURE2_PANEL_E_CONTEXT_GRAY
    assert PANEL_D_METRICS == FIGURE2_PANEL_D_METRICS


def test_composite_uses_letter_aligned_panel_contract(
    candidate_source_root: Path,
) -> None:
    sources = read_composite_sources(candidate_source_root)

    assert PANEL_SET == ("A", "B", "C", "D", "E", "F")
    assert set(sources["D"]["method"]) == {
        "coreot_full",
        "scdot",
        "tacco_ot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    }
    assert set(sources["E"]["map_id"]) == {
        "truth",
        "coreot_full",
        "scdot",
        "tacco_ot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    }


def test_composite_uses_figure2_font_and_display_label_contract(
    candidate_source_root: Path,
) -> None:
    fig = draw_main_figure(read_composite_sources(candidate_source_root))
    texts = [text for text in fig.findobj(match=Text) if text.get_text()]

    compact_labels = PANEL_A_SMALL_TEXT_GIDS | PANEL_B_ENDPOINT_GIDS | PANEL_SPATIAL_ENDPOINT_GIDS
    assert all(
        float(text.get_fontsize()) == PANEL_A_CONDITION_FONT_SIZE
        for text in texts
        if text.get_gid()
        in {"panel-a-reference-omitted-label", "panel-a-restored-reference-label"}
    )
    assert min(
        float(text.get_fontsize())
        for text in texts
        if text.get_gid() in PANEL_A_SMALL_TEXT_GIDS
    ) == pytest.approx(PANEL_A_MIN_FONT_SIZE)
    assert (
        min(float(text.get_fontsize()) for text in texts if text.get_gid() not in compact_labels)
        >= MAIN_FIGURE_MIN_FONT_SIZE
    )
    assert any(text.get_text() == "Seurat" for text in texts)
    assert not any(text.get_text() == "Seurat anchor" for text in texts)
    assert any(
        text.get_text() == "Control-adjusted $\\Delta^{\\mathrm{CA}}$" for text in texts
    )
    assert not any(text.get_text() == "Specificity, $R$" for text in texts)
    plt.close(fig)


def test_panels_c_and_d_use_zero_origin_metric_bars_without_split_points(
    candidate_source_root: Path,
) -> None:
    sources = read_composite_sources(candidate_source_root)
    fig = draw_main_figure(sources)
    metric_axes = [
        axis
        for axis in fig.axes
        if axis.get_title() in {"Stimulated B", "Stimulated NK", "Stimulated DC"}
        and np.isclose(axis.get_position().y0, 0.515)
    ]
    panel_c_axes = [axis for axis in metric_axes if axis.get_position().x0 < 0.6]
    panel_d_axes = [axis for axis in metric_axes if axis.get_position().x0 >= 0.6]

    assert len(panel_c_axes) == len(ENDPOINTS)
    assert len(panel_d_axes) == len(ENDPOINTS)
    assert all(
        axis.get_position(original=True).y0 == pytest.approx(0.515)
        and axis.get_position(original=True).height == pytest.approx(0.185)
        for axis in (*panel_c_axes, *panel_d_axes)
    )
    metric_value_labels = {
        round(text.get_position()[0], 2): text
        for text in fig.texts
        if text.get_text() == "Metric value"
    }
    assert set(metric_value_labels) == {0.32, 0.80}
    assert all(
        text.get_position()[1] == pytest.approx(0.49)
        for text in metric_value_labels.values()
    )
    assert panel_c_axes[0].get_yticklabels()[-2].get_text() == "scmap-\ncluster"
    scmap_label = panel_d_axes[0].get_yticklabels()[-2]
    assert scmap_label.get_text() == "scmap-\ncluster"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert scmap_label.get_window_extent(renderer).x0 > max(
        axis.get_window_extent(renderer).x1 for axis in panel_c_axes
    )
    for axes in (panel_c_axes, panel_d_axes):
        assert all(axis.get_xlim() == (0.0, 1.02) for axis in axes)
        assert all(np.allclose(axis.get_xticks(), (0.0, 0.5, 1.0)) for axis in axes)
        assert all(len(axis.patches) == len(PANEL_F_METHODS) * 2 for axis in axes)
        assert all(
            not any(isinstance(collection, PathCollection) for collection in axis.collections)
            for axis in axes
        )
        for axis in axes:
            gids = {patch.get_gid() for patch in axis.patches}
            assert sum(gid.endswith("-solid") for gid in gids) == len(PANEL_F_METHODS)
            assert sum(gid.endswith("-hollow") for gid in gids) == len(PANEL_F_METHODS)
    plt.close(fig)


def test_spatial_panels_share_columns_and_clear_panel_e_legend(
    candidate_source_root: Path,
) -> None:
    figure = draw_main_figure(read_composite_sources(candidate_source_root))
    expected_rows = {
        0.405: 0.045,
        0.3425: 0.045,
        0.280: 0.045,
        0.180: 0.050,
        0.1125: 0.050,
        0.045: 0.050,
    }
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
    for y_position, axes in row_axes.items():
        positions = [axis.get_position(original=True) for axis in axes]
        assert [position.x0 for position in positions] == pytest.approx(expected_x)
        assert [position.width for position in positions] == pytest.approx([0.128] * 7)
        assert [position.height for position in positions] == pytest.approx(
            [expected_rows[y_position]] * 7
        )

    labels = {
        text.get_gid(): text
        for text in figure.findobj(match=Text)
        if text.get_gid() in PANEL_SPATIAL_ENDPOINT_GIDS
    }
    assert set(labels) == PANEL_SPATIAL_ENDPOINT_GIDS
    assert all(text.get_position()[0] == pytest.approx(0.032) for text in labels.values())
    assert all(text.get_rotation() == pytest.approx(270.0) for text in labels.values())
    assert all(text.get_fontweight() == "normal" for text in labels.values())
    assert all(
        text.get_fontsize() == pytest.approx(PANEL_SPATIAL_ENDPOINT_FONT_SIZE)
        for text in labels.values()
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_e_legend = next(
        legend
        for legend in figure.legends
        if "Method-specific top-ranked $N_+$ cells"
        in [text.get_text() for text in legend.get_texts()]
    )
    legend_box = panel_e_legend.get_window_extent(renderer)
    panel_e_axes = [
        axis
        for y_position in (0.405, 0.3425, 0.280)
        for axis in row_axes[y_position]
    ]
    assert all(
        not legend_box.overlaps(axis.get_window_extent(renderer))
        for axis in panel_e_axes
    )
    plt.close(figure)


def test_panel_b_uses_one_zero_origin_bar_axis_without_split_points(
    candidate_source_root: Path,
) -> None:
    sources = read_composite_sources(candidate_source_root)
    panel_b = sources["C"]
    figure = plt.figure(figsize=(7.0, 9.0))
    _draw_panel_b_restoration(figure, panel_b)

    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_xlim() == (0.0, 1.02)
    assert np.allclose(axis.get_xticks(), (0.0, 0.5, 1.0))
    assert axis.get_xlabel() == "Metric value"
    assert len(axis.patches) == len(ENDPOINTS) * 4
    assert not any(
        isinstance(collection, PathCollection) for collection in axis.collections
    )
    data = panel_b.copy()
    data["heldout_delta_u"] = data["heldout_u_ablated"] - data["heldout_u_full"]
    data["control_delta_u"] = data["control_u_ablated"] - data["control_u_full"]
    expected_rows = (
        "heldout_delta_u",
        "control_delta_u",
        "restoration_specificity",
        "restored_state_conditional_probability",
    )
    patches = {patch.get_gid(): patch for patch in axis.patches}
    for endpoint in ENDPOINTS:
        endpoint_data = data.loc[data["endpoint"].eq(endpoint)]
        for column in expected_rows:
            gid = f"panel-b-bar-{endpoint}-{column}".replace(" ", "_")
            assert patches[gid].get_width() == pytest.approx(
                endpoint_data[column].mean()
            )
    plt.close(figure)


def test_panel_b_endpoint_labels_are_right_rotated_and_smaller(
    candidate_source_root: Path,
) -> None:
    sources = read_composite_sources(candidate_source_root)
    figure = draw_main_figure(sources)
    figure.canvas.draw()
    labels = [
        text
        for text in figure.findobj(match=Text)
        if text.get_gid() in PANEL_B_ENDPOINT_GIDS
    ]

    assert {label.get_text() for label in labels} == {
        "Stimulated B",
        "Stimulated NK",
        "Stimulated DC",
    }
    assert all(label.get_position()[0] == pytest.approx(0.975) for label in labels)
    assert all(label.get_rotation() == pytest.approx(270.0) for label in labels)
    assert all(
        label.get_fontsize() == pytest.approx(PANEL_B_ENDPOINT_FONT_SIZE)
        for label in labels
    )
    renderer = figure.canvas.get_renderer()
    assert all(
        figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1)
        for box in (label.get_window_extent(renderer) for label in labels)
    )
    boxes = [label.get_window_extent(renderer) for label in labels]
    assert all(
        not first.overlaps(second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
    )
    plt.close(figure)


def test_panel_b_row_labels_do_not_overlap_panel_a_schematic(
    candidate_source_root: Path,
) -> None:
    figure = draw_main_figure(read_composite_sources(candidate_source_root))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_a_axis = next(
        axis for axis in figure.axes if not axis.axison and axis.get_position().x0 < 0.1
    )
    panel_a_rightmost_box = max(
        (patch for patch in panel_a_axis.patches if isinstance(patch, FancyBboxPatch)),
        key=lambda patch: patch.get_x() + patch.get_width(),
    )
    panel_b_axis = next(
        axis
        for axis in figure.axes
        if len(axis.patches) == len(ENDPOINTS) * 4
        and axis.get_xlabel() == "Metric value"
    )

    assert all(
        label.get_window_extent(renderer).x0
        > panel_a_rightmost_box.get_window_extent(renderer).x1
        for label in panel_b_axis.get_yticklabels()
    )
    plt.close(figure)


def test_standalone_panels_are_result_side_review_artifacts(
    tmp_path: Path,
    candidate_source_root: Path,
) -> None:
    outputs = generate_panel_images(
        letters=("b",),
        panel_root=tmp_path,
        source_root=candidate_source_root,
    )

    assert outputs["panel_b"] == tmp_path / "figure_3_pbmc_panel_b.png"
    for suffix in ("png", "pdf", "tiff"):
        assert (tmp_path / f"figure_3_pbmc_panel_b.{suffix}").is_file()
    tiff = tmp_path / "figure_3_pbmc_panel_b.tiff"
    first_hash = sha256_file(tiff)
    assert not (tmp_path / "figure_3_pbmc_panel_b.svg").exists()

    generate_panel_images(
        letters=("b",),
        panel_root=tmp_path,
        source_root=candidate_source_root,
    )

    assert sha256_file(tiff) == first_hash


def test_composite_terminology_panel_a_containment_and_panel_d_contract(
    candidate_source_root: Path,
) -> None:
    fig = draw_main_figure(read_composite_sources(candidate_source_root))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(match=Text) if text.get_text()]
    visible = [text.get_text() for text in texts]

    assert "Reference-omitted\ncondition" in visible
    assert "Restored-reference\ncondition" in visible
    assert "AP" in visible
    assert "AUROC" in visible
    assert "Mean prevalence (AP)" in visible
    assert "Reference-omitted cells" in visible
    assert "Restored-state destination fraction" in visible
    assert "Method-specific top-ranked $N_+$ cells" in visible
    assert "Other ranking-cohort cells" in visible
    assert "Outside ranking cohort" in visible
    for panel_a_text in (
        "Separate scenarios",
        "Stimulated B | NK | DC",
        "Five donor splits",
        "Same query cells",
        "All states\nretained",
        "Same reference donors",
        "Stimulated cells omitted\nSame-type controls retained",
        "Stimulated cells restored\nSame-type controls retained",
    ):
        assert panel_a_text in visible
    assert "Evaluation labels" in visible
    assert "Evaluation truth" not in visible
    assert "Spatial localization of top-ranked cells" not in visible
    assert "Label transfer among represented states" not in visible
    for endpoint in ("Stimulated B", "Stimulated NK", "Stimulated DC"):
        assert endpoint in visible
    for retired_endpoint in (
        "B omitted state",
        "NK omitted state",
        "DC omitted state",
    ):
        assert retired_endpoint not in visible
    assert not any(
        any(
            retired in text
            for retired in (
                "Incomplete-reference",
                "incomplete-reference",
                "Average precision",
                "average precision",
                "held-out",
                "Held-out",
                "held out",
                "bars start at",
                "Conditional probability",
                "transported support",
                "Restored-state weight",
            )
        )
        for text in visible
    )

    panel_a_axis = next(
        axis for axis in fig.axes if not axis.axison and axis.get_position().x0 < 0.1
    )
    panel_a_boxes = [
        patch for patch in panel_a_axis.patches if isinstance(patch, FancyBboxPatch)
    ]
    assert len(panel_a_boxes) == 5
    query_box = next(
        patch
        for patch in fig.findobj(match=FancyBboxPatch)
        if patch.get_gid() == "panel-a-query-box"
    )
    omitted_box = next(
        patch
        for patch in fig.findobj(match=FancyBboxPatch)
        if patch.get_gid() == "panel-a-reference-omitted-box"
    )
    restored_box = next(
        patch
        for patch in fig.findobj(match=FancyBboxPatch)
        if patch.get_gid() == "panel-a-restored-reference-box"
    )
    assert omitted_box.get_width() == pytest.approx(restored_box.get_width())
    assert omitted_box.get_height() == pytest.approx(restored_box.get_height())
    assert query_box.get_width() < omitted_box.get_width()
    assert query_box.get_height() > omitted_box.get_height()
    source_box = next(
        patch
        for patch in fig.findobj(match=FancyBboxPatch)
        if patch.get_gid() == "panel-a-source-box"
    )
    split_box = next(
        patch
        for patch in fig.findobj(match=FancyBboxPatch)
        if patch.get_gid() == "panel-a-split-box"
    )
    assert source_box.get_width() == pytest.approx(split_box.get_width())
    assert source_box.get_x() == pytest.approx(split_box.get_x())
    assert source_box.get_y() - (split_box.get_y() + split_box.get_height()) >= 0.06
    label_boxes = [
        text
        for text in texts
        if text.get_gid() in {"panel-a-reference-omitted-label", "panel-a-restored-reference-label"}
    ]
    assert len(label_boxes) == 2
    for label in label_boxes:
        text_box = label.get_window_extent(renderer)
        enclosing = [
            patch.get_window_extent(renderer)
            for patch in (omitted_box, restored_box)
            if patch.get_window_extent(renderer).contains(text_box.x0, text_box.y0)
            and patch.get_window_extent(renderer).contains(text_box.x1, text_box.y1)
        ]
        assert len(enclosing) == 1

    panel_d_axes = [
        axis
        for axis in fig.axes
        if axis.get_xlim() == (0.0, 1.02) and axis.get_position().x0 >= 0.6
    ]
    assert len(panel_d_axes) == 3
    assert all(np.allclose(axis.get_xticks(), (0.0, 0.5, 1.0)) for axis in panel_d_axes)
    assert all(len(axis.patches) == len(PANEL_F_METHODS) * 2 for axis in panel_d_axes)
    assert all(
        not any(isinstance(collection, PathCollection) for collection in axis.collections)
        for axis in panel_d_axes
    )
    plt.close(fig)


def test_composite_panel_a_c_and_legends_have_public_bounds(
    candidate_source_root: Path,
) -> None:
    fig = draw_main_figure(read_composite_sources(candidate_source_root))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox

    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        assert box.x0 > figure_box.x0
        assert box.y0 > figure_box.y0
        assert box.x1 < figure_box.x1
        assert box.y1 < figure_box.y1

    texts = [text for text in fig.findobj(match=Text) if text.get_text()]
    panel_a_axis = next(
        axis for axis in fig.axes if not axis.axison and axis.get_position().x0 < 0.1
    )
    panel_a_boxes = [
        patch.get_window_extent(renderer)
        for patch in panel_a_axis.patches
        if isinstance(patch, FancyBboxPatch)
    ]
    boxed_labels = [
        text
        for text in panel_a_axis.findobj(match=Text)
        if text.get_gid() in PANEL_A_SMALL_TEXT_GIDS
        and text.get_gid() != "panel-a-reference-donor-label"
    ]
    assert boxed_labels
    for text in boxed_labels:
        text_box = text.get_window_extent(renderer)
        assert any(
            box.contains(text_box.x0, text_box.y0)
            and box.contains(text_box.x1, text_box.y1)
            for box in panel_a_boxes
        )

    panel_b_axis = next(
        axis
        for axis in fig.axes
        if len(axis.patches) == len(ENDPOINTS) * 4
        and axis.get_xlabel() == "Metric value"
    )
    xlabel_box = panel_b_axis.xaxis.label.get_window_extent(renderer)
    assert figure_box.contains(xlabel_box.x0, xlabel_box.y0)
    assert figure_box.contains(xlabel_box.x1, xlabel_box.y1)
    visible = {text.get_text() for text in texts}
    assert "Decrease in median\nquery-marginal\ndeficit" not in visible
    assert "Median restored-state\ntransported-label\nweight" not in visible
    plt.close(fig)


def test_main_renderer_records_portable_stable_provenance(
    tmp_path: Path,
    candidate_source_root: Path,
) -> None:
    source_root = tmp_path / "source_data"
    shutil.copytree(candidate_source_root, source_root)
    provenance = {
        "accepted_review_version": "review_v1",
        "generation_base_identity": {
            "kind": "accepted_review_contract",
            "path": "results/manuscript/accepted_review_v1_manifest.yaml",
            "sha256": "1" * 64,
        },
        "sealed_package": {
            "manifest": {
                "path": "results/sealed/manifest.json",
                "sha256": "2" * 64,
            },
            "verification_report": {
                "path": "results/sealed/verification/report.json",
                "sha256": "3" * 64,
            },
        },
    }
    paths = generate_main_figure(
        docs_root=tmp_path / "docs",
        result_root=tmp_path / "results/PBMC/manuscript/figure3",
        source_root=source_root,
        project_root=tmp_path,
        generation_provenance=provenance,
    )
    manifest = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
    serialized = yaml.safe_dump(manifest)

    assert "git_commit" not in serialized
    assert "repository_worktree_dirty" not in serialized
    assert (
        manifest["provenance"]["generation_base_identity"]
        == (provenance["generation_base_identity"])
    )
    renderer = manifest["provenance"]["renderer_source"]
    assert renderer["path"] == "src/coreot/results/pbmc_figure3_composite.py"
    assert len(renderer["sha256"]) == 64
    assert all(not Path(path).is_absolute() for path in manifest["sources"])
    assert all(
        not Path(path).is_absolute()
        for group in manifest["artifacts"].values()
        for path in group.values()
    )
    assert all(not path.endswith(".pdf") for path in manifest["output_sha256"])
    assert set(manifest["artifacts"]["manuscript"]) == {"png", "pdf", "tiff"}
    assert set(manifest["artifacts"]["result"]) == {"png", "pdf", "tiff"}
    assert not any(path.endswith(".svg") for path in manifest["sources"])
    assert manifest["container_metadata_policy"]["authoritative_hashed_formats"] == [
        "png",
        "tiff",
    ]
    assert manifest["container_metadata_policy"]["pdf_hashes_recorded"] is False
    assert (
        manifest["parameters"]["panel_b_endpoint_label_alignment"]
        == "right_edge_rotated_270"
    )
    assert manifest["parameters"]["panel_b_endpoint_font_size_points"] == pytest.approx(
        PANEL_B_ENDPOINT_FONT_SIZE
    )
    assert manifest["parameters"]["panel_b_axis_bounds_fraction"] == pytest.approx(
        [0.54, 0.805, 0.37, 0.15]
    )
    assert manifest["parameters"]["panel_b_encoding"] == {
        "rows": [
            "reference_omitted_cell_median_deficit_decrease",
            "same_type_control_median_deficit_decrease",
            "control_adjusted_median_deficit_decrease",
            "median_restored_state_destination_fraction",
        ],
        "bar": "arithmetic_mean_across_five_donor_splits",
        "whisker": "sample_sd_ddof1",
        "points": "not_rendered",
        "axis": [0.0, 1.0],
        "axis_label": "Metric value",
    }
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
    assert manifest["parameters"]["method_label_layout"] == {
        "panels": ["C", "D"],
        "scmap_cluster_lines": 2,
    }
    assert manifest["parameters"]["spatial_panel_layout"] == {
        "panels": ["E", "F"],
        "method_columns": "aligned",
        "axis_width_fraction": 0.128,
        "panel_e_axis_height_fraction": 0.045,
        "panel_f_axis_height_fraction": 0.050,
        "endpoint_labels": "left_border_rotated_270",
        "endpoint_label_x_fraction": 0.032,
        "endpoint_label_weight": "normal",
        "endpoint_label_font_size_points": PANEL_SPATIAL_ENDPOINT_FONT_SIZE,
        "panel_e_legend_y_fraction": 0.242,
    }
    assert manifest["parameters"]["metric_panel_vertical_alignment"] == {
        "panels": ["C", "D"],
        "axis_y_fraction": 0.515,
        "axis_height_fraction": 0.185,
        "axis_label_y_fraction": 0.49,
        "legend_y_fraction": 0.755,
    }


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
        "scdot",
        "tacco_ot",
        "seurat_anchor",
        "scmap_cluster",
        "chetah",
    ]


def test_panel_crop_bbox_excludes_neighboring_axes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 2), gridspec_kw={"wspace": 0.8})
    axes[0].set(xlabel="Selected x", ylabel="Selected y", title="Selected")
    axes[1].set(xlabel="Neighbor x", ylabel="Neighbor y", title="Neighbor")
    fig.canvas.draw()

    selected_tight = (
        axes[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    )
    neighbor_tight = (
        axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    )
    crop = _panel_crop_bbox_inches(fig, [axes[0]], padding_inches=0.01)

    assert crop.x0 <= selected_tight.x0
    assert crop.x1 >= selected_tight.x1
    assert crop.x1 < neighbor_tight.x0
    plt.close(fig)


def test_standalone_panel_c_crop_contains_all_artists(
    candidate_source_root: Path,
) -> None:
    fig = draw_panel_figure(read_composite_sources(candidate_source_root), "C")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop = _panel_bbox(fig, "C")
    padding = 0.01
    artist_boxes = [
        text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        for text in fig.findobj(match=Text)
        if text.get_visible() and text.get_text()
    ]
    artist_boxes.extend(
        axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for axis in fig.axes
        if axis.get_visible()
    )
    assert artist_boxes
    assert all(
        box.x0 > crop.x0 + padding
        and box.y0 > crop.y0 + padding
        and box.x1 < crop.x1 - padding
        and box.y1 < crop.y1 - padding
        for box in artist_boxes
    )
    plt.close(fig)
