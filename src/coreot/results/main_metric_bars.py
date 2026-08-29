from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FillMetricSpec:
    column: str
    label: str
    filled: bool


def fill_metric_handles(
    metrics: Sequence[FillMetricSpec],
    *,
    color: str,
) -> list[Patch]:
    return [
        Patch(
            facecolor=color if metric.filled else "white",
            edgecolor=color,
            linewidth=1.0,
            label=metric.label,
        )
        for metric in metrics
    ]


def draw_endpoint_metric_facet(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    endpoint_column: str,
    endpoint: str,
    endpoint_label: str,
    method_order: Sequence[str],
    method_labels: Mapping[str, str],
    method_colors: Mapping[str, str],
    metrics: Sequence[FillMetricSpec],
    ink: str,
    grid_color: str,
    show_method_labels: bool,
    prevalence_column: str | None = None,
    font_size: float = 6.5,
) -> None:
    local = frame.loc[frame[endpoint_column].eq(endpoint)]
    positions = np.arange(len(method_order), dtype=float)
    metric_offsets = np.linspace(-0.18, 0.18, len(metrics))

    if prevalence_column is not None:
        prevalence = float(local.groupby("seed")[prevalence_column].first().mean())
        ax.axvline(
            prevalence,
            color="#666666",
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.75,
            zorder=1,
        )

    for method_index, method in enumerate(method_order):
        method_frame = local.loc[local["method"].eq(method)].sort_values("seed")
        method_color = method_colors[method]
        for metric_index, metric in enumerate(metrics):
            values = method_frame[metric.column].to_numpy(dtype=float)
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            center = positions[method_index] + metric_offsets[metric_index]
            bars = ax.barh(
                center,
                mean,
                height=0.30,
                xerr=sample_sd,
                facecolor=method_color if metric.filled else "white",
                edgecolor=method_color,
                linewidth=0.9,
                error_kw={
                    "ecolor": ink,
                    "elinewidth": 0.65,
                    "capsize": 1.5,
                    "capthick": 0.65,
                },
                zorder=2,
            )
            for bar in bars.patches:
                fill = "solid" if metric.filled else "hollow"
                bar.set_gid(
                    f"metric-bar-{endpoint}-{method}-{metric.column}-{fill}".replace(
                        " ", "_"
                    )
                )

    ax.set(
        xlim=(0, 1.02),
        ylim=(len(method_order) - 0.55, -0.55),
        xticks=(0.0, 0.5, 1.0),
        yticks=positions,
        yticklabels=[method_labels[method] for method in method_order]
        if show_method_labels
        else [""] * len(method_order),
        title=endpoint_label,
    )
    ax.title.set_fontsize(font_size)
    ax.title.set_fontweight("normal")
    ax.tick_params(axis="x", labelsize=font_size, length=2.0, pad=1.5)
    ax.tick_params(axis="y", labelsize=font_size, length=0, pad=2.5)
    ax.grid(axis="x", color=grid_color, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
