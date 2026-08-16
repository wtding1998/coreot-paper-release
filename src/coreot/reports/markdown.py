from __future__ import annotations

import math

import pandas as pd


def render_baseline_markdown_report(
    metrics: pd.DataFrame,
    forced_label_summary: pd.DataFrame,
    provenance: dict[str, object] | None = None,
) -> str:
    lines = [
        "# CoRe-OT Baseline Benchmark Report",
        "",
        "This pipeline audit report summarizes the currently implemented pilot outputs. It includes the configured transport baselines, UOT-style methods, calibrated abstention calls, evaluation metrics, and minimal provenance. It is not a manuscript-ready results report.",
        "",
        "## Missing-State Detection Metrics",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were available.")
    else:
        lines.extend(
            _markdown_table(
                metrics,
                ["condition_id", "candidate_set", "method", "score", "metric", "value"],
            )
        )

    lines.extend(["", "## Forced-Label Summary", ""])
    if forced_label_summary.empty:
        lines.append("No forced-label summaries were available.")
    else:
        lines.extend(
            _markdown_table(
                forced_label_summary,
                [
                    "condition_id",
                    "candidate_set",
                    "method",
                    "subset",
                    "forced_label",
                    "n_cells",
                    "mean_max_label_probability",
                    "abstention_rate",
                ],
            )
        )

    if provenance:
        lines.extend(["", "## Run Provenance", ""])
        lines.extend(
            _markdown_table(
                pd.DataFrame(
                    [
                        {"field": field, "value": value}
                        for field, value in sorted(provenance.items())
                    ]
                ),
                ["field", "value"],
            )
        )

    lines.extend(
        [
            "",
            "## Current Limitations",
            "",
            "- This report is an audit artifact for the executable pilot, not final manuscript prose.",
            "- `u_tilde` is a global cross-fitted prior-adjusted residual and remains a secondary diagnostic.",
            "- `coreot_full` uses the configured model-visible matchability prior when it is present; its provenance must be checked separately from this audit report.",
            "- Report-ready figures, HTML output, manuscript-scale summary tables, and multi-label/repeat aggregation are deferred.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    table = frame.loc[:, columns].copy()
    formatted = table.map(_format_value)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return [header, separator, *rows]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6g}"
    return str(value)
