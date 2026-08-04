from __future__ import annotations

"""Generate manuscript tables directly from canonical paper artifacts.

The script reads canonical three-seed summaries and generates validated
machine-readable CSV plus LaTeX and Markdown renderings for:

* Table 1: Task1 mean per-map RMSE across sparse-anchor rates;
* Table 2: Task1 mean per-map MAE and P90 absolute error across sparse rates;
* Table 3: frozen Task1-to-Task2 and Task1-to-Task3 transfer at 1% anchors;
* Table 4: external measured 3.5 GHz grouped-protocol RMSE;
* Table 5: external measured 3.5 GHz few-shot RMSE.

No numerical value is entered manually.
"""

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RATES = (0.005, 0.01, 0.05, 0.10)
METHODS = (
    "wallpath_extra",
    "direct_rf_all_features",
    "direct_extra_all_features",
    "multi_wall_residual_idw",
    "idw",
    "multi_wall",
    "log_distance",
)
METHOD_LABELS = {
    "wallpath_extra": "WallPath-PI",
    "wallpath_rf": "WallPath-RF",
    "wallpath_calibrated": "WallPath-Calibrated",
    "direct_rf_all_features": "Direct RF",
    "direct_extra_all_features": "Direct ExtraTrees",
    "multi_wall_residual_idw": "Multi-wall residual-IDW",
    "idw": "IDW",
    "multi_wall": "Multi-wall",
    "log_distance": "Log-distance",
    "wallpath_residual_extra": "WallPath-PI",
    "wallpath_residual_rf": "WallPath-RF",
    "direct_rf": "Direct RF",
    "direct_extra": "Direct ExtraTrees",
    "multi_wall_linear": "Multi-wall",
}
TABLE1_EXPECTED_WINNER = "direct_extra_all_features"
TABLE2_EXPECTED_WINNERS = {
    "mae": {
        0.005: "direct_rf_all_features",
        0.01: "direct_rf_all_features",
        0.05: "wallpath_extra",
        0.10: "wallpath_extra",
    },
    "p90_ae": {
        0.005: "direct_extra_all_features",
        0.01: "direct_rf_all_features",
        0.05: "wallpath_extra",
        0.10: "wallpath_extra",
    },
}
TABLE1_STEM = "table1_task1_mean_per_map_rmse"
TABLE2_STEM = "table2_task1_mean_per_map_mae_p90"
TABLE3_STEM = "table3_frozen_transfer_main_metrics"
TRANSFER_RATE = 0.01
TRANSFER_TASKS = ("task2", "task3")
TRANSFER_METRICS = ("rmse", "mae", "p90_ae")
TRANSFER_METHODS = (
    "wallpath_extra",
    "wallpath_rf",
    "wallpath_calibrated",
    "direct_rf_all_features",
    "direct_extra_all_features",
    "multi_wall_residual_idw",
    "idw",
    "multi_wall",
    "log_distance",
)
TABLE3_EXPECTED_WINNER = "wallpath_extra"

TABLE4_STEM = "table4_external_3p5ghz_grouped_rmse"
EXTERNAL_METHODS = (
    "wallpath_residual_extra",
    "wallpath_residual_rf",
    "direct_rf",
    "direct_extra",
    "multi_wall_linear",
    "log_distance",
)
EXTERNAL_CONDITIONS = (
    "random_pooled",
    "leave_config_pooled",
    "leave_scenario_pooled",
    "leave_scenario_macro",
)
TABLE4_EXPECTED_WINNERS = {
    "random_pooled": "direct_rf",
    "leave_config_pooled": "wallpath_residual_rf",
    "leave_scenario_pooled": "multi_wall_linear",
    "leave_scenario_macro": "multi_wall_linear",
}

TABLE5_STEM = "table5_external_3p5ghz_fewshot_rmse"
FEWSHOT_RATES = (0.01, 0.05, 0.10)
FEWSHOT_METHODS = (
    "wallpath_fewshot_residual_extra",
    "wallpath_fewshot_residual_rf",
    "direct_rf_with_anchors",
    "direct_extra_with_anchors",
    "bias_calibrated_multi_wall",
    "bias_calibrated_log_distance",
    "multi_wall_linear",
    "log_distance",
)
FEWSHOT_METHOD_LABELS = {
    "wallpath_fewshot_residual_extra": "WallPath-PI + anchors",
    "wallpath_fewshot_residual_rf": "WallPath-RF + anchors",
    "direct_rf_with_anchors": "Direct RF + anchors",
    "direct_extra_with_anchors": "Direct ExtraTrees + anchors",
    "bias_calibrated_multi_wall": "Bias-calibrated multi-wall",
    "bias_calibrated_log_distance": "Bias-calibrated log-distance",
    "multi_wall_linear": "Multi-wall (zero-shot)",
    "log_distance": "Log-distance (zero-shot)",
}
TABLE5_EXPECTED_WINNERS = {
    0.01: "bias_calibrated_multi_wall",
    0.05: "direct_extra_with_anchors",
    0.10: "direct_extra_with_anchors",
}


def rate_token(rate: float) -> str:
    mapping = {0.005: "0p005", 0.01: "0p010", 0.05: "0p050", 0.10: "0p100"}
    for known, token in mapping.items():
        if math.isclose(rate, known, rel_tol=0.0, abs_tol=1e-12):
            return token
    raise ValueError(f"Unsupported sparse rate: {rate}")


def rate_label(rate: float) -> str:
    return f"{rate:.3f}"


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Task1 canonical CSV is missing required columns: {missing}")


def _selected_contract(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame["method"].isin(METHODS)
        & frame["sparse_rate"].apply(
            lambda value: any(
                math.isclose(float(value), rate, rel_tol=0.0, abs_tol=1e-12)
                for rate in RATES
            )
        )
    ].copy()

    expected_pairs = {(rate, method) for rate in RATES for method in METHODS}
    actual_pairs: set[tuple[float, str]] = set()
    for row in selected.itertuples(index=False):
        matched_rate = next(
            rate
            for rate in RATES
            if math.isclose(float(row.sparse_rate), rate, rel_tol=0.0, abs_tol=1e-12)
        )
        pair = (matched_rate, str(row.method))
        if pair in actual_pairs:
            raise ValueError(f"Duplicate Task1 table row for rate/method: {pair}")
        actual_pairs.add(pair)

    missing_pairs = sorted(expected_pairs - actual_pairs)
    unexpected_pairs = sorted(actual_pairs - expected_pairs)
    if missing_pairs or unexpected_pairs:
        raise ValueError(
            "Task1 table source contract failed: "
            f"missing={missing_pairs}, unexpected={unexpected_pairs}"
        )
    if len(selected) != len(expected_pairs):
        raise ValueError(
            f"Expected {len(expected_pairs)} selected rows, found {len(selected)}"
        )
    return selected


def load_task1_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "sparse_rate",
        "method",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "p90_ae_mean",
        "p90_ae_std",
    ]
    _require_columns(frame, required)
    frame = frame.copy()
    numeric = [column for column in required if column not in {"method"}]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    selected = _selected_contract(frame)
    metric_columns = [
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "p90_ae_mean",
        "p90_ae_std",
    ]
    if not np.isfinite(selected[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Task1 tables contain non-finite metric values.")
    for column in ("rmse_std", "mae_std", "p90_ae_std"):
        if (selected[column] < 0).any():
            raise ValueError(f"Task1 tables contain a negative value in {column}.")
    return selected


# Backward-compatible name retained for existing callers/tests.
def load_table1_source(path: Path) -> pd.DataFrame:
    return load_task1_source(path)


def _validate_fixed_transfer_rate(path: Path, *, task: str) -> None:
    frame = pd.read_csv(path, usecols=["sparse_rate"])
    rates = pd.to_numeric(frame["sparse_rate"], errors="raise")
    unique_rates = sorted({float(value) for value in rates})
    if len(unique_rates) != 1 or not math.isclose(
        unique_rates[0], TRANSFER_RATE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{task} transfer source must contain only sparse_rate={TRANSFER_RATE}; "
            f"found {unique_rates}"
        )


def _load_transfer_summary(path: Path, *, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["method"]
    for metric in TRANSFER_METRICS:
        required.extend([f"{metric}_mean", f"{metric}_std"])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{task} transfer summary is missing columns: {missing}")

    frame = frame.copy()
    if "sparse_rate" in frame.columns:
        frame["sparse_rate"] = pd.to_numeric(frame["sparse_rate"], errors="raise")
        rates = sorted({float(value) for value in frame["sparse_rate"]})
        if len(rates) != 1 or not math.isclose(
            rates[0], TRANSFER_RATE, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{task} transfer summary must contain only "
                f"sparse_rate={TRANSFER_RATE}; found {rates}"
            )

    selected = frame[frame["method"].isin(TRANSFER_METHODS)].copy()
    counts = selected["method"].value_counts()
    duplicates = sorted(counts[counts > 1].index.tolist())
    if duplicates:
        raise ValueError(f"Duplicate {task} transfer rows for methods: {duplicates}")

    actual_methods = set(selected["method"].astype(str))
    expected_methods = set(TRANSFER_METHODS)
    missing_methods = sorted(expected_methods - actual_methods)
    if missing_methods or len(selected) != len(TRANSFER_METHODS):
        raise ValueError(
            f"{task} transfer table source contract failed: "
            f"missing_methods={missing_methods}, rows={len(selected)}"
        )

    metric_columns = []
    for metric in TRANSFER_METRICS:
        for suffix in ("mean", "std"):
            column = f"{metric}_{suffix}"
            selected[column] = pd.to_numeric(selected[column], errors="raise")
            metric_columns.append(column)
    if not np.isfinite(selected[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{task} transfer summary contains non-finite values.")
    for metric in TRANSFER_METRICS:
        column = f"{metric}_std"
        if (selected[column] < 0).any():
            raise ValueError(f"{task} transfer summary contains negative {column}.")

    selected["task_id"] = task
    return selected


def load_table3_sources(
    task2_summary: Path,
    task3_summary: Path,
    task2_all_results: Path,
    task3_all_results: Path,
) -> pd.DataFrame:
    _validate_fixed_transfer_rate(task2_all_results, task="task2")
    _validate_fixed_transfer_rate(task3_all_results, task="task3")
    return pd.concat(
        [
            _load_transfer_summary(task2_summary, task="task2"),
            _load_transfer_summary(task3_summary, task="task3"),
        ],
        ignore_index=True,
    )


def _row_for(frame: pd.DataFrame, rate: float, method: str) -> pd.Series:
    rows = frame[
        (frame["method"] == method)
        & frame["sparse_rate"].apply(
            lambda value: math.isclose(
                float(value), rate, rel_tol=0.0, abs_tol=1e-12
            )
        )
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for sparse_rate={rate}, method={method}; found {len(rows)}"
        )
    return rows.iloc[0]


def _unique_winner(frame: pd.DataFrame, rate: float, mean_column: str) -> str:
    rows = pd.DataFrame([_row_for(frame, rate, method) for method in METHODS])
    minimum = float(rows[mean_column].min())
    winner_rows = rows[
        rows[mean_column].apply(
            lambda value: math.isclose(
                float(value), minimum, rel_tol=0.0, abs_tol=1e-12
            )
        )
    ]
    if len(winner_rows) != 1:
        raise ValueError(
            f"Expected a unique {mean_column} winner at sparse rate {rate}, found "
            f"{winner_rows['method'].tolist()}"
        )
    return str(winner_rows.iloc[0]["method"])


def validate_table1_winners(frame: pd.DataFrame) -> dict[float, str]:
    winners: dict[float, str] = {}
    for rate in RATES:
        winner = _unique_winner(frame, rate, "rmse_mean")
        if winner != TABLE1_EXPECTED_WINNER:
            raise ValueError(
                f"Unexpected Task1 RMSE winner at sparse rate {rate}: {winner}; "
                f"expected {TABLE1_EXPECTED_WINNER}"
            )
        winners[rate] = winner
    return winners


# Backward-compatible name retained for existing tests.
def validate_winners(frame: pd.DataFrame) -> dict[float, str]:
    return validate_table1_winners(frame)


def validate_table2_winners(frame: pd.DataFrame) -> dict[str, dict[float, str]]:
    winners: dict[str, dict[float, str]] = {"mae": {}, "p90_ae": {}}
    for metric in ("mae", "p90_ae"):
        for rate in RATES:
            winner = _unique_winner(frame, rate, f"{metric}_mean")
            expected = TABLE2_EXPECTED_WINNERS[metric][rate]
            if winner != expected:
                raise ValueError(
                    f"Unexpected Task1 {metric} winner at sparse rate {rate}: "
                    f"{winner}; expected {expected}"
                )
            winners[metric][rate] = winner
    return winners


def _transfer_row(frame: pd.DataFrame, task: str, method: str) -> pd.Series:
    rows = frame[
        (frame["task_id"] == task)
        & (frame["method"] == method)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one transfer row for task={task}, method={method}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def validate_table3_winners(
    frame: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    winners: dict[str, dict[str, str]] = {
        task: {} for task in TRANSFER_TASKS
    }
    for task in TRANSFER_TASKS:
        for metric in TRANSFER_METRICS:
            rows = pd.DataFrame(
                [_transfer_row(frame, task, method) for method in TRANSFER_METHODS]
            )
            mean_column = f"{metric}_mean"
            minimum = float(rows[mean_column].min())
            winner_rows = rows[
                rows[mean_column].apply(
                    lambda value: math.isclose(
                        float(value), minimum, rel_tol=0.0, abs_tol=1e-12
                    )
                )
            ]
            if len(winner_rows) != 1:
                raise ValueError(
                    f"Expected a unique {task} {metric} winner, found "
                    f"{winner_rows['method'].tolist()}"
                )
            winner = str(winner_rows.iloc[0]["method"])
            if winner != TABLE3_EXPECTED_WINNER:
                raise ValueError(
                    f"Unexpected {task} {metric} winner: {winner}; "
                    f"expected {TABLE3_EXPECTED_WINNER}"
                )
            winners[task][metric] = winner
    return winners


def build_table1_wide(frame: pd.DataFrame, winners: dict[float, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        output: dict[str, object] = {
            "method_id": method,
            "method": METHOD_LABELS[method],
        }
        for rate in RATES:
            source = _row_for(frame, rate, method)
            token = rate_token(rate)
            output[f"rmse_mean_{token}"] = float(source["rmse_mean"])
            output[f"rmse_std_{token}"] = float(source["rmse_std"])
            output[f"is_best_{token}"] = method == winners[rate]
        rows.append(output)
    return pd.DataFrame(rows)


# Backward-compatible name retained for existing tests.
def build_wide_table(frame: pd.DataFrame, winners: dict[float, str]) -> pd.DataFrame:
    return build_table1_wide(frame, winners)


def build_table2_wide(
    frame: pd.DataFrame,
    winners: dict[str, dict[float, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("mae", "p90_ae"):
        metric_label = "MAE" if metric == "mae" else "P90 absolute error"
        for method in METHODS:
            output: dict[str, object] = {
                "metric_id": metric,
                "metric": metric_label,
                "method_id": method,
                "method": METHOD_LABELS[method],
            }
            for rate in RATES:
                source = _row_for(frame, rate, method)
                token = rate_token(rate)
                output[f"mean_{token}"] = float(source[f"{metric}_mean"])
                output[f"std_{token}"] = float(source[f"{metric}_std"])
                output[f"is_best_{token}"] = method == winners[metric][rate]
            rows.append(output)
    return pd.DataFrame(rows)


def build_table3_wide(
    frame: pd.DataFrame,
    winners: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in TRANSFER_METHODS:
        output: dict[str, object] = {
            "method_id": method,
            "method": METHOD_LABELS[method],
        }
        for task in TRANSFER_TASKS:
            source = _transfer_row(frame, task, method)
            for metric in TRANSFER_METRICS:
                output[f"{task}_{metric}_mean"] = float(source[f"{metric}_mean"])
                output[f"{task}_{metric}_std"] = float(source[f"{metric}_std"])
                output[f"{task}_{metric}_is_best"] = (
                    method == winners[task][metric]
                )
        rows.append(output)
    return pd.DataFrame(rows)


def display_value(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def latex_value(mean: float, std: float, *, bold: bool) -> str:
    value = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{value}}}" if bold else value


def markdown_value(mean: float, std: float, *, bold: bool) -> str:
    value = display_value(mean, std)
    return f"**{value}**" if bold else value


def render_table1_markdown(wide: pd.DataFrame) -> str:
    headers = ["Method", *[rate_label(rate) for rate in RATES]]
    lines = [
        "# Table 1: Task1 mean per-map RMSE",
        "",
        (
            "Scene-disjoint evaluation on held-out buildings B21-B25. Values are "
            "mean ± standard deviation across sparse-anchor seeds 11, 22, and 33; "
            "lower is better. The best result at each sparse rate is bold."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * len(RATES))]) + " |",
    ]
    for row in wide.itertuples(index=False):
        values = [str(row.method)]
        for rate in RATES:
            token = rate_token(rate)
            values.append(
                markdown_value(
                    float(getattr(row, f"rmse_mean_{token}")),
                    float(getattr(row, f"rmse_std_{token}")),
                    bold=bool(getattr(row, f"is_best_{token}")),
                )
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "**Metric:** RMSE averaged across evaluation maps (dB).", ""])
    return "\n".join(lines)


# Backward-compatible name retained for existing callers/tests.
def render_markdown(wide: pd.DataFrame) -> str:
    return render_table1_markdown(wide)


def render_table1_latex(wide: pd.DataFrame) -> str:
    column_spec = "l" + "c" * len(RATES)
    header = "Method & " + " & ".join(rate_label(rate) for rate in RATES) + r" \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task~1 scene-disjoint mean per-map RMSE (dB) on held-out buildings B21--B25. Values are mean $\pm$ standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result at each sparse rate is shown in bold.}",
        r"\label{tab:task1-rmse}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in wide.itertuples(index=False):
        values = []
        for rate in RATES:
            token = rate_token(rate)
            values.append(
                latex_value(
                    float(getattr(row, f"rmse_mean_{token}")),
                    float(getattr(row, f"rmse_std_{token}")),
                    bold=bool(getattr(row, f"is_best_{token}")),
                )
            )
        lines.append(f"{row.method} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


# Backward-compatible name retained for existing callers/tests.
def render_latex(wide: pd.DataFrame) -> str:
    return render_table1_latex(wide)


def render_table2_markdown(wide: pd.DataFrame) -> str:
    headers = ["Method", *[rate_label(rate) for rate in RATES]]
    lines = [
        "# Table 2: Task1 mean per-map MAE and P90 absolute error",
        "",
        (
            "Scene-disjoint evaluation on held-out buildings B21-B25. Values are "
            "mean ± standard deviation across sparse-anchor seeds 11, 22, and 33; "
            "lower is better. The best result for each metric and sparse rate is bold."
        ),
    ]
    for metric, heading in (("mae", "Panel A: MAE"), ("p90_ae", "Panel B: P90 absolute error")):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---", *(["---:"] * len(RATES))]) + " |",
            ]
        )
        panel = wide[wide["metric_id"] == metric]
        for row in panel.itertuples(index=False):
            values = [str(row.method)]
            for rate in RATES:
                token = rate_token(rate)
                values.append(
                    markdown_value(
                        float(getattr(row, f"mean_{token}")),
                        float(getattr(row, f"std_{token}")),
                        bold=bool(getattr(row, f"is_best_{token}")),
                    )
                )
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "**Metrics:** MAE and P90 absolute error averaged across evaluation maps (dB).",
            "",
        ]
    )
    return "\n".join(lines)


def render_table2_latex(wide: pd.DataFrame) -> str:
    column_spec = "l" + "c" * len(RATES)
    header = "Method & " + " & ".join(rate_label(rate) for rate in RATES) + r" \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task~1 scene-disjoint mean per-map MAE and P90 absolute error (dB) on held-out buildings B21--B25. Values are mean $\pm$ standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result for each metric and sparse rate is shown in bold.}",
        r"\label{tab:task1-mae-p90}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    panels = (("mae", "MAE"), ("p90_ae", "P90 absolute error"))
    for panel_index, (metric, label) in enumerate(panels):
        lines.append(rf"\multicolumn{{{1 + len(RATES)}}}{{l}}{{\textit{{{label}}}}} \\")
        panel = wide[wide["metric_id"] == metric]
        for row in panel.itertuples(index=False):
            values = []
            for rate in RATES:
                token = rate_token(rate)
                values.append(
                    latex_value(
                        float(getattr(row, f"mean_{token}")),
                        float(getattr(row, f"std_{token}")),
                        bold=bool(getattr(row, f"is_best_{token}")),
                    )
                )
            lines.append(f"{row.method} & " + " & ".join(values) + r" \\")
        if panel_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_table3_markdown(wide: pd.DataFrame) -> str:
    headers = [
        "Method",
        "Task 2 RMSE",
        "Task 2 MAE",
        "Task 2 P90",
        "Task 3 RMSE",
        "Task 3 MAE",
        "Task 3 P90",
    ]
    lines = [
        "# Table 3: Frozen transfer at 1% sparse anchors",
        "",
        (
            "Frozen Task1-to-Task2 and Task1-to-Task3 transfer at sparse rate "
            "0.01. Values are mean ± standard deviation across sparse-anchor "
            "seeds 11, 22, and 33; lower is better. The best result for each "
            "task and metric is bold."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * 6)]) + " |",
    ]
    for row in wide.itertuples(index=False):
        values = [str(row.method)]
        for task in TRANSFER_TASKS:
            for metric in TRANSFER_METRICS:
                values.append(
                    markdown_value(
                        float(getattr(row, f"{task}_{metric}_mean")),
                        float(getattr(row, f"{task}_{metric}_std")),
                        bold=bool(getattr(row, f"{task}_{metric}_is_best")),
                    )
                )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            (
                "**Metrics:** RMSE, MAE, and P90 absolute error averaged across "
                "evaluation maps (dB). Learned estimator weights remain frozen "
                "on the target tasks."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_table3_latex(wide: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen Task~1-to-Task~2 and Task~1-to-Task~3 transfer at 1\% sparse anchors. Values are mean $\pm$ standard deviation across sparse-anchor seeds 11, 22, and 33; lower is better. The best result for each task and metric is shown in bold.}",
        r"\label{tab:frozen-transfer-main}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Task 2} & \multicolumn{3}{c}{Task 3} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & RMSE & MAE & P90 & RMSE & MAE & P90 \\",
        r"\midrule",
    ]
    for row in wide.itertuples(index=False):
        values = []
        for task in TRANSFER_TASKS:
            for metric in TRANSFER_METRICS:
                values.append(
                    latex_value(
                        float(getattr(row, f"{task}_{metric}_mean")),
                        float(getattr(row, f"{task}_{metric}_std")),
                        bold=bool(getattr(row, f"{task}_{metric}_is_best")),
                    )
                )
        lines.append(f"{row.method} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)



def _require_external_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _validate_exact_method_set(
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    *,
    context: str,
) -> None:
    counts = frame["method"].astype(str).value_counts()
    duplicates = sorted(counts[counts > 1].index.tolist())
    actual = set(counts.index.tolist())
    expected = set(methods)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if duplicates or missing or unexpected or len(frame) != len(methods):
        raise ValueError(
            f"{context} method contract failed: duplicates={duplicates}, "
            f"missing={missing}, unexpected={unexpected}, rows={len(frame)}"
        )


def _coerce_external_metrics(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    output = frame.copy()
    metric_columns = list(columns)

    for column in metric_columns:
        output[column] = pd.to_numeric(output[column], errors="raise")

    if not np.isfinite(
        output[metric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{context} contains non-finite metric values.")

    for column in metric_columns:
        if column.endswith("_std") and (output[column] < 0).any():
            raise ValueError(f"{context} contains negative values in {column}.")

    return output


def load_table4_sources(
    random_source: Path,
    grouped_pooled_source: Path,
    grouped_macro_source: Path,
) -> pd.DataFrame:
    random_frame = pd.read_csv(random_source)
    _require_external_columns(
        random_frame,
        ["method", "rmse_mean", "rmse_std"],
        context="External random-split summary",
    )
    random_frame = random_frame[
        ["method", "rmse_mean", "rmse_std"]
    ].copy()
    _validate_exact_method_set(
        random_frame,
        EXTERNAL_METHODS,
        context="External random-split summary",
    )
    random_frame = _coerce_external_metrics(
        random_frame,
        ["rmse_mean", "rmse_std"],
        context="External random-split summary",
    )
    random_frame["condition_id"] = "random_pooled"

    pooled_frame = pd.read_csv(grouped_pooled_source)
    _require_external_columns(
        pooled_frame,
        ["protocol", "method", "rmse_mean", "rmse_std"],
        context="External grouped pooled summary",
    )
    pooled_frame = pooled_frame[
        ["protocol", "method", "rmse_mean", "rmse_std"]
    ].copy()

    expected_protocols = {"leave_config_out", "leave_scenario_out"}
    actual_protocols = set(pooled_frame["protocol"].astype(str))
    if actual_protocols != expected_protocols:
        raise ValueError(
            "External grouped pooled protocol contract failed: "
            f"expected={sorted(expected_protocols)}, "
            f"actual={sorted(actual_protocols)}"
        )

    expected_pairs = {
        (protocol, method)
        for protocol in expected_protocols
        for method in EXTERNAL_METHODS
    }
    actual_pairs = list(
        zip(
            pooled_frame["protocol"].astype(str),
            pooled_frame["method"].astype(str),
        )
    )
    duplicate_pairs = sorted(
        {
            pair
            for pair in actual_pairs
            if actual_pairs.count(pair) > 1
        }
    )
    missing_pairs = sorted(expected_pairs - set(actual_pairs))
    unexpected_pairs = sorted(set(actual_pairs) - expected_pairs)

    if (
        duplicate_pairs
        or missing_pairs
        or unexpected_pairs
        or len(pooled_frame) != len(expected_pairs)
    ):
        raise ValueError(
            "External grouped pooled source contract failed: "
            f"duplicates={duplicate_pairs}, missing={missing_pairs}, "
            f"unexpected={unexpected_pairs}, rows={len(pooled_frame)}"
        )

    pooled_frame = _coerce_external_metrics(
        pooled_frame,
        ["rmse_mean", "rmse_std"],
        context="External grouped pooled summary",
    )
    pooled_frame["condition_id"] = pooled_frame["protocol"].map(
        {
            "leave_config_out": "leave_config_pooled",
            "leave_scenario_out": "leave_scenario_pooled",
        }
    )

    macro_frame = pd.read_csv(grouped_macro_source)
    _require_external_columns(
        macro_frame,
        [
            "protocol",
            "method",
            "rmse_macro_mean",
            "rmse_macro_std",
        ],
        context="External grouped macro summary",
    )
    macro_frame = macro_frame[
        [
            "protocol",
            "method",
            "rmse_macro_mean",
            "rmse_macro_std",
        ]
    ].copy()

    actual_macro_protocols = set(macro_frame["protocol"].astype(str))
    if actual_macro_protocols != expected_protocols:
        raise ValueError(
            "External grouped macro protocol contract failed: "
            f"expected={sorted(expected_protocols)}, "
            f"actual={sorted(actual_macro_protocols)}"
        )

    macro_pairs = list(
        zip(
            macro_frame["protocol"].astype(str),
            macro_frame["method"].astype(str),
        )
    )
    duplicate_macro_pairs = sorted(
        {
            pair
            for pair in macro_pairs
            if macro_pairs.count(pair) > 1
        }
    )
    missing_macro_pairs = sorted(expected_pairs - set(macro_pairs))
    unexpected_macro_pairs = sorted(set(macro_pairs) - expected_pairs)

    if (
        duplicate_macro_pairs
        or missing_macro_pairs
        or unexpected_macro_pairs
        or len(macro_frame) != len(expected_pairs)
    ):
        raise ValueError(
            "External grouped macro source contract failed: "
            f"duplicates={duplicate_macro_pairs}, "
            f"missing={missing_macro_pairs}, "
            f"unexpected={unexpected_macro_pairs}, "
            f"rows={len(macro_frame)}"
        )

    macro_frame = _coerce_external_metrics(
        macro_frame,
        ["rmse_macro_mean", "rmse_macro_std"],
        context="External grouped macro summary",
    )
    macro_frame = macro_frame[
        macro_frame["protocol"] == "leave_scenario_out"
    ].copy()
    macro_frame = macro_frame.rename(
        columns={
            "rmse_macro_mean": "rmse_mean",
            "rmse_macro_std": "rmse_std",
        }
    )
    macro_frame["condition_id"] = "leave_scenario_macro"

    combined = pd.concat(
        [
            random_frame[
                ["condition_id", "method", "rmse_mean", "rmse_std"]
            ],
            pooled_frame[
                ["condition_id", "method", "rmse_mean", "rmse_std"]
            ],
            macro_frame[
                ["condition_id", "method", "rmse_mean", "rmse_std"]
            ],
        ],
        ignore_index=True,
    )

    expected_combined = {
        (condition, method)
        for condition in EXTERNAL_CONDITIONS
        for method in EXTERNAL_METHODS
    }
    actual_combined = set(
        zip(
            combined["condition_id"].astype(str),
            combined["method"].astype(str),
        )
    )
    if actual_combined != expected_combined:
        raise ValueError(
            "External Table 4 combined source contract failed: "
            f"missing={sorted(expected_combined - actual_combined)}, "
            f"unexpected={sorted(actual_combined - expected_combined)}"
        )

    return combined


def load_table5_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "anchor_fraction",
        "method",
        "rmse_mean",
        "rmse_std",
        "rank",
    ]
    _require_external_columns(
        frame,
        required,
        context="External few-shot pooled summary",
    )
    frame = frame[required].copy()
    frame["anchor_fraction"] = pd.to_numeric(
        frame["anchor_fraction"],
        errors="raise",
    )
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise")

    frame = _coerce_external_metrics(
        frame,
        ["rmse_mean", "rmse_std"],
        context="External few-shot pooled summary",
    )

    actual_rates = set(frame["anchor_fraction"].astype(float))
    expected_rates = set(FEWSHOT_RATES)
    if actual_rates != expected_rates:
        raise ValueError(
            "External few-shot anchor-fraction contract failed: "
            f"expected={sorted(expected_rates)}, actual={sorted(actual_rates)}"
        )

    expected_pairs = {
        (rate, method)
        for rate in FEWSHOT_RATES
        for method in FEWSHOT_METHODS
    }
    actual_pairs = list(
        zip(
            frame["anchor_fraction"].astype(float),
            frame["method"].astype(str),
        )
    )
    duplicate_pairs = sorted(
        {
            pair
            for pair in actual_pairs
            if actual_pairs.count(pair) > 1
        }
    )
    missing_pairs = sorted(expected_pairs - set(actual_pairs))
    unexpected_pairs = sorted(set(actual_pairs) - expected_pairs)

    if (
        duplicate_pairs
        or missing_pairs
        or unexpected_pairs
        or len(frame) != len(expected_pairs)
    ):
        raise ValueError(
            "External few-shot source contract failed: "
            f"duplicates={duplicate_pairs}, missing={missing_pairs}, "
            f"unexpected={unexpected_pairs}, rows={len(frame)}"
        )

    for rate in FEWSHOT_RATES:
        ranks = sorted(
            frame.loc[
                frame["anchor_fraction"].apply(
                    lambda value: math.isclose(
                        float(value),
                        rate,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
                "rank",
            ].astype(int)
        )
        if ranks != list(range(1, len(FEWSHOT_METHODS) + 1)):
            raise ValueError(
                f"External few-shot rank contract failed at {rate}: {ranks}"
            )

    return frame


def _external_condition_row(
    frame: pd.DataFrame,
    condition: str,
    method: str,
) -> pd.Series:
    rows = frame[
        (frame["condition_id"] == condition)
        & (frame["method"] == method)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one external grouped row for condition={condition}, "
            f"method={method}; found {len(rows)}"
        )
    return rows.iloc[0]


def validate_table4_winners(frame: pd.DataFrame) -> dict[str, str]:
    winners: dict[str, str] = {}

    for condition in EXTERNAL_CONDITIONS:
        rows = pd.DataFrame(
            [
                _external_condition_row(frame, condition, method)
                for method in EXTERNAL_METHODS
            ]
        )
        minimum = float(rows["rmse_mean"].min())
        winner_rows = rows[
            rows["rmse_mean"].apply(
                lambda value: math.isclose(
                    float(value),
                    minimum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ]
        if len(winner_rows) != 1:
            raise ValueError(
                f"Expected a unique external RMSE winner for {condition}; "
                f"found {winner_rows['method'].tolist()}"
            )

        winner = str(winner_rows.iloc[0]["method"])
        expected = TABLE4_EXPECTED_WINNERS[condition]
        if winner != expected:
            raise ValueError(
                f"Unexpected external RMSE winner for {condition}: "
                f"{winner}; expected {expected}"
            )
        winners[condition] = winner

    return winners


def build_table4_wide(
    frame: pd.DataFrame,
    winners: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for method in EXTERNAL_METHODS:
        output: dict[str, object] = {
            "method_id": method,
            "method": METHOD_LABELS[method],
        }
        for condition in EXTERNAL_CONDITIONS:
            source = _external_condition_row(frame, condition, method)
            output[f"{condition}_mean"] = float(source["rmse_mean"])
            output[f"{condition}_std"] = float(source["rmse_std"])
            output[f"{condition}_is_best"] = (
                method == winners[condition]
            )
        rows.append(output)

    return pd.DataFrame(rows)


def _fewshot_row(
    frame: pd.DataFrame,
    rate: float,
    method: str,
) -> pd.Series:
    rows = frame[
        (frame["method"] == method)
        & frame["anchor_fraction"].apply(
            lambda value: math.isclose(
                float(value),
                rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one external few-shot row for fraction={rate}, "
            f"method={method}; found {len(rows)}"
        )
    return rows.iloc[0]


def validate_table5_winners(frame: pd.DataFrame) -> dict[float, str]:
    winners: dict[float, str] = {}

    for rate in FEWSHOT_RATES:
        rows = pd.DataFrame(
            [_fewshot_row(frame, rate, method) for method in FEWSHOT_METHODS]
        )
        minimum = float(rows["rmse_mean"].min())
        winner_rows = rows[
            rows["rmse_mean"].apply(
                lambda value: math.isclose(
                    float(value),
                    minimum,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ]
        if len(winner_rows) != 1:
            raise ValueError(
                f"Expected a unique external few-shot winner at {rate}; "
                f"found {winner_rows['method'].tolist()}"
            )

        winner = str(winner_rows.iloc[0]["method"])
        expected = TABLE5_EXPECTED_WINNERS[rate]
        if winner != expected:
            raise ValueError(
                f"Unexpected external few-shot winner at {rate}: "
                f"{winner}; expected {expected}"
            )
        winners[rate] = winner

    return winners


def build_table5_wide(
    frame: pd.DataFrame,
    winners: dict[float, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for method in FEWSHOT_METHODS:
        output: dict[str, object] = {
            "method_id": method,
            "method": FEWSHOT_METHOD_LABELS[method],
        }
        for rate in FEWSHOT_RATES:
            source = _fewshot_row(frame, rate, method)
            token = rate_token(rate)
            output[f"rmse_mean_{token}"] = float(source["rmse_mean"])
            output[f"rmse_std_{token}"] = float(source["rmse_std"])
            output[f"is_best_{token}"] = method == winners[rate]
        rows.append(output)

    return pd.DataFrame(rows)


def render_table4_markdown(wide: pd.DataFrame) -> str:
    headers = [
        "Method",
        "Random pooled",
        "Leave-config pooled",
        "Leave-scenario pooled",
        "Leave-scenario macro",
    ]
    lines = [
        "# Table 4: External measured 3.5 GHz grouped evaluation",
        "",
        (
            "Pointwise RMSE in dB. Values are mean ± standard deviation "
            "across seeds 11, 22, and 33; lower is better. Pooled metrics "
            "weight measured points equally, while the scenario macro metric "
            "weights held-out scenarios equally."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:"]) + " |",
    ]

    for row in wide.itertuples(index=False):
        values = [str(row.method)]
        for condition in EXTERNAL_CONDITIONS:
            values.append(
                markdown_value(
                    float(getattr(row, f"{condition}_mean")),
                    float(getattr(row, f"{condition}_std")),
                    bold=bool(getattr(row, f"{condition}_is_best")),
                )
            )
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            (
                "Grouped-protocol standard deviations measure estimator "
                "randomness on fixed folds. The random-split variability also "
                "includes partition variation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_table4_latex(wide: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External measured 3.5~GHz pointwise RMSE (dB). Values are mean $\pm$ standard deviation across seeds 11, 22, and 33; lower is better. Pooled metrics weight measured points equally, whereas the scenario macro metric weights the three held-out scenarios equally.}",
        r"\label{tab:external-grouped-rmse}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& Random & Leave-configuration & \multicolumn{2}{c}{Leave-scenario} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-5}",
        r"Method & Pooled & Pooled & Pooled & Macro \\",
        r"\midrule",
    ]

    for row in wide.itertuples(index=False):
        values = []
        for condition in EXTERNAL_CONDITIONS:
            values.append(
                latex_value(
                    float(getattr(row, f"{condition}_mean")),
                    float(getattr(row, f"{condition}_std")),
                    bold=bool(getattr(row, f"{condition}_is_best")),
                )
            )
        lines.append(f"{row.method} & " + " & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_table5_markdown(wide: pd.DataFrame) -> str:
    headers = [
        "Method",
        "1% anchors",
        "5% anchors",
        "10% anchors",
    ]
    lines = [
        "# Table 5: External measured 3.5 GHz few-shot adaptation",
        "",
        (
            "Leave-scenario-out pooled RMSE in dB on non-anchor target "
            "observations. Values are mean ± standard deviation across "
            "anchor-selection seeds 11, 22, and 33 with model seed fixed at "
            "11; lower is better."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:"]) + " |",
    ]

    for row in wide.itertuples(index=False):
        values = [str(row.method)]
        for rate in FEWSHOT_RATES:
            token = rate_token(rate)
            values.append(
                markdown_value(
                    float(getattr(row, f"rmse_mean_{token}")),
                    float(getattr(row, f"rmse_std_{token}")),
                    bold=bool(getattr(row, f"is_best_{token}")),
                )
            )
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            (
                "Anchor fractions refer to measured rows in each held-out "
                "scenario. All target anchors are excluded from scoring."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_table5_latex(wide: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External measured 3.5~GHz leave-scenario-out few-shot pooled RMSE (dB) on non-anchor target observations. Values are mean $\pm$ standard deviation across anchor-selection seeds 11, 22, and 33 with model seed fixed at 11; lower is better.}",
        r"\label{tab:external-fewshot-rmse}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & 1\% anchors & 5\% anchors & 10\% anchors \\",
        r"\midrule",
    ]

    for row in wide.itertuples(index=False):
        values = []
        for rate in FEWSHOT_RATES:
            token = rate_token(rate)
            values.append(
                latex_value(
                    float(getattr(row, f"rmse_mean_{token}")),
                    float(getattr(row, f"rmse_std_{token}")),
                    bold=bool(getattr(row, f"is_best_{token}")),
                )
            )
        lines.append(f"{row.method} & " + " & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_table4(
    random_source: Path,
    grouped_pooled_source: Path,
    grouped_macro_source: Path,
    output_dir: Path,
) -> list[Path]:
    frame = load_table4_sources(
        random_source,
        grouped_pooled_source,
        grouped_macro_source,
    )
    winners = validate_table4_winners(frame)
    wide = build_table4_wide(frame, winners)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE4_STEM}.csv"
    tex_path = output_dir / f"{TABLE4_STEM}.tex"
    md_path = output_dir / f"{TABLE4_STEM}.md"

    wide.to_csv(csv_path, index=False, float_format="%.15g")
    tex_path.write_text(
        render_table4_latex(wide),
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_table4_markdown(wide),
        encoding="utf-8",
        newline="\n",
    )
    return [csv_path, tex_path, md_path]


def generate_table5(
    fewshot_source: Path,
    output_dir: Path,
) -> list[Path]:
    frame = load_table5_source(fewshot_source)
    winners = validate_table5_winners(frame)
    wide = build_table5_wide(frame, winners)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE5_STEM}.csv"
    tex_path = output_dir / f"{TABLE5_STEM}.tex"
    md_path = output_dir / f"{TABLE5_STEM}.md"

    wide.to_csv(csv_path, index=False, float_format="%.15g")
    tex_path.write_text(
        render_table5_latex(wide),
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_table5_markdown(wide),
        encoding="utf-8",
        newline="\n",
    )
    return [csv_path, tex_path, md_path]


def generate_table1_from_frame(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    winners = validate_table1_winners(frame)
    wide = build_table1_wide(frame, winners)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE1_STEM}.csv"
    tex_path = output_dir / f"{TABLE1_STEM}.tex"
    md_path = output_dir / f"{TABLE1_STEM}.md"
    wide.to_csv(csv_path, index=False, float_format="%.15g")
    tex_path.write_text(render_table1_latex(wide), encoding="utf-8", newline="\n")
    md_path.write_text(render_table1_markdown(wide), encoding="utf-8", newline="\n")
    return [csv_path, tex_path, md_path]


def generate_table2_from_frame(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    winners = validate_table2_winners(frame)
    wide = build_table2_wide(frame, winners)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE2_STEM}.csv"
    tex_path = output_dir / f"{TABLE2_STEM}.tex"
    md_path = output_dir / f"{TABLE2_STEM}.md"
    wide.to_csv(csv_path, index=False, float_format="%.15g")
    tex_path.write_text(render_table2_latex(wide), encoding="utf-8", newline="\n")
    md_path.write_text(render_table2_markdown(wide), encoding="utf-8", newline="\n")
    return [csv_path, tex_path, md_path]


def generate_table3_from_frame(
    frame: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    winners = validate_table3_winners(frame)
    wide = build_table3_wide(frame, winners)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE3_STEM}.csv"
    tex_path = output_dir / f"{TABLE3_STEM}.tex"
    md_path = output_dir / f"{TABLE3_STEM}.md"
    wide.to_csv(csv_path, index=False, float_format="%.15g")
    tex_path.write_text(render_table3_latex(wide), encoding="utf-8", newline="\n")
    md_path.write_text(render_table3_markdown(wide), encoding="utf-8", newline="\n")
    return [csv_path, tex_path, md_path]


def generate_table1(source: Path, output_dir: Path) -> list[Path]:
    return generate_table1_from_frame(load_task1_source(source), output_dir)


def generate_table2(source: Path, output_dir: Path) -> list[Path]:
    return generate_table2_from_frame(load_task1_source(source), output_dir)


def generate_table3(
    task2_summary: Path,
    task3_summary: Path,
    task2_all_results: Path,
    task3_all_results: Path,
    output_dir: Path,
) -> list[Path]:
    frame = load_table3_sources(
        task2_summary,
        task3_summary,
        task2_all_results,
        task3_all_results,
    )
    return generate_table3_from_frame(frame, output_dir)


def generate_all(
    source: Path,
    output_dir: Path,
    *,
    task2_summary: Path | None = None,
    task3_summary: Path | None = None,
    task2_all_results: Path | None = None,
    task3_all_results: Path | None = None,
    external_random: Path | None = None,
    external_grouped_pooled: Path | None = None,
    external_grouped_macro: Path | None = None,
    external_fewshot_pooled: Path | None = None,
) -> list[Path]:
    frame = load_task1_source(source)
    paths = [
        *generate_table1_from_frame(frame, output_dir),
        *generate_table2_from_frame(frame, output_dir),
    ]
    transfer_paths = (
        task2_summary,
        task3_summary,
        task2_all_results,
        task3_all_results,
    )
    if any(path is not None for path in transfer_paths):
        if not all(path is not None for path in transfer_paths):
            raise ValueError(
                "Table 3 generation requires both transfer summaries and both "
                "all-results files."
            )
        paths.extend(
            generate_table3(
                task2_summary,
                task3_summary,
                task2_all_results,
                task3_all_results,
                output_dir,
            )
        )

    external_paths = (
        external_random,
        external_grouped_pooled,
        external_grouped_macro,
        external_fewshot_pooled,
    )
    if any(path is not None for path in external_paths):
        if not all(path is not None for path in external_paths):
            raise ValueError(
                "External Table 4 and Table 5 generation requires the random, "
                "grouped pooled, grouped macro, and few-shot pooled sources."
            )
        paths.extend(
            generate_table4(
                external_random,
                external_grouped_pooled,
                external_grouped_macro,
                output_dir,
            )
        )
        paths.extend(
            generate_table5(
                external_fewshot_pooled,
                output_dir,
            )
        )

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-source", type=Path, required=True)
    parser.add_argument("--task2-summary", type=Path)
    parser.add_argument("--task3-summary", type=Path)
    parser.add_argument("--task2-all-results", type=Path)
    parser.add_argument("--task3-all-results", type=Path)
    parser.add_argument("--external-random", type=Path)
    parser.add_argument("--external-grouped-pooled", type=Path)
    parser.add_argument("--external-grouped-macro", type=Path)
    parser.add_argument("--external-fewshot-pooled", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = generate_all(
        args.task1_source,
        args.output_dir,
        task2_summary=args.task2_summary,
        task3_summary=args.task3_summary,
        task2_all_results=args.task2_all_results,
        task3_all_results=args.task3_all_results,
        external_random=args.external_random,
        external_grouped_pooled=args.external_grouped_pooled,
        external_grouped_macro=args.external_grouped_macro,
        external_fewshot_pooled=args.external_fewshot_pooled,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
