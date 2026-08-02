#!/usr/bin/env python
"""Audit and summarize the six grouped external 3.5 GHz evaluations.

Expected runs
-------------
- leave_config_out, seeds 11/22/33
- leave_scenario_out, seeds 11/22/33

The script is analysis-only. It does not retrain models or alter the six runs.
It produces pooled, macro-across-fold, fold-specific, and paired pointwise
summaries while keeping the grouped protocols distinct.

Important interpretation
------------------------
The external dataset has only two configuration groups and three scenario
groups. Results are therefore descriptive secondary evidence. Seed variation
measures estimator randomness over the same observations; it is not variation
over independently sampled environments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEEDS = (11, 22, 33)
PROTOCOLS = ("leave_config_out", "leave_scenario_out")
METHODS = (
    "log_distance",
    "multi_wall_linear",
    "direct_rf",
    "direct_extra",
    "wallpath_residual_extra",
    "wallpath_residual_rf",
)
METRICS = (
    "rmse",
    "mae",
    "median_ae",
    "p90_ae",
    "p95_ae",
    "bias_db",
    "std_error_db",
)
COMPARISONS = (
    ("wallpath_residual_rf", "direct_rf"),
    ("wallpath_residual_extra", "direct_rf"),
    ("wallpath_residual_rf", "multi_wall_linear"),
    ("wallpath_residual_extra", "multi_wall_linear"),
)
EXPECTED_FOLDS = {
    "leave_config_out": ("holdout_C1", "holdout_C2"),
    "leave_scenario_out": (
        "holdout_Comms",
        "holdout_Library",
        "holdout_SSE",
    ),
}
EXPECTED_GROUP_COUNTS = {
    "leave_config_out": {
        "holdout_C1": 1168,
        "holdout_C2": 1120,
    },
    "leave_scenario_out": {
        "holdout_Comms": 1387,
        "holdout_Library": 687,
        "holdout_SSE": 214,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and summarize six grouped external 3.5 GHz runs."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root containing the six run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_3p5ghz_grouped_summary"),
        help="Directory for consolidated analysis outputs.",
    )
    return parser.parse_args()


def run_dir(root: Path, protocol: str, seed: int) -> Path:
    return root / f"external_3p5ghz_{protocol}_seed{seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def finite_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> bool:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            return False
    return True


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    display = frame.copy()
    display = display.replace({np.nan: ""})

    headers = [str(column) for column in display.columns]
    rows = [
        [str(value).replace("|", r"\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]

    def align(column: pd.Series) -> str:
        numeric = pd.api.types.is_numeric_dtype(column)
        return "---:" if numeric else "---"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(align(display[column]) for column in display.columns)
        + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric_mean_std(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: Iterable[str],
    *,
    ddof: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))
        row["num_seeds"] = int(group["random_seed"].nunique())

        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=ddof))
                if len(values) > 1
                else 0.0
            )
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())

        rows.append(row)

    return pd.DataFrame(rows)


def compute_error_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.percentile(absolute, 90)),
        "p95_ae": float(np.percentile(absolute, 95)),
        "bias_db": float(np.mean(error)),
        "std_error_db": float(np.std(error)),
    }


def improvement_percent(method_value: float, baseline_value: float) -> float:
    if not np.isfinite(baseline_value) or baseline_value == 0:
        return float("nan")
    return float((baseline_value - method_value) / baseline_value * 100.0)


def load_and_audit(
    results_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    audit_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    pooled_frames: list[pd.DataFrame] = []

    common_fingerprint_signature: str | None = None
    feature_signatures: dict[str, str] = {}
    summaries: dict[str, Any] = {}

    for protocol in PROTOCOLS:
        for seed in SEEDS:
            directory = run_dir(results_root, protocol, seed)
            required_paths = {
                "fold": directory / "final_evaluation_results.csv",
                "samples": directory / "per_sample_metrics.csv",
                "pooled": directory / "pooled_evaluation_results.csv",
                "summary": directory / "run_summary.json",
            }

            missing_files = [
                str(path)
                for path in required_paths.values()
                if not path.exists()
            ]
            if missing_files:
                raise FileNotFoundError(
                    "Missing required run outputs:\n"
                    + "\n".join(missing_files)
                )

            folds = pd.read_csv(required_paths["fold"])
            samples = pd.read_csv(required_paths["samples"])
            pooled = pd.read_csv(required_paths["pooled"])
            summary = json.loads(
                required_paths["summary"].read_text(encoding="utf-8")
            )

            require_columns(
                folds,
                [
                    "split",
                    "random_seed",
                    "fold",
                    "method",
                    "rmse",
                    "mae",
                    "p90_ae",
                    "train_count",
                    "test_count",
                ],
                label=str(required_paths["fold"]),
            )
            require_columns(
                samples,
                [
                    "row_uid",
                    "scenario",
                    "config",
                    "method",
                    "target_pl_db",
                    "pred_pl_db",
                    "error_db",
                    "abs_error_db",
                    "split",
                    "random_seed",
                    "fold",
                ],
                label=str(required_paths["samples"]),
            )
            require_columns(
                pooled,
                [
                    "split",
                    "method",
                    "rmse",
                    "mae",
                    "p90_ae",
                    "count",
                    "unique_test_points",
                ],
                label=str(required_paths["pooled"]),
            )

            expected_folds = set(EXPECTED_FOLDS[protocol])
            expected_methods = set(METHODS)
            expected_rows = len(expected_folds) * len(expected_methods)
            expected_sample_rows = 2288 * len(expected_methods)

            fingerprint_signature = stable_json(
                summary.get("file_fingerprints", [])
            )
            if common_fingerprint_signature is None:
                common_fingerprint_signature = fingerprint_signature
            fingerprints_match = (
                fingerprint_signature == common_fingerprint_signature
            )

            feature_signature = stable_json(
                summary.get("feature_columns", [])
            )
            if protocol not in feature_signatures:
                feature_signatures[protocol] = feature_signature
            features_match_within_protocol = (
                feature_signature == feature_signatures[protocol]
            )

            fold_test_counts = (
                folds[
                    ["fold", "test_count"]
                ]
                .drop_duplicates()
                .set_index("fold")["test_count"]
                .astype(int)
                .to_dict()
            )

            checks = {
                "summary_split": summary.get("split") == protocol,
                "summary_seed": summary.get("random_seed") == seed,
                "rows_after_cleaning": (
                    summary.get("rows_after_cleaning") == 2288
                ),
                "rows_removed_total": (
                    summary.get("rows_removed_total") == 5
                ),
                "methods_complete": set(folds["method"]) == expected_methods,
                "folds_complete": set(folds["fold"]) == expected_folds,
                "fold_result_row_count": len(folds) == expected_rows,
                "sample_row_count": len(samples) == expected_sample_rows,
                "pooled_row_count": len(pooled) == len(expected_methods),
                "fold_seed_column": (
                    set(folds["random_seed"].astype(int)) == {seed}
                ),
                "sample_seed_column": (
                    set(samples["random_seed"].astype(int)) == {seed}
                ),
                "fold_split_column": set(folds["split"]) == {protocol},
                "sample_split_column": set(samples["split"]) == {protocol},
                "pooled_split_column": set(pooled["split"]) == {protocol},
                "pooled_methods_complete": (
                    set(pooled["method"]) == expected_methods
                ),
                "finite_fold_metrics": finite_columns(
                    folds,
                    [column for column in METRICS if column in folds.columns],
                ),
                "finite_sample_values": finite_columns(
                    samples,
                    [
                        "target_pl_db",
                        "pred_pl_db",
                        "error_db",
                        "abs_error_db",
                    ],
                ),
                "finite_pooled_metrics": finite_columns(
                    pooled,
                    [column for column in METRICS if column in pooled.columns],
                ),
                "unique_predictions": not samples.duplicated(
                    subset=["row_uid", "method", "fold"]
                ).any(),
                "all_points_once_per_method": (
                    samples.groupby("method")["row_uid"].nunique().eq(2288).all()
                ),
                "test_counts_correct": (
                    fold_test_counts == EXPECTED_GROUP_COUNTS[protocol]
                ),
                "dataset_fingerprints_match": fingerprints_match,
                "features_match_within_protocol": (
                    features_match_within_protocol
                ),
            }

            status = "PASS" if all(checks.values()) else "FAIL"

            audit_row = {
                "protocol": protocol,
                "random_seed": seed,
                "run_dir": str(directory),
                "status": status,
                "fold_rows": int(len(folds)),
                "sample_rows": int(len(samples)),
                "pooled_rows": int(len(pooled)),
                "unique_test_points": int(samples["row_uid"].nunique()),
                "summary_sha256": sha256_file(required_paths["summary"]),
                "fold_csv_sha256": sha256_file(required_paths["fold"]),
                "samples_csv_sha256": sha256_file(required_paths["samples"]),
                "pooled_csv_sha256": sha256_file(required_paths["pooled"]),
            }
            audit_row.update(
                {f"check_{name}": bool(value) for name, value in checks.items()}
            )
            audit_rows.append(audit_row)

            if status != "PASS":
                failed = [name for name, value in checks.items() if not value]
                raise RuntimeError(
                    f"{directory}: audit failed: {failed}"
                )

            folds = folds.copy()
            folds["protocol"] = protocol
            folds["run_dir"] = str(directory)

            samples = samples.copy()
            samples["protocol"] = protocol
            samples["run_dir"] = str(directory)

            pooled = pooled.copy()
            pooled["protocol"] = protocol
            pooled["random_seed"] = seed
            pooled["run_dir"] = str(directory)

            fold_frames.append(folds)
            sample_frames.append(samples)
            pooled_frames.append(pooled)
            summaries[f"{protocol}|{seed}"] = summary

    audit = pd.DataFrame(audit_rows)
    all_folds = pd.concat(fold_frames, ignore_index=True)
    all_samples = pd.concat(sample_frames, ignore_index=True)
    all_pooled = pd.concat(pooled_frames, ignore_index=True)

    return audit, all_folds, all_samples, all_pooled, summaries


def build_macro_per_seed(all_folds: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in METRICS:
        if metric in all_folds.columns:
            aggregations[f"{metric}_macro"] = (metric, "mean")

    return (
        all_folds.groupby(
            ["protocol", "random_seed", "method"],
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(["protocol", "random_seed", "rmse_macro"])
        .reset_index(drop=True)
    )


def build_rankings(
    pooled_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for protocol, group in pooled_summary.groupby("protocol"):
        part = group.copy()
        part["summary_type"] = "pooled_point_level"
        part["rank"] = part["rmse_mean"].rank(
            method="min",
            ascending=True,
        ).astype(int)
        rows.append(part)

    macro_for_rank = macro_summary.rename(
        columns={
            "rmse_macro_mean": "rmse_mean",
            "rmse_macro_std": "rmse_std",
            "rmse_macro_min": "rmse_min",
            "rmse_macro_max": "rmse_max",
        }
    )
    for protocol, group in macro_for_rank.groupby("protocol"):
        part = group[
            [
                "protocol",
                "method",
                "num_seeds",
                "rmse_mean",
                "rmse_std",
                "rmse_min",
                "rmse_max",
            ]
        ].copy()
        part["summary_type"] = "macro_equal_fold"
        part["rank"] = part["rmse_mean"].rank(
            method="min",
            ascending=True,
        ).astype(int)
        rows.append(part)

    ranking = pd.concat(rows, ignore_index=True)
    return ranking.sort_values(
        ["protocol", "summary_type", "rank", "method"]
    ).reset_index(drop=True)


def build_comparisons(
    all_folds: pd.DataFrame,
    all_samples: pd.DataFrame,
    all_pooled: pd.DataFrame,
    macro_per_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_seed_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for protocol in PROTOCOLS:
        protocol_pooled = all_pooled[all_pooled["protocol"] == protocol]
        protocol_macro = macro_per_seed[
            macro_per_seed["protocol"] == protocol
        ]
        protocol_folds = all_folds[all_folds["protocol"] == protocol]
        protocol_samples = all_samples[
            all_samples["protocol"] == protocol
        ]

        for method, baseline in COMPARISONS:
            for seed in SEEDS:
                pooled_method = protocol_pooled[
                    (protocol_pooled["random_seed"] == seed)
                    & (protocol_pooled["method"] == method)
                ].iloc[0]
                pooled_baseline = protocol_pooled[
                    (protocol_pooled["random_seed"] == seed)
                    & (protocol_pooled["method"] == baseline)
                ].iloc[0]

                macro_method = protocol_macro[
                    (protocol_macro["random_seed"] == seed)
                    & (protocol_macro["method"] == method)
                ].iloc[0]
                macro_baseline = protocol_macro[
                    (protocol_macro["random_seed"] == seed)
                    & (protocol_macro["method"] == baseline)
                ].iloc[0]

                per_seed_rows.append(
                    {
                        "protocol": protocol,
                        "random_seed": seed,
                        "method": method,
                        "baseline": baseline,
                        "pooled_method_rmse": float(
                            pooled_method["rmse"]
                        ),
                        "pooled_baseline_rmse": float(
                            pooled_baseline["rmse"]
                        ),
                        "pooled_rmse_improvement_percent": (
                            improvement_percent(
                                float(pooled_method["rmse"]),
                                float(pooled_baseline["rmse"]),
                            )
                        ),
                        "pooled_method_mae": float(
                            pooled_method["mae"]
                        ),
                        "pooled_baseline_mae": float(
                            pooled_baseline["mae"]
                        ),
                        "pooled_mae_improvement_percent": (
                            improvement_percent(
                                float(pooled_method["mae"]),
                                float(pooled_baseline["mae"]),
                            )
                        ),
                        "macro_method_rmse": float(
                            macro_method["rmse_macro"]
                        ),
                        "macro_baseline_rmse": float(
                            macro_baseline["rmse_macro"]
                        ),
                        "macro_rmse_improvement_percent": (
                            improvement_percent(
                                float(macro_method["rmse_macro"]),
                                float(macro_baseline["rmse_macro"]),
                            )
                        ),
                    }
                )

                seed_folds = protocol_folds[
                    protocol_folds["random_seed"] == seed
                ]
                method_folds = seed_folds[
                    seed_folds["method"] == method
                ][["fold", "rmse", "mae"]].rename(
                    columns={
                        "rmse": "method_rmse",
                        "mae": "method_mae",
                    }
                )
                baseline_folds = seed_folds[
                    seed_folds["method"] == baseline
                ][["fold", "rmse", "mae"]].rename(
                    columns={
                        "rmse": "baseline_rmse",
                        "mae": "baseline_mae",
                    }
                )
                merged_folds = method_folds.merge(
                    baseline_folds,
                    on="fold",
                    how="inner",
                    validate="one_to_one",
                )
                for row in merged_folds.itertuples(index=False):
                    fold_rows.append(
                        {
                            "protocol": protocol,
                            "random_seed": seed,
                            "fold": row.fold,
                            "method": method,
                            "baseline": baseline,
                            "method_rmse": float(row.method_rmse),
                            "baseline_rmse": float(row.baseline_rmse),
                            "rmse_difference_db": float(
                                row.baseline_rmse - row.method_rmse
                            ),
                            "rmse_improvement_percent": (
                                improvement_percent(
                                    float(row.method_rmse),
                                    float(row.baseline_rmse),
                                )
                            ),
                            "rmse_winner": (
                                method
                                if row.method_rmse < row.baseline_rmse
                                else baseline
                                if row.baseline_rmse < row.method_rmse
                                else "tie"
                            ),
                            "method_mae": float(row.method_mae),
                            "baseline_mae": float(row.baseline_mae),
                            "mae_improvement_percent": (
                                improvement_percent(
                                    float(row.method_mae),
                                    float(row.baseline_mae),
                                )
                            ),
                        }
                    )

                seed_samples = protocol_samples[
                    protocol_samples["random_seed"] == seed
                ]
                method_samples = seed_samples[
                    seed_samples["method"] == method
                ][
                    [
                        "row_uid",
                        "fold",
                        "target_pl_db",
                        "pred_pl_db",
                        "abs_error_db",
                    ]
                ].rename(
                    columns={
                        "pred_pl_db": "method_prediction",
                        "abs_error_db": "method_abs_error",
                    }
                )
                baseline_samples = seed_samples[
                    seed_samples["method"] == baseline
                ][
                    [
                        "row_uid",
                        "fold",
                        "target_pl_db",
                        "pred_pl_db",
                        "abs_error_db",
                    ]
                ].rename(
                    columns={
                        "target_pl_db": "baseline_target",
                        "pred_pl_db": "baseline_prediction",
                        "abs_error_db": "baseline_abs_error",
                    }
                )
                paired = method_samples.merge(
                    baseline_samples,
                    on=["row_uid", "fold"],
                    how="inner",
                    validate="one_to_one",
                )
                if not np.allclose(
                    paired["target_pl_db"].to_numpy(float),
                    paired["baseline_target"].to_numpy(float),
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"Target mismatch for {protocol}, seed {seed}, "
                        f"{method} vs {baseline}"
                    )

                method_metrics = compute_error_metrics(
                    paired["target_pl_db"].to_numpy(float),
                    paired["method_prediction"].to_numpy(float),
                )
                baseline_metrics = compute_error_metrics(
                    paired["target_pl_db"].to_numpy(float),
                    paired["baseline_prediction"].to_numpy(float),
                )

                method_wins = int(
                    (
                        paired["method_abs_error"]
                        < paired["baseline_abs_error"]
                    ).sum()
                )
                baseline_wins = int(
                    (
                        paired["baseline_abs_error"]
                        < paired["method_abs_error"]
                    ).sum()
                )
                ties = int(len(paired) - method_wins - baseline_wins)

                point_rows.append(
                    {
                        "protocol": protocol,
                        "random_seed": seed,
                        "method": method,
                        "baseline": baseline,
                        "paired_points": int(len(paired)),
                        "method_rmse": method_metrics["rmse"],
                        "baseline_rmse": baseline_metrics["rmse"],
                        "rmse_improvement_percent": improvement_percent(
                            method_metrics["rmse"],
                            baseline_metrics["rmse"],
                        ),
                        "method_mae": method_metrics["mae"],
                        "baseline_mae": baseline_metrics["mae"],
                        "mae_improvement_percent": improvement_percent(
                            method_metrics["mae"],
                            baseline_metrics["mae"],
                        ),
                        "method_point_wins": method_wins,
                        "baseline_point_wins": baseline_wins,
                        "ties": ties,
                        "method_point_win_percent": float(
                            method_wins / len(paired) * 100.0
                        ),
                    }
                )

    return (
        pd.DataFrame(per_seed_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(point_rows),
    )


def summarize_comparisons(
    per_seed: pd.DataFrame,
    fold_level: pd.DataFrame,
    pointwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    keys = ["protocol", "method", "baseline"]
    for group_key, group in per_seed.groupby(keys, sort=True):
        protocol, method, baseline = group_key
        folds = fold_level[
            (fold_level["protocol"] == protocol)
            & (fold_level["method"] == method)
            & (fold_level["baseline"] == baseline)
        ]
        points = pointwise[
            (pointwise["protocol"] == protocol)
            & (pointwise["method"] == method)
            & (pointwise["baseline"] == baseline)
        ]

        fold_means = (
            folds.groupby("fold", as_index=False)
            .agg(
                method_rmse=("method_rmse", "mean"),
                baseline_rmse=("baseline_rmse", "mean"),
                improvement_percent=("rmse_improvement_percent", "mean"),
            )
        )
        fold_wins = int(
            (fold_means["method_rmse"] < fold_means["baseline_rmse"]).sum()
        )

        row = {
            "protocol": protocol,
            "method": method,
            "baseline": baseline,
            "num_seeds": int(group["random_seed"].nunique()),
            "pooled_method_rmse_mean": float(
                group["pooled_method_rmse"].mean()
            ),
            "pooled_method_rmse_std": float(
                group["pooled_method_rmse"].std(ddof=1)
            ),
            "pooled_baseline_rmse_mean": float(
                group["pooled_baseline_rmse"].mean()
            ),
            "pooled_baseline_rmse_std": float(
                group["pooled_baseline_rmse"].std(ddof=1)
            ),
            "pooled_rmse_improvement_mean_percent": float(
                group["pooled_rmse_improvement_percent"].mean()
            ),
            "pooled_rmse_improvement_std_percent": float(
                group["pooled_rmse_improvement_percent"].std(ddof=1)
            ),
            "macro_rmse_improvement_mean_percent": float(
                group["macro_rmse_improvement_percent"].mean()
            ),
            "macro_rmse_improvement_std_percent": float(
                group["macro_rmse_improvement_percent"].std(ddof=1)
            ),
            "seed_wins_pooled_rmse": int(
                (group["pooled_method_rmse"] < group["pooled_baseline_rmse"])
                .sum()
            ),
            "fold_wins_after_seed_average": fold_wins,
            "num_folds": int(len(fold_means)),
            "mean_point_win_percent": float(
                points["method_point_win_percent"].mean()
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["protocol", "baseline", "method"]
    ).reset_index(drop=True)


def round_for_report(
    frame: pd.DataFrame,
    digits: int = 4,
) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        audit,
        all_folds,
        all_samples,
        all_pooled,
        summaries,
    ) = load_and_audit(args.results_root)

    macro_per_seed = build_macro_per_seed(all_folds)

    pooled_summary = metric_mean_std(
        all_pooled,
        ["protocol", "method"],
        METRICS,
    )
    macro_summary = metric_mean_std(
        macro_per_seed,
        ["protocol", "method"],
        [
            f"{metric}_macro"
            for metric in METRICS
            if f"{metric}_macro" in macro_per_seed.columns
        ],
    )
    fold_summary = metric_mean_std(
        all_folds,
        ["protocol", "fold", "method"],
        METRICS,
    )
    rankings = build_rankings(pooled_summary, macro_summary)

    comparison_per_seed, comparison_by_fold, comparison_pointwise = (
        build_comparisons(
            all_folds,
            all_samples,
            all_pooled,
            macro_per_seed,
        )
    )
    comparison_summary = summarize_comparisons(
        comparison_per_seed,
        comparison_by_fold,
        comparison_pointwise,
    )

    outputs = {
        "audit": output_dir / "six_run_audit.csv",
        "all_folds": output_dir / "all_fold_results.csv",
        "all_pooled": output_dir / "all_pooled_results.csv",
        "macro_per_seed": output_dir / "macro_results_per_seed.csv",
        "pooled_summary": (
            output_dir / "three_seed_pooled_mean_std.csv"
        ),
        "macro_summary": (
            output_dir / "three_seed_macro_mean_std.csv"
        ),
        "fold_summary": output_dir / "three_seed_fold_mean_std.csv",
        "rankings": output_dir / "method_rankings.csv",
        "comparison_per_seed": (
            output_dir / "comparison_per_seed.csv"
        ),
        "comparison_by_fold": (
            output_dir / "comparison_by_fold.csv"
        ),
        "comparison_pointwise": (
            output_dir / "comparison_pointwise_per_seed.csv"
        ),
        "comparison_summary": (
            output_dir / "comparison_three_seed_summary.csv"
        ),
        "headline": output_dir / "headline_summary.json",
        "report": output_dir / "external_grouped_summary_report.md",
    }

    audit.to_csv(outputs["audit"], index=False)
    all_folds.to_csv(outputs["all_folds"], index=False)
    all_pooled.to_csv(outputs["all_pooled"], index=False)
    macro_per_seed.to_csv(outputs["macro_per_seed"], index=False)
    pooled_summary.to_csv(outputs["pooled_summary"], index=False)
    macro_summary.to_csv(outputs["macro_summary"], index=False)
    fold_summary.to_csv(outputs["fold_summary"], index=False)
    rankings.to_csv(outputs["rankings"], index=False)
    comparison_per_seed.to_csv(
        outputs["comparison_per_seed"],
        index=False,
    )
    comparison_by_fold.to_csv(
        outputs["comparison_by_fold"],
        index=False,
    )
    comparison_pointwise.to_csv(
        outputs["comparison_pointwise"],
        index=False,
    )
    comparison_summary.to_csv(
        outputs["comparison_summary"],
        index=False,
    )

    headline: dict[str, Any] = {
        "audit_status": (
            "PASS" if (audit["status"] == "PASS").all() else "FAIL"
        ),
        "num_runs": int(len(audit)),
        "seeds": list(SEEDS),
        "protocols": list(PROTOCOLS),
        "rows_after_cleaning": 2288,
        "interpretation_note": (
            "Seed standard deviations reflect estimator randomness on the "
            "same observations. The two configuration groups and three "
            "scenario groups are too few for strong inferential claims."
        ),
        "best_methods": {},
        "wallpath_vs_direct_rf": {},
    }

    for protocol in PROTOCOLS:
        protocol_rank = rankings[
            (rankings["protocol"] == protocol)
            & (rankings["summary_type"] == "pooled_point_level")
        ].sort_values("rank")
        best = protocol_rank.iloc[0]
        headline["best_methods"][protocol] = {
            "pooled_rmse_method": str(best["method"]),
            "pooled_rmse_mean": float(best["rmse_mean"]),
            "pooled_rmse_std": float(best["rmse_std"]),
        }

        protocol_comparisons = comparison_summary[
            (comparison_summary["protocol"] == protocol)
            & (comparison_summary["baseline"] == "direct_rf")
        ]
        headline["wallpath_vs_direct_rf"][protocol] = (
            protocol_comparisons.to_dict(orient="records")
        )

    outputs["headline"].write_text(
        json.dumps(headline, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# External Measured 3.5 GHz Grouped Evaluation",
        "",
        "## Scope",
        "",
        (
            "This report summarizes six grouped evaluations: "
            "leave-configuration-out and leave-scenario-out under random "
            "seeds 11, 22, and 33. The models are trained from scratch on "
            "the measured pointwise dataset; this is not frozen transfer "
            "of the ICASSP radio-map model."
        ),
        "",
        (
            "Seed standard deviations measure estimator randomness over "
            "the same observations. The dataset contains only two "
            "configuration groups and three scenario groups, so the "
            "external results should be interpreted descriptively as "
            "secondary evidence."
        ),
        "",
        "## Six-run audit",
        "",
        markdown_table(
            audit[
                [
                    "protocol",
                    "random_seed",
                    "fold_rows",
                    "sample_rows",
                    "unique_test_points",
                    "status",
                ]
            ]
        ),
        "",
        "## Three-seed pooled point-level RMSE ranking",
        "",
    ]

    pooled_display = rankings[
        rankings["summary_type"] == "pooled_point_level"
    ][
        [
            "protocol",
            "rank",
            "method",
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_max",
        ]
    ]
    report_lines.extend(
        [
            markdown_table(round_for_report(pooled_display)),
            "",
            "## Three-seed macro equal-fold RMSE ranking",
            "",
        ]
    )

    macro_display = rankings[
        rankings["summary_type"] == "macro_equal_fold"
    ][
        [
            "protocol",
            "rank",
            "method",
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_max",
        ]
    ]
    report_lines.extend(
        [
            markdown_table(round_for_report(macro_display)),
            "",
            "## Fold-specific RMSE",
            "",
        ]
    )

    fold_display = fold_summary[
        [
            "protocol",
            "fold",
            "method",
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_max",
        ]
    ].sort_values(["protocol", "fold", "rmse_mean"])
    report_lines.extend(
        [
            markdown_table(round_for_report(fold_display)),
            "",
            "## WallPath comparisons",
            "",
        ]
    )

    comparison_display = comparison_summary[
        [
            "protocol",
            "method",
            "baseline",
            "pooled_method_rmse_mean",
            "pooled_method_rmse_std",
            "pooled_baseline_rmse_mean",
            "pooled_baseline_rmse_std",
            "pooled_rmse_improvement_mean_percent",
            "seed_wins_pooled_rmse",
            "fold_wins_after_seed_average",
            "num_folds",
            "mean_point_win_percent",
        ]
    ]
    report_lines.extend(
        [
            markdown_table(round_for_report(comparison_display)),
            "",
            "## Interpretation limits",
            "",
            (
                "- Pooled metrics weight each measured point equally; macro "
                "metrics weight each held-out configuration or scenario "
                "equally."
            ),
            (
                "- Seeds reuse the same measured observations and grouped "
                "splits. Seed consistency is a stability check, not an "
                "independent-sample significance test."
            ),
            (
                "- With only two configurations and three scenarios, "
                "cluster-based confidence intervals or formal group-level "
                "significance tests would be unstable and should not be "
                "used as strong evidence."
            ),
            (
                "- Pointwise win rates are descriptive because measurements "
                "within the same environment are not guaranteed to be "
                "independent."
            ),
        ]
    )

    outputs["report"].write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 110)
    print("EXTERNAL 3.5 GHz GROUPED SIX-RUN SUMMARY")
    print("=" * 110)
    print("\nAUDIT")
    print(
        audit[
            [
                "protocol",
                "random_seed",
                "fold_rows",
                "sample_rows",
                "unique_test_points",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nPOOLED POINT-LEVEL RMSE RANKING")
    print(
        round_for_report(pooled_display)
        .to_string(index=False)
    )

    print("\nMACRO EQUAL-FOLD RMSE RANKING")
    print(
        round_for_report(macro_display)
        .to_string(index=False)
    )

    print("\nWALLPATH COMPARISONS")
    print(
        round_for_report(comparison_display)
        .to_string(index=False)
    )

    print("\nWROTE")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
