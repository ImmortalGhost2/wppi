#!/usr/bin/env python
"""Summarize measured 3.5 GHz few-shot adaptation against matched zero-shot baselines.

This script does not retrain any model. It joins the completed few-shot
predictions to the seed-11 leave-scenario-out predictions on the exact same
non-anchor rows. This gives matched adaptation gains for each held-out
scenario, anchor fraction, and anchor-selection seed.

Required inputs
---------------
results/external_3p5ghz_fewshot_modelseed11/per_sample_metrics.csv
results/external_3p5ghz_fewshot_modelseed11/final_evaluation_results.csv
results/external_3p5ghz_fewshot_modelseed11/pooled_evaluation_results.csv
results/external_3p5ghz_fewshot_modelseed11/run_summary.json
results/external_3p5ghz_leave_scenario_out_seed11/per_sample_metrics.csv
results/external_3p5ghz_leave_scenario_out_seed11/run_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_BASELINES = {
    "bias_calibrated_log_distance": "log_distance",
    "bias_calibrated_multi_wall": "multi_wall_linear",
    "direct_extra_with_anchors": "direct_extra",
    "direct_rf_with_anchors": "direct_rf",
    "wallpath_fewshot_residual_extra": "wallpath_residual_extra",
    "wallpath_fewshot_residual_rf": "wallpath_residual_rf",
}

ALL_FEWSHOT_METHODS = (
    "log_distance",
    "multi_wall_linear",
    "bias_calibrated_log_distance",
    "bias_calibrated_multi_wall",
    "direct_extra_with_anchors",
    "direct_rf_with_anchors",
    "wallpath_fewshot_residual_extra",
    "wallpath_fewshot_residual_rf",
)

EXPECTED_FOLDS = (
    "holdout_Comms",
    "holdout_Library",
    "holdout_SSE",
)
EXPECTED_FRACTIONS = (0.01, 0.05, 0.10)
EXPECTED_ANCHOR_SEEDS = (11, 22, 33)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze few-shot adaptation using exact matched zero-shot "
            "predictions on the same non-anchor observations."
        )
    )
    parser.add_argument(
        "--fewshot-dir",
        type=Path,
        default=Path("results/external_3p5ghz_fewshot_modelseed11"),
    )
    parser.add_argument(
        "--zero-shot-dir",
        type=Path,
        default=Path(
            "results/external_3p5ghz_leave_scenario_out_seed11"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_3p5ghz_fewshot_summary"),
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.percentile(absolute, 90)),
        "p95_ae": float(np.percentile(absolute, 95)),
        "bias_db": float(np.mean(error)),
        "count": int(len(error)),
    }


def improvement_percent(adapted: float, baseline: float) -> float:
    if baseline == 0 or not np.isfinite(baseline):
        return float("nan")
    return float((baseline - adapted) / baseline * 100.0)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    display = frame.copy().replace({np.nan: ""})
    headers = [str(column) for column in display.columns]

    def rule(column: pd.Series) -> str:
        return "---:" if pd.api.types.is_numeric_dtype(column) else "---"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(rule(display[column]) for column in display.columns)
        + " |",
    ]

    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", r"\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def round_numeric(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(digits)
    return result


def json_default(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values into standard JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fewshot_paths = {
        "samples": args.fewshot_dir / "per_sample_metrics.csv",
        "folds": args.fewshot_dir / "final_evaluation_results.csv",
        "pooled": args.fewshot_dir / "pooled_evaluation_results.csv",
        "summary": args.fewshot_dir / "run_summary.json",
    }
    zero_paths = {
        "samples": args.zero_shot_dir / "per_sample_metrics.csv",
        "summary": args.zero_shot_dir / "run_summary.json",
    }

    for path in [*fewshot_paths.values(), *zero_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    fewshot_samples = pd.read_csv(fewshot_paths["samples"])
    fewshot_folds = pd.read_csv(fewshot_paths["folds"])
    fewshot_pooled = pd.read_csv(fewshot_paths["pooled"])
    fewshot_summary = json.loads(
        fewshot_paths["summary"].read_text(encoding="utf-8")
    )

    zero_samples = pd.read_csv(zero_paths["samples"])
    zero_summary = json.loads(
        zero_paths["summary"].read_text(encoding="utf-8")
    )

    require_columns(
        fewshot_samples,
        [
            "row_uid",
            "fold",
            "method",
            "target_pl_db",
            "pred_pl_db",
            "anchor_fraction",
            "anchor_seed",
            "model_seed",
            "is_anchor",
        ],
        label=str(fewshot_paths["samples"]),
    )
    require_columns(
        zero_samples,
        [
            "row_uid",
            "fold",
            "method",
            "target_pl_db",
            "pred_pl_db",
            "random_seed",
        ],
        label=str(zero_paths["samples"]),
    )

    audit = {
        "fewshot_split": (
            fewshot_summary.get("split") == "leave_scenario_fewshot"
        ),
        "zero_shot_split": (
            zero_summary.get("split") == "leave_scenario_out"
        ),
        "fewshot_model_seed_11": (
            fewshot_summary.get("fewshot_model_seed") == 11
        ),
        "zero_shot_seed_11": (
            zero_summary.get("random_seed") == 11
        ),
        "fewshot_methods_complete": (
            set(fewshot_samples["method"]) == set(ALL_FEWSHOT_METHODS)
        ),
        "folds_complete": (
            set(fewshot_samples["fold"]) == set(EXPECTED_FOLDS)
        ),
        "fractions_complete": (
            set(fewshot_samples["anchor_fraction"].round(2))
            == set(EXPECTED_FRACTIONS)
        ),
        "anchor_seeds_complete": (
            set(fewshot_samples["anchor_seed"].astype(int))
            == set(EXPECTED_ANCHOR_SEEDS)
        ),
        "model_seed_constant": (
            set(fewshot_samples["model_seed"].astype(int)) == {11}
        ),
        "anchors_excluded": fewshot_samples["is_anchor"].eq(False).all(),
        "fewshot_predictions_unique": not fewshot_samples.duplicated(
            subset=[
                "row_uid",
                "fold",
                "method",
                "anchor_fraction",
                "anchor_seed",
                "model_seed",
            ]
        ).any(),
        "zero_shot_predictions_unique": not zero_samples.duplicated(
            subset=["row_uid", "fold", "method"]
        ).any(),
    }

    if not all(audit.values()):
        failed = [name for name, passed in audit.items() if not passed]
        raise RuntimeError(f"Input audit failed: {failed}")

    comparison_rows: list[dict[str, Any]] = []
    matched_prediction_rows: list[pd.DataFrame] = []

    for adapted_method, baseline_method in METHOD_BASELINES.items():
        adapted = fewshot_samples[
            fewshot_samples["method"] == adapted_method
        ].copy()

        baseline = zero_samples[
            zero_samples["method"] == baseline_method
        ][
            [
                "row_uid",
                "fold",
                "target_pl_db",
                "pred_pl_db",
            ]
        ].rename(
            columns={
                "target_pl_db": "baseline_target_pl_db",
                "pred_pl_db": "baseline_pred_pl_db",
            }
        )

        merged = adapted.merge(
            baseline,
            on=["row_uid", "fold"],
            how="left",
            validate="many_to_one",
        )

        if merged["baseline_pred_pl_db"].isna().any():
            missing = merged.loc[
                merged["baseline_pred_pl_db"].isna(),
                ["row_uid", "fold"],
            ].head(10)
            raise RuntimeError(
                f"Missing matched zero-shot predictions for "
                f"{adapted_method} vs {baseline_method}:\n"
                + missing.to_string(index=False)
            )

        if not np.allclose(
            merged["target_pl_db"].to_numpy(float),
            merged["baseline_target_pl_db"].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Target mismatch for {adapted_method} vs {baseline_method}"
            )

        merged["adapted_method"] = adapted_method
        merged["baseline_method"] = baseline_method
        merged["adapted_abs_error"] = np.abs(
            merged["pred_pl_db"] - merged["target_pl_db"]
        )
        merged["baseline_abs_error"] = np.abs(
            merged["baseline_pred_pl_db"]
            - merged["target_pl_db"]
        )
        matched_prediction_rows.append(merged)

        group_columns = [
            "fold",
            "anchor_fraction",
            "anchor_seed",
            "model_seed",
        ]

        for keys, group in merged.groupby(group_columns, sort=True):
            fold, fraction, anchor_seed, model_seed = keys

            adapted_metrics = metrics(
                group["target_pl_db"].to_numpy(float),
                group["pred_pl_db"].to_numpy(float),
            )
            baseline_metrics = metrics(
                group["target_pl_db"].to_numpy(float),
                group["baseline_pred_pl_db"].to_numpy(float),
            )

            adapted_wins = int(
                (
                    group["adapted_abs_error"]
                    < group["baseline_abs_error"]
                ).sum()
            )
            baseline_wins = int(
                (
                    group["baseline_abs_error"]
                    < group["adapted_abs_error"]
                ).sum()
            )

            comparison_rows.append(
                {
                    "fold": fold,
                    "anchor_fraction": float(fraction),
                    "anchor_seed": int(anchor_seed),
                    "model_seed": int(model_seed),
                    "adapted_method": adapted_method,
                    "baseline_method": baseline_method,
                    "test_count": int(len(group)),
                    "adapted_rmse": adapted_metrics["rmse"],
                    "baseline_rmse": baseline_metrics["rmse"],
                    "rmse_improvement_percent": improvement_percent(
                        adapted_metrics["rmse"],
                        baseline_metrics["rmse"],
                    ),
                    "adapted_mae": adapted_metrics["mae"],
                    "baseline_mae": baseline_metrics["mae"],
                    "mae_improvement_percent": improvement_percent(
                        adapted_metrics["mae"],
                        baseline_metrics["mae"],
                    ),
                    "adapted_p90_ae": adapted_metrics["p90_ae"],
                    "baseline_p90_ae": baseline_metrics["p90_ae"],
                    "p90_improvement_percent": improvement_percent(
                        adapted_metrics["p90_ae"],
                        baseline_metrics["p90_ae"],
                    ),
                    "adapted_point_wins": adapted_wins,
                    "baseline_point_wins": baseline_wins,
                    "ties": int(
                        len(group) - adapted_wins - baseline_wins
                    ),
                    "adapted_point_win_percent": float(
                        adapted_wins / len(group) * 100.0
                    ),
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    matched_predictions = pd.concat(
        matched_prediction_rows,
        ignore_index=True,
    )

    per_fraction_rows: list[dict[str, Any]] = []
    for keys, group in comparisons.groupby(
        [
            "anchor_fraction",
            "adapted_method",
            "baseline_method",
        ],
        sort=True,
    ):
        fraction, adapted_method, baseline_method = keys

        pooled_adapted = matched_predictions[
            (matched_predictions["anchor_fraction"] == fraction)
            & (
                matched_predictions["adapted_method"]
                == adapted_method
            )
        ]

        # Each target point is repeated once per anchor seed. Compute one pooled
        # metric per anchor seed first, then summarize across anchor seeds.
        seed_rows: list[dict[str, float]] = []
        for anchor_seed, seed_group in pooled_adapted.groupby(
            "anchor_seed",
            sort=True,
        ):
            adapted_metrics = metrics(
                seed_group["target_pl_db"].to_numpy(float),
                seed_group["pred_pl_db"].to_numpy(float),
            )
            baseline_metrics = metrics(
                seed_group["target_pl_db"].to_numpy(float),
                seed_group["baseline_pred_pl_db"].to_numpy(float),
            )
            seed_rows.append(
                {
                    "anchor_seed": int(anchor_seed),
                    "adapted_rmse": adapted_metrics["rmse"],
                    "baseline_rmse": baseline_metrics["rmse"],
                    "rmse_improvement_percent": improvement_percent(
                        adapted_metrics["rmse"],
                        baseline_metrics["rmse"],
                    ),
                    "adapted_mae": adapted_metrics["mae"],
                    "baseline_mae": baseline_metrics["mae"],
                    "mae_improvement_percent": improvement_percent(
                        adapted_metrics["mae"],
                        baseline_metrics["mae"],
                    ),
                }
            )

        seed_frame = pd.DataFrame(seed_rows)

        fold_mean = (
            group.groupby("fold", as_index=False)
            .agg(
                adapted_rmse=("adapted_rmse", "mean"),
                baseline_rmse=("baseline_rmse", "mean"),
            )
        )

        per_fraction_rows.append(
            {
                "anchor_fraction": float(fraction),
                "adapted_method": adapted_method,
                "baseline_method": baseline_method,
                "adapted_rmse_mean": float(
                    seed_frame["adapted_rmse"].mean()
                ),
                "adapted_rmse_std": float(
                    seed_frame["adapted_rmse"].std(ddof=1)
                ),
                "baseline_rmse_mean": float(
                    seed_frame["baseline_rmse"].mean()
                ),
                "baseline_rmse_std": float(
                    seed_frame["baseline_rmse"].std(ddof=1)
                ),
                "rmse_improvement_mean_percent": float(
                    seed_frame[
                        "rmse_improvement_percent"
                    ].mean()
                ),
                "rmse_improvement_std_percent": float(
                    seed_frame[
                        "rmse_improvement_percent"
                    ].std(ddof=1)
                ),
                "adapted_mae_mean": float(
                    seed_frame["adapted_mae"].mean()
                ),
                "baseline_mae_mean": float(
                    seed_frame["baseline_mae"].mean()
                ),
                "mae_improvement_mean_percent": float(
                    seed_frame[
                        "mae_improvement_percent"
                    ].mean()
                ),
                "anchor_seed_rmse_wins": int(
                    (
                        seed_frame["adapted_rmse"]
                        < seed_frame["baseline_rmse"]
                    ).sum()
                ),
                "scenario_rmse_wins": int(
                    (
                        fold_mean["adapted_rmse"]
                        < fold_mean["baseline_rmse"]
                    ).sum()
                ),
                "num_scenarios": int(len(fold_mean)),
                "mean_point_win_percent": float(
                    group["adapted_point_win_percent"].mean()
                ),
            }
        )

    adaptation_summary = pd.DataFrame(per_fraction_rows).sort_values(
        ["anchor_fraction", "adapted_rmse_mean"]
    ).reset_index(drop=True)

    # Existing pooled few-shot ranking across anchor-selection seeds.
    pooled_ranking = (
        fewshot_pooled.groupby(
            ["anchor_fraction", "method"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            p90_mean=("p90_ae", "mean"),
            p90_std=("p90_ae", "std"),
        )
        .sort_values(["anchor_fraction", "rmse_mean"])
        .reset_index(drop=True)
    )
    pooled_ranking["rank"] = (
        pooled_ranking.groupby("anchor_fraction")["rmse_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    scenario_ranking = (
        fewshot_folds.groupby(
            ["fold", "anchor_fraction", "method"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .sort_values(["fold", "anchor_fraction", "rmse_mean"])
        .reset_index(drop=True)
    )
    scenario_ranking["rank"] = (
        scenario_ranking.groupby(
            ["fold", "anchor_fraction"]
        )["rmse_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    output_paths = {
        "audit": args.output_dir / "fewshot_matched_audit.json",
        "matched_per_scenario_seed": (
            args.output_dir
            / "matched_adaptation_per_scenario_anchor_seed.csv"
        ),
        "adaptation_summary": (
            args.output_dir / "matched_adaptation_summary.csv"
        ),
        "pooled_ranking": (
            args.output_dir / "fewshot_pooled_ranking.csv"
        ),
        "scenario_ranking": (
            args.output_dir / "fewshot_scenario_ranking.csv"
        ),
        "report": args.output_dir / "fewshot_summary_report.md",
    }

    output_paths["audit"].write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": {
                    name: bool(passed)
                    for name, passed in audit.items()
                },
                "fewshot_dir": str(args.fewshot_dir),
                "zero_shot_dir": str(args.zero_shot_dir),
                "matched_method_pairs": METHOD_BASELINES,
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    comparisons.to_csv(
        output_paths["matched_per_scenario_seed"],
        index=False,
    )
    adaptation_summary.to_csv(
        output_paths["adaptation_summary"],
        index=False,
    )
    pooled_ranking.to_csv(
        output_paths["pooled_ranking"],
        index=False,
    )
    scenario_ranking.to_csv(
        output_paths["scenario_ranking"],
        index=False,
    )

    best_by_fraction = (
        pooled_ranking[pooled_ranking["rank"] == 1]
        .sort_values("anchor_fraction")
    )

    report_lines = [
        "# External Measured 3.5 GHz Few-Shot Summary",
        "",
        "## Scope",
        "",
        (
            "The few-shot run uses model seed 11 and three independent "
            "target-anchor selections (11, 22, 33). Metrics exclude all "
            "target anchors. Matched adaptation gains are computed against "
            "the seed-11 zero-shot leave-scenario-out predictions on the "
            "exact same non-anchor observations."
        ),
        "",
        "## Pooled few-shot ranking",
        "",
        markdown_table(
            round_numeric(
                pooled_ranking[
                    [
                        "anchor_fraction",
                        "rank",
                        "method",
                        "rmse_mean",
                        "rmse_std",
                        "mae_mean",
                        "p90_mean",
                    ]
                ]
            )
        ),
        "",
        "## Matched adaptation gains",
        "",
        markdown_table(round_numeric(adaptation_summary)),
        "",
        "## Best pooled method by fraction",
        "",
        markdown_table(
            round_numeric(
                best_by_fraction[
                    [
                        "anchor_fraction",
                        "method",
                        "rmse_mean",
                        "rmse_std",
                    ]
                ]
            )
        ),
        "",
        "## Interpretation limits",
        "",
        (
            "- The three anchor seeds vary target-anchor selection, while "
            "model randomness is fixed."
        ),
        (
            "- Only three held-out scenarios are available. Scenario-win "
            "counts are descriptive, not formal significance tests."
        ),
        (
            "- The external experiment is pointwise measured-data "
            "adaptation and is not frozen transfer of the ICASSP radio-map "
            "model."
        ),
    ]
    output_paths["report"].write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 110)
    print("EXTERNAL 3.5 GHz FEW-SHOT MATCHED SUMMARY")
    print("=" * 110)

    print("\nINPUT AUDIT")
    for name, passed in audit.items():
        print(f"{name:42s}: {'PASS' if passed else 'FAIL'}")

    print("\nPOOLED RANKING")
    print(
        round_numeric(
            pooled_ranking[
                [
                    "anchor_fraction",
                    "rank",
                    "method",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "p90_mean",
                ]
            ]
        ).to_string(index=False)
    )

    print("\nMATCHED ADAPTATION GAINS")
    print(
        round_numeric(adaptation_summary).to_string(index=False)
    )

    print("\nWROTE")
    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
