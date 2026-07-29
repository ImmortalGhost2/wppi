from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

RUNS = {
    "task2": {
        11: Path("results/transfer_task1_to_task2_final_seed11_r001"),
        22: Path("results/transfer_task1_to_task2_final_seed22_r001"),
        33: Path("results/transfer_task1_to_task2_final_seed33_r001"),
    },
    "task3": {
        11: Path("results/transfer_task1_to_task3_final_seed11_r001"),
        22: Path("results/transfer_task1_to_task3_final_seed22_r001"),
        33: Path("results/transfer_task1_to_task3_final_seed33_r001"),
    },
}

EXPECTED_MAPS_PER_RUN = {
    "task2": 750,
    "task3": 5550,
}

EXPECTED_METHODS = [
    "log_distance",
    "multi_wall",
    "idw",
    "multi_wall_residual_idw",
    "direct_rf_geometry",
    "direct_rf_sparse_anchor",
    "direct_rf_all_features",
    "direct_extra_all_features",
    "wallpath_rf",
    "wallpath_extra",
    "wallpath_calibrated",
]

METRICS = [
    "rmse",
    "mae",
    "median_ae",
    "p90_ae",
    "los_rmse",
    "nlos_rmse",
    "high_wall_rmse",
    "unclipped_rmse",
    "non_anchor_rmse",
    "non_anchor_mae",
    "non_anchor_p90_ae",
    "non_anchor_los_rmse",
    "non_anchor_nlos_rmse",
    "free_space_rmse",
    "wall_region_rmse",
]

KEY_METRICS = [
    "rmse",
    "non_anchor_rmse",
    "mae",
    "p90_ae",
    "los_rmse",
    "nlos_rmse",
    "high_wall_rmse",
    "free_space_rmse",
    "wall_region_rmse",
]

KEY_METHODS = [
    "wallpath_extra",
    "direct_rf_all_features",
    "direct_extra_all_features",
    "multi_wall_residual_idw",
]

OUTPUT_DIR = Path("results/task2_task3_transfer_summary")


# =============================================================================
# Helpers
# =============================================================================

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flattened = []
    for column in out.columns:
        if isinstance(column, tuple):
            flattened.append("_".join(str(part) for part in column if str(part)))
        else:
            flattened.append(str(column))
    out.columns = flattened
    return out


def fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def classify_transfer_condition(
    antenna: pd.Series,
    frequency: pd.Series,
) -> np.ndarray:
    return np.select(
        [
            (antenna == 1) & (frequency == 1),
            (antenna == 1) & frequency.isin([2, 3]),
            antenna.isin([2, 3, 4, 5]),
        ],
        [
            "Ant1-f1 source-like",
            "Ant1-f2-f3 frequency shift",
            "Ant2-Ant5 additional antennas",
        ],
        default="unexpected",
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


# =============================================================================
# Load and audit all six experiments
# =============================================================================

def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and summarize the six final Task 2 and Task 3 "
            "frozen-transfer experiments."
        )
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict] = []
    final_frames: list[pd.DataFrame] = []
    subgroup_per_seed_frames: list[pd.DataFrame] = []
    subgroup_count_rows: list[dict] = []

    for task, task_runs in RUNS.items():
        expected_maps = EXPECTED_MAPS_PER_RUN[task]
        expected_sample_rows = expected_maps * len(EXPECTED_METHODS)

        for expected_seed, run_dir in task_runs.items():
            final_path = run_dir / "final_evaluation_results.csv"
            sample_path = run_dir / "per_sample_metrics.csv"
            summary_path = run_dir / "run_summary.json"

            missing_files = [
                str(path)
                for path in (final_path, sample_path, summary_path)
                if not path.exists()
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"{task} seed {expected_seed} is missing:\n"
                    + "\n".join(missing_files)
                )

            final = pd.read_csv(final_path)
            samples = pd.read_csv(sample_path)
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            require_columns(
                final,
                ["method", "sparse_rate", "sparse_seed", *METRICS],
                f"{task} seed {expected_seed} final results",
            )
            require_columns(
                samples,
                ["method", "sample_id", "sparse_rate", "sparse_seed", *METRICS],
                f"{task} seed {expected_seed} per-sample results",
            )

            evaluated_seeds = (
                run_summary.get("extra", {}).get("evaluated_sparse_seeds")
            )
            final_seeds = sorted(
                final["sparse_seed"].astype(int).unique().tolist()
            )
            sample_seeds = sorted(
                samples["sparse_seed"].astype(int).unique().tolist()
            )
            final_rates = sorted(
                final["sparse_rate"].astype(float).unique().tolist()
            )
            sample_rates = sorted(
                samples["sparse_rate"].astype(float).unique().tolist()
            )

            checks = {
                "status_completed": run_summary.get("status") == "completed",
                "summary_seed_correct": evaluated_seeds == [expected_seed],
                "final_seed_correct": final_seeds == [expected_seed],
                "sample_seed_correct": sample_seeds == [expected_seed],
                "final_rate_correct": final_rates == [0.01],
                "sample_rate_correct": sample_rates == [0.01],
                "final_rows_11": len(final) == len(EXPECTED_METHODS),
                "sample_rows_correct": len(samples) == expected_sample_rows,
                "methods_exact": set(final["method"].astype(str))
                == set(EXPECTED_METHODS),
                "all_metrics_present": all(metric in final.columns for metric in METRICS),
                "sample_metrics_present": all(
                    metric in samples.columns for metric in METRICS
                ),
                "no_dense_npz": not any(run_dir.glob("*.npz")),
            }
            run_pass = all(checks.values())

            audit_rows.append(
                {
                    "task": task,
                    "expected_seed": expected_seed,
                    "evaluated_sparse_seeds": str(evaluated_seeds),
                    "final_sparse_seeds": str(final_seeds),
                    "sample_sparse_seeds": str(sample_seeds),
                    "final_rows": len(final),
                    "per_sample_rows": len(samples),
                    **checks,
                    "audit_pass": run_pass,
                    "run_directory": str(run_dir),
                }
            )

            if not run_pass:
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(
                    f"{task} seed {expected_seed} failed audit: {failed}"
                )

            final = final.copy()
            final["task"] = task
            final["verified_seed"] = expected_seed
            final["run_directory"] = str(run_dir)
            final_frames.append(final)

            parsed = samples["sample_id"].str.extract(
                r"B(?P<building>\d+)_"
                r"Ant(?P<antenna>\d+)_"
                r"f(?P<frequency>\d+)_"
                r"S(?P<sample>\d+)"
            )

            if parsed.isna().any().any():
                bad_ids = samples.loc[
                    parsed.isna().any(axis=1), "sample_id"
                ].head(10)
                raise ValueError(
                    f"Could not parse sample IDs in {task} seed {expected_seed}: "
                    f"{bad_ids.tolist()}"
                )

            samples = samples.copy()
            for column in parsed.columns:
                samples[column] = parsed[column].astype(int)

            samples["transfer_condition"] = classify_transfer_condition(
                antenna=samples["antenna"],
                frequency=samples["frequency"],
            )

            if (samples["transfer_condition"] == "unexpected").any():
                unexpected = samples.loc[
                    samples["transfer_condition"] == "unexpected",
                    ["sample_id", "antenna", "frequency"],
                ].drop_duplicates()
                raise ValueError(
                    "Unexpected antenna/frequency combinations:\n"
                    + unexpected.to_string(index=False)
                )

            samples["task"] = task
            samples["verified_seed"] = expected_seed

            count_frame = samples[samples["method"] == "wallpath_extra"]
            counts = count_frame["transfer_condition"].value_counts().to_dict()

            for condition, count in counts.items():
                subgroup_count_rows.append(
                    {
                        "task": task,
                        "verified_seed": expected_seed,
                        "transfer_condition": condition,
                        "map_count": int(count),
                    }
                )

            subgroup_per_seed = (
                samples.groupby(
                    [
                        "task",
                        "verified_seed",
                        "transfer_condition",
                        "method",
                    ],
                    as_index=False,
                )[METRICS]
                .mean()
            )
            subgroup_per_seed_frames.append(subgroup_per_seed)

    audit = pd.DataFrame(audit_rows)
    all_final = pd.concat(final_frames, ignore_index=True)
    subgroup_counts = pd.DataFrame(subgroup_count_rows)
    subgroup_per_seed = pd.concat(
        subgroup_per_seed_frames,
        ignore_index=True,
    )

    # =========================================================================
    # Official task-level three-seed summaries
    # =========================================================================

    task_summary = (
        all_final.groupby(["task", "method"])[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    task_summary = flatten_columns(task_summary)
    task_summary["rmse_rank"] = (
        task_summary.groupby("task")["rmse_mean"]
        .rank(method="min")
        .astype(int)
    )
    task_summary = task_summary.sort_values(
        ["task", "rmse_rank", "method"]
    ).reset_index(drop=True)

    # =========================================================================
    # WallPath versus Direct RF by seed
    # =========================================================================

    pairwise_rows: list[dict] = []

    for task in sorted(RUNS):
        for seed in sorted(RUNS[task]):
            run = all_final[
                (all_final["task"] == task)
                & (all_final["verified_seed"] == seed)
            ].set_index("method")

            wallpath = run.loc["wallpath_extra"]
            direct_rf = run.loc["direct_rf_all_features"]

            for metric in KEY_METRICS:
                wp_value = float(wallpath[metric])
                rf_value = float(direct_rf[metric])
                improvement = 100.0 * (rf_value - wp_value) / rf_value

                pairwise_rows.append(
                    {
                        "task": task,
                        "verified_seed": seed,
                        "metric": metric,
                        "wallpath_extra": wp_value,
                        "direct_rf_all_features": rf_value,
                        "wallpath_improvement_percent": improvement,
                        "winner": (
                            "wallpath_extra"
                            if wp_value < rf_value
                            else "direct_rf_all_features"
                            if rf_value < wp_value
                            else "tie"
                        ),
                    }
                )

    pairwise_by_seed = pd.DataFrame(pairwise_rows)

    # =========================================================================
    # WallPath versus Direct RF task-level summary
    # =========================================================================

    task_index = task_summary.set_index(["task", "method"])
    task_comparison_rows: list[dict] = []

    for task in sorted(RUNS):
        wallpath = task_index.loc[(task, "wallpath_extra")]
        direct_rf = task_index.loc[(task, "direct_rf_all_features")]

        for metric in KEY_METRICS:
            wp_mean = float(wallpath[f"{metric}_mean"])
            wp_std = float(wallpath[f"{metric}_std"])
            rf_mean = float(direct_rf[f"{metric}_mean"])
            rf_std = float(direct_rf[f"{metric}_std"])

            seed_rows = pairwise_by_seed[
                (pairwise_by_seed["task"] == task)
                & (pairwise_by_seed["metric"] == metric)
            ]
            improvement = 100.0 * (rf_mean - wp_mean) / rf_mean

            task_comparison_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "wallpath_mean": wp_mean,
                    "wallpath_std": wp_std,
                    "direct_rf_mean": rf_mean,
                    "direct_rf_std": rf_std,
                    "wallpath_improvement_percent": improvement,
                    "wallpath_seed_wins": int(
                        (seed_rows["winner"] == "wallpath_extra").sum()
                    ),
                    "num_seeds": int(len(seed_rows)),
                }
            )

    task_comparison = pd.DataFrame(task_comparison_rows)

    # =========================================================================
    # Three-seed subgroup summaries
    # =========================================================================

    subgroup_summary = (
        subgroup_per_seed.groupby(
            ["task", "transfer_condition", "method"]
        )[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    subgroup_summary = flatten_columns(subgroup_summary)
    subgroup_summary["rmse_rank"] = (
        subgroup_summary.groupby(
            ["task", "transfer_condition"]
        )["rmse_mean"]
        .rank(method="min")
        .astype(int)
    )
    subgroup_summary = subgroup_summary.sort_values(
        ["task", "transfer_condition", "rmse_rank", "method"]
    ).reset_index(drop=True)

    # =========================================================================
    # Write machine-readable outputs
    # =========================================================================

    paths = {
        "audit": OUTPUT_DIR / "six_run_audit.csv",
        "all_results": OUTPUT_DIR / "all_6_runs_final_results.csv",
        "task_summary": OUTPUT_DIR / "task_level_3seed_mean_std.csv",
        "pairwise": OUTPUT_DIR / "wallpath_vs_direct_rf_by_seed.csv",
        "comparison": OUTPUT_DIR / "wallpath_vs_direct_rf_task_summary.csv",
        "subgroup_counts": OUTPUT_DIR / "subgroup_map_counts.csv",
        "subgroup_seed": OUTPUT_DIR / "subgroup_per_seed.csv",
        "subgroup_summary": OUTPUT_DIR / "subgroup_3seed_mean_std.csv",
        "headline": OUTPUT_DIR / "headline_summary.json",
        "report": OUTPUT_DIR / "transfer_summary_report.md",
    }

    audit.to_csv(paths["audit"], index=False)
    all_final.to_csv(paths["all_results"], index=False)
    task_summary.to_csv(paths["task_summary"], index=False)
    pairwise_by_seed.to_csv(paths["pairwise"], index=False)
    task_comparison.to_csv(paths["comparison"], index=False)
    subgroup_counts.to_csv(paths["subgroup_counts"], index=False)
    subgroup_per_seed.to_csv(paths["subgroup_seed"], index=False)
    subgroup_summary.to_csv(paths["subgroup_summary"], index=False)

    # =========================================================================
    # Headline JSON
    # =========================================================================

    headline: dict[str, dict] = {}

    for task in sorted(RUNS):
        ranking = task_summary[
            task_summary["task"] == task
        ].sort_values("rmse_rank")
        winner = ranking.iloc[0]

        rmse_comparison = task_comparison[
            (task_comparison["task"] == task)
            & (task_comparison["metric"] == "rmse")
        ].iloc[0]

        headline[task] = {
            "best_method_by_rmse": str(winner["method"]),
            "best_rmse_mean": float(winner["rmse_mean"]),
            "best_rmse_std": float(winner["rmse_std"]),
            "wallpath_rmse_mean": float(rmse_comparison["wallpath_mean"]),
            "direct_rf_rmse_mean": float(rmse_comparison["direct_rf_mean"]),
            "wallpath_improvement_vs_direct_rf_percent": float(
                rmse_comparison["wallpath_improvement_percent"]
            ),
            "wallpath_rmse_seed_wins": int(
                rmse_comparison["wallpath_seed_wins"]
            ),
            "num_seeds": int(rmse_comparison["num_seeds"]),
        }

    paths["headline"].write_text(
        json.dumps(headline, indent=2),
        encoding="utf-8",
    )

    # =========================================================================
    # Markdown report
    # =========================================================================

    report_lines: list[str] = [
        "# Task1-to-Task2 and Task1-to-Task3 Transfer Summary",
        "",
        (
            "This report summarizes six matched-seed frozen-transfer "
            "experiments: Task2 and Task3 at seeds 11, 22, and 33."
        ),
        "",
        "## Experiment audit",
        "",
        "| Task | Seed | Runtime sparse seed | Final rows | Per-sample rows | Status |",
        "|---|---:|---|---:|---:|---|",
    ]

    for row in audit.itertuples(index=False):
        status = "PASS" if row.audit_pass else "FAIL"
        report_lines.append(
            f"| {row.task.upper()} | {row.expected_seed} "
            f"| {row.evaluated_sparse_seeds} | {row.final_rows} "
            f"| {row.per_sample_rows} | {status} |"
        )

    for task in sorted(RUNS):
        report_lines.extend(
            [
                "",
                f"## {task.upper()} three-seed ranking",
                "",
                "| Rank | Method | RMSE | MAE | P90 AE | Non-anchor RMSE |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )

        ranking = task_summary[
            task_summary["task"] == task
        ].sort_values("rmse_rank")

        for row in ranking.itertuples(index=False):
            rmse_text = fmt_mean_std(row.rmse_mean, row.rmse_std)
            mae_text = fmt_mean_std(row.mae_mean, row.mae_std)
            p90_text = fmt_mean_std(row.p90_ae_mean, row.p90_ae_std)
            non_anchor_text = fmt_mean_std(
                row.non_anchor_rmse_mean,
                row.non_anchor_rmse_std,
            )
            report_lines.append(
                f"| {row.rmse_rank} | `{row.method}` | {rmse_text} "
                f"| {mae_text} | {p90_text} | {non_anchor_text} |"
            )

        report_lines.extend(
            [
                "",
                f"## {task.upper()}: WallPath-PI versus Direct RF",
                "",
                "| Metric | WallPath | Direct RF | Improvement | Seed wins |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        comparison = task_comparison[
            task_comparison["task"] == task
        ]

        for row in comparison.itertuples(index=False):
            wallpath_text = fmt_mean_std(
                row.wallpath_mean,
                row.wallpath_std,
            )
            direct_rf_text = fmt_mean_std(
                row.direct_rf_mean,
                row.direct_rf_std,
            )
            report_lines.append(
                f"| `{row.metric}` | {wallpath_text} | {direct_rf_text} "
                f"| {row.wallpath_improvement_percent:+.2f}% "
                f"| {row.wallpath_seed_wins}/{row.num_seeds} |"
            )

        report_lines.extend(
            [
                "",
                f"## {task.upper()} transfer-condition summary",
                "",
            ]
        )

        task_subgroups = subgroup_summary[
            subgroup_summary["task"] == task
        ]

        for condition in task_subgroups[
            "transfer_condition"
        ].drop_duplicates():
            report_lines.extend(
                [
                    f"### {condition}",
                    "",
                    (
                        "| Rank | Method | Mean per-map RMSE | "
                        "Mean per-map MAE | Mean per-map P90 AE |"
                    ),
                    "|---:|---|---:|---:|---:|",
                ]
            )

            condition_table = task_subgroups[
                (
                    task_subgroups["transfer_condition"]
                    == condition
                )
                & (task_subgroups["method"].isin(KEY_METHODS))
            ].sort_values("rmse_rank")

            for row in condition_table.itertuples(index=False):
                rmse_text = fmt_mean_std(row.rmse_mean, row.rmse_std)
                mae_text = fmt_mean_std(row.mae_mean, row.mae_std)
                p90_text = fmt_mean_std(row.p90_ae_mean, row.p90_ae_std)
                report_lines.append(
                    f"| {row.rmse_rank} | `{row.method}` | {rmse_text} "
                    f"| {mae_text} | {p90_text} |"
                )

            report_lines.append("")

    report_lines.extend(["## Headline conclusions", ""])

    for task, values in headline.items():
        best_method = values["best_method_by_rmse"]
        best_mean = values["best_rmse_mean"]
        best_std = values["best_rmse_std"]
        improvement = values["wallpath_improvement_vs_direct_rf_percent"]
        seed_wins = values["wallpath_rmse_seed_wins"]
        num_seeds = values["num_seeds"]

        report_lines.append(
            f"- **{task.upper()}:** `{best_method}` ranked first with "
            f"RMSE {best_mean:.4f} ± {best_std:.4f}. "
            f"WallPath improved over Direct RF by {improvement:.2f}% "
            f"and won {seed_wins}/{num_seeds} seeds."
        )

    report_lines.extend(
        [
            "",
            (
                "The official task-level metrics above come from "
                "`final_evaluation_results.csv`. Subgroup metrics are "
                "means of per-map metrics within each condition and should "
                "be labelled as mean per-map values."
            ),
        ]
    )

    paths["report"].write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # Console summary
    # =========================================================================

    print("=" * 110)
    print("SIX-EXPERIMENT TRANSFER SUMMARY")
    print("=" * 110)

    print("\nAUDIT")
    print(
        audit[
            [
                "task",
                "expected_seed",
                "evaluated_sparse_seeds",
                "final_rows",
                "per_sample_rows",
                "audit_pass",
            ]
        ].to_string(index=False)
    )

    for task in sorted(RUNS):
        print("\n" + "=" * 110)
        print(f"{task.upper()} THREE-SEED RANKING")
        print("=" * 110)

        ranking = task_summary[
            task_summary["task"] == task
        ].sort_values("rmse_rank")

        print(
            ranking[
                [
                    "rmse_rank",
                    "method",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "mae_std",
                    "p90_ae_mean",
                    "p90_ae_std",
                    "non_anchor_rmse_mean",
                    "non_anchor_rmse_std",
                ]
            ].to_string(index=False)
        )

        print("\nWALLPATH-PI VS DIRECT RF")
        comparison = task_comparison[
            task_comparison["task"] == task
        ]

        print(
            comparison[
                [
                    "metric",
                    "wallpath_mean",
                    "wallpath_std",
                    "direct_rf_mean",
                    "direct_rf_std",
                    "wallpath_improvement_percent",
                    "wallpath_seed_wins",
                    "num_seeds",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 110)
    print("ALL SIX EXPERIMENTS: PASS")
    print("=" * 110)

    print("\nWROTE:")
    for output_path in paths.values():
        print(output_path)


if __name__ == "__main__":
    main()
