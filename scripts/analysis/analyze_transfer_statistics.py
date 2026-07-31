from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Experiment definition
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

WALLPATH_METHOD = "wallpath_extra"

BASELINE_METHODS = [
    "direct_rf_all_features",
    "multi_wall_residual_idw",
    "direct_extra_all_features",
]

METRICS = [
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

# All listed metrics are errors: lower is better.
LOWER_IS_BETTER = set(METRICS)

DEFAULT_OUTPUT_DIR = Path("results/task2_task3_transfer_statistics")


# =============================================================================
# General helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired map-level and building-cluster statistical analyses "
            "for the six Task2/Task3 frozen-transfer experiments."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=20000,
        help="Number of building-cluster bootstrap replicates (default: 20000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260711,
        help="Random seed for the cluster bootstrap (default: 20260711).",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_ci(low: float, high: float, digits: int = 4) -> str:
    if pd.isna(low) or pd.isna(high):
        return "NA"
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"


def winner_from_difference(difference: float, tolerance: float = 1e-12) -> str:
    """Positive difference means WallPath has lower error and therefore wins."""
    if difference > tolerance:
        return WALLPATH_METHOD
    if difference < -tolerance:
        return "baseline"
    return "tie"


def exact_two_sided_sign_test(positive: int, negative: int) -> float:
    """
    Exact two-sided sign-test p-value, excluding ties.

    With five buildings, the smallest possible two-sided p-value is 0.0625,
    so this test is deliberately reported as a cautious descriptive check.
    """
    n = positive + negative
    if n == 0:
        return float("nan")

    tail = min(positive, negative)
    probability = sum(math.comb(n, i) for i in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


# =============================================================================
# Metadata parsing
# =============================================================================

def parse_sample_metadata(sample_ids: pd.Series, label: str) -> pd.DataFrame:
    """
    Parse building, antenna, and frequency identifiers from sample_id.

    Expected identifiers include fragments such as:
      B21_Ant1_f2_...
    """
    text = sample_ids.astype(str)

    parsed = pd.DataFrame(
        {
            "building": pd.to_numeric(
                text.str.extract(r"(?:^|_)B(\d+)(?:_|$)", expand=False),
                errors="coerce",
            ),
            "antenna": pd.to_numeric(
                text.str.extract(r"(?:^|_)Ant(\d+)(?:_|$)", expand=False),
                errors="coerce",
            ),
            "frequency": pd.to_numeric(
                text.str.extract(r"(?:^|_)f(\d+)(?:_|$)", expand=False),
                errors="coerce",
            ),
        },
        index=sample_ids.index,
    )

    bad_mask = parsed.isna().any(axis=1)
    if bad_mask.any():
        bad_examples = text.loc[bad_mask].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"Could not parse building/antenna/frequency metadata for {label}. "
            f"Examples: {bad_examples}"
        )

    return parsed.astype(int)


def classify_transfer_condition(
    antenna: pd.Series,
    frequency: pd.Series,
) -> np.ndarray:
    """
    Classify target maps using the experiment's antenna/frequency protocol.

    Ant1-f1 remains on unseen buildings, so it is called source-like rather
    than fully in-distribution.
    """
    return np.select(
        [
            (antenna == 1) & (frequency == 1),
            (antenna == 1) & frequency.isin([2, 3]),
            antenna.isin([2, 3, 4, 5]),
        ],
        [
            "Ant1-f1 source-like",
            "Ant1-f2-f3 frequency shift",
            "Ant2-Ant5 additional antennas across frequencies",
        ],
        default="unexpected",
    )


# =============================================================================
# Loading and provenance checks
# =============================================================================

def load_six_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict] = []
    sample_frames: list[pd.DataFrame] = []

    required_methods = {WALLPATH_METHOD, *BASELINE_METHODS}

    for task, task_runs in RUNS.items():
        expected_maps = EXPECTED_MAPS_PER_RUN[task]

        for expected_seed, run_dir in task_runs.items():
            sample_path = run_dir / "per_sample_metrics.csv"
            summary_path = run_dir / "run_summary.json"

            for path in (sample_path, summary_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing required file for {task} seed {expected_seed}: {path}"
                    )

            samples = pd.read_csv(sample_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            require_columns(
                samples,
                ["sample_id", "method", "sparse_seed", "sparse_rate", *METRICS],
                f"{task} seed {expected_seed} per-sample metrics",
            )

            evaluated_seeds = (
                summary.get("extra", {}).get("evaluated_sparse_seeds")
            )
            observed_seeds = sorted(
                samples["sparse_seed"].astype(int).unique().tolist()
            )
            observed_rates = sorted(
                samples["sparse_rate"].astype(float).unique().tolist()
            )
            observed_methods = set(samples["method"].astype(str))
            maps_by_method = (
                samples.groupby("method")["sample_id"].nunique().to_dict()
            )
            duplicate_rows = int(
                samples.duplicated(
                    subset=["sample_id", "method", "sparse_seed"],
                    keep=False,
                ).sum()
            )

            checks = {
                "status_completed": summary.get("status") == "completed",
                "summary_seed_correct": evaluated_seeds == [expected_seed],
                "sample_seed_correct": observed_seeds == [expected_seed],
                "sample_rate_correct": observed_rates == [0.01],
                "required_methods_present": required_methods.issubset(observed_methods),
                "expected_maps_for_required_methods": all(
                    maps_by_method.get(method) == expected_maps
                    for method in required_methods
                ),
                "no_duplicate_method_map_seed_rows": duplicate_rows == 0,
            }
            audit_pass = all(checks.values())

            audit_rows.append(
                {
                    "task": task,
                    "expected_seed": expected_seed,
                    "run_directory": str(run_dir),
                    "evaluated_sparse_seeds": str(evaluated_seeds),
                    "observed_sparse_seeds": str(observed_seeds),
                    "observed_sparse_rates": str(observed_rates),
                    "per_sample_rows": len(samples),
                    "unique_maps": samples["sample_id"].nunique(),
                    "duplicate_rows": duplicate_rows,
                    **checks,
                    "audit_pass": audit_pass,
                }
            )

            if not audit_pass:
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(
                    f"{task} seed {expected_seed} failed provenance audit: {failed}"
                )

            metadata = parse_sample_metadata(
                samples["sample_id"],
                f"{task} seed {expected_seed}",
            )

            samples = samples.copy()
            samples["task"] = task
            samples["verified_seed"] = expected_seed
            samples["building"] = metadata["building"]
            samples["antenna"] = metadata["antenna"]
            samples["frequency"] = metadata["frequency"]
            samples["transfer_condition"] = classify_transfer_condition(
                antenna=samples["antenna"],
                frequency=samples["frequency"],
            )

            unexpected = samples[
                samples["transfer_condition"] == "unexpected"
            ][["sample_id", "antenna", "frequency"]].drop_duplicates()

            if not unexpected.empty:
                raise ValueError(
                    "Unexpected antenna/frequency combinations were found:\n"
                    + unexpected.head(20).to_string(index=False)
                )

            sample_frames.append(samples)

    audit = pd.DataFrame(audit_rows)
    all_samples = pd.concat(sample_frames, ignore_index=True)

    return audit, all_samples


# =============================================================================
# Paired map-level analysis
# =============================================================================

def build_seed_averaged_map_table(all_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Average each method's per-map metric over seeds 11, 22, and 33.

    This creates one row per task, map, and method. Statistical comparisons
    then remain paired by target map while avoiding treating the three seeds
    as three independent datasets.
    """
    selected = all_samples[
        all_samples["method"].isin([WALLPATH_METHOD, *BASELINE_METHODS])
    ].copy()

    grouping = [
        "task",
        "sample_id",
        "building",
        "antenna",
        "frequency",
        "transfer_condition",
        "method",
    ]

    seed_averaged = (
        selected.groupby(grouping, as_index=False)[METRICS]
        .mean()
    )

    seed_counts = (
        selected.groupby(grouping, as_index=False)["verified_seed"]
        .nunique()
        .rename(columns={"verified_seed": "num_seeds"})
    )
    seed_averaged = seed_averaged.merge(seed_counts, on=grouping, how="left")

    if not (seed_averaged["num_seeds"] == 3).all():
        bad = seed_averaged[seed_averaged["num_seeds"] != 3]
        raise RuntimeError(
            "Some map/method combinations do not contain all three seeds:\n"
            + bad.head(20).to_string(index=False)
        )

    return seed_averaged


def build_paired_map_comparisons(
    seed_averaged: pd.DataFrame,
) -> pd.DataFrame:
    index_columns = [
        "task",
        "sample_id",
        "building",
        "antenna",
        "frequency",
        "transfer_condition",
    ]

    paired_rows: list[pd.DataFrame] = []

    wallpath = seed_averaged[
        seed_averaged["method"] == WALLPATH_METHOD
    ].set_index(index_columns)

    for baseline_method in BASELINE_METHODS:
        baseline = seed_averaged[
            seed_averaged["method"] == baseline_method
        ].set_index(index_columns)

        common_index = wallpath.index.intersection(baseline.index)
        if len(common_index) != len(wallpath) or len(common_index) != len(baseline):
            raise RuntimeError(
                f"Map pairing mismatch between {WALLPATH_METHOD} and "
                f"{baseline_method}."
            )

        wallpath_aligned = wallpath.loc[common_index]
        baseline_aligned = baseline.loc[common_index]
        base_frame = common_index.to_frame(index=False)

        for metric in METRICS:
            wp_values = wallpath_aligned[metric].to_numpy(dtype=float)
            baseline_values = baseline_aligned[metric].to_numpy(dtype=float)

            valid_pair = np.isfinite(wp_values) & np.isfinite(baseline_values)
            differences = np.full(len(wp_values), np.nan, dtype=float)
            improvement_pct = np.full(len(wp_values), np.nan, dtype=float)

            differences[valid_pair] = (
                baseline_values[valid_pair] - wp_values[valid_pair]
            )

            nonzero_baseline = valid_pair & (baseline_values != 0)
            improvement_pct[nonzero_baseline] = (
                100.0
                * differences[nonzero_baseline]
                / baseline_values[nonzero_baseline]
            )

            winners = np.full(len(wp_values), "missing", dtype=object)
            winners[valid_pair] = [
                winner_from_difference(value)
                for value in differences[valid_pair]
            ]

            metric_frame = base_frame.copy()
            metric_frame["baseline_method"] = baseline_method
            metric_frame["metric"] = metric
            metric_frame["wallpath_value"] = wp_values
            metric_frame["baseline_value"] = baseline_values
            metric_frame["paired_difference"] = differences
            metric_frame["wallpath_improvement_percent"] = improvement_pct
            metric_frame["valid_pair"] = valid_pair
            metric_frame["winner"] = winners
            paired_rows.append(metric_frame)

    return pd.concat(paired_rows, ignore_index=True)

def summarize_map_win_rates(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    scope_definitions = [
        ("overall", None),
        *[
            (condition, condition)
            for condition in paired["transfer_condition"].drop_duplicates()
        ],
    ]

    for (task, baseline_method, metric), group in paired.groupby(
        ["task", "baseline_method", "metric"],
        sort=True,
    ):
        for scope_label, condition in scope_definitions:
            scoped_all = (
                group
                if condition is None
                else group[group["transfer_condition"] == condition]
            )

            if scoped_all.empty:
                continue

            scoped = scoped_all[scoped_all["valid_pair"]].copy()
            if scoped.empty:
                continue

            wallpath_wins = int((scoped["winner"] == WALLPATH_METHOD).sum())
            baseline_wins = int((scoped["winner"] == "baseline").sum())
            ties = int((scoped["winner"] == "tie").sum())
            valid_total = len(scoped)
            total_candidates = len(scoped_all)
            missing_pairs = total_candidates - valid_total

            mean_wallpath = scoped["wallpath_value"].mean()
            mean_baseline = scoped["baseline_value"].mean()

            rows.append(
                {
                    "task": task,
                    "scope": scope_label,
                    "baseline_method": baseline_method,
                    "metric": metric,
                    "num_candidate_maps": total_candidates,
                    "num_valid_maps": valid_total,
                    "num_missing_pairs": missing_pairs,
                    "wallpath_map_wins": wallpath_wins,
                    "baseline_map_wins": baseline_wins,
                    "ties": ties,
                    "wallpath_map_win_rate": wallpath_wins / valid_total,
                    "baseline_map_win_rate": baseline_wins / valid_total,
                    "mean_wallpath_value": mean_wallpath,
                    "mean_baseline_value": mean_baseline,
                    "mean_paired_difference": scoped["paired_difference"].mean(),
                    "median_paired_difference": scoped["paired_difference"].median(),
                    "mean_wallpath_improvement_percent": (
                        100.0
                        * (mean_baseline - mean_wallpath)
                        / mean_baseline
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Building-level summaries and cluster bootstrap
# =============================================================================

def summarize_buildings(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for scope, scoped_all in [
        ("overall", paired),
        *[
            (
                condition,
                paired[paired["transfer_condition"] == condition],
            )
            for condition in paired["transfer_condition"].drop_duplicates()
        ],
    ]:
        scoped = scoped_all[scoped_all["valid_pair"]].copy()
        if scoped.empty:
            continue

        grouped = (
            scoped.groupby(
                ["task", "building", "baseline_method", "metric"],
                as_index=False,
            )
            .agg(
                num_valid_maps=("sample_id", "nunique"),
                mean_wallpath_value=("wallpath_value", "mean"),
                mean_baseline_value=("baseline_value", "mean"),
                mean_paired_difference=("paired_difference", "mean"),
                median_paired_difference=("paired_difference", "median"),
                wallpath_map_win_rate=(
                    "winner",
                    lambda values: float(
                        np.mean(np.asarray(values) == WALLPATH_METHOD)
                    ),
                ),
            )
        )
        grouped["scope"] = scope
        grouped["wallpath_improvement_percent"] = (
            100.0
            * (
                grouped["mean_baseline_value"]
                - grouped["mean_wallpath_value"]
            )
            / grouped["mean_baseline_value"]
        )
        grouped["building_winner"] = grouped["mean_paired_difference"].map(
            winner_from_difference
        )
        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    columns = [
        "task",
        "scope",
        "building",
        "baseline_method",
        "metric",
        "num_valid_maps",
        "mean_wallpath_value",
        "mean_baseline_value",
        "mean_paired_difference",
        "median_paired_difference",
        "wallpath_improvement_percent",
        "wallpath_map_win_rate",
        "building_winner",
    ]
    return result[columns]


def cluster_bootstrap_from_building_means(
    building_results: pd.DataFrame,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample buildings as clusters and average their mean paired effects.

    The target set contains only B21-B25 (five clusters), so confidence
    intervals are informative but should be described as exploratory.
    """
    if iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1.")

    alpha = 1.0 - confidence
    rows: list[dict] = []

    grouped = building_results.groupby(
        ["task", "scope", "baseline_method", "metric"],
        sort=True,
    )

    for (task, scope, baseline_method, metric), group in grouped:
        group = group.sort_values("building")
        finite_mask = (
            np.isfinite(group["mean_paired_difference"].to_numpy(dtype=float))
            & np.isfinite(
                group["wallpath_improvement_percent"].to_numpy(dtype=float)
            )
        )
        group = group.loc[finite_mask].copy()

        differences = group["mean_paired_difference"].to_numpy(dtype=float)
        improvements = group["wallpath_improvement_percent"].to_numpy(dtype=float)
        buildings = group["building"].astype(int).tolist()

        n_clusters = len(differences)
        if n_clusters == 0:
            continue

        sample_indices = rng.integers(
            low=0,
            high=n_clusters,
            size=(iterations, n_clusters),
        )
        boot_differences = differences[sample_indices].mean(axis=1)
        boot_improvements = improvements[sample_indices].mean(axis=1)

        ci_low, ci_high = np.quantile(
            boot_differences,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        pct_ci_low, pct_ci_high = np.quantile(
            boot_improvements,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )

        positive = int((differences > 0).sum())
        negative = int((differences < 0).sum())
        ties = int((differences == 0).sum())

        rows.append(
            {
                "task": task,
                "scope": scope,
                "baseline_method": baseline_method,
                "metric": metric,
                "num_buildings": n_clusters,
                "buildings": ",".join(f"B{building}" for building in buildings),
                "cluster_mean_paired_difference": differences.mean(),
                "cluster_median_paired_difference": np.median(differences),
                "cluster_mean_improvement_percent": improvements.mean(),
                "bootstrap_iterations": iterations,
                "confidence": confidence,
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
                "improvement_percent_ci_low": pct_ci_low,
                "improvement_percent_ci_high": pct_ci_high,
                "bootstrap_probability_wallpath_better": float(
                    np.mean(boot_differences > 0)
                ),
                "wallpath_building_wins": positive,
                "baseline_building_wins": negative,
                "building_ties": ties,
                "sign_test_p_two_sided": exact_two_sided_sign_test(
                    positive,
                    negative,
                ),
                "difference_ci_excludes_zero": bool(
                    (ci_low > 0) or (ci_high < 0)
                ),
            }
        )

    return pd.DataFrame(rows)

# =============================================================================
# Per-seed consistency
# =============================================================================

def summarize_per_seed_consistency(all_samples: pd.DataFrame) -> pd.DataFrame:
    selected = all_samples[
        all_samples["method"].isin([WALLPATH_METHOD, *BASELINE_METHODS])
    ].copy()

    rows: list[dict] = []

    for (task, seed), run in selected.groupby(
        ["task", "verified_seed"],
        sort=True,
    ):
        index_columns = [
            "sample_id",
            "building",
            "antenna",
            "frequency",
            "transfer_condition",
        ]

        wallpath = run[
            run["method"] == WALLPATH_METHOD
        ].set_index(index_columns)

        for baseline_method in BASELINE_METHODS:
            baseline = run[
                run["method"] == baseline_method
            ].set_index(index_columns)

            common_index = wallpath.index.intersection(baseline.index)
            if len(common_index) != len(wallpath) or len(common_index) != len(baseline):
                raise RuntimeError(
                    f"Per-seed pairing mismatch for {task} seed {seed}, "
                    f"baseline {baseline_method}."
                )

            wallpath_aligned = wallpath.loc[common_index]
            baseline_aligned = baseline.loc[common_index]

            for metric in METRICS:
                wp_all = wallpath_aligned[metric].to_numpy(dtype=float)
                baseline_all = baseline_aligned[metric].to_numpy(dtype=float)
                valid = np.isfinite(wp_all) & np.isfinite(baseline_all)

                wp_values = wp_all[valid]
                baseline_values = baseline_all[valid]
                differences = baseline_values - wp_values

                if len(differences) == 0:
                    mean_wp = float("nan")
                    mean_baseline = float("nan")
                    mean_difference = float("nan")
                    median_difference = float("nan")
                    improvement = float("nan")
                    win_rate = float("nan")
                    winner = "missing"
                else:
                    mean_wp = float(np.mean(wp_values))
                    mean_baseline = float(np.mean(baseline_values))
                    mean_difference = float(np.mean(differences))
                    median_difference = float(np.median(differences))
                    improvement = (
                        100.0
                        * (mean_baseline - mean_wp)
                        / mean_baseline
                    )
                    win_rate = float(np.mean(differences > 0))
                    winner = winner_from_difference(mean_difference)

                rows.append(
                    {
                        "task": task,
                        "verified_seed": int(seed),
                        "baseline_method": baseline_method,
                        "metric": metric,
                        "num_candidate_maps": len(wp_all),
                        "num_valid_maps": len(differences),
                        "num_missing_pairs": len(wp_all) - len(differences),
                        "mean_wallpath_value": mean_wp,
                        "mean_baseline_value": mean_baseline,
                        "mean_paired_difference": mean_difference,
                        "median_paired_difference": median_difference,
                        "wallpath_improvement_percent": improvement,
                        "wallpath_map_win_rate": win_rate,
                        "winner": winner,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# JSON and Markdown reports
# =============================================================================

def build_headline_summary(
    map_win_rates: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> dict:
    headline: dict[str, dict] = {}

    for task in sorted(RUNS):
        task_summary: dict[str, dict] = {}

        for baseline_method in BASELINE_METHODS:
            baseline_summary: dict[str, dict] = {}

            for metric in METRICS:
                map_row = map_win_rates[
                    (map_win_rates["task"] == task)
                    & (map_win_rates["scope"] == "overall")
                    & (map_win_rates["baseline_method"] == baseline_method)
                    & (map_win_rates["metric"] == metric)
                ]
                cluster_row = cluster_bootstrap[
                    (cluster_bootstrap["task"] == task)
                    & (cluster_bootstrap["scope"] == "overall")
                    & (cluster_bootstrap["baseline_method"] == baseline_method)
                    & (cluster_bootstrap["metric"] == metric)
                ]
                seed_rows = per_seed[
                    (per_seed["task"] == task)
                    & (per_seed["baseline_method"] == baseline_method)
                    & (per_seed["metric"] == metric)
                ]

                if map_row.empty or cluster_row.empty:
                    continue

                map_record = map_row.iloc[0]
                cluster_record = cluster_row.iloc[0]

                baseline_summary[metric] = {
                    "mean_per_map_wallpath_value": float(
                        map_record["mean_wallpath_value"]
                    ),
                    "mean_per_map_baseline_value": float(
                        map_record["mean_baseline_value"]
                    ),
                    "mean_per_map_improvement_percent": float(
                        map_record["mean_wallpath_improvement_percent"]
                    ),
                    "wallpath_map_win_rate": float(
                        map_record["wallpath_map_win_rate"]
                    ),
                    "building_cluster_mean_difference": float(
                        cluster_record["cluster_mean_paired_difference"]
                    ),
                    "building_cluster_difference_ci": [
                        float(cluster_record["difference_ci_low"]),
                        float(cluster_record["difference_ci_high"]),
                    ],
                    "wallpath_building_wins": int(
                        cluster_record["wallpath_building_wins"]
                    ),
                    "num_buildings": int(cluster_record["num_buildings"]),
                    "sign_test_p_two_sided": float(
                        cluster_record["sign_test_p_two_sided"]
                    ),
                    "wallpath_seed_wins": int(
                        (seed_rows["winner"] == WALLPATH_METHOD).sum()
                    ),
                    "num_seeds": int(len(seed_rows)),
                }

            task_summary[baseline_method] = baseline_summary

        headline[task] = task_summary

    return headline


def write_markdown_report(
    output_path: Path,
    audit: pd.DataFrame,
    map_win_rates: pd.DataFrame,
    building_results: pd.DataFrame,
    cluster_bootstrap: pd.DataFrame,
    per_seed: pd.DataFrame,
    iterations: int,
    confidence: float,
) -> None:
    lines: list[str] = [
        "# Paired Transfer Statistics",
        "",
        (
            "This report analyzes Task1-to-Task2 and Task1-to-Task3 frozen "
            "transfer using paired per-map metrics. Each map is first averaged "
            "over seeds 11, 22, and 33. Positive paired differences mean that "
            "WallPath-PI has lower error than the comparison method."
        ),
        "",
        (
            "The building-cluster bootstrap resamples the five unseen target "
            "buildings (B21-B25). Because there are only five independent "
            "building clusters, its confidence intervals and sign tests should "
            "be described as exploratory rather than definitive."
        ),
        "",
        "## Provenance audit",
        "",
        "| Task | Seed | Runtime seed | Unique maps | Rows | Status |",
        "|---|---:|---|---:|---:|---|",
    ]

    for row in audit.itertuples(index=False):
        status = "PASS" if row.audit_pass else "FAIL"
        lines.append(
            f"| {row.task.upper()} | {row.expected_seed} "
            f"| {row.evaluated_sparse_seeds} | {row.unique_maps} "
            f"| {row.per_sample_rows} | {status} |"
        )

    direct_rf = "direct_rf_all_features"

    for task in sorted(RUNS):
        lines.extend(
            [
                "",
                f"## {task.upper()}: WallPath-PI versus Direct RF",
                "",
                (
                    "| Metric | Mean per-map WallPath | Mean per-map Direct RF "
                    "| Improvement | Valid-map wins | Building wins | "
                    f"{int(confidence * 100)}% cluster CI for difference "
                    "| Seed wins |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for metric in METRICS:
            map_row = map_win_rates[
                (map_win_rates["task"] == task)
                & (map_win_rates["scope"] == "overall")
                & (map_win_rates["baseline_method"] == direct_rf)
                & (map_win_rates["metric"] == metric)
            ]
            cluster_row = cluster_bootstrap[
                (cluster_bootstrap["task"] == task)
                & (cluster_bootstrap["scope"] == "overall")
                & (cluster_bootstrap["baseline_method"] == direct_rf)
                & (cluster_bootstrap["metric"] == metric)
            ]
            seed_rows = per_seed[
                (per_seed["task"] == task)
                & (per_seed["baseline_method"] == direct_rf)
                & (per_seed["metric"] == metric)
            ]

            if map_row.empty or cluster_row.empty:
                continue

            map_record = map_row.iloc[0]
            cluster_record = cluster_row.iloc[0]
            seed_wins = int((seed_rows["winner"] == WALLPATH_METHOD).sum())

            map_win_text = (
                f"{100.0 * map_record['wallpath_map_win_rate']:.1f}% "
                f"({int(map_record['wallpath_map_wins'])}/"
                f"{int(map_record['num_valid_maps'])})"
            )
            building_win_text = (
                f"{int(cluster_record['wallpath_building_wins'])}/"
                f"{int(cluster_record['num_buildings'])}"
            )
            ci_text = fmt_ci(
                cluster_record["difference_ci_low"],
                cluster_record["difference_ci_high"],
            )

            lines.append(
                f"| `{metric}` "
                f"| {fmt_float(map_record['mean_wallpath_value'])} "
                f"| {fmt_float(map_record['mean_baseline_value'])} "
                f"| {map_record['mean_wallpath_improvement_percent']:+.2f}% "
                f"| {map_win_text} "
                f"| {building_win_text} "
                f"| {ci_text} "
                f"| {seed_wins}/{len(seed_rows)} |"
            )

        lines.extend(
            [
                "",
                "### Transfer-condition breakdown for RMSE",
                "",
                (
                    "| Condition | Mean per-map WallPath | Mean per-map Direct RF "
                    "| Improvement | Valid-map wins | Building wins | Cluster CI |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        condition_rows = map_win_rates[
            (map_win_rates["task"] == task)
            & (map_win_rates["scope"] != "overall")
            & (map_win_rates["baseline_method"] == direct_rf)
            & (map_win_rates["metric"] == "rmse")
        ].sort_values("scope")

        for map_record in condition_rows.itertuples(index=False):
            cluster_row = cluster_bootstrap[
                (cluster_bootstrap["task"] == task)
                & (cluster_bootstrap["scope"] == map_record.scope)
                & (cluster_bootstrap["baseline_method"] == direct_rf)
                & (cluster_bootstrap["metric"] == "rmse")
            ]

            if cluster_row.empty:
                continue

            cluster_record = cluster_row.iloc[0]
            map_win_text = (
                f"{100.0 * map_record.wallpath_map_win_rate:.1f}% "
                f"({map_record.wallpath_map_wins}/{map_record.num_valid_maps})"
            )
            building_win_text = (
                f"{int(cluster_record['wallpath_building_wins'])}/"
                f"{int(cluster_record['num_buildings'])}"
            )
            ci_text = fmt_ci(
                cluster_record["difference_ci_low"],
                cluster_record["difference_ci_high"],
            )

            lines.append(
                f"| {map_record.scope} "
                f"| {fmt_float(map_record.mean_wallpath_value)} "
                f"| {fmt_float(map_record.mean_baseline_value)} "
                f"| {map_record.mean_wallpath_improvement_percent:+.2f}% "
                f"| {map_win_text} "
                f"| {building_win_text} "
                f"| {ci_text} |"
            )

        lines.extend(
            [
                "",
                "### Per-building RMSE",
                "",
                (
                    "| Building | Scope | Mean per-map WallPath | "
                    "Mean per-map Direct RF | Improvement | Winner |"
                ),
                "|---|---|---:|---:|---:|---|",
            ]
        )

        per_building = building_results[
            (building_results["task"] == task)
            & (building_results["scope"] == "overall")
            & (building_results["baseline_method"] == direct_rf)
            & (building_results["metric"] == "rmse")
        ].sort_values("building")

        for row in per_building.itertuples(index=False):
            winner = (
                "WallPath"
                if row.building_winner == WALLPATH_METHOD
                else "Direct RF"
                if row.building_winner == "baseline"
                else "Tie"
            )
            lines.append(
                f"| B{row.building} | overall "
                f"| {fmt_float(row.mean_wallpath_value)} "
                f"| {fmt_float(row.mean_baseline_value)} "
                f"| {row.wallpath_improvement_percent:+.2f}% "
                f"| {winner} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- Task-level values in this report are **means of per-map "
                "metrics**, not the pooled pixel-level metrics stored in "
                "`final_evaluation_results.csv`."
            ),
            (
                "- The three seeds are repeated matched runs over the same "
                "target maps. They are used for stability checks and are first "
                "averaged at map level for the paired analysis."
            ),
            (
                "- Map win rates use only finite paired values for the stated "
                "metric. Maps with undefined region-specific metrics are "
                "reported as missing pairs and excluded from that metric's "
                "denominator."
            ),
            (
                "- Maps within a building are not assumed independent. The "
                f"cluster bootstrap uses {iterations:,} replicates over the "
                "five target buildings."
            ),
            (
                "- With only five buildings, an exact two-sided sign test cannot "
                "reach p < 0.05 even when WallPath wins all five buildings "
                "(minimum p = 0.0625). Avoid overstating formal significance."
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit, all_samples = load_six_runs()
    seed_averaged = build_seed_averaged_map_table(all_samples)
    paired = build_paired_map_comparisons(seed_averaged)
    map_win_rates = summarize_map_win_rates(paired)
    building_results = summarize_buildings(paired)

    rng = np.random.default_rng(args.bootstrap_seed)
    cluster_bootstrap = cluster_bootstrap_from_building_means(
        building_results=building_results,
        iterations=args.bootstrap_iterations,
        confidence=args.confidence,
        rng=rng,
    )

    per_seed = summarize_per_seed_consistency(all_samples)

    headline = build_headline_summary(
        map_win_rates=map_win_rates,
        cluster_bootstrap=cluster_bootstrap,
        per_seed=per_seed,
    )

    output_paths = {
        "audit": args.output_dir / "six_run_statistical_audit.csv",
        "seed_averaged_maps": args.output_dir / "seed_averaged_map_metrics.csv",
        "paired_maps": args.output_dir / "paired_map_comparisons.csv",
        "map_win_rates": args.output_dir / "map_win_rates.csv",
        "building_results": args.output_dir / "building_level_results.csv",
        "cluster_bootstrap": args.output_dir / "building_cluster_bootstrap.csv",
        "per_seed": args.output_dir / "per_seed_paired_summary.csv",
        "json": args.output_dir / "statistical_summary.json",
        "report": args.output_dir / "statistical_report.md",
    }

    audit.to_csv(output_paths["audit"], index=False)
    seed_averaged.to_csv(output_paths["seed_averaged_maps"], index=False)
    paired.to_csv(output_paths["paired_maps"], index=False)
    map_win_rates.to_csv(output_paths["map_win_rates"], index=False)
    building_results.to_csv(output_paths["building_results"], index=False)
    cluster_bootstrap.to_csv(output_paths["cluster_bootstrap"], index=False)
    per_seed.to_csv(output_paths["per_seed"], index=False)
    output_paths["json"].write_text(
        json.dumps(headline, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        output_path=output_paths["report"],
        audit=audit,
        map_win_rates=map_win_rates,
        building_results=building_results,
        cluster_bootstrap=cluster_bootstrap,
        per_seed=per_seed,
        iterations=args.bootstrap_iterations,
        confidence=args.confidence,
    )

    print("=" * 110)
    print("TRANSFER STATISTICAL ANALYSIS")
    print("=" * 110)
    print("\nPROVENANCE AUDIT")
    print(
        audit[
            [
                "task",
                "expected_seed",
                "evaluated_sparse_seeds",
                "unique_maps",
                "per_sample_rows",
                "audit_pass",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("WALLPATH-PI VS DIRECT RF: OVERALL MEAN PER-MAP RESULTS")
    print("=" * 110)

    direct_rf_summary = map_win_rates[
        (map_win_rates["scope"] == "overall")
        & (map_win_rates["baseline_method"] == "direct_rf_all_features")
    ][
        [
            "task",
            "metric",
            "num_valid_maps",
            "num_missing_pairs",
            "mean_wallpath_value",
            "mean_baseline_value",
            "mean_wallpath_improvement_percent",
            "wallpath_map_win_rate",
        ]
    ].sort_values(["task", "metric"])

    print(direct_rf_summary.to_string(index=False))

    print("\n" + "=" * 110)
    print("BUILDING-CLUSTER BOOTSTRAP: WALLPATH-PI VS DIRECT RF")
    print("=" * 110)

    direct_rf_bootstrap = cluster_bootstrap[
        (cluster_bootstrap["scope"] == "overall")
        & (
            cluster_bootstrap["baseline_method"]
            == "direct_rf_all_features"
        )
    ][
        [
            "task",
            "metric",
            "cluster_mean_paired_difference",
            "difference_ci_low",
            "difference_ci_high",
            "wallpath_building_wins",
            "num_buildings",
            "sign_test_p_two_sided",
        ]
    ].sort_values(["task", "metric"])

    print(direct_rf_bootstrap.to_string(index=False))

    print("\n" + "=" * 110)
    print("ANALYSIS COMPLETE")
    print("=" * 110)
    print(
        "Positive paired differences mean lower error for WallPath-PI. "
        "Cluster inference is exploratory because only five target buildings "
        "are available."
    )

    print("\nWROTE:")
    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
