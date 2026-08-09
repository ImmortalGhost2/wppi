#!/usr/bin/env python
"""Optional point-level validation on the 2026 measured 3.5 GHz indoor dataset.

LEGACY / NON-PAPER / HISTORICAL COMPATIBILITY ONLY.
This workflow is not the source of the final manuscript results; use
``scripts/analysis/evaluate_external_3p5ghz.py`` for the canonical pipeline.

This script answers a narrow question for the paper's real-data section: does
WallPath-PI-style residual learning (a physics/geometry baseline plus a learned
residual) also help on *measured point data*? It operates only on the tidy table
produced by ``convert_measured_3p5ghz.py`` and is deliberately independent of the
dense radio-map pipeline. It never reconstructs maps; it fits point regressors
under grouped cross-validation so no scenario/campaign leaks across folds.

Example
-------
python scripts/analysis/evaluate_measured_3p5ghz.py \
  --config configs/config_measured_3p5ghz.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ("log_distance", "multi_wall_point", "direct_rf_point", "residual_rf_point")
DEFAULT_WALL_COLUMNS = (
    "brick_wall_count",
    "wood_wall_count",
    "glass_wall_count",
    "drywall_count",
    "column_count",
)

DEFAULTS: Dict[str, object] = {
    "input_csv": "./data/measured_3p5ghz_processed/measured_points.csv",
    "out_dir": "./results/measured_3p5ghz_validation",
    "group_column": "scenario",
    "fallback_group_column": "campaign_id",
    "target_column": "measured_path_loss_db",
    "distance_column": "distance_m",
    "frequency_column": "frequency_hz",
    "frequency_hz_default": 3.5e9,
    "los_column": "los_nlos",
    "wall_count_columns": list(DEFAULT_WALL_COLUMNS),
    "n_splits": 5,
    "seed": 42,
    "high_wall_count_threshold": 2,
    "make_plots": True,
    "model_params": {
        "use_extra_trees": False,
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "n_jobs": -1,
    },
}


# Metrics
def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    err = y_pred[valid] - y_true[valid]
    abs_err = np.abs(err)
    if err.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "median_ae": float("nan"), "p90_ae": float("nan"), "n": 0}
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "n": int(err.size),
    }


# Feature assembly
class PointData:
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, object]) -> None:
        target_col = str(cfg["target_column"])
        dist_col = str(cfg["distance_column"])
        if target_col not in df.columns:
            raise ValueError(f"Missing target column '{target_col}' in the measured table.")
        if dist_col not in df.columns:
            raise ValueError(f"Missing distance column '{dist_col}' in the measured table.")

        group_col = _resolve_group_column(df, cfg)
        self.group_column = group_col

        work = df.copy()
        work["_y"] = pd.to_numeric(work[target_col], errors="coerce")
        work["_dist"] = pd.to_numeric(work[dist_col], errors="coerce")
        before = len(work)
        work = work[np.isfinite(work["_y"]) & np.isfinite(work["_dist"]) & (work["_dist"] > 0)].reset_index(drop=True)
        self.n_dropped = int(before - len(work))
        if work.empty:
            raise ValueError("No rows with finite target and positive distance remain after cleaning.")

        self.y = work["_y"].to_numpy(dtype=np.float64)
        self.log10_dist = np.log10(work["_dist"].to_numpy(dtype=np.float64))
        self.groups = work[group_col].astype(str).to_numpy()

        self.wall_columns = [c for c in cfg["wall_count_columns"] if c in work.columns]
        self.walls = {c: pd.to_numeric(work[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for c in self.wall_columns}
        self.wall_total = (
            np.sum([self.walls[c] for c in self.wall_columns], axis=0)
            if self.wall_columns
            else np.zeros(len(work), dtype=np.float64)
        )

        freq_col = str(cfg["frequency_column"])
        if freq_col in work.columns:
            freq = pd.to_numeric(work[freq_col], errors="coerce").fillna(float(cfg["frequency_hz_default"]))
        else:
            freq = pd.Series(np.full(len(work), float(cfg["frequency_hz_default"])))
        self.frequency_hz = freq.to_numpy(dtype=np.float64)

        los_col = str(cfg["los_column"])
        if los_col in work.columns:
            self.los_label = work[los_col].astype(str).to_numpy()
            self.los_numeric = np.array([1.0 if str(v).strip().upper() == "LOS" else 0.0 for v in self.los_label])
            self.has_los = bool(np.isin(self.los_label, ["LOS", "NLOS"]).any())
        else:
            self.los_label = None
            self.los_numeric = None
            self.has_los = False

        self.scenario = work["scenario"].astype(str).to_numpy() if "scenario" in work.columns else self.groups
        self._n = len(work)

    def __len__(self) -> int:
        return self._n

    def linear_logdist(self) -> np.ndarray:
        return self.log10_dist.reshape(-1, 1)

    def linear_multiwall(self) -> np.ndarray:
        cols = [self.log10_dist] + [self.walls[c] for c in self.wall_columns]
        return np.column_stack(cols)

    def rf_features(self) -> Tuple[np.ndarray, List[str]]:
        cols = [self.log10_dist, self.frequency_hz]
        names = ["log10_distance", "frequency_hz"]
        for c in self.wall_columns:
            cols.append(self.walls[c])
            names.append(c)
        if self.los_numeric is not None:
            cols.append(self.los_numeric)
            names.append("los_numeric")
        return np.column_stack(cols), names


def _resolve_group_column(df: pd.DataFrame, cfg: Dict[str, object]) -> str:
    primary = str(cfg["group_column"])
    fallback = str(cfg.get("fallback_group_column", "")) if cfg.get("fallback_group_column") else ""
    if primary in df.columns and df[primary].nunique() >= 2:
        return primary
    if fallback and fallback in df.columns and df[fallback].nunique() >= 2:
        return fallback
    if primary in df.columns:
        return primary
    raise ValueError(
        f"Group column '{primary}' is unusable and no valid fallback exists. "
        "Grouped cross-validation needs at least two groups."
    )


# Models
def _make_forest(cfg: Dict[str, object]) -> object:
    mp = dict(DEFAULTS["model_params"])  # type: ignore[arg-type]
    mp.update(cfg.get("model_params", {}) or {})
    kwargs = dict(
        n_estimators=int(mp["n_estimators"]),
        max_depth=int(mp["max_depth"]),
        min_samples_leaf=int(mp["min_samples_leaf"]),
        n_jobs=int(mp["n_jobs"]),
        random_state=int(cfg["seed"]),
    )
    return ExtraTreesRegressor(**kwargs) if mp.get("use_extra_trees") else RandomForestRegressor(**kwargs)


def _predict_fold(data: PointData, train_idx: np.ndarray, test_idx: np.ndarray, cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    y = data.y
    preds: Dict[str, np.ndarray] = {}

    # Log-distance linear fit.
    Xld = data.linear_logdist()
    ld = LinearRegression().fit(Xld[train_idx], y[train_idx])
    preds["log_distance"] = ld.predict(Xld[test_idx])

    # Multi-wall point fit (log-distance plus wall-count terms).
    Xmw = data.linear_multiwall()
    mw = LinearRegression().fit(Xmw[train_idx], y[train_idx])
    preds["multi_wall_point"] = mw.predict(Xmw[test_idx])

    # Direct random forest on point features.
    Xrf, _ = data.rf_features()
    rf = _make_forest(cfg).fit(Xrf[train_idx], y[train_idx])
    preds["direct_rf_point"] = rf.predict(Xrf[test_idx])

    # Residual random forest on the multi-wall baseline residual.
    residual_train = y[train_idx] - mw.predict(Xmw[train_idx])
    rrf = _make_forest(cfg).fit(Xrf[train_idx], residual_train)
    preds["residual_rf_point"] = mw.predict(Xmw[test_idx]) + rrf.predict(Xrf[test_idx])

    return preds


# Cross-validation
def run_cross_validation(data: PointData, cfg: Dict[str, object]) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, int]:
    unique_groups = np.unique(data.groups)
    n_groups = int(unique_groups.size)
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for grouped CV; found {n_groups} in '{data.group_column}'.")
    n_splits = min(int(cfg["n_splits"]), n_groups)
    splitter = GroupKFold(n_splits=n_splits)

    oof = {m: np.full(len(data), np.nan, dtype=np.float64) for m in METHODS}
    fold_rows: List[Dict[str, object]] = []
    indices = np.arange(len(data))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, data.y, groups=data.groups)):
        preds = _predict_fold(data, train_idx, test_idx, cfg)
        for method in METHODS:
            oof[method][test_idx] = preds[method]
            m = error_metrics(data.y[test_idx], preds[method])
            fold_rows.append({"method": method, "fold": fold, **m, "n_test": int(test_idx.size)})

    return oof, pd.DataFrame(fold_rows), n_splits


# Aggregation
def _grouped_rows(method: str, partition: str, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    for value in pd.unique(labels):
        sel = labels == value
        m = error_metrics(y_true[sel], y_pred[sel])
        rows.append({"method": method, "partition": partition, "group": str(value), **m})
    return rows


def _wall_region(wall_total: np.ndarray) -> np.ndarray:
    region = np.where(wall_total <= 0, "0_walls", np.where(wall_total <= 2, "1-2_walls", "3+_walls"))
    return region


def aggregate_metrics(data: PointData, oof: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    wall_region = _wall_region(data.wall_total) if data.wall_columns else None
    for method in METHODS:
        pred = oof[method]
        rows.append({"method": method, "partition": "overall", "group": "all", **error_metrics(data.y, pred)})
        rows.extend(_grouped_rows(method, "scenario", data.y, pred, data.scenario))
        if data.has_los and data.los_label is not None:
            rows.extend(_grouped_rows(method, "los_nlos", data.y, pred, data.los_label))
        if wall_region is not None:
            rows.extend(_grouped_rows(method, "wall_region", data.y, pred, wall_region))
    return pd.DataFrame(rows)


# Plots (optional)
def make_plots(metrics: pd.DataFrame, out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    overall = metrics[metrics["partition"] == "overall"].set_index("method").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(range(len(METHODS)), overall["rmse"].to_numpy(), color="#3b6fb6")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Legacy measured 3.5 GHz validation: pooled point-level RMSE by method")
    fig.tight_layout()
    p = plots_dir / "overall_rmse_by_method.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    scen = metrics[metrics["partition"] == "scenario"]
    if not scen.empty:
        scenarios = sorted(scen["group"].unique())
        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(scenarios)), 4.6))
        width = 0.8 / len(METHODS)
        x = np.arange(len(scenarios))
        for i, method in enumerate(METHODS):
            sub = scen[scen["method"] == method].set_index("group").reindex(scenarios)
            ax.bar(x + i * width, sub["rmse"].to_numpy(), width=width, label=method)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_ylabel("RMSE (dB)")
        ax.set_title("Scenario-wise RMSE by method")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = plots_dir / "scenario_rmse_by_method.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    return written


# Driver
def load_eval_config(config_path: Optional[Path]) -> Dict[str, object]:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy of defaults
    if config_path is not None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        for key, value in raw.items():
            if key == "model_params" and isinstance(value, dict):
                cfg["model_params"].update(value)  # type: ignore[union-attr]
            else:
                cfg[key] = value
    return cfg


def evaluate_measured(cfg: Dict[str, object], *, write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    input_csv = Path(str(cfg["input_csv"])).expanduser()
    if not input_csv.is_absolute():
        input_csv = (REPO_ROOT / input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Measured point table not found: {input_csv}. Run "
            "scripts/converters/convert_measured_3p5ghz.py first."
        )
    df = pd.read_csv(input_csv)
    data = PointData(df, cfg)
    oof, per_fold, n_splits = run_cross_validation(data, cfg)
    metrics = aggregate_metrics(data, oof)

    overall = {
        m: metrics[(metrics["method"] == m) & (metrics["partition"] == "overall")].iloc[0][["rmse", "mae", "median_ae", "p90_ae", "n"]].to_dict()
        for m in METHODS
    }
    best_method = min(METHODS, key=lambda m: overall[m]["rmse"] if np.isfinite(overall[m]["rmse"]) else np.inf)

    summary: Dict[str, object] = {
        "input_csv": str(input_csv),
        "group_column": data.group_column,
        "n_groups": int(np.unique(data.groups).size),
        "n_splits": int(n_splits),
        "n_rows_used": int(len(data)),
        "n_rows_dropped": int(data.n_dropped),
        "wall_count_columns": data.wall_columns,
        "has_los": bool(data.has_los),
        "methods": list(METHODS),
        "overall": overall,
        "best_method_by_rmse": best_method,
    }

    if verbose:
        _print_summary(summary)

    if write_outputs:
        out_dir = Path(str(cfg["out_dir"])).expanduser()
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_dir / "measured_validation_metrics.csv", index=False)
        per_fold.to_csv(out_dir / "measured_validation_per_fold.csv", index=False)
        (out_dir / "measured_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot_paths = make_plots(metrics, out_dir) if cfg.get("make_plots", True) else []
        summary["plots"] = plot_paths
        summary["out_dir"] = str(out_dir)
        if verbose:
            print(f"Wrote metrics, per-fold, and summary to {out_dir} ({len(plot_paths)} plot(s)).")

    summary["_metrics_table"] = metrics
    summary["_per_fold_table"] = per_fold
    return summary


def _print_summary(summary: Dict[str, object]) -> None:
    print("=" * 66)
    print("Measured 3.5 GHz point-level validation (grouped CV)")
    print("=" * 66)
    print(f"Input                : {summary['input_csv']}")
    print(f"Grouping column      : {summary['group_column']} ({summary['n_groups']} groups, {summary['n_splits']} folds)")
    print(f"Rows used / dropped  : {summary['n_rows_used']} / {summary['n_rows_dropped']}")
    print(f"Wall-count columns   : {summary['wall_count_columns']}")
    print(f"LOS/NLOS available   : {summary['has_los']}")
    print("Overall RMSE (dB):")
    for method in summary["methods"]:
        o = summary["overall"][method]
        print(f"  {method:<20} rmse={o['rmse']:.3f}  mae={o['mae']:.3f}  median_ae={o['median_ae']:.3f}  p90_ae={o['p90_ae']:.3f}")
    print(f"Best by RMSE         : {summary['best_method_by_rmse']}")
    print("=" * 66)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-level validation on measured 3.5 GHz indoor data.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config (configs/config_measured_3p5ghz.yaml).")
    parser.add_argument("--input-csv", type=str, default=None, help="Override the measured_points.csv path.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override the output directory.")
    parser.add_argument("--group-column", type=str, default=None, help="Grouping column for CV (default: scenario).")
    parser.add_argument("--n-splits", type=int, default=None, help="Number of grouped CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the forests.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_eval_config(args.config)
    if args.input_csv is not None:
        cfg["input_csv"] = args.input_csv
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.group_column is not None:
        cfg["group_column"] = args.group_column
    if args.n_splits is not None:
        cfg["n_splits"] = args.n_splits
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_plots:
        cfg["make_plots"] = False
    evaluate_measured(cfg, write_outputs=True, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
