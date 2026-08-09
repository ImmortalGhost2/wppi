#!/usr/bin/env python
"""Aggregate WallPath-PI sparse-rate benchmark runs across seeds.

Each manuscript benchmark run (e.g. the ``final_sparse_seed11/22/33`` configs)
produces a ``final_evaluation_results.csv`` with one row per
``(method, sparse_rate, sparse_seed)``. This script reads several such runs,
groups by ``method`` and ``sparse_rate``, and reports the mean and standard
deviation across seeds for the headline metrics, writing both a machine-readable
CSV and a paper-facing Markdown table.

Paper-facing labels follow the project convention that ``wallpath_extra`` is the
proposed **WallPath-PI** method, while ``direct_rf_all_features`` is a
matched-feature direct comparator rather than a proposed method.

CLI
---
python scripts/analysis/summarize_sparse_seed_runs.py \
    results/wallpath_pi_task1_final_seed*_r*/run_1 \
    --out-dir results/summaries/task1_final
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Stable presentation order (proposed method last among the learners). Methods
# not listed here are appended afterwards in first-seen order.
METHOD_ORDER: Tuple[str, ...] = (
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
)

# Paper-facing labels. ``wallpath_extra`` is the proposed WallPath-PI method.
PAPER_LABELS = {
    "log_distance": "Log-distance",
    "multi_wall": "Multi-wall",
    "idw": "IDW",
    "multi_wall_residual_idw": "Multi-wall + IDW",
    "direct_rf_geometry": "Direct RF (geometry)",
    "direct_rf_sparse_anchor": "Direct RF (sparse anchor)",
    "direct_rf_all_features": "Direct RF (all features)",
    "direct_extra_all_features": "Direct ExtraTrees (all features)",
    "wallpath_rf": "WallPath-RF",
    "wallpath_extra": "WallPath-PI",
    "wallpath_calibrated": "WallPath-Calibrated",
}

# (csv column, paper label). Region metrics may be NaN on samples with too few
# pixels; they are dropped per metric rather than imputed.
METRIC_SPECS: Tuple[Tuple[str, str], ...] = (
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("p90_ae", "P90"),
    ("los_rmse", "LOS RMSE"),
    ("nlos_rmse", "NLOS RMSE"),
    ("high_wall_rmse", "High-wall RMSE"),
)

RESULTS_CSV_NAME = "final_evaluation_results.csv"


def paper_label(method: str) -> str:
    """Paper-facing display name for a method (falls back to a tidy default)."""
    m = str(method)
    return PAPER_LABELS.get(m, m.replace("_", " "))


def resolve_results_csv(path: Path) -> Path:
    """Resolve a user-supplied run path to its ``final_evaluation_results.csv``.

    Accepts either the CSV file directly, a run directory containing it, or a
    parent directory with a single ``run_*`` child that contains it.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".csv":
        return path
    candidate = path / RESULTS_CSV_NAME
    if candidate.exists():
        return candidate
    # Allow pointing at an experiment dir with run_* children.
    run_children = sorted(p for p in path.glob("run_*") if (p / RESULTS_CSV_NAME).exists())
    if len(run_children) == 1:
        return run_children[0] / RESULTS_CSV_NAME
    if len(run_children) > 1:
        raise ValueError(
            f"{path} contains multiple runs ({[p.name for p in run_children]}); "
            "pass the specific run directory."
        )
    raise FileNotFoundError(f"No {RESULTS_CSV_NAME} found under {path}.")


def load_runs(run_paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate the ``final_evaluation_results.csv`` of every run.

    Adds a ``source_run`` column (the run directory name) for traceability.
    """
    frames: List[pd.DataFrame] = []
    for raw in run_paths:
        csv_path = resolve_results_csv(Path(raw))
        df = pd.read_csv(csv_path)
        if "method" not in df.columns or "sparse_rate" not in df.columns:
            raise KeyError(f"{csv_path} is missing required columns 'method'/'sparse_rate'.")
        df = df.copy()
        df["source_run"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise ValueError("No run results were loaded.")
    return pd.concat(frames, ignore_index=True)


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    present = list(dict.fromkeys(str(m) for m in methods))
    known = [m for m in METHOD_ORDER if m in present]
    extra = [m for m in present if m not in METHOD_ORDER]
    return known + extra


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each headline metric across seeds, per (method, sparse_rate).

    Standard deviation uses the sample convention (ddof=1) and is NaN when only a
    single seed contributes. ``n_runs`` counts the contributing rows and
    ``seeds`` lists the distinct sparse seeds.
    """
    metric_cols = [c for c, _ in METRIC_SPECS if c in df.columns]
    rows: List[dict] = []
    rates = sorted(df["sparse_rate"].dropna().unique().tolist())
    method_order = _ordered_methods(df["method"].tolist())
    for rate in rates:
        sub_rate = df[df["sparse_rate"] == rate]
        for method in method_order:
            g = sub_rate[sub_rate["method"] == method]
            if g.empty:
                continue
            row: dict = {
                "method": method,
                "method_label": paper_label(method),
                "sparse_rate": float(rate),
                "n_runs": int(len(g)),
            }
            if "sparse_seed" in g.columns:
                seeds = sorted(int(s) for s in g["sparse_seed"].dropna().unique())
                row["seeds"] = ";".join(str(s) for s in seeds)
            for col in metric_cols:
                vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
                row[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
                row[f"{col}_n"] = int(vals.size)
            rows.append(row)
    return pd.DataFrame(rows)


def _format_cell(mean: float, std: float, decimals: int = 3) -> str:
    if not np.isfinite(mean):
        return "-"
    if np.isfinite(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def to_markdown(summary: pd.DataFrame, *, decimals: int = 3) -> str:
    """Render the aggregated summary as paper-facing Markdown (one table/rate)."""
    metrics = [(c, label) for c, label in METRIC_SPECS if f"{c}_mean" in summary.columns]
    lines: List[str] = []
    lines.append("# WallPath-PI sparse-seed benchmark summary")
    lines.append("")
    n_runs = int(summary["n_runs"].max()) if not summary.empty else 0
    all_seeds = sorted({s for v in summary.get("seeds", pd.Series(dtype=str)).dropna() for s in str(v).split(";") if s})
    seed_txt = ", ".join(all_seeds) if all_seeds else "n/a"
    lines.append(
        f"Aggregated across up to {n_runs} seed run(s) (seeds: {seed_txt}). "
        "Cells are mean ± std (dB) across seeds; a single seed shows the mean only."
    )
    lines.append("")
    lines.append(
        "Proposed method: **WallPath-PI** (`wallpath_extra`). "
        "`direct_rf_all_features` is a strong matched-feature direct comparator, not the "
        "proposed method; `wallpath_rf` and `wallpath_calibrated` are WallPath-PI "
        "variants."
    )
    header = "| Method | " + " | ".join(label for _, label in metrics) + " |"
    divider = "|" + "---|" * (len(metrics) + 1)
    for rate in sorted(summary["sparse_rate"].unique().tolist()):
        sub = summary[summary["sparse_rate"] == rate]
        lines.append("")
        lines.append(f"## Sparse rate = {rate:g}")
        lines.append("")
        lines.append(header)
        lines.append(divider)
        for _, r in sub.iterrows():
            cells = [str(r["method_label"])]
            for col, _ in metrics:
                cells.append(_format_cell(r.get(f"{col}_mean", float("nan")), r.get(f"{col}_std", float("nan")), decimals))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize WallPath-PI sparse-rate runs across seeds.")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Run directories (or final_evaluation_results.csv files).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: alongside the first run).")
    parser.add_argument("--out-prefix", type=str, default="sparse_seed_summary", help="Output file stem.")
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places in the Markdown table.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_runs(args.run_dirs)
    summary = aggregate_across_seeds(df)

    out_dir = Path(args.out_dir) if args.out_dir is not None else resolve_results_csv(Path(args.run_dirs[0])).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}.csv"
    md_path = out_dir / f"{args.out_prefix}.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(summary, decimals=int(args.decimals)), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
