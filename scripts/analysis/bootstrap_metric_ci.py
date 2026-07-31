from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, confidence: float = 0.95, seed: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = values[rng.integers(0, len(values), size=len(values))]
        draws[i] = float(np.mean(sample))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(values)),
        "bootstrap_mean": float(np.mean(draws)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals from per_sample_metrics.csv.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to a completed run directory.")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_csv = args.run_dir / "per_sample_metrics.csv"
    if not in_csv.exists():
        raise FileNotFoundError(
            f"Missing {in_csv}; pass a run directory produced by scripts/train.py via --run_dir."
        )
    df = pd.read_csv(in_csv)
    if args.metric not in df.columns:
        raise KeyError(f"Metric '{args.metric}' not found in {in_csv}")
    rows = []
    for (rate, method), g in df.groupby(["sparse_rate", "method"], dropna=False):
        stats = bootstrap_mean_ci(g[args.metric].to_numpy(), n_boot=args.n_boot, confidence=args.confidence, seed=args.seed)
        rows.append({"sparse_rate": float(rate), "method": method, "metric": args.metric, "n": int(len(g)), "n_boot": int(args.n_boot), "confidence": float(args.confidence), **stats})
    out = pd.DataFrame(rows).sort_values(["sparse_rate", "mean"])
    out_path = args.run_dir / f"bootstrap_{args.metric}_ci.csv"
    out.to_csv(out_path, index=False)
    print(out.to_markdown(index=False))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
