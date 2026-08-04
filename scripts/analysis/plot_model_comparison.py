from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.utils.plotter import plot_model_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot model comparison for one sparse rate.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sparse_rate", type=float, default=None)
    parser.add_argument("--metric", type=str, default="rmse")
    args = parser.parse_args()
    out = plot_model_comparison(args.run_dir / "per_sample_metrics.csv", args.run_dir / "plots", sparse_rate=args.sparse_rate, metric=args.metric)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
