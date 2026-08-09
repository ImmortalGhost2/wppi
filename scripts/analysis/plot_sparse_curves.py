from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.utils.plotter import plot_sparse_curves


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RMSE versus sparse sampling rate from a WallPath-PI run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    out = plot_sparse_curves(args.run_dir / "per_sample_metrics.csv", args.run_dir / "plots")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
