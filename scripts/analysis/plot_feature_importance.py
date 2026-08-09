from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.utils.plotter import plot_feature_importance


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot feature importances from a WallPath-PI run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    out = plot_feature_importance(args.run_dir / "feature_importances.csv", args.run_dir / "plots", top_k=args.top_k)
    print(out if out is not None else "No feature importance CSV or no importances available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
