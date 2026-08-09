from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.utils.plotter import plot_prediction_maps


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot prediction and error maps from saved eval_outputs npz.")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["multi_wall", "multi_wall_residual_idw", "direct_extra_all_features", "wallpath_extra"])
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()
    npz = args.npz or (args.run_dir / "eval_outputs_primary.npz")
    out_dir = (args.run_dir / "plots") if args.run_dir is not None else npz.parent
    out = plot_prediction_maps(npz, out_dir, sample_index=args.sample_index, methods=args.methods)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
