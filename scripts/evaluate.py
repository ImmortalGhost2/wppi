from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.training.pipeline import evaluate_saved_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked re-evaluation from a saved WallPath-PI run directory.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to a completed run directory.")
    parser.add_argument("--csv", type=str, default=None, help="Evaluation CSV name relative to data_root, default from locked config.")
    parser.add_argument("--sparse_rate", type=float, default=None)
    parser.add_argument("--sparse_seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_csv = evaluate_saved_run(args.run_dir, csv_name=args.csv, sparse_rate=args.sparse_rate, sparse_seed=args.sparse_seed)
    df = pd.read_csv(out_csv)
    print(df.to_markdown(index=False))
    print(f"Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
