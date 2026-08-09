from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.training.pipeline import run_experiment
from wallpath_pi.utils.config import load_config


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.experiment_name is not None:
        cfg["experiment_name"] = args.experiment_name
        cfg["_full_config"].setdefault("experiment", {})["experiment_name"] = args.experiment_name
    if args.sparse_rates is not None:
        rates = [float(x) for x in args.sparse_rates.split(",") if x.strip()]
        cfg["sparse_rates"] = rates
        cfg["_full_config"].setdefault("train", {})["sparse_rates"] = rates
    if args.sparse_seeds is not None:
        seeds = [int(x) for x in args.sparse_seeds.split(",") if x.strip()]
        cfg["sparse_seeds"] = seeds
        cfg["_full_config"].setdefault("train", {})["sparse_seeds"] = seeds
    cfg["repo_root"] = str(REPO_ROOT.resolve())
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate WallPath-PI baselines and residual models.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to a YAML config file.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--sparse_rates", type=str, default=None, help="Comma-separated override, e.g. 0.02,0.05")
    parser.add_argument("--sparse_seeds", type=str, default=None, help="Comma-separated override, e.g. 11,23")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _apply_overrides(load_config(args.config), args)
    run_dir = run_experiment(cfg)
    print(f"Run complete: {run_dir}")
    print(f"Summary: {run_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
