from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.utils.config import load_config
from wallpath_pi.utils.paths import resolve_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the synthetic WallPath-PI smoke dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to a YAML config file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    data_root = resolve_data_root(cfg["data_root"], repo_root=REPO_ROOT)
    data_root.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(data_root, cfg.get("synthetic", {}))
    print(f"Generated {len(df)} samples under {data_root}")
    print(f"Manifest: {data_root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
