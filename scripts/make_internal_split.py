from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.data.splits import make_fixed_scene_split, make_scene_disjoint_split
from wallpath_pi.utils.config import load_config
from wallpath_pi.utils.paths import resolve_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic scene-disjoint train/validation splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to a YAML config file.")
    parser.add_argument("--group-column", type=str, default=None, help="Override the disjoint group column (default: config or scene_id).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Override the validation group fraction in (0, 1).")
    parser.add_argument("--seed", type=int, default=None, help="Override the deterministic shuffle seed.")
    parser.add_argument("--min-groups", type=int, default=2, help="Fail if fewer than this many unique groups exist.")
    parser.add_argument("--train-scenes", type=str, nargs="*", default=None, help="Fixed train scene_ids (overrides config split.train_scenes).")
    parser.add_argument("--val-scenes", type=str, nargs="*", default=None, help="Fixed validation scene_ids (overrides config split.val_scenes).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    raw = cfg.get("_full_config", {}) or {}
    paths = raw.get("paths", {}) or {}
    split = raw.get("split", {}) or {}

    data_root = resolve_data_root(cfg["data_root"], repo_root=REPO_ROOT)
    input_csv = data_root / "manifest.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing {input_csv}; run scripts/generate_synthetic_data.py first or prepare a manifest.csv.")

    group_column = args.group_column if args.group_column is not None else str(split.get("group_column", "scene_id"))

    train_out = data_root / str(paths.get("train_csv", "train_split.csv"))
    val_out = data_root / str(paths.get("val_csv", "val_split.csv"))
    meta_out = data_root / "split_meta.json"
    summary_out = data_root / "split_summary.json"

    # Fixed scene lists take precedence over val_ratio (CLI overrides config).
    train_scenes = args.train_scenes if args.train_scenes is not None else split.get("train_scenes")
    val_scenes = args.val_scenes if args.val_scenes is not None else split.get("val_scenes")

    if train_scenes and val_scenes:
        meta = make_fixed_scene_split(
            input_csv=input_csv,
            train_out=train_out,
            val_out=val_out,
            meta_out=meta_out,
            train_scenes=list(train_scenes),
            val_scenes=list(val_scenes),
            group_column=group_column,
            min_groups=int(args.min_groups),
            summary_out=summary_out,
        )
        mode = "Fixed scene-disjoint"
    elif train_scenes or val_scenes:
        raise ValueError(
            "A fixed split needs both split.train_scenes and split.val_scenes (or both "
            "--train-scenes and --val-scenes); only one was provided."
        )
    else:
        val_ratio = args.val_ratio if args.val_ratio is not None else float(split.get("val_ratio", 0.25))
        seed = args.seed if args.seed is not None else int(split.get("seed", 42))
        meta = make_scene_disjoint_split(
            input_csv=input_csv,
            train_out=train_out,
            val_out=val_out,
            meta_out=meta_out,
            group_column=group_column,
            val_ratio=val_ratio,
            seed=seed,
            min_groups=int(args.min_groups),
            summary_out=summary_out,
        )
        mode = "Scene-disjoint (val_ratio)"
    print(f"{mode} split complete")
    print(f"Train: {meta['counts']['train']} rows / {meta['groups']['train']} scenes -> {train_out}")
    print(f"Val:   {meta['counts']['val']} rows / {meta['groups']['val']} scenes -> {val_out}")
    print(f"Meta:  {meta_out}")
    print(f"Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
