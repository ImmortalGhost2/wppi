from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_internal_split import main


def _make_manifest(path: Path, scenes, per: int = 2) -> Path:
    rows = []
    for s in scenes:
        for k in range(per):
            sid = f"{s}_S{k}"
            rows.append({"scene_id": s, "sample_id": sid, "scene_path": f"scenes/{sid}.npz"})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(path: Path, data_root: Path, split_block: dict) -> Path:
    cfg = {
        "paths": {
            "data_root": str(data_root),
            "results_root": "./results",
            "train_csv": "train_split.csv",
            "val_csv": "val_split.csv",
            "eval_csv": "val_split.csv",
        },
        "dataset": {"material_ids": [1, 2, 3]},
        "split": split_block,
        "experiment": {"experiment_name": "test_split", "seed": 42},
        "train": {"methods": ["multi_wall"], "sparse_rates": [0.01], "sparse_seeds": [11]},
        "baseline": {"d0_m": 1.0},
        "evaluation": {"metrics": ["rmse"]},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


SCENES = [f"B{i}" for i in range(1, 9)]  # B1..B8


def test_main_fixed_scene_split_from_config(tmp_path):
    data_root = tmp_path / "data"
    _make_manifest(data_root / "manifest.csv", SCENES, per=2)
    cfg = _write_config(
        tmp_path / "cfg.yaml",
        data_root,
        {"group_column": "scene_id", "train_scenes": [f"B{i}" for i in range(1, 7)], "val_scenes": ["B7", "B8"]},
    )
    assert main(["--config", str(cfg)]) == 0

    train = pd.read_csv(data_root / "train_split.csv")
    val = pd.read_csv(data_root / "val_split.csv")
    assert set(val["scene_id"].astype(str)) == {"B7", "B8"}
    assert set(train["scene_id"].astype(str)) == {f"B{i}" for i in range(1, 7)}
    assert set(train["scene_id"]).isdisjoint(set(val["scene_id"]))
    summary = (data_root / "split_summary.json").read_text(encoding="utf-8")
    assert "fixed_scenes" in summary


def test_main_val_ratio_fallback_when_no_fixed_scenes(tmp_path):
    data_root = tmp_path / "data"
    _make_manifest(data_root / "manifest.csv", SCENES, per=2)
    cfg = _write_config(tmp_path / "cfg.yaml", data_root, {"group_column": "scene_id", "val_ratio": 0.25, "seed": 42})
    assert main(["--config", str(cfg)]) == 0

    train = pd.read_csv(data_root / "train_split.csv")
    val = pd.read_csv(data_root / "val_split.csv")
    # Backward-compatible ratio split: disjoint, non-empty, covers all scenes.
    assert not val.empty and not train.empty
    assert set(train["scene_id"]).isdisjoint(set(val["scene_id"]))
    assert set(train["scene_id"].astype(str)) | set(val["scene_id"].astype(str)) == set(SCENES)


def test_main_cli_scene_overrides(tmp_path):
    data_root = tmp_path / "data"
    _make_manifest(data_root / "manifest.csv", SCENES, per=2)
    # Config has val_ratio; CLI fixed lists take precedence.
    cfg = _write_config(tmp_path / "cfg.yaml", data_root, {"group_column": "scene_id", "val_ratio": 0.25, "seed": 42})
    rc = main([
        "--config", str(cfg),
        "--train-scenes", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "--val-scenes", "B8",
    ])
    assert rc == 0
    val = pd.read_csv(data_root / "val_split.csv")
    assert set(val["scene_id"].astype(str)) == {"B8"}


def test_main_only_one_fixed_list_raises(tmp_path):
    data_root = tmp_path / "data"
    _make_manifest(data_root / "manifest.csv", SCENES, per=2)
    cfg = _write_config(tmp_path / "cfg.yaml", data_root, {"group_column": "scene_id", "train_scenes": ["B1", "B2"]})
    with pytest.raises(ValueError, match="both"):
        main(["--config", str(cfg)])
