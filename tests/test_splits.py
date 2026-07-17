from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wallpath_pi.data.splits import make_fixed_scene_split, make_scene_disjoint_split


def _write_manifest(path: Path, num_scenes: int, samples_per_scene: int = 3) -> Path:
    rows = []
    for s in range(num_scenes):
        for k in range(samples_per_scene):
            sample_id = f"scene_{s:03d}_tx_{k:02d}"
            rows.append(
                {
                    "scene_id": f"scene_{s:03d}",
                    "sample_id": sample_id,
                    "scene_path": f"scenes/{sample_id}.npz",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_split_is_scene_disjoint_and_writes_summary(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=8)
    meta = make_scene_disjoint_split(
        input_csv=manifest,
        train_out=tmp_path / "train_split.csv",
        val_out=tmp_path / "val_split.csv",
        meta_out=tmp_path / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.25,
        seed=42,
        summary_out=tmp_path / "split_summary.json",
    )
    train = pd.read_csv(tmp_path / "train_split.csv")
    val = pd.read_csv(tmp_path / "val_split.csv")
    train_scenes = set(train["scene_id"].astype(str))
    val_scenes = set(val["scene_id"].astype(str))

    assert train_scenes.isdisjoint(val_scenes)
    assert train_scenes | val_scenes == {f"scene_{i:03d}" for i in range(8)}
    assert len(train) + len(val) == len(pd.read_csv(manifest))

    summary = json.loads((tmp_path / "split_summary.json").read_text())
    assert summary["group_column"] == "scene_id"
    assert summary["group_disjoint"] is True
    assert summary["samples"]["train"] == int(len(train))
    assert summary["samples"]["val"] == int(len(val))
    assert summary["scenes"]["train"] == len(train_scenes)
    assert summary["scenes"]["val"] == len(val_scenes)
    assert meta["groups"]["val"] == len(val_scenes)


def test_split_is_reproducible_by_seed(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=10)

    def _run(out_dir: Path, seed: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        make_scene_disjoint_split(
            input_csv=manifest,
            train_out=out_dir / "train_split.csv",
            val_out=out_dir / "val_split.csv",
            meta_out=out_dir / "split_meta.json",
            group_column="scene_id",
            val_ratio=0.3,
            seed=seed,
        )
        return json.loads((out_dir / "split_meta.json").read_text())["val_groups"]

    a = _run(tmp_path / "a", seed=7)
    b = _run(tmp_path / "b", seed=7)
    c = _run(tmp_path / "c", seed=99)
    assert a == b
    assert a != c  # a different seed should generally yield a different partition


def test_split_fails_on_too_few_groups(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=3)
    with pytest.raises(ValueError, match="unique 'scene_id'"):
        make_scene_disjoint_split(
            input_csv=manifest,
            train_out=tmp_path / "train_split.csv",
            val_out=tmp_path / "val_split.csv",
            meta_out=tmp_path / "split_meta.json",
            group_column="scene_id",
            val_ratio=0.25,
            seed=42,
            min_groups=5,
        )


def test_split_fails_on_missing_group_column(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=4)
    with pytest.raises(ValueError, match="Missing group column"):
        make_scene_disjoint_split(
            input_csv=manifest,
            train_out=tmp_path / "train_split.csv",
            val_out=tmp_path / "val_split.csv",
            meta_out=tmp_path / "split_meta.json",
            group_column="layout_id",
            val_ratio=0.25,
            seed=42,
        )


def test_split_fails_on_invalid_val_ratio(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=4)
    with pytest.raises(ValueError, match="val_ratio"):
        make_scene_disjoint_split(
            input_csv=manifest,
            train_out=tmp_path / "train_split.csv",
            val_out=tmp_path / "val_split.csv",
            meta_out=tmp_path / "split_meta.json",
            group_column="scene_id",
            val_ratio=1.0,
            seed=42,
        )


def _fixed_split(tmp_path: Path, num_scenes: int, train_scenes, val_scenes):
    manifest = _write_manifest(tmp_path / "manifest.csv", num_scenes=num_scenes, samples_per_scene=3)
    return make_fixed_scene_split(
        input_csv=manifest,
        train_out=tmp_path / "train_split.csv",
        val_out=tmp_path / "val_split.csv",
        meta_out=tmp_path / "split_meta.json",
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        group_column="scene_id",
        summary_out=tmp_path / "split_summary.json",
    )


def test_fixed_scene_split_exact_assignment_and_summary(tmp_path):
    train_scenes = [f"scene_{i:03d}" for i in range(6)]
    val_scenes = [f"scene_{i:03d}" for i in range(6, 8)]
    meta = _fixed_split(tmp_path, num_scenes=8, train_scenes=train_scenes, val_scenes=val_scenes)

    train = pd.read_csv(tmp_path / "train_split.csv")
    val = pd.read_csv(tmp_path / "val_split.csv")
    train_set = set(train["scene_id"].astype(str))
    val_set = set(val["scene_id"].astype(str))

    # Exact requested assignment, disjoint, exact partition of the manifest.
    assert train_set == set(train_scenes)
    assert val_set == set(val_scenes)
    assert train_set.isdisjoint(val_set)
    assert len(train) == 6 * 3 and len(val) == 2 * 3
    assert len(train) + len(val) == len(pd.read_csv(tmp_path / "manifest.csv"))

    assert meta["split_mode"] == "fixed_scenes"
    assert meta["train_groups"] == sorted(train_scenes)
    assert meta["val_groups"] == sorted(val_scenes)

    summary = json.loads((tmp_path / "split_summary.json").read_text())
    assert summary["split_mode"] == "fixed_scenes"
    assert summary["scenes"]["train"] == 6
    assert summary["scenes"]["val"] == 2
    assert summary["group_disjoint"] is True


def test_fixed_scene_split_overlap_raises(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        _fixed_split(tmp_path, num_scenes=3, train_scenes=["scene_000", "scene_001"], val_scenes=["scene_001", "scene_002"])


def test_fixed_scene_split_missing_scene_raises(tmp_path):
    with pytest.raises(ValueError, match="not present"):
        _fixed_split(tmp_path, num_scenes=4, train_scenes=["scene_000", "scene_001"], val_scenes=["scene_002", "scene_999"])


def test_fixed_scene_split_uncovered_scene_raises(tmp_path):
    # scene_004 is assigned to neither list.
    with pytest.raises(ValueError, match="neither"):
        _fixed_split(tmp_path, num_scenes=5, train_scenes=["scene_000", "scene_001"], val_scenes=["scene_002", "scene_003"])


def test_fixed_scene_split_duplicate_entries_raise(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _fixed_split(tmp_path, num_scenes=4, train_scenes=["scene_000", "scene_000", "scene_001"], val_scenes=["scene_002", "scene_003"])

