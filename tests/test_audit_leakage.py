from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wallpath_pi.data.dataset import SceneSample
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.utils.config import load_config

AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "audit_leakage.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_leakage", AUDIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_dataset(tmp_path: Path):
    cfg = load_config(Path("configs/config.yaml"))
    synth_cfg = dict(cfg["synthetic"])
    synth_cfg.update({"num_scenes": 4, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth_cfg)
    make_scene_disjoint_split(
        input_csv=data_root / "manifest.csv",
        train_out=data_root / "train_split.csv",
        val_out=data_root / "val_split.csv",
        meta_out=data_root / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.5,
        seed=7,
    )
    # Point the loaded config at the temp dataset.
    cfg["data_root"] = str(data_root)
    cfg["train_csv"] = "train_split.csv"
    cfg["val_csv"] = "val_split.csv"
    cfg["eval_csv"] = "val_split.csv"
    cfg["min_anchors"] = 4
    return cfg, data_root


def test_leakage_audit_passes_on_clean_split(tmp_path):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)
    out = tmp_path / "audit_out"
    audit = mod.run_leakage_audit(cfg, repo_root=tmp_path, out_dir=out, max_samples=4, write_outputs=True, verbose=False)

    assert audit["overall"] == "PASS"
    assert (out / "leakage_audit.json").exists()
    assert (out / "leakage_audit.txt").exists()
    statuses = {c["id"]: c["status"] for c in audit["checks"]}
    # Core leakage checks must pass.
    for cid in (1, 2, 3, 5, 7, 8):
        assert statuses[cid] in {"PASS", "WARN"}, (cid, statuses[cid])
    assert statuses[1] == "PASS"  # scene-disjoint
    assert statuses[8] == "PASS"  # IDW empty-anchor behavior


def test_leakage_audit_fails_loudly_on_scene_overlap(tmp_path):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)
    # Inject leakage: copy one training scene row into the validation split.
    train = pd.read_csv(data_root / "train_split.csv")
    val = pd.read_csv(data_root / "val_split.csv")
    leaked = pd.concat([val, train.iloc[[0]]], ignore_index=True)
    leaked.to_csv(data_root / "val_split.csv", index=False)

    out = tmp_path / "audit_out"
    audit = mod.run_leakage_audit(cfg, repo_root=tmp_path, out_dir=out, max_samples=4, write_outputs=True, verbose=False)

    assert audit["overall"] == "FAIL"
    check1 = next(c for c in audit["checks"] if c["id"] == 1)
    assert check1["status"] == "FAIL"
    assert check1["metrics"]["overlap_count"] >= 1


def test_leakage_audit_idw_check_is_self_contained(tmp_path):
    mod = _load_audit()
    result = mod.check_idw_empty_anchor_behavior()
    assert result.status == "PASS"
    assert result.check_id == 8


def test_leakage_audit_missing_manifest_raises(tmp_path):
    mod = _load_audit()
    cfg = load_config(Path("configs/config.yaml"))
    cfg["data_root"] = str(tmp_path / "does_not_exist")
    with __import__("pytest").raises(FileNotFoundError):
        mod.run_leakage_audit(cfg, repo_root=tmp_path, write_outputs=False, verbose=False)


def _toy_sample(tmp_path: Path, h: int = 30, w: int = 30) -> SceneSample:
    valid = np.ones((h, w), dtype=bool)
    return SceneSample(
        scene_id="s0", sample_id="s0_tx_00", scene_path=tmp_path / "x.npz",
        path_loss=np.zeros((h, w), dtype=np.float32), valid_mask=valid,
        wall_mask=np.zeros((h, w), dtype=bool), material_map=np.zeros((h, w), dtype=np.int32),
        tx_xy=np.asarray([1.0, 1.0], dtype=np.float32), frequency_hz=3.5e9, resolution_m=0.25,
    )


def test_feature_query_indices_subsets_and_is_reproducible(tmp_path):
    mod = _load_audit()
    sample = _toy_sample(tmp_path)  # 900 valid pixels

    idx, n_valid = mod._feature_query_indices(sample, max_feature_pixels=50, quick=True)
    assert n_valid == 900
    assert idx is not None and idx.size == 50
    assert np.all(np.diff(idx) > 0)  # sorted and unique
    assert idx.min() >= 0 and idx.max() < 900

    idx2, _ = mod._feature_query_indices(sample, max_feature_pixels=50, quick=True)
    assert np.array_equal(idx, idx2)  # reproducible across calls

    none_full, n2 = mod._feature_query_indices(sample, max_feature_pixels=50, quick=False)
    assert none_full is None and n2 == 900  # full mode never subsets

    # When valid pixels already fit under the cap, quick mode also keeps the full table.
    none_small, _ = mod._feature_query_indices(sample, max_feature_pixels=2000, quick=True)
    assert none_small is None


def test_quick_mode_does_not_build_full_dense_feature_tables(tmp_path, monkeypatch):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)
    real_prepare = mod._prepare_sample
    rows_seen: list[int] = []
    qi_seen: list[object] = []

    def spy(sample, config, sparse_rate, sparse_seed, query_indices=None):
        qi_seen.append(query_indices)
        prep = real_prepare(sample, config, sparse_rate, sparse_seed, query_indices=query_indices)
        rows_seen.append(int(prep.feature_table.X.shape[0]))
        return prep

    monkeypatch.setattr(mod, "_prepare_sample", spy)
    audit = mod.run_leakage_audit(
        cfg, repo_root=tmp_path, out_dir=tmp_path / "q", max_samples=4,
        quick=True, max_feature_pixels=20, write_outputs=False, verbose=False,
    )

    assert audit["mode"] == "quick"
    assert audit["max_feature_pixels"] == 20
    assert rows_seen, "feature preparation was never exercised"
    # Every feature table built in quick mode is restricted to the pixel budget.
    assert all(r <= 20 for r in rows_seen)
    assert all(q is not None for q in qi_seen)
    assert audit["n_checked_pixels"] <= 20 * max(1, audit["n_checked_samples"])
    assert audit["overall"] == "PASS"


def test_full_mode_builds_dense_feature_tables(tmp_path, monkeypatch):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)
    real_prepare = mod._prepare_sample
    rows_seen: list[int] = []

    def spy(sample, config, sparse_rate, sparse_seed, query_indices=None):
        prep = real_prepare(sample, config, sparse_rate, sparse_seed, query_indices=query_indices)
        rows_seen.append(int(prep.feature_table.X.shape[0]))
        return prep

    monkeypatch.setattr(mod, "_prepare_sample", spy)
    audit = mod.run_leakage_audit(
        cfg, repo_root=tmp_path, out_dir=tmp_path / "f", max_samples=4,
        quick=False, max_feature_pixels=20, write_outputs=False, verbose=False,
    )

    assert audit["mode"] == "full"
    # Full mode builds the complete dense table, which exceeds the quick budget.
    assert max(rows_seen) > 20
    assert audit["overall"] == "PASS"


def test_quick_mode_limits_samples_and_pixels(tmp_path, monkeypatch):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)  # validation split has 2 samples
    real_prepare = mod._prepare_sample
    prepared_ids: list[str] = []
    rows_seen: list[int] = []

    def spy(sample, config, sparse_rate, sparse_seed, query_indices=None):
        prepared_ids.append(str(sample.sample_id))
        prep = real_prepare(sample, config, sparse_rate, sparse_seed, query_indices=query_indices)
        rows_seen.append(int(prep.feature_table.X.shape[0]))
        return prep

    monkeypatch.setattr(mod, "_prepare_sample", spy)
    audit = mod.run_leakage_audit(
        cfg, repo_root=tmp_path, out_dir=tmp_path / "q", max_samples=1,
        quick=True, max_feature_pixels=15, write_outputs=False, verbose=False,
    )

    # Sample budget: at most one distinct sample is ever prepared.
    assert audit["max_samples"] == 1
    assert audit["n_checked_samples"] == 1
    assert len(set(prepared_ids)) == 1
    # Feature-pixel budget: no feature table exceeds the pixel cap.
    assert audit["max_feature_pixels"] == 15
    assert rows_seen and all(r <= 15 for r in rows_seen)
    assert audit["n_checked_pixels"] <= 15
    assert audit["overall"] == "PASS"


def test_quick_and_full_default_sample_budgets(tmp_path):
    mod = _load_audit()
    cfg, data_root = _build_dataset(tmp_path)

    quick = mod.run_leakage_audit(cfg, repo_root=tmp_path, out_dir=tmp_path / "q", quick=True, write_outputs=False, verbose=False)
    assert quick["mode"] == "quick"
    assert quick["max_samples"] == 3
    assert quick["max_feature_pixels"] == 500

    full = mod.run_leakage_audit(cfg, repo_root=tmp_path, out_dir=tmp_path / "f", quick=False, write_outputs=False, verbose=False)
    assert full["mode"] == "full"
    assert full["max_samples"] == 8
