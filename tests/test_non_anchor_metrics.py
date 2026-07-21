from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.evaluation.metrics import aggregate_metric_rows, compute_metrics
from wallpath_pi.training.pipeline import _evaluate_samples, _train_models_for_rate_seed
from wallpath_pi.utils.config import load_config


# --------------------------------------------------------------------------- #
# Non-anchor metric unit tests.
# --------------------------------------------------------------------------- #
def test_non_anchor_rmse_differs_when_anchors_perfect():
    # 8 valid pixels: 4 anchors (perfect) + 4 non-anchors (error 2).
    target = np.zeros((2, 4), dtype=np.float32)
    pred = np.zeros((2, 4), dtype=np.float32)
    pred[1, :] = 2.0  # non-anchor row has error 2
    valid = np.ones((2, 4), dtype=bool)
    sparse = np.zeros((2, 4), dtype=bool)
    sparse[0, :] = True  # first row are anchors (perfect predictions)

    out = compute_metrics(pred, target, valid, sparse_mask=sparse)
    # All-pixel RMSE is pulled down by the perfect anchors; non-anchor RMSE is not.
    assert out["rmse"] == pytest.approx(np.sqrt(2.0))
    assert out["non_anchor_rmse"] == pytest.approx(2.0)
    assert out["non_anchor_mae"] == pytest.approx(2.0)
    assert out["non_anchor_count"] == 4
    assert out["rmse"] < out["non_anchor_rmse"]


def test_non_anchor_metrics_ignore_anchor_pixels():
    # Anchors have huge error; non-anchors are perfect -> non-anchor RMSE is 0.
    target = np.zeros((1, 6), dtype=np.float32)
    pred = np.zeros((1, 6), dtype=np.float32)
    sparse = np.array([[True, True, False, False, False, False]])
    pred[0, :2] = 100.0  # anchor pixels: huge error
    valid = np.ones((1, 6), dtype=bool)

    out = compute_metrics(pred, target, valid, sparse_mask=sparse)
    assert out["non_anchor_rmse"] == pytest.approx(0.0)
    assert out["non_anchor_count"] == 4
    assert out["rmse"] > 0.0  # all-pixel RMSE still sees the bad anchors


def test_non_anchor_los_nlos_respect_both_masks():
    # 1x8: alternating anchors; LOS on the first half. Anchors carry huge error
    # that must be excluded from the non-anchor LOS/NLOS regions.
    target = np.zeros((1, 8), dtype=np.float32)
    pred = np.zeros((1, 8), dtype=np.float32)
    los = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float32)
    sparse = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=bool)
    # non-anchor LOS pixels (idx 1,3) -> error 3; non-anchor NLOS (idx 5,7) -> error 5.
    pred[0, 1] = pred[0, 3] = 3.0
    pred[0, 5] = pred[0, 7] = 5.0
    pred[0, 0] = pred[0, 2] = pred[0, 4] = pred[0, 6] = 100.0  # anchors: huge error
    valid = np.ones((1, 8), dtype=bool)

    out = compute_metrics(pred, target, valid, los_mask=los, sparse_mask=sparse, min_region_pixels=1)
    assert out["non_anchor_los_rmse"] == pytest.approx(3.0)
    assert out["non_anchor_nlos_rmse"] == pytest.approx(5.0)


def test_non_anchor_no_crash_when_all_valid_are_anchors():
    target = np.zeros((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    sparse = np.ones((2, 2), dtype=bool)  # every valid pixel is an anchor

    out = compute_metrics(pred, target, valid, sparse_mask=sparse, los_mask=np.ones((2, 2), np.float32))
    assert out["non_anchor_count"] == 0
    assert np.isnan(out["non_anchor_rmse"])
    assert np.isnan(out["non_anchor_mae"])
    assert np.isnan(out["non_anchor_p90_ae"])
    assert np.isnan(out["non_anchor_los_rmse"])


def test_metrics_unchanged_without_sparse_mask():
    # Backward compatibility: no sparse_mask -> no non_anchor_* keys.
    target = np.zeros((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    out = compute_metrics(pred, target, np.ones((2, 2), bool))
    assert not any(k.startswith("non_anchor_") for k in out)
    assert not any(k.startswith("free_space_") for k in out)


# --------------------------------------------------------------------------- #
# Free-space sensitivity metric tests.
# --------------------------------------------------------------------------- #
def test_free_space_metrics_split_wall_vs_non_wall():
    target = np.zeros((1, 6), dtype=np.float32)
    pred = np.zeros((1, 6), dtype=np.float32)
    wall = np.array([[True, True, False, False, False, False]])
    pred[0, :2] = 4.0   # wall pixels error 4
    pred[0, 2:] = 1.0   # free-space pixels error 1
    valid = np.ones((1, 6), dtype=bool)

    out = compute_metrics(pred, target, valid, wall_mask=wall)
    assert out["free_space_rmse"] == pytest.approx(1.0)
    assert out["free_space_count"] == 4
    assert out["wall_region_rmse"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Reporting: aggregation keeps the new columns.
# --------------------------------------------------------------------------- #
def test_aggregate_includes_non_anchor_and_free_space_columns():
    rows = [
        {"method": "m", "sparse_rate": 0.05, "sparse_seed": 11, "rmse": 1.5, "non_anchor_rmse": 2.0, "non_anchor_mae": 1.0, "free_space_rmse": 1.0},
        {"method": "m", "sparse_rate": 0.05, "sparse_seed": 11, "rmse": 2.5, "non_anchor_rmse": 4.0, "non_anchor_mae": 3.0, "free_space_rmse": 3.0},
    ]
    agg = aggregate_metric_rows(rows, ["method", "sparse_rate", "sparse_seed"])[0]
    assert agg["non_anchor_rmse"] == pytest.approx(3.0)
    assert "non_anchor_rmse_std" in agg
    assert agg["non_anchor_mae"] == pytest.approx(2.0)
    assert agg["free_space_rmse"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# End-to-end: per-sample rows from the evaluator include the new columns.
# --------------------------------------------------------------------------- #
def _tiny_dataset(tmp_path: Path, methods):
    cfg = load_config(Path("configs/config.yaml"))
    synth = dict(cfg["synthetic"])
    synth.update({"num_scenes": 3, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth)
    make_scene_disjoint_split(
        input_csv=data_root / "manifest.csv",
        train_out=data_root / "train_split.csv",
        val_out=data_root / "val_split.csv",
        meta_out=data_root / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.34,
        seed=7,
    )
    cfg["material_ids"] = [1, 2, 3]
    cfg["min_anchors"] = 4
    cfg["max_train_points_per_sample"] = 150
    cfg["methods"] = list(methods)
    train_samples = list(WallPathManifest(data_root / "train_split.csv", data_root).iter_samples())
    return cfg, train_samples


def test_evaluator_rows_include_non_anchor_and_free_space(tmp_path):
    methods = ["multi_wall", "wallpath_extra"]
    cfg, train_samples = _tiny_dataset(tmp_path, methods)
    models, fnbm = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)
    rows, _payload = _evaluate_samples(train_samples, cfg, 0.05, 7, models, feature_names_by_method=fnbm)

    assert rows, "evaluator produced no rows"
    for col in ("non_anchor_rmse", "non_anchor_mae", "non_anchor_p90_ae", "non_anchor_count",
                "non_anchor_los_rmse", "non_anchor_nlos_rmse",
                "free_space_rmse", "free_space_mae", "free_space_p90_ae", "wall_region_rmse"):
        assert col in rows[0], f"missing {col} in per-sample row"
    # Non-anchor count is strictly fewer than the valid count at a 5% rate.
    assert rows[0]["non_anchor_count"] <= rows[0]["valid_count"]
