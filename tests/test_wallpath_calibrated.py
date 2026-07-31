from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.models.registry import make_regressor
from wallpath_pi.training.pipeline import (
    CALIBRATED_RESIDUAL_METHODS,
    _is_calibration_feature,
    _predictions_for_prepared,
    _prepare_sample,
    _train_models_for_rate_seed,
    _write_npz_payload,
    calibration_feature_names,
)
from wallpath_pi.utils.config import load_config


CALIBRATED = "wallpath_calibrated"


def test_make_regressor_calibrated_is_extra_trees():
    model = make_regressor(CALIBRATED, seed=0)
    assert isinstance(model, ExtraTreesRegressor)


def test_calibrated_is_known_method():
    # The verifier sources its known-method set from the pipeline registry, so
    # the calibrated learner must be recognized end-to-end.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.verify_repo_ready import KNOWN_METHODS

    assert CALIBRATED in KNOWN_METHODS
    assert CALIBRATED in CALIBRATED_RESIDUAL_METHODS


def test_calibration_feature_allowlist():
    allowed = [
        "x_norm", "y_norm",
        "distance_m", "log10_distance_m", "fspl_db",
        "wall_count", "wall_fraction", "los",
        "mat_1_count", "mat_2_count", "mat_3_count",
        "reflectance_sum", "transmittance_sum",
        "anchor_density",
        "anchor_dist_px_nn_1", "anchor_dist_px_nn_2",
        "anchor_dist_px_nn_3", "anchor_dist_px_nn_4",
        "fspl_pred_db", "log_distance_pred_db",
        "multi_wall_pred_db", "multi_wall_residual_idw_pred_db",
    ]
    for name in allowed:
        assert _is_calibration_feature(name) is True, name
    # Anchor-residual label features are forbidden inputs to the calibrated model.
    for forbidden in ("anchor_residual_idw", "anchor_residual_nn_1", "anchor_residual_nn_4"):
        assert _is_calibration_feature(forbidden) is False, forbidden

    full = allowed + ["anchor_residual_idw", "anchor_residual_nn_1", "anchor_residual_nn_2"]
    cal = calibration_feature_names(full)
    assert cal == allowed  # order preserved, residual-label features dropped
    assert not any(n.startswith("anchor_residual") for n in cal)


def _tiny_dataset(tmp_path: Path, methods):
    cfg = load_config(Path("configs/config.yaml"))
    synth_cfg = dict(cfg["synthetic"])
    synth_cfg.update({"num_scenes": 3, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth_cfg)
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
    cfg["max_train_points_per_sample"] = 200
    cfg["methods"] = list(methods)
    train_manifest = WallPathManifest(data_root / "train_split.csv", data_root)
    train_samples = list(train_manifest.iter_samples())
    return cfg, train_samples


def test_calibrated_uses_allowed_feature_subset_and_extra_trees(tmp_path):
    cfg, train_samples = _tiny_dataset(tmp_path, ["wallpath_extra", CALIBRATED])
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)

    assert CALIBRATED in models
    assert isinstance(models[CALIBRATED], ExtraTreesRegressor)

    cal_names = feature_names_by_method[CALIBRATED]
    full_names = feature_names_by_method["wallpath_extra"]
    # Calibrated uses a strict subset that drops anchor-residual label features.
    assert set(cal_names).issubset(set(full_names))
    assert not any(n.startswith("anchor_residual") for n in cal_names)
    assert "multi_wall_residual_idw_pred_db" in cal_names
    assert models[CALIBRATED].n_features_in_ == len(cal_names)


def test_calibrated_predictions_are_finite(tmp_path):
    cfg, train_samples = _tiny_dataset(tmp_path, [CALIBRATED])
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)
    sample = train_samples[0]
    prep = _prepare_sample(sample, cfg, sparse_rate=0.05, sparse_seed=7)
    preds = _predictions_for_prepared(prep, models, [CALIBRATED], feature_names_by_method=feature_names_by_method)
    arr = preds[CALIBRATED]
    assert arr.shape == sample.path_loss.shape
    valid = prep.feature_table.valid_mask
    assert np.isfinite(arr[valid]).all()
    # The calibrated map equals the residual-IDW base plus a finite correction.
    base = prep.multi_wall_residual_idw_pred
    assert np.isfinite((arr - base)[valid]).all()


def test_calibrated_does_not_change_existing_method_outputs(tmp_path):
    existing = ["direct_rf_all_features", "wallpath_rf", "wallpath_extra"]

    cfg_base, train_base = _tiny_dataset(tmp_path / "base", existing)
    models_base, fnbm_base = _train_models_for_rate_seed(train_base, cfg_base, 0.05, 7)

    cfg_cal, train_cal = _tiny_dataset(tmp_path / "cal", existing + [CALIBRATED])
    models_cal, fnbm_cal = _train_models_for_rate_seed(train_cal, cfg_cal, 0.05, 7)

    sample_base = train_base[0]
    sample_cal = train_cal[0]
    prep_base = _prepare_sample(sample_base, cfg_base, sparse_rate=0.05, sparse_seed=7)
    prep_cal = _prepare_sample(sample_cal, cfg_cal, sparse_rate=0.05, sparse_seed=7)

    preds_base = _predictions_for_prepared(prep_base, models_base, existing, feature_names_by_method=fnbm_base)
    preds_cal = _predictions_for_prepared(prep_cal, models_cal, existing, feature_names_by_method=fnbm_cal)

    for method in existing:
        np.testing.assert_array_equal(preds_base[method], preds_cal[method])


def test_calibrated_variable_shape_npz_export(tmp_path):
    shape_a = (4, 4)
    shape_b = (5, 3)
    payload = {
        "predictions_by_method": {
            "wallpath_extra": [np.ones(shape_a, np.float32), np.ones(shape_b, np.float32)],
            CALIBRATED: [np.full(shape_a, 2.0, np.float32), np.full(shape_b, 3.0, np.float32)],
        },
        "targets": [np.full(shape_a, 50.0, np.float32), np.full(shape_b, 60.0, np.float32)],
        "valid_masks": [np.ones(shape_a, np.uint8), np.ones(shape_b, np.uint8)],
        "wall_counts": [np.zeros(shape_a, np.float32), np.zeros(shape_b, np.float32)],
        "los_masks": [np.ones(shape_a, np.float32), np.ones(shape_b, np.float32)],
        "scene_ids": ["scene_0", "scene_1"],
        "sample_ids": ["scene_0_tx_00", "scene_1_tx_00"],
    }
    out = tmp_path / "eval_outputs_calibrated_var.npz"
    _write_npz_payload(payload, out)
    data = np.load(out, allow_pickle=True)
    assert bool(data["variable_shapes"])
    methods = [str(x) for x in data["methods"].tolist()]
    assert CALIBRATED in methods
    mi = methods.index(CALIBRATED)
    assert data["predictions"][mi, 0].shape == shape_a
    assert data["predictions"][mi, 1].shape == shape_b
