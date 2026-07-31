from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.models.registry import make_regressor
from wallpath_pi.training.pipeline import (
    SPARSE_ANCHOR_DIRECT_METHODS,
    _is_sparse_anchor_feature,
    _predictions_for_prepared,
    _prepare_sample,
    _train_models_for_rate_seed,
    sparse_anchor_feature_names,
)
from wallpath_pi.utils.config import load_config


SPARSE_ANCHOR = "direct_rf_sparse_anchor"

# Residual shortcut features the sparse-anchor baseline must never consume.
FORBIDDEN_RESIDUAL_FEATURES = {
    "multi_wall_residual_idw_pred_db",
    "anchor_residual_idw",
    "anchor_residual_nn_1",
    "anchor_residual_nn_2",
    "anchor_residual_nn_3",
    "anchor_residual_nn_4",
}


def test_make_regressor_sparse_anchor_is_random_forest():
    model = make_regressor(SPARSE_ANCHOR, seed=0)
    assert isinstance(model, RandomForestRegressor)


def test_sparse_anchor_is_known_method():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.verify_repo_ready import KNOWN_METHODS

    assert SPARSE_ANCHOR in KNOWN_METHODS
    assert SPARSE_ANCHOR in SPARSE_ANCHOR_DIRECT_METHODS


def test_sparse_anchor_feature_allowlist_excludes_residual_shortcuts():
    # Forbidden residual shortcut features are rejected one by one.
    for forbidden in FORBIDDEN_RESIDUAL_FEATURES:
        assert _is_sparse_anchor_feature(forbidden) is False, forbidden
    # Sparse-anchor observation and geometry/material features are allowed,
    # including the anchor-fitted physics predictions and anchor distances.
    for allowed in (
        "x_norm", "y_norm",
        "distance_m", "log10_distance_m", "fspl_db",
        "wall_count", "wall_fraction", "los",
        "mat_1_count", "mat_2_count", "mat_3_count",
        "reflectance_sum", "transmittance_sum",
        "anchor_density",
        "anchor_dist_px_nn_1", "anchor_dist_px_nn_2",
        "anchor_dist_px_nn_3", "anchor_dist_px_nn_4",
        "fspl_pred_db", "log_distance_pred_db", "multi_wall_pred_db",
    ):
        assert _is_sparse_anchor_feature(allowed) is True, allowed

    full = [
        "x_norm", "distance_m", "fspl_db", "wall_count", "mat_1_count",
        "fspl_pred_db", "log_distance_pred_db", "multi_wall_pred_db",
        "multi_wall_residual_idw_pred_db",
        "anchor_residual_idw", "anchor_residual_nn_1", "anchor_residual_nn_2",
        "anchor_dist_px_nn_1", "anchor_dist_px_nn_2", "anchor_density",
    ]
    sa = sparse_anchor_feature_names(full)
    # Order preserved, residual shortcut features dropped.
    assert sa == [n for n in full if n not in FORBIDDEN_RESIDUAL_FEATURES]
    assert not (set(sa) & FORBIDDEN_RESIDUAL_FEATURES)
    # The baseline still sees genuine sparse-anchor observation information.
    for kept in ("multi_wall_pred_db", "anchor_dist_px_nn_1", "anchor_density"):
        assert kept in sa


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


def test_sparse_anchor_trains_without_forbidden_features(tmp_path):
    cfg, train_samples = _tiny_dataset(tmp_path, ["direct_rf_all_features", SPARSE_ANCHOR])
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)

    assert SPARSE_ANCHOR in models
    assert isinstance(models[SPARSE_ANCHOR], RandomForestRegressor)

    sa_names = feature_names_by_method[SPARSE_ANCHOR]
    full_names = feature_names_by_method["direct_rf_all_features"]

    # The full feature set actually contains forbidden features, otherwise the
    # exclusion would be trivially satisfied.
    assert FORBIDDEN_RESIDUAL_FEATURES & set(full_names)
    # The sparse-anchor baseline uses none of them, and is a strict subset.
    assert not (FORBIDDEN_RESIDUAL_FEATURES & set(sa_names))
    assert set(sa_names).issubset(set(full_names))
    assert len(sa_names) < len(full_names)
    # It still uses anchor-informed observation features.
    assert "multi_wall_pred_db" in sa_names
    assert any(n.startswith("anchor_dist_px_nn_") for n in sa_names)
    assert "anchor_density" in sa_names
    # The fitted model consumed exactly the sparse-anchor columns.
    assert models[SPARSE_ANCHOR].n_features_in_ == len(sa_names)


def test_sparse_anchor_predicts_finite_full_maps(tmp_path):
    cfg, train_samples = _tiny_dataset(tmp_path, [SPARSE_ANCHOR])
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)
    sample = train_samples[0]
    prep = _prepare_sample(sample, cfg, sparse_rate=0.05, sparse_seed=7)
    preds = _predictions_for_prepared(prep, models, [SPARSE_ANCHOR], feature_names_by_method=feature_names_by_method)
    arr = preds[SPARSE_ANCHOR]
    assert arr.shape == sample.path_loss.shape
    assert np.isfinite(arr[prep.feature_table.valid_mask]).all()


def test_sparse_anchor_does_not_change_existing_method_outputs(tmp_path):
    existing = ["direct_rf_geometry", "direct_rf_all_features", "wallpath_rf", "wallpath_extra"]

    cfg_base, train_base = _tiny_dataset(tmp_path / "base", existing)
    models_base, fnbm_base = _train_models_for_rate_seed(train_base, cfg_base, 0.05, 7)

    cfg_sa, train_sa = _tiny_dataset(tmp_path / "sa", existing + [SPARSE_ANCHOR])
    models_sa, fnbm_sa = _train_models_for_rate_seed(train_sa, cfg_sa, 0.05, 7)

    sample_base = train_base[0]
    sample_sa = train_sa[0]
    prep_base = _prepare_sample(sample_base, cfg_base, sparse_rate=0.05, sparse_seed=7)
    prep_sa = _prepare_sample(sample_sa, cfg_sa, sparse_rate=0.05, sparse_seed=7)

    preds_base = _predictions_for_prepared(prep_base, models_base, existing, feature_names_by_method=fnbm_base)
    preds_sa = _predictions_for_prepared(prep_sa, models_sa, existing, feature_names_by_method=fnbm_sa)

    for method in existing:
        np.testing.assert_array_equal(preds_base[method], preds_sa[method])
