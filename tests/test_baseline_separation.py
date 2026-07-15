from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.models.registry import make_regressor
from wallpath_pi.training.pipeline import (
    ALL_FEATURE_DIRECT_METHODS,
    GEOMETRY_DIRECT_METHODS,
    _is_geometry_feature,
    _predictions_for_prepared,
    _prepare_sample,
    _train_models_for_rate_seed,
    geometry_feature_names,
)
from wallpath_pi.utils.config import load_config


# Features that encode a path-loss value or a baseline-corrected prediction and
# must NEVER be used by the geometry-only baseline.
BANNED_PREFIXES = ("anchor_residual",)
BANNED_EXACT = {
    "fspl_pred_db",
    "log_distance_pred_db",
    "multi_wall_pred_db",
    "multi_wall_residual_idw_pred_db",
    "anchor_residual_idw",
}


def _is_banned(name: str) -> bool:
    return name in BANNED_EXACT or any(name.startswith(p) for p in BANNED_PREFIXES)


def test_make_regressor_accepts_new_baseline_names():
    for name in ("direct_rf_all_features", "all_feature_rf", "direct_rf_geometry"):
        model = make_regressor(name, seed=0)
        assert isinstance(model, RandomForestRegressor)


def test_is_geometry_feature_excludes_pred_and_residual():
    # Excluded: baseline-prediction and residual-label features.
    for banned in BANNED_EXACT | {"anchor_residual_nn_0", "anchor_residual_nn_2"}:
        assert _is_geometry_feature(banned) is False, banned
    # Kept: pure geometry / material / anchor-distance features.
    for kept in (
        "x_norm", "y_norm", "distance_m", "log10_distance_m", "fspl_db",
        "los", "wall_count", "wall_fraction", "mat_1_count", "transmittance_sum",
        "reflectance_sum", "anchor_dist_px_nn_0", "anchor_density",
    ):
        assert _is_geometry_feature(kept) is True, kept


def test_geometry_feature_names_drops_banned_keeps_geometry():
    full = [
        "x_norm", "y_norm", "distance_m", "log10_distance_m", "fspl_db",
        "los", "wall_count", "wall_fraction", "mat_1_count", "transmittance_sum",
        "reflectance_sum",
        "fspl_pred_db", "log_distance_pred_db", "multi_wall_pred_db",
        "multi_wall_residual_idw_pred_db",
        "anchor_residual_idw", "anchor_residual_nn_0", "anchor_residual_nn_1",
        "anchor_dist_px_nn_0", "anchor_dist_px_nn_1", "anchor_density",
    ]
    geo = geometry_feature_names(full)
    # No banned feature survives.
    assert not any(_is_banned(n) for n in geo)
    # Strict subset that preserves order.
    assert geo == [n for n in full if n in geo]
    assert set(geo).issubset(set(full))
    # Geometry-only essentials are retained.
    for kept in ("fspl_db", "anchor_dist_px_nn_0", "anchor_density", "mat_1_count"):
        assert kept in geo


def _tiny_dataset(tmp_path: Path):
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
    cfg["methods"] = ["direct_rf_geometry", "direct_rf_all_features", "wallpath_rf"]
    train_manifest = WallPathManifest(data_root / "train_split.csv", data_root)
    train_samples = list(train_manifest.iter_samples())
    return cfg, data_root, train_samples


def test_direct_rf_geometry_excludes_banned_features(tmp_path):
    cfg, _data_root, train_samples = _tiny_dataset(tmp_path)
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)

    assert set(models) == {"direct_rf_geometry", "direct_rf_all_features", "wallpath_rf"}

    full_names = feature_names_by_method["direct_rf_all_features"]
    geo_names = feature_names_by_method["direct_rf_geometry"]

    # The all-feature scenario must actually contain banned features, otherwise
    # the geometry separation would be trivially satisfied.
    assert any(_is_banned(n) for n in full_names)

    # The geometry baseline uses none of the banned features ...
    assert not any(_is_banned(n) for n in geo_names)
    # ... and is a strict subset of the full feature set.
    assert set(geo_names).issubset(set(full_names))
    assert len(geo_names) < len(full_names)

    # The fitted geometry model was trained on exactly the geometry columns.
    assert models["direct_rf_geometry"].n_features_in_ == len(geo_names)
    # The all-feature direct RF and WallPath-PI keep the complete feature set.
    assert models["direct_rf_all_features"].n_features_in_ == len(full_names)
    assert feature_names_by_method["wallpath_rf"] == full_names
    assert models["wallpath_rf"].n_features_in_ == len(full_names)


def test_geometry_baseline_predicts_full_maps(tmp_path):
    cfg, _data_root, train_samples = _tiny_dataset(tmp_path)
    models, feature_names_by_method = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)

    sample = train_samples[0]
    prep = _prepare_sample(sample, cfg, sparse_rate=0.05, sparse_seed=7)
    preds = _predictions_for_prepared(
        prep, models, cfg["methods"], feature_names_by_method=feature_names_by_method
    )
    for method in cfg["methods"]:
        arr = preds[method]
        assert arr.shape == sample.path_loss.shape
        assert np.isfinite(arr[prep.feature_table.valid_mask]).all()
