from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.models.registry import make_regressor
from wallpath_pi.training.pipeline import (
    ALL_FEATURE_DIRECT_EXTRA_METHODS,
    DIRECT_METHODS,
    _predictions_for_prepared,
    _prepare_sample,
    _train_models_for_rate_seed,
)
from wallpath_pi.utils.config import load_config

DIRECT_EXTRA = "direct_extra_all_features"


def test_make_regressor_direct_extra_is_extra_trees():
    model = make_regressor(DIRECT_EXTRA, seed=0)
    assert isinstance(model, ExtraTreesRegressor)
    assert not isinstance(model, RandomForestRegressor)


def test_direct_extra_is_known_method():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.verify_repo_ready import KNOWN_METHODS

    assert DIRECT_EXTRA in KNOWN_METHODS
    assert DIRECT_EXTRA in ALL_FEATURE_DIRECT_EXTRA_METHODS
    assert DIRECT_EXTRA in DIRECT_METHODS


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


def test_direct_extra_uses_full_features_and_predicts_finite(tmp_path):
    cfg, train_samples = _tiny_dataset(tmp_path, ["direct_rf_all_features", DIRECT_EXTRA])
    models, fnbm = _train_models_for_rate_seed(train_samples, cfg, 0.05, 7)

    assert DIRECT_EXTRA in models
    assert isinstance(models[DIRECT_EXTRA], ExtraTreesRegressor)
    # Same feature access as the all-feature direct RF reference (fair contrast).
    assert fnbm[DIRECT_EXTRA] == fnbm["direct_rf_all_features"]
    assert models[DIRECT_EXTRA].n_features_in_ == len(fnbm[DIRECT_EXTRA])

    sample = train_samples[0]
    prep = _prepare_sample(sample, cfg, sparse_rate=0.05, sparse_seed=7)
    preds = _predictions_for_prepared(prep, models, [DIRECT_EXTRA], feature_names_by_method=fnbm)
    arr = preds[DIRECT_EXTRA]
    assert arr.shape == sample.path_loss.shape
    assert np.isfinite(arr[prep.feature_table.valid_mask]).all()


def test_direct_extra_does_not_change_existing_method_outputs(tmp_path):
    existing = ["direct_rf_all_features", "wallpath_rf", "wallpath_extra"]

    cfg_base, train_base = _tiny_dataset(tmp_path / "base", existing)
    models_base, fnbm_base = _train_models_for_rate_seed(train_base, cfg_base, 0.05, 7)

    cfg_new, train_new = _tiny_dataset(tmp_path / "new", existing + [DIRECT_EXTRA])
    models_new, fnbm_new = _train_models_for_rate_seed(train_new, cfg_new, 0.05, 7)

    sample_base = train_base[0]
    sample_new = train_new[0]
    prep_base = _prepare_sample(sample_base, cfg_base, sparse_rate=0.05, sparse_seed=7)
    prep_new = _prepare_sample(sample_new, cfg_new, sparse_rate=0.05, sparse_seed=7)

    preds_base = _predictions_for_prepared(prep_base, models_base, existing, feature_names_by_method=fnbm_base)
    preds_new = _predictions_for_prepared(prep_new, models_new, existing, feature_names_by_method=fnbm_new)

    for method in existing:
        np.testing.assert_array_equal(preds_base[method], preds_new[method])
