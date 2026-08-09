from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from wallpath_pi.geometry.features import build_feature_table
from wallpath_pi.geometry.raster import compute_base_maps
from wallpath_pi.training.pipeline import _chunked_predict, _predictions_for_prepared


def _toy_feature_inputs():
    """Tiny scene with walls, anchors and baseline maps for feature tests."""
    H = W = 8
    wall = np.zeros((H, W), dtype=bool)
    wall[3, 2:6] = True
    material = np.zeros((H, W), dtype=np.int32)
    material[3, 2:6] = 1
    valid = ~wall
    base = compute_base_maps(
        wall, material, tx_xy=(1, 1), frequency_hz=3.5e9, resolution_m=0.5,
        valid_mask=valid, material_ids=[1, 2, 3],
    )
    path_loss = (40.0 + 0.5 * np.arange(H * W).reshape(H, W)).astype(np.float32)
    sample = SimpleNamespace(path_loss=path_loss)
    mw_pred = (35.0 + 0.3 * np.arange(H * W).reshape(H, W)).astype(np.float32)

    sparse = np.zeros((H, W), dtype=bool)
    ys, xs = np.nonzero(valid)
    sel = np.linspace(0, len(ys) - 1, 6).astype(int)
    sparse[ys[sel], xs[sel]] = True

    baseline_pred_maps = {
        "fspl_pred_db": base.fspl,
        "log_distance_pred_db": (30.0 + 0.2 * np.arange(H * W).reshape(H, W)).astype(np.float32),
        "multi_wall_pred_db": mw_pred,
        "multi_wall_residual_idw_pred_db": (mw_pred + 1.0).astype(np.float32),
    }
    feature_cfg = {
        "coordinate_features": True,
        "material_features": True,
        "sparse_anchor_features": True,
        "baseline_prediction_features": True,
        "k_nearest_anchors": 3,
    }
    return sample, base, sparse, mw_pred, feature_cfg, baseline_pred_maps


def _build(sample, base, sparse, mw_pred, feature_cfg, baseline_pred_maps, query_indices=None):
    return build_feature_table(
        sample, base, sparse_mask=sparse, multiwall_pred=mw_pred,
        material_ids=[1, 2, 3], feature_cfg=feature_cfg,
        baseline_pred_maps=baseline_pred_maps, query_indices=query_indices,
    )


def test_subset_feature_table_same_columns_as_full():
    args = _toy_feature_inputs()
    full = _build(*args)
    qi = np.array([0, 3, 7, 11, 25], dtype=np.intp)
    sub = _build(*args, query_indices=qi)
    assert sub.feature_names == full.feature_names
    assert sub.X.shape[1] == full.X.shape[1]
    assert sub.X.shape[0] == qi.size


def test_subset_feature_rows_match_full_rows_on_same_pixels():
    args = _toy_feature_inputs()
    full = _build(*args)
    qi = np.array([0, 3, 7, 11, 25, 40], dtype=np.intp)
    sub = _build(*args, query_indices=qi)
    np.testing.assert_allclose(sub.X, full.X[qi], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sub.y, full.y[qi])
    np.testing.assert_allclose(sub.baseline_values, full.baseline_values[qi])
    np.testing.assert_array_equal(sub.coords_xy, full.coords_xy[qi])


def test_chunked_prediction_equals_nonchunked():
    rng = np.random.default_rng(1)
    H = W = 5
    n = H * W
    valid = np.ones((H, W), dtype=bool)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    model = RandomForestRegressor(n_estimators=8, random_state=0).fit(X, y)

    feature_table = SimpleNamespace(valid_mask=valid, X=X, baseline_values=np.zeros(n, dtype=np.float32))
    zeros = np.zeros((H, W), dtype=np.float32)
    prep = SimpleNamespace(
        sample=SimpleNamespace(path_loss=zeros.copy()),
        feature_table=feature_table,
        fspl_pred=zeros.copy(),
        log_distance_pred=zeros.copy(),
        multi_wall_pred=zeros.copy(),
        idw_pred=zeros.copy(),
        multi_wall_residual_idw_pred=zeros.copy(),
    )

    full = _predictions_for_prepared(prep, {"direct_rf": model}, ["direct_rf"], eval_chunk_size=10_000)
    chunked = _predictions_for_prepared(prep, {"direct_rf": model}, ["direct_rf"], eval_chunk_size=3)
    np.testing.assert_array_equal(full["direct_rf"], chunked["direct_rf"])

    np.testing.assert_array_equal(_chunked_predict(model, X, 2), model.predict(X).astype(np.float32))
