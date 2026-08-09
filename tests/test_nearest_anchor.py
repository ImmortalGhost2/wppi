from __future__ import annotations

import numpy as np

from wallpath_pi.baselines.idw import idw_predict_points
from wallpath_pi.geometry.features import (
    _knn_nearest,
    _knn_nearest_chunked,
    _nearest_anchor_features,
)


def _brute_force_knn(coords, anchor_xy, k):
    """Reference k-NN via an explicit (small) pairwise distance matrix."""
    diff = coords[:, None, :] - anchor_xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    order = np.argsort(dist, axis=1)[:, :k]
    return np.take_along_axis(dist, order, axis=1), order


def test_nearest_anchor_features_shape_and_values_1000x500():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 300, size=(1000, 2)).astype(np.float32)
    anchor_xy = rng.uniform(0, 300, size=(500, 2)).astype(np.float32)
    residuals = rng.normal(size=500).astype(np.float32)
    k = 4

    arr, names = _nearest_anchor_features(coords, anchor_xy, residuals, k=k, eps=1e-6)

    assert arr.shape == (1000, 2 * k)
    assert arr.dtype == np.float32
    assert names == [f"anchor_residual_nn_{i+1}" for i in range(k)] + [
        f"anchor_dist_px_nn_{i+1}" for i in range(k)
    ]

    # Distances must match a brute-force reference (nearest-anchor distances are
    # the second half of the feature block).
    ref_dist, ref_idx = _brute_force_knn(coords, anchor_xy, k)
    np.testing.assert_allclose(arr[:, k:], ref_dist, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(arr[:, :k], residuals[ref_idx], rtol=1e-4, atol=1e-4)


def test_nearest_anchor_features_zero_anchors():
    coords = np.random.default_rng(1).uniform(0, 10, size=(50, 2)).astype(np.float32)
    anchor_xy = np.zeros((0, 2), dtype=np.float32)
    residuals = np.zeros((0,), dtype=np.float32)
    k = 3

    arr, names = _nearest_anchor_features(coords, anchor_xy, residuals, k=k, eps=1e-6)

    assert arr.shape == (50, 2 * k)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)
    assert len(names) == 2 * k


def test_nearest_anchor_features_fewer_than_k_anchors():
    coords = np.random.default_rng(2).uniform(0, 10, size=(20, 2)).astype(np.float32)
    anchor_xy = np.array([[1.0, 1.0], [8.0, 8.0]], dtype=np.float32)
    residuals = np.array([5.0, -7.0], dtype=np.float32)
    k = 4  # more neighbors than anchors

    arr, names = _nearest_anchor_features(coords, anchor_xy, residuals, k=k, eps=1e-6)

    assert arr.shape == (20, 2 * k)
    assert arr.dtype == np.float32
    # Residual columns beyond the 2 real anchors are padded with 0.0.
    assert np.all(arr[:, 2:k] == 0.0)
    # Distance columns beyond the 2 real anchors are a large finite value.
    pad_dist = arr[:, k + 2:]
    assert np.all(np.isfinite(pad_dist))
    assert np.all(pad_dist > arr[:, k:k + 2].max())


def test_knn_chunked_matches_sklearn_path():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 50, size=(300, 2)).astype(np.float32)
    anchor_xy = rng.uniform(0, 50, size=(40, 2)).astype(np.float32)
    k = 5

    d_sk, i_sk = _knn_nearest(coords, anchor_xy, k)
    d_ch, i_ch = _knn_nearest_chunked(coords, anchor_xy, k, chunk_size=32)

    np.testing.assert_allclose(d_sk, d_ch, rtol=1e-4, atol=1e-3)
    # Indices agree wherever distances are not (near-)tied.
    ref_d, _ = _brute_force_knn(coords, anchor_xy, k)
    np.testing.assert_allclose(d_sk, ref_d, rtol=1e-4, atol=1e-3)


def test_idw_chunked_matches_unchunked_toy():
    rng = np.random.default_rng(4)
    anchor_xy = rng.uniform(0, 20, size=(15, 2)).astype(np.float32)
    values = rng.normal(size=15).astype(np.float32)
    query_xy = rng.uniform(0, 20, size=(97, 2)).astype(np.float32)

    full = idw_predict_points(anchor_xy, values, query_xy, chunk_size=10_000)
    chunked = idw_predict_points(anchor_xy, values, query_xy, chunk_size=8)

    assert full.dtype == np.float32
    np.testing.assert_allclose(full, chunked, rtol=1e-5, atol=1e-5)
