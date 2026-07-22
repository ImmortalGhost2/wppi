from __future__ import annotations

import numpy as np
import pytest

from wallpath_pi.baselines.idw import idw_map_from_mask, idw_predict_points
from wallpath_pi.baselines.propagation import fspl_db
from wallpath_pi.data.sparse import make_sparse_mask
from wallpath_pi.evaluation.metrics import compute_metrics
from wallpath_pi.geometry.raster import bresenham_line, compute_base_maps


def test_bresenham_diagonal_endpoints():
    yy, xx = bresenham_line(0, 0, 5, 5)
    assert (yy[0], xx[0]) == (0, 0)
    assert (yy[-1], xx[-1]) == (5, 5)
    assert len(yy) == 6


def test_fspl_known_formula_shape():
    d = np.asarray([[1.0, 10.0]], dtype=np.float32)
    out = fspl_db(d, 1.0e9)
    assert out.shape == d.shape
    assert np.isfinite(out).all()
    assert out[0, 1] > out[0, 0]


def test_sparse_mask_deterministic_and_valid():
    valid = np.ones((10, 10), dtype=bool)
    valid[0, 0] = False
    a = make_sparse_mask(valid, 0.1, seed=7, sample_id="s", min_points=5)
    b = make_sparse_mask(valid, 0.1, seed=7, sample_id="s", min_points=5)
    assert np.array_equal(a, b)
    assert not (a & ~valid).any()
    assert a.sum() >= 5


def test_idw_exact_anchor_value():
    anchors = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    values = np.array([10.0, 20.0], dtype=np.float32)
    query = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pred = idw_predict_points(anchors, values, query)
    assert abs(pred[0] - 10.0) < 1e-6
    assert 10.0 < pred[1] < 20.0


def test_idw_map_empty_anchors_does_not_leak_query_target():
    # Dense target with a strongly non-zero mean; if the default leaked the
    # query-region mean, the output would reflect ~50, not the safe constant.
    value_map = np.full((6, 6), 50.0, dtype=np.float32)
    anchor_mask = np.zeros((6, 6), dtype=bool)  # no anchors
    query_mask = np.ones((6, 6), dtype=bool)
    out = idw_map_from_mask(value_map, anchor_mask, query_mask)
    assert np.all(out == 0.0)
    assert out[query_mask].mean() != 50.0


def test_idw_map_empty_anchors_raises_when_requested():
    value_map = np.full((4, 4), 50.0, dtype=np.float32)
    anchor_mask = np.zeros((4, 4), dtype=bool)
    query_mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        idw_map_from_mask(value_map, anchor_mask, query_mask, on_empty="raise")


def test_residual_idw_empty_anchors_returns_zeros():
    residual_map = np.full((5, 5), 7.0, dtype=np.float32)  # nonzero dense residual
    anchor_mask = np.zeros((5, 5), dtype=bool)
    query_mask = np.ones((5, 5), dtype=bool)
    out = idw_map_from_mask(residual_map, anchor_mask, query_mask, default_value=0.0)
    assert np.all(out == 0.0)


def test_idw_map_with_anchors_unchanged():
    # A single non-zero anchor: every query cell interpolates to that value and
    # does not depend on the dense query-region mean.
    value_map = np.zeros((5, 5), dtype=np.float32)
    value_map[2, 2] = 12.0
    anchor_mask = np.zeros((5, 5), dtype=bool)
    anchor_mask[2, 2] = True
    query_mask = np.ones((5, 5), dtype=bool)
    out = idw_map_from_mask(value_map, anchor_mask, query_mask)
    assert abs(out[2, 2] - 12.0) < 1e-5
    assert np.allclose(out[query_mask], 12.0)


def test_idw_map_two_anchors_exact_and_interpolated():
    value_map = np.zeros((1, 3), dtype=np.float32)
    value_map[0, 0] = 10.0
    value_map[0, 2] = 20.0
    anchor_mask = np.zeros((1, 3), dtype=bool)
    anchor_mask[0, 0] = True
    anchor_mask[0, 2] = True
    query_mask = np.ones((1, 3), dtype=bool)
    out = idw_map_from_mask(value_map, anchor_mask, query_mask)
    assert abs(out[0, 0] - 10.0) < 1e-5
    assert abs(out[0, 2] - 20.0) < 1e-5
    assert 10.0 < out[0, 1] < 20.0


def test_compute_base_maps_counts_wall():
    material = np.zeros((8, 8), dtype=np.int32)
    material[:, 4] = 1
    wall = material > 0
    valid = ~wall
    base = compute_base_maps(wall, material, tx_xy=[1, 1], frequency_hz=3.5e9, resolution_m=0.25, valid_mask=valid, material_ids=[1])
    assert base.material_counts.shape == (1, 8, 8)
    assert base.wall_count[1, 6] > 0
    assert base.los[1, 2] == 1


def test_metrics_zero_error():
    y = np.ones((4, 4), dtype=np.float32) * 7
    m = np.ones_like(y, dtype=bool)
    out = compute_metrics(y, y, m)
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert out["count"] == 16
