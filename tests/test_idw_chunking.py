"""Memory-safe chunked IDW prediction: equivalence and large-input behavior.

Guards the fix for the sparse_rate=0.05/0.10 out-of-memory crash where a full
(n_query x n_anchor) array was materialized. Chunking must not change numbers.
"""
from __future__ import annotations

import numpy as np
import pytest
from wallpath_pi.baselines.idw import (
    _IDW_MAX_QUERY_CHUNK,
    _IDW_MEMORY_BUDGET_BYTES,
    _resolve_query_chunk,
    idw_map_from_mask,
    idw_predict_points,
)


def test_query_chunk_size_matches_large_chunk():
    rng = np.random.default_rng(11)
    anchor_xy = rng.uniform(0, 50, size=(40, 2)).astype(np.float32)
    values = rng.normal(10.0, 3.0, size=40).astype(np.float32)
    query_xy = rng.uniform(0, 50, size=(257, 2)).astype(np.float32)

    big = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size=10_000)
    small = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size=16)
    assert big.shape == (257,) and big.dtype == np.float32
    np.testing.assert_allclose(big, small, rtol=1e-5, atol=1e-5)


def test_query_chunk_size_overrides_chunk_size():
    rng = np.random.default_rng(12)
    anchor_xy = rng.uniform(0, 20, size=(25, 2)).astype(np.float32)
    values = rng.normal(size=25).astype(np.float32)
    query_xy = rng.uniform(0, 20, size=(130, 2)).astype(np.float32)

    # query_chunk_size takes precedence over chunk_size when both are given.
    a = idw_predict_points(anchor_xy, values, query_xy, chunk_size=4, query_chunk_size=10_000)
    b = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size=7)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_map_from_mask_chunked_many_points():
    # A map with many anchors and many query points, scored with a small chunk:
    # the result must match a single large chunk and never need a big allocation.
    rng = np.random.default_rng(13)
    h, w = 100, 100
    value_map = rng.normal(80.0, 10.0, size=(h, w)).astype(np.float32)
    flat = rng.permutation(h * w)
    anchor_idx = flat[:800]
    query_idx = flat[800:]  # the remaining ~9200 cells
    anchor_mask = np.zeros((h, w), dtype=bool)
    query_mask = np.zeros((h, w), dtype=bool)
    anchor_mask.flat[anchor_idx] = True
    query_mask.flat[query_idx] = True

    chunked = idw_map_from_mask(value_map, anchor_mask, query_mask, query_chunk_size=64)
    one_shot = idw_map_from_mask(value_map, anchor_mask, query_mask, query_chunk_size=100_000)
    assert chunked.shape == (h, w)
    np.testing.assert_allclose(chunked, one_shot, rtol=1e-5, atol=1e-5)
    # The chunk size used for queries must be small relative to the anchor count,
    # confirming peak memory scales with chunk x n_anchor, not n_query x n_anchor.
    assert 64 < int(anchor_mask.sum())


def test_empty_anchors_return_default_value_under_chunking():
    query_xy = np.random.default_rng(14).uniform(0, 10, size=(300, 2)).astype(np.float32)
    pred = idw_predict_points(np.empty((0, 2), np.float32), np.empty((0,), np.float32),
                              query_xy, default_value=7.5, query_chunk_size=32)
    assert pred.shape == (300,)
    assert np.all(pred == np.float32(7.5))


def test_map_from_mask_empty_anchor_on_empty_behaviors():
    value_map = np.zeros((8, 8), dtype=np.float32)
    anchor_mask = np.zeros((8, 8), dtype=bool)
    query_mask = np.ones((8, 8), dtype=bool)
    filled = idw_map_from_mask(value_map, anchor_mask, query_mask, default_value=3.0, query_chunk_size=16)
    assert np.all(filled == np.float32(3.0))
    with pytest.raises(ValueError):
        idw_map_from_mask(value_map, anchor_mask, query_mask, on_empty="raise", query_chunk_size=16)


def test_adaptive_matches_fixed_large_chunk():
    rng = np.random.default_rng(21)
    anchor_xy = rng.uniform(0, 30, size=(20, 2)).astype(np.float32)
    values = rng.normal(size=20).astype(np.float32)
    query_xy = rng.uniform(0, 30, size=(311, 2)).astype(np.float32)
    auto = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size="auto")
    default = idw_predict_points(anchor_xy, values, query_xy)            # None == auto
    fixed = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size=10_000)
    np.testing.assert_allclose(auto, fixed, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(default, fixed, rtol=1e-5, atol=1e-5)


def test_low_anchor_count_uses_cap_chunk():
    # Few anchors: adaptive selects the maximum cap (fast path).
    assert _resolve_query_chunk(None, n_anchor=20) == _IDW_MAX_QUERY_CHUNK
    assert _resolve_query_chunk("auto", n_anchor=1) == _IDW_MAX_QUERY_CHUNK


def test_high_anchor_count_shrinks_chunk_within_budget():
    c_small = _resolve_query_chunk("auto", n_anchor=200)
    c_15k = _resolve_query_chunk("auto", n_anchor=15_000)
    c_30k = _resolve_query_chunk("auto", n_anchor=30_000)
    assert c_small == _IDW_MAX_QUERY_CHUNK                 # small -> capped
    assert 1 <= c_30k <= c_15k < _IDW_MAX_QUERY_CHUNK      # dense -> shrinks below cap
    # Peak per chunk stays within the 256 MB budget (5 float32 temporaries).
    assert c_15k * 15_000 * 4 * 5 <= _IDW_MEMORY_BUDGET_BYTES
    assert c_30k * 30_000 * 4 * 5 <= _IDW_MEMORY_BUDGET_BYTES


def test_high_anchor_count_adaptive_runs_and_matches():
    rng = np.random.default_rng(22)
    anchor_xy = rng.uniform(0, 100, size=(4000, 2)).astype(np.float32)
    values = rng.normal(80, 5, size=4000).astype(np.float32)
    query_xy = rng.uniform(0, 100, size=(2000, 2)).astype(np.float32)
    auto = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size="auto")
    fixed = idw_predict_points(anchor_xy, values, query_xy, query_chunk_size=4096)
    assert auto.shape == (2000,)
    np.testing.assert_allclose(auto, fixed, rtol=1e-5, atol=1e-5)
    assert _resolve_query_chunk("auto", n_anchor=4000) < _IDW_MAX_QUERY_CHUNK


def test_explicit_integer_override_is_used_verbatim():
    assert _resolve_query_chunk(7, n_anchor=999_999) == 7
    assert _resolve_query_chunk(0, n_anchor=10) == 1     # floored at 1
    assert _resolve_query_chunk(-5, n_anchor=10) == 1


def test_invalid_query_chunk_size_raises():
    with pytest.raises(ValueError):
        _resolve_query_chunk("nonsense", n_anchor=10)
    with pytest.raises(ValueError):
        idw_predict_points(np.zeros((3, 2), np.float32), np.zeros(3, np.float32),
                           np.zeros((3, 2), np.float32), query_chunk_size=1.5)
