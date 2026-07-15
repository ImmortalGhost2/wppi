from __future__ import annotations

import numpy as np

from wallpath_pi.baselines.propagation import fit_log_distance, fspl_db


def test_fspl_one_meter_one_ghz():
    value = float(fspl_db(np.asarray([1.0], dtype=np.float32), 1.0e9, min_distance_m=1.0)[0])
    assert 32.0 < value < 33.0


def test_log_distance_fit_recovers_slope():
    d = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]], dtype=np.float32)
    y = 40.0 + 2.0 * 10.0 * np.log10(d)
    mask = np.ones_like(d, dtype=bool)
    pred, info = fit_log_distance(d, y.astype(np.float32), mask, d0_m=1.0, min_distance_m=1.0, alpha=0.0)
    assert info.anchor_count == d.size
    assert np.max(np.abs(pred - y)) < 1e-4


def test_log_distance_fallback_ignores_dense_nonanchor_values():
    # Fewer than two anchors forces the degenerate fallback path. The dense,
    # non-anchor labels are set to an extreme value; a leakage-safe fallback must
    # ignore them entirely and fill only from sparse-anchor labels.
    d = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]], dtype=np.float32)
    anchor_value = 50.0
    target = np.full_like(d, 999.0)  # dense non-anchor labels that must be ignored
    target[0, 2] = anchor_value
    mask = np.zeros_like(d, dtype=bool)
    mask[0, 2] = True

    pred, info = fit_log_distance(d, target, mask, d0_m=1.0, min_distance_m=1.0)

    assert info.name == "log_distance_fallback_anchor_mean"
    assert info.anchor_count == 1
    # The fill is the anchor label, never the dense mean (which would be ~620).
    assert np.allclose(pred, anchor_value)

    # Perturbing only the dense non-anchor labels must not change the prediction.
    target_perturbed = target.copy()
    target_perturbed[~mask] = -1234.0
    pred_perturbed, _ = fit_log_distance(d, target_perturbed, mask, d0_m=1.0, min_distance_m=1.0)
    assert np.allclose(pred_perturbed, anchor_value)


def test_log_distance_fallback_zero_anchors_uses_zero():
    # With no usable anchors the fallback fills 0.0 rather than any dense value.
    d = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    target = np.full_like(d, 123.0)
    mask = np.zeros_like(d, dtype=bool)
    pred, info = fit_log_distance(d, target, mask, d0_m=1.0, min_distance_m=1.0)
    assert info.anchor_count == 0
    assert np.allclose(pred, 0.0)
