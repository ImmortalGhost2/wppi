import numpy as np
import pytest

from wallpath_pi.evaluation.metrics import compute_metrics


def test_metrics_known_small_array():
    target = np.zeros((2, 2), dtype=np.float32)
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    out = compute_metrics(pred, target, valid)

    assert out["count"] == 4
    assert out["valid_count"] == 4
    assert out["mae"] == pytest.approx(2.5)
    assert out["bias_db"] == pytest.approx(2.5)  # mean(pred - target)
    assert out["rmse"] == pytest.approx(np.sqrt(7.5))
    assert out["median_ae"] == pytest.approx(2.5)
    assert out["std_error_db"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert out["p90_ae"] == pytest.approx(np.percentile([1, 2, 3, 4], 90))
    assert out["p95_ae"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))


def test_metrics_zero_error_has_zero_bias_and_std():
    y = np.ones((4, 4), dtype=np.float32) * 7.0
    valid = np.ones_like(y, dtype=bool)
    out = compute_metrics(y, y, valid)
    assert out["rmse"] == 0.0
    assert out["bias_db"] == 0.0
    assert out["std_error_db"] == 0.0
    assert out["p95_ae"] == 0.0


def test_metrics_ignore_invalid_and_nonfinite():
    target = np.array([[0.0, 0.0], [0.0, np.nan]], dtype=np.float32)
    pred = np.array([[2.0, 4.0], [np.inf, 100.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, True]], dtype=bool)
    # (1,1) is invalid via NaN target; (1,0) invalid via inf pred.
    out = compute_metrics(pred, target, valid)
    assert out["count"] == 2  # only the two finite, valid pixels
    assert out["mae"] == pytest.approx(3.0)  # mean(|2|, |4|)
    assert out["bias_db"] == pytest.approx(3.0)
    assert np.isfinite(out["rmse"])


def test_metrics_single_valid_pixel_std_is_nan():
    target = np.zeros((1, 3), dtype=np.float32)
    pred = np.array([[5.0, np.nan, np.nan]], dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    out = compute_metrics(pred, target, valid)
    assert out["count"] == 1
    assert np.isnan(out["std_error_db"])  # ddof=1 undefined for n=1
    assert out["bias_db"] == pytest.approx(5.0)


def test_los_nlos_too_few_pixels_returns_nan_not_crash():
    target = np.zeros((1, 10), dtype=np.float32)
    pred = np.ones((1, 10), dtype=np.float32)
    valid = np.ones((1, 10), dtype=bool)
    los = np.zeros((1, 10), dtype=np.float32)
    los[0, :2] = 1.0  # only 2 LOS pixels, below min_region_pixels=5
    out = compute_metrics(pred, target, valid, los_mask=los, min_region_pixels=5)
    assert np.isnan(out["los_rmse"])  # too few LOS pixels
    assert np.isfinite(out["nlos_rmse"])  # 8 NLOS pixels is enough


def test_clipped_pixel_fraction_high_and_low():
    target = np.array([[50.0, 160.0, 80.0, 13.0]], dtype=np.float32)
    pred = target.copy()
    valid = np.ones((1, 4), dtype=bool)
    out = compute_metrics(
        pred, target, valid,
        clip_max_db=160.0, clip_min_db=13.0, treat_clip_max_as_clipped=True,
    )
    # 160 (>= max) and 13 (<= min) are clipped => 2 of 4.
    assert out["clipped_pixel_fraction"] == pytest.approx(0.5)


def test_anchor_count_passthrough():
    y = np.zeros((2, 2), dtype=np.float32)
    valid = np.ones_like(y, dtype=bool)
    out = compute_metrics(y, y, valid, anchor_count=7)
    assert out["anchor_count"] == 7


def test_unclipped_rmse_excludes_clipped_targets():
    target = np.array([[10.0, 160.0, 20.0, 30.0]], dtype=np.float32)
    pred = np.array([[12.0, 200.0, 22.0, 33.0]], dtype=np.float32)
    valid = np.ones((1, 4), dtype=bool)
    out = compute_metrics(pred, target, valid, clip_max_db=160.0, treat_clip_max_as_clipped=True)
    # Unclipped pixels: errors 2, 2, 3 -> rmse over those three.
    expected = np.sqrt(np.mean(np.array([2.0, 2.0, 3.0]) ** 2))
    assert out["unclipped_rmse"] == pytest.approx(expected)
    assert out["unclipped_count"] == 3
