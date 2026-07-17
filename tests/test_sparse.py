from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wallpath_pi.baselines.idw import idw_predict_points
from wallpath_pi.data.sparse import check_sparse_mask_validity, make_sparse_mask
from wallpath_pi.features.sparse import add_sparse_anchor_features


def test_sparse_anchors_are_deterministic_and_valid():
    valid = np.ones((10, 10), dtype=bool)
    valid[0, 0] = False
    a = make_sparse_mask(valid, 0.1, seed=42, sample_id="s", min_points=5)
    b = make_sparse_mask(valid, 0.1, seed=42, sample_id="s", min_points=5)
    assert np.array_equal(a, b)
    assert a.sum() >= 5
    check_sparse_mask_validity(a, valid)


def test_idw_exact_anchor_value():
    q = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    a = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    values = np.array([10.0, 20.0], dtype=np.float32)
    pred = idw_predict_points(a, values, q)
    assert pred[0] == 10.0
    assert 10.0 < pred[1] < 20.0


def _anchor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 2.0, 4.0],
            "y": [0.0, 0.0, 0.0],
            "path_loss_db": [10.0, 20.0, 30.0],
            "multi_wall_pred_db": [1.0, 2.0, 3.0],
        }
    )


def test_add_sparse_anchor_features_valid_behavior_unchanged():
    """A non-empty anchor set produces exact anchor-derived IDW features."""
    df = _anchor_frame()
    out = add_sparse_anchor_features(df, [0, 2], "path_loss_db", "multi_wall_pred_db")
    assert out["is_anchor"].tolist() == [1, 0, 1]
    # IDW reproduces the anchor value exactly at an anchor location.
    assert out.loc[0, "anchor_idw_path_loss_db"] == pytest.approx(10.0)
    assert out.loc[2, "anchor_idw_path_loss_db"] == pytest.approx(30.0)
    # Residual anchors are target-minus-baseline at the anchor rows only.
    assert out.loc[0, "anchor_idw_residual_db"] == pytest.approx(9.0)
    assert out.loc[2, "anchor_idw_residual_db"] == pytest.approx(27.0)


def test_add_sparse_anchor_features_empty_raises_value_error():
    """No anchors must raise, never fall back to a dense-target statistic."""
    df = _anchor_frame()
    with pytest.raises(ValueError, match="at least one sparse anchor"):
        add_sparse_anchor_features(df, [], "path_loss_db", "multi_wall_pred_db")


def test_add_sparse_anchor_features_empty_does_not_read_dense_target():
    """The dense target column is never consulted to build an empty-anchor fill.

    Poisoning every target value with NaN would corrupt any dense-target mean
    fallback; the function must still raise cleanly instead of computing one.
    """
    df = _anchor_frame()
    df["path_loss_db"] = np.nan
    with pytest.raises(ValueError, match="forbidden"):
        add_sparse_anchor_features(df, [], "path_loss_db", "multi_wall_pred_db")


def test_add_sparse_anchor_features_uses_only_anchor_targets():
    """Non-anchor (dense) target values must not influence anchor features."""
    df = _anchor_frame()
    df.loc[1, "path_loss_db"] = np.nan  # non-anchor row poisoned
    out = add_sparse_anchor_features(df, [0, 2], "path_loss_db", "multi_wall_pred_db")
    assert np.isfinite(out["anchor_idw_path_loss_db"]).all()
    assert np.isfinite(out["anchor_idw_residual_db"]).all()


def test_canonical_pipeline_does_not_use_legacy_sparse_helper():
    """The canonical training pipeline never imports the legacy sparse helper."""
    import wallpath_pi.training.pipeline as pipeline

    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "features.sparse" not in source
    assert "features import sparse" not in source
    assert "add_sparse_anchor_features" not in source
