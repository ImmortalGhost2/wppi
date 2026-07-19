"""Legacy sparse-anchor helpers (NON-CANONICAL).

Retained only for backward compatibility; the training pipeline does **not**
import this module. Canonical sparse-anchor handling is split between
``wallpath_pi.data.sparse`` (``make_sparse_mask`` for deterministic mask
generation) and ``wallpath_pi.geometry.features`` (anchor/IDW features built
inside ``build_feature_table``). Prefer those for new code.

``idw_predict`` here delegates to ``baselines.idw.idw_predict_points``.

Deprecated: do not add new call sites; import from ``data.sparse`` /
``geometry.features``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wallpath_pi.baselines.idw import idw_predict_points
from wallpath_pi.utils.hashing import stable_int_hash


def choose_sparse_anchors(df: pd.DataFrame, rate: float, seed: int, sample_id: str, min_points: int = 1) -> np.ndarray:
    n = len(df)
    if n <= 0:
        return np.asarray([], dtype=int)
    k = max(int(min_points), int(np.ceil(float(rate) * n)))
    k = min(k, n)
    rng = np.random.default_rng(stable_int_hash('choose_sparse_anchors', sample_id, rate, seed))
    return np.sort(rng.choice(np.arange(n), size=k, replace=False)).astype(int)


def idw_predict(query_xy: np.ndarray, anchor_xy: np.ndarray, values: np.ndarray, k: int | None = None, power: float = 2.0, eps: float = 1.0e-6):
    # k is accepted for API compatibility; full-anchor IDW is used by default.
    return idw_predict_points(anchor_xy, values, query_xy, power=power, eps=eps)


def add_sparse_anchor_features(
    df: pd.DataFrame,
    anchor_indices: np.ndarray,
    target_col: str,
    baseline_col: str,
    cell_size_m: float = 1.0,
    power: float = 2.0,
    eps: float = 1.0e-6,
) -> pd.DataFrame:
    """Append anchor-derived IDW features for a non-empty sparse anchor set.

    At least one sparse anchor is required. Calling this with an empty
    ``anchor_indices`` raises :class:`ValueError` rather than falling back to a
    dense-target statistic: no scientifically meaningful anchor-derived feature
    can be built without an anchor, and reading the dense (non-anchor) target
    column to synthesize a fill value would leak evaluation labels.
    """
    out = df.copy()
    anchor_indices = np.asarray(anchor_indices, dtype=int)
    out['is_anchor'] = 0
    if len(anchor_indices) == 0:
        raise ValueError(
            "add_sparse_anchor_features requires at least one sparse anchor; "
            "anchor-derived IDW features cannot be constructed from an empty "
            "anchor set. Deriving a fallback value from the dense (non-anchor) "
            f"target column {target_col!r} is forbidden because it would leak "
            "evaluation labels into the feature table."
        )
    out.loc[out.index[anchor_indices], 'is_anchor'] = 1
    q = out[['x', 'y']].to_numpy(dtype=np.float32) * float(cell_size_m)
    a = out.iloc[anchor_indices][['x', 'y']].to_numpy(dtype=np.float32) * float(cell_size_m)
    y = out.iloc[anchor_indices][target_col].to_numpy(dtype=np.float32)
    b = out.iloc[anchor_indices][baseline_col].to_numpy(dtype=np.float32)
    resid = y - b
    out['anchor_idw_path_loss_db'] = idw_predict_points(a, y, q, power=power, eps=eps)
    out['anchor_idw_residual_db'] = idw_predict_points(a, resid, q, power=power, eps=eps)
    return out
