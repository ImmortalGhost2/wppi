"""Canonical deterministic sparse-anchor masks.

``make_sparse_mask`` produces the reproducible anchor mask keyed by
(sample_id, rate, seed) that every method shares. This is the recommended API;
``wallpath_pi.features.sparse`` is a non-canonical compatibility module.
"""
from __future__ import annotations

import numpy as np

from wallpath_pi.utils.hashing import stable_int_hash


def make_sparse_mask(
    valid_mask: np.ndarray,
    sampling_rate: float,
    seed: int,
    sample_id: str,
    min_points: int = 1,
) -> np.ndarray:
    """Select a deterministic sparse-anchor mask over the valid receiver cells.

    The chosen anchors are keyed by ``(sample_id, sampling_rate, seed)`` so every
    method observes the same anchors for a given sample. At least ``min_points``
    cells are kept; ``sampling_rate <= 0`` selects exactly ``min_points`` and
    ``>= 1`` selects all valid cells.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    yy, xx = np.nonzero(valid)
    n_valid = len(yy)
    if n_valid == 0:
        raise ValueError("Cannot sample sparse anchors from an empty valid mask.")
    rate = float(sampling_rate)
    if rate <= 0:
        count = int(min_points)
    elif rate >= 1:
        count = n_valid
    else:
        count = int(np.ceil(n_valid * rate))
    count = max(int(min_points), count)
    count = min(n_valid, count)
    rng_seed = stable_int_hash("sparse_mask", sample_id, sampling_rate, seed)
    rng = np.random.default_rng(rng_seed)
    chosen = rng.choice(np.arange(n_valid), size=count, replace=False)
    mask = np.zeros_like(valid, dtype=bool)
    mask[yy[chosen], xx[chosen]] = True
    return mask


def check_sparse_mask_validity(sparse_mask: np.ndarray, valid_mask: np.ndarray) -> None:
    """Assert that every selected anchor falls on a valid receiver cell.

    Raises ``ValueError`` on a shape mismatch and ``AssertionError`` if any
    anchor lies outside ``valid_mask``.
    """
    sparse = np.asarray(sparse_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if sparse.shape != valid.shape:
        raise ValueError("Sparse mask and valid mask shape mismatch.")
    bad = sparse & ~valid
    if bad.any():
        raise AssertionError("Sparse mask selects invalid receiver cells.")
