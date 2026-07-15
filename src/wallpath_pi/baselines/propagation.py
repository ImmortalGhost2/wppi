"""Canonical propagation baselines: FSPL, fitted log-distance, and multi-wall.

Recommended API for path-loss baselines; this is what the training pipeline
uses. The older ``wallpath_pi.physics.pathloss`` module is a non-canonical
compatibility shim that delegates here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class LinearBaselineResult:
    name: str
    coefficients: np.ndarray
    intercept: float
    feature_names: list[str]
    design_rank: int
    anchor_count: int


def fspl_db(distance_m: np.ndarray, frequency_hz: float, min_distance_m: float = 0.25) -> np.ndarray:
    """Free-space path loss in dB with d in meters and f in Hz."""
    d = np.maximum(np.asarray(distance_m, dtype=np.float64), float(min_distance_m))
    f = max(float(frequency_hz), 1.0)
    return (20.0 * np.log10(d) + 20.0 * np.log10(f) - 147.55).astype(np.float32)


def log_distance_feature(distance_m: np.ndarray, d0_m: float = 1.0, min_distance_m: float = 0.25) -> np.ndarray:
    d = np.maximum(np.asarray(distance_m, dtype=np.float64), float(min_distance_m))
    return (10.0 * np.log10(d / max(float(d0_m), 1.0e-6))).astype(np.float32)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y row mismatch.")
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(X.astype(np.float64), y.astype(np.float64))
    return model


def fit_log_distance(
    distance_m: np.ndarray,
    target_db: np.ndarray,
    fit_mask: np.ndarray,
    d0_m: float = 1.0,
    min_distance_m: float = 0.25,
    alpha: float = 1.0e-6,
) -> tuple[np.ndarray, LinearBaselineResult]:
    feat = log_distance_feature(distance_m, d0_m=d0_m, min_distance_m=min_distance_m)
    mask = np.asarray(fit_mask, dtype=bool)
    if int(mask.sum()) < 2:
        # Leakage-safe fallback: never touch dense non-anchor labels. Fill from
        # the sparse-anchor labels only (or 0.0 if no usable anchors exist).
        y_anchor = np.asarray(target_db, dtype=np.float32)[mask]
        if y_anchor.size > 0 and np.isfinite(y_anchor).any():
            fill = float(np.nanmean(y_anchor))
        else:
            fill = 0.0
        pred = np.full_like(target_db, fill, dtype=np.float32)
        return pred, LinearBaselineResult(
            "log_distance_fallback_anchor_mean",
            np.array([0.0], dtype=np.float32),
            fill,
            ["log_distance"],
            0,
            int(mask.sum()),
        )
    X = feat[mask].reshape(-1, 1)
    y = np.asarray(target_db, dtype=np.float32)[mask]
    model = _fit_ridge(X, y, alpha=alpha)
    pred = model.predict(feat.reshape(-1, 1)).reshape(target_db.shape).astype(np.float32)
    result = LinearBaselineResult(
        name="log_distance",
        coefficients=np.asarray(model.coef_, dtype=np.float32),
        intercept=float(model.intercept_),
        feature_names=["log_distance"],
        design_rank=int(np.linalg.matrix_rank(X)) if X.size else 0,
        anchor_count=int(mask.sum()),
    )
    return pred, result


def fit_multi_wall(
    distance_m: np.ndarray,
    material_counts: np.ndarray,
    target_db: np.ndarray,
    fit_mask: np.ndarray,
    material_ids: Sequence[int],
    d0_m: float = 1.0,
    min_distance_m: float = 0.25,
    alpha: float = 1.0e-6,
) -> tuple[np.ndarray, LinearBaselineResult]:
    logd = log_distance_feature(distance_m, d0_m=d0_m, min_distance_m=min_distance_m)
    counts = np.asarray(material_counts, dtype=np.float32)
    if counts.ndim != 3:
        raise ValueError("material_counts must be [M, H, W].")
    H, W = target_db.shape
    flat_features = [logd.reshape(-1)]
    names = ["log_distance"]
    for idx, mid in enumerate(material_ids):
        flat_features.append(counts[idx].reshape(-1))
        names.append(f"mat_{int(mid)}_count")
    X_all = np.stack(flat_features, axis=1).astype(np.float32)
    mask_flat = np.asarray(fit_mask, dtype=bool).reshape(-1)
    y_all = np.asarray(target_db, dtype=np.float32).reshape(-1)
    if int(mask_flat.sum()) < max(2, len(names)):
        # Fall back to log-distance if too few anchors exist for a stable wall fit.
        pred, ld = fit_log_distance(distance_m, target_db, fit_mask, d0_m=d0_m, min_distance_m=min_distance_m, alpha=alpha)
        return pred, LinearBaselineResult("multi_wall_fallback_log_distance", ld.coefficients, ld.intercept, ld.feature_names, ld.design_rank, ld.anchor_count)
    X = X_all[mask_flat]
    y = y_all[mask_flat]
    model = _fit_ridge(X, y, alpha=alpha)
    pred = model.predict(X_all).reshape(H, W).astype(np.float32)
    result = LinearBaselineResult(
        name="multi_wall",
        coefficients=np.asarray(model.coef_, dtype=np.float32),
        intercept=float(model.intercept_),
        feature_names=names,
        design_rank=int(np.linalg.matrix_rank(X)) if X.size else 0,
        anchor_count=int(mask_flat.sum()),
    )
    return pred, result
