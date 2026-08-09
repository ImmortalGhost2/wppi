"""Legacy standalone path-loss helpers (NON-CANONICAL).

Retained only for backward compatibility; the training pipeline does **not**
import this module. The canonical propagation baselines live in
``wallpath_pi.baselines.propagation`` (FSPL plus the fitted log-distance and
multi-wall baselines). Prefer that module for all new code.

``fspl_db`` here simply delegates to ``baselines.propagation.fspl_db``. The
``LogDistanceModel`` API is a self-contained variant kept for older notebooks;
new code should use ``baselines.propagation.fit_log_distance`` instead.

Deprecated: do not add new call sites; import from ``baselines.propagation``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from wallpath_pi.baselines.propagation import fspl_db as _fspl_db


def fspl_db(distance_m, frequency_hz: float, min_distance_m: float = 0.25):
    """Free-space path loss in dB for distance in meters and frequency in Hz."""
    return _fspl_db(distance_m, frequency_hz, min_distance_m=min_distance_m)


@dataclass
class LogDistanceModel:
    intercept_db: float
    slope_db_per_decade: float
    reference_distance_m: float


def fit_log_distance(distance_m, target_db, reference_distance_m: float = 1.0, ridge_alpha: float = 1.0e-6) -> LogDistanceModel:
    """Fit y = intercept + slope * log10(d / d0)."""
    d = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    y = np.asarray(target_db, dtype=np.float64).reshape(-1)
    mask = np.isfinite(d) & np.isfinite(y)
    if mask.sum() < 2:
        return LogDistanceModel(float(np.nanmean(y)), 0.0, float(reference_distance_m))
    x = np.log10(np.maximum(d[mask], 1.0e-12) / float(reference_distance_m))
    X = np.column_stack([np.ones_like(x), x])
    reg = np.diag([0.0, float(ridge_alpha)])
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y[mask])
    return LogDistanceModel(float(beta[0]), float(beta[1]), float(reference_distance_m))


def predict_log_distance(model: LogDistanceModel, distance_m, reference_distance_m: float | None = None):
    d0 = float(reference_distance_m if reference_distance_m is not None else model.reference_distance_m)
    d = np.asarray(distance_m, dtype=np.float64)
    x = np.log10(np.maximum(d, 1.0e-12) / d0)
    return (float(model.intercept_db) + float(model.slope_db_per_decade) * x).astype(np.float64)
