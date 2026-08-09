from __future__ import annotations

from typing import Dict

import numpy as np


def _masked_values(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    valid = mask & np.isfinite(p) & np.isfinite(t)
    return (p[valid] - t[valid]).astype(np.float32)


def rmse_from_errors(err: np.ndarray) -> float:
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))


def mae_from_errors(err: np.ndarray) -> float:
    if err.size == 0:
        return float("nan")
    return float(np.mean(np.abs(err)))


def _region_rmse(pred: np.ndarray, target: np.ndarray, region: np.ndarray, min_pixels: int) -> float:
    """RMSE over a region, returning NaN (never raising) when the region is too small."""
    region = np.asarray(region, dtype=bool)
    if int(region.sum()) < int(min_pixels):
        return float("nan")
    return rmse_from_errors(_masked_values(pred, target, region))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    los_mask: np.ndarray | None = None,
    wall_count: np.ndarray | None = None,
    high_wall_count_threshold: int = 2,
    clip_max_db: float | None = None,
    clip_min_db: float | None = None,
    treat_clip_max_as_clipped: bool = True,
    anchor_count: int | None = None,
    min_region_pixels: int = 5,
    sparse_mask: np.ndarray | None = None,
    wall_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Error metrics over the valid pixels, with optional region breakdowns.

    The default metrics (``rmse``, ``mae``, ``p90_ae`` ...) are computed over every
    valid pixel for backward compatibility. When ``sparse_mask`` (the per-sample
    sparse-anchor mask) is supplied, additional ``non_anchor_*`` metrics are
    reported over only the *unobserved* pixels (``valid & ~sparse_mask``); anchor
    pixels are observed inputs, so excluding them gives an honest error over the
    pixels the model actually had to predict. When ``wall_mask`` is supplied,
    optional ``free_space_*`` / ``wall_region_rmse`` sensitivity metrics are
    reported over non-wall / wall receiver pixels. Any region with no pixels
    yields NaN rather than raising.
    """
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(target)
    err = _masked_values(pred, target, valid)
    abs_err = np.abs(err)
    out: Dict[str, float] = {
        "rmse": rmse_from_errors(err),
        "mae": mae_from_errors(err),
        "median_ae": float(np.median(abs_err)) if abs_err.size else float("nan"),
        "p90_ae": float(np.percentile(abs_err, 90)) if abs_err.size else float("nan"),
        "p95_ae": float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
        "bias_db": float(np.mean(err)) if err.size else float("nan"),
        "std_error_db": float(np.std(err, ddof=1)) if err.size > 1 else float("nan"),
        "count": int(err.size),
        "valid_count": int(err.size),
    }
    if anchor_count is not None:
        out["anchor_count"] = int(anchor_count)
    if los_mask is not None:
        los = valid & (np.asarray(los_mask) > 0.5)
        nlos = valid & ~los
        out["los_rmse"] = _region_rmse(pred, target, los, min_region_pixels)
        out["nlos_rmse"] = _region_rmse(pred, target, nlos, min_region_pixels)
    if wall_count is not None:
        high = valid & (np.asarray(wall_count) >= int(high_wall_count_threshold))
        out["high_wall_rmse"] = rmse_from_errors(_masked_values(pred, target, high))
    if clip_max_db is not None and treat_clip_max_as_clipped:
        unclipped = valid & (np.asarray(target) < float(clip_max_db) - 1.0e-5)
        out["unclipped_rmse"] = rmse_from_errors(_masked_values(pred, target, unclipped))
        out["unclipped_count"] = int(unclipped.sum())
    clipped_hi = clip_max_db is not None and treat_clip_max_as_clipped
    clipped_lo = clip_min_db is not None
    if (clipped_hi or clipped_lo) and int(valid.sum()) > 0:
        tgt = np.asarray(target, dtype=np.float64)
        clipped = np.zeros_like(valid, dtype=bool)
        if clipped_hi:
            clipped |= valid & (tgt >= float(clip_max_db) - 1.0e-5)
        if clipped_lo:
            clipped |= valid & (tgt <= float(clip_min_db) + 1.0e-5)
        out["clipped_pixel_fraction"] = float(int(clipped.sum()) / int(valid.sum()))
    # Non-anchor (unobserved-pixel) metrics. Anchor pixels are observed inputs at
    # higher sparse rates; excluding them avoids crediting the model for pixels it
    # was simply given. Empty non-anchor region -> NaN, never a crash.
    if sparse_mask is not None:
        non_anchor = valid & ~np.asarray(sparse_mask, dtype=bool)
        na_err = _masked_values(pred, target, non_anchor)
        na_abs = np.abs(na_err)
        out["non_anchor_rmse"] = rmse_from_errors(na_err)
        out["non_anchor_mae"] = mae_from_errors(na_err)
        out["non_anchor_p90_ae"] = float(np.percentile(na_abs, 90)) if na_abs.size else float("nan")
        out["non_anchor_count"] = int(na_err.size)
        if los_mask is not None:
            los_bool = np.asarray(los_mask) > 0.5
            out["non_anchor_los_rmse"] = _region_rmse(pred, target, non_anchor & los_bool, min_region_pixels)
            out["non_anchor_nlos_rmse"] = _region_rmse(pred, target, non_anchor & ~los_bool, min_region_pixels)
    # Optional free-space (non-wall) sensitivity metrics over receiver pixels that
    # are not wall/interface pixels, plus the complementary wall-region RMSE.
    if wall_mask is not None:
        wall_bool = np.asarray(wall_mask, dtype=bool)
        free_space = valid & ~wall_bool
        fs_err = _masked_values(pred, target, free_space)
        fs_abs = np.abs(fs_err)
        out["free_space_rmse"] = rmse_from_errors(fs_err)
        out["free_space_mae"] = mae_from_errors(fs_err)
        out["free_space_p90_ae"] = float(np.percentile(fs_abs, 90)) if fs_abs.size else float("nan")
        out["free_space_count"] = int(fs_err.size)
        out["wall_region_rmse"] = rmse_from_errors(_masked_values(pred, target, valid & wall_bool))
    return out



def aggregate_metric_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    import pandas as pd
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in set(group_keys) and pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for keys, sub in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_keys, keys)}
        for col in metric_cols:
            row[col] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        row["num_rows"] = int(len(sub))
        out.append(row)
    return out
