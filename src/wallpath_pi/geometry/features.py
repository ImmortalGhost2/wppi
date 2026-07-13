"""Canonical feature-table builder for the training pipeline.

``build_feature_table`` assembles coordinate, raster wall, material,
sparse-anchor, and baseline-prediction features. This is the recommended API;
the older ``wallpath_pi.features`` package is non-canonical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from wallpath_pi.baselines.idw import coords_from_mask, idw_predict_points
from wallpath_pi.geometry.raster import BaseMaps, compute_base_maps


@dataclass
class FeatureTable:
    X: np.ndarray
    y: np.ndarray
    coords_xy: np.ndarray
    valid_mask: np.ndarray
    feature_names: list[str]
    baseline_values: np.ndarray


def build_base_maps_for_sample(sample, material_ids: Sequence[int], min_distance_m: float) -> BaseMaps:
    return compute_base_maps(
        wall_mask=sample.wall_mask,
        material_map=sample.material_map,
        tx_xy=sample.tx_xy,
        frequency_hz=sample.frequency_hz,
        resolution_m=sample.resolution_m,
        valid_mask=sample.valid_mask,
        material_ids=material_ids,
        transmittance=sample.transmittance,
        reflectance=sample.reflectance,
        min_distance_m=min_distance_m,
    )


def _knn_nearest_chunked(
    coords: np.ndarray,
    anchor_xy: np.ndarray,
    n_neighbors: int,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked k-nearest-anchor search (fallback when scikit-learn is absent).

    Returns ``(distances, indices)``, both ``(n_query, n_neighbors)`` and sorted
    by ascending distance. Query points are processed in blocks so the largest
    temporary is ``(chunk_size, n_anchor)`` rather than ``(n_query, n_anchor)``.
    """
    coords = np.asarray(coords, dtype=np.float32)
    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    n_query = int(coords.shape[0])
    n_anchor = int(anchor_xy.shape[0])
    n_neighbors = int(min(n_neighbors, n_anchor))

    dist_out = np.empty((n_query, n_neighbors), dtype=np.float32)
    idx_out = np.empty((n_query, n_neighbors), dtype=np.intp)
    ax = anchor_xy[:, 0][None, :]
    ay = anchor_xy[:, 1][None, :]
    for start in range(0, n_query, int(chunk_size)):
        stop = min(start + int(chunk_size), n_query)
        qx = coords[start:stop, 0][:, None]
        qy = coords[start:stop, 1][:, None]
        dx = qx - ax
        dy = qy - ay
        d2 = dx * dx + dy * dy  # (chunk, n_anchor), never (n_query, n_anchor)
        if n_neighbors < n_anchor:
            part = np.argpartition(d2, n_neighbors - 1, axis=1)[:, :n_neighbors]
        else:
            part = np.broadcast_to(np.arange(n_anchor), d2.shape).copy()
        part_d2 = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(part_d2, axis=1)
        idx_out[start:stop] = np.take_along_axis(part, order, axis=1)
        dist_out[start:stop] = np.sqrt(np.maximum(np.take_along_axis(part_d2, order, axis=1), 0.0))
    return dist_out, idx_out


def _knn_nearest(
    coords: np.ndarray,
    anchor_xy: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distances/indices of the ``n_neighbors`` nearest anchors per query.

    Uses :class:`sklearn.neighbors.NearestNeighbors` when available (avoiding any
    full pairwise distance matrix), and a memory-bounded chunked search otherwise.
    Both outputs are ``(n_query, n_neighbors)``, sorted by ascending distance.
    """
    coords = np.asarray(coords, dtype=np.float32)
    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    n_neighbors = int(min(n_neighbors, anchor_xy.shape[0]))
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return _knn_nearest_chunked(coords, anchor_xy, n_neighbors)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(anchor_xy)
    distances, indices = nn.kneighbors(coords, return_distance=True)
    return distances.astype(np.float32), indices.astype(np.intp)


def _nearest_anchor_features(
    coords: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_residuals: np.ndarray,
    k: int,
    eps: float,
) -> tuple[np.ndarray, list[str]]:
    k = int(max(0, k))
    n_query = int(np.asarray(coords).shape[0])
    names = [f"anchor_residual_nn_{i+1}" for i in range(k)] + [f"anchor_dist_px_nn_{i+1}" for i in range(k)]
    if k == 0:
        return np.zeros((n_query, 0), dtype=np.float32), []

    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    anchor_residuals = np.asarray(anchor_residuals, dtype=np.float32).reshape(-1)
    n_anchor = int(anchor_xy.shape[0])
    if n_anchor == 0:
        # No anchors: zero residuals and zero distances (fixed feature dimension).
        return np.zeros((n_query, 2 * k), dtype=np.float32), names

    n_neighbors = min(k, n_anchor)
    dist_k, idx_k = _knn_nearest(coords, anchor_xy, n_neighbors)
    res_k = anchor_residuals[idx_k]

    if n_neighbors < k:
        # Fewer anchors than k: pad residuals with 0.0 and distances with a large
        # finite value, keeping the fixed (2 * k) feature dimension and names.
        pad = k - n_neighbors
        dist_k = np.pad(dist_k, ((0, 0), (0, pad)), constant_values=np.nan)
        res_k = np.pad(res_k, ((0, 0), (0, pad)), constant_values=0.0)

    finite = dist_k[np.isfinite(dist_k)]
    max_dist = float(finite.max()) if finite.size else 1.0
    dist_k = np.nan_to_num(dist_k, nan=max_dist + 1.0)
    arr = np.concatenate([res_k, dist_k], axis=1).astype(np.float32)
    return arr, names


def build_feature_table(
    sample,
    base_maps: BaseMaps,
    sparse_mask: np.ndarray,
    multiwall_pred: np.ndarray,
    material_ids: Iterable[int],
    feature_cfg: Dict[str, object],
    target_mask: np.ndarray | None = None,
    baseline_pred_maps: Dict[str, np.ndarray] | None = None,
    query_indices: np.ndarray | None = None,
) -> FeatureTable:
    valid = np.asarray(base_maps.valid_mask, dtype=bool)
    if target_mask is not None:
        valid = valid & np.asarray(target_mask, dtype=bool)
    yy_all, xx_all = np.nonzero(valid)
    if query_indices is not None:
        # Build features only at the requested query pixels (a subset of the
        # valid pixels, indexed into the row-major nonzero order). Sparse-anchor
        # features below still use the full declared anchor set; only the query
        # rows are restricted. Selecting rows up front avoids materializing the
        # full dense feature table for every training sample.
        qi = np.asarray(query_indices, dtype=np.intp).reshape(-1)
        yy = yy_all[qi]
        xx = xx_all[qi]
        query_mask = np.zeros_like(valid)
        query_mask[yy, xx] = True
    else:
        yy, xx = yy_all, xx_all
        query_mask = valid
    coords = np.stack([xx, yy], axis=1).astype(np.float32)
    H, W = valid.shape
    x_norm = (xx.astype(np.float32) / max(1, W - 1)).reshape(-1, 1)
    y_norm = (yy.astype(np.float32) / max(1, H - 1)).reshape(-1, 1)

    feats = []
    names = []
    if bool(feature_cfg.get("coordinate_features", True)):
        feats.extend([x_norm, y_norm])
        names.extend(["x_norm", "y_norm"])
    dist = base_maps.distance_m[yy, xx].reshape(-1, 1)
    logd = np.log10(np.maximum(dist, 1.0e-6))
    fspl = base_maps.fspl[yy, xx].reshape(-1, 1)
    los = base_maps.los[yy, xx].reshape(-1, 1)
    wall_count = base_maps.wall_count[yy, xx].reshape(-1, 1)
    wall_fraction = base_maps.wall_fraction[yy, xx].reshape(-1, 1)
    feats.extend([dist, logd, fspl, los, wall_count, wall_fraction])
    names.extend(["distance_m", "log10_distance_m", "fspl_db", "los", "wall_count", "wall_fraction"])

    if bool(feature_cfg.get("material_features", True)):
        for i, mid in enumerate(material_ids):
            feats.append(base_maps.material_counts[i][yy, xx].reshape(-1, 1))
            names.append(f"mat_{int(mid)}_count")
        feats.append(base_maps.transmittance_sum[yy, xx].reshape(-1, 1))
        feats.append(base_maps.reflectance_sum[yy, xx].reshape(-1, 1))
        names.extend(["transmittance_sum", "reflectance_sum"])

    if bool(feature_cfg.get("baseline_prediction_features", False)) and baseline_pred_maps:
        # Fitted/analytic baseline predictions as explicit inputs so the direct
        # model is a fair, well-powered competitor. All maps are functions of the
        # sparse anchors and geometry, so they are safe at inference time.
        for name in ("fspl_pred_db", "log_distance_pred_db", "multi_wall_pred_db", "multi_wall_residual_idw_pred_db"):
            bmap = baseline_pred_maps.get(name)
            if bmap is None:
                continue
            feats.append(np.asarray(bmap, dtype=np.float32)[yy, xx].reshape(-1, 1))
            names.append(name)

    sparse = np.asarray(sparse_mask, dtype=bool) & np.asarray(base_maps.valid_mask, dtype=bool)
    anchor_xy = coords_from_mask(sparse)
    anchor_resid_map = np.asarray(sample.path_loss - multiwall_pred, dtype=np.float32)
    anchor_residuals = anchor_resid_map[sparse].astype(np.float32)
    if bool(feature_cfg.get("sparse_anchor_features", True)):
        idw_power = float(feature_cfg.get("idw_power", 2.0))
        idw_eps = float(feature_cfg.get("idw_eps", 1.0e-6))
        default_res = float(np.nanmean(anchor_residuals)) if anchor_residuals.size else 0.0
        idw_res = idw_predict_points(anchor_xy, anchor_residuals, coords, power=idw_power, eps=idw_eps, default_value=default_res)
        feats.append(idw_res.reshape(-1, 1))
        names.append("anchor_residual_idw")
        nearest, nearest_names = _nearest_anchor_features(coords, anchor_xy, anchor_residuals, int(feature_cfg.get("k_nearest_anchors", 4)), idw_eps)
        feats.append(nearest)
        names.extend(nearest_names)
        # Count/density summaries are scalar scene-level values repeated for each row.
        anchor_density = float(anchor_xy.shape[0]) / max(1.0, float(np.asarray(base_maps.valid_mask, dtype=bool).sum()))
        feats.append(np.full((coords.shape[0], 1), anchor_density, dtype=np.float32))
        names.append("anchor_density")

    X = np.concatenate(feats, axis=1).astype(np.float32)
    y = sample.path_loss[yy, xx].astype(np.float32)
    baseline_values = multiwall_pred[yy, xx].astype(np.float32)
    return FeatureTable(X=X, y=y, coords_xy=coords, valid_mask=query_mask, feature_names=names, baseline_values=baseline_values)
