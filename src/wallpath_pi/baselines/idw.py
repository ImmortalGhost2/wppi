from __future__ import annotations

import numpy as np


def coords_from_mask(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    return np.stack([xx, yy], axis=1).astype(np.float32)


# Adaptive query-chunk defaults. The per-chunk (query x anchor) temporaries are
# float32; a conservative multiplier accounts for the several arrays alive at
# once (dx, dy, dist2, exact, weights). The cap keeps low-anchor cases fast.
_IDW_MAX_QUERY_CHUNK = 8192
_IDW_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024
_IDW_TEMP_ARRAYS = 5


def _resolve_query_chunk(
    query_chunk_size: int | str | None,
    n_anchor: int,
    max_chunk: int = _IDW_MAX_QUERY_CHUNK,
    memory_budget_bytes: int = _IDW_MEMORY_BUDGET_BYTES,
    n_temp_arrays: int = _IDW_TEMP_ARRAYS,
) -> int:
    """Resolve the effective query-chunk size.

    An integer ``query_chunk_size`` is used verbatim (floored at 1). ``None`` or
    ``"auto"`` selects adaptive chunking: the chunk equals the memory budget
    divided by the per-query-row cost (``n_anchor`` float32 values times
    ``n_temp_arrays`` temporaries), clamped to ``[1, max_chunk]``. The result is
    large for small anchor sets and shrinks automatically as ``n_anchor`` grows.
    """
    if isinstance(query_chunk_size, bool):
        raise ValueError("query_chunk_size must be a positive int, None, or 'auto'.")
    if isinstance(query_chunk_size, (int, np.integer)):
        return max(1, int(query_chunk_size))
    if not (query_chunk_size is None or (isinstance(query_chunk_size, str) and query_chunk_size.lower() == "auto")):
        raise ValueError(f"query_chunk_size must be a positive int, None, or 'auto'; got {query_chunk_size!r}.")
    cap = max(1, int(max_chunk))
    if int(n_anchor) <= 0:
        return cap
    bytes_per_query_row = int(n_anchor) * 4 * max(1, int(n_temp_arrays))
    auto = int(memory_budget_bytes) // max(1, bytes_per_query_row)
    return max(1, min(cap, auto))


def idw_predict_points(
    anchor_xy: np.ndarray,
    anchor_values: np.ndarray,
    query_xy: np.ndarray,
    power: float = 2.0,
    eps: float = 1.0e-6,
    chunk_size: int = _IDW_MAX_QUERY_CHUNK,
    default_value: float = 0.0,
    query_chunk_size: int | str | None = None,
    memory_budget_bytes: int = _IDW_MEMORY_BUDGET_BYTES,
    n_temp_arrays: int = _IDW_TEMP_ARRAYS,
) -> np.ndarray:
    """Inverse-distance-weighted prediction at ``query_xy`` from sparse anchors.

    Predictions are evaluated in query chunks so that peak memory scales with the
    chunk size times the number of anchors, never ``n_query`` times ``n_anchor``.
    With ``query_chunk_size=None`` or ``"auto"`` the chunk is chosen adaptively
    from ``memory_budget_bytes`` and the anchor count: large (up to ``chunk_size``)
    for small anchor sets and automatically smaller for dense ones. An integer
    ``query_chunk_size`` fixes the chunk exactly, and ``chunk_size`` is the
    maximum cap for the adaptive path. Chunking never changes the numerical
    result. An empty anchor set returns ``default_value`` for every query point.
    """
    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    query_xy = np.asarray(query_xy, dtype=np.float32)
    anchor_values = np.asarray(anchor_values, dtype=np.float32).reshape(-1)
    if anchor_xy.size == 0 or anchor_values.size == 0:
        return np.full((query_xy.shape[0],), float(default_value), dtype=np.float32)
    if anchor_xy.shape[0] != anchor_values.shape[0]:
        raise ValueError("anchor_xy and anchor_values length mismatch.")
    out = np.empty((query_xy.shape[0],), dtype=np.float32)
    p = float(power)
    e = float(eps)
    eff_chunk = _resolve_query_chunk(query_chunk_size, anchor_xy.shape[0], chunk_size, memory_budget_bytes, n_temp_arrays)
    for start in range(0, query_xy.shape[0], eff_chunk):
        q = query_xy[start:start + eff_chunk]
        # Per-coordinate differences keep the temporary 2D (chunk x n_anchor) and
        # never materialize a full (chunk x n_anchor x 2) or (n_query x n_anchor)
        # array, so memory stays bounded for large maps and dense anchor sets.
        dx = q[:, 0][:, None] - anchor_xy[None, :, 0]
        dy = q[:, 1][:, None] - anchor_xy[None, :, 1]
        dist2 = dx * dx + dy * dy
        exact = dist2 <= e * e
        weights = 1.0 / np.maximum(dist2, e * e) ** (p / 2.0)
        pred = (weights @ anchor_values) / np.maximum(weights.sum(axis=1), e)
        if exact.any():
            rows = np.where(exact.any(axis=1))[0]
            for r in rows:
                pred[r] = anchor_values[int(np.argmax(exact[r]))]
        out[start:start + len(q)] = pred.astype(np.float32)
    return out


def idw_map_from_mask(
    value_map: np.ndarray,
    anchor_mask: np.ndarray,
    query_mask: np.ndarray,
    power: float = 2.0,
    eps: float = 1.0e-6,
    default_value: float | None = None,
    on_empty: str = "constant",
    empty_constant: float = 0.0,
    query_chunk_size: int | str | None = None,
) -> np.ndarray:
    """Interpolate a map from anchor observations using IDW.

    The fill value used for the empty-anchor case and for background (non-query)
    cells is derived only from the sparse anchors or from caller-supplied
    constants. It is never taken from the dense ``value_map`` over the query
    region, so evaluation targets cannot leak when anchors are sparse or empty.

    Parameters
    ----------
    default_value:
        Explicit background/fallback fill. If ``None``, a training-safe value is
        used: the mean of the anchor values when anchors exist, otherwise the
        empty-anchor handling below.
    on_empty:
        Behaviour when no anchors are available. ``"constant"`` (default) fills
        with ``default_value`` if given, else ``empty_constant``. ``"raise"``
        raises a ``ValueError``.
    empty_constant:
        Constant used when there are no anchors, no ``default_value``, and
        ``on_empty="constant"``.
    query_chunk_size:
        Forwarded to :func:`idw_predict_points`. ``None`` or ``"auto"`` (default)
        selects adaptive query chunking from the anchor count and memory budget;
        an integer fixes the query-chunk size. It never changes the numerical
        result, only peak memory and speed.
    """
    value_map = np.asarray(value_map, dtype=np.float32)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    query_mask = np.asarray(query_mask, dtype=bool)
    if on_empty not in {"constant", "raise"}:
        raise ValueError(f"idw_map_from_mask: unknown on_empty={on_empty!r} (use 'constant' or 'raise').")

    anchor_xy = coords_from_mask(anchor_mask)
    anchor_values = value_map[anchor_mask]
    if anchor_xy.shape[0] == 0:
        if on_empty == "raise":
            raise ValueError("idw_map_from_mask: no anchors available and on_empty='raise'.")
        fill = float(default_value) if default_value is not None else float(empty_constant)
        return np.full_like(value_map, fill, dtype=np.float32)

    # Anchors exist: choose a fill that depends only on the sparse anchors (or an
    # explicit constant), never on the dense query-region targets.
    fill = float(default_value) if default_value is not None else float(np.mean(anchor_values))
    query_xy = coords_from_mask(query_mask)
    pred_values = idw_predict_points(anchor_xy, anchor_values, query_xy, power=power, eps=eps, default_value=fill, query_chunk_size=query_chunk_size)
    out = np.full_like(value_map, fill, dtype=np.float32)
    yy, xx = np.nonzero(query_mask)
    out[yy, xx] = pred_values
    return out
