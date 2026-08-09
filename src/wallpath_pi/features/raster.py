"""Legacy raster-feature helpers (NON-CANONICAL).

Retained only for backward compatibility; the training pipeline does **not**
import this module. The canonical raster/Bresenham feature extraction lives in
``wallpath_pi.geometry.raster`` (``compute_base_maps``, ``bresenham_line``) and
``wallpath_pi.geometry.features`` (``build_feature_table``). Prefer those.

``bresenham_line`` here delegates to ``geometry.raster``; the DataFrame-oriented
helpers are a thin convenience layer over ``geometry.raster.compute_base_maps``.

Deprecated: do not add new call sites; import from ``geometry.raster`` /
``geometry.features``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from wallpath_pi.geometry.raster import bresenham_line as _bresenham_line, compute_base_maps


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    return _bresenham_line(x0, y0, x1, y1)


@dataclass
class LineMaterialStats:
    los: int
    total_wall_count: int
    counts_by_material: Dict[int, int]


def line_material_stats(material_map: np.ndarray, x0: int, y0: int, x1: int, y1: int, material_ids: Iterable[int] = (1, 2, 3)) -> LineMaterialStats:
    material = np.asarray(material_map, dtype=np.int32)
    yy, xx = _bresenham_line(x0, y0, x1, y1)
    yy = np.clip(yy, 0, material.shape[0] - 1)
    xx = np.clip(xx, 0, material.shape[1] - 1)
    mats = material[yy, xx]
    counts = {int(mid): int(np.sum(mats == int(mid))) for mid in material_ids}
    total = int(np.sum(mats > 0))
    return LineMaterialStats(los=int(total == 0), total_wall_count=total, counts_by_material=counts)


def compute_static_feature_frame(
    material_map: np.ndarray,
    valid_mask: np.ndarray,
    tx_xy: tuple[float, float],
    frequency_hz: float,
    cell_size_m: float,
    material_ids: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    material_ids = [int(m) for m in material_ids]
    material = np.asarray(material_map, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    base = compute_base_maps(
        wall_mask=material > 0,
        material_map=material,
        tx_xy=tx_xy,
        frequency_hz=float(frequency_hz),
        resolution_m=float(cell_size_m),
        valid_mask=valid,
        material_ids=material_ids,
        min_distance_m=float(cell_size_m),
    )
    yy, xx = np.nonzero(valid)
    df = pd.DataFrame({
        'x': xx.astype(int),
        'y': yy.astype(int),
        'distance_m': base.distance_m[valid].astype(float),
        'fspl_db': base.fspl[valid].astype(float),
        'los': base.los[valid].astype(int),
        'wall_count': base.wall_count[valid].astype(float),
        'wall_fraction': base.wall_fraction[valid].astype(float),
    })
    for i, mid in enumerate(material_ids):
        df[f'count_mat_{mid}'] = base.material_counts[i][valid].astype(float)
    return df
