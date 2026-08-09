"""Canonical raster geometry: Bresenham rays and per-sample base maps.

Recommended API for distance/FSPL/LOS/wall-count rasters; this is what the
training pipeline uses. The older ``wallpath_pi.features.raster`` module is a
non-canonical compatibility layer over this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from wallpath_pi.baselines.propagation import fspl_db


@dataclass
class BaseMaps:
    distance_m: np.ndarray
    fspl: np.ndarray
    los: np.ndarray
    wall_count: np.ndarray
    wall_fraction: np.ndarray
    material_counts: np.ndarray
    transmittance_sum: np.ndarray
    reflectance_sum: np.ndarray
    valid_mask: np.ndarray
    feature_names: list[str]


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Return integer pixel coordinates along a line as (yy, xx)."""
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    xs = []
    ys = []
    x, y = x0, y0
    while True:
        xs.append(x)
        ys.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def distance_map(shape: tuple[int, int], tx_xy: Sequence[float], resolution_m: float, min_distance_m: float = 0.25) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    tx_x, tx_y = float(tx_xy[0]), float(tx_xy[1])
    dist = np.sqrt((xx - tx_x) ** 2 + (yy - tx_y) ** 2) * float(resolution_m)
    return np.maximum(dist, float(min_distance_m)).astype(np.float32)


def compute_base_maps(
    wall_mask: np.ndarray,
    material_map: np.ndarray,
    tx_xy: Sequence[float],
    frequency_hz: float,
    resolution_m: float,
    valid_mask: np.ndarray | None = None,
    material_ids: Iterable[int] = (1, 2, 3),
    transmittance: np.ndarray | None = None,
    reflectance: np.ndarray | None = None,
    min_distance_m: float = 0.25,
) -> BaseMaps:
    wall_mask = np.asarray(wall_mask, dtype=bool)
    material_map = np.asarray(material_map, dtype=np.int32)
    if wall_mask.shape != material_map.shape:
        raise ValueError("wall_mask and material_map shape mismatch.")
    H, W = wall_mask.shape
    if valid_mask is None:
        valid = ~wall_mask
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    material_ids = [int(m) for m in material_ids]
    counts = np.zeros((len(material_ids), H, W), dtype=np.float32)
    wall_count = np.zeros((H, W), dtype=np.float32)
    wall_fraction = np.zeros((H, W), dtype=np.float32)
    trans_sum = np.zeros((H, W), dtype=np.float32)
    refl_sum = np.zeros((H, W), dtype=np.float32)

    tx_x = int(np.clip(round(float(tx_xy[0])), 0, W - 1))
    tx_y = int(np.clip(round(float(tx_xy[1])), 0, H - 1))
    trans = np.asarray(transmittance, dtype=np.float32) if transmittance is not None else None
    refl = np.asarray(reflectance, dtype=np.float32) if reflectance is not None else None

    for y in range(H):
        for x in range(W):
            if not valid[y, x]:
                continue
            yy, xx = bresenham_line(tx_x, tx_y, x, y)
            yy = np.clip(yy, 0, H - 1)
            xx = np.clip(xx, 0, W - 1)
            mats = material_map[yy, xx]
            is_wall = mats > 0
            wc = int(is_wall.sum())
            wall_count[y, x] = float(wc)
            wall_fraction[y, x] = float(wc) / max(1.0, float(len(mats)))
            for i, mid in enumerate(material_ids):
                counts[i, y, x] = float(np.sum(mats == mid))
            if trans is not None:
                trans_sum[y, x] = float(trans[yy, xx].sum())
            if refl is not None:
                refl_sum[y, x] = float(refl[yy, xx].sum())

    dist = distance_map((H, W), tx_xy=tx_xy, resolution_m=resolution_m, min_distance_m=min_distance_m)
    fspl = fspl_db(dist, frequency_hz=frequency_hz, min_distance_m=min_distance_m)
    los = ((wall_count <= 0) & valid).astype(np.float32)
    names = ["distance_m", "log_distance", "fspl_db", "los", "wall_count", "wall_fraction"]
    names.extend([f"mat_{m}_count" for m in material_ids])
    names.extend(["transmittance_sum", "reflectance_sum"])
    return BaseMaps(
        distance_m=dist.astype(np.float32),
        fspl=fspl.astype(np.float32),
        los=los.astype(np.float32),
        wall_count=wall_count.astype(np.float32),
        wall_fraction=wall_fraction.astype(np.float32),
        material_counts=counts.astype(np.float32),
        transmittance_sum=trans_sum.astype(np.float32),
        reflectance_sum=refl_sum.astype(np.float32),
        valid_mask=valid.astype(bool),
        feature_names=names,
    )
