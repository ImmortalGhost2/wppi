from __future__ import annotations

import numpy as np

from wallpath_pi.geometry.raster import bresenham_line, compute_base_maps, distance_map


def test_bresenham_diagonal():
    ys, xs = bresenham_line(0, 0, 3, 3)
    assert ys.tolist() == [0, 1, 2, 3]
    assert xs.tolist() == [0, 1, 2, 3]


def test_compute_base_maps_detects_wall_and_los():
    material = np.zeros((7, 7), dtype=np.int32)
    material[3, :] = 2
    wall = material > 0
    valid = ~wall
    base = compute_base_maps(wall, material, tx_xy=(1, 1), frequency_hz=3.5e9, resolution_m=0.5, valid_mask=valid, material_ids=[1, 2])
    assert base.material_counts.shape == (2, 7, 7)
    assert base.wall_count[5, 5] >= 1
    assert base.los[1, 2] == 1
    assert "mat_2_count" in base.feature_names


def test_distance_map_minimum():
    dist = distance_map((5, 5), tx_xy=(2, 2), resolution_m=0.25, min_distance_m=0.25)
    assert dist.shape == (5, 5)
    assert float(dist[2, 2]) == 0.25
