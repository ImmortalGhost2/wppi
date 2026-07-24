"""Compatibility facade over the canonical training pipeline (NON-CANONICAL).

This module only re-exports and lightly wraps the canonical pipeline in
``wallpath_pi.training.pipeline`` (``run_experiment``, ``evaluate_saved_run``,
``_prepare_sample``). It exists so older imports and notebooks keep working.
New code should import directly from ``wallpath_pi.training.pipeline``.

``build_map_frame`` and ``material_ids`` are convenience helpers with no
canonical equivalent and remain safe to use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from wallpath_pi.data.dataset import load_scene_npz, resolve_scene_path
from wallpath_pi.training.pipeline import _prepare_sample, evaluate_saved_run, run_experiment


@dataclass
class BuiltMapFrame:
    frame: pd.DataFrame
    sample_id: str
    scene_id: str


def material_ids(config: Dict[str, Any]) -> list[int]:
    return [int(x) for x in config.get('material_ids', (config.get('dataset', {}) or {}).get('material_ids', [1, 2, 3]))]


def build_map_frame(row, data_root: Path, config: Dict[str, Any], sparse_rate: float = 0.05, seed: int = 11, training: bool = True) -> BuiltMapFrame:
    """Compatibility helper returning a pandas feature frame for one manifest row."""
    row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    scene_col = 'scene_path' if 'scene_path' in row_dict else 'map_path'
    sample = load_scene_npz(resolve_scene_path(data_root, row_dict[scene_col]), row=row_dict)
    prep = _prepare_sample(sample, config, sparse_rate=float(sparse_rate), sparse_seed=int(seed))
    frame = pd.DataFrame(prep.feature_table.X, columns=prep.feature_table.feature_names)
    frame['x'] = prep.feature_table.coords_xy[:, 0]
    frame['y'] = prep.feature_table.coords_xy[:, 1]
    frame['path_loss_db'] = prep.feature_table.y
    frame['multiwall_pred_db'] = prep.feature_table.baseline_values
    frame['residual_db'] = frame['path_loss_db'] - frame['multiwall_pred_db']
    return BuiltMapFrame(frame=frame, sample_id=sample.sample_id, scene_id=sample.scene_id)


__all__ = ['run_experiment', 'evaluate_saved_run', 'build_map_frame', 'material_ids', 'BuiltMapFrame']
