from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from wallpath_pi.utils.hashing import file_hashes, stable_int_hash


def make_scene_disjoint_split(
    input_csv: Path,
    train_out: Path,
    val_out: Path,
    meta_out: Path,
    group_column: str = "scene_id",
    val_ratio: float = 0.25,
    seed: int = 42,
    min_groups: int = 2,
    summary_out: Path | None = None,
) -> Dict[str, object]:
    """Write a deterministic, group-disjoint train/validation split.

    The split guarantees that no value of ``group_column`` (the scene or layout
    identifier by default) appears in both partitions, which is required for an
    honest generalization estimate on real ICASSP data. Raises ``ValueError``
    with an actionable message when the manifest cannot support such a split.
    """
    input_csv = Path(input_csv).expanduser().resolve()
    df = pd.read_csv(input_csv)
    if group_column not in df.columns:
        raise ValueError(
            f"Missing group column '{group_column}' in {input_csv}. "
            f"Available columns: {sorted(df.columns)}."
        )
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in the open interval (0, 1); got {val_ratio}.")
    groups = sorted(df[group_column].astype(str).unique().tolist())
    if len(groups) < max(2, int(min_groups)):
        raise ValueError(
            f"Need at least {max(2, int(min_groups))} unique '{group_column}' values "
            f"for a group-disjoint split, but found {len(groups)} in {input_csv}. "
            "Convert more scenes/layouts or lower --min-groups."
        )
    rng = np.random.default_rng(stable_int_hash("split", input_csv.name, group_column, seed))
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(groups) * float(val_ratio))))
    n_val = min(len(groups) - 1, n_val)
    if n_val < 1 or n_val > len(groups) - 1:
        raise ValueError(
            f"val_ratio={val_ratio} on {len(groups)} groups leaves an empty train or "
            "validation partition. Choose a ratio that keeps at least one group on each side."
        )
    val_groups = set(shuffled[:n_val])
    train_df = df[~df[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
    val_df = df[df[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Split produced an empty train or validation set.")
    overlap = set(train_df[group_column].astype(str)) & set(val_df[group_column].astype(str))
    if overlap:
        raise AssertionError(f"Group leakage detected: {sorted(overlap)}")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    meta = {
        "input_csv": str(input_csv),
        "train_csv": str(train_out),
        "val_csv": str(val_out),
        "group_column": group_column,
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "counts": {"input": int(len(df)), "train": int(len(train_df)), "val": int(len(val_df))},
        "groups": {"input": int(len(groups)), "train": int(train_df[group_column].nunique()), "val": int(val_df[group_column].nunique())},
        "val_groups": sorted(val_groups),
        "hashes": {
            "input_csv": file_hashes(input_csv),
            "train_csv": file_hashes(train_out),
            "val_csv": file_hashes(val_out),
        },
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if summary_out is not None:
        write_split_summary(meta, Path(summary_out))
    return meta


def make_fixed_scene_split(
    input_csv: Path,
    train_out: Path,
    val_out: Path,
    meta_out: Path,
    train_scenes,
    val_scenes,
    group_column: str = "scene_id",
    min_groups: int = 2,
    summary_out: Path | None = None,
) -> Dict[str, object]:
    """Write a split using explicit ``train_scenes`` / ``val_scenes`` lists.

    Unlike :func:`make_scene_disjoint_split` (which samples ``val_ratio`` of the
    scenes at random), this assigns each scene by the exact lists supplied, so
    the partition is fully reproducible and seed-independent. It enforces that
    the two lists are disjoint, that every listed scene exists in the manifest,
    and that the lists together cover every scene in the manifest, so each row
    lands in exactly one split with no silent drops and no scene leakage. Output
    files (``train_split.csv``, ``val_split.csv``, ``split_meta.json`` and the
    optional ``split_summary.json``) match the ``val_ratio`` path.
    """
    input_csv = Path(input_csv).expanduser().resolve()
    df = pd.read_csv(input_csv)
    if group_column not in df.columns:
        raise ValueError(
            f"Missing group column '{group_column}' in {input_csv}. "
            f"Available columns: {sorted(df.columns)}."
        )
    train_list = [str(s) for s in (train_scenes or [])]
    val_list = [str(s) for s in (val_scenes or [])]
    train_set, val_set = set(train_list), set(val_list)
    if not train_set or not val_set:
        raise ValueError("Both train_scenes and val_scenes must be non-empty for a fixed scene split.")
    if len(train_set) != len(train_list) or len(val_set) != len(val_list):
        raise ValueError("train_scenes/val_scenes contain duplicate entries; list each scene once.")
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(
            f"train_scenes and val_scenes overlap on {group_column} values {overlap}; "
            "a scene-disjoint split requires them to be disjoint."
        )
    present = set(df[group_column].astype(str).unique())
    requested = train_set | val_set
    missing = sorted(requested - present)
    if missing:
        raise ValueError(
            f"Requested scenes not present in {input_csv}: {missing}. "
            f"Available {group_column} values: {sorted(present)}."
        )
    uncovered = sorted(present - requested)
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} {group_column} value(s) in {input_csv} are assigned to neither "
            f"train_scenes nor val_scenes: {uncovered}. List every scene so each row belongs to "
            "exactly one split, or omit the fixed lists to use the automatic val_ratio split."
        )
    if len(requested) < max(2, int(min_groups)):
        raise ValueError(
            f"Need at least {max(2, int(min_groups))} unique '{group_column}' values for a "
            f"group-disjoint split; got {len(requested)}."
        )
    train_df = df[df[group_column].astype(str).isin(train_set)].reset_index(drop=True)
    val_df = df[df[group_column].astype(str).isin(val_set)].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Fixed scene split produced an empty train or validation set.")
    leak = set(train_df[group_column].astype(str)) & set(val_df[group_column].astype(str))
    if leak:
        raise AssertionError(f"Group leakage detected: {sorted(leak)}")
    if len(train_df) + len(val_df) != len(df):
        raise AssertionError("Fixed scene split did not partition every manifest row exactly once.")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    meta = {
        "input_csv": str(input_csv),
        "train_csv": str(train_out),
        "val_csv": str(val_out),
        "group_column": group_column,
        "split_mode": "fixed_scenes",
        "val_ratio": None,
        "seed": None,
        "counts": {"input": int(len(df)), "train": int(len(train_df)), "val": int(len(val_df))},
        "groups": {"input": int(len(present)), "train": int(train_df[group_column].nunique()), "val": int(val_df[group_column].nunique())},
        "train_groups": sorted(train_set),
        "val_groups": sorted(val_set),
        "hashes": {
            "input_csv": file_hashes(input_csv),
            "train_csv": file_hashes(train_out),
            "val_csv": file_hashes(val_out),
        },
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if summary_out is not None:
        write_split_summary(meta, Path(summary_out))
    return meta


def write_split_summary(meta: Dict[str, object], summary_out: Path) -> Path:
    """Write a compact split_summary.json with sample and scene counts."""
    counts = meta.get("counts", {})  # type: ignore[assignment]
    groups = meta.get("groups", {})  # type: ignore[assignment]
    summary = {
        "group_column": meta.get("group_column"),
        "split_mode": meta.get("split_mode", "val_ratio"),
        "val_ratio": meta.get("val_ratio"),
        "seed": meta.get("seed"),
        "samples": {
            "input": counts.get("input"),
            "train": counts.get("train"),
            "val": counts.get("val"),
        },
        "scenes": {
            "input": groups.get("input"),
            "train": groups.get("train"),
            "val": groups.get("val"),
        },
        "train_groups": meta.get("train_groups"),
        "val_groups": meta.get("val_groups"),
        "group_disjoint": True,
    }
    summary_out = Path(summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_out


def write_splits(
    df: pd.DataFrame,
    data_root: Path,
    train_csv: str = 'train_split.csv',
    val_csv: str = 'val_split.csv',
    test_csv: str = 'test_split.csv',
    group_column: str = 'scene_id',
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """Write deterministic scene-disjoint train/val/test splits from an in-memory DataFrame."""
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if group_column not in df.columns:
        raise ValueError(f'Missing group column {group_column}')
    groups = sorted(df[group_column].astype(str).unique().tolist())
    if len(groups) < 3:
        # Keep test optional for tiny smoke cases.
        return make_scene_disjoint_split(
            input_csv=(data_root / 'manifest.csv') if (data_root / 'manifest.csv').exists() else _write_temp_manifest(df, data_root),
            train_out=data_root / train_csv,
            val_out=data_root / val_csv,
            meta_out=data_root / 'split_meta.json',
            group_column=group_column,
            val_ratio=val_ratio,
            seed=seed,
        )
    rng = np.random.default_rng(stable_int_hash('split3', group_column, seed))
    shuffled = groups.copy(); rng.shuffle(shuffled)
    n_test = max(1, int(round(len(groups) * float(test_ratio))))
    n_val = max(1, int(round(len(groups) * float(val_ratio))))
    n_test = min(n_test, len(groups) - 2)
    n_val = min(n_val, len(groups) - n_test - 1)
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test+n_val])
    train_groups = set(shuffled[n_test+n_val:])
    parts = {
        train_csv: df[df[group_column].astype(str).isin(train_groups)].reset_index(drop=True),
        val_csv: df[df[group_column].astype(str).isin(val_groups)].reset_index(drop=True),
        test_csv: df[df[group_column].astype(str).isin(test_groups)].reset_index(drop=True),
    }
    for name, part in parts.items():
        part.to_csv(data_root / name, index=False)
    meta = {
        'group_column': group_column,
        'seed': int(seed),
        'counts': {k: int(len(v)) for k, v in parts.items()},
        'train_groups': sorted(train_groups),
        'val_groups': sorted(val_groups),
        'test_groups': sorted(test_groups),
    }
    (data_root / 'split_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def _write_temp_manifest(df: pd.DataFrame, data_root: Path) -> Path:
    path = Path(data_root) / 'manifest.csv'
    df.to_csv(path, index=False)
    return path
