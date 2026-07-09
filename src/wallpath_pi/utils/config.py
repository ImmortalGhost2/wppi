from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config and flatten its sections into a single lookup dict.

    Whitelisted keys from each section (``paths``, ``dataset``, ``train``, and so
    on) are promoted to the top level for convenient access, while the untouched
    document is preserved under ``_full_config`` and its source path under
    ``_config_path``.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Pass a valid path with --config, "
            f"e.g. --config configs/config.yaml (run from the repository root)."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _validate_minimal(raw, path)
    flat: Dict[str, Any] = {}
    _flatten_into(flat, raw)
    flat["_full_config"] = raw
    flat["_config_path"] = str(path)
    return flat


def _set(out: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        out[key] = value


def _flatten_into(out: Dict[str, Any], cfg: Mapping[str, Any]) -> None:
    paths = cfg.get("paths", {}) or {}
    synthetic = cfg.get("synthetic", {}) or {}
    dataset = cfg.get("dataset", {}) or {}
    split = cfg.get("split", {}) or {}
    experiment = cfg.get("experiment", {}) or {}
    train = cfg.get("train", {}) or {}
    baseline = cfg.get("baseline", {}) or {}
    evaluation = cfg.get("evaluation", {}) or {}
    system = cfg.get("system", {}) or {}

    for key in ["data_root", "results_root", "train_csv", "val_csv", "eval_csv"]:
        _set(out, key, paths.get(key))
    _set(out, "cache_root", paths.get("cache_root"))
    _set(out, "synthetic", synthetic)
    _set(out, "dataset", dataset)
    _set(out, "material_ids", dataset.get("material_ids"))
    _set(out, "min_distance_m", dataset.get("min_distance_m"))
    _set(out, "clip_min_db", dataset.get("clip_min_db"))
    _set(out, "clip_max_db", dataset.get("clip_max_db"))
    _set(out, "treat_clip_max_as_clipped", dataset.get("treat_clip_max_as_clipped"))
    _set(out, "feature_set", dataset.get("feature_set"))
    _set(out, "cache_features", dataset.get("cache_features"))

    _set(out, "group_column", split.get("group_column"))
    _set(out, "val_ratio", split.get("val_ratio"))
    _set(out, "split_seed", split.get("seed"))

    _set(out, "experiment_name", experiment.get("experiment_name"))
    _set(out, "seed", experiment.get("seed"))

    for key in [
        "methods", "primary_method", "sparse_rates", "sparse_seeds", "min_anchors",
        "max_train_points_per_sample", "baseline_ridge_alpha", "model_params",
        "prepare_n_jobs", "prepare_backend", "feature_groups_by_method",
    ]:
        _set(out, key, train.get(key))
    _set(out, "baseline", baseline)
    _set(out, "evaluation", evaluation)
    _set(out, "system", system)
    _set(out, "deterministic", system.get("deterministic"))
    _set(out, "num_threads", system.get("num_threads"))


def _validate_minimal(raw: Mapping[str, Any], path: Path) -> None:
    required = {
        "paths.data_root": (raw.get("paths", {}) or {}).get("data_root"),
        "paths.results_root": (raw.get("paths", {}) or {}).get("results_root"),
        "paths.train_csv": (raw.get("paths", {}) or {}).get("train_csv"),
        "paths.val_csv": (raw.get("paths", {}) or {}).get("val_csv"),
        "paths.eval_csv": (raw.get("paths", {}) or {}).get("eval_csv"),
        "dataset.material_ids": (raw.get("dataset", {}) or {}).get("material_ids"),
        "split.group_column": (raw.get("split", {}) or {}).get("group_column"),
        "experiment.experiment_name": (raw.get("experiment", {}) or {}).get("experiment_name"),
        "experiment.seed": (raw.get("experiment", {}) or {}).get("seed"),
        "train.methods": (raw.get("train", {}) or {}).get("methods"),
        "train.sparse_rates": (raw.get("train", {}) or {}).get("sparse_rates"),
        "train.sparse_seeds": (raw.get("train", {}) or {}).get("sparse_seeds"),
        "baseline.d0_m": (raw.get("baseline", {}) or {}).get("d0_m"),
        "evaluation.metrics": (raw.get("evaluation", {}) or {}).get("metrics"),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Config file is missing required keys in {path}:\n  - " + "\n  - ".join(sorted(missing)))


def save_resolved_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    raw = config.get("_full_config", config)
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
