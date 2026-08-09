from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from wallpath_pi.utils.hashing import directory_file_hashes, file_hashes


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return x


def _clean_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in cfg.items():
        if k == "_full_config":
            continue
        out[k] = _jsonable(v)
    return out


def create_run_summary(
    experiment_name: str,
    output_dir: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    timing_info: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(config.get("repo_root", Path.cwd())).expanduser().resolve()
    payload: Dict[str, Any] = {
        "status": "completed",
        "experiment_name": str(experiment_name),
        "env": {
            "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "config": _clean_config(config),
        "metrics": {k: _jsonable(v) for k, v in metrics.items()},
        "artifacts": {"run_dir": str(output_dir)},
    }
    if artifacts:
        payload["artifacts"].update({k: _jsonable(v) for k, v in artifacts.items()})
    if dataset_info:
        payload["dataset_info"] = {k: _jsonable(v) for k, v in dataset_info.items()}
    if timing_info:
        payload["timing"] = {k: _jsonable(v) for k, v in timing_info.items()}
    if extra:
        payload["extra"] = {k: _jsonable(v) for k, v in extra.items()}

    try:
        cfg_json = json.dumps(payload["config"], sort_keys=True).encode("utf-8")
        import hashlib
        payload["config_fingerprint"] = {
            "resolved_config_md5": hashlib.md5(cfg_json).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(cfg_json).hexdigest(),
        }
    except Exception as e:
        payload["config_fingerprint_error"] = str(e)

    cfg_path = config.get("_config_path")
    if cfg_path and Path(cfg_path).exists():
        payload.setdefault("config_fingerprint", {})["config_path"] = str(Path(cfg_path).resolve())
        h = file_hashes(Path(cfg_path))
        payload["config_fingerprint"]["config_md5"] = h["md5"]
        payload["config_fingerprint"]["config_sha256"] = h["sha256"]

    payload["data_fingerprint"] = {}
    data_root = config.get("_resolved_data_root")
    if data_root:
        payload["data_fingerprint"]["data_root"] = str(data_root)
    for key in ["_resolved_train_csv", "_resolved_val_csv", "_resolved_eval_csv"]:
        p = config.get(key)
        if p and Path(p).exists():
            name = key.replace("_resolved_", "")
            payload["data_fingerprint"][name] = str(Path(p).resolve())
            h = file_hashes(Path(p))
            payload["data_fingerprint"][f"{name}_md5"] = h["md5"]
            payload["data_fingerprint"][f"{name}_sha256"] = h["sha256"]

    code_files = [
        repo_root / "src" / "wallpath_pi" / "training" / "pipeline.py",
        repo_root / "src" / "wallpath_pi" / "geometry" / "features.py",
        repo_root / "src" / "wallpath_pi" / "geometry" / "raster.py",
        repo_root / "src" / "wallpath_pi" / "baselines" / "propagation.py",
        repo_root / "src" / "wallpath_pi" / "baselines" / "idw.py",
        repo_root / "src" / "wallpath_pi" / "evaluation" / "metrics.py",
        repo_root / "scripts" / "train.py",
        repo_root / "scripts" / "evaluate.py",
    ]
    payload["code_fingerprint"] = {}
    for path in code_files:
        key = path.name.replace(".", "_")
        payload["code_fingerprint"][f"{key}_path"] = str(path)
        if path.exists():
            h = file_hashes(path)
            payload["code_fingerprint"][f"{key}_md5"] = h["md5"]
            payload["code_fingerprint"][f"{key}_sha256"] = h["sha256"]
        else:
            payload["code_fingerprint"][f"{key}_error"] = "file_not_found"
    payload["source_manifest"] = directory_file_hashes(repo_root / "src")

    resolved_yaml_path = output_dir / "resolved_config.yaml"
    raw_cfg = config.get("_full_config", config)
    resolved_yaml_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary_path, resolved_yaml_path
