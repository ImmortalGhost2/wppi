from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def get_repo_root(start: Union[str, Path] = __file__) -> Path:
    """Walk up from ``start`` to the repository root (the dir holding src/ and configs/)."""
    p = Path(start).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def resolve_against_repo(value: Union[str, Path], repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve ``value`` to an absolute path, treating relative paths as repo-relative."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    root = Path(repo_root).resolve() if repo_root is not None else get_repo_root()
    return (root / p).resolve()


def resolve_data_root(data_root: Union[str, Path], repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the data root, with ``WALLPATH_DATA_ROOT`` taking precedence."""
    env = os.getenv("WALLPATH_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return resolve_against_repo(data_root, repo_root=repo_root)


def resolve_results_root(results_root: Union[str, Path], repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the results root, with ``WALLPATH_RESULTS_ROOT`` taking precedence."""
    env = os.getenv("WALLPATH_RESULTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return resolve_against_repo(results_root, repo_root=repo_root)


def resolve_cache_root(
    cache_root: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the base-map cache directory.

    Precedence: ``WALLPATH_CACHE_ROOT`` env var, then an explicit ``paths.cache_root``,
    then the default ``data/processed/cache/base_maps`` under the repository root.
    """
    env = os.getenv("WALLPATH_CACHE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if cache_root:
        return resolve_against_repo(cache_root, repo_root=repo_root)
    return resolve_against_repo("data/processed/cache/base_maps", repo_root=repo_root)


def resolve_csv_path(data_root: Union[str, Path], csv_value: Union[str, Path]) -> Path:
    """Resolve a manifest CSV path, treating relative values as data-root-relative."""
    p = Path(csv_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(data_root).expanduser().resolve() / p).resolve()


def get_next_run_dir(results_root: Union[str, Path], experiment_name: str) -> Path:
    """Create and return the next sequential ``run_N`` directory for an experiment."""
    model_dir = Path(results_root).expanduser().resolve() / str(experiment_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in model_dir.glob("run_*"):
        if d.is_dir():
            try:
                nums.append(int(d.name.split("_")[1]))
            except Exception:
                pass
    next_num = max(nums) + 1 if nums else 1
    out = model_dir / f"run_{next_num}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def latest_run_dir(results_root: Union[str, Path], experiment_name: str) -> Optional[Path]:
    """Return the most recently modified ``run_*`` directory, or None if none exist."""
    model_dir = Path(results_root).expanduser().resolve() / str(experiment_name)
    runs = [d for d in model_dir.glob("run_*") if d.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def get_data_root(data_root: Union[str, Path], repo_root: Optional[Union[str, Path]] = None) -> Path:
    """Backward-compatible alias for resolve_data_root."""
    return resolve_data_root(data_root, repo_root=repo_root)
