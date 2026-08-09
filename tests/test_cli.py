"""CLI parser smoke tests for the user-facing scripts.

These verify that a new user following the README can invoke each script
without guessing: ``--help`` works, ``--config`` is accepted with a sane
default, and required arguments are enforced. Heavy pipeline functions are
never executed here; only argument parsing and boundary error messages are
exercised.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(rel_path: str):
    """Load a standalone script by file path with a unique module name."""
    path = REPO_ROOT / rel_path
    name = "cli_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONFIG_SCRIPTS = [
    "scripts/train.py",
    "scripts/generate_synthetic_data.py",
    "scripts/make_internal_split.py",
    "scripts/verify_repo_ready.py",
]

ALL_SCRIPTS = CONFIG_SCRIPTS + [
    "scripts/evaluate.py",
    "scripts/analysis/bootstrap_metric_ci.py",
]


@pytest.mark.parametrize("rel_path", ALL_SCRIPTS)
def test_help_exits_zero(rel_path: str):
    module = _load_script(rel_path)
    parser = module.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize("rel_path", CONFIG_SCRIPTS)
def test_config_default_and_override(rel_path: str):
    module = _load_script(rel_path)
    parser = module.build_parser()

    defaults = parser.parse_args([])
    assert Path(defaults.config) == Path("configs/config.yaml")

    custom = parser.parse_args(["--config", "other/cfg.yaml"])
    assert Path(custom.config) == Path("other/cfg.yaml")


def test_train_parses_sparse_overrides():
    module = _load_script("scripts/train.py")
    args = module.build_parser().parse_args(["--sparse_rates", "0.02,0.05", "--sparse_seeds", "11,23"])
    assert args.sparse_rates == "0.02,0.05"
    assert args.sparse_seeds == "11,23"


def test_evaluate_requires_run_dir():
    module = _load_script("scripts/evaluate.py")
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --run_dir is required
    args = parser.parse_args(["--run_dir", "results/run_1", "--csv", "val_split.csv"])
    assert Path(args.run_dir) == Path("results/run_1")
    assert args.csv == "val_split.csv"


def test_make_internal_split_hyphenated_options():
    module = _load_script("scripts/make_internal_split.py")
    args = module.build_parser().parse_args(["--group-column", "scene_id", "--val-ratio", "0.25", "--seed", "7"])
    assert args.group_column == "scene_id"
    assert args.val_ratio == pytest.approx(0.25)
    assert args.seed == 7


def test_bootstrap_metric_ci_defaults():
    module = _load_script("scripts/analysis/bootstrap_metric_ci.py")
    args = module.build_parser().parse_args(["--run_dir", "results/run_1"])
    assert args.metric == "rmse"
    assert args.confidence == pytest.approx(0.95)


def test_load_config_missing_file_message():
    from wallpath_pi.utils.config import load_config

    with pytest.raises(FileNotFoundError) as excinfo:
        load_config(REPO_ROOT / "configs" / "does_not_exist.yaml")
    assert "does_not_exist.yaml" in str(excinfo.value)


def test_evaluate_saved_run_missing_dir_message(tmp_path: Path):
    from wallpath_pi.training.pipeline import evaluate_saved_run

    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate_saved_run(tmp_path / "no_such_run")
    assert "Run directory not found" in str(excinfo.value)


def test_evaluate_saved_run_missing_summary_message(tmp_path: Path):
    from wallpath_pi.training.pipeline import evaluate_saved_run

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate_saved_run(run_dir)
    assert "run_summary.json" in str(excinfo.value)
