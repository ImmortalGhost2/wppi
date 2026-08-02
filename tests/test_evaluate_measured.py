from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "evaluate_measured_3p5ghz.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("evaluate_measured_3p5ghz", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_points(path: Path, n_scenarios: int = 4, per_scenario: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenarios):
        for k in range(per_scenario):
            dist = float(rng.uniform(1.0, 30.0))
            brick = int(rng.integers(0, 3))
            glass = int(rng.integers(0, 2))
            walls = brick + glass
            # Known generative model: log-distance plus per-wall attenuation + noise.
            pl = 32.0 + 21.0 * np.log10(dist) + 6.0 * brick + 3.0 * glass + rng.normal(0, 1.5)
            rows.append(
                {
                    "scenario": f"scn_{s}",
                    "campaign_id": s % 2,
                    "tx_id": f"tx{s}",
                    "rx_id": f"r{s}_{k}",
                    "measured_path_loss_db": pl,
                    "distance_m": dist,
                    "frequency_hz": 3.5e9,
                    "los_nlos": "LOS" if walls == 0 else "NLOS",
                    "brick_wall_count": brick,
                    "glass_wall_count": glass,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_cfg(mod, csv_path: Path, out_dir: Path):
    cfg = mod.load_eval_config(None)
    cfg["input_csv"] = str(csv_path)
    cfg["out_dir"] = str(out_dir)
    cfg["n_splits"] = 4
    # Keep forests small and fast for the test.
    cfg["model_params"].update({"n_estimators": 40, "max_depth": 10, "n_jobs": 1})
    return cfg


def test_measured_eval_writes_outputs_and_is_leakage_safe(tmp_path):
    mod = _load_eval()
    csv = _make_points(tmp_path / "measured_points.csv")
    out = tmp_path / "val"
    summary = mod.evaluate_measured(_base_cfg(mod, csv, out), write_outputs=True, verbose=False)

    assert (out / "measured_validation_metrics.csv").exists()
    assert (out / "measured_validation_per_fold.csv").exists()
    assert (out / "measured_validation_summary.json").exists()

    saved = json.loads((out / "measured_validation_summary.json").read_text())
    assert saved["group_column"] == "scenario"
    assert saved["n_groups"] == 4
    assert saved["n_splits"] == 4
    assert set(saved["methods"]) == set(mod.METHODS)

    per_fold = pd.read_csv(out / "measured_validation_per_fold.csv")
    # Four methods x four folds.
    assert len(per_fold) == len(mod.METHODS) * 4
    # Each grouped fold tests one scenario; folds partition all rows exactly once.
    assert int(per_fold[per_fold["method"] == "log_distance"]["n_test"].sum()) == saved["n_rows_used"]


def test_measured_eval_multiwall_beats_logdistance(tmp_path):
    mod = _load_eval()
    csv = _make_points(tmp_path / "measured_points.csv")
    out = tmp_path / "val"
    summary = mod.evaluate_measured(_base_cfg(mod, csv, out), write_outputs=False, verbose=False)
    overall = summary["overall"]
    # Walls drive attenuation in the generative model, so accounting for them helps.
    assert overall["multi_wall_point"]["rmse"] < overall["log_distance"]["rmse"]
    # Residual and direct forests should at least match the linear wall model.
    assert overall["residual_rf_point"]["rmse"] <= overall["log_distance"]["rmse"] + 1e-6
    assert np.isfinite(overall["direct_rf_point"]["rmse"])


def test_measured_eval_has_scenario_and_los_partitions(tmp_path):
    mod = _load_eval()
    csv = _make_points(tmp_path / "measured_points.csv")
    out = tmp_path / "val"
    mod.evaluate_measured(_base_cfg(mod, csv, out), write_outputs=True, verbose=False)
    metrics = pd.read_csv(out / "measured_validation_metrics.csv")
    partitions = set(metrics["partition"].unique())
    assert {"overall", "scenario", "los_nlos", "wall_region"} <= partitions
    # Scenario partition lists all four scenarios for each method.
    scen = metrics[(metrics["partition"] == "scenario") & (metrics["method"] == "residual_rf_point")]
    assert set(scen["group"]) == {"scn_0", "scn_1", "scn_2", "scn_3"}


def test_measured_eval_group_fallback_when_single_scenario(tmp_path):
    mod = _load_eval()
    # One scenario but two campaigns -> must fall back to campaign_id for grouping.
    rng = np.random.default_rng(1)
    rows = []
    for c in range(3):
        for k in range(30):
            dist = float(rng.uniform(1.0, 20.0))
            rows.append(
                {
                    "scenario": "only_scene",
                    "campaign_id": f"camp_{c}",
                    "rx_id": f"r{c}_{k}",
                    "measured_path_loss_db": 30 + 20 * np.log10(dist) + rng.normal(0, 1.0),
                    "distance_m": dist,
                }
            )
    csv = tmp_path / "measured_points.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = tmp_path / "val"
    cfg = _base_cfg(mod, csv, out)
    cfg["n_splits"] = 3
    summary = mod.evaluate_measured(cfg, write_outputs=False, verbose=False)
    assert summary["group_column"] == "campaign_id"
    assert summary["n_groups"] == 3
