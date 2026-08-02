"""Tests for the external point-wise 3.5 GHz measured-data evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "evaluate_external_3p5ghz.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_external_3p5ghz", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mini_csv(n: int, seed: int, *, drop_elevator: bool = False, drop_pl: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.uniform(1.0, 40.0, n)
    rows = {
        "Coord.": [f"C-{i}" for i in range(n)],
        "Distance (m)": d,
        "Num_brick_wall": rng.integers(0, 3, n),
        "Num_wood_wall": rng.integers(0, 2, n),
        "Num_glass_wall": rng.integers(0, 2, n),
        "Num_drywall": rng.integers(0, 2, n),
        "Num_column": rng.integers(0, 2, n),
        "P_rx (dBm)": -40 - 20 * np.log10(d),
        "PL (dB)": 40 + 20 * np.log10(d) + rng.normal(0, 1, n),
    }
    if not drop_elevator:
        rows["Elevator"] = rng.integers(0, 1, n)
    if drop_pl:
        rows.pop("PL (dB)")
    return pd.DataFrame(rows)


def _make_dataset(root: Path, *, drop_elevator: bool = False) -> Path:
    pl = root / "PL_Data"
    pl.mkdir(parents=True)
    for i, sc in enumerate(("Comms", "Library", "SSE")):
        for j, cf in enumerate(("C1", "C2")):
            df = _mini_csv(14, seed=i * 2 + j, drop_elevator=drop_elevator)
            if sc == "Comms" and cf == "C2":
                df.loc[0, "Coord."] = "C-36"
                df.loc[0, "PL (dB)"] = -60.0
            df.to_csv(pl / f"PL_{sc}_{cf}.csv", index=False)
    return root


def test_load_and_parse(tmp_path):
    mod = _load()
    assert mod.parse_scenario_config("PL_Comms_C1.csv") == ("Comms", "C1")
    df, report = mod.load_external_data(_make_dataset(tmp_path))
    assert set(df["scenario"]) == {"Comms", "Library", "SSE"}
    assert set(df["config"]) == {"C1", "C2"}
    assert {"distance_m", "pl_db", "source_file"}.issubset(df.columns)
    assert report["rows_before_cleaning"] >= report["rows_after_cleaning"]


def test_anomaly_removed(tmp_path):
    mod = _load()
    df, report = mod.load_external_data(_make_dataset(tmp_path))
    bad = df[(df["source_file"] == "PL_Comms_C2.csv") & (df["coord"] == "C-36")]
    assert bad.empty
    assert any(r.get("reason") == "known_physical_anomaly_removed" for r in report["removed"])


def test_random_split_writes_three_files(tmp_path):
    mod = _load()
    out = tmp_path / "out"
    args = mod.build_arg_parser().parse_args([
        "--data-root", str(_make_dataset(tmp_path)), "--out-dir", str(out), "--split", "random"])
    mod.run(args)
    for fn in ("final_evaluation_results.csv", "per_sample_metrics.csv", "run_summary.json"):
        assert (out / fn).exists()


@pytest.mark.parametrize("split", ["leave_scenario_out", "leave_config_out", "random"])
def test_primary_split_matches_requested_split(tmp_path, split):
    mod = _load()
    out = tmp_path / "out"
    args = mod.build_arg_parser().parse_args([
        "--data-root", str(_make_dataset(tmp_path)), "--out-dir", str(out), "--split", split])
    mod.run(args)
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_split"] == split


def test_leave_scenario_out_three_folds(tmp_path):
    mod = _load()
    df, _ = mod.load_external_data(_make_dataset(tmp_path))
    res, _, _ = mod.evaluate_external(df, split="leave_scenario_out", heldout=[], seed=11,
                                      test_size=0.2, include_rf_residual=False)
    assert set(res["fold"]) == {"holdout_Comms", "holdout_Library", "holdout_SSE"}


def test_wallpath_residual_equals_baseline_plus_residual(tmp_path):
    mod = _load()
    df, _ = mod.load_external_data(_make_dataset(tmp_path))
    X, names, _ = mod.build_features(df, "random")
    y = df["pl_db"].to_numpy()
    tr = np.arange(0, len(df) - 10)
    te = np.arange(len(df) - 10, len(df))
    preds = mod.predict_methods(X[tr], y[tr], X[te], names, seed=11, include_rf_residual=False)
    mwc = mod._mw_columns(names)
    mw = Ridge(alpha=1.0).fit(X[tr][:, mwc], y[tr])
    base = mw.predict(X[te][:, mwc])
    extra = ExtraTreesRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=1, random_state=11)
    extra.fit(X[tr], y[tr] - mw.predict(X[tr][:, mwc]))
    np.testing.assert_allclose(preds["wallpath_residual_extra"], base + extra.predict(X[te]))


def test_metrics_correct():
    mod = _load()
    m = mod.compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0]))
    assert m["count"] == 3
    assert m["bias_db"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 0 + 4) / 3))


def test_missing_elevator_ok(tmp_path):
    mod = _load()
    df, _ = mod.load_external_data(_make_dataset(tmp_path, drop_elevator=True))
    assert (df["elevator"] == 0.0).all()


def test_missing_pl_fails(tmp_path):
    mod = _load()

    data_root = _make_dataset(tmp_path)
    pl_dir = data_root / "PL_Data"

    _mini_csv(
        10,
        1,
        drop_pl=True,
    ).to_csv(
        pl_dir / "PL_Comms_C1.csv",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        mod.load_external_data(data_root)

def test_anchor_mask_deterministic_and_monotonic():
    mod = _load()
    m1 = mod.select_anchor_mask(100, mode="fraction", fraction=0.05, count=0, seed=11)
    m2 = mod.select_anchor_mask(100, mode="fraction", fraction=0.05, count=0, seed=11)
    np.testing.assert_array_equal(m1, m2)
    counts = [mod.select_anchor_mask(100, mode="fraction", fraction=f, count=0, seed=11).sum()
              for f in (0.01, 0.05, 0.10)]
    assert counts[0] <= counts[1] <= counts[2]


def test_fewshot_folds_and_excludes_anchors(tmp_path):
    mod = _load()
    df, _ = mod.load_external_data(_make_dataset(tmp_path))
    res, samp, _, _ = mod.evaluate_fewshot(
        df,
        heldout=[],
        settings=[("fraction", 0.10, 0)],
        anchor_seeds=[11],
        model_seed=11,
        include_rf_residual=False,
        include_direct=False,
    )
    assert set(res["fold"]) == {"holdout_Comms", "holdout_Library", "holdout_SSE"}
    # every evaluated sample is a non-anchor in the held-out scenario, anchors excluded
    assert (samp["is_anchor"] == False).all()  # noqa: E712
    assert (samp["scenario"] == samp["heldout_scenario"]).all()
    assert (res["test_count"] > 0).all() and (res["target_anchor_count"] > 0).all()


def test_fewshot_residual_uses_source_plus_anchors_only(tmp_path):
    mod = _load()
    df, _ = mod.load_external_data(_make_dataset(tmp_path))
    X, names, _ = mod.build_features(df, "leave_scenario_fewshot")
    y = df["pl_db"].to_numpy()
    scen = df["scenario"].to_numpy()
    t = np.where(scen == "Comms")[0]
    src = np.where(scen != "Comms")[0]
    a = t[mod.select_anchor_mask(t.size, mode="fraction", fraction=0.10, count=0, seed=11)]
    te = np.setdiff1d(t, a)
    preds, _ = mod.predict_fewshot(X[src], y[src], X[a], y[a], X[te], names, seed=11,
                                   include_rf_residual=False, include_direct=False)
    mwc = mod._mw_columns(names)
    mw = Ridge(alpha=1.0).fit(X[src][:, mwc], y[src])
    Xaug = np.vstack([X[src], X[a]]); yaug = np.concatenate([y[src], y[a]])
    extra = ExtraTreesRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=1, random_state=11)
    extra.fit(Xaug, yaug - mw.predict(Xaug[:, mwc]))
    np.testing.assert_allclose(preds["wallpath_fewshot_residual_extra"], mw.predict(X[te][:, mwc]) + extra.predict(X[te]))
    assert "scenario_Comms" not in names  # no held-out scenario one-hot


def test_fewshot_summary_and_both_modes(tmp_path):
    mod = _load()
    out = tmp_path / "out"
    args = mod.build_arg_parser().parse_args([
        "--data-root", str(_make_dataset(tmp_path)), "--out-dir", str(out), "--split", "leave_scenario_fewshot",
        "--anchor-fractions", "0.10", "--anchor-counts", "5", "--anchor-seeds", "11"])
    mod.run(args)
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "leave_scenario_fewshot"
    assert summary["primary_method"] == "wallpath_fewshot_residual_extra"
    assert summary["evaluation_excludes_target_anchors"] is True
    res = pd.read_csv(out / "final_evaluation_results.csv")
    assert set(res["anchor_mode"]) == {"fraction", "count"}
    assert "target_idw" in summary["skipped_methods"]  # no numeric coords in synthetic data
