from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.training.pipeline import _train_models_for_rate_seed
from wallpath_pi.utils.config import load_config

from scripts.analysis.evaluate_transfer import (
    discover_source_models,
    evaluate_transfer,
    model_filename,
    parse_model_filename,
    resolve_feature_names,
)

RATE = 0.05
SEED = 7
SMALL_PARAMS = {
    "random_forest": {"n_estimators": 6, "max_depth": 5, "min_samples_leaf": 1, "n_jobs": 1},
    "extra_trees": {"n_estimators": 6, "max_depth": 5, "min_samples_leaf": 1, "n_jobs": 1},
}


def test_parse_and_filename_roundtrip():
    assert parse_model_filename("direct_rf_all_features_rate_0p005_seed_11") == ("direct_rf_all_features", 0.005, 11)
    assert parse_model_filename("wallpath_extra_rate_0p01_seed_7") == ("wallpath_extra", 0.01, 7)
    assert parse_model_filename("no_separators_here") is None
    # Filename builder is the exact inverse used by the training pipeline.
    assert model_filename("direct_rf_sparse_anchor", 0.05, 7) == "direct_rf_sparse_anchor_rate_0p05_seed_7.joblib"
    method, rate, seed = parse_model_filename(Path(model_filename("wallpath_extra", 0.005, 11)).stem)
    assert (method, round(rate, 9), seed) == ("wallpath_extra", 0.005, 11)


def test_discover_source_models(tmp_path):
    md = tmp_path / "models"
    md.mkdir()
    joblib.dump({"model": None}, md / "direct_rf_geometry_rate_0p005_seed_11.joblib")
    joblib.dump({"model": None}, md / "wallpath_extra_rate_0p01_seed_11.joblib")
    (md / "unparseable.joblib").write_bytes(b"x")  # no _rate_/_seed_ -> ignored
    index = discover_source_models(md)
    assert ("direct_rf_geometry", 0.005, 11) in index
    assert ("wallpath_extra", 0.01, 11) in index
    assert len(index) == 2


def test_resolve_feature_names_prefers_saved_then_infers():
    full = [
        "x_norm", "y_norm", "distance_m", "fspl_db",
        "multi_wall_pred_db", "anchor_residual_idw", "anchor_dist_px_nn_1", "anchor_density",
    ]
    # Saved feature names always win.
    assert resolve_feature_names(["distance_m", "fspl_db"], "wallpath_extra", full) == ["distance_m", "fspl_db"]
    # Geometry control inference drops baseline-prediction and residual features.
    geo = resolve_feature_names(None, "direct_rf_geometry", full)
    assert "multi_wall_pred_db" not in geo
    assert "anchor_residual_idw" not in geo
    assert "distance_m" in geo
    # All-feature / residual learners default to the full table.
    assert resolve_feature_names(None, "wallpath_extra", full) == full


def _make_source_run(tmp_path: Path, methods) -> Path:
    cfg = load_config(Path("configs/config.yaml"))
    synth = dict(cfg["synthetic"])
    synth.update({"num_scenes": 4, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "source_data"
    generate_synthetic_dataset(data_root, synth)
    make_scene_disjoint_split(
        input_csv=data_root / "manifest.csv",
        train_out=data_root / "train_split.csv",
        val_out=data_root / "val_split.csv",
        meta_out=data_root / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.34,
        seed=7,
    )
    cfg["material_ids"] = [1, 2, 3]
    cfg["min_anchors"] = 4
    cfg["max_train_points_per_sample"] = 150
    cfg["model_params"] = SMALL_PARAMS
    cfg["methods"] = list(methods)
    train_samples = list(WallPathManifest(data_root / "train_split.csv", data_root).iter_samples())
    models, fnbm = _train_models_for_rate_seed(train_samples, cfg, RATE, SEED)

    source_run = tmp_path / "source_run"
    (source_run / "models").mkdir(parents=True)
    for method, model in models.items():
        joblib.dump(
            {"model": model, "feature_names": fnbm[method], "method": method, "sparse_rate": RATE, "sparse_seed": SEED},
            source_run / "models" / model_filename(method, RATE, SEED),
        )
    # Nested resolved-config snapshot so the transfer run can label train_source.
    (source_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg["_full_config"], sort_keys=False), encoding="utf-8"
    )
    return source_run


def _make_target_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    target_data = tmp_path / "target_data"
    synth = dict(raw["synthetic"])
    synth.update({"num_scenes": 3, "samples_per_scene": 1, "image_size": 24, "seed": 999, "overwrite": True})
    generate_synthetic_dataset(target_data, synth)
    make_scene_disjoint_split(
        input_csv=target_data / "manifest.csv",
        train_out=target_data / "train_split.csv",
        val_out=target_data / "val_split.csv",
        meta_out=target_data / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.34,
        seed=3,
    )
    raw["paths"]["data_root"] = str(target_data)
    raw["paths"]["eval_csv"] = "val_split.csv"
    raw["experiment"]["experiment_name"] = "target_task_transfer"
    raw["dataset"]["cache_features"] = False
    raw["train"]["min_anchors"] = 4
    raw["train"]["max_train_points_per_sample"] = 150
    raw["train"]["sparse_rates"] = [RATE]
    raw["train"]["sparse_seeds"] = [SEED]
    cfg_path = tmp_path / "target_config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_transfer_end_to_end(tmp_path):
    source_run = _make_source_run(tmp_path, ["direct_rf_geometry", "direct_rf_all_features", "wallpath_extra"])
    target_config = _make_target_config(tmp_path)
    out_dir = tmp_path / "out"

    evaluate_transfer(source_run=source_run, target_config=target_config, out_dir=out_dir)

    final = pd.read_csv(out_dir / "final_evaluation_results.csv")
    methods = set(final["method"])
    # Non-trained baselines recomputed on the target.
    assert {"log_distance", "multi_wall", "idw", "multi_wall_residual_idw"}.issubset(methods)
    # Frozen source models transferred.
    assert {"direct_rf_geometry", "direct_rf_all_features", "wallpath_extra"}.issubset(methods)
    # Clear transfer labelling.
    assert bool((final["retrained"] == False).all())
    assert (final["train_source"] == "wallpath_pi_synthetic").all()
    assert (final["target_dataset"] == "target_task_transfer").all()
    assert np.isfinite(final["rmse"]).all()

    assert (out_dir / "per_sample_metrics.csv").exists()
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["extra"]["retrained"] is False
    assert summary["extra"]["transfer_evaluation"] is True
    assert summary["extra"]["train_source"] == "wallpath_pi_synthetic"
    assert summary["extra"]["target_dataset"] == "target_task_transfer"


def test_transfer_skips_missing_model_with_warning(tmp_path, capsys):
    source_run = _make_source_run(tmp_path, ["direct_rf_geometry", "wallpath_extra"])
    target_config = _make_target_config(tmp_path)
    out_dir = tmp_path / "out2"

    # Request a method that was never trained in the source run.
    evaluate_transfer(
        source_run=source_run,
        target_config=target_config,
        out_dir=out_dir,
        methods=["multi_wall", "direct_rf_sparse_anchor", "wallpath_extra"],
    )

    final = pd.read_csv(out_dir / "final_evaluation_results.csv")
    methods = set(final["method"])
    assert {"multi_wall", "wallpath_extra"}.issubset(methods)
    assert "direct_rf_sparse_anchor" not in methods  # skipped, not crashed
    err = capsys.readouterr().err
    assert "no source model for method=direct_rf_sparse_anchor" in err


def test_transfer_does_not_write_npz(tmp_path):
    source_run = _make_source_run(tmp_path, ["wallpath_extra"])
    target_config = _make_target_config(tmp_path)
    out_dir = tmp_path / "out3"
    evaluate_transfer(source_run=source_run, target_config=target_config, out_dir=out_dir)
    assert list(out_dir.glob("*.npz")) == []
