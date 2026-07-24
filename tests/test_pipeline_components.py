from __future__ import annotations

from pathlib import Path

import numpy as np

from wallpath_pi.data.dataset import WallPathManifest, load_scene_npz
from wallpath_pi.data.splits import make_scene_disjoint_split
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.training.pipeline import _prepare_sample, _write_npz_payload
from wallpath_pi.utils.config import load_config


def test_prepare_sample_smoke(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    synth_cfg = dict(cfg["synthetic"])
    synth_cfg.update({"num_scenes": 3, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth_cfg)
    make_scene_disjoint_split(
        input_csv=data_root / "manifest.csv",
        train_out=data_root / "train_split.csv",
        val_out=data_root / "val_split.csv",
        meta_out=data_root / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.34,
        seed=7,
    )
    manifest = WallPathManifest(data_root / "train_split.csv", data_root)
    sample = manifest.sample(0)
    cfg["material_ids"] = [1, 2, 3]
    cfg["min_anchors"] = 4
    prep = _prepare_sample(sample, cfg, sparse_rate=0.05, sparse_seed=7)
    assert prep.feature_table.X.shape[0] == int(prep.feature_table.valid_mask.sum())
    assert prep.feature_table.X.shape[1] == len(prep.feature_table.feature_names)
    assert prep.multi_wall_pred.shape == sample.path_loss.shape


def _write_minimal_scene(path: Path, scene_id, sample_id, *, h: int = 6, w: int = 6) -> None:
    np.savez_compressed(
        path,
        path_loss=np.linspace(20.0, 80.0, h * w).reshape(h, w).astype(np.float32),
        valid_mask=np.ones((h, w), dtype=np.uint8),
        material_map=np.zeros((h, w), dtype=np.int16),
        tx_xy=np.asarray([1.0, 2.0], dtype=np.float32),
        frequency_hz=np.asarray(3.5e9, dtype=np.float64),
        resolution_m=np.asarray(0.25, dtype=np.float32),
        scene_id=np.asarray(scene_id),
        sample_id=np.asarray(sample_id),
    )


def test_load_scene_npz_without_pickle(tmp_path):
    # Unicode string metadata.
    p1 = tmp_path / "scene_u.npz"
    _write_minimal_scene(p1, "scene_42", "scene_42_tx_00")
    # NumPy must accept the file with pickling disabled at the loader level.
    raw = np.load(p1, allow_pickle=False)
    assert "path_loss" in raw
    s1 = load_scene_npz(p1)
    assert s1.scene_id == "scene_42"
    assert s1.sample_id == "scene_42_tx_00"
    assert s1.path_loss.shape == (6, 6)
    assert s1.tx_xy.tolist() == [1.0, 2.0]

    # Byte-string metadata must also decode robustly.
    p2 = tmp_path / "scene_b.npz"
    _write_minimal_scene(p2, np.asarray(b"scene_b", dtype="S7"), np.asarray(b"scene_b_tx_01", dtype="S13"))
    s2 = load_scene_npz(p2)
    assert s2.scene_id == "scene_b"
    assert s2.sample_id == "scene_b_tx_01"

    # Manifest row id fallback overrides npz value when present.
    s3 = load_scene_npz(p1, row={"scene_id": "from_row", "sample_id": "row_tx_09"})
    assert s3.scene_id == "from_row"
    assert s3.sample_id == "row_tx_09"


def test_eval_npz_loads_without_pickle(tmp_path):
    h = w = 4
    payload = {
        "predictions_by_method": {
            "fspl": [np.zeros((h, w), np.float32)],
            "wallpath_rf": [np.ones((h, w), np.float32)],
        },
        "targets": [np.full((h, w), 50.0, np.float32)],
        "valid_masks": [np.ones((h, w), np.uint8)],
        "wall_counts": [np.zeros((h, w), np.float32)],
        "los_masks": [np.ones((h, w), np.float32)],
        "scene_ids": ["scene_0"],
        "sample_ids": ["scene_0_tx_00"],
    }
    out = tmp_path / "eval_outputs.npz"
    _write_npz_payload(payload, out)
    data = np.load(out, allow_pickle=False)
    assert data["methods"].dtype.kind in {"U", "S"}
    assert data["scene_ids"].dtype.kind in {"U", "S"}
    assert data["sample_ids"].dtype.kind in {"U", "S"}
    assert [str(x) for x in data["methods"].tolist()] == ["fspl", "wallpath_rf"]
    assert [str(x) for x in data["scene_ids"].tolist()] == ["scene_0"]
    assert data["predictions"].shape == (2, 1, h, w)


def test_eval_npz_handles_variable_shapes(tmp_path):
    # Two samples from buildings with different HxW shapes must not crash export.
    shape_a = (4, 4)
    shape_b = (5, 3)
    payload = {
        "predictions_by_method": {
            "fspl": [np.zeros(shape_a, np.float32), np.zeros(shape_b, np.float32)],
            "wallpath_rf": [np.ones(shape_a, np.float32), np.ones(shape_b, np.float32)],
        },
        "targets": [np.full(shape_a, 50.0, np.float32), np.full(shape_b, 60.0, np.float32)],
        "valid_masks": [np.ones(shape_a, np.uint8), np.ones(shape_b, np.uint8)],
        "wall_counts": [np.zeros(shape_a, np.float32), np.zeros(shape_b, np.float32)],
        "los_masks": [np.ones(shape_a, np.float32), np.ones(shape_b, np.float32)],
        "scene_ids": ["scene_0", "scene_1"],
        "sample_ids": ["scene_0_tx_00", "scene_1_tx_00"],
    }
    out = tmp_path / "eval_outputs_var.npz"
    _write_npz_payload(payload, out)
    data = np.load(out, allow_pickle=True)
    assert bool(data["variable_shapes"])
    assert data["shapes"].tolist() == [list(shape_a), list(shape_b)]
    assert [str(x) for x in data["methods"].tolist()] == ["fspl", "wallpath_rf"]
    predictions = data["predictions"]
    assert predictions.shape == (2, 2)
    assert predictions[0, 0].shape == shape_a
    assert predictions[1, 1].shape == shape_b
    assert data["targets"][0].shape == shape_a
    assert data["targets"][1].shape == shape_b


def _make_split_sample(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    synth_cfg = dict(cfg["synthetic"])
    synth_cfg.update({"num_scenes": 2, "samples_per_scene": 1, "image_size": 24, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth_cfg)
    make_scene_disjoint_split(
        input_csv=data_root / "manifest.csv",
        train_out=data_root / "train_split.csv",
        val_out=data_root / "val_split.csv",
        meta_out=data_root / "split_meta.json",
        group_column="scene_id",
        val_ratio=0.5,
        seed=7,
    )
    manifest = WallPathManifest(data_root / "train_split.csv", data_root)
    sample = manifest.sample(0)
    cfg["material_ids"] = [1, 2, 3]
    cfg["min_anchors"] = 4
    return cfg, sample


def _cfg_with_flag(cfg, value):
    out = dict(cfg)
    out["feature_set"] = dict(cfg["feature_set"])
    out["feature_set"]["baseline_prediction_features"] = value
    return out


BASELINE_FEATURE_NAMES = {
    "fspl_pred_db",
    "log_distance_pred_db",
    "multi_wall_pred_db",
    "multi_wall_residual_idw_pred_db",
}


def test_baseline_prediction_features_included_when_enabled(tmp_path):
    cfg, sample = _make_split_sample(tmp_path)
    prep = _prepare_sample(sample, _cfg_with_flag(cfg, True), sparse_rate=0.1, sparse_seed=7)
    names = set(prep.feature_table.feature_names)
    assert BASELINE_FEATURE_NAMES <= names
    assert prep.feature_table.X.shape[1] == len(prep.feature_table.feature_names)


def test_baseline_prediction_features_excluded_when_disabled(tmp_path):
    cfg, sample = _make_split_sample(tmp_path)
    prep_off = _prepare_sample(sample, _cfg_with_flag(cfg, False), sparse_rate=0.1, sparse_seed=7)
    prep_on = _prepare_sample(sample, _cfg_with_flag(cfg, True), sparse_rate=0.1, sparse_seed=7)
    names_off = set(prep_off.feature_table.feature_names)
    names_on = set(prep_on.feature_table.feature_names)
    # Disabled excludes the baseline group; enabling only adds those exact names.
    assert not (BASELINE_FEATURE_NAMES & names_off)
    assert names_off < names_on
    assert names_on - names_off == BASELINE_FEATURE_NAMES


def test_direct_and_residual_models_train_on_baseline_features(tmp_path):
    from wallpath_pi.models.registry import make_regressor

    cfg, sample = _make_split_sample(tmp_path)
    prep = _prepare_sample(sample, _cfg_with_flag(cfg, True), sparse_rate=0.1, sparse_seed=7)
    table = prep.feature_table
    params = cfg.get("model_params", {})
    # Direct model predicts the target; both models consume the same matrix.
    direct = make_regressor("direct_rf", params=params, seed=0)
    direct.fit(table.X, table.y)
    # Residual model predicts target minus the multi-wall baseline.
    residual = make_regressor("wallpath_rf", params=params, seed=1)
    residual.fit(table.X, table.y - table.baseline_values)
    assert direct.predict(table.X).shape[0] == table.X.shape[0]
    assert residual.predict(table.X).shape[0] == table.X.shape[0]
