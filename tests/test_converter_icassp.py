from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wallpath_pi.data.dataset import WallPathManifest

CONVERTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "converters" / "convert_icassp2025_indoor.py"

PIL_Image = pytest.importorskip("PIL.Image")


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_icassp2025_indoor", CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Transmitter pixel coordinate stored in the Positions CSV columns. The official
# convention is tx_x = row["Y"], tx_y = row["X"], so tx_xy must equal [Y, X].
_POS_X = 4  # 'X' column value (becomes tx_y)
_POS_Y = 7  # 'Y' column value (becomes tx_x)
_WALL_COL = 8


def _make_fake_icassp(raw_root: Path, task: int = 1, n: int = 1, h: int = 16, w: int = 16) -> Path:
    """Build a tiny but official-format Task-1 raw dataset under ``raw_root``."""
    in_dir = raw_root / "Inputs" / f"Task_{task}_ICASSP"
    out_dir = raw_root / "Outputs" / f"Task_{task}_ICASSP"
    pos_dir = raw_root / "Positions"
    bld_dir = raw_root / "Building_Details"
    for d in (in_dir, out_dir, pos_dir, bld_dir):
        d.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        building, ant, freq = 1, 1, 1
        # RGB input: free space is 0; a single non-zero wall column.
        reflectance = np.zeros((h, w), dtype=np.uint8)
        transmittance = np.zeros((h, w), dtype=np.uint8)
        reflectance[:, _WALL_COL] = 120
        transmittance[:, _WALL_COL] = 90
        distance = np.clip(np.hypot(xx - _POS_Y, yy - _POS_X), 0, 255).astype(np.uint8)
        rgb = np.stack([reflectance, transmittance, distance], axis=-1)
        stem = f"B{building}_Ant{ant}_f{freq}_S{i}"
        PIL_Image.fromarray(rgb, mode="RGB").save(in_dir / f"{stem}.png")

        # Grayscale path-loss output.
        path_loss = np.clip(30 + distance.astype(np.int32) // 2, 0, 255).astype(np.uint8)
        PIL_Image.fromarray(path_loss, mode="L").save(out_dir / f"{stem}.png")

    # One Positions CSV per (building, antenna, frequency); row index == sample S#.
    pd.DataFrame({"X": [_POS_X] * n, "Y": [_POS_Y] * n}).to_csv(
        pos_dir / "Positions_B1_Ant1_f1.csv", index=False
    )
    # Building details (W, H).
    pd.DataFrame({"W": [w], "H": [h]}).to_csv(bld_dir / "B1_Details.csv", index=False)
    return raw_root


def test_convert_icassp_task1_writes_and_loads(tmp_path):
    mod = _load_converter()
    raw = _make_fake_icassp(tmp_path / "raw", task=1, n=1)
    out = tmp_path / "out"

    summary = mod.convert_dataset(raw, out, task=1, limit=1, verbose=False)
    assert summary["num_converted"] == 1
    assert summary["task"] == 1

    manifest_path = out / "manifest.csv"
    assert manifest_path.exists()
    cols = list(pd.read_csv(manifest_path).columns)
    assert cols == list(mod.MANIFEST_COLUMNS)

    npz_path = out / "scenes" / "B1_Ant1_f1_S0.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        for key in (
            "path_loss", "valid_mask", "wall_mask", "material_map", "tx_xy",
            "frequency_hz", "resolution_m", "reflectance", "transmittance",
            "distance_img", "scene_id", "sample_id", "antenna_id", "task_id",
            "building_id", "frequency_id",
        ):
            assert key in data.files, f"missing NPZ key: {key}"

        reflectance = data["reflectance"]
        transmittance = data["transmittance"]
        wall_mask = data["wall_mask"].astype(bool)
        expected_wall = (reflectance != 0) | (transmittance != 0)
        assert np.array_equal(wall_mask, expected_wall)
        # Only the wall column is non-zero.
        assert wall_mask[:, _WALL_COL].all()
        assert wall_mask.sum() == reflectance.shape[0]

        # tx_xy = [Y, X]; frequency f1 -> 868 MHz; resolution 0.25 m.
        assert data["tx_xy"].tolist() == [float(_POS_Y), float(_POS_X)]
        assert float(data["frequency_hz"]) == 868.0e6
        assert abs(float(data["resolution_m"]) - 0.25) < 1e-6
        # valid_mask is finite-path-loss only (no automatic wall removal).
        assert data["valid_mask"].astype(bool).all()


def test_convert_icassp_manifest_loads_through_manifest(tmp_path):
    mod = _load_converter()
    raw = _make_fake_icassp(tmp_path / "raw", task=1, n=2)
    out = tmp_path / "out"
    mod.convert_dataset(raw, out, task=1, verbose=False)

    manifest = WallPathManifest(out / "manifest.csv", out)
    samples = list(manifest.iter_samples())
    assert len(samples) == 2
    s = samples[0]
    assert s.scene_id == "B1"
    assert s.valid_mask.shape == s.path_loss.shape
    assert bool((s.material_map[:, _WALL_COL] > 0).all())
    assert s.frequency_hz == 868.0e6


def test_convert_icassp_dry_run_writes_nothing(tmp_path):
    mod = _load_converter()
    raw = _make_fake_icassp(tmp_path / "raw", task=1, n=2)
    out = tmp_path / "out"
    summary = mod.convert_dataset(raw, out, task=1, dry_run=True, verbose=False)
    assert summary["num_samples"] == 2
    assert summary["num_converted"] == 0
    assert summary["dry_run"] is True
    assert not out.exists()


def test_convert_icassp_missing_output_raises(tmp_path):
    mod = _load_converter()
    raw = _make_fake_icassp(tmp_path / "raw", task=1, n=1)
    # Remove the only output image so the input is unpaired.
    (raw / "Outputs" / "Task_1_ICASSP" / "B1_Ant1_f1_S0.png").unlink()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        mod.convert_dataset(raw, out, task=1, verbose=False)


def test_convert_icassp_cli_parser_defaults():
    mod = _load_converter()
    args = mod.build_parser().parse_args(["--raw-root", "data/icassp2025_indoor_raw"])
    assert args.task == 1
    assert abs(args.resolution_m - 0.25) < 1e-6
    assert args.dry_run is False


# load_tx_xy() boundary-tolerance edge case
def _tx_sample(mod, raw_root, *, x, y, s=0, building=1, ant=1, freq=1):
    """Write a Positions CSV whose row ``s`` holds (X=x, Y=y); return the sample.

    The official convention is tx_x = Y and tx_y = X, so ``load_tx_xy`` must
    return ``[y, x]`` for an in-bounds row.
    """
    pos_dir = raw_root / "Positions"
    pos_dir.mkdir(parents=True, exist_ok=True)
    xs = [0.0] * s + [float(x)]
    ys = [0.0] * s + [float(y)]
    pd.DataFrame({"X": xs, "Y": ys}).to_csv(
        pos_dir / f"Positions_B{building}_Ant{ant}_f{freq}.csv", index=False
    )
    return mod.IcasspSample(
        sample_id=f"B{building}_Ant{ant}_f{freq}_S{s}",
        scene_id=f"B{building}",
        task_id=1,
        building_id=building,
        antenna_id=ant,
        frequency_id=freq,
        sample_index=s,
        input_path=raw_root / "in.png",
        output_path=raw_root / "out.png",
    )


def test_load_tx_xy_valid_unchanged_preserves_yx_convention(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # X=4 -> tx_y, Y=7 -> tx_x; both well inside a 16x16 map.
    sample = _tx_sample(mod, raw, x=4, y=7)
    tx = mod.load_tx_xy(raw, sample, (16, 16))  # shape = (height, width)
    # Official convention preserved and value unchanged: tx_xy == [Y, X].
    assert tx.tolist() == [7.0, 4.0]


def test_load_tx_xy_upper_edge_clipped_one_pixel(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # Reproduces the real failure: width=400, height=132, tx_y == height.
    # tx_x = Y = 135 (valid); tx_y = X = 132 == height -> clipped to 131.
    sample = _tx_sample(mod, raw, x=132, y=135)
    tx = mod.load_tx_xy(raw, sample, (132, 400))
    assert tx.tolist() == [135.0, 131.0]


def test_load_tx_xy_lower_edge_clipped_one_pixel(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # tx_x = Y = -1 -> clipped to 0; tx_y = X = 5 (valid).
    sample = _tx_sample(mod, raw, x=5, y=-1)
    tx = mod.load_tx_xy(raw, sample, (10, 20))
    assert tx.tolist() == [0.0, 5.0]


def test_load_tx_xy_far_outside_still_raises(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # tx_y = X = 16 on a height-10 map is 7 px past the last index (9),
    # beyond the 5 px tolerance -> error (never silently clipped).
    sample = _tx_sample(mod, raw, x=16, y=5)
    with pytest.raises(ValueError, match="lies outside"):
        mod.load_tx_xy(raw, sample, (10, 20))


def test_load_tx_xy_far_negative_still_raises(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # tx_x = Y = -6 is 6 px before 0 -> beyond the 5 px tolerance, not clipped.
    sample = _tx_sample(mod, raw, x=5, y=-6)
    with pytest.raises(ValueError, match="lies outside"):
        mod.load_tx_xy(raw, sample, (10, 20))


def test_clip_tx_to_bounds_boundary_tolerance():
    mod = _load_converter()
    clip = mod._clip_tx_to_bounds
    # The default boundary tolerance is 5 px (covers every audited Task 2/3 case).
    assert mod._TX_BOUNDARY_TOL == 5.0
    # In-bounds values are returned unchanged.
    assert clip(5.0, 20) == 5.0
    assert clip(0.0, 20) == 0.0
    assert clip(19.0, 20) == 19.0
    # Up to 5 px past an edge snaps onto the nearest valid index.
    assert clip(20.0, 20) == 19.0   # 1 px
    assert clip(24.0, 20) == 19.0   # 5 px
    assert clip(-1.0, 20) == 0.0
    assert clip(-5.0, 20) == 0.0    # 5 px
    # Just beyond the 5 px tolerance is left unchanged (the caller rejects it).
    assert clip(24.001, 20) == 24.001
    assert clip(-5.001, 20) == -5.001
    assert clip(30.0, 20) == 30.0
    # The tolerance is parameterizable: with tol=1, only a 1 px overflow snaps.
    assert clip(20.0, 20, 1.0) == 19.0
    assert clip(21.0, 20, 1.0) == 21.0  # 2 px > 1 px tol -> unchanged


def test_load_tx_xy_five_pixels_outside_clipped(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    # Worst audited overflow is 5 px: tx_y = X = 14 on a height-10 map is exactly
    # 5 px past the last index (9) -> snapped to 9; tx_x = Y = 5 stays.
    sample = _tx_sample(mod, raw, x=14, y=5)
    tx = mod.load_tx_xy(raw, sample, (10, 20))
    assert tx.tolist() == [5.0, 9.0]


def test_load_tx_xy_respects_custom_tolerance(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "raw"
    sample = _tx_sample(mod, raw, x=14, y=5)  # 5 px past the bottom edge
    # The default 5 px tolerance snaps it; a stricter 1 px tolerance rejects it.
    assert mod.load_tx_xy(raw, sample, (10, 20), tol=5.0).tolist() == [5.0, 9.0]
    with pytest.raises(ValueError, match="lies outside"):
        mod.load_tx_xy(raw, sample, (10, 20), tol=1.0)


def test_cli_tx_boundary_tol_default_and_override():
    mod = _load_converter()
    args = mod.build_parser().parse_args(["--raw-root", "data/icassp2025_indoor_raw"])
    assert args.tx_boundary_tol == 5.0
    args2 = mod.build_parser().parse_args(["--raw-root", "x", "--tx-boundary-tol", "1"])
    assert args2.tx_boundary_tol == 1.0


def _make_icassp_with_tx(raw_root, task, pos_X, pos_Y, h=16, w=20, building=1, ant=1, freq=2):
    """Official-format raw layout with caller-specified Tx columns (X, Y).

    tx_x = Y column, tx_y = X column. Image is (h, w) so width=w, height=h.
    """
    in_dir = raw_root / "Inputs" / f"Task_{task}_ICASSP"
    out_dir = raw_root / "Outputs" / f"Task_{task}_ICASSP"
    pos_dir = raw_root / "Positions"
    for d in (in_dir, out_dir, pos_dir):
        d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w]
    distance = np.clip(np.hypot(xx, yy), 0, 255).astype(np.uint8)
    rgb = np.stack([np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), distance], axis=-1)
    gray = np.clip(30 + distance // 2, 0, 255).astype(np.uint8)
    for i in range(len(pos_X)):
        stem = f"B{building}_Ant{ant}_f{freq}_S{i}"
        PIL_Image.fromarray(rgb, mode="RGB").save(in_dir / f"{stem}.png")
        PIL_Image.fromarray(gray, mode="L").save(out_dir / f"{stem}.png")
    pd.DataFrame({"X": list(pos_X), "Y": list(pos_Y)}).to_csv(
        pos_dir / f"Positions_B{building}_Ant{ant}_f{freq}.csv", index=False
    )
    return raw_root


def test_convert_dataset_reports_and_applies_tx_clip(tmp_path):
    mod = _load_converter()
    # S0 in-bounds; S1 has tx_x = Y = -3 (3 px left of the raster) -> snapped.
    raw = _make_icassp_with_tx(tmp_path / "raw", task=2, pos_X=[8, 8], pos_Y=[8, -3], h=16, w=20)
    out = tmp_path / "out"
    summary = mod.convert_dataset(raw, out, task=2, overwrite=True, verbose=False)

    assert summary["tx_boundary_tol"] == 5.0
    assert summary["tx_clipped"] == 1
    assert summary["tx_clip_samples"] == ["B1_Ant1_f2_S1"]

    # In-bounds sample unchanged (tx_xy = [Y, X]); clipped sample snapped to edge.
    with np.load(out / "scenes" / "B1_Ant1_f2_S0.npz", allow_pickle=False) as d0:
        assert d0["tx_xy"].tolist() == [8.0, 8.0]
    with np.load(out / "scenes" / "B1_Ant1_f2_S1.npz", allow_pickle=False) as d1:
        assert d1["tx_xy"].tolist() == [0.0, 8.0]  # tx_x snapped from -3 to 0


def test_convert_dataset_strict_tolerance_rejects_five_px(tmp_path):
    mod = _load_converter()
    # tx_x = Y = -5 is 5 px left of the raster.
    raw = _make_icassp_with_tx(tmp_path / "raw", task=2, pos_X=[8], pos_Y=[-5], h=16, w=20)
    # The default 5 px tolerance snaps and converts it...
    summary = mod.convert_dataset(raw, tmp_path / "out", task=2, overwrite=True, verbose=False)
    assert summary["tx_clipped"] == 1
    # ...but a stricter 1 px tolerance rejects the same 5 px overflow.
    with pytest.raises(ValueError, match="lies outside"):
        mod.convert_dataset(raw, tmp_path / "out_strict", task=2, tx_boundary_tol=1.0, overwrite=True, verbose=False)


def test_convert_task1_in_bounds_tx_unchanged_backward_compatible(tmp_path):
    mod = _load_converter()
    # Task 1 sample with an in-bounds Tx (the default _make_fake_icassp tx).
    raw = _make_fake_icassp(tmp_path / "raw", task=1, n=1)
    out = tmp_path / "out"
    summary = mod.convert_dataset(raw, out, task=1, verbose=False)
    assert summary["tx_clipped"] == 0  # nothing clipped -> byte-identical to before
    with np.load(out / "scenes" / "B1_Ant1_f1_S0.npz", allow_pickle=False) as d:
        assert d["tx_xy"].tolist() == [float(_POS_Y), float(_POS_X)]
