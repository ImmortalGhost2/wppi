from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

CONVERTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "converters" / "convert_measured_3p5ghz.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_measured_3p5ghz", CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_raw(raw_root: Path):
    raw_root.mkdir(parents=True, exist_ok=True)
    # File 1: explicit LOS column, varied header casing/spacing, GHz frequency.
    pd.DataFrame(
        {
            "Building": ["officeA", "officeA", "officeA"],
            "Config ID": [1, 1, 2],
            "TX": ["tx0", "tx0", "tx1"],
            "Rx_ID": ["r0", "r1", "r2"],
            "Path Loss (dB)": [72.5, 85.1, 99.0],
            "Distance [m]": [3.0, 8.0, 15.0],
            "Frequency (GHz)": [3.5, 3.5, 3.5],
            "LOS/NLOS": ["LOS", "NLOS", "nlos"],
            "Brick Walls": [0, 1, 2],
            "Glass": [0, 0, 1],
            "antenna_height_m": [1.5, 1.5, 1.5],
        }
    ).to_csv(raw_root / "campaign_office.csv", index=False)
    # File 2: no LOS column (must be derived from wall counts), no frequency col.
    pd.DataFrame(
        {
            "scenario": ["lobby", "lobby"],
            "tx_id": ["txA", "txA"],
            "point_id": ["p0", "p1"],
            "pathloss_db": [60.0, 110.0],
            "distance_m": [1.0, 20.0],
            "wood": [0, 3],
            "column": [0, 1],
        }
    ).to_csv(raw_root / "campaign_lobby.csv", index=False)
    return raw_root


def test_measured_converter_writes_table_and_summary(tmp_path):
    mod = _load_converter()
    raw = _make_raw(tmp_path / "raw")
    out = tmp_path / "processed"
    summary = mod.convert_dataset(raw, out, verbose=False)

    points = out / "measured_points.csv"
    assert points.exists()
    assert (out / "measured_points_summary.json").exists()

    df = pd.read_csv(points)
    assert summary["total_rows"] == 5
    assert len(df) == 5
    # Canonical columns present and snake_cased.
    for col in ["scenario", "tx_id", "rx_id", "measured_path_loss_db", "distance_m", "frequency_hz", "los_nlos", "source_file"]:
        assert col in df.columns
    # Frequency converted to Hz and defaulted for the file lacking it.
    assert set(df["frequency_hz"].unique()) == {3.5e9}
    # Extra (unmapped) columns are preserved, not dropped.
    assert "antenna_height_m" in df.columns


def test_measured_converter_normalizes_and_derives_los(tmp_path):
    mod = _load_converter()
    raw = _make_raw(tmp_path / "raw")
    out = tmp_path / "processed"
    mod.convert_dataset(raw, out, verbose=False)
    df = pd.read_csv(out / "measured_points.csv")

    lobby = df[df["scenario"] == "lobby"].sort_values("distance_m")
    # Derived from wall counts: zero walls -> LOS, walls present -> NLOS.
    assert lobby.iloc[0]["los_nlos"] == "LOS"
    assert lobby.iloc[-1]["los_nlos"] == "NLOS"
    # Explicit labels are canonicalized.
    office = df[df["scenario"] == "officeA"]
    assert set(office["los_nlos"]) == {"LOS", "NLOS"}
    # rx_id mapped from "Rx_ID" in file 1 and "point_id" in file 2.
    assert df["rx_id"].notna().all()


def test_measured_converter_summary_statistics(tmp_path):
    mod = _load_converter()
    raw = _make_raw(tmp_path / "raw")
    out = tmp_path / "processed"
    mod.convert_dataset(raw, out, verbose=False)
    summary = json.loads((out / "measured_points_summary.json").read_text())

    assert summary["rows_per_scenario"] == {"officeA": 3, "lobby": 2}
    assert summary["path_loss_db"]["min"] == 60.0
    assert summary["path_loss_db"]["max"] == 110.0
    assert summary["distance_m"]["max"] == 20.0
    assert "brick_wall_count" in summary["available_wall_count_columns"]
    assert "wood_wall_count" in summary["available_wall_count_columns"]


def test_measured_converter_dry_run_writes_nothing(tmp_path):
    mod = _load_converter()
    raw = _make_raw(tmp_path / "raw")
    out = tmp_path / "processed"
    summary = mod.convert_dataset(raw, out, dry_run=True, verbose=False)
    assert summary["total_rows"] == 5
    assert not out.exists()


def test_measured_converter_errors_without_csvs(tmp_path):
    mod = _load_converter()
    raw = tmp_path / "empty"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV files"):
        mod.convert_dataset(raw, tmp_path / "out", verbose=False)
