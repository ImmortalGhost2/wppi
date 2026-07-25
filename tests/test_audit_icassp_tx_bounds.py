from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.audit_icassp_tx_bounds import (
    compute_tx_bounds,
    load_converter,
    main,
    print_summary,
    run_audit,
)

PIL_Image = pytest.importorskip("PIL.Image")


# --------------------------------------------------------------------------- #
# Pure-geometry cases: inside, 1px outside, 4px outside, far outside.
# --------------------------------------------------------------------------- #
def test_compute_inside_is_not_out_of_bounds():
    b = compute_tx_bounds(8, 8, width=16, height=16)
    assert b["out_of_bounds"] is False
    assert b["max_clip_px"] == 0
    assert b["max_clip_m"] == 0
    assert b["within_converter_tol"] is False
    assert b["converter_would_reject"] is False


def test_compute_one_pixel_outside_is_within_tolerance():
    b = compute_tx_bounds(-1, 8, width=16, height=16, resolution_m=0.25, boundary_tol=1.0)
    assert b["out_of_bounds"] is True
    assert b["out_left_px"] == 1
    assert b["max_clip_px"] == 1
    assert b["max_clip_m"] == pytest.approx(0.25)
    assert b["within_converter_tol"] is True
    assert b["converter_would_reject"] is False


def test_compute_four_pixels_outside_is_rejected():
    b = compute_tx_bounds(-4, 8, width=16, height=16, resolution_m=0.25, boundary_tol=1.0)
    assert b["out_left_px"] == 4
    assert b["max_clip_px"] == 4
    assert b["max_clip_m"] == pytest.approx(1.0)
    assert b["within_converter_tol"] is False
    assert b["converter_would_reject"] is True


def test_compute_far_outside():
    b = compute_tx_bounds(8, 40, width=16, height=16, resolution_m=0.25, boundary_tol=1.0)
    assert b["out_bottom_px"] == 25  # 40 - (16 - 1)
    assert b["out_top_px"] == 0
    assert b["max_clip_px"] == 25
    assert b["max_clip_m"] == pytest.approx(6.25)
    assert b["converter_would_reject"] is True


def test_compute_axes_and_edges():
    # tx_x indexes width; tx_y indexes height. Last valid index is size-1.
    assert compute_tx_bounds(15, 15, width=16, height=16)["out_of_bounds"] is False
    assert compute_tx_bounds(16, 0, width=16, height=16)["out_right_px"] == 1
    assert compute_tx_bounds(0, 16, width=16, height=16)["out_bottom_px"] == 1


def test_compute_reproduces_reported_examples():
    # Task 2: B19_Ant1_f2_S33 tx_xy=(-3, 165) on a 180x219 map.
    t2 = compute_tx_bounds(-3, 165, width=180, height=219)
    assert t2["out_left_px"] == 3
    assert t2["max_clip_px"] == 3
    assert t2["max_clip_m"] == pytest.approx(0.75)
    assert t2["converter_would_reject"] is True
    # Task 3: B1_Ant2_f1_S74 tx_xy=(277, 468) on a 348x464 map.
    t3 = compute_tx_bounds(277, 468, width=348, height=464)
    assert t3["out_bottom_px"] == 5  # 468 - 463
    assert t3["max_clip_px"] == 5
    assert t3["max_clip_m"] == pytest.approx(1.25)
    assert t3["converter_would_reject"] is True


# --------------------------------------------------------------------------- #
# End-to-end scan over a synthetic raw layout parsed exactly like the converter.
# --------------------------------------------------------------------------- #
def _make_raw(raw_root: Path, task: int, pos_X, pos_Y, h: int = 16, w: int = 16, building: int = 1, ant: int = 1, freq: int = 1) -> Path:
    """Tiny official-format raw layout. tx_x = Y column, tx_y = X column."""
    in_dir = raw_root / "Inputs" / f"Task_{task}_ICASSP"
    out_dir = raw_root / "Outputs" / f"Task_{task}_ICASSP"
    pos_dir = raw_root / "Positions"
    for d in (in_dir, out_dir, pos_dir):
        d.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(pos_X)):
        stem = f"B{building}_Ant{ant}_f{freq}_S{i}"
        PIL_Image.fromarray(rgb, mode="RGB").save(in_dir / f"{stem}.png")
        PIL_Image.fromarray(gray, mode="L").save(out_dir / f"{stem}.png")
    pd.DataFrame({"X": list(pos_X), "Y": list(pos_Y)}).to_csv(
        pos_dir / f"Positions_B{building}_Ant{ant}_f{freq}.csv", index=False
    )
    return raw_root


def test_run_audit_flags_only_out_of_bounds(tmp_path):
    # Samples S0..S3: inside, 1px left, 4px left, far bottom.
    # tx_x comes from Y, tx_y from X.
    pos_Y = [8, -1, -4, 8]    # -> tx_x
    pos_X = [8, 8, 8, 40]     # -> tx_y
    raw = _make_raw(tmp_path / "raw", task=2, pos_X=pos_X, pos_Y=pos_Y, h=16, w=16, freq=2)

    df, info = run_audit(raw, task=2, resolution_m=0.25, boundary_tol=1.0, converter=load_converter())

    assert info["n_scanned"] == 4
    assert info["n_buildings"] == 1
    assert info["errors"] == []

    ids = set(df["sample_id"])
    assert ids == {"B1_Ant1_f2_S1", "B1_Ant1_f2_S2", "B1_Ant1_f2_S3"}  # S0 (inside) excluded

    by_id = df.set_index("sample_id")
    assert by_id.loc["B1_Ant1_f2_S1", "max_clip_px"] == 1
    assert bool(by_id.loc["B1_Ant1_f2_S1", "within_converter_tol"]) is True
    assert bool(by_id.loc["B1_Ant1_f2_S1", "converter_would_reject"]) is False

    assert by_id.loc["B1_Ant1_f2_S2", "max_clip_px"] == 4
    assert by_id.loc["B1_Ant1_f2_S2", "max_clip_m"] == pytest.approx(1.0)
    assert bool(by_id.loc["B1_Ant1_f2_S2", "converter_would_reject"]) is True

    assert by_id.loc["B1_Ant1_f2_S3", "out_bottom_px"] == 25
    assert by_id.loc["B1_Ant1_f2_S3", "max_clip_m"] == pytest.approx(6.25)
    # Sorted by descending max clip distance -> the far case is first.
    assert df.iloc[0]["sample_id"] == "B1_Ant1_f2_S3"
    # Frequency / dimensions carried through.
    assert (df["frequency_id"] == 2).all()
    assert (df["width"] == 16).all() and (df["height"] == 16).all()


def test_run_audit_all_inside_is_empty(tmp_path):
    raw = _make_raw(tmp_path / "raw", task=1, pos_X=[5, 6], pos_Y=[5, 6], h=16, w=16)
    df, info = run_audit(raw, task=1, resolution_m=0.25, boundary_tol=1.0, converter=load_converter())
    assert df.empty
    assert list(df.columns)[:3] == ["sample_id", "task", "building_id"]

    buf = io.StringIO()
    print_summary(df, task=1, info=info, boundary_tol=1.0, resolution_m=0.25, stream=buf)
    assert "No out-of-bounds transmitter coordinates found." in buf.getvalue()


def test_run_audit_records_errors_without_crashing(tmp_path):
    # Two PNG samples but only one Positions row -> S1 index is out of range.
    raw = _make_raw(tmp_path / "raw", task=3, pos_X=[5], pos_Y=[5], h=16, w=16)
    # Add a second sample image pair without extending the Positions CSV.
    for sub in ("Inputs", "Outputs"):
        stem = "B1_Ant1_f1_S1"
        src = raw / sub / "Task_3_ICASSP" / "B1_Ant1_f1_S0.png"
        (raw / sub / "Task_3_ICASSP" / f"{stem}.png").write_bytes(src.read_bytes())

    df, info = run_audit(raw, task=3, resolution_m=0.25, boundary_tol=1.0, converter=load_converter())
    assert info["n_scanned"] == 2
    assert len(info["errors"]) == 1
    assert info["errors"][0]["sample_id"] == "B1_Ant1_f1_S1"
    assert df.empty  # S0 is inside; S1 errored out


def test_main_writes_csv(tmp_path):
    # S0 inside, S1 is a clean 4px-left case (tx_x = Y = -4, tx_y = X = 8).
    raw = _make_raw(tmp_path / "raw", task=2, pos_X=[8, 8], pos_Y=[8, -4], h=16, w=16, freq=2)
    out_dir = tmp_path / "diag"
    rc = main(["--raw-root", str(raw), "--task", "2", "--out-dir", str(out_dir)])
    assert rc == 0
    out_csv = out_dir / "task2_tx_bounds.csv"
    assert out_csv.exists()
    written = pd.read_csv(out_csv)
    assert set(written["sample_id"]) == {"B1_Ant1_f2_S1"}  # only the 4px case is out of bounds
    assert int(written.iloc[0]["max_clip_px"]) == 4


def test_summary_reports_counts(tmp_path, capsys):
    raw = _make_raw(tmp_path / "raw", task=2, pos_X=[8, 8], pos_Y=[-1, -4], h=16, w=16)
    df, info = run_audit(raw, task=2, resolution_m=0.25, boundary_tol=1.0, converter=load_converter())
    print_summary(df, task=2, info=info, boundary_tol=1.0, resolution_m=0.25)
    out = capsys.readouterr().out
    assert "Out-of-bounds transmitters: 2" in out
    assert "would SNAP" in out and "REJECTS" in out
    assert "By antenna:" in out and "By frequency:" in out
    assert "left: 2" in out  # both cases overflow the left edge
