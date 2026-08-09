#!/usr/bin/env python
"""Audit ICASSP transmitter coordinates against raster bounds (diagnostic only).

Some official ICASSP Task 2 / Task 3 position files store a transmitter pixel
coordinate that lies outside the building raster by more than one pixel, which
the converter currently rejects (it only snaps coordinates within
``_TX_BOUNDARY_TOL`` of the edge). Examples:

* Task 2: ``B19_Ant1_f2_S33`` has ``tx_xy=(-3, 165)`` on a 180x219 map.
* Task 3: ``B1_Ant2_f1_S74`` has ``tx_xy=(277, 468)`` on a 348x464 map.

This tool inspects a task's raw ``Inputs`` / ``Outputs`` / ``Positions`` exactly
the way the converter parses them, loads every transmitter coordinate using the
converter's official ``tx_x = Y, tx_y = X`` convention, and reports every
out-of-bounds case with the per-boundary overflow and the maximum clip distance
needed (in pixels and meters). It changes nothing: the converter is imported
read-only and no NPZ / dataset is written.

CLI
---
python scripts/analysis/audit_icassp_tx_bounds.py \
    --raw-root ./data/icassp2025_indoor_raw --task 2 \
    --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO_ROOT / "scripts" / "converters" / "convert_icassp2025_indoor.py"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "diagnostics"

# Columns written to the CSV, in a stable order.
CSV_COLUMNS: Tuple[str, ...] = (
    "sample_id", "task", "building_id", "antenna_id", "frequency_id", "frequency_hz",
    "sample_index", "width", "height", "tx_x", "tx_y",
    "out_left_px", "out_right_px", "out_top_px", "out_bottom_px",
    "max_clip_px", "max_clip_m", "within_converter_tol", "converter_would_reject",
)


def load_converter():
    """Import the ICASSP converter module read-only (it is not an importable package)."""
    spec = importlib.util.spec_from_file_location("convert_icassp2025_indoor", CONVERTER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load converter module from {CONVERTER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_tx_bounds(
    tx_x: float,
    tx_y: float,
    width: int,
    height: int,
    resolution_m: float = 0.25,
    boundary_tol: float = 1.0,
) -> Dict[str, float]:
    """Per-boundary overflow and maximum clip distance for one transmitter.

    Mirrors the converter's bounds convention: ``tx_x`` indexes the horizontal
    axis (valid ``[0, width - 1]``) and ``tx_y`` the vertical axis (valid
    ``[0, height - 1]``). Overflows are non-negative pixel distances past each
    edge; ``max_clip_px`` is the largest of them and ``max_clip_m`` the same in
    meters. ``within_converter_tol`` marks coordinates the converter would snap
    onto the edge; ``converter_would_reject`` marks those it currently rejects.
    """
    tx_x = float(tx_x)
    tx_y = float(tx_y)
    width = int(width)
    height = int(height)
    out_left = max(0.0, 0.0 - tx_x)
    out_right = max(0.0, tx_x - (width - 1))
    out_top = max(0.0, 0.0 - tx_y)
    out_bottom = max(0.0, tx_y - (height - 1))
    max_px = max(out_left, out_right, out_top, out_bottom)
    return {
        "out_left_px": out_left,
        "out_right_px": out_right,
        "out_top_px": out_top,
        "out_bottom_px": out_bottom,
        "max_clip_px": max_px,
        "max_clip_m": max_px * float(resolution_m),
        "out_of_bounds": bool(max_px > 0.0),
        "within_converter_tol": bool(0.0 < max_px <= float(boundary_tol)),
        "converter_would_reject": bool(max_px > float(boundary_tol)),
    }


def _raw_tx_xy(converter, raw_root: Path, sample, pos_cache: Dict[Tuple[int, int, int], pd.DataFrame]) -> Tuple[float, float]:
    """Raw transmitter ``(tx_x, tx_y)`` using the converter's official convention.

    Identical column convention to ``converter.load_tx_xy`` (``tx_x = Y``,
    ``tx_y = X``) but WITHOUT any edge snapping or bounds rejection, so the true
    out-of-bounds coordinate is preserved for reporting.
    """
    key = (sample.building_id, sample.antenna_id, sample.frequency_id)
    df = pos_cache.get(key)
    if df is None:
        pos_path = raw_root / "Positions" / f"Positions_B{sample.building_id}_Ant{sample.antenna_id}_f{sample.frequency_id}.csv"
        if not pos_path.exists():
            raise FileNotFoundError(f"Missing transmitter positions file: {pos_path}")
        df = pd.read_csv(pos_path)
        if "X" not in df.columns or "Y" not in df.columns:
            raise ValueError(f"{pos_path.name} must contain 'X' and 'Y' columns; found {list(df.columns)}.")
        pos_cache[key] = df
    s = int(sample.sample_index)
    if s < 0 or s >= len(df):
        raise IndexError(f"{sample.sample_id}: sample index S{s} is out of range ({len(df)} rows).")
    row = df.iloc[s]
    # Official convention: tx_x is the 'Y' column, tx_y is the 'X' column.
    return float(row["Y"]), float(row["X"])


def run_audit(
    raw_root: Path,
    task: int,
    resolution_m: float,
    boundary_tol: float,
    converter,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Scan a task and return ``(out_of_bounds_df, info)``.

    ``info`` carries ``n_scanned``, ``n_buildings`` and a list of per-sample
    ``errors`` (missing positions file, bad index, unreadable raster) so a single
    bad sample never aborts the scan.
    """
    raw_root = Path(raw_root).expanduser().resolve()
    samples = converter.discover_samples(raw_root, task)
    freq_map = dict(converter.FREQUENCY_HZ_BY_ID)

    shape_cache: Dict[int, object] = {}  # building_id -> (H, W) or Exception
    pos_cache: Dict[Tuple[int, int, int], pd.DataFrame] = {}
    rows: List[dict] = []
    errors: List[dict] = []

    def shape_for(sample) -> Tuple[int, int]:
        bid = sample.building_id
        if bid not in shape_cache:
            try:
                img = converter.load_image(sample.input_path)
                shape_cache[bid] = (int(img.shape[0]), int(img.shape[1]))
            except Exception as exc:  # cache the failure to avoid re-reading
                shape_cache[bid] = exc
        val = shape_cache[bid]
        if isinstance(val, Exception):
            raise val
        return val  # type: ignore[return-value]

    for sample in samples:
        try:
            height, width = shape_for(sample)
            tx_x, tx_y = _raw_tx_xy(converter, raw_root, sample, pos_cache)
        except Exception as exc:
            errors.append({"sample_id": sample.sample_id, "error": repr(exc)})
            continue
        b = compute_tx_bounds(tx_x, tx_y, width, height, resolution_m, boundary_tol)
        if not b["out_of_bounds"]:
            continue
        rows.append({
            "sample_id": sample.sample_id,
            "task": int(task),
            "building_id": int(sample.building_id),
            "antenna_id": int(sample.antenna_id),
            "frequency_id": int(sample.frequency_id),
            "frequency_hz": freq_map.get(int(sample.frequency_id)),
            "sample_index": int(sample.sample_index),
            "width": int(width),
            "height": int(height),
            "tx_x": tx_x,
            "tx_y": tx_y,
            "out_left_px": b["out_left_px"],
            "out_right_px": b["out_right_px"],
            "out_top_px": b["out_top_px"],
            "out_bottom_px": b["out_bottom_px"],
            "max_clip_px": b["max_clip_px"],
            "max_clip_m": b["max_clip_m"],
            "within_converter_tol": b["within_converter_tol"],
            "converter_would_reject": b["converter_would_reject"],
        })

    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    if not df.empty:
        df = df.sort_values(["max_clip_px", "building_id", "antenna_id", "frequency_id", "sample_index"], ascending=[False, True, True, True, True]).reset_index(drop=True)
    info = {
        "n_scanned": int(len(samples)),
        "n_buildings": int(len({s.building_id for s in samples})),
        "errors": errors,
        "freq_map": freq_map,
    }
    return df, info


def _freq_label(fid: int, hz: Optional[float]) -> str:
    if hz is None:
        return f"f{fid}"
    mhz = float(hz) / 1.0e6
    return f"f{fid} ({mhz / 1000.0:g} GHz)" if mhz >= 1000.0 else f"f{fid} ({mhz:g} MHz)"


def print_summary(df: pd.DataFrame, task: int, info: Dict[str, object], boundary_tol: float, resolution_m: float, stream=None) -> None:
    """Print summary counts by task, antenna, frequency, boundary, and max offset."""
    # Resolve the stream at call time so a redirected/captured sys.stdout is honored.
    stream = stream if stream is not None else sys.stdout
    freq_map = info.get("freq_map", {})  # type: ignore[assignment]
    n_scanned = int(info["n_scanned"])  # type: ignore[index]
    n_buildings = int(info["n_buildings"])  # type: ignore[index]
    errors = info.get("errors", [])  # type: ignore[assignment]

    def p(*a):
        print(*a, file=stream)

    p(f"=== ICASSP Task {task} transmitter-bounds audit ===")
    p(f"Scanned {n_scanned} sample(s) across {n_buildings} building(s); "
      f"resolution {resolution_m:g} m/px; converter snap tolerance {boundary_tol:g} px.")
    n_oob = int(len(df))
    if n_oob == 0:
        p("No out-of-bounds transmitter coordinates found.")
    else:
        pct = 100.0 * n_oob / max(1, n_scanned)
        n_snap = int(df["within_converter_tol"].sum())
        n_reject = int(df["converter_would_reject"].sum())
        p(f"Out-of-bounds transmitters: {n_oob} ({pct:.2f}% of scanned).")
        p(f"  Within converter tolerance, would SNAP (<= {boundary_tol:g} px): {n_snap}")
        p(f"  Beyond tolerance, converter currently REJECTS (> {boundary_tol:g} px): {n_reject}")
        p("By antenna:")
        for ant, c in sorted(Counter(df["antenna_id"]).items()):
            p(f"  Ant{int(ant)}: {c}")
        p("By frequency:")
        for fid, c in sorted(Counter(df["frequency_id"]).items()):
            p(f"  {_freq_label(int(fid), freq_map.get(int(fid)))}: {c}")
        p("By boundary (samples overflowing each edge):")
        p(f"  left: {int((df['out_left_px'] > 0).sum())}  "
          f"right: {int((df['out_right_px'] > 0).sum())}  "
          f"top: {int((df['out_top_px'] > 0).sum())}  "
          f"bottom: {int((df['out_bottom_px'] > 0).sum())}")
        p("Max-offset distribution (ceil px):")
        dist = Counter(int(math.ceil(float(v))) for v in df["max_clip_px"])
        for k in sorted(dist):
            p(f"  {k} px: {dist[k]}")
        worst = df.loc[df["max_clip_px"].idxmax()]
        p(f"Maximum clip distance: {float(worst['max_clip_px']):g} px "
          f"({float(worst['max_clip_m']):g} m) at {worst['sample_id']} "
          f"(tx=({float(worst['tx_x']):g}, {float(worst['tx_y']):g}), "
          f"{int(worst['width'])}x{int(worst['height'])}).")
    if errors:
        p(f"[warn] {len(errors)} sample(s) could not be inspected (e.g. {errors[0]['sample_id']}: {errors[0]['error']}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ICASSP transmitter coordinates against raster bounds (read-only).")
    parser.add_argument("--raw-root", type=Path, required=True, help="Downloaded ICASSP 2025 indoor dataset root.")
    parser.add_argument("--task", type=int, required=True, choices=(1, 2, 3), help="Challenge task to audit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for the CSV report.")
    parser.add_argument("--resolution-m", type=float, default=None, help="Meters per pixel (default: converter's 0.25).")
    parser.add_argument("--boundary-tol", type=float, default=None, help="Edge-snap tolerance in px (default: converter's _TX_BOUNDARY_TOL).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    converter = load_converter()
    resolution_m = float(args.resolution_m) if args.resolution_m is not None else float(getattr(converter, "DEFAULT_RESOLUTION_M", 0.25))
    boundary_tol = float(args.boundary_tol) if args.boundary_tol is not None else float(getattr(converter, "_TX_BOUNDARY_TOL", 1.0))

    df, info = run_audit(args.raw_root, int(args.task), resolution_m, boundary_tol, converter)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"task{int(args.task)}_tx_bounds.csv"
    df.to_csv(out_csv, index=False)

    print_summary(df, int(args.task), info, boundary_tol, resolution_m)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
