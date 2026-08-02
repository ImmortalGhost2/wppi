#!/usr/bin/env python
"""Convert the 2026 measured 3.5 GHz indoor path-loss CSV dataset to a clean table.

LEGACY / NON-PAPER / HISTORICAL COMPATIBILITY ONLY.
This workflow is not the source of the final manuscript results; use
``scripts/analysis/evaluate_external_3p5ghz.py`` for the canonical pipeline.

This dataset is **point-level measured data**, not dense 2D radio maps, so this
converter deliberately stays independent of the dense-map pipeline. It reads the
campaign CSV files, normalizes their heterogeneous headers to snake_case, maps
known synonyms onto a canonical schema (without dropping any original columns),
and writes a single tidy table plus a summary.

Example
-------
python scripts/converters/convert_measured_3p5ghz.py \
  --raw-root data/measured_3p5ghz_raw \
  --out-root data/measured_3p5ghz_processed
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FREQUENCY_HZ = 3.5e9
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "measured_3p5ghz_processed"

WALL_COUNT_FIELDS = (
    "brick_wall_count",
    "wood_wall_count",
    "glass_wall_count",
    "drywall_count",
    "column_count",
)

# Canonical field -> ordered list of accepted normalized aliases (exact match first).
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "scenario": ("scenario", "building", "environment", "env", "site", "venue", "scene", "location"),
    "campaign_id": ("campaign", "campaign_id", "config", "config_id", "setup", "setup_id", "measurement_config"),
    "tx_id": ("tx", "tx_id", "transmitter", "transmitter_id", "tx_name", "bs", "base_station"),
    "rx_id": ("rx", "rx_id", "receiver", "receiver_id", "sample", "sample_id", "point", "point_id", "measurement_id", "meas_id"),
    "measured_path_loss_db": (
        "measured_path_loss_db", "path_loss_db", "pathloss_db", "pl_db", "measured_path_loss",
        "measured_pl_db", "path_loss", "pathloss", "pl", "measured_pl",
    ),
    "distance_m": ("distance_m", "distance", "dist", "dist_m", "d_m", "range", "range_m", "link_distance", "tx_rx_distance", "separation"),
    "frequency_hz": ("frequency_hz", "freq_hz", "f_hz", "frequency", "freq", "carrier_frequency", "fc"),
    "los_nlos": ("los_nlos", "los", "nlos", "condition", "link_type", "propagation", "visibility", "link_condition"),
    "brick_wall_count": ("brick_wall_count", "brick_count", "brick", "brick_walls", "n_brick", "num_brick"),
    "wood_wall_count": ("wood_wall_count", "wood_count", "wood", "wooden", "wood_walls", "n_wood", "num_wood"),
    "glass_wall_count": ("glass_wall_count", "glass_count", "glass", "glass_walls", "window", "windows", "n_glass"),
    "drywall_count": ("drywall_count", "drywall", "dry_wall", "plaster", "gypsum", "partition", "n_drywall"),
    "column_count": ("column_count", "column", "columns", "pillar", "pillars", "concrete_column", "n_column"),
}

CANONICAL_ORDER = (
    "scenario", "campaign_id", "tx_id", "rx_id",
    "measured_path_loss_db", "distance_m", "frequency_hz", "los_nlos",
    *WALL_COUNT_FIELDS,
)


# Header normalization and synonym mapping
def normalize_name(name: str) -> str:
    """Normalize a raw column header to snake_case (units kept as suffix tokens)."""
    text = str(name).strip().lower()
    text = text.replace("%", "_pct").replace("#", "_num")
    text = re.sub(r"[\u00b0\u2103]", "", text)  # strip degree / celsius marks
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_field_lookup() -> Dict[str, str]:
    """Map each accepted alias to its canonical field name."""
    lookup: Dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            lookup.setdefault(alias, canonical)
    return lookup


def map_columns(normalized: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Return (normalized->canonical map, list of unmapped normalized columns).

    Each canonical field binds to the first column that matches one of its
    aliases, so explicit names win and no original column is silently dropped.
    """
    lookup = build_field_lookup()
    mapping: Dict[str, str] = {}
    claimed: set = set()
    for col in normalized:
        canonical = lookup.get(col)
        if canonical is not None and canonical not in claimed:
            mapping[col] = canonical
            claimed.add(canonical)
    unmapped = [c for c in normalized if c not in mapping]
    return mapping, unmapped


# Value coercion
def _coerce_frequency(series: pd.Series, source_name: str) -> pd.Series:
    """Coerce a frequency column to Hz using the header unit, then magnitude."""
    values = pd.to_numeric(series, errors="coerce")
    name = source_name.lower()
    if "ghz" in name:
        return values * 1e9
    if "mhz" in name:
        return values * 1e6
    if "khz" in name:
        return values * 1e3
    if "hz" in name:
        return values
    # No unit in the header: infer from magnitude (3.5 -> GHz, 3500 -> MHz, ...).
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    median = float(finite.median())
    if median < 1e3:
        return values * 1e9
    if median < 1e6:
        return values * 1e6
    return values


def _normalize_los_label(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text in {"1", "true", "yes", "y", "los", "line_of_sight", "line-of-sight"}:
        return "LOS"
    if text in {"0", "false", "no", "n", "nlos", "non_los", "non-los", "nlos_"}:
        return "NLOS"
    if "nlos" in text or "non" in text:
        return "NLOS"
    if "los" in text:
        return "LOS"
    return str(value)


# Per-file conversion
def convert_file(path: Path, default_frequency_hz: float) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read one CSV and return (canonicalized DataFrame, per-file info)."""
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"CSV {path} has no data rows.")
    normalized = [normalize_name(c) for c in raw.columns]
    # Resolve duplicate normalized names defensively.
    seen: Dict[str, int] = {}
    unique_norm: List[str] = []
    for col in normalized:
        if col in seen:
            seen[col] += 1
            unique_norm.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_norm.append(col)
    raw.columns = unique_norm

    mapping, unmapped = map_columns(unique_norm)
    out = pd.DataFrame(index=raw.index)

    for col, canonical in mapping.items():
        if canonical == "frequency_hz":
            out[canonical] = _coerce_frequency(raw[col], col)
        elif canonical == "los_nlos":
            out[canonical] = raw[col].map(_normalize_los_label)
        elif canonical in WALL_COUNT_FIELDS:
            out[canonical] = pd.to_numeric(raw[col], errors="coerce")
        elif canonical in {"measured_path_loss_db", "distance_m"}:
            out[canonical] = pd.to_numeric(raw[col], errors="coerce")
        else:
            out[canonical] = raw[col]

    # Keep every unmapped original column so nothing important is lost.
    for col in unmapped:
        target = col if col not in out.columns else f"extra_{col}"
        out[target] = raw[col]

    # Defaults and derivations.
    if "frequency_hz" not in out.columns:
        out["frequency_hz"] = float(default_frequency_hz)
    else:
        out["frequency_hz"] = out["frequency_hz"].fillna(float(default_frequency_hz))

    present_wall_fields = [f for f in WALL_COUNT_FIELDS if f in out.columns]
    los_source = "explicit" if "los_nlos" in out.columns else None
    if "los_nlos" not in out.columns and present_wall_fields:
        total_walls = out[present_wall_fields].fillna(0).sum(axis=1)
        out["los_nlos"] = np.where(total_walls > 0, "NLOS", "LOS")
        los_source = "derived_from_wall_counts"

    out["source_file"] = path.name

    info = {
        "source_file": path.name,
        "rows": int(len(out)),
        "mapped_columns": {k: v for k, v in mapping.items()},
        "unmapped_columns": unmapped,
        "wall_count_columns": present_wall_fields,
        "los_source": los_source,
    }
    return out, info


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_present = [c for c in CANONICAL_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in canonical_present and c != "source_file"]
    ordered = canonical_present + sorted(extras) + ["source_file"]
    return df[ordered]


# Summary
def summarize(df: pd.DataFrame, file_infos: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_rows": int(len(df)),
        "source_files": [info["source_file"] for info in file_infos],
        "rows_per_source_file": df["source_file"].value_counts().to_dict(),
    }

    if "scenario" in df.columns:
        summary["rows_per_scenario"] = {str(k): int(v) for k, v in df["scenario"].value_counts().items()}
    else:
        summary["rows_per_scenario"] = {}

    def _range(col: str) -> Optional[Dict[str, float]]:
        if col not in df.columns:
            return None
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            return None
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "count": int(finite.size),
        }

    summary["path_loss_db"] = _range("measured_path_loss_db")
    summary["distance_m"] = _range("distance_m")
    summary["frequency_hz_unique"] = sorted(
        {float(v) for v in pd.to_numeric(df.get("frequency_hz", pd.Series(dtype=float)), errors="coerce").dropna().unique()}
    )
    summary["available_wall_count_columns"] = [f for f in WALL_COUNT_FIELDS if f in df.columns]
    if "los_nlos" in df.columns:
        summary["los_nlos_distribution"] = {str(k): int(v) for k, v in df["los_nlos"].value_counts(dropna=False).items()}
    summary["missing_values"] = {col: int(df[col].isna().sum()) for col in df.columns}
    summary["columns"] = list(df.columns)
    summary["per_file"] = file_infos
    return summary


def _print_summary(summary: Dict[str, object]) -> None:
    print("=" * 64)
    print("Measured 3.5 GHz indoor path-loss conversion")
    print("=" * 64)
    print(f"Total rows           : {summary['total_rows']}")
    print(f"Source files         : {summary['source_files']}")
    print(f"Rows per scenario    : {summary['rows_per_scenario']}")
    pl = summary.get("path_loss_db")
    if pl:
        print(f"Path-loss (dB)       : min={pl['min']:.2f}  max={pl['max']:.2f}  mean={pl['mean']:.2f}")
    dist = summary.get("distance_m")
    if dist:
        print(f"Distance (m)         : min={dist['min']:.3f}  max={dist['max']:.3f}  mean={dist['mean']:.3f}")
    print(f"Frequencies (Hz)     : {summary['frequency_hz_unique']}")
    print(f"Wall-count columns   : {summary['available_wall_count_columns']}")
    if "los_nlos_distribution" in summary:
        print(f"LOS/NLOS distribution: {summary['los_nlos_distribution']}")
    missing = {k: v for k, v in summary["missing_values"].items() if v}
    print(f"Columns with missing : {missing if missing else 'none'}")
    print("=" * 64)


# Driver
def convert_dataset(
    raw_root: Path,
    out_root: Path = DEFAULT_OUT_ROOT,
    *,
    pattern: str = "*.csv",
    default_frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """Convert every CSV under ``raw_root`` into one point-level table."""
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    csv_files = sorted(p for p in raw_root.glob(pattern) if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' under {raw_root}.")

    frames: List[pd.DataFrame] = []
    file_infos: List[Dict[str, object]] = []
    for path in csv_files:
        frame, info = convert_file(path, default_frequency_hz)
        frames.append(frame)
        file_infos.append(info)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _order_columns(combined)
    if "measured_path_loss_db" not in combined.columns:
        raise ValueError(
            "No measured path-loss column was found in any CSV. Expected a header "
            f"matching one of: {FIELD_ALIASES['measured_path_loss_db']}."
        )

    summary = summarize(combined, file_infos)

    if verbose:
        _print_summary(summary)

    if not dry_run:
        out_root = Path(out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        points_path = out_root / "measured_points.csv"
        summary_path = out_root / "measured_points_summary.json"
        combined.to_csv(points_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_csv"] = str(points_path)
        summary["output_summary"] = str(summary_path)
        if verbose:
            print(f"Wrote {len(combined)} rows to {points_path}")
            print(f"Wrote summary to {summary_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert measured 3.5 GHz indoor path-loss CSVs to a point table.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder containing the measured CSV files.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output directory for the processed table.")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob for input CSV files (default: *.csv).")
    parser.add_argument("--default-frequency-hz", type=float, default=DEFAULT_FREQUENCY_HZ, help="Frequency used when a CSV omits it.")
    parser.add_argument("--dry-run", action="store_true", help="Summarize without writing files.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    convert_dataset(
        raw_root=args.raw_root,
        out_root=args.out_root,
        pattern=args.pattern,
        default_frequency_hz=args.default_frequency_hz,
        dry_run=args.dry_run,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
