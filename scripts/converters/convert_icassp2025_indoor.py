#!/usr/bin/env python
"""Convert the official ICASSP 2025 Indoor Pathloss Radio Map Prediction dataset.

This converter targets the *official* challenge release layout (IEEE DataPort /
``indoorradiomapchallenge.github.io``) and the WallPath-PI data contract
(see ``docs/DATA_CONTRACT.md``). Unlike a generic importer it does not guess the
folder convention: it reads the documented per-task directory structure and the
documented RGB channel semantics directly.

Official raw layout
--------------------
The downloaded dataset root (``--raw-root``) contains::

    Inputs/Task_1_ICASSP/   Outputs/Task_1_ICASSP/
    Inputs/Task_2_ICASSP/   Outputs/Task_2_ICASSP/
    Inputs/Task_3_ICASSP/   Outputs/Task_3_ICASSP/
    Positions/
    Building_Details/
    Radiation_Patterns/

Each sample file is named ``B#_Ant#_f#_S#.png`` (e.g. ``B5_Ant1_f2_S10.png``).

Channel semantics (official documentation)
------------------------------------------
The input image is RGB at 0.25 m/pixel:

* channel 0 -- normal-incidence **reflectance** (0 for air / free space)
* channel 1 -- normal-incidence **transmittance** (0 for air / free space)
* channel 2 -- physical **distance** from the transmitter to each grid point

The output image is a grayscale path-loss map. Because reflectance and
transmittance are both 0 in free space, walls are detected as *non-zero*
reflectance or transmittance, **not** as ``transmittance < 1``.

Frequencies: ``f1 = 868 MHz``, ``f2 = 1.8 GHz``, ``f3 = 3.5 GHz``.

Transmitter positions come from ``Positions/Positions_B#_Ant#_f#.csv``; following
the official code snippet, ``tx_x = row["Y"]`` and ``tx_y = row["X"]`` (image
coordinates), stored as ``tx_xy = [tx_x, tx_y]``.

Run ``--dry-run`` first to scan and summarize without writing anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "icassp2025_indoor_converted"
DEFAULT_RESOLUTION_M = 0.25

# Frequency id -> carrier frequency in Hz (f1=868 MHz, f2=1.8 GHz, f3=3.5 GHz).
FREQUENCY_HZ_BY_ID: Dict[int, float] = {1: 868.0e6, 2: 1.8e9, 3: 3.5e9}

# Per-task coverage, for the human-readable summary only.
TASK_SPEC: Dict[int, str] = {
    1: "Ant1, f1, B1-B25, S0-S49",
    2: "Ant1, f1-f3, B1-B25, S0-S49",
    3: "Ant1-Ant5, f1-f3, B1-B25 (Ant1: S0-S49, Ant2-Ant5: S0-S79)",
}

# Pixels with |reflectance| or |transmittance| above this are walls/interfaces.
_WALL_EPS = 1.0e-6

# Manifest column order (kept stable for downstream tooling and tests).
MANIFEST_COLUMNS: Tuple[str, ...] = (
    "scene_id",
    "sample_id",
    "scene_path",
    "frequency_hz",
    "resolution_m",
    "height",
    "width",
    "task_id",
    "building_id",
    "antenna_id",
    "frequency_id",
    "sample_index",
)

_FILENAME_RE = re.compile(r"^B(?P<b>\d+)_Ant(?P<ant>\d+)_f(?P<f>\d+)_S(?P<s>\d+)$")


# Sample discovery
@dataclass
class IcasspSample:
    sample_id: str
    scene_id: str
    task_id: int
    building_id: int
    antenna_id: int
    frequency_id: int
    sample_index: int
    input_path: Path
    output_path: Path


def parse_sample_name(stem: str) -> Optional[Dict[str, int]]:
    """Parse a ``B#_Ant#_f#_S#`` stem into integer fields, or ``None``."""
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    return {
        "building_id": int(m.group("b")),
        "antenna_id": int(m.group("ant")),
        "frequency_id": int(m.group("f")),
        "sample_index": int(m.group("s")),
    }


def discover_samples(raw_root: Path, task: int) -> List[IcasspSample]:
    """Discover input/output PNG pairs for ``task`` under the official layout."""
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")

    in_dir = raw_root / "Inputs" / f"Task_{task}_ICASSP"
    out_dir = raw_root / "Outputs" / f"Task_{task}_ICASSP"
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Missing inputs directory for task {task}: {in_dir}")
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Missing outputs directory for task {task}: {out_dir}")

    samples: List[IcasspSample] = []
    for p in sorted(in_dir.glob("*.png")):
        meta = parse_sample_name(p.stem)
        if meta is None:
            raise ValueError(
                f"Input file '{p.name}' does not match the expected 'B#_Ant#_f#_S#.png' pattern."
            )
        out_path = out_dir / p.name
        if not out_path.exists():
            raise FileNotFoundError(
                f"Input '{p.name}' has no matching output path-loss image at {out_path}."
            )
        if meta["frequency_id"] not in FREQUENCY_HZ_BY_ID:
            raise ValueError(
                f"Input '{p.name}' has unknown frequency id f{meta['frequency_id']}; "
                f"expected one of {sorted(FREQUENCY_HZ_BY_ID)}."
            )
        samples.append(
            IcasspSample(
                sample_id=p.stem,
                scene_id=f"B{meta['building_id']}",
                task_id=int(task),
                input_path=p,
                output_path=out_path,
                **meta,
            )
        )

    if not samples:
        raise ValueError(
            f"No ICASSP samples found under {in_dir} (expected 'B#_Ant#_f#_S#.png' files)."
        )
    samples.sort(key=lambda s: (s.building_id, s.antenna_id, s.frequency_id, s.sample_index))
    return samples


# Image and metadata loading
def load_image(path: Path) -> np.ndarray:
    """Read an image as a NumPy array using skimage, imageio, or Pillow."""
    try:
        import skimage.io as skio  # type: ignore
    except ImportError:
        skio = None
    if skio is not None:
        return np.asarray(skio.imread(path))

    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError:
        imageio = None
    if imageio is not None:
        return np.asarray(imageio.imread(path))

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Reading image '{path.name}' requires scikit-image, imageio, or Pillow. "
            "Install one of them (`pip install scikit-image` or `pip install pillow`)."
        ) from exc
    return np.asarray(Image.open(path))


# Tolerance (in pixels) for snapping a near-boundary transmitter coordinate onto
# the raster edge. A small number of official ICASSP Task 2/3 position files
# store a transmitter coordinate that lies slightly outside the raster (for
# example y=132 on a height-132 map, or x=-3). Every Task 2 and Task 3
# coordinate was audited before this correction was added
# (scripts/analysis/audit_icassp_tx_bounds.py; reports in
# results/diagnostics/task2_tx_bounds.csv and task3_tx_bounds.csv): the largest
# overflow observed was 5 px (1.25 m at the 0.25 m/pixel dataset resolution),
# concentrated at the bottom/left raster edges. These are edge artifacts of the
# published position tables, NOT an X/Y axis swap, so coordinates within the
# tolerance are snapped onto the nearest valid index while larger deviations are
# still rejected. The default tolerance is 5.0 px to cover every audited case;
# it is overridable via ``--tx-boundary-tol``.
_TX_BOUNDARY_TOL = 5.0


def _clip_tx_to_bounds(value: float, axis_size: int, tol: float = _TX_BOUNDARY_TOL) -> float:
    """Snap a near-boundary Tx coordinate onto ``[0, axis_size - 1]``.

    This is a *bounded correction* for official near-boundary transmitter
    coordinate artifacts, not a coordinate-system change. It only nudges a
    coordinate that sits at most ``tol`` pixels outside the valid index range
    onto the nearest valid edge index (for example ``value=132`` on a
    height-132 axis becomes ``131``, and ``value=-3`` becomes ``0``). Values
    that are already in bounds are returned exactly unchanged, and values
    farther than ``tol`` pixels outside are returned unchanged so the caller's
    bounds check rejects them loudly. Larger deviations are never silently
    clipped.
    """
    hi = float(axis_size - 1)
    tol = float(tol)
    if -tol <= value < 0.0:
        return 0.0
    if hi < value <= hi + tol:
        return hi
    return value


def load_tx_xy(
    raw_root: Path,
    sample: IcasspSample,
    shape: Tuple[int, int],
    tol: float = _TX_BOUNDARY_TOL,
    clip_report: Optional[List[dict]] = None,
) -> np.ndarray:
    """Read the transmitter pixel coordinate ``[tx_x, tx_y] = [Y, X]`` for a sample.

    The official ICASSP convention is preserved exactly: ``tx_x = row['Y']`` and
    ``tx_y = row['X']``. A coordinate at most ``tol`` pixels outside the raster
    is snapped onto the nearest valid index (a bounded correction for official
    near-boundary artifacts); a coordinate farther out raises ``ValueError`` so
    genuine problems still fail loudly. When ``clip_report`` is provided, each
    snapped coordinate is appended to it so the caller can report how many
    corrections were applied.
    """
    pos_path = (
        raw_root
        / "Positions"
        / f"Positions_B{sample.building_id}_Ant{sample.antenna_id}_f{sample.frequency_id}.csv"
    )
    if not pos_path.exists():
        raise FileNotFoundError(f"Missing transmitter positions file: {pos_path}")
    df = pd.read_csv(pos_path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(
            f"{pos_path.name} must contain 'X' and 'Y' columns; found {list(df.columns)}."
        )

    s = sample.sample_index
    if s < 0 or s >= len(df):
        raise IndexError(
            f"{sample.sample_id}: sample index S{s} is out of range for "
            f"{pos_path.name} ({len(df)} rows)."
        )
    row = df.iloc[s]
    # Official convention: tx_x is the 'Y' column, tx_y is the 'X' column.
    tx_x = float(row["Y"])
    tx_y = float(row["X"])

    height, width = shape
    # Snap near-boundary coordinates (at most ``tol`` px outside) onto the
    # nearest valid index; values farther out are left unchanged so the check
    # below rejects them. This is a bounded correction, never an axis change.
    tx_x_raw, tx_y_raw = tx_x, tx_y
    tx_x = _clip_tx_to_bounds(tx_x, width, tol)
    tx_y = _clip_tx_to_bounds(tx_y, height, tol)
    if not (0.0 <= tx_x <= width - 1) or not (0.0 <= tx_y <= height - 1):
        raise ValueError(
            f"{sample.sample_id}: tx_xy=({tx_x_raw:.2f}, {tx_y_raw:.2f}) lies outside the "
            f"{width}x{height} map by more than the {float(tol):g}px boundary tolerance."
        )
    if clip_report is not None and (tx_x != tx_x_raw or tx_y != tx_y_raw):
        clip_report.append(
            {
                "sample_id": sample.sample_id,
                "tx_x_raw": float(tx_x_raw),
                "tx_y_raw": float(tx_y_raw),
                "tx_x": float(tx_x),
                "tx_y": float(tx_y),
                "width": int(width),
                "height": int(height),
            }
        )
    return np.asarray([tx_x, tx_y], dtype=np.float32)


def load_building_dims(raw_root: Path, building_id: int) -> Optional[Tuple[int, int]]:
    """Return ``(W, H)`` from ``Building_Details/B#_Details.csv`` if available."""
    details = raw_root / "Building_Details" / f"B{building_id}_Details.csv"
    if not details.exists():
        return None
    df = pd.read_csv(details)
    if "W" not in df.columns or "H" not in df.columns or df.empty:
        return None
    return int(df["W"].iloc[0]), int(df["H"].iloc[0])


# Contract array derivation
def derive_wall_mask(reflectance: np.ndarray, transmittance: np.ndarray) -> np.ndarray:
    """Walls are pixels with non-zero reflectance or transmittance (free space is 0)."""
    return (np.abs(reflectance) > _WALL_EPS) | (np.abs(transmittance) > _WALL_EPS)


def derive_material_map(
    reflectance: np.ndarray,
    transmittance: np.ndarray,
    wall_mask: np.ndarray,
) -> np.ndarray:
    """Integer material map: 0 free space, 1 weak, 2 medium, 3 strong interface.

    Interface strength is the combined magnitude ``|reflectance| + |transmittance|``.
    Classes are assigned by tertile thresholds over the non-zero (wall) strengths.
    When too few distinct wall strengths exist, all wall pixels become class 1.
    """
    material = np.zeros(reflectance.shape, dtype=np.int32)
    strength = np.abs(reflectance) + np.abs(transmittance)
    wall_strength = strength[wall_mask]
    if wall_strength.size == 0:
        return material

    if wall_strength.size < 3 or np.unique(wall_strength).size < 3:
        material[wall_mask] = 1
        return material

    q33, q66 = np.quantile(wall_strength, [1.0 / 3.0, 2.0 / 3.0])
    classes = np.ones(reflectance.shape, dtype=np.int32)
    classes[strength > q33] = 2
    classes[strength > q66] = 3
    material[wall_mask] = classes[wall_mask]
    return material


# Conversion
def build_payload(
    raw_root: Path,
    sample: IcasspSample,
    resolution_m: float,
    tx_boundary_tol: float = _TX_BOUNDARY_TOL,
    clip_report: Optional[List[dict]] = None,
) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    """Load one sample and assemble its contract NPZ payload."""
    img = load_image(sample.input_path)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(
            f"{sample.sample_id}: expected an RGB input image (H, W, 3), got shape {img.shape}."
        )
    reflectance = img[:, :, 0].astype(np.float32)
    transmittance = img[:, :, 1].astype(np.float32)
    distance_img = img[:, :, 2].astype(np.float32)

    out_img = load_image(sample.output_path)
    if out_img.ndim == 3:
        out_img = out_img[:, :, 0]
    path_loss = np.asarray(out_img, dtype=np.float32)
    if path_loss.ndim != 2:
        raise ValueError(
            f"{sample.sample_id}: output path-loss map must be 2D, got shape {path_loss.shape}."
        )
    if path_loss.shape != reflectance.shape:
        raise ValueError(
            f"{sample.sample_id}: output shape {path_loss.shape} != input shape {reflectance.shape}."
        )

    shape = (int(path_loss.shape[0]), int(path_loss.shape[1]))
    wall_mask = derive_wall_mask(reflectance, transmittance)
    material_map = derive_material_map(reflectance, transmittance, wall_mask)
    valid_mask = np.isfinite(path_loss)
    tx_xy = load_tx_xy(raw_root, sample, shape, tol=tx_boundary_tol, clip_report=clip_report)
    frequency_hz = FREQUENCY_HZ_BY_ID[sample.frequency_id]

    payload: Dict[str, np.ndarray] = {
        "path_loss": path_loss.astype(np.float32),
        "valid_mask": valid_mask.astype(np.uint8),
        "wall_mask": wall_mask.astype(np.uint8),
        "material_map": material_map.astype(np.int32),
        "tx_xy": tx_xy.astype(np.float32),
        "frequency_hz": np.asarray(float(frequency_hz), dtype=np.float64),
        "resolution_m": np.asarray(float(resolution_m), dtype=np.float32),
        "reflectance": reflectance.astype(np.float32),
        "transmittance": transmittance.astype(np.float32),
        "distance_img": distance_img.astype(np.float32),
        "scene_id": np.asarray(sample.scene_id),
        "sample_id": np.asarray(sample.sample_id),
        "antenna_id": np.asarray(np.int32(sample.antenna_id)),
        "task_id": np.asarray(np.int32(sample.task_id)),
        "building_id": np.asarray(np.int32(sample.building_id)),
        "frequency_id": np.asarray(np.int32(sample.frequency_id)),
    }
    return payload, shape


def _write_npz(out_path: Path, payload: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / (out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(out_path)


def convert_dataset(
    raw_root: Path,
    out_root: Path,
    *,
    task: int = 1,
    limit: Optional[int] = None,
    overwrite: bool = False,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    tx_boundary_tol: float = _TX_BOUNDARY_TOL,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """Convert one task of the ICASSP dataset; return a summary dict.

    Writes ``scenes/<sample_id>.npz`` and ``manifest.csv`` under ``out_root``
    unless ``dry_run`` is set, in which case it only scans and summarizes.
    """
    if int(task) not in (1, 2, 3):
        raise ValueError(f"--task must be 1, 2, or 3; got {task}.")
    raw_root = Path(raw_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()

    samples = discover_samples(raw_root, int(task))
    if limit is not None:
        samples = samples[: int(limit)]

    scene_dir = out_root / "scenes"
    rows: List[Dict[str, object]] = []
    seen_targets: Dict[str, Path] = {}
    checked_buildings: Dict[int, Optional[Tuple[int, int]]] = {}

    shapes: Counter = Counter()
    buildings: set = set()
    antennas: set = set()
    frequencies: set = set()
    wall_fractions: List[float] = []
    building_dim_warnings: List[str] = []
    pl_min = float("inf")
    pl_max = float("-inf")
    pl_sum = 0.0
    pl_count = 0
    written = 0
    skipped_existing = 0
    # Near-boundary transmitter coordinate corrections (bounded clipping); each
    # snapped coordinate is appended here so the summary can report the count.
    tx_clip_report: List[dict] = []

    for sample in samples:
        payload, shape = build_payload(
            raw_root, sample, resolution_m, tx_boundary_tol=tx_boundary_tol, clip_report=tx_clip_report
        )
        out_path = scene_dir / f"{sample.sample_id}.npz"
        rel_path = f"scenes/{sample.sample_id}.npz"

        # Duplicate sample_id protection (two inputs mapping to one target file).
        if sample.sample_id in seen_targets:
            if not (overwrite and seen_targets[sample.sample_id] == out_path):
                raise ValueError(
                    f"Duplicate sample_id '{sample.sample_id}' maps to {out_path}. "
                    "Pass --overwrite to allow rewriting the same target path."
                )
        seen_targets[sample.sample_id] = out_path

        # Optional cross-check against Building_Details W/H (warn-only).
        if sample.building_id not in checked_buildings:
            dims = load_building_dims(raw_root, sample.building_id)
            checked_buildings[sample.building_id] = dims
            if dims is not None and dims != (shape[1], shape[0]):
                building_dim_warnings.append(
                    f"B{sample.building_id}: Building_Details (W,H)={dims} != image (W,H)="
                    f"{(shape[1], shape[0])}"
                )

        shapes[shape] += 1
        buildings.add(sample.building_id)
        antennas.add(sample.antenna_id)
        frequencies.add(float(payload["frequency_hz"]))
        wall_fractions.append(float(np.asarray(payload["wall_mask"], dtype=np.float64).mean()))
        finite = payload["path_loss"][np.isfinite(payload["path_loss"])]
        if finite.size:
            pl_min = min(pl_min, float(finite.min()))
            pl_max = max(pl_max, float(finite.max()))
            pl_sum += float(finite.sum())
            pl_count += int(finite.size)

        if not dry_run:
            if out_path.exists() and not overwrite:
                skipped_existing += 1
            else:
                _write_npz(out_path, payload)
                written += 1

        rows.append(
            {
                "scene_id": sample.scene_id,
                "sample_id": sample.sample_id,
                "scene_path": rel_path,
                "frequency_hz": float(payload["frequency_hz"]),
                "resolution_m": float(resolution_m),
                "height": int(shape[0]),
                "width": int(shape[1]),
                "task_id": int(sample.task_id),
                "building_id": int(sample.building_id),
                "antenna_id": int(sample.antenna_id),
                "frequency_id": int(sample.frequency_id),
                "sample_index": int(sample.sample_index),
            }
        )

    manifest_path = out_root / "manifest.csv"
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(manifest_path, index=False)

    summary: Dict[str, object] = {
        "task": int(task),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "num_samples": len(samples),
        "num_converted": written,
        "skipped_existing": skipped_existing,
        "buildings": sorted(buildings),
        "antennas": sorted(antennas),
        "frequencies_hz": sorted(frequencies),
        "shapes": dict(shapes),
        "pathloss_min": pl_min if pl_count else None,
        "pathloss_max": pl_max if pl_count else None,
        "pathloss_mean": (pl_sum / pl_count) if pl_count else None,
        "wall_fraction_min": min(wall_fractions) if wall_fractions else None,
        "wall_fraction_mean": (sum(wall_fractions) / len(wall_fractions)) if wall_fractions else None,
        "wall_fraction_max": max(wall_fractions) if wall_fractions else None,
        "tx_boundary_tol": float(tx_boundary_tol),
        "tx_clipped": len(tx_clip_report),
        "tx_clip_samples": [c["sample_id"] for c in tx_clip_report],
        "building_dim_warnings": building_dim_warnings,
        "manifest": None if dry_run else str(manifest_path),
        "dry_run": bool(dry_run),
    }
    if verbose:
        _print_summary(summary)
    return summary


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _print_summary(summary: Dict[str, object]) -> None:
    task = int(summary["task"])
    shapes = summary["shapes"]
    shape_str = ", ".join(f"{h}x{w}:{n}" for (h, w), n in sorted(shapes.items())) or "n/a"
    freq_str = ", ".join(f"{v:g}" for v in summary["frequencies_hz"]) or "n/a"  # type: ignore[union-attr]
    print("=" * 64)
    print("ICASSP 2025 Indoor Pathloss conversion " + ("(DRY RUN)" if summary["dry_run"] else ""))
    print("-" * 64)
    print(f"  task                : {task} ({TASK_SPEC.get(task, '?')})")
    print(f"  raw_root            : {summary['raw_root']}")
    print(f"  out_root            : {summary['out_root']}")
    print(f"  samples processed   : {summary['num_samples']}")
    print(f"  NPZ written         : {summary['num_converted']}")
    print(f"  skipped (existing)  : {summary['skipped_existing']}")
    print(f"  buildings           : {summary['buildings']}")
    print(f"  antennas            : {summary['antennas']}")
    print(f"  frequencies (Hz)    : {freq_str}")
    print(f"  image shapes (HxW)  : {shape_str}")
    print(
        "  path-loss min/max/mean : "
        f"{_fmt(summary['pathloss_min'])} / {_fmt(summary['pathloss_max'])} / {_fmt(summary['pathloss_mean'])}"
    )
    print(
        "  wall fraction min/mean/max : "
        f"{_fmt(summary['wall_fraction_min'])} / {_fmt(summary['wall_fraction_mean'])} / "
        f"{_fmt(summary['wall_fraction_max'])}"
    )
    for warning in summary["building_dim_warnings"]:  # type: ignore[union-attr]
        print(f"  WARNING: {warning}")
    print(
        f"  Tx boundary corrections : {summary['tx_clipped']} coordinate(s) clipped "
        f"within tolerance {float(summary['tx_boundary_tol']):g} px; 0 exceeded tolerance."
    )
    if not summary["dry_run"]:
        print(f"  manifest            : {summary['manifest']}")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the official ICASSP 2025 Indoor Pathloss dataset to the WallPath-PI contract."
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Downloaded ICASSP 2025 indoor dataset root.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output directory.")
    parser.add_argument("--task", type=int, choices=(1, 2, 3), default=1, help="Challenge task to convert (default 1).")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many samples.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite NPZ files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and summarize without writing any files.")
    parser.add_argument(
        "--resolution-m", type=float, default=DEFAULT_RESOLUTION_M, help="Meters per pixel (challenge default 0.25)."
    )
    parser.add_argument(
        "--tx-boundary-tol",
        type=float,
        default=_TX_BOUNDARY_TOL,
        help=(
            "Pixel tolerance for snapping near-boundary official transmitter coordinates "
            "onto the nearest valid raster cell (default 5.0). Only coordinates at most "
            "this many pixels outside the raster are clipped; larger deviations still "
            "raise an error. This does not change the official X/Y coordinate convention."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    convert_dataset(
        raw_root=args.raw_root,
        out_root=args.out_root,
        task=args.task,
        limit=args.limit,
        overwrite=args.overwrite,
        resolution_m=args.resolution_m,
        tx_boundary_tol=args.tx_boundary_tol,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
