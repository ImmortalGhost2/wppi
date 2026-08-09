#!/usr/bin/env python
"""External point-wise measured 3.5 GHz path-loss validation for WallPath-PI.

This is a *standalone* validation on an external, point-wise tabular dataset
("Path Loss Dataset from Field Measurements at 3.5 GHz for the Fifth Generation
of Wireless Communications in Indoor Environments"). It is **not** an ICASSP
dense radio-map experiment: there is no map reconstruction, no ICASSP converter,
and no reuse of frozen ICASSP ``.joblib`` models (the feature spaces are
incompatible). The loader applies conservative, fully logged cleaning. It only tests the WallPath-PI *principle* -- a multi-wall /
propagation-informed baseline plus a learned residual correction -- on tabular
measured path loss.

Example
-------
python scripts/analysis/evaluate_external_3p5ghz.py \
  --data-root path/to/extracted/osfstorage-archive \
  --out-dir results/external_3p5ghz_measured_validation \
  --split leave_scenario_out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = ("Comms", "Library", "SSE")
CONFIGS = ("C1", "C2")
PRIMARY_METHOD = "wallpath_residual_extra"
PRIMARY_SPLIT = "leave_scenario_out"
FEWSHOT_PRIMARY_METHOD = "wallpath_fewshot_residual_extra"
ANCHOR_FRACTIONS_DEFAULT = (0.01, 0.05, 0.10)
ANCHOR_SEEDS_DEFAULT = (11, 22, 33)

# Canonical wall/structure feature columns (filled with 0 when a file omits one).
WALL_COLUMNS = (
    "num_brick_wall",
    "num_wood_wall",
    "num_glass_wall",
    "num_drywall",
    "num_column",
    "elevator",
)

CORE_WALL_COLUMNS = (
    "num_brick_wall",
    "num_wood_wall",
    "num_glass_wall",
    "num_drywall",
    "num_column",
)

EXPECTED_FILES = tuple(
    f"PL_{scenario}_{config}.csv"
    for scenario in SCENARIOS
    for config in CONFIGS
)

# Known suspicious anomaly: physically invalid indoor PL of -60 dB.
ANOMALY = {"source_file": "PL_Comms_C2.csv", "coord": "C-36", "pl_db": -60.0}
ANOMALY_TX_DBM = 10.0
ANOMALY_RX_DBM = -70.0  # PL = P_tx - P_rx = 10 - (-70) = 80 dB


# Column standardization
def _norm(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", "_", s.strip())
    return s.strip("_")


# Map normalized raw headers to canonical names. Robust to minor spacing diffs.
_COLUMN_MAP = {
    "coord": "coord", "coord.": "coord", "coordinate": "coord",
    "distance_m": "distance_m", "distance": "distance_m",
    "num_brick_wall": "num_brick_wall", "brick_wall": "num_brick_wall",
    "num_wood_wall": "num_wood_wall", "wood_wall": "num_wood_wall",
    "num_glass_wall": "num_glass_wall", "glass_wall": "num_glass_wall",
    "num_drywall": "num_drywall", "drywall": "num_drywall",
    "num_column": "num_column", "column": "num_column",
    "elevator": "elevator",
    "p_rx_dbm": "p_rx_dbm", "p_rx": "p_rx_dbm", "prx": "p_rx_dbm",
    "pl_db": "pl_db", "pl": "pl_db", "path_loss": "pl_db",
    "x": "rx_x", "y": "rx_y", "rx_x": "rx_x", "rx_y": "rx_y", "pos_x": "rx_x", "pos_y": "rx_y",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for col in df.columns:
        canon = _COLUMN_MAP.get(_norm(col))
        if canon:
            rename[col] = canon
    return df.rename(columns=rename)


def parse_scenario_config(filename: str) -> Tuple[str, str]:
    """Parse 'PL_Comms_C1.csv' -> ('Comms', 'C1')."""
    stem = Path(filename).stem
    m = re.match(r"PL_([A-Za-z]+)_C([12])$", stem)
    if not m:
        raise ValueError(f"Cannot parse scenario/config from file name '{filename}'.")
    return m.group(1), f"C{m.group(2)}"


# Loading and cleaning
def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_external_data(
    data_root: Path,
    *,
    fix_known_anomaly: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load, standardize, and conservatively clean the six PL_Data CSVs.

    Cleaning policy
    ---------------
    1. Require exactly the six expected scenario/configuration files.
    2. Drop rows with missing/non-finite target or non-positive distance.
    3. Treat an absent ``Elevator`` column as a structural zero because the
       source schema includes it only for the Library scenario.
    4. Drop rows with missing values in the five core wall-count columns.
       No unknown wall count is silently converted to zero.
    5. Remove the known Comms-C2/C-36 anomaly. The PL and received-power files
       agree on the impossible source value, so there is no independent basis
       for correcting it to a guessed replacement.
    """
    data_root = Path(data_root)
    pl_dir = data_root / "PL_Data"
    if not pl_dir.is_dir():
        pl_dir = data_root

    discovered = sorted(path.name for path in pl_dir.glob("PL_*.csv"))
    missing_files = sorted(set(EXPECTED_FILES) - set(discovered))
    unexpected_files = sorted(set(discovered) - set(EXPECTED_FILES))
    if missing_files or unexpected_files:
        raise RuntimeError(
            "External dataset file set mismatch. "
            f"Missing={missing_files}; unexpected={unexpected_files}"
        )

    frames: List[pd.DataFrame] = []
    loaded: List[Dict[str, object]] = []
    schema_defaults: List[Dict[str, object]] = []

    for filename in EXPECTED_FILES:
        fp = pl_dir / filename
        scenario, config = parse_scenario_config(fp.name)
        raw = standardize_columns(pd.read_csv(fp))

        required = {"coord", "distance_m", "pl_db", *CORE_WALL_COLUMNS}
        missing_columns = sorted(required - set(raw.columns))
        if missing_columns:
            raise ValueError(
                f"{fp.name}: missing required columns after standardization: "
                f"{missing_columns}"
            )

        raw = raw.copy()
        raw["scenario"] = scenario
        raw["config"] = config
        raw["source_file"] = fp.name
        raw["source_row"] = np.arange(len(raw), dtype=int)
        raw["coord"] = raw["coord"].astype("string").str.strip()

        if "elevator" not in raw.columns:
            raw["elevator"] = 0.0
            schema_defaults.append(
                {
                    "source_file": fp.name,
                    "column": "elevator",
                    "default_value": 0.0,
                    "reason": "column_absent_in_source_schema",
                }
            )

        frames.append(raw)
        loaded.append(
            {
                "file": fp.name,
                "md5": _file_md5(fp),
                "sha256": _file_sha256(fp),
                "rows": int(len(raw)),
            }
        )

    df = pd.concat(frames, ignore_index=True)
    rows_before = int(len(df))

    for column in ["pl_db", "distance_m", *WALL_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["row_uid"] = (
        df["source_file"].astype(str)
        + "::"
        + df["coord"].astype(str)
    )

    duplicate_mask = df.duplicated(subset=["row_uid"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["row_uid", "source_file", "coord", "source_row"],
        ]
        raise RuntimeError(
            "Duplicate source-file/coordinate identifiers detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    removed_records: List[Dict[str, object]] = []

    invalid_target_distance = (
        ~np.isfinite(df["pl_db"])
        | ~np.isfinite(df["distance_m"])
        | (df["distance_m"] <= 0)
    )
    invalid_details = df.loc[
        invalid_target_distance,
        [
            "row_uid",
            "source_file",
            "source_row",
            "coord",
            "distance_m",
            "pl_db",
        ],
    ].to_dict(orient="records")
    if invalid_details:
        removed_records.append(
            {
                "reason": "missing_or_invalid_target_or_distance",
                "count": int(len(invalid_details)),
                "rows": invalid_details,
            }
        )
    df = df.loc[~invalid_target_distance].copy()

    structural_missing = df[list(WALL_COLUMNS)].isna().any(axis=1)
    structural_details = df.loc[
        structural_missing,
        [
            "row_uid",
            "source_file",
            "source_row",
            "coord",
            "distance_m",
            "pl_db",
            *WALL_COLUMNS,
        ],
    ].copy()
    if not structural_details.empty:
        structural_details["missing_columns"] = (
            structural_details[list(WALL_COLUMNS)]
            .isna()
            .apply(
                lambda row: ",".join(
                    column
                    for column, is_missing in row.items()
                    if bool(is_missing)
                ),
                axis=1,
            )
        )
        removed_records.append(
            {
                "reason": "missing_structural_feature",
                "count": int(len(structural_details)),
                "rows": structural_details.to_dict(orient="records"),
            }
        )
    df = df.loc[~structural_missing].copy()

    anomaly_mask = (
        (df["source_file"] == ANOMALY["source_file"])
        & (df["coord"].astype(str) == ANOMALY["coord"])
    )
    anomaly_details = df.loc[
        anomaly_mask,
        [
            "row_uid",
            "source_file",
            "source_row",
            "coord",
            "distance_m",
            "pl_db",
        ],
    ].to_dict(orient="records")

    if anomaly_details:
        removed_records.append(
            {
                "reason": "known_physical_anomaly_removed",
                "count": int(len(anomaly_details)),
                "rows": anomaly_details,
                "note": (
                    "The PL and received-power source files are internally "
                    "consistent with the same physically impossible value; "
                    "there is no independent evidence for a corrected label."
                ),
            }
        )
    df = df.loc[~anomaly_mask].copy()

    if fix_known_anomaly:
        removed_records.append(
            {
                "reason": "deprecated_fix_flag_ignored",
                "count": 0,
                "note": (
                    "--fix-known-anomaly is retained only for CLI compatibility. "
                    "The anomaly is always removed under the conservative policy."
                ),
            }
        )

    if (df[list(WALL_COLUMNS)] < 0).any().any():
        bad = df.loc[
            (df[list(WALL_COLUMNS)] < 0).any(axis=1),
            ["row_uid", *WALL_COLUMNS],
        ]
        raise RuntimeError(
            "Negative structural counts remain after cleaning:\n"
            + bad.to_string(index=False)
        )

    if not np.isfinite(df[["pl_db", "distance_m", *WALL_COLUMNS]]).all().all():
        raise RuntimeError("Non-finite values remain after external-data cleaning.")

    df = df.reset_index(drop=True)

    report = {
        "files_loaded": loaded,
        "expected_files": list(EXPECTED_FILES),
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": int(len(df)),
        "rows_removed_total": int(rows_before - len(df)),
        "schema_defaults": schema_defaults,
        "removed": removed_records,
        "scenario_counts_after_cleaning": {
            str(key): int(value)
            for key, value in df.groupby("scenario").size().to_dict().items()
        },
        "config_counts_after_cleaning": {
            str(key): int(value)
            for key, value in df.groupby("config").size().to_dict().items()
        },
        "cleaning_policy": {
            "invalid_target_or_distance": "drop",
            "absent_elevator_column": "explicit_zero",
            "missing_core_wall_count": "drop",
            "known_anomaly": "drop_without_imputation",
        },
    }
    return df, report


# Features
def build_features(df: pd.DataFrame, split: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return (X, feature_names, log10_distance). Scenario one-hot is omitted for
    leave_scenario_out and config one-hot for leave_config_out to avoid leaking
    held-out identity."""
    dist = df["distance_m"].to_numpy(dtype=np.float64)
    log10_d = np.log10(np.maximum(dist, 1e-3))
    cols: List[np.ndarray] = [log10_d, dist]
    names: List[str] = ["log10_distance", "distance_m"]
    for c in WALL_COLUMNS:
        cols.append(df[c].to_numpy(dtype=np.float64))
        names.append(c)
    if split not in ("leave_scenario_out", "leave_scenario_fewshot"):
        for sc in SCENARIOS:
            cols.append((df["scenario"].to_numpy() == sc).astype(np.float64))
            names.append(f"scenario_{sc}")
    if split != "leave_config_out":
        for cf in CONFIGS:
            cols.append((df["config"].to_numpy() == cf).astype(np.float64))
            names.append(f"config_{cf}")
    return np.column_stack(cols), names, log10_d


def _mw_columns(names: Sequence[str]) -> List[int]:
    keep = {"log10_distance", *WALL_COLUMNS}
    return [i for i, n in enumerate(names) if n in keep]


# Methods
def predict_methods(X_tr, y_tr, X_te, names, *, seed: int, include_rf_residual: bool) -> Dict[str, np.ndarray]:
    ld = [names.index("log10_distance")]
    mw = _mw_columns(names)
    forest = dict(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=1, random_state=seed)
    out: Dict[str, np.ndarray] = {}

    out["log_distance"] = LinearRegression().fit(X_tr[:, ld], y_tr).predict(X_te[:, ld])

    mw_model = Ridge(alpha=1.0).fit(X_tr[:, mw], y_tr)
    out["multi_wall_linear"] = mw_model.predict(X_te[:, mw])

    out["direct_rf"] = RandomForestRegressor(**forest).fit(X_tr, y_tr).predict(X_te)
    out["direct_extra"] = ExtraTreesRegressor(**forest).fit(X_tr, y_tr).predict(X_te)

    mw_tr = mw_model.predict(X_tr[:, mw])
    resid = y_tr - mw_tr
    extra = ExtraTreesRegressor(**forest).fit(X_tr, resid)
    out["wallpath_residual_extra"] = out["multi_wall_linear"] + extra.predict(X_te)
    if include_rf_residual:
        rf = RandomForestRegressor(**forest).fit(X_tr, resid)
        out["wallpath_residual_rf"] = out["multi_wall_linear"] + rf.predict(X_te)
    return out


# Metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    ae = np.abs(err)
    if err.size == 0:
        return {k: float("nan") for k in ("rmse", "mae", "median_ae", "p90_ae", "p95_ae", "bias_db", "std_error_db")} | {"count": 0}
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(ae)),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "bias_db": float(np.mean(err)),
        "std_error_db": float(np.std(err)),
        "count": int(err.size),
    }


# Splits
def make_folds(df: pd.DataFrame, split: str, *, heldout: Sequence[str], seed: int, test_size: float):
    """Yield (fold_name, heldout_scenario, heldout_config, train_idx, test_idx)."""
    idx = np.arange(len(df))
    if split == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(idx)
        n_test = max(1, int(round(test_size * len(idx))))
        yield ("random", "", "", order[n_test:], order[:n_test])
    elif split == "leave_scenario_out":
        scens = list(heldout) if heldout else list(SCENARIOS)
        for sc in scens:
            te = idx[df["scenario"].to_numpy() == sc]
            tr = idx[df["scenario"].to_numpy() != sc]
            if te.size and tr.size:
                yield (f"holdout_{sc}", sc, "", tr, te)
    elif split == "leave_config_out":
        for cf in CONFIGS:
            te = idx[df["config"].to_numpy() == cf]
            tr = idx[df["config"].to_numpy() != cf]
            if te.size and tr.size:
                yield (f"holdout_{cf}", "", cf, tr, te)
    else:
        raise ValueError(f"Unknown split '{split}'.")


# Standard-split driver
def evaluate_external(df: pd.DataFrame, *, split: str, heldout: Sequence[str], seed: int, test_size: float,
                      include_rf_residual: bool) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    X, names, _ = build_features(df, split)
    y = df["pl_db"].to_numpy(dtype=np.float64)
    eval_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    methods: List[str] = []
    for fold_name, hs, hc, tr, te in make_folds(df, split, heldout=heldout, seed=seed, test_size=test_size):
        preds = predict_methods(X[tr], y[tr], X[te], names, seed=seed, include_rf_residual=include_rf_residual)
        methods = list(preds.keys())
        for method, yhat in preds.items():
            m = compute_metrics(y[te], yhat)
            eval_rows.append({"split": split, "random_seed": int(seed), "fold": fold_name,
                              "heldout_scenario": hs, "heldout_config": hc,
                              "method": method, **m, "train_count": int(tr.size),
                              "test_count": int(te.size)})
            sub = df.iloc[te]
            err = yhat - y[te]
            for j, ridx in enumerate(te):
                sample_rows.append({"row_uid": str(sub["row_uid"].iloc[j]),
                                    "scenario": sub["scenario"].iloc[j], "config": sub["config"].iloc[j],
                                    "source_file": sub["source_file"].iloc[j], "coord": str(sub["coord"].iloc[j]),
                                    "distance_m": float(df["distance_m"].iloc[ridx]), "target_pl_db": float(y[ridx]),
                                    "method": method, "pred_pl_db": float(yhat[j]), "error_db": float(err[j]),
                                    "abs_error_db": float(abs(err[j])), "split": split,
                                    "random_seed": int(seed), "fold": fold_name})
    return pd.DataFrame(eval_rows), pd.DataFrame(sample_rows), names


# Few-shot target-scenario calibration
def numeric_coords(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return (n, 2) numeric receiver coordinates if present and complete, else None."""
    if {"rx_x", "rx_y"}.issubset(df.columns):
        x = pd.to_numeric(df["rx_x"], errors="coerce")
        y = pd.to_numeric(df["rx_y"], errors="coerce")
        if x.notna().all() and y.notna().all():
            return np.column_stack([x.to_numpy(float), y.to_numpy(float)])
    return None


def select_anchor_mask(n: int, *, mode: str, fraction: float, count: int, seed: int) -> np.ndarray:
    """Deterministic sparse-anchor mask over n target points; >=1 anchor, >=1 left to score."""
    rng = np.random.default_rng(int(seed))
    k = max(1, int(round(fraction * n))) if mode == "fraction" else int(count)
    k = min(max(1, k), max(1, n - 1))
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def predict_fewshot(X_src, y_src, X_a, y_a, X_t, names, *, coords_a=None, coords_t=None,
                    seed: int, include_rf_residual: bool, include_direct: bool):
    """Few-shot calibration: source-trained models adapted with sparse target anchors.
    Target non-anchor labels are never used. Returns (preds, skipped_reasons)."""
    ld = [names.index("log10_distance")]
    mw = _mw_columns(names)
    forest = dict(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=1, random_state=seed)
    out: Dict[str, np.ndarray] = {}
    skipped: Dict[str, str] = {}

    ld_model = LinearRegression().fit(X_src[:, ld], y_src)
    mw_model = Ridge(alpha=1.0).fit(X_src[:, mw], y_src)
    out["log_distance"] = ld_model.predict(X_t[:, ld])
    out["multi_wall_linear"] = mw_model.predict(X_t[:, mw])

    ld_bias = float(np.mean(y_a - ld_model.predict(X_a[:, ld]))) if len(y_a) else 0.0
    mw_bias = float(np.mean(y_a - mw_model.predict(X_a[:, mw]))) if len(y_a) else 0.0
    out["bias_calibrated_log_distance"] = out["log_distance"] + ld_bias
    out["bias_calibrated_multi_wall"] = out["multi_wall_linear"] + mw_bias

    if coords_a is not None and coords_t is not None:
        d = np.sqrt(((coords_t[:, None, :] - coords_a[None, :, :]) ** 2).sum(-1)) + 1e-6
        w = 1.0 / d ** 2
        out["target_idw"] = (w @ y_a) / w.sum(1)
    else:
        skipped["target_idw"] = "no numeric receiver coordinates available"

    X_aug, y_aug = np.vstack([X_src, X_a]), np.concatenate([y_src, y_a])
    mw_aug = mw_model.predict(X_aug[:, mw])
    extra = ExtraTreesRegressor(**forest).fit(X_aug, y_aug - mw_aug)
    out["wallpath_fewshot_residual_extra"] = mw_model.predict(X_t[:, mw]) + extra.predict(X_t)
    if include_rf_residual:
        rf = RandomForestRegressor(**forest).fit(X_aug, y_aug - mw_aug)
        out["wallpath_fewshot_residual_rf"] = mw_model.predict(X_t[:, mw]) + rf.predict(X_t)
    if include_direct:
        out["direct_extra_with_anchors"] = (
            ExtraTreesRegressor(**forest).fit(X_aug, y_aug).predict(X_t)
        )
        out["direct_rf_with_anchors"] = (
            RandomForestRegressor(**forest).fit(X_aug, y_aug).predict(X_t)
        )
    return out, skipped


def evaluate_fewshot(
    df,
    *,
    heldout,
    settings,
    anchor_seeds,
    model_seed,
    include_rf_residual,
    include_direct,
):
    X, names, _ = build_features(df, "leave_scenario_fewshot")
    y = df["pl_db"].to_numpy(dtype=np.float64)
    coords = numeric_coords(df)
    scen = df["scenario"].to_numpy()
    eval_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    skipped: Dict[str, str] = {}
    scens = list(heldout) if heldout else list(SCENARIOS)
    for sc in scens:
        t_idx = np.where(scen == sc)[0]
        src = np.where(scen != sc)[0]
        if t_idx.size < 2 or src.size == 0:
            continue
        for mode, frac, cnt in settings:
            for sd in anchor_seeds:
                amask = select_anchor_mask(t_idx.size, mode=mode, fraction=frac, count=cnt, seed=sd)
                a_idx, te = t_idx[amask], t_idx[~amask]
                if a_idx.size == 0 or te.size == 0:
                    continue
                ca = coords[a_idx] if coords is not None else None
                ct = coords[te] if coords is not None else None
                preds, sk = predict_fewshot(
                    X[src],
                    y[src],
                    X[a_idx],
                    y[a_idx],
                    X[te],
                    names,
                    coords_a=ca,
                    coords_t=ct,
                    seed=model_seed,
                    include_rf_residual=include_rf_residual,
                    include_direct=include_direct,
                )
                skipped.update(sk)
                for method, yhat in preds.items():
                    m = compute_metrics(y[te], yhat)
                    eval_rows.append({"split": "leave_scenario_fewshot", "fold": f"holdout_{sc}",
                                      "heldout_scenario": sc, "method": method, "anchor_mode": mode,
                                      "anchor_fraction": float(frac) if mode == "fraction" else None,
                                      "anchor_count": int(cnt) if mode == "count" else None,
                                      "anchor_seed": int(sd),
                                      "random_seed": int(model_seed),
                                      "model_seed": int(model_seed),
                                      **m, "train_count": int(src.size + a_idx.size),
                                      "target_anchor_count": int(a_idx.size), "test_count": int(te.size)})
                    sub = df.iloc[te]
                    err = yhat - y[te]
                    for j in range(te.size):
                        sample_rows.append({"row_uid": str(sub["row_uid"].iloc[j]),
                                            "scenario": sub["scenario"].iloc[j], "config": sub["config"].iloc[j],
                                            "source_file": sub["source_file"].iloc[j], "coord": str(sub["coord"].iloc[j]),
                                            "distance_m": float(sub["distance_m"].iloc[j]), "target_pl_db": float(y[te][j]),
                                            "method": method, "pred_pl_db": float(yhat[j]), "error_db": float(err[j]),
                                            "abs_error_db": float(abs(err[j])), "split": "leave_scenario_fewshot",
                                            "fold": f"holdout_{sc}", "heldout_scenario": sc, "anchor_mode": mode,
                                            "anchor_fraction": float(frac) if mode == "fraction" else None,
                                            "anchor_count": int(cnt) if mode == "count" else None,
                                            "anchor_seed": int(sd),
                                            "random_seed": int(model_seed),
                                            "model_seed": int(model_seed),
                                            "is_anchor": False})
    return pd.DataFrame(eval_rows), pd.DataFrame(sample_rows), names, skipped


def pooled_metrics_from_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled point-level metrics from exported per-sample predictions."""
    if samples.empty:
        return pd.DataFrame()

    grouping = ["split", "method"]
    optional = [
        "anchor_mode",
        "anchor_fraction",
        "anchor_count",
        "anchor_seed",
        "random_seed",
        "model_seed",
    ]
    grouping += [column for column in optional if column in samples.columns]

    rows: List[Dict[str, object]] = []
    for keys, group in samples.groupby(grouping, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row.update(
            compute_metrics(
                group["target_pl_db"].to_numpy(float),
                group["pred_pl_db"].to_numpy(float),
            )
        )
        row["unique_test_points"] = int(group["row_uid"].nunique())
        rows.append(row)

    return pd.DataFrame(rows)



def _aggregate(results: pd.DataFrame) -> Dict[str, object]:
    agg: Dict[str, object] = {}
    for method, g in results.groupby("method"):
        agg[method] = {f"{k}_mean": float(g[k].mean()) for k in ("rmse", "mae", "p90_ae")}
        agg[method].update({f"{k}_std": float(g[k].std(ddof=0)) for k in ("rmse", "mae", "p90_ae")})
    return agg


def _fewshot_settings(args: argparse.Namespace) -> List[Tuple[str, float, int]]:
    settings: List[Tuple[str, float, int]] = []
    if args.anchor_fractions:
        settings += [("fraction", float(f), 0) for f in args.anchor_fractions]
    if args.anchor_counts:
        settings += [("count", 0.0, int(c)) for c in args.anchor_counts]
    if not settings:
        settings = [("fraction", float(f), 0) for f in ANCHOR_FRACTIONS_DEFAULT]
    return settings


def run(args: argparse.Namespace) -> Dict[str, object]:
    df, report = load_external_data(Path(args.data_root), fix_known_anomaly=args.fix_known_anomaly)
    common = {
        "experiment": "external_3p5ghz_measured_validation",
        "random_seed": int(args.random_seed),
        "dataset_files": [f["file"] for f in report["files_loaded"]],
        "file_fingerprints": report["files_loaded"],
        "rows_before_cleaning": report["rows_before_cleaning"],
        "rows_after_cleaning": report["rows_after_cleaning"],
        "rows_removed_total": report["rows_removed_total"],
        "cleaning_policy": report["cleaning_policy"],
        "schema_defaults": report["schema_defaults"],
        "scenario_counts_after_cleaning": report["scenario_counts_after_cleaning"],
        "config_counts_after_cleaning": report["config_counts_after_cleaning"],
        "removed_rows": report["removed"],
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }

    if args.split == "leave_scenario_fewshot":
        settings = _fewshot_settings(args)
        anchor_seeds = args.anchor_seeds or list(ANCHOR_SEEDS_DEFAULT)
        results, samples, names, skipped = evaluate_fewshot(
            df,
            heldout=args.heldout_scenario,
            settings=settings,
            anchor_seeds=anchor_seeds,
            model_seed=args.random_seed,
            include_rf_residual=args.include_rf_residual,
            include_direct=args.include_direct_anchors,
        )
        best = {}
        for (f, m, fr, ct), g in results.groupby(["fold", "anchor_mode", "anchor_fraction", "anchor_count"], dropna=False):
            tag = f"frac{float(fr):g}" if m == "fraction" else f"count{int(ct)}"
            best[f"{f}|{tag}"] = str(g.loc[g["rmse"].idxmin(), "method"])
        summary = {**common, "split": "leave_scenario_fewshot", "feature_columns": names,
                   "methods": sorted(results["method"].unique().tolist()),
                   "primary_method": FEWSHOT_PRIMARY_METHOD, "primary_split": "leave_scenario_fewshot",
                   "anchor_modes": sorted({s[0] for s in settings}),
                   "anchor_fractions": [s[1] for s in settings if s[0] == "fraction"],
                   "anchor_counts": [s[2] for s in settings if s[0] == "count"],
                   "anchor_seeds": anchor_seeds,
                   "fewshot_model_seed": int(args.random_seed),
                   "anchor_selection_seed_is_separate_from_model_seed": True,
                   "evaluation_excludes_target_anchors": True,
                   "skipped_methods": skipped,
                   "best_method_by_fold_setting": best, "aggregate": _aggregate(results)}
    else:
        results, samples, names = evaluate_external(
            df, split=args.split, heldout=args.heldout_scenario, seed=args.random_seed,
            test_size=args.test_size, include_rf_residual=args.include_rf_residual)
        best_by_fold = {f: str(g.loc[g["rmse"].idxmin(), "method"]) for f, g in results.groupby("fold")}
        summary = {**common, "feature_columns": names, "split": args.split,
                   "methods": sorted(results["method"].unique().tolist()),
                   "primary_method": PRIMARY_METHOD, "primary_split": args.split,
                   "best_method_by_fold": best_by_fold, "aggregate": _aggregate(results)}

    pooled = pooled_metrics_from_samples(samples)
    summary["pooled_metrics"] = (
        pooled.to_dict(orient="records") if not pooled.empty else []
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "final_evaluation_results.csv", index=False)
    samples.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    pooled.to_csv(out_dir / "pooled_evaluation_results.csv", index=False)
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[external-3p5ghz] split={args.split} methods={summary['methods']} -> {out_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="External point-wise measured 3.5 GHz validation of the WallPath-PI principle.")
    p.add_argument("--data-root", type=Path, required=True, help="Extracted osfstorage-archive root (containing PL_Data/).")
    p.add_argument("--out-dir", type=Path, default=Path("results/external_3p5ghz_measured_validation"))
    p.add_argument("--split", choices=("random", "leave_scenario_out", "leave_config_out", "leave_scenario_fewshot"), default="leave_scenario_out")
    p.add_argument("--heldout-scenario", dest="heldout_scenario", action="append", default=[], choices=list(SCENARIOS),
                   help="Restrict scenario splits to these scenarios (repeatable; default all three).")
    p.add_argument("--random-seed", type=int, default=11, help="Model seed. In few-shot mode this is fixed across all anchor selections.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--anchor-fractions", type=float, nargs="+", default=None, help="Few-shot target anchor fractions, e.g. 0.01 0.05 0.10.")
    p.add_argument("--anchor-counts", type=int, nargs="+", default=None, help="Few-shot target anchor counts, e.g. 5 10 20.")
    p.add_argument("--anchor-seeds", type=int, nargs="+", default=list(ANCHOR_SEEDS_DEFAULT), help="Few-shot target-anchor selection seeds; independent of --random-seed.")
    p.add_argument(
        "--fix-known-anomaly",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The known C-36 anomaly is always "
            "removed because no independent source supports a corrected value."
        ),
    )
    p.add_argument("--include-rf-residual", action="store_true", default=True, help="Also evaluate the RandomForest residual variant.")
    p.add_argument("--include-direct-anchors", action="store_true", default=True, help="Also evaluate matched direct ExtraTrees and RandomForest models with anchors (few-shot).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
