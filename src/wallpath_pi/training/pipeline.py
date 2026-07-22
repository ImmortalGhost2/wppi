"""End-to-end training and evaluation pipeline.

``run_experiment`` and ``evaluate_saved_run`` are the entry points used by
``scripts/train.py`` and ``scripts/evaluate.py``.
``wallpath_pi.experiments.pipeline`` re-exports them for backward compatibility.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from wallpath_pi.baselines.idw import idw_map_from_mask
from wallpath_pi.baselines.propagation import fit_log_distance, fit_multi_wall
from wallpath_pi.data.dataset import SceneSample, WallPathManifest
from wallpath_pi.data.sparse import check_sparse_mask_validity, make_sparse_mask
from wallpath_pi.evaluation.metrics import aggregate_metric_rows, compute_metrics
from wallpath_pi.geometry.cache import load_or_compute_base_maps_for_sample
from wallpath_pi.geometry.features import FeatureTable, build_feature_table
from wallpath_pi.models.registry import feature_importance, make_regressor
from wallpath_pi.utils.paths import get_next_run_dir, resolve_cache_root, resolve_csv_path, resolve_data_root, resolve_results_root
from wallpath_pi.utils.plotter import plot_feature_importance, plot_model_comparison, plot_prediction_maps, plot_sparse_curves
from wallpath_pi.utils.run_summary import create_run_summary
from wallpath_pi.utils.seed import seed_everything


# Method registry.
# Direct (non-residual) random-forest baselines regress path loss directly from
# the feature table. ``all-feature`` variants use every feature (including the
# physics/baseline-prediction and residual-prior features); the ``geometry``
# variant is restricted to features that carry no path-loss target information,
# so it is a fair "plain ML on geometry" control. The residual learners
# (WallPath-PI) instead predict the residual over the multi-wall baseline and
# are left numerically unchanged.
ALL_FEATURE_DIRECT_METHODS = ("direct_rf", "direct_rf_all_features", "all_feature_rf")
GEOMETRY_DIRECT_METHODS = ("direct_rf_geometry",)
# Fair sparse-anchor direct baseline. Like the geometry control it regresses
# path loss directly, but it is additionally allowed to see sparse-anchor
# observation features (the anchor-fitted physics predictions, nearest-anchor
# distances, and anchor density). It is forbidden the WallPath residual shortcut
# features (``multi_wall_residual_idw_pred_db`` and the ``anchor_residual_*``
# features), so it stays a non-WallPath baseline rather than a residual learner.
SPARSE_ANCHOR_DIRECT_METHODS = ("direct_rf_sparse_anchor",)
DIRECT_RF_METHODS = ALL_FEATURE_DIRECT_METHODS + GEOMETRY_DIRECT_METHODS + SPARSE_ANCHOR_DIRECT_METHODS
# All-feature *Extra Trees* direct baseline. Identical setup to
# ``direct_rf_all_features`` (full feature table, direct path-loss target) but
# with an Extra Trees estimator instead of a random forest. It completes the
# direct-vs-residual / RF-vs-ExtraTrees 2x2 alongside ``direct_rf_all_features``
# (RF/direct), ``wallpath_rf`` (RF/residual) and ``wallpath_extra``
# (ExtraTrees/residual), all sharing the same feature access for a fair contrast.
ALL_FEATURE_DIRECT_EXTRA_METHODS = ("direct_extra_all_features", "all_feature_extra")
# Direct learners that score path loss directly (RF and Extra Trees families).
DIRECT_METHODS = DIRECT_RF_METHODS + ALL_FEATURE_DIRECT_EXTRA_METHODS
RESIDUAL_RF_METHODS = ("wallpath_rf", "wallpath_extra")
# Prefix marking config-declared feature-ablation variants. A method named
# ``ablation_*`` is trained exactly like ``wallpath_extra`` (Extra Trees on the
# multi-wall residual target) but its input columns are restricted to the named
# feature groups declared in ``train.feature_groups_by_method``. The prefix is
# reserved: no built-in method uses it, so the convention is additive and
# backward compatible.
ABLATION_METHOD_PREFIX = "ablation_"
# Physics-informed calibrated residual learner. It keeps the multi-wall
# residual-IDW reconstruction as a fixed base prediction and learns an additive
# correction whose target is ``true_path_loss - multi_wall_residual_idw_pred_db``.
# The correction model is an Extra Trees regressor restricted to a curated,
# inference-safe feature subset (geometry/material/anchor-distance features plus
# the analytic/fitted baseline-prediction maps); it never consumes the
# anchor-residual label features used by the other residual learners.
CALIBRATED_RESIDUAL_METHODS = ("wallpath_calibrated",)

# Base prediction map that the calibrated learner corrects.
_CALIBRATION_BASE_FEATURE = "multi_wall_residual_idw_pred_db"

# Exact feature names allowed as inputs to the calibrated correction model.
_CALIBRATION_ALLOWED_EXACT = frozenset({
    "x_norm", "y_norm",
    "distance_m", "log10_distance_m", "fspl_db",
    "wall_count", "wall_fraction", "los",
    "reflectance_sum", "transmittance_sum",
    "anchor_density",
    "fspl_pred_db", "log_distance_pred_db",
    "multi_wall_pred_db", "multi_wall_residual_idw_pred_db",
})


def _is_calibration_feature(name: str) -> bool:
    """Return True for features allowed in the calibrated correction model.

    The allowlist covers coordinate, distance, wall, material-count,
    transmittance/reflectance, anchor-density, nearest-anchor *distance*, and
    baseline-prediction features. It deliberately excludes the anchor-residual
    label features (``anchor_residual_idw`` and ``anchor_residual_nn_*``), which
    encode interpolated path-loss residuals rather than pure geometry.
    """
    n = str(name)
    if n in _CALIBRATION_ALLOWED_EXACT:
        return True
    if n.startswith("mat_") and n.endswith("_count"):
        return True
    if n.startswith("anchor_dist_px_nn_"):
        return True
    return False


def calibration_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Subset of ``feature_names`` allowed for ``wallpath_calibrated`` (order preserved)."""
    return [str(n) for n in feature_names if _is_calibration_feature(str(n))]


# Named feature groups for the WallPath-PI feature-ablation study. Every
# canonical feature produced by ``build_feature_table`` maps to exactly one
# group, so the union of all groups reproduces the full feature table. Unknown
# future features map to ``"other"`` and are simply excluded from any restricted
# subset, which keeps the ablation safe by construction.
FEATURE_GROUP_NAMES = (
    "distance",
    "coordinates",
    "los",
    "wall_count",
    "material",
    "sparse_anchor",
    "sparse_residual",
    "baseline_prediction",
)


def feature_group_of(name: str) -> str:
    """Return the ablation feature group a single feature name belongs to.

    The mapping mirrors the blocks assembled by
    :func:`wallpath_pi.geometry.features.build_feature_table`. ``fspl_db`` is a
    closed-form distance feature and belongs to ``distance``; the fitted/analytic
    ``*_pred_db`` maps belong to ``baseline_prediction``.
    """
    n = str(name)
    if n in ("x_norm", "y_norm"):
        return "coordinates"
    if n in ("distance_m", "log10_distance_m", "fspl_db"):
        return "distance"
    if n == "los":
        return "los"
    if n in ("wall_count", "wall_fraction"):
        return "wall_count"
    if (n.startswith("mat_") and n.endswith("_count")) or n in ("transmittance_sum", "reflectance_sum"):
        return "material"
    if n.endswith("_pred_db"):
        return "baseline_prediction"
    if n.startswith("anchor_dist_px_nn_") or n == "anchor_density":
        return "sparse_anchor"
    if n == "anchor_residual_idw" or n.startswith("anchor_residual_nn_"):
        return "sparse_residual"
    return "other"


def feature_group_columns(feature_names: Sequence[str], enabled_groups: Iterable[str]) -> list[str]:
    """Subset of ``feature_names`` whose group is in ``enabled_groups`` (order preserved)."""
    enabled = {str(g) for g in enabled_groups}
    return [str(n) for n in feature_names if feature_group_of(str(n)) in enabled]


def is_ablation_method(name: str) -> bool:
    """Return True for config-declared feature-ablation variants (``ablation_*``)."""
    return str(name).startswith(ABLATION_METHOD_PREFIX)


def _is_geometry_feature(name: str) -> bool:
    """Return True for features safe to use in the geometry-only baseline.

    Excludes any feature that encodes a path-loss value or a baseline-corrected
    prediction: baseline-prediction features (``*_pred_db``) and residual-label
    features (``anchor_residual_idw`` and ``anchor_residual_nn_*``). Pure
    geometry/material/anchor-distance features (e.g. ``fspl_db`` which is a
    closed-form function of distance and frequency, ``anchor_dist_px_nn_*``,
    ``anchor_density``) are kept.
    """
    n = str(name).lower()
    if "pred" in n:
        return False
    if "residual" in n:
        return False
    return True


def geometry_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Subset of ``feature_names`` allowed for ``direct_rf_geometry``."""
    return [str(n) for n in feature_names if _is_geometry_feature(str(n))]


# Residual shortcut features forbidden to the sparse-anchor direct baseline.
# These encode an interpolated path-loss residual over the multi-wall baseline
# (the WallPath shortcut); a fair non-WallPath baseline must not consume them.
_SPARSE_ANCHOR_FORBIDDEN_EXACT = frozenset({"multi_wall_residual_idw_pred_db"})


def _is_sparse_anchor_feature(name: str) -> bool:
    """Return True for features allowed in ``direct_rf_sparse_anchor``.

    The baseline keeps every geometry/material/ray feature, the anchor-fitted
    physics predictions (``fspl_pred_db``, ``log_distance_pred_db``,
    ``multi_wall_pred_db``), nearest-anchor *distances* (``anchor_dist_px_nn_*``)
    and ``anchor_density``. It excludes the residual shortcut features:
    ``multi_wall_residual_idw_pred_db`` and the anchor-residual labels
    (``anchor_residual_idw`` and ``anchor_residual_nn_*``).
    """
    n = str(name)
    if n in _SPARSE_ANCHOR_FORBIDDEN_EXACT:
        return False
    if n.startswith("anchor_residual"):
        return False
    return True


def sparse_anchor_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Subset of ``feature_names`` allowed for ``direct_rf_sparse_anchor`` (order preserved)."""
    return [str(n) for n in feature_names if _is_sparse_anchor_feature(str(n))]


def _select_feature_columns(X: np.ndarray, full_names: Sequence[str], wanted_names: Sequence[str]) -> np.ndarray:
    """Slice the columns of ``X`` to ``wanted_names`` (order preserved).

    Returns ``X`` unchanged when the wanted names match the full set, so the
    all-feature methods never pay a copy cost and stay numerically identical.
    """
    full = [str(n) for n in full_names]
    wanted = [str(n) for n in wanted_names]
    if wanted == full:
        return X
    index = {n: i for i, n in enumerate(full)}
    cols = [index[n] for n in wanted]
    return X[:, cols]


@dataclass
class SamplePrepared:
    sample: SceneSample
    sparse_mask: np.ndarray
    base_maps: Any
    fspl_pred: np.ndarray
    log_distance_pred: np.ndarray
    multi_wall_pred: np.ndarray
    idw_pred: np.ndarray
    multi_wall_residual_idw_pred: np.ndarray
    feature_table: FeatureTable
    log_distance_info: Any
    multi_wall_info: Any


def _prepare_sample(sample: SceneSample, config: Dict[str, Any], sparse_rate: float, sparse_seed: int, query_indices: np.ndarray | None = None) -> SamplePrepared:
    material_ids = [int(x) for x in config.get("material_ids", [1, 2, 3])]
    min_distance_m = float(config.get("min_distance_m", 0.25))
    base_cfg = config.get("baseline", {}) or {}
    feature_cfg = config.get("feature_set", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}
    min_anchors = int(config.get("min_anchors", 1))
    sparse_mask = make_sparse_mask(sample.valid_mask, sampling_rate=sparse_rate, seed=sparse_seed, sample_id=sample.sample_id, min_points=min_anchors)
    check_sparse_mask_validity(sparse_mask, sample.valid_mask)
    cache_enabled = bool(config.get("cache_features", False))
    cache_root = config.get("_resolved_cache_root") if cache_enabled else None
    base_maps = load_or_compute_base_maps_for_sample(
        sample,
        material_ids=material_ids,
        min_distance_m=min_distance_m,
        cache_root=cache_root,
        enabled=cache_enabled,
    )
    target = sample.path_loss
    fit_mask = sparse_mask & sample.valid_mask & np.isfinite(target)
    alpha = float(config.get("baseline_ridge_alpha", 1.0e-6))
    d0_m = float(base_cfg.get("d0_m", 1.0))
    log_pred, ld_info = fit_log_distance(base_maps.distance_m, target, fit_mask, d0_m=d0_m, min_distance_m=min_distance_m, alpha=alpha)
    mw_pred, mw_info = fit_multi_wall(
        base_maps.distance_m,
        base_maps.material_counts,
        target,
        fit_mask,
        material_ids=material_ids,
        d0_m=d0_m,
        min_distance_m=min_distance_m,
        alpha=alpha,
    )
    idw_power = float(base_cfg.get("idw_power", 2.0))
    idw_eps = float(base_cfg.get("idw_eps", 1.0e-6))
    # ``None``/"auto"/missing -> adaptive IDW chunking; an integer fixes the chunk.
    idw_query_chunk = base_cfg.get("idw_query_chunk_size", None)
    # Training-safe fallback for the degenerate empty-anchor case: the fitted
    # multi-wall baseline (a function of anchors + geometry), never the dense
    # evaluation target. With anchors present, IDW uses anchor-only fills.
    mw_valid = mw_pred[sample.valid_mask]
    idw_empty_fallback = float(np.nanmean(mw_valid)) if (sample.valid_mask.any() and np.isfinite(mw_valid).any()) else 0.0
    idw_pred = idw_map_from_mask(target, fit_mask, sample.valid_mask, power=idw_power, eps=idw_eps, empty_constant=idw_empty_fallback, query_chunk_size=idw_query_chunk)
    residual_map = target - mw_pred
    residual_idw = idw_map_from_mask(residual_map, fit_mask, sample.valid_mask, power=idw_power, eps=idw_eps, default_value=0.0, query_chunk_size=idw_query_chunk)
    mw_resid_idw = (mw_pred + residual_idw).astype(np.float32)
    baseline_pred_maps = {
        "fspl_pred_db": base_maps.fspl,
        "log_distance_pred_db": log_pred,
        "multi_wall_pred_db": mw_pred,
        "multi_wall_residual_idw_pred_db": mw_resid_idw,
    }
    table = build_feature_table(sample, base_maps, sparse_mask=fit_mask, multiwall_pred=mw_pred, material_ids=material_ids, feature_cfg=feature_cfg, baseline_pred_maps=baseline_pred_maps, query_indices=query_indices)
    return SamplePrepared(
        sample=sample,
        sparse_mask=fit_mask,
        base_maps=base_maps,
        fspl_pred=base_maps.fspl,
        log_distance_pred=log_pred,
        multi_wall_pred=mw_pred,
        idw_pred=idw_pred,
        multi_wall_residual_idw_pred=mw_resid_idw,
        feature_table=table,
        log_distance_info=ld_info,
        multi_wall_info=mw_info,
    )


def _select_training_indices(n_valid: int, max_points: int, seed: int, sample_id: str) -> np.ndarray | None:
    """Deterministically choose which valid pixels become training rows.

    Returns ``None`` to indicate "use every valid pixel" (the caller then builds
    the full dense table); otherwise a sorted index array into the row-major
    valid-pixel ordering. The selection is identical to the previous post-hoc
    row subsampling, so training data is unchanged.
    """
    n_valid = int(n_valid)
    max_points = int(max_points)
    if max_points <= 0 or n_valid <= max_points:
        return None
    from wallpath_pi.utils.hashing import stable_int_hash
    rng = np.random.default_rng(stable_int_hash("train_points", sample_id, seed))
    return np.sort(rng.choice(np.arange(n_valid), size=max_points, replace=False))


def _resolve_prepare_n_jobs(config: Dict[str, Any]) -> int:
    """Number of workers for sample-level feature preparation (default 1).

    Reads ``prepare_n_jobs`` from the flattened config (``train.prepare_n_jobs``)
    and falls back to ``system.prepare_n_jobs``. Defaults to 1 so behavior is
    serial and bit-for-bit unchanged unless the option is explicitly set.
    """
    val = config.get("prepare_n_jobs")
    if val is None:
        val = (config.get("system") or {}).get("prepare_n_jobs")
    try:
        n = int(val) if val is not None else 1
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def _resolve_prepare_backend(config: Dict[str, Any]) -> str:
    """joblib backend for parallel preparation (default ``loky`` process pool).

    Set ``train.prepare_backend`` (or ``system.prepare_backend``) to
    ``threading`` to avoid per-worker pickling/memory overhead on Windows.
    """
    val = config.get("prepare_backend")
    if val is None:
        val = (config.get("system") or {}).get("prepare_backend")
    return str(val) if val else "loky"


def _prepare_train_sample_rows(
    sample: SceneSample,
    config: Dict[str, Any],
    sparse_rate: float,
    sparse_seed: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the per-sample training rows (feature matrix, targets, base map).

    Top-level and picklable so it can run under a process-based joblib backend
    on Windows. The selected training pixels come from
    :func:`_select_training_indices`, which is seeded only by the global/sparse
    seed and the stable ``sample_id`` hash, so the rows are independent of worker
    order and identical to the serial path.
    """
    n_valid = int(np.asarray(sample.valid_mask, dtype=bool).sum())
    query_indices = _select_training_indices(n_valid, max_points=max_points, seed=sparse_seed, sample_id=sample.sample_id)
    prep = _prepare_sample(sample, config, sparse_rate, sparse_seed, query_indices=query_indices)
    table = prep.feature_table
    # Per-row multi-wall residual-IDW base prediction, in the same row order as
    # the feature table (the query mask nonzero order is row-major and matches
    # the feature rows). It never uses any dense evaluation label.
    base_resid_idw = prep.multi_wall_residual_idw_pred[table.valid_mask].astype(np.float32)
    return table.X, table.y, table.y - table.baseline_values, base_resid_idw, list(table.feature_names)



def _train_with_log(model: Any, X: np.ndarray, y: np.ndarray, *, method: str, sparse_rate: float, sparse_seed: int):
    """Train a model with terminal-visible progress logging."""
    rows = getattr(X, "shape", ["?"])[0]
    cols = getattr(X, "shape", ["?", "?"])[1] if len(getattr(X, "shape", [])) > 1 else "?"
    print(
        f"[train] start method={method} rate={sparse_rate:g} seed={sparse_seed} rows={rows:,} features={cols}",
        flush=True,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    dt = time.perf_counter() - t0
    print(
        f"[train] done  method={method} rate={sparse_rate:g} seed={sparse_seed} time={dt:.1f}s",
        flush=True,
    )
    return model


def _train_models_for_rate_seed(train_samples: List[SceneSample], config: Dict[str, Any], sparse_rate: float, sparse_seed: int) -> Tuple[Dict[str, Any], dict[str, list[str]]]:
    methods = [str(m) for m in config.get("methods", [])]
    max_points = int(config.get("max_train_points_per_sample", 1500))
    n_jobs = _resolve_prepare_n_jobs(config)
    desc = f"prep r={sparse_rate:g} s={sparse_seed}"
    # Sample preparation is independent and order-deterministic, so it can run in
    # parallel without changing any feature value. joblib.Parallel preserves the
    # input order, so the stacked matrices below are identical to serial mode.
    if n_jobs == 1:
        chunks = [
            _prepare_train_sample_rows(sample, config, sparse_rate, sparse_seed, max_points)
            for sample in tqdm(train_samples, desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0)
        ]
    else:
        backend = _resolve_prepare_backend(config)
        chunks = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(_prepare_train_sample_rows)(sample, config, sparse_rate, sparse_seed, max_points)
            for sample in tqdm(train_samples, desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0)
        )
    X_rows = [c[0] for c in chunks]
    y_direct = [c[1] for c in chunks]
    y_residual = [c[2] for c in chunks]
    base_resid_idw_rows = [c[3] for c in chunks]
    feature_names = chunks[-1][4] if chunks else None
    X = np.concatenate(X_rows, axis=0)
    yd = np.concatenate(y_direct, axis=0)
    yr = np.concatenate(y_residual, axis=0)
    base_resid_idw = np.concatenate(base_resid_idw_rows, axis=0)
    full_names = list(feature_names or [])
    geo_names = geometry_feature_names(full_names)
    X_geo = _select_feature_columns(X, full_names, geo_names)
    sa_names = sparse_anchor_feature_names(full_names)
    cal_names = calibration_feature_names(full_names)
    params = config.get("model_params", {}) or {}
    # Optional feature-ablation declarations: {ablation_method_name: [group, ...]}.
    # Only consulted for ``ablation_*`` methods; absent/empty leaves every
    # built-in method numerically unchanged.
    feature_groups_by_method = config.get("feature_groups_by_method", {}) or {}
    models: Dict[str, Any] = {}
    feature_names_by_method: dict[str, list[str]] = {}
    base_seed = int(config.get("seed", 42)) + int(sparse_seed)
    print(
        f"[train] prepared rate={sparse_rate:g} seed={sparse_seed} rows={X.shape[0]:,} full_features={X.shape[1]} methods={len(methods)}",
        flush=True,
    )
    for method in methods:
        if method in ALL_FEATURE_DIRECT_METHODS:
            # All-feature direct RF (the historical ``direct_rf`` control):
            # full feature table, direct path-loss target, original seed.
            m = make_regressor(method, params=params, seed=base_seed)
            _train_with_log(m, X, yd, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(full_names)
        elif method in ALL_FEATURE_DIRECT_EXTRA_METHODS:
            # All-feature direct Extra Trees: identical setup to
            # ``direct_rf_all_features`` (full feature table, direct path-loss
            # target) but an Extra Trees estimator. Distinct seed offset.
            m = make_regressor(method, params=params, seed=base_seed + 23)
            _train_with_log(m, X, yd, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(full_names)
        elif method in GEOMETRY_DIRECT_METHODS:
            # Geometry-only direct RF: excludes baseline-prediction and
            # residual-prior features so it is a fair plain-ML control.
            m = make_regressor(method, params=params, seed=base_seed + 47)
            _train_with_log(m, X_geo, yd, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(geo_names)
        elif method in SPARSE_ANCHOR_DIRECT_METHODS:
            # Sparse-anchor direct RF: geometry features plus the anchor-fitted
            # physics predictions and anchor-distance/density features, but none
            # of the residual shortcut features. Direct path-loss target.
            m = make_regressor(method, params=params, seed=base_seed + 71)
            X_sa = _select_feature_columns(X, full_names, sa_names)
            _train_with_log(m, X_sa, yd, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(sa_names)
        elif method == "wallpath_rf":
            m = make_regressor("wallpath_rf", params=params, seed=base_seed + 17)
            _train_with_log(m, X, yr, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models["wallpath_rf"] = m
            feature_names_by_method["wallpath_rf"] = list(full_names)
        elif method == "wallpath_extra":
            m = make_regressor("wallpath_extra", params=params, seed=base_seed + 31)
            _train_with_log(m, X, yr, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models["wallpath_extra"] = m
            feature_names_by_method["wallpath_extra"] = list(full_names)
        elif is_ablation_method(method):
            # Feature-ablation variant of WallPath-PI: Extra Trees on the
            # multi-wall residual target, with inputs restricted to the named
            # feature groups from ``feature_groups_by_method`` (full table when
            # unspecified). A deterministic per-name seed keeps each variant
            # reproducible and mutually distinct.
            from wallpath_pi.utils.hashing import stable_int_hash
            groups = feature_groups_by_method.get(method)
            abl_names = feature_group_columns(full_names, groups) if groups else list(full_names)
            X_abl = _select_feature_columns(X, full_names, abl_names)
            abl_seed = base_seed + 90 + int(stable_int_hash("feature_ablation", method) % 10000)
            m = make_regressor("wallpath_extra", params=params, seed=abl_seed)
            _train_with_log(m, X_abl, yr, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(abl_names)
        elif method in CALIBRATED_RESIDUAL_METHODS:
            # Calibrated correction: Extra Trees regressing the residual of the
            # multi-wall residual-IDW base (``true - base``) on the curated,
            # inference-safe feature subset.
            cal_target = yd - base_resid_idw
            m = make_regressor("wallpath_calibrated", params=params, seed=base_seed + 53)
            X_cal = _select_feature_columns(X, full_names, cal_names)
            _train_with_log(m, X_cal, cal_target, method=method, sparse_rate=sparse_rate, sparse_seed=sparse_seed)
            models[method] = m
            feature_names_by_method[method] = list(cal_names)
    return models, feature_names_by_method


def _chunked_predict(model: Any, X: np.ndarray, chunk_size: int) -> np.ndarray:
    """Predict over dense feature rows in bounded-size chunks.

    Numerically identical to ``model.predict(X)`` (rows are independent); the
    chunking only caps peak memory when scoring full-map feature tables.
    """
    n = int(X.shape[0])
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or n <= chunk_size:
        return model.predict(X).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        out[start:stop] = model.predict(X[start:stop]).astype(np.float32)
    return out


def _predictions_for_prepared(prep: SamplePrepared, models: Dict[str, Any], methods: Iterable[str], eval_chunk_size: int = 20000, feature_names_by_method: dict[str, list[str]] | None = None) -> Dict[str, np.ndarray]:
    H, W = prep.sample.path_loss.shape
    valid = prep.feature_table.valid_mask
    names_by_method = feature_names_by_method or {}

    def _model_columns(method: str) -> np.ndarray:
        wanted = names_by_method.get(method)
        if not wanted:
            return prep.feature_table.X
        full_names = list(prep.feature_table.feature_names)
        return _select_feature_columns(prep.feature_table.X, full_names, wanted)

    preds: Dict[str, np.ndarray] = {}
    base_map = {
        "fspl": prep.fspl_pred,
        "log_distance": prep.log_distance_pred,
        "multi_wall": prep.multi_wall_pred,
        "idw": prep.idw_pred,
        "multi_wall_residual_idw": prep.multi_wall_residual_idw_pred,
    }
    for method in methods:
        if method in base_map:
            preds[method] = base_map[method].astype(np.float32)
        elif method in DIRECT_METHODS and method in models:
            arr = np.full((H, W), np.nan, dtype=np.float32)
            arr[valid] = _chunked_predict(models[method], _model_columns(method), eval_chunk_size)
            preds[method] = arr
        elif (method in RESIDUAL_RF_METHODS or is_ablation_method(method)) and method in models:
            arr = np.full((H, W), np.nan, dtype=np.float32)
            residual = _chunked_predict(models[method], _model_columns(method), eval_chunk_size)
            arr[valid] = prep.feature_table.baseline_values + residual
            preds[method] = arr
        elif method in CALIBRATED_RESIDUAL_METHODS and method in models:
            # Final prediction = multi-wall residual-IDW base + learned correction.
            arr = np.full((H, W), np.nan, dtype=np.float32)
            correction = _chunked_predict(models[method], _model_columns(method), eval_chunk_size)
            arr[valid] = prep.multi_wall_residual_idw_pred[valid].astype(np.float32) + correction
            preds[method] = arr
    return preds


def _evaluate_samples(samples: List[SceneSample], config: Dict[str, Any], sparse_rate: float, sparse_seed: int, models: Dict[str, Any], feature_names_by_method: dict[str, list[str]] | None = None) -> Tuple[list[dict], dict[str, np.ndarray]]:
    methods = [str(m) for m in config.get("methods", [])]
    eval_cfg = config.get("evaluation", {}) or {}
    eval_chunk_size = int(eval_cfg.get("eval_chunk_size", 20000))
    export_npz = bool(eval_cfg.get("export_npz", True))
    rows: list[dict] = []
    primary_rate = float(eval_cfg.get("primary_sparse_rate", sparse_rate))
    primary_seed = int(eval_cfg.get("primary_sparse_seed", sparse_seed))
    # Keep a structurally valid but empty payload when NPZ export is disabled.
    # Dense maps are appended below only when export_npz is true.
    export_payload = {
        "predictions_by_method": {},
        "targets": [],
        "valid_masks": [],
        "wall_counts": [],
        "los_masks": [],
        "sparse_masks": [],
        "tx_positions": [],
        "scene_ids": [],
        "sample_ids": [],
    }
    print(f"[eval] start rate={sparse_rate:g} seed={sparse_seed} samples={len(samples)}", flush=True)
    for sample in tqdm(samples, desc=f"eval r={sparse_rate:g} s={sparse_seed}", leave=False, dynamic_ncols=True, mininterval=1.0):
        prep = _prepare_sample(sample, config, sparse_rate, sparse_seed)
        preds = _predictions_for_prepared(prep, models, methods, eval_chunk_size=eval_chunk_size, feature_names_by_method=feature_names_by_method)
        sample_rows: list[dict] = []
        for method, pred in preds.items():
            met = compute_metrics(
                pred,
                sample.path_loss,
                sample.valid_mask,
                los_mask=prep.base_maps.los,
                wall_count=prep.base_maps.wall_count,
                high_wall_count_threshold=int(eval_cfg.get("high_wall_count_threshold", 2)),
                clip_max_db=float(config.get("clip_max_db", 160.0)),
                clip_min_db=config.get("clip_min_db", None),
                treat_clip_max_as_clipped=bool(config.get("treat_clip_max_as_clipped", True)),
                anchor_count=int(prep.sparse_mask.sum()),
                min_region_pixels=int(eval_cfg.get("min_region_pixels", 5)),
                sparse_mask=prep.sparse_mask,
                wall_mask=sample.wall_mask,
            )
            row = {
                "method": method,
                "sparse_rate": float(sparse_rate),
                "sparse_seed": int(sparse_seed),
                "scene_id": sample.scene_id,
                "sample_id": sample.sample_id,
                "anchor_count": int(prep.sparse_mask.sum()),
                "valid_count": int(sample.valid_mask.sum()),
                "mw_anchor_count": int(prep.multi_wall_info.anchor_count),
                "mw_design_rank": int(prep.multi_wall_info.design_rank),
            }
            row.update(met)
            row["anchor_count"] = int(prep.sparse_mask.sum())
            row["valid_count"] = int(sample.valid_mask.sum())
            sample_rows.append(row)
        # Paired per-sample deltas against fixed baselines (negative => method is better).
        ref_rmse = {r["method"]: r["rmse"] for r in sample_rows}
        for r in sample_rows:
            mw = ref_rmse.get("multi_wall")
            if mw is not None and np.isfinite(mw) and np.isfinite(r["rmse"]):
                r["rmse_delta_vs_multi_wall"] = float(r["rmse"] - mw)
            else:
                r["rmse_delta_vs_multi_wall"] = float("nan")
            drf = ref_rmse.get("direct_rf")
            if drf is None:
                for _alias in ALL_FEATURE_DIRECT_METHODS:
                    if _alias in ref_rmse:
                        drf = ref_rmse[_alias]
                        break
            if drf is not None and np.isfinite(drf) and np.isfinite(r["rmse"]):
                r["rmse_delta_vs_direct_rf"] = float(r["rmse"] - drf)
            else:
                r["rmse_delta_vs_direct_rf"] = float("nan")
        rows.extend(sample_rows)
        if export_npz and np.isclose(sparse_rate, primary_rate) and int(sparse_seed) == primary_seed:
            export_payload["targets"].append(sample.path_loss.astype(np.float32))
            export_payload["valid_masks"].append(sample.valid_mask.astype(np.uint8))
            export_payload["wall_counts"].append(prep.base_maps.wall_count.astype(np.float32))
            export_payload["los_masks"].append(prep.base_maps.los.astype(np.float32))
            export_payload["sparse_masks"].append(prep.sparse_mask.astype(np.uint8))
            export_payload["tx_positions"].append(np.asarray(sample.tx_xy, dtype=np.float32).reshape(-1)[:2])
            export_payload["scene_ids"].append(sample.scene_id)
            export_payload["sample_ids"].append(sample.sample_id)
            for method, pred in preds.items():
                export_payload["predictions_by_method"].setdefault(method, []).append(pred.astype(np.float32))
    return rows, export_payload


def _object_array(arrays: list) -> np.ndarray:
    """Pack a list of differently shaped arrays into a 1-D object array."""
    obj = np.empty(len(arrays), dtype=object)
    for i, arr in enumerate(arrays):
        obj[i] = arr
    return obj


def _write_npz_payload(payload: dict, out_path: Path) -> None:
    methods = sorted(payload["predictions_by_method"].keys())
    if not methods or not payload["targets"]:
        return

    targets = payload["targets"]
    uniform = all(t.shape == targets[0].shape for t in targets)

    common: dict[str, np.ndarray] = {
        "methods": np.asarray(methods, dtype=str),
        "scene_ids": np.asarray(payload["scene_ids"], dtype=str),
        "sample_ids": np.asarray(payload["sample_ids"], dtype=str),
    }
    if payload.get("tx_positions"):
        # Tx positions are fixed-length (x, y) per sample and stack regardless.
        common["tx_positions"] = np.stack(payload["tx_positions"], axis=0).astype(np.float32)

    if uniform:
        # Same-shape maps keep the original dense layout so existing readers
        # (which load with allow_pickle=False) are unaffected.
        predictions = np.stack(
            [np.stack(payload["predictions_by_method"][m], axis=0) for m in methods], axis=0
        )
        extra: dict[str, np.ndarray] = {}
        if payload.get("sparse_masks"):
            extra["sparse_masks"] = np.stack(payload["sparse_masks"], axis=0).astype(np.uint8)
        np.savez_compressed(
            out_path,
            predictions=predictions.astype(np.float32),
            targets=np.stack(targets, axis=0).astype(np.float32),
            valid_masks=np.stack(payload["valid_masks"], axis=0).astype(np.uint8),
            wall_counts=np.stack(payload["wall_counts"], axis=0).astype(np.float32),
            los_masks=np.stack(payload["los_masks"], axis=0).astype(np.float32),
            **common,
            **extra,
        )
        return

    # Validation samples can come from buildings with different HxW shapes, which
    # cannot be packed into a single dense tensor. Store the per-sample maps as
    # object arrays plus an explicit shape table. Such files must be read with
    # ``np.load(..., allow_pickle=True)``; the ``variable_shapes`` flag signals it.
    n_samples = len(targets)
    predictions = np.empty((len(methods), n_samples), dtype=object)
    for mi, method in enumerate(methods):
        per_method = payload["predictions_by_method"][method]
        for si, arr in enumerate(per_method):
            predictions[mi, si] = arr.astype(np.float32)
    extra = {}
    if payload.get("sparse_masks"):
        extra["sparse_masks"] = _object_array(
            [m.astype(np.uint8) for m in payload["sparse_masks"]]
        )
    np.savez_compressed(
        out_path,
        predictions=predictions,
        targets=_object_array([t.astype(np.float32) for t in targets]),
        valid_masks=_object_array([m.astype(np.uint8) for m in payload["valid_masks"]]),
        wall_counts=_object_array([w.astype(np.float32) for w in payload["wall_counts"]]),
        los_masks=_object_array([l.astype(np.float32) for l in payload["los_masks"]]),
        shapes=np.asarray([t.shape for t in targets], dtype=np.int64),
        variable_shapes=np.asarray(True),
        **common,
        **extra,
    )


def run_experiment(config: Dict[str, Any]) -> Path:
    repo_root = Path(config.get("repo_root", Path.cwd())).resolve()
    data_root = resolve_data_root(config["data_root"], repo_root=repo_root)
    results_root = resolve_results_root(config["results_root"], repo_root=repo_root)
    train_csv = resolve_csv_path(data_root, config["train_csv"])
    val_csv = resolve_csv_path(data_root, config["val_csv"])
    eval_csv = resolve_csv_path(data_root, config.get("eval_csv", config["val_csv"]))
    config["_resolved_data_root"] = str(data_root)
    config["_resolved_train_csv"] = str(train_csv)
    config["_resolved_val_csv"] = str(val_csv)
    config["_resolved_eval_csv"] = str(eval_csv)
    if bool(config.get("cache_features", False)):
        cache_root = resolve_cache_root(config.get("cache_root"), repo_root=repo_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        config["_resolved_cache_root"] = str(cache_root)
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)), num_threads=config.get("num_threads"))
    run_dir = get_next_run_dir(results_root, str(config.get("experiment_name", "wallpath_pi")))
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()

    train_manifest = WallPathManifest(train_csv, data_root)
    val_manifest = WallPathManifest(val_csv, data_root)
    eval_manifest = WallPathManifest(eval_csv, data_root)
    train_samples = list(train_manifest.iter_samples())
    val_samples = list(val_manifest.iter_samples())
    eval_samples = list(eval_manifest.iter_samples())
    sparse_rates = [float(x) for x in config.get("sparse_rates", [0.05])]
    sparse_seeds = [int(x) for x in config.get("sparse_seeds", [11])]
    all_rows: list[dict] = []
    model_paths = []
    feature_rows = []
    primary_payload = None

    for rate in sparse_rates:
        for sp_seed in sparse_seeds:
            print(f"[stage] rate={rate:g} seed={sp_seed}: train", flush=True)
            models, feature_names_by_method = _train_models_for_rate_seed(train_samples, config, rate, sp_seed)
            for method, model in models.items():
                method_feature_names = feature_names_by_method.get(method, [])
                model_path = run_dir / "models" / f"{method}_rate_{str(rate).replace('.', 'p')}_seed_{sp_seed}.joblib"
                joblib.dump({"model": model, "feature_names": method_feature_names, "method": method, "sparse_rate": rate, "sparse_seed": sp_seed}, model_path)
                model_paths.append(str(model_path.resolve()))
                imp = feature_importance(model, method_feature_names)
                for feat, val in imp.items():
                    feature_rows.append({"method": method, "sparse_rate": rate, "sparse_seed": sp_seed, "feature": feat, "importance": float(val)})
            rows, payload = _evaluate_samples(eval_samples, config, rate, sp_seed, models, feature_names_by_method=feature_names_by_method)
            all_rows.extend(rows)
            eval_cfg = config.get("evaluation", {}) or {}
            if np.isclose(rate, float(eval_cfg.get("primary_sparse_rate", rate))) and int(sp_seed) == int(eval_cfg.get("primary_sparse_seed", sp_seed)):
                primary_payload = payload

    per_sample_csv = run_dir / "per_sample_metrics.csv"
    pd.DataFrame(all_rows).to_csv(per_sample_csv, index=False)
    aggregate_rows = aggregate_metric_rows(all_rows, group_keys=["method", "sparse_rate", "sparse_seed"])
    final_csv = run_dir / "final_evaluation_results.csv"
    pd.DataFrame(aggregate_rows).sort_values(["sparse_rate", "rmse"]).to_csv(final_csv, index=False)
    training_log_csv = run_dir / "training_log.csv"
    pd.DataFrame(aggregate_rows).sort_values(["sparse_rate", "rmse"]).to_csv(training_log_csv, index=False)
    feature_csv = run_dir / "feature_importances.csv"
    pd.DataFrame(feature_rows).to_csv(feature_csv, index=False)
    eval_cfg = config.get("evaluation", {}) or {}
    export_npz = bool(eval_cfg.get("export_npz", True))
    npz_path = run_dir / "eval_outputs_primary.npz"
    # Respect evaluation.export_npz: only write the primary NPZ when enabled.
    # CSV results, feature importances, and run_summary.json are unaffected.
    if export_npz and primary_payload is not None:
        _write_npz_payload(primary_payload, npz_path)

    plot_paths = {}
    try:
        plot_paths["sparse_curves_png"] = str(plot_sparse_curves(per_sample_csv, run_dir / "plots"))
        eval_cfg = config.get("evaluation", {}) or {}
        plot_paths["model_comparison_png"] = str(plot_model_comparison(per_sample_csv, run_dir / "plots", sparse_rate=float(eval_cfg.get("primary_sparse_rate", sparse_rates[0]))))
        if export_npz and npz_path.exists():
            plot_paths["prediction_maps_png"] = str(plot_prediction_maps(npz_path, run_dir / "plots"))
        fi = plot_feature_importance(feature_csv, run_dir / "plots")
        if fi:
            plot_paths["feature_importance_png"] = str(fi)
    except Exception as exc:
        plot_paths["plot_error"] = repr(exc)

    final_df = pd.read_csv(final_csv)
    primary_method = str(config.get("primary_method", "wallpath_rf"))
    primary_rows = final_df[final_df["method"] == primary_method]
    metrics = {}
    if not primary_rows.empty:
        best = primary_rows.sort_values("rmse").iloc[0].to_dict()
        metrics = {f"primary_{k}": v for k, v in best.items() if isinstance(v, (int, float, np.number)) or k in {"method"}}
    dataset_info = {
        "train_samples": int(len(train_samples)),
        "val_samples": int(len(val_samples)),
        "eval_samples": int(len(eval_samples)),
        "train_scenes": int(pd.read_csv(train_csv)[config.get("group_column", "scene_id")].astype(str).nunique()),
        "val_scenes": int(pd.read_csv(val_csv)[config.get("group_column", "scene_id")].astype(str).nunique()),
        "material_ids": [int(x) for x in config.get("material_ids", [1, 2, 3])],
        "sparse_rates": sparse_rates,
        "sparse_seeds": sparse_seeds,
        "methods": [str(m) for m in config.get("methods", [])],
    }
    artifacts = {
        "final_evaluation_results_csv": str(final_csv.resolve()),
        "per_sample_metrics_csv": str(per_sample_csv.resolve()),
        "training_log_csv": str(training_log_csv.resolve()),
        "feature_importances_csv": str(feature_csv.resolve()),
        "eval_outputs_primary_npz": str(npz_path.resolve()) if npz_path.exists() else None,
        "model_paths": model_paths,
    }
    artifacts.update(plot_paths)
    create_run_summary(
        experiment_name=str(config.get("experiment_name", "wallpath_pi")),
        output_dir=run_dir,
        config=config,
        metrics=metrics,
        artifacts=artifacts,
        dataset_info=dataset_info,
        timing_info={
            "start_time": started_at,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - start, 3),
        },
        extra={"primary_method": primary_method},
    )
    return run_dir


def evaluate_saved_run(run_dir: Path, csv_name: str | None = None, sparse_rate: float | None = None, sparse_seed: int | None = None) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}. Pass an existing run with --run_dir.")
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path.name} in {run_dir}. "
            f"This directory does not look like a completed WallPath-PI run."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    config["_full_config"] = yaml_raw = None
    repo_root = Path(config.get("repo_root", run_dir.parents[2] if len(run_dir.parents) > 2 else Path.cwd())).resolve()
    data_root = Path(config.get("_resolved_data_root", config.get("data_root", "data/sample_synthetic"))).resolve()
    eval_csv = resolve_csv_path(data_root, csv_name or config.get("eval_csv", config.get("val_csv", "val_split.csv")))
    eval_cfg = config.get("evaluation", {}) or {}
    rate = float(sparse_rate if sparse_rate is not None else eval_cfg.get("primary_sparse_rate", (config.get("sparse_rates") or [0.05])[0]))
    sp_seed = int(sparse_seed if sparse_seed is not None else eval_cfg.get("primary_sparse_seed", (config.get("sparse_seeds") or [11])[0]))
    manifest = WallPathManifest(eval_csv, data_root)
    samples = list(manifest.iter_samples())
    models = {}
    feature_names_by_method: dict[str, list[str]] = {}
    candidate_methods = list(dict.fromkeys([str(m) for m in (config.get("methods") or [])] + list(DIRECT_RF_METHODS) + list(RESIDUAL_RF_METHODS) + list(CALIBRATED_RESIDUAL_METHODS)))
    for method in candidate_methods:
        path = run_dir / "models" / f"{method}_rate_{str(rate).replace('.', 'p')}_seed_{sp_seed}.joblib"
        if path.exists():
            payload_obj = joblib.load(path)
            models[method] = payload_obj["model"]
            feature_names_by_method[method] = list(payload_obj.get("feature_names") or [])
    rows, payload = _evaluate_samples(samples, config, rate, sp_seed, models, feature_names_by_method=feature_names_by_method)
    tag = f"{Path(eval_csv).stem}_rate_{str(rate).replace('.', 'p')}_seed_{sp_seed}"
    out_csv = run_dir / f"final_evaluation_results_{tag}.csv"
    pd.DataFrame(aggregate_metric_rows(rows, group_keys=["method", "sparse_rate", "sparse_seed"])).sort_values(["sparse_rate", "rmse"]).to_csv(out_csv, index=False)
    out_per = run_dir / f"per_sample_metrics_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_per, index=False)
    out_npz = run_dir / f"eval_outputs_{tag}.npz"
    _write_npz_payload(payload, out_npz)
    return out_csv
