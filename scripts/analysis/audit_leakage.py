#!/usr/bin/env python
"""Leakage audit for WallPath-PI experiments.

Before trusting any reported numbers, this script verifies that the sparse-anchor
sampling, the train/val/eval split, and the evaluation masks are leakage-safe. It
reuses the exact production code paths (``make_sparse_mask``, ``_prepare_sample``,
``idw_map_from_mask``) so the audit reflects what training and evaluation actually
do, not a reimplementation.

CLI
---
python scripts/analysis/audit_leakage.py --config configs/config_icassp2025_task1_full_strong.yaml

By default the audit runs in ``--quick`` mode, which checks at most
``--max-samples`` samples (default 3) and, per sample, a reproducible random
subset of valid receiver pixels (``--max-feature-pixels``, default 500), reusing
cached base maps so it finishes in a few minutes for routine use. Pass ``--full``
to restore the original exhaustive all-pixel audit (default 8 samples). Both
modes run the identical set of leakage checks; only the amount of work differs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.baselines.idw import idw_map_from_mask  # noqa: E402
from wallpath_pi.data.dataset import SceneSample, WallPathManifest  # noqa: E402
from wallpath_pi.data.sparse import make_sparse_mask  # noqa: E402
from wallpath_pi.training.pipeline import _prepare_sample  # noqa: E402
from wallpath_pi.utils.config import load_config  # noqa: E402
from wallpath_pi.utils.paths import resolve_cache_root, resolve_csv_path, resolve_data_root, resolve_results_root  # noqa: E402


class CheckResult:
    def __init__(self, check_id: int, name: str, status: str, detail: str, severity: str = "error", metrics: Optional[Dict] = None):
        self.check_id = check_id
        self.name = name
        self.status = status  # PASS | FAIL | WARN | SKIP
        self.detail = detail
        self.severity = severity
        self.metrics = metrics or {}

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.check_id,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "detail": self.detail,
            "metrics": self.metrics,
        }


def _audit_config(cfg: Dict[str, object], cache_root: Optional[Path] = None) -> Dict[str, object]:
    """Return a side-effect-free copy of the config for prepare-based checks.

    When ``cache_root`` is given, the static base maps are cached on disk and
    reused across the repeated ``_prepare_sample`` calls (e.g. the original and
    perturbed feature tables, which differ only in dense labels). Base-map
    caching depends only on geometry-derived inputs and not on dense
    labels, so reusing it does not alter the leakage-audit comparison.
    """
    audit_cfg = dict(cfg)
    if cache_root is not None:
        audit_cfg["cache_features"] = True
        audit_cfg["_resolved_cache_root"] = str(cache_root)
    else:
        audit_cfg["cache_features"] = False
        audit_cfg.pop("_resolved_cache_root", None)
    return audit_cfg


def _load_samples(manifest: WallPathManifest, cap: int) -> List[SceneSample]:
    n = min(int(cap), len(manifest))
    return [manifest.sample(i) for i in range(n)]


def _feature_query_indices(sample: SceneSample, max_feature_pixels: int, quick: bool):
    """Reproducible random subset of valid-pixel row indices for quick mode.

    Returns ``(indices, n_valid)``. In full mode (or when the sample already has
    at most ``max_feature_pixels`` valid pixels) ``indices`` is ``None``, meaning
    "build features at every valid pixel". Otherwise it is a sorted, unique index
    array into the row-major valid-pixel ordering used by ``build_feature_table``,
    so the audit never materializes the full dense feature table for a sample.
    """
    valid = np.asarray(sample.valid_mask, dtype=bool)
    n_valid = int(valid.sum())
    if not quick or int(max_feature_pixels) <= 0 or n_valid <= int(max_feature_pixels):
        return None, n_valid
    from wallpath_pi.utils.hashing import stable_int_hash
    rng = np.random.default_rng(stable_int_hash("leakage_audit_pixels", sample.sample_id))
    idx = np.sort(rng.choice(np.arange(n_valid), size=int(max_feature_pixels), replace=False))
    return idx, n_valid


# Individual checks
def check_group_overlap(train_mf: WallPathManifest, val_mf: WallPathManifest, group_column: str) -> CheckResult:
    if group_column not in train_mf.df.columns or group_column not in val_mf.df.columns:
        return CheckResult(1, "Train/val group disjointness", "FAIL",
                           f"Group column '{group_column}' missing from a manifest.")
    train_groups = set(train_mf.df[group_column].astype(str))
    val_groups = set(val_mf.df[group_column].astype(str))
    overlap = sorted(train_groups & val_groups)
    if overlap:
        return CheckResult(1, "Train/val group disjointness", "FAIL",
                           f"{len(overlap)} '{group_column}' value(s) appear in BOTH train and val: {overlap[:10]}",
                           metrics={"overlap_count": len(overlap), "overlap_examples": overlap[:10]})
    return CheckResult(1, "Train/val group disjointness", "PASS",
                       f"No '{group_column}' overlap between train ({len(train_groups)}) and val ({len(val_groups)}).",
                       metrics={"train_groups": len(train_groups), "val_groups": len(val_groups)})


def check_anchor_subset_of_valid(samples: List[SceneSample], cfg: Dict[str, object], rate: float, seed: int) -> CheckResult:
    min_anchors = int(cfg.get("min_anchors", 1))
    offenders = []
    for s in samples:
        mask = make_sparse_mask(s.valid_mask, sampling_rate=rate, seed=seed, sample_id=s.sample_id, min_points=min_anchors)
        if bool((mask & ~s.valid_mask.astype(bool)).any()):
            offenders.append(s.sample_id)
    if offenders:
        return CheckResult(2, "Sparse anchors subset of valid_mask", "FAIL",
                           f"{len(offenders)} sample(s) place anchors on invalid cells: {offenders[:10]}")
    return CheckResult(2, "Sparse anchors subset of valid_mask", "PASS",
                       f"All {len(samples)} checked sample(s) keep anchors within valid_mask.")


def check_features_independent_of_dense_labels(samples: List[SceneSample], cfg: Dict[str, object], rate: float, seed: int, query_indices_by_id: Optional[Dict[str, object]] = None, cache_root: Optional[Path] = None) -> List[CheckResult]:
    """Checks 3 & 4: calibration uses only sparse anchors, and dense labels feed
    nothing but the metric computation.

    We perturb the dense target on non-anchor valid pixels and confirm that (3)
    the model-facing feature table and (4) every baseline prediction are unchanged.
    If a dense label leaked into feature extraction or calibration, it would move.

    In quick mode the feature table is built only at a reproducible random subset
    of valid pixels (``query_indices_by_id``); the perturbation still covers every
    non-anchor valid pixel, so the independence guarantee is checked on the
    subset rows without building the full dense table.
    """
    audit_cfg = _audit_config(cfg, cache_root=cache_root)
    rng = np.random.default_rng(12345)
    feature_offenders = []
    baseline_offenders = []
    anchor_offenders = []
    n_evaluated = 0
    for s in samples:
        qi = query_indices_by_id.get(s.sample_id) if query_indices_by_id else None
        prep = _prepare_sample(s, audit_cfg, sparse_rate=rate, sparse_seed=seed, query_indices=qi)
        fit_mask = prep.sparse_mask  # anchors actually used for calibration
        if bool((fit_mask & ~s.valid_mask.astype(bool)).any()):
            anchor_offenders.append(s.sample_id)
            continue
        # Perturb dense labels everywhere EXCEPT the anchors.
        perturb_zone = s.valid_mask.astype(bool) & ~fit_mask
        if not perturb_zone.any():
            continue
        n_evaluated += 1
        new_pl = s.path_loss.copy()
        new_pl[perturb_zone] += rng.normal(50.0, 25.0, size=int(perturb_zone.sum())).astype(new_pl.dtype)
        s2 = dataclasses.replace(s, path_loss=new_pl)
        prep2 = _prepare_sample(s2, audit_cfg, sparse_rate=rate, sparse_seed=seed, query_indices=qi)
        if not np.array_equal(prep.feature_table.X, prep2.feature_table.X):
            feature_offenders.append(s.sample_id)
        same_mw = np.allclose(prep.multi_wall_pred, prep2.multi_wall_pred, equal_nan=True)
        same_log = np.allclose(prep.log_distance_pred, prep2.log_distance_pred, equal_nan=True)
        same_idw = np.allclose(prep.idw_pred, prep2.idw_pred, equal_nan=True)
        if not (same_mw and same_log and same_idw):
            baseline_offenders.append(s.sample_id)

    if feature_offenders or anchor_offenders:
        detail3 = "Feature table moved with dense labels for: " + str((feature_offenders + anchor_offenders)[:10])
        check3 = CheckResult(3, "Sparse-anchor features independent of dense labels", "FAIL", detail3)
    else:
        check3 = CheckResult(3, "Sparse-anchor features independent of dense labels", "PASS",
                             f"Dense-label perturbation left the feature table unchanged on {n_evaluated} sample(s).")

    if baseline_offenders:
        detail4 = f"Baseline predictions moved with dense labels for: {baseline_offenders[:10]}"
        check4 = CheckResult(4, "Dense labels feed only metric computation", "FAIL", detail4)
    else:
        check4 = CheckResult(4, "Dense labels feed only metric computation", "PASS",
                             f"Dense-label perturbation left all baselines unchanged on {n_evaluated} sample(s).")
    return [check3, check4]


def check_no_nan_inf(train_samples: List[SceneSample], val_samples: List[SceneSample], cfg: Dict[str, object], rate: float, seed: int, query_indices_by_id: Optional[Dict[str, object]] = None, cache_root: Optional[Path] = None) -> CheckResult:
    audit_cfg = _audit_config(cfg, cache_root=cache_root)
    offenders = []
    for tag, samples in (("train", train_samples), ("val", val_samples)):
        for s in samples:
            valid = s.valid_mask.astype(bool)
            if valid.any() and not np.isfinite(s.path_loss[valid]).all():
                offenders.append((tag, s.sample_id, "nonfinite_target_on_valid"))
    # Feature tables must also be finite for the model.
    for s in val_samples:
        qi = query_indices_by_id.get(s.sample_id) if query_indices_by_id else None
        prep = _prepare_sample(s, audit_cfg, sparse_rate=rate, sparse_seed=seed, query_indices=qi)
        if not np.isfinite(prep.feature_table.X).all():
            offenders.append(("val", s.sample_id, "nonfinite_feature_row"))
    if offenders:
        return CheckResult(5, "No NaN/Inf on valid training/eval pixels", "FAIL",
                           f"{len(offenders)} issue(s): {offenders[:10]}")
    return CheckResult(5, "No NaN/Inf on valid training/eval pixels", "PASS",
                       "All checked valid targets and feature rows are finite.")


def check_anchor_counts(samples: List[SceneSample], cfg: Dict[str, object], rates: List[float], seeds: List[int]) -> CheckResult:
    min_anchors = int(cfg.get("min_anchors", 1))
    below = []
    warn_small = []
    for rate in rates:
        for seed in seeds:
            for s in samples:
                valid_count = int(s.valid_mask.astype(bool).sum())
                target = min(min_anchors, valid_count)
                mask = make_sparse_mask(s.valid_mask, sampling_rate=rate, seed=seed, sample_id=s.sample_id, min_points=min_anchors)
                count = int(mask.sum())
                if count < target:
                    below.append((s.sample_id, rate, seed, count, target))
                if valid_count < min_anchors:
                    warn_small.append((s.sample_id, valid_count))
    if below:
        return CheckResult(6, "Anchor counts >= min_anchors", "FAIL",
                           f"{len(below)} (sample,rate,seed) cases below the achievable minimum: {below[:8]}")
    if warn_small:
        uniq = sorted(set(warn_small))
        return CheckResult(6, "Anchor counts >= min_anchors", "WARN",
                           f"{len(uniq)} sample(s) have fewer valid pixels than min_anchors={min_anchors}: {uniq[:8]}",
                           severity="warn", metrics={"min_anchors": min_anchors})
    return CheckResult(6, "Anchor counts >= min_anchors", "PASS",
                       f"Every sample reaches min_anchors={min_anchors} across {len(rates)} rate(s) x {len(seeds)} seed(s).",
                       metrics={"min_anchors": min_anchors})


def check_eval_not_in_train(train_mf: WallPathManifest, eval_mf: WallPathManifest) -> CheckResult:
    if "sample_id" not in train_mf.df.columns or "sample_id" not in eval_mf.df.columns:
        return CheckResult(7, "Eval sample_ids absent from training", "FAIL", "Missing 'sample_id' column.")
    train_ids = set(train_mf.df["sample_id"].astype(str))
    eval_ids = set(eval_mf.df["sample_id"].astype(str))
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        return CheckResult(7, "Eval sample_ids absent from training", "FAIL",
                           f"{len(overlap)} eval sample_id(s) also used in training: {overlap[:10]}",
                           metrics={"overlap_count": len(overlap)})
    return CheckResult(7, "Eval sample_ids absent from training", "PASS",
                       f"None of {len(eval_ids)} eval sample_id(s) appear among {len(train_ids)} train sample_id(s).")


def check_idw_empty_anchor_behavior() -> CheckResult:
    """Check 8: with no anchors, IDW fills with a constant independent of the
    dense target mean, and can raise instead of silently filling."""
    rng = np.random.default_rng(7)
    value_map = rng.normal(100.0, 20.0, size=(12, 12)).astype(np.float32)
    empty_anchor = np.zeros((12, 12), dtype=bool)
    query = np.ones((12, 12), dtype=bool)
    const = 7.0
    out = idw_map_from_mask(value_map, empty_anchor, query, empty_constant=const)
    dense_mean = float(np.mean(value_map[query]))

    fills_constant = bool(np.allclose(out, const))
    independent_of_mean = not np.isclose(const, dense_mean)
    # Perturbing dense values must not change the empty-anchor fill.
    out2 = idw_map_from_mask(value_map + 500.0, empty_anchor, query, empty_constant=const)
    stable = bool(np.allclose(out2, const))
    # 'raise' mode must refuse to fill at all.
    raised = False
    try:
        idw_map_from_mask(value_map, empty_anchor, query, on_empty="raise")
    except ValueError:
        raised = True

    if fills_constant and independent_of_mean and stable and raised:
        return CheckResult(8, "IDW empty-anchor fill ignores dense target mean", "PASS",
                           "Empty-anchor IDW fills a caller constant, is invariant to dense values, and supports on_empty='raise'.")
    return CheckResult(8, "IDW empty-anchor fill ignores dense target mean", "FAIL",
                       f"fills_constant={fills_constant} independent_of_mean={independent_of_mean} stable={stable} raises={raised}")


# Driver
def run_leakage_audit(cfg: Dict[str, object], *, repo_root: Path = REPO_ROOT, out_dir: Optional[Path] = None,
                      max_samples: Optional[int] = None, quick: bool = True, max_feature_pixels: int = 500,
                      write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    # Quick mode defaults to a small sample budget so routine audits finish fast;
    # full mode keeps the original exhaustive default.
    if max_samples is None:
        max_samples = 3 if quick else 8
    max_samples = int(max_samples)
    data_root = resolve_data_root(cfg["data_root"], repo_root=repo_root)
    train_csv = resolve_csv_path(data_root, cfg["train_csv"])
    val_csv = resolve_csv_path(data_root, cfg["val_csv"])
    eval_csv = resolve_csv_path(data_root, cfg.get("eval_csv", cfg["val_csv"]))
    for label, path in (("train_csv", train_csv), ("val_csv", val_csv), ("eval_csv", eval_csv)):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}. Run the converter and scripts/make_internal_split.py first.")

    group_column = str(cfg.get("group_column", "scene_id"))
    rates = [float(x) for x in cfg.get("sparse_rates", [0.05])]
    seeds = [int(x) for x in cfg.get("sparse_seeds", [11])]
    primary_rate = rates[0]
    primary_seed = seeds[0]

    train_mf = WallPathManifest(train_csv, data_root)
    val_mf = WallPathManifest(val_csv, data_root)
    eval_mf = WallPathManifest(eval_csv, data_root)
    val_samples = _load_samples(val_mf, max_samples)
    train_samples = _load_samples(train_mf, max_samples)

    # Quick mode restricts feature construction to a reproducible random subset
    # of valid pixels per sample, so the audit never builds the full dense
    # feature table. Full mode keeps the exhaustive all-pixel behavior.
    mode = "quick" if quick else "full"
    query_indices_by_id: Optional[Dict[str, object]] = {} if quick else None
    n_checked_pixels = 0
    for s in val_samples:
        idx, n_valid = _feature_query_indices(s, max_feature_pixels, quick=quick)
        if query_indices_by_id is not None:
            query_indices_by_id[s.sample_id] = idx
        n_checked_pixels += int(idx.size) if idx is not None else int(n_valid)
    n_checked_samples = len(val_samples)
    if verbose:
        print(f"Leakage audit mode: {mode.upper()}")
        print(f"max_samples={max_samples} | max_feature_pixels={int(max_feature_pixels)}")
        print(f"Checked samples: {n_checked_samples} | checked feature pixels: {n_checked_pixels}")

    # Reuse static base maps across the repeated prepare calls (and across runs
    # when a persistent cache_root is configured). This is the dominant audit
    # cost on ICASSP-scale maps; caching is geometry-keyed so results are
    # unchanged. Fall back to a throwaway temp cache when none is configured.
    cache_ctx: Optional[tempfile.TemporaryDirectory] = None
    cache_root_cfg = cfg.get("cache_root")
    if cache_root_cfg:
        cache_root = resolve_cache_root(str(cache_root_cfg), repo_root=repo_root)
    else:
        cache_ctx = tempfile.TemporaryDirectory(prefix="wpi_audit_basemaps_")
        cache_root = Path(cache_ctx.name)
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        results: List[CheckResult] = [
            check_group_overlap(train_mf, val_mf, group_column),
            check_anchor_subset_of_valid(val_samples, cfg, primary_rate, primary_seed),
            *check_features_independent_of_dense_labels(val_samples, cfg, primary_rate, primary_seed, query_indices_by_id=query_indices_by_id, cache_root=cache_root),
            check_no_nan_inf(train_samples, val_samples, cfg, primary_rate, primary_seed, query_indices_by_id=query_indices_by_id, cache_root=cache_root),
            check_anchor_counts(val_samples, cfg, rates, seeds),
            check_eval_not_in_train(train_mf, eval_mf),
            check_idw_empty_anchor_behavior(),
        ]
    finally:
        if cache_ctx is not None:
            cache_ctx.cleanup()
    results.sort(key=lambda r: r.check_id)

    failed = [r for r in results if r.status == "FAIL" and r.severity == "error"]
    overall = "FAIL" if failed else "PASS"

    audit = {
        "config_path": str(cfg.get("_config_path", "")),
        "data_root": str(data_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "eval_csv": str(eval_csv),
        "group_column": group_column,
        "sparse_rates": rates,
        "sparse_seeds": seeds,
        "max_samples_checked": int(max_samples),
        "mode": mode,
        "max_samples": int(max_samples),
        "max_feature_pixels": int(max_feature_pixels),
        "n_checked_samples": int(n_checked_samples),
        "n_checked_pixels": int(n_checked_pixels),
        "n_train": int(len(train_mf)),
        "n_val": int(len(val_mf)),
        "n_eval": int(len(eval_mf)),
        "overall": overall,
        "checks": [r.to_dict() for r in results],
    }

    text = _render_text(audit, results)
    if verbose:
        print(text)

    if write_outputs:
        if out_dir is None:
            out_dir = resolve_results_root(cfg["results_root"], repo_root=repo_root) / "leakage_audit"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        (out_dir / "leakage_audit.txt").write_text(text, encoding="utf-8")
        audit["out_dir"] = str(out_dir)

    return audit


def _render_text(audit: Dict[str, object], results: List[CheckResult]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("WallPath-PI leakage audit")
    lines.append("=" * 70)
    lines.append(f"Config      : {audit['config_path']}")
    lines.append(f"Data root   : {audit['data_root']}")
    lines.append(f"Group column: {audit['group_column']}")
    lines.append(f"Mode        : {str(audit['mode']).upper()} (max_feature_pixels={audit['max_feature_pixels']})")
    lines.append(f"Samples     : train={audit['n_train']} val={audit['n_val']} eval={audit['n_eval']} (max_samples={audit['max_samples']})")
    lines.append(f"Checked     : samples={audit['n_checked_samples']} feature_pixels={audit['n_checked_pixels']}")
    lines.append("-" * 70)
    for r in results:
        lines.append(f"[{r.status:>4}] {r.check_id}. {r.name}")
        lines.append(f"        {r.detail}")
    lines.append("-" * 70)
    lines.append(f"OVERALL: {audit['overall']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit WallPath-PI experiments for data leakage.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <results_root>/leakage_audit).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples checked per split (default: 3 in quick mode, 8 in full mode).")
    parser.add_argument("--quick", dest="quick", action="store_true", default=True,
                        help="Fast audit on a reproducible random pixel subset per sample (default).")
    parser.add_argument("--full", dest="quick", action="store_false",
                        help="Exhaustive audit over every valid pixel (the original behavior).")
    parser.add_argument("--max-feature-pixels", type=int, default=500,
                        help="Max valid pixels per sample for feature checks in quick mode (default: 500).")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    audit = run_leakage_audit(cfg, out_dir=args.out_dir, max_samples=args.max_samples,
                              quick=bool(args.quick), max_feature_pixels=int(args.max_feature_pixels),
                              write_outputs=True, verbose=True)
    return 0 if audit["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
