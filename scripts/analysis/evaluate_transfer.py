#!/usr/bin/env python
"""Zero-parameter cross-task transfer evaluation for WallPath-PI.

Evaluate models that were *already trained* in a completed source run on a
different converted target dataset, without any retraining or fine-tuning. This
answers "how well do Task-1-trained models reconstruct Task-2 / Task-3 radio
maps from sparse anchors?".

Transfer protocol
-----------------
* The frozen source models live under ``<source-run>/models/*.joblib``; each
  artifact stores the fitted estimator, the exact ``feature_names`` it consumes,
  and the ``(method, sparse_rate, sparse_seed)`` it was trained at.
* The target dataset (``--target-config``) is loaded through the *same* feature
  pipeline. Target sparse anchors are simulated from the target dense path-loss
  map (sparse radio-map reconstruction is the task, so anchor observations are
  legitimate inputs); the dense target map is used only to score metrics.
* Non-trained analytic baselines (``log_distance``, ``multi_wall``, ``idw``,
  ``multi_wall_residual_idw``) are recomputed directly on the target data.
* Nothing is retrained. ``run_summary.json`` records ``retrained = false`` and
  the train source / target dataset for unambiguous provenance.

CLI
---
python scripts/analysis/evaluate_transfer.py \
    --source-run <run-dir-containing-models> \
    --target-config configs/config_icassp2025_task2_transfer_unseen_buildings.yaml \
    --out-dir results/transfer/task1_to_task2

``<run-dir-containing-models>`` must be a locally generated run directory that
contains ``models/*.joblib``; the compact tracked evidence runs do not carry
them. See ``docs/RUN_COMMANDS.md`` for a command that locates such a run
automatically.

Optional ``--methods``, ``--sparse-rates`` and ``--sparse-seeds`` override the
defaults (source ML methods + baselines; rates/seeds present in the source
models). A requested model file that is missing for a rate/seed is skipped with
a warning rather than aborting the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.evaluation.metrics import aggregate_metric_rows
from wallpath_pi.training.pipeline import (
    CALIBRATED_RESIDUAL_METHODS,
    GEOMETRY_DIRECT_METHODS,
    SPARSE_ANCHOR_DIRECT_METHODS,
    _evaluate_samples,
    _prepare_sample,
    calibration_feature_names,
    feature_group_columns,
    geometry_feature_names,
    is_ablation_method,
    sparse_anchor_feature_names,
)
from wallpath_pi.utils.config import load_config
from wallpath_pi.utils.paths import resolve_cache_root, resolve_csv_path, resolve_data_root
from wallpath_pi.utils.plotter import plot_model_comparison, plot_sparse_curves
from wallpath_pi.utils.run_summary import create_run_summary
from wallpath_pi.utils.seed import seed_everything

# Analytic baselines recomputed on the target data (no joblib artifact needed).
NON_TRAINED_BASELINES: Tuple[str, ...] = (
    "log_distance",
    "multi_wall",
    "idw",
    "multi_wall_residual_idw",
)

# Presentation order for learned (transferred) methods.
ML_METHOD_ORDER: Tuple[str, ...] = (
    "direct_rf_geometry",
    "direct_rf_sparse_anchor",
    "direct_rf_all_features",
    "direct_extra_all_features",
    "wallpath_rf",
    "wallpath_extra",
    "wallpath_calibrated",
)


def _round_rate(rate: float) -> float:
    return round(float(rate), 9)


def model_filename(method: str, rate: float, seed: int) -> str:
    """Source-run model filename for a ``(method, rate, seed)`` (inverse of training)."""
    return f"{method}_rate_{str(rate).replace('.', 'p')}_seed_{int(seed)}.joblib"


def parse_model_filename(stem: str) -> Optional[Tuple[str, float, int]]:
    """Parse a ``<method>_rate_<rate>_seed_<seed>`` stem into ``(method, rate, seed)``.

    Method names may contain underscores; the ``_rate_`` / ``_seed_`` separators
    are unambiguous because no built-in method name contains them. Returns
    ``None`` when the stem does not match the expected pattern.
    """
    s = str(stem)
    if "_rate_" not in s or "_seed_" not in s:
        return None
    method, _, rest = s.partition("_rate_")
    ratep, _, seedp = rest.partition("_seed_")
    if not method or not ratep or not seedp:
        return None
    try:
        rate = float(ratep.replace("p", "."))
        seed = int(seedp)
    except ValueError:
        return None
    return method, rate, seed


def discover_source_models(models_dir: Path) -> Dict[Tuple[str, float, int], Path]:
    """Index ``models_dir`` as ``{(method, rate, seed): path}`` from filenames."""
    index: Dict[Tuple[str, float, int], Path] = {}
    for path in sorted(Path(models_dir).glob("*.joblib")):
        parsed = parse_model_filename(path.stem)
        if parsed is None:
            continue
        method, rate, seed = parsed
        index[(method, _round_rate(rate), int(seed))] = path
    return index


def resolve_feature_names(
    meta_names: Optional[Sequence[str]],
    method: str,
    full_names: Sequence[str],
    feature_groups_by_method: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    """Feature columns a source model expects, preferring the saved names.

    When the artifact stored ``feature_names`` they are authoritative. Otherwise
    the exact per-method selection used during training is reproduced from the
    target table's full feature names, so the transferred model receives the same
    columns it was fit on.
    """
    if meta_names:
        return [str(n) for n in meta_names]
    m = str(method)
    full = [str(n) for n in full_names]
    if m in GEOMETRY_DIRECT_METHODS:
        return geometry_feature_names(full)
    if m in SPARSE_ANCHOR_DIRECT_METHODS:
        return sparse_anchor_feature_names(full)
    if m in CALIBRATED_RESIDUAL_METHODS:
        return calibration_feature_names(full)
    if is_ablation_method(m):
        groups = (feature_groups_by_method or {}).get(m)
        return feature_group_columns(full, groups) if groups else list(full)
    # All-feature direct (RF / Extra Trees) and residual learners use everything.
    return list(full)


def load_source_config(source_run: Path) -> Tuple[Optional[Dict], str]:
    """Best-effort load of the source run's config for labelling/metadata.

    Prefers ``resolved_config.yaml`` (a nested snapshot, loadable like any
    config), then falls back to the flattened ``config`` block inside
    ``run_summary.json``. Returns ``(config_or_none, origin)``.
    """
    source_run = Path(source_run)
    rc = source_run / "resolved_config.yaml"
    if rc.exists():
        try:
            return load_config(rc), "resolved_config.yaml"
        except Exception:
            try:
                raw = yaml.safe_load(rc.read_text(encoding="utf-8")) or {}
                exp = (raw.get("experiment", {}) or {}).get("experiment_name")
                return {"_full_config": raw, "experiment_name": exp}, "resolved_config.yaml(raw)"
            except Exception:
                pass
    rs = source_run / "run_summary.json"
    if rs.exists():
        try:
            data = json.loads(rs.read_text(encoding="utf-8"))
            cfg = dict(data.get("config") or {})
            cfg.setdefault("experiment_name", data.get("experiment_name"))
            return cfg, "run_summary.json"
        except Exception:
            pass
    return None, "none"


def _ordered_ml(methods) -> List[str]:
    return sorted(
        {str(m) for m in methods},
        key=lambda m: (ML_METHOD_ORDER.index(m) if m in ML_METHOD_ORDER else len(ML_METHOD_ORDER), m),
    )


def evaluate_transfer(
    source_run: Path,
    target_config: Path,
    out_dir: Path,
    methods: Optional[Sequence[str]] = None,
    sparse_rates: Optional[Sequence[float]] = None,
    sparse_seeds: Optional[Sequence[int]] = None,
) -> Path:
    """Run frozen learned-model transfer and write the result artifacts. Target-map sparse anchors may fit analytic baseline and anchor-derived components, but learned estimator weights are not updated."""
    started = time.time()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_run = Path(source_run).expanduser().resolve()
    target_config = Path(target_config).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    models_dir = source_run / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Source run has no models directory: {models_dir}")
    index = discover_source_models(models_dir)
    if not index:
        raise FileNotFoundError(f"No '*.joblib' source models found under {models_dir}")
    source_cfg, source_origin = load_source_config(source_run)

    # Resolve the target config and dataset through the standard pipeline paths.
    config = load_config(target_config)
    config["repo_root"] = str(REPO_ROOT)
    seed_everything(
        int(config.get("seed", 42)),
        deterministic=bool(config.get("deterministic", True)),
        num_threads=config.get("num_threads"),
    )
    data_root = resolve_data_root(config["data_root"], repo_root=REPO_ROOT)
    config["_resolved_data_root"] = str(data_root)
    eval_csv = resolve_csv_path(data_root, config.get("eval_csv", config.get("val_csv")))
    config["_resolved_eval_csv"] = str(eval_csv)
    if bool(config.get("cache_features", False)):
        cache_root = resolve_cache_root(config.get("cache_root"), repo_root=REPO_ROOT)
        cache_root.mkdir(parents=True, exist_ok=True)
        config["_resolved_cache_root"] = str(cache_root)
    target_samples = list(WallPathManifest(eval_csv, data_root).iter_samples())
    if not target_samples:
        raise ValueError(f"No target samples loaded from {eval_csv}")

    # Resolve which rates/seeds/methods to evaluate (defaults come from the
    # discovered source models).
    disc_rates = sorted({r for (_m, r, _s) in index})
    disc_seeds = sorted({s for (_m, _r, s) in index})
    rates = [_round_rate(r) for r in sparse_rates] if sparse_rates else disc_rates
    seeds = [int(s) for s in sparse_seeds] if sparse_seeds else disc_seeds
    avail_ml = _ordered_ml({m for (m, _r, _s) in index})
    if methods:
        requested = [str(m) for m in methods]
    else:
        requested = list(NON_TRAINED_BASELINES) + avail_ml
    requested_baselines = [m for m in requested if m in NON_TRAINED_BASELINES]
    requested_ml = [m for m in requested if m not in NON_TRAINED_BASELINES]

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    ml_by_cell: Dict[str, List[str]] = {}

    for rate in rates:
        for seed in seeds:
            models: Dict[str, object] = {}
            fnbm: Dict[str, Optional[List[str]]] = {}
            cell_ml: List[str] = []
            for m in requested_ml:
                path = index.get((m, _round_rate(rate), int(seed)))
                if path is None:
                    print(f"[warn] no source model for method={m} rate={rate:g} seed={seed}; skipping", file=sys.stderr)
                    continue
                payload = joblib.load(path)
                if isinstance(payload, dict) and "model" in payload:
                    models[m] = payload["model"]
                    fnbm[m] = payload.get("feature_names")
                else:
                    models[m] = payload
                    fnbm[m] = None
                cell_ml.append(m)

            # Validate / infer feature columns against the target feature table.
            if cell_ml:
                prep0 = _prepare_sample(target_samples[0], config, float(rate), int(seed))
                full_names = list(prep0.feature_table.feature_names)
                full_set = set(full_names)
                kept: List[str] = []
                for m in cell_ml:
                    names = resolve_feature_names(fnbm.get(m), m, full_names, config.get("feature_groups_by_method"))
                    missing = [n for n in names if n not in full_set]
                    if missing:
                        print(
                            f"[warn] method={m}: target feature table is missing "
                            f"{len(missing)} expected column(s) (e.g. {missing[:3]}); "
                            "skipping this method (feature mismatch)",
                            file=sys.stderr,
                        )
                        models.pop(m, None)
                        continue
                    fnbm[m] = names
                    kept.append(m)
                cell_ml = kept

            cell_methods = list(requested_baselines) + cell_ml
            if not cell_methods:
                print(f"[warn] no evaluable methods for rate={rate:g} seed={seed}; skipping cell", file=sys.stderr)
                continue

            config["methods"] = cell_methods
            print(f"[transfer] eval rate={rate:g} seed={seed} methods={cell_methods}", flush=True)
            rows, _payload = _evaluate_samples(
                target_samples, config, float(rate), int(seed), models, feature_names_by_method=fnbm
            )
            all_rows.extend(rows)
            ml_by_cell[f"rate={rate:g},seed={seed}"] = cell_ml

    if not all_rows:
        raise RuntimeError(
            "Transfer evaluation produced no rows: no requested method was evaluable "
            "for any rate/seed (check --methods, --sparse-rates/-seeds, and that the "
            "source models exist)."
        )

    # Labels for unambiguous provenance.
    source_exp = ""
    if source_cfg:
        source_exp = str(source_cfg.get("experiment_name") or "")
    train_source = source_exp or source_run.name
    target_dataset = str(config.get("experiment_name") or Path(str(config["data_root"]).rstrip("/\\")).name)

    per_sample_df = pd.DataFrame(all_rows)
    final_rows = aggregate_metric_rows(all_rows, group_keys=["method", "sparse_rate", "sparse_seed"])
    final_df = pd.DataFrame(final_rows).sort_values(["sparse_rate", "rmse"]).reset_index(drop=True)
    for df in (per_sample_df, final_df):
        df["train_source"] = train_source
        df["target_dataset"] = target_dataset
        df["retrained"] = False

    per_sample_csv = out_dir / "per_sample_metrics.csv"
    final_csv = out_dir / "final_evaluation_results.csv"
    per_sample_df.to_csv(per_sample_csv, index=False)
    final_df.to_csv(final_csv, index=False)

    # Method-comparison / sparse-curve plots reuse the standard plotters.
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_paths: Dict[str, str] = {}
    try:
        plot_paths["sparse_curves_png"] = str(plot_sparse_curves(per_sample_csv, plots_dir))
        eval_cfg = config.get("evaluation", {}) or {}
        primary_rate = float(eval_cfg.get("primary_sparse_rate", rates[0]))
        plot_paths["model_comparison_png"] = str(plot_model_comparison(per_sample_csv, plots_dir, sparse_rate=primary_rate))
    except Exception as exc:  # plotting is best-effort, never fatal
        plot_paths["plot_error"] = repr(exc)

    primary_method = str(config.get("primary_method", "wallpath_extra"))
    metrics: Dict[str, object] = {}
    prim = final_df[final_df["method"] == primary_method]
    if not prim.empty:
        best = prim.sort_values("rmse").iloc[0].to_dict()
        metrics = {f"primary_{k}": v for k, v in best.items() if isinstance(v, (int, float, np.number)) or k == "method"}

    extra = {
        "transfer_evaluation": True,
        "retrained": False,
        "train_source": train_source,
        "train_source_run": str(source_run),
        "train_source_config_origin": source_origin,
        "target_dataset": target_dataset,
        "target_data_root": str(data_root),
        "target_config": str(target_config),
        "evaluated_sparse_rates": [float(r) for r in rates],
        "evaluated_sparse_seeds": [int(s) for s in seeds],
        "non_trained_baselines": list(requested_baselines),
        "ml_methods_by_cell": ml_by_cell,
    }
    dataset_info = {
        "eval_samples": int(len(target_samples)),
        "target_data_root": str(data_root),
        "source_run": str(source_run),
        "methods": requested,
        "sparse_rates": [float(r) for r in rates],
        "sparse_seeds": [int(s) for s in seeds],
    }
    artifacts = {
        "final_evaluation_results_csv": str(final_csv.resolve()),
        "per_sample_metrics_csv": str(per_sample_csv.resolve()),
        "source_models_dir": str(models_dir.resolve()),
    }
    artifacts.update(plot_paths)

    # Restore the full requested method list for the config snapshot.
    config["methods"] = requested
    create_run_summary(
        experiment_name=f"transfer_{train_source}_to_{target_dataset}",
        output_dir=out_dir,
        config=config,
        metrics=metrics,
        artifacts=artifacts,
        dataset_info=dataset_info,
        timing_info={
            "start_time": started_at,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - started, 3),
        },
        extra=extra,
    )

    print(f"[transfer] train_source={train_source} -> target_dataset={target_dataset} (retrained=false)")
    print(f"[transfer] wrote {final_csv}")
    print(f"[transfer] wrote {per_sample_csv}")
    print(f"[transfer] wrote {out_dir / 'run_summary.json'}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-task transfer evaluation of frozen WallPath-PI models.")
    parser.add_argument("--source-run", type=Path, required=True, help="Completed source run folder (contains models/*.joblib).")
    parser.add_argument("--target-config", type=Path, required=True, help="Config for the target (converted) dataset.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for transfer results.")
    parser.add_argument("--methods", type=str, nargs="*", default=None, help="Methods to evaluate (default: source ML methods + non-trained baselines).")
    parser.add_argument("--sparse-rates", type=float, nargs="*", default=None, help="Sparse rates (default: those present in the source models).")
    parser.add_argument("--sparse-seeds", type=int, nargs="*", default=None, help="Sparse seeds (default: those present in the source models).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_transfer(
        source_run=args.source_run,
        target_config=args.target_config,
        out_dir=args.out_dir,
        methods=args.methods,
        sparse_rates=args.sparse_rates,
        sparse_seeds=args.sparse_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
