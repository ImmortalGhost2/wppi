import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wallpath_pi.baselines.propagation import fspl_db
from wallpath_pi.data.dataset import load_scene_npz, resolve_scene_path
from wallpath_pi.data.sparse import check_sparse_mask_validity, make_sparse_mask
from wallpath_pi.geometry.raster import bresenham_line
from wallpath_pi.models.registry import make_regressor
from wallpath_pi.training.pipeline import (
    ALL_FEATURE_DIRECT_EXTRA_METHODS,
    ALL_FEATURE_DIRECT_METHODS,
    CALIBRATED_RESIDUAL_METHODS,
    GEOMETRY_DIRECT_METHODS,
    RESIDUAL_RF_METHODS,
    SPARSE_ANCHOR_DIRECT_METHODS,
    _prepare_sample,
    is_ablation_method,
)
from wallpath_pi.utils.config import load_config
from wallpath_pi.utils.paths import resolve_cache_root, resolve_csv_path, resolve_data_root, resolve_results_root

# Analytic/interpolation baselines handled directly by the pipeline.
BASELINE_METHODS = (
    "fspl",
    "log_distance",
    "multi_wall",
    "idw",
    "multi_wall_residual_idw",
)
# Methods the training pipeline actually understands (baselines + ML models).
# The learned-model names are sourced from the pipeline's method registry so
# this check never drifts from what the pipeline can train.
KNOWN_METHODS = (
    set(BASELINE_METHODS)
    | set(ALL_FEATURE_DIRECT_METHODS)
    | set(ALL_FEATURE_DIRECT_EXTRA_METHODS)
    | set(GEOMETRY_DIRECT_METHODS)
    | set(SPARSE_ANCHOR_DIRECT_METHODS)
    | set(RESIDUAL_RF_METHODS)
    | set(CALIBRATED_RESIDUAL_METHODS)
)

# NPZ key aliases accepted by the loader, mirrored here for contract reporting.
ARRAY_ALIASES = {
    "path_loss": ["path_loss", "path_loss_db", "target", "target_db"],
    "valid_mask": ["valid_mask", "mask", "receiver_mask"],
    "material_map": ["material_map", "materials", "building_map"],
    "frequency_hz": ["frequency_hz", "freq_hz"],
    "resolution_m": ["resolution_m", "cell_size_m", "pixel_size_m"],
}


class Reporter:
    """Collects PASS/FAIL/WARN/SKIP results and prints a final report."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> str:
        """Record a check result, print it, and return its status."""
        self.rows.append((name, status, detail))
        line = f"    [{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        return status

    def section(self, label: str) -> None:
        """Print a header introducing a group of checks."""
        print(f"\n{label}")

    @property
    def failed(self) -> bool:
        """Return True if any recorded check has FAIL status."""
        return any(status == "FAIL" for _, status, _ in self.rows)

    def summary(self) -> None:
        """Print the aggregated PASS/FAIL/WARN/SKIP report and final verdict."""
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for _, status, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        print("\n" + "=" * 60)
        print("VERIFICATION REPORT")
        print("=" * 60)
        print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  WARN={counts['WARN']}  SKIP={counts['SKIP']}")
        if counts["FAIL"]:
            print("\n  Failing checks:")
            for name, status, detail in self.rows:
                if status == "FAIL":
                    print(f"    - {name}: {detail}" if detail else f"    - {name}")
        if counts["WARN"]:
            print("\n  Warnings:")
            for name, status, detail in self.rows:
                if status == "WARN":
                    print(f"    - {name}: {detail}" if detail else f"    - {name}")
        verdict = "FAIL" if counts["FAIL"] else "PASS"
        print("\n" + "=" * 60)
        if verdict == "PASS":
            print("RESULT: PASS -- WallPath-PI is ready for experiments.")
        else:
            print("RESULT: FAIL -- fix the failing checks above before training.")
        print("=" * 60)


def _rel(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def _writable_dir(path: Path) -> tuple[bool, str]:
    """Create ``path`` and probe it for write access; return ``(ok, detail)``."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot create {path}: {exc}"
    probe = path / ".wallpath_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"not writable {path}: {exc}"
    return True, ""


def _array_present(keys: set[str], row: dict, name: str) -> str:
    """Return 'npz', 'manifest', or '' depending on where the array is found."""
    aliases = ARRAY_ALIASES.get(name, [name])
    if any(a in keys for a in aliases):
        return "npz"
    # Scalars (frequency_hz, resolution_m) may instead come from manifest columns.
    if name in ("frequency_hz", "resolution_m"):
        for a in aliases:
            if a in row and pd.notna(row[a]):
                return "manifest"
    return ""


def _tx_present(keys: set[str], row: dict) -> str:
    """Return 'npz', 'manifest', or '' for where the transmitter location is found."""
    if "tx_xy" in keys or ("tx_x" in keys and "tx_y" in keys):
        return "npz"
    if "tx_x" in row and "tx_y" in row and pd.notna(row.get("tx_x")) and pd.notna(row.get("tx_y")):
        return "manifest"
    return ""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the verifier."""
    parser = argparse.ArgumentParser(description="Verify WallPath-PI data, features, baselines, leakage guards, and ML smoke fit.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to a YAML config file.")
    parser.add_argument("--max-files", type=int, default=3, help="How many NPZ files per split to load/inspect.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every readiness check and return 0 on success, 1 if any check fails."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    data_root = resolve_data_root(cfg["data_root"], repo_root=REPO_ROOT)
    results_root = resolve_results_root(cfg["results_root"], repo_root=REPO_ROOT)
    train_csv = resolve_csv_path(data_root, cfg["train_csv"])
    val_csv = resolve_csv_path(data_root, cfg["val_csv"])
    eval_csv = resolve_csv_path(data_root, cfg.get("eval_csv", cfg["val_csv"]))
    group_col = str(cfg.get("group_column", "scene_id"))
    cache_features = bool(cfg.get("cache_features", False))
    rep = Reporter()

    print("[CONFIG]")
    print("  config       =", _rel(Path(cfg.get("_config_path", args.config))))
    print("  data_root    =", _rel(data_root))
    print("  results_root =", _rel(results_root))
    print("  train_csv    =", _rel(train_csv))
    print("  val_csv      =", _rel(val_csv))
    # eval_csv is only reported/verified separately when it differs from val_csv
    # (e.g. a transfer or held-out evaluation manifest).
    if eval_csv != val_csv:
        print("  eval_csv     =", _rel(eval_csv))

    rep.section("[1/11] Required directories exist")
    data_ok = data_root.is_dir()
    rep.add("data_root exists", "PASS" if data_ok else "FAIL",
            "" if data_ok else f"missing directory {_rel(data_root)}; run the converter/generator first")

    rep.section("[2/11] Manifest files exist with required columns")
    manifests: dict[str, pd.DataFrame] = {}
    required_cols = {"scene_id", "sample_id", "scene_path"}
    manifest_specs = [("train", train_csv), ("val", val_csv)]
    if eval_csv != val_csv:
        manifest_specs.append(("eval", eval_csv))
    for label, path in manifest_specs:
        if not path.exists():
            rep.add(f"{label} manifest", "FAIL", f"missing {_rel(path)}; run make_internal_split.py")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            rep.add(f"{label} manifest", "FAIL", f"could not read {_rel(path)}: {exc}")
            continue
        if "scene_path" not in df.columns and "map_path" in df.columns:
            df = df.rename(columns={"map_path": "scene_path"})
        missing = required_cols - set(df.columns)
        if missing:
            rep.add(f"{label} manifest columns", "FAIL", f"{path.name} missing {sorted(missing)}")
            continue
        if df.empty:
            rep.add(f"{label} manifest", "FAIL", f"{path.name} has no rows")
            continue
        manifests[label] = df
        rep.add(f"{label} manifest", "PASS", f"{len(df)} rows, {df[group_col].astype(str).nunique()} scenes")

    rep.section("[3/11] All scene_path entries resolve on disk")
    if not manifests:
        rep.add("scene_path resolution", "SKIP", "no readable manifests")
    else:
        for label, df in manifests.items():
            missing_paths = []
            for value in df["scene_path"].astype(str):
                if not resolve_scene_path(data_root, value).exists():
                    missing_paths.append(value)
            if missing_paths:
                shown = ", ".join(missing_paths[:5])
                rep.add(f"{label} scene_path", "FAIL",
                        f"{len(missing_paths)}/{len(df)} missing (e.g. {shown})")
            else:
                rep.add(f"{label} scene_path", "PASS", f"all {len(df)} resolve")

    rep.section("[4/11] A few NPZ files load successfully")
    loaded_samples: list = []
    inspected: list[tuple[set[str], dict]] = []
    if not manifests:
        rep.add("npz load", "SKIP", "no readable manifests")
    else:
        for label, df in manifests.items():
            head = df.head(max(1, int(args.max_files)))
            ok = 0
            for _, row in head.iterrows():
                row_dict = row.to_dict()
                scene_path = resolve_scene_path(data_root, str(row_dict["scene_path"]))
                try:
                    sample = load_scene_npz(scene_path, row=row_dict)
                    loaded_samples.append(sample)
                    with np.load(scene_path, allow_pickle=False) as obj:
                        inspected.append((set(obj.files), row_dict))
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    rep.add(f"{label} npz load", "FAIL", f"{scene_path.name}: {exc}")
            if ok:
                rep.add(f"{label} npz load", "PASS", f"{ok} file(s) loaded")

    rep.section("[5/11] Required arrays present (path_loss, valid_mask, material_map, "
                "wall_mask, tx_xy, frequency_hz, resolution_m)")
    if not inspected:
        rep.add("required arrays", "SKIP", "no NPZ files loaded")
    else:
        required_scalars = ["path_loss", "valid_mask", "material_map", "frequency_hz", "resolution_m"]
        for name in required_scalars:
            sources = [_array_present(keys, row, name) for keys, row in inspected]
            if all(sources):
                via = "manifest" if "manifest" in sources else "npz"
                rep.add(f"array '{name}'", "PASS", f"present (via {via})" if via == "manifest" else "present")
            else:
                n_missing = sources.count("")
                rep.add(f"array '{name}'", "FAIL", f"missing in {n_missing}/{len(inspected)} file(s)")
        # tx_xy (or tx_x/tx_y, or manifest columns)
        tx_sources = [_tx_present(keys, row) for keys, row in inspected]
        if all(tx_sources):
            rep.add("array 'tx_xy'", "PASS", "present" if "manifest" not in tx_sources else "present (via manifest)")
        else:
            rep.add("array 'tx_xy'", "FAIL", f"missing in {tx_sources.count('')}/{len(inspected)} file(s)")
        # wall_mask: present, or safely derivable from material_map > 0
        wall_states = []
        for keys, _ in inspected:
            if "wall_mask" in keys:
                wall_states.append("present")
            elif any(a in keys for a in ARRAY_ALIASES["material_map"]):
                wall_states.append("derived")
            else:
                wall_states.append("missing")
        if "missing" in wall_states:
            rep.add("array 'wall_mask'", "FAIL", "absent and material_map missing")
        elif "derived" in wall_states:
            rep.add("array 'wall_mask'", "WARN", "absent; derived from material_map > 0")
        else:
            rep.add("array 'wall_mask'", "PASS", "present")

    rep.section("[6/11] Optional arrays reported (transmittance, reflectance)")
    if not inspected:
        rep.add("optional arrays", "SKIP", "no NPZ files loaded")
    else:
        for name in ("transmittance", "reflectance"):
            present = [name in keys for keys, _ in inspected]
            if all(present):
                rep.add(f"optional '{name}'", "PASS", "present")
            elif any(present):
                rep.add(f"optional '{name}'", "WARN", "present in some files only")
            else:
                rep.add(f"optional '{name}'", "WARN", "absent (line-integral features will use defaults)")

    rep.section("[7/11] Train/val scene overlap is zero")
    if "train" in manifests and "val" in manifests:
        overlap = sorted(set(manifests["train"][group_col].astype(str)) & set(manifests["val"][group_col].astype(str)))
        if overlap:
            rep.add("scene-disjoint split", "FAIL", f"{len(overlap)} shared scene(s): {overlap[:10]}")
        else:
            rep.add("scene-disjoint split", "PASS", "no shared scenes")
    else:
        rep.add("scene-disjoint split", "SKIP", "need both train and val manifests")

    rep.section("[8/11] Sparse rates and seeds are valid")
    rates = cfg.get("sparse_rates") or []
    seeds = cfg.get("sparse_seeds") or []
    bad_rates = [r for r in rates if not (isinstance(r, (int, float)) and 0.0 < float(r) <= 1.0)]
    bad_seeds = [s for s in seeds if not (isinstance(s, (int, float)) and float(s) == int(s))]
    if not rates:
        rep.add("sparse_rates", "FAIL", "empty; set train.sparse_rates")
    elif bad_rates:
        rep.add("sparse_rates", "FAIL", f"out of (0, 1]: {bad_rates}")
    else:
        rep.add("sparse_rates", "PASS", f"{list(rates)}")
    if not seeds:
        rep.add("sparse_seeds", "FAIL", "empty; set train.sparse_seeds")
    elif bad_seeds:
        rep.add("sparse_seeds", "FAIL", f"non-integer seeds: {bad_seeds}")
    else:
        rep.add("sparse_seeds", "PASS", f"{list(seeds)}")

    rep.section("[9/11] Configured methods are known")
    methods = [str(m) for m in (cfg.get("methods") or [])]
    # ``ablation_*`` variants are config-declared feature-ablation learners
    # (see ``train.feature_groups_by_method``); accept them by prefix.
    unknown = sorted(m for m in set(methods) if m not in KNOWN_METHODS and not is_ablation_method(m))
    if not methods:
        rep.add("methods", "FAIL", "empty; set train.methods")
    elif unknown:
        rep.add("methods", "FAIL", f"unknown {unknown}; known={sorted(KNOWN_METHODS)}")
    else:
        rep.add("methods", "PASS", f"{methods}")
    primary = str(cfg.get("primary_method", "")) if cfg.get("primary_method") else ""
    if primary and primary not in methods:
        rep.add("primary_method", "WARN", f"'{primary}' not in train.methods")

    rep.section("[10/11] Feature cache directory writable (if cache_features=true)")
    if not cache_features:
        rep.add("cache writable", "SKIP", "cache_features=false")
    else:
        cache_root = resolve_cache_root(cfg.get("cache_root"), repo_root=REPO_ROOT)
        ok, detail = _writable_dir(cache_root)
        rep.add("cache writable", "PASS" if ok else "FAIL", _rel(cache_root) if ok else detail)

    rep.section("[11/11] Results directory writable")
    ok, detail = _writable_dir(results_root)
    rep.add("results writable", "PASS" if ok else "FAIL", _rel(results_root) if ok else detail)

    rep.section("[deep] Pipeline build sanity")
    fs = float(fspl_db(np.asarray([1.0], dtype=np.float32), 1.0e9, min_distance_m=1.0)[0])
    if 32.0 < fs < 33.0:
        rep.add("FSPL physics", "PASS", f"FSPL(1 m, 1 GHz) = {fs:.3f} dB")
    else:
        rep.add("FSPL physics", "FAIL", f"expected ~32.45 dB, got {fs:.3f}")
    yy, xx = bresenham_line(0, 0, 5, 5)
    if len(yy) == 6 and yy[-1] == 5 and xx[-1] == 5:
        rep.add("Bresenham raster", "PASS")
    else:
        rep.add("Bresenham raster", "FAIL", "diagonal line endpoints incorrect")

    if not loaded_samples:
        rep.add("feature/baseline build", "SKIP", "no sample loaded")
        rep.add("ML smoke fit", "SKIP", "no sample loaded")
    else:
        sample = loaded_samples[0]
        try:
            rate = float((cfg.get("sparse_rates") or [0.05])[0])
            sp_seed = int((cfg.get("sparse_seeds") or [11])[0])
            sparse = make_sparse_mask(sample.valid_mask, sampling_rate=rate, seed=sp_seed,
                                      sample_id=sample.sample_id, min_points=int(cfg.get("min_anchors", 1)))
            check_sparse_mask_validity(sparse, sample.valid_mask)
            prep = _prepare_sample(sample, cfg, sparse_rate=rate, sparse_seed=sp_seed)
            if int(prep.sparse_mask.sum()) <= 0:
                rep.add("feature/baseline build", "FAIL", "no sparse anchors created")
            elif prep.feature_table.X.shape[0] != int(prep.feature_table.valid_mask.sum()):
                rep.add("feature/baseline build", "FAIL", "feature rows != valid pixels")
            elif not np.isfinite(prep.feature_table.X).all():
                rep.add("feature/baseline build", "FAIL", "feature table contains NaN/Inf")
            else:
                rep.add("feature/baseline build", "PASS",
                        f"{int(prep.sparse_mask.sum())} anchors, "
                        f"{prep.feature_table.X.shape[0]}x{prep.feature_table.X.shape[1]} features")
                params = {"extra_trees": {"n_estimators": 8, "n_jobs": 1, "max_depth": 8, "min_samples_leaf": 1}}
                model = make_regressor("wallpath_extra", params=params, seed=1)
                model.fit(prep.feature_table.X, prep.feature_table.y - prep.feature_table.baseline_values)
                pred = model.predict(prep.feature_table.X[:10])
                if np.all(np.isfinite(pred)):
                    rep.add("ML smoke fit", "PASS", f"pred mean = {float(np.mean(pred)):.4f}")
                else:
                    rep.add("ML smoke fit", "FAIL", "non-finite predictions")
        except Exception as exc:  # noqa: BLE001
            rep.add("feature/baseline build", "FAIL", f"{type(exc).__name__}: {exc}")

    rep.summary()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
