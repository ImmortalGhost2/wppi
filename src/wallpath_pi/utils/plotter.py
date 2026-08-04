"""Plotting helpers and shared method display names."""
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; no window is opened during batch runs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 300 DPI raster output; a vector PDF is saved alongside each figure.
PNG_DPI = 300

# Shared display names so every figure labels methods identically.
DISPLAY_NAMES = {
    "fspl": "FSPL",
    "log_distance": "Log-distance",
    "multi_wall": "Multi-wall",
    "idw": "IDW",
    "multi_wall_residual_idw": "Multi-wall + IDW",
    "direct_rf": "Direct RF",
    "direct_rf_all_features": "Direct RF (all features)",
    "all_feature_rf": "Direct RF (all features)",
    "direct_rf_geometry": "Direct RF (geometry)",
    "direct_rf_sparse_anchor": "Direct RF (sparse anchor)",
    "direct_extra_all_features": "Direct ExtraTrees (all features)",
    "all_feature_extra": "Direct ExtraTrees (all features)",
    "wallpath_rf": "WallPath-RF",
    "wallpath_extra": "WallPath-PI",
    "wallpath_calibrated": "WallPath-Calibrated",
}

# Concise names used in manuscript tables and figures. Diagnostic plots retain
# the qualified DISPLAY_NAMES above by default.
MANUSCRIPT_DISPLAY_NAMES = {
    "wallpath_extra": "WallPath-PI",
    "wallpath_rf": "WallPath-RF",
    "wallpath_calibrated": "WallPath-Calibrated",
    "direct_rf": "Direct RF",
    "direct_rf_all_features": "Direct RF",
    "all_feature_rf": "Direct RF",
    "direct_extra_all_features": "Direct ExtraTrees",
    "all_feature_extra": "Direct ExtraTrees",
}

# Accept loose user input and map it to the canonical internal method keys.
_INPUT_ALIASES = {
    "multiwall": "multi_wall",
    "multiwall_idw": "multi_wall_residual_idw",
    "multi_wall_idw": "multi_wall_residual_idw",
    "wallpath_pi": "wallpath_extra",
    "wallpath": "wallpath_extra",
    "direct_ml": "direct_rf",
}


def display_name(method: str, *, manuscript: bool = False) -> str:
    m = str(method)
    if manuscript:
        return MANUSCRIPT_DISPLAY_NAMES.get(
            m, DISPLAY_NAMES.get(m, m.replace("_", " "))
        )
    return DISPLAY_NAMES.get(m, m.replace("_", " "))


def _normalize_methods(methods: Sequence[str]) -> list[str]:
    return [_INPUT_ALIASES.get(str(m), str(m)) for m in methods]


def _save(fig, out_base: Path, save_pdf: bool = True) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_sparse_curves(
    results_csv: Path, out_dir: Path, *, metric: str = "rmse",
    show_title: bool = True, save_pdf: bool = True,
) -> Path:
    """RMSE versus sparse measurement rate, with error bars across seeds/samples."""
    df = pd.read_csv(results_csv)
    if "sparse_rate" not in df.columns or "method" not in df.columns or metric not in df.columns:
        raise ValueError(f"results CSV must contain 'sparse_rate', 'method', and '{metric}' columns.")
    agg = df.groupby(["method", "sparse_rate"], as_index=False).agg(
        mean=(metric, "mean"), std=(metric, "std"), n=(metric, "size")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, sub in agg.groupby("method"):
        sub = sub.sort_values("sparse_rate")
        yerr = sub["std"].to_numpy()
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        ax.errorbar(
            sub["sparse_rate"].to_numpy(), sub["mean"].to_numpy(), yerr=yerr,
            marker="o", capsize=3, linewidth=1.6, label=display_name(method),
        )
    ax.set_xlabel("Sparse measurement rate (fraction of valid pixels)")
    ax.set_ylabel(f"{metric.upper()} (dB)")
    if show_title:
        ax.set_title("Sparse-budget path-loss reconstruction")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, title="Method", title_fontsize=8)
    fig.tight_layout()
    out_base = Path(out_dir) / "sparse_curves"
    _save(fig, out_base, save_pdf=save_pdf)
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_model_comparison(results_csv: Path, out_dir: Path, sparse_rate: Optional[float] = None,
                          metric: str = "rmse", *, show_title: bool = True,
                          save_pdf: bool = True) -> Path:
    """Per-method mean error (with sample std error bars) at a given sparse rate."""
    df = pd.read_csv(results_csv)
    if metric not in df.columns or "method" not in df.columns:
        raise ValueError(f"results CSV must contain 'method' and '{metric}' columns.")
    if sparse_rate is None:
        sparse_rate = float(df["sparse_rate"].median()) if "sparse_rate" in df.columns else None
    sub = df if sparse_rate is None else df[np.isclose(df["sparse_rate"].astype(float), float(sparse_rate))]
    agg = sub.groupby("method", as_index=False).agg(mean=(metric, "mean"), std=(metric, "std")).sort_values("mean")
    if agg.empty:
        raise ValueError("No rows available for the requested sparse rate.")
    labels = [display_name(m) for m in agg["method"].astype(str)]
    yerr = np.where(np.isfinite(agg["std"].to_numpy()), agg["std"].to_numpy(), 0.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, agg["mean"].astype(float), yerr=yerr, capsize=3, color="#4C72B0")
    ax.set_ylabel(f"{metric.upper()} (dB)")
    if show_title:
        title = "Method comparison" + (f" at sparse rate {sparse_rate:g}" if sparse_rate is not None else "")
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    tag = "all" if sparse_rate is None else str(sparse_rate).replace(".", "p")
    out_base = Path(out_dir) / f"model_comparison_rate_{tag}"
    _save(fig, out_base, save_pdf=save_pdf)
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_region_bars(
    results_csv: Path, out_base: Path, sparse_rate: Optional[float] = None,
    *, show_title: bool = True, save_pdf: bool = True,
) -> Path:
    """Grouped bars of LOS / NLOS / high-wall (and unclipped) RMSE per method."""
    df = pd.read_csv(results_csv)
    if "method" not in df.columns:
        raise ValueError("results CSV must contain a 'method' column.")
    if sparse_rate is None and "sparse_rate" in df.columns:
        sparse_rate = float(df["sparse_rate"].median())
    if sparse_rate is not None and "sparse_rate" in df.columns:
        df = df[np.isclose(df["sparse_rate"].astype(float), float(sparse_rate))]
    region_labels = {
        "los_rmse": "LOS",
        "nlos_rmse": "NLOS",
        "high_wall_rmse": "High-wall",
        "unclipped_rmse": "Unclipped",
    }
    metric_cols = [c for c in region_labels if c in df.columns]
    if not metric_cols:
        raise ValueError("No region metric columns (los_rmse/nlos_rmse/high_wall_rmse/unclipped_rmse) found.")
    grouped = df.groupby("method", as_index=False)[metric_cols].mean()
    grouped = grouped.sort_values(metric_cols[0]).reset_index(drop=True)
    x = np.arange(len(grouped))
    width = 0.8 / max(1, len(metric_cols))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - (len(metric_cols) - 1) / 2) * width, grouped[col].astype(float),
               width=width, label=region_labels[col])
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in grouped["method"].astype(str)], rotation=35, ha="right")
    ax.set_ylabel("RMSE (dB)")
    if show_title:
        title = "Region-specific error" + (f" at sparse rate {sparse_rate:g}" if sparse_rate is not None else "")
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, title="Region", title_fontsize=8)
    fig.tight_layout()
    _save(fig, Path(out_base), save_pdf=save_pdf)
    plt.close(fig)
    return Path(out_base).with_suffix(".png")


def plot_feature_importance(feature_importance_csv: Path, out_dir: Path, top_k: int = 15,
                            *, primary: str = "wallpath_extra", show_title: bool = True,
                            save_pdf: bool = True) -> Optional[Path]:
    """Top-``top_k`` features for the primary model, averaged across seeds/rates.

    Returns None (no crash) when the CSV is missing or empty.
    """
    path = Path(feature_importance_csv)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "feature" not in df.columns or "importance" not in df.columns:
        return None
    # Prefer the primary model's importances; gracefully fall back if absent.
    if "method" in df.columns:
        methods_present = set(df["method"].astype(str))
        chosen = primary if primary in methods_present else None
        if chosen is None:
            for cand in ("wallpath_extra", "direct_rf"):
                if cand in methods_present:
                    chosen = cand
                    break
        if chosen is not None:
            df = df[df["method"].astype(str) == chosen]
        else:
            chosen = "model"
    else:
        chosen = primary
    agg = df.groupby("feature", as_index=False).agg(mean=("importance", "mean"), std=("importance", "std"))
    agg = agg.sort_values("mean", ascending=False).head(int(top_k))
    if agg.empty:
        return None
    order = agg.iloc[::-1]  # largest at top of horizontal bar chart
    xerr = np.where(np.isfinite(order["std"].to_numpy()), order["std"].to_numpy(), 0.0)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.34 * len(order))))
    ax.barh(order["feature"].astype(str), order["mean"].astype(float), xerr=xerr, capsize=2, color="#55A868")
    ax.set_xlabel("Mean feature importance (averaged across seeds)")
    if show_title:
        ax.set_title(f"{display_name(chosen)} feature importance (top {len(order)})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_base = Path(out_dir) / "feature_importance"
    _save(fig, out_base, save_pdf=save_pdf)
    plt.close(fig)
    return out_base.with_suffix(".png")


def _overlay_tx_and_anchors(ax, tx_xy: Optional[np.ndarray], anchor_mask: Optional[np.ndarray], legend: bool) -> None:
    handles = []
    if anchor_mask is not None and np.any(anchor_mask):
        rows, cols = np.nonzero(anchor_mask)
        sc = ax.scatter(cols, rows, s=8, c="white", edgecolors="black", linewidths=0.4, label="Sparse anchors")
        handles.append(sc)
    if tx_xy is not None and np.all(np.isfinite(tx_xy)):
        tx = ax.scatter([float(tx_xy[0])], [float(tx_xy[1])], s=90, marker="*",
                        c="red", edgecolors="black", linewidths=0.6, label="Transmitter")
        handles.append(tx)
    if legend and handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)


def plot_prediction_maps(npz_path: Path, out_dir: Path, sample_index: int = 0,
                         methods: Optional[Iterable[str]] = None, *, primary: str = "wallpath_extra",
                         show_title: bool = True, save_pdf: bool = True) -> Path:
    """Paper-ready prediction figure.

    Panels: true path loss, the multi_wall and direct_rf baselines, WallPath-PI,
    and the WallPath-PI absolute-error map. The transmitter and sparse-anchor
    locations are overlaid when present in the npz. Missing methods are skipped
    without crashing.
    """
    data = np.load(npz_path, allow_pickle=True)
    # ``allow_pickle=True`` is required because variable-shaped (different HxW)
    # validation exports store per-sample maps as object arrays. This is a
    # trusted, locally produced experiment artifact. Fixed-shape exports keep the
    # original dense layout and load identically through the accessors below.
    variable = bool(np.asarray(data["variable_shapes"])) if "variable_shapes" in data else False
    predictions = data["predictions"]
    targets = data["targets"]
    valid_masks = data["valid_masks"]
    method_names = [str(x) for x in data["methods"].tolist()]
    scene_ids = [str(x) for x in data["scene_ids"].tolist()]
    sample_ids = [str(x) for x in data["sample_ids"].tolist()]
    n_samples = len(targets)
    idx = int(np.clip(sample_index, 0, n_samples - 1))

    # Per-sample accessors that work for both stacked (fixed-shape) and object
    # (variable-shape) arrays without ever stacking differently shaped maps.
    def _target(i: int) -> np.ndarray:
        return np.asarray(targets[i], dtype=float)

    def _valid(i: int) -> np.ndarray:
        return np.asarray(valid_masks[i]).astype(bool)

    def _pred(mi: int, i: int) -> np.ndarray:
        return np.asarray(predictions[mi, i], dtype=float)

    sparse_mask = None
    if "sparse_masks" in data:
        sparse_mask = np.asarray(data["sparse_masks"][idx]).astype(bool)
    tx_xy = None
    if "tx_positions" in data:
        tx_xy = np.asarray(data["tx_positions"][idx], dtype=float).reshape(-1)[:2]

    # Decide which method panels to show, in a fixed, paper-friendly order.
    if methods is not None:
        requested = _normalize_methods(list(methods))
    else:
        requested = ["multi_wall", "direct_rf", primary]
    chosen = [m for m in requested if m in method_names]
    prim = primary if primary in method_names else next((m for m in ("wallpath_extra",) if m in method_names), None)

    vmask = _valid(idx)
    target = np.where(vmask, _target(idx), np.nan)
    finite_target = target[np.isfinite(target)]
    if finite_target.size:
        vmin, vmax = np.percentile(finite_target, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    # Build the ordered panel list: (title, array, kind) where kind in {pl, err}.
    panels: list[tuple[str, np.ndarray, str]] = [("True path loss", target, "pl")]
    for m in chosen:
        mi = method_names.index(m)
        pred = np.where(vmask, _pred(mi, idx), np.nan)
        panels.append((display_name(m), pred, "pl"))
    if prim is not None:
        pmi = method_names.index(prim)
        err = np.where(vmask, np.abs(_pred(pmi, idx) - _target(idx)), np.nan)
        panels.append((f"{display_name(prim)} |error|", err, "err"))

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    for j, (title, arr, kind) in enumerate(panels):
        ax = axes[j // ncols][j % ncols]
        if kind == "pl":
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Path loss (dB)", fontsize=8)
        else:
            im = ax.imshow(arr, origin="lower", cmap="magma")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Absolute error (dB)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        _overlay_tx_and_anchors(ax, tx_xy, sparse_mask, legend=(j == 0))
    # Hide any unused axes in the final row.
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    n_anchor = int(sparse_mask.sum()) if sparse_mask is not None else 0
    if show_title:
        fig.suptitle(
            f"{scene_ids[idx]} / {sample_ids[idx]}  (anchors: {n_anchor})",
            y=1.0,
            fontsize=11,
        )
    fig.tight_layout()
    out_base = Path(out_dir) / "prediction_maps"
    _save(fig, out_base, save_pdf=save_pdf)
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_prediction_panel(npz_path: Path, out_base: Path, methods: Optional[Iterable[str]] = None,
                          sample_index: int = 0, *, show_title: bool = True,
                          save_pdf: bool = True) -> Path:
    """Compatibility wrapper that saves a prediction panel at an explicit basename."""
    tmp_dir = Path(out_base).parent
    out = plot_prediction_maps(
        npz_path,
        tmp_dir,
        sample_index=sample_index,
        methods=methods,
        show_title=show_title,
        save_pdf=save_pdf,
    )
    desired = Path(out_base).with_suffix(".png")
    desired_pdf = Path(out_base).with_suffix(".pdf")
    if out != desired:
        out.replace(desired)
        pdf_src = out.with_suffix(".pdf")
        if save_pdf and pdf_src.exists():
            pdf_src.replace(desired_pdf)
    return desired
