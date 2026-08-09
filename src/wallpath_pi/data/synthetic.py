from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from wallpath_pi.geometry.raster import compute_base_maps, distance_map


def _draw_line(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    from wallpath_pi.geometry.raster import bresenham_line
    yy, xx = bresenham_line(x0, y0, x1, y1)
    yy = np.clip(yy, 0, arr.shape[0] - 1)
    xx = np.clip(xx, 0, arr.shape[1] - 1)
    arr[yy, xx] = int(value)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], passes: int = 5) -> np.ndarray:
    z = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    for _ in range(int(passes)):
        z = (
            z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)
        ) / 5.0
    return z.astype(np.float32)


def _choose_tx(valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.nonzero(valid)
    if len(yy) == 0:
        raise ValueError("No valid transmitter cells.")
    idx = int(rng.integers(0, len(yy)))
    return np.asarray([float(xx[idx]), float(yy[idx])], dtype=np.float32)


def generate_scene_sample(
    scene_id: str,
    sample_id: str,
    image_size: int,
    resolution_m: float,
    frequency_hz: float,
    rng: np.random.Generator,
    material_ids: Sequence[int] = (1, 2, 3),
) -> Dict[str, Any]:
    """Generate one synthetic radio-map sample with materials, walls, and path loss.

    The returned payload mirrors the on-disk NPZ contract (path loss, masks,
    material map, transmitter position, and per-cell metadata) so synthetic and
    converted real samples flow through the same pipeline.
    """
    H = W = int(image_size)
    material = np.zeros((H, W), dtype=np.int32)
    # Border walls.
    material[0, :] = 1
    material[-1, :] = 1
    material[:, 0] = 1
    material[:, -1] = 1

    # Interior walls. Each scene has a mix of partial vertical/horizontal walls.
    n_walls = int(rng.integers(5, 10))
    for _ in range(n_walls):
        mat = int(rng.choice(material_ids))
        if rng.random() < 0.5:
            x = int(rng.integers(6, W - 6))
            y0 = int(rng.integers(4, H // 2))
            y1 = int(rng.integers(H // 2, H - 4))
            if rng.random() < 0.45 and (y1 - y0) > 8:
                gap = int(rng.integers(y0 + 3, y1 - 2))
                _draw_line(material, x, y0, x, gap - 2, mat)
                _draw_line(material, x, gap + 2, x, y1, mat)
            else:
                _draw_line(material, x, y0, x, y1, mat)
        else:
            y = int(rng.integers(6, H - 6))
            x0 = int(rng.integers(4, W // 2))
            x1 = int(rng.integers(W // 2, W - 4))
            if rng.random() < 0.45 and (x1 - x0) > 8:
                gap = int(rng.integers(x0 + 3, x1 - 2))
                _draw_line(material, x0, y, gap - 2, y, mat)
                _draw_line(material, gap + 2, y, x1, y, mat)
            else:
                _draw_line(material, x0, y, x1, y, mat)

    wall = material > 0
    valid = ~wall
    tx_xy = _choose_tx(valid, rng)
    transmittance = np.ones((H, W), dtype=np.float32)
    reflectance = np.zeros((H, W), dtype=np.float32)
    for mid, t, r in [(1, 0.65, 0.20), (2, 0.45, 0.35), (3, 0.25, 0.50)]:
        transmittance[material == mid] = t
        reflectance[material == mid] = r

    base = compute_base_maps(
        wall_mask=wall,
        material_map=material,
        tx_xy=tx_xy,
        frequency_hz=float(frequency_hz),
        resolution_m=float(resolution_m),
        valid_mask=valid,
        material_ids=material_ids,
        transmittance=transmittance,
        reflectance=reflectance,
        min_distance_m=float(resolution_m),
    )
    dist = distance_map((H, W), tx_xy=tx_xy, resolution_m=resolution_m, min_distance_m=resolution_m)
    extra_log = 10.0 * float(rng.uniform(0.35, 0.75)) * np.log10(np.maximum(dist, resolution_m) / 1.0)
    wall_losses = np.zeros((H, W), dtype=np.float32)
    loss_per_pixel = {1: 1.1, 2: 1.9, 3: 3.0}
    for idx, mid in enumerate(material_ids):
        wall_losses += base.material_counts[idx] * float(loss_per_pixel.get(int(mid), 1.5))
    yy, xx = np.mgrid[0:H, 0:W]
    phase = rng.uniform(0.0, 2.0 * np.pi)
    structured = 1.8 * np.sin(0.18 * xx + phase) + 1.4 * np.cos(0.15 * yy - phase)
    shadow = 2.2 * _smooth_noise(rng, (H, W), passes=8)
    nlos_bias = (1.0 - base.los) * rng.uniform(1.0, 3.5)
    path_loss = base.fspl + extra_log + wall_losses + structured + shadow + nlos_bias
    path_loss = np.where(valid, path_loss, np.nan)
    path_loss = np.clip(path_loss, 13.0, 160.0).astype(np.float32)
    return {
        "path_loss": path_loss,
        "path_loss_db": path_loss,
        "valid_mask": valid.astype(np.uint8),
        "wall_mask": wall.astype(np.uint8),
        "material_map": material.astype(np.int16),
        "tx_xy": tx_xy.astype(np.float32),
        "frequency_hz": np.asarray(float(frequency_hz), dtype=np.float64),
        "resolution_m": np.asarray(float(resolution_m), dtype=np.float32),
        "cell_size_m": np.asarray(float(resolution_m), dtype=np.float32),
        "tx_x": np.asarray(float(tx_xy[0]), dtype=np.float32),
        "tx_y": np.asarray(float(tx_xy[1]), dtype=np.float32),
        "transmittance": transmittance.astype(np.float32),
        "reflectance": reflectance.astype(np.float32),
        "scene_id": np.asarray(scene_id),
        "sample_id": np.asarray(sample_id),
    }


def generate_synthetic_dataset(data_root: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Write the synthetic smoke dataset under ``data_root`` and return its manifest.

    Scene count, samples per scene, image size, resolution, and frequencies are
    read from ``cfg``; the whole dataset is reproducible from ``cfg['seed']``.
    """
    data_root = Path(data_root).expanduser().resolve()
    scene_dir = data_root / "scenes"
    overwrite = bool(cfg.get("overwrite", True))
    if overwrite and data_root.exists():
        for p in scene_dir.glob("*.npz") if scene_dir.exists() else []:
            p.unlink()
    scene_dir.mkdir(parents=True, exist_ok=True)
    num_scenes = int(cfg.get("num_scenes", 10))
    samples_per_scene = int(cfg.get("samples_per_scene", 2))
    image_size = int(cfg.get("image_size", 48))
    resolution_m = float(cfg.get("resolution_m", 0.25))
    freqs = [float(x) for x in cfg.get("frequencies_hz", [3.5e9])]
    seed = int(cfg.get("seed", 2026))
    rng = np.random.default_rng(seed)
    rows = []
    material_ids = [1, 2, 3]
    for s in range(num_scenes):
        scene_id = f"scene_{s:03d}"
        scene_seed = int(rng.integers(0, 2 ** 32 - 1))
        for k in range(samples_per_scene):
            sample_rng = np.random.default_rng(scene_seed + k * 9973)
            frequency_hz = freqs[k % len(freqs)]
            sample_id = f"{scene_id}_tx_{k:02d}_f_{int(frequency_hz/1e6)}MHz"
            payload = generate_scene_sample(
                scene_id=scene_id,
                sample_id=sample_id,
                image_size=image_size,
                resolution_m=resolution_m,
                frequency_hz=frequency_hz,
                rng=sample_rng,
                material_ids=material_ids,
            )
            rel_path = Path("scenes") / f"{sample_id}.npz"
            out_path = data_root / rel_path
            np.savez_compressed(out_path, **payload)
            rows.append({
                "scene_path": str(rel_path),
                "scene_id": scene_id,
                "sample_id": sample_id,
                "tx_x": float(payload["tx_xy"][0]),
                "tx_y": float(payload["tx_xy"][1]),
                "frequency_hz": float(frequency_hz),
                "resolution_m": float(resolution_m),
            })
    df = pd.DataFrame(rows)
    manifest = data_root / "manifest.csv"
    df.to_csv(manifest, index=False)
    meta = {
        "num_rows": int(len(df)),
        "num_scenes": int(df["scene_id"].nunique()),
        "samples_per_scene": samples_per_scene,
        "image_size": image_size,
        "resolution_m": resolution_m,
        "seed": seed,
    }
    (data_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def generate_sample_dataset(
    out_root: Path,
    num_scenes: int = 8,
    tx_per_scene: int = 2,
    height: int = 48,
    width: int = 48,
    cell_size_m: float = 0.25,
    frequencies_hz: Sequence[float] = (3.5e9,),
    seed: int = 2026,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper that creates the current NPZ/manifest dataset contract.

    Older scripts used `generate_sample_dataset`; the maintained function is
    `generate_synthetic_dataset`. This wrapper keeps both APIs usable.
    """
    if int(height) != int(width):
        raise ValueError('The current synthetic generator expects square maps; use equal height/width.')
    cfg = {
        'num_scenes': int(num_scenes),
        'samples_per_scene': int(tx_per_scene),
        'image_size': int(height),
        'resolution_m': float(cell_size_m),
        'frequencies_hz': [float(x) for x in frequencies_hz],
        'seed': int(seed),
        'overwrite': bool(overwrite),
    }
    df = generate_synthetic_dataset(Path(out_root), cfg)
    # Backwards-compatible alias used by some older helper scripts.
    df.to_csv(Path(out_root).expanduser().resolve() / 'all.csv', index=False)
    return df
