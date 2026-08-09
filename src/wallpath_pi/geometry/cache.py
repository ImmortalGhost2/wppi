"""On-disk caching of static base maps.

``compute_base_maps`` traces a Bresenham ray from the transmitter to every valid
receiver cell. For real ICASSP-scale maps this is expensive and is repeated for
every sparse rate/seed even though the base maps are static for a given sample.
This module caches the static maps so they are computed once per sample.

Cache files are compressed NPZ written without object arrays, so they load with
``np.load(..., allow_pickle=False)``. The cache is file-signature-addressed
rather than content-addressed: the key depends on the sample identity, a cheap
scene-file signature (path, byte size, and modification time), and every
parameter that affects the computed maps. Because the signature reads file
metadata rather than hashing the scene bytes, editing a scene file in place
without changing its size or modification time would not by itself invalidate
the entry.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from wallpath_pi.geometry.features import build_base_maps_for_sample
from wallpath_pi.geometry.raster import BaseMaps

_CACHE_VERSION = "1"

_ARRAY_KEYS = (
    "distance_m",
    "fspl",
    "los",
    "wall_count",
    "wall_fraction",
    "material_counts",
    "transmittance_sum",
    "reflectance_sum",
    "valid_mask",
)


def _scene_signature(scene_path: Path) -> str:
    """A cheap, change-sensitive signature for the scene file (size + mtime)."""
    try:
        st = scene_path.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


def base_maps_cache_key(
    sample,
    material_ids: Sequence[int],
    min_distance_m: float,
) -> str:
    """Deterministic cache key for a sample's base maps.

    Depends on sample/scene identity and every parameter that changes the maps:
    tx position, frequency, resolution, material ids, min distance, and whether
    transmittance/reflectance layers are present.
    """
    tx = np.asarray(sample.tx_xy, dtype=np.float64).reshape(-1)[:2]
    parts = [
        _CACHE_VERSION,
        str(sample.sample_id),
        str(sample.scene_path),
        _scene_signature(Path(sample.scene_path)),
        f"tx={float(tx[0]):.6f},{float(tx[1]):.6f}",
        f"freq={float(sample.frequency_hz):.6f}",
        f"res={float(sample.resolution_m):.6f}",
        "mat=" + ",".join(str(int(m)) for m in material_ids),
        f"mind={float(min_distance_m):.6f}",
        f"trans={sample.transmittance is not None}",
        f"refl={sample.reflectance is not None}",
    ]
    text = "||".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_base_maps_cache(
    cache_path: Union[str, Path],
    base_maps: BaseMaps,
    material_ids: Iterable[int],
) -> Path:
    """Atomically write base maps to a compressed, pickle-free NPZ cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / (cache_path.name + ".tmp")
    arrays = {
        "distance_m": base_maps.distance_m.astype(np.float32),
        "fspl": base_maps.fspl.astype(np.float32),
        "los": base_maps.los.astype(np.float32),
        "wall_count": base_maps.wall_count.astype(np.float32),
        "wall_fraction": base_maps.wall_fraction.astype(np.float32),
        "material_counts": base_maps.material_counts.astype(np.float32),
        "transmittance_sum": base_maps.transmittance_sum.astype(np.float32),
        "reflectance_sum": base_maps.reflectance_sum.astype(np.float32),
        "valid_mask": base_maps.valid_mask.astype(np.uint8),
        "material_ids": np.asarray([int(m) for m in material_ids], dtype=np.int64),
        "feature_names": np.asarray(list(base_maps.feature_names), dtype=str),
    }
    # Write through an open handle so NumPy does not append a second ".npz".
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp_path, cache_path)
    return cache_path


def load_base_maps_cache(cache_path: Union[str, Path]) -> Optional[BaseMaps]:
    """Load base maps from cache, or return ``None`` if missing/unreadable."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if not all(k in data for k in _ARRAY_KEYS):
            return None
        return BaseMaps(
            distance_m=data["distance_m"].astype(np.float32),
            fspl=data["fspl"].astype(np.float32),
            los=data["los"].astype(np.float32),
            wall_count=data["wall_count"].astype(np.float32),
            wall_fraction=data["wall_fraction"].astype(np.float32),
            material_counts=data["material_counts"].astype(np.float32),
            transmittance_sum=data["transmittance_sum"].astype(np.float32),
            reflectance_sum=data["reflectance_sum"].astype(np.float32),
            valid_mask=data["valid_mask"].astype(bool),
            feature_names=[str(x) for x in data["feature_names"].tolist()],
        )
    except Exception:
        # A corrupt or partial cache file is treated as a miss and recomputed.
        return None


def load_or_compute_base_maps_for_sample(
    sample,
    material_ids: Sequence[int],
    min_distance_m: float,
    cache_root: Optional[Union[str, Path]] = None,
    enabled: bool = False,
) -> BaseMaps:
    """Return base maps for a sample, using the on-disk cache when enabled.

    When ``enabled`` is False (the default) or no ``cache_root`` is given, the
    maps are computed directly and nothing is written, so smoke tests stay
    side-effect free unless caching is explicitly turned on.
    """
    material_ids = [int(m) for m in material_ids]
    if not enabled or cache_root is None:
        return build_base_maps_for_sample(sample, material_ids=material_ids, min_distance_m=min_distance_m)

    cache_root = Path(cache_root)
    key = base_maps_cache_key(sample, material_ids, min_distance_m)
    cache_path = cache_root / f"{key}.npz"

    cached = load_base_maps_cache(cache_path)
    if cached is not None:
        return cached

    base_maps = build_base_maps_for_sample(sample, material_ids=material_ids, min_distance_m=min_distance_m)
    try:
        save_base_maps_cache(cache_path, base_maps, material_ids)
    except Exception:
        # Caching is a performance optimization; never fail the run on write error.
        pass
    return base_maps
