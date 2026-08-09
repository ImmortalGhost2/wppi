from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from wallpath_pi.data.dataset import WallPathManifest
from wallpath_pi.geometry import cache as cache_mod
from wallpath_pi.geometry.cache import (
    base_maps_cache_key,
    load_base_maps_cache,
    load_or_compute_base_maps_for_sample,
)
from wallpath_pi.geometry.features import build_base_maps_for_sample
from wallpath_pi.data.synthetic import generate_synthetic_dataset
from wallpath_pi.utils.config import load_config


MATERIAL_IDS = [1, 2, 3]
MIN_DISTANCE_M = 0.25


def _make_sample(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    synth_cfg = dict(cfg["synthetic"])
    synth_cfg.update({"num_scenes": 1, "samples_per_scene": 1, "image_size": 20, "overwrite": True})
    data_root = tmp_path / "data"
    generate_synthetic_dataset(data_root, synth_cfg)
    manifest = WallPathManifest(data_root / "manifest.csv", data_root)
    return manifest.sample(0)


def _assert_base_maps_equal(a, b):
    for field in (
        "distance_m", "fspl", "los", "wall_count", "wall_fraction",
        "material_counts", "transmittance_sum", "reflectance_sum",
    ):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.valid_mask, b.valid_mask)
    assert a.feature_names == b.feature_names


def test_first_call_computes_and_writes_cache(tmp_path):
    sample = _make_sample(tmp_path)
    cache_root = tmp_path / "cache"
    assert not cache_root.exists()
    out = load_or_compute_base_maps_for_sample(
        sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=True,
    )
    files = list(cache_root.glob("*.npz"))
    assert len(files) == 1
    reference = build_base_maps_for_sample(sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M)
    _assert_base_maps_equal(out, reference)
    # The written file is loadable without pickling and matches.
    loaded = load_base_maps_cache(files[0])
    assert loaded is not None
    _assert_base_maps_equal(loaded, reference)


def test_second_call_loads_from_cache(tmp_path, monkeypatch):
    sample = _make_sample(tmp_path)
    cache_root = tmp_path / "cache"
    reference = load_or_compute_base_maps_for_sample(
        sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=True,
    )

    # Any recompute now would be a cache miss; force it to fail so a hit is proven.
    def _boom(*args, **kwargs):
        raise AssertionError("base maps were recomputed instead of loaded from cache")

    monkeypatch.setattr(cache_mod, "build_base_maps_for_sample", _boom)
    out = load_or_compute_base_maps_for_sample(
        sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=True,
    )
    _assert_base_maps_equal(out, reference)


def test_cache_key_invalidates_on_material_ids_and_frequency(tmp_path):
    sample = _make_sample(tmp_path)
    base_key = base_maps_cache_key(sample, MATERIAL_IDS, MIN_DISTANCE_M)

    # Changing material ids changes the key.
    assert base_maps_cache_key(sample, [1, 2], MIN_DISTANCE_M) != base_key
    # Changing frequency changes the key.
    other_freq = dataclasses.replace(sample, frequency_hz=float(sample.frequency_hz) * 2.0)
    assert base_maps_cache_key(other_freq, MATERIAL_IDS, MIN_DISTANCE_M) != base_key
    # Same inputs reproduce the same key.
    assert base_maps_cache_key(sample, MATERIAL_IDS, MIN_DISTANCE_M) == base_key


def test_changed_material_ids_writes_separate_cache_entry(tmp_path):
    sample = _make_sample(tmp_path)
    cache_root = tmp_path / "cache"
    load_or_compute_base_maps_for_sample(
        sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=True,
    )
    load_or_compute_base_maps_for_sample(
        sample, material_ids=[1, 2], min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=True,
    )
    assert len(list(cache_root.glob("*.npz"))) == 2


def test_cache_disabled_writes_nothing(tmp_path):
    sample = _make_sample(tmp_path)
    cache_root = tmp_path / "cache"
    out = load_or_compute_base_maps_for_sample(
        sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M,
        cache_root=cache_root, enabled=False,
    )
    assert not cache_root.exists()
    reference = build_base_maps_for_sample(sample, material_ids=MATERIAL_IDS, min_distance_m=MIN_DISTANCE_M)
    _assert_base_maps_equal(out, reference)


def test_scene_signature_uses_size_and_mtime_metadata(tmp_path):
    """The scene signature is file metadata (size + mtime), not a content hash."""
    scene = tmp_path / "scene.bin"
    scene.write_bytes(b"abc")
    sig = cache_mod._scene_signature(scene)
    assert sig != "missing"
    # The signature is exactly "<size>:<mtime_ns>" of the file metadata.
    st = scene.stat()
    assert sig == f"{st.st_size}:{st.st_mtime_ns}"
    # A missing file yields a sentinel rather than raising.
    assert cache_mod._scene_signature(tmp_path / "absent.bin") == "missing"


def test_cache_key_tracks_scene_file_signature(tmp_path):
    """The key is file-signature-addressed: path, byte size, and mtime all matter."""
    sample = _make_sample(tmp_path)
    scene = tmp_path / "scene_stub.bin"
    scene.write_bytes(b"x" * 16)
    stubbed = dataclasses.replace(sample, scene_path=scene)

    key_initial = base_maps_cache_key(stubbed, MATERIAL_IDS, MIN_DISTANCE_M)
    # Same metadata reproduces the same key.
    assert base_maps_cache_key(stubbed, MATERIAL_IDS, MIN_DISTANCE_M) == key_initial

    # A byte-size change invalidates the entry.
    scene.write_bytes(b"x" * 32)
    key_resized = base_maps_cache_key(stubbed, MATERIAL_IDS, MIN_DISTANCE_M)
    assert key_resized != key_initial

    # Bumping only the modification time also invalidates the entry.
    st = scene.stat()
    os.utime(scene, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    key_touched = base_maps_cache_key(stubbed, MATERIAL_IDS, MIN_DISTANCE_M)
    assert key_touched != key_resized

    # Pointing at a different path also changes the key.
    other = tmp_path / "other_scene.bin"
    scene.replace(other)
    key_moved = base_maps_cache_key(
        dataclasses.replace(sample, scene_path=other), MATERIAL_IDS, MIN_DISTANCE_M
    )
    assert key_moved != key_touched
