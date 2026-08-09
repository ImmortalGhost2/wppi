from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class SceneSample:
    scene_id: str
    sample_id: str
    scene_path: Path
    path_loss: np.ndarray
    valid_mask: np.ndarray
    wall_mask: np.ndarray
    material_map: np.ndarray
    tx_xy: np.ndarray
    frequency_hz: float
    resolution_m: float
    transmittance: np.ndarray | None = None
    reflectance: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.path_loss.shape)


def resolve_scene_path(data_root: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (Path(data_root).resolve() / p).resolve()


def _first_key(obj: np.lib.npyio.NpzFile, keys: list[str]) -> str | None:
    for key in keys:
        if key in obj:
            return key
    return None


def _scalar_from_obj(obj: np.lib.npyio.NpzFile, row: Dict[str, Any], keys: list[str], default: float | None = None) -> float:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return float(row[key])
        if key in obj:
            return float(np.asarray(obj[key]).reshape(-1)[0])
    if default is not None:
        return float(default)
    raise ValueError(f"Missing scalar. Tried keys: {keys}")


def _coerce_to_str(value: Any) -> str:
    """Robustly turn an NPZ string entry into a Python str.

    Handles plain str, bytes/``np.bytes_``, NumPy unicode/byte string arrays,
    0-d scalar arrays, and (defensively) object arrays. Works without pickle.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _coerce_to_str(value.item())
        flat = value.reshape(-1)
        return _coerce_to_str(flat[0]) if flat.size else ""
    if isinstance(value, np.generic):
        return _coerce_to_str(value.item())
    return str(value)


def _id_from_obj(obj: np.lib.npyio.NpzFile, row: Dict[str, Any], key: str, fallback: str) -> str:
    if key in row and pd.notna(row[key]):
        return str(row[key])
    if key in obj:
        return _coerce_to_str(obj[key])
    return fallback


def load_scene_npz(path: Path, row: Dict[str, Any] | None = None) -> SceneSample:
    row = row or {}
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=False)

    pl_key = _first_key(obj, ["path_loss", "path_loss_db", "target", "target_db"])
    valid_key = _first_key(obj, ["valid_mask", "mask", "receiver_mask"])
    material_key = _first_key(obj, ["material_map", "materials", "building_map"])
    if pl_key is None or valid_key is None or material_key is None:
        missing = []
        if pl_key is None:
            missing.append("path_loss/path_loss_db")
        if valid_key is None:
            missing.append("valid_mask")
        if material_key is None:
            missing.append("material_map")
        raise ValueError(f"Scene file {path} missing required arrays: {missing}")

    path_loss = obj[pl_key].astype(np.float32)
    valid = obj[valid_key].astype(bool)
    material = obj[material_key].astype(np.int32)
    wall = obj["wall_mask"].astype(bool) if "wall_mask" in obj else (material > 0)
    if path_loss.shape != valid.shape or path_loss.shape != wall.shape or path_loss.shape != material.shape:
        raise ValueError(f"Scene arrays in {path} have inconsistent shapes.")

    if "tx_xy" in obj:
        tx_xy = obj["tx_xy"].astype(np.float32).reshape(-1)[:2]
    elif "tx_x" in obj and "tx_y" in obj:
        tx_xy = np.asarray([float(np.asarray(obj["tx_x"]).reshape(-1)[0]), float(np.asarray(obj["tx_y"]).reshape(-1)[0])], dtype=np.float32)
    elif "tx_x" in row and "tx_y" in row:
        tx_xy = np.asarray([float(row["tx_x"]), float(row["tx_y"])], dtype=np.float32)
    else:
        raise ValueError(f"Scene file {path} missing tx_xy or tx_x/tx_y.")

    scene_id = _id_from_obj(obj, row, "scene_id", path.stem)
    sample_id = _id_from_obj(obj, row, "sample_id", path.stem)
    trans = obj["transmittance"].astype(np.float32) if "transmittance" in obj else None
    refl = obj["reflectance"].astype(np.float32) if "reflectance" in obj else None
    return SceneSample(
        scene_id=scene_id,
        sample_id=sample_id,
        scene_path=path,
        path_loss=path_loss,
        valid_mask=valid,
        wall_mask=wall,
        material_map=material,
        tx_xy=tx_xy,
        frequency_hz=_scalar_from_obj(obj, row, ["frequency_hz", "freq_hz"]),
        resolution_m=_scalar_from_obj(obj, row, ["resolution_m", "cell_size_m", "pixel_size_m"]),
        transmittance=trans,
        reflectance=refl,
    )


class WallPathManifest:
    REQUIRED_COLUMNS = {"scene_id", "sample_id"}

    def __init__(self, csv_file: str | Path, data_root: str | Path) -> None:
        self.csv_file = Path(csv_file).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.csv_file.exists():
            raise FileNotFoundError(self.csv_file)
        self.df = pd.read_csv(self.csv_file)
        if "scene_path" not in self.df.columns and "map_path" in self.df.columns:
            self.df = self.df.rename(columns={"map_path": "scene_path"})
        missing = (self.REQUIRED_COLUMNS | {"scene_path"}) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required manifest columns in {self.csv_file.name}: {sorted(missing)}")
        if self.df.empty:
            raise ValueError(f"Manifest is empty: {self.csv_file}")

    def __len__(self) -> int:
        return len(self.df)

    def scene_path_for_row(self, idx: int) -> Path:
        return resolve_scene_path(self.data_root, str(self.df.iloc[int(idx)]["scene_path"]))

    def iter_samples(self) -> Iterable[SceneSample]:
        for _, row in self.df.iterrows():
            row_dict = row.to_dict()
            yield load_scene_npz(resolve_scene_path(self.data_root, str(row["scene_path"])), row=row_dict)

    def sample(self, idx: int) -> SceneSample:
        row = self.df.iloc[int(idx)].to_dict()
        return load_scene_npz(resolve_scene_path(self.data_root, str(row["scene_path"])), row=row)

    def scene_ids(self) -> List[str]:
        return self.df["scene_id"].astype(str).tolist()
