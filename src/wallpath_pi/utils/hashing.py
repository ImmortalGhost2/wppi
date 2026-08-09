from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def file_hashes(path: Path) -> Dict[str, str]:
    """Return MD5 and SHA256 for a single file."""
    path = Path(path).expanduser().resolve()
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def stable_int_hash(*parts: object, modulo: int = 2 ** 32) -> int:
    """Deterministic integer hash used for splits and sparse masks."""
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % int(modulo)


def _skip_path(path: Path) -> bool:
    ignored = {".git", "__pycache__", ".venv", ".pytest_cache", "build", "dist", "*.egg-info"}
    parts = set(path.parts)
    return any(part in ignored for part in parts) or any(part.endswith(".egg-info") for part in parts)


def directory_file_hashes(
    root: Path,
    patterns: Iterable[str] = ("*.py", "*.yaml", "*.yml", "*.md", "*.toml", "*.json", "*.csv"),
) -> List[Dict[str, str]]:
    """Return per-file hashes for a reproducibility manifest."""
    root = Path(root).expanduser().resolve()
    rows: List[Dict[str, str]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file() or path in seen or _skip_path(path):
                continue
            seen.add(path)
            h = file_hashes(path)
            rows.append({"path": str(path.relative_to(root)), "md5": h["md5"], "sha256": h["sha256"], "bytes": path.stat().st_size})
    return rows


def write_sha256_manifest(root: Path, out: Path, patterns: Sequence[str] | None = None) -> Path:
    """Write a text SHA256 manifest, compatible with common research artifact practice."""
    root = Path(root).expanduser().resolve()
    out = Path(out)
    if not out.is_absolute():
        out = root / out
    files = directory_file_hashes(root, patterns=patterns or ("*.py", "*.yaml", "*.yml", "*.md", "*.toml", "*.json", "*.csv", "*.npz"))
    lines = [f"{row['sha256']}  {row['path']}" for row in files]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def tree_manifest(
    root: Path,
    patterns: Iterable[str] | None = None,
    exclude_paths: Iterable[Path] = (),
) -> Dict[str, object]:
    """Return a JSON-serializable SHA256/MD5 manifest for a file tree.

    Paths listed in ``exclude_paths`` are omitted after absolute-path
    resolution. This allows a manifest written inside ``root`` to exclude its
    own previous output and remain idempotent across repeated invocations.
    """
    root = Path(root).expanduser().resolve()
    excluded = {
        Path(path).expanduser().resolve()
        for path in exclude_paths
    }
    rows: List[Dict[str, str]] = []
    exclude_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in excluded:
            continue

        rel = path.relative_to(root)
        if any(part in exclude_parts for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if patterns is not None and not any(
            path.match(pattern)
            for pattern in patterns
        ):
            continue

        hashes = file_hashes(path)
        rows.append(
            {
                "path": rel.as_posix(),
                "md5": hashes["md5"],
                "sha256": hashes["sha256"],
            }
        )

    manifest_bytes = json.dumps(rows, sort_keys=True).encode("utf-8")
    return {
        "root": str(root),
        "num_files": len(rows),
        "files": rows,
        "manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
    }
