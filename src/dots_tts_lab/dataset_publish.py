from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from dots_tts_lab.duplicate_graph import canonical_json


class DatasetVersionConflict(RuntimeError):
    pass


def atomic_publish_dataset(
    target: str | Path,
    *,
    build: Callable[[Path], None],
    validate: Callable[[Path], None],
) -> dict[str, Any]:
    destination = Path(target).resolve()
    parent = destination.parent
    if destination == parent or not destination.name:
        raise RuntimeError("Dataset publish target must be a version directory")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=parent,
        )
    ).resolve()
    _assert_safe_staging(staging, destination)
    published = False
    try:
        build(staging)
        validate(staging)
        desired = dataset_tree_inventory(staging)
        if destination.exists():
            if not destination.is_dir():
                raise DatasetVersionConflict(
                    f"Dataset version target is not a directory: {destination}"
                )
            existing = dataset_tree_inventory(destination)
            if existing != desired:
                raise DatasetVersionConflict(
                    "Dataset version already exists with different content: "
                    f"{destination}"
                )
            return {
                "action": "cached",
                "target": str(destination),
                **_inventory_summary(existing),
            }
        os.replace(staging, destination)
        published = True
        return {
            "action": "published",
            "target": str(destination),
            **_inventory_summary(desired),
        }
    finally:
        if not published and staging.exists():
            _assert_safe_staging(staging, destination)
            shutil.rmtree(staging)


def dataset_tree_inventory(root: str | Path) -> dict[str, Any]:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Dataset artifact directory does not exist: {directory}")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Dataset artifacts cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError("Dataset artifact directory is empty")
    tree_sha256 = hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()
    return {"files": files, "tree_sha256": tree_sha256}


def _inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_count": len(inventory["files"]),
        "total_bytes": sum(item["size_bytes"] for item in inventory["files"]),
        "tree_sha256": inventory["tree_sha256"],
    }


def _assert_safe_staging(staging: Path, destination: Path) -> None:
    if (
        staging.parent != destination.parent
        or not staging.name.startswith(f".{destination.name}.")
        or not staging.name.endswith(".staging")
        or staging == destination
    ):
        raise RuntimeError(f"Unsafe dataset staging path: {staging}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
