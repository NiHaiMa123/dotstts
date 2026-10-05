#!/usr/bin/env python3
"""Rewrite frozen Windows JSONL audio paths for the Slice12 Linux container.

The frozen dataset remains byte-for-byte unchanged.  Derived manifests keep the
official three-field schema and add a separate provenance sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_relative_path(value: str) -> PurePosixPath:
    """Extract the path below data/work/standardized on either host OS."""
    parts = PureWindowsPath(value).parts if "\\" in value else PurePosixPath(value).parts
    lowered = [part.casefold() for part in parts]
    marker = ["data", "work", "standardized"]
    for index in range(len(parts) - len(marker) + 1):
        if lowered[index:index + len(marker)] == marker:
            relative = parts[index + len(marker):]
            if not relative:
                break
            return PurePosixPath(*relative)
    raise ValueError(f"Audio path is outside data/work/standardized: {value}")


def convert_manifest(
    source_path: Path,
    output_path: Path,
    *,
    actual_audio_root: Path,
    container_audio_root: PurePosixPath,
) -> dict[str, Any]:
    rows = []
    audio_hashes = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source_row = json.loads(line)
        if set(source_row) != {"fid", "audio", "text"}:
            raise ValueError(f"Unexpected frozen JSONL fields at {source_path}:{line_number}")
        relative = standardized_relative_path(str(source_row["audio"]))
        actual_audio = actual_audio_root.joinpath(*relative.parts)
        if not actual_audio.is_file():
            raise FileNotFoundError(f"Frozen audio is missing: {actual_audio}")
        audio_hashes.append({"fid": str(source_row["fid"]), "sha256": sha256_file(actual_audio)})
        rows.append(
            {
                "audio": str(container_audio_root / relative),
                "fid": str(source_row["fid"]),
                "text": str(source_row["text"]),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "row_count": len(rows),
        "audio_set_sha256": hashlib.sha256(
            json.dumps(audio_hashes, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def build(config_path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 2:
        raise ValueError("Container manifests require Slice12 contract schema_version 2")
    workspace_mount = PurePosixPath(config["runtime"]["workspace_mount"])
    container_audio_root = workspace_mount / "data/work/standardized"
    output_root = (repo_root / config["outputs"]["container_manifest_root"]).resolve()
    actual_audio_root = (repo_root / "data/work/standardized").resolve()
    manifests = {}
    for split, config_key in (("train", "train_manifest"), ("validation", "validation_manifest"), ("test", "test_manifest")):
        source = (repo_root / config["dataset"][config_key]).resolve()
        manifests[split] = convert_manifest(
            source,
            output_root / f"{split}.jsonl",
            actual_audio_root=actual_audio_root,
            container_audio_root=container_audio_root,
        )
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_file(config_path),
        "frozen_tree_sha256": config["dataset"]["frozen_tree_sha256"],
        "container_audio_root": str(container_audio_root),
        "manifests": manifests,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice12/experiment_contract_v2.yaml"))
    args = parser.parse_args()
    print(json.dumps(build(ROOT / args.config), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
