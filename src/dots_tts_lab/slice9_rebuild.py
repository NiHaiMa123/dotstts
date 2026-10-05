"""Rebuild Slice 9 frozen artifacts in isolation and compare bytes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH
from dots_tts_lab.slice9_freeze import (
    DEFAULT_SLICE9_CATALOG_PATH,
    DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH,
    build_slice9_reference_freeze,
)


DEFAULT_SLICE9_REBUILD_RESULT_PATH = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_rebuild_v1.json")
DEFAULT_SLICE9_REFERENCE_DIR = Path("data/references/fuxuan/v1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def rebuild_slice9_reference_artifacts(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    constrained_report_path: str | Path = DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH,
    catalog_path: str | Path = DEFAULT_SLICE9_CATALOG_PATH,
    reference_dir: str | Path = DEFAULT_SLICE9_REFERENCE_DIR,
    output_path: str | Path = DEFAULT_SLICE9_REBUILD_RESULT_PATH,
) -> dict[str, Any]:
    canonical_dir = Path(reference_dir).resolve()
    names = ("manifest.json", "stats.json", "selection_config.json", "checksums.txt")
    with tempfile.TemporaryDirectory(prefix="slice9-rebuild-") as temporary:
        rebuilt_dir = Path(temporary) / "reference"
        build_slice9_reference_freeze(
            config_path=config_path,
            constrained_report_path=constrained_report_path,
            catalog_path=catalog_path,
            reference_dir=rebuilt_dir,
            result_path=Path(temporary) / "freeze_result.json",
        )
        comparisons = {}
        for name in names:
            canonical = canonical_dir / name
            rebuilt = rebuilt_dir / name
            if not canonical.exists() or not rebuilt.exists():
                comparisons[name] = {"match": False, "reason": "missing"}
                continue
            canonical_bytes = canonical.read_bytes()
            rebuilt_bytes = rebuilt.read_bytes()
            comparisons[name] = {
                "match": canonical_bytes == rebuilt_bytes,
                "canonical_sha256": _sha256(canonical),
                "rebuilt_sha256": _sha256(rebuilt),
                "byte_count": len(canonical_bytes),
            }
        mismatches = [name for name, comparison in comparisons.items() if not comparison["match"]]
    if mismatches:
        raise RuntimeError("Slice 9 isolated rebuild differs: " + ", ".join(mismatches))
    result = {
        "schema_version": 1,
        "status": "verified",
        "reference_dir": str(canonical_dir),
        "artifact_count": len(names),
        "byte_identical": True,
        "comparisons": comparisons,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

