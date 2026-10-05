"""Freeze the approved Slice 9 reference pools and their checksums."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import yaml

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_constraints import DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH


DEFAULT_SLICE9_CATALOG_PATH = Path("data/catalog/catalog.sqlite")
DEFAULT_SLICE9_REFERENCE_DIR = Path("data/references/fuxuan/v1")
DEFAULT_SLICE9_FREEZE_RESULT_PATH = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_freeze_v1.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _write_frozen(path: Path, content: bytes) -> bool:
    if path.exists():
        _require(path.read_bytes() == content, f"Frozen artifact content drift; refusing to overwrite {path}")
        return False
    _atomic_write(path, content)
    return True


def build_slice9_reference_freeze(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    constrained_report_path: str | Path = DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH,
    catalog_path: str | Path = DEFAULT_SLICE9_CATALOG_PATH,
    reference_dir: str | Path = DEFAULT_SLICE9_REFERENCE_DIR,
    result_path: str | Path = DEFAULT_SLICE9_FREEZE_RESULT_PATH,
) -> dict[str, Any]:
    config_resolved = Path(config_path).resolve()
    constrained_path = Path(constrained_report_path).resolve()
    config = load_slice9_selection_config(config_resolved)
    constrained = _load_json(constrained_path, label="constrained Slice 9 selection")
    _require(constrained.get("status") == "succeeded", "Constrained selection did not succeed")
    catalog = Catalog(Path(catalog_path).resolve())
    review_rows = catalog.load_slice9_reference_reviews(selection_id=config["selection_id"], selection_version=config["selection_version"])
    approved = {(row["pool_id"], row["asset_sha256"]): row for row in review_rows if row["review_status"] == "approved"}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for pool_id, pool in constrained["pools"].items():
        selected_rows[pool_id] = []
        for rank, item in enumerate(pool["selected"], start=1):
            review = approved.get((pool_id, item["asset_sha256"]))
            _require(review is not None, f"Selected item lacks approved catalog review: {pool_id}/{item['asset_sha256']}")
            selected_rows[pool_id].append({
                "rank": rank,
                "asset_sha256": item["asset_sha256"],
                "fid": item["fid"],
                "audio_relative_path": item["audio_relative_path"],
                "audio_sha256": item["audio_sha256"],
                "speaker_id": item["speaker_id"],
                "emotion_primary": item["emotion_primary"],
                "text_exact": item["text_exact"],
                "text_sha256": item["text_sha256"],
                "text_source": item["text_source"],
                "confidence_tier": item["confidence_tier"],
                "composite_score": item.get("composite_score"),
                "diversity_score": item.get("diversity_score"),
                "review_id": review["review_id"],
                "review_round": review["review_round"],
                "review_batch_id": review["review_batch_id"],
                "review_status": review["review_status"],
            })
    review_snapshot = [
        {key: row[key] for key in ("pool_id", "asset_sha256", "selection_kind", "review_round", "review_status", "review_batch_id", "review_id", "provenance_json")}
        for row in review_rows
    ]
    review_snapshot_sha = _sha256_bytes(_json_bytes(review_snapshot))
    manifest = {
        "schema_version": 1,
        "reference_manifest_schema_version": "slice9_reference_pool_manifest@1",
        "status": "frozen",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "speaker_id": config["input"]["target_speaker_id"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(config_resolved),
            "constrained_report_sha256": _sha256(constrained_path),
            "catalog_review_snapshot_sha256": review_snapshot_sha,
        },
        "pools": {
            pool_id: {
                "purpose": config["pools"][pool_id]["purpose"],
                "top_k": config["pools"][pool_id]["top_k"],
                "selected_count": len(selected_rows[pool_id]),
                "items": selected_rows[pool_id],
            }
            for pool_id in config["pools"]
        },
    }
    stats = {
        "schema_version": 1,
        "stats_schema_version": "slice9_reference_pool_stats@1",
        "status": "frozen",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "pool_count": len(selected_rows),
        "selected_count_total": sum(len(rows) for rows in selected_rows.values()),
        "unique_asset_count": len({row["asset_sha256"] for rows in selected_rows.values() for row in rows}),
        "pool_stats": {
            pool_id: {
                "selected_count": len(rows),
                "asset_sha256s": [row["asset_sha256"] for row in rows],
                "score_min": min(float(row["composite_score"]) for row in rows),
                "score_max": max(float(row["composite_score"]) for row in rows),
                "score_mean": sum(float(row["composite_score"]) for row in rows) / len(rows),
            }
            for pool_id, rows in selected_rows.items()
        },
        "neutral_overlap": sorted({row["asset_sha256"] for row in selected_rows["main_neutral"]} & {row["asset_sha256"] for row in selected_rows["emotion_neutral"]}),
        "review": {"approved_rows": len(review_rows), "selected_rows": sum(len(rows) for rows in selected_rows.values()), "review_batch_ids": sorted({row["review_batch_id"] for row in review_rows})},
    }
    config_payload = yaml.safe_load(config_resolved.read_text(encoding="utf-8"))
    config_json = {"source_path": str(config_resolved), "source_sha256": _sha256(config_resolved), "config": config_payload}
    manifest_bytes = _json_bytes(manifest)
    stats_bytes = _json_bytes(stats)
    config_bytes = _json_bytes(config_json)
    output_dir = Path(reference_dir).resolve()
    files = {"manifest.json": manifest_bytes, "stats.json": stats_bytes, "selection_config.json": config_bytes}
    checksums = "".join(f"{_sha256_bytes(content)}  {name}\n" for name, content in files.items())
    for name, content in files.items():
        _write_frozen(output_dir / name, content)
    _write_frozen(output_dir / "checksums.txt", checksums.encode("ascii"))
    result = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "reference_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "stats_path": str(output_dir / "stats.json"),
        "config_path": str(output_dir / "selection_config.json"),
        "checksums_path": str(output_dir / "checksums.txt"),
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "stats_sha256": _sha256(output_dir / "stats.json"),
        "config_sha256": _sha256(output_dir / "selection_config.json"),
        "checksums_sha256": _sha256(output_dir / "checksums.txt"),
        "artifacts_verified": True,
        "selected_count_total": stats["selected_count_total"],
        "unique_asset_count": stats["unique_asset_count"],
        "neutral_overlap_count": len(stats["neutral_overlap"]),
    }
    result_path_resolved = Path(result_path).resolve()
    _write_frozen(result_path_resolved, _json_bytes(result))
    return {**result, "result_path": str(result_path_resolved)}
