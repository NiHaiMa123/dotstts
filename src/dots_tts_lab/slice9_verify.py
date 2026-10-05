"""Verify frozen Slice 9 reference pools against catalog and files."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_freeze import DEFAULT_SLICE9_CATALOG_PATH, DEFAULT_SLICE9_REFERENCE_DIR


DEFAULT_SLICE9_VERIFY_RESULT_PATH = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_verify_v1.json")
_STANDARDIZED_ROOT = Path("data/work/standardized")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def verify_slice9_reference_pools(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    reference_dir: str | Path = DEFAULT_SLICE9_REFERENCE_DIR,
    catalog_path: str | Path = DEFAULT_SLICE9_CATALOG_PATH,
    output_path: str | Path = DEFAULT_SLICE9_VERIFY_RESULT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    reference_root = Path(reference_dir).resolve()
    manifest_path = reference_root / "manifest.json"
    stats_path = reference_root / "stats.json"
    config_artifact_path = reference_root / "selection_config.json"
    checksums_path = reference_root / "checksums.txt"
    manifest = _load_json(manifest_path, label="frozen Slice 9 manifest")
    stats = _load_json(stats_path, label="frozen Slice 9 stats")
    _require(manifest.get("status") == "frozen" and stats.get("status") == "frozen", "Frozen artifacts are not frozen")
    expected_checksums = {
        "manifest.json": _sha256(manifest_path),
        "stats.json": _sha256(stats_path),
        "selection_config.json": _sha256(config_artifact_path),
    }
    checksum_rows = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        checksum_rows[name] = digest
    _require(checksum_rows == expected_checksums, "Frozen checksums do not match artifact bytes")
    catalog = Catalog(Path(catalog_path).resolve())
    with catalog.read_only_session() as connection:
        dataset_rows = {
            row["asset_sha256"]: dict(row)
            for row in connection.execute(
                """
                SELECT item.*, source.speaker_id AS source_speaker_id
                FROM dataset_item AS item
                JOIN source_location AS source
                  ON source.root_key = item.source_root_key
                 AND source.path_key = item.source_path_key
                WHERE item.dataset_id = ? AND item.dataset_version = ?
                """,
                (config["input"]["dataset_id"], config["input"]["dataset_version"]),
            ).fetchall()
        }
    reviews = catalog.load_slice9_reference_reviews(selection_id=config["selection_id"], selection_version=config["selection_version"])
    latest_reviews = {(row["pool_id"], row["asset_sha256"]): row for row in reviews}
    errors: list[str] = []
    verified_items = 0
    pool_asset_sets: dict[str, set[str]] = {}
    for pool_id, pool in manifest["pools"].items():
        pool_asset_sets[pool_id] = set()
        expected_emotion = config["pools"][pool_id]["emotion_primary"]
        for expected_rank, item in enumerate(pool["items"], start=1):
            asset = item["asset_sha256"]
            pool_asset_sets[pool_id].add(asset)
            if int(item["rank"]) != expected_rank:
                errors.append(f"{pool_id}/{asset}: rank is not sequential")
            row = dataset_rows.get(asset)
            if row is None:
                errors.append(f"{pool_id}/{asset}: missing dataset_item")
                continue
            if row["split"] != "train":
                errors.append(f"{pool_id}/{asset}: split={row['split']}, expected train")
            if row["audio_sha256"] != item["audio_sha256"]:
                errors.append(f"{pool_id}/{asset}: audio SHA mismatch")
            if row["text_sha256"] != item["text_sha256"] or row["text_exact"] != item["text_exact"]:
                errors.append(f"{pool_id}/{asset}: text mismatch")
            if row["source_speaker_id"] != config["input"]["target_speaker_id"] or item["speaker_id"] != config["input"]["target_speaker_id"]:
                errors.append(f"{pool_id}/{asset}: speaker mismatch")
            if row["emotion_primary"] != expected_emotion or item["emotion_primary"] != expected_emotion:
                errors.append(f"{pool_id}/{asset}: pool emotion mismatch")
            audio_path = (_STANDARDIZED_ROOT / item["audio_relative_path"]).resolve()
            if not audio_path.is_file():
                errors.append(f"{pool_id}/{asset}: audio file missing")
            elif _sha256(audio_path) != item["audio_sha256"]:
                errors.append(f"{pool_id}/{asset}: audio file hash mismatch")
            review = latest_reviews.get((pool_id, asset))
            if review is None or review["review_status"] != "approved":
                errors.append(f"{pool_id}/{asset}: missing approved catalog review")
            verified_items += 1
    emotion_ids = [pool_id for pool_id in pool_asset_sets if pool_id.startswith("emotion_")]
    emotion_overlap = []
    for index, left in enumerate(emotion_ids):
        for right in emotion_ids[index + 1 :]:
            overlap = pool_asset_sets[left] & pool_asset_sets[right]
            if overlap:
                emotion_overlap.extend(sorted(overlap))
    if emotion_overlap:
        errors.append("emotion pool overlap: " + ",".join(sorted(set(emotion_overlap))))
    neutral_overlap = sorted(pool_asset_sets.get("main_neutral", set()) & pool_asset_sets.get("emotion_neutral", set()))
    if len(neutral_overlap) > int(config["ranking"]["neutral_overlap_max"] if "neutral_overlap_max" in config.get("ranking", {}) else config["ranking"]["neutral_pool_overlap_max"]):
        errors.append("neutral pool overlap exceeds cap")
    _require(not errors, "Slice 9 reference verification failed: " + "; ".join(errors[:12]))
    result = {
        "schema_version": 1,
        "status": "verified",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "manifest_sha256": _sha256(manifest_path),
        "stats_sha256": _sha256(stats_path),
        "checksums_sha256": _sha256(checksums_path),
        "catalog_path": str(Path(catalog_path).resolve()),
        "verified_item_count": verified_items,
        "verified_pool_count": len(pool_asset_sets),
        "unique_asset_count": len(set().union(*pool_asset_sets.values())),
        "neutral_overlap_count": len(neutral_overlap),
        "errors": errors,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

