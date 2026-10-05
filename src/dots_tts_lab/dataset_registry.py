from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_artifacts import validate_dataset_artifact_set
from dots_tts_lab.dataset_freeze import DatasetFreezeConfig
from dots_tts_lab.dataset_publish import dataset_tree_inventory
from dots_tts_lab.duplicate_graph import canonical_json


_ITEM_FIELDS = (
    "ordinal",
    "fid",
    "asset_sha256",
    "split",
    "analysis_run_id",
    "group_id",
    "source_root_key",
    "source_path_key",
    "derived_id",
    "audio_relative_path",
    "audio_sha256",
    "quality_run_id",
    "quality_decision",
    "quality_reasons_json",
    "text_exact",
    "text_sha256",
    "text_source",
    "emotion_weak_label",
    "emotion_primary",
    "emotion_secondary",
    "intensity",
    "label_source",
    "review_decision_id",
    "review_round",
    "review_batch_id",
    "lineage_json",
)


def publish_dataset_catalog(
    catalog: Catalog,
    root: str | Path,
    *,
    config: DatasetFreezeConfig,
    published_at: str,
) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    validation = validate_frozen_dataset(dataset_root, config=config)
    manifest = _load_json(dataset_root / config.output.manifest_json)
    rows = pq.read_table(dataset_root / config.output.canonical_parquet).to_pylist()
    inventory = dataset_tree_inventory(dataset_root)
    speaker_snapshot_sha256 = _speaker_review_snapshot_sha256(
        catalog, manifest["analysis_run_id"]
    )
    version_values = {
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "analysis_run_id": manifest["analysis_run_id"],
        "candidate_snapshot_sha256": manifest["candidate_snapshot_sha256"],
        "edge_review_snapshot_sha256": manifest["edge_review_snapshot_sha256"],
        "speaker_review_snapshot_sha256": speaker_snapshot_sha256,
        "item_count": manifest["item_count"],
        "total_duration_seconds": manifest["total_duration_seconds"],
        "artifact_set_json": canonical_json(inventory["files"]),
        "artifact_set_sha256": inventory["tree_sha256"],
        "parquet_path": str(dataset_root / config.output.canonical_parquet),
        "parquet_sha256": _sha256_file(
            dataset_root / config.output.canonical_parquet
        ),
        "manifest_path": str(dataset_root / config.output.manifest_json),
        "manifest_sha256": _sha256_file(dataset_root / config.output.manifest_json),
        "checksums_path": str(dataset_root / config.output.checksums),
        "checksums_sha256": _sha256_file(dataset_root / config.output.checksums),
        "published_at": published_at,
    }
    item_values = [{field: row[field] for field in _ITEM_FIELDS} for row in rows]

    with catalog.session() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT * FROM dataset_version
            WHERE dataset_id = ? AND dataset_version = ?
            """,
            (config.dataset_id, config.dataset_version),
        ).fetchone()
        if existing is not None:
            comparable = {key: existing[key] for key in version_values if key != "published_at"}
            requested = {
                key: value for key, value in version_values.items() if key != "published_at"
            }
            if comparable != requested:
                raise RuntimeError(
                    "Dataset catalog version changed without a version bump: "
                    f"{config.dataset_id}@{config.dataset_version}"
                )
            _assert_catalog_items(connection, config, item_values)
            return {
                "action": "cached",
                "speaker_review_snapshot_sha256": speaker_snapshot_sha256,
                **validation,
            }

        columns = tuple(version_values)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO dataset_version ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(version_values[column] for column in columns),
        )
        item_columns = ("dataset_id", "dataset_version", *_ITEM_FIELDS)
        item_placeholders = ", ".join("?" for _ in item_columns)
        connection.executemany(
            f"INSERT INTO dataset_item ({', '.join(item_columns)}) "
            f"VALUES ({item_placeholders})",
            [
                (
                    config.dataset_id,
                    config.dataset_version,
                    *(row[field] for field in _ITEM_FIELDS),
                )
                for row in item_values
            ],
        )
    return {
        "action": "published",
        "speaker_review_snapshot_sha256": speaker_snapshot_sha256,
        **validation,
    }


def validate_frozen_dataset(
    root: str | Path,
    *,
    config: DatasetFreezeConfig,
    catalog: Catalog | None = None,
) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    basic = validate_dataset_artifact_set(dataset_root, config=config)
    parquet_rows = pq.read_table(
        dataset_root / config.output.canonical_parquet
    ).to_pylist()
    seen_audio = {}
    speakers = set()
    for row in parquet_rows:
        audio_path = Path(row["audio_absolute_path"])
        if not audio_path.is_absolute() or not audio_path.is_file():
            raise RuntimeError(f"Frozen audio path is missing: {audio_path}")
        digest = _sha256_file(audio_path)
        if digest != row["audio_sha256"]:
            raise RuntimeError(
                f"Frozen audio SHA-256 drift for {row['asset_sha256']}: {audio_path}"
            )
        if not row["text_exact"].strip() or not row["speaker_id"].strip():
            raise RuntimeError(f"Empty text or speaker for {row['asset_sha256']}")
        if row["audio_absolute_path"] in seen_audio:
            raise RuntimeError("Frozen dataset reuses one audio path for multiple assets")
        seen_audio[row["audio_absolute_path"]] = row["asset_sha256"]
        speakers.add(row["speaker_id"])
    if len(speakers) != 1:
        raise RuntimeError(f"Frozen single-speaker dataset has speakers: {sorted(speakers)}")
    result = {**basic, "audio_hashes_verified": len(parquet_rows), "speakers": sorted(speakers)}
    if catalog is not None:
        with catalog.read_only_session() as connection:
            _assert_catalog_items(connection, config, parquet_rows)
        result["catalog_items_verified"] = len(parquet_rows)
    return result


def _assert_catalog_items(
    connection: Any,
    config: DatasetFreezeConfig,
    expected_rows: list[dict[str, Any]],
) -> None:
    actual_rows = connection.execute(
        """
        SELECT ordinal, fid, asset_sha256, split, analysis_run_id, group_id,
               source_root_key, source_path_key, derived_id, audio_relative_path,
               audio_sha256, quality_run_id, quality_decision,
               quality_reasons_json, text_exact, text_sha256, text_source,
               emotion_weak_label, emotion_primary, emotion_secondary, intensity,
               label_source, review_decision_id, review_round, review_batch_id,
               lineage_json
        FROM dataset_item
        WHERE dataset_id = ? AND dataset_version = ?
        ORDER BY ordinal
        """,
        (config.dataset_id, config.dataset_version),
    ).fetchall()
    actual = [{field: row[field] for field in _ITEM_FIELDS} for row in actual_rows]
    expected = [{field: row[field] for field in _ITEM_FIELDS} for row in expected_rows]
    if actual != expected:
        raise RuntimeError("Parquet and catalog dataset_item rows differ")


def _speaker_review_snapshot_sha256(catalog: Catalog, run_id: str) -> str:
    with catalog.read_only_session() as connection:
        rows = connection.execute(
            """
            SELECT review_id, asset_sha256, review_round, review_status,
                   review_note, created_at, review_batch_id, source_row_index
            FROM dataset_speaker_review current
            WHERE run_id = ? AND review_round = (
                SELECT MAX(candidate.review_round)
                FROM dataset_speaker_review candidate
                WHERE candidate.run_id = current.run_id
                  AND candidate.asset_sha256 = current.asset_sha256
            )
            ORDER BY asset_sha256
            """,
            (run_id,),
        ).fetchall()
    payload = [dict(row) for row in rows]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
