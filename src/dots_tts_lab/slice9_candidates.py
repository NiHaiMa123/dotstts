from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import build_candidate_snapshot, write_candidate_snapshot


DEFAULT_SLICE9_CONFIG_PATH = Path(
    "configs/lab/datasets/fuxuan_slice9_v1.yaml"
)
DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json"
)
DEFAULT_SLICE9_AUDIT_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/slice9_candidate_audit.json"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def load_slice9_selection_config(
    path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"Cannot load Slice 9 config {config_path}: {error}") from error
    _require(isinstance(payload, dict), "Slice 9 config must be a mapping")
    _require(payload.get("schema_version") == 1, "Unsupported Slice 9 config schema")
    _require(
        isinstance(payload.get("selection_id"), str)
        and payload["selection_id"],
        "Slice 9 config selection_id is required",
    )
    _require(
        isinstance(payload.get("selection_version"), int)
        and payload["selection_version"] >= 1,
        "Slice 9 config selection_version must be a positive integer",
    )
    _require(
        isinstance(payload.get("implementation_version"), int)
        and payload["implementation_version"] >= 1,
        "Slice 9 config implementation_version must be a positive integer",
    )

    input_config = payload.get("input")
    _require(isinstance(input_config, dict), "Slice 9 input config is required")
    _require(input_config.get("dataset_id") == "fuxuan", "Slice 9 dataset_id must be fuxuan")
    _require(input_config.get("dataset_version") == 1, "Slice 9 dataset_version must be 1")
    _require_sha256(input_config.get("dataset_tree_sha256"), "dataset_tree_sha256")
    _require(input_config.get("source") == "catalog.dataset_item", "Slice 9 source must be catalog.dataset_item")
    _require(input_config.get("split") == "train", "Slice 9 candidates must use train split")
    _require(
        isinstance(input_config.get("target_speaker_id"), str)
        and input_config["target_speaker_id"].strip(),
        "Slice 9 target_speaker_id is required",
    )
    _require(input_config.get("candidate_snapshot_required") is True, "Slice 9 candidate snapshot is required")

    weights = payload.get("score", {}).get("weights")
    _require(isinstance(weights, dict) and weights, "Slice 9 score weights are required")
    _require(
        all(isinstance(value, (int, float)) and value >= 0 for value in weights.values()),
        "Slice 9 score weights must be non-negative numbers",
    )
    _require(
        abs(sum(float(value) for value in weights.values()) - 1.0) <= 1e-9,
        "Slice 9 score weights must sum to 1",
    )
    normalization = payload.get("score", {}).get("normalization")
    _require(isinstance(normalization, dict), "Slice 9 score normalization is required")
    _require(
        normalization.get("missing_value_is_not_zero") is True,
        "Slice 9 missing values must not be imputed as zero",
    )
    return payload


def _decode_json(value: Any, *, label: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    _require(isinstance(value, str), f"{label} must be JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is invalid JSON: {error}") from error


def load_slice9_catalog_candidates(
    *,
    catalog_path: str | Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    input_config = config["input"]
    dataset_id = str(input_config["dataset_id"])
    dataset_version = int(input_config["dataset_version"])
    expected_tree_sha256 = str(input_config["dataset_tree_sha256"])
    split = str(input_config["split"])
    target_speaker_id = str(input_config["target_speaker_id"])
    catalog = Catalog(Path(catalog_path).resolve())
    with catalog.read_only_session() as connection:
        version = connection.execute(
            """
            SELECT artifact_set_sha256, item_count
            FROM dataset_version
            WHERE dataset_id = ? AND dataset_version = ?
            """,
            (dataset_id, dataset_version),
        ).fetchone()
        _require(version is not None, f"Frozen dataset is not registered: {dataset_id}@{dataset_version}")
        _require(
            str(version["artifact_set_sha256"]) == expected_tree_sha256,
            "Frozen dataset tree SHA-256 differs from Slice 9 config",
        )
        rows = connection.execute(
            """
            SELECT di.ordinal,
                   di.fid,
                   di.asset_sha256,
                   di.split,
                   di.source_root_key,
                   di.source_path_key,
                   di.derived_id,
                   di.audio_relative_path,
                   di.audio_sha256,
                   di.quality_run_id,
                   di.quality_decision,
                   di.quality_reasons_json,
                   di.text_exact,
                   di.text_sha256,
                   di.text_source,
                   di.emotion_weak_label,
                   di.emotion_primary,
                   di.emotion_secondary,
                   di.intensity,
                   di.label_source,
                   di.review_decision_id,
                   di.review_round,
                   di.review_batch_id,
                   di.lineage_json,
                   raw.relative_path AS raw_relative_path,
                   source.relative_path AS source_relative_path,
                   source.speaker_id AS speaker_id,
                   derived.output_sha256 AS derived_output_sha256,
                   derived.duration_seconds AS derived_duration_seconds,
                   derived.output_sample_rate,
                   derived.output_channels,
                   derived.output_subtype
            FROM dataset_item AS di
            JOIN raw_object AS raw
              ON raw.asset_sha256 = di.asset_sha256
            JOIN derived_audio AS derived
              ON derived.derived_id = di.derived_id
             AND derived.asset_sha256 = di.asset_sha256
            JOIN source_location AS source
              ON source.root_key = di.source_root_key
             AND source.path_key = di.source_path_key
             AND source.availability_status = 'available'
            WHERE di.dataset_id = ?
              AND di.dataset_version = ?
              AND di.split = ?
            ORDER BY di.ordinal
            """,
            (dataset_id, dataset_version, split),
        ).fetchall()

    _require(rows, "Slice 9 catalog query returned no train candidates")
    candidates: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for row in rows:
        asset_sha256 = _require_sha256(row["asset_sha256"], "dataset_item.asset_sha256")
        _require(asset_sha256 not in seen_assets, f"Duplicate Slice 9 asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        _require(row["split"] == split, f"Unexpected split for {asset_sha256}")
        _require(row["speaker_id"] == target_speaker_id, f"Unexpected speaker for {asset_sha256}")
        _require(row["derived_output_sha256"] == row["audio_sha256"], f"Derived audio SHA mismatch for {asset_sha256}")
        _require(row["output_sample_rate"] == 48000, f"Unexpected derived sample rate for {asset_sha256}")
        _require(row["output_channels"] == 1, f"Unexpected derived channels for {asset_sha256}")
        _require(row["output_subtype"] == "PCM_24", f"Unexpected derived subtype for {asset_sha256}")
        text_exact = str(row["text_exact"])
        _require(text_exact.strip(), f"Empty frozen text for {asset_sha256}")
        computed_text_sha256 = hashlib.sha256(text_exact.encode("utf-8")).hexdigest()
        _require(computed_text_sha256 == row["text_sha256"], f"Text SHA mismatch for {asset_sha256}")
        quality_reasons = _decode_json(row["quality_reasons_json"], label=f"quality reasons {asset_sha256}")
        lineage = _decode_json(row["lineage_json"], label=f"lineage {asset_sha256}")
        _require(isinstance(quality_reasons, list), f"Quality reasons must be a list for {asset_sha256}")
        _require(isinstance(lineage, dict), f"Lineage must be an object for {asset_sha256}")
        candidates.append(
            {
                "fid": str(row["fid"]),
                "asset_sha256": asset_sha256,
                "raw_relative_path": str(row["raw_relative_path"]),
                "source_root_key": str(row["source_root_key"]),
                "source_path_key": str(row["source_path_key"]),
                "source_relative_path": str(row["source_relative_path"]),
                "speaker_id": str(row["speaker_id"]),
                "derived_id": str(row["derived_id"]),
                "audio_relative_path": str(row["audio_relative_path"]),
                "audio_sha256": str(row["audio_sha256"]),
                "duration_seconds": float(row["derived_duration_seconds"]),
                "quality_run_id": str(row["quality_run_id"]),
                "quality_decision": str(row["quality_decision"]),
                "quality_reasons": quality_reasons,
                "text_exact": text_exact,
                "text_sha256": str(row["text_sha256"]),
                "text_source": str(row["text_source"]),
                "emotion_weak_label": str(row["emotion_weak_label"]),
                "emotion_primary": str(row["emotion_primary"]),
                "emotion_secondary": row["emotion_secondary"],
                "intensity": row["intensity"],
                "label_source": str(row["label_source"]),
                "review_decision_id": row["review_decision_id"],
                "review_round": row["review_round"],
                "review_batch_id": row["review_batch_id"],
                "lineage": lineage,
            }
        )
    expected_item_count = int(version["item_count"])
    _require(len(candidates) <= expected_item_count, "Slice 9 candidate count exceeds frozen dataset count")
    return candidates


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def build_slice9_candidate_snapshot(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    snapshot_path: str | Path | None = None,
    report_path: str | Path | None = DEFAULT_SLICE9_AUDIT_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    candidates = load_slice9_catalog_candidates(catalog_path=catalog_path, config=config)
    snapshot = build_candidate_snapshot(candidates)
    configured_snapshot_path = Path(config["input"]["candidate_snapshot_path"])
    resolved_snapshot_path = Path(snapshot_path or configured_snapshot_path).resolve()
    snapshot_result = write_candidate_snapshot(snapshot, resolved_snapshot_path)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "source": "catalog.dataset_item",
        "split": config["input"]["split"],
        "target_speaker_id": config["input"]["target_speaker_id"],
        "candidate_snapshot_path": str(resolved_snapshot_path),
        "candidate_snapshot_sha256": snapshot["sha256"],
        "candidate_snapshot_file_sha256": _sha256_file(resolved_snapshot_path),
        "candidate_count": len(candidates),
        "total_duration_seconds": snapshot["payload"]["total_duration_seconds"],
        "counts": {
            "emotion_primary": dict(sorted(Counter(item["emotion_primary"] for item in candidates).items())),
            "text_source": dict(sorted(Counter(item["text_source"] for item in candidates).items())),
            "quality_decision": dict(sorted(Counter(item["quality_decision"] for item in candidates).items())),
        },
        "legacy_snapshot_rejected": "data/reports/datasets/fuxuan_v1/audit/candidate_snapshot.json",
        "snapshot_write": snapshot_result,
    }
    resolved_report_path = Path(report_path).resolve() if report_path is not None else None
    if resolved_report_path is not None:
        _atomic_write_json(resolved_report_path, report)
    return {**report, "report_path": str(resolved_report_path) if resolved_report_path else None}

