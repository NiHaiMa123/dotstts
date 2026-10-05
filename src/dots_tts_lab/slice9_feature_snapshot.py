"""Freeze and verify the Slice 9 candidate feature snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    DEFAULT_SLICE9_CONFIG_PATH,
    load_slice9_selection_config,
)


DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json"
)
DEFAULT_REPORTS = {
    "quality": Path("data/reports/datasets/fuxuan_v1/analysis/slice9_quality_features_v1.json"),
    "speaker": Path("data/reports/datasets/fuxuan_v1/analysis/slice9_speaker_features_v1.json"),
    "text": Path("data/reports/datasets/fuxuan_v1/analysis/slice9_text_features_v1.json"),
    "coverage": Path("data/reports/datasets/fuxuan_v1/analysis/slice9_coverage_features_v1.json"),
}


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


def _report_index(report: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    _require(report.get("status") == "succeeded", f"{label} did not succeed")
    items = report.get("items")
    _require(isinstance(items, list), f"{label} items are missing")
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        _require(isinstance(item, dict) and isinstance(item.get("asset_sha256"), str), f"Invalid {label} item")
        asset = item["asset_sha256"]
        _require(asset not in indexed, f"Duplicate {label} asset: {asset}")
        indexed[asset] = item
    return indexed


def build_slice9_feature_snapshot(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    candidate_snapshot_path: str | Path = DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    report_paths: dict[str, str | Path] | None = None,
    output_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
) -> dict[str, Any]:
    config_resolved = Path(config_path).resolve()
    candidate_resolved = Path(candidate_snapshot_path).resolve()
    config = load_slice9_selection_config(config_resolved)
    candidate_snapshot = _load_json(candidate_resolved, label="Slice 9 candidate snapshot")
    candidates = candidate_snapshot.get("items")
    _require(isinstance(candidates, list) and candidate_snapshot.get("item_count") == len(candidates), "Invalid candidate snapshot")
    candidate_sha = _sha256(candidate_resolved)
    reports = {name: Path(path).resolve() for name, path in (report_paths or DEFAULT_REPORTS).items()}
    _require(set(reports) == set(DEFAULT_REPORTS), "Feature reports must be quality/speaker/text/coverage")
    report_payloads = {name: _load_json(path, label=f"{name} feature report") for name, path in reports.items()}
    report_indexes = {name: _report_index(payload, label=f"{name} feature report") for name, payload in report_payloads.items()}
    expected_ids = {str(item["asset_sha256"]) for item in candidates}
    _require(len(expected_ids) == len(candidates), "Candidate snapshot contains duplicate assets")
    for name, index in report_indexes.items():
        _require(set(index) == expected_ids, f"{name} feature report asset set drift")
        _require(report_payloads[name].get("candidate_snapshot_sha256") == candidate_sha, f"{name} report is for a different candidate snapshot")
        _require(report_payloads[name].get("dataset_tree_sha256") == config["input"]["dataset_tree_sha256"], f"{name} dataset tree drift")
    input_hashes = {
        "config_sha256": _sha256(config_resolved),
        "candidate_snapshot_sha256": candidate_sha,
        "feature_reports": {name: _sha256(path) for name, path in sorted(reports.items())},
    }
    freeze_id = hashlib.sha256(json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]
    output_items: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: str(item["asset_sha256"])):
        asset = str(candidate["asset_sha256"])
        quality = report_indexes["quality"][asset]
        speaker = report_indexes["speaker"][asset]
        text = report_indexes["text"][asset]
        coverage = report_indexes["coverage"][asset]
        _require(quality.get("audio_sha256") == candidate.get("audio_sha256"), f"Quality audio SHA drift for {asset}")
        _require(speaker.get("audio_sha256") == candidate.get("audio_sha256"), f"Speaker audio SHA drift for {asset}")
        _require(text.get("text_sha256") == candidate.get("text_sha256"), f"Text SHA drift for {asset}")
        _require(coverage.get("text_sha256") == candidate.get("text_sha256"), f"Coverage text SHA drift for {asset}")
        output_items.append(
            {
                "asset_sha256": asset,
                "fid": candidate["fid"],
                "audio_relative_path": candidate["audio_relative_path"],
                "audio_sha256": candidate["audio_sha256"],
                "speaker_id": candidate["speaker_id"],
                "emotion_primary": candidate["emotion_primary"],
                "emotion_weak_label": candidate["emotion_weak_label"],
                "text_exact": candidate["text_exact"],
                "text_sha256": candidate["text_sha256"],
                "text_source": candidate["text_source"],
                "quality": quality,
                "speaker": speaker,
                "text": text,
                "coverage": coverage,
            }
        )
    result = {
        "schema_version": 1,
        "feature_snapshot_schema_version": "slice9_candidate_features@1",
        "status": "succeeded",
        "freeze_id": freeze_id,
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "implementation_version": config["implementation_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": input_hashes,
        "candidate_count": len(output_items),
        "items": output_items,
    }
    resolved_output = Path(output_path).resolve()
    if resolved_output.exists():
        existing = _load_json(resolved_output, label="existing Slice 9 feature snapshot")
        _require(existing.get("freeze_id") == freeze_id and existing.get("input_hashes") == input_hashes, "Feature snapshot input drift; refusing to overwrite frozen snapshot")
        existing_bytes = resolved_output.read_bytes()
        new_bytes = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _require(existing_bytes == new_bytes, "Frozen feature snapshot content drift; refusing to overwrite")
        return {**existing, "report_path": str(resolved_output), "reused": True}
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output), "reused": False}

