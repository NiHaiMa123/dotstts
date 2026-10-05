from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.dataset_freeze import build_candidate_snapshot
from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CONFIG_PATH,
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    load_slice9_catalog_candidates,
    load_slice9_selection_config,
)
from dots_tts_lab.speaker_embedding import load_speaker_embedding_vectors


DEFAULT_SLICE9_PROVENANCE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/slice9_provenance_audit.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    _require(actual == expected, f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _verify_standardization(
    *,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    report_path: Path,
    integrity_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path, label="standardization report")
    summary = report.get("summary")
    assets = report.get("assets")
    _require(isinstance(summary, dict), "Standardization report summary is required")
    _require(isinstance(assets, list), "Standardization report assets are required")
    _require_equal("standardization status", summary.get("status"), "succeeded")
    standardized = config["standardized_audio"]
    _require_equal("standardization config id", summary.get("config_id"), standardized["config_id"])
    _require_equal("standardization config version", summary.get("config_version"), standardized["config_version"])
    _require_equal("standardization config SHA", summary.get("config_sha256"), standardized["config_sha256"])
    _require_equal("standardization implementation", summary.get("implementation_version"), standardized["implementation_version"])
    integrity = _load_json(integrity_path, label="standardization integrity audit")
    _require_equal("standardization integrity status", integrity.get("status"), "passed")
    _require(not integrity.get("errors"), "Standardization integrity audit contains errors")
    by_asset: dict[str, dict[str, Any]] = {}
    for item in assets:
        _require(isinstance(item, dict), "Standardization asset is not an object")
        asset_sha256 = item.get("asset_sha256")
        _require(isinstance(asset_sha256, str) and len(asset_sha256) == 64, "Invalid standardization asset SHA")
        _require(asset_sha256 not in by_asset, f"Duplicate standardization asset: {asset_sha256}")
        by_asset[asset_sha256] = item

    output_root = Path(str(summary["output_root_path"])).resolve()
    _require(output_root.is_dir(), f"Standardization output root is missing: {output_root}")
    checked_files = 0
    for candidate in candidates:
        asset_sha256 = candidate["asset_sha256"]
        item = by_asset.get(asset_sha256)
        _require(item is not None, f"Standardization report missing asset: {asset_sha256}")
        _require_equal(f"derived id for {asset_sha256}", item.get("derived_id"), candidate["derived_id"])
        _require_equal(f"derived SHA for {asset_sha256}", item.get("output_sha256"), candidate["audio_sha256"])
        _require_equal(f"derived path for {asset_sha256}", item.get("relative_path"), candidate["audio_relative_path"])
        _require_equal(f"derived sample rate for {asset_sha256}", item.get("output_sample_rate"), standardized["sample_rate_hz"])
        _require_equal(f"derived channels for {asset_sha256}", item.get("output_channels"), standardized["channels"])
        _require_equal(f"derived subtype for {asset_sha256}", item.get("output_subtype"), standardized["subtype"])
        artifact = (output_root / str(item["relative_path"])).resolve()
        _require(os.path.commonpath((str(output_root), str(artifact))) == str(output_root), f"Derived path escapes output root: {artifact}")
        _require(artifact.is_file(), f"Standardized artifact is missing: {artifact}")
        _require_equal(f"standardized file SHA for {asset_sha256}", _sha256_file(artifact), candidate["audio_sha256"])
        checked_files += 1
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256_file(report_path),
        "integrity_path": str(integrity_path.resolve()),
        "integrity_sha256": _sha256_file(integrity_path),
        "run_id": summary.get("run_id"),
        "asset_count": len(by_asset),
        "candidate_files_sha256_verified": checked_files,
    }


def _verify_quality(
    *,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path, label="quality report")
    summary = report.get("summary")
    assets = report.get("assets")
    _require(isinstance(summary, dict), "Quality report summary is required")
    _require(isinstance(assets, list), "Quality report assets are required")
    _require_equal("quality status", summary.get("status"), "succeeded")
    provenance = config["provenance"]
    for key in (
        "quality_analysis_id",
        "quality_analysis_version",
        "quality_analysis_config_sha256",
        "quality_policy_id",
        "quality_policy_version",
        "quality_policy_config_sha256",
    ):
        report_key = key.replace("quality_", "", 1)
        _require_equal(f"quality {report_key}", summary.get(report_key), provenance[key])
    by_asset: dict[str, dict[str, Any]] = {}
    for item in assets:
        _require(isinstance(item, dict), "Quality asset is not an object")
        asset_sha256 = item.get("asset_sha256")
        _require(isinstance(asset_sha256, str) and len(asset_sha256) == 64, "Invalid quality asset SHA")
        _require(asset_sha256 not in by_asset, f"Duplicate quality asset: {asset_sha256}")
        by_asset[asset_sha256] = item
    metric_names = (
        "duration_seconds",
        "sample_peak_dbfs",
        "true_peak_estimate_dbtp",
        "rms_dbfs",
        "integrated_loudness_lufs",
        "crest_factor_db",
        "abs_dc_offset",
        "leading_silence_seconds",
        "trailing_silence_seconds",
        "silence_ratio",
        "digital_silence_frame_ratio",
        "snr_proxy_db",
        "near_peak_sample_ratio",
        "flat_top_run_count",
    )
    checked = 0
    for candidate in candidates:
        asset_sha256 = candidate["asset_sha256"]
        item = by_asset.get(asset_sha256)
        _require(item is not None, f"Quality report missing asset: {asset_sha256}")
        _require(item.get("metric_action") != "error", f"Quality metric error for {asset_sha256}")
        _require_equal(f"quality decision for {asset_sha256}", item.get("decision"), candidate["quality_decision"])
        _require_equal(f"quality reasons for {asset_sha256}", item.get("reasons"), candidate["quality_reasons"])
        for metric in metric_names:
            _require(metric in item, f"Quality metric missing {metric} for {asset_sha256}")
        checked += 1
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256_file(report_path),
        "run_id": summary.get("run_id"),
        "asset_count": len(by_asset),
        "candidate_metrics_verified": checked,
    }


def _verify_speaker(
    *,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path, label="speaker embedding report")
    _require_equal("speaker report status", report.get("status"), "succeeded")
    _require_equal("speaker report dimension", report.get("dimension"), 512)
    _require_equal("speaker report dtype", report.get("dtype"), "float32_le")
    provenance = config["provenance"]
    _require_equal("speaker config identity", report.get("speaker_embedding_config_sha256"), provenance["speaker_identity_sha256"])
    features = report.get("features")
    assessments = report.get("assessments")
    _require(isinstance(features, list), "Speaker report features are required")
    _require(isinstance(assessments, list), "Speaker report assessments are required")
    feature_by_asset = {str(item.get("asset_sha256")): item for item in features if isinstance(item, dict)}
    assessment_by_asset = {str(item.get("asset_sha256")): item for item in assessments if isinstance(item, dict)}
    _require(len(feature_by_asset) == len(features), "Speaker report has duplicate/invalid feature assets")
    _require(len(assessment_by_asset) == len(assessments), "Speaker report has duplicate/invalid assessment assets")
    vectors = load_speaker_embedding_vectors(report)
    _require(set(vectors) == set(feature_by_asset), "Speaker cache set differs from speaker report")
    checked = 0
    for candidate in candidates:
        asset_sha256 = candidate["asset_sha256"]
        feature = feature_by_asset.get(asset_sha256)
        assessment = assessment_by_asset.get(asset_sha256)
        _require(feature is not None, f"Speaker feature missing asset: {asset_sha256}")
        _require(assessment is not None, f"Speaker assessment missing asset: {asset_sha256}")
        _require_equal(f"speaker audio SHA for {asset_sha256}", feature.get("audio_sha256"), candidate["audio_sha256"])
        _require_equal(f"speaker feature config for {asset_sha256}", feature.get("speaker_embedding_config_sha256"), provenance["speaker_identity_sha256"])
        _require_equal(f"speaker feature dimension for {asset_sha256}", feature.get("dimension"), 512)
        _require_equal(f"speaker feature dtype for {asset_sha256}", feature.get("dtype"), "float32_le")
        for field in ("center_cosine", "knn_cosine", "outlier_score", "outlier_rank", "neighbor_asset_sha256s", "neighbor_cosines"):
            _require(field in assessment, f"Speaker assessment missing {field} for {asset_sha256}")
        checked += 1
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256_file(report_path),
        "config_identity_sha256": report.get("speaker_embedding_config_sha256"),
        "asset_count": len(feature_by_asset),
        "candidate_vectors_verified": checked,
    }


def _verify_text_provenance(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        asset_sha256 = candidate["asset_sha256"]
        text = str(candidate["text_exact"])
        _require(text.strip(), f"Empty text for {asset_sha256}")
        _require(
            hashlib.sha256(text.encode("utf-8")).hexdigest() == candidate["text_sha256"],
            f"Text SHA drift for {asset_sha256}",
        )
        source = candidate["text_source"]
        if source == "human_review":
            _require(candidate["review_decision_id"], f"Human text missing decision id for {asset_sha256}")
            _require(isinstance(candidate["review_round"], int) and candidate["review_round"] >= 1, f"Human text missing review round for {asset_sha256}")
            _require(candidate["review_batch_id"], f"Human text missing review batch for {asset_sha256}")
            _require(candidate["label_source"] in {"weak_label_confirmed", "human_corrected"}, f"Invalid human label source for {asset_sha256}")
        elif source == "filename_candidate_unreviewed":
            _require(candidate["review_decision_id"] is None, f"Unreviewed text has decision id for {asset_sha256}")
            _require(candidate["review_round"] is None, f"Unreviewed text has review round for {asset_sha256}")
            _require(candidate["review_batch_id"] is None, f"Unreviewed text has review batch for {asset_sha256}")
            _require(candidate["label_source"] == "weak_label_unreviewed", f"Invalid unreviewed label source for {asset_sha256}")
        else:
            raise RuntimeError(f"Unknown text source for {asset_sha256}: {source}")
        counts[source] += 1
    return dict(sorted(counts.items()))


def verify_slice9_provenance(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    snapshot_path: str | Path = DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    standardization_report_path: str | Path = "data/reports/standardization/standardization.json",
    standardization_integrity_path: str | Path = "data/reports/standardization/integrity_audit.json",
    quality_report_path: str | Path = "data/reports/quality/quality.json",
    speaker_report_path: str | Path = "data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json",
    report_path: str | Path = DEFAULT_SLICE9_PROVENANCE_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    candidates = load_slice9_catalog_candidates(catalog_path=catalog_path, config=config)
    resolved_snapshot = Path(snapshot_path).resolve()
    snapshot_bytes = resolved_snapshot.read_bytes()
    snapshot = _load_json(resolved_snapshot, label="Slice 9 candidate snapshot")
    expected_snapshot = build_candidate_snapshot(candidates)
    _require_equal("candidate snapshot bytes", snapshot_bytes, expected_snapshot["canonical_json"].encode("utf-8"))
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    _require_equal("candidate snapshot SHA", snapshot_sha256, expected_snapshot["sha256"])
    _require_equal("candidate snapshot item count", snapshot.get("item_count"), len(candidates))

    standardization = _verify_standardization(
        candidates=candidates,
        config=config,
        report_path=Path(standardization_report_path).resolve(),
        integrity_path=Path(standardization_integrity_path).resolve(),
    )
    quality = _verify_quality(
        candidates=candidates,
        config=config,
        report_path=Path(quality_report_path).resolve(),
    )
    speaker = _verify_speaker(
        candidates=candidates,
        config=config,
        report_path=Path(speaker_report_path).resolve(),
    )
    text_counts = _verify_text_provenance(candidates)
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
        "candidate_count": len(candidates),
        "candidate_snapshot_path": str(resolved_snapshot),
        "candidate_snapshot_sha256": snapshot_sha256,
        "counts": {"text_source": text_counts},
        "standardization": standardization,
        "quality": quality,
        "speaker": speaker,
        "checks": {
            "legacy_277_snapshot_used": False,
            "catalog_snapshot_replayed": True,
            "standardized_audio_hashes_recomputed": True,
            "raw_quality_provenance_verified": True,
            "speaker_cache_hashes_verified": True,
            "text_provenance_verified": True,
        },
    }
    resolved_report = Path(report_path).resolve()
    _write_json(resolved_report, report)
    return {**report, "report_path": str(resolved_report)}
