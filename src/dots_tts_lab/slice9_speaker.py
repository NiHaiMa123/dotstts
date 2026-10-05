from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CONFIG_PATH,
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    load_slice9_selection_config,
)
from dots_tts_lab.speaker_embedding import load_speaker_embedding_vectors


DEFAULT_SLICE9_SPEAKER_FEATURE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_speaker_features_v1.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


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


def _latest_speaker_reviews(catalog_path: str | Path) -> dict[str, dict[str, Any]]:
    catalog = Catalog(Path(catalog_path).resolve())
    with catalog.read_only_session() as connection:
        version = connection.execute(
            """
            SELECT analysis_run_id
            FROM dataset_version
            WHERE dataset_id = 'fuxuan' AND dataset_version = 1
            """
        ).fetchone()
        if version is None:
            return {}
        rows = connection.execute(
            """
            SELECT review_id, asset_sha256, review_round, review_status,
                   review_note, created_at, review_batch_id, source_row_index
            FROM dataset_speaker_review AS current
            WHERE current.run_id = ?
              AND current.review_round = (
                  SELECT MAX(candidate.review_round)
                  FROM dataset_speaker_review AS candidate
                  WHERE candidate.run_id = current.run_id
                    AND candidate.asset_sha256 = current.asset_sha256
              )
            ORDER BY current.asset_sha256
            """,
            (str(version["analysis_run_id"]),),
        ).fetchall()
    return {str(row["asset_sha256"]): dict(row) for row in rows}


def build_slice9_speaker_features(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    snapshot_path: str | Path = DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    speaker_report_path: str | Path = "data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json",
    threshold_report_path: str | Path = "data/reports/datasets/fuxuan_v1/analysis/threshold_calibration_v1.json",
    output_path: str | Path = DEFAULT_SLICE9_SPEAKER_FEATURE_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    snapshot = _load_json(Path(snapshot_path).resolve(), label="Slice 9 candidate snapshot")
    candidates = snapshot.get("items")
    _require(isinstance(candidates, list) and snapshot.get("item_count") == len(candidates), "Invalid Slice 9 candidate snapshot")
    provenance = config["provenance"]
    resolved_speaker = Path(speaker_report_path).resolve()
    _require(_sha256_file(resolved_speaker) == provenance["speaker_report_sha256"], "Speaker report SHA drift")
    resolved_threshold = Path(threshold_report_path).resolve()
    _require(_sha256_file(resolved_threshold) == provenance["threshold_calibration_report_sha256"], "Threshold calibration report SHA drift")
    threshold_report = _load_json(resolved_threshold, label="threshold calibration report")
    _require(threshold_report.get("status") == "succeeded", "Threshold calibration did not succeed")
    _require(threshold_report.get("inputs", {}).get("speaker_report_sha256") == provenance["speaker_report_sha256"], "Threshold report speaker input drift")
    speaker_thresholds = config["gates"]["speaker_review"]
    _require(threshold_report.get("thresholds", {}).get("speaker_center_cosine") == speaker_thresholds["center_cosine_min"], "Speaker center threshold drift")
    _require(threshold_report.get("thresholds", {}).get("speaker_knn_cosine") == speaker_thresholds["knn_cosine_min"], "Speaker kNN threshold drift")
    report = _load_json(resolved_speaker, label="speaker embedding report")
    _require(report.get("status") == "succeeded", "Speaker embedding report did not succeed")
    _require(report.get("speaker_embedding_config_sha256") == provenance["speaker_identity_sha256"], "Speaker embedding identity drift")
    features = report.get("features")
    assessments = report.get("assessments")
    _require(isinstance(features, list) and isinstance(assessments, list), "Speaker report features/assessments are required")
    feature_by_asset = {str(item["asset_sha256"]): item for item in features}
    assessment_by_asset = {str(item["asset_sha256"]): item for item in assessments}
    vectors = load_speaker_embedding_vectors(report)
    _require(set(feature_by_asset) == set(vectors), "Speaker cache and report asset sets differ")
    reviews = _latest_speaker_reviews(catalog_path)
    output_items: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    review_status_counts: Counter[str] = Counter()
    for candidate in candidates:
        asset_sha256 = str(candidate["asset_sha256"])
        _require(str(candidate["speaker_id"]) == str(config["input"]["target_speaker_id"]), f"Non-target speaker in candidate snapshot: {asset_sha256}")
        feature = feature_by_asset.get(asset_sha256)
        assessment = assessment_by_asset.get(asset_sha256)
        _require(feature is not None and assessment is not None, f"Speaker feature/assessment missing: {asset_sha256}")
        _require(feature.get("audio_sha256") == candidate["audio_sha256"], f"Speaker audio SHA drift: {asset_sha256}")
        manual = reviews.get(asset_sha256)
        manual_status = str(manual["review_status"]) if manual else None
        if manual_status:
            review_status_counts[manual_status] += 1
        reasons: list[dict[str, Any]] = []
        if manual_status in {"exclude_wrong_speaker", "exclude_uncertain"}:
            decision = "reject"
            reasons.append({"code": "manual_speaker_exclusion", "severity": "reject", "review_status": manual_status, "review_id": manual["review_id"], "note": manual.get("review_note")})
        else:
            center = float(assessment["center_cosine"])
            knn = float(assessment["knn_cosine"])
            if center < float(speaker_thresholds["center_cosine_min"]):
                reasons.append({"code": "speaker_center_below_threshold", "severity": "review", "actual": center, "threshold": float(speaker_thresholds["center_cosine_min"])})
            if knn < float(speaker_thresholds["knn_cosine_min"]):
                reasons.append({"code": "speaker_knn_below_threshold", "severity": "review", "actual": knn, "threshold": float(speaker_thresholds["knn_cosine_min"])})
            if reasons and manual_status != "confirmed_same_speaker":
                decision = "review"
            else:
                decision = "pass"
                if reasons:
                    reasons.append({"code": "manual_same_speaker_overrides_outlier_review", "severity": "info", "review_id": manual["review_id"]})
        decision_counts[decision] += 1
        for reason in reasons:
            reason_counts[str(reason["code"])] += 1
        output_items.append(
            {
                "asset_sha256": asset_sha256,
                "fid": candidate["fid"],
                "audio_sha256": candidate["audio_sha256"],
                "speaker_id": candidate["speaker_id"],
                "center_cosine": float(assessment["center_cosine"]),
                "knn_cosine": float(assessment["knn_cosine"]),
                "outlier_score": float(assessment["outlier_score"]),
                "outlier_rank": int(assessment["outlier_rank"]),
                "neighbor_asset_sha256s": list(assessment["neighbor_asset_sha256s"]),
                "neighbor_cosines": [float(value) for value in assessment["neighbor_cosines"]],
                "manual_review_status": manual_status,
                "manual_review_id": manual.get("review_id") if manual else None,
                "decision": decision,
                "reasons": reasons,
            }
        )
    output_items.sort(key=lambda item: item["asset_sha256"])
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "candidate_snapshot_sha256": hashlib.sha256(Path(snapshot_path).resolve().read_bytes()).hexdigest(),
        "speaker_report_sha256": _sha256_file(resolved_speaker),
        "speaker_identity_sha256": provenance["speaker_identity_sha256"],
        "threshold_calibration_report_sha256": _sha256_file(resolved_threshold),
        "thresholds": {
            "center_cosine_min": speaker_thresholds["center_cosine_min"],
            "knn_cosine_min": speaker_thresholds["knn_cosine_min"],
            "action": speaker_thresholds["action"],
        },
        "candidate_count": len(output_items),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "manual_review_status_counts": dict(sorted(review_status_counts.items())),
        "items": output_items,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

