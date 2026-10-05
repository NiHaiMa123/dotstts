from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import yaml

from dots_tts_lab.acoustic_fingerprint import load_acoustic_fingerprint_vectors
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.duplicate_graph import canonical_json, load_verified_json
from dots_tts_lab.reports import _atomic_write_text
from dots_tts_lab.speaker_embedding import load_speaker_embedding_vectors
from dots_tts_lab.threshold_calibration import (
    DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    load_threshold_calibration_config,
    load_threshold_calibration_overlay,
)


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Apply versioned duplicate and speaker review decisions."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--fingerprint-report", type=Path, required=True)
    parser.add_argument("--speaker-report", type=Path, required=True)
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review_payload = yaml.safe_load(
        args.review_config.resolve().read_text(encoding="utf-8")
    )
    if not isinstance(review_payload, dict):
        raise RuntimeError("Manual review config must be a mapping")
    overlay = load_threshold_calibration_overlay(args.calibration_config)
    calibration_config = load_threshold_calibration_config(args.calibration_config)
    calibration_report = load_verified_json(
        overlay.calibration_report_path,
        overlay.calibration_report_sha256,
    )
    snapshot = load_verified_json(
        args.candidate_snapshot,
        overlay.candidate_snapshot_sha256,
    )
    fingerprint_report = load_verified_json(
        args.fingerprint_report,
        calibration_config.acoustic.report_sha256,
    )
    speaker_report = load_verified_json(
        args.speaker_report,
        calibration_config.speaker.report_sha256,
    )
    _validate_review_identity(review_payload, overlay)
    candidates = {item["asset_sha256"]: item for item in snapshot["items"]}
    fingerprint_vectors = load_acoustic_fingerprint_vectors(fingerprint_report)
    speaker_vectors = load_speaker_embedding_vectors(speaker_report)
    if set(candidates) != set(fingerprint_vectors) or set(candidates) != set(
        speaker_vectors
    ):
        raise RuntimeError("Feature reports do not cover the frozen candidates")
    fingerprint_metadata = {
        item["asset_sha256"]: item for item in fingerprint_report["features"]
    }
    speaker_metadata = {
        item["asset_sha256"]: item for item in speaker_report["features"]
    }
    features = []
    for asset_sha256 in sorted(candidates):
        candidate = candidates[asset_sha256]
        fingerprint = fingerprint_metadata[asset_sha256]
        speaker = speaker_metadata[asset_sha256]
        if (
            fingerprint["audio_sha256"] != candidate["audio_sha256"]
            or speaker["audio_sha256"] != candidate["audio_sha256"]
        ):
            raise RuntimeError(f"Feature audio identity drift: {asset_sha256}")
        fingerprint_blob = fingerprint_vectors[asset_sha256].astype(
            "<f4", copy=False
        ).tobytes(order="C")
        speaker_blob = speaker_vectors[asset_sha256].astype(
            "<f4", copy=False
        ).tobytes(order="C")
        features.append(
            {
                "asset_sha256": asset_sha256,
                "derived_id": candidate["derived_id"],
                "standardized_sha256": candidate["audio_sha256"],
                "duration_seconds": float(candidate["duration_seconds"]),
                "fingerprint_blob": fingerprint_blob,
                "fingerprint_dimension": 4096,
                "fingerprint_sha256": hashlib.sha256(fingerprint_blob).hexdigest(),
                "speaker_embedding_blob": speaker_blob,
                "speaker_embedding_dimension": 512,
                "speaker_embedding_sha256": hashlib.sha256(speaker_blob).hexdigest(),
            }
        )

    center_calibration = calibration_report["speaker"]["center_cosine"]
    knn_calibration = calibration_report["speaker"]["knn_cosine"]
    assessments = []
    for assessment in speaker_report["assessments"]:
        center = float(assessment["center_cosine"])
        knn = float(assessment["knn_cosine"])
        candidate_outlier = int(
            center < overlay.thresholds.speaker_center_cosine
            or knn < overlay.thresholds.speaker_knn_cosine
        )
        assessments.append(
            {
                "asset_sha256": assessment["asset_sha256"],
                "center_cosine": center,
                "knn_cosine": knn,
                "knn_k": int(speaker_report["effective_knn_k"]),
                "center_robust_z": (
                    center - float(center_calibration["median"])
                )
                / float(center_calibration["robust_sigma"]),
                "knn_robust_z": (knn - float(knn_calibration["median"]))
                / float(knn_calibration["robust_sigma"]),
                "candidate_outlier": candidate_outlier,
                "evidence_json": canonical_json(
                    {
                        "assessment": assessment,
                        "calibration_report_sha256": overlay.calibration_report_sha256,
                        "center_threshold": overlay.thresholds.speaker_center_cosine,
                        "knn_threshold": overlay.thresholds.speaker_knn_cosine,
                    }
                ),
            }
        )

    catalog = Catalog(args.catalog)
    feature_result = catalog.store_dataset_features_and_speaker_assessments(
        run_id=review_payload["analysis_run_id"],
        features=features,
        assessments=assessments,
    )
    with catalog.read_only_session() as connection:
        edge_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT left_asset_sha256, right_asset_sha256, evidence_type,
                       score, threshold, analysis_status, evidence_json
                FROM dataset_similarity_edge
                WHERE run_id = ?
                ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                """,
                (review_payload["analysis_run_id"],),
            )
        ]
    edge_by_key = {
        (
            edge["left_asset_sha256"],
            edge["right_asset_sha256"],
            edge["evidence_type"],
        ): edge
        for edge in edge_rows
    }
    edge_decisions = review_payload.get("edge_decisions")
    if not isinstance(edge_decisions, list) or len(edge_decisions) != len(edge_rows):
        raise RuntimeError("Manual review must decide every duplicate edge")
    edge_review_rows = []
    for index, decision in enumerate(edge_decisions):
        key = (
            decision["left_asset_sha256"],
            decision["right_asset_sha256"],
            decision["evidence_type"],
        )
        if key not in edge_by_key:
            raise RuntimeError(f"Manual review references an unknown edge: {key}")
        _validate_duplicate_resolution(decision)
        edge_review_rows.append(
            {
                "review_id": _review_uuid(review_payload, "edge", index),
                "run_id": review_payload["analysis_run_id"],
                "left_asset_sha256": key[0],
                "right_asset_sha256": key[1],
                "evidence_type": key[2],
                "review_status": decision["review_status"],
                "review_note": canonical_json(
                    {
                        "exclude_asset_sha256s": decision[
                            "exclude_asset_sha256s"
                        ],
                        "note": decision["review_note"],
                        "representative_asset_sha256": decision[
                            "representative_asset_sha256"
                        ],
                    }
                ),
                "created_at": review_payload["reviewed_at"],
                "review_batch_id": review_payload["review_batch_id"] + ":edge",
                "source_row_index": index,
            }
        )
    edge_inserted, edge_ignored = catalog.record_dataset_similarity_edge_reviews(
        edge_review_rows
    )

    calibrated_speaker_candidates = {
        item["asset_sha256"]: int(item["outlier_rank"])
        for item in calibration_report["speaker"]["candidates"]
    }
    speaker_decisions = review_payload.get("speaker_decisions")
    if not isinstance(speaker_decisions, list) or {
        item["asset_sha256"] for item in speaker_decisions
    } != set(calibrated_speaker_candidates):
        raise RuntimeError("Manual review must decide every speaker outlier")
    speaker_review_rows = []
    for index, decision in enumerate(speaker_decisions):
        if calibrated_speaker_candidates[decision["asset_sha256"]] != int(
            decision["outlier_rank"]
        ):
            raise RuntimeError("Speaker outlier rank drift in manual review")
        speaker_review_rows.append(
            {
                "review_id": _review_uuid(review_payload, "speaker", index),
                "run_id": review_payload["analysis_run_id"],
                "asset_sha256": decision["asset_sha256"],
                "review_status": decision["review_status"],
                "review_note": decision["review_note"],
                "created_at": review_payload["reviewed_at"],
                "review_batch_id": review_payload["review_batch_id"] + ":speaker",
                "source_row_index": index,
            }
        )
    speaker_inserted, speaker_ignored = catalog.record_dataset_speaker_reviews(
        speaker_review_rows
    )
    graph_result = catalog.store_dataset_similarity_graph(
        run_id=review_payload["analysis_run_id"],
        candidate_asset_sha256s=sorted(candidates),
        edges=edge_rows,
        created_at=review_payload["reviewed_at"],
    )
    review_config_json = canonical_json(review_payload)
    output = {
        "schema_version": 1,
        "status": "reviewed",
        "review_id": review_payload["review_id"],
        "review_version": review_payload["review_version"],
        "review_config_sha256": hashlib.sha256(
            review_config_json.encode("utf-8")
        ).hexdigest(),
        "analysis_run_id": review_payload["analysis_run_id"],
        "edge_reviews": {
            "inserted": edge_inserted,
            "ignored": edge_ignored,
            "decisions": edge_decisions,
        },
        "speaker_reviews": {
            "inserted": speaker_inserted,
            "ignored": speaker_ignored,
            "decisions": speaker_decisions,
        },
        "feature_persistence": feature_result,
        "group_materialization": graph_result,
    }
    content = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(args.output.resolve(), content)
    print(content, end="")
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _validate_review_identity(review: dict, overlay: object) -> None:
    expected = {
        "schema_version": 1,
        "dataset_id": overlay.dataset_id,
        "dataset_version": overlay.dataset_version,
        "candidate_snapshot_sha256": overlay.candidate_snapshot_sha256,
    }
    for field, value in expected.items():
        if review.get(field) != value:
            raise RuntimeError(f"Manual review {field} identity mismatch")


def _validate_duplicate_resolution(decision: dict) -> None:
    pair = {decision["left_asset_sha256"], decision["right_asset_sha256"]}
    representative = decision.get("representative_asset_sha256")
    excluded = decision.get("exclude_asset_sha256s")
    if decision.get("review_status") != "accepted":
        raise RuntimeError("This manual review config expects accepted duplicate edges")
    if representative not in pair or not isinstance(excluded, list):
        raise RuntimeError("Duplicate representative decision is malformed")
    if set(excluded) != pair - {representative}:
        raise RuntimeError("Duplicate exclusion must remove every non-representative asset")


def _review_uuid(review: dict, kind: str, index: int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{review['review_batch_id']}:{kind}:{index}",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
