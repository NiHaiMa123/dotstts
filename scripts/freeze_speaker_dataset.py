#!/usr/bin/env python3
"""Speaker-scoped dataset freeze orchestrator.

The historical freeze chain (``dataset-audit`` -> ``materialize_duplicate_graph``
-> ``select_reviewed_dataset`` -> ``audit_dataset_split`` -> ``freeze_dataset``)
was built when the catalog held a single speaker, so ``run_dataset_audit``
enforces candidate-set equality against the whole quality run. This script runs
the same fail-closed pipeline over a speaker-scoped candidate subset:

  register build config -> load candidates -> filter speaker_id ->
  verify quality-report consistency -> resolve eligibility ->
  candidate snapshot -> dedup evidence (exact audio/text, near text, acoustic
  fingerprints, speaker embeddings) -> distribution-based threshold calibration
  -> analysis run + similarity graph -> reviewed exclusions -> stratified split
  -> leakage audit -> atomic publish -> catalog registration.

Fail-closed gates: any pending near-duplicate edge, speaker-embedding outlier,
or hash/provenance drift aborts the run and prints what needs human review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from dots_tts_lab.acoustic_fingerprint import (
    build_acoustic_fingerprint_cache,
    load_acoustic_fingerprint_vectors,
    write_acoustic_fingerprint_report,
)
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_analysis import (
    analyze_exact_audio,
    analyze_exact_text,
    analyze_near_text,
    write_exact_audio_report,
    write_exact_text_report,
    write_near_text_report,
)
from dots_tts_lab.dataset_artifacts import (
    build_dataset_artifact_set,
    validate_dataset_artifact_set,
)
from dots_tts_lab.dataset_freeze import (
    build_candidate_snapshot,
    canonical_dataset_config,
    load_dataset_freeze_config,
    resolve_dataset_freeze_candidates,
    write_candidate_snapshot,
)
from dots_tts_lab.dataset_publish import atomic_publish_dataset
from dots_tts_lab.dataset_registry import (
    publish_dataset_catalog,
    validate_frozen_dataset,
)
from dots_tts_lab.dataset_selection import apply_reviewed_exclusions
from dots_tts_lab.dataset_split import (
    load_verified_similarity_groups,
    plan_emotion_stratified_split,
)
from dots_tts_lab.dataset_split_audit import (
    assert_split_audit_passes,
    audit_split_leakage,
)
from dots_tts_lab.duplicate_graph import build_duplicate_edges
from dots_tts_lab.reports import _atomic_write_text
from dots_tts_lab.speaker_embedding import (
    build_speaker_embedding_cache_and_assessment,
    write_speaker_embedding_report,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> dict[str, str]:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, content)
    return {"path": str(path), "sha256": _sha256_text(content)}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _load_quality_report(path: Path, config) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Quality report must be a schema_version 1 object")
    summary = payload.get("summary")
    assets = payload.get("assets")
    if not isinstance(summary, dict) or not isinstance(assets, list):
        raise RuntimeError("Quality report must contain summary and assets")
    if summary.get("status") not in {"succeeded", "completed_with_errors"}:
        raise RuntimeError(
            f"Quality report status must be succeeded/completed_with_errors, "
            f"got {summary.get('status')!r}"
        )
    _require_equal(
        "quality analysis id", summary.get("analysis_id"),
        config.inputs.quality_analysis_id,
    )
    _require_equal(
        "quality analysis version", summary.get("analysis_version"),
        config.inputs.quality_analysis_version,
    )
    _require_equal(
        "quality policy id", summary.get("policy_id"),
        config.inputs.quality_policy_id,
    )
    _require_equal(
        "quality policy version", summary.get("policy_version"),
        config.inputs.quality_policy_version,
    )
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Quality report summary is missing run_id")
    by_asset = {}
    for item in assets:
        sha = item.get("asset_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha in by_asset:
            raise RuntimeError("Quality report assets are malformed")
        by_asset[sha] = item
    return {"summary": summary, "assets_by_sha256": by_asset}


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.sqlite"))
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--standardized-root", type=Path, default=Path("data/work/standardized"))
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args()

    catalog_path = args.catalog.resolve()
    config_path = args.dataset_config.resolve()
    config = load_dataset_freeze_config(config_path)
    config_identity = canonical_dataset_config(config)
    speaker_id = args.speaker_id
    reports_dir = (
        args.reports_dir
        or Path(f"data/reports/datasets/{config.dataset_id}_v{config.dataset_version}")
    ).resolve()
    target = (args.target or Path(config.output.dataset_root)).resolve()

    catalog = Catalog(catalog_path)
    catalog.register_dataset_build_config(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        config_sha256=config_identity["sha256"],
        config_json=config_identity["canonical_json"],
        implementation_version=config.implementation_version,
        source_path=str(config_path),
        registered_at=_now(),
    )

    # Human review decisions persist across reruns: the review file is
    # re-generated with fresh run ids each pass, but prior edge/speaker
    # decisions are carried over and replayed idempotently into the catalog.
    review_config_path = (
        config_path.parent / f"{config.dataset_id}_v{config.dataset_version}_manual_review.yaml"
    )
    prior_edge_decisions: list[dict] = []
    prior_speaker_decisions: list[dict] = []
    if review_config_path.exists():
        prior_review = yaml.safe_load(
            review_config_path.read_text(encoding="utf-8")
        )
        if isinstance(prior_review, dict):
            for key, decision_list in (
                ("edge_decisions", prior_edge_decisions),
                ("speaker_decisions", prior_speaker_decisions),
            ):
                value = prior_review.get(key)
                if value is None:
                    continue
                if not isinstance(value, list) or not all(
                    isinstance(item, dict) for item in value
                ):
                    raise RuntimeError(
                        f"Manual review file has malformed {key}: {review_config_path}"
                    )
                decision_list.extend(value)

    quality_path = Path(
        args.quality_report or config.inputs.quality_report_path
    ).resolve()
    quality = _load_quality_report(quality_path, config)
    quality_run_id = str(quality["summary"]["run_id"])

    rows = catalog.load_dataset_freeze_candidates(
        standardization_config_id=config.inputs.standardization_config_id,
        standardization_config_version=config.inputs.standardization_config_version,
        quality_run_id=quality_run_id,
        review_benchmark_id=config.inputs.review_benchmark_id,
        review_benchmark_version=config.inputs.review_benchmark_version,
    )
    scoped = [row for row in rows if str(row.get("speaker_id")) == speaker_id]
    if not scoped:
        raise RuntimeError(f"No freeze candidates for speaker {speaker_id!r}")
    for row in scoped:
        asset = str(row["asset_sha256"])
        report_item = quality["assets_by_sha256"].get(asset)
        if report_item is None:
            raise RuntimeError(f"Candidate {asset} missing from quality report")
        _require_equal(
            f"quality decision for {asset}",
            row["quality_decision"], report_item.get("decision"),
        )
        _require_equal(
            f"quality reasons for {asset}",
            json.loads(str(row["quality_reasons_json"])), report_item.get("reasons"),
        )

    resolved = resolve_dataset_freeze_candidates(
        scoped, config=config, standardized_root=args.standardized_root
    )
    eligible = resolved["eligible"]
    excluded = resolved["excluded"]
    if not eligible:
        raise RuntimeError("Speaker-scoped freeze has no eligible candidates")

    snapshot = build_candidate_snapshot(eligible)
    snapshot_path = reports_dir / "audit" / "candidate_snapshot.json"
    write_candidate_snapshot(snapshot, snapshot_path)
    snapshot_sha256 = str(snapshot["sha256"])

    analysis_dir = reports_dir / "analysis"
    exact_audio = analyze_exact_audio(eligible)
    exact_audio_info = write_exact_audio_report(exact_audio, analysis_dir / "exact_audio.json")
    exact_text = analyze_exact_text(eligible, config=config.text_similarity)
    exact_text_info = write_exact_text_report(exact_text, analysis_dir / "exact_text.json")
    near_text = analyze_near_text(eligible, config=config.text_similarity)
    near_text_info = write_near_text_report(near_text, analysis_dir / "near_text.json")
    fingerprint = build_acoustic_fingerprint_cache(
        eligible, config=config.acoustic_fingerprint
    )
    fingerprint_info = write_acoustic_fingerprint_report(
        fingerprint["report"], analysis_dir / "acoustic_fingerprints.json"
    )
    speaker = build_speaker_embedding_cache_and_assessment(
        eligible, config=config.speaker_embedding
    )
    speaker_info = write_speaker_embedding_report(
        speaker["report"], analysis_dir / "speaker_embeddings.json"
    )

    # --- threshold calibration from observed distributions -----------------
    near_pairs = near_text.get("pairs") or []
    near_sims = [float(p["similarity"]) for p in near_pairs]
    if near_sims:
        text_threshold = min(0.99, max(near_sims) + 0.02)
        text_rule = "max_observed_pair_similarity_plus_margin"
    else:
        text_threshold = 0.98
        text_rule = "no_near_text_pairs_conservative_default"

    vectors = load_acoustic_fingerprint_vectors(fingerprint["report"])
    assets_sorted = sorted(vectors)
    matrix = np.stack([vectors[a] for a in assets_sorted]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = matrix @ matrix.T
    pair_cosines = [
        float(cosine[i, j])
        for i in range(len(assets_sorted))
        for j in range(i + 1, len(assets_sorted))
    ]
    acoustic_threshold = min(0.999, max(pair_cosines) + 0.01)
    acoustic_rule = "max_background_cosine_plus_margin"

    assessments = speaker["report"]["assessments"]
    center_vals = [a["center_cosine"] for a in assessments]
    knn_vals = [a["knn_cosine"] for a in assessments]

    def _robust_lower_bound(values: list[float], sigma_cutoff: float = 4.5) -> float:
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median))) * 1.4826
        if mad <= 0:
            return float(array.min()) - 0.01
        return median - sigma_cutoff * mad

    center_threshold = _robust_lower_bound(center_vals)
    knn_threshold = _robust_lower_bound(knn_vals)
    outliers = [
        a["asset_sha256"]
        for a in assessments
        if a["center_cosine"] < center_threshold or a["knn_cosine"] < knn_threshold
    ]
    reviewed_speaker_assets = {
        str(decision.get("asset_sha256"))
        for decision in prior_speaker_decisions
        if decision.get("review_status")
        in {
            "confirmed_same_speaker",
            "exclude_wrong_speaker",
            "exclude_uncertain",
        }
    }
    unreviewed_outliers = [
        asset for asset in outliers if asset not in reviewed_speaker_assets
    ]
    if unreviewed_outliers:
        details = [
            {
                "asset_sha256": a["asset_sha256"],
                "center_cosine": a["center_cosine"],
                "knn_cosine": a["knn_cosine"],
                "outlier_rank": a["outlier_rank"],
                "source_relative_path": next(
                    item["source_relative_path"]
                    for item in eligible
                    if item["asset_sha256"] == a["asset_sha256"]
                ),
            }
            for a in assessments
            if a["asset_sha256"] in unreviewed_outliers
        ]
        raise RuntimeError(
            "Speaker-embedding outliers require human review before freeze "
            f"(record speaker_decisions in {review_config_path}): "
            + json.dumps(details, ensure_ascii=False)
        )

    thresholds = {
        "text_near_similarity": float(text_threshold),
        "acoustic_near_cosine": float(acoustic_threshold),
        "speaker_center_cosine": float(center_threshold),
        "speaker_knn_cosine": float(knn_threshold),
    }
    calibration_config_payload = {
        "schema_version": 1,
        "calibration_id": f"{config.dataset_id}_dataset_thresholds",
        "calibration_version": 1,
        "dataset_config_path": str(config_path),
        "dataset_config_sha256": config_identity["sha256"],
        "candidate_snapshot_path": str(snapshot_path),
        "candidate_snapshot_sha256": snapshot_sha256,
        "threshold_rules": {
            "text": text_rule,
            "acoustic": acoustic_rule,
            "speaker": "median_minus_4.5_robust_sigma_lower_bound",
        },
        "output_path": str(analysis_dir / "threshold_calibration_v1.json"),
    }
    calibration_config_path = (
        config_path.parent / f"{config.dataset_id}_v{config.dataset_version}_threshold_calibration.yaml"
    )
    calibration_config_content = yaml.safe_dump(
        calibration_config_payload, allow_unicode=True, sort_keys=False
    )
    _atomic_write_text(calibration_config_path, calibration_config_content)
    calibration_config_sha256 = _sha256_text(calibration_config_content)

    calibration_report = {
        "schema_version": 1,
        "status": "succeeded",
        "calibration_id": calibration_config_payload["calibration_id"],
        "calibration_version": 1,
        "calibration_config_sha256": calibration_config_sha256,
        "candidate_snapshot_sha256": snapshot_sha256,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "speaker_scope": speaker_id,
        "text": {
            "report_path": near_text_info["path"],
            "report_sha256": near_text_info["sha256"],
            "threshold_rule": text_rule,
            "pair_count": len(near_sims),
            "distribution": _distribution(near_sims) if near_sims else None,
            "threshold": thresholds["text_near_similarity"],
        },
        "acoustic": {
            "report_path": fingerprint_info["path"],
            "report_sha256": fingerprint_info["sha256"],
            "threshold_rule": acoustic_rule,
            "scored_pair_count": len(pair_cosines),
            "distribution": _distribution(pair_cosines),
            "threshold": thresholds["acoustic_near_cosine"],
        },
        "speaker": {
            "report_path": speaker_info["path"],
            "report_sha256": speaker_info["sha256"],
            "candidate_rule": "center_or_knn_below_robust_lower_bound",
            "mad_consistency_scale": 1.4826,
            "robust_sigma_cutoff": 4.5,
            "center_cosine": {
                "distribution": _distribution(center_vals),
                "threshold": thresholds["speaker_center_cosine"],
            },
            "knn_cosine": {
                "distribution": _distribution(knn_vals),
                "threshold": thresholds["speaker_knn_cosine"],
            },
            "outlier_count": 0,
        },
        "thresholds": thresholds,
        "policy": {"automatic_exclusion": False},
    }
    calibration_report_path = analysis_dir / "threshold_calibration_v1.json"
    calibration_report_info = _write_json(calibration_report_path, calibration_report)
    catalog.register_dataset_threshold_calibration(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        calibration_id=calibration_config_payload["calibration_id"],
        calibration_version=1,
        calibration_config_sha256=calibration_config_sha256,
        calibration_report_sha256=calibration_report_info["sha256"],
        report_path=calibration_report_info["path"],
        thresholds=thresholds,
        registered_at=_now(),
    )

    # --- analysis run + similarity graph ------------------------------------
    run_id = catalog.begin_dataset_analysis_run(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        candidate_snapshot_sha256=snapshot_sha256,
        calibration_id=calibration_config_payload["calibration_id"],
        calibration_version=1,
        calibration_config_sha256=calibration_config_sha256,
        calibration_report_sha256=calibration_report_info["sha256"],
        started_at=_now(),
    )
    edges_result = build_duplicate_edges(
        candidate_snapshot=snapshot["payload"],
        exact_audio_report=exact_audio,
        exact_text_report=exact_text,
        near_text_report=near_text,
        acoustic_fingerprint_report=fingerprint["report"],
        text_threshold=thresholds["text_near_similarity"],
        acoustic_threshold=thresholds["acoustic_near_cosine"],
        acoustic_duration_ratio_min=config.acoustic_fingerprint.duration_ratio_min,
        source_sha256s={
            "candidate_snapshot": snapshot_sha256,
            "exact_audio": exact_audio_info["sha256"],
            "exact_text": exact_text_info["sha256"],
            "near_text": near_text_info["sha256"],
            "acoustic_fingerprint": fingerprint_info["sha256"],
        },
    )
    catalog.store_dataset_similarity_graph(
        run_id=run_id,
        candidate_asset_sha256s=[item["asset_sha256"] for item in eligible],
        edges=edges_result["edges"],
        created_at=_now(),
    )

    # Replay recorded human edge reviews into the catalog, then let the
    # similarity graph rematerialize groups under the new review snapshot.
    review_batch_id = f"{config.dataset_id}-v{config.dataset_version}-freeze-review"
    edge_review_rows = []
    edge_index = {
        (e["left_asset_sha256"], e["right_asset_sha256"], e["evidence_type"])
        for e in edges_result["edges"]
    }
    pending_edge_index = {
        (e["left_asset_sha256"], e["right_asset_sha256"], e["evidence_type"])
        for e in edges_result["edges"]
        if e["analysis_status"] == "pending_review"
    }
    for row_index, decision in enumerate(prior_edge_decisions):
        key = (
            decision.get("left_asset_sha256"),
            decision.get("right_asset_sha256"),
            decision.get("evidence_type"),
        )
        if key not in edge_index:
            raise RuntimeError(
                f"Edge review decision does not match a graph edge: {key}"
            )
        if key not in pending_edge_index:
            continue
        edge_review_rows.append(
            {
                "review_id": f"{review_batch_id}-edge-{row_index:04d}",
                "run_id": run_id,
                "left_asset_sha256": decision["left_asset_sha256"],
                "right_asset_sha256": decision["right_asset_sha256"],
                "evidence_type": decision["evidence_type"],
                "review_status": decision["review_status"],
                "review_note": decision.get("review_note"),
                "created_at": _now(),
                "review_batch_id": review_batch_id,
                "source_row_index": row_index,
            }
        )
    if edge_review_rows:
        catalog.record_dataset_similarity_edge_reviews(edge_review_rows)
        catalog.store_dataset_similarity_graph(
            run_id=run_id,
            candidate_asset_sha256s=[item["asset_sha256"] for item in eligible],
            edges=edges_result["edges"],
            created_at=_now(),
        )

    reviewed_edge_keys = {
        (r["left_asset_sha256"], r["right_asset_sha256"], r["evidence_type"])
        for r in edge_review_rows
    }
    pending_edges = [
        edge
        for edge in edges_result["edges"]
        if edge["analysis_status"] == "pending_review"
        and (
            edge["left_asset_sha256"],
            edge["right_asset_sha256"],
            edge["evidence_type"],
        )
        not in reviewed_edge_keys
    ]
    if pending_edges:
        details = [
            {
                "evidence_type": e["evidence_type"],
                "score": e["score"],
                "left": e["left_asset_sha256"],
                "right": e["right_asset_sha256"],
                "left_text": next(
                    i["text_exact"] for i in eligible
                    if i["asset_sha256"] == e["left_asset_sha256"]
                ),
                "right_text": next(
                    i["text_exact"] for i in eligible
                    if i["asset_sha256"] == e["right_asset_sha256"]
                ),
            }
            for e in pending_edges
        ]
        raise RuntimeError(
            "Near-duplicate edges require human review before freeze "
            f"(record edge_decisions in {review_config_path}): "
            + json.dumps(details, ensure_ascii=False)
        )

    groups, review_snapshot_sha256 = load_verified_similarity_groups(
        catalog,
        run_id=run_id,
        candidate_snapshot_sha256=snapshot_sha256,
        candidate_count=len(eligible),
    )

    # --- reviewed selection + stratified split ------------------------------
    review_payload = {
        "schema_version": 1,
        "review_id": f"{config.dataset_id}_v{config.dataset_version}_freeze_review",
        "review_version": 1,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "analysis_run_id": run_id,
        "candidate_snapshot_sha256": snapshot_sha256,
        "review_batch_id": review_batch_id,
        "reviewed_at": _now(),
        "reviewer": "freeze_speaker_dataset.py",
        "edge_decisions": [
            decision
            for decision in prior_edge_decisions
            if decision.get("review_status") == "accepted"
        ],
        "speaker_decisions": prior_speaker_decisions,
    }
    # The on-disk file keeps every recorded decision (accepted and rejected)
    # so reruns replay them; the in-memory payload only feeds accepted edges
    # into apply_reviewed_exclusions, which rejects other statuses.
    review_file_payload = {
        **review_payload,
        "edge_decisions": prior_edge_decisions,
    }
    review_content = yaml.safe_dump(
        review_file_payload, allow_unicode=True, sort_keys=False
    )
    _atomic_write_text(review_config_path, review_content)
    review_config_sha256 = _sha256_text(review_content)

    selection = apply_reviewed_exclusions(
        items=list(snapshot["payload"]["items"]),
        groups=groups,
        review=review_payload,
    )
    selection_artifact = {
        "schema_version": 1,
        "status": "selected",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "analysis_run_id": run_id,
        "candidate_snapshot_sha256": snapshot_sha256,
        "edge_review_snapshot_sha256": review_snapshot_sha256,
        "review_config_sha256": review_config_sha256,
        "dataset_config_sha256": config_identity["sha256"],
        **selection,
    }
    freeze_dir = reports_dir / "freeze"
    selection_info = _write_json(freeze_dir / "reviewed_selection.json", selection_artifact)

    split = plan_emotion_stratified_split(
        items=selection["selected_items"],
        groups=selection["selected_groups"],
        split_config=config.split,
        dataset_config_sha256=config_identity["sha256"],
    )
    split_artifact = {
        **split,
        "scope": "reviewed_selection_final_candidate_split",
        "analysis_run_id": run_id,
        "candidate_snapshot_sha256": snapshot_sha256,
        "edge_review_snapshot_sha256": review_snapshot_sha256,
        "review_config_sha256": review_config_sha256,
        "selection_sha256": selection["selection_sha256"],
        "note": (
            "Speaker-scoped freeze split; no sampling or replication applied."
        ),
    }
    split_info = _write_json(freeze_dir / "reviewed_selection_split.json", split_artifact)

    with catalog.read_only_session() as connection:
        stored_edges = [
            dict(row)
            for row in connection.execute(
                """
                SELECT left_asset_sha256, right_asset_sha256, evidence_type,
                       score, threshold, analysis_status
                FROM dataset_similarity_edge
                WHERE run_id = ?
                ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                """,
                (run_id,),
            )
        ]
    audit_report = audit_split_leakage(
        items=selection["selected_items"],
        split_plan=split_artifact,
        similarity_edges=stored_edges,
        text_config=config.text_similarity,
    )
    audit_artifact = {
        **audit_report,
        "analysis_run_id": run_id,
        "selection_sha256": selection["selection_sha256"],
        "assignment_sha256": split["assignment_sha256"],
    }
    assert_split_audit_passes(audit_artifact)
    audit_info = _write_json(freeze_dir / "split_audit.json", audit_artifact)

    # --- atomic publish ------------------------------------------------------
    input_hashes = {
        "reviewed_selection.json": selection_info["sha256"],
        "reviewed_selection_split.json": split_info["sha256"],
        "split_audit.json": audit_info["sha256"],
    }
    publish = atomic_publish_dataset(
        target,
        build=lambda staging: build_dataset_artifact_set(
            staging,
            config=config,
            selection=selection_artifact,
            split_plan=split_artifact,
            split_audit=audit_artifact,
            input_file_sha256s=input_hashes,
            standardized_root=args.standardized_root,
        ),
        validate=lambda staging: validate_dataset_artifact_set(staging, config=config),
    )
    publish["validation"] = validate_dataset_artifact_set(target, config=config)
    publish["deep_validation"] = validate_frozen_dataset(target, config=config)
    published_at = _now()
    publish["catalog"] = publish_dataset_catalog(
        catalog, target, config=config, published_at=published_at
    )
    publish["catalog_validation"] = validate_frozen_dataset(
        target, config=config, catalog=catalog
    )

    scoped_audit = {
        "schema_version": 1,
        "status": "succeeded",
        "scope": "speaker_scoped_freeze",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "speaker_id": speaker_id,
        "analysis_run_id": run_id,
        "candidate_snapshot_sha256": snapshot_sha256,
        "candidate_snapshot_path": str(snapshot_path),
        "counts": {
            "speaker_candidates": len(scoped),
            "eligible": len(eligible),
            "excluded": len(excluded),
            "selected": selection["selected_count"],
            "edges": edges_result["edge_count"],
            "edges_by_evidence": edges_result["edge_counts_by_evidence_type"],
        },
        "exclusions": excluded,
        "provenance": {
            "catalog_path": str(catalog_path),
            "dataset_config_path": str(config_path),
            "dataset_config_sha256": config_identity["sha256"],
            "quality_run_id": quality_run_id,
            "quality_report_path": str(quality_path),
            "calibration_id": calibration_config_payload["calibration_id"],
            "calibration_report_sha256": calibration_report_info["sha256"],
            "selection_sha256": selection["selection_sha256"],
            "assignment_sha256": split["assignment_sha256"],
            "split_audit_sha256": audit_info["sha256"],
        },
    }
    _write_json(reports_dir / "audit" / "dataset_audit.json", scoped_audit)

    print(json.dumps({
        "status": "succeeded",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "speaker_id": speaker_id,
        "analysis_run_id": run_id,
        "candidates": len(scoped),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "selected": selection["selected_count"],
        "splits": split["summary"],
        "publish": publish,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
