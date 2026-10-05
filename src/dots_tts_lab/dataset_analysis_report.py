from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.duplicate_graph import canonical_json, canonical_sha256
from dots_tts_lab.reports import _atomic_write_text
from dots_tts_lab.threshold_calibration import ThresholdCalibrationOverlay


DEFAULT_DATASET_ANALYSIS_REPORT_DIR = Path(
    "data/reports/datasets/fuxuan_v1/analysis"
)

_CSV_FIELDS = (
    "record_type",
    "evidence_type",
    "review_status",
    "included_in_group",
    "left_asset_sha256",
    "left_fid",
    "left_text",
    "left_audio_path",
    "right_asset_sha256",
    "right_fid",
    "right_text",
    "right_audio_path",
    "score",
    "threshold",
    "margin",
    "reasons",
    "neighbor_evidence_json",
)


def build_dataset_analysis_report(
    *,
    catalog: Catalog,
    run_id: str,
    candidate_snapshot: dict[str, Any],
    candidate_snapshot_sha256: str,
    speaker_report: dict[str, Any],
    speaker_report_sha256: str,
    calibration_report: dict[str, Any],
    overlay: ThresholdCalibrationOverlay,
    standardized_root: str | Path,
) -> dict[str, Any]:
    """Build one verified, deterministic duplicate and speaker review report."""
    if candidate_snapshot_sha256 != overlay.candidate_snapshot_sha256:
        raise RuntimeError("Analysis report candidate snapshot differs from calibration")
    if calibration_report.get("status") != "succeeded":
        raise RuntimeError("Analysis report calibration did not succeed")
    if calibration_report.get("speaker", {}).get("candidate_count") is None:
        raise RuntimeError("Analysis report calibration has no speaker candidates")
    expected_speaker_sha256 = calibration_report.get("inputs", {}).get(
        "speaker_report_sha256"
    )
    if expected_speaker_sha256 != speaker_report_sha256:
        raise RuntimeError("Speaker report SHA-256 differs from calibration input")
    if speaker_report.get("status") != "succeeded":
        raise RuntimeError("Speaker report did not succeed")

    items = candidate_snapshot.get("items")
    if not isinstance(items, list) or candidate_snapshot.get("item_count") != len(items):
        raise RuntimeError("Analysis report candidate snapshot count mismatch")
    candidates = {}
    for item in items:
        asset_sha256 = _require_sha256(item.get("asset_sha256"), "candidate asset")
        if asset_sha256 in candidates:
            raise RuntimeError(f"Duplicate analysis report candidate: {asset_sha256}")
        candidates[asset_sha256] = item
    root = Path(standardized_root).resolve()
    audio_cache: dict[str, dict[str, str]] = {}

    def candidate_view(asset_sha256: str) -> dict[str, Any]:
        item = candidates.get(asset_sha256)
        if item is None:
            raise RuntimeError(f"Analysis evidence references unknown asset: {asset_sha256}")
        if asset_sha256 not in audio_cache:
            audio_cache[asset_sha256] = _verified_audio_view(item, root)
        return {
            "asset_sha256": asset_sha256,
            "fid": item.get("fid"),
            "text": item.get("text_exact"),
            "text_source": item.get("text_source"),
            "emotion_primary": item.get("emotion_primary"),
            "duration_seconds": item.get("duration_seconds"),
            **audio_cache[asset_sha256],
        }

    with catalog.read_only_session() as connection:
        run = connection.execute(
            """
            SELECT run.*, binding.calibration_id, binding.calibration_version,
                   binding.calibration_config_sha256,
                   binding.calibration_report_sha256
            FROM dataset_analysis_run AS run
            JOIN dataset_analysis_calibration AS binding
              ON binding.run_id = run.run_id
            WHERE run.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise RuntimeError(f"Dataset analysis run does not exist: {run_id}")
        expected_run_identity = {
            "dataset_id": overlay.dataset_id,
            "dataset_version": overlay.dataset_version,
            "candidate_snapshot_sha256": overlay.candidate_snapshot_sha256,
            "calibration_id": overlay.calibration_id,
            "calibration_version": overlay.calibration_version,
            "calibration_config_sha256": overlay.calibration_config_sha256,
            "calibration_report_sha256": overlay.calibration_report_sha256,
        }
        for field, expected in expected_run_identity.items():
            if run[field] != expected:
                raise RuntimeError(f"Analysis run {field} differs from calibration overlay")
        if int(run["candidate_count"]) != len(candidates):
            raise RuntimeError("Analysis run candidate count mismatch")

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
                (run_id,),
            )
        ]
        latest_reviews = {
            (
                row["left_asset_sha256"],
                row["right_asset_sha256"],
                row["evidence_type"],
            ): dict(row)
            for row in connection.execute(
                """
                SELECT review_id, left_asset_sha256, right_asset_sha256,
                       evidence_type, review_round, review_status,
                       review_note, created_at, review_batch_id,
                       source_row_index
                FROM dataset_similarity_edge_review AS review
                WHERE review.run_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM dataset_similarity_edge_review AS newer
                      WHERE newer.run_id = review.run_id
                        AND newer.left_asset_sha256 = review.left_asset_sha256
                        AND newer.right_asset_sha256 = review.right_asset_sha256
                        AND newer.evidence_type = review.evidence_type
                        AND newer.review_round > review.review_round
                  )
                ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                """,
                (run_id,),
            )
        }
        latest_speaker_reviews = {
            row["asset_sha256"]: dict(row)
            for row in connection.execute(
                """
                SELECT review_id, asset_sha256, review_round, review_status,
                       review_note, created_at, review_batch_id,
                       source_row_index
                FROM dataset_speaker_review AS review
                WHERE review.run_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM dataset_speaker_review AS newer
                      WHERE newer.run_id = review.run_id
                        AND newer.asset_sha256 = review.asset_sha256
                        AND newer.review_round > review.review_round
                  )
                ORDER BY asset_sha256
                """,
                (run_id,),
            )
        }
        group_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT grouped.group_id, grouped.member_count,
                       grouped.edge_review_snapshot_sha256,
                       member.asset_sha256
                FROM dataset_similarity_group AS grouped
                JOIN dataset_similarity_group_member AS member
                  ON member.run_id = grouped.run_id
                 AND member.group_id = grouped.group_id
                WHERE grouped.run_id = ?
                ORDER BY grouped.group_id, member.asset_sha256
                """,
                (run_id,),
            )
        ]

    asset_group_ids, group_summary = _validate_groups(group_rows, set(candidates))
    expected_review_snapshot = hashlib.sha256(
        canonical_json(list(latest_reviews.values())).encode("utf-8")
    ).hexdigest()
    if group_summary["edge_review_snapshot_sha256"] != expected_review_snapshot:
        raise RuntimeError("Materialized groups do not match latest edge reviews")
    duplicate_edges = []
    for edge in edge_rows:
        evidence = json.loads(edge["evidence_json"])
        if canonical_json(evidence) != edge["evidence_json"]:
            raise RuntimeError("Catalog duplicate evidence JSON is not canonical")
        key = (
            edge["left_asset_sha256"],
            edge["right_asset_sha256"],
            edge["evidence_type"],
        )
        review = latest_reviews.get(key)
        exact = edge["analysis_status"] == "accepted_exact"
        review_status = "accepted_exact" if exact else "pending_review"
        if review is not None:
            review_status = review["review_status"]
        included = exact or review_status == "accepted"
        left_view = candidate_view(edge["left_asset_sha256"])
        right_view = candidate_view(edge["right_asset_sha256"])
        if included and (
            asset_group_ids[left_view["asset_sha256"]]
            != asset_group_ids[right_view["asset_sha256"]]
        ):
            raise RuntimeError("Materialized group membership differs from accepted edge state")
        threshold = edge["threshold"]
        duplicate_edges.append(
            {
                "evidence_type": edge["evidence_type"],
                "analysis_status": edge["analysis_status"],
                "review_status": review_status,
                "included_in_group": included,
                "score": float(edge["score"]),
                "threshold": None if threshold is None else float(threshold),
                "margin": None if threshold is None else float(edge["score"] - threshold),
                "reasons": _duplicate_reasons(edge, evidence),
                "left": left_view,
                "right": right_view,
                "evidence": evidence,
                "review": review,
            }
        )

    assessments = speaker_report.get("assessments")
    if not isinstance(assessments, list):
        raise RuntimeError("Speaker report assessments are missing")
    speaker_outliers = []
    for assessment in assessments:
        center = float(assessment["center_cosine"])
        knn = float(assessment["knn_cosine"])
        center_triggered = center < overlay.thresholds.speaker_center_cosine
        knn_triggered = knn < overlay.thresholds.speaker_knn_cosine
        if not center_triggered and not knn_triggered:
            continue
        neighbor_assets = assessment.get("neighbor_asset_sha256s")
        neighbor_cosines = assessment.get("neighbor_cosines")
        if (
            not isinstance(neighbor_assets, list)
            or not isinstance(neighbor_cosines, list)
            or len(neighbor_assets) != len(neighbor_cosines)
        ):
            raise RuntimeError("Speaker neighbor evidence is malformed")
        neighbors = [
            {**candidate_view(asset), "cosine": float(cosine)}
            for asset, cosine in zip(neighbor_assets, neighbor_cosines, strict=True)
        ]
        speaker_outliers.append(
            {
                "review_status": latest_speaker_reviews.get(
                    assessment["asset_sha256"], {}
                ).get("review_status", "pending_review"),
                "review": latest_speaker_reviews.get(assessment["asset_sha256"]),
                "asset": candidate_view(assessment["asset_sha256"]),
                "outlier_rank": int(assessment["outlier_rank"]),
                "outlier_score": float(assessment["outlier_score"]),
                "center_cosine": center,
                "center_threshold": overlay.thresholds.speaker_center_cosine,
                "knn_cosine": knn,
                "knn_threshold": overlay.thresholds.speaker_knn_cosine,
                "reasons": [
                    reason
                    for triggered, reason in (
                        (
                            center_triggered,
                            "center_cosine_below_calibrated_threshold",
                        ),
                        (knn_triggered, "knn_cosine_below_calibrated_threshold"),
                    )
                    if triggered
                ],
                "neighbors": neighbors,
            }
        )
    speaker_outliers.sort(key=lambda item: (item["outlier_rank"], item["asset"]["asset_sha256"]))
    calibrated_candidates = {
        item["asset_sha256"]
        for item in calibration_report["speaker"].get("candidates", [])
    }
    if calibrated_candidates != {
        item["asset"]["asset_sha256"] for item in speaker_outliers
    }:
        raise RuntimeError("Recomputed speaker outliers differ from calibration report")

    edge_counts = {
        evidence_type: sum(
            edge["evidence_type"] == evidence_type for edge in duplicate_edges
        )
        for evidence_type in ("exact_audio", "exact_text", "near_audio", "near_text")
    }
    pending_edge_count = sum(
        edge["review_status"] == "pending_review" for edge in duplicate_edges
    )
    pending_speaker_count = sum(
        outlier["review_status"] == "pending_review" for outlier in speaker_outliers
    )
    speaker_exclusion_count = sum(
        outlier["review_status"]
        in ("exclude_wrong_speaker", "exclude_uncertain")
        for outlier in speaker_outliers
    )
    return {
        "schema_version": 1,
        "status": (
            "review_required"
            if pending_edge_count or pending_speaker_count
            else "reviewed"
        ),
        "summary": {
            "run_id": run_id,
            "dataset_id": overlay.dataset_id,
            "dataset_version": overlay.dataset_version,
            "candidate_count": len(candidates),
            "duplicate_edge_count": len(duplicate_edges),
            "duplicate_edge_counts_by_type": edge_counts,
            "pending_duplicate_review_count": pending_edge_count,
            "speaker_outlier_count": len(speaker_outliers),
            "pending_speaker_review_count": pending_speaker_count,
            "speaker_exclusion_count": speaker_exclusion_count,
            **group_summary,
        },
        "provenance": {
            "catalog_path": str(catalog.path.resolve()),
            "candidate_snapshot_sha256": candidate_snapshot_sha256,
            "speaker_report_sha256": speaker_report_sha256,
            "calibration_id": overlay.calibration_id,
            "calibration_version": overlay.calibration_version,
            "calibration_config_sha256": overlay.calibration_config_sha256,
            "calibration_report_sha256": overlay.calibration_report_sha256,
        },
        "thresholds": overlay.thresholds.model_dump(mode="json"),
        "duplicate_edges": duplicate_edges,
        "speaker_outliers": speaker_outliers,
    }


def write_dataset_analysis_reports(
    report: dict[str, Any],
    output_dir: str | Path = DEFAULT_DATASET_ANALYSIS_REPORT_DIR,
) -> dict[str, str]:
    output = Path(output_dir).resolve()
    json_path = output / "dataset_analysis.json"
    csv_path = output / "dataset_analysis.csv"
    html_path = output / "dataset_analysis.html"
    contents = {
        "json": json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "csv": _analysis_csv(report),
        "html": _analysis_html(report),
    }
    paths = {"json": json_path, "csv": csv_path, "html": html_path}
    for kind, path in paths.items():
        _atomic_write_text(path, contents[kind])
    return {
        f"{kind}_{field}": value
        for kind, path in paths.items()
        for field, value in (
            ("path", str(path)),
            ("sha256", hashlib.sha256(contents[kind].encode("utf-8")).hexdigest()),
        )
    }


def write_portable_dataset_analysis_bundle(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    online_url: str | None = None,
) -> dict[str, Any]:
    """Copy only referenced audio and emit a report without local absolute paths."""
    output = Path(output_dir).resolve()
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    portable = json.loads(json.dumps(report, ensure_ascii=False))
    portable["provenance"]["catalog_path"] = "not_included_in_portable_bundle"
    samples = []
    for edge in portable["duplicate_edges"]:
        samples.extend((edge["left"], edge["right"]))
    for outlier in portable["speaker_outliers"]:
        samples.append(outlier["asset"])
        samples.extend(outlier["neighbors"])

    audio_by_asset: dict[str, tuple[Path, Path]] = {}
    for sample in samples:
        asset_sha256 = _require_sha256(sample["asset_sha256"], "bundle asset")
        source = Path(sample["audio_absolute_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Portable bundle audio does not exist: {source}")
        destination = audio_dir / f"{asset_sha256}.wav"
        existing = audio_by_asset.get(asset_sha256)
        if existing is not None and existing[0] != source:
            raise RuntimeError(f"Bundle asset references multiple audio files: {asset_sha256}")
        audio_by_asset[asset_sha256] = (source, destination)
        relative_path = f"audio/{destination.name}"
        sample["audio_absolute_path"] = relative_path
        sample["audio_url"] = relative_path

    for source, destination in audio_by_asset.values():
        source_sha256 = _sha256_file(source)
        if destination.exists() and _sha256_file(destination) == source_sha256:
            continue
        partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        try:
            shutil.copyfile(source, partial)
            with partial.open("rb+") as copied:
                copied.flush()
                os.fsync(copied.fileno())
            if _sha256_file(partial) != source_sha256:
                raise RuntimeError(f"Portable bundle audio copy drift: {source}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    summary = portable["summary"]
    review_state = (
        "Review decisions are complete. "
        f"Accepted duplicate pairs: "
        f"{summary['duplicate_edge_count'] - summary['pending_duplicate_review_count']}; "
        f"speaker exclusions: {summary['speaker_exclusion_count']}."
        if portable["status"] == "reviewed"
        else "Review decisions are still pending."
    )
    online_line = f"Online report: <{online_url}>\n\n" if online_url else ""
    contents = {
        "index.html": _analysis_html(portable),
        "dataset_analysis.json": (
            json.dumps(portable, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ),
        "dataset_analysis.csv": _analysis_csv(portable),
        "README.md": (
            "# fuxuan Slice 8 review bundle\n\n"
            f"{online_line}"
            "Open `index.html` to inspect the duplicate and speaker review evidence. "
            f"{review_state} This bundle is read-only and includes only the audio "
            "referenced by the report. File integrity is listed in `checksums.txt`.\n"
        ),
        ".gitattributes": "* text eol=lf\n*.wav binary\n",
    }
    for name, content in contents.items():
        _atomic_write_text(output / name, content)
    checksum_rows = []
    artifact_paths = [output / name for name in contents]
    artifact_paths.extend(destination for _, destination in audio_by_asset.values())
    for path in sorted(artifact_paths, key=lambda item: item.relative_to(output).as_posix()):
        checksum_rows.append(f"{_sha256_file(path)}  {path.relative_to(output).as_posix()}")
    _atomic_write_text(output / "checksums.txt", "\n".join(checksum_rows) + "\n")
    return {
        "output_dir": str(output),
        "audio_file_count": len(audio_by_asset),
        "total_audio_bytes": sum(
            destination.stat().st_size for _, destination in audio_by_asset.values()
        ),
        "checksums_sha256": _sha256_file(output / "checksums.txt"),
    }


def _verified_audio_view(item: dict[str, Any], root: Path) -> dict[str, str]:
    relative_value = item.get("audio_relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise RuntimeError("Candidate has no standardized audio path")
    relative = PurePosixPath(relative_value)
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("Candidate audio path escapes standardized root") from error
    if not path.is_file():
        raise FileNotFoundError(f"Review audio does not exist: {path}")
    expected_sha256 = _require_sha256(item.get("audio_sha256"), "candidate audio")
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Review audio SHA-256 drift: {path}")
    return {
        "audio_relative_path": relative.as_posix(),
        "audio_absolute_path": str(path),
        "audio_url": path.as_uri(),
    }


def _validate_groups(
    rows: list[dict[str, Any]], candidate_assets: set[str]
) -> tuple[dict[str, str], dict[str, int]]:
    members_by_group: dict[str, list[str]] = {}
    counts_by_group: dict[str, int] = {}
    snapshots = set()
    for row in rows:
        group_id = row["group_id"]
        members_by_group.setdefault(group_id, []).append(row["asset_sha256"])
        counts_by_group[group_id] = int(row["member_count"])
        snapshots.add(row["edge_review_snapshot_sha256"])
    asset_group_ids = {}
    for group_id, members in members_by_group.items():
        ordered = sorted(members)
        if counts_by_group[group_id] != len(ordered):
            raise RuntimeError("Materialized similarity group count mismatch")
        if canonical_sha256(ordered) != group_id:
            raise RuntimeError("Materialized similarity group ID drift")
        for asset_sha256 in ordered:
            if asset_sha256 in asset_group_ids:
                raise RuntimeError("Asset belongs to multiple similarity groups")
            asset_group_ids[asset_sha256] = group_id
    if set(asset_group_ids) != candidate_assets:
        raise RuntimeError("Materialized similarity groups do not cover candidates")
    if len(snapshots) != 1:
        raise RuntimeError("Materialized similarity groups use mixed review snapshots")
    return asset_group_ids, {
        "group_count": len(members_by_group),
        "singleton_group_count": sum(len(members) == 1 for members in members_by_group.values()),
        "multi_asset_group_count": sum(len(members) > 1 for members in members_by_group.values()),
        "edge_review_snapshot_sha256": next(iter(snapshots)),
    }


def _duplicate_reasons(edge: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    evidence_type = edge["evidence_type"]
    if evidence_type == "exact_audio":
        return ["identical_standardized_audio_bytes"]
    if evidence_type == "exact_text":
        return ["identical_nonempty_normalized_text"]
    reasons = [f"{evidence_type}_score_at_or_above_calibrated_threshold"]
    pair = evidence.get("pair", {})
    reasons.extend(f"risk:{flag}" for flag in pair.get("risk_flags", []))
    return reasons


def _analysis_csv(report: dict[str, Any]) -> str:
    rows = []
    for edge in report["duplicate_edges"]:
        rows.append(
            {
                "record_type": "duplicate_edge",
                "evidence_type": edge["evidence_type"],
                "review_status": edge["review_status"],
                "included_in_group": edge["included_in_group"],
                "left_asset_sha256": edge["left"]["asset_sha256"],
                "left_fid": edge["left"]["fid"],
                "left_text": edge["left"]["text"],
                "left_audio_path": edge["left"]["audio_absolute_path"],
                "right_asset_sha256": edge["right"]["asset_sha256"],
                "right_fid": edge["right"]["fid"],
                "right_text": edge["right"]["text"],
                "right_audio_path": edge["right"]["audio_absolute_path"],
                "score": edge["score"],
                "threshold": edge["threshold"],
                "margin": edge["margin"],
                "reasons": "|".join(edge["reasons"]),
                "neighbor_evidence_json": "",
            }
        )
    for outlier in report["speaker_outliers"]:
        asset = outlier["asset"]
        rows.append(
            {
                "record_type": "speaker_outlier",
                "evidence_type": "speaker_embedding",
                "review_status": outlier["review_status"],
                "included_in_group": False,
                "left_asset_sha256": asset["asset_sha256"],
                "left_fid": asset["fid"],
                "left_text": asset["text"],
                "left_audio_path": asset["audio_absolute_path"],
                "score": outlier["outlier_score"],
                "threshold": "",
                "margin": "",
                "reasons": "|".join(outlier["reasons"]),
                "neighbor_evidence_json": canonical_json(outlier["neighbors"]),
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _analysis_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    if report["status"] == "reviewed":
        accepted_count = sum(
            edge["review_status"] in ("accepted", "accepted_exact")
            for edge in report["duplicate_edges"]
        )
        notice_class = "notice reviewed"
        notice = (
            "人工复核已完成："
            f"{accepted_count} 组重复已确认，"
            f"{summary['speaker_exclusion_count']} 条 speaker 样本已排除；"
            "每张卡片已明确标注保留或排除。"
        )
    else:
        notice_class = "notice pending"
        notice = (
            "这是只读分析报告；黄色 pending_review 表示尚未保存人工结论，"
            "不能据此自动删样本。"
        )
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    duplicate_cards = "".join(_duplicate_card(edge) for edge in report["duplicate_edges"])
    if not duplicate_cards:
        duplicate_cards = '<p class="empty">没有达到阈值的 duplicate edge。</p>'
    speaker_cards = "".join(_speaker_card(item) for item in report["speaker_outliers"])
    if not speaker_cards:
        speaker_cards = '<p class="empty">没有 speaker outlier 候选。</p>'
    provenance = html.escape(
        json.dumps(report["provenance"], ensure_ascii=False, indent=2, sort_keys=True)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fuxuan dataset analysis review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", system-ui, sans-serif; }}
    body {{ max-width: 1280px; margin: 0 auto; padding: 28px; background: #f6f7fb;
      color: #172033; }}
    h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 32px; }}
    .notice {{ padding: 12px 16px; border-left: 4px solid; font-weight: 700; }}
    .notice.pending {{ border-color: #d97706; background: #fff7ed; color: #7a4b00; }}
    .notice.reviewed {{ border-color: #15803d; background: #f0fdf4; color: #166534; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 8px 10px; border: 1px solid #dbe0ea; text-align: left; }}
    th {{ width: 280px; background: #eef1f7; }}
    .card {{ margin: 14px 0; padding: 18px; border: 1px solid #dbe0ea;
      border-radius: 12px; background: white; }}
    .badge {{ display: inline-block; padding: 3px 9px; border-radius: 999px;
      background: #fff0c2; color: #7a4b00; font-weight: 700; }}
    .badge.accepted, .badge.confirmed_same_speaker {{ background: #dcfce7; color: #166534; }}
    .badge.exclude_uncertain, .badge.exclude_wrong_speaker {{
      background: #fee2e2; color: #991b1b; }}
    .score {{ font-variant-numeric: tabular-nums; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
    .sample {{ padding: 12px; border-radius: 9px; background: #f8fafc; min-width: 0; }}
    .sample.keep {{ border: 2px solid #22c55e; background: #f0fdf4; }}
    .sample.exclude {{ border: 2px solid #ef4444; background: #fef2f2; opacity: .88; }}
    .sample-decision {{ margin-bottom: 8px; font-weight: 800; }}
    .sample-decision.keep {{ color: #166534; }}
    .sample-decision.exclude {{ color: #991b1b; }}
    .decision {{ padding: 10px 12px; border-radius: 8px; font-weight: 700; }}
    .decision.keep {{ background: #f0fdf4; color: #166534; }}
    .decision.exclude {{ background: #fef2f2; color: #991b1b; }}
    audio {{ width: 100%; margin: 8px 0; }} code {{ overflow-wrap: anywhere; }}
    .text {{ min-height: 3em; white-space: pre-wrap; }}
    .reason {{ color: #4b5563; }} .empty {{ color: #64748b; }}
    details {{ margin-top: 10px; }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    @media (max-width: 760px) {{
      .pair {{ grid-template-columns: 1fr; }} body {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <h1>fuxuan 数据集分析复核</h1>
  <p class="{notice_class}">{html.escape(notice)}</p>
  <h2>汇总</h2><table><tbody>{summary_rows}</tbody></table>
  <h2>Exact / Near duplicate</h2>{duplicate_cards}
  <h2>Speaker outlier</h2>{speaker_cards}
  <details><summary>Provenance</summary><pre>{provenance}</pre></details>
</body>
</html>
"""


def _duplicate_card(edge: dict[str, Any]) -> str:
    reasons = ", ".join(edge["reasons"])
    score = f'{edge["score"]:.6f}'
    threshold = "n/a" if edge["threshold"] is None else f'{edge["threshold"]:.6f}'
    decision = _duplicate_decision(edge)
    decision_html = ""
    left_state = None
    right_state = None
    if decision is not None:
        representative, excluded, note = decision
        decision_html = (
            '<p class="decision keep">人工结论：确认重复。'
            f'{html.escape(note)}</p>'
        )
        left_state = (
            ("保留此条", "keep")
            if edge["left"]["asset_sha256"] == representative
            else ("排除此条（重复/截短）", "exclude")
        )
        right_state = (
            ("保留此条", "keep")
            if edge["right"]["asset_sha256"] == representative
            else ("排除此条（重复/截短）", "exclude")
        )
        if set(excluded) != {
            sample["asset_sha256"]
            for sample, state in (
                (edge["left"], left_state),
                (edge["right"], right_state),
            )
            if state[1] == "exclude"
        }:
            raise RuntimeError("Duplicate review display decision is inconsistent")
    return (
        '<article class="card">'
        f'<span class="badge {html.escape(edge["review_status"])}">'
        f'{html.escape(edge["review_status"])}</span> '
        f'<strong>{html.escape(edge["evidence_type"])}</strong> '
        f'<span class="score">score {score} / threshold {threshold}</span>'
        f'<p class="reason">{html.escape(reasons)}</p>{decision_html}'
        f'<div class="pair">{_sample_html(edge["left"], left_state)}'
        f'{_sample_html(edge["right"], right_state)}</div>'
        "</article>"
    )


def _speaker_card(outlier: dict[str, Any]) -> str:
    neighbor_html = "".join(
        f'<div class="sample"><strong>cosine {neighbor["cosine"]:.6f}</strong>'
        f'{_sample_html(neighbor)}</div>'
        for neighbor in outlier["neighbors"]
    )
    reasons = ", ".join(outlier["reasons"])
    review_status = outlier["review_status"]
    review = outlier.get("review") or {}
    review_note = review.get("review_note", "")
    excluded = review_status in ("exclude_wrong_speaker", "exclude_uncertain")
    if review_status == "confirmed_same_speaker":
        sample_state = ("保留此条（已确认同一 speaker）", "keep")
        decision_html = (
            '<p class="decision keep">人工结论：保留，确认为同一 speaker。'
            f'{html.escape(review_note)}</p>'
        )
    elif excluded:
        sample_state = ("排除此条", "exclude")
        decision_html = (
            '<p class="decision exclude">人工结论：从冻结数据集排除。'
            f'{html.escape(review_note)}</p>'
        )
    else:
        sample_state = None
        decision_html = ""
    return (
        '<article class="card">'
        f'<span class="badge {html.escape(review_status)}">'
        f'{html.escape(review_status)}</span> '
        f'<strong>rank {outlier["outlier_rank"]}</strong> '
        f'<span class="score">center {outlier["center_cosine"]:.6f} / '
        f'{outlier["center_threshold"]:.6f}; kNN {outlier["knn_cosine"]:.6f} / '
        f'{outlier["knn_threshold"]:.6f}</span>'
        f'<p class="reason">{html.escape(reasons)}</p>{decision_html}'
        f'{_sample_html(outlier["asset"], sample_state)}'
        f'<details><summary>回听 5 个最近邻</summary>{neighbor_html}</details>'
        "</article>"
    )


def _sample_html(
    sample: dict[str, Any], decision: tuple[str, str] | None = None
) -> str:
    audio = '<audio controls preload="none" src="{}"></audio>'.format(
        html.escape(sample["audio_url"], quote=True)
    )
    sample_class = "sample"
    decision_html = ""
    if decision is not None:
        label, state = decision
        sample_class += f" {state}"
        decision_html = (
            f'<div class="sample-decision {state}">{html.escape(label)}</div>'
        )
    return (
        f'<div class="{sample_class}">{decision_html}'
        f'<div><code>{html.escape(str(sample["asset_sha256"]))}</code></div>'
        f'<div>{html.escape(str(sample.get("fid", "")))}</div>'
        f'<div class="text">{html.escape(str(sample.get("text", "")))}</div>'
        f"{audio}"
        f'<div><code>{html.escape(sample["audio_absolute_path"])}</code></div>'
        "</div>"
    )


def _duplicate_decision(
    edge: dict[str, Any],
) -> tuple[str, list[str], str] | None:
    review = edge.get("review")
    if edge.get("review_status") != "accepted" or not isinstance(review, dict):
        return None
    try:
        payload = json.loads(review.get("review_note", ""))
    except (json.JSONDecodeError, TypeError):
        return None
    representative = payload.get("representative_asset_sha256")
    excluded = payload.get("exclude_asset_sha256s")
    note = payload.get("note")
    if (
        not isinstance(representative, str)
        or not isinstance(excluded, list)
        or not all(isinstance(item, str) for item in excluded)
        or not isinstance(note, str)
    ):
        return None
    return representative, excluded, note


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Invalid {label} SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
