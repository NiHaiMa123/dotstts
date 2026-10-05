from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    build_candidate_snapshot,
    canonical_dataset_config,
    load_dataset_freeze_config,
    resolve_dataset_freeze_candidates,
)


DEFAULT_DATASET_AUDIT_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/dataset_audit.json"
)
DEFAULT_CANDIDATE_SNAPSHOT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/candidate_snapshot.json"
)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _load_quality_report(path: Path, *, config: Any) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read quality report {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Quality report must be a schema_version 1 object")
    summary = payload.get("summary")
    assets = payload.get("assets")
    if not isinstance(summary, dict) or not isinstance(assets, list):
        raise RuntimeError("Quality report must contain summary and assets")
    _require_equal("quality status", summary.get("status"), "succeeded")
    _require_equal("quality analysis id", summary.get("analysis_id"), config.inputs.quality_analysis_id)
    _require_equal(
        "quality analysis version",
        summary.get("analysis_version"),
        config.inputs.quality_analysis_version,
    )
    _require_equal("quality policy id", summary.get("policy_id"), config.inputs.quality_policy_id)
    _require_equal(
        "quality policy version",
        summary.get("policy_version"),
        config.inputs.quality_policy_version,
    )
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Quality report summary is missing run_id")
    by_asset: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            raise RuntimeError(f"Quality report asset {index} is not an object")
        asset_sha256 = item.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError(f"Quality report asset {index} has invalid asset_sha256")
        if asset_sha256 in by_asset:
            raise RuntimeError(f"Quality report contains duplicate asset {asset_sha256}")
        by_asset[asset_sha256] = item
    return {"summary": summary, "assets_by_sha256": by_asset}


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


def run_dataset_audit(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    config_path: str | Path = DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    standardized_root: str | Path = "data/work/standardized",
    quality_report_path: str | Path | None = None,
    candidate_snapshot_path: str | Path | None = DEFAULT_CANDIDATE_SNAPSHOT_PATH,
    report_path: str | Path = DEFAULT_DATASET_AUDIT_REPORT_PATH,
) -> dict[str, Any]:
    resolved_catalog_path = Path(catalog_path).resolve()
    resolved_config_path = Path(config_path).resolve()
    config = load_dataset_freeze_config(resolved_config_path)
    config_identity = canonical_dataset_config(config)
    catalog = Catalog(resolved_catalog_path)

    registered = catalog.load_dataset_build_config(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
    )
    if registered is None:
        raise RuntimeError(
            f"Dataset build config is not registered: {config.dataset_id}@{config.dataset_version}"
        )
    _require_equal("registered config SHA-256", registered["config_sha256"], config_identity["sha256"])
    _require_equal("registered config JSON", registered["config_json"], config_identity["canonical_json"])
    _require_equal(
        "registered implementation version",
        int(registered["implementation_version"]),
        config.implementation_version,
    )

    resolved_quality_path = Path(
        quality_report_path or config.inputs.quality_report_path
    ).resolve()
    quality = _load_quality_report(resolved_quality_path, config=config)
    quality_summary = quality["summary"]
    quality_run_id = str(quality_summary["run_id"])
    rows = catalog.load_dataset_freeze_candidates(
        standardization_config_id=config.inputs.standardization_config_id,
        standardization_config_version=config.inputs.standardization_config_version,
        quality_run_id=quality_run_id,
        review_benchmark_id=config.inputs.review_benchmark_id,
        review_benchmark_version=config.inputs.review_benchmark_version,
    )
    row_assets = {str(row["asset_sha256"]) for row in rows}
    quality_assets = set(quality["assets_by_sha256"])
    if row_assets != quality_assets:
        raise RuntimeError(
            "Quality report and provenance-pinned candidate assets differ: "
            f"missing_from_candidates={len(quality_assets - row_assets)}, "
            f"missing_from_report={len(row_assets - quality_assets)}"
        )
    for row in rows:
        asset_sha256 = str(row["asset_sha256"])
        report_item = quality["assets_by_sha256"][asset_sha256]
        _require_equal(
            f"quality decision for {asset_sha256}",
            row["quality_decision"],
            report_item.get("decision"),
        )
        _require_equal(
            f"quality reasons for {asset_sha256}",
            json.loads(str(row["quality_reasons_json"])),
            report_item.get("reasons"),
        )

    resolved = resolve_dataset_freeze_candidates(
        rows,
        config=config,
        standardized_root=standardized_root,
    )
    eligible = resolved["eligible"]
    excluded = resolved["excluded"]
    snapshot = build_candidate_snapshot(eligible)
    resolved_snapshot_path = (
        Path(candidate_snapshot_path).resolve() if candidate_snapshot_path is not None else None
    )
    if resolved_snapshot_path is not None:
        if not resolved_snapshot_path.is_file():
            raise FileNotFoundError(
                f"Frozen candidate snapshot does not exist: {resolved_snapshot_path}"
            )
        frozen_snapshot = resolved_snapshot_path.read_bytes()
        if frozen_snapshot != snapshot["canonical_json"].encode("utf-8"):
            raise RuntimeError(
                "Frozen candidate snapshot differs from current resolved candidates: "
                f"expected {snapshot['sha256']}"
            )

    coverage = catalog.load_dataset_freeze_coverage(
        standardization_config_id=config.inputs.standardization_config_id,
        standardization_config_version=config.inputs.standardization_config_version,
        quality_run_id=quality_run_id,
    )
    coverage.update(
        {
            "quality_report_asset_count": len(quality_assets),
            "candidate_count": len(rows),
            "eligible_count": len(eligible),
            "excluded_count": len(excluded),
        }
    )
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "candidate_snapshot_sha256": snapshot["sha256"],
        "total_duration_seconds": snapshot["payload"]["total_duration_seconds"],
        "coverage": coverage,
        "counts": {
            "quality_decision": dict(sorted(Counter(item["quality_decision"] for item in eligible).items())),
            "text_source": dict(sorted(Counter(item["text_source"] for item in eligible).items())),
            "label_source": dict(sorted(Counter(item["label_source"] for item in eligible).items())),
            "emotion_primary": dict(sorted(Counter(item["emotion_primary"] for item in eligible).items())),
            "exclusion_reason": dict(sorted(Counter(item["reason"] for item in excluded).items())),
        },
        "provenance": {
            "catalog_path": str(resolved_catalog_path),
            "dataset_config_path": str(resolved_config_path),
            "dataset_config_sha256": config_identity["sha256"],
            "dataset_implementation_version": config.implementation_version,
            "standardization": {
                "config_id": config.inputs.standardization_config_id,
                "config_version": config.inputs.standardization_config_version,
                "config_sha256": sorted({item["lineage"]["standardization_config_sha256"] for item in eligible}),
                "implementation_version": sorted({item["lineage"]["standardization_implementation_version"] for item in eligible}),
            },
            "quality": {
                "report_path": str(resolved_quality_path),
                "run_id": quality_run_id,
                "analysis_id": quality_summary["analysis_id"],
                "analysis_version": quality_summary["analysis_version"],
                "analysis_config_sha256": quality_summary["analysis_config_sha256"],
                "implementation_version": quality_summary["implementation_version"],
                "policy_id": quality_summary["policy_id"],
                "policy_version": quality_summary["policy_version"],
                "policy_config_sha256": quality_summary["policy_config_sha256"],
            },
            "review": {
                "benchmark_id": config.inputs.review_benchmark_id,
                "benchmark_version": config.inputs.review_benchmark_version,
                "reviewed_count": sum(item["review_decision_id"] is not None for item in eligible),
                "batch_counts": dict(sorted(Counter(item["review_batch_id"] for item in eligible if item["review_batch_id"] is not None).items())),
            },
            "candidate_snapshot_path": str(resolved_snapshot_path) if resolved_snapshot_path else None,
        },
        "exclusions": excluded,
    }
    resolved_report_path = Path(report_path).resolve()
    _atomic_write_json(resolved_report_path, report)
    return {
        "status": report["status"],
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "candidate_snapshot_sha256": snapshot["sha256"],
        "report_path": str(resolved_report_path),
    }
