from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.quality import (
    DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
    DEFAULT_QUALITY_POLICY_PATH,
    QualityAnalysisConfig,
    QualityPolicy,
    analyze_audio,
    assess_metrics,
    load_quality_analysis_config,
    load_quality_policy,
)
from dots_tts_lab.reports import write_quality_reports

QUALITY_IMPLEMENTATION_VERSION = 1

_DISTRIBUTION_METRICS = (
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
    "noise_floor_proxy_dbfs",
    "speech_level_proxy_dbfs",
    "snr_proxy_db",
    "near_peak_sample_ratio",
    "flat_top_run_count",
)


def _managed_raw_path(raw_root: Path, relative_path: str) -> Path:
    candidate = (raw_root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(f"Raw path escapes managed root: {relative_path!r}") from error
    return candidate


def _metric_identity(
    *, asset_sha256: str, config: QualityAnalysisConfig
) -> dict[str, Any]:
    return {
        "asset_sha256": asset_sha256,
        "analysis_id": config.analysis_id,
        "analysis_version": config.analysis_version,
        "analysis_config_sha256": config.config_sha256(),
        "implementation_version": QUALITY_IMPLEMENTATION_VERSION,
    }


def _assessment_record(
    *,
    asset_sha256: str,
    analysis: QualityAnalysisConfig,
    policy: QualityPolicy,
    assessed_at: str,
    assessment: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_metric_identity(asset_sha256=asset_sha256, config=analysis),
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_config_sha256": policy.config_sha256(),
        "assessed_at": assessed_at,
        "decision": assessment["decision"],
        "reasons_json": json.dumps(
            assessment["reasons"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _distributions(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    distributions: dict[str, dict[str, float]] = {}
    for metric in _DISTRIBUTION_METRICS:
        values = np.asarray(
            [
                float(item[metric])
                for item in items
                if item.get(metric) is not None
                and np.isfinite(float(item[metric]))
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        distributions[metric] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return distributions


def run_quality(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    raw_dir: str | Path = "data/raw/sha256",
    analysis_config_path: str | Path = DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
    policy_path: str | Path = DEFAULT_QUALITY_POLICY_PATH,
    report_dir: str | Path = "data/reports/quality",
    force: bool = False,
) -> dict[str, Any]:
    raw_root = Path(raw_dir).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw object directory not found: {raw_root}")
    analysis_path = Path(analysis_config_path).resolve()
    policy_config_path = Path(policy_path).resolve()
    analysis = load_quality_analysis_config(analysis_path)
    policy = load_quality_policy(policy_config_path)

    catalog = Catalog(catalog_path)
    catalog.initialize()
    started_at = utc_now()
    catalog.register_quality_analysis_config(
        analysis_id=analysis.analysis_id,
        analysis_version=analysis.analysis_version,
        config_sha256=analysis.config_sha256(),
        config_json=analysis.canonical_json(),
        source_path=str(analysis_path),
        registered_at=started_at,
    )
    catalog.register_quality_policy(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        config_sha256=policy.config_sha256(),
        config_json=policy.canonical_json(),
        source_path=str(policy_config_path),
        registered_at=started_at,
    )
    candidates = catalog.load_quality_candidates()
    existing_metrics = catalog.load_quality_metrics(
        analysis_id=analysis.analysis_id,
        analysis_version=analysis.analysis_version,
        analysis_config_sha256=analysis.config_sha256(),
        implementation_version=QUALITY_IMPLEMENTATION_VERSION,
    )
    run_id = catalog.begin_quality_run(
        raw_root_path=str(raw_root),
        analysis_id=analysis.analysis_id,
        analysis_version=analysis.analysis_version,
        analysis_config_sha256=analysis.config_sha256(),
        implementation_version=QUALITY_IMPLEMENTATION_VERSION,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_config_sha256=policy.config_sha256(),
        started_at=started_at,
    )

    try:
        metrics_to_store: list[dict[str, Any]] = []
        assessments: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        assessed_at = utc_now()

        for candidate in candidates:
            asset_sha256 = str(candidate["asset_sha256"])
            raw_relative_path = str(candidate["relative_path"])
            cached = existing_metrics.get(asset_sha256)
            if not force and cached is not None and cached["status"] == "ok":
                metric = dict(cached)
                metric_action = "cached"
            else:
                identity = _metric_identity(
                    asset_sha256=asset_sha256,
                    config=analysis,
                )
                try:
                    path = _managed_raw_path(raw_root, raw_relative_path)
                    measured = analyze_audio(
                        path,
                        subtype=candidate.get("subtype"),
                        config=analysis,
                    )
                    metric = {
                        **identity,
                        "analyzed_at": assessed_at,
                        "error_type": None,
                        "error_message": None,
                        **measured,
                    }
                    metric_action = "analyzed"
                except Exception as error:
                    metric = {
                        **identity,
                        "analyzed_at": assessed_at,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                    metric_action = "error"
                metrics_to_store.append(metric)

            assessment = assess_metrics(metric, policy=policy)
            assessment_record = _assessment_record(
                asset_sha256=asset_sha256,
                analysis=analysis,
                policy=policy,
                assessed_at=assessed_at,
                assessment=assessment,
            )
            assessments.append(assessment_record)
            item = {
                "asset_sha256": asset_sha256,
                "raw_relative_path": raw_relative_path,
                "source_relative_path": candidate.get("source_relative_path"),
                "speaker_id": candidate.get("speaker_id"),
                "emotion_weak_label": candidate.get("emotion_weak_label"),
                "transcript_candidate": candidate.get("transcript_candidate"),
                "metric_action": metric_action,
                "decision": assessment["decision"],
                "reasons": assessment["reasons"],
                "reasons_json": assessment_record["reasons_json"],
                "error_type": metric.get("error_type"),
                "error_message": metric.get("error_message"),
            }
            item.update(
                {
                    key: metric.get(key)
                    for key in _DISTRIBUTION_METRICS
                }
            )
            item.update(
                {
                    "sample_rate": metric.get("sample_rate"),
                    "channels": metric.get("channels"),
                    "frames": metric.get("frames"),
                    "near_peak_sample_count": metric.get(
                        "near_peak_sample_count"
                    ),
                    "max_flat_top_run_samples": metric.get(
                        "max_flat_top_run_samples"
                    ),
                }
            )
            items.append(item)

        finished_at = utc_now()
        decisions = Counter(item["decision"] for item in items)
        reason_counts = Counter(
            reason["code"] for item in items for reason in item["reasons"]
        )
        error_count = sum(item["metric_action"] == "error" for item in items)
        summary = {
            "run_id": run_id,
            "status": "completed_with_errors" if error_count else "succeeded",
            "raw_root_path": str(raw_root),
            "analysis_id": analysis.analysis_id,
            "analysis_version": analysis.analysis_version,
            "analysis_config_sha256": analysis.config_sha256(),
            "implementation_version": QUALITY_IMPLEMENTATION_VERSION,
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "policy_config_sha256": policy.config_sha256(),
            "started_at": started_at,
            "finished_at": finished_at,
            "discovered_count": len(items),
            "analyzed_count": sum(
                item["metric_action"] == "analyzed" for item in items
            ),
            "cached_count": sum(
                item["metric_action"] == "cached" for item in items
            ),
            "pass_count": int(decisions["pass"]),
            "review_count": int(decisions["review"]),
            "reject_count": int(decisions["reject"]),
            "error_count": error_count,
            "reason_counts": dict(sorted(reason_counts.items())),
            "metric_distributions": _distributions(items),
        }
        report_paths = write_quality_reports(
            report_dir,
            summary=summary,
            items=items,
        )
        catalog.complete_quality_run(
            run_id=run_id,
            finished_at=finished_at,
            metrics_to_store=metrics_to_store,
            assessments=assessments,
            items=items,
            summary=summary,
        )
        summary["reports"] = report_paths
        summary["catalog_path"] = str(Path(catalog_path).resolve())
        summary["analysis_config_path"] = str(analysis_path)
        summary["policy_path"] = str(policy_config_path)
        return summary
    except BaseException as error:
        catalog.fail_quality_run(
            run_id=run_id,
            finished_at=utc_now(),
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
