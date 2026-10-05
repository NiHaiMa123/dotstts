"""Explainable, pool-scoped Slice 9 composite scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_feature_snapshot import DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH
from dots_tts_lab.slice9_normalization import DEFAULT_SLICE9_NORMALIZED_REPORT_PATH


DEFAULT_SLICE9_RANKING_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_ranking_v1.json"
)


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


def build_slice9_ranking(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    normalized_report_path: str | Path = DEFAULT_SLICE9_NORMALIZED_REPORT_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_RANKING_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    normalized_path = Path(normalized_report_path).resolve()
    snapshot_path = Path(feature_snapshot_path).resolve()
    normalized = _load_json(normalized_path, label="Slice 9 normalized feature report")
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    _require(normalized.get("status") == "succeeded" and snapshot.get("status") == "succeeded", "Scoring inputs did not succeed")
    weights = {name: float(value) for name, value in config["score"]["weights"].items()}
    _require(set(weights) == set(normalized["feature_names"]), "Score feature names drift")
    output_pools: dict[str, Any] = {}
    for pool_id, pool in normalized["pools"].items():
        scored: list[dict[str, Any]] = []
        for item in pool["items"]:
            normalized_features = item["normalized_features"]
            available = [name for name in weights if normalized_features.get(name) is not None]
            available_weight = sum(weights[name] for name in available)
            _require(available_weight > 0.0, f"No available score weight for {pool_id}/{item['asset_sha256']}")
            contributions = {name: (weights[name] * float(normalized_features[name]) if name in available else None) for name in weights}
            composite = sum(value for value in contributions.values() if value is not None) / available_weight
            reasons: list[dict[str, Any]] = []
            if item.get("gate_status") == "review":
                reasons.append({"code": "gate_review", "severity": "review", "flags": item.get("review_flags", [])})
            if item.get("missing_features"):
                reasons.append({"code": "missing_optional_features", "severity": "info", "features": item["missing_features"]})
            scored.append({
                **item,
                "score_version": "slice9_composite@1",
                "component_scores": normalized_features,
                "component_weights": weights,
                "available_components": available,
                "available_weight": available_weight,
                "total_weight": sum(weights.values()),
                "weighted_contributions": contributions,
                "composite_score": composite,
                "score_reasons": reasons,
            })
        scored.sort(key=lambda item: (-float(item["composite_score"]), item["asset_sha256"]))
        for rank, item in enumerate(scored, start=1):
            item["pool_score_rank"] = rank
        output_pools[pool_id] = {
            "candidate_count": len(scored),
            "available_weight_distribution": {
                "min": min(item["available_weight"] for item in scored),
                "max": max(item["available_weight"] for item in scored),
                "mean": sum(item["available_weight"] for item in scored) / len(scored),
            },
            "items": scored,
        }
    result = {
        "schema_version": 1,
        "ranking_report_schema_version": "slice9_pool_composite_ranking@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(Path(config_path).resolve()),
            "feature_snapshot_sha256": _sha256(snapshot_path),
            "normalized_report_sha256": _sha256(normalized_path),
        },
        "score_definition": {
            "method": "weighted_mean_of_available_pool_normalized_components",
            "weights": weights,
            "missing_feature_policy": "renormalize_available_weights",
            "missing_value_is_not_zero": True,
            "rank_key": "pool_id_score_desc_asset_sha256",
            "global_rank": "forbidden",
        },
        "pool_count": len(output_pools),
        "pools": output_pools,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

