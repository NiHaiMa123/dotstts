"""Build the isolated Slice 9 reference candidate pools."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_feature_snapshot import DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH


DEFAULT_SLICE9_POOL_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_pool_candidates_v1.json"
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


def _reason_codes(reasons: Any) -> list[str]:
    if not isinstance(reasons, list):
        return []
    codes: list[str] = []
    for reason in reasons:
        if isinstance(reason, dict) and isinstance(reason.get("code"), str):
            codes.append(reason["code"])
        elif isinstance(reason, str):
            codes.append(reason)
    return codes


def build_slice9_pool_candidates(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_POOL_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    snapshot_path = Path(feature_snapshot_path).resolve()
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    _require(snapshot.get("status") == "succeeded", "Feature snapshot did not succeed")
    items = snapshot.get("items")
    _require(isinstance(items, list), "Feature snapshot items are missing")
    snapshot_sha = _sha256(snapshot_path)
    pool_config = config.get("pools")
    _require(isinstance(pool_config, dict) and pool_config, "Slice 9 pools are missing")
    pool_items: dict[str, list[dict[str, Any]]] = {pool_id: [] for pool_id in pool_config}
    gate_counts: Counter[str] = Counter()
    for item in items:
        quality_decision = str(item["quality"].get("decision"))
        speaker_decision = str(item["speaker"].get("decision"))
        hard_reasons: list[str] = []
        if quality_decision == "reject":
            hard_reasons.append("quality_reject")
        if speaker_decision == "reject":
            hard_reasons.append("speaker_reject")
        gate_status = "reject" if hard_reasons else ("review" if quality_decision == "review" or speaker_decision == "review" else "pass")
        gate_counts[gate_status] += 1
        base = {
            "asset_sha256": item["asset_sha256"],
            "fid": item["fid"],
            "audio_relative_path": item["audio_relative_path"],
            "audio_sha256": item["audio_sha256"],
            "speaker_id": item["speaker_id"],
            "emotion_primary": item["emotion_primary"],
            "text_sha256": item["text_sha256"],
            "text_exact": item["text_exact"],
            "text_source": item["text_source"],
            "confidence_tier": item["text"]["confidence_tier"],
            "gate_status": gate_status,
            "hard_gate_reasons": hard_reasons,
            "review_flags": sorted(set(_reason_codes(item["quality"].get("reasons"))) | set(_reason_codes(item["speaker"].get("reasons")))),
        }
        if gate_status == "reject":
            continue
        for pool_id, definition in pool_config.items():
            if item["emotion_primary"] == definition["emotion_primary"]:
                pool_items[pool_id].append(dict(base, pool_id=pool_id))
    pools: dict[str, Any] = {}
    shortfalls: list[str] = []
    all_emotion_assets: dict[str, set[str]] = {}
    for pool_id, definition in pool_config.items():
        entries = sorted(pool_items[pool_id], key=lambda entry: entry["asset_sha256"])
        count = len(entries)
        minimum = int(definition["minimum_publish_count"])
        if count < minimum:
            shortfalls.append(pool_id)
        if pool_id.startswith("emotion_"):
            all_emotion_assets[pool_id] = {entry["asset_sha256"] for entry in entries}
        pools[pool_id] = {
            "emotion_primary": definition["emotion_primary"],
            "purpose": definition["purpose"],
            "top_k": int(definition["top_k"]),
            "minimum_publish_count": minimum,
            "boundary_count": int(definition["boundary_count"]),
            "candidate_count": count,
            "meets_minimum": count >= minimum,
            "items": entries,
        }
    emotion_ids = sorted(all_emotion_assets)
    overlap_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(emotion_ids):
        for right in emotion_ids[index + 1 :]:
            overlap = sorted(all_emotion_assets[left] & all_emotion_assets[right])
            if overlap:
                overlap_pairs.append({"left": left, "right": right, "overlap_count": len(overlap), "assets": overlap})
    _require(not overlap_pairs, "Emotion pool asset overlap is forbidden")
    result = {
        "schema_version": 1,
        "pool_report_schema_version": "slice9_pool_candidates@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "feature_snapshot_sha256": snapshot_sha,
        "candidate_count": len(items),
        "hard_gate_decision_counts": dict(sorted(gate_counts.items())),
        "shortfall_pool_ids": shortfalls,
        "pool_count": len(pools),
        "pools": pools,
        "ranking_scope": {
            "global_rank": "forbidden",
            "cross_pool_backfill": "forbidden",
            "cross_pool_normalization": "forbidden",
            "emotion_pool_asset_overlap": "forbidden",
            "neutral_overlap_enforced_at_selection": int(config["ranking"]["neutral_pool_overlap_max"]),
        },
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}
