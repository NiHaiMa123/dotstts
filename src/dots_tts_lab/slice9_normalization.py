"""Pool-local normalization inputs for Slice 9 ranking."""

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
from dots_tts_lab.slice9_pools import DEFAULT_SLICE9_POOL_REPORT_PATH


DEFAULT_SLICE9_NORMALIZED_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_normalized_features_v1.json"
)
_FEATURE_NAMES = (
    "objective_quality",
    "speaker_similarity",
    "text_provenance",
    "silence",
    "duration",
    "phonological_coverage",
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


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _pool_normalize(values: list[float | None], *, lower: float, upper: float, higher_is_better: bool) -> tuple[list[float | None], dict[str, Any]]:
    available = [value for value in values if value is not None]
    if not available:
        return [None for _ in values], {"available_count": 0, "missing_count": len(values), "q_low": None, "q_high": None, "constant": False}
    q_low = _quantile(available, lower)
    q_high = _quantile(available, upper)
    constant = math.isclose(q_low, q_high, rel_tol=0.0, abs_tol=1e-12)
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        if constant:
            scaled = 1.0
        else:
            scaled = _clip01((min(q_high, max(q_low, value)) - q_low) / (q_high - q_low))
        normalized.append(scaled if higher_is_better else 1.0 - scaled)
    return normalized, {
        "available_count": len(available),
        "missing_count": len(values) - len(available),
        "q_low": q_low,
        "q_high": q_high,
        "constant": constant,
        "higher_is_better": higher_is_better,
    }


def _raw_features(item: dict[str, Any]) -> dict[str, float | None]:
    quality = item["quality"]
    q = quality["quality"]
    silence = quality["silence"]
    duration = quality["duration"]
    speaker = item["speaker"]
    text = item["text"]
    coverage = item["coverage"]
    snr = _finite(q.get("snr_proxy_db"))
    loudness = _finite(q.get("integrated_loudness_lufs"))
    dc = _finite(q.get("abs_dc_offset"))
    peak = _finite(q.get("true_peak_estimate_dbtp"))
    quality_raw = None
    if snr is not None and loudness is not None and dc is not None and peak is not None:
        # Fixed, auditable transforms precede pool-local clipping.
        quality_raw = (
            0.55 * _clip01((snr - 12.0) / 60.0)
            + 0.20 * _clip01((loudness + 35.0) / 25.0)
            + 0.15 * (1.0 - _clip01(dc / 0.01))
            + 0.10 * _clip01((-peak) / 12.0)
        )
    center = _finite(speaker.get("center_cosine"))
    knn = _finite(speaker.get("knn_cosine"))
    speaker_raw = (center + knn) / 2.0 if center is not None and knn is not None else None
    tier = text.get("confidence_tier")
    text_raw = {"A": 1.0, "B": 0.5}.get(str(tier))
    silence_ratio = _finite(silence.get("silence_ratio"))
    leading = _finite(silence.get("leading_silence_seconds"))
    trailing = _finite(silence.get("trailing_silence_seconds"))
    silence_raw = None
    if silence_ratio is not None and leading is not None and trailing is not None:
        silence_raw = 0.70 * silence_ratio + 0.15 * _clip01(leading / 0.5) + 0.15 * _clip01(trailing / 0.5)
    effective = _finite(duration.get("effective_seconds"))
    duration_raw = None
    if effective is not None:
        # Configured hard range is (0, 10]; a 5 s clip is the neutral target for coverage.
        duration_raw = _clip01(1.0 - abs(effective - 5.0) / 5.0)
    phonology = coverage["phonology"]
    unique_types = _finite(phonology.get("unique_syllable_type_count"))
    coverage_raw = unique_types if unique_types is not None else None
    return {
        "objective_quality": quality_raw,
        "speaker_similarity": speaker_raw,
        "text_provenance": text_raw,
        "silence": silence_raw,
        "duration": duration_raw,
        "phonological_coverage": coverage_raw,
    }


def build_slice9_normalized_features(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    pool_report_path: str | Path = DEFAULT_SLICE9_POOL_REPORT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_NORMALIZED_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    snapshot_path = Path(feature_snapshot_path).resolve()
    pool_path = Path(pool_report_path).resolve()
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    pools = _load_json(pool_path, label="Slice 9 pool report")
    _require(snapshot.get("status") == "succeeded" and pools.get("status") == "succeeded", "Input report did not succeed")
    snapshot_items = {item["asset_sha256"]: item for item in snapshot.get("items", [])}
    quantile_config = config["score"]["normalization"]
    lower = float(quantile_config["lower_quantile"])
    upper = float(quantile_config["upper_quantile"])
    direction = config["score"]["direction"]
    output_pools: dict[str, Any] = {}
    for pool_id, pool in pools["pools"].items():
        pool_entries = pool["items"]
        raw_rows = []
        raw_by_feature = {name: [] for name in _FEATURE_NAMES}
        for entry in pool_entries:
            asset = entry["asset_sha256"]
            _require(asset in snapshot_items, f"Pool asset missing from feature snapshot: {asset}")
            row_raw = _raw_features(snapshot_items[asset])
            raw_rows.append((entry, row_raw))
            for name in _FEATURE_NAMES:
                raw_by_feature[name].append(row_raw[name])
        normalized_by_feature: dict[str, list[float | None]] = {}
        stats: dict[str, Any] = {}
        for name in _FEATURE_NAMES:
            normalized_by_feature[name], stats[name] = _pool_normalize(
                raw_by_feature[name],
                lower=lower,
                upper=upper,
                higher_is_better=direction[name] != "lower_is_better",
            )
        normalized_items: list[dict[str, Any]] = []
        for index, (entry, raw) in enumerate(raw_rows):
            normalized_items.append({
                **entry,
                "raw_features": raw,
                "normalized_features": {name: normalized_by_feature[name][index] for name in _FEATURE_NAMES},
                "available_features": [name for name in _FEATURE_NAMES if normalized_by_feature[name][index] is not None],
                "missing_features": [name for name in _FEATURE_NAMES if normalized_by_feature[name][index] is None],
            })
        output_pools[pool_id] = {
            "candidate_count": len(normalized_items),
            "normalization": {"method": quantile_config["method"], "lower_quantile": lower, "upper_quantile": upper, "missing_value_is_not_zero": True},
            "feature_stats": stats,
            "items": normalized_items,
        }
    result = {
        "schema_version": 1,
        "normalized_feature_schema_version": "slice9_pool_normalized_features@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(Path(config_path).resolve()),
            "feature_snapshot_sha256": _sha256(snapshot_path),
            "pool_report_sha256": _sha256(pool_path),
        },
        "candidate_count": snapshot.get("candidate_count"),
        "feature_names": list(_FEATURE_NAMES),
        "direction": direction,
        "pools": output_pools,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

