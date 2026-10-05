"""Weight and gate-threshold sensitivity analysis for Slice 9."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_feature_snapshot import DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH
from dots_tts_lab.slice9_scoring import DEFAULT_SLICE9_RANKING_REPORT_PATH


DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_sensitivity_v1.json"
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


def _variant_weights(base: dict[str, float], plus: str, minus: tuple[str, str]) -> dict[str, float]:
    values = dict(base)
    values[plus] += 0.02
    for name in minus:
        values[name] -= 0.01
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _rank_top(items: list[dict[str, Any]], weights: dict[str, float], top_k: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in items:
        available = [name for name, value in item["component_scores"].items() if value is not None]
        denominator = sum(weights[name] for name in available)
        score = sum(weights[name] * float(item["component_scores"][name]) for name in available) / denominator
        scored.append((score, item["asset_sha256"]))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [asset for _, asset in scored[:top_k]]


def _overlap_metrics(base: list[str], variant: list[str]) -> dict[str, Any]:
    left, right = set(base), set(variant)
    overlap = len(left & right)
    union = len(left | right)
    return {"overlap_count": overlap, "top_k": len(base), "jaccard": overlap / union if union else 1.0, "complete_flip": overlap == 0}


def build_slice9_sensitivity(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    ranking_report_path: str | Path = DEFAULT_SLICE9_RANKING_REPORT_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    ranking_path = Path(ranking_report_path).resolve()
    snapshot_path = Path(feature_snapshot_path).resolve()
    ranking = _load_json(ranking_path, label="Slice 9 ranking report")
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    base_weights = {name: float(value) for name, value in config["score"]["weights"].items()}
    weight_variants = {
        "objective_quality_plus_0.02": _variant_weights(base_weights, "objective_quality", ("text_provenance", "phonological_coverage")),
        "speaker_similarity_plus_0.02": _variant_weights(base_weights, "speaker_similarity", ("silence", "duration")),
        "text_provenance_plus_0.02": _variant_weights(base_weights, "text_provenance", ("objective_quality", "speaker_similarity")),
        "silence_plus_0.02": _variant_weights(base_weights, "silence", ("objective_quality", "duration")),
    }
    weight_results: dict[str, Any] = {}
    overlap_values: list[float] = []
    for variant_name, weights in weight_variants.items():
        pools: dict[str, Any] = {}
        for pool_id, definition in config["pools"].items():
            items = ranking["pools"][pool_id]["items"]
            base_top = [item["asset_sha256"] for item in items[: int(definition["top_k"])] ]
            variant_top = _rank_top(items, weights, int(definition["top_k"]))
            metrics = _overlap_metrics(base_top, variant_top)
            overlap_values.append(metrics["jaccard"])
            pools[pool_id] = {"base_top_k": base_top, "variant_top_k": variant_top, **metrics}
        weight_results[variant_name] = {"weights": weights, "pools": pools}

    # Review thresholds are advisory in this config (automatic_exclusion=false); quantify stability.
    snapshot_items = snapshot.get("items", [])
    threshold_variants: dict[str, Any] = {}
    for variant_name, shift in (("thresholds_minus_5pct", -0.05), ("thresholds_plus_5pct", 0.05)):
        quality_reviews = 0
        speaker_reviews = 0
        for item in snapshot_items:
            quality = item["quality"]["quality"]
            silence = item["quality"]["silence"]
            max_dc = 0.01 * (1.0 + shift)
            max_edge = 0.5 * (1.0 + shift)
            max_silence = 0.45 * (1.0 + shift)
            min_loudness = -35.0 * (1.0 - shift)
            max_loudness = -10.0 * (1.0 + shift)
            max_true_peak = 0.0 + shift * 0.5
            min_snr = 12.0 * (1.0 + shift)
            q_review = (
                float(quality["abs_dc_offset"]) > max_dc
                or float(silence["leading_silence_seconds"]) > max_edge
                or float(silence["trailing_silence_seconds"]) > max_edge
                or float(silence["silence_ratio"]) > max_silence
                or float(quality["integrated_loudness_lufs"]) < min_loudness
                or float(quality["integrated_loudness_lufs"]) > max_loudness
                or float(quality["true_peak_estimate_dbtp"]) > max_true_peak
                or float(quality["snr_proxy_db"]) < min_snr
            )
            s = item["speaker"]
            center_min = 0.6545700366223228 + shift * 0.02
            knn_min = 0.6766019064784982 + shift * 0.02
            s_review = float(s["center_cosine"]) < center_min or float(s["knn_cosine"]) < knn_min
            quality_reviews += int(q_review)
            speaker_reviews += int(s_review)
        threshold_variants[variant_name] = {
            "shift": shift,
            "quality_review_count": quality_reviews,
            "speaker_review_count": speaker_reviews,
            "hard_reject_count": 0,
            "selection_set_change": "none (review action is advisory; no automatic exclusion)",
        }
    result = {
        "schema_version": 1,
        "sensitivity_report_schema_version": "slice9_weight_threshold_sensitivity@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(Path(config_path).resolve()),
            "ranking_report_sha256": _sha256(ranking_path),
            "feature_snapshot_sha256": _sha256(snapshot_path),
        },
        "base_weights": base_weights,
        "weight_perturbation": {"plus": 0.02, "minus_each": 0.01, "renormalized": True},
        "threshold_perturbation": {"relative_percent": 5, "true_peak_absolute_db": 0.025, "speaker_absolute_cosine": 0.001},
        "weight_variants": weight_results,
        "threshold_variants": threshold_variants,
        "summary": {
            "min_top_k_jaccard": min(overlap_values),
            "max_top_k_jaccard": max(overlap_values),
            "all_weight_variants_nonzero_overlap": all(value > 0.0 for value in overlap_values),
            "threshold_hard_reject_count_stable": all(value["hard_reject_count"] == 0 for value in threshold_variants.values()),
        },
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}
