"""Deterministic diversity reranking for Slice 9 pools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_feature_snapshot import DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH
from dots_tts_lab.slice9_scoring import DEFAULT_SLICE9_RANKING_REPORT_PATH


DEFAULT_SLICE9_DIVERSITY_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_diversity_rerank_v1.json"
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


def _normalized_text(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def _prosody_vector(item: dict[str, Any]) -> tuple[float, float, float, float]:
    script = item["coverage"]["script"]
    return tuple(float(script.get(key, 0.0)) for key in (
        "punctuation_ratio", "prosody_major_boundary_ratio", "prosody_minor_boundary_ratio", "question_exclamation_count",
    ))


def _prosody_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return math.exp(-distance * 8.0)


def _duration_similarity(left: float, right: float) -> float:
    return math.exp(-abs(left - right) / 0.5)


def build_slice9_diversity_rerank(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    ranking_report_path: str | Path = DEFAULT_SLICE9_RANKING_REPORT_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_DIVERSITY_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    ranking_path = Path(ranking_report_path).resolve()
    snapshot_path = Path(feature_snapshot_path).resolve()
    ranking = _load_json(ranking_path, label="Slice 9 ranking report")
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    _require(ranking.get("status") == "succeeded", "Ranking report did not succeed")
    snapshot_items = {item["asset_sha256"]: item for item in snapshot.get("items", [])}
    output_pools: dict[str, Any] = {}
    for pool_id, pool_definition in config["pools"].items():
        ranked = list(ranking["pools"][pool_id]["items"])
        by_asset = {item["asset_sha256"]: item for item in ranked}
        _require(len(by_asset) == len(ranked), f"Duplicate asset in ranking pool {pool_id}")
        selected: list[dict[str, Any]] = []
        remaining = set(by_asset)
        top_k = int(pool_definition["top_k"])
        while remaining and len(selected) < top_k:
            candidates: list[tuple[float, str, dict[str, Any], dict[str, float]]] = []
            for asset in sorted(remaining):
                item = by_asset[asset]
                base_score = float(item["composite_score"])
                if not selected:
                    max_text = max_duration = max_prosody = 0.0
                else:
                    text = _normalized_text(item["text_exact"])
                    context = snapshot_items[item["asset_sha256"]]
                    duration = float(context["quality"]["duration"]["effective_seconds"])
                    prosody = _prosody_vector(context)
                    similarities = []
                    for previous in selected:
                        similarities.append({
                            "text": _text_similarity(text, _normalized_text(previous["text_exact"])),
                            "duration": _duration_similarity(duration, float(snapshot_items[previous["asset_sha256"]]["quality"]["duration"]["effective_seconds"])),
                            "prosody": _prosody_similarity(prosody, _prosody_vector(snapshot_items[previous["asset_sha256"]])),
                        })
                    max_text = max(value["text"] for value in similarities)
                    max_duration = max(value["duration"] for value in similarities)
                    max_prosody = max(value["prosody"] for value in similarities)
                penalty = 0.15 * max_text + 0.05 * max_duration + 0.05 * max_prosody
                diversity_score = base_score - penalty
                candidates.append((diversity_score, asset, item, {"base_score": base_score, "text_similarity": max_text, "duration_similarity": max_duration, "prosody_similarity": max_prosody, "penalty": penalty}))
            _, asset, item, evidence = min(candidates, key=lambda entry: (-entry[0], entry[1]))
            remaining.remove(asset)
            selected.append({
                **item,
                "diversity_rank": len(selected) + 1,
                "diversity_score": evidence["base_score"] - evidence["penalty"],
                "diversity_evidence": evidence,
            })
        selected_assets = {item["asset_sha256"] for item in selected}
        boundary_count = int(pool_definition["boundary_count"])
        boundary = [item for item in ranked if item["asset_sha256"] not in selected_assets][:boundary_count]
        output_pools[pool_id] = {
            "candidate_count": len(ranked),
            "top_k": top_k,
            "selected_count": len(selected),
            "selected": selected,
            "boundary": boundary,
        }
    result = {
        "schema_version": 1,
        "diversity_report_schema_version": "slice9_deterministic_diversity@1",
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
        "algorithm": {
            "id": "greedy_mmr_text_duration_prosody",
            "version": "1",
            "penalty": {"text_similarity": 0.15, "duration_similarity": 0.05, "prosody_similarity": 0.05},
            "similarity": {"text": "SequenceMatcher(auto junk=false) over NFKC text with whitespace/punctuation removed", "duration": "exp(-abs(delta_seconds)/0.5)", "prosody": "exp(-8*Euclidean distance over punctuation boundary ratios)"},
            "tie_break": "asset_sha256_ascending",
            "scope": "independent per pool; no global rank or cross-pool backfill",
        },
        "pool_count": len(output_pools),
        "pools": output_pools,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}
