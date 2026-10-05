from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CONFIG_PATH,
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    load_slice9_selection_config,
)


DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_text_features_v1.json"
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


def _text_shape(text: str) -> dict[str, Any]:
    chars = [char for char in text if not char.isspace()]
    han_count = sum(1 for char in chars if "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, ""))
    punctuation_count = sum(1 for char in chars if unicodedata.category(char).startswith("P"))
    latin_digit_count = sum(
        1
        for char in chars
        if (unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, ""))
        or unicodedata.category(char).startswith("N")
    )
    denominator = max(1, len(chars))
    normalized = unicodedata.normalize("NFKC", text)
    return {
        "char_count": len(chars),
        "han_char_count": han_count,
        "punctuation_count": punctuation_count,
        "punctuation_ratio": punctuation_count / denominator,
        "latin_digit_count": latin_digit_count,
        "latin_digit_ratio": latin_digit_count / denominator,
        "coverage_view_nfkc": normalized,
    }


def _load_optional_asr(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    resolved = Path(path).resolve()
    report = _load_json(resolved, label="ASR evaluation report")
    _require(report.get("summary", {}).get("status") == "succeeded", "ASR report did not succeed")
    assets = report.get("assets")
    _require(isinstance(assets, list), "ASR report assets are missing")
    by_asset: dict[str, Any] = {}
    for item in assets:
        _require(isinstance(item, dict) and isinstance(item.get("asset_sha256"), str), "Invalid ASR asset")
        _require(item["asset_sha256"] not in by_asset, f"Duplicate ASR asset: {item['asset_sha256']}")
        by_asset[item["asset_sha256"]] = item
    return {"summary": report["summary"], "assets": by_asset}, hashlib.sha256(resolved.read_bytes()).hexdigest()


def build_slice9_text_features(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    snapshot_path: str | Path = DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    asr_report_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    snapshot = _load_json(Path(snapshot_path).resolve(), label="Slice 9 candidate snapshot")
    candidates = snapshot.get("items")
    _require(isinstance(candidates, list) and snapshot.get("item_count") == len(candidates), "Invalid Slice 9 candidate snapshot")
    asr, asr_sha256 = _load_optional_asr(asr_report_path)
    output_items: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    asr_count = 0
    for candidate in candidates:
        asset_sha256 = str(candidate["asset_sha256"])
        text = str(candidate["text_exact"])
        _require(text.strip(), f"Empty text for {asset_sha256}")
        computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        _require(computed_hash == candidate["text_sha256"], f"Text SHA drift for {asset_sha256}")
        source = str(candidate["text_source"])
        if source == "human_review":
            tier = "A"
            provenance = "approved_human_review"
            _require(candidate.get("review_decision_id"), f"Human review decision missing for {asset_sha256}")
            _require(candidate.get("review_round") is not None and candidate.get("review_batch_id"), f"Human review lineage missing for {asset_sha256}")
        elif source == "filename_candidate_unreviewed":
            tier = "B"
            provenance = "filename_candidate_unreviewed"
            _require(candidate.get("review_decision_id") is None, f"Unreviewed text has review decision for {asset_sha256}")
            _require(candidate.get("review_round") is None and candidate.get("review_batch_id") is None, f"Unreviewed text has review lineage for {asset_sha256}")
        else:
            raise RuntimeError(f"Unknown text source for {asset_sha256}: {source}")
        tier_counts[tier] += 1
        shape = _text_shape(text)
        asr_item = asr["assets"].get(asset_sha256) if asr else None
        asr_evidence: dict[str, Any] = {
            "available": False,
            "backend_id": asr["summary"].get("backend_id") if asr else None,
            "report_sha256": asr_sha256,
            "cer": None,
            "edit_distance": None,
            "agreement": None,
            "status": "unknown",
        }
        if asr_item is not None and asr_item.get("reference_exact") == text and asr_item.get("status") == "ok":
            cer = float(asr_item["cer"])
            asr_evidence.update(
                {
                    "available": True,
                    "cer": cer,
                    "edit_distance": int(asr_item["edit_distance"]),
                    "agreement": max(0.0, min(1.0, 1.0 - cer)),
                    "status": "ok",
                }
            )
            asr_count += 1
        elif asr_item is not None:
            asr_evidence["status"] = "reference_mismatch"
        output_items.append(
            {
                "asset_sha256": asset_sha256,
                "fid": candidate["fid"],
                "text_exact": text,
                "text_sha256": candidate["text_sha256"],
                "text_source": source,
                "confidence_tier": tier,
                "provenance": provenance,
                "label_source": candidate["label_source"],
                "review_decision_id": candidate.get("review_decision_id"),
                "review_round": candidate.get("review_round"),
                "review_batch_id": candidate.get("review_batch_id"),
                "shape": shape,
                "asr": asr_evidence,
                "decision": "pass",
                "reasons": [],
            }
        )
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "candidate_snapshot_sha256": hashlib.sha256(Path(snapshot_path).resolve().read_bytes()).hexdigest(),
        "candidate_count": len(output_items),
        "confidence_tier_counts": dict(sorted(tier_counts.items())),
        "asr_report_sha256": asr_sha256,
        "asr_available_count": asr_count,
        "items": sorted(output_items, key=lambda item: item["asset_sha256"]),
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

