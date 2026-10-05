from __future__ import annotations

import difflib
import hashlib
import json
import os
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dots_tts_lab.dataset_freeze import TextSimilarityConfig


DEFAULT_EXACT_AUDIO_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/exact_audio.json"
)
DEFAULT_EXACT_TEXT_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/exact_text.json"
)
DEFAULT_NEAR_TEXT_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/near_text.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_exact_audio(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Rehash standardized bytes and form deterministic exact-audio groups."""
    by_audio_sha256: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_assets: set[str] = set()
    checked_count = 0
    for candidate in candidates:
        asset_sha256 = candidate.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError("Exact-audio candidate has invalid asset_sha256")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate exact-audio candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        audio_path_value = candidate.get("audio_absolute_path")
        if not isinstance(audio_path_value, str) or not audio_path_value:
            raise RuntimeError(f"Missing standardized audio path for {asset_sha256}")
        audio_path = Path(audio_path_value).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Standardized audio missing for exact-audio analysis: {audio_path}"
            )
        expected_sha256 = candidate.get("audio_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(f"Missing standardized audio SHA-256 for {asset_sha256}")
        actual_sha256 = _sha256_file(audio_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Standardized audio SHA-256 drift during exact-audio analysis for "
                f"{asset_sha256}: expected {expected_sha256}, got {actual_sha256}"
            )
        relative_path = candidate.get("audio_relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(f"Missing standardized relative path for {asset_sha256}")
        by_audio_sha256[actual_sha256].append(
            {
                "asset_sha256": asset_sha256,
                "audio_relative_path": relative_path,
            }
        )
        checked_count += 1

    groups = []
    for audio_sha256, members in sorted(by_audio_sha256.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda item: item["asset_sha256"])
        groups.append(
            {
                "group_id": f"exact-audio:{audio_sha256}",
                "evidence_type": "exact_audio",
                "audio_sha256": audio_sha256,
                "member_count": len(members),
                "members": members,
            }
        )
    duplicate_item_count = sum(group["member_count"] for group in groups)
    return {
        "schema_version": 1,
        "status": "succeeded",
        "checked_asset_count": checked_count,
        "unique_audio_sha256_count": len(by_audio_sha256),
        "duplicate_group_count": len(groups),
        "duplicate_item_count": duplicate_item_count,
        "groups": groups,
    }


def write_exact_audio_report(
    report: dict[str, Any],
    path: str | Path = DEFAULT_EXACT_AUDIO_REPORT_PATH,
) -> dict[str, str]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(target),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def normalize_text_for_similarity(
    text: str,
    *,
    config: TextSimilarityConfig,
) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("Text similarity input must be a string")
    unicode_normalized = unicodedata.normalize(config.unicode_form, text)
    casefolded = unicode_normalized.casefold() if config.casefold else unicode_normalized
    removed_characters = []
    kept_characters = []
    removed_prefixes = set(config.remove_unicode_category_prefixes)
    for index, character in enumerate(casefolded):
        category = unicodedata.category(character)
        if category[:1] in removed_prefixes:
            removed_characters.append(
                {
                    "index": index,
                    "character": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "category": category,
                    "name": unicodedata.name(character, "<unnamed>"),
                }
            )
        else:
            kept_characters.append(character)
    normalized = "".join(kept_characters)
    differences = []
    for operation, original_start, original_end, normalized_start, normalized_end in (
        difflib.SequenceMatcher(a=text, b=normalized, autojunk=False).get_opcodes()
    ):
        if operation == "equal":
            continue
        differences.append(
            {
                "operation": operation,
                "original_start": original_start,
                "original_end": original_end,
                "original_text": text[original_start:original_end],
                "normalized_start": normalized_start,
                "normalized_end": normalized_end,
                "normalized_text": normalized[normalized_start:normalized_end],
            }
        )
    return {
        "text_original": text,
        "text_normalized": normalized,
        "normalized_text_sha256": (
            hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if normalized
            else None
        ),
        "changed": normalized != text,
        "unicode_changed": unicode_normalized != text,
        "casefold_changed": casefolded != unicode_normalized,
        "removed_characters": removed_characters,
        "diff": differences,
    }


def analyze_exact_text(
    candidates: Iterable[dict[str, Any]],
    *,
    config: TextSimilarityConfig,
) -> dict[str, Any]:
    """Normalize exact text with an auditable diff and group non-empty matches."""
    by_normalized_sha256: dict[str, list[str]] = defaultdict(list)
    seen_assets: set[str] = set()
    items = []
    empty_items = []
    for candidate in candidates:
        asset_sha256 = candidate.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError("Exact-text candidate has invalid asset_sha256")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate exact-text candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        text = candidate.get("text_exact")
        if not isinstance(text, str):
            raise RuntimeError(f"Missing exact text for {asset_sha256}")
        normalization = normalize_text_for_similarity(text, config=config)
        item = {
            "asset_sha256": asset_sha256,
            "text_source": candidate.get("text_source"),
            **normalization,
        }
        items.append(item)
        normalized_sha256 = normalization["normalized_text_sha256"]
        if normalized_sha256 is None:
            empty_items.append(
                {
                    "asset_sha256": asset_sha256,
                    "reason": "empty_after_normalization",
                    "text_original": text,
                }
            )
            continue
        by_normalized_sha256[normalized_sha256].append(asset_sha256)

    items.sort(key=lambda item: item["asset_sha256"])
    empty_items.sort(key=lambda item: item["asset_sha256"])
    groups = []
    for normalized_sha256, members in sorted(by_normalized_sha256.items()):
        if len(members) < 2:
            continue
        groups.append(
            {
                "group_id": f"exact-text:{normalized_sha256}",
                "evidence_type": "exact_text",
                "normalized_text_sha256": normalized_sha256,
                "member_count": len(members),
                "member_asset_sha256s": sorted(members),
            }
        )
    return {
        "schema_version": 1,
        "status": "succeeded",
        "normalization": {
            "unicode_form": config.unicode_form,
            "casefold": config.casefold,
            "remove_unicode_category_prefixes": list(
                config.remove_unicode_category_prefixes
            ),
        },
        "checked_asset_count": len(items),
        "nonempty_normalized_count": len(items) - len(empty_items),
        "empty_after_normalization_count": len(empty_items),
        "changed_by_normalization_count": sum(item["changed"] for item in items),
        "unique_normalized_text_count": len(by_normalized_sha256),
        "duplicate_group_count": len(groups),
        "duplicate_item_count": sum(group["member_count"] for group in groups),
        "groups": groups,
        "empty_items": empty_items,
        "items": items,
    }


def write_exact_text_report(
    report: dict[str, Any],
    path: str | Path = DEFAULT_EXACT_TEXT_REPORT_PATH,
) -> dict[str, str]:
    return write_exact_audio_report(report, path)


def symmetric_levenshtein_similarity(left: str, right: str) -> dict[str, Any]:
    """Return a direction-independent character edit similarity in [0, 1]."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("Levenshtein similarity inputs must be strings")
    if left == right:
        distance = 0
    elif not left:
        distance = len(right)
    elif not right:
        distance = len(left)
    else:
        if len(left) > len(right):
            left, right = right, left
        previous = list(range(len(left) + 1))
        for right_index, right_character in enumerate(right, start=1):
            current = [right_index]
            for left_index, left_character in enumerate(left, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[left_index] + 1,
                        previous[left_index - 1]
                        + (left_character != right_character),
                    )
                )
            previous = current
        distance = previous[-1]
    maximum_length = max(len(left), len(right))
    similarity = 1.0 if maximum_length == 0 else 1.0 - distance / maximum_length
    return {
        "metric": "symmetric_normalized_levenshtein",
        "edit_distance": distance,
        "similarity": similarity,
    }


def analyze_near_text(
    candidates: Iterable[dict[str, Any]],
    *,
    config: TextSimilarityConfig,
    named_entities: Iterable[str] = (),
    short_text_max_chars: int = 8,
    lexicon_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recall near-text pairs by length and score every pair symmetrically."""
    if short_text_max_chars < 1:
        raise ValueError("short_text_max_chars must be positive")
    normalized_entities = sorted(
        {
            normalized
            for term in named_entities
            if isinstance(term, str)
            and (
                normalized := normalize_text_for_similarity(term, config=config)[
                    "text_normalized"
                ]
            )
        }
    )
    items = []
    skipped_empty_assets = []
    seen_assets: set[str] = set()
    for candidate in candidates:
        asset_sha256 = candidate.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError("Near-text candidate has invalid asset_sha256")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate near-text candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        text = candidate.get("text_exact")
        if not isinstance(text, str):
            raise RuntimeError(f"Missing near-text input for {asset_sha256}")
        normalization = normalize_text_for_similarity(text, config=config)
        normalized = normalization["text_normalized"]
        if not normalized:
            skipped_empty_assets.append(asset_sha256)
            continue
        present_entities = [term for term in normalized_entities if term in normalized]
        items.append(
            {
                "asset_sha256": asset_sha256,
                "text_original": text,
                "text_normalized": normalized,
                "normalized_length": len(normalized),
                "removed_punctuation": any(
                    removed["category"].startswith("P")
                    for removed in normalization["removed_characters"]
                ),
                "named_entities": present_entities,
            }
        )
    items.sort(key=lambda item: (item["normalized_length"], item["asset_sha256"]))

    pairs = []
    exact_pair_count = 0
    length_filtered_pair_count = 0
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            length_ratio = left["normalized_length"] / right["normalized_length"]
            if length_ratio < config.candidate_length_ratio_min:
                length_filtered_pair_count += 1
                continue
            score = symmetric_levenshtein_similarity(
                left["text_normalized"], right["text_normalized"]
            )
            if score["edit_distance"] == 0:
                exact_pair_count += 1
                continue
            risk_flags = []
            if min(left["normalized_length"], right["normalized_length"]) <= (
                short_text_max_chars
            ):
                risk_flags.append("short_text")
            if left["removed_punctuation"] or right["removed_punctuation"]:
                risk_flags.append("punctuation_removed")
            left_entities = set(left["named_entities"])
            right_entities = set(right["named_entities"])
            if left_entities or right_entities:
                risk_flags.append("named_entity_present")
            if left_entities != right_entities:
                risk_flags.append("named_entity_difference")
            output_left = left
            output_right = right
            left_asset = output_left["asset_sha256"]
            right_asset = output_right["asset_sha256"]
            if left_asset > right_asset:
                output_left, output_right = output_right, output_left
                left_asset, right_asset = right_asset, left_asset
                left_entities, right_entities = right_entities, left_entities
            pairs.append(
                {
                    "left_asset_sha256": left_asset,
                    "right_asset_sha256": right_asset,
                    "left_text": output_left["text_original"],
                    "right_text": output_right["text_original"],
                    "left_normalized_text": output_left["text_normalized"],
                    "right_normalized_text": output_right["text_normalized"],
                    "left_named_entities": sorted(left_entities),
                    "right_named_entities": sorted(right_entities),
                    "length_ratio": length_ratio,
                    "edit_distance": score["edit_distance"],
                    "similarity": score["similarity"],
                    "risk_flags": risk_flags,
                }
            )
    pairs.sort(
        key=lambda pair: (
            -pair["similarity"],
            pair["left_asset_sha256"],
            pair["right_asset_sha256"],
        )
    )
    risk_counts = Counter(flag for pair in pairs for flag in pair["risk_flags"])
    return {
        "schema_version": 1,
        "status": "succeeded",
        "metric": "symmetric_normalized_levenshtein",
        "candidate_recall": {
            "method": "all_pairs_with_min_normalized_length_ratio",
            "minimum_length_ratio": config.candidate_length_ratio_min,
        },
        "threshold": config.near_similarity_threshold,
        "calibration_required": config.calibration_required,
        "short_text_max_chars": short_text_max_chars,
        "lexicon_provenance": lexicon_provenance,
        "normalized_named_entities": normalized_entities,
        "input_asset_count": len(seen_assets),
        "scored_asset_count": len(items),
        "skipped_empty_asset_count": len(skipped_empty_assets),
        "skipped_empty_asset_sha256s": sorted(skipped_empty_assets),
        "all_possible_pair_count": len(items) * (len(items) - 1) // 2,
        "length_filtered_pair_count": length_filtered_pair_count,
        "exact_pair_count": exact_pair_count,
        "near_candidate_pair_count": len(pairs),
        "risk_flag_counts": dict(sorted(risk_counts.items())),
        "pairs": pairs,
    }


def write_near_text_report(
    report: dict[str, Any],
    path: str | Path = DEFAULT_NEAR_TEXT_REPORT_PATH,
) -> dict[str, str]:
    return write_exact_audio_report(report, path)
