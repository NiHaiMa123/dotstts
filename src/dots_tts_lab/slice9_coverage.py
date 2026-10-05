"""Deterministic text and Mandarin phonological coverage features for Slice 9."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CONFIG_PATH,
    load_slice9_selection_config,
)


DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_text_features_v1.json"
)
DEFAULT_SLICE9_COVERAGE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_coverage_features_v1.json"
)
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TONE_RE = re.compile(r"^(.*?)([0-5])$")
_MAJOR_PUNCT = frozenset("。！？!?；;：:")
_MINOR_PUNCT = frozenset("，,、：:")


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


def _dependency_info() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("pypinyin")
    except importlib.metadata.PackageNotFoundError:
        return {
            "provider": "pypinyin",
            "required_version": "0.53.0",
            "installed_version": None,
            "available": False,
            "reason": "dependency_not_installed",
        }
    return {
        "provider": "pypinyin",
        "required_version": "0.53.0",
        "installed_version": version,
        "available": True,
        "reason": None,
    }


def _script_features(text: str) -> dict[str, Any]:
    normalized = unicodedata.normalize("NFKC", text)
    chars = [char for char in normalized if not char.isspace()]
    punctuation = [char for char in chars if unicodedata.category(char).startswith("P")]
    han = [char for char in chars if _HAN_RE.fullmatch(char)]
    latin_digit = [
        char
        for char in chars
        if ("LATIN" in unicodedata.name(char, "") and unicodedata.category(char).startswith("L"))
        or unicodedata.category(char).startswith("N")
    ]
    denominator = max(1, len(chars))
    major = [char for char in punctuation if char in _MAJOR_PUNCT]
    minor = [char for char in punctuation if char in _MINOR_PUNCT]
    return {
        "coverage_view_nfkc": normalized,
        "script_char_count": len(chars),
        "han_char_count": len(han),
        "latin_digit_count": len(latin_digit),
        "latin_digit_ratio": len(latin_digit) / denominator,
        "punctuation_count": len(punctuation),
        "punctuation_ratio": len(punctuation) / denominator,
        "prosody_boundary_count": len(punctuation),
        "prosody_major_boundary_count": len(major),
        "prosody_minor_boundary_count": len(minor),
        "prosody_major_boundary_ratio": len(major) / denominator,
        "prosody_minor_boundary_ratio": len(minor) / denominator,
        "question_exclamation_count": sum(char in "！？!?" for char in punctuation),
        "terminal_boundary_count": sum(char in "。！？!?" for char in punctuation),
    }


def _phonology(text: str, dependency: dict[str, Any]) -> dict[str, Any]:
    han_chars = [char for char in unicodedata.normalize("NFKC", text) if _HAN_RE.fullmatch(char)]
    if not dependency["available"]:
        return {
            "pinyin_syllable_count": None,
            "unique_syllable_type_count": None,
            "initial_type_count": None,
            "final_type_count": None,
            "tone_type_count": None,
            "unresolved_char_count": None,
            "coverage_status": "unavailable_dependency" if han_chars else "unsupported_script",
            "unavailable_reason": dependency["reason"],
        }
    try:
        from pypinyin import Style, lazy_pinyin, pinyin
    except (ImportError, ModuleNotFoundError):
        dependency["available"] = False
        dependency["reason"] = "dependency_import_failed"
        return _phonology(text, dependency)
    syllables: list[tuple[str, str, int]] = []
    unresolved = 0
    try:
        normal = pinyin(text, style=Style.TONE3, heteronym=False, strict=False)
        initials = pinyin(text, style=Style.INITIALS, heteronym=False, strict=False)
        finals = pinyin(text, style=Style.FINALS, heteronym=False, strict=False)
        _ = lazy_pinyin  # Keep the provider API pinned and explicit in provenance.
        for char, normal_row, initial_row, final_row in zip(
            unicodedata.normalize("NFKC", text), normal, initials, finals
        ):
            if not _HAN_RE.fullmatch(char):
                continue
            normal_value = normal_row[0] if normal_row else ""
            initial_value = initial_row[0] if initial_row else ""
            final_value = final_row[0] if final_row else ""
            match = _TONE_RE.match(normal_value.lower())
            if not match or not match.group(1):
                unresolved += 1
                continue
            tone = int(match.group(2))
            syllables.append((initial_value or "Ø", final_value or match.group(1), tone))
    except Exception as error:  # pragma: no cover - provider-specific failures are fail-closed.
        return {
            "pinyin_syllable_count": None,
            "unique_syllable_type_count": None,
            "initial_type_count": None,
            "final_type_count": None,
            "tone_type_count": None,
            "unresolved_char_count": None,
            "coverage_status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    unique = set(syllables)
    status = "ok" if unresolved == 0 else "unresolved"
    return {
        "pinyin_syllable_count": len(syllables),
        "unique_syllable_type_count": len(unique),
        "initial_type_count": len({item[0] for item in syllables}),
        "final_type_count": len({item[1] for item in syllables}),
        "tone_type_count": len({item[2] for item in syllables}),
        "unresolved_char_count": unresolved,
        "coverage_status": status,
        "unavailable_reason": None,
    }


def build_slice9_coverage_features(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    text_feature_path: str | Path = DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_COVERAGE_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    text_report_path = Path(text_feature_path).resolve()
    text_report = _load_json(text_report_path, label="Slice 9 text feature report")
    items = text_report.get("items")
    _require(isinstance(items, list), "Slice 9 text feature items are missing")
    _require(isinstance(text_report.get("candidate_snapshot_sha256"), str), "Text feature report candidate snapshot SHA is missing")
    dependency = _dependency_info()
    output_items: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for item in items:
        text = str(item.get("text_exact", ""))
        _require(text.strip(), f"Empty text for {item.get('asset_sha256')}")
        expected_hash = str(item.get("text_sha256", ""))
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_hash, f"Text SHA drift for {item.get('asset_sha256')}")
        script = _script_features(text)
        phonology = _phonology(text, dict(dependency))
        status = str(phonology["coverage_status"])
        status_counts[status] += 1
        output_items.append(
            {
                "asset_sha256": item["asset_sha256"],
                "fid": item["fid"],
                "text_sha256": expected_hash,
                "confidence_tier": item["confidence_tier"],
                "text_source": item["text_source"],
                "script": script,
                "phonology": phonology,
            }
        )
    result = {
        "schema_version": 1,
        "feature_schema_version": "slice9_text_coverage@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "candidate_snapshot_sha256": text_report.get("candidate_snapshot_sha256"),
        "text_feature_report_sha256": hashlib.sha256(text_report_path.read_bytes()).hexdigest(),
        "candidate_count": len(output_items),
        "algorithm": {
            "id": "unicode_nfkc_plus_pypinyin_tone3",
            "version": "1",
            "normalization": "NFKC; whitespace excluded from counts; punctuation retained as boundaries",
            "polyphone_policy": "pypinyin heteronym=false, strict=false; unresolved is explicit",
            "dependency": dependency,
        },
        "coverage_status_counts": dict(sorted(status_counts.items())),
        "items": sorted(output_items, key=lambda entry: entry["asset_sha256"]),
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}
