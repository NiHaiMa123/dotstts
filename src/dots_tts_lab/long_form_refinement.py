from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_contract import LongFormConfig, load_long_form_config
from dots_tts_lab.long_form_features import analyze_segment_samples, read_source_segment
from dots_tts_lab.long_form_pipeline import _render_training_candidate
from dots_tts_lab.long_form_paths import contained_file, validate_output_path
from dots_tts_lab.long_form_render import BoundaryRepairPolicy, render_repaired_spans


LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION = 2
DEFAULT_REFINEMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "long_form"
    / "refinement_v2.yaml"
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SentenceRefinementPolicy(_StrictFrozenModel):
    semantic_split_gap_seconds: float = Field(ge=0.5, le=5.0)
    retained_internal_gap_seconds: float = Field(ge=0.0, le=1.0)
    boundary_padding_seconds: float = Field(ge=0.0, le=0.5)
    join_silence_seconds: float = Field(ge=0.0, le=0.5)
    minimum_active_speech_seconds: float = Field(ge=0.1, le=10.0)
    minimum_output_seconds: float = Field(ge=0.5, le=15.0)
    minimum_text_characters: int = Field(ge=1, le=100)
    maximum_output_seconds: float = Field(ge=0.5, le=15.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SentenceRefinementPolicy":
        if self.retained_internal_gap_seconds >= self.semantic_split_gap_seconds:
            raise ValueError("retained internal gap must be shorter than semantic split gap")
        if self.minimum_output_seconds > self.maximum_output_seconds:
            raise ValueError("minimum output duration cannot exceed maximum")
        return self


class RefinementScreeningPolicy(_StrictFrozenModel):
    accepted_parent_styles: list[str]
    require_dominant_speaker_cluster: bool
    minimum_asr_mean_word_probability: float = Field(ge=0.0, le=1.0)
    minimum_snr_proxy_db: float = Field(ge=0.0, le=60.0)
    maximum_silence_ratio: float = Field(ge=0.0, le=1.0)
    maximum_near_clip_ratio: float = Field(ge=0.0, le=1.0)
    reject_spatial_risk: bool
    reject_dense_background_proxy: bool
    maximum_review_candidates_per_source: int = Field(ge=1, le=500)


class LongFormRefinementConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    sentence: SentenceRefinementPolicy
    screening: RefinementScreeningPolicy
    boundary_repair: BoundaryRepairPolicy | None = None

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_refinement_config(
    path: str | Path = DEFAULT_REFINEMENT_CONFIG_PATH,
) -> LongFormRefinementConfig:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"refinement configuration must be a YAML mapping: {resolved}")
    return LongFormRefinementConfig.model_validate(payload, strict=True)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_characters(text: str) -> int:
    return len(re.sub(r"[^\w\u3400-\u9fff]", "", text, flags=re.UNICODE))


def split_timestamp_segments(
    timestamp_segments: list[dict[str, Any]], *, semantic_gap_seconds: float
) -> list[list[dict[str, Any]]]:
    """Group timestamp segments into utterances separated by a semantic pause."""
    normalized: list[dict[str, Any]] = []
    previous_end = 0.0
    for raw in timestamp_segments:
        if not isinstance(raw, dict):
            raise ValueError("invalid ASR timestamp segment")
        start = float(raw["start"])
        end = float(raw["end"])
        text = str(raw.get("text") or "").strip()
        if start < 0.0 or end <= start or start + 0.05 < previous_end:
            raise ValueError("invalid or non-monotonic ASR timestamp segment")
        if text:
            normalized.append({**raw, "start": start, "end": end, "text": text})
            previous_end = max(previous_end, end)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in normalized:
        if current and segment["start"] - current[-1]["end"] > semantic_gap_seconds:
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def _group_source_spans(
    group: list[dict[str, Any]],
    *,
    parent_start_frame: int,
    parent_end_frame: int,
    sample_rate: int,
    retained_internal_gap_seconds: float,
    padding_seconds: float,
) -> list[dict[str, int]]:
    blocks: list[tuple[float, float]] = []
    for item in group:
        start = float(item["start"])
        end = float(item["end"])
        if blocks and start - blocks[-1][1] <= retained_internal_gap_seconds:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    spans = []
    for index, (start, end) in enumerate(blocks):
        padded_start = max(0.0, start - padding_seconds)
        padded_end = end + padding_seconds
        if index:
            prior_end = blocks[index - 1][1]
            midpoint = (prior_end + start) / 2.0
            padded_start = max(padded_start, midpoint)
        if index + 1 < len(blocks):
            next_start = blocks[index + 1][0]
            midpoint = (end + next_start) / 2.0
            padded_end = min(padded_end, midpoint)
        source_start = max(
            parent_start_frame,
            parent_start_frame + round(padded_start * sample_rate),
        )
        source_end = min(
            parent_end_frame,
            parent_start_frame + round(padded_end * sample_rate),
        )
        if source_end > source_start:
            spans.append(
                {
                    "source_start_frame": source_start,
                    "source_end_frame": source_end,
                }
            )
    return spans


def build_refinement_units(
    parent_segments: list[dict[str, Any]],
    asr_results: list[dict[str, Any]],
    *,
    config: LongFormRefinementConfig,
    source_sample_rate_hz: int,
) -> list[dict[str, Any]]:
    results_by_id = {
        str(item.get("asset_sha256")): item
        for item in asr_results
        if item.get("status") == "ok"
    }
    units: list[dict[str, Any]] = []
    for parent in parent_segments:
        result = results_by_id.get(str(parent["segment_id"]))
        if result is None:
            continue
        metadata = result.get("metadata") or {}
        groups = split_timestamp_segments(
            metadata.get("timestamp_segments") or [],
            semantic_gap_seconds=config.sentence.semantic_split_gap_seconds,
        )
        for group_index, group in enumerate(groups):
            words = [
                word
                for segment in group
                for word in (segment.get("words") or [])
                if isinstance(word, dict) and str(word.get("text") or "")
            ]
            probabilities = [
                float(word["probability"])
                for word in words
                if word.get("probability") is not None
            ]
            text = "".join(str(segment["text"]).strip() for segment in group).strip()
            active_seconds = sum(
                float(segment["end"]) - float(segment["start"])
                for segment in group
            )
            spans = _group_source_spans(
                group,
                parent_start_frame=int(parent["source_start_frame"]),
                parent_end_frame=int(parent["source_end_frame"]),
                sample_rate=source_sample_rate_hz,
                retained_internal_gap_seconds=config.sentence.retained_internal_gap_seconds,
                padding_seconds=config.sentence.boundary_padding_seconds,
            )
            source_frames = sum(
                span["source_end_frame"] - span["source_start_frame"] for span in spans
            )
            join_frames = round(config.sentence.join_silence_seconds * source_sample_rate_hz)
            estimated_frames = source_frames + max(0, len(spans) - 1) * join_frames
            identity_payload = {
                "implementation_version": LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION,
                "parent_segment_id": parent["segment_id"],
                "refinement_config_sha256": config.config_sha256(),
                "source_spans": spans,
            }
            unit_id = hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            units.append(
                {
                    "unit_id": unit_id,
                    "parent_segment_id": parent["segment_id"],
                    "parent_segment_index": parent.get("source_region_index"),
                    "group_index": group_index,
                    "source_spans": spans,
                    "source_start_frame": spans[0]["source_start_frame"] if spans else 0,
                    "source_end_frame": spans[-1]["source_end_frame"] if spans else 0,
                    "estimated_output_seconds": estimated_frames / source_sample_rate_hz,
                    "active_speech_seconds": active_seconds,
                    "asr_candidate_text": text,
                    "text_character_count": _text_characters(text),
                    "asr_word_count": len(words),
                    "asr_mean_word_probability": (
                        sum(probabilities) / len(probabilities) if probabilities else None
                    ),
                    "asr_max_no_speech_probability": max(
                        (float(segment.get("no_speech_prob", 0.0)) for segment in group),
                        default=None,
                    ),
                    "parent": parent,
                }
            )
    return units


def _read_composite(
    source_path: Path,
    source_spans: list[dict[str, int]],
    *,
    join_silence_seconds: float,
    boundary_repair: BoundaryRepairPolicy | None = None,
) -> tuple[np.ndarray, int]:
    if boundary_repair is not None:
        samples, rate, _ = render_repaired_spans(
            source_path, source_spans, join_silence_seconds=join_silence_seconds,
            policy=boundary_repair,
        )
        return samples, rate
    pieces: list[np.ndarray] = []
    sample_rate: int | None = None
    channel_count: int | None = None
    for span in source_spans:
        samples, current_rate = read_source_segment(
            source_path,
            start_frame=int(span["source_start_frame"]),
            end_frame=int(span["source_end_frame"]),
        )
        if sample_rate is None:
            sample_rate = current_rate
            channel_count = int(samples.shape[1])
        elif current_rate != sample_rate or samples.shape[1] != channel_count:
            raise RuntimeError("source audio format changed while reading refinement spans")
        if pieces and join_silence_seconds > 0.0:
            pieces.append(
                np.zeros(
                    (round(join_silence_seconds * sample_rate), int(channel_count)),
                    dtype=np.float32,
                )
            )
        pieces.append(np.asarray(samples, dtype=np.float32))
    if not pieces or sample_rate is None:
        raise ValueError("refinement unit contains no readable source spans")
    return np.concatenate(pieces, axis=0), sample_rate


def _initial_exclusion_reasons(
    unit: dict[str, Any], config: LongFormRefinementConfig
) -> list[str]:
    sentence = config.sentence
    reasons = []
    if unit["active_speech_seconds"] < sentence.minimum_active_speech_seconds:
        reasons.append("too_little_active_speech")
    if unit["estimated_output_seconds"] < sentence.minimum_output_seconds:
        reasons.append("output_too_short")
    if unit["estimated_output_seconds"] > sentence.maximum_output_seconds:
        reasons.append("output_too_long")
    if unit["text_character_count"] < sentence.minimum_text_characters:
        reasons.append("too_few_text_characters")
    if unit["asr_mean_word_probability"] is None:
        reasons.append("asr_probability_unavailable")
    return reasons


def _quality_exclusion_reasons(
    unit: dict[str, Any], features: dict[str, Any], config: LongFormRefinementConfig
) -> list[str]:
    policy = config.screening
    parent = unit["parent"]
    reasons = []
    if not config.sentence.minimum_output_seconds <= float(features["duration_seconds"]) <= config.sentence.maximum_output_seconds:
        reasons.append("rendered_duration_out_of_range")
    if parent.get("style_suggestion") not in policy.accepted_parent_styles:
        reasons.append("parent_style_not_normal")
    if policy.require_dominant_speaker_cluster and not parent.get(
        "speaker_cluster_is_dominant"
    ):
        reasons.append("non_dominant_speaker_cluster")
    probability = unit.get("asr_mean_word_probability")
    if probability is None or probability < policy.minimum_asr_mean_word_probability:
        reasons.append("low_asr_probability")
    snr = features.get("snr_proxy_db")
    if snr is None:
        snr = parent.get("snr_proxy_db")
    if snr is None or float(snr) < policy.minimum_snr_proxy_db:
        reasons.append("low_or_unknown_snr")
    if float(features.get("silence_ratio", 1.0)) > policy.maximum_silence_ratio:
        reasons.append("high_silence_ratio")
    if float(features.get("near_clip_ratio", 1.0)) > policy.maximum_near_clip_ratio:
        reasons.append("near_clipping")
    if policy.reject_spatial_risk and features.get("spatial_review_reasons"):
        reasons.append("spatial_audio_risk")
    if policy.reject_dense_background_proxy and "overlap_or_dense_background_proxy" in features.get(
        "quality_review_reasons", []
    ):
        reasons.append("dense_background_risk")
    return reasons


def _ranking_score(unit: dict[str, Any], features: dict[str, Any]) -> float:
    parent = unit["parent"]
    probability = float(unit.get("asr_mean_word_probability") or 0.0)
    snr = features.get("snr_proxy_db")
    if snr is None:
        snr = parent.get("snr_proxy_db") or 0.0
    speaker = float(parent.get("speaker_center_cosine") or 0.0)
    duration = float(features["duration_seconds"])
    duration_score = max(0.0, 1.0 - abs(duration - 5.0) / 10.0)
    return round(
        probability * 40.0
        + min(max(float(snr), 0.0), 30.0)
        + speaker * 20.0
        + duration_score * 10.0
        - float(features.get("silence_ratio", 1.0)) * 10.0,
        6,
    )


def refine_long_form_manifest(
    manifest_path: str | Path,
    source_path: str | Path,
    *,
    refinement_config_path: str | Path = DEFAULT_REFINEMENT_CONFIG_PATH,
    long_form_config_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    source_file = Path(source_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("asr_status") != "succeeded":
        raise RuntimeError("sentence refinement requires a successful timestamp ASR run")
    if file_sha256(source_file) != manifest.get("source_sha256"):
        raise ValueError("source audio SHA-256 does not match the parent manifest")
    long_form_config: LongFormConfig = (
        load_long_form_config(long_form_config_path)
        if long_form_config_path is not None
        else load_long_form_config()
    )
    if long_form_config.config_sha256() != manifest.get("config_sha256"):
        raise ValueError("long-form configuration does not match the parent manifest")
    config = load_refinement_config(refinement_config_path)
    config_hash = config.config_sha256()
    parent_hash = _manifest_sha256(manifest_file)
    asr_path = manifest_file.parent / "asr_timestamp_results.json"
    asr_hash = file_sha256(asr_path)
    if manifest.get("asr_results_sha256") not in (None, asr_hash):
        raise ValueError("ASR result hash does not match the parent manifest")
    artifact_key = hashlib.sha256(
        f"{parent_hash}:{asr_hash}:{config_hash}:{LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION}".encode()
    ).hexdigest()[:20]
    resolved_output = (
        Path(output_root).resolve()
        if output_root is not None
        else manifest_file.parent / "refinements" / artifact_key
    )
    validate_output_path(
        resolved_output,
        protected_inputs=(source_file, manifest_file, asr_path,
                          Path(__file__).resolve().parents[2] / long_form_config.paths.input_root),
    )
    existing_path = resolved_output / "manifest.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if (
            existing.get("parent_manifest_sha256") != parent_hash
            or existing.get("asr_results_sha256") != asr_hash
            or existing.get("config_sha256") != config_hash
            or existing.get("implementation_version") != LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION
        ):
            raise RuntimeError("refinement output belongs to other inputs; use a new output directory")
        for row in existing["segments"]:
            audio = contained_file(resolved_output, row["derived_relative_path"])
            if file_sha256(audio) != row["derived_audio_sha256"]:
                raise RuntimeError("refinement audio hash drift; use a new output directory")
        audit = json.loads((resolved_output / "refinement_audit.json").read_text(encoding="utf-8"))
        return {
            "schema_version": 1, "manifest_path": str(existing_path),
            "audit_path": str(resolved_output / "refinement_audit.json"),
            **{key: audit[key] for key in ("unit_count", "shortlist_count", "reserve_count", "auto_excluded_count")},
            "cache_action": "cached",
        }
    resolved_output.mkdir(parents=True, exist_ok=True)
    asr_payload = json.loads(asr_path.read_text(encoding="utf-8"))
    sample_rate = int(manifest["source_scan"]["sample_rate_hz"])
    units = build_refinement_units(
        manifest["segments"],
        asr_payload["results"],
        config=config,
        source_sample_rate_hz=sample_rate,
    )
    assessed = []
    for unit in units:
        reasons = _initial_exclusion_reasons(unit, config)
        features = None
        samples = None
        current_rate = sample_rate
        if not reasons:
            samples, current_rate = _read_composite(
                source_file,
                unit["source_spans"],
                join_silence_seconds=config.sentence.join_silence_seconds,
                boundary_repair=config.boundary_repair,
            )
            features = analyze_segment_samples(
                samples,
                sample_rate=current_rate,
                config=long_form_config,
                activity_start_threshold_dbfs=float(
                    manifest["segmentation"]["activity_start_threshold_dbfs"]
                ),
                activity_continue_threshold_dbfs=float(
                    manifest["segmentation"]["activity_continue_threshold_dbfs"]
                ),
            )
            reasons.extend(_quality_exclusion_reasons(unit, features, config))
        assessed.append(
            {
                **unit,
                "sample_rate": current_rate,
                "features": features,
                "automatic_exclusion_reasons": sorted(set(reasons)),
                "ranking_score": (
                    None if features is None else _ranking_score(unit, features)
                ),
            }
        )
        # Keep only metadata. A source can have thousands of rejected candidates.
        del samples
    qualified = sorted(
        (item for item in assessed if not item["automatic_exclusion_reasons"]),
        key=lambda item: (
            -float(item["ranking_score"]),
            int(item["source_start_frame"]),
            str(item["unit_id"]),
        ),
    )
    limit = config.screening.maximum_review_candidates_per_source
    shortlist_ids = {item["unit_id"] for item in qualified[:limit]}
    reserve_ids = {item["unit_id"] for item in qualified[limit:]}
    rows = []
    for item in assessed:
        if item["unit_id"] not in shortlist_ids:
            continue
        parent = item["parent"]
        features = item["features"]
        assert features is not None
        repair = None
        rendered_spans = item["source_spans"]
        if config.boundary_repair is not None:
            samples, current_rate, repair = render_repaired_spans(
                source_file, item["source_spans"],
                join_silence_seconds=config.sentence.join_silence_seconds,
                policy=config.boundary_repair,
            )
            rendered_spans = [
                {key: mapping[key] for key in ("source_start_frame", "source_end_frame")}
                for mapping in repair["mappings"]
            ]
        else:
            samples, current_rate = _read_composite(
                source_file, item["source_spans"],
                join_silence_seconds=config.sentence.join_silence_seconds,
            )
        relative_audio = f"segments/{item['unit_id']}.wav"
        rendered = _render_training_candidate(
            samples,
            sample_rate=current_rate,
            analysis_channel=int(features["analysis_channel"]),
            channel_strategy=str(features["recommended_channel_strategy"]),
            output_path=resolved_output / relative_audio,
            config=long_form_config,
            ensure_zero_edges=repair is not None,
        )
        del samples
        boundary_reason = (
            "asr_sentence_composite"
            if len(item["source_spans"]) > 1
            else "asr_sentence_trim"
        )
        rows.append(
            {
                "schema_version": 1,
                "segment_id": item["unit_id"],
                "source_sha256": manifest["source_sha256"],
                "config_sha256": config_hash,
                "parent_manifest_sha256": parent_hash,
                "parent_segment_id": item["parent_segment_id"],
                "source_start_frame": rendered_spans[0]["source_start_frame"],
                "source_end_frame": rendered_spans[-1]["source_end_frame"],
                "source_spans": rendered_spans,
                "requested_source_spans": item["source_spans"],
                "boundary_repair": repair,
                "source_scan_sample_rate_hz": sample_rate,
                "active_speech_seconds": item["active_speech_seconds"],
                "compressed_internal_pause": len(item["source_spans"]) > 1,
                "boundary_reason": boundary_reason,
                "review_required": True,
                **features,
                **rendered,
                "derived_relative_path": relative_audio,
                "asr_candidate_text": item["asr_candidate_text"],
                "asr_text_status": "candidate",
                "asr_word_count": item["asr_word_count"],
                "asr_mean_word_probability": item["asr_mean_word_probability"],
                "human_confirmed_text": None,
                "text_review_status": "pending",
                "speaker_cluster_id": parent["speaker_cluster_id"],
                "speaker_cluster_size": parent["speaker_cluster_size"],
                "speaker_cluster_is_dominant": parent["speaker_cluster_is_dominant"],
                "speaker_center_cosine": parent.get("speaker_center_cosine"),
                "style_cluster_id": parent["style_cluster_id"],
                "style_suggestion": parent["style_suggestion"],
                "screening_tier": "shortlist",
                "ranking_score": item["ranking_score"],
                "review_reasons": sorted(
                    set(features["review_reasons"] + ["target_reference_required"])
                ),
            }
        )
    rows.sort(key=lambda item: (item["source_start_frame"], item["segment_id"]))
    audit_units = []
    for item in assessed:
        if item["unit_id"] in shortlist_ids:
            tier = "shortlist"
        elif item["unit_id"] in reserve_ids:
            tier = "reserve"
        else:
            tier = "auto_excluded"
        audit_units.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"parent", "samples", "features", "sample_rate"}
            }
            | {
                "screening_tier": tier,
                "features": item["features"],
            }
        )
    audit = {
        "schema_version": 1,
        "implementation_version": LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION,
        "source_sha256": manifest["source_sha256"],
        "parent_manifest_sha256": parent_hash,
        "asr_results_sha256": asr_hash,
        "refinement_config_sha256": config_hash,
        "unit_count": len(audit_units),
        "shortlist_count": len(shortlist_ids),
        "reserve_count": len(reserve_ids),
        "auto_excluded_count": len(audit_units) - len(shortlist_ids) - len(reserve_ids),
        "units": audit_units,
    }
    _atomic_json(resolved_output / "refinement_audit.json", audit)
    refined_manifest = {
        "schema_version": 1,
        "implementation_version": LONG_FORM_REFINEMENT_IMPLEMENTATION_VERSION,
        "source_sha256": manifest["source_sha256"],
        "source_relative_path": manifest["source_relative_path"],
        "config_sha256": config_hash,
        "parent_config_sha256": manifest["config_sha256"],
        "parent_manifest_sha256": parent_hash,
        "asr_results_sha256": asr_hash,
        "source_scan": manifest["source_scan"],
        "segmentation": manifest["segmentation"],
        "asr_status": "succeeded",
        "asr_backend": manifest.get("asr_backend"),
        "segment_count": len(rows),
        "refinement": {
            "config_id": config.config_id,
            "config_version": config.config_version,
            "config_sha256": config_hash,
            "unit_count": len(audit_units),
            "shortlist_count": len(shortlist_ids),
            "reserve_count": len(reserve_ids),
            "auto_excluded_count": len(audit_units) - len(shortlist_ids) - len(reserve_ids),
            "audit_relative_path": "refinement_audit.json",
            "minimum_output_seconds": config.sentence.minimum_output_seconds,
            "maximum_output_seconds": config.sentence.maximum_output_seconds,
        },
        "segments": rows,
    }
    refined_manifest_path = resolved_output / "manifest.json"
    _atomic_json(refined_manifest_path, refined_manifest)
    return {
        "schema_version": 1,
        "manifest_path": str(refined_manifest_path),
        "audit_path": str(resolved_output / "refinement_audit.json"),
        "unit_count": len(audit_units),
        "shortlist_count": len(shortlist_ids),
        "reserve_count": len(reserve_ids),
        "auto_excluded_count": len(audit_units) - len(shortlist_ids) - len(reserve_ids),
        "cache_action": "built",
    }
