from __future__ import annotations

import math
from typing import Any, Iterable

from dots_tts_lab.long_form_contract import LongFormConfig


LONG_FORM_ASR_IMPLEMENTATION_VERSION = 1


def validated_timestamp_words(
    metadata: dict[str, Any], *, audio_duration_seconds: float
) -> list[dict[str, Any]]:
    segments = metadata.get("timestamp_segments")
    if not isinstance(segments, list):
        raise ValueError("ASR result does not contain timestamp_segments")
    words: list[dict[str, Any]] = []
    prior_end = 0.0
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("words"), list):
            raise ValueError("invalid timestamp segment")
        for word in segment["words"]:
            if not isinstance(word, dict):
                raise ValueError("invalid timestamp word")
            start = float(word["start"])
            end = float(word["end"])
            text = str(word["text"])
            probability = float(word["probability"])
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and 0.0 <= start <= end <= audio_duration_seconds + 0.05
                and start + 0.05 >= prior_end
                and math.isfinite(probability)
                and 0.0 <= probability <= 1.0
            ):
                raise ValueError("invalid or non-monotonic ASR word timestamp")
            if text:
                words.append(
                    {
                        "start": max(0.0, start),
                        "end": min(audio_duration_seconds, end),
                        "text": text,
                        "probability": probability,
                    }
                )
            prior_end = max(prior_end, end)
    return words


def candidate_text_for_interval(
    words: Iterable[dict[str, Any]], *, start_seconds: float, end_seconds: float
) -> str:
    selected = []
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        if start_seconds <= midpoint < end_seconds:
            selected.append(str(word["text"]))
    return "".join(selected).strip()


def refine_region_boundaries(
    *,
    source_start_frame: int,
    source_end_frame: int,
    source_sample_rate_hz: int,
    words: list[dict[str, Any]],
    config: LongFormConfig,
) -> list[dict[str, Any]]:
    if source_end_frame <= source_start_frame or source_sample_rate_hz <= 0:
        raise ValueError("invalid source region")
    duration = (source_end_frame - source_start_frame) / source_sample_rate_hz
    minimum = config.segmentation.minimum_seconds
    hard_maximum = config.segmentation.hard_maximum_seconds
    preferred = min(
        config.text.preferred_asr_boundary_seconds,
        config.segmentation.preferred_maximum_seconds,
    )
    if duration <= hard_maximum:
        return [
            {
                "source_start_frame": source_start_frame,
                "source_end_frame": source_end_frame,
                "duration_seconds": duration,
                "boundary_reason": "asr_verified_vad_region",
                "asr_candidate_text": candidate_text_for_interval(
                    words, start_seconds=0.0, end_seconds=duration + 1e-9
                ),
                "review_required": False,
            }
        ]
    usable_ends = sorted(
        {
            min(duration, max(0.0, float(word["end"])))
            for word in words
            if str(word.get("text", "")).strip()
        }
    )
    boundaries = [0.0]
    while duration - boundaries[-1] > hard_maximum:
        current = boundaries[-1]
        candidates = [
            value
            for value in usable_ends
            if current + minimum <= value <= current + hard_maximum
            and duration - value >= minimum
        ]
        if not candidates:
            return []
        at_or_before_preferred = [
            value for value in candidates if value <= current + preferred
        ]
        boundary = (
            max(at_or_before_preferred)
            if at_or_before_preferred
            else min(candidates, key=lambda value: abs(value - (current + preferred)))
        )
        if boundary <= current:
            return []
        boundaries.append(boundary)
    boundaries.append(duration)

    results = []
    for start, end in zip(boundaries, boundaries[1:]):
        start_frame = source_start_frame + round(start * source_sample_rate_hz)
        end_frame = (
            source_end_frame
            if end == duration
            else source_start_frame + round(end * source_sample_rate_hz)
        )
        piece_duration = (end_frame - start_frame) / source_sample_rate_hz
        if not minimum <= piece_duration <= hard_maximum:
            return []
        text = candidate_text_for_interval(
            words, start_seconds=start, end_seconds=end + 1e-9
        )
        results.append(
            {
                "source_start_frame": start_frame,
                "source_end_frame": end_frame,
                "duration_seconds": piece_duration,
                "boundary_reason": "asr_word_boundary",
                "asr_candidate_text": text,
                "review_required": not bool(text),
            }
        )
    return results


def attach_clip_asr_candidate(
    segment: dict[str, Any], *, hypothesis: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    duration = float(segment["duration_seconds"])
    words = validated_timestamp_words(metadata, audio_duration_seconds=duration)
    timestamp_text = candidate_text_for_interval(
        words, start_seconds=0.0, end_seconds=duration + 1e-9
    )
    candidate = timestamp_text or str(hypothesis).strip()
    boundary_flags: list[str] = []
    if words and float(words[0]["start"]) <= 0.02:
        boundary_flags.append("asr_word_touches_start")
    if words and duration - float(words[-1]["end"]) <= 0.02:
        boundary_flags.append("asr_word_touches_end")
    return {
        **segment,
        "asr_candidate_text": candidate,
        "asr_text_status": "candidate" if candidate else "failed",
        "asr_word_count": len(words),
        "asr_mean_word_probability": (
            sum(float(word["probability"]) for word in words) / len(words)
            if words
            else None
        ),
        "asr_boundary_flags": boundary_flags,
        "human_confirmed_text": None,
        "text_review_status": "pending",
    }
