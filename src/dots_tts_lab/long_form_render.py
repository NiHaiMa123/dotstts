from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from dots_tts_lab.long_form_features import read_source_segment


BOUNDARY_REPAIR_IMPLEMENTATION_VERSION = 1


class BoundaryRepairPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    context_seconds: float = Field(default=0.18, ge=0.15, le=0.2)
    zero_crossing_search_seconds: float = Field(default=0.005, ge=0.0, le=0.02)
    edge_fade_seconds: float = Field(default=0.025, ge=0.02, le=0.03)
    join_fade_seconds: float = Field(default=0.015, ge=0.01, le=0.02)


def _nearest_crossing(values: np.ndarray, target: int, radius: int) -> int:
    start, end = max(1, target - radius), min(len(values) - 1, target + radius)
    if start > end:
        return target
    indices = np.arange(start, end + 1)
    crossing = indices[(values[indices] == 0) | (np.signbit(values[indices - 1]) != np.signbit(values[indices]))]
    if not len(crossing):
        return target
    return min((int(index) for index in crossing), key=lambda index: (abs(index - target), index))


def render_repaired_spans(
    source_path: Path,
    source_spans: list[dict[str, int]],
    *,
    join_silence_seconds: float,
    policy: BoundaryRepairPolicy,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Read bounded context, adjust crossings, and de-click each PCM/silence join.

    Trace offsets are in source-rate frames before the final mono/resample render.
    The inserted pause is retained; join fades do not remove or overlap speech.
    """
    info = sf.info(source_path)
    rate = int(info.samplerate)
    context = round(policy.context_seconds * rate)
    radius = round(policy.zero_crossing_search_seconds * rate)
    join_frames = round(join_silence_seconds * rate)
    pieces = []
    mappings = []
    cursor = 0
    for ordinal, span in enumerate(source_spans):
        start, end = int(span["source_start_frame"]), int(span["source_end_frame"])
        if start < 0 or end <= start or end > info.frames or end - start > 15 * rate:
            raise ValueError("invalid or oversized boundary repair source span")
        context_start, context_end = max(0, start - context), min(info.frames, end + context)
        samples, current_rate = read_source_segment(source_path, start_frame=context_start, end_frame=context_end)
        if current_rate != rate:
            raise RuntimeError("source sample rate changed during rendering")
        channel = int(np.argmax(np.mean(np.square(samples, dtype=np.float64), axis=0)))
        guide = samples[:, channel]
        adjusted_start = _nearest_crossing(guide, start - context_start, radius)
        adjusted_end = _nearest_crossing(guide, end - context_start, radius)
        if adjusted_end <= adjusted_start:
            raise ValueError("boundary repair collapsed a source span")
        piece = samples[adjusted_start:adjusted_end].copy()
        fade_in = min(round((policy.edge_fade_seconds if ordinal == 0 else policy.join_fade_seconds) * rate), len(piece) // 2)
        fade_out = min(round((policy.edge_fade_seconds if ordinal == len(source_spans) - 1 else policy.join_fade_seconds) * rate), len(piece) // 2)
        if fade_in:
            piece[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
        if fade_out:
            piece[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
        if pieces and join_frames:
            pieces.append(np.zeros((join_frames, info.channels), dtype=np.float32))
            cursor += join_frames
        mappings.append({
            "source_start_frame": context_start + adjusted_start,
            "source_end_frame": context_start + adjusted_end,
            "context_start_frame": context_start, "context_end_frame": context_end,
            "output_start_frame": cursor, "output_end_frame": cursor + len(piece),
            "fade_in_frames": fade_in, "fade_out_frames": fade_out,
            "zero_crossing_channel": channel,
        })
        pieces.append(piece)
        cursor += len(piece)
    if not pieces or cursor > round(15.5 * rate):
        raise ValueError("empty or oversized boundary repair output")
    return np.concatenate(pieces), rate, {
        "implementation_version": BOUNDARY_REPAIR_IMPLEMENTATION_VERSION,
        "policy": policy.model_dump(mode="json"), "sample_rate_hz": rate,
        "join_silence_frames": join_frames, "frames_before_resample": cursor,
        "mappings": mappings,
    }
