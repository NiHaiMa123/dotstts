from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from dots_tts_lab.long_form_audio import iter_audio_blocks, probe_long_audio
from dots_tts_lab.long_form_contract import LongFormConfig


LONG_FORM_SEGMENTATION_IMPLEMENTATION_VERSION = 1
DB_FLOOR = -120.0


def iter_frame_level_blocks(
    path: str | Path, *, config: LongFormConfig
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    probe = probe_long_audio(path)
    sample_rate = int(probe["sample_rate_hz"])
    frame_length = max(
        1, round(sample_rate * config.segmentation.frame_ms / 1000.0)
    )
    hop_length = max(1, round(sample_rate * config.segmentation.hop_ms / 1000.0))
    carry = np.empty((0, int(probe["channels"])), dtype=np.float32)
    carry_start = 0
    next_frame_start = 0

    for block in iter_audio_blocks(path, block_seconds=config.decode.block_seconds):
        if carry.size:
            combined = np.concatenate((carry, block.samples), axis=0)
            combined_start = carry_start
        else:
            combined = block.samples
            combined_start = block.start_frame
        combined_end = combined_start + len(combined)
        first_start = max(next_frame_start, combined_start)
        last_start = combined_end - frame_length
        if first_start <= last_start:
            global_starts = np.arange(
                first_start, last_start + 1, hop_length, dtype=np.int64
            )
            local_starts = global_starts - combined_start
            per_sample_energy = np.mean(
                np.square(combined, dtype=np.float64), axis=1, dtype=np.float64
            )
            cumulative = np.concatenate(
                ([0.0], np.cumsum(per_sample_energy, dtype=np.float64))
            )
            frame_energy = (
                cumulative[local_starts + frame_length] - cumulative[local_starts]
            ) / frame_length
            levels = 10.0 * np.log10(np.maximum(frame_energy, 10.0 ** (DB_FLOOR / 10.0)))
            ends = np.minimum(global_starts + frame_length, int(probe["frames"]))
            yield global_starts, ends, np.asarray(levels, dtype=np.float32)
            next_frame_start = int(global_starts[-1]) + hop_length

        keep_local = max(0, min(len(combined), next_frame_start - combined_start))
        carry = np.asarray(combined[keep_local:], dtype=np.float32)
        carry_start = combined_start + keep_local


def _histogram_percentile(
    histogram: np.ndarray, edges: np.ndarray, percentile: float
) -> float:
    total = int(np.sum(histogram))
    if total <= 0:
        return DB_FLOOR
    target = max(1, math.ceil(total * percentile / 100.0))
    index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    index = min(index, len(edges) - 2)
    return float((edges[index] + edges[index + 1]) / 2.0)


def estimate_activity_threshold(
    path: str | Path, *, config: LongFormConfig
) -> dict[str, float | int]:
    edges = np.linspace(DB_FLOOR, 0.0, 241, dtype=np.float64)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    frame_count = 0
    for _starts, _ends, levels in iter_frame_level_blocks(path, config=config):
        clipped = np.clip(levels.astype(np.float64), DB_FLOOR, -np.finfo(float).eps)
        counts, _ = np.histogram(clipped, bins=edges)
        histogram += counts
        frame_count += len(levels)
    noise_floor = _histogram_percentile(
        histogram, edges, config.screening.noise_floor_percentile
    )
    start_threshold = min(
        config.screening.maximum_activity_threshold_dbfs,
        max(
            config.screening.absolute_activity_floor_dbfs,
            noise_floor + config.screening.activity_start_margin_db,
        ),
    )
    continue_threshold = min(
        start_threshold,
        max(
            config.screening.absolute_activity_floor_dbfs,
            noise_floor + config.screening.activity_continue_margin_db,
        ),
    )
    return {
        "frame_count": frame_count,
        "noise_floor_proxy_dbfs": noise_floor,
        "activity_start_threshold_dbfs": float(start_threshold),
        "activity_continue_threshold_dbfs": float(continue_threshold),
    }


def _active_regions(
    path: str | Path,
    *,
    config: LongFormConfig,
    start_threshold_dbfs: float,
    continue_threshold_dbfs: float,
) -> list[dict[str, int]]:
    probe = probe_long_audio(path)
    sample_rate = int(probe["sample_rate_hz"])
    merge_gap_frames = round(
        config.segmentation.merge_silence_seconds * sample_rate
    )
    regions: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for starts, ends, levels in iter_frame_level_blocks(path, config=config):
        for start_value, end_value, level_value in zip(starts, ends, levels):
            start = int(start_value)
            end = int(end_value)
            level = float(level_value)
            if current is not None and start - current["end_frame"] > merge_gap_frames:
                regions.append(current)
                current = None
            threshold = (
                start_threshold_dbfs if current is None else continue_threshold_dbfs
            )
            if level <= threshold:
                continue
            if current is None:
                current = {"start_frame": start, "end_frame": end}
            else:
                current["end_frame"] = max(current["end_frame"], end)
    if current is not None:
        regions.append(current)
    return regions


def _pad_and_split_regions(
    regions: list[dict[str, int]],
    *,
    total_frames: int,
    sample_rate: int,
    config: LongFormConfig,
) -> tuple[list[dict[str, Any]], int]:
    minimum_frames = round(config.segmentation.minimum_seconds * sample_rate)
    preferred_frames = round(
        config.segmentation.preferred_maximum_seconds * sample_rate
    )
    hard_maximum_frames = round(
        config.segmentation.hard_maximum_seconds * sample_rate
    )
    padding_frames = round(config.segmentation.boundary_padding_seconds * sample_rate)
    segments: list[dict[str, Any]] = []
    dropped_short_regions = 0

    for region_index, region in enumerate(regions):
        start = max(0, region["start_frame"] - padding_frames)
        end = min(total_frames, region["end_frame"] + padding_frames)
        duration_frames = end - start
        if duration_frames < minimum_frames:
            dropped_short_regions += 1
            continue
        if duration_frames <= hard_maximum_frames:
            boundaries = [start, end]
            reason = "vad_region"
        else:
            piece_count = max(2, math.ceil(duration_frames / preferred_frames))
            while math.ceil(duration_frames / piece_count) > hard_maximum_frames:
                piece_count += 1
            boundaries = [
                start + round(duration_frames * index / piece_count)
                for index in range(piece_count + 1)
            ]
            boundaries[0] = start
            boundaries[-1] = end
            reason = "forced_max_duration_split"
        piece_count = len(boundaries) - 1
        for piece_index, (piece_start, piece_end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            if piece_end - piece_start < minimum_frames:
                dropped_short_regions += 1
                continue
            segments.append(
                {
                    "source_start_frame": int(piece_start),
                    "source_end_frame": int(piece_end),
                    "duration_seconds": (piece_end - piece_start) / sample_rate,
                    "boundary_reason": reason,
                    "review_required": reason != "vad_region",
                    "source_region_index": region_index,
                    "source_region_start_frame": int(start),
                    "source_region_end_frame": int(end),
                    "source_region_piece_index": piece_index,
                    "source_region_piece_count": piece_count,
                }
            )
    return segments, dropped_short_regions


def segment_long_audio(path: str | Path, *, config: LongFormConfig) -> dict[str, Any]:
    probe = probe_long_audio(path)
    threshold = estimate_activity_threshold(path, config=config)
    regions = _active_regions(
        path,
        config=config,
        start_threshold_dbfs=float(threshold["activity_start_threshold_dbfs"]),
        continue_threshold_dbfs=float(threshold["activity_continue_threshold_dbfs"]),
    )
    segments, dropped_short_regions = _pad_and_split_regions(
        regions,
        total_frames=int(probe["frames"]),
        sample_rate=int(probe["sample_rate_hz"]),
        config=config,
    )
    return {
        "schema_version": 1,
        "implementation_version": LONG_FORM_SEGMENTATION_IMPLEMENTATION_VERSION,
        "sample_rate_hz": int(probe["sample_rate_hz"]),
        "source_frames": int(probe["frames"]),
        **threshold,
        "active_region_count": len(regions),
        "candidate_segment_count": len(segments),
        "dropped_short_region_count": dropped_short_regions,
        "segments": segments,
    }
