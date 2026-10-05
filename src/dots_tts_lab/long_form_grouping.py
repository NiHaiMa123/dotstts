from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import soxr

from dots_tts_lab.long_form_contract import LongFormConfig
from dots_tts_lab.speaker_embedding import compute_speaker_embedding


LONG_FORM_GROUPING_IMPLEMENTATION_VERSION = 1
STYLE_FEATURE_FIELDS = (
    "level_dbfs",
    "snr_proxy_db",
    "silence_ratio",
    "spectral_flatness_median",
    "periodicity",
    "pitch_median_hz",
    "pan_standard_deviation",
    "side_to_mid_db",
)


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm <= np.finfo(float).tiny:
        raise ValueError("embedding must be finite and non-zero")
    return values / norm


def extract_speaker_embedding(
    samples: np.ndarray,
    *,
    sample_rate: int,
    config: LongFormConfig,
    encoder: Any,
    channel_strategy: str,
    analysis_channel: int,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or not values.size:
        raise ValueError("speaker input must be non-empty audio")
    if channel_strategy == "mean":
        mono = np.mean(values, axis=1, dtype=np.float32)
    elif channel_strategy in {"mono", "analysis_channel", "review"}:
        if analysis_channel < 0 or analysis_channel >= values.shape[1]:
            raise ValueError("analysis_channel is outside the source channel range")
        mono = values[:, analysis_channel]
    else:
        raise ValueError(f"unsupported channel strategy: {channel_strategy}")
    target_rate = config.speaker.encoder.input_sample_rate
    if sample_rate != target_rate:
        mono = np.asarray(
            soxr.resample(mono, sample_rate, target_rate, quality="HQ"),
            dtype=np.float32,
        )
    return compute_speaker_embedding(
        mono,
        sample_rate=target_rate,
        config=config.speaker.encoder,
        encoder=encoder,
    )


def cluster_speaker_embeddings(
    vectors: Sequence[np.ndarray],
    *,
    durations: Sequence[float],
    cosine_link_threshold: float,
    knn_k: int,
    minimum_cluster_size: int,
    reference_vector: np.ndarray | None = None,
    reference_minimum_cosine: float,
) -> list[dict[str, Any]]:
    if len(vectors) != len(durations) or not vectors:
        raise ValueError("speaker vectors and durations must be non-empty and aligned")
    matrix = np.stack([_normalized(vector) for vector in vectors])
    similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
    parent = list(range(len(matrix)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            if similarities[left, right] >= cosine_link_threshold:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(matrix)):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(
        components.values(),
        key=lambda members: (-sum(float(durations[i]) for i in members), min(members)),
    )
    cluster_ids = {
        member: f"speaker_{cluster_index:03d}"
        for cluster_index, members in enumerate(ordered)
        for member in members
    }
    cluster_centers: dict[str, np.ndarray] = {}
    for members in ordered:
        cluster_id = cluster_ids[members[0]]
        cluster_centers[cluster_id] = _normalized(np.mean(matrix[members], axis=0))
    reference = None if reference_vector is None else _normalized(reference_vector)

    results: list[dict[str, Any]] = []
    for index, vector in enumerate(matrix):
        cluster_id = cluster_ids[index]
        members = next(group for group in ordered if index in group)
        neighbors = np.delete(similarities[index], index)
        effective_k = min(knn_k, len(neighbors))
        knn_cosine = (
            float(np.mean(np.sort(neighbors)[-effective_k:]))
            if effective_k
            else 1.0
        )
        reference_cosine = (
            None if reference is None else float(np.dot(vector, reference))
        )
        if reference_cosine is None:
            target_status = "reference_required"
        elif reference_cosine >= reference_minimum_cosine:
            target_status = "target_match"
        else:
            target_status = "target_review"
        results.append(
            {
                "speaker_cluster_id": cluster_id,
                "speaker_cluster_size": len(members),
                "speaker_cluster_is_dominant": cluster_id == "speaker_000",
                "speaker_cluster_is_small": len(members) < minimum_cluster_size,
                "speaker_center_cosine": float(
                    np.dot(vector, cluster_centers[cluster_id])
                ),
                "speaker_knn_cosine": knn_cosine,
                "target_reference_cosine": reference_cosine,
                "target_speaker_status": target_status,
            }
        )
    return results


def estimate_pitch_periodicity(
    samples: np.ndarray, *, sample_rate: int, analysis_channel: int
) -> dict[str, float | None]:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        mono = values[:, analysis_channel]
    elif values.ndim == 1:
        mono = values
    else:
        raise ValueError("pitch input must be mono or frames-by-channels")
    analysis_rate = 16000
    if sample_rate != analysis_rate:
        mono = np.asarray(
            soxr.resample(mono, sample_rate, analysis_rate, quality="HQ"),
            dtype=np.float32,
        )
    frame_length = round(0.04 * analysis_rate)
    hop_length = round(0.02 * analysis_rate)
    minimum_lag = max(1, analysis_rate // 500)
    maximum_lag = min(frame_length - 2, analysis_rate // 60)
    periodicities: list[float] = []
    pitches: list[float] = []
    for start in range(0, max(1, len(mono) - frame_length + 1), hop_length):
        frame = mono[start : start + frame_length].astype(np.float64)
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame -= np.mean(frame)
        energy = float(np.dot(frame, frame))
        if energy <= 1e-8:
            continue
        lags = np.arange(minimum_lag, maximum_lag + 1)
        scores = np.asarray(
            [
                np.dot(frame[:-lag], frame[lag:])
                / max(
                    1e-12,
                    math.sqrt(
                        float(np.dot(frame[:-lag], frame[:-lag]))
                        * float(np.dot(frame[lag:], frame[lag:]))
                    ),
                )
                for lag in lags
            ]
        )
        best = int(np.argmax(scores))
        periodicity = float(np.clip(scores[best], 0.0, 1.0))
        periodicities.append(periodicity)
        if periodicity >= 0.25:
            pitches.append(analysis_rate / float(lags[best]))
    return {
        "periodicity": float(np.median(periodicities)) if periodicities else 0.0,
        "pitch_median_hz": float(np.median(pitches)) if pitches else None,
    }


def _style_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    raw = np.asarray(
        [
            [float(row[field]) if row.get(field) is not None else np.nan for field in STYLE_FEATURE_FIELDS]
            for row in rows
        ],
        dtype=np.float64,
    )
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_rows, missing_columns = np.where(~np.isfinite(raw))
    raw[missing_rows, missing_columns] = medians[missing_columns]
    deviations = np.median(np.abs(raw - medians), axis=0)
    fallback = np.std(raw, axis=0)
    scales = np.where(deviations > 1e-8, deviations * 1.4826, fallback)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return (raw - medians) / scales


def cluster_style_features(
    rows: Sequence[dict[str, Any]], *, maximum_groups: int, minimum_group_size: int
) -> list[str]:
    if not rows:
        return []
    matrix = _style_matrix(rows)
    group_count = min(maximum_groups, max(1, len(rows) // minimum_group_size))
    center_indices = [int(np.argmin(np.sum(np.square(matrix), axis=1)))]
    while len(center_indices) < group_count:
        distances = np.min(
            np.stack(
                [np.sum(np.square(matrix - matrix[index]), axis=1) for index in center_indices]
            ),
            axis=0,
        )
        distances[center_indices] = -1.0
        center_indices.append(int(np.argmax(distances)))
    centers = matrix[center_indices].copy()
    labels = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(50):
        distances = np.stack(
            [np.sum(np.square(matrix - center), axis=1) for center in centers], axis=1
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and _ > 0:
            break
        labels = updated
        for index in range(group_count):
            members = matrix[labels == index]
            if len(members):
                centers[index] = np.mean(members, axis=0)
    ordering = sorted(
        range(group_count),
        key=lambda index: (-int(np.sum(labels == index)), int(np.flatnonzero(labels == index)[0])),
    )
    remap = {old: new for new, old in enumerate(ordering)}
    return [f"style_{remap[int(label)]:03d}" for label in labels]


def suggest_styles(
    rows: Sequence[dict[str, Any]],
    *,
    config: LongFormConfig,
    speaker_groups: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != len(speaker_groups):
        raise ValueError("style rows and speaker groups must be aligned")
    levels = [float(row["level_dbfs"]) for row in rows if row.get("level_dbfs") is not None]
    corpus_median = float(np.median(levels)) if levels else -120.0
    group_ids = cluster_style_features(
        rows,
        maximum_groups=config.style.maximum_groups,
        minimum_group_size=config.style.minimum_group_size,
    )
    results: list[dict[str, Any]] = []
    for row, speaker, group_id in zip(rows, speaker_groups, group_ids):
        reasons: list[str] = []
        if row.get("spatial_review_reasons"):
            suggestion = "binaural_3d"
            reasons.append("spatial_audio_features")
        elif (
            float(row.get("periodicity") or 0.0)
            <= config.style.whisper_maximum_periodicity
            and float(row.get("spectral_flatness_median") or 0.0)
            >= config.style.whisper_minimum_spectral_flatness
        ):
            suggestion = "whisper"
            reasons.append("low_periodicity_high_flatness")
        elif (
            row.get("level_dbfs") is not None
            and float(row["level_dbfs"])
            <= corpus_median - config.style.soft_below_corpus_median_db
        ):
            suggestion = "soft"
            reasons.append("level_below_corpus_median")
        elif not speaker["speaker_cluster_is_dominant"]:
            suggestion = "unknown"
            reasons.append("non_dominant_speaker_cluster")
        else:
            suggestion = "normal"
        requires_review = (
            suggestion not in config.style.accepted_styles
            or speaker["target_speaker_status"] != "target_match"
            or speaker["speaker_cluster_is_small"]
        )
        results.append(
            {
                "style_cluster_id": group_id,
                "style_suggestion": suggestion,
                "style_reasons": reasons,
                "requires_review": requires_review,
            }
        )
    return results
