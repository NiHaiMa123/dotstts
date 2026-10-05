from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_contract import LongFormConfig


LONG_FORM_FEATURE_IMPLEMENTATION_VERSION = 1
DB_FLOOR = -120.0


def _db(value: float) -> float | None:
    if value <= 0.0 or not math.isfinite(value):
        return None
    return float(20.0 * math.log10(value))


def read_source_segment(
    path: str | Path, *, start_frame: int, end_frame: int
) -> tuple[np.ndarray, int]:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("invalid source segment frame range")
    with sf.SoundFile(str(Path(path).resolve()), mode="r") as source:
        if end_frame > source.frames:
            raise ValueError("source segment frame range exceeds the source audio")
        source.seek(start_frame)
        samples = source.read(
            frames=end_frame - start_frame,
            dtype="float32",
            always_2d=True,
        )
        sample_rate = int(source.samplerate)
    if len(samples) != end_frame - start_frame:
        raise RuntimeError("decoded segment frame count mismatch")
    if not np.all(np.isfinite(samples)):
        raise ValueError("source segment contains NaN or infinite samples")
    return samples, sample_rate


def _frame_levels(
    mono: np.ndarray, *, sample_rate: int, frame_ms: float, hop_ms: float
) -> np.ndarray:
    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(mono) <= frame_length:
        energy = np.asarray([np.mean(np.square(mono, dtype=np.float64))])
    else:
        starts = np.arange(0, len(mono) - frame_length + 1, hop_length)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.square(mono, dtype=np.float64), dtype=np.float64))
        )
        energy = (cumulative[starts + frame_length] - cumulative[starts]) / frame_length
    return np.asarray(
        10.0 * np.log10(np.maximum(energy, 10.0 ** (DB_FLOOR / 10.0))),
        dtype=np.float32,
    )


def _spectral_flatness_median(mono: np.ndarray, sample_rate: int) -> float:
    frame_length = max(128, round(sample_rate * 0.04))
    hop_length = max(64, round(sample_rate * 0.02))
    if len(mono) < frame_length:
        source = np.pad(mono, (0, frame_length - len(mono)))
        starts = np.asarray([0])
    else:
        source = mono
        starts = np.arange(0, len(mono) - frame_length + 1, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    values: list[np.ndarray] = []
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset : offset + 128]
        frames = np.stack(
            [source[start : start + frame_length] for start in batch_starts]
        )
        power = np.square(np.abs(np.fft.rfft(frames * window, axis=1))) + 1e-12
        geometric = np.exp(np.mean(np.log(power), axis=1))
        arithmetic = np.mean(power, axis=1)
        values.append(np.asarray(geometric / arithmetic, dtype=np.float32))
    return float(np.median(np.concatenate(values)))


def _stereo_features(
    samples: np.ndarray, *, sample_rate: int, config: LongFormConfig
) -> dict[str, Any]:
    channels = samples.shape[1]
    rms = np.sqrt(np.mean(np.square(samples, dtype=np.float64), axis=0))
    channel_rms_dbfs = [_db(float(value)) for value in rms]
    finite = [value for value in channel_rms_dbfs if value is not None]
    level_difference = (
        max(finite) - min(finite) if len(finite) >= 2 else 0.0 if finite else None
    )
    if channels != 2:
        return {
            "channel_rms_dbfs": channel_rms_dbfs,
            "channel_level_difference_db": level_difference,
            "stereo_correlation": None,
            "pan_standard_deviation": None,
            "side_to_mid_db": None,
            "recommended_channel_strategy": "mono" if channels == 1 else "review",
            "spatial_review_reasons": [] if channels == 1 else ["multichannel_source"],
        }

    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(correlation):
        correlation = 0.0
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(np.square(mid))))
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    side_to_mid_db = _db(side_rms / max(mid_rms, np.finfo(float).tiny))

    window_frames = max(
        1, round(config.screening.spatial_window_seconds * sample_rate)
    )
    pan_values: list[float] = []
    for start in range(0, len(samples), window_frames):
        window = samples[start : start + window_frames].astype(np.float64)
        if not len(window):
            continue
        window_energy = np.mean(np.square(window), axis=0)
        total = float(np.sum(window_energy))
        if total > np.finfo(float).tiny:
            pan_values.append(float((window_energy[1] - window_energy[0]) / total))
    pan_standard_deviation = float(np.std(pan_values)) if pan_values else 0.0

    reasons: list[str] = []
    if (
        level_difference is not None
        and level_difference > config.screening.maximum_channel_level_difference_db
    ):
        reasons.append("channel_level_imbalance")
    if correlation < config.screening.minimum_channel_correlation_for_safe_mean:
        reasons.append("low_stereo_correlation")
    if pan_standard_deviation > config.screening.maximum_pan_standard_deviation:
        reasons.append("moving_stereo_position")
    if (
        side_to_mid_db is not None
        and side_to_mid_db > config.screening.maximum_side_to_mid_db
    ):
        reasons.append("side_dominant_audio")
    return {
        "channel_rms_dbfs": channel_rms_dbfs,
        "channel_level_difference_db": level_difference,
        "stereo_correlation": correlation,
        "pan_standard_deviation": pan_standard_deviation,
        "side_to_mid_db": side_to_mid_db,
        "recommended_channel_strategy": "mean" if not reasons else "review",
        "spatial_review_reasons": reasons,
    }


def analyze_segment_samples(
    samples: np.ndarray,
    *,
    sample_rate: int,
    config: LongFormConfig,
    activity_start_threshold_dbfs: float,
    activity_continue_threshold_dbfs: float,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or not values.size:
        raise ValueError("segment samples must be a non-empty frames-by-channels array")
    if not np.all(np.isfinite(values)):
        raise ValueError("segment samples contain NaN or infinite values")

    channel_energy = np.mean(np.square(values, dtype=np.float64), axis=0)
    analysis_channel = int(np.argmax(channel_energy))
    mono = values[:, analysis_channel]
    frame_levels = _frame_levels(
        mono,
        sample_rate=sample_rate,
        frame_ms=config.segmentation.frame_ms,
        hop_ms=config.segmentation.hop_ms,
    )
    active = frame_levels > activity_continue_threshold_dbfs
    noise = frame_levels[frame_levels <= activity_start_threshold_dbfs]
    speech = frame_levels[frame_levels > activity_start_threshold_dbfs]
    noise_floor = float(np.percentile(noise, 50.0)) if noise.size else None
    speech_level = float(np.percentile(speech, 90.0)) if speech.size else None
    snr_proxy = (
        None
        if noise_floor is None or speech_level is None
        else max(0.0, speech_level - noise_floor)
    )
    peak = float(np.max(np.abs(values)))
    near_clip_amplitude = 10.0 ** (config.screening.near_clip_level_dbfs / 20.0)
    near_clip_ratio = float(np.mean(np.abs(values) >= near_clip_amplitude))
    silence_ratio = float(1.0 - np.mean(active))
    flatness = _spectral_flatness_median(mono, sample_rate)
    continuous_activity_ratio = float(np.mean(active))
    overlap_risk_proxy = float(
        min(1.0, 0.55 * continuous_activity_ratio + 0.45 * min(1.0, flatness / 0.5))
    )

    quality_reasons: list[str] = []
    if snr_proxy is None:
        quality_reasons.append("snr_proxy_unavailable")
    elif snr_proxy < config.screening.minimum_snr_proxy_db:
        quality_reasons.append("low_snr_proxy")
    if silence_ratio > config.screening.maximum_silence_ratio:
        quality_reasons.append("high_silence_ratio")
    if near_clip_ratio > config.screening.maximum_near_clip_ratio:
        quality_reasons.append("near_clipping")
    if overlap_risk_proxy >= config.screening.overlap_proxy_review_score:
        quality_reasons.append("overlap_or_dense_background_proxy")

    stereo = _stereo_features(values, sample_rate=sample_rate, config=config)
    level_dbfs = stereo["channel_rms_dbfs"][analysis_channel]
    review_reasons = quality_reasons + list(stereo["spatial_review_reasons"])
    return {
        "schema_version": 1,
        "implementation_version": LONG_FORM_FEATURE_IMPLEMENTATION_VERSION,
        "sample_rate_hz": sample_rate,
        "channels": int(values.shape[1]),
        "frames": int(values.shape[0]),
        "duration_seconds": len(values) / sample_rate,
        "analysis_channel": analysis_channel,
        "level_dbfs": level_dbfs,
        "sample_peak_dbfs": _db(peak),
        "silence_ratio": silence_ratio,
        "noise_floor_proxy_dbfs": noise_floor,
        "speech_level_proxy_dbfs": speech_level,
        "snr_proxy_db": snr_proxy,
        "near_clip_ratio": near_clip_ratio,
        "spectral_flatness_median": flatness,
        "continuous_activity_ratio": continuous_activity_ratio,
        "overlap_risk_proxy": overlap_risk_proxy,
        "overlap_proxy_is_calibrated": False,
        **stereo,
        "quality_review_reasons": quality_reasons,
        "review_reasons": sorted(set(review_reasons)),
    }


def analyze_source_segment(
    path: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    config: LongFormConfig,
    activity_start_threshold_dbfs: float,
    activity_continue_threshold_dbfs: float,
) -> dict[str, Any]:
    samples, sample_rate = read_source_segment(
        path, start_frame=start_frame, end_frame=end_frame
    )
    return analyze_segment_samples(
        samples,
        sample_rate=sample_rate,
        config=config,
        activity_start_threshold_dbfs=activity_start_threshold_dbfs,
        activity_continue_threshold_dbfs=activity_continue_threshold_dbfs,
    )
