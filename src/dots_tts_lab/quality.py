from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import resample_poly

DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "quality"
    / "signal_analysis_v1.yaml"
)
DEFAULT_QUALITY_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "quality"
    / "training_source_review_v1.yaml"
)

_PCM_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
}
_DB_FLOOR = -120.0


class _CanonicalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class QualityAnalysisConfig(_CanonicalConfig):
    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    analysis_version: int = Field(ge=1)
    frame_length_ms: float = Field(gt=0, le=100)
    hop_length_ms: float = Field(gt=0, le=100)
    silence_threshold_dbfs: float = Field(ge=-120, le=0)
    true_peak_oversample: int = Field(ge=1, le=16)
    flat_top_min_level_dbfs: float = Field(ge=-12, le=0)
    flat_top_tolerance_lsb: float = Field(gt=0, le=8)
    flat_top_min_run_samples: int = Field(ge=2, le=1000)
    noise_percentile: float = Field(ge=0, le=50)
    speech_percentile: float = Field(ge=50, le=100)

    @model_validator(mode="after")
    def validate_frame_and_percentiles(self) -> QualityAnalysisConfig:
        if self.hop_length_ms > self.frame_length_ms:
            raise ValueError("hop_length_ms cannot exceed frame_length_ms")
        if self.noise_percentile >= self.speech_percentile:
            raise ValueError("noise_percentile must be below speech_percentile")
        return self


class QualityPolicy(_CanonicalConfig):
    schema_version: Literal[1]
    policy_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    policy_version: int = Field(ge=1)
    min_sample_rate_hz: int = Field(gt=0)
    min_duration_seconds: float = Field(ge=0)
    max_duration_seconds: float = Field(gt=0)
    max_abs_dc_offset: float = Field(ge=0, le=1)
    max_leading_silence_seconds: float = Field(ge=0)
    max_trailing_silence_seconds: float = Field(ge=0)
    max_silence_ratio: float = Field(ge=0, le=1)
    max_flat_top_run_count: int = Field(ge=0)
    min_integrated_loudness_lufs: float
    max_integrated_loudness_lufs: float
    max_true_peak_estimate_dbtp: float
    min_snr_proxy_db: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> QualityPolicy:
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must exceed the minimum")
        if self.max_integrated_loudness_lufs <= self.min_integrated_loudness_lufs:
            raise ValueError(
                "max_integrated_loudness_lufs must exceed the minimum"
            )
        return self


def _load_yaml_model(path: str | Path, model_type):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Quality configuration must be a YAML mapping: {config_path}")
    return model_type.model_validate(payload, strict=True)


def load_quality_analysis_config(
    path: str | Path = DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
) -> QualityAnalysisConfig:
    return _load_yaml_model(path, QualityAnalysisConfig)


def load_quality_policy(
    path: str | Path = DEFAULT_QUALITY_POLICY_PATH,
) -> QualityPolicy:
    return _load_yaml_model(path, QualityPolicy)


def _to_db(value: float) -> float | None:
    if value <= 0 or not np.isfinite(value):
        return None
    return float(20.0 * np.log10(value))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _frame_dbfs(
    data: np.ndarray, *, sample_rate: int, config: QualityAnalysisConfig
) -> np.ndarray:
    power = np.mean(np.square(data, dtype=np.float64), axis=1)
    frame_length = max(1, round(sample_rate * config.frame_length_ms / 1000.0))
    hop_length = max(1, round(sample_rate * config.hop_length_ms / 1000.0))
    if len(power) <= frame_length:
        means = np.asarray([float(np.mean(power))], dtype=np.float64)
    else:
        starts = np.arange(0, len(power) - frame_length + 1, hop_length)
        if starts[-1] != len(power) - frame_length:
            starts = np.append(starts, len(power) - frame_length)
        cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
        means = (cumulative[starts + frame_length] - cumulative[starts]) / frame_length
    rms = np.sqrt(np.maximum(means, 0.0))
    floor = 10.0 ** (_DB_FLOOR / 20.0)
    return 20.0 * np.log10(np.maximum(rms, floor))


def _silence_metrics(
    frame_dbfs: np.ndarray,
    *,
    frame_count: int,
    sample_rate: int,
    config: QualityAnalysisConfig,
) -> tuple[float, float, float]:
    frame_length = max(1, round(sample_rate * config.frame_length_ms / 1000.0))
    hop_length = max(1, round(sample_rate * config.hop_length_ms / 1000.0))
    active = frame_dbfs > config.silence_threshold_dbfs
    silence_ratio = float(np.mean(~active))
    if not np.any(active):
        duration = frame_count / sample_rate
        return duration, duration, silence_ratio
    active_indices = np.flatnonzero(active)
    leading = active_indices[0] * hop_length / sample_rate
    active_end = min(
        frame_count,
        active_indices[-1] * hop_length + frame_length,
    )
    trailing = max(0.0, (frame_count - active_end) / sample_rate)
    return float(leading), float(trailing), silence_ratio


def _flat_top_metrics(
    data: np.ndarray, *, subtype: str | None, config: QualityAnalysisConfig
) -> tuple[int, int, int]:
    bits = _PCM_BITS.get(subtype or "")
    tolerance = (
        config.flat_top_tolerance_lsb / (2 ** (bits - 1))
        if bits is not None
        else 1e-7
    )
    min_level = 10.0 ** (config.flat_top_min_level_dbfs / 20.0)
    run_lengths: list[int] = []
    near_peak_count = 0
    for channel in range(data.shape[1]):
        samples = data[:, channel]
        near_peak = np.abs(samples) >= min_level
        near_peak_count += int(np.count_nonzero(near_peak))
        flat_pairs = (
            near_peak[:-1]
            & near_peak[1:]
            & (np.abs(np.diff(samples)) <= tolerance)
        )
        padded = np.concatenate(([False], flat_pairs, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        run_lengths.extend(int(end - start + 1) for start, end in zip(starts, ends))
    qualifying = [
        length
        for length in run_lengths
        if length >= config.flat_top_min_run_samples
    ]
    return (
        near_peak_count,
        len(qualifying),
        max(qualifying, default=0),
    )


def analyze_audio(
    path: str | Path,
    *,
    subtype: str | None,
    config: QualityAnalysisConfig,
) -> dict[str, Any]:
    audio_path = Path(path)
    data, sample_rate = sf.read(
        str(audio_path),
        dtype="float64",
        always_2d=True,
    )
    if data.size == 0 or data.shape[0] == 0:
        raise ValueError("Audio contains no frames")
    if not np.all(np.isfinite(data)):
        raise ValueError("Audio contains NaN or infinite samples")

    frame_count, channels = data.shape
    duration_seconds = frame_count / sample_rate
    sample_peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    dc_offset = float(np.max(np.abs(np.mean(data, axis=0))))
    oversampled = resample_poly(
        data,
        up=config.true_peak_oversample,
        down=1,
        axis=0,
        padtype="line",
    )
    true_peak = float(np.max(np.abs(oversampled)))

    loudness_data = data[:, 0] if channels == 1 else data
    if duration_seconds >= 0.4:
        loudness = _finite_or_none(
            pyln.Meter(sample_rate).integrated_loudness(loudness_data)
        )
    else:
        loudness = None

    frame_levels = _frame_dbfs(data, sample_rate=sample_rate, config=config)
    leading, trailing, silence_ratio = _silence_metrics(
        frame_levels,
        frame_count=frame_count,
        sample_rate=sample_rate,
        config=config,
    )
    active_levels = frame_levels[frame_levels > config.silence_threshold_dbfs]
    noise_levels = frame_levels[
        (frame_levels > _DB_FLOOR + 1.0)
        & (frame_levels <= config.silence_threshold_dbfs)
    ]
    noise_floor = (
        float(np.percentile(noise_levels, config.noise_percentile))
        if noise_levels.size
        else None
    )
    speech_level = (
        float(np.percentile(active_levels, config.speech_percentile))
        if active_levels.size
        else None
    )
    near_peak_count, flat_top_runs, max_flat_top_run = _flat_top_metrics(
        data,
        subtype=subtype,
        config=config,
    )
    sample_peak_dbfs = _to_db(sample_peak)
    rms_dbfs = _to_db(rms)

    return {
        "status": "ok",
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "frames": int(frame_count),
        "duration_seconds": float(duration_seconds),
        "sample_peak_dbfs": sample_peak_dbfs,
        "true_peak_estimate_dbtp": _to_db(true_peak),
        "rms_dbfs": rms_dbfs,
        "integrated_loudness_lufs": loudness,
        "crest_factor_db": (
            None
            if sample_peak_dbfs is None or rms_dbfs is None
            else sample_peak_dbfs - rms_dbfs
        ),
        "abs_dc_offset": dc_offset,
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "silence_ratio": silence_ratio,
        "digital_silence_frame_ratio": float(
            np.mean(frame_levels <= _DB_FLOOR + 1.0)
        ),
        "noise_floor_proxy_dbfs": noise_floor,
        "speech_level_proxy_dbfs": speech_level,
        "snr_proxy_db": (
            None
            if noise_floor is None or speech_level is None
            else max(0.0, speech_level - noise_floor)
        ),
        "near_peak_sample_count": near_peak_count,
        "near_peak_sample_ratio": near_peak_count / data.size,
        "flat_top_run_count": flat_top_runs,
        "max_flat_top_run_samples": max_flat_top_run,
    }


def assess_metrics(metrics: dict[str, Any], *, policy: QualityPolicy) -> dict[str, Any]:
    if metrics.get("status") != "ok":
        return {
            "decision": "reject",
            "reasons": [
                {
                    "code": "analysis_error",
                    "severity": "reject",
                    "message": metrics.get("error_message") or "analysis failed",
                }
            ],
        }

    reasons: list[dict[str, Any]] = []

    def minimum(metric: str, threshold: float, code: str) -> None:
        actual = metrics.get(metric)
        if actual is not None and actual < threshold:
            reasons.append(
                {
                    "code": code,
                    "severity": "review",
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                    "relation": "minimum",
                }
            )

    def maximum(metric: str, threshold: float, code: str) -> None:
        actual = metrics.get(metric)
        if actual is not None and actual > threshold:
            reasons.append(
                {
                    "code": code,
                    "severity": "review",
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                    "relation": "maximum",
                }
            )

    minimum("sample_rate", policy.min_sample_rate_hz, "low_sample_rate")
    minimum("duration_seconds", policy.min_duration_seconds, "too_short")
    maximum("duration_seconds", policy.max_duration_seconds, "too_long")
    maximum("abs_dc_offset", policy.max_abs_dc_offset, "dc_offset")
    maximum(
        "leading_silence_seconds",
        policy.max_leading_silence_seconds,
        "long_leading_silence",
    )
    maximum(
        "trailing_silence_seconds",
        policy.max_trailing_silence_seconds,
        "long_trailing_silence",
    )
    maximum("silence_ratio", policy.max_silence_ratio, "high_silence_ratio")
    maximum(
        "flat_top_run_count",
        policy.max_flat_top_run_count,
        "possible_hard_clipping",
    )
    minimum(
        "integrated_loudness_lufs",
        policy.min_integrated_loudness_lufs,
        "low_integrated_loudness",
    )
    maximum(
        "integrated_loudness_lufs",
        policy.max_integrated_loudness_lufs,
        "high_integrated_loudness",
    )
    maximum(
        "true_peak_estimate_dbtp",
        policy.max_true_peak_estimate_dbtp,
        "true_peak_above_limit",
    )
    minimum("snr_proxy_db", policy.min_snr_proxy_db, "low_snr_proxy")
    return {"decision": "review" if reasons else "pass", "reasons": reasons}
