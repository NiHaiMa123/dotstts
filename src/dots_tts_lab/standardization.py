from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import resample_poly


DEFAULT_STANDARDIZATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "standardize"
    / "training_audio_v1.yaml"
)


class StandardizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    target_sample_rate_hz: int = Field(gt=0)
    mono_mix_strategy: Literal["mean"]
    resampler: Literal["soxr"]
    resampler_quality: Literal["HQ"]
    output_subtype: Literal["PCM_24"]
    silence_frame_ms: float = Field(gt=0, le=100)
    silence_hop_ms: float = Field(gt=0, le=100)
    silence_threshold_dbfs: float = Field(ge=-120, le=0)
    trim_trigger_seconds: float = Field(ge=0)
    edge_padding_seconds: float = Field(ge=0)
    true_peak_oversample: int = Field(ge=1, le=16)
    target_true_peak_dbtp: float = Field(ge=-12, le=0)
    loudness_normalization: Literal[False]
    dc_offset_removal: Literal[False]

    @model_validator(mode="after")
    def validate_ranges(self) -> StandardizationConfig:
        if self.silence_hop_ms > self.silence_frame_ms:
            raise ValueError("silence_hop_ms cannot exceed silence_frame_ms")
        if self.edge_padding_seconds >= self.trim_trigger_seconds:
            raise ValueError("edge padding must be smaller than the trim trigger")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_standardization_config(
    path: str | Path = DEFAULT_STANDARDIZATION_CONFIG_PATH,
) -> StandardizationConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Standardization configuration must be a YAML mapping: {config_path}"
        )
    return StandardizationConfig.model_validate(payload, strict=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derived_identity(
    *, asset_sha256: str, config: StandardizationConfig, implementation_version: int
) -> str:
    payload = json.dumps(
        {
            "asset_sha256": asset_sha256,
            "config_sha256": config.config_sha256(),
            "implementation_version": implementation_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dbfs(value: float) -> float | None:
    if value <= 0 or not np.isfinite(value):
        return None
    return float(20.0 * np.log10(value))


def true_peak_estimate(data: np.ndarray, oversample: int) -> float:
    if data.size == 0:
        return 0.0
    if oversample == 1:
        return float(np.max(np.abs(data)))
    oversampled = resample_poly(data, oversample, 1)
    return float(np.max(np.abs(oversampled)))


def _active_bounds(
    data: np.ndarray, sample_rate: int, config: StandardizationConfig
) -> tuple[int, int] | None:
    frame_length = max(1, round(sample_rate * config.silence_frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * config.silence_hop_ms / 1000.0))
    if len(data) <= frame_length:
        starts = np.asarray([0], dtype=np.int64)
        means = np.asarray([float(np.mean(np.square(data, dtype=np.float64)))])
    else:
        starts = np.arange(0, len(data) - frame_length + 1, hop_length)
        if starts[-1] != len(data) - frame_length:
            starts = np.append(starts, len(data) - frame_length)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.square(data, dtype=np.float64), dtype=np.float64))
        )
        means = (cumulative[starts + frame_length] - cumulative[starts]) / frame_length
    threshold = 10.0 ** (config.silence_threshold_dbfs / 20.0)
    active = np.sqrt(np.maximum(means, 0.0)) > threshold
    if not np.any(active):
        return None
    indices = np.flatnonzero(active)
    return int(starts[indices[0]]), int(
        min(len(data), starts[indices[-1]] + frame_length)
    )


def conservative_trim(
    data: np.ndarray, sample_rate: int, config: StandardizationConfig
) -> tuple[np.ndarray, int, int]:
    bounds = _active_bounds(data, sample_rate, config)
    if bounds is None:
        return data, 0, 0
    active_start, active_end = bounds
    trigger = round(config.trim_trigger_seconds * sample_rate)
    padding = round(config.edge_padding_seconds * sample_rate)
    start = max(0, active_start - padding) if active_start > trigger else 0
    trailing_silence = len(data) - active_end
    end = min(len(data), active_end + padding) if trailing_silence > trigger else len(data)
    if end <= start:
        return data, 0, 0
    return data[start:end], start, len(data) - end


def standardize_array(
    data: np.ndarray,
    source_sample_rate: int,
    config: StandardizationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(data, dtype=np.float64)
    if source.ndim == 1:
        source = source[:, np.newaxis]
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("Decoded audio must contain at least one frame and channel")
    mono = np.mean(source, axis=1, dtype=np.float64)
    if source_sample_rate == config.target_sample_rate_hz:
        resampled = mono.copy()
    else:
        resampled = np.asarray(
            soxr.resample(
                mono,
                source_sample_rate,
                config.target_sample_rate_hz,
                quality=config.resampler_quality,
            ),
            dtype=np.float64,
        )
    trimmed, leading_removed, trailing_removed = conservative_trim(
        resampled, config.target_sample_rate_hz, config
    )
    peak_before_gain = true_peak_estimate(trimmed, config.true_peak_oversample)
    target_peak = 10.0 ** (config.target_true_peak_dbtp / 20.0)
    gain = min(1.0, target_peak / peak_before_gain) if peak_before_gain > 0 else 1.0
    output = trimmed * gain
    gain_db = 0.0 if gain == 1.0 else float(20.0 * math.log10(gain))
    return output, {
        "leading_samples_removed": leading_removed,
        "trailing_samples_removed": trailing_removed,
        "gain_applied_db": gain_db,
    }


def atomic_write_pcm24(
    path: Path, data: np.ndarray, sample_rate: int, true_peak_oversample: int
) -> tuple[str, int, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        sf.write(str(temporary_path), data, sample_rate, subtype="PCM_24", format="WAV")
        with temporary_path.open("r+b") as output_file:
            os.fsync(output_file.fileno())
        info = sf.info(str(temporary_path))
        if (
            info.samplerate != sample_rate
            or info.channels != 1
            or info.frames != len(data)
            or info.subtype != "PCM_24"
        ):
            raise RuntimeError(f"Derived WAV verification failed: {temporary_path}")
        decoded, decoded_rate = sf.read(
            str(temporary_path), dtype="float64", always_2d=False
        )
        if decoded_rate != sample_rate or len(decoded) != len(data):
            raise RuntimeError(f"Derived WAV decode verification failed: {temporary_path}")
        output_hash = file_sha256(temporary_path)
        size_bytes = temporary_path.stat().st_size
        os.replace(temporary_path, path)
        metrics = {
            "output_sample_peak_dbfs": _dbfs(float(np.max(np.abs(decoded)))),
            "output_true_peak_estimate_dbtp": _dbfs(
                true_peak_estimate(decoded, true_peak_oversample)
            ),
        }
        return output_hash, size_bytes, metrics
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
