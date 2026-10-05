from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Protocol

import numpy as np
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_audio import (
    file_sha256,
    iter_audio_blocks,
    probe_long_audio,
)
from dots_tts_lab.long_form_paths import validate_output_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFILTER_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "prefilter_v1.yaml"
)
LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION = 2

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class PrefilterVadConfig(_StrictFrozenModel):
    backend: Literal["silero_onnx"]
    package_version: Literal["6.2.0"]
    sample_rate_hz: Literal[16000]
    window_samples: Literal[512]
    speech_threshold: float = Field(gt=0.0, lt=1.0)
    continue_threshold_ratio: float = Field(gt=0.0, le=1.0)
    min_speech_seconds: float = Field(ge=0.1, le=10.0)
    merge_gap_seconds: float = Field(ge=0.0, le=2.0)
    region_max_seconds: float = Field(gt=1.0, le=34.0)


class PrefilterStereoConfig(_StrictFrozenModel):
    window_seconds: float = Field(ge=0.1, le=1.0)
    max_channel_level_difference_db: float = Field(ge=0.0, le=60.0)
    min_correlation_for_safe_mean: float = Field(ge=-1.0, le=1.0)
    max_side_to_mid_db: float = Field(ge=-60.0, le=20.0)
    max_pan_standard_deviation: float = Field(ge=0.0, le=1.0)
    spatial_risk_fraction_quarantine: float = Field(gt=0.0, le=1.0)


class PrefilterEventConfig(_StrictFrozenModel):
    hop_samples: int = Field(ge=64, le=1024)
    transient_zscore: float = Field(ge=1.0, le=20.0)
    max_transients_per_second: float = Field(ge=0.05, le=20.0)


class PrefilterPaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "PrefilterPaths":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError("prefilter outputs cannot be written under data/inbox")
        return self


class PrefilterConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    block_seconds: float = Field(ge=5.0, le=120.0)
    vad: PrefilterVadConfig
    stereo: PrefilterStereoConfig
    events: PrefilterEventConfig
    paths: PrefilterPaths

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_prefilter_config(
    path: str | Path = DEFAULT_PREFILTER_CONFIG_PATH,
) -> PrefilterConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"prefilter config must be a YAML mapping: {config_path}")
    return PrefilterConfig.model_validate(payload, strict=True)


class SpeechDetector(Protocol):
    """Streaming VAD contract: per-window probabilities, state carried across
    calls within one source and explicitly reset between sources."""

    def reset(self) -> None: ...

    def probabilities(self, samples_16k_mono: np.ndarray) -> np.ndarray: ...


class SileroOnnxVAD:
    def __init__(self, config: PrefilterVadConfig) -> None:
        import silero_vad

        if silero_vad.__version__ != config.package_version:
            raise RuntimeError(
                f"silero-vad {silero_vad.__version__} != pinned {config.package_version}"
            )
        self._model = silero_vad.load_silero_vad(onnx=True)
        self._window = config.window_samples
        self._rate = config.sample_rate_hz

    def reset(self) -> None:
        self._model.reset_states()

    def probabilities(self, samples_16k_mono: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples_16k_mono, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        whole = samples.size // self._window
        remainder = samples.size - whole * self._window
        frame_count = whole + (1 if remainder else 0)
        padded = np.zeros(frame_count * self._window, dtype=np.float32)
        padded[: samples.size] = samples
        probs = np.zeros(frame_count, dtype=np.float32)
        import torch

        for index in range(frame_count):
            frame = padded[index * self._window : (index + 1) * self._window]
            probs[index] = float(
                self._model(torch.from_numpy(frame), self._rate).item()
            )
        return probs


@dataclass(frozen=True)
class Region:
    region_index: int
    source_start_frame: int
    source_end_frame: int
    mean_vad_probability: float
    max_vad_probability: float
    speech_ratio: float
    spatial_risk_fraction: float
    transients_per_second: float
    status: Literal["usable", "prefilter_quarantine"]
    reasons: list[str]


def build_speech_regions(
    probabilities: np.ndarray,
    *,
    window_hop_seconds: float,
    config: PrefilterVadConfig,
) -> list[tuple[int, int]]:
    """Merge per-window VAD probabilities into contiguous speech regions.

    Returns (start_window, end_window) index pairs. Hysteresis: a region starts
    at speech_threshold and continues while probability stays above
    threshold * continue_ratio; gaps up to merge_gap_seconds merge; regions
    longer than region_max_seconds split at the lowest-probability valley.
    """
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return []
    start_threshold = config.speech_threshold
    continue_threshold = config.speech_threshold * config.continue_threshold_ratio
    merge_gap_windows = round(config.merge_gap_seconds / window_hop_seconds)
    min_windows = max(1, round(config.min_speech_seconds / window_hop_seconds))
    max_windows = max(1, round(config.region_max_seconds / window_hop_seconds))

    segments: list[tuple[int, int]] = []
    open_start: int | None = None
    for index, prob in enumerate(probs):
        if prob >= start_threshold and open_start is None:
            open_start = index
        elif (
            open_start is not None
            and prob < continue_threshold
            and index + merge_gap_windows < probs.size
            and not np.any(probs[index : index + merge_gap_windows + 1] >= start_threshold)
        ):
            segments.append((open_start, index))
            open_start = None
    if open_start is not None:
        segments.append((open_start, probs.size))

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= merge_gap_windows:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    merged = [seg for seg in merged if seg[1] - seg[0] >= min_windows]

    bounded: list[tuple[int, int]] = []
    for start, end in merged:
        while end - start > max_windows:
            limit = start + max_windows
            valley = start + max_windows // 2 + int(
                np.argmin(probs[start + max_windows // 2 : limit])
            )
            if valley <= start:
                valley = limit
            bounded.append((start, valley))
            start = valley
        bounded.append((start, end))
    return bounded


@dataclass
class _StereoWindow:
    index: int
    level_difference_db: float
    correlation: float | None
    pan: float
    side_to_mid_db: float


def _stereo_windows(samples: np.ndarray, sample_rate: int, window_seconds: float) -> list[_StereoWindow]:
    window_frames = max(1, round(window_seconds * sample_rate))
    output: list[_StereoWindow] = []
    total = samples.shape[0]
    for start in range(0, total, window_frames):
        block = samples[start : start + window_frames]
        if block.shape[0] < 8 or block.shape[1] != 2:
            continue
        left = block[:, 0].astype(np.float64)
        right = block[:, 1].astype(np.float64)
        left_rms = float(np.sqrt(np.mean(left * left)))
        right_rms = float(np.sqrt(np.mean(right * right)))
        if left_rms <= 0 or right_rms <= 0:
            continue
        level_diff = abs(20.0 * math.log10(left_rms / right_rms))
        denom = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
        correlation = (
            float(np.sum(left * right) / denom) if denom > np.finfo(float).tiny else None
        )
        pan = (right_rms - left_rms) / max(left_rms + right_rms, 1e-12)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mid_rms = float(np.sqrt(np.mean(mid * mid)))
        side_rms = float(np.sqrt(np.mean(side * side)))
        side_to_mid = (
            20.0 * math.log10(side_rms / mid_rms)
            if mid_rms > 0 and side_rms > 0
            else -120.0
        )
        output.append(
            _StereoWindow(
                index=start // window_frames,
                level_difference_db=level_diff,
                correlation=correlation,
                pan=float(pan),
                side_to_mid_db=float(side_to_mid),
            )
        )
    return output


def _transient_strengths(mono_16k: np.ndarray, hop: int) -> np.ndarray:
    """Positive derivative of smoothed energy: a cheap impact/click proxy.

    The envelope is smoothed over ~5 frames so sustained tones/voiced speech do
    not produce per-cycle wiggles. Not an event classifier — it only flags
    transient-dense regions for the later event gate.
    """
    if mono_16k.size < hop * 4:
        return np.zeros(0, dtype=np.float32)
    frames = mono_16k[: mono_16k.size - mono_16k.size % hop].reshape(-1, hop)
    # peak (not mean) energy per frame: short impacts stay visible while
    # sustained speech keeps a flat envelope
    energy = np.log1p(np.max(frames * frames, axis=1) * 1e6)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(energy, kernel, mode="same")
    derivative = np.diff(smooth, prepend=smooth[0])
    derivative[derivative < 0] = 0.0
    return derivative.astype(np.float32)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _source_dir(work_root: Path, source_sha256: str) -> Path:
    return work_root / source_sha256[:2] / source_sha256


def prefilter_completed(
    work_root: Path, source_sha256: str, config: PrefilterConfig
) -> bool:
    state_path = _source_dir(work_root, source_sha256) / "state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("status") == "completed"
        and state.get("config_sha256") == config.config_sha256()
        and state.get("implementation_version")
        == LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION
        and state.get("source_sha256") == source_sha256
    )


def prefilter_source(
    source_path: str | Path,
    *,
    config: PrefilterConfig,
    detector: SpeechDetector,
) -> dict[str, Any]:
    """Stream one source through neural VAD + raw-L/R + transient scans.

    Memory stays bounded by block_seconds; the detector state carries across
    blocks inside this file and must be reset by the caller between files.
    """
    source = Path(source_path).resolve()
    probe = probe_long_audio(source)
    source_hash = file_sha256(source)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    source_dir = _source_dir(work_root, source_hash)
    source_dir.mkdir(parents=True, exist_ok=True)
    state_path = source_dir / "state.json"

    if prefilter_completed(work_root, source_hash, config):
        regions_path = source_dir / "regions.json"
        return {
            "status": "cached",
            "source_sha256": source_hash,
            "regions_path": str(regions_path),
        }

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION,
            "started_at": _now(),
        },
    )
    try:
        vad_rate = config.vad.sample_rate_hz
        window_hop = config.vad.window_samples / vad_rate
        stereo_windows: list[_StereoWindow] = []
        vad_probs: list[float] = []
        transient_rows: list[np.ndarray] = []
        source_rate = int(probe["sample_rate_hz"])
        channels = int(probe["channels"])
        for block in iter_audio_blocks(source, block_seconds=config.block_seconds):
            samples = block.samples
            if channels == 2:
                stereo_windows.extend(
                    _stereo_windows(samples, source_rate, config.stereo.window_seconds)
                )
            mono = samples.mean(axis=1, dtype=np.float32)
            if source_rate != vad_rate:
                mono_16k = soxr.resample(mono, source_rate, vad_rate, quality="VHQ")
            else:
                mono_16k = mono
            vad_probs.extend(detector.probabilities(mono_16k).tolist())
            transient_rows.append(
                _transient_strengths(mono_16k, config.events.hop_samples)
            )
        strengths = (
            np.concatenate(transient_rows)
            if transient_rows
            else np.zeros(0, dtype=np.float32)
        )
        median = float(np.median(strengths)) if strengths.size else 0.0
        mad = float(np.median(np.abs(strengths - median))) if strengths.size else 0.0
        # floor scales with the strongest onset so near-flat envelopes do not
        # turn sub-1e-6 wiggles into fake transients
        peak = float(np.max(strengths)) if strengths.size else 0.0
        transient_threshold = median + config.events.transient_zscore * max(
            mad, 0.02 * peak, 1e-6
        )
        transient_times = np.nonzero(strengths > transient_threshold)[0] * (
            config.events.hop_samples / vad_rate
        )

        regions_windows = build_speech_regions(
            np.asarray(vad_probs, dtype=np.float32),
            window_hop_seconds=window_hop,
            config=config.vad,
        )
        probs_array = np.asarray(vad_probs, dtype=np.float32)
        stereo_window_seconds = config.stereo.window_seconds
        pan_values = np.asarray([w.pan for w in stereo_windows], dtype=np.float64)
        pan_std = float(np.std(pan_values)) if pan_values.size else 0.0

        total_frames = int(probe["frames"])
        total_seconds = total_frames / source_rate
        regions: list[Region] = []
        for index, (start_w, end_w) in enumerate(regions_windows):
            start_seconds = start_w * window_hop
            # The final VAD window can overshoot EOF; clamp to real audio.
            end_seconds = min(end_w * window_hop, total_seconds)
            duration = max(0.0, end_seconds - start_seconds)
            region_probs = probs_array[start_w:end_w]
            sw_start = int(start_seconds / stereo_window_seconds)
            sw_end = int(math.ceil(end_seconds / stereo_window_seconds))
            local = stereo_windows[sw_start:sw_end]
            risky = 0
            for window in local:
                flagged = (
                    window.level_difference_db
                    > config.stereo.max_channel_level_difference_db
                    or (
                        window.correlation is not None
                        and window.correlation
                        < config.stereo.min_correlation_for_safe_mean
                    )
                    or window.side_to_mid_db > config.stereo.max_side_to_mid_db
                    or abs(window.pan) > config.stereo.max_pan_standard_deviation
                )
                if flagged:
                    risky += 1
            spatial_fraction = risky / len(local) if local else 0.0
            in_region = transient_times[
                (transient_times >= start_seconds) & (transient_times < end_seconds)
            ]
            density = float(in_region.size) / duration if duration > 0 else 0.0
            reasons: list[str] = []
            if spatial_fraction > config.stereo.spatial_risk_fraction_quarantine:
                reasons.append("spatial_risk_region")
            if density > config.events.max_transients_per_second:
                reasons.append("transient_dense_region")
            status: Literal["usable", "prefilter_quarantine"] = (
                "prefilter_quarantine" if reasons else "usable"
            )
            regions.append(
                Region(
                    region_index=index,
                    source_start_frame=round(start_seconds * source_rate),
                    source_end_frame=min(
                        round(end_seconds * source_rate), total_frames
                    ),
                    mean_vad_probability=float(np.mean(region_probs)),
                    max_vad_probability=float(np.max(region_probs)),
                    speech_ratio=float(np.mean(region_probs >= config.vad.speech_threshold)),
                    spatial_risk_fraction=spatial_fraction,
                    transients_per_second=density,
                    status=status,
                    reasons=reasons,
                )
            )

        manifest = {
            "schema_version": 1,
            "implementation_version": LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION,
            "source_sha256": source_hash,
            "source_relative_path": str(source.relative_to(ROOT))
            if source.is_relative_to(ROOT)
            else str(source),
            "config_sha256": config.config_sha256(),
            "vad": {
                "backend": config.vad.backend,
                "package_version": config.vad.package_version,
                "sample_rate_hz": vad_rate,
                "window_samples": config.vad.window_samples,
                "window_hop_seconds": window_hop,
            },
            "probe": probe,
            "pan_standard_deviation": pan_std,
            "transient_threshold": transient_threshold,
            "transient_median": median,
            "regions": [
                {
                    "region_index": r.region_index,
                    "source_start_frame": r.source_start_frame,
                    "source_end_frame": r.source_end_frame,
                    "duration_seconds": round(
                        (r.source_end_frame - r.source_start_frame) / source_rate, 6
                    ),
                    "mean_vad_probability": round(r.mean_vad_probability, 6),
                    "max_vad_probability": round(r.max_vad_probability, 6),
                    "speech_ratio": round(r.speech_ratio, 6),
                    "spatial_risk_fraction": round(r.spatial_risk_fraction, 6),
                    "transients_per_second": round(r.transients_per_second, 6),
                    "status": r.status,
                    "reasons": r.reasons,
                }
                for r in regions
            ],
            "usable_region_count": sum(1 for r in regions if r.status == "usable"),
            "quarantine_region_count": sum(
                1 for r in regions if r.status == "prefilter_quarantine"
            ),
        }
        _atomic_write_json(source_dir / "regions.json", manifest)
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "completed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION,
                "region_count": len(regions),
                "usable_region_count": manifest["usable_region_count"],
                "finished_at": _now(),
            },
        )
        return {
            "status": "completed",
            "source_sha256": source_hash,
            "region_count": len(regions),
            "usable_region_count": manifest["usable_region_count"],
            "regions_path": str(source_dir / "regions.json"),
        }
    except Exception as error:
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_PREFILTER_IMPLEMENTATION_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": _now(),
            },
        )
        raise


def run_prefilter(
    source_paths: list[str | Path],
    *,
    config: PrefilterConfig,
    detector: SpeechDetector | None = None,
) -> dict[str, Any]:
    """Multi-source entry: one detector instance is reused across sources and
    explicitly reset between files, satisfying the model-reuse contract."""
    if detector is None:
        detector = SileroOnnxVAD(config.vad)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    for partial in work_root.rglob("*.partial"):
        partial.unlink()
    results = []
    for source in source_paths:
        detector.reset()
        results.append(prefilter_source(source, config=config, detector=detector))
    return {
        "status": "completed",
        "config_sha256": config.config_sha256(),
        "sources": results,
    }
