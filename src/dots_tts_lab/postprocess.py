from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyloudnorm as pyln
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.ndimage import median_filter
from scipy.signal import butter, fftconvolve, istft, sosfilt, sosfiltfilt, stft

from dots_tts_lab.standardization import true_peak_estimate

ROOT = Path(__file__).resolve().parents[2]
LEGACY_FUXUAN_VOICE_POLISH_CONFIG_PATH = (
    ROOT / "configs/lab/postprocess/fuxuan_voice_polish_nearfield_v1.yaml"
)
DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH = (
    ROOT / "configs/lab/postprocess/fuxuan_voice_polish_roughness_control_v2.yaml"
)
DEFAULT_BRIGHTNESS_PROFILE = "E_half_brightness_same_pitch"
DEFAULT_BRIGHTNESS_CROSSOVER_HZ = 3_000.0
DEFAULT_BRIGHTNESS_GAIN_DB = 0.68
DEFAULT_TRUE_PEAK_CEILING_DBTP = -1.0


class EdgeTrimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    silence_frame_ms: float = Field(gt=0, le=100)
    silence_hop_ms: float = Field(gt=0, le=100)
    silence_threshold_dbfs: float = Field(ge=-120, le=0)
    trim_trigger_seconds: float = Field(gt=0)
    preserve_leading_seconds: float = Field(ge=0)
    preserve_trailing_seconds: float = Field(ge=0)
    fade_seconds: float = Field(ge=0, le=0.05)
    output_subtype: Literal["PCM_24"]

    @model_validator(mode="after")
    def validate_ranges(self) -> EdgeTrimConfig:
        if self.silence_hop_ms > self.silence_frame_ms:
            raise ValueError("silence_hop_ms cannot exceed silence_frame_ms")
        if max(self.preserve_leading_seconds, self.preserve_trailing_seconds) >= self.trim_trigger_seconds:
            raise ValueError("preserved edge silence must be shorter than the trim trigger")
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ParametricEqBand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    center_hz: float = Field(gt=20.0, le=20_000.0)
    gain_db: float = Field(ge=-6.0, le=6.0)
    q: float = Field(gt=0.1, le=10.0)


class ParallelCompressorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    threshold_dbfs: float = Field(ge=-60.0, le=0.0)
    ratio: float = Field(ge=1.0, le=20.0)
    attack_ms: float = Field(gt=0.0, le=500.0)
    release_ms: float = Field(gt=0.0, le=2_000.0)
    makeup_gain_db: float = Field(ge=-12.0, le=12.0)
    mix: float = Field(ge=0.0, le=1.0)
    gate_dbfs: float = Field(ge=-100.0, le=0.0)
    frame_ms: float = Field(gt=0.0, le=100.0)


class DeesserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    low_hz: float = Field(gt=20.0, le=20_000.0)
    high_hz: float = Field(gt=20.0, le=20_000.0)
    threshold_dbfs: float = Field(ge=-80.0, le=0.0)
    ratio: float = Field(ge=1.0, le=20.0)
    maximum_reduction_db: float = Field(ge=0.0, le=12.0)
    attack_ms: float = Field(gt=0.0, le=500.0)
    release_ms: float = Field(gt=0.0, le=2_000.0)
    frame_ms: float = Field(gt=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_band(self) -> DeesserConfig:
        if self.high_hz <= self.low_hz:
            raise ValueError("deesser high_hz must exceed low_hz")
        return self


class DynamicEqBandConfig(DeesserConfig):
    """A narrow dynamic attenuation band used only when its energy gets hot."""

    band_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")


class ExciterConfig(BaseModel):
    """Optional harmonic exciter for high-frequency detail synthesis.

    A source band above ``source_low_hz`` is driven through an asymmetric tanh
    (``bias`` introduces even harmonics alongside odd ones), band-limited to
    the ``harmonic_low_hz``–``harmonic_high_hz`` octave region, and mixed back
    at ``mix`` relative to the source band energy — synthesized sparkle rather
    than amplified noise floor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_low_hz: float = Field(gt=500.0, le=12_000.0)
    drive: float = Field(gt=0.0, le=20.0)
    bias: float = Field(ge=0.0, le=1.0)
    harmonic_low_hz: float = Field(gt=500.0, le=20_000.0)
    harmonic_high_hz: float = Field(gt=500.0, le=22_000.0)
    mix: float = Field(ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_band(self) -> ExciterConfig:
        if self.harmonic_high_hz <= self.harmonic_low_hz:
            raise ValueError("exciter harmonic_high_hz must exceed harmonic_low_hz")
        if self.harmonic_low_hz < self.source_low_hz:
            raise ValueError("exciter harmonic band must sit above source_low_hz")
        return self


class AmbienceConfig(BaseModel):
    """Optional small-room ambience for de-closening a dry take.

    A synthetic exponentially-decaying noise impulse response is band-limited
    to ``band_low_hz``–``band_high_hz`` (no low-end reverb mud), delayed by
    ``pre_delay_ms`` so the direct voice stays forward, convolved, and mixed
    back at ``mix`` relative to dry energy. Adds depth, not audible echo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rt60_ms: float = Field(gt=40.0, le=800.0)
    pre_delay_ms: float = Field(ge=0.0, le=80.0)
    band_low_hz: float = Field(gt=80.0, le=2_000.0)
    band_high_hz: float = Field(gt=2_000.0, le=20_000.0)
    mix: float = Field(ge=0.0, le=0.5)
    tail: bool = True

    @model_validator(mode="after")
    def validate_band(self) -> AmbienceConfig:
        if self.band_high_hz <= self.band_low_hz:
            raise ValueError("ambience band_high_hz must exceed band_low_hz")
        return self


class SpectralStabilizerConfig(BaseModel):
    """Optional temporal spectral stabilizer for frame-to-frame envelope warble.

    Neural vocoders can emit a per-frame spectral wobble perceived as a
    gritty/rough texture ("磨砂感") that static EQ cannot reach. This block
    median-filters each STFT magnitude band across a short window — the median
    preserves consonant edges while damping sub-100 ms envelope flutter — then
    resynthesizes with the original phase.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    low_hz: float = Field(gt=200.0, le=4_000.0)
    high_hz: float = Field(gt=4_000.0, le=20_000.0)
    frames: int = Field(ge=3, le=15)
    strength: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_band(self) -> "SpectralStabilizerConfig":
        if self.high_hz <= self.low_hz:
            raise ValueError("stabilizer high_hz must exceed low_hz")
        if self.frames % 2 == 0:
            raise ValueError("stabilizer frames must be odd")
        return self


class PromptLoudnessCalibrationConfig(BaseModel):
    """Anchor batch outputs to the reference prompt audio's integrated loudness.

    The batch runner renders the prompt text once, measures the polished probe
    against the reference file, and applies the resulting fixed gain to every
    output so per-line dynamics are preserved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    maximum_gain_db: float = Field(ge=0.0, le=12.0)


class VoicePolishConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    low_cut_hz: float | None = Field(default=None, ge=20.0, le=500.0)
    brightness_crossover_hz: float = Field(gt=20.0, le=20_000.0)
    brightness_gain_db: float = Field(ge=-6.0, le=6.0)
    body_eq: ParametricEqBand
    presence_eq: ParametricEqBand
    parallel_compressor: ParallelCompressorConfig
    dynamic_eq_bands: list[DynamicEqBandConfig] = Field(default_factory=list, max_length=4)
    deesser: DeesserConfig
    match_input_loudness: bool
    target_loudness_lufs: float | None = Field(default=None, ge=-30.0, le=-10.0)
    prompt_loudness_calibration: PromptLoudnessCalibrationConfig | None = None
    exciter: ExciterConfig | None = None
    ambience: AmbienceConfig | None = None
    stabilizer: SpectralStabilizerConfig | None = None
    maximum_loudness_adjustment_db: float = Field(ge=0.0, le=6.0)
    true_peak_ceiling_dbtp: float = Field(ge=-12.0, le=0.0)
    true_peak_oversample: int = Field(ge=1, le=16)

    @model_validator(mode="after")
    def validate_loudness_mode(self) -> VoicePolishConfig:
        if self.match_input_loudness and self.target_loudness_lufs is not None:
            raise ValueError(
                "target_loudness_lufs cannot be set when match_input_loudness is true"
            )
        if self.prompt_loudness_calibration is not None and self.match_input_loudness:
            raise ValueError(
                "prompt_loudness_calibration cannot be combined with match_input_loudness"
            )
        return self

    def canonical_sha256(self) -> str:
        values = self.model_dump(mode="json")
        # Preserve the v1 hash contract when newly added optional controls are unused.
        if not values["dynamic_eq_bands"]:
            values.pop("dynamic_eq_bands")
        if values["target_loudness_lufs"] is None:
            values.pop("target_loudness_lufs")
        if values["prompt_loudness_calibration"] is None:
            values.pop("prompt_loudness_calibration")
        if values["low_cut_hz"] is None:
            values.pop("low_cut_hz")
        if values["exciter"] is None:
            values.pop("exciter")
        if values["ambience"] is None:
            values.pop("ambience")
        if values["stabilizer"] is None:
            values.pop("stabilizer")
        payload = json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_edge_trim_config(path: str | Path) -> EdgeTrimConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Edge-trim configuration must be a YAML mapping")
    return EdgeTrimConfig.model_validate(payload, strict=True)


def load_voice_polish_config(path: str | Path) -> VoicePolishConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Voice-polish configuration must be a YAML mapping")
    return VoicePolishConfig.model_validate(payload, strict=True)


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Voice polish requires finite, non-empty mono audio")
    return values


def _db(value: float) -> float | None:
    if value <= 0.0 or not np.isfinite(value):
        return None
    return float(20.0 * np.log10(value))


def _integrated_loudness(audio: np.ndarray, sample_rate: int) -> float | None:
    try:
        value = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def measure_integrated_loudness(
    audio: np.ndarray,
    sample_rate: int,
) -> float | None:
    """Integrated loudness (LUFS) of mono or stereo audio; None if unmeasurable."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = values.squeeze()
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return None
    if sample_rate <= 0:
        return None
    return _integrated_loudness(values, sample_rate)


def _brightness_unlimited(
    audio: np.ndarray,
    sample_rate: int,
    *,
    crossover_hz: float,
    gain_db: float,
) -> np.ndarray:
    if not 0.0 < crossover_hz < sample_rate / 2:
        raise ValueError("brightness crossover must be below the Nyquist frequency")
    high_pass = butter(
        2,
        crossover_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    high_band = sosfiltfilt(high_pass, audio)
    return audio + (10.0 ** (gain_db / 20.0) - 1.0) * high_band


def apply_default_brightness(
    audio: np.ndarray,
    sample_rate: int,
    *,
    crossover_hz: float = DEFAULT_BRIGHTNESS_CROSSOVER_HZ,
    gain_db: float = DEFAULT_BRIGHTNESS_GAIN_DB,
    true_peak_ceiling_dbtp: float = DEFAULT_TRUE_PEAK_CEILING_DBTP,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    """Apply the accepted E brightness profile without changing pitch or timing."""
    source = _mono_audio(audio)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    output = _brightness_unlimited(
        source,
        sample_rate,
        crossover_hz=crossover_hz,
        gain_db=gain_db,
    )
    peak_before = true_peak_estimate(output, oversample=4)
    ceiling_linear = 10.0 ** (true_peak_ceiling_dbtp / 20.0)
    attenuation_db = 0.0
    if peak_before > ceiling_linear:
        gain = ceiling_linear / peak_before
        output *= gain
        attenuation_db = float(20.0 * np.log10(gain))
    peak_after = true_peak_estimate(output, oversample=4)
    return output, {
        "profile": DEFAULT_BRIGHTNESS_PROFILE,
        "crossover_hz": crossover_hz,
        "high_band_gain_db": gain_db,
        "true_peak_ceiling_dbtp": true_peak_ceiling_dbtp,
        "peak_safety_attenuation_db": attenuation_db,
        "output_true_peak_dbtp": _db(peak_after),
    }


def _peaking_eq_sos(
    sample_rate: int,
    *,
    center_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    if center_hz >= sample_rate / 2:
        raise ValueError("EQ center frequency must be below the Nyquist frequency")
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * center_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)
    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cosine
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha / amplitude
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _frame_levels_db(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float,
) -> tuple[np.ndarray, int]:
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    frame_count = (audio.size + frame_size - 1) // frame_size
    padded_size = frame_count * frame_size
    padded = np.pad(audio, (0, padded_size - audio.size))
    rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size)), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12)), frame_size


def _smooth_gain_db(
    target_db: np.ndarray,
    *,
    frame_seconds: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    attack = np.exp(-frame_seconds / (attack_ms / 1000.0))
    release = np.exp(-frame_seconds / (release_ms / 1000.0))
    smoothed = np.empty_like(target_db)
    previous = 0.0
    for index, target in enumerate(target_db):
        coefficient = attack if target < previous else release
        previous = coefficient * previous + (1.0 - coefficient) * target
        smoothed[index] = previous
    return smoothed


def _expand_frame_values(values: np.ndarray, frame_size: int, size: int) -> np.ndarray:
    return np.repeat(values, frame_size)[:size]


def _parallel_compress(
    audio: np.ndarray,
    sample_rate: int,
    config: ParallelCompressorConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    levels, frame_size = _frame_levels_db(audio, sample_rate, config.frame_ms)
    over_threshold = np.maximum(levels - config.threshold_dbfs, 0.0)
    compression_db = -(1.0 - 1.0 / config.ratio) * over_threshold
    target_wet_gain_db = np.where(
        levels > config.gate_dbfs,
        compression_db + config.makeup_gain_db,
        0.0,
    )
    smoothed_db = _smooth_gain_db(
        target_wet_gain_db,
        frame_seconds=frame_size / sample_rate,
        attack_ms=config.attack_ms,
        release_ms=config.release_ms,
    )
    wet_gain = 10.0 ** (
        _expand_frame_values(smoothed_db, frame_size, audio.size) / 20.0
    )
    wet = audio * wet_gain
    output = (1.0 - config.mix) * audio + config.mix * wet
    return output, {
        "mix": config.mix,
        "maximum_compression_db": float(np.min(compression_db)),
        "minimum_wet_gain_db": float(np.min(smoothed_db)),
        "maximum_wet_gain_db": float(np.max(smoothed_db)),
    }


def _deess(
    audio: np.ndarray,
    sample_rate: int,
    config: DeesserConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    if config.high_hz >= sample_rate / 2:
        raise ValueError("deesser high_hz must be below the Nyquist frequency")
    band_filter = butter(
        2,
        [config.low_hz, config.high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    sibilance = sosfiltfilt(band_filter, audio)
    levels, frame_size = _frame_levels_db(sibilance, sample_rate, config.frame_ms)
    over_threshold = np.maximum(levels - config.threshold_dbfs, 0.0)
    target_reduction_db = -(1.0 - 1.0 / config.ratio) * over_threshold
    target_reduction_db = np.maximum(
        target_reduction_db,
        -config.maximum_reduction_db,
    )
    smoothed_db = _smooth_gain_db(
        target_reduction_db,
        frame_seconds=frame_size / sample_rate,
        attack_ms=config.attack_ms,
        release_ms=config.release_ms,
    )
    gain = 10.0 ** (
        _expand_frame_values(smoothed_db, frame_size, audio.size) / 20.0
    )
    output = audio + sibilance * (gain - 1.0)
    return output, {
        "maximum_reduction_db": float(-np.min(smoothed_db)),
        "active_frame_fraction": float(np.mean(target_reduction_db < -0.01)),
    }


def _excite(
    audio: np.ndarray,
    sample_rate: int,
    config: ExciterConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    if config.source_low_hz >= sample_rate / 2 or config.harmonic_high_hz >= sample_rate / 2:
        raise ValueError("exciter bands must sit below the Nyquist frequency")
    source = sosfiltfilt(
        butter(4, config.source_low_hz, btype="highpass", fs=sample_rate, output="sos"),
        audio,
    )
    source_rms = float(np.sqrt(np.mean(source**2)))
    if source_rms <= 0.0:
        return audio, {"mix": config.mix, "harmonic_level_dbfs": None}
    driven = (
        np.tanh(config.drive * (source / source_rms) + config.bias)
        - np.tanh(config.bias)
    )
    harmonics = sosfiltfilt(
        butter(
            2,
            [config.harmonic_low_hz, config.harmonic_high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ),
        driven,
    )
    harmonic_rms = float(np.sqrt(np.mean(harmonics**2)))
    if harmonic_rms <= 0.0:
        return audio, {"mix": config.mix, "harmonic_level_dbfs": None}
    output = audio + config.mix * source_rms * (harmonics / harmonic_rms)
    return output, {
        "mix": config.mix,
        "harmonic_level_dbfs": _db(config.mix * source_rms),
    }


def _ambience(
    audio: np.ndarray,
    sample_rate: int,
    config: AmbienceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.band_high_hz >= sample_rate / 2:
        raise ValueError("ambience band_high_hz must be below the Nyquist frequency")
    tail_samples = int(round(config.rt60_ms / 1000.0 * sample_rate))
    delay_samples = int(round(config.pre_delay_ms / 1000.0 * sample_rate))
    rng = np.random.default_rng(20261007)
    ir = rng.standard_normal(tail_samples) * np.exp(
        -6.907755 * np.arange(tail_samples) / max(tail_samples - 1, 1)
    )
    ir = sosfiltfilt(
        butter(
            2,
            [config.band_low_hz, config.band_high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        ),
        ir,
    )
    energy = float(np.sqrt(np.sum(ir**2)))
    if energy <= 0.0:
        return audio, {"mix": config.mix, "tail_samples_added": 0}
    ir = np.concatenate((np.zeros(delay_samples), ir / energy))
    wet = fftconvolve(audio, ir)
    if config.tail:
        output_length = audio.size + delay_samples + tail_samples - 1
    else:
        output_length = audio.size
    output = np.pad(audio, (0, output_length - audio.size))
    output += config.mix * wet[:output_length]
    return output, {
        "mix": config.mix,
        "rt60_ms": config.rt60_ms,
        "tail_samples_added": output_length - audio.size,
    }


def _spectral_stabilize(
    audio: np.ndarray,
    sample_rate: int,
    config: SpectralStabilizerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.high_hz >= sample_rate / 2:
        raise ValueError("stabilizer high_hz must be below the Nyquist frequency")
    nperseg = 2048
    hop = nperseg // 4
    _, _, z = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(z)
    phase = z / np.maximum(magnitude, 1e-12)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate)
    band = (freqs >= config.low_hz) & (freqs <= config.high_hz)
    log_mag = np.log(magnitude + 1e-12)
    smoothed = median_filter(log_mag, size=(1, config.frames), mode="reflect")
    blended = np.where(
        band[:, None],
        (1.0 - config.strength) * log_mag + config.strength * smoothed,
        log_mag,
    )
    _, rebuilt = istft(
        np.exp(blended) * phase,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary=True,
    )
    output = np.zeros(audio.size)
    take = min(audio.size, rebuilt.size)
    output[:take] = rebuilt[:take]
    return output, {
        "low_hz": config.low_hz,
        "high_hz": config.high_hz,
        "frames": config.frames,
        "strength": config.strength,
    }


def apply_fuxuan_voice_polish(
    audio: np.ndarray,
    sample_rate: int,
    config: VoicePolishConfig | None = None,
    *,
    include_brightness: bool = True,
    calibration_gain_db: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the versioned near-field voice chain used by Fuxuan final outputs."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    resolved = config or load_voice_polish_config(
        DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH
    )
    output = _mono_audio(audio).copy()
    input_true_peak = true_peak_estimate(output, resolved.true_peak_oversample)
    input_loudness = _integrated_loudness(output, sample_rate)
    if resolved.low_cut_hz is not None:
        if resolved.low_cut_hz >= sample_rate / 2:
            raise ValueError("low_cut_hz must be below the Nyquist frequency")
        output = sosfiltfilt(
            butter(
                2,
                resolved.low_cut_hz,
                btype="highpass",
                fs=sample_rate,
                output="sos",
            ),
            output,
        )
    stabilizer_details: dict[str, Any] = {"strength": 0.0}
    if resolved.stabilizer is not None:
        output, stabilizer_details = _spectral_stabilize(
            output, sample_rate, resolved.stabilizer
        )
    if include_brightness:
        output = _brightness_unlimited(
            output,
            sample_rate,
            crossover_hz=resolved.brightness_crossover_hz,
            gain_db=resolved.brightness_gain_db,
        )

    for band in (resolved.body_eq, resolved.presence_eq):
        output = sosfilt(
            _peaking_eq_sos(
                sample_rate,
                center_hz=band.center_hz,
                gain_db=band.gain_db,
                q=band.q,
            ),
            output,
        )

    output, compression = _parallel_compress(
        output,
        sample_rate,
        resolved.parallel_compressor,
    )
    dynamic_eq: list[dict[str, Any]] = []
    for band in resolved.dynamic_eq_bands:
        output, attenuation = _deess(output, sample_rate, band)
        dynamic_eq.append(
            {
                **band.model_dump(mode="json"),
                **attenuation,
            }
        )
    exciter_details: dict[str, Any] = {"mix": 0.0, "harmonic_level_dbfs": None}
    if resolved.exciter is not None:
        output, exciter_details = _excite(output, sample_rate, resolved.exciter)
    output, deesser = _deess(output, sample_rate, resolved.deesser)
    ambience_details: dict[str, Any] = {"mix": 0.0, "tail_samples_added": 0}
    if resolved.ambience is not None:
        output, ambience_details = _ambience(output, sample_rate, resolved.ambience)

    loudness_before_matching = _integrated_loudness(output, sample_rate)
    loudness_adjustment_db = 0.0
    loudness_target = (
        input_loudness if resolved.match_input_loudness else resolved.target_loudness_lufs
    )
    if calibration_gain_db is not None:
        loudness_adjustment_db = float(calibration_gain_db)
        output *= 10.0 ** (loudness_adjustment_db / 20.0)
        loudness_mode = "prompt_calibrated"
    elif loudness_target is not None and loudness_before_matching is not None:
        requested_adjustment = loudness_target - loudness_before_matching
        loudness_adjustment_db = float(
            np.clip(
                requested_adjustment,
                -resolved.maximum_loudness_adjustment_db,
                resolved.maximum_loudness_adjustment_db,
            )
        )
        output *= 10.0 ** (loudness_adjustment_db / 20.0)
        loudness_mode = (
            "match_input" if resolved.match_input_loudness else "target_lufs"
        )
    else:
        loudness_mode = "none"

    peak_before_safety = true_peak_estimate(output, resolved.true_peak_oversample)
    ceiling = 10.0 ** (resolved.true_peak_ceiling_dbtp / 20.0)
    safety_attenuation_db = 0.0
    if peak_before_safety > ceiling:
        gain = ceiling / peak_before_safety
        output *= gain
        safety_attenuation_db = float(20.0 * np.log10(gain))
    output_peak = true_peak_estimate(output, resolved.true_peak_oversample)
    return output, {
        "profile": resolved.config_id,
        "config_version": resolved.config_version,
        "config_sha256": resolved.canonical_sha256(),
        "brightness_profile": (
            DEFAULT_BRIGHTNESS_PROFILE if include_brightness else "already_applied"
        ),
        "low_cut_hz": resolved.low_cut_hz,
        "brightness_crossover_hz": resolved.brightness_crossover_hz,
        "brightness_gain_db": resolved.brightness_gain_db,
        "body_eq": resolved.body_eq.model_dump(mode="json"),
        "presence_eq": resolved.presence_eq.model_dump(mode="json"),
        "parallel_compressor": {
            **resolved.parallel_compressor.model_dump(mode="json"),
            **compression,
        },
        "dynamic_eq_bands": dynamic_eq,
        "exciter": exciter_details,
        "ambience": ambience_details,
        "stabilizer": stabilizer_details,
        "deesser": {
            **resolved.deesser.model_dump(mode="json"),
            **deesser,
        },
        "match_input_loudness": resolved.match_input_loudness,
        "target_loudness_lufs": resolved.target_loudness_lufs,
        "loudness_mode": loudness_mode,
        "calibration_gain_db": calibration_gain_db,
        "input_loudness_lufs": input_loudness,
        "pre_match_loudness_lufs": loudness_before_matching,
        "maximum_loudness_adjustment_db": resolved.maximum_loudness_adjustment_db,
        "loudness_adjustment_db": loudness_adjustment_db,
        "input_true_peak_dbtp": _db(input_true_peak),
        "pre_safety_true_peak_dbtp": _db(peak_before_safety),
        "true_peak_ceiling_dbtp": resolved.true_peak_ceiling_dbtp,
        "safety_attenuation_db": safety_attenuation_db,
        "output_true_peak_dbtp": _db(output_peak),
    }


def _active_bounds(
    audio: np.ndarray,
    sample_rate: int,
    config: EdgeTrimConfig,
) -> tuple[int, int] | None:
    frame = max(1, round(sample_rate * config.silence_frame_ms / 1000.0))
    hop = max(1, round(sample_rate * config.silence_hop_ms / 1000.0))
    if audio.size <= frame:
        starts = np.asarray([0], dtype=np.int64)
    else:
        starts = np.arange(0, audio.size - frame + 1, hop, dtype=np.int64)
        final = audio.size - frame
        if starts[-1] != final:
            starts = np.append(starts, final)
    squared = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    rms = np.sqrt(
        np.maximum((cumulative[starts + frame] - cumulative[starts]) / frame, 0.0)
    )
    threshold = 10.0 ** (config.silence_threshold_dbfs / 20.0)
    indices = np.flatnonzero(rms > threshold)
    if not indices.size:
        return None
    return int(starts[indices[0]]), int(min(audio.size, starts[indices[-1]] + frame))


def safe_edge_trim(
    audio: np.ndarray,
    sample_rate: int,
    config: EdgeTrimConfig,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Safe edge trim requires non-empty mono audio")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    bounds = _active_bounds(values, sample_rate, config)
    if bounds is None:
        return values.copy(), {
            "all_silent": True,
            "leading_samples_removed": 0,
            "trailing_samples_removed": 0,
        }
    active_start, active_end = bounds
    trigger = round(config.trim_trigger_seconds * sample_rate)
    leading_keep = round(config.preserve_leading_seconds * sample_rate)
    trailing_keep = round(config.preserve_trailing_seconds * sample_rate)
    start = max(0, active_start - leading_keep) if active_start > trigger else 0
    trailing_silence = values.size - active_end
    end = min(values.size, active_end + trailing_keep) if trailing_silence > trigger else values.size
    if end <= start:
        raise RuntimeError("Safe edge trim produced invalid bounds")
    output = values[start:end].copy()
    fade = min(round(config.fade_seconds * sample_rate), output.size // 2)
    if fade > 0 and start > 0:
        output[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=True)
    if fade > 0 and end < values.size:
        output[-fade:] *= np.linspace(1.0, 0.0, fade, endpoint=True)
    return output, {
        "all_silent": False,
        "leading_samples_removed": int(start),
        "trailing_samples_removed": int(values.size - end),
    }
