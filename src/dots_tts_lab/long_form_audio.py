from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import soxr

from dots_tts_lab.long_form_contract import LongFormConfig


LONG_FORM_AUDIO_IMPLEMENTATION_VERSION = 1
HASH_BLOCK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class AudioBlock:
    index: int
    start_frame: int
    end_frame: int
    sample_rate_hz: int
    samples: np.ndarray


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_BLOCK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_long_audio(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    info = sf.info(str(source_path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Invalid or empty audio metadata: {source_path}")
    return {
        "absolute_path": str(source_path),
        "format": str(info.format),
        "subtype": str(info.subtype),
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
    }


def iter_audio_blocks(
    path: str | Path, *, block_seconds: float
) -> Iterator[AudioBlock]:
    if block_seconds <= 0:
        raise ValueError("block_seconds must be positive")
    source_path = Path(path).resolve()
    with sf.SoundFile(str(source_path), mode="r") as audio:
        block_frames = max(1, round(float(block_seconds) * audio.samplerate))
        start_frame = 0
        block_index = 0
        while start_frame < audio.frames:
            samples = audio.read(
                frames=min(block_frames, audio.frames - start_frame),
                dtype="float32",
                always_2d=True,
            )
            if samples.size == 0:
                break
            if not np.all(np.isfinite(samples)):
                raise ValueError(
                    f"Audio block {block_index} contains NaN or infinite samples"
                )
            end_frame = start_frame + int(samples.shape[0])
            yield AudioBlock(
                index=block_index,
                start_frame=start_frame,
                end_frame=end_frame,
                sample_rate_hz=int(audio.samplerate),
                samples=samples,
            )
            start_frame = end_frame
            block_index += 1
        if start_frame != audio.frames:
            raise RuntimeError(
                f"Decoded frame count mismatch: expected {audio.frames}, got {start_frame}"
            )


def ensure_48k_derivative(
    source: str | Path,
    *,
    work_root: str | Path,
    target_rate: int = 48000,
    block_seconds: float = 30.0,
) -> tuple[Path, str]:
    """Return a 48 kHz pipeline source for any input rate.

    Non-48 kHz sources are resampled block-wise (channels preserved) into
    ``<work_root>/normalized/<orig_sha12>.48k.wav`` plus a sidecar JSON
    recording the original path/sha256/rate — frame coordinates downstream
    then refer to the derivative, whose own sha256 becomes the pipeline
    source hash. Already-48 kHz sources are returned unchanged.
    """
    source = Path(source).resolve()
    probe = probe_long_audio(source)
    original_sha = file_sha256(source)
    if probe["sample_rate_hz"] == target_rate:
        return source, original_sha

    norm_dir = Path(work_root) / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    derivative = norm_dir / f"{original_sha[:12]}.48k.wav"
    sidecar = norm_dir / f"{original_sha[:12]}.json"
    if derivative.is_file() and sidecar.is_file():
        return derivative, file_sha256(derivative)

    partial = derivative.with_suffix(derivative.suffix + ".partial")
    src_rate = int(probe["sample_rate_hz"])
    written = 0
    with sf.SoundFile(str(source), mode="r") as src, sf.SoundFile(
        str(partial), mode="w", format="WAV",
        samplerate=target_rate, channels=int(probe["channels"]),
        subtype="PCM_24",
    ) as dst:
        block_frames = max(1, round(block_seconds * src_rate))
        while True:
            block = src.read(block_frames, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            out = np.asarray(
                soxr.resample(block, src_rate, target_rate, quality="VHQ"),
                dtype=np.float32,
            )
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            dst.write(out)
            written += int(out.shape[0])
    partial.replace(derivative)
    import json as _json
    sidecar.write_text(_json.dumps({
        "schema_version": 1,
        "kind": "normalized_48k_derivative",
        "original_path": str(source),
        "original_sha256": original_sha,
        "original_sample_rate_hz": src_rate,
        "target_sample_rate_hz": target_rate,
        "channels": probe["channels"],
        "resampler": "soxr:VHQ",
        "frames_written": written,
        "derivative_sha256": file_sha256(derivative),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return derivative, file_sha256(derivative)


def _to_dbfs(amplitude: float) -> float | None:
    if amplitude <= 0.0 or not math.isfinite(amplitude):
        return None
    return float(20.0 * math.log10(amplitude))


def _stereo_correlation(
    count: int,
    sums: np.ndarray,
    square_sums: np.ndarray,
    stereo_product_sum: float,
) -> float | None:
    if count <= 1 or len(sums) != 2:
        return None
    covariance = stereo_product_sum - float(sums[0] * sums[1]) / count
    left_variance = float(square_sums[0] - sums[0] * sums[0] / count)
    right_variance = float(square_sums[1] - sums[1] * sums[1] / count)
    denominator = math.sqrt(max(0.0, left_variance) * max(0.0, right_variance))
    if denominator <= np.finfo(float).tiny:
        return None
    return float(max(-1.0, min(1.0, covariance / denominator)))


def scan_long_audio(path: str | Path, *, config: LongFormConfig) -> dict[str, Any]:
    source_path = Path(path).resolve()
    probe = probe_long_audio(source_path)
    channel_count = int(probe["channels"])
    sample_count = 0
    sums = np.zeros(channel_count, dtype=np.float64)
    square_sums = np.zeros(channel_count, dtype=np.float64)
    peak = 0.0
    near_clip_count = 0
    stereo_product_sum = 0.0
    maximum_block_frames = 0
    maximum_block_bytes = 0
    block_count = 0
    near_clip_amplitude = 10.0 ** (config.screening.near_clip_level_dbfs / 20.0)

    for block in iter_audio_blocks(
        source_path, block_seconds=config.decode.block_seconds
    ):
        samples = block.samples
        if samples.shape[1] != channel_count:
            raise RuntimeError("Channel count changed while decoding source audio")
        block_count += 1
        maximum_block_frames = max(maximum_block_frames, int(samples.shape[0]))
        maximum_block_bytes = max(maximum_block_bytes, int(samples.nbytes))
        sample_count += int(samples.shape[0])
        values = np.asarray(samples, dtype=np.float64)
        sums += np.sum(values, axis=0, dtype=np.float64)
        square_sums += np.sum(np.square(values), axis=0, dtype=np.float64)
        peak = max(peak, float(np.max(np.abs(samples))))
        near_clip_count += int(np.count_nonzero(np.abs(samples) >= near_clip_amplitude))
        if channel_count == 2:
            stereo_product_sum += float(
                np.sum(values[:, 0] * values[:, 1], dtype=np.float64)
            )

    if sample_count != int(probe["frames"]):
        raise RuntimeError("Streaming scan did not decode the complete source")
    channel_rms = np.sqrt(square_sums / max(1, sample_count))
    channel_rms_dbfs = [_to_dbfs(float(value)) for value in channel_rms]
    finite_channel_levels = [value for value in channel_rms_dbfs if value is not None]
    channel_level_difference_db = (
        max(finite_channel_levels) - min(finite_channel_levels)
        if len(finite_channel_levels) >= 2
        else 0.0
        if finite_channel_levels
        else None
    )
    stereo_correlation = _stereo_correlation(
        sample_count, sums, square_sums, stereo_product_sum
    )
    spatial_review_reasons: list[str] = []
    if channel_count > 2:
        spatial_review_reasons.append("multichannel_source")
    if (
        channel_level_difference_db is not None
        and channel_level_difference_db
        > config.screening.maximum_channel_level_difference_db
    ):
        spatial_review_reasons.append("channel_level_imbalance")
    if (
        stereo_correlation is not None
        and stereo_correlation
        < config.screening.minimum_channel_correlation_for_safe_mean
    ):
        spatial_review_reasons.append("low_stereo_correlation")

    return {
        "schema_version": 1,
        "implementation_version": LONG_FORM_AUDIO_IMPLEMENTATION_VERSION,
        "source_sha256": file_sha256(source_path),
        **probe,
        "block_seconds": config.decode.block_seconds,
        "block_count": block_count,
        "maximum_decoded_block_frames": maximum_block_frames,
        "maximum_decoded_block_bytes": maximum_block_bytes,
        "decoded_frames": sample_count,
        "sample_peak_dbfs": _to_dbfs(peak),
        "channel_rms_dbfs": channel_rms_dbfs,
        "near_clip_ratio": near_clip_count / max(1, sample_count * channel_count),
        "stereo_correlation": stereo_correlation,
        "channel_level_difference_db": channel_level_difference_db,
        "spatial_review_reasons": spatial_review_reasons,
    }

