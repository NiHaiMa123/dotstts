from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import soundfile as sf
import soxr
from scipy.signal.windows import hann

from dots_tts_lab.dataset_freeze import AcousticFingerprintConfig


FINGERPRINT_IMPLEMENTATION_VERSION = 1
FINGERPRINT_DIMENSION = 4096
DEFAULT_FINGERPRINT_CACHE_ROOT = Path(
    "data/cache/dataset_features/acoustic_fingerprint"
)
DEFAULT_FINGERPRINT_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/acoustic_fingerprints.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acoustic_fingerprint_identity(
    config: AcousticFingerprintConfig,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "algorithm": "edge_trim_rms_logmel_timepool_bandcenter_l2",
        "implementation_version": FINGERPRINT_IMPLEMENTATION_VERSION,
        "cache_format_version": 2,
        "config": config.model_dump(mode="json"),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "soxr": soxr.__version__,
        },
        "resampler_quality": "HQ",
        "mel_scale": "htk",
        "stft_implementation": "numpy_sliding_window_rfft",
        "stft_boundary": None,
        "stft_padded": False,
        "log_floor": 1e-10,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "payload": payload,
        "canonical_json": canonical_json,
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def _trim_edges(audio: np.ndarray, *, top_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio), initial=0.0))
    if not math.isfinite(peak) or peak <= np.finfo(np.float32).tiny:
        raise RuntimeError("Acoustic fingerprint input is silent or non-finite")
    threshold = peak * 10.0 ** (-top_db / 20.0)
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if active.size == 0:
        raise RuntimeError("Acoustic fingerprint input is empty after edge trim")
    return audio[int(active[0]) : int(active[-1]) + 1]


def _htk_mel_filterbank(
    *, sample_rate: int, n_fft: int, n_mels: int
) -> np.ndarray:
    frequency_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    maximum_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(0.0, maximum_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    filters = np.zeros((n_mels, frequency_bins.size), dtype=np.float64)
    for index in range(n_mels):
        lower, center, upper = hz_points[index : index + 3]
        rising = (frequency_bins - lower) / max(center - lower, np.finfo(float).eps)
        falling = (upper - frequency_bins) / max(upper - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def compute_acoustic_fingerprint(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: AcousticFingerprintConfig,
) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError("Acoustic fingerprint input must contain audio samples")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Acoustic fingerprint input contains non-finite samples")
    values = _trim_edges(values, top_db=config.trim_top_db)
    if sample_rate != config.sample_rate:
        values = np.asarray(
            soxr.resample(values, sample_rate, config.sample_rate, quality="HQ"),
            dtype=np.float64,
        )
    rms = float(np.sqrt(np.mean(np.square(values))))
    if not math.isfinite(rms) or rms <= np.finfo(np.float32).tiny:
        raise RuntimeError("Acoustic fingerprint input has invalid RMS")
    target_rms = 10.0 ** (config.rms_target_dbfs / 20.0)
    values = values * (target_rms / rms)
    if values.size < config.window_length:
        values = np.pad(values, (0, config.window_length - values.size))

    window = hann(config.window_length, sym=False)
    frames = np.lib.stride_tricks.sliding_window_view(
        values, config.window_length
    )[:: config.hop_length]
    spectrum = np.fft.rfft(
        frames * window[np.newaxis, :],
        n=config.n_fft,
        axis=1,
    )
    power = np.square(np.abs(spectrum.T), dtype=np.float64)
    mel = _htk_mel_filterbank(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
    ) @ power
    log_mel = np.log(np.maximum(mel, 1e-10))
    source_positions = np.linspace(0.0, 1.0, log_mel.shape[1])
    target_positions = np.linspace(0.0, 1.0, config.pooled_frames)
    pooled = np.stack(
        [
            np.interp(target_positions, source_positions, frequency_band)
            for frequency_band in log_mel
        ]
    )
    pooled -= pooled.mean(axis=1, keepdims=True)
    flattened = pooled.reshape(-1)
    expected_dimension = config.n_mels * config.pooled_frames
    if expected_dimension != FINGERPRINT_DIMENSION:
        raise RuntimeError(
            "Fingerprint config dimension is "
            f"{expected_dimension}, expected {FINGERPRINT_DIMENSION}"
        )
    norm = float(np.linalg.norm(flattened))
    if not math.isfinite(norm) or norm <= np.finfo(np.float32).tiny:
        raise RuntimeError("Acoustic fingerprint collapsed to a zero vector")
    return np.asarray(flattened / norm, dtype="<f4")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("Cosine inputs must have the same non-empty shape")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("Cosine inputs must be finite")
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("Cosine inputs must have non-zero norm")
    return float(np.clip(np.dot(left_values, right_values) / denominator, -1.0, 1.0))


def load_acoustic_fingerprint_vectors(
    report: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load and revalidate every content-addressed vector named by a report."""
    if report.get("status") != "succeeded":
        raise RuntimeError("Acoustic fingerprint report did not succeed")
    if report.get("dimension") != FINGERPRINT_DIMENSION:
        raise RuntimeError("Acoustic fingerprint report dimension mismatch")
    if report.get("dtype") != "float32_le":
        raise RuntimeError("Acoustic fingerprint report dtype mismatch")
    features = report.get("features")
    if not isinstance(features, list) or report.get("feature_count") != len(features):
        raise RuntimeError("Acoustic fingerprint report feature count mismatch")

    vectors: dict[str, np.ndarray] = {}
    expected_size = FINGERPRINT_DIMENSION * np.dtype("<f4").itemsize
    for feature in features:
        asset_sha256 = feature.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError("Acoustic fingerprint report has an invalid asset SHA-256")
        if asset_sha256 in vectors:
            raise RuntimeError(f"Duplicate acoustic fingerprint asset: {asset_sha256}")
        cache_path_value = feature.get("cache_path")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            raise RuntimeError(f"Acoustic fingerprint cache path missing: {asset_sha256}")
        cache_path = Path(cache_path_value).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Acoustic fingerprint cache missing: {cache_path}")
        blob = cache_path.read_bytes()
        if len(blob) != expected_size:
            raise RuntimeError(f"Invalid acoustic fingerprint cache size: {cache_path}")
        if hashlib.sha256(blob).hexdigest() != feature.get("fingerprint_sha256"):
            raise RuntimeError(f"Acoustic fingerprint cache SHA-256 drift: {asset_sha256}")
        vector = np.frombuffer(blob, dtype="<f4").copy()
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.all(np.isfinite(vector)) or not math.isclose(
            norm, 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise RuntimeError(f"Invalid acoustic fingerprint vector: {asset_sha256}")
        vectors[asset_sha256] = vector
    return vectors


def load_or_compute_acoustic_fingerprint(
    candidate: dict[str, Any],
    *,
    config: AcousticFingerprintConfig,
    cache_root: str | Path = DEFAULT_FINGERPRINT_CACHE_ROOT,
) -> dict[str, Any]:
    asset_sha256 = candidate.get("asset_sha256")
    if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
        raise RuntimeError("Fingerprint candidate has invalid asset_sha256")
    expected_audio_sha256 = candidate.get("audio_sha256")
    if not isinstance(expected_audio_sha256, str) or len(expected_audio_sha256) != 64:
        raise RuntimeError(f"Fingerprint candidate {asset_sha256} has invalid audio_sha256")
    audio_path_value = candidate.get("audio_absolute_path")
    if not isinstance(audio_path_value, str) or not audio_path_value:
        raise RuntimeError(f"Fingerprint candidate {asset_sha256} has no audio path")
    audio_path = Path(audio_path_value).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Fingerprint audio does not exist: {audio_path}")
    actual_audio_sha256 = _sha256_file(audio_path)
    if actual_audio_sha256 != expected_audio_sha256:
        raise RuntimeError(
            f"Fingerprint input SHA-256 drift for {asset_sha256}: "
            f"expected {expected_audio_sha256}, got {actual_audio_sha256}"
        )

    identity = acoustic_fingerprint_identity(config)
    cache_path = (
        Path(cache_root).resolve()
        / identity["sha256"]
        / expected_audio_sha256[:2]
        / f"{expected_audio_sha256}.f32"
    )
    metadata_path = cache_path.with_suffix(".json")
    expected_size = FINGERPRINT_DIMENSION * np.dtype("<f4").itemsize
    if cache_path.exists() != metadata_path.exists():
        raise RuntimeError(f"Incomplete fingerprint cache entry: {cache_path}")
    if cache_path.exists():
        blob = cache_path.read_bytes()
        if len(blob) != expected_size:
            raise RuntimeError(f"Invalid cached fingerprint size: {cache_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid fingerprint cache metadata: {metadata_path}") from error
        expected_metadata = {
            "schema_version": 1,
            "asset_audio_sha256": expected_audio_sha256,
            "fingerprint_config_sha256": identity["sha256"],
            "fingerprint_sha256": hashlib.sha256(blob).hexdigest(),
            "dimension": FINGERPRINT_DIMENSION,
            "dtype": "float32_le",
        }
        if metadata != expected_metadata:
            raise RuntimeError(f"Fingerprint cache content hash or metadata drift: {cache_path}")
        vector = np.frombuffer(blob, dtype="<f4").copy()
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.all(np.isfinite(vector)) or not math.isclose(
            norm, 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise RuntimeError(f"Invalid cached fingerprint vector: {cache_path}")
        action = "cached"
    else:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        vector = compute_acoustic_fingerprint(
            audio,
            sample_rate=int(sample_rate),
            config=config,
        )
        blob = vector.astype("<f4", copy=False).tobytes(order="C")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.partial")
        try:
            with partial.open("xb") as output:
                output.write(blob)
                output.flush()
                os.fsync(output.fileno())
            if cache_path.exists():
                if cache_path.read_bytes() != blob:
                    raise RuntimeError(f"Conflicting fingerprint cache entry: {cache_path}")
                partial.unlink()
            else:
                partial.replace(cache_path)
        finally:
            if partial.exists():
                partial.unlink()
        metadata = {
            "schema_version": 1,
            "asset_audio_sha256": expected_audio_sha256,
            "fingerprint_config_sha256": identity["sha256"],
            "fingerprint_sha256": hashlib.sha256(blob).hexdigest(),
            "dimension": FINGERPRINT_DIMENSION,
            "dtype": "float32_le",
        }
        metadata_content = (
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        metadata_partial = metadata_path.with_name(
            f".{metadata_path.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            with metadata_partial.open("x", encoding="utf-8", newline="\n") as output:
                output.write(metadata_content)
                output.flush()
                os.fsync(output.fileno())
            if metadata_path.exists():
                if metadata_path.read_text(encoding="utf-8") != metadata_content:
                    raise RuntimeError(
                        f"Conflicting fingerprint cache metadata: {metadata_path}"
                    )
                metadata_partial.unlink()
            else:
                metadata_partial.replace(metadata_path)
        finally:
            if metadata_partial.exists():
                metadata_partial.unlink()
        action = "computed"
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    return {
        "asset_sha256": asset_sha256,
        "audio_sha256": expected_audio_sha256,
        "fingerprint_config_sha256": identity["sha256"],
        "fingerprint_sha256": blob_sha256,
        "dimension": FINGERPRINT_DIMENSION,
        "dtype": "float32_le",
        "cache_path": str(cache_path),
        "cache_metadata_path": str(metadata_path),
        "action": action,
        "vector": vector,
    }


def build_acoustic_fingerprint_cache(
    candidates: Iterable[dict[str, Any]],
    *,
    config: AcousticFingerprintConfig,
    cache_root: str | Path = DEFAULT_FINGERPRINT_CACHE_ROOT,
) -> dict[str, Any]:
    seen_assets: set[str] = set()
    features = []
    actions: Counter[str] = Counter()
    for candidate in candidates:
        asset_sha256 = candidate.get("asset_sha256")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate fingerprint candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        feature = load_or_compute_acoustic_fingerprint(
            candidate,
            config=config,
            cache_root=cache_root,
        )
        actions[feature["action"]] += 1
        features.append(
            {
                key: value
                for key, value in feature.items()
                if key not in {"vector", "action"}
            }
        )
    features.sort(key=lambda item: item["asset_sha256"])
    identity = acoustic_fingerprint_identity(config)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "fingerprint_identity": identity["payload"],
        "fingerprint_config_sha256": identity["sha256"],
        "feature_count": len(features),
        "dimension": FINGERPRINT_DIMENSION,
        "dtype": "float32_le",
        "features": features,
    }
    return {"report": report, "actions": dict(sorted(actions.items()))}


def write_acoustic_fingerprint_report(
    report: dict[str, Any],
    path: str | Path = DEFAULT_FINGERPRINT_REPORT_PATH,
) -> dict[str, str]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(target),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
