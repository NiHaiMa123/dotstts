from __future__ import annotations

from typing import Any

import numpy as np
import librosa

from dots_tts_lab.postprocess import measure_integrated_loudness
from dots_tts_lab.standardization import true_peak_estimate

VOICED_FRAME_MS = 2048
VOICED_HOP = 512
VOICED_RMS_PERCENTILE = 40.0
VOICED_F0_MIN = 80.0
VOICED_F0_MAX = 400.0
REFERENCE_BAND_HZ = (200.0, 4_000.0)
FLATNESS_BANDS_HZ = ((4_000.0, 8_000.0), (8_000.0, 12_000.0))
TEMPORAL_BAND_HZ = (2_000.0, 9_000.0)


def voiced_frame_mask(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed voiced-frame selection: RMS above the 40th percentile AND yin f0 in
    80–400 Hz on 2048-sample frames at 512 hop. Returns (mask, f0_hz)."""
    x = np.asarray(audio, dtype=np.float64)
    f0 = librosa.yin(
        x,
        fmin=VOICED_F0_MIN,
        fmax=VOICED_F0_MAX,
        sr=sample_rate,
        frame_length=VOICED_FRAME_MS,
        hop_length=VOICED_HOP,
    )
    rms = librosa.feature.rms(
        y=x, frame_length=VOICED_FRAME_MS, hop_length=VOICED_HOP
    )[0]
    voiced = (f0 > VOICED_F0_MIN) & (rms > np.percentile(rms, VOICED_RMS_PERCENTILE))
    return voiced, f0


def band_relative_energy_db(
    spectra_power: np.ndarray, freqs: np.ndarray, band_hz: tuple[float, float]
) -> float:
    """Mean power in band minus mean power in the 200–4000 Hz voice band, dB."""
    ref = spectra_power[(freqs >= REFERENCE_BAND_HZ[0]) & (freqs < REFERENCE_BAND_HZ[1])]
    band = spectra_power[(freqs >= band_hz[0]) & (freqs < band_hz[1])]
    ref_power = float(ref.mean()) if ref.size else np.nan
    band_power = float(band.mean()) if band.size else np.nan
    if not np.isfinite(ref_power) or not np.isfinite(band_power):
        return float("nan")
    return 10.0 * np.log10(band_power + 1e-20) - 10.0 * np.log10(ref_power + 1e-20)


def _spectral_flatness(frame_power: np.ndarray) -> float:
    mean = float(np.mean(frame_power) + 1e-20)
    geo = float(np.exp(np.mean(np.log(frame_power + 1e-20))))
    return geo / mean


def _spectral_crest(frame_power: np.ndarray) -> float:
    mean = float(np.mean(frame_power) + 1e-20)
    return float(np.max(frame_power)) / mean


def _spectral_entropy(frame_power: np.ndarray) -> float:
    p = frame_power / (np.sum(frame_power) + 1e-20)
    entropy = float(-np.sum(p * np.log(p + 1e-20)))
    return entropy / float(np.log(frame_power.size))


def compute_grit_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Objective metrics for the suoming grit diagnostic.

    Voiced frames are selected once via the fixed rule in ``voiced_frame_mask``
    and reused for flatness/crest/entropy; temporal delta is measured over all
    frames since silence transitions are also diagnostically relevant.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("grit metrics require non-empty mono audio")
    rms = float(np.sqrt(np.mean(x**2)))
    voiced, _ = voiced_frame_mask(x, sample_rate)

    spec = np.abs(librosa.stft(x, n_fft=VOICED_FRAME_MS, hop_length=VOICED_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=VOICED_FRAME_MS)
    voiced_spec = spec[:, voiced] if np.any(voiced) else spec

    metrics: dict[str, Any] = {
        "duration_s": float(x.size / sample_rate),
        "rms_dbfs": float(20.0 * np.log10(rms + 1e-12)),
        "integrated_lufs": measure_integrated_loudness(x, sample_rate),
        "true_peak_dbtp": float(
            20.0 * np.log10(true_peak_estimate(x, 4) + 1e-12)
        ),
        "voiced_frame_count": int(np.count_nonzero(voiced)),
        "total_frame_count": int(spec.shape[1]),
    }
    for lo, hi in ((4_000.0, 8_000.0), (8_000.0, 12_000.0), (12_000.0, 18_000.0)):
        metrics[f"band_{int(lo)}_{int(hi)}_rel_db"] = band_relative_energy_db(
            voiced_spec.mean(axis=1), freqs, (lo, hi)
        )
    for lo, hi in FLATNESS_BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        band_frames = voiced_spec[mask]
        flat = np.array([_spectral_flatness(f) for f in band_frames.T])
        crest = np.array([_spectral_crest(f) for f in band_frames.T])
        entropy = np.array([_spectral_entropy(f) for f in band_frames.T])
        key = f"{int(lo)}_{int(hi)}"
        metrics[f"flatness_{key}_mean"] = float(np.mean(flat))
        metrics[f"flatness_{key}_median"] = float(np.median(flat))
        metrics[f"flatness_{key}_p90"] = float(np.percentile(flat, 90))
        metrics[f"crest_{key}_median"] = float(np.median(crest))
        metrics[f"entropy_{key}_median"] = float(np.median(entropy))

    band = (freqs >= TEMPORAL_BAND_HZ[0]) & (freqs < TEMPORAL_BAND_HZ[1])
    log_mag = 20.0 * np.log10(np.abs(librosa.stft(
        x, n_fft=VOICED_FRAME_MS, hop_length=VOICED_HOP
    ))[band] + 1e-9)
    delta = np.abs(np.diff(log_mag.mean(axis=0)))
    metrics["temporal_delta_2k_9k_median_db"] = float(np.median(delta))
    metrics["temporal_delta_2k_9k_p90_db"] = float(np.percentile(delta, 90))
    return metrics
