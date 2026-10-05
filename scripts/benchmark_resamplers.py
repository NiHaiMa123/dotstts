from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import soundfile as sf
import soxr
import torch
import torchaudio
from scipy.signal import resample_poly


TARGET_RATE = 48_000
METHODS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {}


def method(name: str):
    def register(function):
        METHODS[name] = function
        return function

    return register


@method("soxr_hq")
def resample_soxr(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    return np.asarray(
        soxr.resample(data, source_rate, target_rate, quality="HQ"),
        dtype=np.float64,
    )


@method("scipy_polyphase")
def resample_scipy(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(data, target_rate // divisor, source_rate // divisor),
        dtype=np.float64,
    )


@method("torchaudio_sinc")
def resample_torchaudio(
    data: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float64))
    output = torchaudio.functional.resample(
        tensor,
        source_rate,
        target_rate,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    )
    return output.numpy()


def analytical_signal(sample_rate: int, seconds: float = 2.0) -> np.ndarray:
    count = round(sample_rate * seconds)
    time_axis = np.arange(count, dtype=np.float64) / sample_rate
    frequencies = (137.0, 431.0, 997.0, 2_137.0, 4_901.0, 8_003.0, 12_007.0)
    amplitudes = (0.18, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04)
    phases = (0.1, 0.8, 1.4, 2.2, 2.8, 0.4, 1.9)
    result = np.zeros_like(time_axis)
    for frequency, amplitude, phase in zip(frequencies, amplitudes, phases):
        result += amplitude * np.sin(2.0 * np.pi * frequency * time_axis + phase)
    return result


def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(reference), len(candidate))
    # Exclude filter startup/ending transients from steady-state accuracy.
    margin = min(round(TARGET_RATE * 0.1), max(0, count // 10))
    if count <= 2 * margin:
        margin = 0
    return reference[margin : count - margin or None], candidate[margin : count - margin or None]


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = align(reference, candidate)
    error = reference - candidate
    return float(
        10.0
        * np.log10(
            np.sum(np.square(reference, dtype=np.float64))
            / np.sum(np.square(error, dtype=np.float64))
        )
    )


def tone_amplitude(data: np.ndarray, sample_rate: int, frequency: float) -> float:
    margin = min(round(sample_rate * 0.1), len(data) // 10)
    selected = data[margin : len(data) - margin]
    time_axis = np.arange(len(selected), dtype=np.float64) / sample_rate
    sine = np.sin(2.0 * np.pi * frequency * time_axis)
    cosine = np.cos(2.0 * np.pi * frequency * time_axis)
    return float(2.0 * np.hypot(selected @ sine, selected @ cosine) / len(selected))


def synthetic_benchmark(repeats: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    reference = analytical_signal(TARGET_RATE)
    for source_rate in (36_000, 44_100):
        source = analytical_signal(source_rate)
        near_nyquist = source_rate * 0.46
        source_time = np.arange(round(source_rate * 2.0), dtype=np.float64) / source_rate
        tone = 0.5 * np.sin(2.0 * np.pi * near_nyquist * source_time)
        image_frequency = source_rate - near_nyquist
        for name, function in METHODS.items():
            output = function(source, source_rate, TARGET_RATE)
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                function(source, source_rate, TARGET_RATE)
                timings.append(time.perf_counter() - started)
            tone_output = function(tone, source_rate, TARGET_RATE)
            wanted = tone_amplitude(tone_output, TARGET_RATE, near_nyquist)
            image = tone_amplitude(tone_output, TARGET_RATE, image_frequency)
            rows.append(
                {
                    "method": name,
                    "source_rate": source_rate,
                    "input_frames": len(source),
                    "output_frames": len(output),
                    "expected_output_frames": round(len(source) * TARGET_RATE / source_rate),
                    "snr_db": snr_db(reference, output),
                    "near_nyquist_gain_db": float(20.0 * np.log10(wanted / 0.5)),
                    "image_rejection_db": float(20.0 * np.log10(max(image, 1e-15) / wanted)),
                    "median_runtime_ms": statistics.median(timings) * 1000.0,
                }
            )
    return rows


def real_audio_benchmark(
    paths: list[Path], repeats: int
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        data, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
        mono = np.mean(data, axis=1)
        for name, function in METHODS.items():
            output = function(mono, sample_rate, TARGET_RATE)
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                function(mono, sample_rate, TARGET_RATE)
                timings.append(time.perf_counter() - started)
            rows.append(
                {
                    "method": name,
                    "path": str(path),
                    "source_rate": sample_rate,
                    "input_frames": len(mono),
                    "output_frames": len(output),
                    "expected_output_frames": round(len(mono) * TARGET_RATE / sample_rate),
                    "median_runtime_ms": statistics.median(timings) * 1000.0,
                    "realtime_factor": statistics.median(timings)
                    / (len(mono) / sample_rate),
                    "output_peak": float(np.max(np.abs(output))),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soxr": soxr.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "target_rate": TARGET_RATE,
        },
        "method_notes": {
            "soxr_hq": "python-soxr/libsoxr HQ",
            "scipy_polyphase": "scipy.signal.resample_poly defaults",
            "torchaudio_sinc": "Kaiser sinc, width=64, rolloff and beta matching torchaudio high-quality preset",
        },
        "synthetic": synthetic_benchmark(args.repeats),
        "real_audio": real_audio_benchmark(args.audio, args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
