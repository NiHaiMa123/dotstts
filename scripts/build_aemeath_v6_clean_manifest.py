#!/usr/bin/env python3
"""Build the Aemeath v6 training manifest by dropping breathy v4 clips.

The frozen ``datasets/aemeath/v4`` files are not modified. A clip is excluded
when the median spectral flatness of its voiced frames, measured from 1 kHz to
6 kHz, is above 0.08. That is the tail whose flow-matching targets are noise
between harmonics. Validation stays on the original v4 manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "aemeath" / "v4" / "train.jsonl"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "work" / "aemeath" / "manifests" / "v6_flatness_le_0p08"
)
FLATNESS_MAX = 0.08


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def _voiced_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float] | None:
    frame = int(0.02 * sample_rate)
    if len(audio) >= frame:
        frame_count = len(audio) // frame
        rms = np.sqrt(
            np.mean(audio[: frame_count * frame].reshape(frame_count, frame) ** 2, axis=1)
            + 1e-12
        )
        active = np.flatnonzero(rms > 10 ** (-40 / 20))
        if active.size:
            start = max(0, int(active[0]) * frame - frame)
            stop = min(len(audio), (int(active[-1]) + 1) * frame + frame)
            audio = audio[start:stop]
    window_size = int(0.04 * sample_rate)
    hop = int(0.01 * sample_rate)
    if len(audio) < window_size:
        return None
    window = np.hanning(window_size)
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
    flatness: list[float] = []
    harmonic_to_noise: list[float] = []
    for start in range(0, len(audio) - window_size, hop):
        frame_audio = audio[start : start + window_size]
        if np.sqrt(np.mean(frame_audio**2) + 1e-12) < 10 ** (-35 / 20):
            continue
        centered = frame_audio - frame_audio.mean()
        correlation = np.correlate(centered, centered, mode="full")
        correlation = correlation[len(correlation) // 2 :]
        correlation = correlation / (correlation[0] + 1e-12)
        lag_min = int(sample_rate / 450)
        lag_max = min(int(sample_rate / 70), len(correlation) - 1)
        if lag_max <= lag_min:
            continue
        peak = float(np.max(correlation[lag_min:lag_max]))
        if peak < 0.35:
            continue
        peak = min(peak, 0.999)
        harmonic_to_noise.append(10 * np.log10(peak / (1 - peak)))
        power = np.abs(np.fft.rfft(frame_audio * window)) ** 2 + 1e-12
        band = power[(frequencies >= 1000) & (frequencies <= 6000)]
        flatness.append(float(np.exp(np.mean(np.log(band))) / (np.mean(band) + 1e-12)))
    if not flatness:
        return None
    return {
        "flatness_1k_6k": float(np.median(flatness)),
        "hnr_db": float(np.median(harmonic_to_noise)),
        "duration_seconds": round(len(audio) / sample_rate, 3),
    }


def build_manifest(
    source: Path,
    output_dir: Path,
    *,
    flatness_max: float = FLATNESS_MAX,
) -> dict[str, object]:
    kept_lines: list[str] = []
    excluded: list[dict[str, object]] = []
    kept: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = Path(record["audio"])
            audio, sample_rate = _load_mono(audio_path)
            metrics = _voiced_metrics(audio, sample_rate)
            if metrics is None:
                raise RuntimeError(f"No voiced frames in {audio_path}")
            entry = {
                "fid": record["fid"],
                "text": record["text"],
                "audio": record["audio"],
                **metrics,
            }
            if metrics["flatness_1k_6k"] > flatness_max:
                excluded.append(entry)
                continue
            kept.append(entry)
            kept_lines.append(json.dumps(record, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    train_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    (output_dir / "excluded.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in excluded),
        encoding="utf-8",
    )
    summary = {
        "source_manifest": str(source),
        "flatness_max": flatness_max,
        "flatness_band_hz": [1000, 6000],
        "source_count": len(kept) + len(excluded),
        "kept_count": len(kept),
        "excluded_count": len(excluded),
        "kept_duration_seconds": round(sum(item["duration_seconds"] for item in kept), 3),
        "excluded_duration_seconds": round(
            sum(item["duration_seconds"] for item in excluded), 3
        ),
        "train_manifest": str(train_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flatness-max", type=float, default=FLATNESS_MAX)
    args = parser.parse_args()
    summary = build_manifest(
        args.source,
        args.output_dir,
        flatness_max=args.flatness_max,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
