#!/usr/bin/env python3
"""Build the v7 Aemeath train manifest from the screened v4 split only.

Frozen ``datasets/aemeath/v4`` is not modified. Clips whose voiced spectrum is
a broadband veil are dropped. A shouted tail that continues through a silence
gap is cropped only for the one line whose cut and transcript were already
checked by ear; other loud tails are dropped instead of guessing the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "aemeath" / "v4" / "train.jsonl"
OUTPUT_DIR = ROOT / "data" / "work" / "aemeath" / "manifests" / "v7_official"
AUDIO_DIR = ROOT / "data" / "work" / "aemeath" / "v7_official" / "audio"
PROMPT_WAV = ROOT / "data" / "work" / "aemeath" / "v7_official" / "prompt_cropped_before_quba.wav"

FLATNESS_MAX = 0.08
KNOWN_CROP_FID = "7219e17ad342b6ee025c8cae497dcb1372960cdded6c78e9767cd612c7ccfc26"
KNOWN_CUT_SECONDS = 7.965
KNOWN_TEXT_SUFFIX = "去吧！"


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def voiced_flatness(audio: np.ndarray, sample_rate: int) -> float | None:
    window_size = int(0.04 * sample_rate)
    hop = int(0.01 * sample_rate)
    if len(audio) < window_size:
        return None
    window = np.hanning(window_size)
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
    flatness: list[float] = []
    for start in range(0, len(audio) - window_size, hop):
        frame = audio[start : start + window_size]
        if np.sqrt(np.mean(frame**2) + 1e-12) < 10 ** (-35 / 20):
            continue
        centered = frame - frame.mean()
        correlation = np.correlate(centered, centered, mode="full")
        correlation = correlation[len(correlation) // 2 :]
        correlation = correlation / (correlation[0] + 1e-12)
        lag_min = int(sample_rate / 450)
        lag_max = min(int(sample_rate / 70), len(correlation) - 1)
        if lag_max <= lag_min or float(np.max(correlation[lag_min:lag_max])) < 0.35:
            continue
        power = np.abs(np.fft.rfft(frame * window)) ** 2 + 1e-12
        band = power[(frequencies >= 1000) & (frequencies <= 6000)]
        flatness.append(float(np.exp(np.mean(np.log(band))) / (np.mean(band) + 1e-12)))
    if not flatness:
        return None
    return float(np.median(flatness))


def smear_spans(audio: np.ndarray, sample_rate: int) -> list[tuple[float, float, float]]:
    frame_n = int(0.02 * sample_rate)
    hits: list[tuple[float, float]] = []
    for start in range(0, len(audio) - frame_n, frame_n):
        frame = audio[start : start + frame_n]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        db = 20.0 * np.log10(rms)
        if db <= -28:
            continue
        centered = frame - frame.mean()
        correlation = np.correlate(centered, centered, mode="full")
        correlation = correlation[len(correlation) // 2 :]
        correlation = correlation / (correlation[0] + 1e-12)
        lag_min = int(sample_rate / 450)
        lag_max = min(int(sample_rate / 70), len(correlation) - 1)
        if lag_max <= lag_min:
            continue
        peak = float(np.max(correlation[lag_min:lag_max]))
        window = np.hanning(len(frame))
        power = np.abs(np.fft.rfft(frame * window)) ** 2 + 1e-12
        frequencies = np.fft.rfftfreq(len(frame), 1 / sample_rate)
        band = power[(frequencies >= 800) & (frequencies <= 7000)]
        flat = float(np.exp(np.mean(np.log(band))) / (np.mean(band) + 1e-12))
        if peak < 0.45 and flat > 0.30:
            hits.append((start / sample_rate, db))
    if not hits:
        return []
    groups: list[list[tuple[float, float]]] = [[hits[0]]]
    for hit in hits[1:]:
        if hit[0] - groups[-1][-1][0] <= 0.04:
            groups[-1].append(hit)
        else:
            groups.append([hit])
    spans: list[tuple[float, float, float]] = []
    for group in groups:
        duration = group[-1][0] + 0.02 - group[0][0]
        if duration >= 0.18:
            spans.append((group[0][0], group[-1][0] + 0.02, max(item[1] for item in group)))
    return spans


def crop_known(audio: np.ndarray, sample_rate: int, text: str) -> tuple[np.ndarray, str]:
    if not text.endswith(KNOWN_TEXT_SUFFIX):
        raise RuntimeError("Known clip text does not end with the shouted suffix")
    cropped_text = text[: -len(KNOWN_TEXT_SUFFIX)]
    if not cropped_text.endswith("学院"):
        raise RuntimeError("Known clip text does not end on 学院 before the shout")
    cut = int(round(KNOWN_CUT_SECONDS * sample_rate))
    if cut <= 0 or cut >= len(audio):
        raise RuntimeError("Known clip cut is outside the file")
    cropped = audio[:cut]
    tail_n = int(0.015 * sample_rate)
    tail = cropped[-tail_n:]
    tail_db = 20.0 * np.log10(float(np.sqrt(np.mean(tail**2) + 1e-12)))
    if tail_db > -33:
        raise RuntimeError(f"Known clip cut still includes the burst: {tail_db:.1f} dB")
    return cropped, cropped_text


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError(f"Screened train manifest is missing: {SOURCE}")
    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 179:
        raise RuntimeError(f"Expected the screened 179-clip train split, found {len(records)}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    kept_lines: list[str] = []
    excluded: list[dict[str, object]] = []
    crops: list[dict[str, object]] = []
    prompt_written = False

    for record in records:
        fid = str(record["fid"])
        text = str(record["text"])
        audio_path = Path(str(record["audio"]))
        audio, sample_rate = load_mono(audio_path)
        flatness = voiced_flatness(audio, sample_rate)
        spans = smear_spans(audio, sample_rate)
        duration = len(audio) / sample_rate
        entry: dict[str, object] = {
            "fid": fid,
            "text": text,
            "audio": str(audio_path),
            "flatness_1k_6k": None if flatness is None else round(flatness, 4),
            "duration_seconds": round(duration, 3),
            "smear_spans": [
                {"start": round(start, 3), "end": round(end, 3), "db_max": round(db, 1)}
                for start, end, db in spans
            ],
        }
        if fid == KNOWN_CROP_FID:
            cropped, cropped_text = crop_known(audio, sample_rate, text)
            cropped_flatness = voiced_flatness(cropped, sample_rate)
            cropped_spans = [
                span
                for span in smear_spans(cropped, sample_rate)
                if span[1] < (len(cropped) / sample_rate) - 0.05
            ]
            if cropped_flatness is None or cropped_flatness > FLATNESS_MAX or cropped_spans:
                entry["reason"] = "known_crop_still_bad"
                entry["cropped_flatness"] = cropped_flatness
                excluded.append(entry)
                continue
            destination = AUDIO_DIR / f"{fid}.wav"
            sf.write(destination, cropped, sample_rate, subtype="PCM_24")
            sf.write(PROMPT_WAV, cropped, sample_rate, subtype="PCM_24")
            prompt_written = True
            kept_lines.append(
                json.dumps(
                    {"fid": fid, "audio": str(destination), "text": cropped_text},
                    ensure_ascii=False,
                )
            )
            crops.append(
                {
                    "fid": fid,
                    "old_text": text,
                    "new_text": cropped_text,
                    "cut_seconds": KNOWN_CUT_SECONDS,
                    "new_seconds": round(len(cropped) / sample_rate, 3),
                    "flatness_1k_6k": round(cropped_flatness, 4),
                }
            )
            continue

        reasons: list[str] = []
        if flatness is None or flatness > FLATNESS_MAX:
            reasons.append("voiced_flatness")
        if any(span[1] < duration - 0.45 for span in spans):
            reasons.append("internal_smear")
        elif spans:
            reasons.append("tail_smear")
        if reasons:
            entry["reason"] = "+".join(reasons)
            excluded.append(entry)
            continue
        kept_lines.append(json.dumps({"fid": fid, "audio": str(audio_path), "text": text}, ensure_ascii=False))

    if not prompt_written:
        raise RuntimeError("The checked reference line was not cropped into the prompt")
    if not kept_lines:
        raise RuntimeError("No training clips left")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train.jsonl").write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "excluded.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in excluded),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "crops.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in crops),
        encoding="utf-8",
    )
    summary = {
        "source_manifest": str(SOURCE),
        "source_count": len(records),
        "kept_count": len(kept_lines),
        "excluded_count": len(excluded),
        "crop_count": len(crops),
        "excluded_reasons": {
            reason: sum(1 for item in excluded if item["reason"] == reason)
            for reason in sorted({str(item["reason"]) for item in excluded})
        },
        "train_manifest": str(OUTPUT_DIR / "train.jsonl"),
        "prompt_wav": str(PROMPT_WAV),
        "prompt_text": crops[0]["new_text"] if crops else None,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
