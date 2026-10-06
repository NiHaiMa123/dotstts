"""守岸人诊断指标 (Phase 5): speaker cosine + LUFS/RMS/peak + 4-12kHz flatness 分布"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import stft

from dots_tts_lab.long_form_identity import load_identity_config
from dots_tts_lab.speaker_embedding import (
    load_or_compute_speaker_embedding,
    load_speaker_encoder,
)

OUT = ROOT / "outputs/diagnostics/shouanren_zero_shot_v1"
RAW = OUT / "raw"
REPORT = OUT / "report"
IDENTITY_CONFIG = ROOT / "configs/lab/long_form/identity_v1.yaml"

REFS = {
    "R1": ROOT / "data/inbox/守岸人/中立_neutral/【中立_neutral】或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因…….wav",
    "R2": ROOT / "data/work/standardized/56/56784f46414575ff5789a2857fcb764ab5dcad40ce7522790997cb9d889418cd.wav",
    "R3": ROOT / "data/work/standardized/6b/6bdb59898ba90334918498b7a86cdbf24e8c30a06caf4b90ab17581d069610e9.wav",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate(path: Path):
    return {
        "asset_sha256": hashlib.sha256(str(path).encode()).hexdigest(),
        "audio_sha256": sha256_file(path),
        "audio_absolute_path": str(path),
    }


def embed(path: Path, *, encoder, encoder_config, cache_root):
    return load_or_compute_speaker_embedding(
        candidate(path), config=encoder_config, encoder=encoder, cache_root=cache_root
    )["vector"].astype(np.float64)


def cos(a, b) -> float:
    return float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def flatness_stats(path: Path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    f, t, Z = stft(a, sr, nperseg=2048, noverlap=1024)
    P = np.abs(Z) ** 2
    lo = P[(f >= 200) & (f <= 2000)].sum(0)
    voiced = lo > np.percentile(lo, 60)
    band = (f >= 4000) & (f <= 12000)
    B = P[band][:, voiced] + 1e-14
    flat_frames = np.exp(np.log(B).mean(0)) / B.mean(0)
    meter = pyloudnorm.Meter(sr)
    lufs = float(meter.integrated_loudness(a)) if len(a) > sr * 0.4 else None
    return {
        "integrated_lufs": None if lufs is None else round(lufs, 2),
        "rms_dbfs": round(20 * np.log10(np.sqrt((a**2).mean()) + 1e-12), 2),
        "peak_dbfs": round(20 * np.log10(np.abs(a).max() + 1e-12), 2),
        "flatness_4_12k_mean": round(float(flat_frames.mean()), 4),
        "flatness_4_12k_median": round(float(np.median(flat_frames)), 4),
        "flatness_4_12k_p90": round(float(np.percentile(flat_frames, 90)), 4),
        "flatness_4_12k_max": round(float(flat_frames.max()), 4),
        "duration_seconds": round(len(a) / sr, 3),
    }


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    encoder_config = load_identity_config(IDENTITY_CONFIG).encoder
    encoder = load_speaker_encoder(encoder_config)
    cache_root = OUT / "speaker_embedding_cache"

    ref_vecs = {}
    for rid, p in REFS.items():
        ref_vecs[rid] = embed(p, encoder=encoder, encoder_config=encoder_config,
                              cache_root=cache_root)

    rows = {}
    for name, p in REFS.items():
        rows[f"ref_{name}"] = {"path": str(p), **flatness_stats(p)}

    samples = sorted(RAW.glob("*.wav")) + [OUT / "webui_final" / "T1.wav"]
    for p in samples:
        if not p.is_file():
            continue
        vec = embed(p, encoder=encoder, encoder_config=encoder_config,
                    cache_root=cache_root)
        row = {
            "path": str(p.relative_to(ROOT)),
            **flatness_stats(p),
            "cos_vs_R1": round(cos(vec, ref_vecs["R1"]), 4),
            "cos_vs_R2": round(cos(vec, ref_vecs["R2"]), 4),
            "cos_vs_R3": round(cos(vec, ref_vecs["R3"]), 4),
        }
        rows[p.stem] = row
        print(f"{p.stem:22s} cosR1={row['cos_vs_R1']:.3f} cosR2={row['cos_vs_R2']:.3f} "
              f"flat={row['flatness_4_12k_median']:.3f}/{row['flatness_4_12k_p90']:.3f} "
              f"lufs={row['integrated_lufs']}")

    (REPORT / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2,
                   default=lambda o: float(o)), encoding="utf-8")
    print("written:", REPORT / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
