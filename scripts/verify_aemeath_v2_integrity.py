#!/usr/bin/env python3
"""Post-freeze integrity verification for datasets/aemeath/v2.

Two advisory checks over the frozen selection (train/val/test jsonl):

1. FX scan — flags baked-in scene effects that would hurt TTS training:
   - comms/bandpass filter (energy collapsed to the 300-3400 Hz band)
   - long reverberant tail (slow energy decay after the final voiced frame)
   Metrics are corpus-relative: items beyond a robust z cutoff are flagged,
   so the detector adapts to this speaker's own recording conditions.

2. ASR agreement — transcribes each clip with faster-whisper and compares
   against the unpacked game text. The unpack text stays authoritative;
   mismatches are reported for triage, not auto-corrected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


def _norm_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        c for c in t if not unicodedata.category(c).startswith(("P", "Z", "C"))
    )


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad <= 0:
        mad = float(np.std(values)) or 1.0
    return (values - median) / mad


def fx_metrics(path: Path) -> dict[str, float]:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < sr // 4:
        return {"skip": 1.0}

    spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    total = float(spec.sum()) or 1e-12
    low = float(spec[freqs < 300].sum()) / total
    high = float(spec[freqs > 3400].sum()) / total
    mid = float(spec[(freqs >= 300) & (freqs <= 3400)].sum()) / total

    # energy envelope (10 ms hop)
    hop = sr // 100
    frames = len(data) // hop
    env = np.sqrt(
        np.mean(data[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12
    )
    env_db = 20 * np.log10(env + 1e-9)
    voiced = env_db > env_db.max() - 35
    if voiced.sum() < 10:
        return {"skip": 1.0}
    last_voiced = int(np.nonzero(voiced)[0][-1])
    tail = env_db[last_voiced:]
    tail_len_s = len(tail) * hop / sr
    # decay slope of the tail; a shallow slope over a long tail means reverb
    if len(tail) > 3:
        slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    else:
        slope = -120.0
    silence_tail_ratio = float(np.mean(~voiced[last_voiced:])) if last_voiced < len(voiced) - 1 else 0.0

    # inter-word echo: autocorrelation peak of the envelope at 60-400 ms lag
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1 :]
    ac /= ac[0] or 1e-12
    lag_lo, lag_hi = 6, 40
    echo_peak = float(ac[lag_lo:lag_hi].max()) if len(ac) > lag_hi else 0.0

    return {
        "band_low_ratio": low,
        "band_mid_ratio": mid,
        "band_high_ratio": high,
        "tail_seconds": tail_len_s,
        "tail_slope_db_per_10ms": slope,
        "silence_tail_ratio": silence_tail_ratio,
        "echo_peak_60_400ms": echo_peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/aemeath/v2"))
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports/aemeath_v2_integrity"))
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--fx-z-cutoff", type=float, default=4.0)
    parser.add_argument("--cer-flag", type=float, default=0.30)
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for split in ("train", "validation", "test"):
        path = args.dataset_root / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["split"] = split
                items.append(rec)
    print(f"items: {len(items)}")

    # --- FX scan -------------------------------------------------------------
    metrics = []
    for item in items:
        m = fx_metrics(Path(item["audio"]))
        m.update({"fid": item["fid"], "split": item["split"], "text": item["text"]})
        metrics.append(m)

    valid = [m for m in metrics if not m.get("skip")]
    flagged_fx = []
    zscores: dict[str, np.ndarray] = {}
    for key in ("band_mid_ratio", "tail_seconds", "tail_slope_db_per_10ms",
                "echo_peak_60_400ms"):
        vals = np.array([m[key] for m in valid], dtype=np.float64)
        zscores[key] = _robust_z(vals)
    for m, i in zip(valid, range(len(valid))):
        if zscores["band_mid_ratio"][i] > args.fx_z_cutoff:
            flagged_fx.append({**m, "flag": "band_mid_ratio", "z": float(zscores["band_mid_ratio"][i])})
        if zscores["echo_peak_60_400ms"][i] > args.fx_z_cutoff:
            flagged_fx.append({**m, "flag": "echo_peak_60_400ms", "z": float(zscores["echo_peak_60_400ms"][i])})
        # reverb needs both a long tail AND a shallow decay slope; a long tail
        # with steep decay is just a fade-out/padded silence, not ambience
        if (
            zscores["tail_seconds"][i] > args.fx_z_cutoff
            and zscores["tail_slope_db_per_10ms"][i] > args.fx_z_cutoff
        ):
            flagged_fx.append({**m, "flag": "reverb_tail", "z": float(min(
                zscores["tail_seconds"][i], zscores["tail_slope_db_per_10ms"][i]
            ))})
    fx_by_fid = {}
    for f in flagged_fx:
        fx_by_fid.setdefault(f["fid"], []).append(
            {"metric": f["flag"], "z": round(f["z"], 2)}
        )
    fx_report = [
        {
            "fid": m["fid"],
            "split": m["split"],
            "text": m["text"],
            "flags": fx_by_fid.get(m["fid"], []),
            "metrics": {k: round(float(v), 4) for k, v in m.items() if k not in ("fid", "split", "text")},
        }
        for m in metrics
    ]
    (args.report_dir / "fx_report.json").write_text(
        json.dumps(fx_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"FX flagged: {len(fx_by_fid)}/{len(metrics)}")

    # --- ASR agreement --------------------------------------------------------
    asr_report = []
    if not args.skip_asr:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.whisper_model, device=args.device, compute_type=args.compute_type
        )
        for i, item in enumerate(items):
            segments, _ = model.transcribe(
                item["audio"], language="zh", beam_size=5,
                vad_filter=True,
            )
            hyp = "".join(seg.text for seg in segments)
            ref_n = _norm_text(item["text"])
            hyp_n = _norm_text(hyp)
            cer = (
                _levenshtein(ref_n, hyp_n) / max(len(ref_n), 1)
            )
            asr_report.append(
                {
                    "fid": item["fid"],
                    "split": item["split"],
                    "ref": item["text"],
                    "hyp": hyp,
                    "cer": round(cer, 4),
                    "match": cer <= args.cer_flag,
                }
            )
            if (i + 1) % 50 == 0:
                print(f"asr {i + 1}/{len(items)}")
        (args.report_dir / "asr_report.json").write_text(
            json.dumps(asr_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mism = [r for r in asr_report if not r["match"]]
        print(f"ASR mismatch (CER>{args.cer_flag}): {len(mism)}/{len(asr_report)}")

    summary = {
        "dataset_root": str(args.dataset_root),
        "item_count": len(items),
        "fx_flagged": [
            {"fid": m["fid"], "text": m["text"], "flags": fx_by_fid[m["fid"]]}
            for m in metrics
            if m["fid"] in fx_by_fid
        ],
        "asr_mismatches": [
            r for r in asr_report if not r["match"]
        ],
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "fx_flagged": len(fx_by_fid),
        "asr_checked": len(asr_report),
        "asr_mismatch": len(summary["asr_mismatches"]),
        "report_dir": str(args.report_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
