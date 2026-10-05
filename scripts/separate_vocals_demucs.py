"""Chunked vocal extraction for long recordings (bounded memory, original untouched)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="htdemucs_ft")
    parser.add_argument("--chunk-sec", type=float, default=60.0)
    parser.add_argument("--ctx-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import julius
    from demucs.api import Separator

    src = args.input.resolve()
    info = sf.info(str(src))
    sr_in, channels, total = info.samplerate, info.channels, info.frames
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.vocals.flac"
    part_path = out_path.with_suffix(".part")
    prov_path = out_dir / f"{src.stem}.vocals.provenance.json"

    print(f"[probe] {src.name}: {total / sr_in:.1f}s {sr_in}Hz {channels}ch", flush=True)
    print(f"[hash] sha256 ...", flush=True)
    src_sha = sha256_file(src)
    print(f"[hash] {src_sha}", flush=True)

    separator = Separator(model=args.model, device=args.device, split=True, overlap=0.25, progress=False)
    sr_model = separator.samplerate
    core = int(args.chunk_sec * sr_in)
    ctx = int(args.ctx_sec * sr_in)
    n_chunks = (total + core - 1) // core
    stem_key = "vocals"
    assert stem_key in separator._model.sources, separator._model.sources

    t0 = time.time()
    with sf.SoundFile(str(src), "r") as fin, sf.SoundFile(
        str(part_path), "w", samplerate=sr_in, channels=channels, format="FLAC", subtype="PCM_24"
    ) as fout:
        for i in range(n_chunks):
            a, b = i * core, min((i + 1) * core, total)
            ra, rb = max(a - ctx, 0), min(b + ctx, total)
            fin.seek(ra)
            block = fin.read(rb - ra, dtype="float32", always_2d=True).T
            wav = torch.from_numpy(block)
            _, stems = separator.separate_tensor(wav, sr_in)
            voc = stems[stem_key]
            lead = a - ra
            trail = rb - b
            m0 = int(round(lead / sr_in * sr_model))
            m1 = voc.shape[1] - int(round(trail / sr_in * sr_model))
            voc = voc[:, m0:m1]
            voc = julius.resample_frac(voc, sr_model, sr_in)
            need = b - a
            if voc.shape[1] >= need:
                voc = voc[:, :need]
            else:
                voc = torch.nn.functional.pad(voc, (0, need - voc.shape[1]))
            fout.write(voc.T.cpu().numpy().astype(np.float32))
            done = min((i + 1) * core, total)
            rate = done / (time.time() - t0)
            eta = (total - done) / max(rate, 1e-9)
            print(f"[chunk] {i + 1}/{n_chunks} {done / sr_in:.0f}s eta {eta:.0f}s", flush=True)
            del wav, stems, voc, block
            if args.device == "cuda":
                torch.cuda.empty_cache()

    part_path.replace(out_path)
    provenance = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION_VERSION,
        "operation": "demucs_vocal_separation",
        "model": args.model,
        "demucs_version": __import__("demucs").__version__,
        "source_path": str(src),
        "source_sha256": src_sha,
        "source_duration_seconds": total / sr_in,
        "source_sample_rate": sr_in,
        "output_path": str(out_path),
        "output_sample_rate": sr_in,
        "chunk_seconds": args.chunk_sec,
        "context_seconds": args.ctx_sec,
        "split": True,
        "overlap": 0.25,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
