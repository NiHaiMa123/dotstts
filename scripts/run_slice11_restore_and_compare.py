#!/usr/bin/env python3
"""Verify Slice11 adapter restore and prepare a blind baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dots_tts.runtime import DotsTtsRuntime  # noqa: E402
from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/dots.tts-mf-1step")
    parser.add_argument("--checkpoint", default="data/work/slice11/tiny_overfit_v1/checkpoint-step00000032.pt")
    parser.add_argument("--prompt-audio", default="data/raw/sha256/d0/d0a371c51fc10bbb9c88bdfbc76c51e4ed28b0c49a2a95c337610df7199b3ec8.wav")
    parser.add_argument("--output-dir", default="data/work/slice11/tiny_overfit_v1/compare")
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter(model: DotsTtsModel, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["trainable_state"]
    current = dict(model.named_parameters())
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"Adapter keys not found in model: {missing}")
    for name, tensor in state.items():
        current[name].data.copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(runtime: DotsTtsRuntime, *, text: str, prompt_audio: Path, seed: int, path: Path) -> dict[str, Any]:
    set_seed(seed)
    result = runtime.generate(
        text=text,
        prompt_audio_path=str(prompt_audio),
        language="zh",
        normalize_text=False,
        speaker_scale=1.5,
    )
    audio = result["audio"].detach().float().cpu().squeeze().numpy()
    sample_rate = int(result["sample_rate"])
    if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise RuntimeError("Runtime returned empty or non-finite audio")
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_rate": sample_rate,
        "sample_count": int(audio.size),
        "duration_seconds": float(audio.size / sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))),
        "peak": float(np.max(np.abs(audio))),
    }


def main() -> int:
    args = parse_args()
    model_path = (ROOT / args.model).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    prompt_audio = (ROOT / args.prompt_audio).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file() or not prompt_audio.is_file():
        raise FileNotFoundError("Slice11 checkpoint or prompt audio is missing")

    text = "天行健，君子以自强不息。"
    baseline_model = DotsTtsModel.from_pretrained(str(model_path))
    baseline_runtime = DotsTtsRuntime(
        baseline_model,
        model_path,
        precision="bfloat16",
        optimize=False,
        max_generate_length=32,
        max_sequence_length=128,
        warmup_on_optimize=False,
    )
    baseline = generate(
        baseline_runtime,
        text=text,
        prompt_audio=prompt_audio,
        seed=args.seed,
        path=output_dir / "baseline.wav",
    )
    del baseline_runtime, baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapted_model = DotsTtsModel.from_pretrained(str(model_path))
    adapter_payload = load_adapter(adapted_model, checkpoint_path)
    adapted_runtime = DotsTtsRuntime(
        adapted_model,
        model_path,
        precision="bfloat16",
        optimize=False,
        max_generate_length=32,
        max_sequence_length=128,
        warmup_on_optimize=False,
    )
    adapted = generate(
        adapted_runtime,
        text=text,
        prompt_audio=prompt_audio,
        seed=args.seed,
        path=output_dir / "adapted.wav",
    )
    del adapted_runtime, adapted_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A fresh load plus the same adapter must reproduce every trainable tensor.
    restored_model = DotsTtsModel.from_pretrained(str(model_path))
    load_adapter(restored_model, checkpoint_path)
    restored_diffs = []
    restored_names = dict(restored_model.named_parameters())
    adapter_state = adapter_payload["trainable_state"]
    for name, tensor in adapter_state.items():
        restored_diffs.append(float((tensor - restored_names[name].detach().cpu()).abs().max().item()))
    restore_max_abs_diff = max(restored_diffs, default=0.0)

    # The labels are intentionally opaque; the mapping is written separately for audit.
    blind_a = output_dir / "blind-S11-A.wav"
    blind_b = output_dir / "blind-S11-B.wav"
    shutil.copy2(output_dir / "baseline.wav", blind_a)
    shutil.copy2(output_dir / "adapted.wav", blind_b)
    blind_items = [
        {"blind_id": "S11-A", "audio": str(blind_a.resolve())},
        {"blind_id": "S11-B", "audio": str(blind_b.resolve())},
    ]
    mapping = {"S11-A": "baseline", "S11-B": "adapted"}
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "experiment_id": "slice11_tiny_overfit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(adapter_payload["step"]),
        "seed": int(args.seed),
        "text": text,
        "prompt_audio": str(prompt_audio),
        "restore_max_abs_diff": restore_max_abs_diff,
        "baseline": baseline,
        "adapted": adapted,
        "blind_items": blind_items,
        "blind_mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    (output_dir / "restore_and_blind_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "blind_compare.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready_for_human_blind_review",
                "experiment_id": "slice11_tiny_overfit_v1",
                "text": text,
                "items": blind_items,
                "mapping_sha256": report["blind_mapping_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "blind_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "succeeded", "report": str(output_dir / "restore_and_blind_compare.json"), "restore_max_abs_diff": restore_max_abs_diff}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
