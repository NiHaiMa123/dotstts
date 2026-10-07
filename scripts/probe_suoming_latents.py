"""Capture generated AudioVAE latents for base vs step-500 LoRA renders.

Same text/prompt/seed/steps; hooks VocoderInference.stream_step to collect
the latent windows that flow into the streaming decoder. Writes:

    outputs/diagnostics/suoming_grit_v1/latents/base_latents.npy
    outputs/diagnostics/suoming_grit_v1/latents/lora_latents.npy
    outputs/diagnostics/suoming_grit_v1/latents/{base,lora}_render.wav
    outputs/diagnostics/suoming_grit_v1/latents/latent_metrics.json
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_suoming_grit import _free_runtime, _render  # noqa: E402
from dots_tts_lab.fuxuan_batch import load_runtime  # noqa: E402
from dots_tts_lab.voice_registry import (  # noqa: E402
    DEFAULT_VOICE_REGISTRY_PATH,
    load_voice_registry,
)

OUT = ROOT / "outputs" / "diagnostics" / "suoming_grit_v1" / "latents"
PROFILE = "suoming_step500_v1"


def _latent_stats(lat: np.ndarray) -> dict:
    """lat: (T, C) latent frames."""
    delta = np.diff(lat, axis=0)
    per_ch_std = lat.std(axis=0)
    frame_delta_rms = np.sqrt((delta**2).mean(axis=1))
    # temporal FFT band ratio per channel (high half of band vs low half)
    t = lat - lat.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(t, axis=0))
    n = spec.shape[0]
    hi = spec[n // 2 :, :].mean()
    lo = spec[2 : n // 2, :].mean()
    return {
        "frames": int(lat.shape[0]),
        "per_ch_std_mean": float(per_ch_std.mean()),
        "per_ch_std_p90": float(np.quantile(per_ch_std, 0.9)),
        "frame_delta_rms_median": float(np.median(frame_delta_rms)),
        "frame_delta_rms_p90": float(np.quantile(frame_delta_rms, 0.9)),
        "temporal_hf_ratio": float(hi / max(lo, 1e-9)),
        "channel_corr_mean": float(
            np.abs(np.corrcoef(lat[: min(2000, lat.shape[0])].T)).mean()
        ),
    }


def _capture(rt, profile, registry, tag: str) -> dict:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

    captured: list[torch.Tensor] = []
    orig = VocoderInference.decode_latents

    def wrapped(self, latents, *a, **kw):
        captured.append(latents.detach().float().cpu())
        return orig(self, latents, *a, **kw)

    VocoderInference.decode_latents = wrapped
    try:
        audio = _render(rt, profile, registry, 16, OUT / f"{tag}_render.wav")
    finally:
        VocoderInference.decode_latents = orig
    # captured chunks: (B, T, C) or (B, C, T) — normalize to (T_total, C=128)
    mats = []
    for c in captured:
        t = c[0]
        if t.size(0) == 128:
            t = t.transpose(0, 1)
        mats.append(t)
    lat = torch.cat(mats, dim=0).numpy()  # (T, C)
    np.save(OUT / f"{tag}_latents.npy", lat)
    stats = _latent_stats(lat)
    print(f"[{tag}] latent frames={lat.shape[0]} stats={stats}", flush=True)
    return stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = load_voice_registry(DEFAULT_VOICE_REGISTRY_PATH)
    profile = registry.profiles[PROFILE]

    results: dict[str, object] = {}

    kw = dict(
        base_model=registry.path(profile.base_model.path),
        base_revision=profile.base_model.revision,
        precision=profile.runtime.precision,
        optimize=profile.runtime.optimize,
        max_generate_length=profile.runtime.max_generate_length,
        max_sequence_length=profile.runtime.max_sequence_length,
        vocoder_merge_steps=profile.runtime.vocoder_merge_steps,
        warmup_on_optimize=profile.runtime.warmup_on_optimize,
        merge_lora=profile.runtime.merge_lora,
    )

    rt = load_runtime(adapter=None, **kw)
    results["base"] = _capture(rt, profile, registry, "base")
    _free_runtime(rt)

    rt = load_runtime(adapter=registry.path(profile.adapter.path), **kw)
    results["lora_step500"] = _capture(rt, profile, registry, "lora")
    _free_runtime(rt)

    (OUT / "latent_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
