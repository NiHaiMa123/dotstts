"""Suoming "grit" root-cause isolation diagnostic.

Runs the fixed case matrix from DEVIN_SUOMING_GRIT_DIAGNOSTIC.md:
raw LoRA/base renders at varied steps/precision/optimize, the current v11
production chain, a minimal-processing control, plus stabilizer/ambience
isolation checks on the validation reference. Writes WAVs, metrics.json,
figures, and a listening page under outputs/diagnostics/suoming_grit_v1/.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dots_tts_lab.fuxuan_batch import load_runtime, render_segments, run_batch
from dots_tts_lab.grit_diagnostics import compute_grit_metrics
from dots_tts_lab.postprocess import (
    VoicePolishConfig,
    apply_fuxuan_voice_polish,
    load_edge_trim_config,
    load_voice_polish_config,
)
from dots_tts_lab.voice_registry import (
    DEFAULT_VOICE_REGISTRY_PATH,
    load_voice_registry,
)

OUT = ROOT / "outputs/diagnostics/suoming_grit_v1"
LISTEN = OUT / "listen"
REF_WAV = ROOT / (
    "data/work/standardized/fb/"
    "fb8e3c080298ab2d0a404789e6973d68cf5af7bfadb8486851fae677f1aac1ca.wav"
)
TARGET_TEXT = (
    "午后凉风拂过，雨云渐聚，细雨敲在屋檐上。我坐在窗边听雨，"
    "拭去玄朱锁上的薄尘。这确实是有些凄迷的场景，但……我很喜欢。"
)

MINIMAL_POLISH = {
    "schema_version": 1,
    "config_id": "grit_minimal_control",
    "config_version": 1,
    "low_cut_hz": 100.0,
    "brightness_crossover_hz": 6000.0,
    "brightness_gain_db": 0.0,
    "body_eq": {"center_hz": 340.0, "gain_db": 0.0, "q": 0.9},
    "presence_eq": {"center_hz": 3200.0, "gain_db": 0.0, "q": 1.0},
    "parallel_compressor": {
        "threshold_dbfs": -24.0, "ratio": 3.0, "attack_ms": 12.0,
        "release_ms": 140.0, "makeup_gain_db": 0.0, "mix": 0.0,
        "gate_dbfs": -45.0, "frame_ms": 10.0,
    },
    "deesser": {
        "low_hz": 5000.0, "high_hz": 10000.0, "threshold_dbfs": -30.0,
        "ratio": 2.0, "maximum_reduction_db": 0.0, "attack_ms": 5.0,
        "release_ms": 80.0, "frame_ms": 10.0,
    },
    "match_input_loudness": True,
    "maximum_loudness_adjustment_db": 6.0,
    "true_peak_ceiling_dbtp": -1.5,
    "true_peak_oversample": 4,
}


def _free_runtime(runtime) -> None:
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _render(rt, profile, registry, num_steps: int, out_path: Path) -> np.ndarray:
    audio, rate = render_segments(
        rt,
        [TARGET_TEXT],
        edge_trim_config=load_edge_trim_config(
            registry.path(profile.postprocess.edge_trim_config)
        ),
        base_seed=profile.generation.base_seed,
        pause_ms=profile.generation.pause_ms,
        num_steps=num_steps,
        prompt_audio_path=registry.path(profile.prompt.audio_path),
        prompt_text=profile.prompt.text,
        language=profile.generation.language,
        template_name=profile.generation.template_name,
        normalize_text=profile.generation.normalize_text,
        soften_emphasis=profile.generation.soften_emphasis,
        speaker_scale=profile.generation.speaker_scale,
        ode_method=profile.generation.ode_method,
        guidance_scale=profile.generation.guidance_scale,
        cancelled=None,
    )
    sf.write(out_path, audio, rate, subtype="PCM_24")
    return audio


def _write_case(name: str, audio: np.ndarray, rate: int, metrics: dict) -> None:
    path = LISTEN / f"{name}.wav"
    sf.write(path, audio, rate, subtype="PCM_24")
    metrics[name] = compute_grit_metrics(audio, rate)


def _mean_log_spectrum(audio: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(librosa.stft(audio, n_fft=4096, hop_length=512)) ** 2
    return librosa.fft_frequencies(sr=rate, n_fft=4096), spec.mean(axis=1)


def main() -> int:
    LISTEN.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "target_text": TARGET_TEXT,
        "fixed": {
            "base_seed": 42, "guidance_scale": 1.2, "speaker_scale": 1.5,
            "language": "chinese", "template": "tts", "normalize_text": False,
            "ode_method": "euler", "soften_emphasis": True,
        },
        "voiced_frame_rule": (
            "n_fft=2048 hop=512 @48kHz; voiced = yin f0 in 80-400Hz "
            "AND rms > 40th percentile"
        ),
        "cases": {},
    }
    registry = load_voice_registry(DEFAULT_VOICE_REGISTRY_PATH)
    profile = registry.profiles["suoming_step500_v1"]
    cases: dict[str, object] = metrics["cases"]

    ref_audio, ref_rate = sf.read(REF_WAV, dtype="float64")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    shutil.copyfile(REF_WAV, LISTEN / "00_reference_validation.wav")
    cases["REF"] = compute_grit_metrics(
        librosa.resample(ref_audio, orig_sr=ref_rate, target_sr=48000)
        if ref_rate != 48000 else ref_audio, 48000
    )
    print("[diag] REF measured", flush=True)

    # --- LoRA runtime: cases 1, 2, 5 share BF16 + optimize ---
    lora_cases_pending = not all(
        (LISTEN / f"{n}.wav").is_file()
        for n in (
            "01_lora16_bf16_opt_raw",
            "02_lora32_bf16_opt_raw",
            "05_lora16_current_v11",
        )
    )
    if lora_cases_pending:
        rt = load_runtime(
            base_model=registry.path(profile.base_model.path),
            base_revision=profile.base_model.revision,
            adapter=registry.path(profile.adapter.path),
            precision=profile.runtime.precision,
            optimize=profile.runtime.optimize,
            max_generate_length=profile.runtime.max_generate_length,
            max_sequence_length=profile.runtime.max_sequence_length,
            vocoder_merge_steps=profile.runtime.vocoder_merge_steps,
            warmup_on_optimize=profile.runtime.warmup_on_optimize,
            merge_lora=profile.runtime.merge_lora,
        )
        for name, steps in (
            ("01_lora16_bf16_opt_raw", 16),
            ("02_lora32_bf16_opt_raw", 32),
        ):
            if not (LISTEN / f"{name}.wav").is_file():
                _render(rt, profile, registry, steps, LISTEN / f"{name}.wav")
        if not (LISTEN / "05_lora16_current_v11.wav").is_file():
            in_dir, out_dir = OUT / "case5_in", OUT / "case5_out"
            in_dir.mkdir(parents=True, exist_ok=True)
            (in_dir / "05.txt").write_text(TARGET_TEXT, encoding="utf-8")
            summary = run_batch(
                rt,
                input_dir=in_dir,
                output_dir=out_dir,
                edge_trim_config=load_edge_trim_config(
                    registry.path(profile.postprocess.edge_trim_config)
                ),
                base_seed=profile.generation.base_seed,
                pause_ms=profile.generation.pause_ms,
                max_chars=profile.generation.max_chars,
                num_steps=profile.generation.num_steps,
                prompt_audio_path=registry.path(profile.prompt.audio_path),
                prompt_text=profile.prompt.text,
                language=profile.generation.language,
                template_name=profile.generation.template_name,
                normalize_text=profile.generation.normalize_text,
                soften_emphasis=profile.generation.soften_emphasis,
                speaker_scale=profile.generation.speaker_scale,
                ode_method=profile.generation.ode_method,
                guidance_scale=profile.generation.guidance_scale,
                force=True,
                voice_polish_config=load_voice_polish_config(
                    registry.path(profile.postprocess.voice_polish_config)
                ),
            )
            produced = next(out_dir.glob("*.wav"))
            shutil.copyfile(produced, LISTEN / "05_lora16_current_v11.wav")
        _free_runtime(rt)
    for name in (
        "01_lora16_bf16_opt_raw",
        "02_lora32_bf16_opt_raw",
        "05_lora16_current_v11",
    ):
        _write_case(
            name, sf.read(LISTEN / f"{name}.wav", dtype="float64")[0], 48000, cases
        )
    print("[diag] cases 1/2/5 measured", flush=True)

    # --- Case 3: FP32 eager is impractically slow on RTX 5080 (sm_120):
    # no TF32 tensor-core path by default; a single ~10s utterance exceeded
    # >10 min at 100% GPU during a prior attempt, so record BLOCKED per task
    # spec and run the BF16-eager auxiliary case instead. ---
    case3_status = (
        "BLOCKED: FP32 eager exceeded practical runtime on RTX 5080 "
        "(sm_120, no TF32 path engaged; >10min for one utterance). "
        "Ran BF16-eager auxiliary case 03b instead."
    )
    rt3 = None
    if not (LISTEN / "03b_lora32_bf16_eager_raw.wav").is_file():
        rt3 = load_runtime(
            base_model=registry.path(profile.base_model.path),
            base_revision=profile.base_model.revision,
            adapter=registry.path(profile.adapter.path),
            precision="bfloat16",
            optimize=False,
            max_generate_length=profile.runtime.max_generate_length,
            max_sequence_length=profile.runtime.max_sequence_length,
            vocoder_merge_steps=profile.runtime.vocoder_merge_steps,
            warmup_on_optimize=False,
            merge_lora=profile.runtime.merge_lora,
        )
        _render(
            rt3, profile, registry, 32,
            LISTEN / "03b_lora32_bf16_eager_raw.wav",
        )
    _write_case(
        "03b_lora32_bf16_eager_raw",
        sf.read(LISTEN / "03b_lora32_bf16_eager_raw.wav", dtype="float64")[0],
        48000,
        cases,
    )
    cases["_case3_status"] = case3_status
    print(f"[diag] case3 status: {case3_status}", flush=True)
    if rt3 is not None:
        _free_runtime(rt3)

    # --- Case 4: base SOAR without LoRA ---
    if not (LISTEN / "04_base32_bf16_raw.wav").is_file():
        rt4 = load_runtime(
            base_model=registry.path(profile.base_model.path),
            base_revision=profile.base_model.revision,
            adapter=None,
            precision="bfloat16",
            optimize=True,
            max_generate_length=profile.runtime.max_generate_length,
            max_sequence_length=profile.runtime.max_sequence_length,
            vocoder_merge_steps=profile.runtime.vocoder_merge_steps,
            warmup_on_optimize=True,
        )
        _render(rt4, profile, registry, 32, LISTEN / "04_base32_bf16_raw.wav")
        _free_runtime(rt4)
    _write_case(
        "04_base32_bf16_raw",
        sf.read(LISTEN / "04_base32_bf16_raw.wav", dtype="float64")[0],
        48000,
        cases,
    )
    print("[diag] case4 done", flush=True)

    # --- Case 6: minimal chain on the raw case with lowest temporal delta ---
    raw_names = [
        k for k in cases
        if isinstance(cases[k], dict) and ("raw" in k) and ("base" not in k)
    ]
    best_raw_name = min(
        raw_names,
        key=lambda k: cases[k]["temporal_delta_2k_9k_median_db"],
    )
    best_audio, _ = sf.read(LISTEN / f"{best_raw_name}.wav", dtype="float64")
    minimal_cfg = VoicePolishConfig.model_validate(MINIMAL_POLISH, strict=True)
    minimal_out, _ = apply_fuxuan_voice_polish(
        best_audio, 48000, minimal_cfg, calibration_gain_db=0.0
    )
    _write_case("06_best_raw_minimal", minimal_out, 48000, cases)
    cases["_case6_source"] = best_raw_name
    print(f"[diag] case6 from {best_raw_name}", flush=True)

    # --- Isolation checks on the clean reference ---
    v11 = load_voice_polish_config(
        registry.path(profile.postprocess.voice_polish_config)
    )
    ref48 = (
        librosa.resample(ref_audio, orig_sr=ref_rate, target_sr=48000)
        if ref_rate != 48000 else ref_audio
    )
    checks: dict[str, object] = {}

    stab_only = v11.model_copy(update={
        "brightness_gain_db": 0.0,
        "body_eq": v11.body_eq.model_copy(update={"gain_db": 0.0}),
        "presence_eq": v11.presence_eq.model_copy(update={"gain_db": 0.0}),
        "parallel_compressor": v11.parallel_compressor.model_copy(
            update={"mix": 0.0}
        ),
        "deesser": v11.deesser.model_copy(update={"maximum_reduction_db": 0.0}),
        "ambience": None,
        "exciter": None,
        "target_loudness_lufs": None,
        "prompt_loudness_calibration": None,
        "match_input_loudness": True,
        "low_cut_hz": None,
    })
    sf.write(OUT / "stabilizer_reference_before.wav", ref48, 48000,
             subtype="PCM_24")
    after, _ = apply_fuxuan_voice_polish(
        ref48, 48000, stab_only, calibration_gain_db=0.0
    )
    sf.write(OUT / "stabilizer_reference_after.wav", after, 48000,
             subtype="PCM_24")
    checks["stabilizer_only"] = {
        "before": compute_grit_metrics(ref48, 48000),
        "after": compute_grit_metrics(after, 48000),
    }
    print("[diag] stabilizer check done", flush=True)

    amb_only = v11.model_copy(update={
        "brightness_gain_db": 0.0,
        "body_eq": v11.body_eq.model_copy(update={"gain_db": 0.0}),
        "presence_eq": v11.presence_eq.model_copy(update={"gain_db": 0.0}),
        "parallel_compressor": v11.parallel_compressor.model_copy(
            update={"mix": 0.0}
        ),
        "deesser": v11.deesser.model_copy(update={"maximum_reduction_db": 0.0}),
        "stabilizer": None,
        "exciter": None,
        "target_loudness_lufs": None,
        "prompt_loudness_calibration": None,
        "match_input_loudness": True,
        "low_cut_hz": None,
    })
    sf.write(OUT / "ambience_reference_before.wav", ref48, 48000,
             subtype="PCM_24")
    after_amb, _ = apply_fuxuan_voice_polish(
        ref48, 48000, amb_only, calibration_gain_db=0.0
    )
    sf.write(OUT / "ambience_reference_after.wav", after_amb, 48000,
             subtype="PCM_24")
    checks["ambience_only"] = {
        "before": compute_grit_metrics(ref48, 48000),
        "after": compute_grit_metrics(after_amb, 48000),
    }
    metrics["isolation_checks"] = checks
    print("[diag] ambience check done", flush=True)

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, color in (
        ("REF", "black"),
        ("01_lora16_bf16_opt_raw", "tab:blue"),
        ("02_lora32_bf16_opt_raw", "tab:green"),
        ("04_base32_bf16_raw", "tab:orange"),
    ):
        src = ref48 if name == "REF" else sf.read(
            LISTEN / f"{name}.wav", dtype="float64"
        )[0]
        f, p = _mean_log_spectrum(src, 48000)
        ax.plot(f, 10 * np.log10(p + 1e-20), label=name, color=color, lw=1)
    ax.set_xlim(0, 20000); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title("Mean log spectrum"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_mean_spectrum.png", dpi=110)
    plt.close(fig)

    fig2, axes = plt.subplots(3, 1, figsize=(14, 9), sharey=True)
    for axx, name in zip(
        axes, ("REF", "01_lora16_bf16_opt_raw", "05_lora16_current_v11")
    ):
        src = ref48 if name == "REF" else sf.read(
            LISTEN / f"{name}.wav", dtype="float64"
        )[0]
        sgram = librosa.amplitude_to_db(
            np.abs(librosa.stft(src, n_fft=2048, hop_length=256)), ref=np.max
        )
        librosa.display.specshow(
            sgram, sr=48000, hop_length=256, x_axis="time", y_axis="hz",
            ax=axx, cmap="magma", vmax=0, vmin=-70,
        )
        axx.set_ylim(4000, 12000); axx.set_title(name)
    fig2.tight_layout(); fig2.savefig(OUT / "fig_spectrogram_4_12k.png", dpi=110)
    plt.close(fig2)
    print("[diag] figures done", flush=True)

    # --- Listening page ---
    case_desc = {
        "00_reference_validation": "验证集原声（无处理）",
        "01_lora16_bf16_opt_raw": "LoRA / Euler16 / BF16 / optimize / 无polish",
        "02_lora32_bf16_opt_raw": "LoRA / Euler32 / BF16 / optimize / 无polish",
        "03_lora32_fp32_eager_raw": "LoRA / Euler32 / FP32 / eager / 无polish",
        "03b_lora32_bf16_eager_raw": "LoRA / Euler32 / BF16 / eager / 无polish（FP32备选）",
        "04_base32_bf16_raw": "官方SOAR无LoRA / Euler32 / BF16 / optimize / 无polish",
        "05_lora16_current_v11": "当前正式路径（v11全链）",
        "06_best_raw_minimal": f"极简链（{best_raw_name} + 100Hz低切+响度+真峰）",
    }
    rows = []
    for wav in sorted(LISTEN.glob("*.wav")):
        desc = case_desc.get(wav.stem, wav.stem)
        rows.append(
            f"""<div class="case"><h3>{wav.stem}</h3><p>{desc}</p>
<audio controls src="{wav.name}"></audio>
<div class="scores">
<label>清澈度 <input type="number" min="1" max="5"></label>
<label>磨砂/颗粒 <input type="number" min="1" max="5"></label>
<label>像锁暝 <input type="number" min="1" max="5"></label>
<label>备注 <input type="text" size="40"></label>
</div></div>"""
        )
    (LISTEN / "index.html").write_text(
        "<html><meta charset='utf-8'><body><h2>锁暝磨砂诊断试听包</h2>"
        + "\n".join(rows)
        + "<style>.case{margin:18px 0;padding:12px;border:1px solid #ccc}"
        ".scores label{margin-right:14px}input[type=number]{width:56px}</style>"
        "</body></html>",
        encoding="utf-8",
    )

    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[diag] all done → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
