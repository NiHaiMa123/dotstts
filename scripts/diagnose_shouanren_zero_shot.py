"""守岸人 zero-shot / MF 盲听诊断样本生成 (DEVIN_SHOUANREN_ZERO_SHOT_LISTENING.md Phase 1-4)

直接调用 DotsTtsRuntime，不走 voice-polish；固定 seed=42。
每条样本旁边写 metadata JSON；失败记录异常不静默。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf
import torch

from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything

OUT_ROOT = ROOT / "outputs/diagnostics/shouanren_zero_shot_v1"
RAW_DIR = OUT_ROOT / "raw"
META_DIR = OUT_ROOT / "metadata"
SEED = 42
SPEAKER_SCALE = 1.5
LANGUAGE = "chinese"
TEMPLATE = "tts"

T1 = "森林里的小动物们都说，月亮最近失眠了。"
T2 = "我睡不着。风把云朵吹散了，天空太亮，我找不到做梦的枕头。"

R1 = {
    "id": "R1",
    "audio": ROOT / "data/inbox/守岸人/中立_neutral/【中立_neutral】或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因…….wav",
    "text": "或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因……",
}
R2 = {
    "id": "R2",
    "audio": ROOT / "data/work/standardized/56/56784f46414575ff5789a2857fcb764ab5dcad40ce7522790997cb9d889418cd.wav",
    "text": "欢迎回到，这片属于你的海岸。",
}
R3 = {
    "id": "R3",
    "audio": ROOT / "data/work/standardized/6b/6bdb59898ba90334918498b7a86cdbf24e8c30a06caf4b90ab17581d069610e9.wav",
    "text": "各位启程前，请先好好休养歇息一番。有什么想去的地方，尽可以逛逛。我要暂代云骑事务，无法奉陪了。",
}

MODELS = {
    "S": {
        "label": "SOAR",
        "path": ROOT / "pretrained_models/dots.tts-soar",
        "sampling": {"ode_method": "euler", "num_steps": 16, "guidance_scale": 1.2},
    },
    "M": {
        "label": "MF",
        "path": ROOT / "pretrained_models/dots.tts-mf",
        "sampling": {"ode_method": "euler", "num_steps": 4, "guidance_scale": 0.0},
    },
    "O": {
        "label": "MF1",
        "path": ROOT / "models/dots.tts-mf-1step",
        "sampling": {"ode_method": "euler", "num_steps": 1, "guidance_scale": 0.0},
    },
}

TEXTS = {"T1": T1, "T2": T2}
REFS = {"R1": R1, "R2": R2, "R3": R3}

CASES: list[dict] = []
for model_key, ref_key, text_key in [
    ("S", "R1", "T1"), ("S", "R1", "T2"), ("S", "R2", "T1"), ("S", "R2", "T2"),
    ("M", "R1", "T1"), ("M", "R1", "T2"), ("M", "R2", "T1"), ("M", "R2", "T2"),
    ("O", "R1", "T1"), ("O", "R1", "T2"), ("O", "R2", "T1"), ("O", "R2", "T2"),
]:
    CASES.append({
        "case_id": f"{model_key}-{ref_key}-{text_key}",
        "model": model_key, "ref": ref_key, "text": text_key,
        "prompt_text_override": None,
    })
# Phase 3 sanity checks
CASES.append({"case_id": "C1-O-R3-T1", "model": "O", "ref": "R3", "text": "T1", "prompt_text_override": None})
CASES.append({"case_id": "C2-O-R1-T1-xvec", "model": "O", "ref": "R1", "text": "T1",
              "prompt_text_override": "NONE_SENTINEL"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha(path: Path) -> str | None:
    cfg = path / "config.json"
    return sha256_file(cfg) if cfg.is_file() else None


def run_case(runtime: DotsTtsRuntime, case: dict) -> dict:
    ref = REFS[case["ref"]]
    target = TEXTS[case["text"]]
    sampling = dict(MODELS[case["model"]]["sampling"])
    prompt_text = ref["text"] if case["prompt_text_override"] != "NONE_SENTINEL" else None

    seed_everything(SEED)
    t0 = time.perf_counter()
    result = runtime.generate(
        text=target,
        prompt_audio_path=str(ref["audio"]),
        prompt_text=prompt_text,
        language=LANGUAGE,
        template_name=TEMPLATE,
        speaker_scale=SPEAKER_SCALE,
        **sampling,
    )
    wall = time.perf_counter() - t0

    wav = result["audio"]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    wav = wav.reshape(-1)
    sr = int(result["sample_rate"])

    out_path = RAW_DIR / f"{case['case_id']}.wav"
    sf.write(out_path, wav, sr, subtype="PCM_24")
    out_sha = sha256_file(out_path)

    return {
        "case_id": case["case_id"],
        "status": "ok",
        "model_label": MODELS[case["model"]]["label"],
        "model_path": str(MODELS[case["model"]]["path"]),
        "model_config_sha256": config_sha(MODELS[case["model"]]["path"]),
        "prompt_audio": str(ref["audio"]),
        "prompt_audio_sha256": sha256_file(ref["audio"]),
        "prompt_text": prompt_text,
        "target_text": target,
        "seed": SEED,
        "speaker_scale": SPEAKER_SCALE,
        "sampling_args": sampling,
        "sample_rate": sr,
        "duration_seconds": round(len(wav) / sr, 4),
        "wall_seconds": round(wall, 3),
        "rtf": round(wall / max(len(wav) / sr, 1e-6), 4),
        "output_path": str(out_path.relative_to(ROOT)),
        "output_sha256": out_sha,
    }


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    runtimes: dict[str, DotsTtsRuntime] = {}
    results = []
    try:
        for case in CASES:
            mk = case["model"]
            if mk not in runtimes:
                mpath = MODELS[mk]["path"]
                if not (mpath / "model.safetensors").is_file():
                    meta = {"case_id": case["case_id"], "status": "blocked",
                            "error": f"missing model weights: {mpath / 'model.safetensors'}"}
                    (META_DIR / f"{case['case_id']}.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    results.append(meta)
                    print(f"BLOCKED {case['case_id']}: missing weights")
                    continue
                runtimes[mk] = DotsTtsRuntime.from_pretrained(str(mpath), precision="bfloat16")
            try:
                meta = run_case(runtimes[mk], case)
            except Exception as exc:  # noqa: BLE001 - 失败必须落盘
                meta = {"case_id": case["case_id"], "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8)}
            (META_DIR / f"{case['case_id']}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(meta)
            print(meta["status"].upper(), case["case_id"],
                  f"{meta.get('duration_seconds','-')}s rtf={meta.get('rtf','-')}")
    finally:
        for rt in runtimes.values():
            del rt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDONE: {ok}/{len(results)} ok")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
