from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import yaml

from dots_tts_lab.long_form_contract import load_long_form_config
from dots_tts_lab.long_form_features import analyze_segment_samples


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/lab/long_form/denoise_ab_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(_json_bytes(value))
    os.replace(partial, path)


def _resolved(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


@dataclass(frozen=True)
class DenoiseABConfig:
    raw: dict[str, Any]
    path: Path
    sha256: str

    @property
    def selection(self) -> dict[str, Any]:
        return self.raw["selection"]

    @property
    def enhancement(self) -> dict[str, Any]:
        return self.raw["enhancement"]

    @property
    def work_root(self) -> Path:
        return _resolved(self.raw["paths"]["work_root"])

    @property
    def report_root(self) -> Path:
        return _resolved(self.raw["paths"]["report_root"])


def load_denoise_ab_config(path: str | Path = DEFAULT_CONFIG) -> DenoiseABConfig:
    resolved = _resolved(path)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("denoise AB config must be a mapping")
    if raw.get("schema_version") != 1 or raw.get("config_id") != "long_form_denoise_ab":
        raise ValueError("unsupported denoise AB config")
    selection = raw.get("selection") or {}
    counts = [
        int(selection.get("high_noise_floor_cases", 0)),
        int(selection.get("medium_noise_floor_cases", 0)),
        int(selection.get("clean_control_cases", 0)),
    ]
    if sum(counts) != int(selection.get("total_cases", -1)) or any(x < 0 for x in counts):
        raise ValueError("denoise AB selection counts do not match total_cases")
    enhancement = raw.get("enhancement") or {}
    if enhancement.get("backend") != "deepfilternet":
        raise ValueError("only the deepfilternet backend is supported")
    attenuation = float(enhancement.get("attenuation_limit_db", 0.0))
    if not 0.0 < attenuation <= 20.0:
        raise ValueError("attenuation_limit_db must be in (0, 20]")
    return DenoiseABConfig(raw=raw, path=resolved, sha256=_sha256(resolved))


def _manifest_complete(path: Path, manifest_sha256: str) -> bool:
    snapshot = path.parent / "reviews/review_snapshot.json"
    if not snapshot.exists():
        return False
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    return bool(payload.get("complete")) and payload.get("manifest_sha256") == manifest_sha256


def discover_candidates(
    source_root: str | Path, *, pending_sources_only: bool = True
) -> list[dict[str, Any]]:
    root = _resolved(source_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/*/refinements/*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = _sha256(manifest_path)
        if pending_sources_only and _manifest_complete(manifest_path, manifest_hash):
            continue
        for segment in payload.get("segments", []):
            audio_path = (manifest_path.parent / segment["derived_relative_path"]).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            rows.append(
                {
                    **segment,
                    "source_relative_path": payload["source_relative_path"],
                    "source_sha256": payload["source_sha256"],
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "audio_path": str(audio_path),
                    "activity_start_threshold_dbfs": payload["segmentation"][
                        "activity_start_threshold_dbfs"
                    ],
                    "activity_continue_threshold_dbfs": payload["segmentation"][
                        "activity_continue_threshold_dbfs"
                    ],
                }
            )
    return rows


def _round_robin(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["source_sha256"]), []).append(row)
    ordered_sources = sorted(groups)
    selected: list[dict[str, Any]] = []
    while ordered_sources and len(selected) < count:
        remaining: list[str] = []
        for source in ordered_sources:
            if groups[source] and len(selected) < count:
                selected.append(groups[source].pop(0))
            if groups[source]:
                remaining.append(source)
        ordered_sources = remaining
    return selected


def select_ab_cases(candidates: list[dict[str, Any]], config: DenoiseABConfig) -> list[dict[str, Any]]:
    selection = config.selection
    high_threshold = float(selection["high_noise_floor_dbfs"])
    clean_threshold = float(selection["clean_control_noise_floor_dbfs"])
    valid = [row for row in candidates if row.get("noise_floor_proxy_dbfs") is not None]
    high = sorted(
        (row for row in valid if float(row["noise_floor_proxy_dbfs"]) > high_threshold),
        key=lambda row: (-float(row["noise_floor_proxy_dbfs"]), str(row["segment_id"])),
    )
    medium = sorted(
        (
            row
            for row in valid
            if clean_threshold < float(row["noise_floor_proxy_dbfs"]) <= high_threshold
        ),
        key=lambda row: (-float(row["noise_floor_proxy_dbfs"]), str(row["segment_id"])),
    )
    clean = sorted(
        (row for row in valid if float(row["noise_floor_proxy_dbfs"]) <= clean_threshold),
        key=lambda row: (float(row["noise_floor_proxy_dbfs"]), str(row["segment_id"])),
    )
    buckets = (
        ("high", high, int(selection["high_noise_floor_cases"])),
        ("medium", medium, int(selection["medium_noise_floor_cases"])),
        ("clean_control", clean, int(selection["clean_control_cases"])),
    )
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for bucket, rows, count in buckets:
        chosen = _round_robin(rows, count)
        if len(chosen) != count:
            raise RuntimeError(f"not enough candidates for {bucket}: need {count}, found {len(chosen)}")
        for row in chosen:
            segment_id = str(row["segment_id"])
            if segment_id in used:
                raise RuntimeError("duplicate segment selected for denoise AB")
            used.add(segment_id)
            output.append({**row, "noise_bucket": bucket})
    return output


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _metrics(samples: np.ndarray, sample_rate: int, row: dict[str, Any]) -> dict[str, Any]:
    return analyze_segment_samples(
        samples,
        sample_rate=sample_rate,
        config=load_long_form_config(),
        activity_start_threshold_dbfs=float(row["activity_start_threshold_dbfs"]),
        activity_continue_threshold_dbfs=float(row["activity_continue_threshold_dbfs"]),
    )


def _build_review_html(manifest: dict[str, Any], output_path: Path) -> None:
    public = {
        "schema_version": 1,
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "items": [
            {
                "case_id": row["case_id"],
                "noise_bucket": row["noise_bucket"],
                "source_relative_path": row["source_relative_path"],
                "asr_candidate_text": row.get("asr_candidate_text") or "",
                "noise_floor_proxy_dbfs": row.get("noise_floor_proxy_dbfs"),
                "snr_proxy_db": row.get("snr_proxy_db"),
                "a": row["blind"]["a_relative_path"],
                "b": row["blind"]["b_relative_path"],
            }
            for row in manifest["cases"]
        ],
    }
    data = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>长音频降噪盲听 AB</title>
<style>body{{font:15px system-ui;margin:24px;background:#f5f7fb;color:#172033}}main{{max-width:1050px;margin:auto}}.card{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin:14px 0}}audio{{width:100%}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}button{{padding:8px 12px;margin:4px;border:1px solid #9aa7bb;border-radius:8px;background:white}}button.on{{background:#2457d6;color:white}}.status{{font-weight:700}}small{{color:#667085}}textarea{{width:100%;min-height:52px}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>长音频降噪盲听 AB</h1><p>请戴耳机比较 A/B。重点听持续底噪、人声清晰度、齿音、尾音，以及水声/金属感。A/B 顺序已随机隐藏。</p>
<p class=\"status\" id=\"summary\"></p><div id=\"items\"></div>
<button id=\"exportBtn\">导出完整审核 JSON</button></main>
<script>const DATA={data}; const key=`denoise-ab:${{DATA.manifest_sha256}}`; let saved=JSON.parse(localStorage.getItem(key)||'{{}}');
const itemsEl=document.getElementById('items');const summaryEl=document.getElementById('summary');const exportEl=document.getElementById('exportBtn');
const esc=s=>String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function save(id,value){{saved[id]={{...(saved[id]||{{}}),...value,updated_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(saved));render()}}
function render(){{let done=0;itemsEl.innerHTML=DATA.items.map((x,i)=>{{const v=saved[x.case_id]||{{}};if(v.decision)done++;return `<section class=card><h3>${{i+1}}. ${{esc(x.asr_candidate_text||x.source_relative_path)}}</h3><small>${{esc(x.noise_bucket)}} · 噪声底 ${{Number(x.noise_floor_proxy_dbfs).toFixed(1)}} dBFS · SNR ${{Number(x.snr_proxy_db).toFixed(1)}} dB</small><div class=pair><div><b>A</b><audio controls preload=metadata src=\"${{encodeURI(x.a)}}\"></audio></div><div><b>B</b><audio controls preload=metadata src=\"${{encodeURI(x.b)}}\"></audio></div></div><div>${{['a_better','b_better','same','both_bad'].map(k=>`<button class=\"${{v.decision===k?'on':''}}\" onclick=\"save('${{x.case_id}}',{{decision:'${{k}}'}})\">${{{{a_better:'A 更好',b_better:'B 更好',same:'差不多',both_bad:'都有问题'}}[k]}}</button>`).join('')}}</div><textarea placeholder=\"可选备注：水声、金属感、齿音受损等\" onchange=\"save('${{x.case_id}}',{{note:this.value}})\">${{esc(v.note||'')}}</textarea></section>`}}).join('');summaryEl.textContent=`已完成 ${{done}} / ${{DATA.items.length}}`}}
exportEl.onclick=()=>{{if(Object.keys(saved).filter(k=>saved[k].decision).length!==DATA.items.length){{alert('还有未选择的项目');return}}const payload={{schema_version:1,config_sha256:DATA.config_sha256,manifest_sha256:DATA.manifest_sha256,exported_at:new Date().toISOString(),decisions:DATA.items.map(x=>({{case_id:x.case_id,...saved[x.case_id]}}))}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));a.download='long-form-denoise-ab-decisions.json';a.click();URL.revokeObjectURL(a.href)}};render();</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def rebuild_denoise_ab_review(manifest_path: str | Path) -> dict[str, str]:
    resolved = _resolved(manifest_path)
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    output = resolved.parent / "review.html"
    _build_review_html(manifest, output)
    return {"review_path": str(output), "sha256": _sha256(output)}


def build_denoise_ab(
    *,
    source_root: str | Path = "data/work/long_form/sources",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from df.enhance import enhance, get_model_basedir, init_df, load_audio

    config = load_denoise_ab_config(config_path)
    work_root = config.work_root
    report_root = config.report_root
    work_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    for partial in list(work_root.rglob("*.partial")) + list(report_root.rglob("*.partial")):
        partial.unlink()
    state_path = work_root / "state.json"
    _atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "config_sha256": config.sha256,
            "started_at_unix": time.time(),
        },
    )
    try:
        candidates = discover_candidates(
            source_root,
            pending_sources_only=bool(config.selection["pending_sources_only"]),
        )
        selected = select_ab_cases(candidates, config)
        model_name = str(config.enhancement["model_name"])
        model_dir = Path(get_model_basedir(model_name)).resolve()
        initialized = init_df(
            str(model_dir),
            post_filter=bool(config.enhancement["post_filter"]),
            log_level="ERROR",
            log_file=None,
        )
        if len(initialized) == 4:
            model, df_state, _, epoch = initialized
        elif len(initialized) == 3:
            # DeepFilterNet 0.5.6 predates the epoch value in this public API.
            model, df_state, _ = initialized
            epoch = None
        else:
            raise RuntimeError("unsupported DeepFilterNet init_df return contract")
        cases: list[dict[str, Any]] = []
        for index, row in enumerate(selected, start=1):
            case_id = f"case_{index:03d}"
            case_root = report_root / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            raw_path = case_root / "raw.wav"
            enhanced_path = case_root / "enhanced.wav"
            shutil.copyfile(row["audio_path"], raw_path)
            audio, meta = load_audio(str(raw_path), sr=df_state.sr())
            enhanced = enhance(
                model,
                df_state,
                audio,
                pad=bool(config.enhancement["compensate_delay"]),
                atten_lim_db=float(config.enhancement["attenuation_limit_db"]),
            )
            samples = enhanced.detach().cpu().numpy().T
            sf.write(enhanced_path, samples, int(meta.sample_rate), subtype="PCM_24")
            raw_samples, raw_rate = sf.read(raw_path, dtype="float32", always_2d=True)
            enhanced_samples, enhanced_rate = sf.read(
                enhanced_path, dtype="float32", always_2d=True
            )
            if raw_rate != enhanced_rate or len(raw_samples) != len(enhanced_samples):
                raise RuntimeError("DeepFilterNet output duration or sample rate changed")
            raw_hash = _sha256(raw_path)
            enhanced_hash = _sha256(enhanced_path)
            enhanced_is_a = int(
                hashlib.sha256((row["segment_id"] + config.sha256).encode()).hexdigest(), 16
            ) % 2 == 0
            cases.append(
                {
                    "case_id": case_id,
                    "segment_id": row["segment_id"],
                    "source_relative_path": row["source_relative_path"],
                    "source_sha256": row["source_sha256"],
                    "manifest_path": row["manifest_path"],
                    "manifest_sha256": row["manifest_sha256"],
                    "noise_bucket": row["noise_bucket"],
                    "asr_candidate_text": row.get("asr_candidate_text"),
                    "noise_floor_proxy_dbfs": row.get("noise_floor_proxy_dbfs"),
                    "snr_proxy_db": row.get("snr_proxy_db"),
                    "raw": {
                        "relative_path": f"{case_id}/raw.wav",
                        "sha256": raw_hash,
                        "metrics": _metrics(raw_samples, raw_rate, row),
                    },
                    "enhanced": {
                        "relative_path": f"{case_id}/enhanced.wav",
                        "sha256": enhanced_hash,
                        "metrics": _metrics(enhanced_samples, enhanced_rate, row),
                    },
                    "blind": {
                        "a_kind": "enhanced" if enhanced_is_a else "raw",
                        "b_kind": "raw" if enhanced_is_a else "enhanced",
                        "a_relative_path": f"{case_id}/{'enhanced.wav' if enhanced_is_a else 'raw.wav'}",
                        "b_relative_path": f"{case_id}/{'raw.wav' if enhanced_is_a else 'enhanced.wav'}",
                    },
                }
            )
        manifest = {
            "schema_version": 1,
            "config_sha256": config.sha256,
            "backend": config.enhancement,
            "backend_model_epoch": epoch,
            "backend_model_tree_sha256": _tree_sha256(model_dir),
            "case_count": len(cases),
            "cases": cases,
        }
        manifest_bytes = _json_bytes(manifest)
        manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_json(report_root / "manifest.json", manifest)
        _build_review_html(manifest, report_root / "review.html")
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "awaiting_human_review",
                "config_sha256": config.sha256,
                "case_count": len(cases),
                "review_path": str(report_root / "review.html"),
                "finished_at_unix": time.time(),
            },
        )
        return {
            "status": "awaiting_human_review",
            "case_count": len(cases),
            "manifest_path": str(report_root / "manifest.json"),
            "review_path": str(report_root / "review.html"),
            "state_path": str(state_path),
        }
    except Exception as error:
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "config_sha256": config.sha256,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_unix": time.time(),
            },
        )
        raise
