from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_strict_gate import SourceSpan


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PACK_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "reference_pack_suisui_v1.yaml"
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"

NegativeKind = Literal[
    "same_actor_roleplay",
    "whispered_text",
    "nearfield_binaural",
    "event_over_speech",
    "other_speaker_normal",
]
REQUIRED_NEGATIVE_KINDS: tuple[NegativeKind, ...] = (
    "same_actor_roleplay",
    "whispered_text",
    "nearfield_binaural",
    "event_over_speech",
)
PackStatus = Literal["draft_pending_confirmation", "confirmed"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class ClipBinding(_StrictFrozenModel):
    clip_id: str = Field(pattern=SHA256_PATTERN)
    audio_relative_path: str
    audio_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sample_rate_hz: int = Field(gt=0)
    source_spans: list[SourceSpan] = Field(min_length=1)
    duration_seconds: float = Field(gt=0.0)
    text_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("audio_relative_path")
    @classmethod
    def validate_audio_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class ReferenceClip(ClipBinding):
    """A confirmed normal-voice reference: identity, style and freedom from
    events are all human-confirmed under the v1 labeling spec."""

    confirmed_review_batch_id: str = Field(min_length=1)
    confirmed_at: str = Field(min_length=1)


class NegativeExample(ClipBinding):
    kind: NegativeKind
    note: str = Field(default="")


class ReferencePack(_StrictFrozenModel):
    schema_version: Literal[1]
    pack_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    pack_version: int = Field(ge=1)
    voice_id: str = Field(min_length=1)
    status: PackStatus
    clips: list[ReferenceClip] = Field(min_length=6, max_length=12)
    negatives: list[NegativeExample] = Field(default_factory=list)
    created_from_review_batches: list[str] = Field(min_length=1)
    created_at: str = Field(min_length=1)
    labeling_spec: str

    @field_validator("labeling_spec")
    @classmethod
    def validate_spec_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_pack(self) -> "ReferencePack":
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("reference clips must be unique")
        for clip in self.clips:
            if not 3.0 <= clip.duration_seconds <= 10.0:
                raise ValueError(
                    f"reference clip {clip.clip_id[:12]} duration {clip.duration_seconds} "
                    "outside the 3-10 second contract"
                )
        if self.status == "confirmed":
            covered = {negative.kind for negative in self.negatives}
            missing = [kind for kind in REQUIRED_NEGATIVE_KINDS if kind not in covered]
            if missing:
                raise ValueError(
                    "confirmed packs require hard negatives of kinds: " + ", ".join(missing)
                )
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def pack_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_reference_pack(path: str | Path) -> ReferencePack:
    pack_path = Path(path).resolve()
    with pack_path.open("r", encoding="utf-8") as pack_file:
        payload = yaml.safe_load(pack_file)
    if not isinstance(payload, dict):
        raise ValueError(f"reference pack must be a YAML mapping: {pack_path}")
    declared_sha256 = payload.pop("pack_sha256", None)
    pack = ReferencePack.model_validate(payload, strict=True)
    if declared_sha256 is not None and declared_sha256 != pack.pack_sha256():
        raise ValueError(
            f"declared pack_sha256 {declared_sha256} does not match content "
            f"{pack.pack_sha256()}: {pack_path}"
        )
    return pack


def propose_reference_clips(
    export_manifest_path: str | Path,
    *,
    max_clips: int = 12,
    minimum_seconds: float = 3.0,
    maximum_seconds: float = 10.0,
) -> list[dict]:
    """Rank already-exported normal clips as draft reference candidates.

    Only the human re-confirmation step can turn a proposal into a pack; this
    helper never marks anything confirmed. Selection prefers mid-length clips
    spread across the source timeline so references cover several sentence
    shapes rather than one repeated phrase.
    """
    manifest_path = Path(export_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows") or manifest.get("segments") or []
    manifest_rate = (
        manifest.get("source_sample_rate_hz")
        or manifest.get("output_sample_rate_hz")
        or 48000
    )
    eligible = [
        row
        for row in rows
        if row.get("style") == "normal"
        and row.get("audio_sha256")
        and minimum_seconds <= float(row.get("duration_seconds", 0.0)) <= maximum_seconds
    ]
    eligible.sort(
        key=lambda row: (
            abs(float(row["duration_seconds"]) - 6.0),
            str(row["audio_sha256"]),
        )
    )
    chosen: list[dict] = []
    used_spans: list[tuple[int, int]] = []
    for row in eligible:
        if len(chosen) >= max_clips:
            break
        spans = [
            (int(span["source_start_frame"]), int(span["source_end_frame"]))
            for span in row.get("source_spans", [])
        ]
        if not spans:
            spans = [(int(row["source_start_frame"]), int(row["source_end_frame"]))]
        row_rate = int(row.get("source_sample_rate_hz") or manifest_rate)
        if any(
            abs(start - used_start) < 5 * row_rate
            for start, _ in spans
            for used_start, _ in used_spans
        ):
            continue
        used_spans.extend(spans)
        chosen.append(
            {
                "audio_relative_path": row["audio_relative_path"],
                "audio_sha256": row["audio_sha256"],
                "source_sha256": row["source_sha256"],
                "source_sample_rate_hz": row_rate,
                "source_spans": [
                    {
                        "source_start_frame": start,
                        "source_end_frame": end,
                    }
                    for start, end in spans
                ],
                "duration_seconds": float(row["duration_seconds"]),
                "text_sha256": row.get("text_sha256"),
                "origin_review_batch_id": row.get("review_batch_id"),
            }
        )
    return chosen


def _segment_binding(manifest: dict, segment: dict, manifest_dir: Path) -> dict:
    spans = segment.get("source_spans") or [
        {
            "source_start_frame": int(segment["source_start_frame"]),
            "source_end_frame": int(segment["source_end_frame"]),
        }
    ]
    audio_rel = (
        manifest_dir.relative_to(ROOT) / segment["derived_relative_path"]
    ).as_posix()
    text = segment.get("asr_candidate_text") or ""
    return {
        "clip_id": segment["segment_id"],
        "audio_relative_path": audio_rel,
        "audio_sha256": segment["derived_audio_sha256"],
        "source_sha256": manifest["source_sha256"],
        "source_sample_rate_hz": int(
            segment.get("derived_sample_rate_hz") or segment.get("sample_rate_hz") or 48000
        ),
        "source_spans": [
            {
                "source_start_frame": int(s["source_start_frame"]),
                "source_end_frame": int(s["source_end_frame"]),
            }
            for s in spans
        ],
        "duration_seconds": float(segment["duration_seconds"]),
        "text_sha256": (
            hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        ),
    }


def _round_robin_by_source(
    rows: list[tuple[dict, dict, dict, Path]], count: int
) -> list[tuple[dict, dict, dict, Path]]:
    groups: dict[str, list[tuple[dict, dict, dict, Path]]] = {}
    for row in rows:
        groups.setdefault(row[0]["source_sha256"], []).append(row)
    ordered = sorted(groups)
    selected: list[tuple[dict, dict, dict, Path]] = []
    while ordered and len(selected) < count:
        remaining = []
        for source in ordered:
            if groups[source] and len(selected) < count:
                selected.append(groups[source].pop(0))
            if groups[source]:
                remaining.append(source)
        ordered = remaining
    return selected


def propose_negative_candidates(
    sources_root: str | Path = "data/work/long_form/sources",
    *,
    per_kind: int = 6,
    exclude_ids: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Scout hard negatives from the auto-screened candidate pool.

    Heuristics only rank suspicious segments for human confirmation; nothing
    here asserts the label is true. Returns kind -> ranked candidate list.
    """
    root = (ROOT / sources_root).resolve()
    excluded = exclude_ids or set()
    manifests = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append((manifest, manifest_path.parent))

    def has_text(seg: dict) -> bool:
        return bool((seg.get("asr_candidate_text") or "").strip()) and int(
            seg.get("asr_word_count") or 0
        ) >= 3

    pools: dict[str, list[tuple[dict, dict, dict, Path]]] = {
        kind: [] for kind in REQUIRED_NEGATIVE_KINDS
    }
    pools["other_speaker_normal"] = []
    for manifest, manifest_dir in manifests:
        for seg in manifest.get("segments", []):
            if seg.get("segment_id") in excluded:
                continue
            style = seg.get("style_suggestion")
            spatial = set(seg.get("spatial_review_reasons") or [])
            reasons = set(seg.get("review_reasons") or [])
            row = (manifest, seg, manifest_dir, manifest_dir)
            if style == "binaural_3d" and (
                "side_dominant_audio" in spatial
                or "low_stereo_correlation" in spatial
                or "channel_level_imbalance" in spatial
                or "moving_stereo_position" in spatial
            ):
                pools["nearfield_binaural"].append(row)
            if (
                style in ("soft", "unknown")
                and has_text(seg)
                and float(seg.get("periodicity") or 1.0) < 0.5
            ):
                pools["whispered_text"].append(row)
            if (
                seg.get("speaker_cluster_is_dominant")
                and style != "normal"
                and has_text(seg)
            ):
                pools["same_actor_roleplay"].append(row)
            if (
                style == "normal"
                and has_text(seg)
                and (
                    float(seg.get("overlap_risk_proxy") or 0.0) >= 0.6
                    or float(seg.get("near_clip_ratio") or 0.0) > 0.0
                )
            ):
                pools["event_over_speech"].append(row)
            if (
                seg.get("speaker_cluster_is_small")
                or "non_dominant_speaker_cluster" in reasons
            ) and style == "normal" and has_text(seg):
                pools["other_speaker_normal"].append(row)

    sorters = {
        "nearfield_binaural": lambda m, s: (
            "moving_stereo_position" in (s.get("spatial_review_reasons") or []),
            -float(s.get("side_to_mid_db") or -99.0),
            float(s.get("stereo_correlation") or 1.0),
        ),
        "whispered_text": lambda m, s: (
            float(s.get("periodicity") or 1.0),
            -float(s.get("spectral_flatness_median") or 0.0),
        ),
        "same_actor_roleplay": lambda m, s: -abs(
            float(s.get("pitch_median_hz") or 0.0) - 200.0
        ),
        "event_over_speech": lambda m, s: -float(s.get("overlap_risk_proxy") or 0.0),
        "other_speaker_normal": lambda m, s: float(s.get("speaker_center_cosine") or 1.0),
    }
    result: dict[str, list[dict]] = {}
    for kind, rows in pools.items():
        ranked = sorted(rows, key=lambda r: (sorters[kind](r[0], r[1]), r[1]["segment_id"]))
        chosen = _round_robin_by_source(ranked, per_kind)
        result[kind] = [
            {
                **_segment_binding(manifest, seg, manifest_dir),
                "kind": kind,
                "evidence": {
                    "style_suggestion": seg.get("style_suggestion"),
                    "spatial_review_reasons": seg.get("spatial_review_reasons") or [],
                    "review_reasons": seg.get("review_reasons") or [],
                    "asr_candidate_text": seg.get("asr_candidate_text"),
                    "periodicity": seg.get("periodicity"),
                    "spectral_flatness_median": seg.get("spectral_flatness_median"),
                    "overlap_risk_proxy": seg.get("overlap_risk_proxy"),
                    "speaker_center_cosine": seg.get("speaker_center_cosine"),
                    "pitch_median_hz": seg.get("pitch_median_hz"),
                },
            }
            for manifest, seg, manifest_dir, _ in chosen
        ]
    return result


_NEGATIVE_LABELS = {
    "same_actor_roleplay": "同演员角色音",
    "whispered_text": "有字耳语",
    "nearfield_binaural": "近讲/双耳",
    "event_over_speech": "人声夹拍击",
    "other_speaker_normal": "他人正常声",
}


def build_reference_review_html(
    proposal_path: str | Path,
    output_path: str | Path,
) -> dict[str, str]:
    """Self-contained confirmation page for a draft reference proposal.

    Each clip can be confirmed as a normal reference, demoted to a typed hard
    negative, or rejected outright. Export emits the JSON contract consumed by
    ``build_long_form_reference_pack.py freeze``.
    """
    proposal_file = Path(proposal_path).resolve()
    proposal = json.loads(proposal_file.read_text(encoding="utf-8"))
    if proposal.get("status") != "draft_pending_confirmation":
        raise ValueError("only draft_pending_confirmation proposals can be reviewed")
    audio_root = proposal_file.parent
    if proposal.get("source_export_manifest"):
        audio_root = Path(proposal["source_export_manifest"]).resolve().parent
    out = Path(output_path).resolve()
    items = []
    for index, clip in enumerate(proposal["candidates"], start=1):
        source = clip["audio_relative_path"]
        resolved_audio = (audio_root / source).resolve()
        audio_src = Path(
            os.path.relpath(resolved_audio, out.parent)
        ).as_posix()
        items.append(
            {
                "index": index,
                "clip_id": clip["audio_sha256"],
                "audio_src": audio_src,
                "duration_seconds": clip["duration_seconds"],
                "source_spans": clip["source_spans"],
                "source_sample_rate_hz": clip["source_sample_rate_hz"],
                "binding": clip,
            }
        )
    public = {
        "schema_version": 1,
        "voice_id": proposal.get("voice_id"),
        "proposal_sha256": _sha256_text(proposal_file.read_text(encoding="utf-8")),
        "labeling_spec": proposal.get("labeling_spec"),
        "required_clips": {"min": 6, "max": 12},
        "required_negative_kinds": list(REQUIRED_NEGATIVE_KINDS),
        "negative_labels": _NEGATIVE_LABELS,
        "items": items,
    }
    data = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    page = _REFERENCE_REVIEW_HTML.replace("__DATA__", data)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {"review_path": str(out), "sha256": _sha256_text(page)}


def build_negative_review_html(
    candidates: dict[str, list[dict]],
    output_path: str | Path,
    *,
    sources_root: str | Path = "data/work/long_form/sources",
    save_name: str = "reference-negatives",
) -> dict[str, str]:
    """Confirmation page for scouted hard negatives.

    Every item must still be listened to: the heuristics only nominate, the
    human verdict is what turns a candidate into a labeled negative.
    """
    out = Path(output_path).resolve()
    items = []
    for kind, rows in candidates.items():
        for row in rows:
            audio_abs = (ROOT / row["audio_relative_path"]).resolve()
            items.append(
                {
                    "kind": kind,
                    "kind_label": _NEGATIVE_LABELS[kind],
                    "clip_id": row["clip_id"],
                    "audio_src": Path(
                        os.path.relpath(audio_abs, out.parent)
                    ).as_posix(),
                    "duration_seconds": row["duration_seconds"],
                    "evidence": row["evidence"],
                    "binding": {
                        key: row[key]
                        for key in (
                            "clip_id",
                            "audio_relative_path",
                            "audio_sha256",
                            "source_sha256",
                            "source_sample_rate_hz",
                            "source_spans",
                            "duration_seconds",
                            "text_sha256",
                        )
                    },
                }
            )
    public = {
        "schema_version": 1,
        "negative_labels": _NEGATIVE_LABELS,
        "items": items,
        "save_name": save_name,
    }
    data = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    page = _NEGATIVE_REVIEW_HTML.replace("__DATA__", data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {"review_path": str(out), "sha256": _sha256_text(page)}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_REFERENCE_REVIEW_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>正常参考包确认</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fb;color:#172033}main{max-width:1050px;margin:auto}.card{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin:14px 0}audio{width:100%}button{padding:8px 12px;margin:4px;border:1px solid #9aa7bb;border-radius:8px;background:white;cursor:pointer}button.on{background:#2457d6;color:white}button.neg.on{background:#b42318;color:white}.status{font-weight:700}.warn{color:#b42318}small{color:#667085}textarea{width:100%;min-height:44px;margin-top:6px}.check{color:#475467;font-size:13px}</style></head>
<body><main><h1>正常参考包确认（draft v1）</h1>
<p><b>操作：</b>1) 逐条点播放听 → 2) 每条三选一：「确认为 normal 参考」/「负例:某类型」/「排除」→ 3) 全部决定后点底部「导出确认 JSON」，文件自动保存到固定位置，告诉 Devin 即可，进度刷新不丢。</p>
<p>每条请戴耳机确认四点：① 目标演员<b>正常说话</b>声线（非角色腔/耳语/轻声/气声）② 空间安全（无近讲、双耳左右移动）③ 无事件（拍击/摩擦/音乐/笑哭/他人声）④ 完整自然句、首尾没切字。</p>
<p>不满足但可以当<b>难负例</b>的，标成对应类型；完全没用的选「排除」。冻结要求：确认 ≥6 条且四类难负例（同演员角色音/有字耳语/近讲双耳/人声夹拍击）各至少 1 条。</p>
<p class="status" id="summary"></p><p class="warn" id="problems"></p><div id="items"></div>
<button id="exportBtn">导出确认 JSON</button></main>
<script>
const DATA=__DATA__;
const key=`refpack:${DATA.proposal_sha256}`;let saved=JSON.parse(localStorage.getItem(key)||'{}');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const NL=DATA.negative_labels;
function save(id,value){saved[id]={...(saved[id]||{}),...value,updated_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(saved));render()}
function spanStr(it){return it.source_spans.map(s=>((s.source_start_frame/it.source_sample_rate_hz).toFixed(2)+'–'+(s.source_end_frame/it.source_sample_rate_hz).toFixed(2)+'s')).join(', ')}
function validate(){const clips=Object.values(saved).filter(v=>v.decision==='reference');const negs=Object.values(saved).filter(v=>v.decision==='negative');const kinds=new Set(negs.map(v=>v.negative_kind));const problems=[];if(clips.length<6)problems.push(`确认参考 ${clips.length}/6 不足`);DATA.required_negative_kinds.forEach(k=>{if(!kinds.has(k))problems.push(`缺负例:${NL[k]}`)});return problems}
function render(){let ref=0,neg=0,rej=0,und=0;document.getElementById('items').innerHTML=DATA.items.map(x=>{const v=saved[x.clip_id]||{};if(v.decision==='reference')ref++;else if(v.decision==='negative')neg++;else if(v.decision==='reject')rej++;else und++;const negBtns=Object.keys(NL).map(k=>`<button class="neg ${v.decision==='negative'&&v.negative_kind===k?'on':''}" onclick="save('${x.clip_id}',{decision:'negative',negative_kind:'${k}'})">负例:${NL[k]}</button>`).join('');return `<section class=card><h3>${x.index}. ${x.duration_seconds.toFixed(2)}s · 源 ${spanStr(x)}</h3><audio controls preload=metadata src="${encodeURI(x.audio_src)}"></audio><div><button class="${v.decision==='reference'?'on':''}" onclick="save('${x.clip_id}',{decision:'reference',negative_kind:null})">确认为 normal 参考</button>${negBtns}<button class="${v.decision==='reject'?'on':''}" onclick="save('${x.clip_id}',{decision:'reject',negative_kind:null})">排除</button></div><textarea placeholder="备注（可选）" onchange="save('${x.clip_id}',{note:this.value})">${esc(v.note||'')}</textarea></section>`}).join('');const problems=validate();document.getElementById('summary').textContent=`参考 ${ref} · 负例 ${neg} · 排除 ${rej} · 未定 ${und}`;document.getElementById('problems').textContent=und?`还有 ${und} 条未决定`:(problems.length?problems.join('；'):'可以导出')}
document.getElementById('exportBtn').onclick=()=>{const und=Object.keys(saved).filter(k=>!saved[k].decision).length+ (DATA.items.length-Object.keys(saved).length);if(und){alert(`还有 ${und} 条未决定`);return}const clips=[],negatives=[],prior=new Set();DATA.items.forEach(x=>{const v=saved[x.clip_id]||{};if(!v.decision)return;const b={...x.binding};if(b.origin_review_batch_id)prior.add(b.origin_review_batch_id);const entry={clip_id:x.clip_id,...b};if(v.decision==='reference')clips.push(entry);else if(v.decision==='negative')negatives.push({...entry,kind:v.negative_kind,note:v.note||''})});const payload={schema_version:1,review_batch_id:crypto.randomUUID(),confirmed_at:new Date().toISOString(),labeling_spec:DATA.labeling_spec,proposal_sha256:DATA.proposal_sha256,prior_batch_ids:[...prior],clips,negatives};const body=JSON.stringify(payload,null,2);fetch('/save/reference-pack-confirmations',{method:'POST',headers:{'Content-Type':'application/json'},body}).then(r=>{if(!r.ok)throw new Error();return r.json()}).then(r=>alert('已保存到固定路径：\\n'+r.saved+'\\n\\n告诉 Devin 即可')).catch(()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download='reference-pack-confirmations.json';a.click();URL.revokeObjectURL(a.href)})};render();
</script></body></html>"""


_NEGATIVE_REVIEW_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>难负例确认</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fb;color:#172033}main{max-width:1050px;margin:auto}.card{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin:14px 0}audio{width:100%}button{padding:8px 12px;margin:4px;border:1px solid #9aa7bb;border-radius:8px;background:white;cursor:pointer}button.on{background:#2457d6;color:white}button.no.on{background:#b42318;color:white}.status{font-weight:700}.warn{color:#b42318}small{color:#667085}textarea{width:100%;min-height:40px;margin-top:6px}.ev{color:#475467;font-size:13px}h2{margin-top:28px}</style></head>
<body><main><h1>难负例确认</h1>
<p><b>操作：</b>1) 按类型分组，逐条点播放听 → 2) 真的是该类负例点「确认是」，不像就点「不是」→ 3) 四类各至少确认 1 条后点「导出负例 JSON」，文件自动保存到固定位置，告诉 Devin 即可，进度刷新不丢。</p>
<p>这些条目由启发式提名，<b>必须逐条听过</b>：确认它真的是该类负例（保留）或不是（丢弃）。四类必需：同演员角色音 / 有字耳语 / 近讲双耳 / 人声夹拍击。每组卡片上方的「已确认 n」是该类已确认数量。</p>
<p class="status" id="summary"></p><p class="warn" id="problems"></p><div id="items"></div>
<button id="exportBtn">导出负例 JSON</button></main>
<script>
const DATA=__DATA__;const key='negscout:'+(DATA.save_name||'v1');let saved=JSON.parse(localStorage.getItem(key)||'{}');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ev=e=>{const parts=[];if(e.style_suggestion)parts.push('style='+e.style_suggestion);(e.spatial_review_reasons||[]).forEach(r=>parts.push(r));(e.review_reasons||[]).forEach(r=>parts.push(r));if(e.periodicity!=null)parts.push('periodicity='+Number(e.periodicity).toFixed(2));if(e.spectral_flatness_median!=null)parts.push('flat='+Number(e.spectral_flatness_median).toFixed(3));if(e.overlap_risk_proxy!=null)parts.push('overlap='+Number(e.overlap_risk_proxy).toFixed(2));if(e.speaker_center_cosine!=null)parts.push('cos='+Number(e.speaker_center_cosine).toFixed(2));if(e.pitch_median_hz!=null)parts.push('f0='+Number(e.pitch_median_hz).toFixed(0)+'Hz');return parts.join(' · ')};
function save(id,value){saved[id]={...(saved[id]||{}),...value,updated_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(saved));render()}
function render(){let yes=0,no=0,und=0;const byKind={};DATA.items.forEach(x=>{const v=saved[x.clip_id]||{};if(v.decision==='yes'){yes++;byKind[x.kind]=(byKind[x.kind]||0)+1}else if(v.decision==='no')no++;else und++});const groups={};DATA.items.forEach(x=>{(groups[x.kind]=groups[x.kind]||[]).push(x)});
document.getElementById('items').innerHTML=Object.entries(groups).map(([kind,rows])=>`<h2>${DATA.negative_labels[kind]}（已确认 ${byKind[kind]||0}）</h2>`+rows.map(x=>{const v=saved[x.clip_id]||{};return `<section class=card><h3>${x.duration_seconds.toFixed(2)}s · ${esc(x.evidence.asr_candidate_text||'(无文本)')}</h3><small class=ev>${esc(ev(x.evidence))}</small><audio controls preload=metadata src="${encodeURI(x.audio_src)}"></audio><div><button class="${v.decision==='yes'?'on':''}" onclick="save('${x.clip_id}',{decision:'yes'})">确认是「${x.kind_label}」</button><button class="no ${v.decision==='no'?'on':''}" onclick="save('${x.clip_id}',{decision:'no'})">不是</button></div><textarea placeholder="备注（可选）" onchange="save('${x.clip_id}',{note:this.value})">${esc(v.note||'')}</textarea></section>`}).join('')).join('');
const missing=['same_actor_roleplay','whispered_text','nearfield_binaural','event_over_speech'].filter(k=>!(byKind[k]>0));document.getElementById('summary').textContent=`确认 ${yes} · 不是 ${no} · 未定 ${und}`;document.getElementById('problems').textContent=missing.length?`缺负例类型：${missing.map(k=>DATA.negative_labels[k]).join('、')}`:'四类已覆盖'}
document.getElementById('exportBtn').onclick=()=>{const negatives=DATA.items.filter(x=>(saved[x.clip_id]||{}).decision==='yes').map(x=>({...x.binding,kind:x.kind,note:(saved[x.clip_id]||{}).note||''}));const payload={schema_version:1,exported_at:new Date().toISOString(),negatives};const body=JSON.stringify(payload,null,2);fetch('/save/'+(DATA.save_name||'reference-negatives'),{method:'POST',headers:{'Content-Type':'application/json'},body}).then(r=>{if(!r.ok)throw new Error();return r.json()}).then(r=>alert('已保存到固定路径：\\n'+r.saved+'\\n\\n告诉 Devin 即可')).catch(()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download=(DATA.save_name||'reference-negatives')+'.json';a.click();URL.revokeObjectURL(a.href)})};render();
</script></body></html>"""

