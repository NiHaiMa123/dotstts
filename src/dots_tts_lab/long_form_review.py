from __future__ import annotations

import hashlib
import html
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dots_tts_lab.long_form_paths import contained_file, validate_output_path


STYLE_LABELS = {
    "normal",
    "soft",
    "whisper",
    "roleplay",
    "binaural_3d",
    "singing",
    "non_speech",
    "unknown",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _review_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for index, segment in enumerate(manifest["segments"]):
        items.append(
            {
                "index": index,
                "segment_id": segment["segment_id"],
                "audio": segment["derived_relative_path"].replace("\\", "/"),
                "start_seconds": segment["source_start_frame"]
                / segment["source_scan_sample_rate_hz"]
                if "source_scan_sample_rate_hz" in segment
                else segment["source_start_frame"]
                / manifest["source_scan"]["sample_rate_hz"],
                "duration_seconds": segment["duration_seconds"],
                "asr_candidate_text": segment.get("asr_candidate_text") or "",
                "asr_mean_word_probability": segment.get("asr_mean_word_probability"),
                "style_suggestion": segment["style_suggestion"],
                "style_cluster_id": segment["style_cluster_id"],
                "speaker_cluster_id": segment["speaker_cluster_id"],
                "speaker_cluster_size": segment["speaker_cluster_size"],
                "speaker_cluster_is_dominant": segment["speaker_cluster_is_dominant"],
                "snr_proxy_db": segment.get("snr_proxy_db"),
                "stereo_correlation": segment.get("stereo_correlation"),
                "pan_standard_deviation": segment.get("pan_standard_deviation"),
                "boundary_reason": segment["boundary_reason"],
                "screening_tier": segment.get("screening_tier"),
                "ranking_score": segment.get("ranking_score"),
                "compressed_internal_pause": segment.get(
                    "compressed_internal_pause", False
                ),
                "review_reasons": segment["review_reasons"],
            }
        )
    return items


def build_long_form_review(
    manifest_path: str | Path, *, output_path: str | Path | None = None
) -> dict[str, Any]:
    resolved_manifest = Path(manifest_path).resolve()
    resolved_output = validate_output_path(
        Path(output_path) if output_path is not None else resolved_manifest.with_name("review.html"),
        protected_inputs=(resolved_manifest,),
    )
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if manifest.get("asr_status") != "succeeded":
        raise RuntimeError("review page requires a completed timestamp ASR manifest")
    items = _review_items(manifest)
    manifest_sha256 = _sha256(resolved_manifest)
    payload = {
        "schema_version": 1,
        "source_sha256": manifest["source_sha256"],
        "source_relative_path": manifest["source_relative_path"],
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": manifest_sha256,
        "refinement": manifest.get("refinement"),
        "items": items,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    styles_json = json.dumps(sorted(STYLE_LABELS), ensure_ascii=False)
    title = html.escape(f"长音频审核 · {manifest['source_relative_path']}")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;--amber:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#0d1117ee;border-bottom:1px solid var(--line);padding:14px 20px}}
h1{{font-size:19px;margin:0 0 6px}}.summary,.muted{{color:var(--muted)}}.toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
button,select,input,textarea{{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px}}
button{{cursor:pointer}}button.primary{{background:#1f6feb;border-color:#1f6feb}}button.good{{background:#238636;border-color:#238636}}
main{{max-width:1100px;margin:18px auto;padding:0 16px}}.nav{{display:grid;grid-template-columns:1fr auto auto;gap:8px;margin-bottom:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}}.badges{{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}}
.badge{{border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--muted)}}.badge.warn{{color:#f2cc60;border-color:#9e6a03}}
audio{{width:100%;margin:12px 0}}label{{display:block;margin-top:12px;color:var(--muted)}}textarea{{width:100%;min-height:88px;resize:vertical}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.status{{font-weight:700}}.saved{{color:var(--green)}}.unsaved{{color:var(--amber)}}
.decision-row{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:12px 0}}.decision-row label{{margin:0;color:var(--text)}}
.reason{{font-family:ui-monospace,Consolas,monospace;color:#f2cc60}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><h1>{title}</h1><div id="summary" class="summary"></div><div class="toolbar">
<select id="filter"><option value="all">全部</option><option value="priority">优先参考候选</option><option value="unsaved">未保存</option><option value="normal">建议 normal</option><option value="main">主说话人簇</option><option value="spatial">空间音频风险</option><option value="forced">强制切口</option></select>
<button id="importBtn">导入审核进度</button><input id="importFile" type="file" accept="application/json" hidden>
<button id="exportBtn" class="primary">导出审核决定</button></div></header>
<main><div class="nav"><select id="itemSelect"></select><button id="prev">上一条</button><button id="next">下一条</button></div>
<section class="card"><div id="position" class="muted"></div><div id="badges" class="badges"></div><audio id="audio" controls preload="none"></audio>
<div id="reasons"></div><div class="decision-row"><label><input type="radio" name="decision" value="keep"> 保留</label><label><input type="radio" name="decision" value="reject"> 排除</label><label><input type="radio" name="decision" value="undecided"> 未决定</label><label><input id="reference" type="checkbox"> 设为目标音色参考</label></div>
<div class="grid"><div><label for="style">确认风格</label><select id="style"></select></div><div><label for="note">备注</label><input id="note" type="text"></div><div><label>本条状态</label><div id="itemStatus" class="status unsaved">未保存</div></div></div>
<label for="transcript">确认文本（保留时不能为空）</label><textarea id="transcript"></textarea>
<div class="toolbar"><button id="save" class="good">保存本条</button><button id="saveNext">保存并下一条</button></div></section></main>
<script>
const DATA={data_json}; const STYLES={styles_json};
const storageKey=`long-form-review:${{DATA.source_sha256}}:${{DATA.manifest_sha256}}`;
let saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}'); let current=0; let visible=[]; let referenceId=saved.__reference_segment_id||null;
const $=id=>document.getElementById(id);
for(const style of STYLES) $('style').append(new Option(style,style));
function stateFor(item){{return saved[item.segment_id]||null}}
function matches(item,filter){{const s=stateFor(item); if(filter==='all')return true;if(filter==='priority')return item.style_suggestion==='normal'&&item.speaker_cluster_is_dominant&&item.boundary_reason!=='forced_max_duration_split'&&!item.review_reasons.some(x=>x.includes('stereo')||x.includes('spatial')||x.includes('side_'))&&(item.asr_mean_word_probability??0)>=0.75;if(filter==='unsaved')return !s;if(filter==='normal')return item.style_suggestion==='normal';if(filter==='main')return item.speaker_cluster_is_dominant;if(filter==='spatial')return item.review_reasons.some(x=>x.includes('stereo')||x.includes('spatial')||x.includes('side_'));if(filter==='forced')return item.boundary_reason==='forced_max_duration_split';return true}}
function refreshVisible(keepId){{visible=DATA.items.filter(x=>matches(x,$('filter').value)); $('itemSelect').innerHTML=''; visible.forEach((x,i)=>$('itemSelect').append(new Option(`${{i+1}} · ${{x.asr_candidate_text.slice(0,28)||'(空文本)'}}`,x.segment_id))); current=Math.max(0,visible.findIndex(x=>x.segment_id===keepId)); if(current<0)current=0; render()}}
function summary(){{const decisions=Object.entries(saved).filter(([k])=>!k.startsWith('__')).map(([,v])=>v);const keep=decisions.filter(x=>x.decision==='keep').length;const reject=decisions.filter(x=>x.decision==='reject').length;const undecided=decisions.filter(x=>x.decision==='undecided').length;const refinement=DATA.refinement?` · 候补 ${{DATA.refinement.reserve_count}} · 自动排除 ${{DATA.refinement.auto_excluded_count}}`:'';$('summary').textContent=`首轮审核 ${{DATA.items.length}}${{refinement}} · 已保存 ${{decisions.length}}/${{DATA.items.length}} · 保留 ${{keep}} · 排除 ${{reject}} · 未决定 ${{undecided}} · 目标参考 ${{referenceId?'已选择':'未选择'}}`;}}
function render(){{summary();if(!visible.length){{$('position').textContent='当前筛选没有条目';$('audio').removeAttribute('src');return}}const item=visible[current];$('itemSelect').value=item.segment_id;$('position').textContent=`第 ${{item.index+1}}/${{DATA.items.length}} 条 · 来源 ${{item.start_seconds.toFixed(2)}}s · 时长 ${{item.duration_seconds.toFixed(2)}}s`;$('badges').innerHTML=[`风格 ${{item.style_suggestion}}`,`说话人 ${{item.speaker_cluster_id}} (${{item.speaker_cluster_size}})`,`ASR ${{item.asr_mean_word_probability==null?'n/a':item.asr_mean_word_probability.toFixed(2)}}`,`边界 ${{item.boundary_reason}}`,item.compressed_internal_pause?'已压缩句内长停顿':null].filter(Boolean).map((x,i)=>`<span class="badge ${{i===3&&item.boundary_reason.includes('forced')?'warn':''}}">${{x}}</span>`).join('');$('reasons').innerHTML=item.review_reasons.length?`<div class="reason">需复核：${{item.review_reasons.join(' · ')}}</div>`:'<div class="muted">无自动风险标记</div>';$('audio').src=item.audio;const s=stateFor(item);document.querySelector(`input[name=decision][value=${{s?.decision||'undecided'}}]`).checked=true;$('style').value=s?.confirmed_style||item.style_suggestion;$('transcript').value=s?.confirmed_text??item.asr_candidate_text;$('note').value=s?.note||'';$('reference').checked=referenceId===item.segment_id;$('itemStatus').textContent=s?'已保存':'未保存';$('itemStatus').className=`status ${{s?'saved':'unsaved'}}`;}}
function save(goNext=false){{if(!visible.length)return;const item=visible[current];const decision=document.querySelector('input[name=decision]:checked').value;const text=$('transcript').value.trim();if(decision==='keep'&&!text){{alert('保留片段必须确认文本');return}}if($('reference').checked&&decision!=='keep'){{alert('目标音色参考必须同时选择“保留”');return}}saved[item.segment_id]={{segment_id:item.segment_id,decision,confirmed_style:$('style').value,confirmed_text:text,note:$('note').value.trim(),saved_at:new Date().toISOString()}};if($('reference').checked)referenceId=item.segment_id;else if(referenceId===item.segment_id)referenceId=null;saved.__reference_segment_id=referenceId;localStorage.setItem(storageKey,JSON.stringify(saved));render();if(goNext&&current<visible.length-1){{current++;render()}}}}
$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);$('prev').onclick=()=>{{if(current>0)current--;render()}};$('next').onclick=()=>{{if(current<visible.length-1)current++;render()}};$('itemSelect').onchange=e=>{{current=visible.findIndex(x=>x.segment_id===e.target.value);render()}};$('filter').onchange=()=>refreshVisible(visible[current]?.segment_id);$('reference').onchange=()=>{{if($('reference').checked){{referenceId=visible[current].segment_id}}else if(referenceId===visible[current].segment_id)referenceId=null}};
$('exportBtn').onclick=()=>{{const decisions=Object.entries(saved).filter(([k])=>!k.startsWith('__')).map(([,v])=>v);const undecided=decisions.filter(x=>x.decision==='undecided').length;const reference=decisions.find(x=>x.segment_id===referenceId);if(decisions.length!==DATA.items.length){{alert(`还有 ${{DATA.items.length-decisions.length}} 条未保存，不能导出完整审核结果`);return}}if(undecided){{alert(`还有 ${{undecided}} 条处于“未决定”，请明确选择保留或排除`);return}}if(!referenceId||!reference||reference.decision!=='keep'){{alert('请选择一条已保存为“保留”的目标音色参考');return}}const payload={{schema_version:1,review_batch_id:crypto.randomUUID(),source_sha256:DATA.source_sha256,config_sha256:DATA.config_sha256,manifest_sha256:DATA.manifest_sha256,reference_segment_id:referenceId,exported_at:new Date().toISOString(),decisions}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='long-form-review-decisions.json';a.click();URL.revokeObjectURL(a.href)}};
$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{{const p=JSON.parse(await e.target.files[0].text());if(p.source_sha256!==DATA.source_sha256||p.manifest_sha256!==DATA.manifest_sha256){{alert('审核文件与当前页面不匹配');return}}for(const d of p.decisions||[])saved[d.segment_id]=d;referenceId=p.reference_segment_id||referenceId;saved.__reference_segment_id=referenceId;localStorage.setItem(storageKey,JSON.stringify(saved));refreshVisible(visible[current]?.segment_id)}};
refreshVisible();
</script></body></html>"""
    _atomic_text(resolved_output, document)
    return {
        "schema_version": 1,
        "review_path": str(resolved_output),
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": manifest_sha256,
        "item_count": len(items),
    }


def build_long_form_review_index(
    work_root: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    resolved_work = Path(work_root).resolve()
    resolved_output = validate_output_path(output_path)
    manifests = sorted(
        resolved_work.glob("sources/*/*/refinements/*/manifest.json"),
        key=lambda path: str(path).casefold(),
    )
    rows = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        review_path = manifest_path.with_name("review.html")
        snapshot_path = manifest_path.parent / "reviews" / "review_snapshot.json"
        current_hash = _sha256(manifest_path)
        snapshot = None
        if snapshot_path.is_file():
            try:
                candidate = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if candidate.get("manifest_sha256") == current_hash:
                    snapshot = candidate
            except (OSError, json.JSONDecodeError):
                snapshot = None
        complete = bool(snapshot and snapshot.get("complete") is True)
        duration = sum(float(item["duration_seconds"]) for item in manifest["segments"])
        refinement = manifest.get("refinement") or {}
        rows.append(
            {
                "source_relative_path": manifest["source_relative_path"],
                "source_sha256": manifest["source_sha256"],
                "manifest_sha256": current_hash,
                "review_path": str(review_path),
                "review_uri": review_path.as_uri(),
                "item_count": int(manifest["segment_count"]),
                "duration_seconds": duration,
                "reserve_count": int(refinement.get("reserve_count", 0)),
                "auto_excluded_count": int(refinement.get("auto_excluded_count", 0)),
                "status": "complete" if complete else "pending",
                "reviewed_count": int(snapshot.get("reviewed_count", 0)) if snapshot else 0,
            }
        )
    rows.sort(key=lambda row: (row["status"] == "complete", row["source_relative_path"]))
    pending = [row for row in rows if row["status"] == "pending"]
    complete = [row for row in rows if row["status"] == "complete"]
    cards = []
    for row in rows:
        status_label = "已完成" if row["status"] == "complete" else "待审核"
        status_class = "done" if row["status"] == "complete" else "pending"
        cards.append(
            "<article class='card'>"
            f"<div class='status {status_class}'>{status_label}</div>"
            f"<h2>{html.escape(row['source_relative_path'])}</h2>"
            f"<p>{row['item_count']} 条 · {row['duration_seconds'] / 60:.2f} 分钟"
            f" · 候补 {row['reserve_count']} · 自动排除 {row['auto_excluded_count']}</p>"
            f"<a href='{html.escape(row['review_uri'], quote=True)}'>打开审核页</a>"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>岁岁长音频审核总览</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--amber:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:980px;margin:30px auto;padding:0 18px}}h1{{margin-bottom:6px}}.summary{{color:var(--muted);margin-bottom:20px}}
.grid{{display:grid;gap:12px}}.card{{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 20px}}
h2{{font-size:17px;margin:0 90px 6px 0}}p{{color:var(--muted);margin:0 0 9px}}a{{color:var(--blue);font-weight:700}}
.status{{position:absolute;right:18px;top:16px;border:1px solid;border-radius:999px;padding:2px 9px}}.done{{color:var(--green)}}.pending{{color:var(--amber)}}
</style></head><body><main><h1>岁岁长音频审核总览</h1>
<div class="summary">待审核 {len(pending)} 个源、{sum(row['item_count'] for row in pending)} 条、{sum(row['duration_seconds'] for row in pending) / 60:.2f} 分钟；已完成 {len(complete)} 个源。</div>
<div class="grid">{''.join(cards)}</div></main></body></html>"""
    _atomic_text(resolved_output, document)
    payload = {
        "schema_version": 1,
        "review_index_path": str(resolved_output),
        "source_count": len(rows),
        "pending_source_count": len(pending),
        "pending_item_count": sum(row["item_count"] for row in pending),
        "pending_duration_seconds": sum(row["duration_seconds"] for row in pending),
        "complete_source_count": len(complete),
        "sources": rows,
    }
    _atomic_text(
        resolved_output.with_suffix(".json"),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return payload


def apply_long_form_review(
    manifest_path: str | Path,
    decisions_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    decisions_file = Path(decisions_path).resolve()
    target_dir = validate_output_path(
        Path(output_dir) if output_dir else manifest_file.parent / "reviews",
        protected_inputs=(manifest_file, decisions_file),
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload = json.loads(decisions_file.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest_file)
    for field, expected in (
        ("source_sha256", manifest["source_sha256"]),
        ("config_sha256", manifest["config_sha256"]),
        ("manifest_sha256", manifest_sha256),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"review {field} does not match the manifest")
    batch_id = payload.get("review_batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("review_batch_id is required")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")
    segments = {item["segment_id"]: item for item in manifest["segments"]}
    known = set(segments)
    binding = {
        "source_sha256": manifest["source_sha256"],
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": manifest_sha256,
    }
    normalized = []
    seen = set()
    for decision in decisions:
        segment_id = decision.get("segment_id")
        if segment_id not in known or segment_id in seen:
            raise ValueError("review contains an unknown or duplicate segment_id")
        seen.add(segment_id)
        action = decision.get("decision")
        style = decision.get("confirmed_style")
        text = str(decision.get("confirmed_text") or "").strip()
        if action not in {"keep", "reject", "undecided"} or style not in STYLE_LABELS:
            raise ValueError("review contains an invalid decision or style")
        if action == "keep" and not text:
            raise ValueError("kept segments require confirmed text")
        segment = segments[segment_id]
        audio = contained_file(manifest_file.parent, segment["derived_relative_path"])
        audio_hash = segment.get("derived_audio_sha256")
        if not audio_hash or _sha256(audio) != audio_hash:
            raise ValueError(f"review audio hash drift: {segment_id}")
        normalized.append(
            {
                **binding,
                "derived_audio_sha256": audio_hash,
                "segment_id": segment_id,
                "decision": action,
                "confirmed_style": style,
                "confirmed_text": text,
                "note": str(decision.get("note") or "").strip(),
            }
        )
    reference_id = payload.get("reference_segment_id")
    if reference_id is not None:
        selected = next((item for item in normalized if item["segment_id"] == reference_id), None)
        if selected is None or selected["decision"] != "keep":
            raise ValueError("reference_segment_id must be kept in the same review batch")
    canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload_sha256 = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    log_path = target_dir / "decisions.jsonl"
    prior_events = []
    prior_content = ""
    if log_path.is_file():
        prior_content = log_path.read_bytes().decode("utf-8")
        prior_events = [json.loads(line) for line in prior_content.splitlines() if line.strip()]
    # Legacy events have no manifest binding. Only an unchanged, matching snapshot
    # can anchor their latest decisions; never stamp the current hash onto them blindly.
    legacy = [event for event in prior_events if "manifest_sha256" not in event]
    anchored = {}
    if legacy:
        snapshot_path = target_dir / "review_snapshot.json"
        if not snapshot_path.is_file():
            raise ValueError("legacy review log has no binding snapshot; use a new review directory")
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in binding.items()):
            raise ValueError("legacy review snapshot does not match manifest; use a new review directory")
        for decision in previous.get("decisions", []):
            anchored[(decision["segment_id"], decision["review_batch_id"], decision["payload_sha256"])] = decision
    replay_events = []
    for event in prior_events:
        if "manifest_sha256" in event:
            if any(event.get(key) != value for key, value in binding.items()):
                raise ValueError("review history belongs to another manifest; use a new review directory")
            if event["segment_id"] not in segments or event.get("derived_audio_sha256") != segments[event["segment_id"]].get("derived_audio_sha256"):
                raise ValueError("review history audio binding mismatch")
            replay_events.append(event)
        else:
            anchor = anchored.get((event["segment_id"], event["review_batch_id"], event["payload_sha256"]))
            if anchor is None:
                continue  # Superseded legacy event; its latest decision is anchored below.
            if any(anchor.get(key) != value for key, value in event.items() if key != "schema_version"):
                raise ValueError("legacy review snapshot differs from its event log")
            segment = segments.get(event["segment_id"])
            if segment is None or not segment.get("derived_audio_sha256"):
                raise ValueError("legacy review contains an unknown or unbound audio segment")
            replay_events.append({**event, **binding, "schema_version": 2,
                                  "derived_audio_sha256": segment["derived_audio_sha256"]})
    if legacy and not replay_events:
        raise ValueError("legacy snapshot cannot anchor review history; use a new review directory")
    matching = [event for event in prior_events if event["review_batch_id"] == batch_id]
    if matching:
        if matching[0]["payload_sha256"] != payload_sha256:
            raise RuntimeError("review_batch_id was already used for different content")
        action = "cached"
    else:
        applied_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        new_events = [
            {
                "schema_version": 2,
                "review_batch_id": batch_id,
                "payload_sha256": payload_sha256,
                "applied_at": applied_at,
                "reference_segment_id": reference_id,
                **decision,
            }
            for decision in normalized
        ]
        prior_events.extend(new_events)
        replay_events.extend(new_events)
        target_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            log_path,
            prior_content + ("\n" if prior_content and not prior_content.endswith("\n") else "")
            + "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in new_events),
        )
        action = "applied"
    latest = {}
    latest_reference = None
    for event in replay_events:
        latest[event["segment_id"]] = event
        if event.get("reference_segment_id") is not None:
            latest_reference = event["reference_segment_id"]
    if latest_reference and latest.get(latest_reference, {}).get("decision") != "keep":
        latest_reference = None
    snapshot = {
        "schema_version": 2,
        "source_sha256": manifest["source_sha256"],
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": manifest_sha256,
        "reviewed_count": len(latest),
        "total_count": len(known),
        "complete": bool(known) and set(latest) == known and all(item["decision"] != "undecided" for item in latest.values()),
        "reference_segment_id": latest_reference,
        "decisions": [latest[key] for key in sorted(latest)],
    }
    _atomic_text(target_dir / "review_snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {**snapshot, "action": action, "log_path": str(log_path)}
