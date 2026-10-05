"""Corpus review web app for the Aemeath dataset.

Serves a local page listing every item in a frozen dataset so a human can
listen and mark keep / exclude decisions. Decisions persist to a JSON file
and can be exported into the next version's manual-review YAML
(speaker_decisions: exclude_uncertain), which freeze_speaker_dataset.py
consumes via apply_reviewed_exclusions.

Usage:
    .venv/Scripts/python.exe scripts/review_corpus_web.py \
        --dataset datasets/aemeath/v3 --port 8765
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DECISIONS_PATH = ROOT / "data/reports/review_web/aemeath_corpus_decisions.json"
VALID_DECISIONS = {"keep", "exclude", "undecided"}

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Aemeath 语料审核 v5</title>
<meta http-equiv="Cache-Control" content="no-store">
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:8px 16px;display:flex;gap:12px;align-items:center;border-bottom:1px solid #333;z-index:10;flex-wrap:wrap}
button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px;padding:4px 10px;cursor:pointer}
button.on{background:#2d5a2d;border-color:#4a4}
button.exc{background:#5a2d2d;border-color:#a44}
#bar{height:6px;background:#333;border-radius:3px;flex:1;min-width:120px}
#bar>i{display:block;height:100%;background:#4a4;border-radius:3px}
table{width:100%;border-collapse:collapse}
td,th{padding:6px 8px;border-bottom:1px solid #2a2a2a;vertical-align:middle}
tr.cur{background:#26262e}
tr.exc .txt{opacity:.45;text-decoration:line-through}
tr.keep .txt{color:#9d9}
td.c{white-space:nowrap}
.txt{max-width:640px}
audio{height:28px;width:220px}
input.note{background:#1c1c1c;border:1px solid #333;color:#ccc;border-radius:4px;padding:3px 6px;width:160px}
.flt input{background:#1c1c1c;border:1px solid #333;color:#eee;border-radius:4px;padding:4px 8px}
small{color:#888}
</style></head><body>
<header>
 <b>Aemeath 语料审核</b><span id=prog></span><div id=bar><i></i></div>
 <span class=flt><input id=q placeholder="搜索文本" size=12>
 <select id=fs><option value=>全部split</option><option>train</option><option>validation</option><option>test</option></select>
 <select id=fd><option value=all>全部</option><option value=undecided>未审</option><option value=keep>保留</option><option value=exclude>剔除</option></select></span>
 <button onclick="exportDecisions()">导出 v4 审核文件</button><small id=exp></small>
</header>
<table><thead><tr><th>#</th><th>播放</th><th>文本</th><th>时长</th><th>split</th><th>情绪</th><th>判定</th><th>备注</th></tr></thead><tbody id=tb></tbody></table>
<script>
document.getElementById('prog').textContent='boot';
window.onerror=function(m,s,l,c){document.getElementById('prog').textContent='JS错误: '+m+' @'+l+':'+c;return false;};
window.onunhandledrejection=function(e){document.getElementById('prog').textContent='Promise拒绝: '+(e.reason&&e.reason.message||e.reason);};
let items=[], decisions={}, cur=0;
const $=s=>document.querySelector(s);
async function load(){
  try{
    const ri=await fetch('/api/items');
    const ii=await ri.text();
    document.getElementById('prog').textContent='items http '+ri.status+' len '+ii.length;
    items=JSON.parse(ii);
    const rd=await fetch('/api/decisions');
    decisions=JSON.parse(await rd.text());
    document.getElementById('prog').textContent='loaded '+items.length+' items, rendering...';
    render();
  }catch(e){
    document.getElementById('prog').textContent='加载失败: '+(e&&e.message||e)+' '+(e&&e.stack||'').split(String.fromCharCode(10))[0];
  }
}
function render(){
  const q=$('#q').value, fs=$('#fs').value, fd=$('#fd').value;
  const tb=$('#tb'); tb.innerHTML='';
  items.forEach((it,i)=>{
    try{
    const d=decisions[it.asset]||{};
    const st=d.status||'undecided';
    if(q&&!it.text.includes(q))return;
    if(fs&&it.split!==fs)return;
    if(fd!=='all'&&st!==fd)return;
    const tr=document.createElement('tr');
    tr.id='r'+i; tr.className=(i===cur?'cur ':'')+(st==='exclude'?'exc':st==='keep'?'keep':'');
    tr.innerHTML=`<td>${i+1}</td>
      <td><audio controls preload=none src="/api/audio/${it.asset}"></audio></td>
      <td class=txt>${it.text}</td>
      <td>${it.duration.toFixed(1)}s</td><td>${it.split}</td><td><small>${it.emotion||''}</small></td>
      <td class=c>
        <button class="${st==='keep'?'on':''}" onclick="set(${i},'keep')">留</button>
        <button class="${st==='exclude'?'exc':''}" onclick="set(${i},'exclude')">剔</button>
        <button onclick="set(${i},'undecided')">清</button></td>
      <td><input class=note value="${(d.note||'').replace(/"/g,'&quot;')}" onchange="note(${i},this.value)"></td>`;
    tr.onclick=e=>{if(e.target.tagName!=='AUDIO'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='BUTTON'){cur=i;render();}};
    tb.appendChild(tr);
    }catch(e){const tr2=document.createElement('tr');tr2.innerHTML='<td colspan=8>row '+i+' err: '+e+'</td>';tb.appendChild(tr2);}
  });
  const done=items.filter(it=>(decisions[it.asset]||{}).status==='keep'||(decisions[it.asset]||{}).status==='exclude').length;
  const exc=items.filter(it=>(decisions[it.asset]||{}).status==='exclude').length;
  $('#prog').textContent=`已审 ${done}/${items.length}（剔除 ${exc}）`;
  $('#bar i').style.width=(100*done/items.length)+'%';
}
async function set(i,status){
  const it=items[i];
  const note=(decisions[it.asset]&&decisions[it.asset].note)||'';
  const r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({asset_sha256:it.asset,status,note})});
  decisions[it.asset]=await r.json();
  cur=i+1<items.length?i+1:i; render();
}
async function note(i,v){
  const it=items[i];
  const st=(decisions[it.asset]&&decisions[it.asset].status)||'undecided';
  const r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({asset_sha256:it.asset,status:st,note:v})});
  decisions[it.asset]=await r.json();
}
async function exportDecisions(){
  const r=await fetch('/api/export',{method:'POST'});
  const j=await r.json();
  $('#exp').textContent=r.ok?`已导出 ${j.excluded} 条剔除 → ${j.path}`:('失败: '+JSON.stringify(j));
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='j'||e.key==='ArrowDown'){cur=Math.min(cur+1,items.length-1);render();scroll();}
  if(e.key==='k'||e.key==='ArrowUp'){cur=Math.max(cur-1,0);render();scroll();}
  if(e.key==='1')set(cur,'keep');
  if(e.key==='2')set(cur,'exclude');
  if(e.key===' '){e.preventDefault();const a=document.querySelector('#r'+cur+' audio');a&&(a.paused?a.play():a.pause());}
});
function scroll(){const el=document.getElementById('r'+cur);el&&el.scrollIntoView({block:'nearest'});}
$('#q').oninput=render;$('#fs').onchange=render;$('#fd').onchange=render;
load();
</script></body></html>"""


class DecisionIn(BaseModel):
    asset_sha256: str
    status: str
    note: str = ""


def _load_items(dataset_dir: Path) -> list[dict[str, Any]]:
    table = pq.read_table(dataset_dir / "dataset.parquet")
    items = []
    for row in table.to_pylist():
        items.append(
            {
                "asset": row["asset_sha256"],
                "fid": row["fid"],
                "text": row["text_exact"],
                "split": row["split"],
                "emotion": row["emotion_primary"],
                "duration": float(row["duration_seconds"]),
                "audio": row["audio_absolute_path"],
            }
        )
    items.sort(key=lambda item: (item["split"] != "train", item["fid"]))
    return items


def _load_decisions() -> dict[str, dict[str, Any]]:
    if DECISIONS_PATH.is_file():
        data = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
        return data.get("decisions", {})
    return {}


def _save_decisions(decisions: dict[str, dict[str, Any]]) -> None:
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "corpus_review_decisions",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "decisions": decisions,
    }
    DECISIONS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _export_manual_review(
    items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    prior_review_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    prior_speaker: list[dict[str, Any]] = []
    prior_edges: list[dict[str, Any]] = []
    if prior_review_path.is_file():
        prior = yaml.safe_load(prior_review_path.read_text(encoding="utf-8"))
        if isinstance(prior, dict):
            prior_speaker = list(prior.get("speaker_decisions") or [])
            prior_edges = list(prior.get("edge_decisions") or [])
    prior_assets = {str(d.get("asset_sha256")) for d in prior_speaker}
    known = {item["asset"]: item for item in items}
    new_decisions: list[dict[str, Any]] = []
    overridden: list[str] = []
    skipped: list[str] = []
    for asset, decision in decisions.items():
        if asset not in known:
            skipped.append(asset)
            continue
        if decision.get("status") != "exclude":
            continue
        note = decision.get("note") or "web corpus review: user excluded"
        if asset in prior_assets:
            # A fresh human exclusion overrides a prior carry-over decision
            # (e.g. confirmed_same_speaker): drop the stale entry so the new
            # one is the single decision apply_reviewed_exclusions sees.
            prior_speaker = [
                d for d in prior_speaker if str(d.get("asset_sha256")) != asset
            ]
            overridden.append(asset)
        new_decisions.append(
            {
                "asset_sha256": asset,
                "review_status": "exclude_uncertain",
                "review_note": note,
            }
        )
    payload = {
        "schema_version": 1,
        "review_id": "aemeath_v4_corpus_web_review",
        "review_version": 1,
        "dataset_id": "aemeath",
        "dataset_version": 4,
        "review_batch_id": "aemeath-v4-corpus-web-review",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": "review_corpus_web.py",
        "edge_decisions": prior_edges,
        "speaker_decisions": prior_speaker + new_decisions,
    }
    out_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "path": str(out_path),
        "carried_over": len(prior_speaker),
        "excluded": len(new_decisions),
        "overridden_prior": overridden,
        "skipped_unknown": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/aemeath/v3",
        help="Frozen dataset directory containing dataset.parquet.",
    )
    parser.add_argument(
        "--prior-review",
        type=Path,
        default=ROOT / "configs/lab/datasets/aemeath_v3_manual_review.yaml",
        help="Previous version's manual review YAML to carry decisions over.",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=ROOT / "configs/lab/datasets/aemeath_v4_manual_review.yaml",
        help="Manual review YAML written by the export button.",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    items = _load_items(dataset_dir)
    decisions = _load_decisions()
    lock = threading.Lock()

    app = FastAPI(title="Aemeath corpus review")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/items")
    def api_items() -> list[dict[str, Any]]:
        return [{k: v for k, v in item.items() if k != "audio"} for item in items]

    @app.get("/api/decisions")
    def api_decisions() -> dict[str, dict[str, Any]]:
        return decisions

    @app.get("/api/audio/{asset}")
    def api_audio(asset: str) -> FileResponse:
        for item in items:
            if item["asset"] == asset:
                path = Path(item["audio"])
                if not path.is_file():
                    raise HTTPException(404, "audio missing")
                return FileResponse(path, media_type="audio/wav")
        raise HTTPException(404, "unknown asset")

    @app.post("/api/decision")
    def api_decision(decision: DecisionIn) -> dict[str, Any]:
        if decision.status not in VALID_DECISIONS:
            raise HTTPException(400, "invalid status")
        if not any(item["asset"] == decision.asset_sha256 for item in items):
            raise HTTPException(404, "unknown asset")
        with lock:
            entry = {
                "status": decision.status,
                "note": decision.note,
                "decided_at": datetime.now(timezone.utc).isoformat(),
            }
            if decision.status == "undecided":
                decisions.pop(decision.asset_sha256, None)
            else:
                decisions[decision.asset_sha256] = entry
            _save_decisions(decisions)
            return decisions.get(decision.asset_sha256, entry)

    @app.post("/api/export")
    def api_export() -> dict[str, Any]:
        return _export_manual_review(
            items, decisions, args.prior_review, args.export_path
        )

    print(f"Serving {len(items)} items at http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
