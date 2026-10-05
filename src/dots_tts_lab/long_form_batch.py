from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from dots_tts_lab.long_form_strict_gate import (
    GateVerdict,
    StrictGateConfig,
    evaluate_candidate,
)

ROOT = Path(__file__).resolve().parents[2]
LONG_FORM_BATCH_IMPLEMENTATION_VERSION = 1


def _verdict(
    gate: str,
    status: str,
    *,
    reasons: list[str] | None = None,
    evaluator: str,
    calibrated: bool = False,
) -> GateVerdict:
    return GateVerdict(
        schema_version=1,
        gate=gate,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        reasons=reasons or [],
        evidence={},
        evaluator=evaluator,
        calibrated=calibrated,
        recorded_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
    )


def assemble_sentence_candidates(
    transcript_manifest: Mapping[str, Any],
    *,
    routes_by_region: Mapping[int, Mapping[str, Any]],
    identity_by_region: Mapping[int, Mapping[str, Any]],
    style_event_by_region: Mapping[int, Mapping[str, Any]],
    gate_config: StrictGateConfig,
    region_of_span: Mapping[tuple[int, int], int],
) -> list[dict[str, Any]]:
    """Join transcript sentence spans with every evidence layer.

    Each span keeps its raw evidence and receives GateVerdicts under the
    current (uncalibrated) evaluators — unknown evidence stays unknown, so
    nothing here promotes a span to pass on its own.
    """
    candidates: list[dict[str, Any]] = []
    for span in transcript_manifest.get("sentences", []):
        key = (int(span["source_start_frame"]), int(span["source_end_frame"]))
        region_index = region_of_span.get(key)
        route_row = routes_by_region.get(region_index) if region_index is not None else None
        identity = (
            identity_by_region.get(region_index) if region_index is not None else None
        )
        style = (
            style_event_by_region.get(region_index)
            if region_index is not None
            else None
        )

        verdicts: dict[str, GateVerdict] = {}
        # G0 source integrity: hash-bound spans from the transcript manifest.
        verdicts["G0"] = _verdict(
            "G0", "pass",
            evaluator="transcript_manifest",
            reasons=["source-hash-bound span"],
        )
        # G1 speech presence: VAD-derived region membership is evidence, not
        # a calibrated pass.
        verdicts["G1"] = _verdict(
            "G1", "unknown",
            evaluator="prefilter_vad",
            reasons=["vad evidence present; threshold uncalibrated"],
        )
        verdicts["G2"] = _verdict(
            "G2",
            "fail" if (identity or {}).get("status") == "speaker_change_suspect"
            else "unknown",
            evaluator="campp_timeline",
            reasons=["speaker timeline uncalibrated"],
        )
        verdicts["G3"] = _verdict(
            "G3", "unknown",
            evaluator="campp_target_cosine",
            reasons=["target identity threshold uncalibrated"],
        )
        verdicts["G4"] = _verdict(
            "G4", "unknown",
            evaluator="style_distribution",
            reasons=["normal-style threshold uncalibrated"],
        )
        verdicts["G5"] = _verdict(
            "G5",
            "fail" if (style or {}).get("prefilter_spatial_risk_fraction", 0) > 0
            else "unknown",
            evaluator="spatial_lr_evidence",
            reasons=["spatial threshold uncalibrated"],
        )
        event_risk_max = None
        ev = (style or {}).get("event_evidence") or {}
        if ev.get("risk_scores"):
            event_risk_max = max(
                (
                    float(s["max"])
                    for s in ev["risk_scores"].values()
                    if s.get("max") is not None
                ),
                default=None,
            )
        verdicts["G6"] = _verdict(
            "G6",
            "fail" if span.get("overlaps_quarantine") else "unknown",
            evaluator="event_panns",
            reasons=(
                ["span crosses prefilter-quarantined audio"]
                if span.get("overlaps_quarantine")
                else ["event thresholds uncalibrated"]
            ),
        )
        verdicts["G7"] = _verdict(
            "G7",
            "fail" if span.get("status") not in ("candidate",)
            else (
                "fail"
                if span.get("text_agreement", {}).get("status") not in ("match",)
                else "unknown"
            ),
            evaluator="dual_asr",
            reasons=[
                f"span status {span.get('status')}; agreement "
                f"{span.get('text_agreement', {}).get('status')}"
            ],
        )
        verdicts["G8"] = _verdict(
            "G8", "unknown",
            evaluator="dnsmos_quality",
            reasons=["quality threshold uncalibrated"],
        )

        outcome = evaluate_candidate(verdicts, gate_config)
        span_id = hashlib.sha256(
            json.dumps(
                {
                    "source_sha256": transcript_manifest.get("source_sha256"),
                    "start": span["source_start_frame"],
                    "end": span["source_end_frame"],
                    "text": span.get("primary_text"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                "candidate_id": span_id,
                "source_start_frame": span["source_start_frame"],
                "source_end_frame": span["source_end_frame"],
                "duration_seconds": span["duration_seconds"],
                "primary_text": span.get("primary_text"),
                "text_agreement": span.get("text_agreement", {}).get("status"),
                "boundary_clean": span.get("boundary_clean"),
                "overlaps_quarantine": span.get("overlaps_quarantine"),
                "span_status": span.get("status"),
                "region_index": region_index,
                "route": (route_row or {}).get("route"),
                "route_output": (route_row or {}).get("output"),
                "identity_status": (identity or {}).get("status"),
                "event_risk_max": event_risk_max,
                "disposition": outcome.disposition,
                "failed_gates": outcome.failed_gates,
                "unknown_gates": outcome.unknown_gates,
                "verdicts": {
                    gate: verdict.status for gate, verdict in verdicts.items()
                },
            }
        )
    return candidates


def rank_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    """Deterministic ordering: agreed+duration-valid spans first, then by
    event risk ascending, then shorter coverage gaps. Used for both the
    review order and the budget assignment — never promotes quarantined
    items."""
    def score(row: Mapping[str, Any]) -> tuple:
        in_range = 3.0 <= float(row.get("duration_seconds", 0)) <= 12.0
        return (
            0 if row.get("text_agreement") == "match" else 1,
            0 if in_range else 1,
            float(row.get("event_risk_max") or 99.0),
            -float(row.get("duration_seconds") or 0),
            str(row.get("candidate_id")),
        )

    return [
        str(row["candidate_id"])
        for row in sorted(candidates, key=score)
    ]


def dedupe_by_text(
    ranked_ids: list[str], candidates: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Drop later spans whose normalized text duplicates an earlier one."""
    text_of = {
        str(row["candidate_id"]): (row.get("primary_text") or "").strip()
        for row in candidates
    }
    seen: set[str] = set()
    kept: list[str] = []
    for candidate_id in ranked_ids:
        text = text_of.get(candidate_id, "")
        key = text if text else candidate_id
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate_id)
    return kept


_STRICT_REVIEW_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>严格候选审核 · {title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#14161a;color:#e8e8e8}}
h1{{font-size:1.1rem}} .muted{{color:#888}} .warn{{color:#e0a050}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{border:1px solid #333;padding:.35rem .5rem;vertical-align:top}}
tr.gray td{{opacity:.75}} .badge{{display:inline-block;background:#262a31;border:1px solid #3a3f47;border-radius:4px;padding:0 .35rem;margin:.1rem .15rem .1rem 0;font-size:.75rem}}
.badge.bad{{border-color:#7a3535;color:#e08080}} .badge.good{{border-color:#2f5f3f;color:#7fd79a}}
audio{{width:240px;height:32px}} button{{background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:.35rem .8rem;cursor:pointer}}
button.ghost{{background:#3a3f47}} .diff{{color:#e0a050}}
details{{margin:.5rem 0}} .toolbar{{position:sticky;top:0;background:#14161a;padding:.5rem 0;border-bottom:1px solid #333;z-index:5}}
</style></head><body>
<div class="toolbar">
<h1>严格候选审核 — {title}</h1>
<div class="muted">{summary}</div>
<div style="margin:.4rem 0">
<button onclick="exportJson()">导出审核 JSON</button>
<button class="ghost" onclick="toggleGray()">显示/隐藏灰区（默认隐藏）</button>
<span id="saveMsg" class="muted"></span></div>
<div class="muted" style="font-size:.8rem">判定 = 训练可用确认。灰区条目仅供校准抽样（≤6/批），默认不计入候选。
导出写入 data/reports/long_form/incoming/{save_name}</div>
</div>
<table id="tbl"><thead><tr>
<th>#</th><th>音频(raw/成品)</th><th>文本(主/副)</th><th>时长</th><th>证据</th><th>处置</th><th>判定</th>
</tr></thead><tbody></tbody></table>
<script>
const DATA = {payload};
const SAVE_NAME = {save_name_js};
const stateKey = 'strict-review:' + DATA.source_sha256;
let state = JSON.parse(localStorage.getItem(stateKey) || '{{}}');
let showGray = false;
function esc(s){{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML}}
function itemRow(item, i){{
  const gray = item.disposition !== 'strict_candidate';
  if (gray && !showGray) return '';
  const dec = (state[item.candidate_id]||{{}}).decision || '';
  const badges = [];
  badges.push(`<span class="badge">${{item.disposition}}</span>`);
  if (item.failed_gates?.length) badges.push(`<span class="badge bad">fail ${{item.failed_gates.join(',')}}</span>`);
  if (item.unknown_gates?.length) badges.push(`<span class="badge">${{item.unknown_gates.length}} unknown</span>`);
  if (item.overlaps_quarantine) badges.push('<span class="badge bad">覆盖隔离区</span>');
  if (item.boundary_clean === false) badges.push('<span class="badge bad">脏边界</span>');
  if (item.event_risk_max != null && item.event_risk_max >= 0.5) badges.push(`<span class="badge bad">事件风险 ${{item.event_risk_max.toFixed(2)}}</span>`);
  if (item.text_agreement === 'match') badges.push('<span class="badge good">双ASR一致</span>');
  const sec = item.secondary_text && item.secondary_text !== item.primary_text
    ? `<div class="diff">副: ${{esc(item.secondary_text)}}</div>` : '';
  const routed = item.routed_audio ? `<br><audio controls src="${{item.routed_audio}}"></audio>` : '';
  const opts = ['confirm','reject','gray'].map(v =>
    `<label><input type="radio" name="d${{i}}" value="${{v}}" ${{dec===v?'checked':''}} onchange="setDec('${{item.candidate_id}}','${{v}}')">${{v==='confirm'?'确认':v==='reject'?'拒绝':'灰区'}}</label>`).join(' ');
  return `<tr class="${{gray?'gray':''}}"><td>${{i+1}}</td>
    <td><audio controls src="${{item.raw_audio}}"></audio>${{routed}}<div class="muted" style="font-size:.7rem">${{item.route||''}}</div></td>
    <td>${{esc(item.primary_text)}}${{sec}}</td>
    <td>${{item.duration_seconds.toFixed(2)}}s</td>
    <td>${{badges.join('')}}</td>
    <td class="muted" style="font-size:.75rem">${{(item.route_reasons||[]).join('; ')}}</td>
    <td>${{opts}}</td></tr>`;
}}
function render(){{
  document.querySelector('#tbl tbody').innerHTML = DATA.items.map(itemRow).join('');
}}
function setDec(id, v){{ state[id] = {{decision: v}}; localStorage.setItem(stateKey, JSON.stringify(state)); }}
function toggleGray(){{ showGray = !showGray; render(); }}
function exportJson(){{
  const payload = {{schema_version:1, page:'strict_review', source_sha256:DATA.source_sha256,
    batch_sha256:DATA.batch_sha256, exported_at:new Date().toISOString(), decisions:state}};
  if (location.protocol.startsWith('http')) {{
    fetch('/save/' + SAVE_NAME, {{method:'POST', body: JSON.stringify(payload)}})
      .then(r => r.json()).then(r => {{ document.getElementById('saveMsg').textContent = ' 已保存: ' + r.saved; }})
      .catch(e => fallbackDownload(payload));
  }} else fallbackDownload(payload);
}}
function fallbackDownload(payload){{
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));
  a.download = SAVE_NAME + '.json'; a.click();
}}
render();
</script></body></html>"""


def build_strict_review_html(
    *,
    title: str,
    source_sha256: str,
    items: list[dict[str, Any]],
    save_name: str,
    batch_sha256: str,
) -> str:
    """Render the strict-candidate review page.

    Items carry raw_audio (always) and routed_audio (when the region has a
    completed routed output) as server-relative paths. Quarantined/gray rows
    render dimmed and hidden by default.
    """
    n_strict = sum(1 for i in items if i["disposition"] == "strict_candidate")
    summary = (
        f"共 {len(items)} 句 · strict_candidate {n_strict} · "
        f"灰区 {len(items) - n_strict}（默认隐藏）"
    )
    payload = {
        "source_sha256": source_sha256,
        "batch_sha256": batch_sha256,
        "items": items,
    }
    return _STRICT_REVIEW_PAGE.format(
        title=title,
        summary=summary,
        payload=json.dumps(payload, ensure_ascii=False),
        save_name=save_name,
        save_name_js=json.dumps(save_name),
    )


_TEXT_REVIEW_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>候选文本校订 · {title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#14161a;color:#e8e8e8}}
h1{{font-size:1.1rem}} .muted{{color:#888}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{border:1px solid #333;padding:.35rem .5rem;vertical-align:top}}
.badge{{display:inline-block;background:#262a31;border:1px solid #3a3f47;border-radius:4px;padding:0 .35rem;margin:.1rem;font-size:.75rem}}
.badge.bad{{border-color:#7a3535;color:#e08080}} .badge.good{{border-color:#2f5f3f;color:#7fd79a}}
audio{{width:220px;height:32px}} textarea{{width:100%;background:#1c1f24;color:#e8e8e8;border:1px solid #3a3f47;border-radius:4px;font-size:.85rem;padding:.3rem}}
button{{background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:.35rem .8rem;cursor:pointer}}
button.ghost{{background:#3a3f47;font-size:.75rem;padding:.15rem .5rem}}
.toolbar{{position:sticky;top:0;background:#14161a;padding:.5rem 0;border-bottom:1px solid #333;z-index:5}}
.alt{{color:#9ab4e0;font-size:.78rem;cursor:pointer}} .alt:hover{{text-decoration:underline}}
</style></head><body>
<div class="toolbar">
<h1>候选文本校订 — {title}</h1>
<div class="muted">{summary}</div>
<div style="margin:.4rem 0">
<button onclick="exportJson()">导出校订 JSON</button>
<span id="saveMsg" class="muted"></span></div>
<div class="muted" style="font-size:.8rem">每条：听音频 → 文本框里直接改正（或点蓝色候选文本套用）→ 无需单独保存。
导出的 text_final 即人工确认文本，写入 data/reports/long_form/incoming/{save_name}</div>
</div>
<table id="tbl"><thead><tr>
<th>#</th><th>音频</th><th>时长</th><th>一致性</th><th>最终文本（可编辑）</th>
</tr></thead><tbody></tbody></table>
<script>
const DATA = {payload};
const SAVE_NAME = {save_name_js};
const stateKey = 'strict-text:' + DATA.source_sha256;
let state = JSON.parse(localStorage.getItem(stateKey) || '{{}}');
function esc(s){{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML}}
function getText(item){{return (state[item.candidate_id]||{{}}).text ?? item.primary_text ?? ''}}
function setText(id, v){{ state[id] = {{text: v, edited: true}}; localStorage.setItem(stateKey, JSON.stringify(state)); updateCount(); }}
function applyAlt(id, text, idx){{ const ta = document.getElementById('ta'+idx); ta.value = text; setText(id, text); }}
function itemRow(item, i){{
  const agreed = item.text_agreement === 'match';
  const badge = agreed ? '<span class="badge good">双ASR一致</span>' : '<span class="badge bad">' + esc(item.text_agreement||'?') + '</span>';
  const alt = (!agreed && item.secondary_text)
    ? `<div class="alt" onclick="applyAlt('${{item.candidate_id}}', this.dataset.t, ${{i}})" data-t="${{esc(item.secondary_text)}}">副: ${{esc(item.secondary_text)}}</div>` : '';
  return `<tr><td>${{i+1}}</td>
    <td><audio controls src="${{item.raw_audio}}"></audio>${{item.routed_audio?'<br><audio controls src="'+item.routed_audio+'"></audio>':''}}</td>
    <td>${{item.duration_seconds.toFixed(2)}}s</td>
    <td>${{badge}}</td>
    <td><textarea id="ta${{i}}" rows="2" oninput="setText('${{item.candidate_id}}', this.value)">${{esc(getText(item))}}</textarea>${{alt}}</td></tr>`;
}}
function render(){{ document.querySelector('#tbl tbody').innerHTML = DATA.items.map(itemRow).join(''); updateCount(); }}
function updateCount(){{ const n = Object.keys(state).length; document.getElementById('saveMsg').textContent = ' 已编辑 ' + n + '/' + DATA.items.length; }}
function exportJson(){{
  const decisions = {{}};
  DATA.items.forEach(it => {{ decisions[it.candidate_id] = {{text_final: getText(it), edited: !!(state[it.candidate_id]||{{}}).edited, text_agreement: it.text_agreement}}; }});
  const payload = {{schema_version:1, page:'strict_text_review', source_sha256:DATA.source_sha256,
    batch_sha256:DATA.batch_sha256, exported_at:new Date().toISOString(), texts:decisions}};
  if (location.protocol.startsWith('http')) {{
    fetch('/save/' + SAVE_NAME, {{method:'POST', body: JSON.stringify(payload)}})
      .then(r => r.json()).then(r => {{ document.getElementById('saveMsg').textContent = ' 已保存: ' + r.saved; }})
      .catch(e => fallbackDownload(payload));
  }} else fallbackDownload(payload);
}}
function fallbackDownload(payload){{
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));
  a.download = SAVE_NAME + '.json'; a.click();
}}
render();
</script></body></html>"""


def build_text_review_html(
    *,
    title: str,
    source_sha256: str,
    items: list[dict[str, Any]],
    save_name: str,
    batch_sha256: str,
) -> str:
    """Text-correction page for human-confirmed candidates.

    Every row shows the primary ASR text in an editable box; the secondary
    hypothesis is a one-click alternate when they disagree. The exported
    ``text_final`` is the human-confirmed transcript bound to the candidate.
    """
    n_mismatch = sum(
        1 for i in items if i.get("text_agreement") != "match"
    )
    summary = (
        f"{len(items)} 条已确认候选 · 其中 {n_mismatch} 条双 ASR 不一致需校订"
    )
    payload = {
        "source_sha256": source_sha256,
        "batch_sha256": batch_sha256,
        "items": items,
    }
    return _TEXT_REVIEW_PAGE.format(
        title=title,
        summary=summary,
        payload=json.dumps(payload, ensure_ascii=False),
        save_name=save_name,
        save_name_js=json.dumps(save_name),
    )
