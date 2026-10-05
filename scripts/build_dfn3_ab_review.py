from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import file_sha256

ROOT = Path(__file__).resolve().parents[1]

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>DFN3 对照验收 · {title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#14161a;color:#e8e8e8}}
h1{{font-size:1.1rem}} h2{{font-size:.95rem;color:#9ab4e0;margin-top:1.2rem}}
.muted{{color:#888}} table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{border:1px solid #333;padding:.35rem .5rem;vertical-align:top}}
audio{{width:280px;height:32px;display:block;margin:.15rem 0}}
button{{background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:.35rem .8rem;cursor:pointer}}
.toolbar{{position:sticky;top:0;background:#14161a;padding:.5rem 0;border-bottom:1px solid #333;z-index:5}}
.txt{{max-width:260px;font-size:.82rem}}
label{{white-space:nowrap}}
</style></head><body>
<div class="toolbar">
<h1>DFN3 对照验收 — {title}</h1>
<div class="muted">{summary}</div>
<div style="margin:.4rem 0"><button onclick="exportJson()">导出验收 JSON</button>
<span id="saveMsg" class="muted"></span></div>
<div class="muted" style="font-size:.8rem">每条是一句完整语音：听原音 / 8dB / 12dB 三版 →
按尾音完整性/事件残留/音色变化/边界干净判结论。导出写入 incoming/{save_name}</div>
</div>
<div id="root"></div>
<script>
const DATA = {payload};
const SAVE_NAME = {save_name_js};
const stateKey = 'dfn3-ab-v3:' + DATA.source_sha256;
let state = JSON.parse(localStorage.getItem(stateKey) || '{{}}');
function esc(s){{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML}}
const VERDICTS = [['db12_best','12dB 最干净且无损'],['db8_best','8dB 最干净且无损'],['tie','差别不大'],['dfn3_worse','两档都变差/伤内容'],['raw_bad','原音本身不合格']];
function groupHtml(g){{
  let h = `<h2>${{g.group}}</h2><table><thead><tr><th>句</th><th>文本</th><th>原音</th><th>DFN3 8dB</th><th>DFN3 12dB</th><th>结论</th></tr></thead><tbody>`;
  g.items.forEach((it) => {{
    const cur = (state[it.id]||{{}}).verdict || '';
    const opts = VERDICTS.map(v => `<label><input type="radio" name="r${{it.id}}" value="${{v[0]}}" ${{cur===v[0]?'checked':''}} onchange="setV('${{it.id}}','${{v[0]}}')">${{v[1]}}</label>`).join('<br>');
    h += `<tr><td>${{it.sentence_index}}<br><span class="muted">${{it.duration_seconds.toFixed(2)}}s</span></td>
      <td class="txt">${{esc(it.primary_text)}}<br><span class="muted">副: ${{esc(it.secondary_text||'—')}}</span></td>
      <td><audio controls src="${{it.raw_audio}}"></audio></td>
      <td><audio controls src="${{it.dfn3_audio}}"></audio></td>
      <td><audio controls src="${{it.dfn3_12db_audio}}"></audio></td>
      <td>${{opts}}</td></tr>`;
  }});
  return h + '</tbody></table>';
}}
function render(){{ document.getElementById('root').innerHTML = DATA.groups.map(groupHtml).join(''); updateCount(); }}
function setV(id, v){{ state[id] = {{verdict: v}}; localStorage.setItem(stateKey, JSON.stringify(state)); updateCount(); }}
function updateCount(){{ document.getElementById('saveMsg').textContent = ' 已判 ' + Object.keys(state).length + '/' + DATA.total; }}
function exportJson(){{
  const payload = {{schema_version:1, page:'dfn3_ab_acceptance_v3', source_sha256:DATA.source_sha256,
    level:'sentence', arms:['raw','dfn3_8db','dfn3_12db'],
    exported_at:new Date().toISOString(), verdicts:state}};
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the sentence-level raw/DFN3 A/B acceptance page "
        "(LF-13D): ≤12 groups — 4 clean, 4 steady-noise, 4 hard negatives."
    )
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    source_hash = file_sha256(source)
    sha12 = source_hash[:12]
    ab_dir = ROOT / "data/work/long_form/denoise_ab" / sha12

    meta = json.loads((ab_dir / "sentence_meta.json").read_text(encoding="utf-8"))
    run_dir = (
        ROOT / "data/work/long_form/enhancement/dfn3"
        / sha12[:2] / source_hash
    )
    run_dir_12db = (
        ROOT / "data/work/long_form/enhancement/dfn3_dev12db"
        / sha12[:2] / source_hash
    )

    audio_root = ROOT / f"data/reports/long_form/dfn3_ab/{sha12}/audio_v2"
    audio_root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict]] = {
        "clean": [], "steady_noise": [], "hard_negative": []
    }
    for rid_s, m in sorted(meta.items(), key=lambda kv: int(kv[0])):
        rid = int(rid_s)
        dfn3_wav = run_dir / f"region_{rid:05d}.wav"
        dfn3_12db_wav = run_dir_12db / f"region_{rid:05d}.wav"
        if not dfn3_wav.is_file() or not dfn3_12db_wav.is_file():
            raise RuntimeError(f"missing DFN3 output for region {rid}")
        raw_name = f"s{rid:05d}_raw.wav"
        raw_path = audio_root / raw_name
        if not raw_path.is_file():
            regions = json.loads(
                (ab_dir / "sentence_regions.json").read_text(encoding="utf-8")
            )["regions"]
            row = next(r for r in regions if r["region_index"] == rid)
            start, end = row["source_start_frame"], row["source_end_frame"]
            with sf.SoundFile(str(source), mode="r") as f:
                f.seek(start)
                block = f.read(end - start, dtype="float32", always_2d=True)
                rate = int(f.samplerate)
            sf.write(
                str(raw_path), block.mean(axis=1).astype(np.float32),
                rate, subtype="PCM_16",
            )
        groups[m["group"]].append({
            "id": str(rid),
            "sentence_index": m["sentence_index"],
            "duration_seconds": m["duration_seconds"],
            "primary_text": m["primary_text"],
            "secondary_text": m.get("secondary_text"),
            "raw_audio": f"/data/reports/long_form/dfn3_ab/{sha12}/audio_v2/{raw_name}",
            "dfn3_audio": (
                f"/data/work/long_form/enhancement/dfn3/{sha12[:2]}"
                f"/{source_hash}/region_{rid:05d}.wav"
            ),
            "dfn3_12db_audio": (
                f"/data/work/long_form/enhancement/dfn3_dev12db/{sha12[:2]}"
                f"/{source_hash}/region_{rid:05d}.wav"
            ),
        })

    payload = {
        "source_sha256": source_hash,
        "groups": [{"group": n, "items": items} for n, items in groups.items()],
        "total": sum(len(v) for v in groups.values()),
    }
    summary = (
        f"12 句三列对照 原音/8dB/12dB（4 干净 / 4 稳态底噪 / 4 难负例）· "
        f"源 {source.name}"
    )
    html = _PAGE.format(
        title=source.name,
        summary=summary,
        payload=json.dumps(payload, ensure_ascii=False),
        save_name=f"reference-dfn3-ab-v3-{sha12}",
        save_name_js=json.dumps(f"reference-dfn3-ab-v3-{sha12}"),
    )
    out = ROOT / f"data/reports/long_form/dfn3_ab_v3_{sha12}.html"
    out.write_text(html, encoding="utf-8")
    print(json.dumps({
        "status": "completed", "page": str(out),
        "items": {g: [i["sentence_index"] for i in v] for g, v in groups.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
