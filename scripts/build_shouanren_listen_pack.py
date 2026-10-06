"""守岸人盲听包 (Phase 6): 匿名化 A01.. + 评分页 + answer_key + 响度匹配副本"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diagnostics/shouanren_zero_shot_v1"
RAW = OUT / "raw"
LISTEN = OUT / "listen"
LM = LISTEN / "level_matched"

import numpy as np
import pyloudnorm
import soundfile as sf

SAMPLES = [
    # (匿名说明仅为分组展示用, 真实 case 在 answer_key)
    "S-R1-T1", "S-R1-T2", "S-R2-T1", "S-R2-T2",
    "M-R1-T1", "M-R1-T2", "M-R2-T1", "M-R2-T2",
    "O-R1-T1", "O-R1-T2", "O-R2-T1", "O-R2-T2",
    "C1-O-R3-T1", "C2-O-R1-T1-xvec",
]
EXTRA = {"webui_profile_final": OUT / "webui_final" / "T1.wav"}

T1 = "森林里的小动物们都说，月亮最近失眠了。"
T2 = "我睡不着。风把云朵吹散了，天空太亮，我找不到做梦的枕头。"
TEXT_OF = {"T1": T1, "T2": T2}


def target_text(case_id: str) -> str:
    return TEXT_OF["T2" if case_id.endswith("T2") else "T1"]


def level_match(src: Path, dst: Path, target_lufs: float = -20.0):
    a, sr = sf.read(src, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    lufs = pyloudnorm.Meter(sr).integrated_loudness(a)
    g = 10 ** ((target_lufs - lufs) / 20)
    out = np.clip(a * g, -1.0, 1.0)
    sf.write(dst, out, sr, subtype="PCM_24")


def main() -> int:
    LISTEN.mkdir(parents=True, exist_ok=True)
    LM.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261006)

    ids = [f"A{i+1:02d}" for i in range(len(SAMPLES) + len(EXTRA))]
    rng.shuffle(ids)
    order = rng.sample(range(len(ids)), len(ids))  # display order permutation

    entries = []
    answer_key = {}
    all_items = [(cid, RAW / f"{cid}.wav", "core") for cid in SAMPLES] + [
        (cid, path, "webui_path") for cid, path in EXTRA.items()
    ]
    for (case_id, src, group), anon in zip(all_items, ids):
        dst = LISTEN / f"{anon}.wav"
        shutil.copy2(src, dst)
        lm_dst = LM / f"{anon}.wav"
        level_match(src, lm_dst)
        answer_key[anon] = {"case_id": case_id, "group": group,
                            "source": str(src.relative_to(ROOT))}
        entries.append({"anon": anon, "file": f"{anon}.wav",
                        "lm_file": f"level_matched/{anon}.wav",
                        "target_text": target_text(case_id)})
    entries = [entries[i] for i in order]

    (LISTEN / "manifest.json").write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    (LISTEN / "answer_key.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8")

    html = HTML_TEMPLATE.replace("__ENTRIES__", json.dumps(entries, ensure_ascii=False))
    (LISTEN / "index.html").write_text(html, encoding="utf-8")
    print(f"listen pack: {len(entries)} clips -> {LISTEN}")
    return 0


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>守岸人盲听 v1</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#14161a;color:#e8e8ea}
h1{font-size:20px}.tgt{color:#9db4d0;font-size:14px;margin:14px 0 4px}
.card{background:#1d2026;border:1px solid #333;border-radius:8px;padding:12px;margin:8px 0}
audio{width:100%;height:36px}
.rate{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:13px}
.rate label{display:flex;align-items:center;gap:6px}
select,textarea{background:#111;color:#e8e8ea;border:1px solid #444;border-radius:4px;padding:2px 6px}
textarea{width:100%;margin-top:6px;min-height:34px}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:10px 20px;font-size:15px;cursor:pointer}
.note{color:#8a8f98;font-size:12px}
.lm{cursor:pointer;color:#6fa8ff;font-size:12px;margin-left:8px}
</style></head><body>
<h1>守岸人盲听 v1</h1>
<p class="note">每条三个评分：像守岸人(1-5) / 沙·粗糙(1不沙-5非常沙) / 自然度(1-5)。LM 开关切换响度匹配副本(仅调音量)。听完点底部导出。</p>
<div id="app"></div>
<button id="export">导出评分 JSON</button>
<script>
const ENTRIES = __ENTRIES__;
const state = JSON.parse(localStorage.getItem('shouanren_blind_v1')||'{}');
const app = document.getElementById('app');
let curTgt = null;
for(const e of ENTRIES){
  if(e.target_text!==curTgt){curTgt=e.target_text;
    const d=document.createElement('div');d.className='tgt';d.textContent='目标文本：'+curTgt;app.appendChild(d);}
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<b>${e.anon}</b><span class="lm" data-a="${e.anon}">[LM]</span><br>
    <audio controls preload="none" src="${e.file}" data-raw="${e.file}" data-lm="${e.lm_file}"></audio>
    <div class="rate">
      <label>像守岸人 <select data-k="sim"><option value="">-</option>${[1,2,3,4,5].map(v=>`<option>${v}</option>`).join('')}</select></label>
      <label>沙/粗糙 <select data-k="rough"><option value="">-</option>${[1,2,3,4,5].map(v=>`<option>${v}</option>`).join('')}</select></label>
      <label>自然度 <select data-k="nat"><option value="">-</option>${[1,2,3,4,5].map(v=>`<option>${v}</option>`).join('')}</select></label>
    </div>
    <textarea placeholder="备注" data-k="note"></textarea>`;
  const s=state[e.anon]||{};
  c.querySelectorAll('select,textarea').forEach(el=>{
    if(s[el.dataset.k])el.value=s[el.dataset.k];
    el.addEventListener('change',()=>{state[e.anon]=state[e.anon]||{};state[e.anon][el.dataset.k]=el.value;
      localStorage.setItem('shouanren_blind_v1',JSON.stringify(state));});
  });
  c.querySelector('.lm').addEventListener('click',ev=>{
    const au=c.querySelector('audio');
    au.src = au.src.endsWith(e.file)? e.lm_file : e.file; au.play();
  });
  app.appendChild(c);
}
document.getElementById('export').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify({ratings:state},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='shouanren_blind_v1.json';a.click();
});
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
