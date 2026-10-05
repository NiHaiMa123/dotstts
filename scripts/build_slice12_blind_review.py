#!/usr/bin/env python3
"""Build a deterministic paired blind review for Slice12 control vs LoRA."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "data/reports/datasets/fuxuan_v1/slice12"
MANIFEST_SHA256 = "ddfb42d6211f1c882bf77fe3c273dc7423fd52a8be3824a5650e66a6f43927b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relative_to_report(path: Path) -> str:
    return os.path.relpath(path, REPORT_ROOT).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-generation",
        type=Path,
        default=REPORT_ROOT / "soar_control_generation.jsonl",
    )
    parser.add_argument(
        "--trained-generation",
        type=Path,
        default=REPORT_ROOT / "soar_lora_generation.jsonl",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "data/work/slice12/blind_review_v1",
    )
    parser.add_argument("--output-stem", default="slice12_blind_review")
    parser.add_argument("--storage-key", default="slice12-blind-review-v1")
    parser.add_argument("--download-name", default="slice12-blind-review-decisions.json")
    parser.add_argument(
        "--mapping-salt",
        default="",
        help="Optional review-specific salt so a later candidate does not reuse an earlier A/B mapping.",
    )
    args = parser.parse_args()

    control_path = args.control_generation.resolve()
    trained_path = args.trained_generation.resolve()
    work_root = args.work_root.resolve()
    control_rows = load_jsonl(control_path)
    trained_rows = load_jsonl(trained_path)
    control = {
        (int(row["ordinal"]), row["sentence_id"], int(row["seed"])): row
        for row in control_rows
    }
    trained = {
        (int(row["ordinal"]), row["sentence_id"], int(row["seed"])): row
        for row in trained_rows
    }
    if set(control) != set(trained) or len(control) != 72:
        raise RuntimeError("Control and trained generation matrices do not match")
    if any(row.get("status") != "ok" for row in [*control_rows, *trained_rows]):
        raise RuntimeError("Blind review requires successful generation rows")

    sentence_order = [
        "neutral_report",
        "named_entity",
        "long_decision",
        "date_number",
        "short_ack",
        "out_of_domain",
    ]
    seeds = [20260908, 20260909]
    work_root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for ordinal, sentence_id in enumerate(sentence_order, start=1):
        for seed_index, seed in enumerate(seeds, start=1):
            source_rows = {
                "control": control[(ordinal, sentence_id, seed)],
                "trained": trained[(ordinal, sentence_id, seed)],
            }
            pair_id = f"P{ordinal:02d}-{seed_index}"
            mapping = ["control", "trained"]
            mapping_material = f"{MANIFEST_SHA256}|{pair_id}"
            if args.mapping_salt:
                mapping_material = f"{MANIFEST_SHA256}|{args.mapping_salt}|{pair_id}"
            swap_key = hashlib.sha256(mapping_material.encode()).digest()[0]
            if swap_key % 2:
                mapping.reverse()
            blind_paths: dict[str, Path] = {}
            for label, role in zip(("A", "B"), mapping, strict=True):
                source = ROOT / source_rows[role]["output_path"]
                if not source.is_file() or sha256_file(source) != source_rows[role]["output_sha256"]:
                    raise RuntimeError(f"Generated audio hash mismatch: {source}")
                target = work_root / f"{pair_id}-{label}.wav"
                shutil.copy2(source, target)
                blind_paths[label] = target
            items.append(
                {
                    "pair_id": pair_id,
                    "sentence_id": sentence_id,
                    "text": source_rows["control"]["text"],
                    "seed": seed,
                    "audio_a": relative_to_report(blind_paths["A"]),
                    "audio_b": relative_to_report(blind_paths["B"]),
                }
            )
            key_rows.append(
                {
                    "pair_id": pair_id,
                    "ordinal": ordinal,
                    "sentence_id": sentence_id,
                    "seed": seed,
                    "A": mapping[0],
                    "B": mapping[1],
                    "control_sha256": source_rows["control"]["output_sha256"],
                    "trained_sha256": source_rows["trained"]["output_sha256"],
                }
            )

    manifest_path = REPORT_ROOT / f"{args.output_stem}.json"
    key_path = REPORT_ROOT / f"{args.output_stem}_key.json"
    html_path = REPORT_ROOT / f"{args.output_stem}.html"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pending_manual_review",
                "benchmark_manifest_sha256": MANIFEST_SHA256,
                "mapping_salt_sha256": (
                    hashlib.sha256(args.mapping_salt.encode()).hexdigest()
                    if args.mapping_salt
                    else None
                ),
                "control_generation": relative_to_report(control_path),
                "trained_generation": relative_to_report(trained_path),
                "pair_count": len(items),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "sealed_key",
                "control_generation_sha256": sha256_file(control_path),
                "trained_generation_sha256": sha256_file(trained_path),
                "items": key_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cards = []
    for item in items:
        cards.append(
            "<article data-id='{pair_id}'><h2>{pair_id} · {sentence}</h2>"
            "<p>{text}</p><div class='audios'><label>A<audio controls preload='none' src='{a}'></audio></label>"
            "<label>B<audio controls preload='none' src='{b}'></audio></label></div>"
            "<div class='review'><label>整体偏好 <select class='choice'><option value=''>未选择</option>"
            "<option value='A'>A 更好</option><option value='B'>B 更好</option><option value='tie'>相当</option>"
            "<option value='both_bad'>都不可用</option></select></label>"
            "<label>前导静音 <select class='silence'><option value=''>未选择</option><option value='A_worse'>A 更差</option>"
            "<option value='B_worse'>B 更差</option><option value='same'>相当</option><option value='neither'>都可接受</option></select></label>"
            "<input class='note' placeholder='可懂度、音色、杂音或其他备注'><button onclick='saveOne(this)'>保存本条</button>"
            "<span class='state'>未保存</span></div></article>".format(
                pair_id=html.escape(item["pair_id"]),
                sentence=html.escape(item["sentence_id"]),
                text=html.escape(item["text"]),
                a=html.escape(item["audio_a"]),
                b=html.escape(item["audio_b"]),
            )
        )
    page = """<!doctype html><meta charset='utf-8'><title>Slice12 SOAR vs LoRA 盲审</title>
<style>body{font:14px system-ui;margin:2rem auto;max-width:1050px;color:#17202a}header{position:sticky;top:0;background:#fff;padding:.8rem 0;border-bottom:1px solid #ddd;z-index:2}article{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0}.audios{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.audios label{font-weight:700}.audios audio{display:block;width:100%;margin-top:.35rem}.review{display:grid;grid-template-columns:1fr 1fr 2fr auto;gap:.6rem;align-items:end;margin-top:.8rem}.review label{display:grid;gap:.3rem}.review input,.review select,.review button{padding:.45rem}.state{grid-column:1/-1;color:#68707a}.state.saved{color:#087f5b;font-weight:700}.global{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}button{cursor:pointer}@media(max-width:760px){.audios,.review{grid-template-columns:1fr}}</style>
<header><h1>Slice12 配对盲审</h1><p>共 12 对，覆盖 6 个参考、6 类句子和 2 个 seed。只按听感选择，不查看 sealed key；导出后再统一解盲。</p><div class='global'><strong id='progress'>已保存 0/12</strong><label>审听状态 <select id='reviewStatus'><option value=''>未选择</option><option value='review_complete'>已审完，可以解盲</option><option value='needs_more_review'>需要更多试听</option></select></label><button onclick='exportReviews()'>导出审核结果</button></div></header>
""" + "\n".join(cards) + """
<script>const STORE='slice12-blind-review-v1';function read(){try{return JSON.parse(localStorage.getItem(STORE)||'{"ratings":{}}')}catch(e){return {ratings:{}}}}function write(v){localStorage.setItem(STORE,JSON.stringify(v))}function update(){const r=read(),n=Object.keys(r.ratings||{}).length;document.getElementById('progress').textContent=`已保存 ${n}/12`;document.getElementById('reviewStatus').value=r.review_status||''}function saveOne(btn){const a=btn.closest('article'),id=a.dataset.id,c=a.querySelector('.choice').value,s=a.querySelector('.silence').value,state=a.querySelector('.state');if(!c||!s){state.textContent='请先完成两项选择';return}const r=read();r.ratings=r.ratings||{};r.ratings[id]={preference:c,leading_silence:s,note:a.querySelector('.note').value,saved_at:new Date().toISOString()};write(r);state.textContent='已保存 '+new Date(r.ratings[id].saved_at).toLocaleString();state.classList.add('saved');update()}function restore(){const r=read();document.querySelectorAll('article').forEach(a=>{const v=(r.ratings||{})[a.dataset.id];if(!v)return;a.querySelector('.choice').value=v.preference||'';a.querySelector('.silence').value=v.leading_silence||'';a.querySelector('.note').value=v.note||'';const s=a.querySelector('.state');s.textContent='已保存 '+new Date(v.saved_at).toLocaleString();s.classList.add('saved')});update()}document.getElementById('reviewStatus').addEventListener('change',e=>{const r=read();r.review_status=e.target.value;r.review_status_saved_at=new Date().toISOString();write(r);update()});function exportReviews(){const r=read();if(Object.keys(r.ratings||{}).length!==12||!r.review_status){alert('请保存全部 12 条并选择审听状态');return}const payload={schema_version:1,status:'manual_review_export',ratings:r.ratings,review_status:r.review_status,review_status_saved_at:r.review_status_saved_at},blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='slice12-blind-review-decisions.json';a.click();URL.revokeObjectURL(u)}restore();</script>
"""
    page = page.replace("slice12-blind-review-v1", args.storage_key)
    page = page.replace("slice12-blind-review-decisions.json", args.download_name)
    html_path.write_text(page, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "pending_manual_review",
                "pair_count": len(items),
                "html": str(html_path),
                "key": str(key_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
