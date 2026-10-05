from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(config_path: Path) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    generation = load_jsonl((repo_root / outputs["generation_manifest_path"]).resolve())
    ranking = json.loads((repo_root / outputs["ranking_path"]).resolve().read_text(encoding="utf-8"))
    if ranking.get("status") != "succeeded":
        raise RuntimeError("Cannot build blind listening package from incomplete ranking")
    # One fixed seed and the same neutral sentence make the primary blind block
    # a controlled comparison. Additional sentence blocks test transfer without
    # revealing the model-closed-loop score or static rank to the listener.
    selected = [row for row in generation if row["seed"] == int(config["seeds"][0])]
    sentence_order = [sentence["sentence_id"] for sentence in config["test_sentences"]]
    by_sentence_asset = {(row["sentence_id"], row["asset_sha256"]): row for row in selected}
    assets = sorted({row["asset_sha256"] for row in selected})
    if len(assets) != int(ranking["candidate_count"]):
        raise RuntimeError("Blind listening candidate count does not match ranking")

    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for sentence_id in sentence_order:
        ordered = sorted(assets, key=lambda asset: hashlib.sha256(f"{outputs['benchmark_manifest_sha256']}|{sentence_id}|{asset}".encode()).hexdigest())
        for position, asset in enumerate(ordered, start=1):
            row = by_sentence_asset[(sentence_id, asset)]
            blind_id = f"{sentence_id[:4].upper()}-{position:02d}"
            items.append({
                "blind_id": blind_id,
                "sentence_id": sentence_id,
                "seed": int(row["seed"]),
                "text": row["text"],
                "audio_path": row["output_path"],
            })
            candidate = next(candidate for candidate in ranking["candidates"] if candidate["asset_sha256"] == asset)
            key.append({
                "blind_id": blind_id,
                "sentence_id": sentence_id,
                "asset_sha256": asset,
                "ordinal": candidate["ordinal"],
                "static_rank": candidate["static_rank"],
                "closed_loop_rank": candidate["closed_loop_rank"],
                "closed_loop_score": candidate["closed_loop_score"],
                "pool_ids": candidate["pool_ids"],
            })

    root = (repo_root / outputs["run_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    blind_path = (repo_root / outputs["blind_listening_path"]).resolve()
    blind_path.write_text(json.dumps({"schema_version": 1, "status": "ready", "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"], "instructions": "同一 sentence_id 内按盲号试听；不要依据文件名推断排名。", "items": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    key_path = (repo_root / outputs["blind_listening_key_path"]).resolve()
    key_path.write_text(json.dumps({"schema_version": 1, "status": "sealed_key", "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"], "items": key}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    review_template_path = (repo_root / outputs["blind_review_template_path"]).resolve()
    review_template_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pending_manual_review",
                "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
                "ratings": [
                    {"blind_id": item["blind_id"], "sentence_id": item["sentence_id"], "rating": None, "note": None, "saved_at": None}
                    for item in items
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rows = []
    for sentence_id in sentence_order:
        sentence_items = [item for item in items if item["sentence_id"] == sentence_id]
        controls = "\n".join(
            "<article data-blind-id='{id}'><label><strong>{id}</strong></label><audio controls preload='none' src='{src}'></audio><div class='review'><select class='rating'><option value=''>未选择</option><option value='keep'>保留</option><option value='reject'>淘汰</option><option value='uncertain'>不确定</option></select><input class='note' placeholder='备注（可选）'><button onclick='saveOne(this)'>保存本条</button><span class='state'>未保存</span></div></article>".format(
                id=html.escape(item["blind_id"]),
                src=html.escape(os.path.relpath(repo_root / item["audio_path"], (repo_root / outputs["blind_listening_html_path"]).resolve().parent).replace("\\", "/")),
            )
            for item in sentence_items
        )
        rows.append(f"<section><h2>{html.escape(sentence_id)}</h2><p>{html.escape(sentence_items[0]['text'])}</p>{controls}</section>")
    html_path = (repo_root / outputs["blind_listening_html_path"]).resolve()
    html_path.write_text("<!doctype html><meta charset='utf-8'><title>Slice10 blind listening</title><style>body{font:14px system-ui;margin:2rem;max-width:1100px}section{border-top:1px solid #ddd;padding:1rem 0}article{display:grid;grid-template-columns:100px 1fr;gap:.5rem 1rem;align-items:center;margin:.7rem 0}.review{grid-column:2;display:flex;gap:.4rem;align-items:center}.review input{flex:1;padding:.3rem}.state{color:#666;font-size:.85rem}.state.saved{color:#087f5b;font-weight:600}audio{width:100%}button{padding:.3rem .7rem}#export{position:fixed;right:2rem;top:1.2rem}</style><button id='export' onclick='exportReviews()'>导出审核结果</button><h1>Slice10 盲听包</h1><p>每个区块内文本、seed 相同；请只记录盲号的可懂度、音质、音色一致性和明显失败，不要查看 sealed key。选择评级后点击“保存本条”，状态会显示“已保存”。</p>" + "\n".join(rows) + "<script>const STORE='slice10-blind-review-v1';function read(){try{return JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){return {}}}function save(){localStorage.setItem(STORE,JSON.stringify(read()))}function saveOne(btn){const a=btn.closest('article'),id=a.dataset.blindId,r=read(),rating=a.querySelector('.rating').value;if(!rating){a.querySelector('.state').textContent='请先选择评级';return}r[id]={rating,note:a.querySelector('.note').value,saved_at:new Date().toISOString()};localStorage.setItem(STORE,JSON.stringify(r));const s=a.querySelector('.state');s.textContent='已保存 '+new Date(r[id].saved_at).toLocaleString();s.classList.add('saved')}function restore(){const r=read();document.querySelectorAll('article[data-blind-id]').forEach(a=>{const v=r[a.dataset.blindId];if(!v)return;a.querySelector('.rating').value=v.rating||'';a.querySelector('.note').value=v.note||'';const s=a.querySelector('.state');s.textContent='已保存 '+new Date(v.saved_at).toLocaleString();s.classList.add('saved')})}function exportReviews(){const blob=new Blob([JSON.stringify({schema_version:1,status:'manual_review_export',ratings:read()},null,2)],{type:'application/json'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='slice10-blind-review-decisions.json';a.click();URL.revokeObjectURL(u)}restore();</script>\n", encoding="utf-8", newline="\n")
    return {"status": "ready", "item_count": len(items), "sentence_count": len(sentence_order), "blind_listening_path": str(blind_path), "blind_listening_html_path": str(html_path), "blind_listening_key_path": str(key_path), "blind_review_template_path": str(review_template_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a blinded Slice10 listening package.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
