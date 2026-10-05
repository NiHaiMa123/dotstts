from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_or_default(value: Any, default: float) -> float:
    """Return a numeric metric without treating the valid value 0 as missing."""
    return default if value is None else float(value)


def run(config_path: Path, decisions_path: Path) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    key_path = (repo_root / outputs["blind_listening_key_path"]).resolve()
    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    if decisions.get("schema_version") != 1 or decisions.get("status") != "manual_review_export":
        raise ValueError("Unsupported Slice10 manual decision export")
    ratings = decisions.get("ratings")
    if not isinstance(ratings, dict):
        raise ValueError("Manual decision export must contain a ratings mapping")
    key_items = key_payload.get("items", [])
    key_by_id = {str(item["blind_id"]): item for item in key_items}
    rating_ids = set(ratings)
    key_ids = set(key_by_id)
    if rating_ids != key_ids:
        raise RuntimeError(f"Manual decision IDs mismatch: missing={len(key_ids-rating_ids)}, unexpected={len(rating_ids-key_ids)}")
    allowed = {"keep", "reject", "uncertain"}
    for blind_id, value in ratings.items():
        if not isinstance(value, dict) or value.get("rating") not in allowed:
            raise ValueError(f"Invalid rating for {blind_id}")
        if not value.get("saved_at"):
            raise ValueError(f"Missing saved_at for {blind_id}")
        datetime.fromisoformat(str(value["saved_at"]).replace("Z", "+00:00"))
    if any(value["rating"] == "uncertain" for value in ratings.values()):
        raise RuntimeError("Slice10 manual gate still contains uncertain ratings")
    if key_payload.get("benchmark_manifest_sha256") != outputs["benchmark_manifest_sha256"]:
        raise RuntimeError("Blind key benchmark manifest hash mismatch")

    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for blind_id, value in ratings.items():
        key = key_by_id[blind_id]
        by_asset[str(key["asset_sha256"])].append({
            "blind_id": blind_id,
            "sentence_id": key["sentence_id"],
            "rating": value["rating"],
            "note": value.get("note"),
            "saved_at": value["saved_at"],
        })
    expected_sentences = int(config["manual_gate"]["required_sentence_count"])
    minimum_keep = int(config["manual_gate"]["minimum_keep_count"])
    ranking = json.loads((repo_root / outputs["ranking_path"]).resolve().read_text(encoding="utf-8"))
    ranking_by_asset = {row["asset_sha256"]: row for row in ranking["candidates"]}
    candidates = []
    for asset, rows in sorted(by_asset.items()):
        if len(rows) != expected_sentences:
            raise RuntimeError(f"Candidate {asset} has {len(rows)} ratings; expected {expected_sentences}")
        keep_count = sum(row["rating"] == "keep" for row in rows)
        reject_count = sum(row["rating"] == "reject" for row in rows)
        status = "approved_majority" if keep_count >= minimum_keep and keep_count > reject_count else "rejected_majority"
        if keep_count == expected_sentences:
            status = "approved_unanimous"
        if reject_count == expected_sentences:
            status = "rejected_unanimous"
        row = dict(ranking_by_asset[asset])
        row.update({
            "manual_keep_count": keep_count,
            "manual_reject_count": reject_count,
            "manual_uncertain_count": 0,
            "manual_keep_rate": keep_count / expected_sentences,
            "manual_status": status,
            "manual_ratings": rows,
        })
        candidates.append(row)

    final_order = sorted(
        candidates,
        key=lambda row: (
            not row["manual_status"].startswith("approved"),
            -int(row["manual_keep_count"]),
            -float(row["closed_loop_score"]),
            numeric_or_default(row["mean_cer"], 1.0),
            row["asset_sha256"],
        ),
    )
    for index, row in enumerate(final_order, start=1):
        row["final_rank"] = index
    approved = [row for row in final_order if row["manual_status"].startswith("approved")]
    review_report = {
        "schema_version": 1,
        "status": "succeeded",
        "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
        "decision_source_path": str(decisions_path.resolve()),
        "decision_source_sha256": sha256_file(decisions_path),
        "blind_key_sha256": sha256_file(key_path),
        "candidate_policy": config["manual_gate"]["candidate_policy"],
        "required_sentence_count": expected_sentences,
        "minimum_keep_count": minimum_keep,
        "rating_count": len(ratings),
        "rating_counts": {rating: sum(value["rating"] == rating for value in ratings.values()) for rating in sorted(allowed)},
        "candidate_count": len(candidates),
        "approved_candidate_count": len(approved),
        "rejected_candidate_count": len(candidates) - len(approved),
        "candidates": candidates,
    }
    report_path = (repo_root / outputs["manual_review_report_path"]).resolve()
    report_path.write_text(json.dumps(review_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    final_payload = {
        "schema_version": 1,
        "status": "succeeded",
        "benchmark_id": config["benchmark_id"],
        "benchmark_version": config["benchmark_version"],
        "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
        "ranking_policy": "approved majority first; manual keep count, closed-loop score, CER, asset SHA-256",
        "approved_candidate_count": len(approved),
        "candidate_count": len(candidates),
        "candidates": final_order,
        "pool_rankings": {
            pool: [row["asset_sha256"] for row in approved if pool in row["pool_ids"]]
            for pool in sorted({pool for row in candidates for pool in row["pool_ids"]})
        },
    }
    final_path = (repo_root / outputs["final_ranking_path"]).resolve()
    final_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    csv_path = (repo_root / outputs["final_ranking_csv_path"]).resolve()
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["final_rank", "manual_status", "manual_keep_count", "manual_reject_count", "asset_sha256", "pool_ids", "closed_loop_rank", "closed_loop_score", "static_rank", "mean_cer", "mean_speaker_cosine"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in final_order:
            writer.writerow({field: (" ".join(row["pool_ids"]) if field == "pool_ids" else row.get(field)) for field in fields})
    html_path = (repo_root / outputs["final_ranking_html_path"]).resolve()
    rows_html = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2%}</td><td>{:.4f}</td><td>{:.4f}</td><td>{}</td></tr>".format(row["final_rank"], html.escape(row["manual_status"]), row["manual_keep_count"], row["manual_reject_count"], row["manual_keep_rate"], row["closed_loop_score"], numeric_or_default(row["mean_cer"], 1.0), html.escape(row["asset_sha256"][:12]))
        for row in final_order
    )
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><title>Slice10 final ranking</title><style>body{{font:14px system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.4rem}}th{{background:#f3f4f6}}</style><h1>Slice10 final ranking after blind review</h1><p>多数票规则：{expected_sentences} 个句子中至少 {minimum_keep} 个 keep，且 keep 多于 reject；闭环分数与静态分数仍单独保留。</p><table><tr><th>最终#</th><th>人工状态</th><th>keep</th><th>reject</th><th>keep率</th><th>闭环分</th><th>均值 CER</th><th>资产</th></tr>" + rows_html + "</table>\n", encoding="utf-8", newline="\n")
    return {"status": "succeeded", "rating_count": len(ratings), "candidate_count": len(candidates), "approved_candidate_count": len(approved), "manual_review_report_path": str(report_path), "final_ranking_path": str(final_path), "final_ranking_html_path": str(html_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the exported Slice10 blind-listening decisions.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.decisions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
