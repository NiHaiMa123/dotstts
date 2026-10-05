#!/usr/bin/env python3
"""Build the Slice12 fixed baseline matrix from the approved Slice10 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lab/slice12/experiment_contract_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    source = json.loads((ROOT / cfg["dataset"]["source_manifest"]).read_text(encoding="utf-8"))
    ranking = json.loads((ROOT / cfg["dataset"]["reference_ranking"]).read_text(encoding="utf-8"))
    source_by_asset: dict[str, dict] = {}
    for pool in source["pools"].values():
        for item in pool["items"]:
            source_by_asset.setdefault(str(item["asset_sha256"]), item)
    approved = [
        row for row in ranking["candidates"]
        if str(row.get("manual_status", "")).startswith("approved")
    ]
    if len(approved) != int(cfg["dataset"]["approved_reference_count"]):
        raise RuntimeError(f"Expected {cfg['dataset']['approved_reference_count']} approved references, got {len(approved)}")
    rows: list[dict] = []
    for ordinal, candidate in enumerate(sorted(approved, key=lambda r: int(r["final_rank"])), start=1):
        asset = str(candidate["asset_sha256"])
        item = source_by_asset.get(asset)
        if item is None:
            raise KeyError(f"Approved asset missing from source manifest: {asset}")
        standardized = ROOT / "data" / "work" / "standardized" / str(item["audio_relative_path"])
        if not standardized.is_file():
            raise FileNotFoundError(f"Standardized reference missing: {standardized}")
        actual_sha = sha256_file(standardized)
        if actual_sha != str(item["audio_sha256"]):
            raise RuntimeError(f"Standardized SHA drift for {asset}: expected {item['audio_sha256']}, got {actual_sha}")
        rows.append(
            {
                "asset_sha256": asset,
                "audio_relative_path": str(item["audio_relative_path"]),
                "audio_sha256": str(item["audio_sha256"]),
                "ordinal": ordinal,
                "pool_ids": list(candidate.get("pool_ids", [])),
                "prompt_text_sha256": str(item["text_sha256"]),
                "review_status": str(item["review_status"]),
                "speaker_id": str(item["speaker_id"]),
                "text_exact": str(item["text_exact"]),
                "text_sha256": str(item["text_sha256"]),
                "final_rank": int(candidate["final_rank"]),
                "static_composite_score": float(candidate["static_composite_score"]),
            }
        )
    output_name = cfg["outputs"].get("reference_manifest", cfg["outputs"].get("baseline_manifest"))
    if not output_name:
        raise ValueError("Slice12 contract must declare outputs.reference_manifest")
    output = ROOT / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "experiment_id": cfg["experiment_id"],
        "reference_count": len(rows),
        "sentence_count": len(cfg["evaluation"]["sentences"]),
        "seed_count": len(cfg["evaluation"]["seeds"]),
        "expected_generation_count": len(rows) * len(cfg["evaluation"]["sentences"]) * len(cfg["evaluation"]["seeds"]),
        "manifest_path": str(output.resolve()),
        "manifest_sha256": sha256_file(output),
        "source_manifest_sha256": sha256_file(ROOT / cfg["dataset"]["source_manifest"]),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
