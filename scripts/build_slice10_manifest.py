from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Slice10 config must be a schema_version=1 mapping")
    return payload


def build(config_path: Path) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = load_config(config_path)
    selection = config["selection"]
    source_manifest = (repo_root / selection["manifest_path"]).resolve()
    if sha256_file(source_manifest) != str(selection["manifest_sha256"]).lower():
        raise RuntimeError("Slice9 reference manifest SHA-256 drift")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "frozen":
        raise RuntimeError("Slice9 reference manifest is not frozen")

    standardized_root = (repo_root / selection["standardized_audio_root"]).resolve()
    by_asset: dict[str, dict[str, Any]] = {}
    pool_ids: dict[str, list[str]] = {}
    for pool_id, pool in sorted(source.get("pools", {}).items()):
        for item in pool.get("items", []):
            asset = str(item["asset_sha256"])
            if asset in by_asset:
                previous = by_asset[asset]
                if previous["audio_sha256"] != item["audio_sha256"] or previous["text_exact"] != item["text_exact"]:
                    raise RuntimeError(f"Conflicting duplicate candidate: {asset}")
            else:
                relative = str(item["audio_relative_path"])
                audio_path = (standardized_root / relative).resolve()
                audio_path.relative_to(standardized_root)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Standardized reference is missing: {audio_path}")
                actual_audio = sha256_file(audio_path)
                if actual_audio != str(item["audio_sha256"]).lower():
                    raise RuntimeError(
                        f"Reference audio SHA-256 drift for {asset}: expected {item['audio_sha256']}, got {actual_audio}"
                    )
                by_asset[asset] = {
                    "asset_sha256": asset,
                    "audio_sha256": str(item["audio_sha256"]),
                    "audio_relative_path": relative,
                    "speaker_id": str(item["speaker_id"]),
                    "text_exact": str(item["text_exact"]),
                    "text_sha256": str(item["text_sha256"]),
                    "review_status": str(item["review_status"]),
                    "static_composite_score": float(item["composite_score"]),
                }
            pool_ids.setdefault(asset, []).append(pool_id)

    if not by_asset:
        raise RuntimeError("Slice10 manifest contains no candidates")
    if any(row["speaker_id"] != selection["target_speaker_id"] for row in by_asset.values()):
        raise RuntimeError("Slice10 candidate speaker mismatch")
    if any(row["review_status"] != "approved" for row in by_asset.values()):
        raise RuntimeError("Slice10 candidate set contains unapproved references")

    rows = []
    for ordinal, asset in enumerate(sorted(by_asset), start=1):
        row = dict(by_asset[asset])
        row["ordinal"] = ordinal
        row["pool_ids"] = sorted(set(pool_ids[asset]))
        row["prompt_text_sha256"] = hashlib.sha256(row["text_exact"].encode("utf-8")).hexdigest()
        rows.append(row)

    outputs = config["outputs"]
    output_path = (repo_root / outputs["benchmark_manifest_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output_path.write_text(content, encoding="utf-8", newline="\n")
    manifest_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    summary = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "benchmark_version": config["benchmark_version"],
        "config_path": str(config_path.resolve()),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": str(selection["manifest_sha256"]).lower(),
        "benchmark_manifest_path": str(output_path),
        "benchmark_manifest_sha256": manifest_hash,
        "candidate_count": len(rows),
        "pool_counts": {pool: sum(pool in row["pool_ids"] for row in rows) for pool in sorted({p for ids in pool_ids.values() for p in ids})},
        "unique_audio_count": len({row["audio_sha256"] for row in rows}),
        "status": "succeeded",
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Slice10 reference benchmark manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    args = parser.parse_args()
    summary = build(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
