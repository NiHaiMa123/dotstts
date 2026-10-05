#!/usr/bin/env python3
"""Build a deterministic 8-row tiny manifest from the frozen Fuxuan v1 references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-manifest",
        default="data/references/fuxuan/v1/manifest.json",
    )
    parser.add_argument(
        "--output",
        default="data/reports/datasets/fuxuan_v1/slice11/tiny_overfit_train.jsonl",
    )
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 8 or args.count > 32:
        raise ValueError("Slice11 tiny manifest count must be in [8, 32].")
    root = Path(__file__).resolve().parents[1]
    source_path = (root / args.reference_manifest).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    selected: list[dict] = []
    seen: set[str] = set()
    for pool in payload["pools"].values():
        for item in pool["items"]:
            fid = str(item["fid"])
            if fid in seen:
                continue
            audio_path = root / "data" / "raw" / "sha256" / fid[:2] / f"{fid}.wav"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing frozen source audio: {audio_path}")
            seen.add(fid)
            selected.append(
                {
                    "fid": fid,
                    "audio": str(audio_path.resolve()),
                    "text": str(item["text_exact"]),
                    "pool": str(item.get("emotion_primary", "unknown")),
                    "source_audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                }
            )
            if len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Only found {len(selected)} usable records; wanted {args.count}.")

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fout:
        for row in selected:
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "slice11_tiny_overfit_v1",
                "count": len(selected),
                "source_manifest": str(source_path),
                "records": selected,
                "selection_policy": "first unique usable record per frozen pool traversal",
                "synthetic_repetition": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "metadata": str(metadata_path), "count": len(selected)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
