from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.quality_report.read_text(encoding="utf-8"))
    assets = list(report["assets"])

    selected: dict[str, dict] = {}

    def add(asset: dict) -> None:
        selected.setdefault(asset["asset_sha256"], asset)

    # Preserve every known resampling, long-tail, and positive true-peak edge case.
    for asset in assets:
        if (
            asset.get("sample_rate") != 44_100
            or (asset.get("true_peak_estimate_dbtp") or -999.0) > 0.0
            or (asset.get("trailing_silence_seconds") or 0.0) > 0.5
        ):
            add(asset)

    # The sad class has only six clips; cover all of them in the prototype.
    for asset in assets:
        if asset.get("emotion_weak_label") == "难过_sad":
            add(asset)

    minimums = {"中立_neutral": 4, "开心_happy": 4, "生气_angry": 10}
    for emotion, minimum in minimums.items():
        candidates = sorted(
            (asset for asset in assets if asset.get("emotion_weak_label") == emotion),
            key=lambda asset: (
                asset.get("decision") != "review",
                -(asset.get("silence_ratio") or 0.0),
                asset["asset_sha256"],
            ),
        )
        for candidate in candidates:
            current = sum(
                item.get("emotion_weak_label") == emotion for item in selected.values()
            )
            if current >= minimum:
                break
            add(candidate)

    if len(selected) != 24:
        raise RuntimeError(f"Expected 24 prototype assets, selected {len(selected)}")
    ordered = sorted(selected.values(), key=lambda asset: asset["asset_sha256"])
    payload = {
        "schema_version": 1,
        "selection": "all known 36 kHz, >0 dBTP, >0.5 s tail and sad assets; balanced diagnostic fill",
        "count": len(ordered),
        "emotion_counts": {
            emotion: sum(item.get("emotion_weak_label") == emotion for item in ordered)
            for emotion in sorted(
                {str(item.get("emotion_weak_label")) for item in ordered}
            )
        },
        "assets": [
            {
                "asset_sha256": item["asset_sha256"],
                "source_relative_path": item.get("source_relative_path"),
                "emotion_weak_label": item.get("emotion_weak_label"),
                "sample_rate": item.get("sample_rate"),
                "true_peak_estimate_dbtp": item.get("true_peak_estimate_dbtp"),
                "trailing_silence_seconds": item.get("trailing_silence_seconds"),
                "decision": item.get("decision"),
                "reasons": item.get("reasons"),
            }
            for item in ordered
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["emotion_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
