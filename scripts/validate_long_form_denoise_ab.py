from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_contract import load_long_form_config
from dots_tts_lab.speaker_embedding import compute_speaker_embedding, load_speaker_encoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a long-form denoise AB package.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("data/reports/long_form/denoise_ab_v1/manifest.json"),
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if manifest["case_count"] != len(cases) or len({x["segment_id"] for x in cases}) != len(cases):
        raise RuntimeError("invalid or duplicate denoise AB cases")
    config = load_long_form_config()
    encoder = load_speaker_encoder(config.speaker.encoder)
    cosines: list[float] = []
    noise_floor_deltas: list[float] = []
    bucket_deltas: dict[str, list[float]] = {}
    results = []
    for case in cases:
        loaded = {}
        for kind in ("raw", "enhanced"):
            item = case[kind]
            path = (root / item["relative_path"]).resolve()
            path.relative_to(root)
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"audio missing or hash drift: {path}")
            info = sf.info(path)
            if info.samplerate != 48000 or info.channels != 1 or info.subtype != "PCM_24":
                raise RuntimeError(f"unexpected audio contract: {path}: {info}")
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            loaded[kind] = (path, samples, int(sample_rate))
        if len(loaded["raw"][1]) != len(loaded["enhanced"][1]):
            raise RuntimeError(f"duration drift: {case['case_id']}")
        vectors = {}
        for kind in ("raw", "enhanced"):
            _, samples, sample_rate = loaded[kind]
            vectors[kind] = compute_speaker_embedding(
                samples,
                sample_rate=sample_rate,
                config=config.speaker.encoder,
                encoder=encoder,
            )
        cosine = float(np.dot(vectors["raw"], vectors["enhanced"]))
        cosines.append(cosine)
        before = case["raw"]["metrics"].get("noise_floor_proxy_dbfs")
        after = case["enhanced"]["metrics"].get("noise_floor_proxy_dbfs")
        delta = None if before is None or after is None else float(after) - float(before)
        if delta is not None:
            noise_floor_deltas.append(delta)
            bucket_deltas.setdefault(case["noise_bucket"], []).append(delta)
        blind = case["blind"]
        if {blind["a_kind"], blind["b_kind"]} != {"raw", "enhanced"}:
            raise RuntimeError(f"invalid blind mapping: {case['case_id']}")
        results.append(
            {
                "case_id": case["case_id"],
                "speaker_cosine_raw_vs_enhanced": cosine,
                "noise_floor_delta_db": delta,
            }
        )
    report = {
        "schema_version": 1,
        "status": "passed_technical_checks",
        "decision": "human_review_required",
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256(manifest_path),
        "case_count": len(cases),
        "audio_file_count": len(cases) * 2,
        "speaker_cosine": {
            "minimum": min(cosines),
            "p50": percentile(cosines, 50),
            "mean": float(np.mean(cosines)),
            "below_0_95": sum(value < 0.95 for value in cosines),
        },
        "noise_floor_delta_db": {
            "available_count": len(noise_floor_deltas),
            "p50": percentile(noise_floor_deltas, 50) if noise_floor_deltas else None,
            "mean": float(np.mean(noise_floor_deltas)) if noise_floor_deltas else None,
        },
        "noise_floor_delta_by_bucket": {
            bucket: {
                "count": len(values),
                "improved_count": sum(value < 0.0 for value in values),
                "p50": percentile(values, 50),
                "mean": float(np.mean(values)),
                "minimum": min(values),
                "maximum": max(values),
            }
            for bucket, values in sorted(bucket_deltas.items())
        },
        "results": results,
    }
    output = root / "automatic_validation.json"
    atomic_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
