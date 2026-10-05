from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from run_asr_backend import prepare_native_libraries


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def compact(text: str) -> str:
    return "".join(character for character in text.strip() if not character.isspace())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ASR on raw and enhanced AB clips.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("data/reports/long_form/denoise_ab_v1/manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lab/asr/backends/faster_whisper_long_form_v1.yaml"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepare_native_libraries(config, repo)
    from faster_whisper import WhisperModel

    model_path = (
        repo / "data/work/models/asr/faster_whisper" / str(config["model_revision"])
    ).resolve()
    model = WhisperModel(str(model_path), device="cuda", compute_type=config["parameters"]["compute_type"])
    results = []
    exact_count = 0
    total_distance = 0
    total_chars = 0
    for case in manifest["cases"]:
        hypotheses = {}
        for kind in ("raw", "enhanced"):
            path = root / case[kind]["relative_path"]
            segments, _ = model.transcribe(
                str(path),
                language=config["parameters"]["language"],
                beam_size=int(config["parameters"]["beam_size"]),
                vad_filter=bool(config["parameters"]["vad_filter"]),
                condition_on_previous_text=bool(config["parameters"]["condition_on_previous_text"]),
                word_timestamps=False,
            )
            hypotheses[kind] = "".join(segment.text for segment in segments).strip()
        raw = compact(hypotheses["raw"])
        enhanced = compact(hypotheses["enhanced"])
        edit_distance = distance(raw, enhanced)
        exact = raw == enhanced
        exact_count += int(exact)
        total_distance += edit_distance
        total_chars += max(len(raw), len(enhanced), 1)
        results.append(
            {
                "case_id": case["case_id"],
                "raw_hypothesis": hypotheses["raw"],
                "enhanced_hypothesis": hypotheses["enhanced"],
                "exact_match": exact,
                "character_edit_distance": edit_distance,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "backend_id": config["backend_id"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "case_count": len(results),
        "exact_match_count": exact_count,
        "aggregate_character_change_rate": total_distance / total_chars,
        "results": results,
    }
    output = root / "asr_validation.json"
    atomic_json(output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()

