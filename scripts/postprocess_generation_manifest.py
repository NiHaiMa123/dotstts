from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from dots_tts_lab.postprocess import load_edge_trim_config, safe_edge_trim

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_pcm24(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(str(temporary), audio, sample_rate, subtype="PCM_24", format="WAV")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(
    source_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    config_path: Path,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    output_manifest = output_manifest.resolve()
    config = load_edge_trim_config(config_path.resolve())
    rows = load_jsonl(source_manifest)
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("Postprocessing requires a non-empty successful generation manifest")
    output_rows: list[dict[str, Any]] = []
    trimmed_count = 0
    for row in rows:
        source = (ROOT / row["output_path"]).resolve()
        if not source.is_file() or sha256_file(source) != row["output_sha256"]:
            raise RuntimeError(f"Source audio hash mismatch: {source}")
        audio, sample_rate = sf.read(str(source), dtype="float64", always_2d=False)
        processed, details = safe_edge_trim(audio, int(sample_rate), config)
        target = output_root / f"{row['job_id']}.wav"
        atomic_pcm24(target, processed, int(sample_rate))
        output_hash = sha256_file(target)
        source_hash_after = sha256_file(source)
        if source_hash_after != row["output_sha256"]:
            raise RuntimeError(f"Source audio changed during postprocessing: {source}")
        trimmed = bool(
            details["leading_samples_removed"] or details["trailing_samples_removed"]
        )
        trimmed_count += int(trimmed)
        sidecar = {
            "schema_version": 1,
            "operation": "safe_edge_trim",
            "config_sha256": config.canonical_sha256(),
            "source_path": source.relative_to(ROOT).as_posix(),
            "source_sha256": row["output_sha256"],
            "output_path": target.relative_to(ROOT).as_posix(),
            "output_sha256": output_hash,
            "sample_rate": int(sample_rate),
            "source_sample_count": int(len(audio)),
            "output_sample_count": int(len(processed)),
            **details,
        }
        write_json(target.with_suffix(".json"), sidecar)
        generation_seconds = row.get("generation_seconds")
        duration = len(processed) / int(sample_rate)
        output_rows.append(
            {
                **row,
                "source_output_path": row["output_path"],
                "source_output_sha256": row["output_sha256"],
                "output_path": target.relative_to(ROOT).as_posix(),
                "output_sha256": output_hash,
                "sample_count": int(len(processed)),
                "duration_seconds": duration,
                "rtf": (
                    float(generation_seconds) / max(duration, 1e-9)
                    if generation_seconds is not None
                    else row.get("rtf")
                ),
                "postprocess": sidecar,
            }
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "source_manifest": source_manifest.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_manifest": output_manifest.relative_to(ROOT).as_posix(),
        "output_manifest_sha256": sha256_file(output_manifest),
        "config_sha256": config.canonical_sha256(),
        "item_count": len(output_rows),
        "trimmed_count": trimmed_count,
        "unchanged_count": len(output_rows) - trimmed_count,
    }
    write_json(output_manifest.with_suffix(".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lab/postprocess/generated_audio_edge_trim_v1.yaml"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.source_manifest, args.output_root, args.output_manifest, args.config),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
