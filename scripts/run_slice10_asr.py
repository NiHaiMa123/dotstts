from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def configured_asr_python(config_path: Path) -> Path | None:
    """Resolve the optional isolated ASR interpreter declared by the benchmark."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured = config.get("metrics", {}).get("asr_python")
    if not configured:
        return None
    return (config_path.parents[3] / str(configured)).resolve()


def run(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    metrics = config["metrics"]
    generation_path = (repo_root / outputs["generation_manifest_path"]).resolve()
    asr_path = (repo_root / outputs["asr_output_path"]).resolve()
    rows = load_jsonl(generation_path)
    if not rows:
        raise RuntimeError("Slice10 generation manifest is empty")
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("Slice10 ASR requires every generation row to be successful")

    existing = {} if force or not asr_path.is_file() else {
        row["job_id"]: row for row in load_jsonl(asr_path)
    }
    model_root = (repo_root / metrics["asr_model_root"] / metrics["asr_backend_id"] / metrics["asr_model_revision"]).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"ASR model snapshot is missing: {model_root}")

    # faster-whisper/CTranslate2 resolves cuBLAS through Windows' DLL search
    # order.  The isolated ASR environment intentionally reuses the torch
    # runtime shipped with the project instead of carrying another 400 MB DLL.
    torch_lib = (repo_root / ".venv/Lib/site-packages/torch/lib").resolve()
    cublas = torch_lib / "cublas64_12.dll"
    if not cublas.is_file():
        raise FileNotFoundError(f"Pinned cuBLAS runtime is missing: {cublas}")
    os.environ["PATH"] = os.pathsep.join([str(torch_lib), os.environ.get("PATH", "")])

    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(str(model_root), device="cuda", compute_type="float16")
    load_seconds = time.perf_counter() - started
    result_rows = []
    for index, row in enumerate(rows, start=1):
        prior = existing.get(row["job_id"])
        output_path = (repo_root / row["output_path"]).resolve()
        output_sha = row.get("output_sha256")
        if not output_path.is_file() or sha256_file(output_path) != output_sha:
            raise RuntimeError(f"Generated audio hash mismatch: {output_path}")
        if prior and prior.get("output_sha256") == output_sha and prior.get("status") == "ok" and not force:
            result_rows.append(prior)
            continue
        item = {
            "schema_version": 1,
            "job_id": row["job_id"],
            "ordinal": row["ordinal"],
            "asset_sha256": row["asset_sha256"],
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "seed": row["seed"],
            "output_path": row["output_path"],
            "output_sha256": output_sha,
        }
        try:
            item_started = time.perf_counter()
            segments, info = model.transcribe(
                str(output_path),
                language="zh",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            materialized = list(segments)
            item.update(
                {
                    "status": "ok",
                    "hypothesis": "".join(segment.text for segment in materialized).strip(),
                    "asr_seconds": time.perf_counter() - item_started,
                    "detected_language": info.language,
                    "language_probability": info.language_probability,
                    "segment_count": len(materialized),
                }
            )
        except Exception as error:
            item.update({"status": "asr_error", "error_type": type(error).__name__, "error_message": str(error)})
        result_rows.append(item)
        asr_path.parent.mkdir(parents=True, exist_ok=True)
        asr_path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in result_rows), encoding="utf-8", newline="\n")
        print(f"[{index}/{len(rows)}] {row['job_id']} {item['status']}")

    result_rows.sort(key=lambda value: (int(value["ordinal"]), value["sentence_id"], int(value["seed"])))
    asr_path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in result_rows), encoding="utf-8", newline="\n")
    summary = {
        "schema_version": 1,
        "asr_backend_id": metrics["asr_backend_id"],
        "asr_model_id": metrics["asr_model_id"],
        "asr_model_revision": metrics["asr_model_revision"],
        "model_load_seconds": load_seconds,
        "expected_count": len(rows),
        "result_count": len(result_rows),
        "success_count": sum(value.get("status") == "ok" for value in result_rows),
        "error_count": sum(value.get("status") != "ok" for value in result_rows),
        "asr_output_path": str(asr_path),
        "status": "succeeded" if len(result_rows) == len(rows) and all(value.get("status") == "ok" for value in result_rows) else "incomplete",
    }
    asr_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe Slice10 generated audio with the pinned faster-whisper model.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asr_python = configured_asr_python(args.config)
    if asr_python is not None and Path(sys.executable).resolve() != asr_python:
        if not asr_python.is_file():
            raise FileNotFoundError(f"Configured Slice10 ASR interpreter is missing: {asr_python}")
        command = [str(asr_python), str(Path(__file__).resolve()), "--config", str(args.config)]
        if args.force:
            command.append("--force")
        return subprocess.call(command)
    print(json.dumps(run(args.config, force=args.force), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
