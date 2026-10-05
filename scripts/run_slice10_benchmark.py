from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSONL row in {path}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)


def relative_output_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_jobs(config: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for candidate in manifest_rows:
        for sentence in config["test_sentences"]:
            for seed in config["seeds"]:
                job_id = f"a{int(candidate['ordinal']):03d}-{candidate['asset_sha256'][:12]}__{sentence['sentence_id']}__s{int(seed)}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "ordinal": int(candidate["ordinal"]),
                        "asset_sha256": candidate["asset_sha256"],
                        "audio_sha256": candidate["audio_sha256"],
                        "audio_relative_path": candidate["audio_relative_path"],
                        "pool_ids": candidate["pool_ids"],
                        "speaker_id": candidate["speaker_id"],
                        "prompt_text": candidate["text_exact"],
                        "prompt_text_sha256": candidate["prompt_text_sha256"],
                        "sentence_id": sentence["sentence_id"],
                        "text": sentence["text"],
                        "purpose": sentence["purpose"],
                        "seed": int(seed),
                    }
                )
    return jobs


def run(config_path: Path, *, force: bool = False) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Slice10 config must be a schema_version=1 mapping")
    outputs = config["outputs"]
    manifest_path = (repo_root / outputs["benchmark_manifest_path"]).resolve()
    if sha256_file(manifest_path) != str(outputs["benchmark_manifest_sha256"]).lower():
        raise RuntimeError("Benchmark manifest SHA-256 drift; rebuild and repin config")
    manifest_rows = load_jsonl(manifest_path)
    if not manifest_rows:
        raise RuntimeError("Benchmark manifest is empty")
    jobs = build_jobs(config, manifest_rows)
    generation_root = (repo_root / outputs["generation_root"]).resolve()
    standardized_root = (repo_root / config["selection"]["standardized_audio_root"]).resolve()
    record_path = (repo_root / outputs["generation_manifest_path"]).resolve()
    config_hash = canonical_sha256(config)
    accepted_config_hashes = {config_hash}
    original_config_hash = config.get("provenance", {}).get("generation_config_sha256")
    if original_config_hash:
        accepted_config_hashes.add(str(original_config_hash))
    model = config["model"]
    model_path = (repo_root / model["local_path"]).resolve()
    config_file = (repo_root / model["model_config_path"]).resolve()
    if not model_path.is_dir() or not config_file.is_file():
        raise FileNotFoundError(f"Pinned TTS model is unavailable: {model_path}")
    if sha256_file(config_file) != str(model["model_config_sha256"]).lower():
        raise RuntimeError("Pinned TTS model config SHA-256 drift")

    adapter = model.get("adapter")
    adapter_dir = None
    if adapter is not None:
        if adapter.get("format") != "dots_tts_trainable_delta":
            raise ValueError(f"Unsupported model adapter format: {adapter.get('format')!r}")
        adapter_dir = (repo_root / adapter["path"]).resolve()
        expected_adapter_files = {
            "trainable_model.safetensors": adapter["weights_sha256"],
            "trainable_model.json": adapter["metadata_sha256"],
        }
        for filename, expected_hash in expected_adapter_files.items():
            artifact_path = adapter_dir / filename
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Pinned trainable delta is incomplete: {artifact_path}")
            if sha256_file(artifact_path) != str(expected_hash).lower():
                raise RuntimeError(f"Pinned trainable delta SHA-256 drift: {artifact_path}")

    existing_rows = [] if force else load_jsonl(record_path)
    by_job: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if row.get("config_sha256") not in (None, *accepted_config_hashes):
            raise RuntimeError("Existing Slice10 generation records use a different config")
        if row.get("benchmark_manifest_sha256") not in (None, outputs["benchmark_manifest_sha256"]):
            raise RuntimeError("Existing Slice10 generation records use a different manifest")
        by_job[str(row["job_id"])] = row

    runtime_kwargs = {
        "precision": str(model["precision"]),
        "optimize": bool(model.get("optimize", False)),
        "max_generate_length": int(model["max_generate_length"]),
        "max_sequence_length": int(model["max_sequence_length"]),
        "vocoder_merge_steps": int(model.get("vocoder_merge_steps", 4)),
        "warmup_on_optimize": bool(model.get("warmup_on_optimize", True)),
    }
    if adapter_dir is None:
        runtime = DotsTtsRuntime.from_pretrained(str(model_path), **runtime_kwargs)
    else:
        runtime = DotsTtsRuntime.from_pretrained_with_trainable_delta(
            str(model_path),
            adapter_dir,
            merge_lora=bool(adapter.get("merge_for_inference", False)),
            **runtime_kwargs,
        )
    sampling = model.get("sampling", {})
    sampling_options = {
        "ode_method": sampling.get("ode_method"),
        "num_steps": sampling.get("num_steps"),
        "guidance_scale": sampling.get("guidance_scale"),
    }
    for index, job in enumerate(jobs, start=1):
        prior = by_job.get(job["job_id"])
        if prior and prior.get("status") == "ok" and not force:
            output_path = (repo_root / str(prior["output_path"])).resolve()
            if output_path.is_file() and sha256_file(output_path) == prior.get("output_sha256"):
                continue

        output_path = generation_root / f"a{job['ordinal']:03d}-{job['asset_sha256'][:12]}" / job["sentence_id"] / f"seed-{job['seed']}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_audio = (standardized_root / job["audio_relative_path"]).resolve()
        row = {
            **job,
            "schema_version": 1,
            "config_sha256": config_hash,
            "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
            "model_adapter": adapter,
            "runtime_options": runtime_kwargs,
            "sampling_options": sampling_options,
            "precision": model["precision"],
            "output_path": relative_output_path(repo_root, output_path),
            "started_at": now(),
        }
        try:
            if not prompt_audio.is_file() or sha256_file(prompt_audio) != job["audio_sha256"]:
                raise RuntimeError("prompt audio is missing or SHA-256 does not match benchmark manifest")
            seed_everything(job["seed"])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = runtime.generate(
                text=job["text"],
                prompt_audio_path=str(prompt_audio),
                prompt_text=job["prompt_text"],
                language=str(model["language"]),
                speaker_scale=float(model["speaker_scale"]),
                normalize_text=False,
                **sampling_options,
            )
            elapsed = time.perf_counter() - started
            audio = result["audio"].detach().float().cpu().squeeze().numpy()
            sample_rate = int(result["sample_rate"])
            if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
                raise RuntimeError("runtime returned empty or non-finite audio")
            sf.write(str(output_path), audio, sample_rate, subtype="PCM_24")
            row.update(
                {
                    "status": "ok",
                    "output_sha256": sha256_file(output_path),
                    "sample_rate": sample_rate,
                    "sample_count": int(audio.size),
                    "duration_seconds": float(audio.size / sample_rate),
                    "generation_seconds": float(elapsed),
                    "rtf": float(elapsed / max(audio.size / sample_rate, 1e-9)),
                    "request_id": result.get("fid"),
                    "peak_torch_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                }
            )
        except Exception as error:  # keep the full matrix auditable and resumable
            row.update(
                {
                    "status": "generation_error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "finished_at": now(),
                }
            )
        else:
            row["finished_at"] = now()
        by_job[job["job_id"]] = row
        write_jsonl(record_path, [by_job[j["job_id"]] for j in jobs if j["job_id"] in by_job])
        print(f"[{index}/{len(jobs)}] {job['job_id']} {row['status']}")

    ordered = [by_job[job["job_id"]] for job in jobs if job["job_id"] in by_job]
    write_jsonl(record_path, ordered)
    summary = {
        "schema_version": 1,
        "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
        "config_sha256": config_hash,
        "expected_count": len(jobs),
        "record_count": len(ordered),
        "success_count": sum(row.get("status") == "ok" for row in ordered),
        "failure_count": sum(row.get("status") != "ok" for row in ordered),
        "generation_manifest_path": str(record_path),
        "status": "succeeded" if len(ordered) == len(jobs) else "incomplete",
    }
    summary_path = record_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic Slice10 TTS reference benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    args = parser.parse_args()
    print(json.dumps(run(args.config, force=args.force), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
