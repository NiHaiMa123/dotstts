from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def strip_nonlinguistic(text: str) -> str:
    """Drop rich-transcription markers so a hypothesis carries transcript text only.

    SenseVoice turns its `<|HAPPY|>`-style tags into emoji, which are neither
    spoken content nor punctuation. Leaving them in the hypothesis charges every
    tag as an insertion error and makes CER incomparable across backends.
    """
    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "So"
    ).strip()


def managed(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path


def canonical_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_native_libraries(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Put backend-declared native library directories on PATH before import.

    CTranslate2 resolves cuBLAS through the default Windows search order rather
    than through the torch loader, so the directory has to be on PATH before the
    backend module loads. Leaving it undeclared meant faster-whisper only ran
    while a torch-enabled venv happened to be active, which is not
    reproducible. Declaring the dependency also lets a missing library fail with
    an actionable message instead of a bare `cublas64_12.dll is not found`.
    """
    directories = [str(entry) for entry in config.get("native_library_dirs") or []]
    required = [str(entry) for entry in config.get("required_native_libraries") or []]
    resolved = []
    for entry in directories:
        directory = (repo_root / entry).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Declared native library directory is missing: {directory}"
            )
        resolved.append(directory)
    for library in required:
        if not any((directory / library).is_file() for directory in resolved):
            searched = ", ".join(str(directory) for directory in resolved) or "<none>"
            raise FileNotFoundError(
                f"Required native library {library} not found in: {searched}"
            )
    if resolved:
        os.environ["PATH"] = os.pathsep.join(
            [str(directory) for directory in resolved] + [os.environ.get("PATH", "")]
        )
    return {
        "native_library_dirs": [str(directory) for directory in resolved],
        "required_native_libraries": required,
    }


def run_faster_whisper(
    model_path: Path, audio_paths: list[Path], parameters: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        str(model_path),
        device="cuda",
        compute_type=parameters["compute_type"],
    )
    load_seconds = time.perf_counter() - started
    outputs = []
    for path in audio_paths:
        item_started = time.perf_counter()
        segments, info = model.transcribe(
            str(path),
            language=parameters["language"],
            beam_size=int(parameters["beam_size"]),
            vad_filter=bool(parameters["vad_filter"]),
            condition_on_previous_text=bool(parameters["condition_on_previous_text"]),
            word_timestamps=bool(parameters["word_timestamps"]),
        )
        materialized = list(segments)
        timestamp_segments = []
        if bool(parameters["word_timestamps"]):
            for segment in materialized:
                timestamp_segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": str(segment.text),
                        "avg_logprob": float(segment.avg_logprob),
                        "no_speech_prob": float(segment.no_speech_prob),
                        "words": [
                            {
                                "start": float(word.start),
                                "end": float(word.end),
                                "text": str(word.word),
                                "probability": float(word.probability),
                            }
                            for word in (segment.words or [])
                        ],
                    }
                )
        outputs.append(
            {
                "hypothesis": "".join(segment.text for segment in materialized).strip(),
                "runtime_seconds": time.perf_counter() - item_started,
                "detected_language": info.language,
                "metadata": {
                    "language_probability": info.language_probability,
                    "segment_count": len(materialized),
                    **(
                        {"timestamp_segments": timestamp_segments}
                        if bool(parameters["word_timestamps"])
                        else {}
                    ),
                },
            }
        )
    return outputs, {
        "model_load_seconds": load_seconds,
        # CTranslate2 allocates outside the torch caching allocator, so a torch
        # peak reading would be meaningless rather than merely unavailable.
        "peak_torch_cuda_bytes": None,
        "peak_memory_note": "ctranslate2 backend: torch allocator not involved",
    }


def run_sensevoice(
    model_path: Path, audio_paths: list[Path], parameters: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    construction: dict[str, Any] = {
        "model": str(model_path),
        "device": "cuda:0",
        "disable_update": True,
    }
    # An explicit null keeps VAD off, which is what 3-10 s sentence clips need.
    if parameters["vad_model"] is not None:
        construction["vad_model"] = str(parameters["vad_model"])
    model = AutoModel(**construction)
    load_seconds = time.perf_counter() - started
    outputs = []
    for path in audio_paths:
        item_started = time.perf_counter()
        result = model.generate(
            input=str(path),
            cache={},
            language=parameters["language"],
            use_itn=bool(parameters["use_itn"]),
            batch_size=int(parameters["batch_size"]),
        )[0]
        raw = str(result["text"])
        rich = rich_transcription_postprocess(raw)
        outputs.append(
            {
                "hypothesis": strip_nonlinguistic(rich),
                "runtime_seconds": time.perf_counter() - item_started,
                "detected_language": parameters["language"],
                "metadata": {"backend_text_raw": raw, "backend_text_rich": rich},
            }
        )
    return outputs, {
        "model_load_seconds": load_seconds,
        "peak_torch_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }


def run_qwen3_asr(
    model_path: Path, audio_paths: list[Path], parameters: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if parameters["dtype"] == "bfloat16" else torch.float16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=parameters.get("device", "cuda:0"),
        attn_implementation=parameters.get("attention", "sdpa"),
        max_inference_batch_size=int(parameters["max_inference_batch_size"]),
        max_new_tokens=int(parameters["max_new_tokens"]),
    )
    load_seconds = time.perf_counter() - started
    outputs = []
    for path in audio_paths:
        item_started = time.perf_counter()
        result = model.transcribe(
            audio=str(path),
            language=parameters["language"],
            return_time_stamps=bool(parameters["timestamps"]),
        )[0]
        outputs.append(
            {
                "hypothesis": result.text,
                "runtime_seconds": time.perf_counter() - item_started,
                "detected_language": result.language,
                "metadata": {},
            }
        )
    return outputs, {
        "model_load_seconds": load_seconds,
        "peak_torch_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }


RUNNERS = {
    "faster_whisper": run_faster_whisper,
    "sensevoice": run_sensevoice,
    "qwen3_asr": run_qwen3_asr,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backend_id = str(config["backend_id"])
    if backend_id not in RUNNERS:
        raise ValueError(f"Unsupported backend: {backend_id}")
    manifest_path = args.manifest.resolve()
    manifest_sha256 = file_sha256(manifest_path)
    manifest = load_manifest(manifest_path)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    audio_root = args.audio_root.resolve()
    audio_paths = [
        managed(audio_root, str(item["derived_relative_path"])) for item in manifest
    ]
    for item, audio_path in zip(manifest, audio_paths):
        if not audio_path.is_file():
            raise FileNotFoundError(f"Benchmark audio is missing: {audio_path}")
        if file_sha256(audio_path) != item["derived_sha256"]:
            raise RuntimeError(
                f"Benchmark audio hash mismatch for {item['asset_sha256']}: "
                f"{audio_path}"
            )
    revision = str(config["model_revision"])
    model_path = (args.model_root / backend_id / revision).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model snapshot not found: {model_path}")
    parameters = dict(config["parameters"])
    if backend_id == "qwen3_asr":
        parameters["device"] = config["device"]
    native_libraries = prepare_native_libraries(
        config, Path(__file__).resolve().parents[1]
    )
    started_at = now()
    run_started = time.perf_counter()
    outputs, runtime_metadata = RUNNERS[backend_id](model_path, audio_paths, parameters)
    if len(outputs) != len(manifest):
        raise RuntimeError("Backend output count does not match manifest")
    if file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Benchmark manifest changed while inference was running")
    results = []
    for item, output in zip(manifest, outputs):
        results.append(
            {
                "asset_sha256": item["asset_sha256"],
                "hypothesis": output["hypothesis"],
                "runtime_seconds": output["runtime_seconds"],
                "detected_language": output["detected_language"],
                "metadata": output["metadata"],
                "status": "ok",
            }
        )
    package_name = str(config["package"]).split("==", 1)[0]
    payload = {
        "schema_version": 1,
        "backend_id": backend_id,
        "backend_version": importlib.metadata.version(package_name),
        "model_id": config["model_id"],
        "model_revision": revision,
        "model_license": config["model_license"],
        "inference_config": parameters,
        "inference_config_sha256": canonical_sha256(parameters),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "started_at": started_at,
        "finished_at": now(),
        "wall_runtime_seconds": time.perf_counter() - run_started,
        "runtime_metadata": {
            **runtime_metadata,
            "verified_audio_count": len(audio_paths),
        },
        "capabilities": config["capabilities"],
        "environment": {
            "python": os.sys.version,
            "executable": os.sys.executable,
            **native_libraries,
        },
        "results": results,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "results"}, ensure_ascii=False, indent=2))
    print(f"result_count={len(results)} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
