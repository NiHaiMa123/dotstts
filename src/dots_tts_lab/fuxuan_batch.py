from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from dots_tts_lab.postprocess import (
    DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH,
    EdgeTrimConfig,
    VoicePolishConfig,
    apply_fuxuan_voice_polish,
    load_edge_trim_config,
    load_voice_polish_config,
    safe_edge_trim,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "inputs/fuxuan_text"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/fuxuan_audio"
DEFAULT_BASE_MODEL = ROOT / "pretrained_models/dots.tts-soar"
DEFAULT_ADAPTER = (
    ROOT / "data/work/slice12/soar_lora_v1/checkpoint-00000400/model"
)
DEFAULT_PROMPT_AUDIO = ROOT / (
    "data/work/standardized/6b/"
    "6bdb59898ba90334918498b7a86cdbf24e8c30a06caf4b90ab17581d069610e9.wav"
)
DEFAULT_PROMPT_TEXT = (
    "各位启程前，请先好好休养歇息一番。有什么想去的地方，尽可以逛逛。"
    "我要暂代云骑事务，无法奉陪了。"
)
DEFAULT_EDGE_TRIM_CONFIG = (
    ROOT / "configs/lab/postprocess/generated_audio_edge_trim_v1.yaml"
)
DEFAULT_NUM_STEPS = 16

_SENTENCE_PATTERN = re.compile(r".+?(?:[。！？!?；;]+[”’」』）》】]*|$)")
_SOFT_BREAKS = "，、,：:"

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


class BatchCancelled(RuntimeError):
    """Raised when a batch is cancelled between synthesis operations."""


def _raise_if_cancelled(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise BatchCancelled("Batch generation was cancelled")


def _report_progress(
    progress_callback: ProgressCallback | None,
    **event: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        split_at = max((window.rfind(mark) + 1 for mark in _SOFT_BREAKS), default=0)
        if split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_text(text: str, *, max_chars: int = 120) -> list[str]:
    """Split text deterministically while keeping sentence punctuation."""
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    segments: list[str] = []
    for paragraph in normalized.split("\n"):
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact:
            continue
        matches = [match.group(0).strip() for match in _SENTENCE_PATTERN.finditer(compact)]
        for match in matches:
            if match:
                segments.extend(_split_long_segment(match, max_chars))
    return segments


def discover_text_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


def output_path_for(text_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / text_path.relative_to(input_dir).with_suffix(".wav")


def _as_mono_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    audio = np.asarray(value, dtype=np.float64).squeeze()
    if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise RuntimeError("Runtime returned empty, non-mono, or non-finite audio")
    return audio


def render_segments(
    runtime: Any,
    segments: list[str],
    *,
    edge_trim_config: EdgeTrimConfig,
    base_seed: int,
    pause_ms: int,
    num_steps: int = 10,
    prompt_audio_path: str | Path = DEFAULT_PROMPT_AUDIO,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    language: str = "chinese",
    template_name: str = "tts",
    normalize_text: bool = False,
    speaker_scale: float = 1.5,
    ode_method: str = "euler",
    guidance_scale: float = 1.2,
    progress_callback: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> tuple[np.ndarray, int]:
    if not segments:
        raise ValueError("Text file has no non-whitespace content")
    if pause_ms < 0 or pause_ms > 5_000:
        raise ValueError("pause_ms must be between 0 and 5000")
    if num_steps < 1 or num_steps > 100:
        raise ValueError("num_steps must be between 1 and 100")
    if not prompt_text.strip():
        raise ValueError("prompt_text cannot be empty")
    resolved_prompt_audio = Path(prompt_audio_path).expanduser().resolve()
    if not resolved_prompt_audio.is_file():
        raise FileNotFoundError(f"Prompt audio is missing: {resolved_prompt_audio}")

    from dots_tts.utils.util import seed_everything

    rendered: list[np.ndarray] = []
    sample_rate: int | None = None
    for index, segment in enumerate(segments):
        _raise_if_cancelled(cancelled)
        _report_progress(
            progress_callback,
            stage="generating_segment",
            segment_index=index + 1,
            segment_count=len(segments),
            text=segment,
        )
        seed_everything(base_seed + index)
        result = runtime.generate(
            text=segment,
            prompt_audio_path=str(resolved_prompt_audio),
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            speaker_scale=speaker_scale,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            normalize_text=normalize_text,
        )
        _raise_if_cancelled(cancelled)
        current_rate = int(result["sample_rate"])
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise RuntimeError(
                f"Runtime sample rate changed within one file: {sample_rate} -> {current_rate}"
            )
        audio, _ = safe_edge_trim(
            _as_mono_numpy(result["audio"]),
            current_rate,
            edge_trim_config,
        )
        rendered.append(audio)
        _report_progress(
            progress_callback,
            stage="segment_completed",
            segment_index=index + 1,
            segment_count=len(segments),
        )

    assert sample_rate is not None
    pause = np.zeros(round(sample_rate * pause_ms / 1000.0), dtype=np.float64)
    parts: list[np.ndarray] = []
    for index, audio in enumerate(rendered):
        if index and pause.size:
            parts.append(pause)
        parts.append(audio)
    return np.concatenate(parts), sample_rate


def write_final_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial.wav")
    try:
        sf.write(partial, audio, sample_rate, format="WAV", subtype="PCM_24")
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def run_batch(
    runtime: Any,
    *,
    input_dir: Path,
    output_dir: Path,
    edge_trim_config: EdgeTrimConfig,
    base_seed: int = 20260908,
    pause_ms: int = 250,
    max_chars: int = 120,
    num_steps: int = DEFAULT_NUM_STEPS,
    prompt_audio_path: str | Path = DEFAULT_PROMPT_AUDIO,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    language: str = "chinese",
    template_name: str = "tts",
    normalize_text: bool = False,
    speaker_scale: float = 1.5,
    ode_method: str = "euler",
    guidance_scale: float = 1.2,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
    voice_polish_config: VoicePolishConfig | None = None,
) -> dict[str, Any]:
    text_files = discover_text_files(input_dir)
    resolved_voice_polish = voice_polish_config or load_voice_polish_config(
        DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH
    )
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_count": len(text_files),
        "generated": [],
        "skipped": [],
        "errors": [],
        "final_processing": resolved_voice_polish.model_dump(mode="json"),
        "final_processing_sha256": resolved_voice_polish.canonical_sha256(),
        "generation": {
            "base_seed": base_seed,
            "num_steps": num_steps,
            "prompt_audio_path": str(Path(prompt_audio_path).expanduser().resolve()),
            "prompt_text": prompt_text,
            "language": language,
            "template_name": template_name,
            "normalize_text": normalize_text,
            "speaker_scale": speaker_scale,
            "ode_method": ode_method,
            "guidance_scale": guidance_scale,
        },
    }
    _report_progress(
        progress_callback,
        stage="discovered",
        file_count=len(text_files),
    )
    for file_index, text_path in enumerate(text_files, start=1):
        _raise_if_cancelled(cancelled)
        relative = text_path.relative_to(input_dir)
        output_path = output_path_for(text_path, input_dir, output_dir)
        if output_path.is_file() and not force:
            summary["skipped"].append(relative.as_posix())
            _report_progress(
                progress_callback,
                stage="file_skipped",
                file_index=file_index,
                file_count=len(text_files),
                input=relative.as_posix(),
            )
            continue
        try:
            text = text_path.read_text(encoding="utf-8-sig")
            segments = split_text(text, max_chars=max_chars)
            _report_progress(
                progress_callback,
                stage="file_started",
                file_index=file_index,
                file_count=len(text_files),
                input=relative.as_posix(),
                segment_count=len(segments),
            )

            def report_segment(event: dict[str, Any]) -> None:
                _report_progress(
                    progress_callback,
                    **event,
                    file_index=file_index,
                    file_count=len(text_files),
                    input=relative.as_posix(),
                )

            audio, sample_rate = render_segments(
                runtime,
                segments,
                edge_trim_config=edge_trim_config,
                base_seed=base_seed,
                pause_ms=pause_ms,
                num_steps=num_steps,
                prompt_audio_path=prompt_audio_path,
                prompt_text=prompt_text,
                language=language,
                template_name=template_name,
                normalize_text=normalize_text,
                speaker_scale=speaker_scale,
                ode_method=ode_method,
                guidance_scale=guidance_scale,
                progress_callback=report_segment,
                cancelled=cancelled,
            )
            _raise_if_cancelled(cancelled)
            _report_progress(
                progress_callback,
                stage="finalizing_file",
                file_index=file_index,
                file_count=len(text_files),
                input=relative.as_posix(),
            )
            audio, processing = apply_fuxuan_voice_polish(
                audio,
                sample_rate,
                resolved_voice_polish,
            )
            write_final_audio(output_path, audio, sample_rate)
            summary["generated"].append(
                {
                    "input": relative.as_posix(),
                    "output": output_path.relative_to(output_dir).as_posix(),
                    "segments": len(segments),
                    "duration_seconds": audio.size / sample_rate,
                    "processing": processing,
                }
            )
            _report_progress(
                progress_callback,
                stage="file_completed",
                file_index=file_index,
                file_count=len(text_files),
                input=relative.as_posix(),
                output=output_path.relative_to(output_dir).as_posix(),
            )
        except BatchCancelled:
            raise
        except Exception as error:
            summary["errors"].append(
                {"input": relative.as_posix(), "error": f"{type(error).__name__}: {error}"}
            )
    return summary


def load_runtime(
    *,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    base_revision: str | None = None,
    adapter: str | Path | None = DEFAULT_ADAPTER,
    precision: str = "bfloat16",
    optimize: bool = False,
    max_generate_length: int = 128,
    max_sequence_length: int = 1024,
    vocoder_merge_steps: int = 4,
    warmup_on_optimize: bool = False,
    merge_lora: bool = True,
) -> Any:
    from dots_tts.runtime import DotsTtsRuntime

    resolved_base = Path(base_model).expanduser().resolve()
    required_paths: list[Path] = [resolved_base]
    resolved_adapter: Path | None = None
    if adapter is not None:
        resolved_adapter = Path(adapter).expanduser().resolve()
        required_paths.append(resolved_adapter)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Fuxuan assets are missing: " + ", ".join(missing))
    if resolved_adapter is None:
        return DotsTtsRuntime.from_pretrained(
            str(resolved_base),
            revision=base_revision,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
            max_sequence_length=max_sequence_length,
            vocoder_merge_steps=vocoder_merge_steps,
            warmup_on_optimize=warmup_on_optimize,
        )
    return DotsTtsRuntime.from_pretrained_with_trainable_delta(
        str(resolved_base),
        resolved_adapter,
        revision=base_revision,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
        max_sequence_length=max_sequence_length,
        vocoder_merge_steps=vocoder_merge_steps,
        warmup_on_optimize=warmup_on_optimize,
        merge_lora=merge_lora,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one final Fuxuan WAV for every UTF-8 TXT file."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--pause-ms", type=int, default=250)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    text_files = discover_text_files(input_dir)
    pending = [
        path
        for path in text_files
        if args.force or not output_path_for(path, input_dir, output_dir).is_file()
    ]
    if not text_files:
        print(json.dumps({"status": "no_input", "input_dir": str(input_dir)}, ensure_ascii=False))
        return 0
    if not pending:
        print(
            json.dumps(
                {
                    "status": "nothing_to_do",
                    "input_count": len(text_files),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0

    runtime = load_runtime()
    summary = run_batch(
        runtime,
        input_dir=input_dir,
        output_dir=output_dir,
        edge_trim_config=load_edge_trim_config(DEFAULT_EDGE_TRIM_CONFIG),
        base_seed=args.seed,
        pause_ms=args.pause_ms,
        max_chars=args.max_chars,
        num_steps=args.num_steps,
        force=args.force,
    )
    summary["status"] = "failed" if summary["errors"] else "succeeded"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
