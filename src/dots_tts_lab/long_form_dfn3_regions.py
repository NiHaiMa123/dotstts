from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_audio import file_sha256, probe_long_audio
from dots_tts_lab.long_form_dfn3 import Dfn3ProductionConfig
from dots_tts_lab.long_form_paths import validate_output_path

ROOT = Path(__file__).resolve().parents[2]
DFN3_REGION_IMPLEMENTATION_VERSION = 3


@dataclass(frozen=True)
class EnhanceWindow:
    """A bounded enhancement window in source-frame coordinates.

    The model sees [context_start, context_end); after delay-compensated
    enhancement the context margins are discarded and only
    [core_start, core_end) is kept, so the emitted stream is exactly the
    concatenation of consecutive cores.
    """

    window_index: int
    core_start_frame: int
    core_end_frame: int
    context_start_frame: int
    context_end_frame: int

    @property
    def core_frames(self) -> int:
        return self.core_end_frame - self.core_start_frame

    @property
    def context_frames(self) -> int:
        return self.context_end_frame - self.context_start_frame

    def core_output_bounds(self) -> tuple[int, int]:
        """Offsets inside the delay-compensated context-length output."""
        offset = self.core_start_frame - self.context_start_frame
        return offset, offset + self.core_frames


def plan_windows(
    region_start_frame: int,
    region_end_frame: int,
    *,
    core_max_frames: int,
    context_max_frames: int,
    total_source_frames: int,
) -> list[EnhanceWindow]:
    """Split a region into non-overlapping cores each wrapped by real context.

    Cores tile [region_start, region_end) contiguously so the emitted audio is
    a gap-free concatenation. Context is clamped at file boundaries — never
    zero-padded — and a short tail core absorbs the remainder rather than
    overflowing core_max.
    """
    if not 0 <= region_start_frame < region_end_frame <= total_source_frames:
        raise ValueError("region bounds must satisfy 0 <= start < end <= total")
    if core_max_frames <= 0 or context_max_frames < 0:
        raise ValueError("invalid window limits")
    windows: list[EnhanceWindow] = []
    cursor = region_start_frame
    index = 0
    while cursor < region_end_frame:
        core_end = min(cursor + core_max_frames, region_end_frame)
        windows.append(
            EnhanceWindow(
                window_index=index,
                core_start_frame=cursor,
                core_end_frame=core_end,
                context_start_frame=max(0, cursor - context_max_frames),
                context_end_frame=min(total_source_frames, core_end + context_max_frames),
            )
        )
        index += 1
        cursor = core_end
    return windows


@dataclass(frozen=True)
class SpanPlacement:
    span_start_frame: int
    span_end_frame: int
    window_index: int | None  # None → the span crosses a core boundary (seam)


def classify_sentence_spans(
    spans: list[tuple[int, int]], windows: list[EnhanceWindow]
) -> list[SpanPlacement]:
    """Locate each sentence span inside a single window's core.

    A span not fully inside one core crosses a processing seam: the caller
    must reprocess it through ``sentence_window`` or quarantine it — stitching
    a clipped word across window outputs is not allowed.
    """
    placements: list[SpanPlacement] = []
    for start, end in spans:
        hit = None
        for window in windows:
            if window.core_start_frame <= start and end <= window.core_end_frame:
                hit = window.window_index
                break
        placements.append(
            SpanPlacement(
                span_start_frame=start, span_end_frame=end, window_index=hit
            )
        )
    return placements


def sentence_window(
    span_start_frame: int,
    span_end_frame: int,
    *,
    core_max_frames: int,
    context_max_frames: int,
    total_source_frames: int,
) -> EnhanceWindow | None:
    """Dedicated window for a seam-crossing sentence.

    Returns None when the sentence itself exceeds core_max — such spans cannot
    be reprocessed atomically and must be quarantined.
    """
    if span_end_frame - span_start_frame > core_max_frames:
        return None
    return EnhanceWindow(
        window_index=0,
        core_start_frame=span_start_frame,
        core_end_frame=span_end_frame,
        context_start_frame=max(0, span_start_frame - context_max_frames),
        context_end_frame=min(
            total_source_frames, span_end_frame + context_max_frames
        ),
    )


def verify_delay_alignment(
    enhance_fn: Callable[[np.ndarray], np.ndarray], *, sample_rate: int = 48000
) -> int:
    """Measure residual delay of the enhancement path with an impulse.

    With compensate_delay=True the output must be time-aligned to the input;
    returns the measured residual delay in samples (0 expected).
    """
    probe = np.zeros(sample_rate, dtype=np.float32)
    impulse_at = sample_rate // 4
    probe[impulse_at] = 1.0
    output = np.asarray(enhance_fn(probe), dtype=np.float32).reshape(-1)
    if output.size != probe.size or not np.all(np.isfinite(output)):
        raise RuntimeError("enhancement output failed length/finite checks")
    correlation = np.correlate(output, probe, mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (probe.size - 1)
    return lag


class EnhanceFn(Protocol):
    def __call__(self, context_mono_48k: np.ndarray) -> np.ndarray: ...


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_bytes(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        writer(partial)
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    def write(target: Path) -> None:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write_bytes(path, write)


def _source_run_dir(work_root: Path, source_sha256: str) -> Path:
    return work_root / source_sha256[:2] / source_sha256


def _mono_route(
    block: np.ndarray, *, downmix_allowed: bool
) -> np.ndarray | None:
    """Route raw L/R evidence to mono under the safe_mean_only policy.

    Returns None when a downmix is required but not permitted — the region is
    then blocked rather than silently flattened.
    """
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32)
    if block.shape[1] == 2:
        if not downmix_allowed:
            return None
        return block.mean(axis=1, dtype=np.float32)
    return None  # multichannel >2 never downmixes


def enhance_source_regions(
    source_path: str | Path,
    regions: list[dict[str, Any]],
    *,
    config: Dfn3ProductionConfig,
    enhance_fn: EnhanceFn,
    spatial_gate: dict[int, bool] | None = None,
    run_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Enhance regions through bounded context windows with atomic commits.

    - ``regions`` are dicts with ``region_index``/``source_start_frame``/
      ``source_end_frame`` (prefilter manifest rows).
    - ``spatial_gate`` maps region_index → downmix allowed; absent entries mean
      "no evidence" and block stereo downmix (fail closed).
    - Each region commits independently; a re-run resumes after the last
      committed region and never reprocesses finished ones.
    """
    source = Path(source_path).resolve()
    probe = probe_long_audio(source)
    source_hash = file_sha256(source)
    total_frames = int(probe["frames"])
    source_rate = int(probe["sample_rate_hz"])
    if source_rate != config.output.final_sample_rate_hz:
        raise RuntimeError(
            f"source rate {source_rate} != pinned {config.output.final_sample_rate_hz}"
        )
    core_max = round(config.chunking.core_max_seconds * source_rate)
    context_max = round(config.chunking.context_max_seconds * source_rate)

    work_root = validate_output_path(ROOT / config.output.work_root)
    run_dir = (run_state_dir or _source_run_dir(work_root, source_hash)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    records_path = run_dir / "regions.json"

    state: dict[str, Any] = {"completed_regions": []}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {"completed_regions": []}
    if state.get("config_sha256") not in (None, config.config_sha256()):
        raise RuntimeError(
            "resume refused: run was produced under a different config hash"
        )
    done: set[int] = set(state.get("completed_regions", []))

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": DFN3_REGION_IMPLEMENTATION_VERSION,
            "completed_regions": sorted(done),
            "started_at": _now(),
        },
    )

    records: dict[int, dict[str, Any]] = {}
    if records_path.is_file():
        try:
            for row in json.loads(records_path.read_text(encoding="utf-8")).get(
                "regions", []
            ):
                records[int(row["region_index"])] = row
        except (OSError, json.JSONDecodeError, KeyError):
            records = {}

    try:
        for region in regions:
            region_index = int(region["region_index"])
            start = int(region["source_start_frame"])
            end = int(region["source_end_frame"])
            if region_index in done:
                if region_index not in records:
                    # A kill between the per-region state commit and the
                    # records.json write leaves a committed output with no
                    # manifest row — rebuild it from the artifact on disk.
                    out_path = run_dir / f"region_{region_index:05d}.wav"
                    if not out_path.is_file():
                        raise RuntimeError(
                            f"region {region_index} marked complete but "
                            f"output missing: {out_path}"
                        )
                    info = sf.info(str(out_path))
                    records[region_index] = {
                        "region_index": region_index,
                        "source_sha256": source_hash,
                        "core_start_frame": start,
                        "core_end_frame": end,
                        "channel_route": "rebuilt_record",
                        "windows": [],
                        "status": "enhanced",
                        "output": {
                            "relative_path": out_path.name,
                            "sample_rate_hz": int(info.samplerate),
                            "channels": int(info.channels),
                            "frames": int(info.frames),
                            "sha256": file_sha256(out_path),
                        },
                    }
                continue
            end = int(region["source_end_frame"])
            windows = plan_windows(
                start,
                end,
                core_max_frames=core_max,
                context_max_frames=context_max,
                total_source_frames=total_frames,
            )
            downmix_allowed = bool(
                spatial_gate and spatial_gate.get(region_index, False)
            )
            channel_route = "mono_source" if probe["channels"] == 1 else (
                "safe_mean" if downmix_allowed else "blocked"
            )
            region_record: dict[str, Any] = {
                "region_index": region_index,
                "source_sha256": source_hash,
                "core_start_frame": start,
                "core_end_frame": end,
                "channel_route": channel_route,
                "windows": [],
            }
            if channel_route == "blocked":
                region_record["status"] = "blocked_spatial"
                records[region_index] = region_record
                continue

            out_path = run_dir / f"region_{region_index:05d}.wav"
            pieces: list[np.ndarray] = []
            failed: str | None = None
            with sf.SoundFile(str(source), mode="r") as audio_file:
                for window in windows:
                    audio_file.seek(window.context_start_frame)
                    context_block = audio_file.read(
                        window.context_frames, dtype="float32", always_2d=True
                    )
                    mono = _mono_route(
                        context_block, downmix_allowed=downmix_allowed
                    )
                    if mono is None:
                        failed = "downmix_not_permitted"
                        break
                    try:
                        enhanced = np.asarray(
                            enhance_fn(mono), dtype=np.float32
                        ).reshape(-1)
                    except Exception as error:
                        failed = f"backend_failure: {type(error).__name__}: {error}"
                        break
                    if (
                        enhanced.size != window.context_frames
                        or not np.all(np.isfinite(enhanced))
                    ):
                        failed = (
                            f"output_mismatch: expected {window.context_frames} "
                            f"finite samples, got {enhanced.size}"
                        )
                        break
                    offset, offset_end = window.core_output_bounds()
                    core = enhanced[offset:offset_end]
                    if core.size != window.core_frames:
                        failed = "core_slice_mismatch"
                        break
                    pieces.append(core)
                    region_record["windows"].append(
                        {
                            "window_index": window.window_index,
                            "core_start_frame": window.core_start_frame,
                            "core_end_frame": window.core_end_frame,
                            "context_start_frame": window.context_start_frame,
                            "context_end_frame": window.context_end_frame,
                            "status": "enhanced",
                        }
                    )
            if failed is None:
                enhanced_region = np.concatenate(pieces)
                if enhanced_region.size != end - start:
                    failed = "assembled_length_mismatch"
            if failed is not None:
                region_record["status"] = "failed"
                region_record["failure"] = failed
                records[region_index] = region_record
                _atomic_write_json(
                    state_path,
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "source_sha256": source_hash,
                        "config_sha256": config.config_sha256(),
                        "implementation_version": DFN3_REGION_IMPLEMENTATION_VERSION,
                        "completed_regions": sorted(done),
                        "failed_region": region_index,
                        "failed_at": _now(),
                    },
                )
                raise RuntimeError(
                    f"region {region_index} enhancement failed: {failed}"
                )

            def write_wav(target: Path) -> None:
                sf.write(
                    str(target),
                    enhanced_region,
                    config.output.final_sample_rate_hz,
                    subtype=config.output.final_subtype,
                    format="WAV",
                )

            _atomic_write_bytes(out_path, write_wav)
            region_record["status"] = "enhanced"
            region_record["output"] = {
                "relative_path": out_path.name,
                "sample_rate_hz": config.output.final_sample_rate_hz,
                "channels": config.output.final_channels,
                "frames": int(enhanced_region.size),
                "sha256": file_sha256(out_path),
            }
            records[region_index] = region_record
            done.add(region_index)
            _atomic_write_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "source_sha256": source_hash,
                    "config_sha256": config.config_sha256(),
                    "implementation_version": DFN3_REGION_IMPLEMENTATION_VERSION,
                    "completed_regions": sorted(done),
                    "updated_at": _now(),
                },
            )
    finally:
        _atomic_write_json(
            records_path,
            {
                "schema_version": 1,
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": DFN3_REGION_IMPLEMENTATION_VERSION,
                "regions": [records[key] for key in sorted(records)],
            },
        )

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "completed",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": DFN3_REGION_IMPLEMENTATION_VERSION,
            "completed_regions": sorted(done),
            "finished_at": _now(),
        },
    )
    return {
        "status": "completed",
        "source_sha256": source_hash,
        "run_dir": str(run_dir),
        "enhanced_regions": sorted(done),
        "blocked_regions": sorted(
            key for key, row in records.items() if row.get("status") == "blocked_spatial"
        ),
        "records_path": str(records_path),
    }


def make_df_enhancer(config: Dfn3ProductionConfig) -> EnhanceFn:
    """Build the production enhance function inside the DFN3 image.

    init_df runs once (model reuse); every enhance() call processes a fresh
    context buffer, so per-window inference state resets by construction.
    """
    import torch
    from df.enhance import enhance, init_df

    model_dir = (
        ROOT / config.model_files.cache_root /
        str(Path(config.model_files.config_relative_path).parent)
    ).resolve()
    model, df_state, _ = init_df(
        model_base_dir=str(model_dir),
        post_filter=config.backend.post_filter,
        log_level="WARNING",
        config_allow_defaults=True,
    )
    atten = config.backend.attenuation_limit_db

    def run(context_mono_48k: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(
            np.asarray(context_mono_48k, dtype=np.float32)[None, :]
        )
        with torch.no_grad():
            enhanced = enhance(
                model,
                df_state,
                tensor,
                pad=True,
                atten_lim_db=atten,
            )
        return enhanced.cpu().numpy().astype(np.float32)

    return run
