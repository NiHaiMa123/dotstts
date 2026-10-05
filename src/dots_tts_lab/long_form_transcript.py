from __future__ import annotations

import hashlib
import json
import os
import subprocess
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Literal

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_features import read_source_segment
from dots_tts_lab.long_form_paths import validate_output_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPT_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "transcript_v1.yaml"
)
LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION = 4

SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SENTENCE_END = "。！？!?…—~"
_DIGIT_CHARS = "0123456789０１２３４５６７８９零一二三四五六七八九十百千万亿两"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class TranscriptBackend(_StrictFrozenModel):
    """One ASR backend: an isolated venv python plus a pinned backend config."""

    backend_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")
    env_python: str
    backend_config: str

    @field_validator("env_python", "backend_config")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class TranscriptAsrSegment(_StrictFrozenModel):
    """How usable regions are grouped into ASR transcription units."""

    merge_gap_seconds: float = Field(ge=0.0, le=5.0)
    max_seconds: float = Field(gt=1.0, le=34.0)
    audio_sample_rate_hz: Literal[16000]
    audio_subtype: Literal["PCM_16"]
    transcribe_quarantined: bool = False


class TranscriptSentence(_StrictFrozenModel):
    """Final natural-sentence candidate bounds (LF-12F policy: 3–12 s)."""

    min_seconds: float = Field(ge=0.5, le=12.0)
    max_seconds: float = Field(gt=3.0, le=30.0)
    pause_split_seconds: float = Field(ge=0.2, le=3.0)
    edge_margin_seconds: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "TranscriptSentence":
        if self.min_seconds >= self.max_seconds:
            raise ValueError("sentence min_seconds must be < max_seconds")
        return self


class TranscriptPaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "TranscriptPaths":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError(
                    "transcript outputs cannot be written under data/inbox"
                )
        return self


class TranscriptConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    primary: TranscriptBackend  # must provide word timestamps (aligner choice)
    secondary: TranscriptBackend  # independent text, no timestamps required
    asr_segment: TranscriptAsrSegment
    sentence: TranscriptSentence
    paths: TranscriptPaths

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_transcript_config(
    path: str | Path = DEFAULT_TRANSCRIPT_CONFIG_PATH,
) -> TranscriptConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"transcript config must be a YAML mapping: {config_path}")
    return TranscriptConfig.model_validate(payload, strict=True)


def normalize_zh_text(text: str) -> str:
    """Agreement normalization: NFKC, strip whitespace/punctuation/symbols,
    case-fold Latin. Digits and particles are NOT folded away — a 3-vs-三
    divergence is a real disagreement and must surface as one."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in normalized if not unicodedata.category(ch).startswith(("P", "Z", "S"))
    )


def compare_hypotheses(primary: str, secondary: str) -> dict[str, Any]:
    """Exact-match agreement after normalization plus a coarse disagreement
    class. Anything but ``match`` keeps the candidate text unconfirmed."""
    a = normalize_zh_text(primary)
    b = normalize_zh_text(secondary)
    if a == b and a:
        return {"status": "match", "normalized_text": a}
    if not a or not b:
        return {"status": "coverage_gap", "normalized_text": a or b}
    diff_chars = {ch for ch in a + b if (a.count(ch) != b.count(ch))}
    reason = "text_mismatch"
    if diff_chars and all(ch in _DIGIT_CHARS or ch.isdigit() for ch in diff_chars):
        reason = "digit_disagreement"
    elif any(ch.isascii() and ch.isalpha() for ch in diff_chars):
        reason = "latin_or_name_disagreement"
    return {
        "status": reason,
        "normalized_text_primary": a,
        "normalized_text_secondary": b,
    }


def group_regions_for_asr(
    regions: list[dict[str, Any]],
    *,
    merge_gap_seconds: float,
    max_seconds: float,
    sample_rate: int,
    transcribe_quarantined: bool = False,
) -> list[dict[str, int]]:
    """Merge speech regions into bounded ASR transcription units.

    With ``transcribe_quarantined`` off (default) only ``usable`` regions
    group together and quarantined regions break the chain. When on, every
    speech region is transcribed as evidence — quarantine is not rejection —
    and each resulting sentence span is later flagged with
    ``overlaps_quarantine`` when its range touches a quarantined region."""
    usable = sorted(
        (
            int(r["source_start_frame"]),
            int(r["source_end_frame"]),
        )
        for r in regions
        if transcribe_quarantined or r.get("status") == "usable"
    )
    gap = round(merge_gap_seconds * sample_rate)
    limit = round(max_seconds * sample_rate)
    groups: list[dict[str, int]] = []
    for start, end in usable:
        if (
            groups
            and start - groups[-1]["end"] <= gap
            and end - groups[-1]["start"] <= limit
        ):
            groups[-1]["end"] = end
        else:
            groups.append({"start": start, "end": end})
    return groups


def extract_sentence_spans(
    words: list[dict[str, Any]],
    *,
    segment_start_frame: int,
    segment_end_frame: int,
    sample_rate: int,
    config: TranscriptSentence,
) -> list[dict[str, Any]]:
    """Group word-level timestamps into natural-sentence spans.

    Splits on sentence-ending punctuation and pauses ≥ pause_split_seconds.
    Spans longer than max_seconds split at the deepest internal pause; spans
    still too long or shorter than min_seconds are returned with a reject
    status — they are evidence, not candidates.
    """
    ordered = sorted(
        (
            {
                "start": float(w["start"]),
                "end": float(w["end"]),
                "text": str(w.get("text", "")),
            }
            for w in words
            if w.get("start") is not None and w.get("end") is not None
        ),
        key=lambda w: w["start"],
    )
    if not ordered:
        return []

    # Accumulate words until the span reaches min_seconds, then emit at the
    # next natural boundary (sentence-ending punctuation or a real pause).
    # Slow ASMR speech legitimately has ~0.5 s gaps between every token, so a
    # bare pause is not a sentence boundary by itself.
    chunks: list[list[dict[str, Any]]] = [[]]
    for word in ordered:
        chunk = chunks[-1]
        if chunk:
            span_duration = chunk[-1]["end"] - chunk[0]["start"]
            gap = word["start"] - chunk[-1]["end"]
            prev_text = chunk[-1]["text"].strip()
            punctuated = bool(prev_text) and prev_text[-1] in _SENTENCE_END
            natural_boundary = punctuated or gap >= config.pause_split_seconds
            if natural_boundary and (
                span_duration >= config.min_seconds
                or (punctuated and gap >= config.pause_split_seconds)
            ):
                chunks.append([])
                chunk = chunks[-1]
        chunk.append(word)

    spans: list[dict[str, Any]] = []
    min_s, max_s = config.min_seconds, config.max_seconds

    def emit(chunk: list[dict[str, Any]]) -> None:
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        duration = end - start
        text = "".join(w["text"] for w in chunk).strip()
        if duration < min_s:
            status = "too_short"
        elif duration <= max_s:
            status = "candidate"
        else:
            status = "too_long"
        spans.append(
            {
                "source_start_frame": segment_start_frame
                + round(start * sample_rate),
                "source_end_frame": segment_start_frame + round(end * sample_rate),
                "duration_seconds": round(duration, 6),
                "primary_text": text,
                "word_count": len(chunk),
                "status": status,
            }
        )

    for chunk in chunks:
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        if end - start <= max_s:
            emit(chunk)
            continue
        # split over-long chunks at the deepest internal pause
        remaining = chunk
        while remaining and remaining[-1]["end"] - remaining[0]["start"] > max_s:
            cut_at = remaining[0]["start"] + max_s
            gaps = [
                (remaining[i + 1]["start"] - remaining[i]["end"], i)
                for i in range(len(remaining) - 1)
                if remaining[i + 1]["start"] <= cut_at
            ]
            if not gaps:
                emit(remaining)
                break
            split_index = max(gaps)[1] + 1
            emit(remaining[:split_index])
            remaining = remaining[split_index:]
        if remaining:
            emit(remaining)

    # clamp into the ASR segment bounds; drop degenerate spans
    out = []
    for span in spans:
        span["source_start_frame"] = max(
            segment_start_frame, min(span["source_start_frame"], segment_end_frame)
        )
        span["source_end_frame"] = max(
            span["source_start_frame"],
            min(span["source_end_frame"], segment_end_frame),
        )
        if span["source_end_frame"] > span["source_start_frame"]:
            out.append(span)
    return out


def _word_boundary_clean(span: dict[str, Any], words: list[dict[str, Any]], sample_rate: int, segment_start_frame: int) -> bool:
    """A span's edges must sit on word boundaries or silence, not inside a
    word — clipping a character is a critical defect."""
    if not words:
        return False
    margin = 0.04  # 40 ms tolerance: an edge may sit on a word boundary, but
    # strictly inside a word means a clipped character — a critical defect
    s = (span["source_start_frame"] - segment_start_frame) / sample_rate
    e = (span["source_end_frame"] - segment_start_frame) / sample_rate
    for word in words:
        ws, we = float(word["start"]), float(word["end"])
        if (ws + margin < s < we - margin) or (ws + margin < e < we - margin):
            return False
    return True


# ---------------------------------------------------------------------------
# Backend invocation
# ---------------------------------------------------------------------------

BackendFn = Callable[[Path, list[dict[str, Any]]], dict[str, Any]]


def _materialize_asr_audio(
    source: Path,
    units: list[dict[str, int]],
    *,
    out_dir: Path,
    sample_rate: int,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, unit in enumerate(units):
        samples, rate = read_source_segment(
            source, start_frame=unit["start"], end_frame=unit["end"]
        )
        mono = (
            samples.mean(axis=1) if samples.ndim == 2 else samples
        ).astype(np.float32)
        if rate != sample_rate:
            import soxr

            mono = np.asarray(
                soxr.resample(mono, rate, sample_rate, quality="HQ"),
                dtype=np.float32,
            )
        name = f"unit_{index:05d}.wav"
        target = out_dir / name
        sf.write(str(target), mono, sample_rate, subtype="PCM_16")
        rows.append(
            {
                "asset_sha256": hashlib.sha256(
                    f"{unit['start']}:{unit['end']}".encode()
                ).hexdigest(),
                "derived_relative_path": name,
                "derived_sha256": file_sha256(target),
                "source_start_frame": unit["start"],
                "source_end_frame": unit["end"],
            }
        )
    return rows


def subprocess_backend(backend: TranscriptBackend) -> BackendFn:
    """Production backend call: reuses scripts/run_asr_backend.py inside the
    pinned per-backend venv so ASR deps never enter the main environment."""

    def run(audio_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
        work_dir = audio_dir.parent
        manifest_path = work_dir / f"{backend.backend_id}_manifest.jsonl"
        output_path = work_dir / f"{backend.backend_id}_results.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "asset_sha256": row["asset_sha256"],
                            "derived_relative_path": row["derived_relative_path"],
                            "derived_sha256": row["derived_sha256"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        command = [
            str((ROOT / backend.env_python).resolve()),
            str((ROOT / "scripts" / "run_asr_backend.py").resolve()),
            "--config",
            str((ROOT / backend.backend_config).resolve()),
            "--manifest",
            str(manifest_path.resolve()),
            "--audio-root",
            str(audio_dir.resolve()),
            "--model-root",
            str((ROOT / "data" / "work" / "models" / "asr").resolve()),
            "--output",
            str(output_path.resolve()),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{backend.backend_id} ASR failed: "
                + (completed.stderr.strip() or completed.stdout.strip())[-2000:]
            )
        return json.loads(output_path.read_text(encoding="utf-8"))

    return run


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _source_dir(work_root: Path, source_sha256: str) -> Path:
    return work_root / source_sha256[:2] / source_sha256


def transcript_completed(
    work_root: Path, source_sha256: str, config: TranscriptConfig
) -> bool:
    state_path = _source_dir(work_root, source_sha256) / "state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("status") == "completed"
        and state.get("config_sha256") == config.config_sha256()
        and state.get("implementation_version")
        == LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION
        and state.get("source_sha256") == source_sha256
    )


def transcribe_source(
    source_path: str | Path,
    regions: list[dict[str, Any]],
    *,
    config: TranscriptConfig,
    primary_fn: BackendFn,
    secondary_fn: BackendFn,
) -> dict[str, Any]:
    """Dual-ASR + sentence slicing for one source.

    The primary backend supplies word timestamps (the chosen zh aligner);
    the secondary supplies an independent hypothesis on each final span.
    Agreement must be exact after normalization — anything else leaves the
    text unconfirmed, and ASR output is never treated as confirmed text.
    """
    source = Path(source_path).resolve()
    source_hash = file_sha256(source)
    probe = sf.info(str(source))
    sample_rate = int(probe.samplerate)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    source_dir = _source_dir(work_root, source_hash)
    source_dir.mkdir(parents=True, exist_ok=True)
    state_path = source_dir / "state.json"

    if transcript_completed(work_root, source_hash, config):
        return {
            "status": "cached",
            "source_sha256": source_hash,
            "transcript_path": str(source_dir / "transcript.json"),
        }

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION,
            "started_at": _now(),
        },
    )
    try:
        quarantine_ranges = [
            (int(r["source_start_frame"]), int(r["source_end_frame"]))
            for r in regions
            if r.get("status") != "usable"
        ]
        units = group_regions_for_asr(
            regions,
            merge_gap_seconds=config.asr_segment.merge_gap_seconds,
            max_seconds=config.asr_segment.max_seconds,
            sample_rate=sample_rate,
            transcribe_quarantined=config.asr_segment.transcribe_quarantined,
        )
        audio_dir = source_dir / "asr_units"
        unit_rows = _materialize_asr_audio(
            source,
            units,
            out_dir=audio_dir,
            sample_rate=config.asr_segment.audio_sample_rate_hz,
        )
        primary = primary_fn(audio_dir, unit_rows)
        by_asset = {
            item.get("asset_sha256"): item
            for item in primary.get("results", primary.get("outputs", []))
        }

        sentence_units: list[dict[str, Any]] = []
        for row in unit_rows:
            item = by_asset.get(row["asset_sha256"]) or {}
            words = [
                word
                for segment in (item.get("metadata") or {}).get(
                    "timestamp_segments", []
                )
                for word in segment.get("words", [])
            ]
            spans = extract_sentence_spans(
                words,
                segment_start_frame=row["source_start_frame"],
                segment_end_frame=row["source_end_frame"],
                sample_rate=sample_rate,
                config=config.sentence,
            )
            primary_text = str(item.get("hypothesis", ""))
            for span in spans:
                span["asr_unit"] = {
                    "start": row["source_start_frame"],
                    "end": row["source_end_frame"],
                }
                span["boundary_clean"] = _word_boundary_clean(
                    span, words, sample_rate, row["source_start_frame"]
                )
                span["primary_unit_text"] = primary_text
                sentence_units.append(span)
        # secondary pass over the final sentence spans only
        sentence_audio_dir = source_dir / "asr_sentences"
        sentence_audio_rows = _materialize_asr_audio(
            source,
            [
                {"start": s["source_start_frame"], "end": s["source_end_frame"]}
                for s in sentence_units
            ],
            out_dir=sentence_audio_dir,
            sample_rate=config.asr_segment.audio_sample_rate_hz,
        )
        secondary = secondary_fn(sentence_audio_dir, sentence_audio_rows)
        secondary_by_asset = {
            item.get("asset_sha256"): item
            for item in secondary.get("results", secondary.get("outputs", []))
        }

        records: list[dict[str, Any]] = []
        for span, audio_row in zip(sentence_units, sentence_audio_rows):
            secondary_item = secondary_by_asset.get(audio_row["asset_sha256"]) or {}
            agreement = compare_hypotheses(
                span["primary_text"], str(secondary_item.get("hypothesis", ""))
            )
            overlaps_quarantine = any(
                span["source_start_frame"] < qe and qs < span["source_end_frame"]
                for qs, qe in quarantine_ranges
            )
            status = span["status"]
            if status == "candidate":
                if not span["boundary_clean"]:
                    status = "dirty_boundary"
                elif overlaps_quarantine:
                    status = "covers_quarantine"
                elif agreement["status"] != "match":
                    status = agreement["status"]
            records.append(
                {
                    "source_start_frame": span["source_start_frame"],
                    "source_end_frame": span["source_end_frame"],
                    "duration_seconds": span["duration_seconds"],
                    "status": status,
                    "boundary_clean": span["boundary_clean"],
                    "overlaps_quarantine": overlaps_quarantine,
                    "primary_text": span["primary_text"],
                    "primary_text_sha256": hashlib.sha256(
                        span["primary_text"].encode("utf-8")
                    ).hexdigest(),
                    "secondary_text": str(secondary_item.get("hypothesis", "")),
                    "secondary_text_sha256": hashlib.sha256(
                        str(secondary_item.get("hypothesis", "")).encode("utf-8")
                    ).hexdigest(),
                    "text_agreement": agreement,
                    "asr_unit": span["asr_unit"],
                    "candidate_text_is_confirmed": False,
                    "requires_final_recheck": True,
                }
            )

        manifest = {
            "schema_version": 1,
            "implementation_version": LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION,
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "asr_unit_count": len(unit_rows),
            "primary_backend": config.primary.backend_id,
            "secondary_backend": config.secondary.backend_id,
            "sentences": records,
            "candidate_count": sum(
                1 for r in records if r["status"] == "candidate"
            ),
            "agreed_count": sum(
                1
                for r in records
                if r["text_agreement"].get("status") == "match"
            ),
        }
        _atomic_write_json(source_dir / "transcript.json", manifest)
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "completed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION,
                "sentence_count": len(records),
                "finished_at": _now(),
            },
        )
        return {
            "status": "completed",
            "source_sha256": source_hash,
            "sentence_count": len(records),
            "candidate_count": manifest["candidate_count"],
            "agreed_count": manifest["agreed_count"],
            "transcript_path": str(source_dir / "transcript.json"),
        }
    except Exception as error:
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_TRANSCRIPT_IMPLEMENTATION_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": _now(),
            },
        )
        raise


def run_transcript(
    jobs: list[tuple[str | Path, list[dict[str, Any]]]],
    *,
    config: TranscriptConfig,
    primary_fn: BackendFn | None = None,
    secondary_fn: BackendFn | None = None,
) -> dict[str, Any]:
    """Multi-source entry; default backends invoke the pinned per-env ASR."""
    primary_call = primary_fn or subprocess_backend(config.primary)
    secondary_call = secondary_fn or subprocess_backend(config.secondary)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    for partial in work_root.rglob("*.partial"):
        partial.unlink()
    results = [
        transcribe_source(
            source,
            regions,
            config=config,
            primary_fn=primary_call,
            secondary_fn=secondary_call,
        )
        for source, regions in jobs
    ]
    return {
        "status": "completed",
        "config_sha256": config.config_sha256(),
        "sources": results,
    }
