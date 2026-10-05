from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
import soundfile as sf
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.dataset_freeze import SpeakerEmbeddingConfig
from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.speaker_embedding import compute_speaker_embedding

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "identity_v1.yaml"
)
LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION = 2

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class IdentityWindows(_StrictFrozenModel):
    window_seconds: float = Field(ge=0.5, le=8.0)
    hop_seconds: float = Field(ge=0.1, le=4.0)
    min_window_rms_dbfs: float = Field(ge=-120.0, le=0.0)
    min_voiced_windows_for_stability: int = Field(ge=1, le=64)


class IdentityTimeline(_StrictFrozenModel):
    """Speaker-turn detection inside one source. Thresholds here are
    structural (change-point and prototype matching); they produce evidence,
    never a pass verdict."""

    turn_change_cosine: float = Field(ge=-1.0, le=1.0)
    prototype_match_cosine: float = Field(ge=-1.0, le=1.0)
    min_turn_seconds: float = Field(ge=0.2, le=30.0)
    overlap_ambiguity_margin: float = Field(ge=0.0, le=0.5)


class IdentityTargets(_StrictFrozenModel):
    """Reference scoring. Decision thresholds stay unset until calibrated —
    this stage emits cosine statistics only, and the strict gate keeps G3 at
    ``unknown`` until a calibration report binds real values."""

    reference_pack_path: str
    reference_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_audio_root: str
    max_reference_seconds: float = Field(ge=1.0, le=30.0)

    @field_validator("reference_pack_path", "reference_audio_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class IdentityPaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "IdentityPaths":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError("identity outputs cannot be written under data/inbox")
        return self


class IdentityConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    windows: IdentityWindows
    timeline: IdentityTimeline
    targets: IdentityTargets
    encoder: SpeakerEmbeddingConfig
    paths: IdentityPaths

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_identity_config(
    path: str | Path = DEFAULT_IDENTITY_CONFIG_PATH,
) -> IdentityConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"identity config must be a YAML mapping: {config_path}")
    return IdentityConfig.model_validate(payload, strict=True)


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm <= np.finfo(float).tiny:
        raise ValueError("embedding must be finite and non-zero")
    return values / norm


@dataclass(frozen=True)
class ReferenceVectors:
    normal: np.ndarray  # (n_refs, dim) L2-normalized
    negatives_by_kind: dict[str, np.ndarray]
    pack_sha256: str


def load_reference_vectors(
    config: IdentityConfig, *, encoder: Any
) -> ReferenceVectors:
    """Embed every confirmed clip of the frozen pack; drift aborts loudly."""
    pack_path = (ROOT / config.targets.reference_pack_path).resolve()
    pack = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict) or pack.get("status") != "confirmed":
        raise RuntimeError("reference pack must be a confirmed pack")
    if pack.get("pack_sha256") != config.targets.reference_pack_sha256:
        raise RuntimeError(
            "reference pack sha256 mismatch: "
            f"{pack.get('pack_sha256')} != {config.targets.reference_pack_sha256}"
        )
    audio_root = (ROOT / config.targets.reference_audio_root).resolve()

    def embed_clip(clip: dict[str, Any]) -> np.ndarray:
        relative = clip["audio_relative_path"]
        candidates = [(ROOT / relative).resolve(), (audio_root / relative).resolve()]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"reference clip audio missing: {relative}")
        if file_sha256(path) != clip["audio_sha256"]:
            raise RuntimeError(f"reference clip audio drift: {relative}")
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        limit = round(config.targets.max_reference_seconds * rate)
        return _normalized(
            compute_speaker_embedding(
                mono[:limit],
                sample_rate=int(rate),
                config=config.encoder,
                encoder=encoder,
            )
        )

    normal = np.stack([embed_clip(c) for c in pack["clips"]])
    negatives: dict[str, list[np.ndarray]] = {}
    for neg in pack.get("negatives", []):
        negatives.setdefault(neg["kind"], []).append(embed_clip(neg))
    return ReferenceVectors(
        normal=normal,
        negatives_by_kind={k: np.stack(v) for k, v in negatives.items()},
        pack_sha256=pack["pack_sha256"],
    )


@dataclass(frozen=True)
class WindowEmbedding:
    source_start_frame: int
    source_end_frame: int
    rms_dbfs: float
    vector: np.ndarray | None  # None → window below energy gate, not embedded


def embed_region_windows(
    samples: np.ndarray,
    *,
    region_start_frame: int,
    sample_rate: int,
    config: IdentityConfig,
    encoder: Any,
) -> list[WindowEmbedding]:
    """Embed fixed-hop windows across one region. Only voiced windows get a
    vector; the caller sees coverage explicitly instead of silent gaps."""
    window = round(config.windows.window_seconds * sample_rate)
    hop = round(config.windows.hop_seconds * sample_rate)
    min_rms = 10.0 ** (config.windows.min_window_rms_dbfs / 20.0)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    target_rate = int(config.encoder.input_sample_rate)
    if sample_rate != target_rate:
        embed_mono = soxr.resample(
            np.asarray(mono, dtype=np.float32),
            sample_rate,
            target_rate,
            quality="VHQ",
        )
        window_t = round(config.windows.window_seconds * target_rate)
        rate_ratio = target_rate / sample_rate
    else:
        embed_mono = mono
        window_t = window
        rate_ratio = 1.0
    out: list[WindowEmbedding] = []
    for start in range(0, max(1, samples.shape[0] - window + 1), hop):
        chunk = mono[start : start + window]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        vector = None
        if rms >= min_rms and np.all(np.isfinite(chunk)):
            start_t = round(start * rate_ratio)
            chunk_t = embed_mono[start_t : start_t + window_t]
            vector = _normalized(
                compute_speaker_embedding(
                    np.asarray(chunk_t, dtype=np.float32),
                    sample_rate=target_rate,
                    config=config.encoder,
                    encoder=encoder,
                )
            )
        out.append(
            WindowEmbedding(
                source_start_frame=region_start_frame + start,
                source_end_frame=region_start_frame + start + window,
                rms_dbfs=20.0 * float(np.log10(max(rms, 1e-12))),
                vector=vector,
            )
        )
    return out


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_label: str
    start_window: int
    end_window: int  # exclusive
    source_start_frame: int
    source_end_frame: int


def _build_prototype_labels(
    turn_vectors: list[np.ndarray],
    existing: list[np.ndarray],
    match_cosine: float,
) -> tuple[list[str], list[np.ndarray]]:
    """Greedy nearest-prototype labelling within one file. Labels are scoped
    to this source — ``speaker_000`` never claims a cross-file identity."""
    labels: list[str] = []
    prototypes = [p for p in existing]
    for vector in turn_vectors:
        best = -1.0
        best_index = -1
        for index, proto in enumerate(prototypes):
            cosine = float(np.dot(vector, proto))
            if cosine > best:
                best = cosine
                best_index = index
        if best_index >= 0 and best >= match_cosine:
            merged = _normalized((prototypes[best_index] + vector) / 2.0)
            prototypes[best_index] = merged
            labels.append(f"speaker_{best_index:03d}")
        else:
            prototypes.append(vector)
            labels.append(f"speaker_{len(prototypes) - 1:03d}")
    return labels, prototypes


def build_speaker_timeline(
    windows: list[WindowEmbedding],
    *,
    config: IdentityConfig,
    prototypes: list[np.ndarray],
) -> tuple[list[SpeakerTurn], list[np.ndarray], dict[str, Any]]:
    """Split a region's window embeddings into speaker turns.

    A turn boundary is drawn where adjacent voiced-window cosine drops below
    ``turn_change_cosine``; sub-threshold micro-turns are merged back. Returns
    (turns, updated prototypes, diagnostics).
    """
    voiced = [(i, w) for i, w in enumerate(windows) if w.vector is not None]
    diagnostics: dict[str, Any] = {
        "window_count": len(windows),
        "voiced_window_count": len(voiced),
    }
    if len(voiced) < 2:
        return [], prototypes, diagnostics

    indices = [i for i, _ in voiced]
    vectors = [w.vector for _, w in voiced]
    adjacent = [
        float(np.dot(vectors[k], vectors[k + 1])) for k in range(len(vectors) - 1)
    ]
    boundaries = [0]
    for k, cosine in enumerate(adjacent):
        if cosine < config.timeline.turn_change_cosine:
            boundaries.append(k + 1)
    boundaries.append(len(vectors))

    raw_turns = [
        (boundaries[t], boundaries[t + 1]) for t in range(len(boundaries) - 1)
    ]
    # merge turns shorter than min_turn_seconds into the better-matching side
    min_windows = max(
        1, round(config.timeline.min_turn_seconds / config.windows.hop_seconds)
    )
    merged: list[tuple[int, int]] = []
    for start, end in raw_turns:
        if merged and end - start < min_windows:
            prev_start, prev_end = merged[-1]
            left = _normalized(
                np.mean([vectors[i] for i in range(prev_start, prev_end)], axis=0)
            )
            candidate = _normalized(
                np.mean([vectors[i] for i in range(start, end)], axis=0)
            )
            if float(np.dot(candidate, left)) >= config.timeline.prototype_match_cosine * 0.8:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))

    turn_vectors = [
        _normalized(np.mean([vectors[i] for i in range(s, e)], axis=0))
        for s, e in merged
    ]
    labels, updated = _build_prototype_labels(
        turn_vectors, prototypes, config.timeline.prototype_match_cosine
    )
    turns = [
        SpeakerTurn(
            speaker_label=label,
            start_window=indices[s],
            end_window=indices[e - 1] + 1,
            source_start_frame=windows[indices[s]].source_start_frame,
            source_end_frame=windows[indices[e - 1]].source_end_frame,
        )
        for label, (s, e) in zip(labels, merged)
    ]
    diagnostics["adjacent_cosine_min"] = min(adjacent) if adjacent else None
    diagnostics["adjacent_cosine_p10"] = (
        float(np.quantile(adjacent, 0.1)) if adjacent else None
    )
    diagnostics["turn_count"] = len(turns)
    diagnostics["speaker_labels"] = sorted({t.speaker_label for t in turns})
    return turns, updated, diagnostics


def score_region_identity(
    windows: list[WindowEmbedding],
    turns: list[SpeakerTurn],
    references: ReferenceVectors,
    *,
    config: IdentityConfig,
) -> dict[str, Any]:
    """Produce G2/G3 evidence for one region. No pass/fail is declared here:
    uncalibrated thresholds mean the output is evidence, and downstream gates
    decide."""
    voiced = [w for w in windows if w.vector is not None]
    record: dict[str, Any] = {
        "window_count": len(windows),
        "voiced_window_count": len(voiced),
        "turn_count": len(turns),
        "speaker_labels": sorted({t.speaker_label for t in turns}),
        "turns": [
            {
                "speaker_label": t.speaker_label,
                "source_start_frame": t.source_start_frame,
                "source_end_frame": t.source_end_frame,
            }
            for t in turns
        ],
    }
    if not voiced:
        record["status"] = "insufficient_speech"
        return record

    matrix = np.stack([w.vector for w in voiced])
    normal_cos = matrix @ references.normal.T  # (windows, n_refs)
    record["target_cosine_median"] = float(np.median(np.max(normal_cos, axis=1)))
    record["target_cosine_min"] = float(np.min(np.max(normal_cos, axis=1)))
    record["target_cosine_p10"] = float(
        np.quantile(np.max(normal_cos, axis=1), 0.1)
    )
    neg_max: dict[str, float] = {}
    for kind, vectors in references.negatives_by_kind.items():
        neg_max[kind] = float(np.max(matrix @ vectors.T))
    record["negative_cosine_max_by_kind"] = neg_max

    adjacent = [
        float(np.dot(matrix[k], matrix[k + 1])) for k in range(len(matrix) - 1)
    ]
    record["adjacent_cosine_min"] = min(adjacent) if adjacent else None

    # Overlap suspicion: windows that match no running prototype well are
    # ambiguous — could be a second voice mixed in. This is a heuristic flag,
    # not a calibrated overlap detector.
    if len(turns) >= 2 or (adjacent and record["adjacent_cosine_min"] < config.timeline.turn_change_cosine):
        record["speaker_change_suspect"] = True
    else:
        record["speaker_change_suspect"] = False
    dominant_share = 0.0
    if turns:
        counts: dict[str, int] = {}
        for t in turns:
            counts[t.speaker_label] = counts.get(t.speaker_label, 0) + (
                t.end_window - t.start_window
            )
        dominant_share = max(counts.values()) / max(1, sum(counts.values()))
    record["dominant_turn_share"] = dominant_share
    record["status"] = "single_speaker_candidate" if len(turns) <= 1 and len(voiced) >= config.windows.min_voiced_windows_for_stability else (
        "speaker_change_suspect" if len(turns) > 1 else "insufficient_speech"
    )
    return record


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


def identity_completed(
    work_root: Path, source_sha256: str, config: IdentityConfig
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
        == LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION
        and state.get("source_sha256") == source_sha256
    )


def identify_source(
    source_path: str | Path,
    regions: list[dict[str, Any]],
    *,
    config: IdentityConfig,
    encoder: Any,
    references: ReferenceVectors,
) -> dict[str, Any]:
    """Build the speaker timeline + target evidence for one source.

    ``regions`` should be the prefilter manifest rows; every region gets an
    identity record, including quarantined ones (their evidence is still
    useful for the event/style gates)."""
    source = Path(source_path).resolve()
    source_hash = file_sha256(source)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    source_dir = _source_dir(work_root, source_hash)
    source_dir.mkdir(parents=True, exist_ok=True)
    state_path = source_dir / "state.json"

    if identity_completed(work_root, source_hash, config):
        return {
            "status": "cached",
            "source_sha256": source_hash,
            "identity_path": str(source_dir / "identity.json"),
        }

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION,
            "started_at": _now(),
        },
    )
    try:
        info = sf.info(str(source))
        sample_rate = int(info.samplerate)
        prototypes: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        with sf.SoundFile(str(source), mode="r") as audio_file:
            for region in sorted(
                regions, key=lambda row: int(row["source_start_frame"])
            ):
                start = int(region["source_start_frame"])
                end = int(region["source_end_frame"])
                audio_file.seek(start)
                block = audio_file.read(
                    end - start, dtype="float32", always_2d=True
                )
                windows = embed_region_windows(
                    block,
                    region_start_frame=start,
                    sample_rate=sample_rate,
                    config=config,
                    encoder=encoder,
                )
                turns, prototypes, diagnostics = build_speaker_timeline(
                    windows, config=config, prototypes=prototypes
                )
                record = score_region_identity(
                    windows, turns, references, config=config
                )
                record["region_index"] = int(region["region_index"])
                record["source_start_frame"] = start
                record["source_end_frame"] = end
                record["prefilter_status"] = region.get("status", "usable")
                record["diagnostics"] = diagnostics
                records.append(record)

        manifest = {
            "schema_version": 1,
            "implementation_version": LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION,
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "reference_pack_sha256": references.pack_sha256,
            "speaker_labels_seen": sorted(
                {
                    label
                    for row in records
                    for label in row.get("speaker_labels", [])
                }
            ),
            "regions": records,
        }
        _atomic_write_json(source_dir / "identity.json", manifest)
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "completed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION,
                "region_count": len(records),
                "finished_at": _now(),
            },
        )
        return {
            "status": "completed",
            "source_sha256": source_hash,
            "region_count": len(records),
            "identity_path": str(source_dir / "identity.json"),
        }
    except Exception as error:
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_IDENTITY_IMPLEMENTATION_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": _now(),
            },
        )
        raise


def run_identity(
    jobs: list[tuple[str | Path, list[dict[str, Any]]]],
    *,
    config: IdentityConfig,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Multi-source entry. The encoder loads once and is reused across
    sources; speaker prototypes reset per file by construction."""
    if encoder is None:
        from dots_tts_lab.speaker_embedding import load_speaker_encoder

        encoder = load_speaker_encoder(config.encoder)
    references = load_reference_vectors(config, encoder=encoder)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    for partial in work_root.rglob("*.partial"):
        partial.unlink()
    results = []
    for source, regions in jobs:
        results.append(
            identify_source(
                source,
                regions,
                config=config,
                encoder=encoder,
                references=references,
            )
        )
    return {
        "status": "completed",
        "config_sha256": config.config_sha256(),
        "sources": results,
    }
