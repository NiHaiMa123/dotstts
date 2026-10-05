from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_features import (
    _spectral_flatness_median,
    read_source_segment,
)
from dots_tts_lab.long_form_grouping import estimate_pitch_periodicity
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_prefilter import _transient_strengths

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STYLE_EVENT_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "style_event_v1.yaml"
)
LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION = 2

SHA256_PATTERN = r"^[0-9a-f]{64}$"

STYLE_FEATURE_FIELDS = (
    "level_dbfs",
    "silence_ratio",
    "spectral_flatness_median",
    "periodicity",
    "pitch_median_hz",
    "side_to_mid_db",
    "pan_standard_deviation",
    "stereo_correlation",
    "transients_per_second",
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class StyleEventTargets(_StrictFrozenModel):
    reference_pack_path: str
    reference_pack_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_audio_root: str
    max_reference_seconds: float = Field(ge=1.0, le=30.0)

    @field_validator("reference_pack_path", "reference_audio_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class StyleEventEvents(_StrictFrozenModel):
    """Coarse event tagging. ``backend: none`` keeps the event gate at
    unknown — every region reports missing evidence rather than passing."""

    backend: Literal["panns_cnn14", "none"]
    sample_rate_hz: Literal[32000] = 32000
    min_tag_seconds: float = Field(ge=0.5, le=5.0)
    labels_csv_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    frame_seconds: float = Field(ge=0.005, le=0.1)
    risk_label_patterns: list[str] = Field(min_length=1)
    # Stationary/background noise classes — what a denoiser could remove.
    # Separate from risk_label_patterns (contamination events → quarantine).
    noise_label_patterns: list[str] = Field(default_factory=list)
    transient_hop_samples: int = Field(ge=64, le=1024)
    transient_zscore: float = Field(ge=1.0, le=20.0)


class StyleEventPaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "StyleEventPaths":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError(
                    "style/event outputs cannot be written under data/inbox"
                )
        return self


class StyleEventConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    targets: StyleEventTargets
    events: StyleEventEvents
    paths: StyleEventPaths

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_style_event_config(
    path: str | Path = DEFAULT_STYLE_EVENT_CONFIG_PATH,
) -> StyleEventConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"style/event config must be a YAML mapping: {config_path}")
    return StyleEventConfig.model_validate(payload, strict=True)


def _dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(rms, 1e-12))


def region_style_features(
    samples: np.ndarray, *, sample_rate: int, transient_hop: int = 128
) -> dict[str, float | None]:
    """Compact style profile over one region. All features are cheap DSP
    proxies — they characterise voice style and spatial/event context; none
    alone proves normality."""
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    mono = samples.mean(axis=1, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))

    frame = max(1, round(sample_rate * 0.04))
    hop = max(1, round(sample_rate * 0.02))
    n_frames = max(1, (len(mono) - frame) // hop + 1) if len(mono) > frame else 1
    levels = []
    for i in range(n_frames):
        chunk = mono[i * hop : i * hop + frame]
        if chunk.size:
            levels.append(float(np.mean(chunk.astype(np.float64) ** 2)))
    level_db = np.asarray([10.0 * math.log10(max(e, 1e-12)) for e in levels])
    active = level_db > (float(np.max(level_db)) - 30.0)
    silence_ratio = float(1.0 - np.mean(active)) if active.size else 1.0

    pitch = estimate_pitch_periodicity(
        samples if samples.ndim == 2 else mono,
        sample_rate=sample_rate,
        analysis_channel=0,
    )

    side_to_mid = None
    pan_std = None
    correlation = None
    if samples.shape[1] == 2:
        left = samples[:, 0].astype(np.float64)
        right = samples[:, 1].astype(np.float64)
        denom = math.sqrt(
            float(np.sum(left * left)) * float(np.sum(right * right))
        )
        correlation = (
            float(np.sum(left * right) / denom)
            if denom > np.finfo(float).tiny
            else None
        )
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mid_rms = float(np.sqrt(np.mean(mid * mid)))
        side_rms = float(np.sqrt(np.mean(side * side)))
        side_to_mid = (
            20.0 * math.log10(side_rms / mid_rms)
            if mid_rms > 0 and side_rms > 0
            else -120.0
        )
        win = max(1, round(0.25 * sample_rate))
        pans = []
        for start in range(0, samples.shape[0], win):
            block = samples[start : start + win]
            energy = np.mean(np.square(block.astype(np.float64)), axis=0)
            total = float(np.sum(energy))
            if total > 1e-12:
                pans.append(float((energy[1] - energy[0]) / total))
        pan_std = float(np.std(pans)) if pans else 0.0

    mono_16k = (
        mono
        if sample_rate == 16000
        else np.asarray(
            soxr.resample(mono, sample_rate, 16000, quality="VHQ"),
            dtype=np.float32,
        )
    )
    strengths = _transient_strengths(mono_16k, transient_hop)
    duration = len(mono) / sample_rate
    transients = (
        float(np.count_nonzero(strengths > np.median(strengths) * 4 + 1e-6))
        / duration
        if duration > 0 and strengths.size
        else 0.0
    )

    return {
        "level_dbfs": _dbfs(rms),
        "silence_ratio": silence_ratio,
        "spectral_flatness_median": _spectral_flatness_median(mono, sample_rate),
        "periodicity": float(pitch["periodicity"]),
        "pitch_median_hz": pitch["pitch_median_hz"],
        "side_to_mid_db": side_to_mid,
        "pan_standard_deviation": pan_std,
        "stereo_correlation": correlation,
        "transients_per_second": transients,
    }


def _feature_vector(features: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(features[field])
            if features.get(field) is not None and math.isfinite(float(features[field]))
            else np.nan
            for field in STYLE_FEATURE_FIELDS
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class StyleAnchors:
    """Per-kind feature distributions from the confirmed reference pack."""

    normal_matrix: np.ndarray  # (n_clips, n_features)
    negative_matrices: dict[str, np.ndarray]
    pack_sha256: str


def _load_pack_clip_audio(
    clip: dict[str, Any], *, audio_root: Path, max_seconds: float
) -> tuple[np.ndarray, int]:
    relative = clip["audio_relative_path"]
    candidates = [(ROOT / relative).resolve(), (audio_root / relative).resolve()]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"reference clip audio missing: {relative}")
    if file_sha256(path) != clip["audio_sha256"]:
        raise RuntimeError(f"reference clip audio drift: {relative}")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    limit = round(max_seconds * rate)
    return audio[:limit], int(rate)


def load_style_anchors(config: StyleEventConfig) -> StyleAnchors:
    pack_path = (ROOT / config.targets.reference_pack_path).resolve()
    pack = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict) or pack.get("status") != "confirmed":
        raise RuntimeError("reference pack must be a confirmed pack")
    if pack.get("pack_sha256") != config.targets.reference_pack_sha256:
        raise RuntimeError("reference pack sha256 mismatch")
    audio_root = (ROOT / config.targets.reference_audio_root).resolve()

    def matrix_for(clips: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        for clip in clips:
            audio, rate = _load_pack_clip_audio(
                clip,
                audio_root=audio_root,
                max_seconds=config.targets.max_reference_seconds,
            )
            rows.append(
                _feature_vector(region_style_features(audio, sample_rate=rate))
            )
        return np.stack(rows)

    normal = matrix_for(pack["clips"])
    negatives: dict[str, np.ndarray] = {}
    for neg in pack.get("negatives", []):
        audio, rate = _load_pack_clip_audio(
            neg,
            audio_root=audio_root,
            max_seconds=config.targets.max_reference_seconds,
        )
        vector = _feature_vector(region_style_features(audio, sample_rate=rate))
        negatives.setdefault(neg["kind"], []).append(vector)
    return StyleAnchors(
        normal_matrix=normal,
        negative_matrices={k: np.stack(v) for k, v in negatives.items()},
        pack_sha256=pack["pack_sha256"],
    )


def _distribution_distance(
    vector: np.ndarray, anchors: np.ndarray
) -> dict[str, float]:
    """Robust per-feature z-distance to an anchor set: median/MAD over the
    anchors' own distribution, then worst and mean |z|. Columns with no finite
    anchor data (e.g. pitch absent in whispers) contribute nothing; missing
    values in the query vector fall back to the anchor median."""
    col_has_data = np.isfinite(anchors).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(anchors, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        mads = np.nanmedian(np.abs(anchors - medians), axis=0)
    stds = np.ones(anchors.shape[1])
    if col_has_data.any():
        stds[col_has_data] = np.nanstd(anchors[:, col_has_data], axis=0)
    scales = np.where(mads > 1e-8, mads * 1.4826, stds)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    usable = col_has_data
    missing_z = 2.0  # unmeasurable-but-expected feature counts as off-distribution
    filled = np.where(
        np.isfinite(vector),
        vector,
        np.where(usable, medians - missing_z * scales, medians),
    )
    z = np.abs(filled - medians) / scales
    z = np.where(usable, z, 0.0)
    count = max(1, int(np.count_nonzero(usable)))
    anchor_rows = np.where(np.isfinite(anchors), anchors, medians[None, :])
    per_anchor = np.sum(
        np.abs(filled[None, :] - anchor_rows) / scales * usable[None, :],
        axis=1,
    ) / count
    return {
        "max_z": float(np.max(z)),
        "mean_z": float(np.sum(z) / count),
        "nearest_anchor_mean_z": float(np.min(per_anchor)),
    }


def score_style_evidence(
    features: dict[str, Any], anchors: StyleAnchors
) -> dict[str, Any]:
    """Style distances: closeness to the normal reference distribution and to
    each hard-negative kind. Evidence only — thresholds stay uncalibrated."""
    vector = _feature_vector(features)
    normal = _distribution_distance(vector, anchors.normal_matrix)
    negatives = {
        kind: _distribution_distance(vector, matrix)
        for kind, matrix in anchors.negative_matrices.items()
    }
    nearest_kind = min(
        negatives, key=lambda kind: negatives[kind]["nearest_anchor_mean_z"]
    ) if negatives else None
    return {
        "normal_max_z": normal["max_z"],
        "normal_mean_z": normal["mean_z"],
        "normal_nearest_anchor_mean_z": normal["nearest_anchor_mean_z"],
        "negative_distance_by_kind": {
            kind: round(dist["nearest_anchor_mean_z"], 6)
            for kind, dist in negatives.items()
        },
        "nearest_negative_kind": nearest_kind,
        "nearest_negative_distance": (
            round(negatives[nearest_kind]["nearest_anchor_mean_z"], 6)
            if nearest_kind
            else None
        ),
    }


class EventBackend(Protocol):
    def status(self) -> str: ...

    def tag(self, mono: np.ndarray, *, sample_rate: int) -> dict[str, Any]: ...


class NoEventBackend:
    def status(self) -> str:
        return "unavailable"

    def tag(self, mono: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
        return {"status": "unavailable"}


class PannsEventBackend:
    """PANNs CNN14 AudioSet tagger. Checkpoint and label files are hash-pinned
    in the config; a drift aborts instead of silently tagging with a
    different model."""

    def __init__(self, config: StyleEventEvents) -> None:
        if config.backend != "panns_cnn14":
            raise RuntimeError("PannsEventBackend requires backend=panns_cnn14")
        home = Path.home() / "panns_data"
        labels_csv = home / "class_labels_indices.csv"
        checkpoint = home / "Cnn14_mAP=0.431.pth"
        for label, path, expected in (
            ("labels", labels_csv, config.labels_csv_sha256),
            ("checkpoint", checkpoint, config.checkpoint_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"PANNs {label} file missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"PANNs {label} sha256 drift: {actual} != {expected}"
                )
        from panns_inference import AudioTagging
        from panns_inference.config import labels as panns_labels

        self._labels = list(panns_labels)
        self._at = AudioTagging(checkpoint_path=str(checkpoint), device="cpu")
        self._rate = config.sample_rate_hz
        self._frame_seconds = config.frame_seconds
        self._risk_patterns = [p.lower() for p in config.risk_label_patterns]
        self._noise_patterns = [p.lower() for p in config.noise_label_patterns]

    def status(self) -> str:
        return "ready"

    def tag(self, mono: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
        audio = np.asarray(mono, dtype=np.float32).reshape(-1)
        if sample_rate != self._rate:
            audio = np.asarray(
                soxr.resample(audio, sample_rate, self._rate, quality="HQ"),
                dtype=np.float32,
            )
        clipwise, framewise = self._at.inference(audio[None, :])
        clipwise = np.asarray(clipwise).reshape(-1)
        framewise = np.asarray(framewise)  # (T, classes)
        top = np.argsort(-clipwise)[:5]

        def pattern_scores(
            patterns: list[str],
        ) -> dict[str, dict[str, Any]]:
            scores: dict[str, dict[str, Any]] = {}
            for pattern in patterns:
                indices = [
                    i
                    for i, name in enumerate(self._labels)
                    if pattern in name.lower()
                ]
                if not indices:
                    continue
                column = framewise[:, indices].max(axis=1)
                scores[pattern] = {
                    "max": float(column.max()) if column.size else 0.0,
                    "at_seconds": (
                        round(float(np.argmax(column)) * self._frame_seconds, 3)
                        if column.size
                        else None
                    ),
                    "matched_labels": [self._labels[i] for i in indices],
                }
            return scores

        risk_scores = pattern_scores(self._risk_patterns)
        noise_scores = pattern_scores(self._noise_patterns)
        speech_idx = [
            i for i, name in enumerate(self._labels) if name.lower() == "speech"
        ]
        speech = framewise[:, speech_idx[0]] if speech_idx else None
        return {
            "status": "ok",
            "top_labels": [
                {"label": self._labels[i], "score": round(float(clipwise[i]), 4)}
                for i in top
            ],
            "risk_scores": risk_scores,
            "noise_scores": noise_scores,
            "speech_score_median": (
                float(np.median(speech)) if speech is not None else None
            ),
            "speech_score_max": (
                float(speech.max()) if speech is not None else None
            ),
        }


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


def style_event_completed(
    work_root: Path, source_sha256: str, config: StyleEventConfig
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
        == LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION
        and state.get("source_sha256") == source_sha256
    )


def analyze_source_regions(
    source_path: str | Path,
    regions: list[dict[str, Any]],
    *,
    config: StyleEventConfig,
    anchors: StyleAnchors,
    event_backend: EventBackend,
) -> dict[str, Any]:
    """Emit per-region style + event + spatial evidence. Nothing here returns
    a pass verdict — ``evidence_missing`` when any required signal is absent.
    """
    source = Path(source_path).resolve()
    source_hash = file_sha256(source)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    source_dir = _source_dir(work_root, source_hash)
    source_dir.mkdir(parents=True, exist_ok=True)
    state_path = source_dir / "state.json"

    if style_event_completed(work_root, source_hash, config):
        return {
            "status": "cached",
            "source_sha256": source_hash,
            "evidence_path": str(source_dir / "style_event.json"),
        }

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION,
            "started_at": _now(),
        },
    )
    try:
        backend_status = event_backend.status()
        records: list[dict[str, Any]] = []
        for region in sorted(
            regions, key=lambda row: int(row["source_start_frame"])
        ):
            start = int(region["source_start_frame"])
            end = int(region["source_end_frame"])
            samples, sample_rate = read_source_segment(
                source, start_frame=start, end_frame=end
            )
            features = region_style_features(
                samples, sample_rate=sample_rate,
                transient_hop=config.events.transient_hop_samples,
            )
            style = score_style_evidence(features, anchors)
            # Event tagging needs ≥min_tag_seconds for the CNN receptive
            # field; short regions borrow real context from both sides and
            # record the expanded window instead of padding silence.
            min_frames = round(config.events.min_tag_seconds * sample_rate)
            context_padded = False
            event_start, event_end = start, end
            if end - start < min_frames:
                probe = sf.info(str(source))
                deficit = min_frames - (end - start)
                event_start = max(0, start - deficit // 2)
                event_end = min(int(probe.frames), end + (deficit - (start - event_start)))
                event_start = max(0, event_start - (min_frames - (event_end - event_start)))
                context_padded = True
            if event_end - event_start != end - start:
                event_samples, _ = read_source_segment(
                    source, start_frame=event_start, end_frame=event_end
                )
                mono = (
                    event_samples.mean(axis=1)
                    if event_samples.ndim == 2
                    else event_samples
                ).astype(np.float32)
            else:
                mono = (
                    samples.mean(axis=1)
                    if samples.ndim == 2
                    else samples
                ).astype(np.float32)
            event = event_backend.tag(mono, sample_rate=sample_rate)
            event["event_window_start_frame"] = event_start
            event["event_window_end_frame"] = event_end
            event["context_padded"] = context_padded
            record = {
                "region_index": int(region["region_index"]),
                "source_start_frame": start,
                "source_end_frame": end,
                "prefilter_status": region.get("status", "usable"),
                "prefilter_reasons": region.get("reasons", []),
                "prefilter_spatial_risk_fraction": region.get(
                    "spatial_risk_fraction"
                ),
                "prefilter_transients_per_second": region.get(
                    "transients_per_second"
                ),
                "style_features": {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in features.items()
                },
                "style_evidence": style,
                "event_backend": backend_status,
                "event_evidence": event,
                "status": (
                    "evidence_complete"
                    if event.get("status") == "ok"
                    else "evidence_missing"
                ),
            }
            records.append(record)

        manifest = {
            "schema_version": 1,
            "implementation_version": LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION,
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "reference_pack_sha256": anchors.pack_sha256,
            "event_backend": backend_status,
            "regions": records,
        }
        _atomic_write_json(source_dir / "style_event.json", manifest)
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "completed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION,
                "region_count": len(records),
                "finished_at": _now(),
            },
        )
        return {
            "status": "completed",
            "source_sha256": source_hash,
            "region_count": len(records),
            "evidence_path": str(source_dir / "style_event.json"),
        }
    except Exception as error:
        _atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "implementation_version": LONG_FORM_STYLE_EVENT_IMPLEMENTATION_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": _now(),
            },
        )
        raise


def run_style_event(
    jobs: list[tuple[str | Path, list[dict[str, Any]]]],
    *,
    config: StyleEventConfig,
    event_backend: EventBackend | None = None,
) -> dict[str, Any]:
    if event_backend is None:
        if config.events.backend == "panns_cnn14":
            event_backend = PannsEventBackend(config.events)
        else:
            event_backend = NoEventBackend()
    anchors = load_style_anchors(config)
    work_root = validate_output_path(ROOT / config.paths.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    for partial in work_root.rglob("*.partial"):
        partial.unlink()
    results = [
        analyze_source_regions(
            source,
            regions,
            config=config,
            anchors=anchors,
            event_backend=event_backend,
        )
        for source, regions in jobs
    ]
    return {
        "status": "completed",
        "config_sha256": config.config_sha256(),
        "event_backend": event_backend.status(),
        "sources": results,
    }
