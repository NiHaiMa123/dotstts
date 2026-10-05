from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.dataset_freeze import SpeakerEmbeddingConfig


DEFAULT_LONG_FORM_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "long_form"
    / "training_source_v1.yaml"
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
StyleLabel = Literal[
    "normal",
    "soft",
    "whisper",
    "roleplay",
    "binaural_3d",
    "singing",
    "non_speech",
    "unknown",
]
ReviewAction = Literal["review"]
RunStatus = Literal[
    "pending",
    "running",
    "interrupted",
    "awaiting_review",
    "awaiting_asr",
    "completed",
    "failed",
]
SegmentStatus = Literal[
    "candidate",
    "recommended_keep",
    "review_required",
    "approved",
    "rejected",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class LongFormPaths(_StrictFrozenModel):
    input_root: str
    work_root: str
    report_root: str

    @field_validator("input_root", "work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "LongFormPaths":
        input_path = PurePosixPath(self.input_root)
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == input_path or input_path in output_path.parents:
                raise ValueError("long-form outputs cannot be written under input_root")
        return self


class DecodeConfig(_StrictFrozenModel):
    block_seconds: float = Field(ge=1.0, le=120.0)
    analysis_sample_rate_hz: int = Field(ge=8000, le=48000)
    output_sample_rate_hz: Literal[48000]
    output_subtype: Literal["PCM_24"]
    max_parallel_sources: int = Field(ge=1, le=4)


class SegmentationConfig(_StrictFrozenModel):
    frame_ms: float = Field(ge=10.0, le=100.0)
    hop_ms: float = Field(ge=5.0, le=100.0)
    minimum_seconds: float = Field(ge=0.5)
    preferred_maximum_seconds: float = Field(gt=0.5)
    hard_maximum_seconds: float = Field(gt=0.5, le=15.0)
    merge_silence_seconds: float = Field(ge=0.0, le=2.0)
    boundary_padding_seconds: float = Field(ge=0.0, le=0.5)
    minimum_speech_ratio: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SegmentationConfig":
        if self.hop_ms > self.frame_ms:
            raise ValueError("segmentation hop_ms cannot exceed frame_ms")
        if not (
            self.minimum_seconds
            <= self.preferred_maximum_seconds
            <= self.hard_maximum_seconds
        ):
            raise ValueError(
                "segment duration must satisfy minimum <= preferred <= hard maximum"
            )
        return self


class ScreeningConfig(_StrictFrozenModel):
    absolute_activity_floor_dbfs: float = Field(ge=-120.0, le=-20.0)
    activity_start_margin_db: float = Field(ge=0.0, le=40.0)
    activity_continue_margin_db: float = Field(ge=0.0, le=40.0)
    noise_floor_percentile: float = Field(ge=0.0, le=50.0)
    maximum_activity_threshold_dbfs: float = Field(ge=-60.0, le=-10.0)
    minimum_snr_proxy_db: float = Field(ge=0.0, le=60.0)
    maximum_silence_ratio: float = Field(ge=0.0, le=1.0)
    near_clip_level_dbfs: float = Field(ge=-12.0, le=0.0)
    maximum_near_clip_ratio: float = Field(ge=0.0, le=1.0)
    maximum_channel_level_difference_db: float = Field(ge=0.0, le=60.0)
    minimum_channel_correlation_for_safe_mean: float = Field(ge=-1.0, le=1.0)
    spatial_window_seconds: float = Field(ge=0.05, le=2.0)
    maximum_pan_standard_deviation: float = Field(ge=0.0, le=1.0)
    maximum_side_to_mid_db: float = Field(ge=-60.0, le=20.0)
    overlap_proxy_review_score: float = Field(ge=0.0, le=1.0)
    uncertain_signal_action: ReviewAction
    suspected_overlap_action: ReviewAction
    suspected_noise_or_music_action: ReviewAction
    suspected_spatial_audio_action: ReviewAction

    @model_validator(mode="after")
    def validate_vad_hysteresis(self) -> "ScreeningConfig":
        if self.activity_continue_margin_db > self.activity_start_margin_db:
            raise ValueError("activity continue margin cannot exceed start margin")
        return self


class SpeakerPolicy(_StrictFrozenModel):
    require_reference_for_auto_keep: Literal[True]
    mismatch_action: ReviewAction
    uncertain_action: ReviewAction
    cosine_link_threshold: float = Field(ge=-1.0, le=1.0)
    minimum_cluster_size: int = Field(ge=2)
    knn_k: int = Field(ge=1)
    target_reference_minimum_cosine: float = Field(ge=-1.0, le=1.0)
    encoder: SpeakerEmbeddingConfig


class StylePolicy(_StrictFrozenModel):
    accepted_styles: list[StyleLabel]
    review_styles: list[StyleLabel]
    excluded_styles: list[StyleLabel]
    maximum_groups: int = Field(ge=1, le=12)
    minimum_group_size: int = Field(ge=2)
    soft_below_corpus_median_db: float = Field(ge=0.0, le=30.0)
    whisper_maximum_periodicity: float = Field(ge=0.0, le=1.0)
    whisper_minimum_spectral_flatness: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_partition(self) -> "StylePolicy":
        accepted = set(self.accepted_styles)
        review = set(self.review_styles)
        excluded = set(self.excluded_styles)
        all_styles = {
            "normal",
            "soft",
            "whisper",
            "roleplay",
            "binaural_3d",
            "singing",
            "non_speech",
            "unknown",
        }
        if accepted != {"normal"}:
            raise ValueError("v1 may auto-accept only the normal style")
        if accepted & review or accepted & excluded or review & excluded:
            raise ValueError("style policy groups must not overlap")
        if accepted | review | excluded != all_styles:
            raise ValueError("style policy must classify every supported style")
        return self


class TextPolicy(_StrictFrozenModel):
    asr_is_candidate_only: Literal[True]
    require_human_confirmation_for_training: Literal[True]
    use_timestamp_backend_for_boundaries_only: Literal[True]
    timestamp_backend_config: str = Field(min_length=1)
    timestamp_python: str = Field(min_length=1)
    model_root: str = Field(min_length=1)
    preferred_asr_boundary_seconds: float = Field(ge=2.0, le=15.0)


class RecoveryPolicy(_StrictFrozenModel):
    atomic_segment_writes: Literal[True]
    resume_completed_sources: Literal[True]
    mark_running_as_interrupted_on_startup: Literal[True]
    delete_partial_files_on_startup: Literal[True]


class LongFormConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    paths: LongFormPaths
    decode: DecodeConfig
    segmentation: SegmentationConfig
    screening: ScreeningConfig
    speaker: SpeakerPolicy
    style: StylePolicy
    text: TextPolicy
    recovery: RecoveryPolicy

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class LongFormSourceState(_StrictFrozenModel):
    schema_version: Literal[1]
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_relative_path: str = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    status: RunStatus
    started_at: str
    updated_at: str
    segment_count: int = Field(ge=0)
    error_type: str | None = None
    error_message: str | None = None


class LongFormSegmentRecord(_StrictFrozenModel):
    schema_version: Literal[1]
    segment_id: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0, le=15.0)
    status: SegmentStatus
    style: StyleLabel
    review_reasons: list[str]
    asr_candidate_text: str | None = None
    human_confirmed_text: str | None = None

    @model_validator(mode="after")
    def validate_frame_range(self) -> "LongFormSegmentRecord":
        if self.source_end_frame <= self.source_start_frame:
            raise ValueError("source_end_frame must exceed source_start_frame")
        expected = (
            self.source_end_frame - self.source_start_frame
        ) / self.source_sample_rate_hz
        if abs(expected - self.duration_seconds) > 1.0 / self.source_sample_rate_hz:
            raise ValueError("duration_seconds does not match the source frame range")
        if self.status == "approved" and not self.human_confirmed_text:
            raise ValueError("approved training segments require human_confirmed_text")
        return self


def load_long_form_config(
    path: str | Path = DEFAULT_LONG_FORM_CONFIG_PATH,
) -> LongFormConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Long-form configuration must be a YAML mapping: {config_path}")
    return LongFormConfig.model_validate(payload, strict=True)


def segment_identity(
    *,
    source_sha256: str,
    source_start_frame: int,
    source_end_frame: int,
    config_sha256: str,
    implementation_version: int,
) -> str:
    payload = json.dumps(
        {
            "config_sha256": config_sha256,
            "implementation_version": implementation_version,
            "source_end_frame": source_end_frame,
            "source_sha256": source_sha256,
            "source_start_frame": source_start_frame,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
