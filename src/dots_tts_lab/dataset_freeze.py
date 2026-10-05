from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_DATASET_FREEZE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "datasets"
    / "fuxuan_v1.yaml"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetInputs(_StrictFrozenModel):
    standardization_config_id: str = Field(min_length=1)
    standardization_config_version: int = Field(ge=1)
    standardization_config_path: str = Field(min_length=1)
    quality_analysis_id: str = Field(min_length=1)
    quality_analysis_version: int = Field(ge=1)
    quality_policy_id: str = Field(min_length=1)
    quality_policy_version: int = Field(ge=1)
    quality_report_path: str = Field(min_length=1)
    review_benchmark_id: str = Field(min_length=1)
    review_benchmark_version: int = Field(ge=1)


class DatasetEligibility(_StrictFrozenModel):
    allow_quality_decisions: list[Literal["pass"]]
    quality_review_requires_latest_status: Literal["approved"]
    exclude_latest_review_statuses: list[Literal["rejected", "pending"]]
    allow_unreviewed_filename_text: bool
    require_nonempty_text: bool
    require_available_source: bool
    require_verified_standardized_hash: bool

    @model_validator(mode="after")
    def validate_fail_closed_policy(self) -> "DatasetEligibility":
        if self.allow_quality_decisions != ["pass"]:
            raise ValueError("allow_quality_decisions must be exactly ['pass']")
        if set(self.exclude_latest_review_statuses) != {"rejected", "pending"}:
            raise ValueError(
                "exclude_latest_review_statuses must contain rejected and pending"
            )
        if not (
            self.require_nonempty_text
            and self.require_available_source
            and self.require_verified_standardized_hash
        ):
            raise ValueError("dataset v1 eligibility integrity checks cannot be disabled")
        return self


class TextSimilarityConfig(_StrictFrozenModel):
    unicode_form: Literal["NFKC"]
    casefold: bool
    remove_unicode_category_prefixes: list[Literal["P", "Z", "C"]]
    candidate_length_ratio_min: float = Field(gt=0.0, le=1.0)
    near_similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_required: bool


class AcousticFingerprintConfig(_StrictFrozenModel):
    sample_rate: int = Field(ge=8000)
    trim_top_db: float = Field(gt=0.0)
    rms_target_dbfs: float = Field(lt=0.0)
    n_fft: int = Field(ge=64)
    window_length: int = Field(ge=1)
    hop_length: int = Field(ge=1)
    n_mels: int = Field(ge=8)
    pooled_frames: int = Field(ge=8)
    duration_ratio_min: float = Field(gt=0.0, le=1.0)
    near_cosine_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    storage_dtype: Literal["float32_le"]
    calibration_required: bool

    @model_validator(mode="after")
    def validate_frame_parameters(self) -> "AcousticFingerprintConfig":
        if self.window_length > self.n_fft:
            raise ValueError("window_length cannot exceed n_fft")
        if self.hop_length > self.window_length:
            raise ValueError("hop_length cannot exceed window_length")
        return self


class SpeakerEmbeddingConfig(_StrictFrozenModel):
    model_family: Literal["campplus"]
    model_id: Literal["dots-studio/dots.tts-mf-1step"]
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weights_path: str = Field(min_length=1)
    weights_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_config_path: str = Field(min_length=1)
    model_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    license: Literal["Apache-2.0"]
    input_sample_rate: Literal[48000]
    embedding_size: Literal[512]
    max_audio_seconds: Literal[0.0]
    inference_device: Literal["cpu"]
    inference_dtype: Literal["float32"]
    l2_normalize: Literal[True]
    knn_k: int = Field(ge=1)
    center_cosine_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    knn_cosine_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    calibration_required: bool


class SplitConfig(_StrictFrozenModel):
    train_ratio: float = Field(gt=0.0, lt=1.0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    seed: int = Field(ge=0)
    strata_field: Literal["emotion_primary"]
    group_evidence: list[
        Literal[
            "exact_audio",
            "exact_text",
            "reviewed_near_audio",
            "reviewed_near_text",
        ]
    ]
    low_resource_label: str = Field(min_length=1)
    low_resource_min_groups_per_eval_split: int = Field(ge=1)
    balance_fields: list[
        Literal["item_count", "emotion_primary", "duration_seconds", "group_count"]
    ]

    @model_validator(mode="after")
    def validate_split_contract(self) -> "SplitConfig":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-12:
            raise ValueError("split ratios must sum to 1.0")
        if len(set(self.group_evidence)) != len(self.group_evidence):
            raise ValueError("group_evidence cannot contain duplicates")
        if set(self.balance_fields) != {
            "item_count",
            "emotion_primary",
            "duration_seconds",
            "group_count",
        }:
            raise ValueError("balance_fields must contain the complete v1 objective")
        return self


class DatasetOutputConfig(_StrictFrozenModel):
    dataset_root: str = Field(min_length=1)
    canonical_parquet: str = Field(min_length=1)
    manifest_json: str = Field(min_length=1)
    stats_json: str = Field(min_length=1)
    train_jsonl: str = Field(min_length=1)
    validation_jsonl: str = Field(min_length=1)
    test_jsonl: str = Field(min_length=1)
    frozen_config: str = Field(min_length=1)
    checksums: str = Field(min_length=1)
    trainer_jsonl_fields: list[str]
    trainer_audio_paths: Literal["absolute"]

    @field_validator(
        "dataset_root",
        "canonical_parquet",
        "manifest_json",
        "stats_json",
        "train_jsonl",
        "validation_jsonl",
        "test_jsonl",
        "frozen_config",
        "checksums",
    )
    @classmethod
    def validate_relative_posix_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("dataset output paths must be relative POSIX paths")
        return value

    @field_validator("trainer_jsonl_fields")
    @classmethod
    def validate_trainer_fields(cls, value: list[str]) -> list[str]:
        if value != ["fid", "audio", "text"]:
            raise ValueError("trainer_jsonl_fields must be exactly fid, audio, text")
        return value


class DatasetFreezeConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    dataset_version: int = Field(ge=1)
    implementation_version: Literal[1]
    inputs: DatasetInputs
    eligibility: DatasetEligibility
    text_similarity: TextSimilarityConfig
    acoustic_fingerprint: AcousticFingerprintConfig
    speaker_embedding: SpeakerEmbeddingConfig
    split: SplitConfig
    output: DatasetOutputConfig

    @model_validator(mode="after")
    def validate_calibration_state(self) -> "DatasetFreezeConfig":
        fields = (
            (
                self.text_similarity.calibration_required,
                self.text_similarity.near_similarity_threshold,
                "text_similarity",
            ),
            (
                self.acoustic_fingerprint.calibration_required,
                self.acoustic_fingerprint.near_cosine_threshold,
                "acoustic_fingerprint",
            ),
            (
                self.speaker_embedding.calibration_required,
                self.speaker_embedding.center_cosine_threshold,
                "speaker center",
            ),
            (
                self.speaker_embedding.calibration_required,
                self.speaker_embedding.knn_cosine_threshold,
                "speaker knn",
            ),
        )
        for calibration_required, threshold, name in fields:
            if calibration_required == (threshold is not None):
                raise ValueError(
                    f"{name} must have null threshold exactly while calibration_required is true"
                )
        return self

    def assert_freeze_ready(self) -> None:
        pending = []
        if self.text_similarity.calibration_required:
            pending.append("text_similarity")
        if self.acoustic_fingerprint.calibration_required:
            pending.append("acoustic_fingerprint")
        if self.speaker_embedding.calibration_required:
            pending.append("speaker_embedding")
        if pending:
            raise RuntimeError(
                "Dataset config is audit-only until thresholds are calibrated: "
                + ", ".join(pending)
            )


def load_dataset_freeze_config(
    path: str | Path = DEFAULT_DATASET_FREEZE_CONFIG_PATH,
) -> DatasetFreezeConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset freeze config must be a YAML mapping: {config_path}")
    return DatasetFreezeConfig.model_validate(payload, strict=True)


def canonical_dataset_config(config: DatasetFreezeConfig) -> dict[str, str]:
    canonical_json = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "canonical_json": canonical_json,
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(row: dict[str, Any], field: str, *, asset_sha256: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing {field} for dataset asset {asset_sha256}")
    return value


def _parse_quality_reasons(value: Any, *, asset_sha256: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid quality_reasons_json for dataset asset {asset_sha256}"
        ) from error
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise RuntimeError(
            f"quality_reasons_json must be a list of objects for {asset_sha256}"
        )
    return parsed


def resolve_dataset_freeze_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    config: DatasetFreezeConfig,
    standardized_root: str | Path = "data/work/standardized",
) -> dict[str, list[dict[str, Any]]]:
    """Resolve final text/labels and verify files before analysis can begin."""
    root = Path(standardized_root).resolve()
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for row in rows:
        asset_sha256 = _require_string(row, "asset_sha256", asset_sha256="<unknown>")
        if len(asset_sha256) != 64 or any(c not in "0123456789abcdef" for c in asset_sha256):
            raise RuntimeError(f"Invalid asset SHA-256 in dataset candidates: {asset_sha256}")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate dataset candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)

        required_fields = (
            "raw_relative_path",
            "source_root_key",
            "source_path_key",
            "source_relative_path",
            "speaker_id",
            "emotion_weak_label",
            "parser_profile_id",
            "parser_config_sha256",
            "derived_id",
            "derived_relative_path",
            "derived_output_sha256",
            "standardization_config_sha256",
            "quality_run_id",
            "quality_analysis_id",
            "quality_analysis_config_sha256",
            "quality_policy_id",
            "quality_policy_config_sha256",
            "quality_decision",
        )
        for field in required_fields:
            _require_string(row, field, asset_sha256=asset_sha256)
        for field in (
            "parser_profile_version",
            "standardization_implementation_version",
            "quality_analysis_version",
            "quality_implementation_version",
            "quality_policy_version",
        ):
            if not isinstance(row.get(field), int):
                raise RuntimeError(f"Missing {field} for dataset asset {asset_sha256}")

        if row.get("source_availability_status") != "available":
            raise RuntimeError(f"Dataset source is not available for {asset_sha256}")
        relative_audio = PurePosixPath(str(row["derived_relative_path"]).replace("\\", "/"))
        if relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise RuntimeError(f"Unsafe standardized path for {asset_sha256}: {relative_audio}")
        audio_path = (root / Path(*relative_audio.parts)).resolve()
        if not audio_path.is_relative_to(root):
            raise RuntimeError(f"Standardized path escapes root for {asset_sha256}")
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Standardized audio missing for dataset asset {asset_sha256}: {audio_path}"
            )
        expected_audio_sha256 = str(row["derived_output_sha256"])
        actual_audio_sha256 = _sha256_file(audio_path)
        if actual_audio_sha256 != expected_audio_sha256:
            raise RuntimeError(
                "Standardized audio SHA-256 drift for dataset asset "
                f"{asset_sha256}: expected {expected_audio_sha256}, got {actual_audio_sha256}"
            )

        quality_decision = str(row["quality_decision"])
        quality_reasons = _parse_quality_reasons(
            row.get("quality_reasons_json"), asset_sha256=asset_sha256
        )
        has_review = row.get("review_decision_id") is not None
        review_status = row.get("review_status") if has_review else None
        if has_review and review_status in config.eligibility.exclude_latest_review_statuses:
            excluded.append(
                {
                    "asset_sha256": asset_sha256,
                    "reason": f"review_{review_status}",
                }
            )
            continue
        if quality_decision == "reject":
            excluded.append({"asset_sha256": asset_sha256, "reason": "quality_reject"})
            continue
        if quality_decision == "review" and review_status != (
            config.eligibility.quality_review_requires_latest_status
        ):
            excluded.append(
                {
                    "asset_sha256": asset_sha256,
                    "reason": "quality_review_without_latest_approval",
                }
            )
            continue
        if quality_decision not in {*config.eligibility.allow_quality_decisions, "review", "reject"}:
            raise RuntimeError(
                f"Unknown quality decision for dataset asset {asset_sha256}: {quality_decision}"
            )

        if has_review:
            if review_status != "approved":
                raise RuntimeError(
                    f"Invalid latest review status for dataset asset {asset_sha256}: {review_status}"
                )
            text_decision = row.get("review_text_decision")
            if text_decision not in {"accept_reference", "accept_edited"}:
                raise RuntimeError(
                    f"Approved dataset review has invalid text decision for {asset_sha256}"
                )
            text_exact = _require_string(
                row, "review_text_final", asset_sha256=asset_sha256
            ).strip()
            emotion_primary = _require_string(
                row, "review_emotion_primary", asset_sha256=asset_sha256
            )
            label_source = row.get("review_label_source")
            if label_source not in {"weak_label_confirmed", "human_corrected"}:
                raise RuntimeError(
                    f"Invalid human label_source for dataset asset {asset_sha256}"
                )
            review_round = row.get("review_round")
            if not isinstance(review_round, int) or review_round < 1:
                raise RuntimeError(f"Invalid review_round for dataset asset {asset_sha256}")
            review_batch_id = _require_string(
                row, "review_export_batch_id", asset_sha256=asset_sha256
            )
            text_source = "human_review"
        else:
            if not config.eligibility.allow_unreviewed_filename_text:
                excluded.append(
                    {"asset_sha256": asset_sha256, "reason": "unreviewed_text_not_allowed"}
                )
                continue
            text_exact = _require_string(
                row, "transcript_candidate", asset_sha256=asset_sha256
            ).strip()
            emotion_primary = str(row["emotion_weak_label"])
            label_source = "weak_label_unreviewed"
            review_round = None
            review_batch_id = None
            text_source = "filename_candidate_unreviewed"

        eligible.append(
            {
                "fid": asset_sha256,
                "asset_sha256": asset_sha256,
                "raw_relative_path": str(row["raw_relative_path"]),
                "source_root_key": str(row["source_root_key"]),
                "source_path_key": str(row["source_path_key"]),
                "source_relative_path": str(row["source_relative_path"]),
                "speaker_id": str(row["speaker_id"]),
                "derived_id": str(row["derived_id"]),
                "audio_relative_path": relative_audio.as_posix(),
                "audio_absolute_path": str(audio_path),
                "audio_sha256": actual_audio_sha256,
                "duration_seconds": float(row["derived_duration_seconds"]),
                "quality_run_id": str(row["quality_run_id"]),
                "quality_decision": quality_decision,
                "quality_reasons": quality_reasons,
                "text_exact": text_exact,
                "text_sha256": hashlib.sha256(text_exact.encode("utf-8")).hexdigest(),
                "text_source": text_source,
                "emotion_weak_label": str(row["emotion_weak_label"]),
                "emotion_primary": emotion_primary,
                "emotion_secondary": row.get("review_emotion_secondary") if has_review else None,
                "intensity": row.get("review_intensity") if has_review else None,
                "label_source": label_source,
                "review_decision_id": str(row["review_decision_id"]) if has_review else None,
                "review_round": review_round,
                "review_batch_id": review_batch_id,
                "lineage": {
                    "metadata_profile_id": str(row["parser_profile_id"]),
                    "metadata_profile_version": int(row["parser_profile_version"]),
                    "metadata_config_sha256": str(row["parser_config_sha256"]),
                    "standardization_config_sha256": str(
                        row["standardization_config_sha256"]
                    ),
                    "standardization_implementation_version": int(
                        row["standardization_implementation_version"]
                    ),
                    "quality_analysis_id": str(row["quality_analysis_id"]),
                    "quality_analysis_version": int(row["quality_analysis_version"]),
                    "quality_analysis_config_sha256": str(
                        row["quality_analysis_config_sha256"]
                    ),
                    "quality_implementation_version": int(
                        row["quality_implementation_version"]
                    ),
                    "quality_policy_id": str(row["quality_policy_id"]),
                    "quality_policy_version": int(row["quality_policy_version"]),
                    "quality_policy_config_sha256": str(
                        row["quality_policy_config_sha256"]
                    ),
                    "review_benchmark_id": config.inputs.review_benchmark_id
                    if has_review
                    else None,
                    "review_benchmark_version": config.inputs.review_benchmark_version
                    if has_review
                    else None,
                },
            }
        )
    eligible.sort(key=lambda item: item["asset_sha256"])
    excluded.sort(key=lambda item: item["asset_sha256"])
    return {"eligible": eligible, "excluded": excluded}


_CANDIDATE_SNAPSHOT_FIELDS = (
    "fid",
    "asset_sha256",
    "raw_relative_path",
    "source_root_key",
    "source_path_key",
    "source_relative_path",
    "speaker_id",
    "derived_id",
    "audio_relative_path",
    "audio_sha256",
    "duration_seconds",
    "quality_run_id",
    "quality_decision",
    "quality_reasons",
    "text_exact",
    "text_sha256",
    "text_source",
    "emotion_weak_label",
    "emotion_primary",
    "emotion_secondary",
    "intensity",
    "label_source",
    "review_decision_id",
    "review_round",
    "review_batch_id",
    "lineage",
)


def build_candidate_snapshot(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    items = []
    seen: set[str] = set()
    for candidate in candidates:
        asset_sha256 = _require_string(
            candidate, "asset_sha256", asset_sha256="<unknown>"
        )
        if asset_sha256 in seen:
            raise RuntimeError(f"Duplicate asset in candidate snapshot: {asset_sha256}")
        seen.add(asset_sha256)
        missing = [field for field in _CANDIDATE_SNAPSHOT_FIELDS if field not in candidate]
        if missing:
            raise RuntimeError(
                f"Candidate snapshot item {asset_sha256} is missing fields: {missing}"
            )
        items.append({field: candidate[field] for field in _CANDIDATE_SNAPSHOT_FIELDS})
    items.sort(key=lambda item: item["asset_sha256"])
    payload = {
        "schema_version": 1,
        "item_count": len(items),
        "total_duration_seconds": sum(float(item["duration_seconds"]) for item in items),
        "items": items,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "payload": payload,
        "canonical_json": canonical_json,
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def write_candidate_snapshot(
    snapshot: dict[str, Any], path: str | Path
) -> dict[str, Any]:
    target = Path(path).resolve()
    expected = str(snapshot["canonical_json"]).encode("utf-8")
    expected_sha256 = str(snapshot["sha256"])
    if hashlib.sha256(expected).hexdigest() != expected_sha256:
        raise ValueError("Candidate snapshot canonical JSON does not match its SHA-256")
    if target.exists():
        actual = target.read_bytes()
        if actual != expected:
            raise RuntimeError(
                f"Candidate snapshot content changed without a version bump: {target}"
            )
        return {"path": str(target), "sha256": expected_sha256, "action": "cached"}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    if _sha256_file(target) != expected_sha256:
        raise RuntimeError(f"Candidate snapshot verification failed after write: {target}")
    return {"path": str(target), "sha256": expected_sha256, "action": "written"}
