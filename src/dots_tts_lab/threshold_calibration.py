from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import soundfile as sf
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field

from dots_tts_lab.acoustic_fingerprint import (
    compute_acoustic_fingerprint,
    cosine_similarity,
    load_acoustic_fingerprint_vectors,
)
from dots_tts_lab.dataset_freeze import (
    canonical_dataset_config,
    load_dataset_freeze_config,
)


DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH = Path(
    "configs/lab/datasets/fuxuan_v1_threshold_calibration.yaml"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReviewedTextPair(_StrictModel):
    left_asset_sha256: str = Field(pattern=_SHA256_PATTERN)
    right_asset_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_note: str = Field(min_length=1)


class TextCalibrationConfig(_StrictModel):
    report_path: str = Field(min_length=1)
    report_sha256: str = Field(pattern=_SHA256_PATTERN)
    threshold_rule: Literal["midpoint_min_positive_max_background"]
    positive_pairs: list[ReviewedTextPair] = Field(min_length=2)


class AcousticCalibrationConfig(_StrictModel):
    report_path: str = Field(min_length=1)
    report_sha256: str = Field(pattern=_SHA256_PATTERN)
    samples_per_emotion: int = Field(ge=2)
    gain_factor: float = Field(gt=0.0, lt=1.0)
    leading_silence_seconds: float = Field(gt=0.0)
    trailing_silence_seconds: float = Field(gt=0.0)
    resample_intermediate_rate: int = Field(ge=8000)
    threshold_rule: Literal["midpoint_min_positive_max_background"]


class SpeakerCalibrationConfig(_StrictModel):
    report_path: str = Field(min_length=1)
    report_sha256: str = Field(pattern=_SHA256_PATTERN)
    mad_consistency_scale: float = Field(gt=0.0)
    robust_sigma_cutoff: float = Field(gt=0.0)
    candidate_rule: Literal["center_or_knn_below_robust_lower_bound"]


class ThresholdCalibrationConfig(_StrictModel):
    schema_version: Literal[1]
    calibration_id: str = Field(min_length=1)
    calibration_version: int = Field(ge=1)
    dataset_config_path: str = Field(min_length=1)
    dataset_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_snapshot_path: str = Field(min_length=1)
    candidate_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    text: TextCalibrationConfig
    acoustic: AcousticCalibrationConfig
    speaker: SpeakerCalibrationConfig
    output_path: str = Field(min_length=1)

    def identity(self) -> dict[str, str]:
        canonical_json = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "canonical_json": canonical_json,
            "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        }


class CalibratedThresholds(_StrictModel):
    text_near_similarity: float = Field(ge=0.0, le=1.0)
    acoustic_near_cosine: float = Field(ge=0.0, le=1.0)
    speaker_center_cosine: float = Field(ge=-1.0, le=1.0)
    speaker_knn_cosine: float = Field(ge=-1.0, le=1.0)


class ThresholdCalibrationOverlay(_StrictModel):
    calibration_id: str
    calibration_version: int
    calibration_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_report_path: str
    dataset_id: str
    dataset_version: int
    dataset_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    thresholds: CalibratedThresholds


def load_threshold_calibration_config(
    path: str | Path = DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
) -> ThresholdCalibrationConfig:
    payload = yaml.safe_load(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Threshold calibration config must be a YAML mapping")
    return ThresholdCalibrationConfig.model_validate(payload, strict=True)


def load_threshold_calibration_overlay(
    config_path: str | Path = DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
) -> ThresholdCalibrationOverlay:
    config = load_threshold_calibration_config(config_path)
    dataset_config = load_dataset_freeze_config(config.dataset_config_path)
    dataset_identity = canonical_dataset_config(dataset_config)
    if dataset_identity["sha256"] != config.dataset_config_sha256:
        raise RuntimeError("Dataset config SHA-256 drift before calibration overlay")
    report_path = Path(config.output_path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Calibration report does not exist: {report_path}")
    report_sha256 = _sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "status": "succeeded",
        "calibration_id": config.calibration_id,
        "calibration_version": config.calibration_version,
        "calibration_config_sha256": config.identity()["sha256"],
        "candidate_snapshot_sha256": config.candidate_snapshot_sha256,
        "dataset_id": dataset_config.dataset_id,
        "dataset_version": dataset_config.dataset_version,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise RuntimeError(
                f"Calibration overlay {field} mismatch: "
                f"expected {value!r}, got {report.get(field)!r}"
            )
    thresholds = CalibratedThresholds.model_validate(
        report.get("thresholds"), strict=True
    )
    detailed_thresholds = {
        "text_near_similarity": report["text"]["threshold"],
        "acoustic_near_cosine": report["acoustic"]["threshold"],
        "speaker_center_cosine": report["speaker"]["center_cosine"]["threshold"],
        "speaker_knn_cosine": report["speaker"]["knn_cosine"]["threshold"],
    }
    if thresholds.model_dump() != detailed_thresholds:
        raise RuntimeError("Calibration summary thresholds differ from report details")
    policy = report.get("policy")
    if not isinstance(policy, dict) or policy.get("automatic_exclusion") is not False:
        raise RuntimeError("Calibration overlay must remain review-only")
    return ThresholdCalibrationOverlay(
        calibration_id=config.calibration_id,
        calibration_version=config.calibration_version,
        calibration_config_sha256=config.identity()["sha256"],
        calibration_report_sha256=report_sha256,
        calibration_report_path=str(report_path),
        dataset_id=dataset_config.dataset_id,
        dataset_version=dataset_config.dataset_version,
        dataset_config_sha256=config.dataset_config_sha256,
        candidate_snapshot_sha256=config.candidate_snapshot_sha256,
        thresholds=thresholds,
    )


def assert_dataset_ready_with_calibration(
    dataset_config: Any,
    overlay: ThresholdCalibrationOverlay,
) -> CalibratedThresholds:
    identity = canonical_dataset_config(dataset_config)
    if (
        overlay.dataset_id != dataset_config.dataset_id
        or overlay.dataset_version != dataset_config.dataset_version
        or overlay.dataset_config_sha256 != identity["sha256"]
    ):
        raise RuntimeError("Calibration overlay does not belong to this dataset config")
    return overlay.thresholds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_json(path_value: str, expected_sha256: str) -> tuple[Path, Any]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration input does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Calibration input SHA-256 drift for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values or not np.all(np.isfinite(values)):
        raise RuntimeError("Calibration distribution must contain finite values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "max": float(array.max()),
    }


def separated_midpoint_threshold(
    positive_scores: list[float], background_scores: list[float]
) -> dict[str, float]:
    positive_min = min(positive_scores)
    background_max = max(background_scores)
    if positive_min <= background_max:
        raise RuntimeError(
            "Calibration distributions overlap: "
            f"positive_min={positive_min}, background_max={background_max}"
        )
    return {
        "positive_min": positive_min,
        "background_max": background_max,
        "separation_margin": positive_min - background_max,
        "threshold": (positive_min + background_max) / 2.0,
    }


def robust_lower_threshold(
    values: list[float], *, consistency_scale: float, sigma_cutoff: float
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_sigma = consistency_scale * mad
    return {
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "sigma_cutoff": sigma_cutoff,
        "threshold": median - sigma_cutoff * robust_sigma,
    }


def select_stratified_duration_samples(
    items: list[dict[str, Any]], *, samples_per_emotion: int
) -> list[dict[str, Any]]:
    by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_emotion[str(item["emotion_primary"])].append(item)
    selected = []
    for emotion in sorted(by_emotion):
        ordered = sorted(
            by_emotion[emotion],
            key=lambda item: (float(item["duration_seconds"]), item["asset_sha256"]),
        )
        count = min(samples_per_emotion, len(ordered))
        indices = sorted(
            set(
                np.rint(np.linspace(0, len(ordered) - 1, count))
                .astype(int)
                .tolist()
            )
        )
        selected.extend(ordered[index] for index in indices)
    return selected


def _calibrate_text(
    report: dict[str, Any], config: TextCalibrationConfig
) -> dict[str, Any]:
    by_pair = {
        tuple(sorted((pair["left_asset_sha256"], pair["right_asset_sha256"]))): pair
        for pair in report["pairs"]
    }
    positive_keys = set()
    positives = []
    for reviewed in config.positive_pairs:
        key = tuple(
            sorted((reviewed.left_asset_sha256, reviewed.right_asset_sha256))
        )
        if key in positive_keys:
            raise RuntimeError(f"Duplicate reviewed text calibration pair: {key}")
        positive_keys.add(key)
        pair = by_pair.get(key)
        if pair is None:
            raise RuntimeError(f"Reviewed text calibration pair was not recalled: {key}")
        positives.append({**pair, "review_note": reviewed.review_note})
    positive_scores = [float(pair["similarity"]) for pair in positives]
    background = [
        float(pair["similarity"])
        for key, pair in by_pair.items()
        if key not in positive_keys
    ]
    separation = separated_midpoint_threshold(positive_scores, background)
    threshold = separation["threshold"]
    return {
        "metric": report["metric"],
        "threshold_rule": config.threshold_rule,
        **separation,
        "positive_distribution": distribution_summary(positive_scores),
        "background_distribution": distribution_summary(background),
        "reviewed_positive_pairs": positives,
        "candidate_pair_count_at_threshold": sum(
            float(pair["similarity"]) >= threshold for pair in report["pairs"]
        ),
    }


def _calibrate_acoustic(
    *,
    snapshot: dict[str, Any],
    report: dict[str, Any],
    dataset_config: Any,
    config: AcousticCalibrationConfig,
) -> dict[str, Any]:
    items = snapshot["items"]
    vectors = load_acoustic_fingerprint_vectors(report)
    selected = select_stratified_duration_samples(
        items, samples_per_emotion=config.samples_per_emotion
    )
    standardized_root = Path("data/work/standardized").resolve()
    positive_records = []
    for item in selected:
        relative_path = PurePosixPath(item["audio_relative_path"])
        audio_path = (standardized_root / Path(*relative_path.parts)).resolve()
        if _sha256_file(audio_path) != item["audio_sha256"]:
            raise RuntimeError(f"Calibration audio SHA-256 drift: {audio_path}")
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        variants = {
            "gain": audio * config.gain_factor,
            "edge_silence": np.pad(
                audio,
                (
                    (
                        round(sample_rate * config.leading_silence_seconds),
                        round(sample_rate * config.trailing_silence_seconds),
                    ),
                    (0, 0),
                ),
            ),
            "resample_roundtrip": soxr.resample(
                soxr.resample(
                    audio[:, 0],
                    sample_rate,
                    config.resample_intermediate_rate,
                    quality="HQ",
                ),
                config.resample_intermediate_rate,
                sample_rate,
                quality="HQ",
            ),
        }
        for transform, transformed in variants.items():
            variant_vector = compute_acoustic_fingerprint(
                transformed,
                sample_rate=sample_rate,
                config=dataset_config.acoustic_fingerprint,
            )
            positive_records.append(
                {
                    "asset_sha256": item["asset_sha256"],
                    "emotion_primary": item["emotion_primary"],
                    "transform": transform,
                    "cosine": cosine_similarity(
                        vectors[item["asset_sha256"]], variant_vector
                    ),
                }
            )
    ordered = sorted(items, key=lambda item: item["asset_sha256"])
    matrix = np.stack([vectors[item["asset_sha256"]] for item in ordered]).astype(
        np.float64
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity_matrix = matrix @ matrix.T
    background_records = []
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            duration_ratio = min(
                float(left["duration_seconds"]), float(right["duration_seconds"])
            ) / max(float(left["duration_seconds"]), float(right["duration_seconds"]))
            if duration_ratio < dataset_config.acoustic_fingerprint.duration_ratio_min:
                continue
            background_records.append(
                {
                    "left_asset_sha256": left["asset_sha256"],
                    "right_asset_sha256": right["asset_sha256"],
                    "duration_ratio": duration_ratio,
                    "cosine": float(similarity_matrix[left_index, right_index]),
                }
            )
    positive_scores = [record["cosine"] for record in positive_records]
    background_scores = [record["cosine"] for record in background_records]
    separation = separated_midpoint_threshold(positive_scores, background_scores)
    threshold = separation["threshold"]
    background_records.sort(
        key=lambda record: (
            -record["cosine"],
            record["left_asset_sha256"],
            record["right_asset_sha256"],
        )
    )
    return {
        "metric": "fingerprint_cosine",
        "threshold_rule": config.threshold_rule,
        **separation,
        "sample_selection": {
            "method": "per_emotion_even_duration_quantiles",
            "requested_per_emotion": config.samples_per_emotion,
            "selected_count": len(selected),
            "emotion_counts": dict(
                sorted(Counter(item["emotion_primary"] for item in selected).items())
            ),
            "asset_sha256s": [item["asset_sha256"] for item in selected],
        },
        "transforms": config.model_dump(
            mode="json",
            exclude={"report_path", "report_sha256", "threshold_rule"},
        ),
        "positive_distribution": distribution_summary(positive_scores),
        "positive_by_transform": {
            transform: distribution_summary(
                [
                    record["cosine"]
                    for record in positive_records
                    if record["transform"] == transform
                ]
            )
            for transform in ("gain", "edge_silence", "resample_roundtrip")
        },
        "positive_records": positive_records,
        "background_distribution": distribution_summary(background_scores),
        "candidate_pair_count_at_threshold": sum(
            record["cosine"] >= threshold for record in background_records
        ),
        "top_background_pairs": background_records[:20],
    }


def _calibrate_speaker(
    report: dict[str, Any], config: SpeakerCalibrationConfig
) -> dict[str, Any]:
    assessments = report["assessments"]
    center_values = [float(item["center_cosine"]) for item in assessments]
    knn_values = [float(item["knn_cosine"]) for item in assessments]
    center = robust_lower_threshold(
        center_values,
        consistency_scale=config.mad_consistency_scale,
        sigma_cutoff=config.robust_sigma_cutoff,
    )
    knn = robust_lower_threshold(
        knn_values,
        consistency_scale=config.mad_consistency_scale,
        sigma_cutoff=config.robust_sigma_cutoff,
    )
    candidates = [
        item
        for item in assessments
        if float(item["center_cosine"]) < center["threshold"]
        or float(item["knn_cosine"]) < knn["threshold"]
    ]
    candidates.sort(
        key=lambda item: (-float(item["outlier_score"]), item["asset_sha256"])
    )
    return {
        "threshold_rule": config.candidate_rule,
        "mad_consistency_scale": config.mad_consistency_scale,
        "robust_sigma_cutoff": config.robust_sigma_cutoff,
        "center_cosine": {
            **center,
            "distribution": distribution_summary(center_values),
        },
        "knn_cosine": {
            **knn,
            "distribution": distribution_summary(knn_values),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def run_threshold_calibration(
    config_path: str | Path = DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = load_threshold_calibration_config(resolved_config_path)
    identity = config.identity()
    dataset_config = load_dataset_freeze_config(config.dataset_config_path)
    if canonical_dataset_config(dataset_config)["sha256"] != config.dataset_config_sha256:
        raise RuntimeError("Dataset config SHA-256 drift before threshold calibration")
    snapshot_path, snapshot = _load_verified_json(
        config.candidate_snapshot_path, config.candidate_snapshot_sha256
    )
    text_path, text_report = _load_verified_json(
        config.text.report_path, config.text.report_sha256
    )
    acoustic_path, acoustic_report = _load_verified_json(
        config.acoustic.report_path, config.acoustic.report_sha256
    )
    speaker_path, speaker_report = _load_verified_json(
        config.speaker.report_path, config.speaker.report_sha256
    )
    text = _calibrate_text(text_report, config.text)
    acoustic = _calibrate_acoustic(
        snapshot=snapshot,
        report=acoustic_report,
        dataset_config=dataset_config,
        config=config.acoustic,
    )
    speaker = _calibrate_speaker(speaker_report, config.speaker)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "calibration_id": config.calibration_id,
        "calibration_version": config.calibration_version,
        "dataset_id": dataset_config.dataset_id,
        "dataset_version": dataset_config.dataset_version,
        "calibration_config_sha256": identity["sha256"],
        "candidate_snapshot_sha256": config.candidate_snapshot_sha256,
        "inputs": {
            "calibration_config_path": str(resolved_config_path),
            "dataset_config_path": str(Path(config.dataset_config_path).resolve()),
            "dataset_config_sha256": config.dataset_config_sha256,
            "candidate_snapshot_path": str(snapshot_path),
            "text_report_path": str(text_path),
            "text_report_sha256": config.text.report_sha256,
            "acoustic_report_path": str(acoustic_path),
            "acoustic_report_sha256": config.acoustic.report_sha256,
            "speaker_report_path": str(speaker_path),
            "speaker_report_sha256": config.speaker.report_sha256,
        },
        "thresholds": {
            "text_near_similarity": text["threshold"],
            "acoustic_near_cosine": acoustic["threshold"],
            "speaker_center_cosine": speaker["center_cosine"]["threshold"],
            "speaker_knn_cosine": speaker["knn_cosine"]["threshold"],
        },
        "text": text,
        "acoustic": acoustic,
        "speaker": speaker,
        "policy": {
            "near_duplicate_result": "pending_review_only",
            "speaker_outlier_result": "pending_review_only",
            "automatic_exclusion": False,
            "dataset_config_promotion_required": True,
        },
    }
    output = _atomic_write_json(Path(config.output_path).resolve(), report)
    return {
        "status": "succeeded",
        "calibration_id": config.calibration_id,
        "calibration_version": config.calibration_version,
        "thresholds": report["thresholds"],
        "text_candidate_count": text["candidate_pair_count_at_threshold"],
        "acoustic_candidate_count": acoustic["candidate_pair_count_at_threshold"],
        "speaker_candidate_count": speaker["candidate_count"],
        "report": output,
    }
