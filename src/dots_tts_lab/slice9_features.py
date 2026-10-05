from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from dots_tts_lab.quality import (
    analyze_audio,
    load_quality_analysis_config,
)
from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_CONFIG_PATH,
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    load_slice9_selection_config,
)


DEFAULT_SLICE9_QUALITY_FEATURE_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_quality_features_v1.json"
)
_DB_FLOOR = -120.0
_QUALITY_METRICS = (
    "sample_peak_dbfs",
    "true_peak_estimate_dbtp",
    "rms_dbfs",
    "integrated_loudness_lufs",
    "crest_factor_db",
    "abs_dc_offset",
    "noise_floor_proxy_dbfs",
    "speech_level_proxy_dbfs",
    "snr_proxy_db",
    "near_peak_sample_ratio",
    "flat_top_run_count",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _frame_dbfs(data: np.ndarray, *, sample_rate: int, frame_length_ms: float, hop_length_ms: float) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    frame_length = max(1, round(sample_rate * frame_length_ms / 1000.0))
    hop_length = max(1, round(sample_rate * hop_length_ms / 1000.0))
    power = np.square(values, dtype=np.float64)
    if len(power) <= frame_length:
        means = np.asarray([float(np.mean(power))], dtype=np.float64)
    else:
        starts = np.arange(0, len(power) - frame_length + 1, hop_length)
        if starts[-1] != len(power) - frame_length:
            starts = np.append(starts, len(power) - frame_length)
        cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
        means = (cumulative[starts + frame_length] - cumulative[starts]) / frame_length
    floor = 10.0 ** (_DB_FLOOR / 20.0)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.maximum(means, 0.0)), floor))


def _internal_silence_metrics(
    frame_dbfs: np.ndarray,
    *,
    frame_length_ms: float,
    hop_length_ms: float,
    threshold_dbfs: float,
) -> dict[str, Any]:
    active = frame_dbfs > threshold_dbfs
    if not np.any(active):
        return {
            "active_frame_count": 0,
            "internal_silence_run_count": 0,
            "max_internal_silence_seconds": 0.0,
            "all_silent": True,
        }
    first = int(np.flatnonzero(active)[0])
    last = int(np.flatnonzero(active)[-1])
    between = ~active[first : last + 1]
    padded = np.concatenate(([False], between, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    run_lengths = [int(end - start) for start, end in zip(starts, ends)]
    hop_seconds = hop_length_ms / 1000.0
    return {
        "active_frame_count": int(np.count_nonzero(active)),
        "internal_silence_run_count": len(run_lengths),
        "max_internal_silence_seconds": float(max(run_lengths, default=0) * hop_seconds),
        "all_silent": False,
    }


def _runtime_trim(path: Path, *, target_sample_rate: int) -> dict[str, Any]:
    waveform, sample_rate = librosa.load(str(path), sr=None, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    _require(waveform.size > 0, "Runtime prompt audio is empty")
    trimmed, indices = librosa.effects.trim(waveform, top_db=30)
    start, end = int(indices[0]), int(indices[1])
    if int(sample_rate) != target_sample_rate:
        import torch

        from dots_tts.utils.audio import high_quality_resample

        tensor = torch.from_numpy(np.asarray(trimmed, dtype=np.float32)).reshape(1, -1)
        trimmed = high_quality_resample(
            tensor,
            orig_sr=int(sample_rate),
            target_sr=int(target_sample_rate),
        ).detach().cpu().numpy().reshape(-1)
    return {
        "source_sample_rate_hz": int(sample_rate),
        "source_samples": int(waveform.size),
        "trimmed_samples": int(np.asarray(trimmed).size),
        "leading_seconds": float(start / sample_rate),
        "trailing_seconds": float(max(0, waveform.size - end) / sample_rate),
        "effective_seconds": float(np.asarray(trimmed).size / target_sample_rate),
    }


def _quality_reasons(metrics: dict[str, Any], quality_review: dict[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    def maximum(metric: str, threshold: float, code: str) -> None:
        value = metrics.get(metric)
        if value is not None and float(value) > threshold:
            reasons.append({"code": code, "metric": metric, "actual": float(value), "threshold": threshold, "relation": "maximum"})

    def minimum(metric: str, threshold: float, code: str) -> None:
        value = metrics.get(metric)
        if value is not None and float(value) < threshold:
            reasons.append({"code": code, "metric": metric, "actual": float(value), "threshold": threshold, "relation": "minimum"})

    maximum("abs_dc_offset", float(quality_review["max_abs_dc_offset"]), "dc_offset")
    maximum("leading_silence_seconds", float(quality_review["max_leading_silence_seconds"]), "long_leading_silence")
    maximum("trailing_silence_seconds", float(quality_review["max_trailing_silence_seconds"]), "long_trailing_silence")
    maximum("silence_ratio", float(quality_review["max_silence_ratio"]), "high_silence_ratio")
    maximum("flat_top_run_count", float(quality_review["max_flat_top_run_count"]), "possible_hard_clipping")
    minimum("integrated_loudness_lufs", float(quality_review["min_integrated_loudness_lufs"]), "low_integrated_loudness")
    maximum("integrated_loudness_lufs", float(quality_review["max_integrated_loudness_lufs"]), "high_integrated_loudness")
    maximum("true_peak_estimate_dbtp", float(quality_review["max_true_peak_estimate_dbtp"]), "true_peak_above_limit")
    minimum("snr_proxy_db", float(quality_review["min_snr_proxy_db"]), "low_snr_proxy")
    return reasons


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def build_slice9_quality_features(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    snapshot_path: str | Path = DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    standardization_report_path: str | Path = "data/reports/standardization/standardization.json",
    output_path: str | Path = DEFAULT_SLICE9_QUALITY_FEATURE_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    snapshot = _load_json(Path(snapshot_path).resolve(), label="Slice 9 candidate snapshot")
    items = snapshot.get("items")
    _require(isinstance(items, list) and snapshot.get("item_count") == len(items), "Invalid Slice 9 candidate snapshot")
    standardization_report = _load_json(Path(standardization_report_path).resolve(), label="standardization report")
    summary = standardization_report.get("summary")
    assets = standardization_report.get("assets")
    _require(isinstance(summary, dict) and summary.get("status") == "succeeded", "Standardization report did not succeed")
    _require(isinstance(assets, list), "Standardization assets are missing")
    standardized = config["standardized_audio"]
    _require(summary.get("config_sha256") == standardized["config_sha256"], "Standardization config drift")
    output_root = Path(str(summary["output_root_path"])).resolve()
    by_asset = {str(asset["asset_sha256"]): asset for asset in assets if isinstance(asset, dict)}
    analysis_config = load_quality_analysis_config()
    quality_review = config["gates"]["quality_review"]
    frame_length_ms = float(analysis_config.frame_length_ms)
    hop_length_ms = float(analysis_config.hop_length_ms)
    threshold_dbfs = float(analysis_config.silence_threshold_dbfs)
    output_items: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for candidate in items:
        asset_sha256 = str(candidate["asset_sha256"])
        standardized_item = by_asset.get(asset_sha256)
        _require(standardized_item is not None, f"Standardization asset missing: {asset_sha256}")
        _require(standardized_item.get("relative_path") == candidate["audio_relative_path"], f"Standardization path drift: {asset_sha256}")
        audio_path = (output_root / str(standardized_item["relative_path"])).resolve()
        _require(os.path.commonpath((str(output_root), str(audio_path))) == str(output_root), f"Audio path escapes standardized root: {audio_path}")
        _require(audio_path.is_file(), f"Standardized audio missing: {audio_path}")
        info = sf.info(str(audio_path))
        metrics = analyze_audio(str(audio_path), subtype=info.subtype, config=analysis_config)
        _require(metrics.get("status") == "ok", f"Quality analysis failed: {asset_sha256}")
        waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        frame_dbfs = _frame_dbfs(
            waveform,
            sample_rate=int(sample_rate),
            frame_length_ms=frame_length_ms,
            hop_length_ms=hop_length_ms,
        )
        silence = _internal_silence_metrics(
            frame_dbfs,
            frame_length_ms=frame_length_ms,
            hop_length_ms=hop_length_ms,
            threshold_dbfs=threshold_dbfs,
        )
        runtime = _runtime_trim(audio_path, target_sample_rate=int(config["runtime_contract"]["model_sample_rate_hz"]))
        duration = {
            "source_duration_seconds": float(candidate.get("duration_seconds", 0.0)),
            "standardized_duration_seconds": float(metrics["duration_seconds"]),
            "effective_seconds": runtime["effective_seconds"],
            "trimmed_fraction": (
                runtime["effective_seconds"] / float(metrics["duration_seconds"])
                if float(metrics["duration_seconds"]) > 0
                else None
            ),
            "patch_size": 4,
            "hop_size": 1920,
            "patch_count": math.ceil(runtime["effective_seconds"] * int(config["runtime_contract"]["model_sample_rate_hz"]) / (4 * 1920)),
        }
        reasons: list[dict[str, Any]] = []
        hard_reject = False
        if silence["all_silent"]:
            hard_reject = True
            reasons.append({"code": "all_silent_audio", "severity": "reject"})
        if duration["effective_seconds"] <= float(config["gates"]["hard"]["effective_duration_min_seconds"]):
            hard_reject = True
            reasons.append({"code": "effective_duration_empty", "severity": "reject", "actual": duration["effective_seconds"]})
        if duration["effective_seconds"] > float(config["gates"]["hard"]["effective_duration_max_seconds"]):
            hard_reject = True
            reasons.append({"code": "effective_duration_too_long", "severity": "reject", "actual": duration["effective_seconds"], "threshold": 10.0})
        review_reasons = _quality_reasons(metrics, quality_review)
        reasons.extend({"severity": "review", **reason} for reason in review_reasons)
        decision = "reject" if hard_reject else ("review" if review_reasons else "pass")
        decision_counts[decision] += 1
        for reason in reasons:
            reason_counts[str(reason["code"])] += 1
        output_items.append(
            {
                "asset_sha256": asset_sha256,
                "fid": candidate["fid"],
                "audio_sha256": candidate["audio_sha256"],
                "audio_relative_path": candidate["audio_relative_path"],
                "emotion_primary": candidate["emotion_primary"],
                "quality": {key: metrics.get(key) for key in ("sample_rate", "channels", "frames", "duration_seconds", *_QUALITY_METRICS)},
                "silence": {
                    "frame_length_ms": frame_length_ms,
                    "hop_length_ms": hop_length_ms,
                    "threshold_dbfs": threshold_dbfs,
                    "leading_silence_seconds": metrics.get("leading_silence_seconds"),
                    "trailing_silence_seconds": metrics.get("trailing_silence_seconds"),
                    "silence_ratio": metrics.get("silence_ratio"),
                    "digital_silence_frame_ratio": metrics.get("digital_silence_frame_ratio"),
                    **silence,
                },
                "runtime_trim": runtime,
                "duration": duration,
                "decision": decision,
                "reasons": reasons,
            }
        )
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "candidate_snapshot_sha256": hashlib.sha256(Path(snapshot_path).resolve().read_bytes()).hexdigest(),
        "quality_analysis_id": analysis_config.analysis_id,
        "quality_analysis_version": analysis_config.analysis_version,
        "quality_analysis_config_sha256": analysis_config.config_sha256(),
        "implementation_version": 1,
        "candidate_count": len(output_items),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "items": sorted(output_items, key=lambda item: item["asset_sha256"]),
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, report)
    return {**report, "report_path": str(resolved_output)}

