from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soxr

from dots_tts_lab.inventory import DEFAULT_EXTENSIONS, discover_audio_files
from dots_tts_lab.long_form_asr import attach_clip_asr_candidate
from dots_tts_lab.long_form_audio import scan_long_audio
from dots_tts_lab.long_form_contract import (
    LongFormConfig,
    load_long_form_config,
    segment_identity,
)
from dots_tts_lab.long_form_features import (
    analyze_segment_samples,
    read_source_segment,
)
from dots_tts_lab.long_form_grouping import (
    cluster_speaker_embeddings,
    estimate_pitch_periodicity,
    suggest_styles,
)
from dots_tts_lab.long_form_segmentation import segment_long_audio
from dots_tts_lab.speaker_embedding import (
    load_or_compute_speaker_embedding,
    load_speaker_encoder,
)
from dots_tts_lab.standardization import atomic_write_pcm24, true_peak_estimate
from dots_tts_lab.long_form_paths import validate_output_path


LONG_FORM_PIPELINE_IMPLEMENTATION_VERSION = 2


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_long_form_work_root(work_root: Path) -> dict[str, int]:
    resolved = validate_output_path(work_root)
    resolved.mkdir(parents=True, exist_ok=True)
    partial_count = 0
    interrupted_count = 0
    for partial in resolved.rglob("*.partial"):
        if partial.is_file():
            partial.resolve().relative_to(resolved)
            partial.unlink()
            partial_count += 1
    for state_path in resolved.rglob("state.json"):
        state_path.resolve().relative_to(resolved)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") == "running":
            state["status"] = "interrupted"
            state["updated_at"] = _now()
            state["interrupted_reason"] = "previous_process_ended_while_running"
            _atomic_write_json(state_path, state)
            interrupted_count += 1
    return {
        "deleted_partial_count": partial_count,
        "marked_interrupted_count": interrupted_count,
    }


def _render_training_candidate(
    samples: np.ndarray,
    *,
    sample_rate: int,
    analysis_channel: int,
    channel_strategy: str,
    output_path: Path,
    config: LongFormConfig,
    ensure_zero_edges: bool = False,
) -> dict[str, Any]:
    if samples.shape[1] == 1:
        mono = samples[:, 0]
        rendered_strategy = "mono"
    elif channel_strategy == "mean":
        mono = np.mean(samples, axis=1, dtype=np.float32)
        rendered_strategy = "mean"
    else:
        mono = samples[:, analysis_channel]
        rendered_strategy = f"analysis_channel_{analysis_channel}"
    if sample_rate != config.decode.output_sample_rate_hz:
        mono = np.asarray(
            soxr.resample(
                mono,
                sample_rate,
                config.decode.output_sample_rate_hz,
                quality="HQ",
            ),
            dtype=np.float64,
        )
    else:
        mono = np.asarray(mono, dtype=np.float64)
    peak = true_peak_estimate(mono, 4)
    target = 10.0 ** (-1.0 / 20.0)
    gain = min(1.0, target / peak) if peak > 0.0 else 1.0
    rendered = mono * gain
    if ensure_zero_edges and len(rendered):
        # Resampler ringing/roundoff must not reintroduce nonzero file endpoints.
        rendered[0] = 0.0
        rendered[-1] = 0.0
    output_sha256, size_bytes, output_metrics = atomic_write_pcm24(
        output_path,
        rendered,
        config.decode.output_sample_rate_hz,
        4,
    )
    return {
        "derived_audio_sha256": output_sha256,
        "derived_audio_size_bytes": size_bytes,
        "derived_sample_rate_hz": config.decode.output_sample_rate_hz,
        "derived_frames": len(rendered),
        "rendered_channel_strategy": rendered_strategy,
        "render_gain_db": 0.0 if gain == 1.0 else 20.0 * np.log10(gain),
        **output_metrics,
    }


def _validated_cached_manifest(
    source_root: Path, *, source_sha256: str, config_sha256: str
) -> dict[str, Any] | None:
    state_path = source_root / "state.json"
    manifest_path = source_root / "manifest.json"
    if not state_path.is_file() or not manifest_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        state.get("status") not in {"awaiting_review", "completed", "awaiting_asr", "failed", "interrupted"}
        or manifest.get("implementation_version") != LONG_FORM_PIPELINE_IMPLEMENTATION_VERSION
        or manifest.get("source_sha256") != source_sha256
        or manifest.get("config_sha256") != config_sha256
    ):
        return None
    items = manifest.get("segments")
    if not isinstance(items, list):
        return None
    for item in items:
        relative = item.get("derived_relative_path")
        expected = item.get("derived_audio_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return None
        audio_path = (source_root / relative).resolve()
        try:
            audio_path.relative_to(source_root.resolve())
        except ValueError:
            return None
        if not audio_path.is_file() or _sha256_file(audio_path) != expected:
            return None
    return manifest


def _run_timestamp_asr(
    *,
    repo_root: Path,
    config: LongFormConfig,
    source_root: Path,
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = source_root / "asr_manifest.jsonl"
    lines = [
        json.dumps(
            {
                "asset_sha256": row["segment_id"],
                "derived_relative_path": Path(row["derived_relative_path"]).name,
                "derived_sha256": row["derived_audio_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in segment_rows
    ]
    _atomic_write_text(manifest_path, "\n".join(lines) + "\n")
    output_path = source_root / "asr_attempt_results.json"
    command = [
        str((repo_root / config.text.timestamp_python).resolve()),
        str((repo_root / "scripts" / "run_asr_backend.py").resolve()),
        "--config",
        str((repo_root / config.text.timestamp_backend_config).resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--audio-root",
        str((source_root / "segments").resolve()),
        "--model-root",
        str((repo_root / config.text.model_root).resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "timestamp ASR failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-4000:]
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("capabilities", {}).get("timestamps_enabled") is not True:
        raise RuntimeError("timestamp ASR completed without timestamps enabled")
    return payload


def _materialize_segments(
    source_path: Path, *, source_sha256: str, config_sha256: str,
    source_root: Path, work_root: Path, config: LongFormConfig, encoder: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segmentation = segment_long_audio(source_path, config=config)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    speaker_features: list[dict[str, Any]] = []
    active_encoder = encoder or load_speaker_encoder(config.speaker.encoder)
    for boundary in segmentation["segments"]:
        segment_id = segment_identity(
            source_sha256=source_sha256,
            source_start_frame=int(boundary["source_start_frame"]),
            source_end_frame=int(boundary["source_end_frame"]),
            config_sha256=config_sha256,
            implementation_version=LONG_FORM_PIPELINE_IMPLEMENTATION_VERSION,
        )
        samples, sample_rate = read_source_segment(
            source_path,
            start_frame=int(boundary["source_start_frame"]),
            end_frame=int(boundary["source_end_frame"]),
        )
        features = analyze_segment_samples(
            samples,
            sample_rate=sample_rate,
            config=config,
            activity_start_threshold_dbfs=float(
                segmentation["activity_start_threshold_dbfs"]
            ),
            activity_continue_threshold_dbfs=float(
                segmentation["activity_continue_threshold_dbfs"]
            ),
        )
        pitch = estimate_pitch_periodicity(
            samples,
            sample_rate=sample_rate,
            analysis_channel=int(features["analysis_channel"]),
        )
        relative_audio = f"segments/{segment_id}.wav"
        output_path = source_root / relative_audio
        rendered = _render_training_candidate(
            samples,
            sample_rate=sample_rate,
            analysis_channel=int(features["analysis_channel"]),
            channel_strategy=str(features["recommended_channel_strategy"]),
            output_path=output_path,
            config=config,
        )
        speaker_feature = load_or_compute_speaker_embedding(
            {
                "asset_sha256": segment_id,
                "audio_sha256": rendered["derived_audio_sha256"],
                "audio_absolute_path": str(output_path.resolve()),
            },
            config=config.speaker.encoder,
            encoder=active_encoder,
            cache_root=work_root / "cache" / "speaker_embedding",
        )
        vectors.append(speaker_feature.pop("vector"))
        speaker_features.append(speaker_feature)
        reasons = list(features["review_reasons"])
        if boundary["review_required"]:
            reasons.append(str(boundary["boundary_reason"]))
        rows.append(
            {
                "schema_version": 1,
                "segment_id": segment_id,
                "source_sha256": source_sha256,
                "config_sha256": config_sha256,
                **boundary,
                **features,
                **pitch,
                **rendered,
                "derived_relative_path": relative_audio,
                "speaker_embedding": speaker_feature,
                "review_reasons": sorted(set(reasons)),
                "asr_text_status": "not_run",
                "asr_candidate_text": None,
                "human_confirmed_text": None,
                "text_review_status": "pending",
            }
        )
    groups = cluster_speaker_embeddings(
        vectors,
        durations=[float(row["duration_seconds"]) for row in rows],
        cosine_link_threshold=config.speaker.cosine_link_threshold,
        knn_k=config.speaker.knn_k,
        minimum_cluster_size=config.speaker.minimum_cluster_size,
        reference_vector=None,
        reference_minimum_cosine=config.speaker.target_reference_minimum_cosine,
    ) if rows else []
    styles = suggest_styles(rows, config=config, speaker_groups=groups) if rows else []
    for row, group, style in zip(rows, groups, styles):
        row.update(group)
        row.update(style)
        row["review_reasons"] = sorted(
            set(
                row["review_reasons"]
                + (["target_reference_required"] if group["target_speaker_status"] == "reference_required" else [])
                + (["small_speaker_cluster"] if group["speaker_cluster_is_small"] else [])
                + style["style_reasons"]
            )
        )

    return rows, {key: value for key, value in segmentation.items() if key != "segments"}


def _attached_asr(row: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None or result.get("status") != "ok":
        return None
    try:
        attached = attach_clip_asr_candidate(
            row, hypothesis=str(result["hypothesis"]), metadata=dict(result["metadata"]),
        )
    except (KeyError, ValueError, TypeError):
        return None
    if attached["asr_text_status"] != "candidate" or attached["asr_word_count"] == 0:
        return None
    return attached


def _complete_timestamp_asr(
    summary: dict[str, Any], *, source_root: Path, repo_root: Path, config: LongFormConfig,
) -> None:
    results_path = source_root / "asr_timestamp_results.json"
    prior: dict[str, Any] = {}
    if (
        results_path.is_file()
        and summary.get("asr_results_sha256") == _sha256_file(results_path)
    ):
        prior = json.loads(results_path.read_text(encoding="utf-8"))
    old_results = {item["asset_sha256"]: item for item in prior.get("results", [])}
    rows = summary["segments"]
    pending = [row for row in rows if _attached_asr(row, old_results.get(row["segment_id"])) is None]
    payload = prior
    if pending:
        payload = _run_timestamp_asr(
            repo_root=repo_root, config=config, source_root=source_root, segment_rows=pending,
        )
        new_results = {item["asset_sha256"]: item for item in payload["results"]}
        for row in pending:
            key = row["segment_id"]
            old_results[key] = new_results.get(key, {"asset_sha256": key, "status": "failed"})
    combined = []
    for row in rows:
        result = old_results.get(row["segment_id"])
        attached = _attached_asr(row, result)
        row["review_reasons"] = [reason for reason in row["review_reasons"] if reason != "asr_failed"]
        if attached is None:
            row.update(asr_text_status="failed", asr_candidate_text=None)
            row["review_reasons"].append("asr_failed")
            result = {"asset_sha256": row["segment_id"], "status": "failed"}
        else:
            row.update(attached)
            row["review_reasons"] = sorted(
                (set(row["review_reasons"]) - {"asr_failed"}) | set(row.get("asr_boundary_flags", []))
            )
        combined.append(result)
    success = sum(row["asr_text_status"] == "candidate" for row in rows)
    failure = len(rows) - success
    summary.update(
        asr_status="succeeded" if failure == 0 else ("partial" if success else "failed"),
        asr_success_count=success, asr_failure_count=failure,
        asr_backend={key: payload.get(key) for key in (
            "backend_id", "backend_version", "model_id", "model_revision", "inference_config_sha256"
        )},
    )
    _atomic_write_json(results_path, {**payload, "results": combined})
    summary["asr_results_sha256"] = _sha256_file(results_path)


def _process_source(
    source_path: Path, *, input_root: Path, repo_root: Path, work_root: Path,
    config: LongFormConfig, skip_asr: bool, force: bool, encoder: Any | None,
) -> dict[str, Any]:
    scan = scan_long_audio(source_path, config=config)
    source_sha256 = str(scan["source_sha256"])
    config_sha256 = config.config_sha256()
    backend_config_sha256 = _sha256_file(repo_root / config.text.timestamp_backend_config)
    identity = {
        "source_sha256": source_sha256, "config_sha256": config_sha256,
        "implementation_version": LONG_FORM_PIPELINE_IMPLEMENTATION_VERSION,
        "asr_backend_config_sha256": backend_config_sha256,
    }
    if force:
        identity["forced_run_id"] = uuid.uuid4().hex
    run_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    source_root = work_root / "sources" / run_key[:2] / run_key
    validate_output_path(source_root, protected_inputs=(input_root, repo_root / config.paths.input_root))
    state_path = source_root / "state.json"
    manifest_path = source_root / "manifest.json"
    relative_source = source_path.relative_to(input_root).as_posix()
    cached = _validated_cached_manifest(
        source_root, source_sha256=source_sha256, config_sha256=config_sha256,
    )
    if manifest_path.exists() and cached is None:
        raise RuntimeError("source cache is invalid; use --force to build a separate run")
    if cached is not None and (skip_asr or cached.get("asr_status") == "succeeded"):
        if not skip_asr:
            asr_path = source_root / "asr_timestamp_results.json"
            if (
                any(row.get("asr_text_status") != "candidate" for row in cached["segments"])
                or not asr_path.is_file()
                or cached.get("asr_results_sha256") != _sha256_file(asr_path)
            ):
                raise RuntimeError("completed ASR cache is invalid; use --force for a separate run")
        return {**cached, "source_relative_path": relative_source,
                "cache_action": "cached", "manifest_path": str(manifest_path)}
    state = {
        "schema_version": 1, **identity, "source_relative_path": relative_source,
        "status": "running", "stage": "materialize", "started_at": _now(), "updated_at": _now(),
    }
    _atomic_write_json(state_path, state)
    try:
        if cached is None:
            rows, segmentation = _materialize_segments(
                source_path, source_sha256=source_sha256, config_sha256=config_sha256,
                source_root=source_root, work_root=work_root, config=config, encoder=encoder,
            )
            summary = {
                "schema_version": 1, **identity, "source_relative_path": relative_source,
                "source_scan": scan, "segmentation": segmentation, "segment_count": len(rows),
                "segments": rows, "asr_status": "skipped", "asr_backend": None,
                "cache_action": "built",
            }
            # Commit materialized audio before invoking ASR, so backend crashes can resume.
            _atomic_write_json(manifest_path, summary)
        else:
            summary = cached
        if not skip_asr:
            state.update(stage="asr", updated_at=_now())
            _atomic_write_json(state_path, state)
            _complete_timestamp_asr(summary, source_root=source_root, repo_root=repo_root, config=config)
        _atomic_write_json(manifest_path, summary)
        pending = not skip_asr and summary["asr_status"] != "succeeded"
        state.update(
            status="awaiting_asr" if pending else "awaiting_review",
            stage="asr" if pending else "review", updated_at=_now(),
            segment_count=summary["segment_count"], manifest_path=str(manifest_path),
        )
        _atomic_write_json(state_path, state)
        return {**summary, "cache_action": "resumed" if cached else "built",
                "manifest_path": str(manifest_path)}
    except Exception as error:
        state.update(status="failed", updated_at=_now(),
                     error_type=type(error).__name__, error_message=str(error))
        _atomic_write_json(state_path, state)
        raise


def run_long_form_pipeline(
    input_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    work_root: str | Path | None = None,
    report_root: str | Path | None = None,
    skip_asr: bool = False,
    force: bool = False,
    limit_sources: int | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_long_form_config(config_path) if config_path else load_long_form_config()
    resolved_input = Path(input_dir).resolve()
    if not resolved_input.is_dir():
        raise NotADirectoryError(f"Long-form input directory does not exist: {resolved_input}")
    resolved_work = (
        Path(work_root).resolve()
        if work_root is not None
        else (repo_root / config.paths.work_root).resolve()
    )
    resolved_report = (
        Path(report_root).resolve()
        if report_root is not None
        else (repo_root / config.paths.report_root).resolve()
    )
    protected = (resolved_input, repo_root / config.paths.input_root)
    validate_output_path(resolved_work, protected_inputs=protected)
    validate_output_path(resolved_report, protected_inputs=protected)
    if limit_sources is not None and limit_sources < 1:
        raise ValueError("limit_sources must be positive")
    recovery = recover_long_form_work_root(resolved_work)
    sources = discover_audio_files(resolved_input, extensions=DEFAULT_EXTENSIONS)
    if limit_sources is not None:
        sources = sources[:limit_sources]
    results = []
    for source in sources:
        results.append(
            _process_source(
                source,
                input_root=resolved_input,
                repo_root=repo_root,
                work_root=resolved_work,
                config=config,
                skip_asr=skip_asr,
                force=force,
                encoder=encoder,
            )
        )
    index = {
        "schema_version": 1,
        "config_id": config.config_id,
        "config_version": config.config_version,
        "config_sha256": config.config_sha256(),
        "input_root": str(resolved_input),
        "work_root": str(resolved_work),
        "report_root": str(resolved_report),
        "recovery": recovery,
        "source_count": len(results),
        "segment_count": sum(int(result["segment_count"]) for result in results),
        "sources": [
            {
                "source_sha256": result["source_sha256"],
                "source_relative_path": result["source_relative_path"],
                "segment_count": result["segment_count"],
                "asr_status": result["asr_status"],
                "cache_action": result["cache_action"],
                "manifest_path": result["manifest_path"],
                "asr_failure_count": result.get("asr_failure_count", 0),
            }
            for result in results
        ],
        "created_at": _now(),
    }
    _atomic_write_json(resolved_report / "index.json", index)
    return index
