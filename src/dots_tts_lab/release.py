from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dots_tts_lab.postprocess import load_edge_trim_config, safe_edge_trim
from dots_tts_lab.standardization import true_peak_estimate

ROOT = Path(__file__).resolve().parents[2]


class ReleaseSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generation_manifest: str
    generation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acceptance_report: str
    acceptance_required: bool


class ReleaseFormatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    container: Literal["WAV", "FLAC", "MP3"]
    subtype: Literal["PCM_24", "MPEG_LAYER_III"]
    bitrate_mode: Literal["VARIABLE", "CONSTANT", "AVERAGE"] | None = None
    compression_level: float | None = Field(default=None, ge=0.0, le=1.0)


class ReleasePresetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    description: str
    edge_trim: bool
    loudness_normalization: bool
    target_loudness_lufs: float | None
    maximum_gain_db: float | None = Field(default=None, ge=0.0, le=30.0)
    safety_limiter: bool
    true_peak_ceiling_dbtp: float | None = Field(default=None, ge=-12.0, le=0.0)
    formats: list[Literal["wav", "flac", "mp3"]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_processing_contract(self) -> ReleasePresetConfig:
        if self.loudness_normalization != (self.target_loudness_lufs is not None):
            raise ValueError("target_loudness_lufs must match loudness_normalization")
        if self.loudness_normalization != (self.maximum_gain_db is not None):
            raise ValueError("maximum_gain_db must match loudness_normalization")
        if self.safety_limiter != (self.true_peak_ceiling_dbtp is not None):
            raise ValueError("true_peak_ceiling_dbtp must match safety_limiter")
        return self


class ReleaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    config_id: str
    config_version: int = Field(ge=1)
    source: ReleaseSourceConfig
    model: dict[str, Any]
    postprocess: dict[str, str]
    presets: dict[str, ReleasePresetConfig]
    formats: dict[str, ReleaseFormatConfig]
    output_root: str

    @model_validator(mode="after")
    def validate_references(self) -> ReleaseConfig:
        missing = {
            name
            for preset in self.presets.values()
            for name in preset.formats
            if name not in self.formats
        }
        if missing:
            raise ValueError(f"Undefined release formats: {sorted(missing)}")
        return self


def load_release_config(path: str | Path) -> ReleaseConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Release configuration must be a YAML mapping")
    return ReleaseConfig.model_validate(payload, strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float | None:
    try:
        value = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float | None:
    peak = true_peak_estimate(audio, oversample)
    if peak <= 0.0 or not np.isfinite(peak):
        return None
    return float(20.0 * np.log10(peak))


def process_audio(
    audio: np.ndarray,
    sample_rate: int,
    preset: ReleasePresetConfig,
    *,
    edge_trim_config,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(audio, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError("Release processing requires finite, non-empty mono audio")
    output = source.copy()
    trim = {
        "all_silent": False,
        "leading_samples_removed": 0,
        "trailing_samples_removed": 0,
    }
    if preset.edge_trim:
        output, trim = safe_edge_trim(output, sample_rate, edge_trim_config)

    loudness_before = integrated_loudness(output, sample_rate)
    peak_before = true_peak_dbtp(output)
    requested_gain_db = 0.0
    applied_gain_db = 0.0
    limited_by_true_peak = False
    if preset.loudness_normalization:
        if loudness_before is None:
            raise RuntimeError("Cannot normalize loudness for silent or invalid audio")
        requested_gain_db = float(preset.target_loudness_lufs - loudness_before)
        applied_gain_db = min(requested_gain_db, float(preset.maximum_gain_db))
    if preset.safety_limiter and peak_before is not None:
        peak_safe_gain = float(preset.true_peak_ceiling_dbtp - peak_before)
        if applied_gain_db > peak_safe_gain:
            applied_gain_db = peak_safe_gain
            limited_by_true_peak = True
    output *= 10.0 ** (applied_gain_db / 20.0)
    return output, {
        "edge_trim": bool(preset.edge_trim),
        **trim,
        "loudness_normalization": bool(preset.loudness_normalization),
        "loudness_before_lufs": loudness_before,
        "target_loudness_lufs": preset.target_loudness_lufs,
        "requested_gain_db": requested_gain_db,
        "applied_gain_db": applied_gain_db,
        "safety_limiter": bool(preset.safety_limiter),
        "safety_limiter_mode": "peak_safe_global_gain" if preset.safety_limiter else None,
        "limited_by_true_peak": limited_by_true_peak,
        "true_peak_before_dbtp": peak_before,
        "true_peak_ceiling_dbtp": preset.true_peak_ceiling_dbtp,
    }


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_copy(source: Path, target: Path) -> None:
    temporary = _temporary_path(target)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Any) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_audio(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    format_config: ReleaseFormatConfig,
) -> None:
    temporary = _temporary_path(path)
    try:
        sf.write(
            str(temporary),
            audio,
            sample_rate,
            format=format_config.container,
            subtype=format_config.subtype,
            compression_level=format_config.compression_level,
            bitrate_mode=format_config.bitrate_mode,
        )
        decoded, decoded_rate = sf.read(
            str(temporary), dtype="float64", always_2d=False
        )
        if decoded_rate != sample_rate or decoded.ndim != 1 or decoded.size == 0:
            raise RuntimeError(f"Encoded audio verification failed: {temporary}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def encode_peak_safe(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    format_config: ReleaseFormatConfig,
    ceiling_dbtp: float | None,
) -> tuple[np.ndarray, float]:
    encode_audio = audio
    encoding_attenuation_db = 0.0
    for _ in range(4):
        _write_audio(path, encode_audio, sample_rate, format_config)
        decoded, _ = sf.read(str(path), dtype="float64", always_2d=False)
        peak = true_peak_dbtp(decoded)
        if ceiling_dbtp is None or peak is None or peak <= ceiling_dbtp + 1e-3:
            return decoded, encoding_attenuation_db
        correction = ceiling_dbtp - peak - 0.01
        encoding_attenuation_db += correction
        encode_audio = audio * 10.0 ** (encoding_attenuation_db / 20.0)
    raise RuntimeError(f"Unable to satisfy encoded true-peak ceiling: {path}")


def extension_for(container: str) -> str:
    return {"WAV": "wav", "FLAC": "flac", "MP3": "mp3"}[container]


def build_sidecar(
    *,
    row: dict[str, Any],
    preset_name: str,
    run_label: str | None,
    preset: ReleasePresetConfig,
    source_manifest: Path,
    source_manifest_sha256: str,
    source_path: Path,
    config_path: Path,
    config_sha256: str,
    output_path: Path,
    format_name: str,
    format_config: ReleaseFormatConfig,
    processing: dict[str, Any],
    decoded: np.ndarray,
    sample_rate: int,
    encoding_attenuation_db: float,
) -> dict[str, Any]:
    info = sf.info(str(output_path))
    return {
        "schema_version": 1,
        "status": "exported",
        "preset": preset_name,
        "run_label": run_label,
        "preset_contract": preset.model_dump(mode="json"),
        "release_config_path": config_path.relative_to(ROOT).as_posix(),
        "release_config_sha256": config_sha256,
        "source_manifest_path": source_manifest.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": source_manifest_sha256,
        "source_audio_path": source_path.relative_to(ROOT).as_posix(),
        "source_audio_sha256": row["output_sha256"],
        "job": {
            key: row.get(key)
            for key in (
                "job_id",
                "ordinal",
                "sentence_id",
                "text",
                "purpose",
                "seed",
                "request_id",
            )
        },
        "reference": {
            key: row.get(key)
            for key in (
                "asset_sha256",
                "audio_relative_path",
                "audio_sha256",
                "prompt_text",
                "prompt_text_sha256",
                "speaker_id",
                "pool_ids",
            )
        },
        "model": {
            key: row.get(key)
            for key in (
                "model_id",
                "model_revision",
                "model_adapter",
                "precision",
                "runtime_options",
                "sampling_options",
            )
        },
        "processing": {
            **processing,
            "encoding_safety_attenuation_db": encoding_attenuation_db,
            "total_gain_db": processing["applied_gain_db"]
            + encoding_attenuation_db,
            "output_loudness_lufs": integrated_loudness(decoded, sample_rate),
            "output_true_peak_dbtp": true_peak_dbtp(decoded),
        },
        "output": {
            "format_name": format_name,
            "container": format_config.container,
            "subtype": info.subtype,
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_seconds": info.duration,
        },
    }


def export_preset(
    config_path: Path,
    preset_name: str,
    *,
    limit: int | None = None,
    job_ids: set[str] | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_release_config(config_path)
    if preset_name not in config.presets:
        raise ValueError(f"Unknown release preset: {preset_name}")
    if (limit is not None or job_ids is not None) and run_label is None:
        raise ValueError("Filtered exports require run_label to protect canonical manifests")
    if run_label is not None and not re.fullmatch(r"[a-z][a-z0-9_.-]{0,31}", run_label):
        raise ValueError("run_label must be a safe lowercase identifier")
    preset = config.presets[preset_name]
    source_manifest = (ROOT / config.source.generation_manifest).resolve()
    if sha256_file(source_manifest) != config.source.generation_manifest_sha256:
        raise RuntimeError("Accepted generation manifest hash mismatch")
    acceptance = (ROOT / config.source.acceptance_report).resolve()
    acceptance_report = json.loads(acceptance.read_text(encoding="utf-8"))
    if config.source.acceptance_required and not acceptance_report.get("accepted"):
        raise RuntimeError("Slice 12 acceptance report is not accepted")
    edge_config_path = (ROOT / config.postprocess["edge_trim_config"]).resolve()
    edge_config = load_edge_trim_config(edge_config_path)
    if edge_config.canonical_sha256() != config.postprocess["edge_trim_config_sha256"]:
        raise RuntimeError("Edge-trim configuration hash mismatch")

    rows = load_jsonl(source_manifest)
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("Release source manifest must contain only successful rows")
    if job_ids is not None:
        rows = [row for row in rows if row["job_id"] in job_ids]
        missing = job_ids - {row["job_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown job IDs: {sorted(missing)}")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No release rows selected")

    config_hash = sha256_file(config_path)
    source_manifest_hash = sha256_file(source_manifest)
    output_root = (
        ROOT
        / config.output_root
        / ((Path("runs") / run_label / preset_name) if run_label else preset_name)
    ).resolve()
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        source_path = (ROOT / row["output_path"]).resolve()
        if sha256_file(source_path) != row["output_sha256"]:
            raise RuntimeError(f"Source audio hash mismatch: {source_path}")
        source_audio, sample_rate = sf.read(
            str(source_path), dtype="float64", always_2d=False
        )
        if preset_name in {"raw", "training"}:
            processed = source_audio
            processing = {
                "edge_trim": False,
                "all_silent": False,
                "leading_samples_removed": 0,
                "trailing_samples_removed": 0,
                "loudness_normalization": False,
                "loudness_before_lufs": integrated_loudness(
                    source_audio, sample_rate
                ),
                "target_loudness_lufs": None,
                "requested_gain_db": 0.0,
                "applied_gain_db": 0.0,
                "safety_limiter": False,
                "safety_limiter_mode": None,
                "limited_by_true_peak": False,
                "true_peak_before_dbtp": true_peak_dbtp(source_audio),
                "true_peak_ceiling_dbtp": None,
            }
        else:
            processed, processing = process_audio(
                source_audio,
                sample_rate,
                preset,
                edge_trim_config=edge_config,
            )

        for format_name in preset.formats:
            format_config = config.formats[format_name]
            extension = extension_for(format_config.container)
            output_path = output_root / f"{row['job_id']}.{extension}"
            if preset_name in {"raw", "training"}:
                if format_config.container != "WAV":
                    raise RuntimeError("Passthrough presets only support WAV")
                atomic_copy(source_path, output_path)
                decoded = source_audio
                encoding_attenuation_db = 0.0
                if sha256_file(output_path) != row["output_sha256"]:
                    raise RuntimeError("Passthrough export is not byte-identical")
            else:
                decoded, encoding_attenuation_db = encode_peak_safe(
                    output_path,
                    processed,
                    sample_rate,
                    format_config,
                    preset.true_peak_ceiling_dbtp,
                )
            sidecar = build_sidecar(
                row=row,
                preset_name=preset_name,
                run_label=run_label,
                preset=preset,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_hash,
                source_path=source_path,
                config_path=config_path,
                config_sha256=config_hash,
                output_path=output_path,
                format_name=format_name,
                format_config=format_config,
                processing=processing,
                decoded=decoded,
                sample_rate=sample_rate,
                encoding_attenuation_db=encoding_attenuation_db,
            )
            sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
            atomic_json(sidecar_path, sidecar)
            manifest_rows.append(
                {
                    "schema_version": 1,
                    "preset": preset_name,
                    "job_id": row["job_id"],
                    "format": format_name,
                    "output_path": sidecar["output"]["path"],
                    "output_sha256": sidecar["output"]["sha256"],
                    "sidecar_path": sidecar_path.relative_to(ROOT).as_posix(),
                    "sidecar_sha256": sha256_file(sidecar_path),
                }
            )
    manifest_path = output_root / "manifest.jsonl"
    atomic_jsonl(manifest_path, manifest_rows)
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "preset": preset_name,
        "run_label": run_label,
        "source_item_count": len(rows),
        "output_count": len(manifest_rows),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "release_config_sha256": config_hash,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
