from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping

import numpy as np
import soundfile as sf
import soxr
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_paths import validate_output_path

ROOT = Path(__file__).resolve().parents[2]
LONG_FORM_QUALITY_IMPLEMENTATION_VERSION = 1
DEFAULT_QUALITY_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "quality_v1.yaml"
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
DNSMOS_SAMPLING_RATE = 16000
DNSMOS_INPUT_SECONDS = 9.01


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or not path.parts
        or ".." in path.parts
    ):
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class QualityModels(_StrictFrozenModel):
    """Pinned DNSMOS P.835 ONNX artifacts — drift aborts the run."""

    model_root: str
    primary_relative_path: str
    primary_sha256: str = Field(pattern=SHA256_PATTERN)
    p808_relative_path: str
    p808_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("model_root", "primary_relative_path", "p808_relative_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class QualityThresholds(_StrictFrozenModel):
    """Uncalibrated until LF-12G fixes them on the dev split — evidence only."""

    dfn3_min_ovrl_gain: float | None = None
    dfn3_min_sig_gain: float | None = None
    calibrated: bool = False


class QualityPaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "QualityPaths":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError(
                    "quality outputs cannot be written under data/inbox"
                )
        return self


class QualityConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    models: QualityModels
    thresholds: QualityThresholds
    paths: QualityPaths

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_quality_config(
    path: str | Path = DEFAULT_QUALITY_CONFIG_PATH,
) -> QualityConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(
            f"quality configuration must be a YAML mapping: {config_path}"
        )
    return QualityConfig.model_validate(payload, strict=True)


def _polyfit_nonpersonalized(
    sig_raw: float, bak_raw: float, ovr_raw: float
) -> tuple[float, float, float]:
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return float(p_sig(sig_raw)), float(p_bak(bak_raw)), float(p_ovr(ovr_raw))


class DnsmosBackend:
    """DNSMOS P.835 scorer matching microsoft/DNS-Challenge dnsmos_local.py.

    Clips shorter than 9.01 s are self-loop padded exactly like upstream;
    scores on such clips carry ``padded: true`` and the region report marks
    them ``short_clip`` so padding sensitivity is visible, not hidden.
    """

    def __init__(self, config: QualityModels) -> None:
        import onnxruntime as ort

        root = (ROOT / config.model_root).resolve()
        for label, rel, expected in (
            ("primary", config.primary_relative_path, config.primary_sha256),
            ("p808", config.p808_relative_path, config.p808_sha256),
        ):
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(f"DNSMOS {label} model missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"DNSMOS {label} sha256 drift: {actual} != {expected}"
                )
        self._primary = ort.InferenceSession(
            str(root / config.primary_relative_path),
            providers=["CPUExecutionProvider"],
        )
        self._p808 = ort.InferenceSession(
            str(root / config.p808_relative_path),
            providers=["CPUExecutionProvider"],
        )
        self._len = int(DNSMOS_INPUT_SECONDS * DNSMOS_SAMPLING_RATE)

    @staticmethod
    def _melspec(audio: np.ndarray) -> np.ndarray:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio, sr=DNSMOS_SAMPLING_RATE, n_fft=321, hop_length=160,
            n_mels=120,
        )
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel.T

    def score(self, audio: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
        clip = np.asarray(audio, dtype=np.float32).reshape(-1)
        if clip.size == 0:
            raise ValueError("cannot score empty audio")
        if sample_rate != DNSMOS_SAMPLING_RATE:
            clip = np.asarray(
                soxr.resample(clip, sample_rate, DNSMOS_SAMPLING_RATE, quality="HQ"),
                dtype=np.float32,
            )
        actual_seconds = clip.size / DNSMOS_SAMPLING_RATE
        padded = clip.size < self._len
        while clip.size < self._len:
            clip = np.append(clip, clip)
        num_hops = int(np.floor(clip.size / DNSMOS_SAMPLING_RATE) - DNSMOS_INPUT_SECONDS) + 1
        sig_raw: list[float] = []
        bak_raw: list[float] = []
        ovr_raw: list[float] = []
        p808: list[float] = []
        for idx in range(num_hops):
            seg = clip[
                idx * DNSMOS_SAMPLING_RATE:
                int((idx + DNSMOS_INPUT_SECONDS) * DNSMOS_SAMPLING_RATE)
            ]
            if seg.size < self._len:
                continue
            features = seg.astype(np.float32)[None, :]
            p808_features = self._melspec(seg[:-160]).astype(np.float32)[None, :, :]
            mos_sig, mos_bak, mos_ovr = self._primary.run(
                None, {"input_1": features}
            )[0][0]
            p808_mos = self._p808.run(None, {"input_1": p808_features})[0][0][0]
            sig_raw.append(float(mos_sig))
            bak_raw.append(float(mos_bak))
            ovr_raw.append(float(mos_ovr))
            p808.append(float(p808_mos))
        sig, bak, ovr = _polyfit_nonpersonalized(
            float(np.mean(sig_raw)), float(np.mean(bak_raw)), float(np.mean(ovr_raw))
        )
        return {
            "sig": sig,
            "bak": bak,
            "ovr": ovr,
            "p808": float(np.mean(p808)),
            "sig_raw": float(np.mean(sig_raw)),
            "bak_raw": float(np.mean(bak_raw)),
            "ovr_raw": float(np.mean(ovr_raw)),
            "num_hops": num_hops,
            "clip_seconds": round(actual_seconds, 3),
            "padded": padded,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(rate)


def _read_source_range(
    source: Path, start_frame: int, end_frame: int
) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(source), mode="r") as audio_file:
        audio_file.seek(start_frame)
        block = audio_file.read(end_frame - start_frame, dtype="float32", always_2d=True)
    return block.mean(axis=1).astype(np.float32), int(audio_file.samplerate)


def score_region_pair(
    region: Mapping[str, Any],
    *,
    source_path: Path,
    run_dir: Path,
    backend: DnsmosBackend,
    risk_scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one routed region: raw source range vs routed output.

    Deltas measure in-domain gain of the chosen route over the untouched
    source. Event risk scores (when supplied) are joined so noise gain can be
    separated from Foley damage — a BAK gain with a SIG loss on an
    event-heavy region reads as enhancement damaging content, not cleaning
    noise.
    """
    start = int(region["source_start_frame"])
    end = int(region["source_end_frame"])
    raw_audio, raw_rate = _read_source_range(source_path, start, end)
    raw_scores = backend.score(raw_audio, sample_rate=raw_rate)

    row: dict[str, Any] = {
        "region_index": int(region["region_index"]),
        "route": region.get("route"),
        "source_start_frame": start,
        "source_end_frame": end,
        "duration_seconds": round((end - start) / raw_rate, 3),
        "raw_scores": raw_scores,
        "short_clip": bool(raw_scores["padded"]),
    }
    output = region.get("output") or {}
    rel = output.get("relative_path")
    if region.get("status") == "completed" and rel:
        out_audio, out_rate = _read_mono((run_dir / rel).resolve())
        out_scores = backend.score(out_audio, sample_rate=out_rate)
        row["routed_scores"] = out_scores
        row["delta"] = {
            key: round(out_scores[key] - raw_scores[key], 4)
            for key in ("sig", "bak", "ovr", "p808")
        }
        row["output_sha256_verified"] = (
            file_sha256((run_dir / rel).resolve()) == output.get("sha256")
        )
    if risk_scores:
        row["event_risk_max"] = max(
            (
                float(entry["max"])
                for entry in risk_scores.values()
                if isinstance(entry, Mapping) and entry.get("max") is not None
            ),
            default=None,
        )
    return row


def run_quality_acceptance(
    source_path: str | Path,
    routes_manifest: Mapping[str, Any],
    *,
    config: QualityConfig,
    backend: DnsmosBackend | None = None,
    event_risk_by_region: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score every routed region of one source and write the report."""
    source = Path(source_path).resolve()
    source_hash = file_sha256(source)
    if routes_manifest.get("source_sha256") != source_hash:
        raise RuntimeError("routes manifest belongs to a different source")

    report_root = validate_output_path(ROOT / config.paths.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    backend = backend or DnsmosBackend(config.models)
    rows: list[dict[str, Any]] = []
    run_dir = Path(routes_manifest.get("run_dir", "")).resolve()
    for region in routes_manifest.get("regions", []):
        rows.append(
            score_region_pair(
                region,
                source_path=source,
                run_dir=run_dir,
                backend=backend,
                risk_scores=(
                    (event_risk_by_region or {}).get(int(region["region_index"]))
                    or {}
                ).get("risk_scores"),
            )
        )

    scored = [r for r in rows if "delta" in r]
    summary = {
        "region_count": len(rows),
        "scored_pairs": len(scored),
        "short_clip_fraction": (
            round(sum(1 for r in scored if r["short_clip"]) / len(scored), 4)
            if scored
            else None
        ),
    }
    for route in ("raw", "dfn3_denoised"):
        subset = [r for r in scored if r["route"] == route]
        if subset:
            summary[f"{route}_mean_delta"] = {
                key: round(
                    float(np.mean([r["delta"][key] for r in subset])), 4
                )
                for key in ("sig", "bak", "ovr", "p808")
            }
    eventy = [
        r for r in scored
        if r.get("event_risk_max") is not None and r["event_risk_max"] >= 0.5
    ]
    if eventy:
        summary["event_risky_mean_delta"] = {
            key: round(
                float(np.mean([r["delta"][key] for r in eventy])), 4
            )
            for key in ("sig", "bak", "ovr")
        }

    report = {
        "schema_version": 1,
        "implementation_version": LONG_FORM_QUALITY_IMPLEMENTATION_VERSION,
        "source_sha256": source_hash,
        "config_sha256": config.config_sha256(),
        "thresholds_calibrated": config.thresholds.calibrated,
        "generated_at": _now(),
        "summary": summary,
        "regions": rows,
    }
    report_path = report_root / f"quality_{source_hash[:12]}.json"
    _atomic_write_json(report_path, report)
    return {
        "status": "completed",
        "source_sha256": source_hash,
        "report_path": str(report_path),
        "summary": summary,
    }
