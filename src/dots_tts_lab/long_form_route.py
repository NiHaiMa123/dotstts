from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Literal, Mapping

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.long_form_audio import file_sha256, probe_long_audio
from dots_tts_lab.long_form_dfn3 import Dfn3ProductionConfig
from dots_tts_lab.long_form_dfn3_regions import (
    EnhanceFn,
    enhance_source_regions,
)
from dots_tts_lab.long_form_paths import validate_output_path

ROOT = Path(__file__).resolve().parents[2]
LONG_FORM_ROUTE_IMPLEMENTATION_VERSION = 2
DEFAULT_ROUTE_CONFIG_PATH = ROOT / "configs" / "lab" / "long_form" / "route_v1.yaml"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
RouteName = Literal["raw", "dfn3_denoised", "skipped_prefilter", "blocked_spatial"]


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


class RouteRouting(_StrictFrozenModel):
    """When a usable region needs denoising. ``dfn3_noise_threshold`` is an
    uncalibrated knob — until LF-12G fixes it on the dev split it is evidence
    only, and missing noise evidence keeps the region on the raw route
    (clean source preferred; DFN3 only where justified)."""

    dfn3_noise_threshold: float = Field(ge=0.0, le=1.0)
    threshold_calibrated: bool = False


class RouteOutput(_StrictFrozenModel):
    work_root: str
    final_sample_rate_hz: Literal[48000]
    final_subtype: Literal["PCM_16"]

    @field_validator("work_root")
    @classmethod
    def validate_work_root(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "RouteOutput":
        inbox = PurePosixPath("data/inbox")
        output_path = PurePosixPath(self.work_root)
        if output_path == inbox or inbox in output_path.parents:
            raise ValueError("route outputs cannot be written under data/inbox")
        return self


class RouteConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    routing: RouteRouting
    output: RouteOutput

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_route_config(
    path: str | Path = DEFAULT_ROUTE_CONFIG_PATH,
) -> RouteConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"route configuration must be a YAML mapping: {config_path}")
    return RouteConfig.model_validate(payload, strict=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _atomic_write_bytes(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        writer(partial)
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    def write(target: Path) -> None:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write_bytes(path, write)


def _source_run_dir(work_root: Path, source_sha256: str) -> Path:
    return work_root / source_sha256[:2] / source_sha256


def _noise_max(event_evidence: Mapping[str, Any] | None) -> float | None:
    """Highest stationary-noise score across the configured label groups."""
    if not event_evidence:
        return None
    scores = event_evidence.get("noise_scores")
    if not isinstance(scores, Mapping) or not scores:
        return None
    values = [
        float(entry["max"])
        for entry in scores.values()
        if isinstance(entry, Mapping) and entry.get("max") is not None
    ]
    return max(values) if values else None


def decide_route(
    region: Mapping[str, Any],
    *,
    event_evidence: Mapping[str, Any] | None,
    downmix_allowed: bool,
    source_channels: int,
    config: RouteRouting,
) -> tuple[RouteName, list[str], float | None]:
    """Pick the enhancement route for one region.

    - non-usable prefilter regions are never routed (skipped, not processed);
    - a stereo region without spatial downmix permission is blocked — no
      silent flattening;
    - stationary-noise evidence at/above the threshold routes to DFN3;
    - everything else takes the raw route, including missing noise evidence.
    Returns (route, reasons, noise_max).
    """
    reasons: list[str] = []
    if region.get("status") != "usable":
        return "skipped_prefilter", ["prefilter_not_usable"], None
    if source_channels == 2 and not downmix_allowed:
        return "blocked_spatial", ["stereo_downmix_not_permitted"], None
    if source_channels not in (1, 2):
        return "blocked_spatial", [f"unsupported_channels:{source_channels}"], None
    noise = _noise_max(event_evidence)
    if noise is not None and noise >= config.dfn3_noise_threshold:
        reasons.append(f"noise_max {noise:.3f} >= {config.dfn3_noise_threshold}")
        return "dfn3_denoised", reasons, noise
    if noise is None:
        reasons.append("noise_evidence_missing_default_raw")
    return "raw", reasons, noise


def _mono_route(block: np.ndarray, *, downmix_allowed: bool) -> np.ndarray | None:
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32)
    if block.shape[1] == 2:
        if not downmix_allowed:
            return None
        return block.mean(axis=1, dtype=np.float32)
    return None


def run_routes(
    source_path: str | Path,
    regions: list[dict[str, Any]],
    *,
    config: RouteConfig,
    dfn3_config: Dfn3ProductionConfig,
    enhance_fn: EnhanceFn | None,
    event_evidence_by_region: Mapping[int, Mapping[str, Any]] | None = None,
    spatial_gate: Mapping[int, bool] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Route each region to raw or DFN3 and materialise region audio.

    Raw regions are cut from the source (mono-safe); DFN3 regions are
    delegated to ``enhance_source_regions`` which owns its own atomic
    commits and resume inside ``run_dir/dfn3``. A DFN3 failure aborts the
    run — there is no silent fallback onto the raw route.
    """
    source = Path(source_path).resolve()
    probe = probe_long_audio(source)
    source_hash = file_sha256(source)
    source_rate = int(probe["sample_rate_hz"])
    source_channels = int(probe["channels"])
    if source_rate != config.output.final_sample_rate_hz:
        raise RuntimeError(
            f"source rate {source_rate} != pinned {config.output.final_sample_rate_hz}"
        )

    work_root = validate_output_path(ROOT / config.output.work_root)
    run_dir = (run_dir or _source_run_dir(work_root, source_hash)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    manifest_path = run_dir / "routes.json"
    raw_dir = run_dir / "raw"

    state: dict[str, Any] = {"completed_regions": []}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {"completed_regions": []}
    if state.get("config_sha256") not in (None, config.config_sha256()):
        raise RuntimeError(
            "resume refused: run was produced under a different route config hash"
        )
    done: set[int] = set(state.get("completed_regions", []))

    records: dict[int, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            for row in json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "regions", []
            ):
                records[int(row["region_index"])] = row
        except (OSError, json.JSONDecodeError, KeyError):
            records = {}

    decisions: dict[int, tuple[RouteName, list[str], float | None]] = {}
    dfn3_regions: list[dict[str, Any]] = []
    for region in regions:
        region_index = int(region["region_index"])
        evidence = (event_evidence_by_region or {}).get(region_index)
        downmix_allowed = bool(
            spatial_gate and spatial_gate.get(region_index, False)
        )
        route, reasons, noise = decide_route(
            region,
            event_evidence=evidence,
            downmix_allowed=downmix_allowed,
            source_channels=source_channels,
            config=config.routing,
        )
        decisions[region_index] = (route, reasons, noise)
        if route == "dfn3_denoised":
            dfn3_regions.append(region)

    # DFN3 subset: delegated run owns its commits/resume under run_dir/dfn3.
    dfn3_records: dict[int, dict[str, Any]] = {}
    if dfn3_regions:
        if enhance_fn is None:
            raise RuntimeError(
                "dfn3 route requested but no enhancement backend was provided"
            )
        # Pass every dfn3-routed region, not just the pending ones: the
        # delegated run must also rebuild manifest rows for regions whose
        # output committed before a mid-run kill (state vs records gap).
        if dfn3_regions:
            enhance_source_regions(
                source,
                dfn3_regions,
                config=dfn3_config,
                enhance_fn=enhance_fn,
                spatial_gate=dict(spatial_gate or {}),
                run_state_dir=run_dir / "dfn3",
            )
        dfn3_manifest = run_dir / "dfn3" / "regions.json"
        if dfn3_manifest.is_file():
            for row in json.loads(dfn3_manifest.read_text(encoding="utf-8")).get(
                "regions", []
            ):
                dfn3_records[int(row["region_index"])] = row

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_ROUTE_IMPLEMENTATION_VERSION,
            "completed_regions": sorted(done),
            "started_at": _now(),
        },
    )

    try:
        for region in regions:
            region_index = int(region["region_index"])
            route, reasons, noise = decisions[region_index]
            start = int(region["source_start_frame"])
            end = int(region["source_end_frame"])
            record: dict[str, Any] = {
                "region_index": region_index,
                "source_sha256": source_hash,
                "source_start_frame": start,
                "source_end_frame": end,
                "route": route,
                "route_reasons": reasons,
                "noise_max": noise,
                "threshold_calibrated": config.routing.threshold_calibrated,
            }
            if route in ("skipped_prefilter", "blocked_spatial"):
                record["status"] = route
                records[region_index] = record
                continue
            if region_index in done:
                record["status"] = "completed"
                record["output"] = records.get(region_index, {}).get("output")
                records[region_index] = record
                continue

            if route == "raw":
                out_path = raw_dir / f"region_{region_index:05d}.wav"
                with sf.SoundFile(str(source), mode="r") as audio_file:
                    audio_file.seek(start)
                    block = audio_file.read(
                        end - start, dtype="float32", always_2d=True
                    )
                mono = _mono_route(block, downmix_allowed=bool(
                    spatial_gate and spatial_gate.get(region_index, False)
                ))
                if mono is None:
                    record["status"] = "blocked_spatial"
                    records[region_index] = record
                    continue

                def write_raw(target: Path, data: np.ndarray = mono) -> None:
                    sf.write(
                        str(target),
                        data,
                        config.output.final_sample_rate_hz,
                        subtype=config.output.final_subtype,
                        format="WAV",
                    )

                _atomic_write_bytes(out_path, write_raw)
                record["status"] = "completed"
                record["output"] = {
                    "relative_path": f"raw/{out_path.name}",
                    "sample_rate_hz": config.output.final_sample_rate_hz,
                    "channels": 1,
                    "frames": int(mono.size),
                    "sha256": file_sha256(out_path),
                }
                records[region_index] = record
                done.add(region_index)
            else:  # dfn3_denoised
                dfn3_row = dfn3_records.get(region_index)
                if (
                    dfn3_row is None
                    or dfn3_row.get("status") != "enhanced"
                    or not dfn3_row.get("output")
                ):
                    record["status"] = "failed"
                    record["failure"] = "dfn3_region_not_enhanced"
                    records[region_index] = record
                    raise RuntimeError(
                        f"region {region_index} dfn3 route has no enhanced output"
                    )
                record["status"] = "completed"
                record["output"] = {
                    "relative_path": f"dfn3/{dfn3_row['output']['relative_path']}",
                    "sample_rate_hz": dfn3_row["output"]["sample_rate_hz"],
                    "channels": dfn3_row["output"]["channels"],
                    "frames": dfn3_row["output"]["frames"],
                    "sha256": dfn3_row["output"]["sha256"],
                }
                records[region_index] = record
                done.add(region_index)
            _atomic_write_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "source_sha256": source_hash,
                    "config_sha256": config.config_sha256(),
                    "implementation_version": LONG_FORM_ROUTE_IMPLEMENTATION_VERSION,
                    "completed_regions": sorted(done),
                    "updated_at": _now(),
                },
            )
    finally:
        _atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "source_sha256": source_hash,
                "config_sha256": config.config_sha256(),
                "dfn3_config_sha256": dfn3_config.config_sha256(),
                "implementation_version": LONG_FORM_ROUTE_IMPLEMENTATION_VERSION,
                "regions": [records[key] for key in sorted(records)],
            },
        )

    _atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "status": "completed",
            "source_sha256": source_hash,
            "config_sha256": config.config_sha256(),
            "implementation_version": LONG_FORM_ROUTE_IMPLEMENTATION_VERSION,
            "completed_regions": sorted(done),
            "finished_at": _now(),
        },
    )
    return {
        "status": "completed",
        "source_sha256": source_hash,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "route_counts": {
            name: sum(1 for r in records.values() if r.get("route") == name)
            for name in ("raw", "dfn3_denoised", "skipped_prefilter", "blocked_spatial")
        },
    }
