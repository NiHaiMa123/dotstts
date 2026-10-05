from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DFN3_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "deepfilternet3_v2.yaml"
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class Dfn3BackendConfig(_StrictFrozenModel):
    name: Literal["deepfilternet"]
    package_version: Literal["0.5.6"]
    model_name: Literal["DeepFilterNet3"]
    attenuation_limit_db: float = Field(gt=0.0, le=20.0)
    post_filter: Literal[False]
    compensate_delay: Literal[True]
    expected_image_tag: str = Field(min_length=1)


class Dfn3ModelFiles(_StrictFrozenModel):
    """Pinned model fingerprint; a hash drift is an error, never a silent swap."""

    cache_root: str
    config_relative_path: str
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    weights_relative_path: str
    weights_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("cache_root", "config_relative_path", "weights_relative_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class Dfn3Chunking(_StrictFrozenModel):
    core_max_seconds: float = Field(gt=0.0, le=30.0)
    context_max_seconds: float = Field(ge=0.0, le=2.0)
    reset_state_per_window: Literal[True]
    model_reuse_across_windows: Literal[True]
    seam_policy: Literal["reprocess_sentence_else_quarantine"]


class Dfn3Channels(_StrictFrozenModel):
    spatial_gate_before_downmix: Literal[True]
    downmix_rule: Literal["safe_mean_only"]
    preserve_lr_evidence: Literal[True]


class Dfn3Output(_StrictFrozenModel):
    work_root: str
    report_root: str
    final_sample_rate_hz: Literal[48000]
    final_subtype: Literal["PCM_24"]
    final_channels: Literal[1]

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def keep_outputs_outside_inbox(self) -> "Dfn3Output":
        inbox = PurePosixPath("data/inbox")
        for output in (self.work_root, self.report_root):
            output_path = PurePosixPath(output)
            if output_path == inbox or inbox in output_path.parents:
                raise ValueError("dfn3 outputs cannot be written under data/inbox")
        return self


class Dfn3FailurePolicy(_StrictFrozenModel):
    on_backend_failure: Literal["block_candidate"]
    on_missing_weights: Literal["abort_run"]
    on_output_mismatch: Literal["block_candidate"]
    atomic_region_commit: Literal[True]


class Dfn3ProductionConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    backend: Dfn3BackendConfig
    model_files: Dfn3ModelFiles
    chunking: Dfn3Chunking
    channels: Dfn3Channels
    output: Dfn3Output
    failure_policy: Dfn3FailurePolicy

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_dfn3_config(path: str | Path = DEFAULT_DFN3_CONFIG_PATH) -> Dfn3ProductionConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"dfn3 production config must be a YAML mapping: {config_path}")
    return Dfn3ProductionConfig.model_validate(payload, strict=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Dfn3VerificationError(RuntimeError):
    pass


def verify_model_files(config: Dfn3ProductionConfig) -> dict[str, Any]:
    cache_root = (ROOT / config.model_files.cache_root).resolve()
    report: dict[str, Any] = {"cache_root": str(cache_root), "files": {}}
    problems: list[str] = []
    for label, relative, expected in (
        ("config", config.model_files.config_relative_path, config.model_files.config_sha256),
        ("weights", config.model_files.weights_relative_path, config.model_files.weights_sha256),
    ):
        path = (cache_root / relative).resolve()
        entry: dict[str, Any] = {"path": str(path), "expected_sha256": expected}
        if not path.is_file():
            entry["status"] = "missing"
            problems.append(f"{label} file missing: {path}")
        else:
            actual = _file_sha256(path)
            entry["actual_sha256"] = actual
            if actual != expected:
                entry["status"] = "hash_mismatch"
                problems.append(f"{label} sha256 mismatch: {actual} != {expected}")
            else:
                entry["status"] = "verified"
        report["files"][label] = entry
    report["status"] = "verified" if not problems else "failed"
    if problems:
        raise Dfn3VerificationError("; ".join(problems))
    return report


def verify_backend(
    config: Dfn3ProductionConfig,
    *,
    package_version: str | None = None,
    image_tag: str | None = None,
) -> dict[str, Any]:
    """Verify the pinned backend before any production run.

    Weight/config hashes must match exactly. The package check is skipped when
    deepfilternet is not importable in the current environment (production runs
    inside the dedicated image); when a version is supplied it must equal the
    pinned requirement.
    """
    report = verify_model_files(config)
    report["backend"] = config.backend.model_dump(mode="json")
    if package_version is not None:
        report["package_version"] = package_version
        if package_version != config.backend.package_version:
            raise Dfn3VerificationError(
                f"deepfilternet {package_version} != pinned {config.backend.package_version}"
            )
    else:
        report["package_version"] = "not_checked_in_this_env"
    if image_tag is not None:
        report["image_tag"] = image_tag
        if image_tag != config.backend.expected_image_tag:
            raise Dfn3VerificationError(
                f"image {image_tag} != pinned {config.backend.expected_image_tag}"
            )
    report["status"] = "verified"
    return report


def compute_run_hash(
    config: Dfn3ProductionConfig,
    *,
    source_sha256: str,
    core_start_frame: int,
    core_end_frame: int,
    channel_route: str,
    implementation_version: int,
) -> str:
    payload = json.dumps(
        {
            "channel_route": channel_route,
            "config_sha256": config.config_sha256(),
            "core_end_frame": core_end_frame,
            "core_start_frame": core_start_frame,
            "implementation_version": implementation_version,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
