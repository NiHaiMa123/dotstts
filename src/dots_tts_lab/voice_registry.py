from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOICE_REGISTRY_PATH = ROOT / "configs/voices/registry.yaml"


class BaseModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    revision: str | None = None


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    format: Literal["dots_tts_trainable_delta"]
    training_step: int = Field(ge=1)
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    audio_path: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1)


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    precision: Literal["float32", "float16", "bfloat16"]
    optimize: bool
    max_generate_length: int = Field(ge=1, le=4096)
    max_sequence_length: int = Field(ge=1, le=8192)
    vocoder_merge_steps: int = Field(ge=1, le=32)
    warmup_on_optimize: bool
    merge_lora: bool


class GenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    language: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    normalize_text: bool
    soften_emphasis: bool = False
    speaker_scale: float = Field(ge=0.0, le=5.0)
    ode_method: str = Field(min_length=1)
    num_steps: int = Field(ge=1, le=100)
    guidance_scale: float = Field(ge=0.0, le=10.0)
    base_seed: int = Field(ge=0)
    pause_ms: int = Field(ge=0, le=5000)
    max_chars: int = Field(ge=20, le=1000)


class PostprocessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    edge_trim_config: str
    edge_trim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    voice_polish_config: str
    voice_polish_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IOSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    input_dir: str
    output_dir: str


class VoiceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    profile_version: int = Field(ge=1)
    model_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    display_name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    base_model: BaseModelSpec
    adapter: AdapterSpec | None = None
    prompt: PromptSpec
    runtime: RuntimeSpec
    generation: GenerationSpec
    postprocess: PostprocessSpec
    io: IOSpec

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> VoiceProfile:
        if self.runtime.warmup_on_optimize and not self.runtime.optimize:
            raise ValueError("warmup_on_optimize requires optimize=true")
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VoiceRegistrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    registry_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    registry_version: int = Field(ge=1)
    default_model_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    profiles: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class LoadedVoiceRegistry:
    root: Path
    spec: VoiceRegistrySpec
    profiles: dict[str, VoiceProfile]

    @property
    def default_model_id(self) -> str:
        return self.spec.default_model_id

    def get(self, model_id: str) -> VoiceProfile:
        try:
            return self.profiles[model_id]
        except KeyError as error:
            raise KeyError(f"Unknown voice model: {model_id}") from error

    def path(self, value: str) -> Path:
        resolved = (self.root / value).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"Voice profile path escapes the project root: {value}") from error
        return resolved

    def public_models(self) -> list[dict[str, object]]:
        return [
            {
                "id": profile.model_id,
                "label": profile.display_name,
                "description": profile.description,
                "profile_version": profile.profile_version,
                "profile_sha256": profile.canonical_sha256(),
                "input_dir": profile.io.input_dir,
                "output_dir": profile.io.output_dir,
            }
            for profile in self.profiles.values()
        ]

    def output_dirs(self) -> list[Path]:
        return [self.path(profile.io.output_dir) for profile in self.profiles.values()]

    def input_dirs(self) -> list[Path]:
        return [self.path(profile.io.input_dir) for profile in self.profiles.values()]


def _read_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} checksum mismatch: expected {expected_sha256}, got {actual}"
        )


def validate_voice_profile_artifacts(
    registry: LoadedVoiceRegistry,
    profile: VoiceProfile,
) -> None:
    base_model = registry.path(profile.base_model.path)
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model is missing: {base_model}")
    if profile.adapter is not None:
        adapter = registry.path(profile.adapter.path)
        if not adapter.is_dir():
            raise FileNotFoundError(f"Adapter is missing: {adapter}")
        _require_file(
            adapter / "trainable_model.safetensors",
            profile.adapter.weights_sha256,
            "Adapter weights",
        )
        _require_file(
            adapter / "trainable_model.json",
            profile.adapter.metadata_sha256,
            "Adapter metadata",
        )
    _require_file(
        registry.path(profile.prompt.audio_path),
        profile.prompt.audio_sha256,
        "Prompt audio",
    )
    _require_file(
        registry.path(profile.postprocess.edge_trim_config),
        profile.postprocess.edge_trim_sha256,
        "Edge-trim config",
    )
    _require_file(
        registry.path(profile.postprocess.voice_polish_config),
        profile.postprocess.voice_polish_sha256,
        "Voice-polish config",
    )


def load_voice_registry(
    path: str | Path = DEFAULT_VOICE_REGISTRY_PATH,
    *,
    project_root: str | Path = ROOT,
    validate_artifacts: bool = True,
) -> LoadedVoiceRegistry:
    root = Path(project_root).expanduser().resolve()
    registry_path = Path(path).expanduser().resolve()
    spec = VoiceRegistrySpec.model_validate(_read_yaml(registry_path), strict=True)
    profiles: dict[str, VoiceProfile] = {}
    for relative in spec.profiles:
        profile_path = (root / relative).resolve()
        try:
            profile_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Profile path escapes the project root: {relative}") from error
        profile = VoiceProfile.model_validate(_read_yaml(profile_path), strict=True)
        if profile.model_id in profiles:
            raise ValueError(f"Duplicate voice model id: {profile.model_id}")
        profiles[profile.model_id] = profile
    if spec.default_model_id not in profiles:
        raise ValueError(f"Default model is not registered: {spec.default_model_id}")
    loaded = LoadedVoiceRegistry(root=root, spec=spec, profiles=profiles)
    if validate_artifacts:
        for profile in profiles.values():
            validate_voice_profile_artifacts(loaded, profile)
    return loaded
