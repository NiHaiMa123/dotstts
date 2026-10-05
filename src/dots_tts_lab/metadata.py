from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_METADATA_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "metadata_profiles"
    / "speaker_emotion_filename_v1.yaml"
)

_DIRECTORY_SELECTOR = re.compile(r"^level_(?P<level>[1-9][0-9]*)_directory$")


class MetadataProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    profile_version: int = Field(ge=1)
    speaker_from: str
    emotion_from: str | None
    filename_pattern: str
    filename_emotion_group: str = "emotion"
    transcript_group: str = "text"
    trim_values: bool = True
    conflict_policy: Literal["review_required"] = "review_required"

    @field_validator("speaker_from", "emotion_from")
    @classmethod
    def validate_directory_selector(cls, value: str | None) -> str | None:
        if value is not None and _DIRECTORY_SELECTOR.fullmatch(value) is None:
            raise ValueError(
                "directory selector must look like level_1_directory"
            )
        return value

    @model_validator(mode="after")
    def validate_filename_pattern(self) -> MetadataProfile:
        try:
            pattern = re.compile(self.filename_pattern)
        except re.error as error:
            raise ValueError(f"invalid filename_pattern: {error}") from error
        required_groups = {self.transcript_group}
        if self.emotion_from is not None:
            required_groups.add(self.filename_emotion_group)
        missing = sorted(required_groups - set(pattern.groupindex))
        if missing:
            raise ValueError(
                "filename_pattern is missing named groups: " + ", ".join(missing)
            )
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_metadata_profile(
    path: str | Path = DEFAULT_METADATA_PROFILE_PATH,
) -> MetadataProfile:
    profile_path = Path(path).resolve()
    with profile_path.open("r", encoding="utf-8") as profile_file:
        payload = yaml.safe_load(profile_file)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Metadata profile must contain a YAML mapping: {profile_path}"
        )
    return MetadataProfile.model_validate(payload, strict=True)


def _directory_value(relative_path: Path, selector: str | None) -> str | None:
    if selector is None:
        return None
    match = _DIRECTORY_SELECTOR.fullmatch(selector)
    assert match is not None
    directory_parts = relative_path.parts[:-1]
    index = int(match.group("level")) - 1
    return directory_parts[index] if index < len(directory_parts) else None


def _clean(value: str | None, *, trim: bool) -> str | None:
    if value is None:
        return None
    cleaned = value.strip() if trim else value
    return cleaned or None


def parse_metadata(
    relative_path: Path, *, profile: MetadataProfile
) -> dict[str, Any]:
    speaker_id = _clean(
        _directory_value(relative_path, profile.speaker_from),
        trim=profile.trim_values,
    )
    directory_emotion = _clean(
        _directory_value(relative_path, profile.emotion_from),
        trim=profile.trim_values,
    )
    pattern = re.compile(profile.filename_pattern)
    match = pattern.fullmatch(relative_path.stem)
    filename_emotion = _clean(
        (
            None
            if match is None
            else match.groupdict().get(profile.filename_emotion_group)
        ),
        trim=profile.trim_values,
    )
    transcript = _clean(
        None if match is None else match.groupdict().get(profile.transcript_group),
        trim=profile.trim_values,
    )

    errors: list[str] = []
    reviews: list[str] = []
    if speaker_id is None:
        errors.append(f"missing speaker from {profile.speaker_from}")
    if profile.emotion_from is not None and directory_emotion is None:
        errors.append(f"missing emotion from {profile.emotion_from}")
    if match is None:
        errors.append("filename does not match configured pattern")
    elif transcript is None:
        errors.append("filename transcript is empty")
    if (
        directory_emotion is not None
        and filename_emotion is not None
        and directory_emotion != filename_emotion
    ):
        reviews.append(
            "directory emotion does not match filename emotion: "
            f"{directory_emotion!r} != {filename_emotion!r}"
        )

    metadata_status = (
        "error" if errors else "review_required" if reviews else "ok"
    )
    return {
        "speaker_id": speaker_id,
        "emotion_weak_label": directory_emotion or filename_emotion,
        "directory_emotion_weak_label": directory_emotion,
        "filename_emotion_weak_label": filename_emotion,
        "transcript_candidate": transcript,
        "parse_status": "error" if errors else "ok",
        "parse_error": "; ".join(errors) if errors else None,
        "metadata_status": metadata_status,
        "metadata_review_reason": "; ".join(reviews) if reviews else None,
        "parser_profile_id": profile.profile_id,
        "parser_profile_version": profile.profile_version,
        "parser_config_sha256": profile.config_sha256(),
    }
