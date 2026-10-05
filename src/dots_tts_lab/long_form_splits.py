from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_CONFIG_PATH = ROOT / "configs" / "lab" / "long_form" / "split_v1.yaml"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SplitName = Literal["reference", "development", "holdout"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceAssignment(_StrictFrozenModel):
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_relative_path: str = Field(min_length=1)
    split: SplitName
    reason: str = Field(min_length=1)


class SplitConfig(_StrictFrozenModel):
    """Whole-source isolation between reference, development and holdout sets.

    When a source is too scarce for a dedicated split the assignment may fall
    back to non-adjacent time blocks, but the reason must say so explicitly so
    the extrapolation limit is recorded rather than implied.
    """

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    assignments: list[SourceAssignment] = Field(min_length=1)
    notes: str = Field(default="")

    @model_validator(mode="after")
    def validate_assignments(self) -> "SplitConfig":
        seen: dict[str, SplitName] = {}
        for assignment in self.assignments:
            previous = seen.get(assignment.source_sha256)
            if previous is not None:
                raise ValueError(
                    f"source {assignment.source_sha256[:12]} assigned to both "
                    f"{previous} and {assignment.split}"
                )
            seen[assignment.source_sha256] = assignment.split
        splits = {assignment.split for assignment in self.assignments}
        if "holdout" not in splits or "development" not in splits:
            raise ValueError("split config needs at least one development and one holdout source")
        return self

    def split_of(self, source_sha256: str) -> SplitName:
        for assignment in self.assignments:
            if assignment.source_sha256 == source_sha256:
                return assignment.split
        raise KeyError(f"source not covered by split config: {source_sha256}")


def load_split_config(path: str | Path = DEFAULT_SPLIT_CONFIG_PATH) -> SplitConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"split configuration must be a YAML mapping: {config_path}")
    return SplitConfig.model_validate(payload, strict=True)


class SplitRecord(_StrictFrozenModel):
    """Minimal evidence a candidate carries into the isolation check."""

    candidate_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    text_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    pair_group: str | None = None


def check_split_isolation(
    records: list[SplitRecord],
    config: SplitConfig,
    *,
    adjacency_seconds: float = 30.0,
) -> list[str]:
    """List every pair of records that illegally crosses a split boundary.

    Adjacent slices from one source, raw/denoised pairs sharing a pair_group,
    and duplicate normalized texts must never land in different splits;
    otherwise the holdout stops measuring generalization.
    """
    violations: list[str] = []
    ordered = sorted(records, key=lambda record: record.candidate_id)
    for index, left in enumerate(ordered):
        left_split = config.split_of(left.source_sha256)
        for right in ordered[index + 1 :]:
            right_split = config.split_of(right.source_sha256)
            if left_split == right_split:
                continue
            reasons: list[str] = []
            if left.source_sha256 == right.source_sha256:
                gap_frames = max(
                    right.source_start_frame - left.source_end_frame,
                    left.source_start_frame - right.source_end_frame,
                )
                overlaps = not (
                    right.source_start_frame >= left.source_end_frame
                    or left.source_start_frame >= right.source_end_frame
                )
                if overlaps or gap_frames <= adjacency_seconds * left.source_sample_rate_hz:
                    reasons.append("adjacent_or_overlapping_spans")
            if left.pair_group and left.pair_group == right.pair_group:
                reasons.append("paired_raw_enhanced_group")
            if (
                left.text_sha256
                and left.text_sha256 == right.text_sha256
            ):
                reasons.append("duplicate_text")
            if reasons:
                violations.append(
                    f"{left.candidate_id[:16]}({left_split}) <-> "
                    f"{right.candidate_id[:16]}({right_split}): {','.join(sorted(set(reasons)))}"
                )
    return violations
