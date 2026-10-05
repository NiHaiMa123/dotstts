from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRICT_GATE_CONFIG_PATH = (
    ROOT / "configs" / "lab" / "long_form" / "strict_gate_v1.yaml"
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"

GateId = Literal["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
CONTENT_GATES: tuple[GateId, ...] = (
    "G0",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
)
ALL_GATES: tuple[GateId, ...] = CONTENT_GATES + ("G9",)

GateStatus = Literal["pass", "fail", "unknown"]
EnhancementRoute = Literal["raw", "dfn3_denoised"]
StrictDisposition = Literal[
    "strict_candidate",
    "auto_verified",
    "reserve_budget",
    "quarantine",
    "reject",
    "calibration_sample",
]
GateMode = Literal["calibration", "production"]
CertificationStatus = Literal["none", "pending", "certified"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or not path.parts or ".." in path.parts:
        raise ValueError("paths must be non-empty repository-relative paths")
    return path.as_posix().rstrip("/")


class SourceSpan(_StrictFrozenModel):
    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceSpan":
        if self.source_end_frame <= self.source_start_frame:
            raise ValueError("source_end_frame must exceed source_start_frame")
        return self


class GateVerdict(_StrictFrozenModel):
    schema_version: Literal[1]
    gate: GateId
    status: GateStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    evaluator: str = Field(min_length=1)
    calibrated: bool = True
    recorded_at: str = Field(min_length=1)


class CandidateBinding(_StrictFrozenModel):
    """Everything a decision binds to; any change invalidates old approvals."""

    schema_version: Literal[1]
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sample_rate_hz: int = Field(gt=0)
    source_spans: list[SourceSpan] = Field(min_length=1)
    final_audio_sha256: str = Field(pattern=SHA256_PATTERN)
    enhancement_route: EnhancementRoute
    gate_config_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_pack_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    calibration_report_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    text_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ApprovalBinding(_StrictFrozenModel):
    """A stored human or automatic approval that may only attach to an identical binding."""

    schema_version: Literal[1]
    decision: Literal["human_confirmed", "auto_verified"]
    batch_or_report_sha256: str = Field(pattern=SHA256_PATTERN)
    recorded_at: str = Field(min_length=1)
    bound: CandidateBinding


class ThresholdSpec(_StrictFrozenModel):
    value: float | None = None
    calibrated: bool = False


class BatchBudget(_StrictFrozenModel):
    total_candidates: int = Field(ge=1, le=24)
    per_source: int = Field(ge=1, le=6)
    calibration_batch_max: int = Field(ge=1, le=6)


class CertificationPolicy(_StrictFrozenModel):
    status: CertificationStatus
    report_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    report_path: str | None = None

    @field_validator("report_path")
    @classmethod
    def validate_report_path(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_relative_path(value)
        return value

    @model_validator(mode="after")
    def validate_certification(self) -> "CertificationPolicy":
        if self.status == "certified" and (self.report_sha256 is None or self.report_path is None):
            raise ValueError("certified status requires report_sha256 and report_path")
        if self.status != "certified" and self.report_sha256 is not None:
            raise ValueError("report_sha256 is only allowed once certification is certified")
        return self


class StrictGateThresholds(_StrictFrozenModel):
    identity: ThresholdSpec
    normal: ThresholdSpec
    event: dict[str, ThresholdSpec] = Field(default_factory=dict)
    noise: ThresholdSpec
    quality: ThresholdSpec
    alignment: ThresholdSpec

    def uncalibrated_names(self) -> list[str]:
        names: list[str] = []
        for name in ("identity", "normal", "noise", "quality", "alignment"):
            spec: ThresholdSpec = getattr(self, name)
            if not spec.calibrated or spec.value is None:
                names.append(name)
        for name, spec in sorted(self.event.items()):
            if not spec.calibrated or spec.value is None:
                names.append(f"event.{name}")
        return names


class StrictGateConfig(_StrictFrozenModel):
    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    mode: GateMode
    required_gates: list[GateId]
    batch_budget: BatchBudget
    thresholds: StrictGateThresholds
    certification: CertificationPolicy
    paths: "StrictGatePaths"

    @field_validator("required_gates")
    @classmethod
    def validate_required_gates(cls, value: list[GateId]) -> list[GateId]:
        if sorted(set(value)) != sorted(ALL_GATES) or len(value) != len(ALL_GATES):
            raise ValueError("required_gates must list every gate G0-G9 exactly once")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "StrictGateConfig":
        if self.mode == "production" and self.thresholds.uncalibrated_names():
            raise ValueError(
                "production mode requires every threshold to be calibrated: "
                + ", ".join(self.thresholds.uncalibrated_names())
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


class StrictGatePaths(_StrictFrozenModel):
    work_root: str
    report_root: str

    @field_validator("work_root", "report_root")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_relative_path(value)


class GateOutcome(_StrictFrozenModel):
    disposition: StrictDisposition
    failed_gates: list[GateId] = Field(default_factory=list)
    unknown_gates: list[GateId] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _verdict(verdicts: Mapping[str, GateVerdict], gate: GateId) -> GateVerdict | None:
    value = verdicts.get(gate)
    if value is not None and value.gate != gate:
        raise ValueError(f"verdict stored under {gate} declares gate {value.gate}")
    return value


def evaluate_candidate(
    verdicts: Mapping[str, GateVerdict],
    config: StrictGateConfig,
) -> GateOutcome:
    """Combine G0-G9 verdicts into one disposition.

    A hard fail rejects the candidate regardless of other gates' scores; missing
    or unknown evidence quarantines it. Content passes only reach reviewable
    states once every required content gate has a passing, calibrated verdict.
    """
    failed: list[GateId] = []
    unknown: list[GateId] = []
    reasons: list[str] = []
    for gate in CONTENT_GATES:
        verdict = _verdict(verdicts, gate)
        if verdict is None:
            unknown.append(gate)
            reasons.append(f"{gate}: required evidence missing")
            continue
        if verdict.status == "fail":
            failed.append(gate)
            reasons.extend(f"{gate}: {reason}" for reason in verdict.reasons)
        elif verdict.status == "unknown":
            unknown.append(gate)
            reasons.extend(
                f"{gate}: {reason}" for reason in verdict.reasons or ["evidence insufficient"]
            )
        elif not verdict.calibrated and config.mode == "production":
            unknown.append(gate)
            reasons.append(f"{gate}: uncalibrated evaluator cannot pass in production mode")
    if failed:
        return GateOutcome(
            disposition="reject",
            failed_gates=failed,
            unknown_gates=unknown,
            reasons=reasons,
        )
    if unknown:
        return GateOutcome(
            disposition="quarantine", unknown_gates=unknown, reasons=reasons
        )
    g9 = _verdict(verdicts, "G9")
    if g9 is None or g9.status != "pass" or (
        not g9.calibrated and config.mode == "production"
    ):
        reason = "G9: certification evidence missing" if g9 is None else "; ".join(
            f"G9: {reason}" for reason in g9.reasons or ["domain evidence invalid"]
        )
        return GateOutcome(disposition="quarantine", unknown_gates=["G9"], reasons=[reason])
    if config.mode == "calibration":
        return GateOutcome(
            disposition="calibration_sample",
            reasons=["all content gates passed under calibration mode"],
        )
    if config.certification.status == "certified":
        return GateOutcome(
            disposition="auto_verified",
            reasons=[f"certified by report {config.certification.report_sha256}"],
        )
    return GateOutcome(
        disposition="strict_candidate",
        reasons=["content gates passed; joint certification pending"],
    )


def apply_batch_budget(
    ranked_candidate_ids: Sequence[str],
    source_of: Mapping[str, str],
    budget: BatchBudget,
) -> dict[str, StrictDisposition]:
    """Assign strict_candidate vs reserve_budget under the batch caps.

    Input order is the caller's diversity/quality ranking; the caps never pull
    quarantined or rejected items back into review.
    """
    assignments: dict[str, StrictDisposition] = {}
    per_source_count: dict[str, int] = {}
    total = 0
    for candidate_id in ranked_candidate_ids:
        source = source_of[candidate_id]
        if total < budget.total_candidates and (
            per_source_count.get(source, 0) < budget.per_source
        ):
            assignments[candidate_id] = "strict_candidate"
            per_source_count[source] = per_source_count.get(source, 0) + 1
            total += 1
        else:
            assignments[candidate_id] = "reserve_budget"
    return assignments


def verify_approval_binding(
    approval: ApprovalBinding, candidate: CandidateBinding
) -> list[str]:
    """Return mismatch reasons; empty means the approval may attach to this candidate.

    Any change to audio, spans, route, gate config, reference pack, calibration
    report, or text prevents an old approval from migrating silently.
    """
    mismatches: list[str] = []
    old = approval.bound.model_dump(mode="json")
    new = candidate.model_dump(mode="json")
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            mismatches.append(key)
    if approval.decision == "auto_verified" and (
        candidate.calibration_report_sha256 is None
        or candidate.calibration_report_sha256 != approval.batch_or_report_sha256
    ):
        if "calibration_report_sha256" not in mismatches:
            mismatches.append("calibration_report_sha256")
    return mismatches


def load_strict_gate_config(
    path: str | Path = DEFAULT_STRICT_GATE_CONFIG_PATH,
) -> StrictGateConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"strict gate configuration must be a YAML mapping: {config_path}")
    return StrictGateConfig.model_validate(payload, strict=True)
