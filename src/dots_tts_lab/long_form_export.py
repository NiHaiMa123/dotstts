from __future__ import annotations

import hashlib
import json
import os
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dots_tts_lab.long_form_paths import validate_output_path


LONG_FORM_EXPORT_IMPLEMENTATION_VERSION = 1
DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "long_form"
    / "export_suisui_v1.yaml"
)


class LongFormExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    config_version: int = Field(ge=1)
    voice_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    speaker_id: str = Field(min_length=1)
    emotion_primary: str = Field(min_length=1)
    accepted_styles: list[Literal["normal"]]
    output_root: str = Field(min_length=1)

    @field_validator("accepted_styles")
    @classmethod
    def normal_only(cls, value: list[str]) -> list[str]:
        if value != ["normal"]:
            raise ValueError("the default training fragment may contain only normal speech")
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_long_form_export_config(
    path: str | Path = DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH,
) -> LongFormExportConfig:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"long-form export config must be a mapping: {resolved}")
    return LongFormExportConfig.model_validate(payload, strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _normalized_text(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(character)[0] not in {"P", "Z", "C"}
    )


def _load_inputs(
    manifest_path: Path, review_snapshot_path: Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest_bytes = manifest_path.read_bytes()
    snapshot_bytes = review_snapshot_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    snapshot = json.loads(snapshot_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    for field in ("source_sha256", "config_sha256"):
        if snapshot.get(field) != manifest.get(field):
            raise ValueError(f"review snapshot {field} does not match manifest")
    if snapshot.get("manifest_sha256") != manifest_sha256:
        raise ValueError("review snapshot does not match current manifest bytes")
    if snapshot.get("complete") is not True:
        raise RuntimeError("long-form review is incomplete")
    if snapshot.get("schema_version", 1) >= 2:
        segments = {row["segment_id"]: row for row in manifest["segments"]}
        for decision in snapshot.get("decisions", []):
            row = segments.get(decision.get("segment_id"))
            if row is None or any(
                decision.get(key) != expected for key, expected in (
                    ("source_sha256", manifest["source_sha256"]),
                    ("config_sha256", manifest["config_sha256"]),
                    ("manifest_sha256", manifest_sha256),
                    ("derived_audio_sha256", row.get("derived_audio_sha256")),
                )
            ):
                raise ValueError("review decision binding does not match manifest/audio")
    return manifest, snapshot, manifest_sha256, snapshot_sha256


def _selected_rows(
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    config: LongFormExportConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = {item["segment_id"]: item for item in manifest["segments"]}
    decisions = snapshot.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(segments):
        raise RuntimeError("review decision set is incomplete")
    selected = []
    excluded = []
    seen = set()
    for decision in decisions:
        segment_id = str(decision.get("segment_id"))
        if segment_id in seen or segment_id not in segments:
            raise RuntimeError("review contains duplicate or unknown segment")
        seen.add(segment_id)
        action = decision.get("decision")
        style = decision.get("confirmed_style")
        text = str(decision.get("confirmed_text") or "").strip()
        if action not in {"keep", "reject"}:
            raise RuntimeError("review contains an unresolved decision")
        if action == "keep" and not text:
            raise RuntimeError("kept segment has empty confirmed text")
        if action != "keep":
            excluded.append({"segment_id": segment_id, "reason": "human_rejected"})
            continue
        if style not in config.accepted_styles:
            excluded.append(
                {
                    "segment_id": segment_id,
                    "reason": "style_not_in_default_training_set",
                    "confirmed_style": style,
                }
            )
            continue
        contract = manifest.get("refinement", {})
        minimum = float(contract.get("minimum_output_seconds", 1.5))
        maximum = float(contract.get("maximum_output_seconds", 15.0))
        if not minimum <= float(segments[segment_id]["duration_seconds"]) <= maximum:
            raise RuntimeError("kept segment violates the manifest duration contract")
        selected.append(
            {
                "segment": segments[segment_id],
                "decision": decision,
                "confirmed_text": text,
                "confirmed_style": style,
            }
        )
    reference_id = snapshot.get("reference_segment_id")
    if reference_id not in {item["segment"]["segment_id"] for item in selected}:
        raise RuntimeError("target reference is not in the selected normal set")
    audio_hashes = [item["segment"]["derived_audio_sha256"] for item in selected]
    text_keys = [_normalized_text(item["confirmed_text"]) for item in selected]
    if len(audio_hashes) != len(set(audio_hashes)):
        raise RuntimeError("selected set contains exact duplicate audio")
    if len(text_keys) != len(set(text_keys)):
        raise RuntimeError("selected set contains exact duplicate text")
    selected.sort(
        key=lambda item: (
            int(item["segment"]["source_start_frame"]),
            item["segment"]["segment_id"],
        )
    )
    excluded.sort(key=lambda item: item["segment_id"])
    return selected, excluded


def _build_export(
    staging: Path,
    final_root: Path,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    manifest_sha256: str,
    snapshot_sha256: str,
    config: LongFormExportConfig,
) -> dict[str, Any]:
    selected, excluded = _selected_rows(manifest, snapshot, config=config)
    rows = []
    source_root = manifest_path.parent
    for ordinal, item in enumerate(selected, start=1):
        segment = item["segment"]
        source_audio = (source_root / segment["derived_relative_path"]).resolve()
        source_audio.relative_to(source_root.resolve())
        if _sha256_file(source_audio) != segment["derived_audio_sha256"]:
            raise RuntimeError(f"reviewed audio hash drift: {segment['segment_id']}")
        info = sf.info(source_audio)
        contract = manifest.get("refinement", {})
        if not float(contract.get("minimum_output_seconds", 1.5)) <= info.duration <= float(contract.get("maximum_output_seconds", 15.0)):
            raise RuntimeError("reviewed audio violates the manifest duration contract")
        relative_audio = f"audio/{segment['segment_id']}.wav"
        destination = staging / relative_audio
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_audio, destination)
        with destination.open("r+b") as copied:
            copied.flush()
            os.fsync(copied.fileno())
        if _sha256_file(destination) != segment["derived_audio_sha256"]:
            raise RuntimeError(f"copied audio hash mismatch: {segment['segment_id']}")
        text = item["confirmed_text"]
        rows.append(
            {
                "ordinal": ordinal,
                "fid": segment["segment_id"],
                "audio_relative_path": relative_audio,
                "audio_absolute_path": str((final_root / relative_audio).resolve()),
                "audio_sha256": segment["derived_audio_sha256"],
                "duration_seconds": segment["duration_seconds"],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "speaker_id": config.speaker_id,
                "voice_id": config.voice_id,
                "emotion_primary": config.emotion_primary,
                "style": item["confirmed_style"],
                "source_sha256": manifest["source_sha256"],
                "source_start_frame": segment["source_start_frame"],
                "source_end_frame": segment["source_end_frame"],
                "source_spans": segment["source_spans"],
                **({"boundary_repair": segment["boundary_repair"],
                    "requested_source_spans": segment["requested_source_spans"]}
                   if segment.get("boundary_repair") else {}),
                "parent_segment_id": segment["parent_segment_id"],
                "review_batch_id": item["decision"]["review_batch_id"],
                "review_payload_sha256": item["decision"]["payload_sha256"],
            }
        )
    trainer_content = "".join(
        _canonical_json(
            {"audio": row["audio_absolute_path"], "fid": row["fid"], "text": row["text"]}
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    catalog_content = "".join(
        _canonical_json(
            {
                "asset_sha256": row["audio_sha256"],
                "audio_relative_path": row["audio_relative_path"],
                "duration_seconds": row["duration_seconds"],
                "emotion_weak_label": row["emotion_primary"],
                "source_segment_id": row["fid"],
                "speaker_id": row["speaker_id"],
                "transcript_candidate": row["text"],
            }
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    _write_bytes(staging / "trainer.jsonl", trainer_content)
    _write_bytes(staging / "catalog_candidates.jsonl", catalog_content)
    _write_bytes(
        staging / "excluded.json",
        (json.dumps(excluded, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    reference_id = snapshot["reference_segment_id"]
    reference = next(row for row in rows if row["fid"] == reference_id)
    _write_bytes(
        staging / "reference.json",
        (json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    row_set_sha256 = hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()
    export_manifest = {
        "schema_version": 1,
        "implementation_version": LONG_FORM_EXPORT_IMPLEMENTATION_VERSION,
        "config_id": config.config_id,
        "config_version": config.config_version,
        "config_sha256": config.config_sha256(),
        "voice_id": config.voice_id,
        "speaker_id": config.speaker_id,
        "emotion_primary": config.emotion_primary,
        "source_sha256": manifest["source_sha256"],
        "parent_manifest_sha256": manifest_sha256,
        "review_snapshot_sha256": snapshot_sha256,
        "reference_segment_id": reference_id,
        "item_count": len(rows),
        "total_duration_seconds": sum(float(row["duration_seconds"]) for row in rows),
        "excluded_count": len(excluded),
        "row_set_sha256": row_set_sha256,
        "rows": rows,
    }
    _write_bytes(
        staging / "manifest.json",
        (json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    checksummed = [
        path
        for path in staging.rglob("*")
        if path.is_file() and path.name != "checksums.txt"
    ]
    checksums = "".join(
        f"{_sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
        for path in sorted(checksummed, key=lambda value: value.relative_to(staging).as_posix())
    )
    _write_bytes(staging / "checksums.txt", checksums.encode("utf-8"))
    return export_manifest


def validate_long_form_export(
    root: str | Path,
    *,
    config_path: str | Path = DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH,
) -> dict[str, Any]:
    resolved = Path(root).resolve()
    config = load_long_form_export_config(config_path)
    manifest = json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config.config_sha256():
        raise RuntimeError("long-form export config hash mismatch")
    checksum_rows = (resolved / "checksums.txt").read_text(encoding="utf-8").splitlines()
    for line in checksum_rows:
        digest, separator, relative = line.partition("  ")
        if not separator:
            raise RuntimeError("malformed long-form checksum row")
        path = (resolved / relative).resolve()
        path.relative_to(resolved)
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(f"long-form export checksum mismatch: {relative}")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != manifest.get("item_count"):
        raise RuntimeError("long-form export item count mismatch")
    if hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest() != manifest.get(
        "row_set_sha256"
    ):
        raise RuntimeError("long-form export row set hash mismatch")
    trainer = [
        json.loads(line)
        for line in (resolved / "trainer.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(trainer) != len(rows):
        raise RuntimeError("trainer JSONL item count mismatch")
    for row, trainer_row in zip(rows, trainer):
        audio = resolved / row["audio_relative_path"]
        info = sf.info(audio)
        if (info.samplerate, info.channels, info.subtype) != (48000, 1, "PCM_24"):
            raise RuntimeError(f"invalid training audio format: {audio.name}")
        if _sha256_file(audio) != row["audio_sha256"]:
            raise RuntimeError(f"training audio hash mismatch: {audio.name}")
        if trainer_row != {"audio": row["audio_absolute_path"], "fid": row["fid"], "text": row["text"]}:
            raise RuntimeError(f"trainer row mismatch: {row['fid']}")
    return {
        "status": "succeeded",
        "root": str(resolved),
        "item_count": len(rows),
        "total_duration_seconds": manifest["total_duration_seconds"],
        "reference_segment_id": manifest["reference_segment_id"],
        "manifest_sha256": _sha256_file(resolved / "manifest.json"),
        "checksums_sha256": _sha256_file(resolved / "checksums.txt"),
    }


def export_long_form_dataset(
    manifest_path: str | Path,
    review_snapshot_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH,
    target: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_file = Path(manifest_path).resolve()
    snapshot_file = Path(review_snapshot_path).resolve()
    config = load_long_form_export_config(config_path)
    final_root = (
        Path(target).resolve()
        if target is not None
        else (repo_root / config.output_root).resolve()
    )
    validate_output_path(final_root, protected_inputs=(manifest_file, snapshot_file))
    manifest, snapshot, manifest_sha256, snapshot_sha256 = _load_inputs(
        manifest_file, snapshot_file
    )
    if final_root.exists():
        validation = validate_long_form_export(final_root, config_path=config_path)
        existing = json.loads((final_root / "manifest.json").read_text(encoding="utf-8"))
        if (
            existing.get("parent_manifest_sha256") != manifest_sha256
            or existing.get("review_snapshot_sha256") != snapshot_sha256
        ):
            raise RuntimeError("immutable long-form export target already exists for other inputs")
        return {**validation, "action": "cached"}
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = final_root.with_name(f".{final_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir()
    try:
        _build_export(
            staging,
            final_root,
            manifest_path=manifest_file,
            manifest=manifest,
            snapshot=snapshot,
            manifest_sha256=manifest_sha256,
            snapshot_sha256=snapshot_sha256,
            config=config,
        )
        validate_long_form_export(staging, config_path=config_path)
        os.replace(staging, final_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**validate_long_form_export(final_root, config_path=config_path), "action": "built"}
