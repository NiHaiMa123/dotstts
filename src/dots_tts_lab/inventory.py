from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.metadata import (
    DEFAULT_METADATA_PROFILE_PATH,
    MetadataProfile,
    load_metadata_profile,
    parse_metadata,
)
from dots_tts_lab.reports import write_inventory_reports

DEFAULT_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg"})
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_LEGACY_WINDOWS_PATH_WARNING = 240


class FileChangedDuringScan(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def normalize_root_key(path: Path) -> str:
    resolved = str(path.resolve())
    return os.path.normcase(resolved) if os.name == "nt" else resolved


def normalize_path_key(relative_path: Path) -> str:
    portable = relative_path.as_posix()
    return portable.casefold() if os.name == "nt" else portable


def discover_audio_files(
    root: Path, *, extensions: Iterable[str] = DEFAULT_EXTENSIONS
) -> list[Path]:
    normalized_extensions = {
        (
            extension.casefold()
            if extension.startswith(".")
            else f".{extension.casefold()}"
        )
        for extension in extensions
    }
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.casefold() in normalized_extensions
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def hash_file(path: Path) -> tuple[str, os.stat_result]:
    last_error: FileChangedDuringScan | None = None
    for _ in range(2):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            while block := file_obj.read(_HASH_CHUNK_SIZE):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (
            after.st_size,
            after.st_mtime_ns,
        ):
            return digest.hexdigest(), after
        last_error = FileChangedDuringScan(
            f"File changed while hashing: {path!s}"
        )
    assert last_error is not None
    raise last_error


def probe_audio(path: Path) -> dict[str, Any]:
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.channels <= 0 or info.frames < 0:
            raise ValueError(f"Invalid audio metadata: {info!r}")
        return {
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_seconds": float(info.duration),
            "probe_status": "ok",
            "probe_error_type": None,
            "probe_error_message": None,
        }
    except Exception as error:
        return {
            "format": None,
            "subtype": None,
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "duration_seconds": None,
            "probe_status": "error",
            "probe_error_type": type(error).__name__,
            "probe_error_message": str(error),
        }


def scan_file(
    path: Path, *, root: Path, metadata_profile: MetadataProfile
) -> dict[str, Any]:
    relative_path = path.relative_to(root)
    record: dict[str, Any] = {
        "relative_path": relative_path.as_posix(),
        "path_key": normalize_path_key(relative_path),
        "absolute_path": str(path.resolve()),
        "path_length": len(str(path.resolve())),
        "path_warning": None,
    }
    if os.name == "nt" and record["path_length"] >= _LEGACY_WINDOWS_PATH_WARNING:
        record["path_warning"] = (
            f"absolute path length {record['path_length']} approaches the legacy "
            "Windows MAX_PATH limit"
        )

    record.update(parse_metadata(relative_path, profile=metadata_profile))

    try:
        sha256, stat_result = hash_file(path)
        record.update(
            {
                "sha256": sha256,
                "size_bytes": int(stat_result.st_size),
                "mtime_ns": int(stat_result.st_mtime_ns),
                "hash_status": "ok",
                "hash_error_type": None,
                "hash_error_message": None,
            }
        )
    except Exception as error:
        try:
            stat_result = path.stat()
        except OSError:
            stat_result = None
        record.update(
            {
                "sha256": None,
                "size_bytes": (
                    None if stat_result is None else int(stat_result.st_size)
                ),
                "mtime_ns": (
                    None if stat_result is None else int(stat_result.st_mtime_ns)
                ),
                "hash_status": "error",
                "hash_error_type": type(error).__name__,
                "hash_error_message": str(error),
            }
        )

    if record["hash_status"] == "ok":
        record.update(probe_audio(path))
    else:
        record.update(
            {
                "format": None,
                "subtype": None,
                "sample_rate": None,
                "channels": None,
                "frames": None,
                "duration_seconds": None,
                "probe_status": "error",
                "probe_error_type": "HashFailed",
                "probe_error_message": "Audio probe skipped because hashing failed.",
            }
        )

    errors = [
        message
        for message in (
            record.get("parse_error"),
            record.get("hash_error_message"),
            record.get("probe_error_message"),
        )
        if message
    ]
    record["scan_status"] = (
        "error"
        if errors
        else "review_required"
        if record["metadata_status"] == "review_required"
        else "ok"
    )
    record["scan_error"] = "; ".join(errors) if errors else None
    record["scan_review_reason"] = record.get("metadata_review_reason")
    return record


def classify_changes(
    records: list[dict[str, Any]], previous: dict[str, dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    current_keys = {record["path_key"] for record in records}
    previous_active = {
        path_key: value
        for path_key, value in previous.items()
        if value["availability_status"] == "available"
    }
    missing_keys = sorted(set(previous_active) - current_keys)
    missing_by_hash: dict[str, list[str]] = defaultdict(list)
    for path_key in missing_keys:
        sha256 = previous_active[path_key].get("asset_sha256")
        if sha256:
            missing_by_hash[str(sha256)].append(path_key)

    matched_missing: set[str] = set()
    for record in records:
        prior = previous.get(record["path_key"])
        if prior is None or prior["availability_status"] == "missing":
            record["change_kind"] = "added"
        elif prior.get("asset_sha256") != record.get("sha256"):
            record["change_kind"] = "changed"
        else:
            record["change_kind"] = "unchanged"

        if record["change_kind"] != "added" or not record.get("sha256"):
            continue
        candidates = missing_by_hash.get(record["sha256"], [])
        source_key = next(
            (candidate for candidate in candidates if candidate not in matched_missing),
            None,
        )
        if source_key is not None:
            matched_missing.add(source_key)
            record["change_kind"] = "moved"
            record["moved_from"] = previous_active[source_key]["relative_path"]

    missing_records = [
        {
            "relative_path": previous_active[path_key]["relative_path"],
            "path_key": path_key,
            "sha256": previous_active[path_key].get("asset_sha256"),
            "change_kind": "missing",
        }
        for path_key in missing_keys
        if path_key not in matched_missing
    ]
    return missing_keys, missing_records


def build_summary(
    *,
    run_id: str,
    root: Path,
    records: list[dict[str, Any]],
    missing_records: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    metadata_profile: MetadataProfile,
) -> dict[str, Any]:
    change_counts = Counter(record["change_kind"] for record in records)
    emotion_counts = Counter(
        record["emotion_weak_label"]
        for record in records
        if record.get("emotion_weak_label")
    )
    sample_rate_counts = Counter(
        str(record["sample_rate"])
        for record in records
        if record.get("sample_rate") is not None
    )
    total_duration = sum(
        float(record["duration_seconds"] or 0.0) for record in records
    )
    error_count = sum(record["scan_status"] == "error" for record in records)
    review_required_count = sum(
        record["scan_status"] == "review_required" for record in records
    )
    return {
        "run_id": run_id,
        "status": "completed_with_errors" if error_count else "succeeded",
        "root_path": str(root),
        "started_at": started_at,
        "finished_at": finished_at,
        "discovered_count": len(records),
        "readable_count": sum(record["probe_status"] == "ok" for record in records),
        "error_count": error_count,
        "review_required_count": review_required_count,
        "added_count": int(change_counts["added"]),
        "changed_count": int(change_counts["changed"]),
        "moved_count": int(change_counts["moved"]),
        "missing_count": len(missing_records),
        "unchanged_count": int(change_counts["unchanged"]),
        "total_duration_seconds": round(total_duration, 6),
        "total_duration_minutes": round(total_duration / 60.0, 6),
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "sample_rate_counts": dict(sorted(sample_rate_counts.items())),
        "path_warning_count": sum(bool(record["path_warning"]) for record in records),
        "parser_profile_id": metadata_profile.profile_id,
        "parser_profile_version": metadata_profile.profile_version,
        "parser_config_sha256": metadata_profile.config_sha256(),
    }


def run_inventory(
    input_dir: str | Path = "data/inbox",
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    report_dir: str | Path = "data/reports/inventory",
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    metadata_profile_path: str | Path = DEFAULT_METADATA_PROFILE_PATH,
) -> dict[str, Any]:
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Inventory input directory not found: {root!s}")

    profile_path = Path(metadata_profile_path).resolve()
    metadata_profile = load_metadata_profile(profile_path)
    catalog = Catalog(catalog_path)
    catalog.initialize()
    root_path = str(root)
    root_key = normalize_root_key(root)
    started_at = utc_now()
    catalog.register_metadata_profile(
        profile_id=metadata_profile.profile_id,
        profile_version=metadata_profile.profile_version,
        config_sha256=metadata_profile.config_sha256(),
        config_json=metadata_profile.canonical_json(),
        source_path=str(profile_path),
        registered_at=started_at,
    )
    run_id = catalog.begin_run(
        root_path=root_path,
        root_key=root_key,
        started_at=started_at,
    )

    try:
        previous = catalog.load_locations(root_key=root_key)
        files = discover_audio_files(root, extensions=extensions)
        records = [
            scan_file(path, root=root, metadata_profile=metadata_profile)
            for path in files
        ]
        missing_path_keys, missing_records = classify_changes(records, previous)
        finished_at = utc_now()
        summary = build_summary(
            run_id=run_id,
            root=root,
            records=records,
            missing_records=missing_records,
            started_at=started_at,
            finished_at=finished_at,
            metadata_profile=metadata_profile,
        )
        report_paths = write_inventory_reports(
            report_dir,
            summary=summary,
            records=records,
            missing_records=missing_records,
        )
        catalog.complete_run(
            run_id=run_id,
            root_path=root_path,
            root_key=root_key,
            finished_at=finished_at,
            records=records,
            missing_path_keys=missing_path_keys,
            summary=summary,
        )
        summary["reports"] = report_paths
        summary["catalog_path"] = str(Path(catalog_path).resolve())
        summary["metadata_profile_path"] = str(profile_path)
        return summary
    except BaseException as error:
        catalog.fail_run(
            run_id=run_id,
            finished_at=utc_now(),
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
