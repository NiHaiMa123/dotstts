from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import (
    DEFAULT_EXTENSIONS,
    normalize_root_key,
    run_inventory,
    utc_now,
)
from dots_tts_lab.metadata import DEFAULT_METADATA_PROFILE_PATH
from dots_tts_lab.reports import write_ingest_reports

_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_SAFE_EXTENSION = re.compile(r"^\.[a-z0-9]{1,10}$")
_PARTIAL_NAME = re.compile(r"^\.[0-9a-f]{64}\..+\.partial$")


class RawIntegrityError(RuntimeError):
    pass


class SourceChangedDuringImport(RuntimeError):
    pass


def _acquire_raw_store_lock(raw_root: Path):
    raw_root.mkdir(parents=True, exist_ok=True)
    lock_file = (raw_root / ".ingest.lock").open("a+b")
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock_file.close()
        raise RuntimeError(f"Another ingest process holds the raw store: {raw_root}")
    return lock_file


def _release_raw_store_lock(lock_file) -> None:
    try:
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _cleanup_orphan_partials(raw_root: Path) -> int:
    removed = 0
    for path in raw_root.rglob("*.partial"):
        if path.is_file() and _PARTIAL_NAME.fullmatch(path.name):
            path.unlink()
            removed += 1
    return removed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while block := file_obj.read(_COPY_CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _copy_stream(source, destination) -> str:
    digest = hashlib.sha256()
    while block := source.read(_COPY_CHUNK_SIZE):
        destination.write(block)
        digest.update(block)
    return digest.hexdigest()


def _canonical_extension(relative_path: str) -> str:
    extension = Path(relative_path).suffix.casefold()
    return extension if _SAFE_EXTENSION.fullmatch(extension) else ".bin"


def _managed_path(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Raw object path escapes the managed root: {relative_path!r}"
        ) from error
    return candidate


def _source_path(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path.replace("/", os.sep))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Source path escapes the inventory root: {relative_path!r}"
        ) from error
    return candidate


def _verify_existing(path: Path, *, expected_sha256: str, expected_size: int) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RawIntegrityError(
            f"Existing raw object has size {actual_size}, "
            f"expected {expected_size}: {path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RawIntegrityError(
            "Existing raw object failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual_sha256}: {path}"
        )


def _atomic_copy_verified(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{expected_sha256}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        before = source.stat()
        with source.open("rb") as source_file, os.fdopen(
            descriptor, "wb"
        ) as destination_file:
            descriptor = -1
            copied_sha256 = _copy_stream(source_file, destination_file)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        after = source.stat()

        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SourceChangedDuringImport(f"Source changed during import: {source}")
        if after.st_size != expected_size:
            raise SourceChangedDuringImport(
                f"Source size changed since inventory: {source}"
            )
        if copied_sha256 != expected_sha256:
            raise SourceChangedDuringImport(
                "Source content changed since inventory: "
                f"expected {expected_sha256}, got {copied_sha256}: {source}"
            )

        persisted_sha256 = _sha256_file(temporary_path)
        if persisted_sha256 != expected_sha256:
            raise RawIntegrityError(
                "Temporary raw object failed post-write SHA-256 verification: "
                f"expected {expected_sha256}, got {persisted_sha256}"
            )

        os.replace(temporary_path, target)
        _verify_existing(
            target,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _raw_relative_path(sha256: str, extension: str) -> str:
    return f"{sha256[:2]}/{sha256}{extension}"


def _materialize_asset(
    *,
    sha256: str,
    candidates: list[dict[str, Any]],
    source_root: Path,
    raw_root: Path,
    existing_raw: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str | None, dict[str, str]]:
    expected_size = int(candidates[0]["asset_size_bytes"])
    if existing_raw is not None:
        relative_path = str(existing_raw["relative_path"])
        extension = str(existing_raw["extension"])
    else:
        extension = _canonical_extension(str(candidates[0]["relative_path"]))
        relative_path = _raw_relative_path(sha256, extension)

    target = _managed_path(raw_root, relative_path)
    if target.exists():
        _verify_existing(
            target,
            expected_sha256=sha256,
            expected_size=expected_size,
        )
        action = "reused"
        used_source = None
        source_errors: dict[str, str] = {}
    else:
        source_errors = {}
        used_source = None
        for candidate in candidates:
            source = _source_path(source_root, str(candidate["relative_path"]))
            try:
                _atomic_copy_verified(
                    source,
                    target,
                    expected_sha256=sha256,
                    expected_size=expected_size,
                )
                action = "imported"
                used_source = str(candidate["relative_path"])
                break
            except (OSError, RawIntegrityError, SourceChangedDuringImport) as error:
                source_errors[str(candidate["relative_path"])] = (
                    f"{type(error).__name__}: {error}"
                )
        else:
            raise OSError(
                "No source location could materialize the raw object; "
                + " | ".join(
                    f"{path}: {message}" for path, message in source_errors.items()
                )
            )

    raw_object = {
        "asset_sha256": sha256,
        "relative_path": relative_path,
        "extension": extension,
        "size_bytes": expected_size,
        "absolute_path": str(target),
    }
    return raw_object, action, used_source, source_errors


def run_ingest(
    input_dir: str | Path = "data/inbox",
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    raw_dir: str | Path = "data/raw/sha256",
    inventory_report_dir: str | Path = "data/reports/inventory",
    report_dir: str | Path = "data/reports/ingest",
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    metadata_profile_path: str | Path = DEFAULT_METADATA_PROFILE_PATH,
) -> dict[str, Any]:
    source_root = Path(input_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Ingest input directory not found: {source_root}")
    raw_root = Path(raw_dir).resolve()

    inventory_summary = run_inventory(
        source_root,
        catalog_path=catalog_path,
        report_dir=inventory_report_dir,
        extensions=extensions,
        metadata_profile_path=metadata_profile_path,
    )

    catalog = Catalog(catalog_path)
    catalog.initialize()
    root_key = normalize_root_key(source_root)
    candidates = catalog.load_import_candidates(root_key=root_key)
    existing_raw_objects = catalog.load_raw_objects()
    started_at = utc_now()
    run_id = catalog.begin_import_run(
        root_path=str(source_root),
        root_key=root_key,
        raw_root_path=str(raw_root),
        started_at=started_at,
    )

    lock_file = None
    try:
        lock_file = _acquire_raw_store_lock(raw_root)
        orphan_partial_count = _cleanup_orphan_partials(raw_root)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            grouped[str(candidate["asset_sha256"])].append(candidate)

        raw_objects: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        imported_count = 0
        reused_count = 0
        bytes_written = 0

        for sha256, asset_candidates in sorted(grouped.items()):
            try:
                raw_object, action, used_source, source_errors = _materialize_asset(
                    sha256=sha256,
                    candidates=asset_candidates,
                    source_root=source_root,
                    raw_root=raw_root,
                    existing_raw=existing_raw_objects.get(sha256),
                )
                raw_objects.append(raw_object)
                if action == "imported":
                    imported_count += 1
                    bytes_written += int(raw_object["size_bytes"])
                else:
                    reused_count += 1
                for candidate in asset_candidates:
                    source_relative_path = str(candidate["relative_path"])
                    source_error = source_errors.get(source_relative_path)
                    item_action = (
                        "error"
                        if source_error is not None
                        else (
                            "imported"
                            if action == "imported"
                            and source_relative_path == used_source
                            else "reused"
                        )
                    )
                    items.append(
                        {
                            "root_key": root_key,
                            "path_key": candidate["path_key"],
                            "asset_sha256": sha256,
                            "source_relative_path": source_relative_path,
                            "original_name": Path(
                                str(candidate["relative_path"])
                            ).name,
                            "raw_relative_path": raw_object["relative_path"],
                            "action": item_action,
                            "error_type": (
                                None
                                if source_error is None
                                else source_error.partition(":")[0]
                            ),
                            "error_message": source_error,
                        }
                    )
            except Exception as error:
                for candidate in asset_candidates:
                    items.append(
                        {
                            "root_key": root_key,
                            "path_key": candidate["path_key"],
                            "asset_sha256": sha256,
                            "source_relative_path": candidate["relative_path"],
                            "original_name": Path(
                                str(candidate["relative_path"])
                            ).name,
                            "raw_relative_path": None,
                            "action": "error",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )

        finished_at = utc_now()
        error_count = sum(item["action"] == "error" for item in items)
        inventory_error_count = int(inventory_summary["error_count"])
        summary = {
            "run_id": run_id,
            "status": (
                "completed_with_errors"
                if error_count or inventory_error_count
                else "succeeded"
            ),
            "root_path": str(source_root),
            "raw_root_path": str(raw_root),
            "storage_mode": "copy",
            "started_at": started_at,
            "finished_at": finished_at,
            "discovered_count": len(candidates),
            "unique_asset_count": len(grouped),
            "imported_count": imported_count,
            "reused_count": reused_count,
            "error_count": error_count,
            "bytes_written": bytes_written,
            "inventory_run_id": inventory_summary["run_id"],
            "inventory_status": inventory_summary["status"],
            "inventory_error_count": inventory_error_count,
            "orphan_partial_count": orphan_partial_count,
        }
        report_paths = write_ingest_reports(
            report_dir,
            summary=summary,
            items=items,
            raw_objects=raw_objects,
        )
        catalog.complete_import_run(
            run_id=run_id,
            finished_at=finished_at,
            raw_objects=raw_objects,
            items=items,
            summary=summary,
        )
        summary["reports"] = report_paths
        summary["catalog_path"] = str(Path(catalog_path).resolve())
        return summary
    except BaseException as error:
        catalog.fail_import_run(
            run_id=run_id,
            finished_at=utc_now(),
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if lock_file is not None:
            _release_raw_store_lock(lock_file)
