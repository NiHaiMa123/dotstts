from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.reports import write_standardization_reports
from dots_tts_lab.standardization import (
    DEFAULT_STANDARDIZATION_CONFIG_PATH,
    atomic_write_pcm24,
    derived_identity,
    file_sha256,
    load_standardization_config,
    standardize_array,
)


STANDARDIZATION_IMPLEMENTATION_VERSION = 1


def _managed_path(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes managed root: {relative_path!r}") from error
    return candidate


def _cache_is_valid(root: Path, record: dict[str, Any]) -> bool:
    try:
        path = _managed_path(root, str(record["relative_path"]), label="Derived")
        return (
            path.is_file()
            and path.stat().st_size == int(record["size_bytes"])
            and file_sha256(path) == str(record["output_sha256"])
        )
    except (OSError, ValueError):
        return False


def _record_item(
    *, candidate: dict[str, Any], derived_id: str, action: str
) -> dict[str, Any]:
    return {
        "asset_sha256": str(candidate["asset_sha256"]),
        "derived_id": derived_id,
        "source_relative_path": candidate.get("source_relative_path"),
        "raw_relative_path": str(candidate["relative_path"]),
        "speaker_id": candidate.get("speaker_id"),
        "emotion_weak_label": candidate.get("emotion_weak_label"),
        "transcript_candidate": candidate.get("transcript_candidate"),
        "action": action,
    }


def run_standardization(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    raw_dir: str | Path = "data/raw/sha256",
    output_dir: str | Path = "data/work/standardized",
    config_path: str | Path = DEFAULT_STANDARDIZATION_CONFIG_PATH,
    report_dir: str | Path = "data/reports/standardization",
    force: bool = False,
    asset_sha256s: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw object directory not found: {raw_root}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    resolved_config_path = Path(config_path).resolve()
    config = load_standardization_config(resolved_config_path)
    config_sha256 = config.config_sha256()
    selected_assets = None if asset_sha256s is None else set(asset_sha256s)

    catalog = Catalog(catalog_path)
    catalog.initialize()
    started_at = utc_now()
    catalog.register_standardization_config(
        config_id=config.config_id,
        config_version=config.config_version,
        config_sha256=config_sha256,
        config_json=config.canonical_json(),
        source_path=str(resolved_config_path),
        registered_at=started_at,
    )
    candidates = catalog.load_standardization_candidates()
    if selected_assets is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["asset_sha256"] in selected_assets
        ]
        missing = selected_assets - {
            str(candidate["asset_sha256"]) for candidate in candidates
        }
        if missing:
            raise ValueError(
                "Requested asset SHA-256 values are not present in raw: "
                + ", ".join(sorted(missing))
            )
    if limit is not None:
        candidates = candidates[:limit]
    existing = catalog.load_derived_audio(
        config_id=config.config_id,
        config_version=config.config_version,
        config_sha256=config_sha256,
        implementation_version=STANDARDIZATION_IMPLEMENTATION_VERSION,
    )
    run_id = catalog.begin_standardization_run(
        raw_root_path=str(raw_root),
        output_root_path=str(output_root),
        config_id=config.config_id,
        config_version=config.config_version,
        config_sha256=config_sha256,
        implementation_version=STANDARDIZATION_IMPLEMENTATION_VERSION,
        started_at=started_at,
    )

    try:
        derived_to_store: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        for candidate in candidates:
            asset_sha256 = str(candidate["asset_sha256"])
            derived_id = derived_identity(
                asset_sha256=asset_sha256,
                config=config,
                implementation_version=STANDARDIZATION_IMPLEMENTATION_VERSION,
            )
            relative_output = f"{derived_id[:2]}/{derived_id}.wav"
            cached = existing.get(asset_sha256)
            if (
                not force
                and cached is not None
                and cached["derived_id"] == derived_id
                and _cache_is_valid(output_root, cached)
            ):
                item = _record_item(
                    candidate=candidate, derived_id=derived_id, action="cached"
                )
                item.update(cached)
                item["derived_relative_path"] = cached["relative_path"]
                items.append(item)
                continue

            action = "rebuilt" if cached is not None else "built"
            item = _record_item(candidate=candidate, derived_id=derived_id, action=action)
            item["derived_relative_path"] = relative_output
            try:
                source_path = _managed_path(
                    raw_root, str(candidate["relative_path"]), label="Raw"
                )
                data, source_rate = sf.read(
                    str(source_path), dtype="float64", always_2d=True
                )
                output, processing = standardize_array(data, source_rate, config)
                output_path = _managed_path(
                    output_root, relative_output, label="Derived"
                )
                output_hash, size_bytes, output_metrics = atomic_write_pcm24(
                    output_path,
                    output,
                    config.target_sample_rate_hz,
                    config.true_peak_oversample,
                )
                completed_at = utc_now()
                derived = {
                    "derived_id": derived_id,
                    "asset_sha256": asset_sha256,
                    "config_id": config.config_id,
                    "config_version": config.config_version,
                    "config_sha256": config_sha256,
                    "implementation_version": STANDARDIZATION_IMPLEMENTATION_VERSION,
                    "relative_path": relative_output,
                    "output_sha256": output_hash,
                    "size_bytes": size_bytes,
                    "created_at": completed_at,
                    "last_verified_at": completed_at,
                    "source_sample_rate": source_rate,
                    "source_channels": int(data.shape[1]),
                    "source_frames": int(data.shape[0]),
                    "output_sample_rate": config.target_sample_rate_hz,
                    "output_channels": 1,
                    "output_frames": int(len(output)),
                    "duration_seconds": len(output) / config.target_sample_rate_hz,
                    **processing,
                    **output_metrics,
                    "output_subtype": config.output_subtype,
                    "created_run_id": run_id,
                }
                derived_to_store.append(derived)
                item.update(derived)
                item["derived_relative_path"] = relative_output
            except Exception as error:
                item["action"] = "error"
                item["error_type"] = type(error).__name__
                item["error_message"] = str(error)
            items.append(item)

        finished_at = utc_now()
        error_count = sum(item["action"] == "error" for item in items)
        summary = {
            "run_id": run_id,
            "status": "completed_with_errors" if error_count else "succeeded",
            "raw_root_path": str(raw_root),
            "output_root_path": str(output_root),
            "config_id": config.config_id,
            "config_version": config.config_version,
            "config_sha256": config_sha256,
            "implementation_version": STANDARDIZATION_IMPLEMENTATION_VERSION,
            "started_at": started_at,
            "finished_at": finished_at,
            "discovered_count": len(items),
            "built_count": sum(item["action"] == "built" for item in items),
            "cached_count": sum(item["action"] == "cached" for item in items),
            "rebuilt_count": sum(item["action"] == "rebuilt" for item in items),
            "trimmed_count": sum(
                (item.get("leading_samples_removed", 0) or 0) > 0
                or (item.get("trailing_samples_removed", 0) or 0) > 0
                for item in items
            ),
            "attenuated_count": sum(
                (item.get("gain_applied_db", 0.0) or 0.0) < 0 for item in items
            ),
            "error_count": error_count,
            "output_duration_minutes": sum(
                float(item.get("duration_seconds", 0.0) or 0.0) for item in items
            )
            / 60.0,
            "output_size_bytes": sum(
                int(item.get("size_bytes", 0) or 0) for item in items
            ),
            "max_output_true_peak_estimate_dbtp": max(
                (
                    float(item["output_true_peak_estimate_dbtp"])
                    for item in items
                    if item.get("output_true_peak_estimate_dbtp") is not None
                ),
                default=None,
            ),
        }
        report_paths = write_standardization_reports(
            report_dir, summary=summary, items=items
        )
        catalog.complete_standardization_run(
            run_id=run_id,
            finished_at=finished_at,
            derived_to_store=derived_to_store,
            items=items,
            summary=summary,
        )
        summary["reports"] = report_paths
        summary["catalog_path"] = str(Path(catalog_path).resolve())
        summary["config_path"] = str(resolved_config_path)
        return summary
    except BaseException as error:
        catalog.fail_standardization_run(
            run_id=run_id,
            finished_at=utc_now(),
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
