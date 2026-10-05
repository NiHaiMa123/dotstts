"""Append the user-confirmed Slice 9 listening decisions to the catalog."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_reports import DEFAULT_SLICE9_REVIEW_JSON_PATH


DEFAULT_SLICE9_CATALOG_PATH = Path("data/catalog/catalog.sqlite")
DEFAULT_SLICE9_REVIEW_RESULT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/audit/slice9_reference_review_import_v1.json"
)
_REVIEW_NAMESPACE = uuid.UUID("1f9fbf29-61c6-4d11-9f8d-ef5dca1b3e6c")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_slice9_reference_reviews(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    review_report_path: str | Path = DEFAULT_SLICE9_REVIEW_JSON_PATH,
    catalog_path: str | Path = DEFAULT_SLICE9_CATALOG_PATH,
    output_path: str | Path = DEFAULT_SLICE9_REVIEW_RESULT_PATH,
    review_status: str = "approved",
    review_note: str = "user_confirmed_no_issues",
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    report_path = Path(review_report_path).resolve()
    report = _load_json(report_path, label="Slice 9 manual review report")
    _require(report.get("manual_review_required") is True, "Review report is not a Slice 9 manual-review artifact")
    _require(review_status in {"approved", "rejected", "uncertain"}, "Invalid Slice 9 review status")
    rows = report.get("rows")
    _require(isinstance(rows, list) and rows, "Slice 9 review rows are missing")
    batch_id = "slice9-reference-review-v1"
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    # The batch id and report hash define one immutable review export.  Keeping
    # this timestamp deterministic makes an identical replay a true no-op.
    created_at = "2026-09-08T00:00:00Z"
    review_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pool_id = str(row["pool_id"])
        _require(pool_id in config["pools"], f"Unknown Slice 9 pool in review report: {pool_id}")
        review_rows.append({
            "review_id": str(uuid.uuid5(_REVIEW_NAMESPACE, f"{batch_id}:{index}:{pool_id}:{row['asset_sha256']}")),
            "selection_id": config["selection_id"],
            "selection_version": config["selection_version"],
            "asset_sha256": row["asset_sha256"],
            "pool_id": pool_id,
            "selection_kind": row["selection_kind"],
            "review_status": review_status,
            "review_note": review_note,
            "created_at": created_at,
            "review_batch_id": batch_id,
            "source_row_index": index,
            "provenance_json": json.dumps({
                "review_report_sha256": report_sha,
                "review_report_path": str(report_path),
                "manual_confirmation": "用户确认报告内候选没有问题",
                "audio_sha256": row["audio_sha256"],
                "text_sha256": hashlib.sha256(str(row["text_exact"]).encode("utf-8")).hexdigest(),
                "selection_kind": row["selection_kind"],
            }, ensure_ascii=False, sort_keys=True),
        })
    catalog = Catalog(Path(catalog_path).resolve())
    inserted, ignored = catalog.record_slice9_reference_reviews(review_rows)
    result = {
        "schema_version": 1,
        "status": "reviewed",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "review_batch_id": batch_id,
        "review_status": review_status,
        "created_at": created_at,
        "review_report_sha256": report_sha,
        "row_count": len(review_rows),
        "inserted": inserted,
        "ignored_idempotent_replay": ignored,
        "catalog_path": str(Path(catalog_path).resolve()),
    }
    _atomic_write_json(Path(output_path).resolve(), result)
    return {**result, "report_path": str(Path(output_path).resolve())}
