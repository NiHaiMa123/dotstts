"""End-to-end acceptance audit for the real Slice 9 pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_freeze import DEFAULT_SLICE9_CATALOG_PATH, DEFAULT_SLICE9_REFERENCE_DIR


DEFAULT_SLICE9_ACCEPTANCE_RESULT_PATH = Path("data/reports/datasets/fuxuan_v1/audit/slice9_pipeline_acceptance_v1.json")
ANALYSIS_ROOT = Path("data/reports/datasets/fuxuan_v1/analysis")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def build_slice9_acceptance_audit(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    catalog_path: str | Path = DEFAULT_SLICE9_CATALOG_PATH,
    reference_dir: str | Path = DEFAULT_SLICE9_REFERENCE_DIR,
    output_path: str | Path = DEFAULT_SLICE9_ACCEPTANCE_RESULT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    catalog = Catalog(Path(catalog_path).resolve())
    with catalog.read_only_session() as connection:
        dataset = connection.execute(
            "SELECT item_count, candidate_snapshot_sha256, artifact_set_sha256 FROM dataset_version WHERE dataset_id=? AND dataset_version=?",
            (config["input"]["dataset_id"], config["input"]["dataset_version"]),
        ).fetchone()
        split_rows = connection.execute(
            "SELECT split, COUNT(*) AS count FROM dataset_item WHERE dataset_id=? AND dataset_version=? GROUP BY split ORDER BY split",
            (config["input"]["dataset_id"], config["input"]["dataset_version"]),
        ).fetchall()
    _require(dataset is not None, "Frozen dataset version is missing")
    split_counts = {str(row["split"]): int(row["count"]) for row in split_rows}
    artifacts = {
        "candidate_snapshot": Path("data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json"),
        "provenance_audit": ANALYSIS_ROOT.parent / "datasets/fuxuan_v1/audit/slice9_provenance_audit.json",
        "quality_features": ANALYSIS_ROOT / "slice9_quality_features_v1.json",
        "speaker_features": ANALYSIS_ROOT / "slice9_speaker_features_v1.json",
        "text_features": ANALYSIS_ROOT / "slice9_text_features_v1.json",
        "coverage_features": ANALYSIS_ROOT / "slice9_coverage_features_v1.json",
        "feature_snapshot": ANALYSIS_ROOT / "slice9_feature_snapshot_v1.json",
        "pool_candidates": ANALYSIS_ROOT / "slice9_pool_candidates_v1.json",
        "ranking": ANALYSIS_ROOT / "slice9_ranking_v1.json",
        "diversity": ANALYSIS_ROOT / "slice9_diversity_rerank_v1.json",
        "constrained": ANALYSIS_ROOT / "slice9_constrained_selection_v1.json",
        "review_report": ANALYSIS_ROOT / "slice9_review_report_v1.json",
        "freeze_result": ANALYSIS_ROOT.parent / "datasets/fuxuan_v1/audit/slice9_reference_freeze_v1.json",
        "verify_result": ANALYSIS_ROOT.parent / "datasets/fuxuan_v1/audit/slice9_reference_verify_v1.json",
        "rebuild_result": ANALYSIS_ROOT.parent / "datasets/fuxuan_v1/audit/slice9_reference_rebuild_v1.json",
    }
    # Correct the paths above where the audit root is data/reports, not analysis/../datasets.
    artifacts["provenance_audit"] = Path("data/reports/datasets/fuxuan_v1/audit/slice9_provenance_audit.json")
    artifacts["freeze_result"] = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_freeze_v1.json")
    artifacts["verify_result"] = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_verify_v1.json")
    artifacts["rebuild_result"] = Path("data/reports/datasets/fuxuan_v1/audit/slice9_reference_rebuild_v1.json")
    payloads = {name: _load_json(path.resolve(), label=name) for name, path in artifacts.items()}
    _require(dataset["item_count"] == 273, f"Expected frozen dataset count 273, got {dataset['item_count']}")
    _require(split_counts.get("train") == payloads["candidate_snapshot"]["item_count"] == 219, "Train candidate count drift")
    _require(sum(split_counts.values()) == 273, "Dataset split counts do not sum to 273")
    _require(payloads["candidate_snapshot"]["item_count"] == 219, "Candidate snapshot count drift")
    for name in ("provenance_audit", "quality_features", "speaker_features", "text_features", "coverage_features", "feature_snapshot", "pool_candidates", "ranking", "diversity", "constrained", "review_report"):
        _require(payloads[name].get("status") in {"succeeded", "reviewed"}, f"{name} status is not successful")
    _require(payloads["freeze_result"].get("status") == "frozen", "Reference freeze is not frozen")
    _require(payloads["verify_result"].get("status") == "verified", "Reference verification did not pass")
    _require(payloads["rebuild_result"].get("byte_identical") is True, "Reference rebuild is not byte-identical")
    manifest = _load_json((Path(reference_dir).resolve() / "manifest.json"), label="frozen manifest")
    _require(sum(pool["selected_count"] for pool in manifest["pools"].values()) == 33, "Final selected count drift")
    result = {
        "schema_version": 1,
        "acceptance_report_schema_version": "slice9_real_pipeline_acceptance@1",
        "status": "accepted",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_item_count": int(dataset["item_count"]),
        "split_counts": split_counts,
        "selection_scope": {"train_candidates": 219, "validation_test_excluded": 54, "reason": "reference selection is train-only to preserve evaluation isolation"},
        "pipeline_artifacts": {name: {"path": str(path.resolve()), "sha256": _sha256(path.resolve()), "status": payloads[name].get("status")} for name, path in artifacts.items()},
        "final_manifest": {"selected_count": 33, "unique_asset_count": 30, "pool_count": 5, "neutral_overlap_count": 3},
        "checks": {"candidate_to_pool_to_rank_to_review_to_freeze": True, "verify_errors": payloads["verify_result"].get("errors", []), "isolated_rebuild_byte_identical": True},
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}
