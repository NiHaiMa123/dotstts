from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    canonical_dataset_config,
    load_dataset_freeze_config,
)
from dots_tts_lab.dataset_selection import apply_reviewed_exclusions
from dots_tts_lab.dataset_split import (
    load_verified_similarity_groups,
    plan_emotion_stratified_split,
)
from dots_tts_lab.duplicate_graph import canonical_json, load_verified_json
from dots_tts_lab.reports import _atomic_write_text


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Apply reviewed exclusions and build the final-candidate split plan."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot-sha256", required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    )
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_dataset_freeze_config(args.dataset_config)
    config_identity = canonical_dataset_config(config)
    snapshot = load_verified_json(
        args.candidate_snapshot,
        args.candidate_snapshot_sha256,
    )
    items = snapshot.get("items")
    if not isinstance(items, list) or snapshot.get("item_count") != len(items):
        raise RuntimeError("Candidate snapshot count mismatch")
    review = yaml.safe_load(args.review_config.resolve().read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise RuntimeError("Manual review config must be a mapping")
    if (
        review.get("analysis_run_id") != args.run_id
        or review.get("candidate_snapshot_sha256")
        != args.candidate_snapshot_sha256
    ):
        raise RuntimeError("Manual review identity differs from selection inputs")
    groups, review_snapshot_sha256 = load_verified_similarity_groups(
        Catalog(args.catalog),
        run_id=args.run_id,
        candidate_snapshot_sha256=args.candidate_snapshot_sha256,
        candidate_count=len(items),
    )
    selection = apply_reviewed_exclusions(
        items=items,
        groups=groups,
        review=review,
    )
    review_config_sha256 = hashlib.sha256(
        canonical_json(review).encode("utf-8")
    ).hexdigest()
    selection_artifact = {
        "schema_version": 1,
        "status": "selected",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "analysis_run_id": args.run_id,
        "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
        "edge_review_snapshot_sha256": review_snapshot_sha256,
        "review_config_sha256": review_config_sha256,
        "dataset_config_sha256": config_identity["sha256"],
        **selection,
    }
    selection_content = (
        json.dumps(selection_artifact, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    _atomic_write_text(args.selection_output.resolve(), selection_content)

    split = plan_emotion_stratified_split(
        items=selection["selected_items"],
        groups=selection["selected_groups"],
        split_config=config.split,
        dataset_config_sha256=config_identity["sha256"],
    )
    split_artifact = {
        **split,
        "scope": "reviewed_selection_final_candidate_split",
        "analysis_run_id": args.run_id,
        "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
        "edge_review_snapshot_sha256": review_snapshot_sha256,
        "review_config_sha256": review_config_sha256,
        "selection_sha256": selection["selection_sha256"],
        "note": (
            "Final candidate assignment after reviewed duplicate and speaker "
            "exclusions; no sampling or replication was applied."
        ),
    }
    split_content = (
        json.dumps(split_artifact, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    _atomic_write_text(args.split_output.resolve(), split_content)
    result = {
        "status": "succeeded",
        "selection_output_sha256": hashlib.sha256(
            selection_content.encode("utf-8")
        ).hexdigest(),
        "split_output_sha256": hashlib.sha256(
            split_content.encode("utf-8")
        ).hexdigest(),
        "selection_sha256": selection["selection_sha256"],
        "assignment_sha256": split["assignment_sha256"],
        "candidate_count": selection["candidate_count"],
        "selected_count": selection["selected_count"],
        "excluded_count": selection["excluded_count"],
        "oversampled_count": selection["oversampled_count"],
        "duplicate_resolution_count": selection["duplicate_resolution_count"],
        "split_summary": split["summary"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
