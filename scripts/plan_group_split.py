from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    canonical_dataset_config,
    load_dataset_freeze_config,
)
from dots_tts_lab.dataset_split import (
    load_verified_similarity_groups,
    plan_emotion_stratified_split,
    plan_group_aware_split,
)
from dots_tts_lab.duplicate_graph import load_verified_json
from dots_tts_lab.reports import _atomic_write_text


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Build a deterministic pre-stratification group split plan."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot-sha256", required=True)
    parser.add_argument("--stratify-emotion", action="store_true")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
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
    groups, review_snapshot_sha256 = load_verified_similarity_groups(
        Catalog(args.catalog),
        run_id=args.run_id,
        candidate_snapshot_sha256=args.candidate_snapshot_sha256,
        candidate_count=len(items),
    )
    planner = (
        plan_emotion_stratified_split
        if args.stratify_emotion
        else plan_group_aware_split
    )
    plan = planner(
        items=items,
        groups=groups,
        split_config=config.split,
        dataset_config_sha256=config_identity["sha256"],
    )
    output = {
        **plan,
        "analysis_run_id": args.run_id,
        "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
        "edge_review_snapshot_sha256": review_snapshot_sha256,
        "note": _plan_note(args.stratify_emotion),
    }
    content = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(args.output.resolve(), content)
    result = {
        "status": "succeeded",
        "output_path": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "assignment_sha256": plan["assignment_sha256"],
        "candidate_count": plan["candidate_count"],
        "group_count": plan["group_count"],
        "summary": plan["summary"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _plan_note(stratified: bool) -> str:
    if stratified:
        return (
            "Emotion-stratified pre-exclusion check only; final freeze will rerun "
            "after applying reviewed duplicate and speaker exclusions."
        )
    return (
        "Pre-stratification contract check only; final freeze will rerun after "
        "manual exclusions and emotion-strata balancing."
    )


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
