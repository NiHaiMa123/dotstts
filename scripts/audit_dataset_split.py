from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    load_dataset_freeze_config,
)
from dots_tts_lab.dataset_split_audit import (
    assert_split_audit_passes,
    audit_split_leakage,
)
from dots_tts_lab.duplicate_graph import load_verified_json
from dots_tts_lab.reports import _atomic_write_text


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Audit a reviewed dataset split for cross-split leakage."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--split-plan-sha256", required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = load_verified_json(args.selection, args.selection_sha256)
    split_plan = load_verified_json(args.split_plan, args.split_plan_sha256)
    if (
        selection.get("analysis_run_id") != args.run_id
        or split_plan.get("analysis_run_id") != args.run_id
        or split_plan.get("selection_sha256") != selection.get("selection_sha256")
    ):
        raise RuntimeError("Split audit input identities differ")
    with Catalog(args.catalog).read_only_session() as connection:
        edges = [
            dict(row)
            for row in connection.execute(
                """
                SELECT left_asset_sha256, right_asset_sha256, evidence_type,
                       score, threshold, analysis_status
                FROM dataset_similarity_edge
                WHERE run_id = ?
                ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                """,
                (args.run_id,),
            )
        ]
    config = load_dataset_freeze_config(args.dataset_config)
    audit = audit_split_leakage(
        items=selection["selected_items"],
        split_plan=split_plan,
        similarity_edges=edges,
        text_config=config.text_similarity,
    )
    output = {
        **audit,
        "analysis_run_id": args.run_id,
        "selection_file_sha256": args.selection_sha256,
        "selection_sha256": selection["selection_sha256"],
        "split_plan_file_sha256": args.split_plan_sha256,
        "assignment_sha256": split_plan["assignment_sha256"],
    }
    content = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(args.output.resolve(), content)
    assert_split_audit_passes(audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_path": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "audit_sha256": audit["audit_sha256"],
                "item_count": audit["item_count"],
                "split_counts": audit["split_counts"],
                "checks": audit["checks"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
