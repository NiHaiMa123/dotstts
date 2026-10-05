from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from dots_tts_lab.dataset_artifacts import (
    build_dataset_artifact_set,
    validate_dataset_artifact_set,
)
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    load_dataset_freeze_config,
)
from dots_tts_lab.dataset_publish import dataset_tree_inventory
from dots_tts_lab.duplicate_graph import load_verified_json
from dots_tts_lab.reports import _atomic_write_text


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Independently rebuild and byte-compare a frozen dataset version."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--split-plan-sha256", required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--split-audit-sha256", required=True)
    parser.add_argument(
        "--dataset-config", type=Path, default=DEFAULT_DATASET_FREEZE_CONFIG_PATH
    )
    parser.add_argument(
        "--standardized-root", type=Path, default=Path("data/work/standardized")
    )
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("data/work"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = load_dataset_freeze_config(args.dataset_config)
    selection = load_verified_json(args.selection, args.selection_sha256)
    split_plan = load_verified_json(args.split_plan, args.split_plan_sha256)
    split_audit = load_verified_json(args.split_audit, args.split_audit_sha256)
    published = dataset_tree_inventory(args.published_root)
    args.work_root.resolve().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="slice8-independent-rebuild-", dir=args.work_root.resolve()
    ) as temp_dir:
        rebuilt_root = Path(temp_dir) / "v1"
        build_dataset_artifact_set(
            rebuilt_root,
            config=config,
            selection=selection,
            split_plan=split_plan,
            split_audit=split_audit,
            input_file_sha256s={
                "reviewed_selection.json": args.selection_sha256,
                "reviewed_selection_split.json": args.split_plan_sha256,
                "split_audit.json": args.split_audit_sha256,
            },
            standardized_root=args.standardized_root,
        )
        validation = validate_dataset_artifact_set(rebuilt_root, config=config)
        rebuilt = dataset_tree_inventory(rebuilt_root)
    if rebuilt != published:
        raise RuntimeError("Independent rebuild differs from published dataset v1")
    report = {
        "schema_version": 1,
        "status": "passed",
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "normalization": {"allowed_runtime_fields": []},
        "artifact_count": len(published["files"]),
        "tree_sha256": published["tree_sha256"],
        "files": published["files"],
        "validation": validation,
    }
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(args.report.resolve(), content)
    result = {
        "status": "passed",
        "report": str(args.report.resolve()),
        "report_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "artifact_count": report["artifact_count"],
        "tree_sha256": report["tree_sha256"],
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
