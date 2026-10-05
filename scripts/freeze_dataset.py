from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from dots_tts_lab.dataset_artifacts import (
    build_dataset_artifact_set,
    validate_dataset_artifact_set,
)
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    load_dataset_freeze_config,
)
from dots_tts_lab.dataset_publish import atomic_publish_dataset
from dots_tts_lab.dataset_registry import (
    publish_dataset_catalog,
    validate_frozen_dataset,
)
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.duplicate_graph import load_verified_json


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Build and atomically publish an immutable dataset artifact set."
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
    parser.add_argument("--target", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()

    config = load_dataset_freeze_config(args.dataset_config)
    selection = load_verified_json(args.selection, args.selection_sha256)
    split_plan = load_verified_json(args.split_plan, args.split_plan_sha256)
    split_audit = load_verified_json(args.split_audit, args.split_audit_sha256)
    target = (args.target or Path(config.output.dataset_root)).resolve()
    input_hashes = {
        "reviewed_selection.json": args.selection_sha256,
        "reviewed_selection_split.json": args.split_plan_sha256,
        "split_audit.json": args.split_audit_sha256,
    }

    result = atomic_publish_dataset(
        target,
        build=lambda staging: build_dataset_artifact_set(
            staging,
            config=config,
            selection=selection,
            split_plan=split_plan,
            split_audit=split_audit,
            input_file_sha256s=input_hashes,
            standardized_root=args.standardized_root,
        ),
        validate=lambda staging: validate_dataset_artifact_set(
            staging, config=config
        ),
    )
    result["validation"] = validate_dataset_artifact_set(target, config=config)
    result["deep_validation"] = validate_frozen_dataset(target, config=config)
    if args.catalog is not None:
        published_at = dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        catalog = Catalog(args.catalog)
        result["catalog"] = publish_dataset_catalog(
            catalog,
            target,
            config=config,
            published_at=published_at,
        )
        result["catalog_validation"] = validate_frozen_dataset(
            target, config=config, catalog=catalog
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
