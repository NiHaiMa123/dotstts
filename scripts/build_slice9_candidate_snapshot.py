from __future__ import annotations

import argparse
import json

from dots_tts_lab.slice9_candidates import (
    DEFAULT_SLICE9_AUDIT_REPORT_PATH,
    DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH,
    DEFAULT_SLICE9_CONFIG_PATH,
    build_slice9_candidate_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Slice 9 train-only candidate snapshot from catalog.dataset_item."
    )
    parser.add_argument("--catalog", default="data/catalog/catalog.sqlite")
    parser.add_argument("--config", default=str(DEFAULT_SLICE9_CONFIG_PATH))
    parser.add_argument("--snapshot", default=str(DEFAULT_SLICE9_CANDIDATE_SNAPSHOT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_SLICE9_AUDIT_REPORT_PATH))
    args = parser.parse_args()
    summary = build_slice9_candidate_snapshot(
        catalog_path=args.catalog,
        config_path=args.config,
        snapshot_path=args.snapshot,
        report_path=args.report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

