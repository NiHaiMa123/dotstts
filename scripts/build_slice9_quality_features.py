from __future__ import annotations

import argparse
import json

from dots_tts_lab.slice9_features import (
    DEFAULT_SLICE9_QUALITY_FEATURE_REPORT_PATH,
    build_slice9_quality_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Slice 9 standardized quality/silence/duration features.")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--snapshot", default="data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json")
    parser.add_argument("--standardization-report", default="data/reports/standardization/standardization.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_QUALITY_FEATURE_REPORT_PATH))
    args = parser.parse_args()
    summary = build_slice9_quality_features(
        config_path=args.config,
        snapshot_path=args.snapshot,
        standardization_report_path=args.standardization_report,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

