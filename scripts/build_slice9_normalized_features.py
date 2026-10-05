from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_normalization import (
    DEFAULT_SLICE9_NORMALIZED_REPORT_PATH,
    build_slice9_normalized_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pool-local Slice 9 normalized features")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--pool-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_pool_candidates_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_NORMALIZED_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_normalized_features(
        config_path=Path(args.config),
        feature_snapshot_path=Path(args.feature_snapshot),
        pool_report_path=Path(args.pool_report),
        output_path=Path(args.output),
    )
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "pool_counts": {pool_id: pool["candidate_count"] for pool_id, pool in result["pools"].items()},
        "missing_by_pool": {pool_id: {name: stats["missing_count"] for name, stats in pool["feature_stats"].items()} for pool_id, pool in result["pools"].items()},
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

