from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_scoring import DEFAULT_SLICE9_RANKING_REPORT_PATH, build_slice9_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explainable Slice 9 pool-scoped ranking")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--normalized-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_normalized_features_v1.json")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_RANKING_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_ranking(
        config_path=Path(args.config),
        normalized_report_path=Path(args.normalized_report),
        feature_snapshot_path=Path(args.feature_snapshot),
        output_path=Path(args.output),
    )
    print(json.dumps({
        "status": result["status"],
        "pool_count": result["pool_count"],
        "pool_counts": {pool_id: pool["candidate_count"] for pool_id, pool in result["pools"].items()},
        "available_weight": {pool_id: pool["available_weight_distribution"] for pool_id, pool in result["pools"].items()},
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

