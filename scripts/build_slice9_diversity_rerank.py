from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_diversity import DEFAULT_SLICE9_DIVERSITY_REPORT_PATH, build_slice9_diversity_rerank


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Slice 9 diversity rerank")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--ranking-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_ranking_v1.json")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_DIVERSITY_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_diversity_rerank(
        config_path=Path(args.config),
        ranking_report_path=Path(args.ranking_report),
        feature_snapshot_path=Path(args.feature_snapshot),
        output_path=Path(args.output),
    )
    print(json.dumps({
        "status": result["status"],
        "pool_counts": {pool_id: {"candidate_count": pool["candidate_count"], "selected_count": pool["selected_count"], "boundary_count": len(pool["boundary"])} for pool_id, pool in result["pools"].items()},
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
