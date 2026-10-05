from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_constraints import DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH, build_slice9_constrained_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Slice 9 reviewed duplicate and neutral overlap constraints")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--diversity-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_diversity_rerank_v1.json")
    parser.add_argument("--ranking-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_ranking_v1.json")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--duplicate-report", default="data/reports/datasets/fuxuan_v1/analysis/dataset_analysis.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_constrained_selection(
        config_path=Path(args.config),
        diversity_report_path=Path(args.diversity_report),
        ranking_report_path=Path(args.ranking_report),
        feature_snapshot_path=Path(args.feature_snapshot),
        duplicate_report_path=Path(args.duplicate_report),
        output_path=Path(args.output),
    )
    print(json.dumps({
        "status": result["status"],
        "duplicate_group_count": result["duplicate_group_count"],
        "reviewed_duplicate_group_count": result["reviewed_duplicate_group_count"],
        "neutral_overlap": result["neutral_overlap"],
        "pool_counts": {pool_id: {"selected": pool["selected_count"], "replacement_count": pool["replacement_count"], "meets_top_k": pool["meets_top_k"]} for pool_id, pool in result["pools"].items()},
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

