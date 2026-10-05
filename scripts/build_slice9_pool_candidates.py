from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_pools import (
    DEFAULT_SLICE9_POOL_REPORT_PATH,
    build_slice9_pool_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated Slice 9 candidate pools")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_POOL_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_pool_candidates(
        config_path=Path(args.config),
        feature_snapshot_path=Path(args.feature_snapshot),
        output_path=Path(args.output),
    )
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "hard_gate_decision_counts": result["hard_gate_decision_counts"],
        "shortfall_pool_ids": result["shortfall_pool_ids"],
        "pool_counts": {pool_id: pool["candidate_count"] for pool_id, pool in result["pools"].items()},
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
