from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_sensitivity import DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH, build_slice9_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Slice 9 weight and threshold sensitivity analysis")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--ranking-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_ranking_v1.json")
    parser.add_argument("--feature-snapshot", default="data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_sensitivity(
        config_path=Path(args.config),
        ranking_report_path=Path(args.ranking_report),
        feature_snapshot_path=Path(args.feature_snapshot),
        output_path=Path(args.output),
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"], "threshold_variants": result["threshold_variants"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

