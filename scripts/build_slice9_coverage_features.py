from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_coverage import (
    DEFAULT_SLICE9_COVERAGE_REPORT_PATH,
    DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH,
    build_slice9_coverage_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Slice 9 text/pinyin coverage features")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--text-report", default=str(DEFAULT_SLICE9_TEXT_FEATURE_REPORT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_COVERAGE_REPORT_PATH))
    args = parser.parse_args()
    result = build_slice9_coverage_features(
        config_path=Path(args.config),
        text_feature_path=Path(args.text_report),
        output_path=Path(args.output),
    )
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "coverage_status_counts", "report_path")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

