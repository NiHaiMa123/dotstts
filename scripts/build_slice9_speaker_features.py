from __future__ import annotations

import argparse
import json

from dots_tts_lab.slice9_speaker import (
    DEFAULT_SLICE9_SPEAKER_FEATURE_REPORT_PATH,
    build_slice9_speaker_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Slice 9 CAM++ speaker features and manual exclusion decisions.")
    parser.add_argument("--catalog", default="data/catalog/catalog.sqlite")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--snapshot", default="data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json")
    parser.add_argument("--speaker-report", default="data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json")
    parser.add_argument("--threshold-report", default="data/reports/datasets/fuxuan_v1/analysis/threshold_calibration_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_SPEAKER_FEATURE_REPORT_PATH))
    args = parser.parse_args()
    summary = build_slice9_speaker_features(
        catalog_path=args.catalog,
        config_path=args.config,
        snapshot_path=args.snapshot,
        speaker_report_path=args.speaker_report,
        threshold_report_path=args.threshold_report,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

