from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_analysis_report import (
    DEFAULT_DATASET_ANALYSIS_REPORT_DIR,
    build_dataset_analysis_report,
    write_dataset_analysis_reports,
)
from dots_tts_lab.duplicate_graph import load_verified_json
from dots_tts_lab.threshold_calibration import (
    DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    load_threshold_calibration_overlay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate verified JSON, CSV, and HTML dataset analysis reports."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot-sha256", required=True)
    parser.add_argument("--speaker-report", type=Path, required=True)
    parser.add_argument("--speaker-report-sha256", required=True)
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=Path("data/work/standardized"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_ANALYSIS_REPORT_DIR,
    )
    args = parser.parse_args()

    overlay = load_threshold_calibration_overlay(args.calibration_config)
    candidate_snapshot = load_verified_json(
        args.candidate_snapshot,
        args.candidate_snapshot_sha256,
    )
    speaker_report = load_verified_json(
        args.speaker_report,
        args.speaker_report_sha256,
    )
    calibration_report = load_verified_json(
        overlay.calibration_report_path,
        overlay.calibration_report_sha256,
    )
    report = build_dataset_analysis_report(
        catalog=Catalog(args.catalog),
        run_id=args.run_id,
        candidate_snapshot=candidate_snapshot,
        candidate_snapshot_sha256=args.candidate_snapshot_sha256,
        speaker_report=speaker_report,
        speaker_report_sha256=args.speaker_report_sha256,
        calibration_report=calibration_report,
        overlay=overlay,
        standardized_root=args.standardized_root,
    )
    outputs = write_dataset_analysis_reports(report, args.output_dir)
    print(
        json.dumps(
            {"status": report["status"], "summary": report["summary"], **outputs},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
