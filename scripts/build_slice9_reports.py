from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_reports import (
    DEFAULT_SLICE9_CSV_PATH,
    DEFAULT_SLICE9_HTML_PATH,
    DEFAULT_SLICE9_REVIEW_JSON_PATH,
    build_slice9_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Slice 9 JSON/CSV/HTML manual review reports")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--constrained-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_constrained_selection_v1.json")
    parser.add_argument("--sensitivity-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_sensitivity_v1.json")
    parser.add_argument("--json-output", default=str(DEFAULT_SLICE9_REVIEW_JSON_PATH))
    parser.add_argument("--csv-output", default=str(DEFAULT_SLICE9_CSV_PATH))
    parser.add_argument("--html-output", default=str(DEFAULT_SLICE9_HTML_PATH))
    args = parser.parse_args()
    result = build_slice9_reports(
        config_path=Path(args.config),
        constrained_report_path=Path(args.constrained_report),
        sensitivity_report_path=Path(args.sensitivity_report),
        output_json_path=Path(args.json_output),
        output_csv_path=Path(args.csv_output),
        output_html_path=Path(args.html_output),
    )
    print(json.dumps({"status": result["status"], "row_count": result["row_count"], "pool_summaries": result["pool_summaries"], "report_paths": result["report_paths"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

