from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_review import (
    DEFAULT_SLICE9_CATALOG_PATH,
    DEFAULT_SLICE9_REVIEW_RESULT_PATH,
    apply_slice9_reference_reviews,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Slice 9 reference listening decisions to catalog")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--review-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_review_report_v1.json")
    parser.add_argument("--catalog", default=str(DEFAULT_SLICE9_CATALOG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_REVIEW_RESULT_PATH))
    parser.add_argument("--review-status", choices=("approved", "rejected", "uncertain"), default="approved")
    parser.add_argument("--review-note", default="user_confirmed_no_issues")
    args = parser.parse_args()
    result = apply_slice9_reference_reviews(
        config_path=Path(args.config),
        review_report_path=Path(args.review_report),
        catalog_path=Path(args.catalog),
        output_path=Path(args.output),
        review_status=args.review_status,
        review_note=args.review_note,
    )
    print(json.dumps({key: result[key] for key in ("status", "review_batch_id", "review_status", "row_count", "inserted", "ignored_idempotent_replay", "report_path")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

