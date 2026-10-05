from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_freeze import (
    DEFAULT_SLICE9_CATALOG_PATH,
    DEFAULT_SLICE9_FREEZE_RESULT_PATH,
    DEFAULT_SLICE9_REFERENCE_DIR,
    build_slice9_reference_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze approved Slice 9 reference pools")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--constrained-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_constrained_selection_v1.json")
    parser.add_argument("--catalog", default=str(DEFAULT_SLICE9_CATALOG_PATH))
    parser.add_argument("--reference-dir", default=str(DEFAULT_SLICE9_REFERENCE_DIR))
    parser.add_argument("--result", default=str(DEFAULT_SLICE9_FREEZE_RESULT_PATH))
    args = parser.parse_args()
    result = build_slice9_reference_freeze(
        config_path=Path(args.config),
        constrained_report_path=Path(args.constrained_report),
        catalog_path=Path(args.catalog),
        reference_dir=Path(args.reference_dir),
        result_path=Path(args.result),
    )
    print(json.dumps({key: result[key] for key in ("status", "selected_count_total", "unique_asset_count", "neutral_overlap_count", "manifest_sha256", "stats_sha256", "config_sha256", "checksums_sha256", "reference_dir", "result_path")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

