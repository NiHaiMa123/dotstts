from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_freeze import DEFAULT_SLICE9_CATALOG_PATH, DEFAULT_SLICE9_REFERENCE_DIR
from dots_tts_lab.slice9_rebuild import DEFAULT_SLICE9_REBUILD_RESULT_PATH, rebuild_slice9_reference_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Slice 9 frozen artifacts in a temporary directory")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--constrained-report", default="data/reports/datasets/fuxuan_v1/analysis/slice9_constrained_selection_v1.json")
    parser.add_argument("--catalog", default=str(DEFAULT_SLICE9_CATALOG_PATH))
    parser.add_argument("--reference-dir", default=str(DEFAULT_SLICE9_REFERENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_REBUILD_RESULT_PATH))
    args = parser.parse_args()
    result = rebuild_slice9_reference_artifacts(
        config_path=Path(args.config), constrained_report_path=Path(args.constrained_report), catalog_path=Path(args.catalog), reference_dir=Path(args.reference_dir), output_path=Path(args.output)
    )
    print(json.dumps({"status": result["status"], "artifact_count": result["artifact_count"], "byte_identical": result["byte_identical"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

