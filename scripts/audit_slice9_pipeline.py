from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_acceptance import DEFAULT_SLICE9_ACCEPTANCE_RESULT_PATH, build_slice9_acceptance_audit
from dots_tts_lab.slice9_freeze import DEFAULT_SLICE9_CATALOG_PATH, DEFAULT_SLICE9_REFERENCE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the real Slice 9 candidate-to-freeze pipeline")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--catalog", default=str(DEFAULT_SLICE9_CATALOG_PATH))
    parser.add_argument("--reference-dir", default=str(DEFAULT_SLICE9_REFERENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_ACCEPTANCE_RESULT_PATH))
    args = parser.parse_args()
    result = build_slice9_acceptance_audit(config_path=Path(args.config), catalog_path=Path(args.catalog), reference_dir=Path(args.reference_dir), output_path=Path(args.output))
    print(json.dumps({"status": result["status"], "dataset_item_count": result["dataset_item_count"], "split_counts": result["split_counts"], "selection_scope": result["selection_scope"], "final_manifest": result["final_manifest"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

