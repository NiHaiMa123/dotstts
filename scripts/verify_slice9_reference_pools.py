from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_freeze import DEFAULT_SLICE9_CATALOG_PATH, DEFAULT_SLICE9_REFERENCE_DIR
from dots_tts_lab.slice9_verify import DEFAULT_SLICE9_VERIFY_RESULT_PATH, verify_slice9_reference_pools


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Slice 9 reference pools")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--reference-dir", default=str(DEFAULT_SLICE9_REFERENCE_DIR))
    parser.add_argument("--catalog", default=str(DEFAULT_SLICE9_CATALOG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_VERIFY_RESULT_PATH))
    args = parser.parse_args()
    result = verify_slice9_reference_pools(
        config_path=Path(args.config), reference_dir=Path(args.reference_dir), catalog_path=Path(args.catalog), output_path=Path(args.output)
    )
    print(json.dumps({key: result[key] for key in ("status", "verified_item_count", "verified_pool_count", "unique_asset_count", "neutral_overlap_count", "report_path")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

