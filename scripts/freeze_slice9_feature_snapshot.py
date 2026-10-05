from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.slice9_feature_snapshot import (
    DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    build_slice9_feature_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Slice 9 candidate feature snapshot")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--candidate-snapshot", default="data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json")
    parser.add_argument("--output", default=str(DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH))
    args = parser.parse_args()
    result = build_slice9_feature_snapshot(
        config_path=Path(args.config),
        candidate_snapshot_path=Path(args.candidate_snapshot),
        output_path=Path(args.output),
    )
    print(json.dumps({key: result[key] for key in ("status", "freeze_id", "candidate_count", "input_hashes", "report_path", "reused")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

