from __future__ import annotations

import argparse
import json

from dots_tts_lab.slice9_provenance import (
    DEFAULT_SLICE9_PROVENANCE_REPORT_PATH,
    verify_slice9_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Slice 9 catalog, standardized audio, quality, speaker and text provenance."
    )
    parser.add_argument("--catalog", default="data/catalog/catalog.sqlite")
    parser.add_argument("--config", default="configs/lab/datasets/fuxuan_slice9_v1.yaml")
    parser.add_argument("--snapshot", default="data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json")
    parser.add_argument("--standardization-report", default="data/reports/standardization/standardization.json")
    parser.add_argument("--standardization-integrity", default="data/reports/standardization/integrity_audit.json")
    parser.add_argument("--quality-report", default="data/reports/quality/quality.json")
    parser.add_argument("--speaker-report", default="data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json")
    parser.add_argument("--report", default=str(DEFAULT_SLICE9_PROVENANCE_REPORT_PATH))
    args = parser.parse_args()
    summary = verify_slice9_provenance(
        catalog_path=args.catalog,
        config_path=args.config,
        snapshot_path=args.snapshot,
        standardization_report_path=args.standardization_report,
        standardization_integrity_path=args.standardization_integrity,
        quality_report_path=args.quality_report,
        speaker_report_path=args.speaker_report,
        report_path=args.report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

