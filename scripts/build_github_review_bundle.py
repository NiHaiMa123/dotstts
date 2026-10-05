from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.dataset_analysis_report import (
    write_portable_dataset_analysis_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a portable GitHub review bundle with referenced audio only."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--online-url")
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    result = write_portable_dataset_analysis_bundle(
        report,
        args.output_dir,
        online_url=args.online_url,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
