from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_review import build_long_form_review_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local index for long-form review pages.")
    parser.add_argument("--work-root", type=Path, default=Path("data/work/long_form"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/long_form/review_index.html"))
    args = parser.parse_args()
    print(
        json.dumps(
            build_long_form_review_index(args.work_root, output_path=args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
