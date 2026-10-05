from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_review import build_long_form_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a long-form audio review page.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_long_form_review(args.manifest, output_path=args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

