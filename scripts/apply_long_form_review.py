from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_review import apply_long_form_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply append-only long-form review decisions.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_long_form_review(args.manifest, args.decisions, output_dir=args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

