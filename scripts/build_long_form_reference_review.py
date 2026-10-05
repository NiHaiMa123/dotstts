from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_reference import build_reference_review_html


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the reference-pack confirmation page (LF-12A)."
    )
    parser.add_argument(
        "--proposal",
        default="data/reports/long_form/reference_pack_proposal_v1.json",
    )
    parser.add_argument(
        "--output",
        default="data/reports/long_form/reference_pack_review_v1.html",
    )
    args = parser.parse_args()
    result = build_reference_review_html(args.proposal, args.output)
    print(f"review page: {result['review_path']}")
    print(f"sha256: {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
