from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_reference import (
    build_negative_review_html,
    propose_negative_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scout hard negatives and build their confirmation page (LF-12A)."
    )
    parser.add_argument(
        "--sources-root", default="data/work/long_form/sources"
    )
    parser.add_argument("--per-kind", type=int, default=6)
    parser.add_argument(
        "--exclude",
        default=None,
        help="JSON file(s) with already-judged clip_ids to skip (e.g. a prior negatives export or the generated page data).",
    )
    parser.add_argument(
        "--only-kind",
        default=None,
        help="Only scout one kind, e.g. nearfield_binaural.",
    )
    parser.add_argument(
        "--output",
        default="data/reports/long_form/reference_negative_review_v1.html",
    )
    parser.add_argument(
        "--save-name",
        default="reference-negatives",
        help="Fixed incoming filename the page POSTs to (without .json).",
    )
    args = parser.parse_args()
    exclude_ids: set[str] = set()
    if args.exclude:
        import json

        for path in args.exclude.split(","):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            for item in payload.get("negatives", []):
                exclude_ids.add(item["clip_id"])
            for item in payload.get("items", []):
                exclude_ids.add(item["clip_id"])
            for item in payload.get("candidates", []):
                exclude_ids.add(item.get("clip_id") or item.get("audio_sha256"))
    candidates = propose_negative_candidates(
        args.sources_root, per_kind=args.per_kind, exclude_ids=exclude_ids
    )
    if args.only_kind:
        candidates = {args.only_kind: candidates.get(args.only_kind, [])}
    for kind, rows in candidates.items():
        print(f"{kind}: {len(rows)} candidates")
    result = build_negative_review_html(
        candidates, args.output, save_name=args.save_name
    )
    print(f"review page: {result['review_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
