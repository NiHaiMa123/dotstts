from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_batch import build_text_review_html

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the text-correction page for human-confirmed "
        "candidates (LF-12H follow-up)."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--decisions",
        required=True,
        help="strict-review decisions JSON (incoming/strict-review-*.json)",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    source_hash = file_sha256(source)
    sha12 = source_hash[:12]

    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    confirmed = {
        cid for cid, d in decisions.get("decisions", {}).items()
        if d.get("decision") == "confirm"
    }

    transcript = json.loads(
        (
            ROOT / "data/work/long_form/transcript" / sha12[:2] / source_hash
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    review_page = json.loads(
        (
            ROOT / "data/reports/long_form/incoming"
            / f"strict-review-{sha12}.json"
        ).read_text(encoding="utf-8")
    ) if (
        ROOT / "data/reports/long_form/incoming" / f"strict-review-{sha12}.json"
    ).is_file() else {"batch_sha256": decisions.get("batch_sha256")}

    import hashlib

    def cid(span: dict) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "source_sha256": transcript.get("source_sha256"),
                    "start": span["source_start_frame"],
                    "end": span["source_end_frame"],
                    "text": span.get("primary_text"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    items = []
    for span in transcript.get("sentences", []):
        span_id = cid(span)
        if span_id not in confirmed:
            continue
        items.append(
            {
                "candidate_id": span_id,
                "raw_audio": (
                    f"/data/reports/long_form/strict_review/{sha12}"
                    f"/audio/{span_id[:16]}_raw.wav"
                ),
                "routed_audio": (
                    f"/data/reports/long_form/strict_review/{sha12}"
                    f"/audio/{span_id[:16]}_routed.wav"
                    if (
                        ROOT / "data/reports/long_form/strict_review" / sha12
                        / "audio" / f"{span_id[:16]}_routed.wav"
                    ).is_file()
                    else None
                ),
                "primary_text": span.get("primary_text"),
                "secondary_text": span.get("secondary_text"),
                "text_agreement": span.get("text_agreement", {}).get("status"),
                "duration_seconds": span["duration_seconds"],
            }
        )
    items.sort(key=lambda i: (i["text_agreement"] == "match", -i["duration_seconds"]))

    html = build_text_review_html(
        title=source.name,
        source_sha256=source_hash,
        items=items,
        save_name=f"strict-review-text-{sha12}",
        batch_sha256=str(review_page.get("batch_sha256") or ""),
    )
    out = ROOT / "data/reports/long_form" / f"strict_text_{sha12}.html"
    out.write_text(html, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "page": str(out),
        "items": len(items),
        "needs_correction": sum(
            1 for i in items if i["text_agreement"] != "match"
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
