#!/usr/bin/env python3
"""Validate and unblind a completed Slice12 paired listening review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY = ROOT / "data/reports/datasets/fuxuan_v1/slice12/slice12_blind_review_key.json"
DEFAULT_OUTPUT = ROOT / "data/reports/datasets/fuxuan_v1/slice12"
PREFERENCES = {"A", "B", "tie", "both_bad"}
SILENCE_CHOICES = {"A_worse", "B_worse", "same", "neither"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions-name", default="slice12_blind_review_decisions.json")
    parser.add_argument("--report-name", default="slice12_manual_review.json")
    parser.add_argument("--candidate-id", default="step500_candidate")
    args = parser.parse_args()

    decisions_path = args.decisions.resolve()
    key_path = args.key.resolve()
    decisions = load_object(decisions_path)
    key = load_object(key_path)
    if decisions.get("schema_version") != 1 or decisions.get("status") != "manual_review_export":
        raise ValueError("Unsupported Slice12 blind review export")
    if decisions.get("review_status") != "review_complete":
        raise ValueError("Blind review is not marked complete")
    ratings = decisions.get("ratings")
    if not isinstance(ratings, dict):
        raise ValueError("Blind review ratings must be an object")
    key_rows = key.get("items")
    if key.get("status") != "sealed_key" or not isinstance(key_rows, list):
        raise ValueError("Invalid Slice12 sealed key")
    by_pair = {str(row["pair_id"]): row for row in key_rows}
    if len(by_pair) != 12 or set(ratings) != set(by_pair):
        raise ValueError("Blind review must contain exactly the 12 sealed pairs")

    preference_counts = {"control": 0, "trained": 0, "tie": 0, "both_bad": 0}
    silence_counts = {"control_worse": 0, "trained_worse": 0, "same": 0, "neither": 0}
    unblinded: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        rating = ratings[pair_id]
        preference = rating.get("preference")
        silence = rating.get("leading_silence")
        if preference not in PREFERENCES:
            raise ValueError(f"Invalid preference for {pair_id}: {preference!r}")
        if silence not in SILENCE_CHOICES:
            raise ValueError(f"Invalid leading-silence choice for {pair_id}: {silence!r}")
        key_row = by_pair[pair_id]
        preferred_role = None
        if preference in {"A", "B"}:
            preferred_role = key_row[preference]
            preference_counts[preferred_role] += 1
        else:
            preference_counts[preference] += 1
        worse_silence_role = None
        if silence in {"A_worse", "B_worse"}:
            worse_silence_role = key_row[silence[0]]
            silence_counts[f"{worse_silence_role}_worse"] += 1
        else:
            silence_counts[silence] += 1
        unblinded.append(
            {
                "pair_id": pair_id,
                "ordinal": int(key_row["ordinal"]),
                "sentence_id": key_row["sentence_id"],
                "seed": int(key_row["seed"]),
                "A": key_row["A"],
                "B": key_row["B"],
                "preference": preference,
                "preferred_role": preferred_role,
                "leading_silence": silence,
                "worse_silence_role": worse_silence_role,
                "note": rating.get("note") or "",
                "saved_at": rating.get("saved_at"),
            }
        )

    decisive = preference_counts["control"] + preference_counts["trained"]
    trained_win_rate = preference_counts["trained"] / max(1, decisive)
    if preference_counts["trained"] > preference_counts["control"]:
        gate_status = "accepted"
        recommendation = f"accept_{args.candidate_id}"
    elif preference_counts["control"] > preference_counts["trained"]:
        gate_status = "rejected"
        recommendation = f"reject_{args.candidate_id}"
    else:
        gate_status = "inconclusive"
        recommendation = "collect_more_manual_reviews"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_decisions_path = output_dir / args.decisions_name
    canonical_decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "decision_source_sha256": sha256_file(decisions_path),
        "canonical_decisions_sha256": sha256_file(canonical_decisions_path),
        "sealed_key_sha256": sha256_file(key_path),
        "pair_count": len(unblinded),
        "preference_counts": preference_counts,
        "decisive_pair_count": decisive,
        "trained_win_rate_among_decisive": trained_win_rate,
        "leading_silence_counts": silence_counts,
        "manual_gate": {
            "status": gate_status,
            "recommendation": recommendation,
            "reason": (
                "trained wins fewer decisive pairs than control"
                if gate_status == "rejected"
                else "manual preference result"
            ),
        },
        "items": unblinded,
    }
    report_path = output_dir / args.report_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["manual_gate"]["status"],
                "recommendation": report["manual_gate"]["recommendation"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
