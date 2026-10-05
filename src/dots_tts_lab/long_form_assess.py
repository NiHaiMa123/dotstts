from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

ASSESS_IMPLEMENTATION_VERSION = 1


def _manifest(work: str, source_hash: str, name: str) -> Path:
    return (
        ROOT / "data/work/long_form" / work
        / source_hash[:2] / source_hash / name
    )


def _region_max(scores: dict | None) -> float:
    return max(
        (float(e["max"]) for e in (scores or {}).values()
         if e.get("max") is not None),
        default=0.0,
    )


def assess_source(
    source_hash: str,
    *,
    file_cosine: float | None,
    file_negative_cosines: dict[str, float] | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Per-file KEEP/DISCARD triage after Phase-A evidence stages.

    Combines prefilter/identity/style_event/transcript manifests plus a
    file-level speaker cosine (embedding of ~20 s concatenated usable
    speech vs the frozen normal-reference centroid — region-level cosines
    are unstable on whispered material). Thresholds here are triage aids,
    not calibrated gate thresholds; the strict gates still decide final
    candidacy.
    """
    work = work_root or (ROOT / "data/work/long_form")
    sha12 = source_hash[:12]
    prefilter = json.loads(
        _manifest("prefilter", source_hash, "regions.json")
        .read_text(encoding="utf-8")
    )
    identity = json.loads(
        _manifest("identity", source_hash, "identity.json")
        .read_text(encoding="utf-8")
    )
    style = json.loads(
        _manifest("style_event", source_hash, "style_event.json")
        .read_text(encoding="utf-8")
    )
    transcript_path = _manifest("transcript", source_hash, "transcript.json")
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript_path.is_file() else {"sentences": []}
    )

    regions = prefilter.get("regions", [])
    usable = [r for r in regions if r.get("status") == "usable"]
    usable_seconds = sum(
        float(r.get("duration_seconds") or 0.0) for r in usable
    )

    ident = {r["region_index"]: r for r in identity.get("regions", [])}
    usable_cos = [
        ident[r["region_index"]]
        for r in usable
        if r["region_index"] in ident
        and ident[r["region_index"]].get("target_cosine_median") is not None
    ]
    target_median = sorted(
        float(r["target_cosine_median"]) for r in usable_cos
    )

    style_rows = {r["region_index"]: r for r in style.get("regions", [])}
    negative_kinds: dict[str, int] = {}
    event_risks: list[float] = []
    for r in usable:
        row = style_rows.get(r["region_index"])
        if not row:
            continue
        kind = (row.get("style_evidence") or {}).get(
            "nearest_negative_kind"
        )
        if kind:
            negative_kinds[kind] = negative_kinds.get(kind, 0) + 1
        event_risks.append(
            _region_max(
                (row.get("event_evidence") or {}).get("risk_scores")
            )
        )

    sentences = transcript.get("sentences", [])
    eligible = [
        s for s in sentences
        if 3.0 <= float(s.get("duration_seconds") or 0.0) <= 12.0
        and not s.get("overlaps_quarantine")
        and s.get("boundary_clean")
        and (s.get("text_agreement") or {}).get("status") == "match"
    ]
    eligible_seconds = sum(
        float(s["duration_seconds"]) for s in eligible
    )

    reasons: list[str] = []
    if file_cosine is None:
        verdict = "discard"
        reasons.append("no_identity_evidence")
    else:
        # Same-timbre check vs the approved reference pack: the true target
        # speaker scored 0.864 on a 20 s usable-speech concatenation while
        # same-actor roleplay negatives reached 0.813 — treat <0.75 as a
        # different voice/register, 0.75–0.80 as borderline.
        if file_cosine < 0.75:
            reasons.append(f"file_cosine_low:{file_cosine:.3f}")
        elif file_cosine < 0.80:
            reasons.append(f"file_cosine_borderline:{file_cosine:.3f}")
        if usable_seconds < 30.0:
            reasons.append(f"usable_speech_too_short:{usable_seconds:.0f}s")
        if eligible_seconds < 10.0:
            reasons.append(
                f"strict_eligible_seconds:{eligible_seconds:.1f}"
            )
        verdict = "discard" if (
            file_cosine < 0.75
            or usable_seconds < 30.0
            or eligible_seconds < 10.0
        ) else "keep"

    return {
        "schema_version": 1,
        "assessment_version": ASSESS_IMPLEMENTATION_VERSION,
        "source_sha256": source_hash,
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "usable_seconds": round(usable_seconds, 1),
            "usable_regions": len(usable),
            "file_cosine_normal": file_cosine,
            "file_cosine_negatives": file_negative_cosines or {},
            "region_cosine_median": (
                target_median[len(target_median) // 2]
                if target_median else None
            ),
            "nearest_negative_kinds": negative_kinds,
            "event_risk_median": (
                sorted(event_risks)[len(event_risks) // 2]
                if event_risks else None
            ),
            "strict_eligible_sentences": len(eligible),
            "strict_eligible_seconds": round(eligible_seconds, 1),
        },
    }
